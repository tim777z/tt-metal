// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "impl/streaming_profiler/ops_csv.hpp"

#include <algorithm>
#include <cstdio>
#include <string>

namespace tt::tt_metal::streaming_profiler {

namespace api = experimental::streaming_profiler;

OpsCsvConsumer::OpsCsvConsumer(const std::string& path) : file_(path, "ops CSV") {}

void OpsCsvConsumer::operator()(const Batch& batch) {
    for (const auto& z : batch.zones()) {
        if (z.runtime_id() == 0 || !z.site().name.ends_with("-KERNEL")) {
            continue;
        }
        const api::Core c = z.core();
        const bool eth = c.processor >= api::Processor::ERISC0;
        const uint32_t risc = static_cast<uint32_t>(eth ? api::Processor::ERISC0 : c.processor);
        const CoreKey core_key{
            .eth = eth, .y = static_cast<uint32_t>(c.logical.y), .x = static_cast<uint32_t>(c.logical.x)};
        // The wrapper zone never nests, so the k-th one on a lane for a program is its k-th execution.
        uint32_t& completed = executions_seen_[{
            .chip = static_cast<uint32_t>(c.chip_id),
            .core = core_key,
            .processor = static_cast<uint32_t>(c.processor),
            .program = z.runtime_id()}];
        OpAgg& op = ops_[{.chip = static_cast<uint32_t>(c.chip_id), .program = z.runtime_id(), .execution = completed}];
        completed++;
        op.start_cycles = std::min(op.start_cycles, z.start_device_cycles());
        op.end_cycles = std::max(op.end_cycles, z.end_device_cycles());
        const int64_t start = z.start_tsc();
        const int64_t end = z.end_tsc();
        Span& core = op.cores[core_key];
        op.latest_core_start_tsc = std::max(op.latest_core_start_tsc, start);
        op.risc_start_tsc[risc] = std::min(op.risc_start_tsc[risc], start);
        core.start = std::min(core.start, start);
        op.risc_end_tsc[risc] = std::max(op.risc_end_tsc[risc], end);
        core.end = std::max(core.end, end);
    }
}

void OpsCsvConsumer::write_csv() {
    FILE* const out = file_.begin([](FILE* file) {
        std::fputs(
            "DEVICE ID,GLOBAL CALL COUNT,EXECUTION,CORE COUNT,DEVICE KERNEL START CYCLE,DEVICE KERNEL END CYCLE,"
            "DEVICE KERNEL DURATION [ns],DEVICE KERNEL DURATION DM START [ns],"
            "DEVICE KERNEL DURATION PER CORE MIN [ns],DEVICE KERNEL DURATION PER CORE MAX [ns],"
            "DEVICE KERNEL DURATION PER CORE AVG [ns],DEVICE KERNEL FIRST TO LAST START [ns],"
            "DEVICE BRISC KERNEL DURATION [ns],DEVICE NCRISC KERNEL DURATION [ns],"
            "DEVICE TRISC0 KERNEL DURATION [ns],DEVICE TRISC1 KERNEL DURATION [ns],"
            "DEVICE TRISC2 KERNEL DURATION [ns],DEVICE ERISC KERNEL DURATION [ns]\n",
            file);
    });
    for (const auto& [key, op] : ops_) {
        auto elapsed_ns = [](int64_t start, int64_t end) {
            return end > start ? static_cast<double>(end - start) * api::NsPerTscTick() : 0.0;
        };
        const auto risc_start = [&](api::Processor processor) {
            return op.risc_start_tsc[static_cast<uint32_t>(processor)];
        };
        const auto risc_ns = [&](api::Processor processor) {
            return elapsed_ns(risc_start(processor), op.risc_end_tsc[static_cast<uint32_t>(processor)]);
        };
        const int64_t start_tsc = std::ranges::min(op.risc_start_tsc);
        const int64_t end_tsc = std::ranges::max(op.risc_end_tsc);
        const int64_t dm_start_tsc = std::min(
            {risc_start(api::Processor::BRISC),
             risc_start(api::Processor::NCRISC),
             risc_start(api::Processor::ERISC0)});
        double core_min = 0.0, core_max = 0.0, core_sum = 0.0;
        uint32_t core_n = 0;
        for (const auto& [core, span] : op.cores) {
            if (span.end <= span.start) {
                continue;
            }
            const double core_ns = elapsed_ns(span.start, span.end);
            core_min = core_n == 0 ? core_ns : std::min(core_min, core_ns);
            core_max = std::max(core_max, core_ns);
            core_sum += core_ns;
            core_n++;
        }
        std::fprintf(
            out,
            "%u,%u,%u,%u,%llu,%llu,%.0f,%.0f,%.0f,%.0f,%.0f,%.0f,%.0f,%.0f,%.0f,%.0f,%.0f,",
            key.chip,
            key.program,
            key.execution,
            core_n,
            static_cast<unsigned long long>(op.start_cycles),
            static_cast<unsigned long long>(op.end_cycles),
            elapsed_ns(start_tsc, end_tsc),
            elapsed_ns(dm_start_tsc, end_tsc),
            core_min,
            core_max,
            core_n != 0 ? core_sum / core_n : 0.0,
            elapsed_ns(start_tsc, op.latest_core_start_tsc),
            risc_ns(api::Processor::BRISC),
            risc_ns(api::Processor::NCRISC),
            risc_ns(api::Processor::TRISC0),
            risc_ns(api::Processor::TRISC1),
            risc_ns(api::Processor::TRISC2));
        // The classic report leaves the column empty for an op with no eth kernel zone.
        if (op.risc_end_tsc[static_cast<uint32_t>(api::Processor::ERISC0)] != INT64_MIN) {
            std::fprintf(out, "%.0f", risc_ns(api::Processor::ERISC0));
        }
        std::fputc('\n', out);
    }
    std::fflush(out);
    ops_.clear();
    executions_seen_.clear();
}

}  // namespace tt::tt_metal::streaming_profiler
