// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// The sync check's device workloads, one per subcommand: host_sync, multicast and pingpong. Each opens the system
// mesh with the streaming profiler on and checks the timeline it records. Needs a Blackhole system.

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <ranges>
#include <set>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

#include <fmt/format.h>
#include <tt-metalium/allocator.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/experimental/fabric/control_plane.hpp>
#include <tt-metalium/experimental/fabric/fabric.hpp>
#include <tt-metalium/experimental/streaming_profiler.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/kernel_types.hpp>
#include <tt-metalium/mesh_buffer.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/system_mesh.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt_stl/assert.hpp>

#include "hostdev/streaming_profiler_common.h"
#include "impl/context/metal_context.hpp"
#include "kernels/multicast_role.hpp"
#include "llrt/tt_cluster.hpp"

// What the workloads share: opening the system mesh, the matmul and NoC load kernels with their
// calibration against the device, and the per-core flag a kernel sets when it stops waiting for a peer.
namespace streaming_profiler_workload {

inline constexpr std::string_view kKernelDir = "tests/tt_metal/tools/profiler/streaming_profiler_sync/kernels/";

constexpr uint32_t kLoadBytes = 8192;
constexpr uint32_t kLoadPartners = 8;
constexpr uint32_t kLoadTileBytes = 2048;
constexpr uint32_t kPartnerStepX = 3, kPartnerStepY = 5;

// One region of every worker's L1, from the allocator: the kernels' flag, where word 0 is the flag, word 1 the round
// a kernel gave up waiting on (0 if none) and word 2 the host poke kernel's ack, the multicast's round values and the
// load's scratch.
struct WorkerL1 {
    static constexpr uint32_t kFlagBytes = 64;
    static constexpr uint32_t kValuesBytes = 64 * 1024;
    static constexpr uint32_t kLoadScratchBytes = 4 * kLoadBytes;
    std::shared_ptr<tt::tt_metal::distributed::MeshBuffer> buffer;
    uint32_t flag() const { return static_cast<uint32_t>(buffer->address()); }
    uint32_t ack() const { return flag() + 8; }
    uint32_t values() const { return flag() + kFlagBytes; }
    uint32_t load_scratch() const { return values() + kValuesBytes; }
};

// One page per L1 bank, and every worker is one bank, so each worker holds the region at the same address.
inline WorkerL1 reserve_worker_l1(tt::tt_metal::distributed::MeshDevice& mesh) {
    constexpr uint32_t kPerCore = WorkerL1::kFlagBytes + WorkerL1::kValuesBytes + WorkerL1::kLoadScratchBytes;
    const uint32_t banks = mesh.allocator()->get_num_banks(tt::tt_metal::BufferType::L1);
    return {tt::tt_metal::distributed::MeshBuffer::create(
        tt::tt_metal::distributed::ReplicatedBufferConfig{.size = uint64_t{kPerCore} * banks},
        tt::tt_metal::distributed::DeviceLocalBufferConfig{
            .page_size = kPerCore, .buffer_type = tt::tt_metal::BufferType::L1},
        &mesh)};
}

using Clock = std::chrono::steady_clock;
inline double seconds_since(Clock::time_point start) {
    return std::chrono::duration<double>(Clock::now() - start).count();
}

inline double median(std::vector<double> values) {
    std::ranges::nth_element(values, values.begin() + static_cast<std::ptrdiff_t>(values.size() / 2));
    return values[values.size() / 2];
}

inline double run_once(
    tt::tt_metal::distributed::MeshCommandQueue& cq, tt::tt_metal::distributed::MeshWorkload& workload) {
    const auto start = Clock::now();
    tt::tt_metal::distributed::EnqueueMeshWorkload(cq, workload, /*blocking=*/false);
    tt::tt_metal::distributed::Finish(cq);
    return seconds_since(start);
}

inline std::shared_ptr<tt::tt_metal::distributed::MeshDevice> open_system_mesh(const char* tag) {
    auto mesh_device = tt::tt_metal::distributed::MeshDevice::create(
        tt::tt_metal::distributed::MeshDeviceConfig(tt::tt_metal::distributed::SystemMesh::instance().shape()),
        DEFAULT_L1_SMALL_SIZE,
        DEFAULT_TRACE_REGION_SIZE,
        /*num_command_queues=*/1);
    if (!tt::tt_metal::experimental::streaming_profiler::IsActive()) {
        std::fprintf(stderr, "[%s] needs TT_METAL_STREAMING_PROFILER=1\n", tag);
        mesh_device->close();
        return nullptr;
    }
    return mesh_device;
}

inline tt::tt_metal::CoreRange worker_grid(tt::tt_metal::distributed::MeshDevice& mesh) {
    const tt::tt_metal::CoreCoord grid = mesh.compute_with_storage_grid_size();
    return tt::tt_metal::CoreRange(tt::tt_metal::CoreCoord(0, 0), tt::tt_metal::CoreCoord(grid.x - 1, grid.y - 1));
}

struct LoadSpec {
    uint32_t bursts = 1, mm_iters = 0, dm_iters = 0, idle_cycles = 0;
};

inline void add_load(
    tt::tt_metal::Program& program,
    tt::tt_metal::distributed::MeshDevice& mesh,
    const WorkerL1& worker_l1,
    const tt::tt_metal::CoreRangeSet& cores,
    const LoadSpec& spec,
    const std::set<tt::tt_metal::CoreCoord>& avoid = {}) {
    const std::string kernels(kKernelDir);
    const tt::tt_metal::CoreCoord grid = mesh.compute_with_storage_grid_size();
    for (const tt::CBIndex cb_index : {tt::CBIndex::c_0, tt::CBIndex::c_1, tt::CBIndex::c_16}) {
        tt::tt_metal::CreateCircularBuffer(
            program,
            cores,
            tt::tt_metal::CircularBufferConfig(2 * kLoadTileBytes, {{cb_index, tt::DataFormat::Float16_b}})
                .set_page_size(cb_index, kLoadTileBytes));
    }
    const tt::tt_metal::KernelHandle compute = tt::tt_metal::CreateKernel(
        program,
        kernels + "load_compute.cpp",
        cores,
        tt::tt_metal::ComputeConfig{.math_fidelity = tt::tt_metal::MathFidelity::HiFi4});
    const tt::tt_metal::KernelHandle dm0 = tt::tt_metal::CreateKernel(
        program,
        kernels + "load_dm.cpp",
        cores,
        tt::tt_metal::DataMovementConfig{
            .processor = tt::tt_metal::DataMovementProcessor::RISCV_0, .noc = tt::tt_metal::NOC::RISCV_0_default});
    const tt::tt_metal::KernelHandle dm1 = tt::tt_metal::CreateKernel(
        program,
        kernels + "load_dm.cpp",
        cores,
        tt::tt_metal::DataMovementConfig{
            .processor = tt::tt_metal::DataMovementProcessor::RISCV_1, .noc = tt::tt_metal::NOC::RISCV_1_default});
    for (const tt::tt_metal::CoreRange& range : cores.ranges()) {
        for (const tt::tt_metal::CoreCoord& core : range) {
            tt::tt_metal::SetRuntimeArgs(program, compute, core, {spec.bursts, spec.mm_iters, spec.idle_cycles});
            std::vector<uint32_t> args = {
                worker_l1.load_scratch(), kLoadBytes, spec.bursts, spec.dm_iters, spec.idle_cycles, kLoadPartners};
            for (uint32_t partner = 1; partner <= kLoadPartners; partner++) {
                tt::tt_metal::CoreCoord partner_core(
                    (core.x + kPartnerStepX * partner) % grid.x, (core.y + kPartnerStepY * partner) % grid.y);
                while (avoid.contains(partner_core)) {
                    partner_core.x = (partner_core.x + 1) % grid.x;
                }
                const tt::tt_metal::CoreCoord partner_noc = mesh.worker_core_from_logical_core(partner_core);
                args.push_back(kernel_profiler::word_of(kernel_profiler::NocXy{
                    .x = static_cast<uint32_t>(partner_noc.x), .y = static_cast<uint32_t>(partner_noc.y)}));
            }
            tt::tt_metal::SetRuntimeArgs(program, dm0, core, args);
            tt::tt_metal::SetRuntimeArgs(program, dm1, core, args);
        }
    }
}

inline tt::tt_metal::distributed::MeshWorkload make_load(
    tt::tt_metal::distributed::MeshDevice& mesh,
    const WorkerL1& worker_l1,
    const tt::tt_metal::CoreRangeSet& cores,
    const LoadSpec& spec) {
    tt::tt_metal::Program program = tt::tt_metal::CreateProgram();
    add_load(program, mesh, worker_l1, cores, spec);
    tt::tt_metal::distributed::MeshWorkload workload;
    workload.add_program(tt::tt_metal::distributed::MeshCoordinateRange(mesh.shape()), std::move(program));
    return workload;
}

// Seconds per HiFi4 matmul block and per NoC load round on the whole worker grid, each timed on its second run.
struct LoadRate {
    double mm_s = 0.0, dm_s = 0.0;
    // `base` with iteration counts set so each RISC's load runs for about `seconds`.
    LoadSpec lasting(double seconds, LoadSpec base = {}) const {
        base.mm_iters = std::max<uint32_t>(1, static_cast<uint32_t>(std::lround(seconds / mm_s)));
        base.dm_iters = std::max<uint32_t>(1, static_cast<uint32_t>(std::lround(seconds / dm_s)));
        return base;
    }
};

inline LoadRate calibrate_load(tt::tt_metal::distributed::MeshDevice& mesh, const WorkerL1& worker_l1) {
    const auto timed = [&](const LoadSpec& spec) {
        tt::tt_metal::distributed::MeshWorkload workload = make_load(mesh, worker_l1, worker_grid(mesh), spec);
        run_once(mesh.mesh_command_queue(), workload);
        return run_once(mesh.mesh_command_queue(), workload);
    };
    const LoadSpec matmul_only{.mm_iters = 2000}, noc_only{.dm_iters = 200};
    return {.mm_s = timed(matmul_only) / matmul_only.mm_iters, .dm_s = timed(noc_only) / noc_only.dm_iters};
}

struct FlagCore {
    tt::tt_metal::IDevice* device;
    tt::tt_metal::CoreCoord core;
};

inline void clear_flags(const WorkerL1& worker_l1, const std::vector<FlagCore>& cores) {
    std::vector<uint32_t> zero = {0, 0};
    for (const FlagCore& flag_core : cores) {
        tt::tt_metal::detail::WriteToDeviceL1(flag_core.device, flag_core.core, worker_l1.flag(), zero);
    }
}

inline uint32_t count_gave_up(const WorkerL1& worker_l1, const std::vector<FlagCore>& cores, const char* tag) {
    uint32_t gave_up = 0;
    for (const FlagCore& flag_core : cores) {
        std::vector<uint32_t> words(2, 0);
        tt::tt_metal::detail::ReadFromDeviceL1(flag_core.device, flag_core.core, worker_l1.flag(), 8, words);
        if (words[1] != 0) {
            std::printf(
                "[%s] chip %d core (%zu,%zu) gave up waiting for round %u (flag %u)\n",
                tag,
                flag_core.device->id(),
                flag_core.core.x,
                flag_core.core.y,
                words[1],
                words[0]);
            gave_up++;
        }
    }
    return gave_up;
}

}  // namespace streaming_profiler_workload

using namespace tt;
using namespace tt::tt_metal;
namespace sp = tt::tt_metal::experimental::streaming_profiler;
using namespace streaming_profiler_workload;

// Checks the device-to-host part of the timeline. For each round the host writes a round number into one worker's L1
// on every chip and polls for that worker's ack, and the worker records an HOST_RX zone in between. Placed on the host
// timeline, every zone must start after the host began its write and before it read the ack. The window is the host's
// MMIO round trip, a few microseconds wide, so the check is loose: it catches a host mapping off by more than that.
namespace host_sync {

namespace {
constexpr uint32_t kRounds = 2000;
constexpr auto kAckTimeout = std::chrono::seconds(1);
// Spaced so the capture runs long enough for the sync check to measure, and spans seconds of clock drift.
constexpr auto kRoundGap = std::chrono::milliseconds(1);

struct Window {
    Clock::time_point before, after;
};

double to_us(Clock::duration duration) { return std::chrono::duration<double, std::micro>(duration).count(); }
}  // namespace

int run(int, char**) {
    std::map<uint32_t, std::vector<Clock::time_point>> starts;
    uint64_t dropped_bytes = 0;
    auto sub = sp::RegisterCallback(
        [&](const sp::Batch<sp::RecordType::Zones>& batch) {
            dropped_bytes += batch.dropped_bytes();
            for (const sp::Zone& zone : batch.zones()) {
                if (std::string_view(zone.site().name) == "HOST_RX") {
                    starts[zone.core().chip_id].push_back(zone.start_time());
                }
            }
        },
        "host_sync");

    auto mesh_device = open_system_mesh("host_sync");
    if (!mesh_device) {
        return 1;
    }
    const WorkerL1 worker_l1 = reserve_worker_l1(*mesh_device);
    const CoreCoord core(0, 0);
    std::vector<FlagCore> cores;
    std::vector<uint32_t> zero(4, 0);
    for (IDevice* device : mesh_device->get_devices()) {
        cores.push_back({device, core});
        detail::WriteToDeviceL1(device, core, worker_l1.flag(), zero);
    }

    Program program = CreateProgram();
    const auto kid = CreateKernel(
        program,
        std::string(kKernelDir) + "host_poke_dm.cpp",
        core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});
    SetRuntimeArgs(program, kid, core, {worker_l1.flag(), kRounds});
    distributed::MeshWorkload workload;
    workload.add_program(distributed::MeshCoordinateRange(mesh_device->shape()), std::move(program));
    distributed::MeshCommandQueue& cq = mesh_device->mesh_command_queue();
    distributed::EnqueueMeshWorkload(cq, workload, /*blocking=*/false);

    // ReadFromDeviceL1 barriers every core on the chip first, which takes about 0.5 ms and would widen each window by
    // that much, so the ack poll reads the word directly.
    const auto& cluster = MetalContext::instance().get_cluster();
    std::map<uint32_t, std::vector<Window>> windows;
    bool timed_out = false;
    std::vector<uint32_t> round_word(1, 0);
    for (uint32_t round = 1; round <= kRounds && !timed_out; round++) {
        std::this_thread::sleep_for(kRoundGap);
        round_word[0] = round;
        for (const FlagCore& flag_core : cores) {
            const tt_cxy_pair ack_core(
                flag_core.device->id(),
                flag_core.device->virtual_core_from_logical_core(flag_core.core, CoreType::WORKER));
            const Clock::time_point before = Clock::now();
            detail::WriteToDeviceL1(flag_core.device, flag_core.core, worker_l1.flag(), round_word);
            while (true) {
                uint32_t ack = 0;
                cluster.read_core(&ack, sizeof(ack), ack_core, worker_l1.ack());
                if (ack == round) {
                    break;
                }
                if (Clock::now() - before > kAckTimeout) {
                    std::printf("[host_sync] chip %d gave no ack for round %u\n", flag_core.device->id(), round);
                    timed_out = true;
                    break;
                }
            }
            windows[flag_core.device->id()].push_back({before, Clock::now()});
        }
    }
    distributed::Finish(cq);
    const uint32_t gave_up = count_gave_up(worker_l1, cores, "host_sync");
    mesh_device->close();
    sub = {};

    size_t failed = 0;
    for (const auto& [chip, chip_windows] : windows) {
        const std::vector<Clock::time_point>& zone_starts = starts[chip];
        if (zone_starts.size() != chip_windows.size()) {
            std::printf(
                "[host_sync] chip %u: FAIL, %zu zones for %zu rounds\n", chip, zone_starts.size(), chip_windows.size());
            failed++;
            continue;
        }
        size_t outside = 0;
        std::vector<double> after_write, before_ack, width;
        for (size_t k = 0; k < chip_windows.size(); k++) {
            const double write_to_zone_us = to_us(zone_starts[k] - chip_windows[k].before);
            const double zone_to_ack_us = to_us(chip_windows[k].after - zone_starts[k]);
            outside += write_to_zone_us < 0.0 || zone_to_ack_us < 0.0 ? 1 : 0;
            after_write.push_back(write_to_zone_us);
            before_ack.push_back(zone_to_ack_us);
            width.push_back(to_us(chip_windows[k].after - chip_windows[k].before));
        }
        const bool pass = outside == 0;
        failed += pass ? 0 : 1;
        std::printf(
            "[host_sync] chip %u: %s, %zu of %zu zones outside their window; zone after the write p50 %.2f us (min "
            "%.2f), before the ack p50 %.2f us (min %.2f); window p50 %.2f us\n",
            chip,
            pass ? "PASS" : "FAIL",
            outside,
            chip_windows.size(),
            median(after_write),
            std::ranges::min(after_write),
            median(before_ack),
            std::ranges::min(before_ack),
            median(width));
    }
    const bool pass = failed == 0 && !timed_out && gave_up == 0 && dropped_bytes == 0;
    std::printf(
        "[host_sync] %s: %zu of %zu chips failed, %u cores gave up, %llu bytes dropped%s\n",
        pass ? "PASS" : "FAIL",
        failed,
        windows.size(),
        gave_up,
        static_cast<unsigned long long>(dropped_bytes),
        timed_out ? ", an ack timed out" : "");
    return pass ? 0 : 1;
}

}  // namespace host_sync

// Checks each chip's timeline core to core. One Tensix core per chip broadcasts rounds over each NoC, and every worker
// records the arrival as an MC_RX zone. A multicast takes a fixed 9 cycles per router hop, so once each arrival is on
// the host timeline and its hops are subtracted, every core should report the same instant. The test fails if any
// core's mean differs from the others' by a cycle or more, a round goes missing or records are dropped. The broadcasts
// run at the start and again after kIdleSeconds, so the check covers the timeline as the clocks drift.
namespace multicast {

namespace {
constexpr uint32_t kRounds = 4000;
static_assert(16 * (kRounds + 1) <= WorkerL1::kValuesBytes);
static_assert(kRounds % 2 == 0);
// 9 cycles per router hop (BlackholeA0/NoC README.md; measured 9.00).
constexpr double kCyclesPerHop = 9.0;
constexpr auto kIdleSeconds = std::chrono::seconds(10);

struct Arrival {
    int64_t tsc;
    int64_t ticks;
    uint16_t chip;
    uint8_t x, y;
    uint8_t physical_x, physical_y;
};

distributed::MeshWorkload make_multicast(
    distributed::MeshDevice& mesh, const WorkerL1& worker_l1, uint32_t runtime_id) {
    const CoreRange cores = worker_grid(mesh);
    const CoreCoord &lo = cores.start_coord, &hi = cores.end_coord;
    const CoreCoord vlo = mesh.worker_core_from_logical_core(lo);
    const CoreCoord vhi = mesh.worker_core_from_logical_core(hi);
    const uint32_t num_dests = static_cast<uint32_t>(cores.size() - 1);
    Program program = CreateProgram();
    program.set_runtime_id(runtime_id);
    const auto kid = CreateKernel(
        program,
        std::string(kKernelDir) + "multicast_dm.cpp",
        CoreRangeSet(cores),
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});
    for (const CoreCoord& core : cores) {
        const MulticastRole role =
            core == lo ? MulticastRole::Noc0Source : (core == hi ? MulticastRole::Noc1Source : MulticastRole::Receiver);
        // A NoC 1 rectangle runs from its start corner at the high coordinates down to the low ones.
        const CoreCoord& rect_start = role == MulticastRole::Noc1Source ? vhi : vlo;
        const CoreCoord& rect_end = role == MulticastRole::Noc1Source ? vlo : vhi;
        SetRuntimeArgs(
            program,
            kid,
            core,
            {static_cast<uint32_t>(role),
             static_cast<uint32_t>(rect_start.x),
             static_cast<uint32_t>(rect_start.y),
             static_cast<uint32_t>(rect_end.x),
             static_cast<uint32_t>(rect_end.y),
             num_dests,
             worker_l1.flag(),
             worker_l1.values(),
             kRounds});
    }
    distributed::MeshWorkload workload;
    workload.add_program(distributed::MeshCoordinateRange(mesh.shape()), std::move(program));
    return workload;
}

uint32_t multicast(distributed::MeshDevice& mesh, const WorkerL1& worker_l1, uint32_t runtime_id) {
    const CoreRange grid = worker_grid(mesh);
    std::vector<FlagCore> cores;
    for (IDevice* device : mesh.get_devices()) {
        for (const CoreCoord& core : grid) {
            cores.push_back({device, core});
        }
    }
    clear_flags(worker_l1, cores);
    distributed::MeshWorkload workload = make_multicast(mesh, worker_l1, runtime_id);
    run_once(mesh.mesh_command_queue(), workload);
    return count_gave_up(worker_l1, cores, "multicast");
}

struct Verdict {
    double worst_span_ticks = 0.0;
    bool incomplete = false;
};

using Round = std::map<CoreCoord, const Arrival*>;
struct Lane {
    CoreCoord src;
    std::vector<Round> rounds;
};
struct LaneKey {
    uint32_t chip = 0, noc = 0;
    auto operator<=>(const LaneKey&) const = default;
};
struct Lanes {
    std::map<LaneKey, Lane> by_chip_noc;
    bool incomplete = false;
};

// The NoC 0 source sends the odd rounds and the NoC 1 source the even ones, and a source never receives its own
// broadcast.
size_t round_of_arrival(
    const CoreCoord& core, size_t arrival_index, const CoreCoord& noc0_source, const CoreCoord& noc1_source) {
    if (core == noc0_source) {
        return 2 * arrival_index + 2;
    }
    if (core == noc1_source) {
        return 2 * arrival_index + 1;
    }
    return arrival_index + 1;
}

Lanes group_lanes(const std::vector<Arrival>& arrivals, const CoreCoord& grid, size_t num_chips) {
    std::map<uint32_t, std::vector<const Arrival*>> by_chip;
    for (const Arrival& arrival : arrivals) {
        by_chip[arrival.chip].push_back(&arrival);
    }
    const CoreCoord noc0_source{0, 0}, noc1_source{grid.x - 1, grid.y - 1};
    Lanes lanes;
    lanes.incomplete = by_chip.size() != num_chips;
    for (auto& [chip, chip_arrivals] : by_chip) {
        std::map<CoreCoord, std::vector<const Arrival*>> by_core;
        for (const Arrival* arrival : chip_arrivals) {
            by_core[CoreCoord{arrival->x, arrival->y}].push_back(arrival);
        }
        bool counts_ok = by_core.size() == grid.x * grid.y;
        for (auto& [core, list] : by_core) {
            std::ranges::sort(list, {}, &Arrival::ticks);
            const size_t want = core == noc0_source || core == noc1_source ? kRounds / 2 : kRounds;
            if (list.size() != want) {
                std::printf(
                    "[multicast] chip %u core (%zu,%zu): %zu arrivals, expected %zu\n",
                    chip,
                    core.x,
                    core.y,
                    list.size(),
                    want);
                counts_ok = false;
            }
        }
        if (!counts_ok) {
            lanes.incomplete = true;
            continue;
        }
        Lane& noc0_lane = lanes.by_chip_noc[{.chip = chip, .noc = 0}];
        Lane& noc1_lane = lanes.by_chip_noc[{.chip = chip, .noc = 1}];
        const Arrival* noc0_source_arrival = by_core.at(noc0_source).front();
        const Arrival* noc1_source_arrival = by_core.at(noc1_source).front();
        noc0_lane.src = CoreCoord(noc0_source_arrival->physical_x, noc0_source_arrival->physical_y);
        noc1_lane.src = CoreCoord(noc1_source_arrival->physical_x, noc1_source_arrival->physical_y);
        noc0_lane.rounds.resize(kRounds / 2);
        noc1_lane.rounds.resize(kRounds / 2);
        for (const auto& [core, list] : by_core) {
            for (size_t k = 0; k < list.size(); k++) {
                const size_t round = round_of_arrival(core, k, noc0_source, noc1_source);
                ((round & 1u) ? noc0_lane : noc1_lane).rounds[(round - 1) / 2][core] = list[k];
            }
        }
    }
    lanes.incomplete = lanes.incomplete || lanes.by_chip_noc.empty();
    return lanes;
}

struct LaneSpan {
    size_t cores = 0, rounds = 0;
    double ns_per_tick = 0.0, span_ns = 0.0, noise_ns = 0.0;
};

LaneSpan solve_lane(uint32_t chip, uint32_t noc, const Lane& lane) {
    const double ns_per_tsc = sp::NsPerTscTick();
    const std::vector<Round>& rounds = lane.rounds;
    const CoreCoord src = lane.src;
    std::vector<CoreCoord> cores;
    std::vector<double> hops;
    for (const auto& [core, arrival] : rounds.front()) {
        const CoreCoord physical(arrival->physical_x, arrival->physical_y);
        const int hops_x = noc == 0 ? static_cast<int>(physical.x) - static_cast<int>(src.x)
                                    : static_cast<int>(src.x) - static_cast<int>(physical.x);
        const int hops_y = noc == 0 ? static_cast<int>(physical.y) - static_cast<int>(src.y)
                                    : static_cast<int>(src.y) - static_cast<int>(physical.y);
        TT_FATAL(
            hops_x >= 0 && hops_y >= 0,
            "chip {} NoC {}: core {} is behind the source {}",
            chip,
            noc,
            physical.str(),
            src.str());
        cores.push_back(core);
        hops.push_back(static_cast<double>(hops_x + hops_y));
    }
    const size_t num_cores = cores.size(), num_rounds = rounds.size();
    std::vector<std::vector<double>> residual_ns(num_cores, std::vector<double>(num_rounds));
    double ns_per_tick_sum = 0.0;
    for (size_t k = 0; k < num_rounds; k++) {
        const size_t neighbour_round = k + 1 < num_rounds ? k + 1 : k - 1;
        const Arrival* ref = rounds[k].at(cores[0]);
        const Arrival* ref_in_neighbour = rounds[neighbour_round].at(cores[0]);
        const double round_ns_per_tick = static_cast<double>(ref_in_neighbour->tsc - ref->tsc) * ns_per_tsc /
                                         static_cast<double>(ref_in_neighbour->ticks - ref->ticks);
        ns_per_tick_sum += round_ns_per_tick;
        double round_residual_sum = 0.0;
        for (size_t i = 0; i < num_cores; i++) {
            const Arrival* arrival = rounds[k].at(cores[i]);
            residual_ns[i][k] =
                static_cast<double>(arrival->tsc - ref->tsc) * ns_per_tsc - kCyclesPerHop * hops[i] * round_ns_per_tick;
            round_residual_sum += residual_ns[i][k];
        }
        for (size_t i = 0; i < num_cores; i++) {
            residual_ns[i][k] -= round_residual_sum / static_cast<double>(num_cores);
        }
    }
    const double ns_per_tick = ns_per_tick_sum / static_cast<double>(num_rounds);
    double min_mean_ns = std::numeric_limits<double>::max(), max_mean_ns = std::numeric_limits<double>::lowest();
    double stderr_ns = 0.0;
    for (size_t i = 0; i < num_cores; i++) {
        double sum = 0.0, sum2 = 0.0;
        for (const double residual : residual_ns[i]) {
            sum += residual;
            sum2 += residual * residual;
        }
        const double mean = sum / static_cast<double>(num_rounds);
        stderr_ns = std::max(
            stderr_ns, std::sqrt(std::max(sum2 / static_cast<double>(num_rounds) - mean * mean, 0.0) / num_rounds));
        min_mean_ns = std::min(min_mean_ns, mean);
        max_mean_ns = std::max(max_mean_ns, mean);
    }
    return LaneSpan{
        .cores = num_cores,
        .rounds = num_rounds,
        .ns_per_tick = ns_per_tick,
        .span_ns = max_mean_ns - min_mean_ns,
        .noise_ns = stderr_ns};
}

Verdict check(const std::vector<Arrival>& arrivals, const CoreCoord& grid, size_t num_chips) {
    const Lanes lanes = group_lanes(arrivals, grid, num_chips);
    Verdict verdict{.incomplete = lanes.incomplete};
    std::printf("chip noc cores rounds  ns/tick   span ns (ticks)  noise ns\n");
    for (const auto& [key, lane] : lanes.by_chip_noc) {
        const LaneSpan span = solve_lane(key.chip, key.noc, lane);
        const double span_ticks = span.span_ns / span.ns_per_tick;
        verdict.worst_span_ticks = std::max(verdict.worst_span_ticks, span_ticks);
        std::printf(
            "%4u %3u %5zu  %5zu  %.4f   %6.3f (%5.2f)   %6.3f\n",
            key.chip,
            key.noc,
            span.cores,
            span.rounds,
            span.ns_per_tick,
            span.span_ns,
            span_ticks,
            span.noise_ns);
    }
    return verdict;
}

struct PhaseResult {
    std::string name;
    uint32_t gave_up;
};
}  // namespace

int run(int argc, char** argv) {
    if (argc > 1) {
        std::printf("usage: %s\n", argv[0]);
        return 1;
    }

    std::map<uint32_t, std::vector<Arrival>> by_phase;
    uint64_t dropped_bytes = 0;
    auto sub = sp::RegisterCallback(
        [&](const sp::Batch<sp::RecordType::Zones>& batch) {
            dropped_bytes += batch.dropped_bytes();
            for (const sp::Zone& zone : batch.zones()) {
                if (std::string_view(zone.site().name) != "MC_RX") {
                    continue;
                }
                const sp::Core core = zone.core();
                by_phase[zone.runtime_id()].push_back(Arrival{
                    zone.start_tsc(),
                    static_cast<int64_t>(zone.start_device_cycles()),
                    static_cast<uint16_t>(core.chip_id),
                    static_cast<uint8_t>(core.logical.x),
                    static_cast<uint8_t>(core.logical.y),
                    static_cast<uint8_t>(core.physical.x),
                    static_cast<uint8_t>(core.physical.y)});
            }
        },
        "multicast");

    auto mesh_device = open_system_mesh("multicast");
    if (!mesh_device) {
        return 1;
    }
    const CoreCoord grid = mesh_device->compute_with_storage_grid_size();
    const size_t num_chips = mesh_device->get_devices().size();

    const WorkerL1 worker_l1 = reserve_worker_l1(*mesh_device);
    std::printf(
        "[multicast] %zux%zu Tensix cores x %u rounds (%u a NoC) on %zu chips\n",
        grid.x,
        grid.y,
        kRounds,
        kRounds / 2,
        num_chips);
    std::fflush(stdout);

    const std::vector<std::string> phases = {"start", fmt::format("after {} s idle", kIdleSeconds.count())};
    std::vector<PhaseResult> results;
    for (size_t k = 0; k < phases.size(); k++) {
        if (k != 0) {
            std::this_thread::sleep_for(kIdleSeconds);
        }
        results.push_back({phases[k], multicast(*mesh_device, worker_l1, static_cast<uint32_t>(k + 1))});
        std::printf("[multicast] phase %zu/%zu %s done\n", k + 1, phases.size(), results.back().name.c_str());
        std::fflush(stdout);
    }
    mesh_device->close();
    sub = {};

    double worst_span_ticks = 0.0;
    uint32_t total_gave_up = 0;
    size_t failed = 0;
    for (size_t k = 0; k < results.size(); k++) {
        const PhaseResult& result = results[k];
        std::printf("[multicast] phase %zu/%zu %s\n", k + 1, results.size(), result.name.c_str());
        const auto id = static_cast<uint32_t>(k + 1);
        const Verdict verdict = check(by_phase[id], grid, num_chips);
        by_phase.erase(id);
        const bool pass = verdict.worst_span_ticks < 1.0 && result.gave_up == 0 && !verdict.incomplete;
        std::printf(
            "[multicast] %s: %s, worst span %.2f ticks; %u cores gave up%s\n",
            result.name.c_str(),
            pass ? "PASS" : "FAIL",
            verdict.worst_span_ticks,
            result.gave_up,
            verdict.incomplete ? "; SOME CHIPS MISSING OR WITH WRONG ARRIVAL COUNTS" : "");
        worst_span_ticks = std::max(worst_span_ticks, verdict.worst_span_ticks);
        total_gave_up += result.gave_up;
        failed += pass ? 0 : 1;
    }
    const bool pass = failed == 0 && dropped_bytes == 0;
    std::printf(
        "[multicast] %s: worst core-to-core span of the device-local timeline %.2f ticks over %zu phases, %zu failed; "
        "%u cores gave up, %llu bytes dropped\n",
        pass ? "PASS" : "FAIL",
        worst_span_ticks,
        phases.size(),
        failed,
        total_gave_up,
        static_cast<unsigned long long>(dropped_bytes));
    return pass ? 0 : 1;
}

}  // namespace multicast

// Fabric traffic under the profiler. Over 2D fabric, one worker on each side of every linked chip pair ping-pongs
// atomic increments for --seconds, recording a PP_TX zone per send and a PP_RX zone per arrival, then again with --load
// while the other workers run matmul and NoC load. It fails if a kernel gives up waiting for its peer or any round's
// zones don't all reach the host. As a sanity check of the synced timeline, it also fails if a pair's first rounds
// break causality or either direction's one-way time is far from half the round trip. --idle just opens the mesh and
// sleeps, for the sync check's idle arm.
// The sync gate runs it with the sync check on, so its traffic shares the routers with the link sync.
namespace pingpong {

namespace {
constexpr uint32_t kRounds = 20000;  // a pass long enough (~25 ms) that the host work between passes is the minority

struct Pair {
    uint32_t chip_a, chip_b;
    CoreCoord core_a, core_b;
    uint32_t link_a, link_b;
};

constexpr uint32_t kTimedRounds = 1000;
struct Stamp {
    uint64_t cycles;
    int64_t tsc;
    bool tx;
};
// Every zone is counted; the first kTimedRounds rounds' zones are also kept for the timeline check.
struct Stamps {
    size_t tx = 0, rx = 0;
    std::vector<Stamp> first;
};
using StampsByCore = std::map<std::pair<uint32_t, CoreCoord>, Stamps>;
struct Chip {
    distributed::MeshCoordinate coord;
    IDevice* device;
    tt::tt_fabric::FabricNodeId node;
};
using Chips = std::map<uint32_t, Chip>;

std::vector<Pair> find_pairs(distributed::MeshDevice& mesh_device, const Chips& chips) {
    const CoreCoord grid = mesh_device.compute_with_storage_grid_size();
    std::map<uint32_t, uint32_t> next_worker;
    const auto take_worker = [&](uint32_t chip) {
        const uint32_t worker = next_worker[chip]++;
        TT_FATAL(worker < grid.x * grid.y, "chip {} has more fabric neighbours than worker cores", chip);
        return CoreCoord(worker % grid.x, worker / grid.x);
    };
    std::vector<Pair> pairs;
    for (const auto& [id_a, chip_a] : chips) {
        for (const auto& [id_b, chip_b] : chips) {
            if (id_a >= id_b || tt::tt_fabric::get_neighbor_eth_directions(chip_a.node, chip_b.node).empty()) {
                continue;
            }
            const auto links_ab = tt::tt_fabric::get_forwarding_link_indices(chip_a.node, chip_b.node);
            const auto links_ba = tt::tt_fabric::get_forwarding_link_indices(chip_b.node, chip_a.node);
            TT_FATAL(!links_ab.empty() && !links_ba.empty(), "no fabric link between chips {} and {}", id_a, id_b);
            pairs.push_back(Pair{id_a, id_b, take_worker(id_a), take_worker(id_b), links_ab.front(), links_ba.front()});
        }
    }
    return pairs;
}

void arm(
    Program& program,
    distributed::MeshDevice& mesh_device,
    const WorkerL1& worker_l1,
    const CoreCoord& core,
    uint32_t role,
    const CoreCoord& peer,
    const tt::tt_fabric::FabricNodeId& src,
    const tt::tt_fabric::FabricNodeId& dst,
    uint32_t link_idx) {
    const auto kid = CreateKernel(
        program,
        std::string(kKernelDir) + "pingpong_fabric_dm.cpp",
        core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});
    const CoreCoord vpeer = mesh_device.worker_core_from_logical_core(peer);
    std::vector<uint32_t> args = {
        role,
        static_cast<uint32_t>(vpeer.x),
        static_cast<uint32_t>(vpeer.y),
        worker_l1.flag(),
        kRounds,
        static_cast<uint32_t>(dst.chip_id),
        static_cast<uint32_t>(dst.mesh_id.get())};
    tt::tt_fabric::append_fabric_connection_rt_args(src, dst, link_idx, program, core, args);
    SetRuntimeArgs(program, kid, core, args);
}

distributed::MeshWorkload build_workload(
    distributed::MeshDevice& mesh_device,
    const WorkerL1& worker_l1,
    const std::vector<Pair>& pairs,
    const Chips& chips,
    const std::optional<LoadSpec>& load) {
    std::map<uint32_t, Program> programs;
    std::map<uint32_t, std::set<CoreCoord>> ends;
    for (const uint32_t chip : std::views::keys(chips)) {
        programs.emplace(chip, CreateProgram());
    }
    for (const Pair& pair : pairs) {
        const auto &node_a = chips.at(pair.chip_a).node, &node_b = chips.at(pair.chip_b).node;
        arm(programs.at(pair.chip_a), mesh_device, worker_l1, pair.core_a, 0, pair.core_b, node_a, node_b, pair.link_a);
        arm(programs.at(pair.chip_b), mesh_device, worker_l1, pair.core_b, 1, pair.core_a, node_b, node_a, pair.link_b);
        ends[pair.chip_a].insert(pair.core_a);
        ends[pair.chip_b].insert(pair.core_b);
    }
    distributed::MeshWorkload workload;
    const CoreRange workers = worker_grid(mesh_device);
    for (auto& [chip, program] : programs) {
        if (load) {
            std::vector<CoreRange> others;
            for (const CoreCoord& core : workers) {
                if (!ends[chip].contains(core)) {
                    others.emplace_back(core);
                }
            }
            add_load(program, mesh_device, worker_l1, CoreRangeSet(others), *load, ends[chip]);
        }
        const auto& coord = chips.at(chip).coord;
        workload.add_program(distributed::MeshCoordinateRange(coord, coord), std::move(program));
    }
    return workload;
}

uint32_t count_missing(const std::vector<Pair>& pairs, const StampsByCore& by_core, size_t expected) {
    uint32_t failures = 0;
    for (const Pair& pair : pairs) {
        const auto found_a = by_core.find({pair.chip_a, pair.core_a});
        const auto found_b = by_core.find({pair.chip_b, pair.core_b});
        const size_t atx = found_a == by_core.end() ? 0 : found_a->second.tx;
        const size_t arx = found_a == by_core.end() ? 0 : found_a->second.rx;
        const size_t btx = found_b == by_core.end() ? 0 : found_b->second.tx;
        const size_t brx = found_b == by_core.end() ? 0 : found_b->second.rx;
        if (atx != expected || brx != expected || btx != expected || arx != expected) {
            std::printf(
                "chip %u - chip %u: stamps %zu/%zu/%zu/%zu of %zu\n",
                pair.chip_a,
                pair.chip_b,
                atx,
                brx,
                btx,
                arx,
                expected);
            failures++;
        }
    }
    return failures;
}
// Each direction's ping and reply times on the synced timeline, against the round trip minus the turnaround, which each
// end measures on its own clock. A sync error between two chips pushes one direction's one-way times up and the other's
// down. On the 8-chip LoudBox the ping takes 54% of that span and the reply 46% on every pair, whichever chip sends, so
// a 35-65% band leaves the fabric room and still catches any sync error above about 0.2 us.
constexpr double kLegBand = 0.15;

uint32_t check_timeline(const std::vector<Pair>& pairs, StampsByCore& by_core) {
    const double ns_per_tsc = sp::NsPerTscTick();
    uint32_t failures = 0;
    for (const Pair& pair : pairs) {
        std::vector<Stamp>& stamps_a = by_core[{pair.chip_a, pair.core_a}].first;
        std::vector<Stamp>& stamps_b = by_core[{pair.chip_b, pair.core_b}].first;
        for (std::vector<Stamp>* stamps : {&stamps_a, &stamps_b}) {
            std::ranges::sort(*stamps, {}, &Stamp::cycles);
        }
        enum Leg : size_t { kPingAB, kPingBA, kReplyAB, kReplyBA, kLegs };
        std::array<std::vector<double>, kLegs> legs;
        std::vector<double> base;
        uint32_t acausal = 0, misread = 0;
        for (size_t k = 0; k < std::min(stamps_a.size(), stamps_b.size()) / 2; k++) {
            // Round k + 1: core_b (role 1) sends first on odd rounds, core_a on even ones.
            const bool b_first = (k & 1u) == 0;
            const Stamp* sender = &(b_first ? stamps_b : stamps_a)[2 * k];
            const Stamp* replier = &(b_first ? stamps_a : stamps_b)[2 * k];
            if (!sender[0].tx || sender[1].tx || replier[0].tx || !replier[1].tx) {
                misread++;
                continue;
            }
            const double ping_sent = sender[0].tsc * ns_per_tsc, reply_received = sender[1].tsc * ns_per_tsc;
            const double ping_received = replier[0].tsc * ns_per_tsc, reply_sent = replier[1].tsc * ns_per_tsc;
            acausal += (ping_received <= ping_sent ? 1u : 0u) + (reply_received <= reply_sent ? 1u : 0u);
            legs[b_first ? kPingBA : kPingAB].push_back(ping_received - ping_sent);
            legs[b_first ? kReplyAB : kReplyBA].push_back(reply_received - reply_sent);
            base.push_back((reply_received - ping_sent) - (reply_sent - ping_received));
        }
        if (std::ranges::any_of(legs, [](const std::vector<double>& leg) { return leg.empty(); })) {
            std::printf("chip %u - chip %u: FAIL, no timed rounds in some direction\n", pair.chip_a, pair.chip_b);
            failures++;
            continue;
        }
        const double full = median(base);
        std::array<double, kLegs> leg_median{};
        bool pair_ok = acausal == 0 && misread == 0;
        for (size_t i = 0; i < legs.size(); i++) {
            leg_median[i] = median(legs[i]);
            pair_ok = pair_ok && std::abs(leg_median[i] / full - 0.5) <= kLegBand;
        }
        std::printf(
            "chip %u - chip %u: %s, of %.0f ns round trip less turnaround: ping %.0f / %.0f ns, reply %.0f / %.0f ns "
            "(a to b / b to a); %u acausal, %u misread of %zu rounds\n",
            pair.chip_a,
            pair.chip_b,
            pair_ok ? "ok" : "FAIL",
            full,
            leg_median[kPingAB],
            leg_median[kPingBA],
            leg_median[kReplyAB],
            leg_median[kReplyBA],
            acausal,
            misread,
            base.size() + misread);
        failures += pair_ok ? 0 : 1;
    }
    return failures;
}
}  // namespace

int run(int argc, char** argv) {
    double seconds = 10.0;
    bool with_load = false, idle = false;
    for (int i = 1; i < argc; i++) {
        const std::string_view arg = argv[i];
        if (arg == "--seconds" && i + 1 < argc) {
            seconds = std::strtod(argv[++i], nullptr);
        } else if (arg == "--load" && !idle) {
            with_load = true;
        } else if (arg == "--idle" && !with_load) {
            idle = true;
        } else {
            std::fprintf(stderr, "usage: %s [--seconds S] [--load | --idle]\n", argv[0]);
            return 2;
        }
    }
    StampsByCore by_core;
    sp::Callback sub;
    if (!idle) {
        sub = sp::RegisterCallback(
            [&](const sp::Batch<sp::RecordType::Zones>& batch) {
                for (const sp::Zone& zone : batch.zones()) {
                    const std::string_view name = zone.site().name;
                    if (name == "PP_TX" || name == "PP_RX") {
                        Stamps& stamps = by_core[{zone.core().chip_id, zone.core().logical}];
                        (name == "PP_TX" ? stamps.tx : stamps.rx)++;
                        if (stamps.first.size() < 2 * kTimedRounds) {
                            stamps.first.push_back({zone.start_device_cycles(), zone.start_tsc(), name == "PP_TX"});
                        }
                    }
                }
            },
            "pingpong-fabric");
        tt::tt_fabric::SetFabricConfig(tt::tt_fabric::FabricConfig::FABRIC_2D);
    }
    auto mesh_device = open_system_mesh("pingpong-fabric");
    if (!mesh_device) {
        return 1;
    }

    Chips chips;
    if (!idle) {
        for (const auto& coord : distributed::MeshCoordinateRange(mesh_device->shape())) {
            IDevice* device = mesh_device->get_device(coord);
            chips.emplace(
                device->id(),
                Chip{coord, device, tt::tt_fabric::get_fabric_node_id_from_physical_chip_id(device->id())});
        }
    }
    const std::vector<Pair> pairs = find_pairs(*mesh_device, chips);
    std::vector<FlagCore> flag_cores;
    for (const Pair& pair : pairs) {
        flag_cores.push_back({chips.at(pair.chip_a).device, pair.core_a});
        flag_cores.push_back({chips.at(pair.chip_b).device, pair.core_b});
    }
    distributed::MeshCommandQueue& cq = mesh_device->mesh_command_queue();
    const WorkerL1 worker_l1 = reserve_worker_l1(*mesh_device);
    uint32_t failures = 0, passes = 0;
    if (idle) {
        std::printf("[pingpong-fabric] idle, %zu chips, %.1f s\n", mesh_device->num_devices(), seconds);
        std::fflush(stdout);
        std::this_thread::sleep_for(std::chrono::duration<double>(seconds));
    } else {
        TT_FATAL(!pairs.empty(), "no fabric-linked chip pairs");
        std::printf(
            "[pingpong-fabric] %zu linked pairs x %u rounds on %zu chips\n", pairs.size(), kRounds, chips.size());
        std::fflush(stdout);
        const auto run_passes = [&](distributed::MeshWorkload& workload) {
            const auto start = Clock::now();
            double last_pass_s = 0.0;
            do {
                clear_flags(worker_l1, flag_cores);
                last_pass_s = run_once(cq, workload);
                failures += count_gave_up(worker_l1, flag_cores, "pingpong-fabric");
                passes++;
            } while (seconds_since(start) < seconds);
            return last_pass_s;
        };
        distributed::MeshWorkload unloaded = build_workload(*mesh_device, worker_l1, pairs, chips, std::nullopt);
        const double unloaded_s = run_passes(unloaded);
        std::printf("[pingpong-fabric] %u unloaded passes, %.1f ms each\n", passes, unloaded_s * 1e3);
        if (with_load) {
            distributed::MeshWorkload loaded = build_workload(
                *mesh_device, worker_l1, pairs, chips, calibrate_load(*mesh_device, worker_l1).lasting(unloaded_s));
            const uint32_t before = passes;
            const double loaded_s = run_passes(loaded);
            std::printf("[pingpong-fabric] %u loaded passes, %.1f ms each\n", passes - before, loaded_s * 1e3);
        }
    }
    std::fflush(stdout);
    mesh_device->close();
    sub = {};
    failures += count_missing(pairs, by_core, static_cast<size_t>(passes) * kRounds);
    failures += check_timeline(pairs, by_core);
    return failures == 0 ? 0 : 1;
}

}  // namespace pingpong

int main(int argc, char** argv) {
    const std::string_view workload = argc > 1 ? argv[1] : "";
    if (workload == "host_sync") {
        return host_sync::run(argc - 1, argv + 1);
    }
    if (workload == "multicast") {
        return multicast::run(argc - 1, argv + 1);
    }
    if (workload == "pingpong") {
        return pingpong::run(argc - 1, argv + 1);
    }
    std::fprintf(stderr, "usage: %s host_sync | multicast | pingpong [--seconds S] [--load | --idle]\n", argv[0]);
    return 2;
}
