// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "impl/streaming_profiler/tracy_consumer.hpp"

#if defined(TRACY_ENABLE)

#include <algorithm>
#include <cstring>

#include <fmt/format.h>
#include <common/TracyTTDeviceData.hpp>
#include <tracy/Tracy.hpp>
#include <client/TracyProfiler.hpp>

namespace tt::tt_metal::streaming_profiler {

namespace api = experimental::streaming_profiler;

namespace {

constexpr size_t kSrclocTableInitial = 1024;

// By physical coordinate: an eth core's logical coordinate can equal a Tensix core's.
uint64_t lane_key(const api::Core& core) {
    return (static_cast<uint64_t>(core.chip_id) << 32) | ((static_cast<uint64_t>(core.physical.x) & 0xFFFu) << 20) |
           ((static_cast<uint64_t>(core.physical.y) & 0xFFFu) << 8) | (static_cast<uint64_t>(core.processor) & 0xFFu);
}

uint64_t srcloc_key(std::string_view name, uint32_t processor) {
    return (static_cast<uint64_t>(name.size()) << 32) | processor;
}

size_t srcloc_hash(const char* name, uint64_t key) {
    return (((reinterpret_cast<uintptr_t>(name) >> 3) ^ key) * 0x9E3779B97F4A7C15ull) >> 32;
}

// Tracy's RiscType has only one ERISC. The second ERISC uses a value of the 6-bit RISC field that no RiscType uses,
// which gives its row its own thread id.
constexpr tracy::RiscType kTracyRisc[kProcessorCount] = {
    tracy::RiscType::BRISC,
    tracy::RiscType::NCRISC,
    tracy::RiscType::TRISC_0,
    tracy::RiscType::TRISC_1,
    tracy::RiscType::TRISC_2,
    tracy::RiscType::ERISC,
    static_cast<tracy::RiscType>(32 + static_cast<uint8_t>(tracy::RiscType::ERISC))};
constexpr uint32_t kProcessorColor[kProcessorCount] = {
    tracy::Color::Orange2,
    tracy::Color::SeaGreen3,
    tracy::Color::SkyBlue3,
    tracy::Color::Turquoise2,
    tracy::Color::CadetBlue1,
    tracy::Color::Yellow3,
    tracy::Color::Yellow2};

tracy::TTDeviceMarker device_marker(const api::Core& core) {
    tracy::TTDeviceMarker marker;
    marker.chip_id = core.chip_id;
    marker.core_x = core.logical.x;
    marker.core_y = core.logical.y;
    marker.risc = kTracyRisc[static_cast<uint32_t>(core.processor)];
    return marker;
}

constexpr int64_t kRoomForRecordsBeforeAnchorNs = 2'000'000'000;

int64_t origin_margin_ns(int64_t anchor_tracy) {
    const int64_t anchor_ns =
        static_cast<int64_t>(static_cast<double>(anchor_tracy - TracyGetBaseTime()) * TracyGetTimerMul());
    return std::clamp<int64_t>(anchor_ns, 0, kRoomForRecordsBeforeAnchorNs);
}

}  // namespace

TracyConsumer::TracyConsumer() :
    anchor_tracy_(tracy::Profiler::GetTime()),
    origin_margin_ns_(origin_margin_ns(anchor_tracy_)),
    srcloc_table_(kSrclocTableInitial) {}

void TracyConsumer::operator()(const Batch& batch) {
    for (const api::Zone& z : batch.zones()) {
        push_zone(z.core(), z.site().name, z.start_tsc(), z.end_tsc());
    }
    for (const api::TimestampedData& d : batch.timestamped_data()) {
        push_marker(d.core(), d.site().name, d.tsc(), d.runtime_id(), d.payload());
    }
    for (const api::Event& e : batch.events()) {
        push_marker(e.core(), e.site().name, e.tsc(), e.runtime_id(), {});
    }
}

// A record's TSC comes from host_sync's tsc_now(), the same counter Tracy's GetTime() reads (rdtsc on x86-64,
// CLOCK_MONOTONIC_RAW elsewhere), so timeline_ns can subtract a GetTime() anchor from it.
int64_t TracyConsumer::timeline_ns(int64_t tsc) const {
    return static_cast<int64_t>(static_cast<double>(tsc - anchor_tracy_) * TracyGetTimerMul()) + origin_margin_ns_;
}

TracyConsumer::Lane TracyConsumer::lane(const Core& core) {
    const uint64_t key = lane_key(core);
    if (key == lane_key_) {
        return lane_hit_;
    }
    Lane ln;
    ln.processor = static_cast<uint32_t>(core.processor);
    auto [it, fresh] = cores_.try_emplace(key & ~uint64_t{0xFF});
    CoreEntry& ce = it->second;
    if (fresh) {
        ce.ctx = TracyTTContext();
        // Everything the sink emits goes through this thread's lock-free queue, and its FIFO order keeps the context
        // ahead of the zones that refer to it.
        TracyTTContextPopulateCalibratedLockfree(ce.ctx, anchor_tracy_, static_cast<double>(origin_margin_ns_), 1.0);
        const std::string name = fmt::format(
            "Device: {}, {}Logical ({},{}) Physical ({},{})",
            core.chip_id,
            core.processor >= api::Processor::ERISC0 ? "Ethernet " : "",
            core.logical.x,
            core.logical.y,
            core.physical.x,
            core.physical.y);
        TracyTTContextNameLockfree(ce.ctx, name.c_str(), name.size());
    }
    if ((ce.named & (1u << ln.processor)) == 0) {
        // Use the marker path's thread id (TTDeviceMarker::get_thread_id) so zones and markers share each RISC's row.
        ce.thread[ln.processor] = device_marker(core).get_thread_id();
        tracy::SetThreadName(ce.thread[ln.processor], kProcessorNames[ln.processor]);
        ce.named |= static_cast<uint8_t>(1u << ln.processor);
    }
    ln.ctx = ce.ctx;
    ln.thread = ce.thread[ln.processor];
    lane_key_ = key;
    lane_hit_ = ln;
    return ln;
}

const void* TracyConsumer::srcloc(std::string_view name, uint32_t processor) {
    static constexpr char kUnnamed[] = "";
    if (name.data() == nullptr) {
        name = kUnnamed;  // a null address would read as an empty table slot
    }
    const uint64_t key = srcloc_key(name, processor);
    const size_t mask = srcloc_table_.size() - 1;
    const char* const name_ptr = name.data();  // NOLINT(bugprone-suspicious-stringview-data-usage)
    for (size_t i = srcloc_hash(name_ptr, key) & mask;; i = (i + 1) & mask) {
        const SrclocEntry& e = srcloc_table_[i];
        if (e.name == name_ptr && e.key == key) {
            return e.srcloc;
        }
        if (e.name == nullptr) {
            return srcloc_slow(name, name_ptr, key, processor);
        }
    }
}

const void* TracyConsumer::srcloc_slow(std::string_view name, const char* name_ptr, uint64_t key, uint32_t processor) {
    // Same colors as getMarkerColor (TracyTTDevice.hpp): Tomato3 for PROFILER-keyword names, then the per-RISC palette.
    const uint32_t color =
        name.find("PROFILER") != std::string_view::npos ? tracy::Color::Tomato3 : kProcessorColor[processor];
    const std::string skey = fmt::format("{}#{:08x}", name, color);
    auto it = srclocs_.find(skey);
    if (it == srclocs_.end()) {
        char* nm_copy = new char[name.size() + 1];
        std::memcpy(nm_copy, name.data(), name.size());
        nm_copy[name.size()] = 0;
        auto* sl = new tracy::SourceLocationData{nm_copy, "kernel_profiler", "kernel_profiler", 0, color};
        it = srclocs_.emplace(skey, sl).first;
    }
    auto insert = [](std::vector<SrclocEntry>& table, const SrclocEntry& e) {
        const size_t mask = table.size() - 1;
        size_t i = srcloc_hash(e.name, e.key) & mask;
        while (table[i].name != nullptr) {
            i = (i + 1) & mask;
        }
        table[i] = e;
    };
    if ((srcloc_count_ + 1) * 2 > srcloc_table_.size()) {
        std::vector<SrclocEntry> old = std::move(srcloc_table_);
        srcloc_table_.assign(old.size() * 2, {});
        for (const SrclocEntry& e : old) {
            if (e.name != nullptr) {
                insert(srcloc_table_, e);
            }
        }
    }
    insert(srcloc_table_, {name_ptr, key, it->second});
    srcloc_count_++;
    return it->second;
}

void TracyConsumer::push_zone(const Core& core, std::string_view name, int64_t start_tsc, int64_t end_tsc) {
    const int64_t start_ns = std::max<int64_t>(timeline_ns(start_tsc), 0);
    const int64_t end_ns = std::max(timeline_ns(end_tsc), start_ns);
    const Lane ln = lane(core);
    TracyTTPushZone(
        ln.ctx,
        static_cast<const tracy::SourceLocationData*>(srcloc(name, ln.processor)),
        ln.thread,
        static_cast<uint64_t>(start_ns),
        static_cast<uint64_t>(end_ns));
}

void TracyConsumer::push_marker(
    const Core& core, std::string_view name, int64_t tsc, uint32_t runtime_id, std::span<const uint64_t> values) {
    const int64_t timestamp_ns = std::max<int64_t>(timeline_ns(tsc), 0);
    TracyTTCtx ctx = lane(core).ctx;
    tracy::TTDeviceMarker marker = device_marker(core);
    // Tracy's fallback color asserts on a RiscType it doesn't know.
    marker.color = kProcessorColor[static_cast<uint32_t>(core.processor)];
    marker.timestamp = static_cast<uint64_t>(timestamp_ns);
    marker.runtime_host_id = runtime_id;
    marker.marker_type = values.empty() ? tracy::TTDeviceMarkerType::EVENT : tracy::TTDeviceMarkerType::DATA;
    marker.marker_name = std::string(name);
    marker.file = "kernel_profiler";
    marker.line = 0;
    if (!values.empty()) {
        marker.data = values[0];
    }
    if (values.size() > 1) {
        marker.data_high = values[1];
    }
#ifdef TRACY_TT_HAS_FULL_DEPS
    for (size_t i = 2; i < values.size(); i++) {
        marker.meta_data[fmt::format("value{}", i)] = values[i];
    }
#endif
    TracyTTPushMarkerLockfree(ctx, marker);
}

}  // namespace tt::tt_metal::streaming_profiler

#endif
