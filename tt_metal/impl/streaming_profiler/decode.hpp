// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <array>
#include <span>
#include <cstdint>
#include <vector>

#include <tt-metalium/experimental/streaming_profiler.hpp>

#include "impl/streaming_profiler/spsc_marker_decode.hpp"
#include "impl/streaming_profiler/capture_context.hpp"

namespace tt::tt_metal::streaming_profiler {

static_assert(sizeof(experimental::streaming_profiler::Record) == profiler::kSpscEventBytes);
static_assert(sizeof(experimental::streaming_profiler::PointRecord) == profiler::kSpscEventBytes);
static_assert(sizeof(experimental::streaming_profiler::Zone) == profiler::kSpscZoneBytes);
static_assert(sizeof(experimental::streaming_profiler::Event) == profiler::kSpscEventBytes);
static_assert(sizeof(experimental::streaming_profiler::TimestampedData) == profiler::kSpscDataBytes);
static_assert(std::is_trivially_copyable_v<experimental::streaming_profiler::Zone>);
static_assert(std::is_trivially_copyable_v<experimental::streaming_profiler::Event>);
static_assert(profiler::kSpscNRiscDecode == static_cast<size_t>(experimental::streaming_profiler::Processor::ERISC0));

// 64x64 covers every supported grid; a coordinate outside it is unknown.
struct CoreTable {
    static constexpr uint16_t kNone = 0xFFFF;
    static constexpr uint32_t kCoordBits = 6;
    static constexpr uint32_t kCoordMask = (1u << kCoordBits) - 1;
    static constexpr size_t kSlots = size_t{1} << (2 * kCoordBits);
    static constexpr uint32_t kOutsideGrid = ~((kCoordMask << 16) | kCoordMask);
    std::vector<uint16_t> slot = std::vector<uint16_t>(kSlots, kNone);
    static uint32_t index(uint32_t xy) { return (((xy >> 16) & kCoordMask) << kCoordBits) | (xy & kCoordMask); }
    uint16_t& operator[](uint32_t xy) { return slot[index(xy)]; }
    uint32_t find(uint32_t xy) const { return (xy & kOutsideGrid) != 0 ? kNone : slot[index(xy)]; }
    void load(std::span<const CaptureContext::Device::Tile> tiles) {
        slot.assign(kSlots, kNone);
        for (uint32_t core = 0; core < tiles.size(); core++) {
            (*this)[tiles[core].xy] = static_cast<uint16_t>(core);
        }
    }
};

struct SpscLane {
    // The producer guarantees the first zone after a launch or rewind is an absolute ZONE_ATOMIC; a resync recovers at
    // the next one.
    uint64_t cursor = 0;
    uint64_t last_ts = 0;  // lanes emit in end order, so a step back is a torn read
    // A regression repairs that record in place, so it is valid only while last_rec_seq is the current batch.
    uint8_t* last_rec = nullptr;
    uint64_t last_rec_seq = 0;
    uint32_t last_rec_zone = 0;
    uint32_t timer_hi = 0;
    uint32_t prog = 0;
    uint32_t seeded = 0;
    uint32_t need_state = 1;
    uint32_t need_anchor = 1;
};

struct StreamDecoder {
    uint64_t order_regressions = 0;

    struct Out {
        uint8_t* zones;
        uint8_t* events;
        uint8_t* data;
        uint8_t* values;
    };
    struct Produced {
        uint32_t zones, events, data, values;
        // A lane's tick plus its tile offset in the chip's eth wall domain, or INT64_MIN when it decoded none.
        int64_t newest_ticks;
        uint64_t stalls;
    };
    // A packet that becomes a record is at least two words, so each kind takes at most words / 2 records, plus the
    // whole quads the block kernels write past their last record. Values take two words each, plus per frame the 32
    // bytes a point kernel stores past its last value.
    struct Capacity {
        size_t zones, events, data, values;
    };
    static_assert(std::ranges::all_of(profiler::kFormats, [](const profiler::PacketFormat& format) {
        return format.kind == profiler::Kind::Sticky || format.words >= 2;
    }));
    static constexpr Capacity out_capacity(size_t words, uint32_t frames) {
        const size_t records = words / 2 + profiler::kSpscSinkSlackRecs;
        return Capacity{
            .zones = records * profiler::kSpscZoneBytes,
            .events = records * profiler::kSpscEventBytes,
            .data = records * profiler::kSpscDataBytes,
            .values = words * 4 + size_t{32} * frames};
    }

    void open(const CaptureContext::Device& dev);
    // Size `out` with out_capacity(). A later call may repair each lane's last record in place, so keep the output
    // writable and undelivered until commit().
    Produced decode_frames(const uint32_t* frames, std::span<const uint32_t> frame_words, Out out);

    void commit() { batch_seq_++; }

    const CoreTable& core_of_xy() const { return core_of_xy_; }

private:
    std::vector<SpscLane> lanes_;
    std::vector<profiler::SpscLaneConsts> consts_;
    // Per core, padded to one vector.
    std::vector<std::array<uint32_t, 8>> heads_;
    CoreTable core_of_xy_;
    std::vector<profiler::SpscRecConsts> rec_consts_;
    uint64_t batch_seq_ = 0;
};

}  // namespace tt::tt_metal::streaming_profiler
