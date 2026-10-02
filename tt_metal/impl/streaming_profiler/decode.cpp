// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "impl/streaming_profiler/decode.hpp"

#include <algorithm>
#include <bit>
#include <cstring>
#include <limits>

#include <tt_stl/assert.hpp>

namespace tt::tt_metal::streaming_profiler {

namespace {

profiler::SpscRecConsts record_consts(const experimental::streaming_profiler::Core& core, int64_t offset) {
    TT_FATAL(
        core.logical.x < 256 && core.logical.y < 256 && core.physical.x < 256 && core.physical.y < 256 &&
            core.chip_id < 65536,
        "streaming profiler: core {} does not fit a record",
        core.physical.str());
    return profiler::SpscRecConsts{
        .coords =
            {static_cast<uint32_t>(core.logical.x) | (static_cast<uint32_t>(core.logical.y) << 8) |
                 (static_cast<uint32_t>(core.physical.x) << 16) | (static_cast<uint32_t>(core.physical.y) << 24),
             core.chip_id | (static_cast<uint32_t>(core.processor) << 16)},
        .offset = static_cast<uint64_t>(offset)};
}

constexpr uint64_t kEpoch = 1ull << 32;

// A wall-clock read is either right or exactly 2^32 high (kernel_profiler_streaming.hpp read_wall_clock), so the
// previous record borrowed the next epoch.
// noinline: inlined at its sites, the repair body raises the walk's register pressure.
__attribute__((noinline)) bool repair_prev_record(uint8_t* rec_start, bool zone, uint64_t prev_ts, uint64_t next_ts) {
    if (prev_ts < kEpoch || prev_ts - next_ts > kEpoch || rec_start == nullptr) {
        return false;
    }
    uint64_t* rec = reinterpret_cast<uint64_t*>(rec_start);
    if (zone && rec[profiler::kSpscQwDuration] >= kEpoch) {
        rec[profiler::kSpscQwDuration] -= kEpoch;
    } else if (rec[profiler::kSpscQwTimestamp] >= kEpoch) {
        rec[profiler::kSpscQwTimestamp] -= kEpoch;
    } else {
        return false;
    }
    return true;
}

// A dur_hi zone's end may legitimately precede the previous record's when that record is the stall zone raised by
// this zone's own ring reservation.
template <profiler::PacketFormat F>
__attribute__((noinline)) uint64_t
run_repairs(const uint32_t* src, uint32_t n, uint8_t* first, uint64_t th_hi, uint64_t ts0, const SpscLane& lane) {
    constexpr uint32_t kRecBytes = profiler::spsc_rec_bytes<F>;
    uint64_t order_regressions = 0;
    if constexpr (F.has_dur_hi()) {
        for (uint32_t k = 0; k < n; k++) {
            const uint32_t* packet = src + F.words * k;
            if (packet[F.dur_hi] != 0xFFFFFFFFu) {
                continue;
            }
            const uint64_t end = profiler::spsc_ts_at<F>(src, k, th_hi);
            const uint64_t dur = (static_cast<uint64_t>(packet[F.dur_hi]) << 32) | packet[F.dur_lo];
            if (end >= dur + kEpoch) {
                uint64_t* rec = reinterpret_cast<uint64_t*>(first + kRecBytes * k);
                rec[profiler::kSpscQwDuration] = dur + kEpoch;
                rec[profiler::kSpscQwTimestamp] = end - rec[profiler::kSpscQwDuration];
            } else {
                order_regressions++;
            }
        }
    }
    const auto step = [&](uint64_t prev_ts, uint64_t next_ts, uint8_t* prev_rec, bool prev_zone) {
        if (!(next_ts < prev_ts)) {
            return;
        }
        if constexpr (F.has_dur_hi()) {
            const uint64_t* prev = reinterpret_cast<const uint64_t*>(prev_rec);
            if (prev != nullptr &&
                (static_cast<uint32_t>(prev[profiler::kSpscQwIds]) & PP_LOW27_MASK) == profiler::kSpscStallZoneId) {
                order_regressions++;
                return;
            }
        }
        const bool fixed = repair_prev_record(prev_rec, prev_zone, prev_ts, next_ts);
        order_regressions += !fixed;
    };
    step(lane.last_ts, ts0, lane.last_rec, lane.last_rec_zone != 0);
    if constexpr (!F.delta16) {
        for (uint32_t k = 1; k < n; k++) {
            step(
                profiler::spsc_ts_at<F>(src, k - 1, th_hi),
                profiler::spsc_ts_at<F>(src, k, th_hi),
                first + kRecBytes * (k - 1),
                F.kind == profiler::Kind::Zone);
        }
    }
    return order_regressions;
}

template <profiler::PacketFormat F>
constexpr uint32_t sticky_value(const uint32_t* src) {
    static_assert(F.kind == profiler::Kind::Sticky);
    return F.value_word == 0 ? src[0] & PP_LOW27_MASK : src[F.value_word];
}

}  // namespace

void StreamDecoder::open(const CaptureContext::Device& dev) {
    const size_t num_cores = dev.lanes.size() / profiler::kSpscNRiscDecode;
    lanes_.assign(num_cores * profiler::kSpscNRiscDecode, {});
    consts_.assign(lanes_.size(), {});
    heads_.assign(num_cores, {});
    core_of_xy_.load(dev.tiles);
    rec_consts_.reserve(dev.lanes.size());
    for (size_t lane = 0; lane < dev.lanes.size(); lane++) {
        rec_consts_.push_back(
            record_consts(dev.lanes[lane], dev.tiles[lane / profiler::kSpscNRiscDecode].clock_offset));
    }
}

// Decoding starts at whichever is later, the head mirror or the start of the extent. The mirror falls behind after an
// upstream loss, which the decoder accepts, and the extent falls behind after a late head write-back, whose overlap it
// skips.
StreamDecoder::Produced StreamDecoder::decode_frames(
    const uint32_t* frames, std::span<const uint32_t> frame_words, Out out) {
    namespace kp = kernel_profiler;
    using namespace profiler;
    // The kernels store through byte pointers, which would force a member reached through `this` to be reloaded after
    // every store.
    const SpscRecConsts* const lane_consts = rec_consts_.data();
    const uint64_t seq = batch_seq_;
    uint8_t* const zones_out = out.zones;
    uint8_t* const events_out = out.events;
    uint8_t* const data_out = out.data;
    uint64_t* const values_out = reinterpret_cast<uint64_t*>(out.values);
    uint64_t zone_bytes = 0;
    uint64_t value_count_out = 0;
    // Both point offsets share one register, data's in the high half. An array indexed by packet kind would live in
    // memory and put every update on a store-to-load chain.
    uint64_t point_offsets = 0;
    uint64_t stalls = 0;
    uint64_t unrepaired = 0;
    int64_t newest_ticks = std::numeric_limits<int64_t>::min();

    const uint32_t* frame = frames;
    for (const uint32_t frame_len : frame_words) {
        const uint32_t* ctrl = frame + kp::SPSC_SPAN_PREFIX_WORDS;
        const uint32_t core = core_of_xy_.find(frame[kp::SPSC_PREFIX_XY]);
        TT_FATAL(
            core != CoreTable::kNone,
            "streaming profiler: frame from NoC core {:#x}, which the capture did not seed",
            frame[kp::SPSC_PREFIX_XY]);
        uint32_t* const heads = heads_[core].data();
        SpscLane* const core_lanes = lanes_.data() + size_t{core} * kSpscNRiscDecode;
        SpscLaneConsts* const core_consts = consts_.data() + size_t{core} * kSpscNRiscDecode;
        const simde__m256i tail_v =
            simde_mm256_loadu_si256(reinterpret_cast<const simde__m256i*>(ctrl + kp::SPSC_WIRE_TAIL_0));
        const simde__m256i idle = simde_mm256_and_si256(
            simde_mm256_cmpeq_epi32(
                tail_v, simde_mm256_loadu_si256(reinterpret_cast<const simde__m256i*>(frame + kp::SPSC_PREFIX_HEAD_0))),
            simde_mm256_cmpeq_epi32(tail_v, simde_mm256_loadu_si256(reinterpret_cast<const simde__m256i*>(heads))));
        uint32_t busy = ~static_cast<uint32_t>(simde_mm256_movemask_ps(simde_mm256_castsi256_ps(idle))) &
                        ((1u << kSpscNRiscDecode) - 1u);
        uint64_t frame_ts = 0;
        uint32_t word_offset = kp::SPSC_SPAN_PREFIX_WORDS + kp::SPSC_SPAN_WIRE_CTRL_WORDS;
        for (; busy != 0; busy &= busy - 1) {
            const uint32_t risc = static_cast<uint32_t>(std::countr_zero(busy));
            const uint32_t lane_index = core * kSpscNRiscDecode + risc;
            SpscLane& lane = core_lanes[risc];
            const uint32_t tail = ctrl[kp::SPSC_WIRE_TAIL_0 + risc];
            const uint32_t start = frame[kp::SPSC_PREFIX_HEAD_0 + risc];
            const uint32_t extent = tail - start;
            const uint32_t* words = nullptr;
            // A nearly full run that wraps arrives as the whole ring image, so its pad is phased for ring offset 0 and
            // its payload advances by the full ring.
            const bool ring_ordered = extent != 0 && kp::spsc_span_wrap_image(start, extent, kSpscRingCap);
            if (extent != 0) {
                word_offset += kp::spsc_span_pack_pad(ring_ordered ? 0u : start, word_offset);
                words = frame + word_offset;
                word_offset += ring_ordered ? kSpscRingCap : extent;
                TT_FATAL(
                    word_offset <= frame_len,
                    "streaming profiler: frame control block places lane {} {} words past the frame's {}",
                    lane_index,
                    word_offset - frame_len,
                    frame_len);
            }
            uint32_t head;
            if (lane.seeded == 0) {
                lane.seeded = 1;
                head = start;
            } else {
                head = heads[risc];
                if (static_cast<int32_t>(start - head) > 0) {
                    head = start;
                    lane.need_state = 1;
                }
            }
            heads[risc] = tail;
            const uint32_t run = tail - head;
            if (run == 0) {
                continue;
            }
            SpscLaneConsts& consts = core_consts[risc];
            uint32_t& timer_hi = lane.timer_hi;
            uint32_t& prog = lane.prog;
            uint64_t& cursor = lane.cursor;
            uint32_t linear[kSpscRingCap];
            if (ring_ordered) {
                const uint32_t ring_head = head & kSpscRingMask;
                const uint32_t first = kSpscRingCap - ring_head < run ? kSpscRingCap - ring_head : run;
                std::memcpy(linear, words + ring_head, first * sizeof(uint32_t));
                if (first < run) {
                    std::memcpy(linear + first, words, (run - first) * sizeof(uint32_t));
                }
                words = linear;
            } else {
                words += extent - run;
            }
            const uint32_t* const readable_end = ring_ordered ? words + run : frame + frame_len;
            uint32_t pos = 0;
            uint32_t& awaiting_anchor = lane.need_anchor;
            if (lane.need_state) {
                // The slots hold the state at the frame's tail. A sticky packet inside the run means the words before
                // it ran on an earlier value that isn't recorded here, so the run is picked up from its last sticky.
                timer_hi = ctrl[kp::SPSC_WIRE_TIMER_0 + risc];
                prog = ctrl[kp::spsc_wire_prog_word(risc)];
                uint32_t k = 0;
                while (k < run) {
                    const uint32_t type = pp_type(words[k]);
                    uint32_t packet_words = kSpscWordsOfType[type];
                    if (packet_words == 0) {
                        break;
                    }
                    if (kSpscDataMaskOfType[type] != 0) {
                        if (k + kSpscDataFormat.size_word >= run) {
                            break;
                        }
                        packet_words += (words[k + kSpscDataFormat.size_word] >> kSpscDataFormat.size_shift) &
                                        kSpscDataFormat.size_mask;
                    }
                    spsc_for_format<spsc_is_sticky>(type, [&]<PacketFormat F>() __attribute__((always_inline)) {
                        if (k + F.words <= run) {
                            (F.sets == PacketFormat::Sets::TimerHi ? timer_hi : prog) = sticky_value<F>(words + k);
                        }
                        pos = k + packet_words;
                    });
                    k += packet_words;
                }
                lane.need_state = 0;
                awaiting_anchor = 1;
                spsc_lane_consts(consts, lane_consts[lane_index], timer_hi, prog);
            }
            uint64_t& lane_ts = lane.last_ts;
            if (lane.last_rec_seq != seq) {
                lane.last_rec = nullptr;
                lane.last_rec_seq = seq;
            }
            uint8_t*& lane_rec = lane.last_rec;
            uint32_t& lane_rec_zone = lane.last_rec_zone;
            const auto zones_emitted = [&](uint32_t n) {
                zone_bytes += kSpscZoneBytes * n;
                lane_rec = zones_out + zone_bytes - kSpscZoneBytes;
                lane_rec_zone = true;
            };
            while (pos < run) {
                const uint32_t* const src = words + pos;
                const uint32_t type = pp_type(src[0]);
                const uint32_t readable = static_cast<uint32_t>(readable_end - src);
                const uint32_t left = run - pos;
                uint32_t got = 0;
                if ((kSpscPointTypes >> type) & 1u) {
                    // One branch for all of them, so random alternation between them doesn't mispredict.
                    bool blocked = false;
                    spsc_for_each_format<spsc_is_point>([&]<PacketFormat F>() __attribute__((always_inline)) {
                        if (!blocked && left >= 4u * F.words && readable >= 8u && spsc_run4<F>(src)) {
                            blocked = true;
                            uint8_t* const first = events_out + static_cast<uint32_t>(point_offsets);
                            const auto block = spsc_block<F>(src, readable, left / F.words, cursor, consts, first);
                            point_offsets += kSpscEventBytes * block.n;
                            if (block.n != 0) {
                                const uint64_t ts0 = spsc_ts_at<F>(src, 0, consts.th_hi);
                                if (__builtin_expect(ts0 < lane_ts || block.regress != 0, 0)) {
                                    unrepaired += run_repairs<F>(src, block.n, first, consts.th_hi, ts0, lane);
                                }
                                lane_ts = block.ts_last;
                                lane_rec = events_out + static_cast<uint32_t>(point_offsets) - kSpscEventBytes;
                                lane_rec_zone = false;
                                got = F.words * block.n;
                            }
                        }
                    });
                    if (!blocked) {
                        const uint32_t data_mask = kSpscDataMaskOfType[type];
                        const uint32_t size_word = std::min<uint32_t>(kSpscDataFormat.size_word, readable - 1u);
                        const uint32_t value_count =
                            ((src[size_word] >> kSpscDataFormat.size_shift) & kSpscDataFormat.size_mask) & data_mask;
                        const uint32_t packet_words = kSpscWordsOfType[type] + value_count;
                        if (left >= packet_words) {
                            const uint64_t timestamp = consts.th_hi | src[kSpscDataFormat.ts_lo];
                            const uint32_t half_shift = data_mask & 32u;
                            uint8_t* const dst = (data_mask ? data_out : events_out) +
                                                 static_cast<uint32_t>(point_offsets >> half_shift);
                            if (__builtin_expect(timestamp < lane_ts, 0)) {
                                unrepaired += run_repairs<kSpscDataFormat>(src, 1, dst, consts.th_hi, timestamp, lane);
                            }
                            lane_ts = timestamp;
                            value_count_out +=
                                spsc_point(src, readable, value_count, consts, dst, values_out + value_count_out);
                            point_offsets += static_cast<uint64_t>(
                                                 kSpscEventBytes + ((kSpscDataBytes - kSpscEventBytes) & data_mask))
                                             << half_shift;
                            lane_rec = dst;
                            lane_rec_zone = false;
                            got = packet_words;
                        }
                    }
                } else {
                    spsc_for_format<spsc_is_zone_or_sticky>(type, [&]<PacketFormat F>() __attribute__((always_inline)) {
                        if (left < F.words) {
                            return;
                        }
                        if constexpr (F.kind == Kind::Sticky) {
                            const uint32_t value = sticky_value<F>(src);
                            if constexpr (F.sets == PacketFormat::Sets::TimerHi) {
                                timer_hi = value;
                                spsc_lane_consts_th(consts, timer_hi);
                            } else {
                                prog = value;
                                spsc_lane_consts_prog(consts, prog);
                            }
                            got = F.words;
                        } else if constexpr (F.delta16) {
                            const bool long_run = left >= 4u * F.words && pp_type(src[3u * F.words]) == F.type;
                            const auto block =
                                long_run ? spsc_block<F>(
                                               src, readable, left / F.words, cursor, consts, zones_out + zone_bytes)
                                         : spsc_delta16_short<F>(
                                               src, left / F.words, cursor, consts, zones_out + zone_bytes);
                            if (block.n != 0) {
                                // Without a cursor the run decodes into the scratch space it would have used and isn't
                                // counted.
                                if (!awaiting_anchor) {
                                    const uint64_t ts0 = cursor + (src[1] >> 16);
                                    if (__builtin_expect(ts0 < lane_ts, 0)) {
                                        unrepaired += run_repairs<F>(
                                            src, block.n, zones_out + zone_bytes, consts.th_hi, ts0, lane);
                                    }
                                    lane_ts = block.ts_last;
                                    zones_emitted(block.n);
                                }
                                cursor = block.ts_last;
                                got = F.words * block.n;
                            }
                        } else {
                            uint8_t* const first = zones_out + zone_bytes;
                            const bool same_type_follows = left > F.words && pp_type(src[F.words]) == F.type;
                            const SpscBlockResult block =
                                same_type_follows ? spsc_block<F>(src, readable, left / F.words, cursor, consts, first)
                                                  : spsc_one<F>(src, readable, consts, first);
                            if (block.n == 0) {
                                return;
                            }
                            stalls += block.stalls;
                            const uint64_t ts0 = spsc_ts_at<F>(src, 0, consts.th_hi);
                            if (__builtin_expect(
                                    (F.has_dur_hi() && block.wrapped != 0) || ts0 < lane_ts || block.regress != 0, 0)) {
                                unrepaired += run_repairs<F>(src, block.n, first, consts.th_hi, ts0, lane);
                            }
                            lane_ts = block.ts_last;
                            zones_emitted(block.n);
                            if constexpr (F.reanchor) {
                                cursor = block.ts_last;
                                awaiting_anchor = 0;
                            }
                            got = F.words * block.n;
                        }
                    });
                }
                TT_FATAL(
                    got != 0,
                    "streaming profiler: undecodable word {:#010x} at offset {} of lane {}'s run of {}",
                    src[0],
                    pos,
                    lane_index,
                    run);
                pos += got;
            }
            frame_ts = std::max(frame_ts, lane_ts);
        }
        if (frame_ts != 0) {
            newest_ticks = std::max(
                newest_ticks,
                static_cast<int64_t>(frame_ts) + static_cast<int64_t>(lane_consts[core * kSpscNRiscDecode].offset));
        }
        frame += frame_len;
    }
    order_regressions += unrepaired;
    return Produced{
        static_cast<uint32_t>(zone_bytes / kSpscZoneBytes),
        static_cast<uint32_t>(static_cast<uint32_t>(point_offsets) / kSpscEventBytes),
        static_cast<uint32_t>((point_offsets >> 32) / kSpscDataBytes),
        static_cast<uint32_t>(value_count_out),
        newest_ticks,
        stalls};
}

}  // namespace tt::tt_metal::streaming_profiler
