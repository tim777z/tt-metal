// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "hostdev/streaming_profiler_common.h"
#include "internal/ethernet/eth_ptp.hpp"

namespace sampler {
namespace eth_ptp = tt::tt_metal::eth_ptp;
FORCE_INLINE uint32_t xorshift(uint32_t& state) {
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    return state;
}
// The branch predictor is off for a pass, since a branch taken on the previous pass would otherwise mispredict on this
// one.
FORCE_INLINE void branch_predictor(bool on) {
    if (on) {
        asm volatile("csrrci zero, 0x7c0, 2" ::: "memory");
    } else {
        asm volatile("csrrsi zero, 0x7c0, 2" ::: "memory");
    }
}
// One 64-byte-aligned copy of the pass serves both calibration and sampling, so calibration measures exactly the reads
// that sample, whatever code surrounds them.
__attribute__((noinline, aligned(64))) uint32_t pass(uint32_t (&walls)[3], uint32_t (&refclks)[4]) {
    uint32_t blocks[3][4];
    const auto read = [](uint32_t(&block)[4]) __attribute__((always_inline)) {
        block[1] = eth_ptp::kRefclkLo.read();
        block[2] = eth_ptp::kRefclkLo.read();
        block[0] = eth_ptp::kWallClockLo.read();
        block[3] = eth_ptp::kRefclkLo.read();
    };
    uint32_t step_after = 0;
    branch_predictor(false);
    read(blocks[2]);
    read(blocks[0]);
#pragma GCC unroll 20
    for (uint32_t k = 0; k < 20; k++) {
        uint32_t(&cur)[4] = blocks[(k + 1) % 3];
        const uint32_t(&prev)[4] = blocks[k % 3];
        const uint32_t(&prev2)[4] = blocks[(k + 2) % 3];
        read(cur);
        if (prev[3] != prev2[3]) {
            walls[0] = prev2[0];
            walls[1] = prev[0];
            walls[2] = cur[0];
            refclks[0] = prev2[3];
            refclks[1] = prev[1];
            refclks[2] = prev[2];
            refclks[3] = prev[3];
            step_after = k + 1;
            break;
        }
    }
    branch_predictor(true);
    return step_after;
}
// The tracker's range, 6, is its block length in cycles, so each pass starts at a uniform phase against the blocks. The
// ruler's range also covers the refclk's advance (20 ns, 16-27 cycles across the AICLK range); without that, the loop's
// own length phase-locks the passes to the refclk and each chip's updates settle on one AICLK cycle of the crossing.
template <uint32_t Range>
FORCE_INLINE uint32_t draw(uint32_t& walk) {
    static_assert(Range >= 1 && Range <= 64);
    return ((xorshift(walk) >> 16) * Range) >> 16;
}
// Runs `count` of Length nops, one cycle each. The stream has its own pad, whose target it computes once for all its
// passes.
template <uint32_t Length>
FORCE_INLINE void nops(uint32_t count) {
    asm volatile(
        ".option push\n\t.option norvc\n\t"
        "slli t1, %[count], 2\n\t"
        "auipc t0, 0\n\t"
        "addi t0, t0, 16 + %[n] * 4\n\t"
        "sub t0, t0, t1\n\t"
        "jr t0\n\t"
        ".rept %[n]\n\tnop\n\t.endr\n\t"
        ".option pop"
        :
        : [count] "r"(count), [n] "i"(Length)
        : "t0", "t1", "memory");
}
template <uint32_t Range>
FORCE_INLINE void pad(uint32_t& walk) {
    nops<Range>(draw<Range>(walk));
}
constexpr uint32_t kTrackerPadRange = 6;
constexpr uint32_t kRulerPadRange = 64;
constexpr int32_t kDropGap = -128;
constexpr uint32_t kPeriodCalibrationPasses = 1024;
constexpr uint32_t kMinCalibrationUpdates = 1u << 16;
constexpr uint32_t kMaxCalibrationUpdates = 1u << 26;
constexpr uint32_t kMaxScaledUpdates = 1u << 20;
// Each of the three gaps' positions in eighths, one signed byte per gap in one word, since the stream takes it in a
// register.
struct GapPos8 {
    uint32_t word = 0;
    static constexpr GapPos8 of(int32_t gap0, int32_t gap1, int32_t gap2) {
        return {
            (static_cast<uint32_t>(gap0) & 0xFFu) | (static_cast<uint32_t>(gap1) & 0xFFu) << 8 |
            (static_cast<uint32_t>(gap2) & 0xFFu) << 16};
    }
    FORCE_INLINE constexpr int32_t at(uint32_t gap) const { return static_cast<int8_t>(word >> (8 * gap)); }
};
struct Table {
    uint32_t period;
    GapPos8 pos8;
};
// Runs one pass and returns 1 if it wrote an update to slot. With period 0 an update is the length of the block after
// its step, which never follows the pad. Otherwise an update needs blocks period apart and a gap whose position isn't
// kDropGap, and holds the new refclk and the step's wall time in eighths: the block's wall read plus that position.
FORCE_INLINE uint32_t run(uint32_t period, GapPos8 pos8, volatile tt_l1_ptr uint32_t* slot, uint32_t& walk) {
    uint32_t walls[3], refclks[4];
    pad<kRulerPadRange>(walk);
    if (pass(walls, refclks) == 0) {
        return 0;
    }
    if (period == 0) {
        slot[0] = walls[2] - walls[1];
        return 1;
    }
    const uint32_t gap = refclks[1] != refclks[0] ? 0u : refclks[2] != refclks[1] ? 1u : 2u;
    const int32_t gap_pos8 = pos8.at(gap);
    if (walls[1] - walls[0] != period || walls[2] - walls[1] != period || gap_pos8 == kDropGap) {
        return 0;
    }
    slot[0] = refclks[gap + 1];
    slot[1] = (walls[1] << 3) + static_cast<uint32_t>(gap_pos8);
    return 1;
}
constexpr uint32_t kPeriodBins = 64;
constexpr uint32_t kControlPollMask = 1023u;
// `run_pass(period, pos8, slot)` runs one pass of the kernel's sampling reads and returns 1 if it wrote an update.
template <class RunPass>
Table calibrate(RunPass run_pass, volatile tt_l1_ptr kernel_profiler::RelayCtrl* ctl) {
    Table table{};
    static uint32_t hist[kPeriodBins];
    uint32_t slot_words[2];
    volatile tt_l1_ptr uint32_t* slot = slot_words;
    for (uint32_t i = 0; i < kPeriodCalibrationPasses; i++) {
        if (run_pass(0, GapPos8{}, slot) != 0) {
            hist[slot[0] & (kPeriodBins - 1u)]++;
        }
    }
    for (uint32_t i = 1; i < kPeriodBins; i++) {
        table.period = hist[i] > hist[table.period] ? i : table.period;
    }
    uint32_t updates_per_gap[3] = {}, total = 0;
    for (uint32_t i = 1; ctl->stop == 0u; i++) {
        // Gap p sits p eighths in, so an update's low three bits name its gap.
        if (run_pass(table.period, GapPos8::of(0, 1, 2), slot) != 0) {
            updates_per_gap[slot[1] & 7u]++;
            total++;
        }
        if ((i & kControlPollMask) == 0u) {
            ctl->heartbeat++;
            invalidate_l1_cache();
            if ((total >= kMinCalibrationUpdates && ctl->go != 0u) || total >= kMaxCalibrationUpdates) {
                break;
            }
        }
    }
    while (ctl->go == 0u && ctl->stop == 0u) {
        ctl->heartbeat++;
        invalidate_l1_cache();
    }
    uint32_t scale_shift = 0;
    while ((total >> scale_shift) >= kMaxScaledUpdates) {
        scale_shift++;
    }
    const uint32_t scaled_total = total >> scale_shift;
    if (scaled_total == 0) {
        return table;
    }
    int32_t width64[3];
    for (uint32_t gap = 0; gap < 3; gap++) {
        width64[gap] =
            static_cast<int32_t>((64u * table.period * (updates_per_gap[gap] >> scale_shift)) / scaled_total);
    }
    const int32_t r2_pos64 = -64;  // the read just before the wall read, in 64ths of a cycle from it
    const int32_t centre64[3] = {
        r2_pos64 - width64[1] - width64[0] / 2, r2_pos64 - width64[1] / 2, r2_pos64 + width64[2] / 2};
    table.pos8 = GapPos8::of((centre64[0] + 4) >> 3, (centre64[1] + 4) >> 3, (centre64[2] + 4) >> 3);
    return table;
}
}  // namespace sampler
