// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// AICLK only takes whole FBDIV values, each a multiple of the crystal the refclk counts, so wherever it holds, the wall
// clock gains exactly k8/8 ticks per refclk tick, k8 = FBDIV * 8 / (REFDIV * postdiv0), and its samples lie on one
// line of that slope to within a sample's width.

#include <algorithm>
#include <atomic>
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "hostdev/streaming_profiler_common.h"
#include "tt_metal/impl/streaming_profiler/kernels/eth_sync_ring.hpp"

constexpr uint32_t kCtrlAddr = get_named_compile_time_arg_val("ctrl_addr");
constexpr uint32_t kSyncRingAddr = get_named_compile_time_arg_val("sync_ring_addr");
constexpr uint32_t kSampleRingAddr = get_named_compile_time_arg_val("sample_ring_addr");

namespace kp = kernel_profiler;

// A window becomes a point at least once a millisecond, well within a SyncLocalStep's 16-bit refclk field.
constexpr uint32_t kPointTicks = kp::kEthRefclkHz / 1000;
// A sample is placed to within half its read pair, at most 1.5 cycles, so a sample twice that far from the line is off
// it.
constexpr int32_t kOffEighths = 24;
// The sampler sees nearly every refclk update, one per 80 ns (4 ticks). A step off the line adds an eighth of a tick
// per update, so after eight samples (32 ticks) it has left the line. A sample only joins a window once the
// kHoldSamples samples after it are on the line too.
constexpr uint32_t kHoldSamples = 8;
// Off a line, every kGroupSamples consecutive samples (~320 ns, over which a clock walk bends the wall ~0.06 cycles)
// become one point, their centroid. A new line opens once kSteadyPoints such points (~4 us) fit one k8; a PLL step
// takes ~1.3 us, so it can't fit a line in that span.
constexpr uint32_t kGroupSamples = 4;
constexpr uint32_t kSteadyPoints = 12;
constexpr uint32_t kMaxWindowSamples = 1u << 23;

using Out = SyncRingWriter<kCtrlAddr, kSyncRingAddr>;
constexpr kp::SyncMeta kMetaLocal{.kind = kp::SyncKind::Local};

struct Anchor {
    uint64_t refclk = 0, wall8 = 0;
    uint32_t refclk_lo = 0, wall8_lo = 0;
    uint64_t widen_refclk(uint32_t lo) const {
        return refclk + static_cast<uint64_t>(static_cast<int64_t>(static_cast<int32_t>(lo - refclk_lo)));
    }
    uint64_t widen_wall8(uint32_t lo) const {
        return wall8 + static_cast<uint64_t>(static_cast<int64_t>(static_cast<int32_t>(lo - wall8_lo)));
    }
    void rebase(uint32_t new_refclk_lo, uint32_t new_wall8_lo) {
        refclk = widen_refclk(new_refclk_lo);
        wall8 = widen_wall8(new_wall8_lo);
        refclk_lo = new_refclk_lo;
        wall8_lo = new_wall8_lo;
    }
};

using Sample = kp::SyncSample;

struct Model {
    // The line the samples are on (k8 0 when there's none): through `at` at k8 eighths of a tick per refclk tick, with
    // the mean residue of its `count` samples so far.
    struct Line {
        uint32_t k8 = 0;
        Sample at{};
        int32_t mean = 0;
        int64_t sum = 0;
        uint32_t count = 0;
    };
    // The samples on the line since the last point. sum_residue is their residues minus the line's mean; only a window
    // close moves the mean, so it fits in 32 bits.
    struct Window {
        Sample first{};
        uint32_t sum_refclk_from_first = 0;
        int32_t sum_residue = 0;
        uint32_t count = 0, size = 1;
        uint32_t last_refclk = 0;
    };
    // A held sample's residue is recomputed when it joins the window, since a line change empties the hold.
    struct Hold {
        Sample samples[kHoldSamples];
        uint32_t begin = 0, count = 0;
    };
    // Off a line: the samples since the last point, and the last kSteadyPoints points.
    struct Group {
        Sample first{}, last{};
        uint32_t sum_refclk_from_first = 0, sum_wall8_from_first = 0, count = 0;
    };
    struct Recent {
        Sample points[kSteadyPoints];
        uint32_t count = 0, head = 0;
    };

    Line line;
    uint32_t slope = 0;
    Window window;
    Hold hold;
    Sample off_sample{};
    bool has_off = false;
    Group group;
    Recent recent;
    Anchor anchor;
    Out out;

    FORCE_INLINE int32_t residue(Sample sample) const {
        return static_cast<int32_t>((sample.wall8 - line.at.wall8) - line.k8 * (sample.refclk - line.at.refclk));
    }
    void point(uint32_t refclk, uint32_t wall8, uint32_t k8) {
        out.add(anchor.widen_refclk(refclk), anchor.widen_wall8(wall8), k8, slope, kMetaLocal);
    }
    __attribute__((noinline)) void close_window() {
        const uint32_t count = window.count;
        const uint32_t mean_refclk_from_first = (window.sum_refclk_from_first + count / 2u) / count;
        const int32_t half_count = static_cast<int32_t>(count / 2u);
        // The window's error against its first sample: the sum of each sample's residue minus the first's.
        const int32_t error_sum =
            window.sum_residue - static_cast<int32_t>(count) * (residue(window.first) - line.mean);
        const int32_t mean_error =
            (error_sum + (error_sum < 0 ? -half_count : half_count)) / static_cast<int32_t>(count);
        anchor.rebase(window.first.refclk, window.first.wall8);
        point(
            window.first.refclk + mean_refclk_from_first,
            window.first.wall8 + line.k8 * mean_refclk_from_first + static_cast<uint32_t>(mean_error),
            line.k8);
        line.sum += static_cast<int64_t>(window.sum_residue) + static_cast<int64_t>(count) * line.mean;
        line.count += count;
        line.mean = static_cast<int32_t>(line.sum / static_cast<int64_t>(line.count));
        window.count = 0;
        window.size = window.size < kMaxWindowSamples ? window.size * 2u : window.size;
    }
    FORCE_INLINE void window_add(Sample sample, int32_t sample_residue) {
        if (window.count == 0) {
            window.first = sample;
            window.sum_refclk_from_first = 0;
            window.sum_residue = 0;
        }
        const uint32_t refclk_from_first = sample.refclk - window.first.refclk;
        window.sum_refclk_from_first += refclk_from_first;
        window.sum_residue += sample_residue - line.mean;
        window.last_refclk = sample.refclk;
        if (++window.count == window.size || refclk_from_first >= kPointTicks) {
            close_window();
        }
    }
    __attribute__((noinline)) void close_group() {
        Group& closing = group;
        const uint32_t span = closing.last.refclk - closing.first.refclk;
        if (span != 0) {
            slope = ((closing.last.wall8 - closing.first.wall8) + span / 2u) / span;
        }
        // Every group but the capture's last closes full, so the division is nearly always by the constant.
        const auto divide = [&](uint32_t sum) {
            return closing.count == kGroupSamples ? sum / kGroupSamples : sum / closing.count;
        };
        const uint32_t mean_refclk_from_first = divide(closing.sum_refclk_from_first),
                       refclk_remainder = closing.sum_refclk_from_first - mean_refclk_from_first * closing.count;
        const uint32_t mean_wall8_from_first =
            divide(closing.sum_wall8_from_first - slope * refclk_remainder + closing.count / 2u);
        anchor.rebase(closing.first.refclk, closing.first.wall8);
        const Sample centroid{
            closing.first.refclk + mean_refclk_from_first, closing.first.wall8 + mean_wall8_from_first};
        point(centroid.refclk, centroid.wall8, 0);
        recent.points[(recent.head + recent.count) % kSteadyPoints] = centroid;
        if (recent.count < kSteadyPoints) {
            recent.count++;
        } else {
            recent.head = (recent.head + 1) % kSteadyPoints;
        }
        closing.count = 0;
        if (recent.count == kSteadyPoints) {
            try_line();
        }
    }
    FORCE_INLINE void group_add(Sample sample) {
        if (group.count == 0) {
            group.first = sample;
            group.sum_refclk_from_first = 0;
            group.sum_wall8_from_first = 0;
        }
        group.sum_refclk_from_first += sample.refclk - group.first.refclk;
        group.sum_wall8_from_first += sample.wall8 - group.first.wall8;
        group.last = sample;
        if (++group.count == kGroupSamples) {
            close_group();
        }
    }
    __attribute__((noinline)) void try_line() {
        const Sample oldest = recent.points[recent.head];
        const Sample& newest = recent.points[(recent.head + kSteadyPoints - 1) % kSteadyPoints];
        const uint32_t span = newest.refclk - oldest.refclk;
        if (span == 0) {
            return;
        }
        const uint32_t k8 = ((newest.wall8 - oldest.wall8) + span / 2u) / span;
        int32_t residues[kSteadyPoints];
        int32_t sum = 0;
        for (uint32_t i = 0; i < kSteadyPoints; i++) {
            const Sample& centroid = recent.points[(recent.head + i) % kSteadyPoints];
            residues[i] =
                static_cast<int32_t>((centroid.wall8 - oldest.wall8) - k8 * (centroid.refclk - oldest.refclk));
            sum += residues[i];
        }
        const int32_t mean = sum / static_cast<int32_t>(kSteadyPoints);
        for (uint32_t i = 0; i < kSteadyPoints; i++) {
            if (static_cast<uint32_t>(residues[i] - mean + kOffEighths) > 2u * kOffEighths) {
                return;
            }
        }
        line.k8 = k8;
        line.at = oldest;
        line.mean = mean;
        line.sum = sum;
        line.count = kSteadyPoints;
        slope = k8;
        window.count = 0;
        window.size = 1;
        hold.count = 0;
        has_off = false;
    }
    // End the line with a point where its samples last held it. Otherwise the chord from the last window's centroid, up
    // to half a window back, into the first group would leave the line well before the clock did.
    __attribute__((noinline)) void end_line(Sample sample) {
        if (window.count != 0) {
            close_window();
        }
        if (line.count > kSteadyPoints) {
            const uint32_t last_refclk = window.last_refclk;
            point(
                last_refclk,
                line.at.wall8 + line.k8 * (last_refclk - line.at.refclk) + static_cast<uint32_t>(line.mean),
                line.k8);
        }
        line.k8 = 0;
        recent.count = 0;
        recent.head = 0;
        group.count = 0;
        for (uint32_t i = 0; i < hold.count; i++) {
            group_add(hold.samples[(hold.begin + i) % kHoldSamples]);
        }
        hold.count = 0;
        group_add(off_sample);
        group_add(sample);
        has_off = false;
    }
    FORCE_INLINE void on_sample(Sample sample) {
        if (line.k8 == 0) {
            group_add(sample);
            return;
        }
        const int32_t sample_residue = residue(sample);
        if (static_cast<uint32_t>(sample_residue - line.mean + kOffEighths) > 2u * kOffEighths) {
            if (has_off) {
                end_line(sample);
                return;
            }
            off_sample = sample;
            has_off = true;
            return;
        }
        has_off = false;
        if (hold.count == kHoldSamples) {
            window_add(hold.samples[hold.begin], residue(hold.samples[hold.begin]));
            hold.begin = (hold.begin + 1) % kHoldSamples;
            hold.count--;
        }
        hold.samples[(hold.begin + hold.count) % kHoldSamples] = sample;
        hold.count++;
    }
    // Feeds the samples [cursor, end) in ring order. On a line, with a full hold and nothing pending off it, the state
    // a sample touches stays in registers; anything else goes through on_sample.
    void feed(const volatile tt_l1_ptr Sample* cursor, const volatile tt_l1_ptr Sample* end) {
        const auto load = [](const volatile tt_l1_ptr Sample* slot) { return Sample{slot->refclk, slot->wall8}; };
        while (cursor != end) {
            if (line.k8 == 0 || has_off || hold.count != kHoldSamples) {
                on_sample(load(cursor));
                cursor++;
                continue;
            }
            const uint32_t k8 = line.k8, line_refclk = line.at.refclk, line_wall8 = line.at.wall8;
            int32_t mean = line.mean, window_sum_residue = window.sum_residue;
            uint32_t window_first_refclk = window.first.refclk,
                     window_sum_refclk_from_first = window.sum_refclk_from_first, window_count = window.count,
                     window_size = window.size, hold_begin = hold.begin, window_last_refclk = window.last_refclk;
            bool left = false;
            // Load each sample's words and hold slot an iteration ahead, so their latency hides behind the math. The
            // one read past the end lands in L1 and is never used.
            Sample sample = load(cursor), held = hold.samples[hold_begin];
            for (; cursor != end; cursor++) {
                const Sample next_sample = load(cursor + 1);
                const uint32_t next_hold_begin = (hold_begin + 1) % kHoldSamples;
                const Sample next_held = hold.samples[next_hold_begin];
                const int32_t sample_residue =
                    static_cast<int32_t>((sample.wall8 - line_wall8) - k8 * (sample.refclk - line_refclk));
                if (static_cast<uint32_t>(sample_residue - mean + kOffEighths) > 2u * kOffEighths) {
                    left = true;
                    break;
                }
                hold.samples[hold_begin] = sample;
                hold_begin = next_hold_begin;
                if (window_count == 0) {
                    window_first_refclk = held.refclk;
                    window.first.wall8 = held.wall8;
                    window_sum_refclk_from_first = 0;
                    window_sum_residue = 0;
                }
                const uint32_t refclk_from_first = held.refclk - window_first_refclk;
                window_sum_refclk_from_first += refclk_from_first;
                window_sum_residue +=
                    static_cast<int32_t>((held.wall8 - line_wall8) - k8 * (held.refclk - line_refclk)) - mean;
                window_last_refclk = held.refclk;
                if (++window_count == window_size || refclk_from_first >= kPointTicks) {
                    window.first.refclk = window_first_refclk;
                    window.sum_refclk_from_first = window_sum_refclk_from_first;
                    window.sum_residue = window_sum_residue;
                    window.count = window_count;
                    window.last_refclk = window_last_refclk;
                    close_window();
                    window_count = window.count;
                    window_size = window.size;
                    mean = line.mean;
                }
                sample = next_sample;
                held = next_held;
            }
            window.first.refclk = window_first_refclk;
            window.sum_refclk_from_first = window_sum_refclk_from_first;
            window.sum_residue = window_sum_residue;
            window.count = window_count;
            hold.begin = hold_begin;
            window.last_refclk = window_last_refclk;
            if (left) {
                on_sample(load(cursor));
                cursor++;
            }
        }
    }
    void finish() {
        if (line.k8 != 0) {
            for (uint32_t i = 0; i < hold.count; i++) {
                const Sample& sample = hold.samples[(hold.begin + i) % kHoldSamples];
                window_add(sample, residue(sample));
            }
            hold.count = 0;
            if (window.count != 0) {
                close_window();
            }
        } else if (group.count != 0) {
            close_group();
        }
    }
};

void kernel_main() {
    volatile tt_l1_ptr kp::RelayCtrl* ctrl = Out::ctrl();
    volatile tt_l1_ptr kp::SyncSampleRing* ring =
        reinterpret_cast<volatile tt_l1_ptr kp::SyncSampleRing*>(kSampleRingAddr);
    Model model;
    uint32_t next = 0;
    do {
        invalidate_l1_cache();
    } while (ring->tail == 0 && ring->done == 0);
    invalidate_l1_cache();
    const uint64_t base_refclk = ring->refclk, base_wall8 = ring->wall8;
    model.anchor = Anchor{
        .refclk = base_refclk,
        .wall8 = base_wall8,
        .refclk_lo = static_cast<uint32_t>(base_refclk),
        .wall8_lo = static_cast<uint32_t>(base_wall8)};
    constexpr uint32_t kChunkSamples = 64, kRingSamples = kp::kSyncSampleRingSamples;
    uint32_t stop = 0;
    while (true) {
        invalidate_l1_cache();
        const bool done = ring->done != 0;
        stop = ctrl->stop != 0u ? kp::kSyncHeadStop : stop;
        const uint32_t tail = ring->tail;
        const uint32_t end = next + std::min(tail - next, kChunkSamples);
        const uint32_t next_slot = next % kRingSamples;
        const uint32_t before_wrap = std::min(end - next, kRingSamples - next_slot);
        model.feed(&ring->samples[next_slot], &ring->samples[next_slot + before_wrap]);
        model.feed(&ring->samples[0], &ring->samples[end - next - before_wrap]);
        next = end;
        std::atomic_thread_fence(std::memory_order_release);
        ring->head = next + stop;
        if (done && next == tail) {
            break;
        }
    }
    model.finish();
    model.out.close();
}
