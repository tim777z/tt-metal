// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Placement error peaks within a few microseconds of a clock change, so every update near one is kept. Elsewhere only
// one in kSyncRulerKeepEvery is, and the host weights those by that.

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "hostdev/streaming_profiler_common.h"
#include "tt_metal/impl/streaming_profiler/kernels/eth_clock_sampling.hpp"
#include "tt_metal/impl/streaming_profiler/kernels/eth_sync_ring.hpp"

constexpr uint32_t kCtrlAddr = get_named_compile_time_arg_val("ctrl_addr");
constexpr uint32_t kSyncRingAddr = get_named_compile_time_arg_val("sync_ring_addr");

namespace kp = kernel_profiler;
namespace eth_ptp = tt::tt_metal::eth_ptp;

constexpr uint32_t kHistory = 8;      // ~9 us of updates held back, so a change's run-up goes out dense
constexpr uint32_t kDenseAfter = 64;  // ~70 us of updates at kHistory's rate go out dense after a change
constexpr int64_t kOffLine8 = 16;     // two ticks: a steady update sits within a few eighths of the line
constexpr uint32_t kStopPollMask = 255u;

using Out = SyncRingWriter<kCtrlAddr, kSyncRingAddr>;
constexpr kp::SyncMeta kMetaDense{.dense = 1, .kind = kp::SyncKind::Ruler};
constexpr kp::SyncMeta kMetaThin{.kind = kp::SyncKind::Ruler};

void kernel_main() {
    volatile tt_l1_ptr kp::RelayCtrl* ctrl = Out::ctrl();
    uint32_t calibration_walk = eth_ptp::kWallClockLo.read() | 1u;
    sampler::Table table = sampler::calibrate(
        [&](uint32_t period, sampler::GapPos8 pos8, volatile tt_l1_ptr uint32_t* slot) {
            return sampler::run(period, pos8, slot, calibration_walk);
        },
        ctrl);
    // Only the adjacent pair's gap is kept.
    table.pos8 = sampler::GapPos8::of(sampler::kDropGap, table.pos8.at(1), sampler::kDropGap);
    Out out;
    struct Held {
        uint64_t refclk, wall8;
    };
    Held held[kHistory];
    uint32_t held_count = 0, held_next = 0, thin = 0, dense_left = 0;
    int64_t k8 = 0;
    const auto release = [&](const Held& update) {
        if (dense_left != 0) {
            out.add(update.refclk, update.wall8, 0, static_cast<uint32_t>(k8), kMetaDense);
        } else if (++thin == kp::kSyncRulerKeepEvery) {
            thin = 0;
            out.add(update.refclk, update.wall8, 0, static_cast<uint32_t>(k8), kMetaThin);
        }
    };
    const auto release_held = [&] {
        for (; held_count != 0; held_count--) {
            release(held[(held_next + kHistory - held_count) % kHistory]);
        }
    };
    if (ctrl->stop == 0u) {
        const eth_ptp::Instant start = eth_ptp::read_instant();
        uint64_t refclk = start.refclk, wall8 = start.wall() << 3;
        uint32_t refclk_lo = static_cast<uint32_t>(refclk), wall8_lo = static_cast<uint32_t>(wall8);
        uint32_t iter = 0, walk = start.wall_lo | 1u;
        while (true) {
            uint32_t update[2];
            if (sampler::run(table.period, table.pos8, update, walk) != 0) {
                refclk += update[0] - refclk_lo;
                wall8 += update[1] - wall8_lo;
                refclk_lo = update[0];
                wall8_lo = update[1];
                if (held_count != 0) {
                    const Held& prev = held[(held_next + kHistory - 1) % kHistory];
                    const auto refclk_step = static_cast<int64_t>(refclk - prev.refclk),
                               wall8_step = static_cast<int64_t>(wall8 - prev.wall8);
                    const int64_t off_line8 = wall8_step - k8 * refclk_step;
                    if (k8 == 0 || off_line8 > kOffLine8 || off_line8 < -kOffLine8) {
                        if (k8 != 0) {
                            dense_left = kDenseAfter;
                            release_held();
                        }
                        k8 = refclk_step > 0 ? (wall8_step + refclk_step / 2) / refclk_step : 0;
                    }
                }
                if (held_count == kHistory) {
                    release(held[held_next]);
                    held_count--;
                }
                held[held_next] = {refclk, wall8};
                held_next = (held_next + 1) % kHistory;
                held_count++;
                if (dense_left != 0) {
                    dense_left--;
                }
            }
            if ((++iter & kStopPollMask) != 0u) {
                continue;
            }
            invalidate_l1_cache();
            if (ctrl->stop != 0u) {
                break;
            }
        }
        release_held();
    }
    out.close();
}
