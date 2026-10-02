// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Host-only: the sync engine and clock map against a synthetic truth model, with no device. Chain feeds three
// chips joined in a chain by two links, with chip 0 switching AICLK partway, chip 1 silent for longer than its refclk
// counter's 24-bit wrap, link streams with dropped and late stamps, and every counter a year past power-on. It checks
// every placement on the host timeline to within 0.5 ns and every steady_clock time to within 1 ns. Retention overflows
// a series' capacity and checks where records before, between and after the kept nodes land.

#include <cmath>
#include <cstdint>

#include <gtest/gtest.h>

#include "impl/streaming_profiler/service.hpp"
#include "impl/streaming_profiler/sync/engine.hpp"
#include "impl/streaming_profiler/sync/host_sync.hpp"

using namespace tt::tt_metal;
using namespace tt::tt_metal::streaming_profiler;

namespace {

using kernel_profiler::SyncRole;

constexpr double kRefHz = kernel_profiler::kEthRefclkHz;
// The test's readings are exact, so a placement is off only by its rounding to a TSC tick (0.33 ns).
constexpr double kTol = 0.5;
constexpr double kF0 = 1.35e9;
constexpr double kSlow = 26.875 / 27.0;  // chip 0's AICLK after its DVFS switch: one 1/8 step of the PLL multiple
constexpr double kTauSwitch = 0.300;
constexpr double kOneWay = 1.0e-6;
constexpr double kTurn = 350e-9;
// Every clock reads as it would a year after power-on, so each count is far past 2^53 of its units.
constexpr int64_t kYear = int64_t{365} * 86400;
constexpr int64_t kRef0[3] = {kYear * 50'000'000, kYear * 50'000'000 + 1'000'000, kYear * 50'000'000 + 3'000'000};
constexpr int64_t kWall0[3] = {
    kYear * 1'350'000'000 + 1'000'000'000,
    kYear * 1'350'000'000 + 7'000'000'000,
    kYear * 1'350'000'000 + 4'000'000'000};
constexpr int64_t kTsc0 = kYear * 3'000'000'000;
constexpr int64_t kHost0 = kYear * 1'000'000'000;
constexpr double kTicksPerNs = 3.0;

double refclk(double tau) { return kRefHz * tau; }
double wall(int chip, double tau) {
    if (chip != 0 || tau <= kTauSwitch) {
        return kF0 * tau;
    }
    return kF0 * kTauSwitch + kSlow * kF0 * (tau - kTauSwitch);
}
double tsc(double tau) { return tau * 1e9 * kTicksPerNs; }
double host_ns(double tau) { return tau * 1e9; }
int64_t wall_tick(int chip, double tau) { return kWall0[chip] + std::llround(wall(chip, tau)); }
uint64_t hw_stamp(int chip, double tau) {
    return static_cast<uint64_t>(kRef0[chip] * kNsPerRefclk + std::llround(refclk(tau) * kNsPerRefclk));
}

void feed_local(SyncEngine& sync, uint32_t dev, uint64_t refclk, uint64_t wall8, uint32_t wall_per_refclk_eighths) {
    sync.on_record(
        dev,
        0,
        {.local = {
             .meta = kernel_profiler::word_of(
                 kernel_profiler::SyncMeta{.count = 1, .kind = kernel_profiler::SyncKind::Local}),
             .rates = kernel_profiler::word_of(kernel_profiler::SyncLocalRates{
                 .k8 = {static_cast<uint8_t>(wall_per_refclk_eighths)},
                 .base_k8 = static_cast<uint8_t>(wall_per_refclk_eighths)}),
             .first_refclk = refclk,
             .first_wall8 = wall8}});
}
void feed_link(SyncEngine& sync, uint32_t dev, uint32_t core, uint32_t round, SyncRole role, uint64_t stamp) {
    sync.on_record(
        dev,
        core,
        {.link = {
             .meta = kernel_profiler::word_of(
                 kernel_profiler::SyncMeta{.role = role, .kind = kernel_profiler::SyncKind::Link}),
             .round = round,
             .first = stamp,
             .count = 1}});
}

}  // namespace

TEST(StreamingProfilerSyncEngine, Chain) {
    CaptureContext ctx;
    for (uint32_t chip = 0; chip < 3; chip++) {
        ctx.devices.push_back({.chip_id = chip});
    }
    ctx.links.push_back(CaptureContext::Link{.dev_a = 0, .dev_b = 1, .core_a = 0, .core_b = 0});
    ctx.links.push_back(CaptureContext::Link{.dev_a = 1, .dev_b = 2, .core_a = 1, .core_b = 0});
    ClockMap map(3, ClockMap::kSeriesNodes, {.root_refclk = kRef0[0], .tsc = kTsc0});
    for (double tau : {0.0, 0.6}) {
        map.append_host(
            HostNode{.at = refclk(tau), .value = tsc(tau), .tangent = kTicksPerNs * 1e9 / kRefHz}, refclk(tau + 0.6));
    }
    for (double tau : {0.0, 1.0}) {
        service().steady().append(kTsc0 + std::llround(tsc(tau)), kHost0 + std::llround(host_ns(tau)));
    }

    SyncEngine sync(ctx, map);
    // Chip 1 goes silent from 0.40 to 0.75 s, longer than the refclk's 24-bit period.
    constexpr uint32_t kK8Fast = 216, kK8Slow = 215;  // 27.0 and 26.875 wall ticks per refclk tick, in eighths
    const auto point = [&](int chip, double tau, uint32_t wall_per_refclk_eighths) {
        feed_local(
            sync,
            static_cast<uint32_t>(chip),
            static_cast<uint64_t>(kRef0[chip] + std::llround(refclk(tau))),
            static_cast<uint64_t>(8 * kWall0[chip] + std::llround(8.0 * wall(chip, tau))),
            wall_per_refclk_eighths);
    };
    bool switched = false;
    for (int k = 0; k < 1000; k++) {
        const double tau = k * 1e-3;
        if (!switched && tau > kTauSwitch) {
            point(0, kTauSwitch - 1e-6, kK8Fast);
            point(0, kTauSwitch + 50e-6, kK8Slow);
            point(0, kTauSwitch + 150e-6, kK8Slow);
            switched = true;
        }
        for (int c = 0; c < 3; c++) {
            if (c == 1 && tau > 0.40 && tau < 0.75) {
                continue;
            }
            point(c, tau, c == 0 && tau > kTauSwitch ? kK8Slow : kK8Fast);
        }
    }
    // Each link runs 300 rounds 1 ms apart over the same span, longer than the solve's 250 ms window, with its stream
    // damaged the way a lapped consumer or a full ring damages it.
    constexpr uint32_t kLateRound = 100, kLateArrival = 105;
    const auto burst = [&](uint32_t snd_dev, uint32_t snd_core, uint32_t rcv_dev, uint32_t rcv_core) {
        const auto receiver = [&](uint32_t round) {
            const double sent = 0.020 + round * 1e-3;
            feed_link(sync, rcv_dev, rcv_core, round, SyncRole::T0, hw_stamp(snd_dev, sent));
            feed_link(sync, rcv_dev, rcv_core, round, SyncRole::T1, hw_stamp(rcv_dev, sent + kOneWay));
        };
        for (uint32_t k = 0; k < 300; k++) {
            const double sent = 0.020 + k * 1e-3;
            const double echo_out = sent + kOneWay + kTurn, echo_in = sent + 2 * kOneWay + kTurn;
            const bool echo_lost = k % 11 == 5;
            const bool receive_lost = k % 7 == 3;
            feed_link(sync, snd_dev, snd_core, k, SyncRole::T1B, hw_stamp(rcv_dev, echo_out));
            if (!echo_lost) {
                feed_link(sync, snd_dev, snd_core, k, SyncRole::T2, hw_stamp(snd_dev, echo_in));
            }
            if (!receive_lost && k != kLateRound) {
                receiver(k);
            }
            if (k == kLateArrival) {
                receiver(kLateRound);
            }
        }
    };
    burst(/*snd*/ 0, 0, /*rcv*/ 1, 0);
    burst(/*snd*/ 1, 1, /*rcv*/ 2, 0);
    sync.on_capture_end();

    ClockMap::Reader reader = map.reader();
    const auto placed_tsc = [&](int chip, double tau) {
        return map.place_host(reader, static_cast<uint32_t>(chip), wall_tick(chip, tau));
    };
    const auto placed_ns = [&](int chip, double tau) {
        return (static_cast<double>(placed_tsc(chip, tau) - kTsc0) - tsc(tau)) / kTicksPerNs;
    };
    const auto steady_ns = [&](int chip, double tau) {
        return static_cast<double>(steady_mono_ns(placed_tsc(chip, tau)) - kHost0);
    };
    for (double tau : {0.050, 0.150, 0.280}) {
        EXPECT_NEAR(placed_ns(0, tau), 0.0, kTol) << "chip0 root pre-switch, tau " << tau;
    }
    for (double tau : {0.400, 0.700, 0.950}) {
        EXPECT_NEAR(placed_ns(0, tau), 0.0, kTol) << "chip0 root post-switch, tau " << tau;
    }
    for (double tau : {0.050, 0.500, 0.950}) {
        EXPECT_NEAR(placed_ns(1, tau), 0.0, kTol) << "chip1 one hop, tau " << tau;
        EXPECT_NEAR(placed_ns(2, tau), 0.0, kTol) << "chip2 two hops, tau " << tau;
    }
    for (int chip : {0, 1, 2}) {
        for (double tau : {0.050, 0.500, 0.950}) {
            EXPECT_NEAR(steady_ns(chip, tau), host_ns(tau), 1.0) << "steady chip" << chip << ", tau " << tau;
        }
    }
}

TEST(StreamingProfilerSyncEngine, Retention) {
    constexpr uint32_t kNodes = 1u << 14;
    constexpr uint32_t chip = 3;
    ClockMap map(chip + 1, kNodes, {});
    ClockMap::Reader map_reader = map.reader();
    constexpr int64_t step = 1000;
    // The first segment is steeper than the rest, which alternate in slope, and no node's tangent matches a chord, so
    // the retired first node, a kept node and a mid-series chord each place differently. Every value is a multiple of
    // 2^-5 below 2^43, where a double resolves 2^-10, so the map's arithmetic on them is exact.
    const auto value = [](uint32_t node) {
        return node == 0 ? 7.5e12 - 62.5 : 7.5e12 + 31.25 * node + 3.90625 * (node % 2);
    };
    const auto tangent = [](uint32_t node) { return 0.041015625 + 0.001953125 * (node % 3); };
    const uint32_t node_count = kNodes + 1;
    for (uint32_t i = 0; i < node_count; i++) {
        map.append(chip, SyncNode{.at = static_cast<int64_t>(i) * step, .value = value(i), .tangent = tangent(i)});
    }
    const auto root = [&](int64_t tick) { return map.root_offset(map_reader, chip, tick, 0.0).value_or(0.0); };
    const uint32_t mid = node_count / 2;
    EXPECT_NEAR(root(static_cast<int64_t>(node_count - 1) * step), value(node_count - 1), 1e-3) << "the newest node";
    EXPECT_NEAR(root(static_cast<int64_t>(mid) * step + step / 2), (value(mid) + value(mid + 1)) / 2, 1e-3)
        << "a node mid-series";
    EXPECT_NEAR(root(0), value(1) - tangent(1) * step, 1e-3) << "the retired first node, on the oldest kept tangent";
}
