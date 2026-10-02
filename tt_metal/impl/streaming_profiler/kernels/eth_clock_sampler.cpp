// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <atomic>
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "hostdev/streaming_profiler_common.h"
#include "internal/ethernet/eth_ptp.hpp"
#include "tt_metal/impl/streaming_profiler/kernels/eth_clock_sampling.hpp"

namespace sampler {
constexpr uint32_t kPadSlots = 256;
// The registers come in as data, so the compiler holds their addresses in registers instead of rebuilding them between
// blocks.
struct StreamArgs {
    eth_ptp::Reg<> refclk;
    eth_ptp::Reg<> wall;
    Table table;
    volatile uint32_t* slots;
    uint32_t mask;
    volatile uint32_t* head;
    uint32_t limit, iters;
    const uint32_t* pads;
    volatile uint32_t* tail_word;
    uint32_t tail;
};

/**
 * @brief Streams refclk updates into args.slots from args.tail on and returns the tail after the last one it stored.
 *
 * Update n goes to slot n & args.mask. Each pass first runs args.pads[tail % kPadSlots] / 4 nops, then reads blocks of
 * refclk, refclk, wall, refclk until the refclk steps. An update whose two blocks each took args.table.period wall
 * ticks is stored as its new refclk and the wall time of its step in eighths, the block's wall read plus its gap's
 * position in args.table.pos8; one that did not is dropped. With period 0 every update is stored, unpublished, as its
 * block's wall length, for calibration. Otherwise, before each stored update the tail it follows is written to
 * *args.tail_word, so every published sample was stored a handler earlier.
 *
 * At args.limit the stream reloads *args.head, the model's position, and moves the limit args.mask past it, waiting
 * there while the ring is full. The first pass and the pass after a reload skip their pad. It returns when the head is
 * kSyncHeadStop past the model's position, or once args.iters passes have stored nothing (0 counts 2^32).
 */
// -Os lays the calibration path in line and merges the handlers' tails, and the extra taken branches, which cost more
// with the branch predictor off, made the stream miss every other update at 800 MHz.
__attribute__((noinline, aligned(64), optimize("no-crossjumping", "reorder-blocks-algorithm=stc"))) inline uint32_t
sampler_stream(const StreamArgs& args) {
    struct Block {
        uint32_t first_refclk, second_refclk, wall, last_refclk;
    };
    enum class Step : uint8_t { Stored, Refill, Rejected };
    const eth_ptp::Reg<> refclk = args.refclk, wall = args.wall;
    const uint32_t period = args.table.period, gap_pos8 = args.table.pos8.word, mask = args.mask;
    volatile uint32_t* const slots = args.slots;
    volatile uint32_t* const head = args.head;
    uint32_t limit = args.limit, iters = args.iters;
    const uint32_t* const pads = args.pads;
    volatile uint32_t* const tail_word = args.tail_word;
    uint32_t tail = args.tail;
    const auto read = [&]() __attribute__((always_inline)) {
        Block block;
        block.first_refclk = refclk.read();
        block.second_refclk = refclk.read();
        block.wall = wall.read();
        block.last_refclk = refclk.read();
        return block;
    };
    // The pad jumps pad bytes back from the end of its nops. The asm defines the label, so it must be emitted once.
    uint32_t nops_end;
    asm("lla %0, .Lsampler_stream_nops_end" : "=r"(nops_end));
    uint32_t pad = 0;
    branch_predictor(false);
    // The pass starts on a cache line, so its blocks sit where they do in every build. Passing pad through keeps its
    // load ahead of the alignment, so nothing lands between the alignment and the pass.
    asm volatile(".p2align 6" : "+r"(pad));
    // Odds this low keep a step at the last check a single taken branch into its handler.
    const auto stepped = [](const Block& before, const Block& last) __attribute__((always_inline)) {
        return __builtin_expect_with_probability(last.last_refclk != before.last_refclk, 0, 0.001);
    };
    // A step between before's and last's final reads is placed by last's reads, and now times the block after it.
    const auto store = [&](const Block& before, const Block& last, const Block& now)
                           __attribute__((always_inline)) -> Step {
        const uint32_t step_block_ticks = last.wall - before.wall, next_block_ticks = now.wall - last.wall;
        volatile uint32_t* const slot = slots + 2 * (tail & mask);
        if (__builtin_expect(period != 0, 1)) {
            if (((step_block_ticks ^ period) | (next_block_ticks ^ period)) != 0) {
                return Step::Rejected;
            }
            const uint32_t gap = static_cast<uint32_t>(last.first_refclk == before.last_refclk)
                                 << (last.second_refclk == last.first_refclk);
            slot[0] = last.last_refclk;
            slot[1] = (last.wall << 3) + static_cast<uint32_t>(static_cast<int8_t>(gap_pos8 >> (8 * gap)));
            *tail_word = tail;
            return ++tail == limit ? Step::Refill : Step::Stored;
        }
        slot[0] = next_block_ticks;
        return ++tail == limit ? Step::Refill : Step::Stored;
    };
    // Reads blocks until the refclk steps, and stores the step. Three blocks rotating by name keep each check's reads
    // in the registers they landed in; a single rotating site or an array makes the compiler copy them into one
    // handler's registers.
    const auto run = [&]() __attribute__((always_inline)) -> Step {
        Block a = read(), b = read(), c;
#pragma GCC unroll 6
        for (uint32_t i = 0; i < 6; i++) {
            c = read();
            if (stepped(a, b)) {
                return store(a, b, c);
            }
            a = read();
            if (stepped(b, c)) {
                return store(b, c, a);
            }
            b = read();
            if (stepped(c, a)) {
                return store(c, a, b);
            }
        }
        return Step::Rejected;
    };
pass:
    asm volatile(
        ".option push\n\t.option norvc\n\t"
        "sub t0, %[end], %[pad]\n\t"
        "jr t0\n\t"
        ".rept %[n]\n\tnop\n\t.endr\n"
        ".Lsampler_stream_nops_end:\n\t"
        ".option pop"
        :
        : [end] "r"(nops_end), [pad] "r"(pad), [n] "i"(kTrackerPadRange - 1)
        : "t0", "memory");
    pad = pads[tail & (kPadSlots - 1)];
    switch (run()) {
        case Step::Stored: goto pass;
        case Step::Refill: goto refill;
        case Step::Rejected: goto reject;
    }
reject:
    if (--iters != 0) {
        goto pass;
    }
    goto done;
refill:
    for (;;) {
        const uint32_t model_head = *head;
        if (__builtin_expect(static_cast<int32_t>(tail - model_head) < 0, 0)) {
            goto done;
        }
        limit = model_head + mask;
        if (limit != tail) {
            break;
        }
    }
    pad = 0;
    goto pass;
done:
    branch_predictor(true);
    return tail;
}
}  // namespace sampler

constexpr uint32_t kCtrlAddr = get_named_compile_time_arg_val("ctrl_addr");
constexpr uint32_t kSampleRingAddr = get_named_compile_time_arg_val("sample_ring_addr");

namespace kp = kernel_profiler;
namespace eth_ptp = tt::tt_metal::eth_ptp;

void kernel_main() {
    volatile tt_l1_ptr kp::RelayCtrl* ctrl = reinterpret_cast<volatile tt_l1_ptr kp::RelayCtrl*>(kCtrlAddr);
    volatile tt_l1_ptr kp::SyncSampleRing* ring =
        reinterpret_cast<volatile tt_l1_ptr kp::SyncSampleRing*>(kSampleRingAddr);
    // The stream's pads in bytes, 4 per nop, uniform over kTrackerPadRange nops, drawn once so an update's pad is one
    // load.
    static uint32_t pads[sampler::kPadSlots];
    uint32_t walk = eth_ptp::kWallClockLo.read() | 1u;
    for (uint32_t& pad : pads) {
        pad = 4u * sampler::draw<sampler::kTrackerPadRange>(walk);
    }
    // Calibration's updates aren't samples, so they publish to a word the model never reads, and each call ends at its
    // first update, whose reload finds the head at kSyncHeadStop.
    uint32_t unpublished = 0;
    sampler::StreamArgs args{
        .refclk = eth_ptp::kRefclkLo,
        .wall = eth_ptp::kWallClockLo,
        .head = &ring->head,
        .pads = pads,
        .tail_word = &unpublished};
    ring->head = kp::kSyncHeadStop;
    const sampler::Table table = sampler::calibrate(
        [&](uint32_t period, sampler::GapPos8 pos8, volatile tt_l1_ptr uint32_t* slot) {
            const uint32_t from = args.tail;
            // Field by field: copying the structs goes through the stack at -Os.
            args.table.period = period;
            args.table.pos8.word = pos8.word;
            args.slots = slot;
            args.iters = 2;
            args.limit = from + 1;
            args.tail = sampler_stream(args);
            return args.tail - from;
        },
        ctrl);
    if (ctrl->stop == 0u) {
        const eth_ptp::Instant start = eth_ptp::read_instant();
        ring->refclk = start.refclk;
        ring->wall8 = start.wall() << 3;
        ring->head = 0;
        args.table.period = table.period;
        args.table.pos8.word = table.pos8.word;
        args.slots = &ring->samples[0].refclk;
        args.mask = kp::kSyncSampleRingSamples - 1;
        args.iters = 0;
        args.tail_word = &ring->tail;
        args.tail = 0;
        args.limit = args.mask;
        const uint32_t tail = sampler_stream(args);
        std::atomic_thread_fence(std::memory_order_release);
        ring->tail = tail;
    }
    std::atomic_thread_fence(std::memory_order_release);
    ring->done = 1;
}
