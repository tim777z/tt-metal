// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// ActiveEthPtpStamps' kernel: one end of a link's stamped frame exchange, as initiator or echo.

#include <cstddef>
#include <cstdint>

#include "internal/ethernet/dataflow_api.h"
#include "internal/ethernet/eth_ptp.hpp"
#include "eth_ptp_stamps.hpp"

namespace eth_ptp = tt::tt_metal::eth_ptp;
using namespace eth_ptp_stamps;

constexpr bool kInitiator = get_compile_time_arg_val(0) != 0;
// A TX queue and header row that neither the firmware nor the fabric uses (eth_ptp.hpp), and any TCAM row and label.
constexpr uint32_t kTxq = 2, kHeaderRow = 3, kTcamRow = 63, kLabel = 0x15;
// A destination no firmware frame has (eth_ptp.hpp), and a rule that matches only it.
constexpr uint64_t kStampFrameDa = 0x02A5'A5A5'A5A5ull;
// A 1 in the mask means don't care, so the rule compares only the four 0xA5 bytes.
constexpr eth_ptp::RxTcamNonIpPattern kStampRowValues{.da = {0xA5A5'A500u, 0x0000'00A5u}};
constexpr eth_ptp::RxTcamNonIpPattern kStampRowMask{
    .sa = {~0u, ~0u, ~0u, ~0u},
    .da = {0x0000'00FFu, 0xFFFF'FF00u, ~0u, ~0u},
    .addr_flags = {.augmented_da = 0xF, .augmented_sa = 0xF},
    .ethertype = {.value = 0xFFFF, .augmented = 0xF},
    .priority = {.pcp = 7}};
// The MAC writes the egress stamp into `stamp`.
struct Frame {
    eth_ptp::FrameStampSlot stamp;
    uint32_t key;
    uint32_t echo_key;
};
static_assert(
    eth_ptp::kFrameStampOffsetBytes + eth_ptp::kFrameStampBytes <= offsetof(Frame, key) &&
    sizeof(Frame) <= kFrameBytes);
using StampQueue = eth_ptp::TxQueue<kTxq>;
static eth_ptp::TxHeaderRow<kTxq, kHeaderRow> g_header;
static eth_ptp::RxStampRule<kTcamRow, kLabel> g_rule;
static constexpr uint32_t kSpins = 1u << 20;
constexpr uint32_t kKeyTag = 0x5A000000u;

inline void send(volatile tt_l1_ptr Frame* frame) {
    eth_ptp::clear_frame_stamp(frame->stamp);
    const uint32_t addr = reinterpret_cast<uint32_t>(frame);
    internal_::eth_send_packet<false>(kTxq, addr >> 4, addr >> 4, kFrameBytes >> 4);
    while (internal_::eth_txq_is_busy(kTxq)) {
    }
}

struct Ingress {
    uint64_t stamp = 0;
    uint32_t count = 0;
};

Ingress take_ingress() {
    Ingress ingress;
    eth_ptp::RxStampFifo::drain<kLabel>([&](uint64_t stamp) {
        if (ingress.count++ == 0) {
            ingress.stamp = stamp;
        }
    });
    return ingress;
}

template <typename Pred>
bool wait_for(Pred&& pred) {
    for (uint32_t spin = 0; spin < kSpins; spin++) {
        invalidate_l1_cache();
        if (pred()) {
            return true;
        }
    }
    return false;
}

void kernel_main() {
    const uint32_t base = get_arg_val<uint32_t>(0);
    volatile tt_l1_ptr Result* result = reinterpret_cast<volatile tt_l1_ptr Result*>(base + kResultOffset);
    volatile tt_l1_ptr Frame* frame = reinterpret_cast<volatile tt_l1_ptr Frame*>(base + kFrameOffset);
    frame->key = 0;
    frame->echo_key = 0;
    result->done = 0;
    result->header_select_before = eth_ptp::word_of(eth_ptp::txq_header_select(kTxq).read());
    result->no_match_before = eth_ptp::word_of(eth_ptp::kRxNoMatchActions.read());
    eth_ptp::restart_ptp_timer();
    g_rule.install(kStampRowValues, kStampRowMask);
    g_header.install(kStampFrameDa);
    StampQueue::arm_in_frame();

    if constexpr (kInitiator) {
        eth_send_bytes(base, base, 16);
        eth_wait_for_receiver_done();
    } else {
        eth_wait_for_bytes(16);
        eth_receiver_channel_done(0);
    }

    uint32_t unstamped = 0, rx_missing = 0, rx_extra = 0, i = 0;
    for (; i < kRounds; i++) {
        const uint32_t key = kKeyTag | (i + 1);
        if constexpr (kInitiator) {
            frame->key = key;
            frame->echo_key = 0;
            send(frame);
            if (!wait_for([&] { return frame->echo_key == key; })) {
                break;
            }
        } else {
            if (!wait_for([&] { return frame->key == key; })) {
                break;
            }
        }
        const uint64_t egress = eth_ptp::frame_stamp_ns(frame->stamp);
        const Ingress ingress = take_ingress();
        rx_missing += ingress.count == 0;
        rx_extra += ingress.count > 1 ? ingress.count - 1 : 0;
        unstamped += egress == 0;
        result->stamps[i].egress = egress;
        result->stamps[i].ingress = ingress.stamp;
        if constexpr (!kInitiator) {
            frame->echo_key = key;
            send(frame);
        }
    }

    StampQueue::disarm();
    g_header.restore();
    g_rule.remove();
    result->header_select_after = eth_ptp::word_of(eth_ptp::txq_header_select(kTxq).read());
    result->no_match_after = eth_ptp::word_of(eth_ptp::kRxNoMatchActions.read());
    result->unstamped = unstamped;
    result->rx_missing = rx_missing;
    result->rx_extra = rx_extra;
    result->rounds = i;
    result->done = kDone;
}
