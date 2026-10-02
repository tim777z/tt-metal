// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Blackhole Ethernet IEEE 1588 hardware: the PTP timer, the TX queues' timestamping, the TX header table, the RX
// classifier's stamp rules, and the RX and TX timestamp FIFOs, plus the refclk and ERISC wall clock reads that go with
// them.
//
// These blocks belong to the tile: a second user on the same tile would take the first one's FIFO entries, and
// restarting the PTP timer restarts every user's PTP time. Setup is cold and out of line, so an -O3 caller keeps it out
// of its hot code; per-frame operations are inline.

#pragma once

#if !defined(ARCH_BLACKHOLE)
#error "eth_ptp.hpp is Blackhole only"
#endif

#include <cstdint>

#include "internal/ethernet/dataflow_api.h"
#include "internal/ethernet/tt_eth_ss_regs.h"
#include "internal/risc_attribs.h"

namespace tt::tt_metal::eth_ptp {

constexpr uint32_t kRefclkHz = 50'000'000u;
constexpr uint32_t kNsPerRefclkTick = 1'000'000'000u / kRefclkHz;

template <typename T>
constexpr uint32_t word_of(const T& value) {
    static_assert(sizeof(T) == sizeof(uint32_t));
    return __builtin_bit_cast(uint32_t, value);
}

// A register bound to its layout. A layout struct names every bit, reserved ones included, so that a value built with
// designated initializers has all its other bits zero.
template <typename T = uint32_t>
struct Reg {
    static_assert(sizeof(T) == sizeof(uint32_t));
    uint32_t addr;
    FORCE_INLINE T read() const { return __builtin_bit_cast(T, *reinterpret_cast<volatile uint32_t*>(addr)); }
    FORCE_INLINE void write(T value) const { *reinterpret_cast<volatile uint32_t*>(addr) = word_of(value); }
};

template <typename T = uint32_t>
constexpr Reg<T> risc_reg(uint32_t offset) {
    return {ETH_RISC_REGS_START + offset};
}
template <typename T = uint32_t>
constexpr Reg<T> ptp_timer_reg(uint32_t offset) {
    return {ETH_PTP_TIMER_REGS_START + offset};
}
template <typename T = uint32_t>
constexpr Reg<T> txq_reg(uint32_t queue, uint32_t offset) {
    return {ETH_TXQ0_REGS_START + queue * ETH_TXQ_REGS_SIZE + offset};
}
template <typename T = uint32_t>
constexpr Reg<T> tx_header_reg(uint32_t row, uint32_t offset) {
    return {ETH_TXPKT_CFG_REGS_START + row * ETH_TXPKT_CFG_REGS_SIZE + offset};
}
template <typename T = uint32_t>
constexpr Reg<T> rx_classifier_reg(uint32_t offset, uint32_t word = 0) {
    return {ETH_RX_CLASSIFIER_REGS_START + offset + word * sizeof(uint32_t)};
}
template <typename T = uint32_t>
constexpr Reg<T> rx_stamp_reg(uint32_t offset) {
    return {ETH_RX_TH_REGS_START + offset};
}
template <typename T = uint32_t>
constexpr Reg<T> mac_reg(uint32_t offset) {
    return {ETH_MAC_REGS_START + offset};
}

// ---------------------------------------------------------------------------------------------------------------------
// The refclk and the ERISC wall clock

// The refclk count lives in the PTP timer block. Reading a counter's low word latches its high word.
constexpr Reg<> kRefclkLo = ptp_timer_reg(ETH_PTP_TIMER_CFR_LO);
constexpr Reg<> kRefclkHi = ptp_timer_reg(ETH_PTP_TIMER_CFR_HI);
// The wall clock's low word latches WALL_CLOCK_1_AT; WALL_CLOCK_1 is live.
constexpr Reg<> kWallClockLo = risc_reg(ETH_RISC_WALL_CLOCK_0);
constexpr Reg<> kWallClockHi = risc_reg(ETH_RISC_WALL_CLOCK_1_AT);

FORCE_INLINE uint64_t read_latched(Reg<> lo_reg, Reg<> hi_reg) {
    const uint32_t lo = lo_reg.read();
    const uint32_t hi = hi_reg.read();
    return (static_cast<uint64_t>(hi) << 32) | lo;
}
FORCE_INLINE uint64_t read_refclk() { return read_latched(kRefclkLo, kRefclkHi); }

// The wall clock in the two words it was read as, since most callers need only the low one, and the refclk read with
// it.
struct Instant {
    uint32_t wall_lo, wall_hi;
    uint64_t refclk;
    uint64_t wall() const { return (static_cast<uint64_t>(wall_hi) << 32) | wall_lo; }
};
// The wall clock's low read latches its high word for only a few cycles, and the refclk read between the two takes
// longer, so a low word that wrapped before the high read would tear the pair by 2^32.
FORCE_INLINE Instant read_instant() {
    Instant instant;
    while (true) {
        const uint32_t wall_hi_before = kWallClockHi.read();
        instant.wall_lo = kWallClockLo.read();
        const uint32_t refclk_lo = kRefclkLo.read();
        instant.wall_hi = kWallClockHi.read();
        const uint32_t refclk_hi = kRefclkHi.read();
        if (instant.wall_hi == wall_hi_before) {
            instant.refclk = (static_cast<uint64_t>(refclk_hi) << 32) | refclk_lo;
            return instant;
        }
    }
}

// The refclk's low word just after it changed, and the wall clock's low word read in the same loop pass.
struct RefclkUpdate {
    uint32_t wall_lo, refclk_lo;
};
FORCE_INLINE RefclkUpdate await_refclk_update() {
    uint32_t prev = kRefclkLo.read();
    while (true) {
        const uint32_t wall_lo = kWallClockLo.read();
        const uint32_t refclk_lo = kRefclkLo.read();
        if (refclk_lo != prev) {
            return {wall_lo, refclk_lo};
        }
        prev = refclk_lo;
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// The PTP timer

struct PtpTimerCtrl {
    uint32_t enable : 1;
    uint32_t rsvd : 31;
};
// The PTP ns the timer adds per refclk tick, in 8.16 fixed point.
struct PtpTickIncrement {
    uint32_t ns_q8_16 : 24;
    uint32_t rsvd : 8;
};
struct PtpUpdateRequest {
    uint32_t request : 1;
    uint32_t rsvd : 31;
};
struct PtpUpdateStat {
    uint32_t tick_increment_pending : 1;
    uint32_t time_pending : 1;
    uint32_t rsvd0 : 6;
    uint32_t tick_increment_ack : 1;
    uint32_t time_ack : 1;
    uint32_t rsvd1 : 6;
    uint32_t tick_increment_late : 1;
    uint32_t time_late : 1;
    uint32_t rsvd2 : 14;
};
// How many PTP ns the SYNC timers run ahead of the main one.
struct PtpSyncOffset {
    uint32_t ns : 8;
    uint32_t rsvd : 24;
};

constexpr Reg<PtpTimerCtrl> kPtpTimerCtrl = ptp_timer_reg<PtpTimerCtrl>(ETH_PTP_TIMER_CTRL);
// A scheduled update: at refclk count kPtpUpdateAtRefclk*, the timer takes the requested tick increment and time.
constexpr Reg<> kPtpUpdateAtRefclkLo = ptp_timer_reg(ETH_PTP_TIMER_FUTURE_CFR_LO);
constexpr Reg<> kPtpUpdateAtRefclkHi = ptp_timer_reg(ETH_PTP_TIMER_FUTURE_CFR_HI);
constexpr Reg<PtpTickIncrement> kPtpNextTickIncrement = ptp_timer_reg<PtpTickIncrement>(ETH_PTP_TIMER_FUTURE_PTI);
constexpr Reg<> kPtpNextTimeLo = ptp_timer_reg(ETH_PTP_TIMER_FUTURE_TIMESTAMP_LO);
constexpr Reg<> kPtpNextTimeHi = ptp_timer_reg(ETH_PTP_TIMER_FUTURE_TIMESTAMP_HI);
constexpr Reg<PtpUpdateRequest> kPtpUpdateTickIncrement = ptp_timer_reg<PtpUpdateRequest>(ETH_PTP_TIMER_UPDATE_PTI);
constexpr Reg<PtpUpdateRequest> kPtpUpdateTime = ptp_timer_reg<PtpUpdateRequest>(ETH_PTP_TIMER_UPDATE_TIMESTAMP);
constexpr Reg<PtpUpdateStat> kPtpUpdateStat = ptp_timer_reg<PtpUpdateStat>(ETH_PTP_TIMER_UPDATE_STAT);
constexpr Reg<PtpSyncOffset> kPtpSyncOffset = ptp_timer_reg<PtpSyncOffset>(ETH_PTP_TIMER_SYNC_OFFSET1);
constexpr Reg<PtpTickIncrement> kPtpTickIncrement = ptp_timer_reg<PtpTickIncrement>(ETH_PTP_TIMER_PTI_STAT);
// The PTP time as 32-bit seconds and 32-bit ns, and as 64-bit ns; the SYNC timers' copies run kPtpSyncOffset ahead.
constexpr Reg<> kPtpSecondsNsLo = ptp_timer_reg(ETH_PTP_TIMER_32S_32NS_LO);
constexpr Reg<> kPtpSecondsNsHi = ptp_timer_reg(ETH_PTP_TIMER_32S_32NS_HI);
constexpr Reg<> kPtpNsLo = ptp_timer_reg(ETH_PTP_TIMER_64NS_LO);
constexpr Reg<> kPtpNsHi = ptp_timer_reg(ETH_PTP_TIMER_64NS_HI);
constexpr Reg<> kPtpSyncSecondsNsLo = ptp_timer_reg(ETH_PTP_TIMER_SYNC_32S_32NS_LO);
constexpr Reg<> kPtpSyncSecondsNsHi = ptp_timer_reg(ETH_PTP_TIMER_SYNC_32S_32NS_HI);
constexpr Reg<> kPtpSyncNsLo = ptp_timer_reg(ETH_PTP_TIMER_SYNC_64NS_LO);
constexpr Reg<> kPtpSyncNsHi = ptp_timer_reg(ETH_PTP_TIMER_SYNC_64NS_HI);

FORCE_INLINE uint64_t read_ptp_ns() { return read_latched(kPtpNsLo, kPtpNsHi); }

// Far enough ahead that the scheduled restart is still in the future once all of its setup writes have landed.
constexpr uint32_t kPtpRestartLeadTicks = 5000;  // 100 us

// Restarts the tile's PTP time at 0 a little ahead, counting kNsPerRefclkTick per refclk tick, and returns PTP ns
// minus refclk ns from then on. Every PTP stamp the tile takes restarts with it.
__attribute__((noinline, cold)) inline int64_t restart_ptp_timer() {
    kPtpTimerCtrl.write({.enable = 1});
    const uint64_t restart_at = read_refclk() + kPtpRestartLeadTicks;
    kPtpUpdateAtRefclkLo.write(static_cast<uint32_t>(restart_at));
    kPtpUpdateAtRefclkHi.write(static_cast<uint32_t>(restart_at >> 32));
    kPtpNextTickIncrement.write({.ns_q8_16 = kNsPerRefclkTick << 16});
    kPtpNextTimeLo.write(0);
    kPtpNextTimeHi.write(0);
    kPtpUpdateTickIncrement.write({.request = 1});
    kPtpUpdateTime.write({.request = 1});
    PtpUpdateStat stat{};
    while (!(stat.tick_increment_ack && stat.time_ack)) {
        stat = kPtpUpdateStat.read();
    }
    kPtpUpdateTickIncrement.write({});
    kPtpUpdateTime.write({});
    // The restart lands one tick after the scheduled one.
    return -static_cast<int64_t>(restart_at + 1) * kNsPerRefclkTick;
}

// ---------------------------------------------------------------------------------------------------------------------
// TX queues: which packets get a send-time stamp, and how

enum class StampMode : uint32_t { None, OneStepCorrection, OneStepInFrame, TwoStepFifo };

struct TxqStampControl {
    StampMode mode : 3;
    uint32_t rsvd0 : 13;
    uint32_t in_frame_offset : 6;  // in 2-byte units; see kFrameStampOffsetBytes
    uint32_t rsvd1 : 10;
};

// The TX header table row each kind of software-issued packet uses.
struct TxqHeaderSelect {
    uint32_t raw : 4;
    uint32_t reg_write : 4;
    uint32_t packet : 4;
    uint32_t rsvd : 20;
};

constexpr Reg<TxqHeaderSelect> txq_header_select(uint32_t queue) {
    return txq_reg<TxqHeaderSelect>(queue, ETH_TXQ_TXPKT_CFG_SEL_SW);
}
constexpr Reg<TxqStampControl> txq_stamp_control(uint32_t queue) {
    return txq_reg<TxqStampControl>(queue, ETH_TXQ_TIMESTAMP);
}
// The tag a two-step stamp carries into the TX timestamp FIFO.
constexpr Reg<> txq_two_step_tag_lo(uint32_t queue) { return txq_reg(queue, ETH_TXQ_RX_TIMESTAMP_LO); }
constexpr Reg<> txq_two_step_tag_hi(uint32_t queue) { return txq_reg(queue, ETH_TXQ_RX_TIMESTAMP_HI); }
constexpr Reg<> txq_word_count(uint32_t queue) { return txq_reg(queue, ETH_TXQ_WORD_CNT); }

// The queue samples its stamp control 24 to 32 cycles after a send command (measured), so arming just after the
// command still covers that packet. It reports idle before the MAC has taken the setting, though, so to stamp only
// chosen packets, disarm once the last one's time is in.
template <uint32_t Queue>
struct TxQueue {
    static_assert(Queue < NUM_ETH_QUEUES);
    static FORCE_INLINE void arm_two_step(uint64_t tag) {
        txq_two_step_tag_lo(Queue).write(static_cast<uint32_t>(tag));
        txq_two_step_tag_hi(Queue).write(static_cast<uint32_t>(tag >> 32));
        txq_stamp_control(Queue).write(TxqStampControl{.mode = StampMode::TwoStepFifo});
    }
    static FORCE_INLINE void arm_in_frame() {
        txq_stamp_control(Queue).write(
            TxqStampControl{.mode = StampMode::OneStepInFrame, .in_frame_offset = kInFrameOffsetSetting});
    }
    static FORCE_INLINE void disarm() { txq_stamp_control(Queue).write(TxqStampControl{.mode = StampMode::None}); }
    // In 96-byte units on the wire: one for a keepalive or a packet of up to 64 bytes of payload, two up to 128.
    static FORCE_INLINE uint32_t words_sent() { return txq_word_count(Queue).read(); }

    static constexpr uint32_t kInFrameOffsetSetting = 2;
};

// A one-step stamp is 16 high bits and then the 64-bit PTP time in ns, big-endian. It starts 2 bytes past the queue's
// offset setting, which counts 2-byte units (measured). Keepalives are stamped like any other packet: a stamp near the
// start of the payload lands in their padding, but at offset 40, past its end, every keepalive was lost.
constexpr uint32_t kFrameStampOffsetBytes = 2 * TxQueue<0>::kInFrameOffsetSetting + 2;
constexpr uint32_t kFrameStampBytes = 10;
constexpr uint32_t kFrameStampTimeWord = (kFrameStampOffsetBytes + 2) / sizeof(uint32_t);
static_assert((kFrameStampOffsetBytes + 2) % sizeof(uint32_t) == 0);

// The start of a frame, where the MAC writes its one-step stamp. A type of its own, so that a store to it doesn't make
// the compiler reload a client's own uint32_t members.
struct FrameStampSlot {
    uint32_t words[(kFrameStampOffsetBytes + kFrameStampBytes + 3) / sizeof(uint32_t)];
};

FORCE_INLINE uint64_t frame_stamp_ns(const volatile FrameStampSlot& slot) {
    return (static_cast<uint64_t>(__builtin_bswap32(slot.words[kFrameStampTimeWord])) << 32) |
           __builtin_bswap32(slot.words[kFrameStampTimeWord + 1]);
}
// A frame the MAC didn't stamp keeps whatever its slot held, so a slot reused for sending is cleared first.
FORCE_INLINE void clear_frame_stamp(volatile FrameStampSlot& slot) {
    slot.words[kFrameStampTimeWord] = 0;
    slot.words[kFrameStampTimeWord + 1] = 0;
}

// ---------------------------------------------------------------------------------------------------------------------
// The TX header table

// The firmware sends each TX queue with the header row of the same number, rows 0 to 2, and leaves rows 3 to 9 free.
// install() points Queue at Row, a copy of the queue's own row except for the destination `da`.
template <uint32_t Queue, uint32_t Row>
class TxHeaderRow {
public:
    __attribute__((noinline, cold)) void install(uint64_t da) {
        select_before_ = txq_header_select(Queue).read();
        for (uint32_t reg :
             {ETH_TXPKT_CFG_INSERT_CTL,
              ETH_TXPKT_CFG_CUSTOM_HDR,
              ETH_TXPKT_CFG_MAC_SA_LO,
              ETH_TXPKT_CFG_MAC_SA_HI,
              ETH_TXPKT_CFG_ETHERTYPE,
              ETH_TXPKT_CFG_VLAN1,
              ETH_TXPKT_CFG_VLAN2}) {
            tx_header_reg(Row, reg).write(tx_header_reg(Queue, reg).read());
        }
        tx_header_reg(Row, ETH_TXPKT_CFG_MAC_DA_LO).write(static_cast<uint32_t>(da));
        tx_header_reg(Row, ETH_TXPKT_CFG_MAC_DA_HI).write(static_cast<uint32_t>(da >> 32));
        txq_header_select(Queue).write({.raw = Row, .reg_write = Row, .packet = Row});
    }
    __attribute__((noinline, cold)) void restore() const { txq_header_select(Queue).write(select_before_); }

private:
    TxqHeaderSelect select_before_{};
};

// ---------------------------------------------------------------------------------------------------------------------
// The RX classifier: TCAM rows match frames, and each row's flow-table row says what to do with them

struct RxFlowActions {
    uint32_t rx_queue : 2;
    uint32_t drop : 1;
    uint32_t strip_headers : 1;
    uint32_t record_rx_time : 1;
    uint32_t prepend_sw_metadata : 1;
    uint32_t prepend_hw_metadata : 1;
    uint32_t rsvd : 25;
};

struct RxTcamRowMapping {
    uint32_t priority : 3;  // the larger wins
    uint32_t rsvd0 : 13;
    uint32_t ftable_row : 6;
    uint32_t rsvd1 : 10;
};
constexpr uint32_t kRxTcamTopPriority = 7;

struct RxTcamRowUpdate {
    uint32_t row : 6;
    uint32_t rsvd0 : 2;
    uint32_t enable : 1;
    uint32_t rsvd1 : 7;
    uint32_t write : 1;
    uint32_t rsvd2 : 14;
    uint32_t go : 1;
};

enum class RxTcamTupleType : uint32_t { NonIp = 0 };

// Each update_* bit has the row take that part of its pattern from the write registers; the rest is left unchanged.
struct RxTcamUpdate {
    uint32_t row : 6;
    uint32_t rsvd0 : 2;
    uint32_t mask : 1;  // the row's mask bits rather than its values
    uint32_t write : 1;
    uint32_t not_ip : 1;
    uint32_t rsvd1 : 5;
    uint32_t update_protocol : 1;
    uint32_t update_dst_port : 1;
    uint32_t update_src_port : 1;
    uint32_t update_da : 1;
    uint32_t update_sa : 1;
    uint32_t update_row_kind : 1;
    uint32_t update_ethertype : 1;
    uint32_t update_l2_priority : 1;
    uint32_t rsvd2 : 7;
    uint32_t go : 1;
};

struct RxTcamNonIpAddrFlags {
    uint32_t augmented_da : 4;
    uint32_t rsvd0 : 12;
    uint32_t augmented_sa : 4;
    uint32_t rsvd1 : 12;
};

struct RxTcamEthertype {
    uint32_t value : 16;
    uint32_t augmented : 4;
    uint32_t rsvd : 12;
};

struct RxTcamPriority {
    uint32_t pcp : 3;
    uint32_t rsvd : 29;
};

struct RxFtableUpdate {
    uint32_t row : 6;
    uint32_t rsvd0 : 2;
    uint32_t write : 1;
    uint32_t rsvd1 : 22;
    uint32_t go : 1;
};

constexpr Reg<RxFlowActions> kRxNoMatchActions = rx_classifier_reg<RxFlowActions>(ETH_RX_CLASSIFIER_NO_MATCH_ACTIONS);
constexpr Reg<RxTcamRowMapping> rx_tcam_row_mapping(uint32_t row) {
    return rx_classifier_reg<RxTcamRowMapping>(ETH_RX_CLASSIFIER_TCAM_ROW_MAPPING, row);
}
constexpr Reg<RxTcamRowUpdate> kRxTcamRowUpdate = rx_classifier_reg<RxTcamRowUpdate>(ETH_RX_CLASSIFIER_TCAM_ROW_UPDATE);
constexpr Reg<RxTcamTupleType> kRxTcamTupleTypeWrite =
    rx_classifier_reg<RxTcamTupleType>(ETH_RX_CLASSIFIER_TCAM_TUPLE_TYPE_WRITE);
constexpr Reg<> rx_tcam_sa_write(uint32_t word) { return rx_classifier_reg(ETH_RX_CLASSIFIER_TCAM_SA_WRITE, word); }
constexpr Reg<> rx_tcam_da_write(uint32_t word) { return rx_classifier_reg(ETH_RX_CLASSIFIER_TCAM_DA_WRITE, word); }
constexpr Reg<RxTcamNonIpAddrFlags> kRxTcamNonIpAddrFlagsWrite =
    rx_classifier_reg<RxTcamNonIpAddrFlags>(ETH_RX_CLASSIFIER_TCAM_NON_IP_ADDR_FLAGS_WRITE);
constexpr Reg<RxTcamEthertype> kRxTcamEthertypeWrite =
    rx_classifier_reg<RxTcamEthertype>(ETH_RX_CLASSIFIER_TCAM_ETHERTYPE_WRITE);
constexpr Reg<RxTcamPriority> kRxTcamPriorityWrite =
    rx_classifier_reg<RxTcamPriority>(ETH_RX_CLASSIFIER_TCAM_PRIORITY_WRITE);
constexpr Reg<RxTcamUpdate> kRxTcamUpdate = rx_classifier_reg<RxTcamUpdate>(ETH_RX_CLASSIFIER_TCAM_UPDATE);
constexpr Reg<> kRxFtableLabel = rx_classifier_reg(ETH_RX_CLASSIFIER_FTABLE_LABELS);
constexpr Reg<RxFlowActions> kRxFtableActions = rx_classifier_reg<RxFlowActions>(ETH_RX_CLASSIFIER_FTABLE_ACTIONS);
constexpr Reg<> kRxFtableVlan = rx_classifier_reg(ETH_RX_CLASSIFIER_FTABLE_VLAN);
constexpr Reg<> kRxFtableSwMetadata = rx_classifier_reg(ETH_RX_CLASSIFIER_FTABLE_SW_METADATA);
constexpr Reg<RxFtableUpdate> kRxFtableUpdate = rx_classifier_reg<RxFtableUpdate>(ETH_RX_CLASSIFIER_FTABLE_UPDATE);

// A non-IP TCAM pattern. The address fields are four words wide, the classifier's IPv6 width. The firmware sends only
// broadcast, 01:00:.. multicast and 02:00:.. unicast frames.
struct RxTcamNonIpPattern {
    uint32_t sa[4];
    uint32_t da[4];
    RxTcamNonIpAddrFlags addr_flags;
    RxTcamEthertype ethertype;
    RxTcamPriority priority;
};

// ---------------------------------------------------------------------------------------------------------------------
// The RX timestamp FIFO: each entry is a received frame's arrival time and the label of the rule that recorded it

struct RxStampLabel {
    uint32_t label : 5;
    uint32_t recorded_by_rule : 1;  // set in every entry a rule records (measured), so not part of the label
    uint32_t rsvd : 25;
    uint32_t valid : 1;
};
struct RxStampStatus {
    uint32_t entries : 4;
    uint32_t rsvd0 : 12;
    uint32_t full : 1;
    uint32_t nearly_full : 1;
    uint32_t empty : 1;
    uint32_t rsvd1 : 13;
};
// The same register as RxStampStatus, written.
struct RxStampCommand {
    uint32_t rsvd : 30;
    uint32_t flush : 1;
    uint32_t pop : 1;
};

constexpr Reg<> kRxStampTimeLo = rx_stamp_reg(ETH_RX_TH_TS_LOW);
constexpr Reg<> kRxStampTimeHi = rx_stamp_reg(ETH_RX_TH_TS_HIGH);
constexpr Reg<RxStampLabel> kRxStampLabel = rx_stamp_reg<RxStampLabel>(ETH_RX_TH_TS_LABEL);
constexpr Reg<RxStampStatus> kRxStampStatus = rx_stamp_reg<RxStampStatus>(ETH_RX_TH_STATUS);
constexpr Reg<RxStampCommand> kRxStampCommand = rx_stamp_reg<RxStampCommand>(ETH_RX_TH_STATUS);

// A frame's time is recorded as it starts to arrive, so once its data is visible in L1, its time is already waiting.
struct RxStampFifo {
    static constexpr uint32_t kDepth = 16;
    struct Entry {
        uint64_t time_ns;
        uint32_t label;
        bool valid;
    };
    // The head can be read before it is removed, and removing it takes a pop of 1, then of 0 (measured).
    static FORCE_INLINE bool pop(Entry& entry) {
        if (kRxStampStatus.read().empty) {
            return false;
        }
        const uint32_t lo = kRxStampTimeLo.read();
        const uint32_t hi = kRxStampTimeHi.read();
        const RxStampLabel label = kRxStampLabel.read();
        kRxStampCommand.write({.pop = 1});
        kRxStampCommand.write({});
        entry.time_ns = (static_cast<uint64_t>(hi) << 32) | lo;
        entry.label = label.label;
        entry.valid = label.valid;
        return true;
    }
    // Exactly `count` entries, and not full at any point since the last pop.
    static FORCE_INLINE bool holds_exactly(uint32_t count) {
        constexpr RxStampStatus kCounted{.entries = 0xF, .full = 1};
        return (word_of(kRxStampStatus.read()) & word_of(kCounted)) == count;
    }
    static FORCE_INLINE void flush() { kRxStampCommand.write({.flush = 1}); }
    // Pops up to kDepth entries and hands each of `label`'s arrival times to `sink`; returns how many it handed over.
    template <uint32_t Label, typename Sink>
    static FORCE_INLINE uint32_t drain(Sink&& sink) {
        Entry entry;
        uint32_t matched = 0;
        for (uint32_t i = 0; i < kDepth && pop(entry); i++) {
            if (entry.valid && entry.label == Label) {
                sink(entry.time_ns);
                matched++;
            }
        }
        return matched;
    }
};

// Records the arrival time of every frame that matches `values` where `mask` is 0, under kLabel, with TCAM row Row
// and its flow-table row.
template <uint32_t Row, uint32_t Label>
class RxStampRule {
public:
    static_assert(Label < 32);  // flow-table labels are five bits
    static constexpr uint32_t kLabel = Label;

    // Also flushes the FIFO and stops unmatched frames recording their times.
    __attribute__((noinline, cold)) void install(const RxTcamNonIpPattern& values, const RxTcamNonIpPattern& mask) {
        no_match_before_ = kRxNoMatchActions.read();
        RxStampFifo::flush();
        RxFlowActions no_match = no_match_before_;
        no_match.record_rx_time = 0;
        kRxNoMatchActions.write(no_match);
        write_tcam_row(values, false);
        write_tcam_row(mask, true);
        rx_tcam_row_mapping(Row).write({.priority = kRxTcamTopPriority, .ftable_row = Row});
        write_flow({.record_rx_time = 1}, Label);
        kRxTcamRowUpdate.write({.row = Row, .enable = 1, .write = 1, .go = 1});
    }
    // Disables the row, empties its flow-table row, restores the unmatched frames' actions and flushes the FIFO.
    __attribute__((noinline, cold)) void remove() const {
        kRxTcamRowUpdate.write({.row = Row, .write = 1, .go = 1});
        write_flow({}, 0);
        kRxNoMatchActions.write(no_match_before_);
        RxStampFifo::flush();
    }

private:
    FORCE_INLINE static void write_tcam_row(const RxTcamNonIpPattern& pattern, bool mask) {
        kRxTcamTupleTypeWrite.write(RxTcamTupleType::NonIp);
        kRxTcamEthertypeWrite.write(pattern.ethertype);
        kRxTcamPriorityWrite.write(pattern.priority);
        for (uint32_t word = 0; word < 4; word++) {
            rx_tcam_sa_write(word).write(pattern.sa[word]);
        }
        for (uint32_t word = 0; word < 4; word++) {
            rx_tcam_da_write(word).write(pattern.da[word]);
        }
        kRxTcamNonIpAddrFlagsWrite.write(pattern.addr_flags);
        kRxTcamUpdate.write(
            {.row = Row,
             .mask = mask,
             .write = 1,
             .not_ip = 1,
             .update_da = 1,
             .update_sa = 1,
             .update_row_kind = 1,
             .update_ethertype = 1,
             .update_l2_priority = 1,
             .go = 1});
    }
    FORCE_INLINE static void write_flow(RxFlowActions actions, uint32_t label) {
        kRxFtableActions.write(actions);
        kRxFtableVlan.write(0);
        kRxFtableLabel.write(label);
        kRxFtableSwMetadata.write(0);
        kRxFtableUpdate.write({.row = Row, .write = 1, .go = 1});
    }

    RxFlowActions no_match_before_{};
};

// ---------------------------------------------------------------------------------------------------------------------
// The MAC: send-time stamping, and the TX timestamp FIFO a two-step stamp lands in

struct MacTxCfg {
    uint32_t rsvd0 : 1;
    uint32_t tx_enable : 1;
    uint32_t rsvd1 : 2;
    uint32_t stamp_fifo_every_packet : 1;  // 0: only packets armed for a two-step stamp
    uint32_t rsvd2 : 6;
    uint32_t origin_timestamp_mode : 1;
    uint32_t rsvd3 : 20;
};
struct MacTxInt {
    uint32_t rsvd0 : 2;
    uint32_t stamp_fifo_full : 1;
    uint32_t stamp_fifo_not_empty : 1;
    uint32_t stamp_offset_error : 1;
    uint32_t rsvd1 : 27;
};

constexpr Reg<MacTxCfg> kMacTxCfg = mac_reg<MacTxCfg>(ETH_MAC_TX_CFG);
constexpr Reg<> kMacTxDelay = mac_reg(ETH_MAC_TX_DELAY);  // PTP ns added to every send-time stamp
constexpr Reg<MacTxInt> kMacTxInt = mac_reg<MacTxInt>(ETH_MAC_TX_INT);
constexpr Reg<MacTxInt> kMacTxIntRaw = mac_reg<MacTxInt>(ETH_MAC_TX_INT_RAW);
constexpr Reg<> kMacTxStampFifoFullThreshold = mac_reg(ETH_MAC_TS_FIFO_FULL_THRESH);
// An entry's four words are its tag's low and high words and then its time's, not the order the vendor's guide gives
// (measured). Reading the first pops the entry, and an empty FIFO reads all ones there.
constexpr Reg<> kMacTxStampTagLo = mac_reg(ETH_MAC_TS_FIFO_0);
constexpr Reg<> kMacTxStampTagHi = mac_reg(ETH_MAC_TS_FIFO_1);
constexpr Reg<> kMacTxStampTimeLo = mac_reg(ETH_MAC_TS_FIFO_2);
constexpr Reg<> kMacTxStampTimeHi = mac_reg(ETH_MAC_TS_FIFO_3);

struct TxStampFifo {
    static constexpr uint32_t kEmptyTagLo = 0xFFFFFFFFu;
    // A two-step stamp's frame is told by its tag's low word, so pop() reads only that half of the tag.
    struct Entry {
        uint32_t tag_lo;
        uint64_t time_ns;
    };
    static FORCE_INLINE bool pop(Entry& entry) {
        const uint32_t tag_lo = kMacTxStampTagLo.read();
        if (tag_lo == kEmptyTagLo) {
            return false;
        }
        const uint32_t time_lo = kMacTxStampTimeLo.read();
        const uint32_t time_hi = kMacTxStampTimeHi.read();
        entry.tag_lo = tag_lo;
        entry.time_ns = (static_cast<uint64_t>(time_hi) << 32) | time_lo;
        return true;
    }
    static FORCE_INLINE bool empty() { return !kMacTxIntRaw.read().stamp_fifo_not_empty; }
    __attribute__((noinline, cold)) static void clear() {
        Entry entry;
        while (pop(entry)) {
        }
    }
    // Pops every entry and hands the send time of each tagged `tag_lo` to `sink`; returns how many it handed over.
    template <typename Sink>
    static FORCE_INLINE uint32_t drain(uint32_t tag_lo, Sink&& sink) {
        Entry entry;
        uint32_t matched = 0;
        while (pop(entry)) {
            if (entry.tag_lo == tag_lo) {
                sink(entry.time_ns);
                matched++;
            }
        }
        return matched;
    }
};

}  // namespace tt::tt_metal::eth_ptp
