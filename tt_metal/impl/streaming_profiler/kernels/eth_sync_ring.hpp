// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <atomic>
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "hostdev/streaming_profiler_common.h"

template <uint32_t CtrlAddr, uint32_t RingAddr>
struct SyncRingWriter {
    uint32_t tail = 0, dropped = 0;
    uint32_t points = 0, from_first[kernel_profiler::kSyncLocalPoints - 1] = {};
    kernel_profiler::SyncMeta meta{};
    kernel_profiler::SyncLocalRates rates{};
    uint64_t first_refclk = 0, first_wall8 = 0;

    static volatile tt_l1_ptr kernel_profiler::RelayCtrl* ctrl() {
        return reinterpret_cast<volatile tt_l1_ptr kernel_profiler::RelayCtrl*>(CtrlAddr);
    }
    void emit() {
        invalidate_l1_cache();
        if (tail - ctrl()->sync_head >= kernel_profiler::kSyncRingRecords) {
            dropped++;
            return;
        }
        volatile tt_l1_ptr kernel_profiler::SyncLocalRecord* record =
            &reinterpret_cast<volatile tt_l1_ptr kernel_profiler::SyncRecord*>(
                 RingAddr)[tail % kernel_profiler::kSyncRingRecords]
                 .local;
        kernel_profiler::SyncMeta counted_meta = meta;
        counted_meta.count = points;
        record->meta = kernel_profiler::word_of(counted_meta);
        record->rates = kernel_profiler::word_of(rates);
        record->first_refclk = first_refclk;
        record->first_wall8 = first_wall8;
        record->from_first[0] = from_first[0];
        record->from_first[1] = from_first[1];
        std::atomic_thread_fence(std::memory_order_release);
        ctrl()->sync_tail = ++tail;
    }
    void flush() {
        if (points != 0) {
            emit();
            points = 0;
        }
    }
    void close() {
        flush();
        ctrl()->dropped_sync = dropped;
        std::atomic_thread_fence(std::memory_order_release);
        ctrl()->done = kernel_profiler::kRelayDoneWord;
    }
    __attribute__((noinline)) void add(
        uint64_t refclk, uint64_t wall8, uint32_t k8, uint32_t base_k8, kernel_profiler::SyncMeta record_meta) {
        if (points != 0) {
            if (kernel_profiler::word_of(record_meta) == kernel_profiler::word_of(meta)) {
                const int32_t wall_off = static_cast<int32_t>(
                    static_cast<uint32_t>(wall8) - static_cast<uint32_t>(first_wall8) -
                    rates.base_k8 * static_cast<uint32_t>(refclk - first_refclk));
                if (kernel_profiler::sync_local_step_fits(refclk - first_refclk, wall_off)) {
                    from_first[points - 1] = kernel_profiler::word_of(kernel_profiler::SyncLocalStep{
                        .refclk_from_first = static_cast<uint32_t>(refclk - first_refclk), .wall_off = wall_off});
                    rates.k8[points] = static_cast<uint8_t>(k8);
                    if (++points == kernel_profiler::kSyncLocalPoints) {
                        flush();
                    }
                    return;
                }
            }
            flush();
        }
        first_refclk = refclk;
        first_wall8 = wall8;
        rates.k8[0] = static_cast<uint8_t>(k8);
        rates.base_k8 = static_cast<uint8_t>(base_k8);
        meta = record_meta;
        points = 1;
    }
};
