// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// NoC load for the sync workloads: bursts of writes to and reads from partner cores, each followed by an idle spin.

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "hostdev/streaming_profiler_common.h"

constexpr uint32_t kFirstPartnerArg = 6;

void kernel_main() {
    const uint32_t scratch = get_arg_val<uint32_t>(0);
    const uint32_t bytes = get_arg_val<uint32_t>(1);
    const uint32_t bursts = get_arg_val<uint32_t>(2);
    const uint32_t iters = get_arg_val<uint32_t>(3);
    const uint32_t idle = get_arg_val<uint32_t>(4);
    const uint32_t num_partners = get_arg_val<uint32_t>(5);
    const auto partner = [&](uint32_t i, uint32_t addr) {
        const auto xy = kernel_profiler::word_as<kernel_profiler::NocXy>(
            get_arg_val<uint32_t>(kFirstPartnerArg + i % num_partners));
        return get_noc_addr(xy.x, xy.y, addr);
    };
    const uint32_t landing = scratch + (2 + noc_index) * bytes;
    volatile uint32_t* const wall_clock = reinterpret_cast<volatile uint32_t*>(RISCV_DEBUG_REG_WALL_CLOCK_L);
    for (uint32_t burst = 0; burst < bursts; burst++) {
        for (uint32_t i = 0; i < iters; i++) {
            for (uint32_t j = 0; j < 4; j++) {
                noc_async_write(scratch, partner(4 * i + j, landing), bytes);
                noc_async_read(partner(4 * i + j + 1, scratch), scratch + bytes, bytes);
            }
            noc_async_write_barrier();
            noc_async_read_barrier();
        }
        for (const uint32_t idle_start = *wall_clock; *wall_clock - idle_start < idle;) {
        }
    }
}
