// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Compute load for the sync workloads: bursts of matmul blocks, each followed by an idle spin, so the chip's current
// steps up and down (di/dt) and throttling moves AICLK.

#include <cstdint>
#include "api/compute/matmul.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "risc_common.h"

constexpr uint32_t kMatmulsPerBlock = 8;

void kernel_main() {
    const uint32_t bursts = get_arg_val<uint32_t>(0);
    const uint32_t iters = get_arg_val<uint32_t>(1);
    const uint32_t idle = get_arg_val<uint32_t>(2);
    volatile uint32_t* const wall_clock = reinterpret_cast<volatile uint32_t*>(RISCV_DEBUG_REG_WALL_CLOCK_L);
    constexpr auto in0 = tt::CBIndex::c_0;
    constexpr auto in1 = tt::CBIndex::c_1;
    constexpr auto out = tt::CBIndex::c_16;
    compute_kernel_hw_startup<SrcOrder::Reverse>(in0, in1, out);
    matmul_init(in0, in1);
    for (uint32_t burst = 0; burst < bursts; burst++) {
        for (uint32_t i = 0; i < iters; i++) {
            tile_regs_acquire();
            for (uint32_t j = 0; j < kMatmulsPerBlock; j++) {
                matmul_tiles(in0, in1, 0, 0, 0);
            }
            tile_regs_commit();
            tile_regs_wait();
            tile_regs_release();
        }
        for (const uint32_t idle_start = *wall_clock; *wall_clock - idle_start < idle;) {
        }
    }
}
