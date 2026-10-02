// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

#include <tt-metalium/experimental/streaming_profiler.hpp>

namespace tt::tt_metal::streaming_profiler {

inline constexpr size_t kProcessorCount = static_cast<size_t>(experimental::streaming_profiler::Processor::ERISC1) + 1;
inline constexpr std::array<const char*, kProcessorCount> kProcessorNames = {
    "BRISC", "NCRISC", "TRISC_0", "TRISC_1", "TRISC_2", "ERISC_0", "ERISC_1"};

// Immutable once the receiver starts.
struct CaptureContext {
    struct Device {
        struct Tile {
            uint32_t xy = 0;  // kernel_profiler::NocXy
            // The tracker's wall tick minus this core's. Every tile keeps its own wall clock on the one AICLK, so it's
            // one integer for the capture.
            int64_t clock_offset = 0;
        };
        // profiler::kSpscNRiscDecode lanes per tile, in tile order and RISC order; an eth tile fills its unused lanes
        // with ERISC1.
        std::vector<experimental::streaming_profiler::Core> lanes;
        std::vector<Tile> tiles;  // by core index
        uint32_t chip_id = 0;
        // The tracker's wall tick minus the sync check ruler's, or 0 without a ruler.
        int64_t ruler_offset = 0;
    };
    std::vector<Device> devices;
    struct Link {
        uint32_t dev_a = 0, dev_b = 0;
        uint32_t core_a = 0, core_b = 0;
        CoreCoord eth_a, eth_b;
    };
    std::vector<Link> links;
    bool sync_check = false;
};

}  // namespace tt::tt_metal::streaming_profiler
