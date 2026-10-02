// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstdint>
#include <vector>

#include <tt-metalium/core_coord.hpp>
#include "hostdev/streaming_profiler_common.h"

namespace tt::tt_metal {

class Hal;
class MetalContext;

namespace streaming_profiler {

bool can_capture(const MetalContext& mc);

// Without fabric the profiler runs the ends as resident kernels; with fabric the routers on the chosen links run them
// under LINK_SYNC_ROLE in fabric_erisc_router.cpp.
namespace link_sync {

struct Link {
    uint32_t chip_a = 0, chip_b = 0;  // chip_a < chip_b; chip_a's end sends
    CoreCoord eth_a, eth_b;
};
// Every connected eth pair between the two chips or, with fabric on, only those on the fabric's active channels.
std::vector<Link> links_between(MetalContext& mc, uint32_t chip_x, uint32_t chip_y);

kernel_profiler::LinkSyncRole role_of(MetalContext& mc, uint32_t chip, const CoreCoord& eth_logical);
// The link end's kernel_profiler::LinkSyncL1 on every active eth core: the top of ACTIVE_ETH UNRESERVED, the same place
// whether a resident kernel or a router runs the end.
uint32_t l1_addr(const Hal& hal);

}  // namespace link_sync

}  // namespace streaming_profiler
}  // namespace tt::tt_metal
