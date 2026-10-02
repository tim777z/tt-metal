// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "impl/streaming_profiler/sync/link_sync.hpp"

#include <algorithm>
#include <tuple>
#include <vector>

#include <tt-metalium/experimental/fabric/control_plane.hpp>
#include <tt-metalium/experimental/fabric/fabric_types.hpp>
#include <umd/device/types/core_coordinates.hpp>

#include "context/metal_context.hpp"
#include "llrt/metal_soc_descriptor.hpp"
#include "llrt/tt_cluster.hpp"

namespace tt::tt_metal::streaming_profiler {

bool can_capture(const MetalContext& mc) {
    return mc.rtoptions().get_streaming_profiler_enabled() && mc.hal().get_arch() == tt::ARCH::BLACKHOLE &&
           mc.hal().has_programmable_core_type(HalProgrammableCoreType::DRAM);
}

namespace link_sync {

namespace {

bool eligible(MetalContext& mc, uint32_t chip, const CoreCoord& eth_logical) {
    if (mc.get_fabric_config() == tt_fabric::FabricConfig::DISABLED) {
        return true;
    }
    const auto& control_plane = mc.get_control_plane();
    const auto& soc = mc.get_cluster().get_soc_desc(static_cast<ChipId>(chip));
    return std::ranges::any_of(
        control_plane.get_active_fabric_eth_channels(
            control_plane.get_fabric_node_id_from_physical_chip_id(static_cast<ChipId>(chip))),
        [&](const auto& chan_dir) {
            return soc.get_eth_core_for_channel(chan_dir.first, CoordSystem::LOGICAL) == eth_logical;
        });
}

}  // namespace

std::vector<Link> links_between(MetalContext& mc, uint32_t chip_x, uint32_t chip_y) {
    std::vector<Link> out;
    const uint32_t lo = std::min(chip_x, chip_y), hi = std::max(chip_x, chip_y);
    const auto& cluster = mc.get_cluster();
    const auto by_peer = cluster.get_ethernet_cores_grouped_by_connected_chips(static_cast<ChipId>(lo));
    const auto it = by_peer.find(static_cast<ChipId>(hi));
    if (it == by_peer.end()) {
        return out;
    }
    for (const CoreCoord& eth_a : it->second) {
        const CoreCoord eth_b =
            std::get<1>(cluster.get_connected_ethernet_core(std::make_tuple(static_cast<ChipId>(lo), eth_a)));
        if (eligible(mc, lo, eth_a) && eligible(mc, hi, eth_b)) {
            out.push_back(Link{.chip_a = lo, .chip_b = hi, .eth_a = eth_a, .eth_b = eth_b});
        }
    }
    return out;
}

kernel_profiler::LinkSyncRole role_of(MetalContext& mc, uint32_t chip, const CoreCoord& eth_logical) {
    for (const auto& [peer, cores] :
         mc.get_cluster().get_ethernet_cores_grouped_by_connected_chips(static_cast<ChipId>(chip))) {
        if (std::ranges::find(cores, eth_logical) == cores.end()) {
            continue;
        }
        for (const Link& link : links_between(mc, chip, static_cast<uint32_t>(peer))) {
            if (link.chip_a == chip && link.eth_a == eth_logical) {
                return kernel_profiler::LinkSyncRole::Sender;
            }
            if (link.chip_b == chip && link.eth_b == eth_logical) {
                return kernel_profiler::LinkSyncRole::Receiver;
            }
        }
    }
    return kernel_profiler::LinkSyncRole::None;
}

uint32_t l1_addr(const Hal& hal) {
    return hal.get_dev_addr(HalProgrammableCoreType::ACTIVE_ETH, HalL1MemAddrType::UNRESERVED) +
           hal.get_dev_size(HalProgrammableCoreType::ACTIVE_ETH, HalL1MemAddrType::UNRESERVED) -
           sizeof(kernel_profiler::LinkSyncL1);
}

}  // namespace link_sync

}  // namespace tt::tt_metal::streaming_profiler
