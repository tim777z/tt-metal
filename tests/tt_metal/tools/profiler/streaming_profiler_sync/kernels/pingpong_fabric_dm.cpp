// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// The fabric ping-pong kernel: each round, one end sends an atomic increment over the fabric (PP_TX) and waits for the
// reply, and the other waits (PP_RX on arrival) and replies; the ends swap roles every round.

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "fabric/fabric_edm_packet_header.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/edm_fabric_worker_adapters.hpp"
#include "tt_metal/fabric/hw/inc/noc_addr.h"
#include "tt_metal/fabric/hw/inc/tt_fabric_api.h"
#include "tt_metal/fabric/hw/inc/packet_header_pool.h"
#include "tools/profiler/kernel_profiler.hpp"

constexpr uint32_t kSpinLimit = 1u << 26;

void kernel_main() {
    using namespace tt::tt_fabric;
    size_t arg_index = 0;
    const uint32_t role = get_arg_val<uint32_t>(arg_index++);
    const uint32_t peer_x = get_arg_val<uint32_t>(arg_index++);
    const uint32_t peer_y = get_arg_val<uint32_t>(arg_index++);
    const uint32_t flag_addr = get_arg_val<uint32_t>(arg_index++);
    const uint32_t rounds = get_arg_val<uint32_t>(arg_index++);
    const uint32_t dst_chip = get_arg_val<uint32_t>(arg_index++);
    const uint32_t dst_mesh = get_arg_val<uint32_t>(arg_index++);
    auto conn = WorkerToFabricEdmSender::build_from_args<ProgrammableCoreType::TENSIX>(arg_index);
    conn.open();

    volatile tt_l1_ptr uint32_t* flag = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(flag_addr);
    auto* hdr = PacketHeaderPool::allocate_header();
    fabric_set_unicast_route(
        (HybridMeshPacketHeader*)hdr, static_cast<uint16_t>(dst_chip), static_cast<uint16_t>(dst_mesh));
    hdr->to_noc_unicast_atomic_inc(
        NocUnicastAtomicIncCommandHeader{safe_get_noc_addr(peer_x, peer_y, flag_addr, 0), 1});

    auto wait = [&](uint32_t round) {
        for (uint32_t polls = 0;; polls++) {
            invalidate_l1_cache();
            if (*flag >= round) {
                return true;
            }
            if (polls == kSpinLimit) {
                flag[1] = round;
                return false;
            }
        }
    };
    auto send = [&] { conn.send_payload_flush_non_blocking_from_address((uint32_t)hdr, sizeof(PACKET_HEADER_TYPE)); };
    for (uint32_t round = 1; round <= rounds; round++) {
        if ((round & 1u) == role) {
            conn.wait_for_empty_write_slot();
            {
                DeviceZoneScopedN("PP_TX");
                send();
            }
            if (!wait(round)) {
                break;
            }
            {
                DeviceZoneScopedN("PP_RX");
            }
        } else {
            if (!wait(round)) {
                break;
            }
            {
                DeviceZoneScopedN("PP_RX");
            }
            conn.wait_for_empty_write_slot();
            {
                DeviceZoneScopedN("PP_TX");
                send();
            }
        }
    }
    conn.close();
    noc_async_full_barrier();
}
