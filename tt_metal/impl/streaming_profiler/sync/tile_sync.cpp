// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "impl/streaming_profiler/sync/tile_sync.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <map>
#include <optional>
#include <set>
#include <string_view>

#include <tt-metalium/allocator.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/program.hpp>
#include <tt-metalium/tt_metal.hpp>

#include "context/metal_context.hpp"
#include "hostdev/streaming_profiler_common.h"
#include "impl/kernels/kernel.hpp"
#include "impl/streaming_profiler/device_programs.hpp"
#include "impl/streaming_profiler/sync/link_sync.hpp"
#include "impl/streaming_profiler/service.hpp"
#include "impl/streaming_profiler/sync/engine.hpp"
#include "llrt/hal.hpp"
#include "llrt/metal_soc_descriptor.hpp"
#include "llrt/tt_cluster.hpp"

namespace tt::tt_metal::streaming_profiler {

int64_t TileClocks::offset(CoreType type, const CoreCoord& logical) const {
    const auto it = std::ranges::find_if(
        tiles, [&](const TileClock& tile) { return tile.type == type && tile.logical == logical; });
    TT_FATAL(
        it != tiles.end(), "streaming profiler: tile ({},{}) is not in its chip's tile clocks", logical.x, logical.y);
    return it->offset;
}

namespace {

constexpr uint32_t kReps = 256;

// A reader's reads of one partner over one NoC. median2 is the median, over the reads, of 2 * (partner wall - bracket
// midpoint), in the clocks' low words. doubled_offset() is that widened by coarse, the whole-clock difference.
struct Reading {
    uint32_t partner, noc;
    // On the both-NoC reader's extra reads, the index of its read of the same partner over the other NoC.
    std::optional<uint32_t> other_noc_read;
    int32_t median2 = 0;
    int64_t coarse = 0;
    int64_t doubled_offset() const {
        return 2 * coarse + static_cast<int32_t>(static_cast<uint32_t>(median2) - static_cast<uint32_t>(2 * coarse));
    }
};

// A mirrored pair's doubled offsets differ by, and a both-NoC pair's add up to, four times the tiles' offset.
constexpr double kDoubledPairPerTick = 4.0;

struct Tile : CoreCoords {
    CoreType type;
    HalProgrammableCoreType core;
    uint32_t scratch = 0;
    uint64_t host_scratch = 0;
    std::vector<Reading> reads;
};

// NoC 0 runs towards higher raw coordinates, NoC 1 towards lower.
bool aligned(const CoreCoord& reader, const CoreCoord& partner) {
    return reader != partner && (reader.x == partner.x || reader.y == partner.y);
}
bool upward(const CoreCoord& reader, const CoreCoord& partner) {
    return reader.x == partner.x ? partner.y > reader.y : partner.x > reader.x;
}

std::string_view kind_name(CoreType type) {
    return type == CoreType::WORKER ? "Tensix" : type == CoreType::DRAM ? "DRAM" : "eth";
}

// A profiler ring can't be scratch, even zeroed afterwards: the first frames depend on its contents.
std::vector<Tile> plan_tiles(IDevice* device, ContextId ctx) {
    auto& mc = MetalContext::instance(ctx);
    auto& cluster = mc.get_cluster();
    const auto& hal = mc.hal();
    const uint32_t chip = static_cast<uint32_t>(device->id());
    const auto& soc = cluster.get_soc_desc(chip);
    std::vector<Tile> tiles;
    const auto add = [&](CoreType type, HalProgrammableCoreType core, const CoreCoord& logical, uint32_t scratch) {
        const uint64_t host = core == HalProgrammableCoreType::DRAM
                                  ? hal.get_dev_noc_addr(core, HalL1MemAddrType::UNRESERVED) +
                                        (scratch - hal.get_dev_addr(core, HalL1MemAddrType::UNRESERVED))
                                  : scratch;
        tiles.push_back(Tile{{locate(cluster, chip, logical, type)}, type, core, scratch, host});
    };
    const auto add_unreserved = [&](CoreType type, HalProgrammableCoreType core, const CoreCoord& logical) {
        add(type, core, logical, hal.get_dev_addr(core, HalL1MemAddrType::UNRESERVED));
    };
    const uint32_t user_l1 = static_cast<uint32_t>(device->allocator()->get_base_allocator_addr(HalMemType::L1));
    TT_FATAL(
        hal.get_dev_size(HalProgrammableCoreType::TENSIX, HalL1MemAddrType::PROFILER) >=
            kernel_profiler::PROFILER_L1_CONTROL_BUFFER_SIZE + sizeof(kernel_profiler::TileNetScratch),
        "streaming profiler: a profiler L1 region cannot hold the tile clock scratch");
    const uint32_t dispatch_scratch =
        static_cast<uint32_t>(hal.get_dev_addr(HalProgrammableCoreType::TENSIX, HalL1MemAddrType::PROFILER)) +
        kernel_profiler::PROFILER_L1_CONTROL_BUFFER_SIZE;
    const CoreCoord compute = device->compute_with_storage_grid_size();
    const CoreCoord grid = soc.get_grid_size(CoreType::TENSIX);
    for (uint32_t y = 0; y < grid.y; y++) {
        for (uint32_t x = 0; x < grid.x; x++) {
            const bool is_compute = x < compute.x && y < compute.y;
            add(CoreType::WORKER, HalProgrammableCoreType::TENSIX, {x, y}, is_compute ? user_l1 : dispatch_scratch);
        }
    }
    if (hal.has_programmable_core_type(HalProgrammableCoreType::IDLE_ETH)) {
        for (const CoreCoord& logical : sorted_yx(device->get_inactive_ethernet_cores())) {
            add_unreserved(CoreType::ETH, HalProgrammableCoreType::IDLE_ETH, logical);
        }
    }
    if (hal.has_programmable_core_type(HalProgrammableCoreType::ACTIVE_ETH)) {
        for (const CoreCoord& logical :
             sorted_yx(device->get_active_ethernet_cores(/*skip_reserved_tunnel_cores=*/false))) {
            add_unreserved(CoreType::ETH, HalProgrammableCoreType::ACTIVE_ETH, logical);
        }
    }
    if (hal.has_programmable_core_type(HalProgrammableCoreType::DRAM)) {
        // The cluster gives a DRAM core's translated coordinate as its physical one. The NoC 0 grid position, which
        // says which Tensix row it shares, comes from the SoC descriptor in the same order.
        const std::vector<CoreCoord> logical = soc.get_metal_dram_cores(CoordSystem::LOGICAL);
        const std::vector<CoreCoord> noc0 = soc.get_metal_dram_cores(CoordSystem::NOC0);
        for (size_t i = 0; i < logical.size(); i++) {
            if (dram_view_endpoint_noc_mask(soc, logical[i]) == 0) {
                add_unreserved(CoreType::DRAM, HalProgrammableCoreType::DRAM, logical[i]);
                tiles.back().phys = noc0[i];
            }
        }
    }
    // One idle eth tile also reads every eth tile in its row over the other NoC. Around the ring the two NoCs' hops
    // cancel, and the initiator's own latency is the same for every target.
    const auto both_noc_reader =
        static_cast<uint32_t>(std::ranges::find(tiles, HalProgrammableCoreType::IDLE_ETH, &Tile::core) - tiles.begin());
    for (uint32_t r = 0; r < tiles.size(); r++) {
        Tile& reader = tiles[r];
        for (uint32_t partner = 0; partner < tiles.size(); partner++) {
            if (aligned(reader.phys, tiles[partner].phys)) {
                reader.reads.push_back(Reading{partner, upward(reader.phys, tiles[partner].phys) ? 0u : 1u});
            }
        }
        if (r == both_noc_reader) {
            for (uint32_t i = 0, direct_reads = static_cast<uint32_t>(reader.reads.size()); i < direct_reads; i++) {
                const Reading& direct = reader.reads[i];
                if (tiles[direct.partner].type == CoreType::ETH) {
                    reader.reads.push_back(Reading{direct.partner, direct.noc ^ 1u, i});
                }
            }
        }
        TT_FATAL(
            reader.reads.size() <= kernel_profiler::kTileNetMaxPartners,
            "streaming profiler: tile ({},{}) has {} row and column partners, the table holds {}",
            reader.logical.x,
            reader.logical.y,
            reader.reads.size(),
            kernel_profiler::kTileNetMaxPartners);
    }
    return tiles;
}

KernelHandle create_tile_kernel(Program& program, HalProgrammableCoreType core, const CoreRangeSet& cores) {
    const char* src = "tt_metal/impl/streaming_profiler/kernels/tile_sync.cpp";
    switch (core) {
        case HalProgrammableCoreType::TENSIX:
            return CreateKernel(
                program,
                src,
                cores,
                DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});
        case HalProgrammableCoreType::IDLE_ETH:
            return CreateKernel(
                program, src, cores, EthernetConfig{.eth_mode = Eth::IDLE, .noc = NOC::RISCV_0_default});
        case HalProgrammableCoreType::ACTIVE_ETH:
            return CreateKernel(program, src, cores, EthernetConfig{.noc = NOC::RISCV_0_default});
        case HalProgrammableCoreType::DRAM: return CreateKernel(program, src, cores, DramConfig{.noc = NOC::NOC_0});
        case HalProgrammableCoreType::DISPATCH:
        case HalProgrammableCoreType::COUNT: break;
    }
    TT_THROW("Unreachable");
}

// If a later step throws, the kernels stay up waiting on their go words.
void read_network(IDevice* device, ContextId ctx, std::vector<Tile>& tiles) {
    auto& mc = MetalContext::instance(ctx);
    auto& cluster = mc.get_cluster();
    const auto& hal = mc.hal();
    const uint32_t chip = static_cast<uint32_t>(device->id());
    // Neither the nonce nor its inverse may read as a zeroed table or a go word.
    constexpr auto kLastGo = static_cast<uint32_t>(kernel_profiler::TileNetGo::Exit);
    auto nonce = static_cast<uint32_t>(std::chrono::steady_clock::now().time_since_epoch().count());
    while (nonce <= kLastGo || ~nonce <= kLastGo) {
        nonce++;
    }
    struct Kind {
        std::set<CoreRange> cores;
        std::optional<Program> program;
        KernelHandle kernel = 0;
    };
    std::map<HalProgrammableCoreType, Kind> kinds;
    for (const Tile& tile : tiles) {
        kinds[tile.core].cores.insert(CoreRange(tile.logical, tile.logical));
    }
    for (auto& [core, kind] : kinds) {
        kind.program = CreateProgram();
        kind.kernel = create_tile_kernel(*kind.program, core, CoreRangeSet(kind.cores));
    }
    const auto table_of = [](const Tile& tile) {
        return tile.host_scratch + offsetof(kernel_profiler::TileNetScratch, table);
    };
    // Firmware takes its ring position from the control vector at the session's first launch (this one), so a stale
    // tail would put it thousands of words ahead.
    for (const Tile& tile : tiles) {
        std::vector<uint32_t> args{tile.scratch, kReps, nonce, static_cast<uint32_t>(tile.reads.size())};
        for (const Reading& read : tile.reads) {
            const CoreCoord& partner = tiles[read.partner].virt;
            args.push_back(kernel_profiler::word_of(kernel_profiler::TileNetRead{
                .x = static_cast<uint32_t>(partner.x), .y = static_cast<uint32_t>(partner.y), .noc = read.noc}));
        }
        const Kind& kind = kinds[tile.core];
        SetRuntimeArgs(*kind.program, kind.kernel, tile.logical, args);
        zero_l1(cluster, chip, tile.virt, table_of(tile), 2 * sizeof(uint32_t));
        zero_profiler_control(cluster, chip, tile.virt, host_l1_addr(hal, tile.core, HalL1MemAddrType::PROFILER));
    }
    for (auto& [core, kind] : kinds) {
        launch_resident(device, *kind.program);
    }
    for (Tile& tile : tiles) {
        const tt_cxy_pair core(chip, tile.virt);
        kernel_profiler::TileNetTable table{};
        const auto table_bytes = static_cast<uint32_t>(
            offsetof(kernel_profiler::TileNetTable, partner) +
            tile.reads.size() * sizeof(kernel_profiler::TileNetPartner));
        const auto await = [&](uint32_t ready) {
            const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
            do {
                cluster.read_core(&table, table_bytes, core, table_of(tile));
                TT_FATAL(
                    table.ready == ready || std::chrono::steady_clock::now() < deadline,
                    "streaming profiler: device {} tile {} did not post its tile clock table",
                    chip,
                    tile.virt.str());
            } while (table.ready != ready);
        };
        await(nonce);
        const kernel_profiler::TileNetGo go_measure = kernel_profiler::TileNetGo::Measure;
        cluster.write_core(&go_measure, sizeof(go_measure), core, table_of(tile));
        await(~nonce);
        for (size_t i = 0; i < tile.reads.size(); i++) {
            const kernel_profiler::TileNetPartner& posted = table.partner[i];
            tile.reads[i].median2 = posted.median2;
            tile.reads[i].coarse = posted.coarse;
        }
    }
    for (const Tile& tile : tiles) {
        const kernel_profiler::TileNetGo go_exit = kernel_profiler::TileNetGo::Exit;
        cluster.write_core(&go_exit, sizeof(go_exit), tt_cxy_pair(chip, tile.virt), table_of(tile));
    }
    for (auto& [core, kind] : kinds) {
        detail::WaitProgramDone(device, *kind.program, false);
    }
    // Fast dispatch's go signal reaches every core and a host-launched kernel leaves its slot valid, so restore the
    // firmware's initial launch message.
    for (const Tile& tile : tiles) {
        auto msg = hal.get_dev_msgs_factory(tile.core).create<dev_msgs::launch_msg_t>();
        cluster.write_core(
            msg.data(),
            static_cast<uint32_t>(msg.size()),
            tt_cxy_pair(chip, tile.virt),
            host_l1_addr(hal, tile.core, HalL1MemAddrType::LAUNCH));
    }
}

bool is_active_eth(const Tile& tile) { return tile.core == HalProgrammableCoreType::ACTIVE_ETH; }

// plan_tiles lists the Tensix grid first, so tile 0 is its first Tensix tile, which every other tile is placed from.
constexpr uint32_t kGroundTile = 0;

// Every tile but the active eth ones, from the mirrored pairs, as ticks from the first Tensix tile. An active eth
// tile's reads leave its NIU about 6 cycles later than an idle eth tile's, which would bias a mirrored pair with one by
// a tick and a half.
std::vector<double> place_mirrored(const std::vector<Tile>& tiles, uint32_t chip) {
    const auto tile_count = static_cast<uint32_t>(tiles.size());
    std::vector<std::optional<size_t>> unknown_of(tile_count);
    std::vector<uint32_t> tile_of_unknown;
    for (uint32_t i = 0; i < tile_count; i++) {
        if (i != kGroundTile && !is_active_eth(tiles[i])) {
            unknown_of[i] = tile_of_unknown.size();
            tile_of_unknown.push_back(i);
        }
    }
    std::vector<PotentialEdge> edges;
    std::vector<double> offsets;
    for (uint32_t reader = 0; reader < tile_count; reader++) {
        for (const Reading& there : tiles[reader].reads) {
            if (there.other_noc_read || there.noc != 0 || is_active_eth(tiles[reader]) ||
                is_active_eth(tiles[there.partner])) {
                continue;
            }
            const Reading& back = *std::ranges::find(tiles[there.partner].reads, reader, &Reading::partner);
            edges.push_back({.sender = unknown_of[there.partner], .receiver = unknown_of[reader]});
            offsets.push_back(
                static_cast<double>(there.doubled_offset() - back.doubled_offset()) / kDoubledPairPerTick);
        }
    }
    const Potential potential = solve_potential(edges, offsets, tile_of_unknown.size());
    TT_FATAL(
        !potential.unreached,
        "streaming profiler: device {} {} tile ({},{}) has no chain of mirrored pairs to the first Tensix tile",
        chip,
        kind_name(tiles[tile_of_unknown[*potential.unreached]].type),
        tiles[tile_of_unknown[*potential.unreached]].logical.x,
        tiles[tile_of_unknown[*potential.unreached]].logical.y);
    std::vector<double> ticks(tile_count, 0.0);
    for (size_t u = 0; u < tile_of_unknown.size(); u++) {
        ticks[tile_of_unknown[u]] = potential.x[u];
    }
    return ticks;
}

// Each active eth tile from its both-NoC reading, shifted onto the mirrored placement by the idle eth tiles that have
// both-NoC readings too.
void place_active_eth(const std::vector<Tile>& tiles, uint32_t chip, std::vector<double>& ticks) {
    std::vector<std::optional<double>> estimate(tiles.size());
    double shift = 0.0;
    size_t refs = 0;
    for (const Tile& reader : tiles) {
        for (const Reading& read : reader.reads) {
            if (!read.other_noc_read) {
                continue;
            }
            estimate[read.partner] =
                static_cast<double>(reader.reads[*read.other_noc_read].doubled_offset() + read.doubled_offset()) /
                kDoubledPairPerTick;
            if (!is_active_eth(tiles[read.partner])) {
                shift += ticks[read.partner] - *estimate[read.partner];
                refs++;
            }
        }
    }
    for (uint32_t t = 0; t < tiles.size(); t++) {
        if (!is_active_eth(tiles[t])) {
            continue;
        }
        TT_FATAL(
            refs != 0 && estimate[t],
            "streaming profiler: device {} active eth tile ({},{}) has no both-NoC reading to place it from",
            chip,
            tiles[t].logical.x,
            tiles[t].logical.y);
        ticks[t] = shift / static_cast<double>(refs) + *estimate[t];
    }
}

}  // namespace

void measure_tile_clocks(IDevice* device, ContextId ctx) {
    const auto& mc = MetalContext::instance(ctx);
    const uint32_t chip = static_cast<uint32_t>(device->id());
    if (!can_capture(mc) || service().tile_clocks(ctx, chip) != nullptr) {
        return;
    }
    std::vector<Tile> tiles = plan_tiles(device, ctx);
    read_network(device, ctx, tiles);
    std::vector<double> ticks = place_mirrored(tiles, chip);
    place_active_eth(tiles, chip, ticks);
    TileClocks clocks;
    for (size_t i = 0; i < tiles.size(); i++) {
        clocks.tiles.push_back(TileClock{tiles[i].type, tiles[i].logical, std::llround(ticks[i])});
    }
    service().set_tile_clocks(ctx, chip, std::move(clocks));
}

}  // namespace tt::tt_metal::streaming_profiler
