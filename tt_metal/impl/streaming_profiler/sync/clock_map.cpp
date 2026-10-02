// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "impl/streaming_profiler/sync/clock_map.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <deque>
#include <limits>
#include <string>
#include <utility>

#include <tt-logger/tt-logger.hpp>
#include <tt_stl/assert.hpp>

#include "impl/streaming_profiler/sync/host_sync.hpp"

namespace tt::tt_metal::streaming_profiler {

template <typename Key>
void ClockSeries<Key>::append(const Node& node) {
    if (node.at <= last.at) {
        return;
    }
    if (!full_warned && nodes.size() == nodes.capacity()) {
        full_warned = true;
        log_warning(
            tt::LogMetal,
            "[streaming profiler] clock sync: {}'s series is full at {} points, so records older than its oldest point "
            "are placed less accurately",
            name,
            nodes.capacity());
    }
    last = node;
    nodes.writer().publish(last);
    extend(node.at);
}
template struct ClockSeries<int64_t>;
template struct ClockSeries<double>;

namespace {

// The index of the first node after `key`, or nullopt if the writer overwrote a node mid-search. The search starts at
// `hint` and doubles its step until it brackets `key`, then bisects: records from interleaved cores land a few nodes
// from the last one, which a search over the whole series would reach through ~20 cold nodes.
template <typename Key>
std::optional<uint64_t> first_after(
    const ClockSeries<Key>& series, uint64_t oldest, uint64_t end, Key key, uint64_t hint) {
    bool overwritten = false;
    const auto after = [&](uint64_t index) {
        typename ClockSeries<Key>::Node node{};
        overwritten = overwritten || !series.nodes.read_at(index, node);
        return node.at > key;
    };
    // Every node before `lo` is at or before `key`; `hi` is the end or a node after it.
    uint64_t lo = oldest, hi = end;
    const uint64_t start = std::clamp(hint, oldest, end - 1);
    if (after(start)) {
        hi = start;
        for (uint64_t step = 1; lo < hi; step *= 2) {
            const uint64_t probe = start - std::min(step, start - oldest);
            if (!after(probe)) {
                lo = probe + 1;
                break;
            }
            hi = probe;
        }
    } else {
        lo = start + 1;
        for (uint64_t step = 1; lo < hi; step *= 2) {
            const uint64_t probe = std::min(start + step, end - 1);
            if (after(probe)) {
                hi = probe;
                break;
            }
            lo = probe + 1;
        }
    }
    while (lo < hi) {
        const uint64_t mid = lo + (hi - lo) / 2;
        if (after(mid)) {
            hi = mid;
        } else {
            lo = mid + 1;
        }
    }
    return overwritten ? std::nullopt : std::optional(lo);
}

// The segment that maps `key`, given `next`, the first node after it; nullopt if the writer overwrote one of its nodes.
template <typename Key>
std::optional<Segment<Key>> segment_at(
    const ClockSeries<Key>& series, uint64_t oldest, uint64_t end, uint64_t next, Key key, Key cover) {
    using Node = typename ClockSeries<Key>::Node;
    Node below{}, above{};
    if ((next > oldest && !series.nodes.read_at(next - 1, below)) ||
        (next < end && !series.nodes.read_at(next, above))) {
        return std::nullopt;
    }
    if (next == oldest) {
        return Segment<Key>{
            .from = std::numeric_limits<Key>::lowest(),
            .to = above.at,
            .origin = above.at,
            .value = above.value,
            .slope = above.tangent,
            .hint = next};
    }
    if (next < end) {
        const double chord = (above.value - below.value) / static_cast<double>(above.at - below.at);
        return Segment<Key>{
            .from = below.at,
            .to = above.at,
            .origin = below.at,
            .value = below.value,
            .slope = series.between == Between::Chord ? chord : below.tangent,
            .hint = next};
    }
    if (key <= cover) {
        return Segment<Key>{
            .from = below.at,
            .to = cover,
            .origin = below.at,
            .value = below.value,
            .slope = below.tangent,
            .hint = next};
    }
    // Only the sync engine's own reads of a still-growing series land past the cover; the segment holds no key, so the
    // next read past the cover searches again.
    return Segment<Key>{.origin = below.at, .value = below.value, .slope = below.tangent, .hint = next};
}

// Maps `key` through the reader's segment, or through the segment found for it. A read that loses to the writer
// retries from the series' new oldest node.
template <typename Key>
std::optional<double> lookup(const ClockSeries<Key>& series, Segment<Key>& seg, Key key) noexcept {
    if (seg.holds(key)) {
        return seg.at(key);
    }
    while (true) {
        const Key cover = series.cover.load(std::memory_order_acquire);
        const uint64_t oldest = series.nodes.oldest();
        const uint64_t end = series.nodes.published();
        if (end == oldest) {
            seg = Segment<Key>{};
            return std::nullopt;
        }
        if (const std::optional<uint64_t> next = first_after(series, oldest, end, key, seg.hint)) {
            if (const std::optional<Segment<Key>> found = segment_at(series, oldest, end, *next, key, cover)) {
                seg = *found;
                return seg.at(key);
            }
        }
    }
}

}  // namespace

ClockMap::ClockMap(size_t devices, uint32_t series_nodes, ClockBases bases) :
    host_(series_nodes, "the host", Between::Tangent), root_base_(bases.root_refclk), tsc_base_(bases.tsc) {
    for (size_t d = 0; d < devices; d++) {
        chips_.emplace_back(series_nodes, "device " + std::to_string(d));
    }
}

ClockMap::Reader ClockMap::reader() const { return Reader(chips_.size(), tsc_base_); }

void ClockMap::append(uint32_t dev, SyncNode node) {
    TT_FATAL(
        std::isfinite(node.value) && std::isfinite(node.tangent) && node.tangent > 0.0,
        "streaming profiler: placement node for device {} at wall {} is not a rate: root {} tangent {}",
        dev,
        node.at,
        node.value,
        node.tangent);
    chips_[dev].append(node);
}

void ClockMap::finish(uint32_t dev) { chips_[dev].extend(std::numeric_limits<int64_t>::max()); }

void ClockMap::append_host(HostNode node, double until) {
    TT_FATAL(
        std::isfinite(node.at) && std::isfinite(node.value) && std::isfinite(node.tangent) && node.tangent > 0.0,
        "streaming profiler: host placement node at refclk {} is not a rate: tsc {} tangent {}",
        node.at,
        node.value,
        node.tangent);
    host_.append(node);
    host_.extend(until);
}

// A chip's records are final once both maps are: up to the chip series' cover, and up to the wall tick whose root time
// is the host series' cover. Past the host cover a record would take whichever node the next burst brings, so two
// records at the same tick could land a line's move apart.
int64_t ClockMap::cover_ticks(uint32_t dev) const noexcept {
    const ClockSeries<int64_t>& chip = chips_[dev];
    const int64_t chip_cover = chip.cover.load(std::memory_order_acquire);
    if (chip_cover == std::numeric_limits<int64_t>::max()) {
        return chip_cover;
    }
    SyncNode newest{};
    const uint64_t end = chip.nodes.published();
    if (end == chip.nodes.oldest() || !chip.nodes.read_at(end - 1, newest)) {
        return chip_cover;
    }
    const double host_cover = host_.cover.load(std::memory_order_acquire);
    const double host_cover_wall = static_cast<double>(newest.at) + (host_cover - newest.value) / newest.tangent;
    if (host_cover_wall >= static_cast<double>(chip_cover)) {
        return chip_cover;
    }
    if (host_cover_wall <= static_cast<double>(std::numeric_limits<int64_t>::min())) {
        return std::numeric_limits<int64_t>::min();
    }
    return static_cast<int64_t>(std::floor(host_cover_wall));
}

bool ClockMap::has_host_nodes() const noexcept { return host_.nodes.size() != 0; }

std::optional<double> ClockMap::root_offset(Reader& reader, uint32_t dev, int64_t wall, double frac) const noexcept {
    Segment<int64_t>& seg = reader.chips_[dev];
    const std::optional<double> root = lookup(chips_[dev], seg, wall);
    return root ? std::optional(*root + seg.slope * frac) : std::nullopt;
}

std::optional<double> ClockMap::host_tsc_offset(Reader& reader, double root) const noexcept {
    return lookup(host_, reader.host_, root);
}

int64_t ClockMap::place_slow(Reader& reader, uint32_t dev, int64_t wall) const {
    const std::optional<double> root = lookup(chips_[dev], reader.chips_[dev], wall);
    TT_FATAL(root, "streaming profiler: device {} has no placement node for its wall tick {}", dev, wall);
    const std::optional<double> tsc = lookup(host_, reader.host_, *root);
    TT_FATAL(tsc, "streaming profiler: device {}'s wall tick {} has no host placement node", dev, wall);
    return tsc_base_ + round_nearest(*tsc);
}

void SteadyClock::append(int64_t tsc, int64_t mono_ns) {
    const SyncNode& last = steady_.last;
    if (tsc <= last.at) {
        return;
    }
    const int64_t base = base_ ? *base_ : *(base_ = mono_ns);
    const double value = static_cast<double>(mono_ns - base);
    const double ns_per_tick = last.at == std::numeric_limits<int64_t>::lowest()
                                   ? 1.0 / tsc_ticks_per_ns()
                                   : (value - last.value) / static_cast<double>(tsc - last.at);
    TT_FATAL(ns_per_tick > 0.0, "streaming profiler: steady clock node at tsc {} is not a rate: {}", tsc, ns_per_tick);
    steady_.append(SyncNode{.at = tsc, .value = value, .tangent = ns_per_tick});
}

std::optional<int64_t> SteadyClock::ns(int64_t tsc) const noexcept {
    constinit thread_local Segment<int64_t> t_segment{};
    const std::optional<double> rel = lookup(steady_, t_segment, tsc);
    return rel ? std::optional(*base_ + ClockMap::round_nearest(*rel)) : std::nullopt;
}

}  // namespace tt::tt_metal::streaming_profiler
