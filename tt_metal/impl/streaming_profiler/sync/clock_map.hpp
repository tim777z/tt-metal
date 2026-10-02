// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <limits>
#include <optional>
#include <string>
#include <vector>

#include "tt_metal/common/broadcast_ring.hpp"

namespace tt::tt_metal::streaming_profiler {

// A point of a piecewise-linear map from Key to a double: at key `at` the map is `value`, with slope `tangent`.
template <typename Key>
struct ClockNode {
    Key at{};
    double value = 0.0;
    double tangent = 0.0;
};
using SyncNode = ClockNode<int64_t>;
using HostNode = ClockNode<double>;

struct ClockBases {
    int64_t root_refclk = 0, tsc = 0;
};

// How a series maps a key between two nodes: on the straight line between them, or on the earlier node's tangent.
enum class Between { Chord, Tangent };

// A piecewise-linear map through its nodes, written by one thread and read lock-free by any. Keys up to `cover` map
// for good: no later node changes them.
template <typename Key>
struct ClockSeries {
    using Node = ClockNode<Key>;
    ClockSeries(uint32_t series_nodes, std::string name, Between between = Between::Chord) :
        nodes(series_nodes), name(std::move(name)), between(between) {}
    BroadcastRing<Node, 4096> nodes;
    std::string name;
    Between between;
    Node last{.at = std::numeric_limits<Key>::lowest()};  // the writer's own copy
    alignas(64) std::atomic<Key> cover{std::numeric_limits<Key>::lowest()};
    bool full_warned = false;

    void append(const Node& node);
    void extend(Key until) {
        if (until > cover.load(std::memory_order_relaxed)) {
            cover.store(until, std::memory_order_release);
        }
    }
};
extern template struct ClockSeries<int64_t>;
extern template struct ClockSeries<double>;

// One linear piece of a series, over keys [from, to]. A reader keeps the last one it found, so most lookups skip the
// search.
template <typename Key>
struct Segment {
    Key from = std::numeric_limits<Key>::max();
    Key to = std::numeric_limits<Key>::lowest();
    Key origin{};
    double value = 0.0;
    double slope = 0.0;
    uint64_t hint = 0;  // the index of the first node after `from`, where the next search starts
    bool holds(Key key) const noexcept { return key >= from && key <= to; }
    double at(Key key) const noexcept { return value + slope * static_cast<double>(key - origin); }
};

// Each chip's series maps its wall ticks onto the root chip's refclk, and the host series maps root refclk ticks onto
// the host TSC. One thread writes, and reads never lock or block it (BroadcastRing::read_at).
class ClockMap {
public:
    // At 24 B a node, 1.5 GB at most per chip. A chip takes a node at least every millisecond (kPointTicks in
    // kernels/eth_clock_model.cpp), so a series holds about 18.6 hours of a steady clock.
    static constexpr uint32_t kSeriesNodes = 1u << 26;

    // Placement state for one ClockMap, owned and used by a single thread.
    class Reader {
    public:
        Reader() = default;

    private:
        friend class ClockMap;
        Reader(size_t devices, int64_t tsc_base) : chips_(devices), tsc_base_(tsc_base) {}
        std::vector<Segment<int64_t>> chips_;
        Segment<double> host_;
        int64_t tsc_base_ = 0;
    };

    ClockMap(size_t devices, uint32_t series_nodes, ClockBases bases);
    ClockMap(const ClockMap&) = delete;
    ClockMap& operator=(const ClockMap&) = delete;
    Reader reader() const;

    // A node at or before its series' last node is dropped.
    void append(uint32_t dev, SyncNode node);
    // The host series places every key from `node` up to `until` on its tangent, so a record's host time never waits
    // for a later burst.
    void append_host(HostNode node, double until);
    // Ends the chip's series, which releases every record held for it.
    void finish(uint32_t dev);
    int64_t root_base() const noexcept { return root_base_; }
    int64_t tsc_base() const noexcept { return tsc_base_; }

    std::optional<double> root_offset(Reader& reader, uint32_t dev, int64_t wall, double frac) const noexcept;
    int64_t place_host(Reader& reader, uint32_t dev, int64_t wall) const {
        if (const std::optional<int64_t> tsc = on_segments(reader.chips_[dev], reader.host_, reader.tsc_base_, wall)) {
            return *tsc;
        }
        return place_slow(reader, dev, wall);
    }
    std::optional<double> host_tsc_offset(Reader& reader, double root) const noexcept;
    bool has_host_nodes() const noexcept;
    // The wall tick below which the chip's records have final host times.
    int64_t cover_ticks(uint32_t dev) const noexcept;

    // One instruction, where std::llround is a call.
    static int64_t round_nearest(double x) noexcept { return static_cast<int64_t>(std::nearbyint(x)); }

private:
    static std::optional<int64_t> on_segments(
        const Segment<int64_t>& chip, const Segment<double>& host, int64_t tsc_base, int64_t wall) {
        if (chip.holds(wall)) {
            const double root = chip.at(wall);
            if (host.holds(root)) {
                return tsc_base + round_nearest(host.at(root));
            }
        }
        return std::nullopt;
    }
    int64_t place_slow(Reader& reader, uint32_t dev, int64_t wall) const;

    std::deque<ClockSeries<int64_t>> chips_;
    ClockSeries<double> host_;
    const int64_t root_base_, tsc_base_;
};

// Every capture's host sync appends its pairs from the Service's one sync thread. Read lock-free from any thread.
class SteadyClock {
public:
    SteadyClock() = default;
    SteadyClock(const SteadyClock&) = delete;
    SteadyClock& operator=(const SteadyClock&) = delete;
    void append(int64_t tsc, int64_t mono_ns);
    std::optional<int64_t> ns(int64_t tsc) const noexcept;

private:
    ClockSeries<int64_t> steady_{ClockMap::kSeriesNodes, "the host steady clock"};
    std::optional<int64_t> base_;
};

}  // namespace tt::tt_metal::streaming_profiler
