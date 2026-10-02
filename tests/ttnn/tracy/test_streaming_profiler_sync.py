# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""The streaming profiler's clock sync gates.

test_streaming_profiler_sync_check is the accuracy gate. Each workload runs as a subprocess with the streaming profiler
and its sync check on (TT_METAL_STREAMING_PROFILER_SYNC_CHECK=1), which puts a ruler on one more idle eth core per chip
and logs, at each capture's end, the chip-to-chip error of the global timeline measured against held-out link rounds.
The test asserts that every chip pair was measured, p50 <= 1 ns, p99.9 <= 3.5 ns and max <= 6 ns with every reading
counted, the link sync's precision and parallel links' agreement within their bounds, no sync warnings in the log, and,
for the FF1 matmul and SDPA di/dt tests, that AICLK swung at least 100 MHz so DVFS was exercised. The workloads: an idle
mesh, fabric ping-pong with and without load, a multicast core-to-core check, the FF1 matmul and SDPA di/dt tests, two
ttnn CCL ops, and host round trips that bound where the host timeline puts a device zone. Needs a multi-chip Blackhole
system (an 8-chip LoudBox in CI); about 4 minutes.

test_streaming_profiler_fabric_overhead bounds the profiler's cost to the fabric. It runs test_tt_fabric's unicast
microbench (latency and bandwidth over linear, mesh and ring routes) with the profiler off as the baseline, then on,
then with the sync check, and bounds the latency and bandwidth each adds per test. The sync-check arm also gates the
sync's accuracy. test_streaming_profiler_fabric_eth_zones turns on Ethernet-core zones
(TT_METAL_STREAMING_PROFILER_ETH=1) and checks that the linear, ring and mesh routers still build and run and that eth
zones reach the zone CSV. Both need four or more Blackhole chips; about 2 minutes warm, 5 cold.

Run as a script, this file is the sync check's CCL workload (ccl_workload).
"""

from __future__ import annotations

import argparse
import csv
import functools
import os
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path

import pytest

from tools.tracy.common import TT_METAL_HOME

# ~1.25x and ~1.4x the worst an 8-chip LoudBox has shown over three runs of the suite: p99.9 2.78 ns, max 4.38 ns with
# every reading counted.
P999_BOUND_NS = 3.5
# The typical error: p50 ran 0.41-0.72 ns over 163 sync-check captures (09-29 and 09-30), so 1.0 ns is ~40% above it.
P50_BOUND_NS = 1.0
MAX_BOUND_NS = 6.0
# Link sync precision over 22 captures (sync check, fabric overhead, torus): the worst link's round scatter ran
# 0.51-0.63 ns and the worst window's fit at its centre 0.13-0.17 ns, so the bounds sit ~25% and ~50% above them.
ROUND_SCATTER_BOUND_NS = 0.8
WINDOW_FIT_BOUND_NS = 0.25
# The worst pair of parallel links put their chips' offset 1.26-2.68 ns apart over 31 captures across 8 link-ups (mean
# 1.84, sd 0.41); a link-up redraws it, so 4 ns is the one-sided 99.9% prediction bound with the 8 link-ups as samples.
PARALLEL_LINKS_BOUND_NS = 4.0

HEADLINE = re.compile(
    r"sync check: chip-to-chip error of the global timeline, a bound, over (\d+) of \d+ chip pairs and \d+ samples: "
    r"\|err\| p50 ([0-9.]+), p99 [0-9.]+, p99\.9 ([0-9.]+), max ([0-9.]+) ns \(([^)]*)\)"
)
CHIP_LINE = re.compile(r"sync check chip (\d+): \d+ readings against the reference, .*?max ([0-9.]+) ns")
AICLK_LINE = re.compile(r"sync check chip (\d+) AICLK: (.*?)(?: \(\w+\.cpp:\d+\))?$", re.MULTILINE)
AICLK_RANGE = re.compile(r"sync check chip \d+ AICLK: mean \d+ MHz, sd \d+, (\d+)-(\d+) MHz")
FORBIDDEN = [
    "no chip pair measured",  # sync/check.cpp
    "the clock map could not place",  # sync/check.cpp
    "sync stream",  # service.cpp
    "not solved",  # sync/engine.cpp
    "overflows region",  # tt_elffile.cpp
    "the clock tracker's sync ring",  # device_programs.cpp
    "the sync check's ruler",  # device_programs.cpp
    "TT_FATAL",  # tt_stl/assert.hpp
]
PROBE_TIMEOUT_S = 300
# CI's logger colors every message, which puts an escape code right before a line's first word.
ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;]*m")


def built(rel: Path) -> Path:
    """The build output `rel`, under `build` (CI) or `build_Release` (a local build_metal.sh run)."""
    for build_dir in ("build", "build_Release"):
        candidate = Path(TT_METAL_HOME) / build_dir / rel
        if candidate.exists():
            return candidate
    return Path(TT_METAL_HOME) / "build" / rel


def run(args: list[str | Path], env_extra: dict[str, str], timeout: float) -> str:
    """Run `args` with `env_extra` over the caller's environment minus its TT_METAL_STREAMING_PROFILER* variables, and
    return stdout then stderr. Skips the test if the profiler was asked for but never started; fails it on a nonzero
    exit or a timeout."""
    env = {name: value for name, value in os.environ.items() if not name.startswith("TT_METAL_STREAMING_PROFILER")}
    env.update(env_extra)
    args = [str(arg) for arg in args]
    try:
        proc = subprocess.run(args, env=env, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired as timeout_error:
        # On POSIX the output captured before the timeout comes back as bytes even with text=True.
        out = "".join(
            stream.decode(errors="replace") if isinstance(stream, bytes) else stream or ""
            for stream in (timeout_error.stdout, timeout_error.stderr)
        )
        out = ANSI_ESCAPE.sub("", out)
        pytest.fail(f"{args[0]} timed out after {timeout} s:\n{out[-4000:]}")
    out = ANSI_ESCAPE.sub("", proc.stdout + proc.stderr)
    if "TT_METAL_STREAMING_PROFILER" in env_extra and "[streaming profiler] active on" not in out:
        pytest.skip("streaming profiler did not start (not Blackhole / no DRAM programmable cores)")
    assert proc.returncode == 0, f"{args[0]} exited {proc.returncode}:\n{out[-4000:]}"
    return out


@functools.cache
def system() -> tuple[str, int]:
    """Read in a subprocess so the pytest parent never takes the PCIe lock."""
    code = "import ttnn; print('SYSTEM', ttnn.get_arch_name(), ttnn.get_num_devices())"
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=PROBE_TIMEOUT_S)
    match = re.search(r"^SYSTEM (\S+) (\d+)$", proc.stdout, re.MULTILINE)
    assert match, f"no system probe output:\n{proc.stdout[-2000:]}\n{proc.stderr[-2000:]}"
    return match.group(1).lower(), int(match.group(2))


def skip_unless_blackhole(min_chips: int) -> None:
    arch, chips = system()
    if arch != "blackhole":
        pytest.skip(f"the streaming profiler runs on Blackhole, not {arch}")
    if chips < min_chips:
        pytest.skip(f"needs at least {min_chips} chips, this system has {chips}")


def aiclk_swing_mhz(out: str) -> int:
    """The widest range, in MHz, that any chip's AICLK covered in any of the run's captures."""
    return max((int(hi) - int(lo) for lo, hi in AICLK_RANGE.findall(out)), default=0)


PRECISION = re.compile(
    r"link sync precision over (\d+) of (\d+) links: round scatter ([0-9.]+) ns \(links' mean ([0-9.]+) ns\), "
    r"fit at the window centre ([0-9.]+) ns \((chip \d+ -> chip \d+)\), worst window ([0-9.]+) ns"
)


def check_link_precision(out: str) -> None:
    """Assert every capture in `out` measured all its links, within the round scatter and window fit bounds."""
    reports = list(PRECISION.finditer(out))
    assert reports, f"no link sync precision report:\n{out[-4000:]}"
    for report in reports:
        print(f"\n[link-precision] {report.group(0)}")
        measured, links = int(report.group(1)), int(report.group(2))
        assert measured == links, f"precision measured on {measured} of {links} links"
        scatter, window = float(report.group(3)), float(report.group(7))
        assert scatter <= ROUND_SCATTER_BOUND_NS, f"round scatter {scatter} ns > {ROUND_SCATTER_BOUND_NS} ns"
        assert window <= WINDOW_FIT_BOUND_NS, f"worst window fit {window} ns > {WINDOW_FIT_BOUND_NS} ns"


PARALLEL = re.compile(r"parallel link agreement over \d+ link pairs: rms [0-9.]+ ns, worst ([0-9.]+) ns \([^)]*\)")


def check_parallel_links(out: str) -> None:
    """Assert `out` reports its captures' parallel link agreement, each within the bound."""
    reports = list(PARALLEL.finditer(out))
    assert reports, f"no parallel link agreement report:\n{out[-4000:]}"
    for report in reports:
        print(f"\n[parallel-links] {report.group(0)}")
        worst = float(report.group(1))
        assert worst <= PARALLEL_LINKS_BOUND_NS, f"parallel links {worst} ns apart > {PARALLEL_LINKS_BOUND_NS} ns"


def check_sync_accuracy(out: str) -> None:
    headlines = list(HEADLINE.finditer(out))
    assert headlines, f"no sync check report:\n{out[-4000:]}"
    for k, headline in enumerate(headlines):
        block = out[headline.start() : headlines[k + 1].start() if k + 1 < len(headlines) else len(out)]
        chips = {int(chip): float(max_ns) for chip, max_ns in CHIP_LINE.findall(block)}
        assert len(chips) == system()[1], f"the report covers chips {sorted(chips)} of {system()[1]}"
        measured, p50, p999, worst, where = headline.groups()
        print(f"\n[sync-check] {headline.group(0)}")
        per_chip = " ".join(f"c{chip} {max_ns:.2f}" for chip, max_ns in sorted(chips.items()))
        print(f"[sync-check] per chip max ns: {per_chip}")
        for chip, clock in AICLK_LINE.findall(block):
            print(f"[sync-check] chip {chip} AICLK: {clock}")
        num_chips = len(chips)
        all_pairs = num_chips * (num_chips - 1) // 2
        assert int(measured) == all_pairs, f"{measured} chip pairs measured for {num_chips} chips"
        assert float(p50) <= P50_BOUND_NS, f"p50 {p50} ns > {P50_BOUND_NS} ns ({headline.group(0)})"
        assert float(p999) <= P999_BOUND_NS, f"p99.9 {p999} ns > {P999_BOUND_NS} ns ({headline.group(0)})"
        assert float(worst) <= MAX_BOUND_NS, f"max {worst} ns > {MAX_BOUND_NS} ns ({where})"
    check_link_precision(out)
    check_parallel_links(out)
    check_log_clean(out)


def check_log_clean(out: str) -> None:
    """Fail if `out` has any line the profiler logs only when the sync went wrong."""
    for bad in FORBIDDEN:
        offending = "\n".join(line for line in out.splitlines() if bad in line)
        assert bad not in out, f"'{bad}' in the log:\n{offending[:4000]}"


SYNC_WORKLOADS = built(Path("test/tt_metal/tools/profiler/test_streaming_profiler_sync_workloads"))
CCL = [Path(sys.executable), Path(__file__)]
DIDT = [Path(sys.executable), "-m", "pytest", "-s"]
DIDT_TESTS = Path(__file__).parents[2] / "didt"
SECONDS = 20
WORKLOADS = {
    "idle": [SYNC_WORKLOADS, "pingpong", "--idle", "--seconds", str(SECONDS)],
    "host_sync": [SYNC_WORKLOADS, "host_sync"],
    "fabric_traffic_load": [SYNC_WORKLOADS, "pingpong", "--load"],
    "multicast": [SYNC_WORKLOADS, "multicast"],
    "didt_ff1_matmul": [
        *DIDT,
        f"{DIDT_TESTS}/test_ff1_matmul.py::test_ff1_matmul",
        "-k",
        "all and without_gelu",
        "--didt-workload-iterations",
        "3000",
    ],
    "didt_sdpa": [
        *DIDT,
        f"{DIDT_TESTS}/test_sdpa_op.py::test_sdpa_op",
        "-k",
        "all and bf16_HiFi2",
        "--didt-workload-iterations",
        "500",
    ],
    "ccl_all_gather_ring": [*CCL, "--op", "all_gather", "--fabric", "ring", "--seconds", str(SECONDS)],
    "ccl_all_reduce_2d_load": [*CCL, "--op", "all_reduce", "--fabric", "2d", "--load", "4", "--seconds", str(SECONDS)],
}
SYNC_TIMEOUT_S = 300
# The di/dt tests are here for the DVFS their throttling causes: some chip's AICLK swings over 100 MHz in each, against
# the 6 MHz an idle chip jitters by.
DVFS_WORKLOADS = {"didt_ff1_matmul", "didt_sdpa"}
MIN_DVFS_SWING_MHZ = 100
ENV = {"TT_METAL_STREAMING_PROFILER": "1", "TT_METAL_STREAMING_PROFILER_SYNC_CHECK": "1"}


@pytest.mark.timeout(PROBE_TIMEOUT_S + SYNC_TIMEOUT_S + 60)
@pytest.mark.parametrize("workload", list(WORKLOADS))
def test_streaming_profiler_sync_check(workload):
    skip_unless_blackhole(min_chips=2)
    exe = WORKLOADS[workload][0]
    assert exe.exists(), f"{exe} not built (build with --build-tests)"
    out = run(WORKLOADS[workload], ENV, SYNC_TIMEOUT_S)
    for line in out.splitlines():
        if "[host_sync]" in line:
            print(line)
    check_sync_accuracy(out)
    if workload in DVFS_WORKLOADS:
        swing = aiclk_swing_mhz(out)
        assert swing >= MIN_DVFS_SWING_MHZ, f"AICLK moved at most {swing} MHz on any chip, so no DVFS was exercised"


FABRIC_BIN = built(Path("test/tt_metal/tt_fabric/test_infra/test_tt_fabric"))
CONFIG = (
    Path(TT_METAL_HOME) / "tests/tt_metal/tt_fabric/test_infra/test_yamls/test_fabric_ubench_at_least_2x2_mesh.yaml"
)
FABRIC_TIMEOUT_S = 180
ARMS = {
    "off": {},
    "on": {"TT_METAL_STREAMING_PROFILER": "1"},
    "sync_check": {"TT_METAL_STREAMING_PROFILER": "1", "TT_METAL_STREAMING_PROFILER_SYNC_CHECK": "1"},
    "eth_zones": {"TT_METAL_STREAMING_PROFILER": "1", "TT_METAL_STREAMING_PROFILER_ETH": "1"},
}
# Latency added, in ns: the median over tests of p50 and p99, and the worst test's p99 and max. Each latency test takes
# 256 samples, so its p99 is about its third-slowest packet and moves ~90 ns run to run. Over 20 runs on an 8-chip
# LoudBox the profiler added, mean +- sd per run, +0.6 +- 3.7 ns to the median p50, +55 +- 29 to the median p99, +245
# +- 93 to the worst test's p99 and +527 +- 51 to the worst test's max; each bound is mean + 3.67 sd, the one-sided
# 99.9% prediction bound for one more run. Over four runs the sync check added at most +366 ns to the median p99,
# +608 to a test's p99 and +680 to a max.
LATENCY_BOUNDS = {"on": (15.0, 165.0, 590.0, 715.0), "sync_check": (30.0, 700.0, 1000.0, 1200.0)}
# Over seven runs on an 8-chip LoudBox the median bandwidth ratio was 1.023-1.025 and no test fell below 0.997 with the
# profiler on (1.009-1.010 and 0.984 with the sync check). Both arms' bounds sit 0.028 below their mean median ratio and
# 0.009 below their mean worst test.
BANDWIDTH_BOUNDS = {"on": (0.996, 0.989), "sync_check": (0.982, 0.976)}

RESULTS = re.compile(r"=== Latency Test Results for (\S+) ===")
RUNNING = re.compile(r"Running Test: (\S+)")
BANDWIDTH = re.compile(r"BW \(GB/s\)=([0-9.]+)")
# A row of the latency table tt_fabric_test_latency_results.cpp:462 logs; the third column is Net Latency.
ROW = re.compile(r"\b(Max|P99|P50)\s+[0-9.]+\s+[0-9.]+\s+([0-9.]+)\s+[0-9.]+")


def run_fabric_ubench(env_extra: dict, name_glob: str = "*Unicast*") -> tuple[dict, dict, str]:
    out = run([FABRIC_BIN, "--test_config", CONFIG, "--filter", f"name.{name_glob}"], env_extra, FABRIC_TIMEOUT_S)
    latency, bandwidth, name, running = {}, {}, None, None
    for line in out.splitlines():
        if match := RESULTS.search(line):
            name = match.group(1)
            latency[name] = {}
        elif name and (match := ROW.search(line)):
            latency[name][match.group(1)] = float(match.group(2))
        if match := RUNNING.search(line):
            running = match.group(1)
        elif running and not running.startswith("Latency") and (match := BANDWIDTH.search(line)):
            bandwidth.setdefault(running, []).append(float(match.group(1)))
    latency = {test: rows for test, rows in latency.items() if len(rows) == 3}
    bandwidth = {test: statistics.mean(samples) for test, samples in bandwidth.items()}
    assert latency or bandwidth, f"no latency or bandwidth results:\n{out[-4000:]}"
    return latency, bandwidth, out


def assert_same_tests(kind: str, results: dict, base: dict) -> None:
    assert set(results) == set(base), f"{kind} tests differ: {sorted(set(results) ^ set(base))}"


@pytest.fixture(scope="module")
def baseline():
    skip_unless_blackhole(min_chips=4)
    assert FABRIC_BIN.exists(), f"{FABRIC_BIN} not built (build with --build-tests)"
    return run_fabric_ubench(ARMS["off"])


@pytest.mark.timeout(PROBE_TIMEOUT_S + 2 * FABRIC_TIMEOUT_S + 60)
@pytest.mark.parametrize("arm", ["on", "sync_check"])
def test_streaming_profiler_fabric_overhead(baseline, arm):
    base_latency, base_bandwidth, _ = baseline
    latency, bandwidth, out = run_fabric_ubench(ARMS[arm])
    assert_same_tests("latency", latency, base_latency)
    assert_same_tests("bandwidth", bandwidth, base_bandwidth)
    p50 = {test: latency[test]["P50"] - base_latency[test]["P50"] for test in base_latency}
    p99 = {test: latency[test]["P99"] - base_latency[test]["P99"] for test in base_latency}
    max_added = {test: latency[test]["Max"] - base_latency[test]["Max"] for test in base_latency}
    ratio = {test: bandwidth[test] / base_bandwidth[test] for test in base_bandwidth}
    p50_bound, median_bound, p99_bound, max_bound = LATENCY_BOUNDS[arm]
    median_ratio, worst_ratio = BANDWIDTH_BOUNDS[arm]
    table = "\n".join(
        f"  {test}: p50 {p50[test]:+.0f} ns, p99 {p99[test]:+.0f} ns, max {max_added[test]:+.0f} ns"
        for test in sorted(base_latency)
    )
    worst = "\n".join(
        f"  {test}: {ratio[test]:.3f}" for test in sorted(ratio, key=lambda test: (ratio[test], test))[:8]
    )
    print(f"\n[fabric-overhead {arm}] latency added:\n{table}")
    print(
        f"[fabric-overhead {arm}] bandwidth ratio: median {statistics.median(ratio.values()):.4f}, "
        f"worst {min(ratio.values()):.4f}; worst tests:\n{worst}"
    )
    assert statistics.median(p50.values()) <= p50_bound, f"median p50 added > {p50_bound} ns:\n{table}"
    assert statistics.median(p99.values()) <= median_bound, f"median p99 added > {median_bound} ns:\n{table}"
    assert max(p99.values()) <= p99_bound, f"a test's p99 added > {p99_bound} ns:\n{table}"
    assert max(max_added.values()) <= max_bound, f"a test's max added > {max_bound} ns:\n{table}"
    assert statistics.median(ratio.values()) >= median_ratio, f"median bandwidth ratio < {median_ratio}:\n{worst}"
    assert min(ratio.values()) >= worst_ratio, f"a test's bandwidth ratio < {worst_ratio}:\n{worst}"
    if arm == "sync_check":
        check_sync_accuracy(out)
    else:
        check_link_precision(out)
        check_log_clean(out)


@pytest.mark.timeout(PROBE_TIMEOUT_S + 2 * FABRIC_TIMEOUT_S + 60)
def test_streaming_profiler_fabric_eth_zones(baseline, tmp_path):
    """With Ethernet-core zones on, the latency tests' routers on linear, ring and mesh fabric each fit their kernel
    config buffer and run, and their zones reach the zone CSV. No other gate turns eth zones on."""
    base_latency, _, _ = baseline
    zone_csv = tmp_path / "zones.csv"
    latency, _, _ = run_fabric_ubench(
        ARMS["eth_zones"] | {"TT_METAL_STREAMING_PROFILER_ZONE_CSV": str(zone_csv)}, name_glob="Latency*Unicast"
    )
    assert_same_tests("latency", latency, base_latency)
    with zone_csv.open() as zone_file:
        zone_file.readline()
        riscs = {row["RISC processor type"] for row in csv.DictReader(zone_file, skipinitialspace=True)}
    assert "ERISC" in riscs, f"no ERISC zone in the zone CSV, only {sorted(riscs)}"


CCL_OPS_PER_SYNC = 20


def ccl_workload() -> None:
    """A ttnn CCL op in a loop on every chip of the box for a fixed time, optionally with matmuls between the ops."""
    import torch
    import ttnn

    fabrics = {
        "ring": (ttnn.FabricConfig.FABRIC_1D_RING, ttnn.Topology.Ring),
        "2d": (ttnn.FabricConfig.FABRIC_2D, ttnn.Topology.Linear),
    }
    parser = argparse.ArgumentParser(description=ccl_workload.__doc__)
    parser.add_argument("--op", choices=["all_gather", "all_reduce"], required=True)
    parser.add_argument("--fabric", choices=list(fabrics), required=True)
    parser.add_argument("--seconds", type=float, default=20.0)
    parser.add_argument("--load", type=int, default=0, help="matmuls before each CCL op")
    args = parser.parse_args()

    fabric, topology = fabrics[args.fabric]
    rows, cols = sorted(tuple(ttnn._ttnn.multi_device.SystemMeshDescriptor().shape()), reverse=True)
    ttnn.set_fabric_config(fabric)
    mesh = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(rows, cols))

    def to_mesh(host_tensor: torch.Tensor, mapper) -> ttnn.Tensor:
        return ttnn.from_torch(
            host_tensor,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh,
            mesh_mapper=mapper,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    ccl_input = to_mesh(
        torch.randn([rows, cols, 512, 2048]).bfloat16(),
        ttnn.ShardTensor2dMesh(mesh, dims=(0, 1), mesh_shape=(rows, cols)),
    )
    tensors = [ccl_input]
    if args.load:
        matmul_a = to_mesh(torch.randn([1, 1, 2048, 4096]).bfloat16(), ttnn.ReplicateTensorToMesh(mesh))
        matmul_b = to_mesh(torch.randn([1, 1, 4096, 4096]).bfloat16(), ttnn.ReplicateTensorToMesh(mesh))
        tensors += [matmul_a, matmul_b]

    ops = 0
    end = time.monotonic() + args.seconds
    while time.monotonic() < end:
        for _ in range(CCL_OPS_PER_SYNC):
            for _ in range(args.load):
                ttnn.matmul(matmul_a, matmul_b)
            if args.op == "all_gather":
                ttnn.all_gather(ccl_input, dim=3, cluster_axis=0, topology=topology)
            else:
                ttnn.all_reduce(ccl_input, cluster_axis=0, topology=topology)
        ttnn.synchronize_device(mesh)
        ops += CCL_OPS_PER_SYNC
    print(f"{args.op} over {args.fabric} fabric, load {args.load}: {ops} ops in {args.seconds:.0f} s")
    for tensor in tensors:
        ttnn.deallocate(tensor)
    ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    ccl_workload()
