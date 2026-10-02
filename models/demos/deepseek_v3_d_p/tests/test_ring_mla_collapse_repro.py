# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

import os
import time

os.environ.setdefault("PREFILL_STEM_PROBE", "1")
os.environ.setdefault("PREFILL_STEM_PROBE_CUT", "1e6")
os.environ.setdefault("PREFILL_MODEL", "kimi_k2_7")

import pytest
from loguru import logger

import ttnn
from models.common.utility_functions import is_blackhole
from models.demos.common.prefill.adapter import PrefillRunParams, get_adapter
from models.demos.common.prefill.runners.runner_utils import load_trace_token_ids, open_mesh_device
from models.demos.deepseek_v3_d_p.tt.mla import mla as mla_module

MESH_SHAPE = (8, 4)
CHUNK_SIZE = 5120


def _env_int(name, default):
    return int(os.environ.get(name, default))


@pytest.mark.timeout(0)
def test_ring_mla_collapse_repro():
    adapter = get_adapter(os.environ["PREFILL_MODEL"])
    max_seq_len = _env_int("REPRO_MAX_SEQ_LEN", 256000)
    num_users = _env_int("REPRO_NUM_USERS", 84)
    num_requests = _env_int("REPRO_REQUESTS", 100)
    sync_per_chunk = os.environ.get("REPRO_SYNC_PER_CHUNK", "1") == "1"
    stop_on_hit = os.environ.get("REPRO_STOP_ON_HIT", "1") == "1"
    n_chunks = max_seq_len // CHUNK_SIZE
    assert n_chunks * CHUNK_SIZE == max_seq_len, f"max_seq_len {max_seq_len} is not a whole number of chunks"

    trace_dir = os.environ.get("PREFILL_TRACE_DIR", adapter.prefill_trace_default)
    pool = load_trace_token_ids(trace_dir, max_seq_len)
    assert pool, f"no token_ids in {trace_dir}"
    pool = (pool * -(-max_seq_len // len(pool)))[:max_seq_len]

    mesh_device = open_mesh_device(MESH_SHAPE, adapter.model_config, l1_small_size=adapter.l1_small_size)
    try:
        hf_config = adapter.load_hf_config()
        hf_config.max_seq_len = max_seq_len
        params = PrefillRunParams(
            mesh_shape=MESH_SHAPE,
            num_layers=1,
            first_layer_idx=0,
            is_first_rank=True,
            is_last_rank=True,
            max_seq_len=max_seq_len,
            chunk_size=CHUNK_SIZE,
            num_users=num_users,
            capacity_factor=8,
            num_links=2 if is_blackhole() else 1,
            gate_mode_name=os.environ.get("PREFILL_GATE_FALLBACK_MODE", adapter.default_gate_mode),
            kv_only_last_layer=False,
            weight_cache_path=adapter.weight_cache_path(MESH_SHAPE),
            sparse_kv_cache_format=adapter.default_sparse_kv_cache_format,
            use_trace=False,
        )
        runtime = adapter.build_runtime(mesh_device=mesh_device, hf_config=hf_config, params=params)
        kv_caches = adapter.allocate_kv_cache(mesh_device=mesh_device, hf_config=hf_config, params=params)
        runtime.compile(kv_caches)

        device_ids = [
            [mesh_device.get_device_id(ttnn.MeshCoordinate(r, c)) for c in range(MESH_SHAPE[1])]
            for r in range(MESH_SHAPE[0])
        ]
        logger.info(f"[repro] device ids by mesh (row, col): {device_ids}")
        logger.info(
            f"[repro] {num_requests} requests x {n_chunks} chunks, users={num_users}, "
            f"max_seq_len={max_seq_len}, sync_per_chunk={sync_per_chunk}"
        )

        hits = mla_module._STEM_PROBE_CALLS
        t0 = time.perf_counter()
        first_hit = None
        for req in range(num_requests):
            for c in range(n_chunks):
                start = c * CHUNK_SIZE
                inp = runtime.make_chunk_input(pool[start : start + CHUNK_SIZE])
                runtime.prefill_chunk(
                    inp,
                    kv_caches,
                    slot_id=0,
                    actual_start=start,
                    actual_end=start + CHUNK_SIZE,
                    request_id=req * n_chunks + c,
                )
                if sync_per_chunk:
                    ttnn.synchronize_device(mesh_device)
                if first_hit is None:
                    dirty_tags = [k[1] for k, v in hits.items() if k[0] == "hits" and v]
                    if dirty_tags:
                        first_hit = (req, c)
                        logger.error(
                            f"[repro] HIT {dirty_tags} at request {req} chunk {c} (kv [{start}, {start + CHUNK_SIZE}))"
                        )
            dirty = {k[1]: v for k, v in hits.items() if k[0] == "hits" and v}
            logger.info(
                f"[repro] request {req + 1}/{num_requests} done at {time.perf_counter() - t0:.0f}s, "
                f"dirty probe calls per tag: {dirty or 'none'}"
            )
            if first_hit is not None and stop_on_hit:
                break
        ttnn.synchronize_device(mesh_device)
    finally:
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
        ttnn.close_mesh_device(mesh_device)

    dirty = {k[1]: v for k, v in mla_module._STEM_PROBE_CALLS.items() if k[0] == "hits" and v}
    assert (
        not dirty
    ), f"stem probe saw exploded rows (dirty calls per tag): {dirty}, first hit at (request, chunk) {first_hit}"
