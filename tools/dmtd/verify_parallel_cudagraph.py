#!/usr/bin/env python3
"""Verify the attention backend can capture a parallel-layer CUDA graph.

The parallel-layer plan hinges on one property: a batch of `num_reqs` requests
each contributing `tau` rows must be capturable as a FULL CUDA graph, including
with `causal=False`. `resolve_cudagraph_mode_and_sizes` only validates the main
model's single `decode_query_len`, so it says nothing about a second uniform
query length -- hence this probe.

It boots a real DMTD checkpoint, then for each `causal` setting:
  1. builds parallel-layer attention metadata at `max_query_len = tau`,
  2. runs the 28 parallel layers eagerly to get a reference,
  3. captures the same call in a CUDA graph and replays it,
  4. requires the replayed output to equal the eager one bit-for-bit.

Exits non-zero if any step fails.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any

MODEL_ROOT = os.environ.get("DMTD_MODEL_ROOT", "/workspace/parallel-eval/models")


def _probe(worker: Any, tau: int, num_reqs: int) -> dict:
    import torch

    from vllm.forward_context import set_forward_context
    from vllm.v1.attention.backend import AttentionCGSupport
    from vllm.v1.worker.gpu.attn_utils import (
        build_attn_metadata,
        build_slot_mappings_by_layer,
    )

    runner = worker.model_runner
    model = runner.model
    state = runner.model_state
    device = runner.device
    kv_cache_config = runner.kv_cache_config
    attn_groups = runner.attn_groups
    block_tables = runner.block_tables

    builder = attn_groups[0][0].get_metadata_builder(0)
    backend = attn_groups[0][0].backend
    report: dict[str, Any] = {
        "backend": backend.__name__,
        "supports_non_causal": bool(backend.supports_non_causal()),
        "cudagraph_support": builder.get_cudagraph_support(
            runner.vllm_config, builder.kv_cache_spec
        ).name,
        "uniform_batch_ok": (
            builder.get_cudagraph_support(
                runner.vllm_config, builder.kv_cache_spec
            ).value
            >= AttentionCGSupport.UNIFORM_BATCH.value
        ),
        "num_metadata_builders": len(attn_groups[0][0].metadata_builders),
        "tau": tau,
        "num_reqs": num_reqs,
        "cases": {},
    }

    num_rows = num_reqs * tau
    parallel_layers = model.model.layers[: state.num_parallel_layers]
    layer_names = [layer.self_attn.attn.layer_name for layer in parallel_layers]

    # A uniform batch: every request contributes exactly `tau` rows, mirroring
    # what a cycle-head step plans.
    query_start_cpu = torch.arange(num_reqs + 1, dtype=torch.int32) * tau
    query_start_gpu = query_start_cpu.to(device)
    seq_lens = torch.full((num_reqs,), tau, dtype=torch.int32, device=device)
    positions = torch.arange(num_rows, dtype=torch.int64, device=device) % tau
    input_ids = torch.full(
        (num_rows,), state.mask_token_id, dtype=torch.int32, device=device
    )
    dummy_block_tables = block_tables.get_dummy_block_tables(num_reqs)
    # PAD_SLOT_ID everywhere: this probe must not write into the real KV cache.
    slot_mappings = block_tables.get_dummy_slot_mappings(num_rows)
    slot_mappings_by_layer = build_slot_mappings_by_layer(
        slot_mappings, kv_cache_config
    )

    for causal in (True, False):
        metadata = build_attn_metadata(
            attn_groups=attn_groups,
            num_reqs=num_reqs,
            num_tokens=num_rows,
            query_start_loc_gpu=query_start_gpu,
            query_start_loc_cpu=query_start_cpu,
            max_query_len=tau,
            seq_lens=seq_lens,
            max_seq_len=runner.max_model_len,
            block_tables=dummy_block_tables,
            slot_mappings=slot_mappings,
            kv_cache_config=kv_cache_config,
            for_cudagraph_capture=True,
            causal=causal,
            metadata_builder_idx=1,
        )

        def run() -> torch.Tensor:
            context = torch.cuda.current_stream()  # noqa: F841  (keep stream alive)
            hidden = model.model.embed_input_ids(input_ids)
            residual = None
            for layer in parallel_layers:
                hidden, residual = layer(positions, hidden, residual)
            return hidden + residual

        with set_forward_context(
            metadata,
            runner.vllm_config,
            num_tokens=num_rows,
            slot_mapping=slot_mappings_by_layer,
        ):
            eager = run().clone()

            # Warm up on a side stream before capture, as vLLM's own capture
            # path does; the first call may allocate lazily.
            replay_out = run()
            torch.cuda.synchronize()

            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                captured = run()
            graph.replay()
            torch.cuda.synchronize()

        max_diff = (captured - eager).abs().max().item()
        report["cases"][f"causal={causal}"] = {
            "layer_names": len(layer_names),
            "rows": num_rows,
            "captured_matches_eager": bool(torch.equal(captured, eager)),
            "max_abs_diff": max_diff,
            "replay_finite": bool(torch.isfinite(replay_out).all().item()),
        }

    return report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="Bidirectional-Parallel-Refresh")
    ap.add_argument("--num-reqs", type=int, default=2)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.6)
    args = ap.parse_args()

    from vllm import LLM

    path = f"{MODEL_ROOT}/{args.variant}"
    llm = LLM(
        model=path,
        trust_remote_code=True,
        dtype="bfloat16",
        enable_prefix_caching=False,
        max_model_len=256,
        max_num_seqs=max(args.num_reqs, 4),
        gpu_memory_utilization=args.gpu_memory_utilization,
        disable_log_stats=True,
    )
    tau = llm.llm_engine.vllm_config.model_config.hf_config.mtp_horizon
    reports = llm.llm_engine.collective_rpc(
        lambda worker: _probe(worker, tau, args.num_reqs)
    )
    report = reports[0]
    print(json.dumps(report, indent=2))

    ok = report["uniform_batch_ok"] and report["supports_non_causal"]
    for name, case in report["cases"].items():
        if not case["captured_matches_eager"]:
            print(f"FAIL: {name} replay diverged (max_abs_diff={case['max_abs_diff']})")
            ok = False
    print("\nW0 VERDICT:", "PASS" if ok else "FAIL")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
