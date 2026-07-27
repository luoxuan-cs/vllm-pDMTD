#!/usr/bin/env python3
"""Ablation of the DMTD compilation optimizations, per variant.

Ladder, matching what the engine now supports:
  0_eager                  enforce_eager=True, async_scheduling=False
  1_cudagraph_seq          FULL_DECODE_ONLY graphs for the sequential layers
  2_cudagraph_head         + graphs for the parallel layers (cycle-head steps)
  3_cudagraph_head_async   + async scheduling

Every config runs in its own subprocess so GPU memory from one never leaks into
the next, and all of them use the same fixed input/output lengths.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

MODEL_ROOT = os.environ.get("DMTD_MODEL_ROOT", "/workspace/parallel-eval/models")
VARIANTS = [
    "Causal-Parallel-Norefresh",
    "Causal-Parallel-Refresh",
    "Bidirectional-Parallel-Norefresh",
    "Bidirectional-Parallel-Refresh",
]
DISABLE_PARALLEL_CG = "VLLM_DMTD_DISABLE_PARALLEL_CUDAGRAPH"
# Each entry is (LLM kwargs, extra environment for the worker subprocess).
CONFIGS: dict[str, tuple[dict, dict[str, str]]] = {
    "0_eager": (dict(enforce_eager=True, async_scheduling=False), {}),
    "1_cudagraph_seq": (
        dict(enforce_eager=False, async_scheduling=False),
        {DISABLE_PARALLEL_CG: "1"},
    ),
    "2_cudagraph_head": (dict(enforce_eager=False, async_scheduling=False), {}),
    "3_cudagraph_head_async": (dict(enforce_eager=False, async_scheduling=True), {}),
}


@dataclass
class RunResult:
    variant: str
    config: str
    enforce_eager: bool
    async_scheduling: bool
    cudagraph_mode: str
    parallel_graphs: int
    parallel_replays: int
    batch_size: int
    input_len: int
    output_len: int
    total_output_tokens: int
    wall_seconds: float
    tokens_per_second: float


def _prompts(model_path: str, batch_size: int, input_len: int):
    from transformers import AutoTokenizer

    from vllm import TokensPrompt

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    seed = (
        "Write a detailed technical explanation of parallel decoding for "
        "transformer language models. Cover prefill, decode, KV cache, "
        "memory bandwidth, and speculative methods. "
    )
    seed_ids = tokenizer.encode(seed, add_special_tokens=False)
    ids = (seed_ids * ((input_len // len(seed_ids)) + 2))[:input_len]
    return [TokensPrompt(prompt_token_ids=list(ids)) for _ in range(batch_size)]


def _parallel_graph_stats(worker) -> dict[str, int]:
    """Total captured graphs and replays across the parallel-layer groups."""
    managers = worker.model_runner.model_state._parallel_graph_managers.values()
    return {
        "graphs": sum(len(m.graphs) for m in managers),
        "replays": sum(m.num_replays for m in managers),
    }


def bench_one(args: argparse.Namespace) -> RunResult:
    import torch

    from vllm import LLM, SamplingParams

    options, _ = CONFIGS[args.worker_config]
    model_path = f"{MODEL_ROOT}/{args.worker_variant}"
    max_model_len = args.input_len + args.output_len + 64
    prompts = _prompts(model_path, args.batch_size, args.input_len)
    sampling = SamplingParams(
        temperature=0.0, max_tokens=args.output_len, ignore_eos=True
    )

    llm = LLM(
        model=model_path,
        trust_remote_code=True,
        dtype="bfloat16",
        enable_prefix_caching=False,
        max_model_len=max_model_len,
        max_num_seqs=args.batch_size,
        max_num_batched_tokens=max(max_model_len, args.batch_size * args.input_len),
        gpu_memory_utilization=args.gpu_memory_utilization,
        disable_log_stats=True,
        **options,
    )
    vllm_config = llm.llm_engine.vllm_config

    for _ in range(args.warmup_rounds):
        llm.generate(prompts, sampling, use_tqdm=False)
    torch.cuda.synchronize()

    start = time.perf_counter()
    total_out = 0
    for _ in range(args.timed_rounds):
        outputs = llm.generate(prompts, sampling, use_tqdm=False)
        total_out += sum(len(o.outputs[0].token_ids) for o in outputs)
    torch.cuda.synchronize()
    wall = time.perf_counter() - start

    stats = llm.llm_engine.collective_rpc(_parallel_graph_stats)[0]
    return RunResult(
        variant=args.worker_variant,
        config=args.worker_config,
        enforce_eager=bool(options["enforce_eager"]),
        async_scheduling=bool(vllm_config.scheduler_config.async_scheduling),
        cudagraph_mode=str(vllm_config.compilation_config.cudagraph_mode),
        parallel_graphs=stats["graphs"],
        parallel_replays=stats["replays"],
        batch_size=args.batch_size,
        input_len=args.input_len,
        output_len=args.output_len,
        total_output_tokens=total_out,
        wall_seconds=wall,
        tokens_per_second=total_out / wall if wall > 0 else 0.0,
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-len", type=int, default=512)
    p.add_argument("--output-len", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--warmup-rounds", type=int, default=1)
    p.add_argument("--timed-rounds", type=int, default=2)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    p.add_argument("--variants", nargs="*", default=VARIANTS)
    p.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--worker-variant", default="")
    p.add_argument("--worker-config", default="")
    p.add_argument("--worker-out", default="")
    p.add_argument(
        "--out-json",
        default="/workspace/vllm-pDMTD/tools/dmtd/bench_dmtd_ablation_results.json",
    )
    return p.parse_args()


def _print_table(results: list[RunResult]) -> None:
    by_variant: dict[str, dict[str, RunResult]] = {}
    for r in results:
        by_variant.setdefault(r.variant, {})[r.config] = r

    config_names = list(CONFIGS)
    header = ["variant"] + [f"{name} tok/s" for name in config_names] + ["vs eager"]
    rows = []
    for variant, per_config in by_variant.items():
        eager = per_config.get("0_eager")
        best = per_config.get(config_names[-1])
        speedup = (
            best.tokens_per_second / eager.tokens_per_second
            if eager and best and eager.tokens_per_second
            else 0.0
        )
        cells = [
            f"{per_config[name].tokens_per_second:.2f}" if name in per_config else "-"
            for name in config_names
        ]
        rows.append([variant] + cells + [f"{speedup:.2f}x"])
    widths = [max(len(h), max(len(r[i]) for r in rows)) for i, h in enumerate(header)]
    print(" | ".join(h.ljust(widths[i]) for i, h in enumerate(header)))
    print("-+-".join("-" * w for w in widths))
    for row in rows:
        print(" | ".join(row[i].ljust(widths[i]) for i in range(len(header))))


def main() -> None:
    args = parse_args()
    if args.worker:
        result = bench_one(args)
        Path(args.worker_out).write_text(json.dumps(asdict(result), indent=2) + "\n")
        print(json.dumps(asdict(result), indent=2), flush=True)
        gc.collect()
        raise SystemExit(0)

    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    work_dir = out_path.parent / ".bench_dmtd_ablation_tmp"
    work_dir.mkdir(parents=True, exist_ok=True)

    results: list[RunResult] = []
    for variant in args.variants:
        for config in CONFIGS:
            print(f"\n=== {variant} / {config} ===", flush=True)
            worker_out = work_dir / f"{variant}.{config}.json"
            cmd = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--worker",
                "--worker-variant",
                variant,
                "--worker-config",
                config,
                "--worker-out",
                str(worker_out),
                "--input-len",
                str(args.input_len),
                "--output-len",
                str(args.output_len),
                "--batch-size",
                str(args.batch_size),
                "--warmup-rounds",
                str(args.warmup_rounds),
                "--timed-rounds",
                str(args.timed_rounds),
                "--gpu-memory-utilization",
                str(args.gpu_memory_utilization),
            ]
            env = dict(os.environ)
            env.pop(DISABLE_PARALLEL_CG, None)
            env.update(CONFIGS[config][1])
            proc = subprocess.run(cmd, check=False, env=env)
            if proc.returncode != 0:
                raise SystemExit(
                    f"Worker failed for {variant}/{config} (rc={proc.returncode})"
                )
            results.append(RunResult(**json.loads(worker_out.read_text())))

    print("\n=== Summary ===", flush=True)
    _print_table(results)
    out_path.write_text(
        json.dumps(
            {
                "config": dict(
                    input_len=args.input_len,
                    output_len=args.output_len,
                    batch_size=args.batch_size,
                    warmup_rounds=args.warmup_rounds,
                    timed_rounds=args.timed_rounds,
                ),
                "results": [asdict(r) for r in results],
            },
            indent=2,
        )
        + "\n"
    )
    print(f"\nWrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
