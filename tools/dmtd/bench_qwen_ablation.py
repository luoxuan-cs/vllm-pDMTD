#!/usr/bin/env python3
"""Ablation benchmark: start from Qwen3-4B with ALL vLLM optimizations that
DMTD is forced to give up (torch.compile, CUDA graphs, async scheduling),
then remove them one at a time to measure each optimization's contribution.

DMTDQwen3ForCausalLMConfig.verify_and_update_config (see
vllm/model_executor/models/config.py) unconditionally forces:
  - enforce_eager=True, compilation_config.mode=NONE, cudagraph_mode=NONE
  - rejects enable_prefix_caching
  - rejects async_scheduling
  - rejects speculative_config, quant_config, PP>1, CP>1, DBO, KV connectors

Of these, only {torch.compile, CUDA graphs, async scheduling} are meaningful
to ablate on a single GPU, bf16, no-draft-model, no-quant setup (the rest
either require multi-GPU/multi-node or change numerics). Prefix caching is
intentionally NOT included here: its benefit is entirely a function of how
much prompt content is shared across requests in real traffic, and using
identical repeated prompts (as this fixed-length benchmark does) would give
it an unbounded, unrealistic advantage rather than a meaningful fixed
percentage.

Each named config is run in an isolated subprocess (fresh CUDA context) so
GPU memory from one config never leaks into the next.
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

DEFAULT_MODEL_PATH = os.environ.get("QWEN3_4B_PATH", "/workspace/models/Qwen3-4B")

# name -> (enforce_eager, compilation_config dict or None, async_scheduling)
# compilation_config=None means "let vLLM's default optimization_level (O2)
# decide" (only meaningful when enforce_eager=False).
CONFIGS: list[tuple[str, dict]] = [
    (
        "0_eager_baseline",
        dict(
            enforce_eager=True,
            compilation_config=None,
            async_scheduling=False,
        ),
    ),
    (
        "1_cudagraph_only",
        dict(
            enforce_eager=False,
            compilation_config={"mode": 0, "cudagraph_mode": "FULL"},
            async_scheduling=False,
        ),
    ),
    (
        "2_compile_only",
        dict(
            enforce_eager=False,
            compilation_config={"mode": 3, "cudagraph_mode": "NONE"},
            async_scheduling=False,
        ),
    ),
    (
        "3_compile_and_cudagraph",
        dict(
            enforce_eager=False,
            compilation_config=None,  # default O2: VLLM_COMPILE + FULL_AND_PIECEWISE
            async_scheduling=False,
        ),
    ),
    (
        "4_all_incl_async_scheduling",
        dict(
            enforce_eager=False,
            compilation_config=None,
            async_scheduling=True,
        ),
    ),
]


@dataclass
class RunResult:
    name: str
    enforce_eager: bool
    compilation_mode: str
    cudagraph_mode: str
    async_scheduling: bool
    batch_size: int
    input_len: int
    output_len: int
    warmup_rounds: int
    timed_rounds: int
    total_output_tokens: int
    wall_seconds: float
    tokens_per_second: float
    latency_per_request_ms: float
    peak_gpu_mem_gb: float


def _build_prompts(tokenizer_path: str, batch_size: int, input_len: int):
    from transformers import AutoTokenizer

    from vllm import TokensPrompt

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    seed = (
        "Write a detailed technical explanation of parallel decoding for "
        "transformer language models. Cover prefill, decode, KV cache, "
        "memory bandwidth, and speculative methods. "
    )
    seed_ids = tokenizer.encode(seed, add_special_tokens=False)
    ids = (seed_ids * ((input_len // len(seed_ids)) + 2))[:input_len]
    return [TokensPrompt(prompt_token_ids=list(ids)) for _ in range(batch_size)]


def bench_one_inprocess(
    name: str,
    model_path: str,
    engine_kwargs: dict,
    *,
    batch_size: int,
    input_len: int,
    output_len: int,
    warmup_rounds: int,
    timed_rounds: int,
    max_model_len: int,
    gpu_memory_utilization: float,
) -> RunResult:
    import torch

    from vllm import LLM, SamplingParams

    prompts = _build_prompts(model_path, batch_size, input_len)
    sampling = SamplingParams(temperature=0.0, max_tokens=output_len, ignore_eos=True)

    enforce_eager = engine_kwargs["enforce_eager"]
    compilation_config = engine_kwargs["compilation_config"]
    async_scheduling = engine_kwargs["async_scheduling"]

    llm_kwargs = dict(
        model=model_path,
        trust_remote_code=True,
        dtype="bfloat16",
        enforce_eager=enforce_eager,
        enable_prefix_caching=False,
        max_model_len=max_model_len,
        max_num_seqs=batch_size,
        max_num_batched_tokens=max(max_model_len, batch_size * input_len),
        gpu_memory_utilization=gpu_memory_utilization,
        disable_log_stats=True,
        async_scheduling=async_scheduling,
    )
    if compilation_config is not None:
        llm_kwargs["compilation_config"] = compilation_config

    llm = LLM(**llm_kwargs)

    vllm_config = llm.llm_engine.vllm_config
    actual_mode = str(vllm_config.compilation_config.mode)
    actual_cudagraph = str(vllm_config.compilation_config.cudagraph_mode)
    actual_async = bool(vllm_config.scheduler_config.async_scheduling)

    for _ in range(warmup_rounds):
        llm.generate(prompts, sampling, use_tqdm=False)
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    start = time.perf_counter()
    total_out = 0
    for _ in range(timed_rounds):
        outputs = llm.generate(prompts, sampling, use_tqdm=False)
        total_out += sum(len(o.outputs[0].token_ids) for o in outputs)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    wall = time.perf_counter() - start

    peak = 0.0
    if torch.cuda.is_available():
        peak = torch.cuda.max_memory_allocated() / (1024**3)
    tps = total_out / wall if wall > 0 else 0.0
    latency_ms = (wall / (batch_size * timed_rounds)) * 1000.0

    return RunResult(
        name=name,
        enforce_eager=enforce_eager,
        compilation_mode=actual_mode,
        cudagraph_mode=actual_cudagraph,
        async_scheduling=actual_async,
        batch_size=batch_size,
        input_len=input_len,
        output_len=output_len,
        warmup_rounds=warmup_rounds,
        timed_rounds=timed_rounds,
        total_output_tokens=total_out,
        wall_seconds=wall,
        tokens_per_second=tps,
        latency_per_request_ms=latency_ms,
        peak_gpu_mem_gb=peak,
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default=DEFAULT_MODEL_PATH)
    p.add_argument("--input-len", type=int, default=512)
    p.add_argument("--output-len", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--warmup-rounds", type=int, default=2)
    p.add_argument("--timed-rounds", type=int, default=3)
    p.add_argument("--max-model-len", type=int, default=0)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    p.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--worker-config", type=str, default="")
    p.add_argument("--worker-out", type=str, default="")
    p.add_argument(
        "--out-json",
        default="/workspace/vllm-pDMTD/tools/dmtd/bench_qwen_ablation_results.json",
    )
    return p.parse_args()


def _run_worker(args: argparse.Namespace) -> int:
    max_model_len = args.max_model_len or (args.input_len + args.output_len + 64)
    engine_kwargs = dict(CONFIGS)[args.worker_config]
    result = bench_one_inprocess(
        args.worker_config,
        args.model,
        engine_kwargs,
        batch_size=args.batch_size,
        input_len=args.input_len,
        output_len=args.output_len,
        warmup_rounds=args.warmup_rounds,
        timed_rounds=args.timed_rounds,
        max_model_len=max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
    Path(args.worker_out).write_text(json.dumps(asdict(result), indent=2) + "\n")
    print(json.dumps(asdict(result), indent=2), flush=True)
    gc.collect()
    return 0


def _print_table(results: list[RunResult]) -> None:
    full = next(
        (r for r in results if r.name == "4_all_incl_async_scheduling"), results[-1]
    )
    headers = ("config", "compile", "cudagraph", "async_sched", "tok/s", "vs_full")
    rows = []
    for r in results:
        vs_full = (
            r.tokens_per_second / full.tokens_per_second
            if full.tokens_per_second
            else 0.0
        )
        rows.append(
            (
                r.name,
                r.compilation_mode,
                r.cudagraph_mode,
                str(r.async_scheduling),
                f"{r.tokens_per_second:.2f}",
                f"{vs_full:.2f}x",
            )
        )
    widths = [
        max(len(h), max(len(row[i]) for row in rows)) for i, h in enumerate(headers)
    ]
    print(" | ".join(h.ljust(widths[i]) for i, h in enumerate(headers)))
    print("-+-".join("-" * w for w in widths))
    for row in rows:
        print(" | ".join(row[i].ljust(widths[i]) for i in range(len(headers))))


def main() -> None:
    args = parse_args()
    if args.worker:
        raise SystemExit(_run_worker(args))

    max_model_len = args.max_model_len or (args.input_len + args.output_len + 64)
    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    work_dir = out_path.parent / ".bench_qwen_ablation_tmp"
    work_dir.mkdir(parents=True, exist_ok=True)

    print(
        json.dumps(
            dict(
                model=args.model,
                input_len=args.input_len,
                output_len=args.output_len,
                batch_size=args.batch_size,
                warmup_rounds=args.warmup_rounds,
                timed_rounds=args.timed_rounds,
                max_model_len=max_model_len,
            ),
            indent=2,
        ),
        flush=True,
    )

    results: list[RunResult] = []
    for name, _ in CONFIGS:
        print(f"\n=== Running {name} ===", flush=True)
        worker_out = work_dir / f"{name}.json"
        cmd = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--worker",
            "--worker-config",
            name,
            "--worker-out",
            str(worker_out),
            "--model",
            args.model,
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
            "--max-model-len",
            str(max_model_len),
            "--gpu-memory-utilization",
            str(args.gpu_memory_utilization),
        ]
        proc = subprocess.run(cmd, check=False)
        if proc.returncode != 0:
            raise SystemExit(f"Worker failed for {name} (rc={proc.returncode})")
        payload = json.loads(worker_out.read_text())
        results.append(RunResult(**payload))

    print("\n=== Summary ===", flush=True)
    _print_table(results)

    out = {
        "config": dict(
            model=args.model,
            input_len=args.input_len,
            output_len=args.output_len,
            batch_size=args.batch_size,
            warmup_rounds=args.warmup_rounds,
            timed_rounds=args.timed_rounds,
            max_model_len=max_model_len,
        ),
        "results": [asdict(r) for r in results],
    }
    out_path.write_text(json.dumps(out, indent=2) + "\n")
    print(f"\nWrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
