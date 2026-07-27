#!/usr/bin/env python3
"""Fair vLLM speed comparison: 4 DMTD variants + baseline Qwen3-4B.

Fixed input_len / output_len / batch_size for every model.
Each model runs in an isolated subprocess so GPU memory is fully released.
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

DEFAULT_MODELS = [
    (
        "Qwen3-4B",
        os.environ.get("QWEN3_4B_PATH", "/workspace/models/Qwen3-4B"),
    ),
    (
        "Causal-Parallel-Norefresh",
        "/workspace/parallel-eval/models/Causal-Parallel-Norefresh",
    ),
    (
        "Causal-Parallel-Refresh",
        "/workspace/parallel-eval/models/Causal-Parallel-Refresh",
    ),
    (
        "Bidirectional-Parallel-Norefresh",
        "/workspace/parallel-eval/models/Bidirectional-Parallel-Norefresh",
    ),
    (
        "Bidirectional-Parallel-Refresh",
        "/workspace/parallel-eval/models/Bidirectional-Parallel-Refresh",
    ),
]


@dataclass
class RunResult:
    name: str
    model_path: str
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

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path, trust_remote_code=True
    )
    seed = (
        "Write a detailed technical explanation of parallel decoding for "
        "transformer language models. Cover prefill, decode, KV cache, "
        "memory bandwidth, and speculative methods. "
    )
    seed_ids = tokenizer.encode(seed, add_special_tokens=False)
    if not seed_ids:
        raise RuntimeError(f"Tokenizer produced empty ids for {tokenizer_path}")
    ids = (seed_ids * ((input_len // len(seed_ids)) + 2))[:input_len]
    assert len(ids) == input_len
    return [TokensPrompt(prompt_token_ids=list(ids)) for _ in range(batch_size)]


def bench_one_inprocess(
    name: str,
    model_path: str,
    *,
    batch_size: int,
    input_len: int,
    output_len: int,
    warmup_rounds: int,
    timed_rounds: int,
    max_model_len: int,
    gpu_memory_utilization: float,
    enforce_eager: bool,
) -> RunResult:
    import torch

    from vllm import LLM, SamplingParams

    if not Path(model_path).exists():
        raise FileNotFoundError(f"[{name}] model path missing: {model_path}")

    prompts = _build_prompts(model_path, batch_size, input_len)
    sampling = SamplingParams(
        temperature=0.0,
        max_tokens=output_len,
        ignore_eos=True,
    )

    llm = LLM(
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
    )

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

    expected = batch_size * output_len * timed_rounds
    if total_out != expected:
        print(
            f"[warn] {name}: expected {expected} output tokens, got {total_out}",
            flush=True,
        )

    tps = total_out / wall if wall > 0 else 0.0
    latency_ms = (wall / (batch_size * timed_rounds)) * 1000.0
    return RunResult(
        name=name,
        model_path=model_path,
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


def _print_table(results: list[RunResult]) -> None:
    if not results:
        return
    baseline = next((r for r in results if r.name == "Qwen3-4B"), results[0])
    headers = (
        "model",
        "tok/s",
        "vs_base",
        "lat_ms/req",
        "peak_GB",
        "wall_s",
    )
    rows = []
    for r in results:
        speedup = (
            r.tokens_per_second / baseline.tokens_per_second
            if baseline.tokens_per_second > 0
            else 0.0
        )
        rows.append(
            (
                r.name,
                f"{r.tokens_per_second:.2f}",
                f"{speedup:.2f}x",
                f"{r.latency_per_request_ms:.1f}",
                f"{r.peak_gpu_mem_gb:.2f}",
                f"{r.wall_seconds:.2f}",
            )
        )
    widths = [
        max(len(h), max(len(row[i]) for row in rows))
        for i, h in enumerate(headers)
    ]
    line = " | ".join(h.ljust(widths[i]) for i, h in enumerate(headers))
    sep = "-+-".join("-" * widths[i] for i in range(len(headers)))
    print(line)
    print(sep)
    for row in rows:
        print(" | ".join(row[i].ljust(widths[i]) for i in range(len(headers))))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-len", type=int, default=512)
    p.add_argument("--output-len", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--warmup-rounds", type=int, default=1)
    p.add_argument("--timed-rounds", type=int, default=2)
    p.add_argument("--max-model-len", type=int, default=0)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    p.add_argument(
        "--enforce-eager",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "True compares both eager. Pass --no-enforce-eager to compare each "
            "at its best: DMTD now captures CUDA graphs for the sequential "
            "layers and for the parallel-layer groups."
        ),
    )
    p.add_argument(
        "--only",
        nargs="*",
        default=None,
        help="Optional subset of model names to run.",
    )
    p.add_argument(
        "--worker",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    p.add_argument("--worker-name", type=str, default="")
    p.add_argument("--worker-path", type=str, default="")
    p.add_argument("--worker-out", type=str, default="")
    p.add_argument(
        "--out-json",
        type=str,
        default="/workspace/vllm-pDMTD/tools/dmtd/bench_compare_results.json",
    )
    return p.parse_args()


def _run_worker(args: argparse.Namespace) -> int:
    max_model_len = args.max_model_len or (args.input_len + args.output_len + 64)
    result = bench_one_inprocess(
        args.worker_name,
        args.worker_path,
        batch_size=args.batch_size,
        input_len=args.input_len,
        output_len=args.output_len,
        warmup_rounds=args.warmup_rounds,
        timed_rounds=args.timed_rounds,
        max_model_len=max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=args.enforce_eager,
    )
    Path(args.worker_out).write_text(json.dumps(asdict(result), indent=2) + "\n")
    print(json.dumps(asdict(result), indent=2), flush=True)
    # Explicitly drop references before process exit.
    del result
    gc.collect()
    return 0


def main() -> None:
    args = parse_args()
    if args.worker:
        raise SystemExit(_run_worker(args))

    max_model_len = args.max_model_len or (args.input_len + args.output_len + 64)
    models = DEFAULT_MODELS
    if args.only:
        wanted = set(args.only)
        models = [m for m in models if m[0] in wanted]
        missing = wanted - {m[0] for m in models}
        if missing:
            raise SystemExit(f"Unknown model names: {sorted(missing)}")

    config = {
        "input_len": args.input_len,
        "output_len": args.output_len,
        "batch_size": args.batch_size,
        "warmup_rounds": args.warmup_rounds,
        "timed_rounds": args.timed_rounds,
        "max_model_len": max_model_len,
        "enforce_eager": args.enforce_eager,
        "gpu_memory_utilization": args.gpu_memory_utilization,
    }
    print(json.dumps(config, indent=2), flush=True)

    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    work_dir = out_path.parent / ".bench_compare_tmp"
    work_dir.mkdir(parents=True, exist_ok=True)

    results: list[RunResult] = []
    for name, path in models:
        print(f"\n=== Running {name} ===", flush=True)
        print(f"path={path}", flush=True)
        worker_out = work_dir / f"{name}.json"
        cmd = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--worker",
            "--worker-name",
            name,
            "--worker-path",
            path,
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
            "--max-model-len",
            str(max_model_len),
            "--gpu-memory-utilization",
            str(args.gpu_memory_utilization),
        ]
        if args.enforce_eager:
            cmd.append("--enforce-eager")
        else:
            cmd.append("--no-enforce-eager")

        proc = subprocess.run(cmd, check=False)
        if proc.returncode != 0:
            raise SystemExit(f"Worker failed for {name} (rc={proc.returncode})")
        payload = json.loads(worker_out.read_text())
        results.append(RunResult(**payload))

    print("\n=== Summary ===", flush=True)
    _print_table(results)
    out = {"config": config, "results": [asdict(r) for r in results]}
    out_path.write_text(json.dumps(out, indent=2) + "\n")
    print(f"\nWrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
