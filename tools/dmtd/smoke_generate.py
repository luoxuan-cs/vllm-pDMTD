#!/usr/bin/env python3
"""Deterministic generation smoke check across DMTD variants.

Prints the greedy token ids for a couple of fixed prompts so that a refactor
can be compared against a previously recorded run (see --expect-json).
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

VARIANTS = [
    "Causal-Parallel-Norefresh",
    "Causal-Parallel-Refresh",
    "Bidirectional-Parallel-Norefresh",
    "Bidirectional-Parallel-Refresh",
]
MODEL_ROOT = os.environ.get("DMTD_MODEL_ROOT", "/workspace/parallel-eval/models")


def run_variant(
    name: str,
    max_tokens: int,
    prompt_lens: list[int],
    *,
    enforce_eager: bool | None = None,
    async_scheduling: bool | None = None,
) -> dict:
    from transformers import AutoTokenizer

    from vllm import LLM, SamplingParams, TokensPrompt

    path = f"{MODEL_ROOT}/{name}"
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    seed_ids = tokenizer.encode(
        "Parallel decoding reduces the number of layers each token traverses. "
        "This smoke test pins greedy outputs so refactors stay verifiable. ",
        add_special_tokens=False,
    )
    prompts = []
    for length in prompt_lens:
        ids = (seed_ids * ((length // len(seed_ids)) + 2))[:length]
        prompts.append(TokensPrompt(prompt_token_ids=ids))

    llm_kwargs = dict(
        model=path,
        trust_remote_code=True,
        dtype="bfloat16",
        enable_prefix_caching=False,
        max_model_len=max(prompt_lens) + max_tokens + 16,
        max_num_seqs=len(prompts),
        gpu_memory_utilization=0.85,
        disable_log_stats=True,
    )
    if enforce_eager is not None:
        llm_kwargs["enforce_eager"] = enforce_eager
    if async_scheduling is not None:
        llm_kwargs["async_scheduling"] = async_scheduling
    llm = LLM(**llm_kwargs)
    outputs = llm.generate(
        prompts,
        SamplingParams(temperature=0.0, max_tokens=max_tokens, ignore_eos=True),
        use_tqdm=False,
    )
    return {
        "prompt_lens": prompt_lens,
        "token_ids": [list(o.outputs[0].token_ids) for o in outputs],
        "texts": [o.outputs[0].text for o in outputs],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True, choices=VARIANTS)
    ap.add_argument("--max-tokens", type=int, default=16)
    ap.add_argument("--prompt-lens", default="5,17")
    ap.add_argument(
        "--enforce-eager", action=argparse.BooleanOptionalAction, default=None
    )
    ap.add_argument(
        "--async-scheduling", action=argparse.BooleanOptionalAction, default=None
    )
    ap.add_argument("--out-json", default="")
    ap.add_argument(
        "--expect-json",
        default="",
        help="Compare against a previously recorded run and exit non-zero on drift.",
    )
    args = ap.parse_args()

    prompt_lens = [int(x) for x in args.prompt_lens.split(",") if x]
    result = run_variant(
        args.variant,
        args.max_tokens,
        prompt_lens,
        enforce_eager=args.enforce_eager,
        async_scheduling=args.async_scheduling,
    )
    print(json.dumps({args.variant: result}, indent=2, ensure_ascii=False))

    if args.out_json:
        path = Path(args.out_json)
        existing = json.loads(path.read_text()) if path.exists() else {}
        existing[args.variant] = result
        path.write_text(json.dumps(existing, indent=2, ensure_ascii=False) + "\n")
        print(f"Wrote {path}")

    if args.expect_json:
        expected = json.loads(Path(args.expect_json).read_text())[args.variant]
        if expected["token_ids"] != result["token_ids"]:
            raise SystemExit(
                f"[{args.variant}] token drift\n"
                f"expected={expected['token_ids']}\ngot     ={result['token_ids']}"
            )
        print(f"[{args.variant}] matches recorded baseline")


if __name__ == "__main__":
    main()
