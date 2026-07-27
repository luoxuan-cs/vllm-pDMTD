# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""End-to-end DMTD tests against a real checkpoint.

NOTE: the checkpoints under ``parallel-eval/models/`` were trained under the
old semantics (cycle heads anchored to absolute sequence position, prefill
also driven through the masked shadow branch). This engine now runs prefill
as a plain vanilla causal forward and anchors cycle heads to each request's
own prompt length instead, so these checkpoints are expected to produce
*different* generations than before -- that is not a bug, it is the whole
point of the change, and retraining to match is a separate, out-of-scope
follow-up. Comparing against the checkpoint's own ``full_recompute_greedy``
(which still implements the *old* masked-prefill semantics) is therefore no
longer a meaningful correctness oracle.

What these tests check instead are internal self-consistency properties of
the new engine that must hold for *any* checkpoint, independent of whether
its weights were trained for this exact inference scheme:
- generation runs to completion without crashing, across prefill, decode,
  and multi-request batching;
- batching multiple independent requests together in one step produces
  *exactly* the same per-request output as running each one alone -- the
  regression check for the historical "batched attention corrupts non-head
  slots" bug, now exercised with real weights through the grouped
  causal/non-causal attention calls instead of the old per-request loop;
- splitting one request's prompt into chunked-prefill steps produces exactly
  the same output as prefilling it in one shot, since prefill is now a plain
  causal forward that must be invariant to how its input happens to be
  chunked.
"""

import gc
import os
from pathlib import Path

import pytest
import torch
from transformers import AutoTokenizer

from tests.utils import large_gpu_mark
from vllm import LLM, SamplingParams, TokensPrompt

MODEL_PATH = Path(
    os.getenv(
        "DMTD_MODEL_PATH",
        "/workspace/parallel-eval/models/Causal-Parallel-Norefresh",
    )
)


def _require_checkpoint() -> str:
    if not (MODEL_PATH / "model.safetensors.index.json").is_file():
        pytest.skip(f"DMTD checkpoint is not available at {MODEL_PATH}")
    return str(MODEL_PATH)


def _prompt_token_ids(model_path: str, lengths: tuple[int, ...]) -> list[list[int]]:
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    ids = tokenizer.encode(
        "A concise explanation of parallel language model decoding begins with "
        "a causal shadow sequence and continues with sequential refinement.",
        add_special_tokens=False,
    )
    if len(ids) < max(lengths):
        raise AssertionError("The deterministic test prompt tokenized too short")
    return [ids[:length] for length in lengths]


def _generate(llm: LLM, prompts: list[list[int]], max_tokens: int) -> list[list[int]]:
    outputs = llm.generate(
        [TokensPrompt(prompt_token_ids=ids) for ids in prompts],
        SamplingParams(temperature=0, max_tokens=max_tokens, ignore_eos=True),
    )
    return [list(output.outputs[0].token_ids) for output in outputs]


@large_gpu_mark(min_gb=32)
@pytest.mark.forked
@pytest.mark.parametrize("max_new_tokens", [2, 5, 9])
def test_real_weight_generation_completes(max_new_tokens: int):
    """Smoke test: prefill (vanilla causal) + decode (DMTD cycle mechanism)
    both run to completion, for prompts of varying length relative to tau,
    without crashing, and produce valid in-vocabulary token ids."""
    model_path = _require_checkpoint()
    prompts = _prompt_token_ids(model_path, (1, 3, 4, 5))

    llm = LLM(
        model=model_path,
        trust_remote_code=False,
        enforce_eager=True,
        tensor_parallel_size=1,
        max_model_len=256,
        max_num_seqs=len(prompts),
        enable_prefix_caching=False,
        kv_cache_memory_bytes=4 * 1024**3,
    )
    actual = _generate(llm, prompts, max_new_tokens)

    vocab_size = llm.llm_engine.model_config.get_vocab_size()
    for token_ids in actual:
        assert len(token_ids) == max_new_tokens
        assert all(0 <= tok < vocab_size for tok in token_ids)


@large_gpu_mark(min_gb=32)
@pytest.mark.forked
def test_continuous_batching_matches_individual_generation():
    """Two requests with different prompt lengths (so different cycle
    phases) scheduled together must produce exactly the same per-request
    output as running each one alone -- batching independent requests into
    one grouped causal/non-causal call must never change the result."""
    model_path = _require_checkpoint()
    prompts = _prompt_token_ids(model_path, (3, 5))
    max_new_tokens = 6

    llm = LLM(
        model=model_path,
        trust_remote_code=False,
        enforce_eager=True,
        tensor_parallel_size=1,
        max_model_len=128,
        max_num_seqs=2,
        enable_prefix_caching=False,
        kv_cache_memory_bytes=4 * 1024**3,
    )

    together = _generate(llm, prompts, max_new_tokens)
    alone = [_generate(llm, [prompt], max_new_tokens)[0] for prompt in prompts]

    assert together == alone


@large_gpu_mark(min_gb=32)
@pytest.mark.forked
def test_chunked_prefill_matches_unchunked_prefill():
    """Prefill is a plain causal forward, so splitting it into chunks must
    not change the result versus prefilling in one shot."""
    model_path = _require_checkpoint()
    (prompt,) = _prompt_token_ids(model_path, (17,))
    max_new_tokens = 6

    def run(*, max_num_batched_tokens: int, enable_chunked_prefill: bool) -> list[int]:
        llm = LLM(
            model=model_path,
            trust_remote_code=False,
            enforce_eager=True,
            tensor_parallel_size=1,
            max_model_len=128,
            max_num_seqs=1,
            max_num_batched_tokens=max_num_batched_tokens,
            enable_chunked_prefill=enable_chunked_prefill,
            enable_prefix_caching=False,
            kv_cache_memory_bytes=4 * 1024**3,
            # This test builds two engines in one process, and the first one's
            # allocations are not fully returned to the driver by the time the
            # second starts up. The KV cache size is pinned above, so this only
            # relaxes the startup free-memory check.
            gpu_memory_utilization=0.4,
        )
        result = _generate(llm, [prompt], max_new_tokens)[0]
        del llm
        gc.collect()
        torch.cuda.empty_cache()
        return result

    unchunked = run(max_num_batched_tokens=256, enable_chunked_prefill=False)
    chunked = run(max_num_batched_tokens=8, enable_chunked_prefill=True)

    assert chunked == unchunked


def _run_with_engine_options(
    prompts, max_new_tokens, *, collect_stats: bool = False, **llm_options
):
    model_path = _require_checkpoint()
    llm = LLM(
        model=model_path,
        trust_remote_code=False,
        tensor_parallel_size=1,
        max_model_len=128,
        max_num_seqs=len(prompts),
        enable_prefix_caching=False,
        kv_cache_memory_bytes=4 * 1024**3,
        # Two engines are built per test below, and the first one's allocations
        # are not fully returned before the second starts up. KV size is pinned,
        # so this only relaxes the startup free-memory check.
        gpu_memory_utilization=0.4,
        **llm_options,
    )
    result = _generate(llm, prompts, max_new_tokens)
    stats = None
    if collect_stats:
        stats = llm.llm_engine.collective_rpc(_parallel_graph_stats)[0]
    del llm
    gc.collect()
    torch.cuda.empty_cache()
    return (result, stats) if collect_stats else result


def _parallel_graph_stats(worker) -> dict[str, int]:
    """Captured-graph and replay counts for each parallel-layer group."""
    managers = worker.model_runner.model_state._parallel_graph_managers
    return {
        kind: {"graphs": len(m.graphs), "replays": m.num_replays}
        for kind, m in managers.items()
    }


@large_gpu_mark(min_gb=32)
@pytest.mark.forked
def test_cudagraph_decode_matches_eager():
    """The sequential-layer forward is shape-invariant, so it replays for cycle
    heads and mid-cycle steps alike, and the parallel-layer groups replay graphs
    of their own. If any of that dropped work or read a stale buffer the tokens
    would move, so require exact equality over a rollout spanning several cycles.

    The replay count is asserted too: without it this passes trivially whenever
    the graphs silently never engage."""
    model_path = _require_checkpoint()
    prompts = _prompt_token_ids(model_path, (3, 7))
    max_new_tokens = 12

    eager = _run_with_engine_options(prompts, max_new_tokens, enforce_eager=True)
    graphed, stats = _run_with_engine_options(
        prompts, max_new_tokens, enforce_eager=False, collect_stats=True
    )

    assert graphed == eager
    assert stats, "no parallel-layer graph manager was created"
    for kind, counts in stats.items():
        assert counts["graphs"] > 0, f"{kind}: no graph captured"
        assert counts["replays"] > 0, f"{kind}: graphs captured but never replayed"


@large_gpu_mark(min_gb=32)
@pytest.mark.forked
def test_async_scheduling_matches_sync_scheduling():
    """Async scheduling lets the scheduler run a step ahead of sampling. The
    cycle planners only ever look at token positions, and the shadow block's
    real token ids are gathered from the GPU-resident history, so results must
    not change."""
    model_path = _require_checkpoint()
    prompts = _prompt_token_ids(model_path, (3, 7))
    max_new_tokens = 12

    sync = _run_with_engine_options(prompts, max_new_tokens, async_scheduling=False)
    async_ = _run_with_engine_options(prompts, max_new_tokens, async_scheduling=True)

    assert async_ == sync
