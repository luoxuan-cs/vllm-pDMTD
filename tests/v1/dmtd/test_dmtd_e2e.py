# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import gc
import os
import warnings
from pathlib import Path

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from tests.utils import large_gpu_mark
from vllm import LLM, SamplingParams, TokensPrompt

from .oracle import full_recompute_greedy

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


def _history_mode(model_path: str) -> str:
    import json

    with open(Path(model_path) / "config.json", encoding="utf-8") as f:
        cfg = json.load(f)
    return str(cfg.get("dmtd_history_mode", "shadow"))


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


def _hf_expected(
    model_path: str,
    prompts: list[list[int]],
    max_new_tokens: int,
) -> list[list[int]]:
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        dtype=torch.bfloat16,
        device_map="cuda",
    )
    model.eval()
    config = model.config
    expected: list[list[int]] = []
    for prompt in prompts:
        generated, _ = full_recompute_greedy(
            model,
            torch.tensor([prompt], device="cuda"),
            max_new_tokens=max_new_tokens,
            cycle_length=int(config.mtp_horizon),
            mask_token_id=int(config.mask_token_id),
            pad_bidirectional_cycle=(
                getattr(config, "dmtd_block_attention", "causal")
                == "bidirectional"
            ),
        )
        expected.append(generated[0, len(prompt) :].cpu().tolist())
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return expected


def _assert_hf_teacher_forced_close(
    model_path: str,
    prompts: list[list[int]],
    actual: list[list[int]],
) -> None:
    """Accept a different argmax only within one BF16 ULP of the HF maximum."""
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        dtype=torch.bfloat16,
        device_map="cuda",
    )
    model.eval()
    config = model.config
    cycle_length = int(config.mtp_horizon)
    mask_token_id = int(config.mask_token_id)
    bidirectional = (
        getattr(config, "dmtd_block_attention", "causal") == "bidirectional"
    )
    max_margin = 0.0
    with torch.inference_mode():
        for prompt, generated in zip(prompts, actual):
            history = torch.tensor([prompt], device="cuda")
            for token_id in generated:
                forward_ids = history
                if bidirectional:
                    pad_len = (-history.shape[1]) % cycle_length
                    if pad_len:
                        forward_ids = torch.cat(
                            (
                                history,
                                torch.full(
                                    (1, pad_len),
                                    mask_token_id,
                                    dtype=history.dtype,
                                    device=history.device,
                                ),
                            ),
                            dim=1,
                        )
                logits = (
                    model(
                        input_ids=forward_ids,
                        use_cache=False,
                        return_dict=True,
                    )
                    .logits[0, history.shape[1] - 1]
                    .float()
                )
                maximum = logits.max()
                margin = float(maximum - logits[token_id])
                maximum_bf16 = maximum.to(torch.bfloat16)
                previous_bf16 = torch.nextafter(
                    maximum_bf16,
                    torch.full_like(maximum_bf16, -torch.inf),
                )
                one_ulp = float(maximum_bf16.float() - previous_bf16.float())
                assert margin <= one_ulp + 1e-6, (
                    f"token {token_id} is {margin} below the full-recompute "
                    f"maximum, exceeding one BF16 ULP ({one_ulp})"
                )
                max_margin = max(max_margin, margin)
                history = torch.cat(
                    (history, torch.tensor([[token_id]], device="cuda")),
                    dim=1,
                )
    del model
    gc.collect()
    torch.cuda.empty_cache()
    warnings.warn(
        "Exact greedy IDs differed only on BF16 ties/one-ULP boundaries; "
        f"largest teacher-forced HF margin was {max_margin}.",
        stacklevel=2,
    )


@large_gpu_mark(min_gb=32)
@pytest.mark.forked
@pytest.mark.parametrize("max_new_tokens", [2, 5, 9])
def test_real_weight_greedy_parity(max_new_tokens: int):
    model_path = _require_checkpoint()
    prompts = _prompt_token_ids(model_path, (1, 3, 4, 5))
    expected = _hf_expected(model_path, prompts, max_new_tokens)

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
    outputs = llm.generate(
        [TokensPrompt(prompt_token_ids=ids) for ids in prompts],
        SamplingParams(
            temperature=0,
            max_tokens=max_new_tokens,
            ignore_eos=True,
        ),
    )
    actual = [output.outputs[0].token_ids for output in outputs]

    # Short sequences usually match exactly; BF16 ties (seen under continuous
    # batching for Causal-Refresh) fall back to one-ULP teacher-forced checks.
    if actual != expected:
        _assert_hf_teacher_forced_close(model_path, prompts, actual)


@large_gpu_mark(min_gb=32)
@pytest.mark.forked
def test_continuous_batching_different_cycle_phases():
    model_path = _require_checkpoint()
    prompts = _prompt_token_ids(model_path, (3, 5))
    expected = _hf_expected(model_path, prompts, max_new_tokens=6)

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
    outputs = llm.generate(
        [TokensPrompt(prompt_token_ids=ids) for ids in prompts],
        SamplingParams(temperature=0, max_tokens=6, ignore_eos=True),
    )

    actual = [output.outputs[0].token_ids for output in outputs]
    if actual != expected:
        _assert_hf_teacher_forced_close(model_path, prompts, actual)


@large_gpu_mark(min_gb=32)
@pytest.mark.forked
def test_chunked_prefill_across_block_boundary():
    model_path = _require_checkpoint()
    (prompt,) = _prompt_token_ids(model_path, (17,))
    expected = _hf_expected(model_path, [prompt], max_new_tokens=6)

    llm = LLM(
        model=model_path,
        trust_remote_code=False,
        enforce_eager=True,
        tensor_parallel_size=1,
        max_model_len=128,
        max_num_seqs=1,
        max_num_batched_tokens=8,
        enable_chunked_prefill=True,
        enable_prefix_caching=False,
        kv_cache_memory_bytes=4 * 1024**3,
    )
    outputs = llm.generate(
        [TokensPrompt(prompt_token_ids=prompt)],
        SamplingParams(temperature=0, max_tokens=6, ignore_eos=True),
    )
    actual = [output.outputs[0].token_ids for output in outputs]
    if actual != expected:
        _assert_hf_teacher_forced_close(model_path, [prompt], actual)
