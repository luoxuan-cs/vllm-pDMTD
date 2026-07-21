#!/usr/bin/env python3
"""Small correctness-baseline throughput comparison for DMTD."""

import gc
import json
import os
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from tests.v1.dmtd.oracle import full_recompute_greedy
from vllm import LLM, SamplingParams, TokensPrompt

model_path = os.getenv(
    "DMTD_MODEL_PATH",
    "/workspace/parallel-eval/models/Causal-Parallel-Norefresh",
)
max_new_tokens = int(os.getenv("DMTD_BENCH_TOKENS", "32"))
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
seed_ids = tokenizer.encode(
    "Parallel decoding should preserve model semantics while reducing latency. "
    "This deterministic benchmark compares the reference path with the native path.",
    add_special_tokens=False,
)
prompts = [seed_ids[:length] for length in (8, 12, 16, 20)]

hf_model = AutoModelForCausalLM.from_pretrained(
    model_path,
    trust_remote_code=True,
    dtype=torch.bfloat16,
    device_map="cuda",
).eval()
torch.cuda.synchronize()
start = time.perf_counter()
for prompt in prompts:
    full_recompute_greedy(
        hf_model,
        torch.tensor([prompt], device="cuda"),
        max_new_tokens=max_new_tokens,
        cycle_length=int(hf_model.config.mtp_horizon),
        mask_token_id=int(hf_model.config.mask_token_id),
        pad_bidirectional_cycle=(
            getattr(hf_model.config, "dmtd_block_attention", "causal")
            == "bidirectional"
        ),
    )
torch.cuda.synchronize()
hf_seconds = time.perf_counter() - start
del hf_model
gc.collect()
torch.cuda.empty_cache()

llm = LLM(
    model=model_path,
    trust_remote_code=False,
    enforce_eager=True,
    enable_prefix_caching=False,
    max_model_len=256,
    max_num_seqs=len(prompts),
    kv_cache_memory_bytes=4 * 1024**3,
)
sampling = SamplingParams(
    temperature=0,
    max_tokens=max_new_tokens,
    ignore_eos=True,
)
torch.cuda.synchronize()
start = time.perf_counter()
outputs = llm.generate(
    [TokensPrompt(prompt_token_ids=prompt) for prompt in prompts],
    sampling,
    use_tqdm=False,
)
torch.cuda.synchronize()
vllm_seconds = time.perf_counter() - start
num_output_tokens = sum(len(output.outputs[0].token_ids) for output in outputs)

result = {
    "model": model_path,
    "prompts": len(prompts),
    "output_tokens": num_output_tokens,
    "hf_full_recompute_seconds": hf_seconds,
    "hf_full_recompute_tokens_per_second": num_output_tokens / hf_seconds,
    "vllm_eager_seconds": vllm_seconds,
    "vllm_eager_tokens_per_second": num_output_tokens / vllm_seconds,
    "speedup": hf_seconds / vllm_seconds,
}
print(json.dumps(result, indent=2, sort_keys=True))
