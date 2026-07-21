# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from tests.v1.core.utils import create_requests
from vllm.config import (
    CacheConfig,
    ModelConfig,
    ParallelConfig,
    SchedulerConfig,
    VllmConfig,
)
from vllm.transformers_utils.configs.dmtd_qwen3 import DMTDQwen3Config
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
)
from vllm.v1.structured_output import StructuredOutputManager

BLOCK_SIZE = 16
NUM_BLOCKS = 8


def _create_dmtd_scheduler(
    tmp_path,
    *,
    enable_prefix_caching: bool = False,
    dmtd_history_mode: str = "shadow",
    dmtd_block_attention: str = "causal",
) -> Scheduler:
    config = DMTDQwen3Config(
        architectures=["DMTDQwen3ForCausalLM"],
        vocab_size=128,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=36,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        num_parallel_layers=28,
        num_sequential_layers=8,
        mtp_horizon=4,
        mask_token_id=127,
        max_position_embeddings=128,
        dmtd_history_mode=dmtd_history_mode,
        dmtd_block_attention=dmtd_block_attention,
    )
    config.save_pretrained(tmp_path)
    model_config = ModelConfig(
        model=str(tmp_path),
        runner="generate",
        max_model_len=100,
        trust_remote_code=False,
        skip_tokenizer_init=True,
    )
    scheduler_config = SchedulerConfig(
        max_num_seqs=16,
        max_num_batched_tokens=8192,
        max_model_len=model_config.max_model_len,
        is_encoder_decoder=model_config.is_encoder_decoder,
    )
    cache_config = CacheConfig(
        block_size=BLOCK_SIZE,
        gpu_memory_utilization=0.9,
        cache_dtype="auto",
        enable_prefix_caching=enable_prefix_caching,
    )
    vllm_config = VllmConfig(
        scheduler_config=scheduler_config,
        model_config=model_config,
        cache_config=cache_config,
        parallel_config=ParallelConfig(),
    )
    kv_cache_config = KVCacheConfig(
        num_blocks=NUM_BLOCKS,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["layer"],
                FullAttentionSpec(
                    block_size=BLOCK_SIZE,
                    num_kv_heads=1,
                    head_size=1,
                    dtype=torch.float32,
                ),
            )
        ],
    )
    cache_config.num_gpu_blocks = NUM_BLOCKS
    return Scheduler(
        vllm_config=vllm_config,
        kv_cache_config=kv_cache_config,
        block_size=BLOCK_SIZE,
        log_stats=True,
        structured_output_manager=StructuredOutputManager(vllm_config),
    )


def test_dmtd_reserves_cycle_lookahead_without_spec_decode(tmp_path):
    scheduler = _create_dmtd_scheduler(tmp_path)

    assert scheduler.vllm_config.speculative_config is None
    assert scheduler.num_lookahead_tokens == 3
    assert scheduler.vllm_config.scheduler_config.async_scheduling is False
    assert scheduler._dmtd_refresh_cycle_align is False

    (request,) = create_requests(
        num_requests=1,
        num_tokens=BLOCK_SIZE,
        block_size=BLOCK_SIZE,
    )
    scheduler.add_request(request)
    output = scheduler.schedule()

    assert output.scheduled_spec_decode_tokens == {}
    assert output.num_scheduled_tokens[request.request_id] == BLOCK_SIZE
    assert len(output.scheduled_new_reqs[0].block_ids[0]) == 2


def test_dmtd_shadow_window_fits_allocated_blocks(tmp_path):
    scheduler = _create_dmtd_scheduler(tmp_path)
    (request,) = create_requests(
        num_requests=1,
        num_tokens=BLOCK_SIZE,
        block_size=BLOCK_SIZE,
    )
    scheduler.add_request(request)
    output = scheduler.schedule()

    block_ids = output.scheduled_new_reqs[0].block_ids[0]
    shadow_positions = range(BLOCK_SIZE, BLOCK_SIZE + 3)
    assert all(pos // BLOCK_SIZE < len(block_ids) for pos in shadow_positions)


def test_dmtd_rejects_prefix_caching(tmp_path):
    with pytest.raises(ValueError, match="prefix caching"):
        _create_dmtd_scheduler(tmp_path, enable_prefix_caching=True)


def test_dmtd_refresh_cycle_aligned_prefill_clips_to_tau(tmp_path):
    """Refresh history_mode=real: first schedule clips to one cycle (≤4)."""
    scheduler = _create_dmtd_scheduler(tmp_path, dmtd_history_mode="real")
    assert scheduler._dmtd_refresh_cycle_align is True
    assert scheduler._dmtd_cycle_length == 4
    assert scheduler.num_lookahead_tokens == 3

    (request,) = create_requests(
        num_requests=1,
        num_tokens=5,
        block_size=BLOCK_SIZE,
    )
    scheduler.add_request(request)
    output = scheduler.schedule()

    assert output.num_scheduled_tokens[request.request_id] == 4


def test_dmtd_norefresh_does_not_clip_prefill_to_cycle(tmp_path):
    """Default shadow mode keeps scheduling the full prompt in one chunk."""
    scheduler = _create_dmtd_scheduler(tmp_path, dmtd_history_mode="shadow")
    assert scheduler._dmtd_refresh_cycle_align is False

    (request,) = create_requests(
        num_requests=1,
        num_tokens=5,
        block_size=BLOCK_SIZE,
    )
    scheduler.add_request(request)
    output = scheduler.schedule()

    assert output.num_scheduled_tokens[request.request_id] == 5


@pytest.mark.parametrize(
    ("prompt_len", "computed", "budget", "expected"),
    [
        (5, 0, 8192, 4),
        (5, 4, 8192, 1),
        (3, 0, 8192, 3),
        (8, 0, 2, 2),
        (8, 3, 8192, 1),
    ],
)
def test_dmtd_refresh_cycle_split_helper(
    tmp_path, prompt_len, computed, budget, expected
):
    scheduler = _create_dmtd_scheduler(tmp_path, dmtd_history_mode="real")
    (request,) = create_requests(
        num_requests=1,
        num_tokens=prompt_len,
        block_size=BLOCK_SIZE,
    )
    request.num_computed_tokens = computed
    clipped = scheduler._dmtd_refresh_cycle_aligned_split(
        request, min(prompt_len - computed, budget), num_computed_tokens=computed
    )
    assert clipped == expected
