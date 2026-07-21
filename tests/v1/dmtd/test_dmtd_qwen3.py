# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from vllm.model_executor.models.config import DMTDQwen3ForCausalLMConfig
from vllm.transformers_utils.configs.dmtd_qwen3 import DMTDQwen3Config
from vllm.v1.attention.backends.flash_attn import FlashAttentionMetadata
from vllm.v1.attention.backends.registry import AttentionBackendEnum
from vllm.v1.worker.gpu.model_states.dmtd_qwen3 import (
    DMTDQwen3ModelState,
    RefreshCyclePlanner,
    _make_dmtd_bidirectional_real_history_mask,
    dmtd_bidirectional_shadow_history_mask,
)


def _tiny_config(**overrides) -> DMTDQwen3Config:
    kwargs = dict(
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
    )
    kwargs.update(overrides)
    return DMTDQwen3Config(**kwargs)


def _state(*, block_attention: str = "causal") -> DMTDQwen3ModelState:
    state = object.__new__(DMTDQwen3ModelState)
    state.tau = 4
    state.max_model_len = 64
    state.block_attention = block_attention
    state._cycle_base = np.full(2, -1, dtype=np.int64)
    state._next_computed = np.full(2, -1, dtype=np.int64)
    state._buffer_valid = np.zeros((2, 4), dtype=np.bool_)
    state._parallel_real_len = np.zeros(2, dtype=np.int64)
    return state


def _refresh_state() -> DMTDQwen3ModelState:
    state = _state()
    state.mask_token_id = 127
    state.history_mode = "real"
    return state


def _batch(
    *,
    start: int,
    query_len: int,
    is_prefilling: bool,
) -> SimpleNamespace:
    return SimpleNamespace(
        num_reqs=1,
        num_tokens=query_len,
        idx_mapping_np=np.array([0], dtype=np.int32),
        num_computed_tokens_np=np.array([start], dtype=np.int32),
        num_scheduled_tokens=np.array([query_len], dtype=np.int32),
        query_start_loc_np=np.array([0, query_len], dtype=np.int32),
        is_prefilling_np=np.array([is_prefilling], dtype=np.bool_),
    )


def test_config_rejects_non_checkpoint_layout():
    with pytest.raises(ValueError, match="parallel / 8 sequential"):
        DMTDQwen3Config(
            vocab_size=128,
            num_hidden_layers=4,
            num_parallel_layers=2,
            num_sequential_layers=2,
            mtp_horizon=4,
            mask_token_id=127,
        )


def test_config_accepts_real_causal_and_default_shadow_causal():
    refresh = _tiny_config(dmtd_history_mode="real", dmtd_block_attention="causal")
    assert refresh.dmtd_history_mode == "real"
    assert refresh.dmtd_block_attention == "causal"

    norefresh = _tiny_config()
    assert norefresh.dmtd_history_mode == "shadow"
    assert norefresh.dmtd_block_attention == "causal"


def test_model_config_enables_flex_only_for_bidirectional_parallel():
    hf_config = _tiny_config(dmtd_block_attention="bidirectional")
    vllm_config = MagicMock()
    vllm_config.model_config.hf_config = hf_config
    vllm_config.attention_config.backend = None
    vllm_config.attention_config.use_non_causal = False
    vllm_config.speculative_config = None
    vllm_config.quant_config = None
    vllm_config.cache_config.enable_prefix_caching = False
    vllm_config.parallel_config.pipeline_parallel_size = 1
    vllm_config.parallel_config.decode_context_parallel_size = 1
    vllm_config.parallel_config.prefill_context_parallel_size = 1
    vllm_config.parallel_config.enable_dbo = False
    vllm_config.kv_transfer_config = None
    vllm_config.scheduler_config.async_scheduling = False
    vllm_config.use_v2_model_runner = True
    vllm_config.compilation_config = MagicMock()

    DMTDQwen3ForCausalLMConfig.verify_and_update_config(vllm_config)

    assert vllm_config.attention_config.backend == AttentionBackendEnum.FLEX_ATTENTION
    # The custom parallel-layer metadata supplies its bidirectional same-cycle
    # mask. Model-wide non-causal mode would incorrectly make the sequential
    # layers bidirectional.
    assert vllm_config.attention_config.use_non_causal is False


def test_model_config_rejects_non_flex_bidirectional_backend():
    hf_config = _tiny_config(dmtd_block_attention="bidirectional")
    vllm_config = MagicMock()
    vllm_config.model_config.hf_config = hf_config
    vllm_config.attention_config.backend = AttentionBackendEnum.FLASH_ATTN

    with pytest.raises(ValueError, match="FLEX_ATTENTION"):
        DMTDQwen3ForCausalLMConfig.verify_and_update_config(vllm_config)


def test_parallel_metadata_ownership_preserves_shared_layer_aliases():
    state = _state(block_attention="bidirectional")
    state._parallel_layer_names = ["layer.0", "layer.1"]

    owned = object()
    metadata = MagicMock()
    metadata.make_owned_copy.return_value = owned
    result = state._own_parallel_metadata({"layer.0": metadata, "layer.1": metadata})

    metadata.make_owned_copy.assert_called_once_with()
    assert result == {"layer.0": owned, "layer.1": owned}


def test_flash_attention_metadata_owned_copy_detaches_builder_tensors():
    metadata = FlashAttentionMetadata(
        num_actual_tokens=2,
        max_query_len=2,
        query_start_loc=torch.tensor([0, 2]),
        max_seq_len=4,
        seq_lens=torch.tensor([4]),
        block_table=torch.tensor([[1, 2]]),
        slot_mapping=torch.tensor([8, 9]),
        use_cascade=False,
        common_prefix_len=0,
        cu_prefix_query_lens=None,
        prefix_kv_lens=None,
        suffix_kv_lens=None,
    )

    owned = metadata.make_owned_copy()
    metadata.seq_lens.fill_(99)
    metadata.block_table.fill_(99)
    metadata.slot_mapping.fill_(99)

    assert owned.seq_lens.tolist() == [4]
    assert owned.block_table.tolist() == [[1, 2]]
    assert owned.slot_mapping.tolist() == [8, 9]


def test_bidirectional_norefresh_mask_is_full_only_within_cycle():
    q = torch.tensor([4, 4, 5, 7, 8])
    kv = torch.tensor([7, 8, 7, 4, 4])
    visible = dmtd_bidirectional_shadow_history_mask(
        torch.zeros_like(q), torch.zeros_like(q), q, kv
    )
    assert visible.tolist() == [True, False, True, True, True]


def test_bidirectional_refresh_mask_keeps_refresh_queries_causal():
    # Merged parallel-layer forward at cycle 1: refresh positions 0..3, shadow positions 4..7.
    mask = _make_dmtd_bidirectional_real_history_mask(shadow_cycle=torch.tensor(1))
    q = torch.tensor([0, 1, 4, 4, 7])
    kv = torch.tensor([3, 3, 7, 3, 4])
    visible = mask(torch.zeros_like(q), torch.zeros_like(q), q, kv)
    # refresh q0/q1 cannot see future refresh; shadow q4 sees future shadow
    # and refreshed history; later shadow remains fully connected.
    assert visible.tolist() == [False, False, True, True, True]


def test_bidirectional_norefresh_prefill_expands_touched_cycles():
    state = _state(block_attention="bidirectional")
    shadow = state._make_shadow_batch(_batch(start=0, query_len=5, is_prefilling=True))

    assert shadow.positions == list(range(8))
    assert shadow.token_abs_positions == [0, -1, -1, -1, 4, -1, -1, -1]
    assert shadow.direct_indices == [0, 1, 2, 3, 4]
    assert state._cycle_base[0] == 4
    assert state._buffer_valid[0].all()


def test_bidirectional_norefresh_rebuilds_full_cycle_after_buffer_loss():
    state = _state(block_attention="bidirectional")
    state._next_computed[0] = 2
    shadow = state._make_shadow_batch(_batch(start=2, query_len=1, is_prefilling=False))

    assert shadow.positions == [0, 1, 2, 3]
    assert shadow.token_abs_positions == [0, -1, -1, -1]
    assert shadow.direct_indices == [-1]
    assert state._buffer_valid[0].all()


def test_prefill_masks_non_heads_and_retains_partial_cycle_state():
    state = _state()
    shadow = state._make_shadow_batch(_batch(start=0, query_len=5, is_prefilling=True))

    assert shadow.positions == [0, 1, 2, 3, 4]
    assert shadow.real_input_indices == [0, -1, -1, -1, 4]
    assert shadow.direct_indices == [0, 1, 2, 3, 4]
    assert state._cycle_base[0] == 4
    assert state._buffer_valid[0].tolist() == [True, False, False, False]


def test_decode_fills_missing_shadow_suffix_once():
    state = _state()
    state._make_shadow_batch(_batch(start=0, query_len=5, is_prefilling=True))

    suffix = state._make_shadow_batch(_batch(start=5, query_len=1, is_prefilling=False))
    assert suffix.positions == [5, 6, 7]
    assert suffix.real_input_indices == [-1, -1, -1]
    assert suffix.seq_lens == [8]
    assert suffix.direct_indices == [-1]

    buffered = state._make_shadow_batch(
        _batch(start=6, query_len=1, is_prefilling=False)
    )
    assert buffered.positions == []
    assert buffered.seq_phases == [2]


def test_decode_cycle_head_builds_full_shadow_block():
    state = _state()
    state._next_computed[0] = 8
    shadow = state._make_shadow_batch(_batch(start=8, query_len=1, is_prefilling=False))

    assert shadow.positions == [8, 9, 10, 11]
    assert shadow.real_input_indices == [0, -1, -1, -1]
    assert shadow.seq_lens == [12]
    assert state._cycle_base[0] == 8
    assert state._buffer_valid[0].all()


def test_remove_request_discards_mid_cycle_buffer():
    state = _state()
    state._req_id_to_index = {"finished": 0}
    state._cycle_base[0] = 8
    state._next_computed[0] = 10
    state._buffer_valid[0].fill(True)
    state._parallel_real_len[0] = 8

    state.remove_request("finished")

    assert state._cycle_base[0] == -1
    assert state._next_computed[0] == -1
    assert not state._buffer_valid[0].any()
    assert int(state._parallel_real_len[0]) == 0


def test_recompute_position_mismatch_rebuilds_cycle_state():
    state = _state()
    state._cycle_base[0] = 8
    state._next_computed[0] = 12
    state._buffer_valid[0].fill(True)

    rebuilt = state._make_shadow_batch(
        _batch(start=8, query_len=1, is_prefilling=False)
    )

    assert rebuilt.positions == [8, 9, 10, 11]
    assert state._cycle_base[0] == 8
    assert state._next_computed[0] == 9


def test_refresh_first_cycle_shadow_only_batch_shape():
    """First Refresh cycle: parallel-layer forward is shadow4 only; parallel_real_len stays 0."""
    state = _refresh_state()
    plan = RefreshCyclePlanner(state).plan(
        _batch(start=0, query_len=1, is_prefilling=False)
    )

    assert plan.positions == [0, 1, 2, 3]
    assert plan.refresh_positions == []
    assert plan.token_abs_positions == [0, -1, -1, -1]
    assert plan.write_shadow_indices == [0, 1, 2, 3]
    assert int(state._parallel_real_len[0]) == 0
    assert state._cycle_base[0] == 0
    assert state._buffer_valid[0].all()


def test_refresh_steady_cycle_head_refresh_then_shadow_batch_shape(monkeypatch):
    """Steady Refresh cycle head: merged refresh4 + shadow [head,M,M,M]."""
    monkeypatch.setenv("DMTD_REFRESH_BACKEND", "merged")
    state = _refresh_state()
    state._next_computed[0] = 4
    state._parallel_real_len[0] = 0

    plan = RefreshCyclePlanner(state).plan(
        _batch(start=4, query_len=1, is_prefilling=False)
    )

    # Default merged backend packs refresh then shadow into one parallel-layer batch.
    assert plan.positions == [0, 1, 2, 3, 4, 5, 6, 7]
    assert plan.token_abs_positions == [0, 1, 2, 3, 4, -1, -1, -1]
    assert plan.refresh_positions == []
    assert plan.write_shadow_indices == [4, 5, 6, 7]
    assert int(state._parallel_real_len[0]) == 4
    assert state._cycle_base[0] == 4
    assert state._buffer_valid[0].all()


def test_refresh_steady_cycle_two_pass_batch_shape(monkeypatch):
    """two_pass backend builds separate refresh4 + shadow4 metadata lists."""
    monkeypatch.setenv("DMTD_REFRESH_BACKEND", "two_pass")
    state = _refresh_state()
    state._next_computed[0] = 4
    state._parallel_real_len[0] = 0

    plan = RefreshCyclePlanner(state).plan(
        _batch(start=4, query_len=1, is_prefilling=False)
    )

    assert plan.refresh_positions == [0, 1, 2, 3]
    assert plan.positions == [4, 5, 6, 7]
    assert plan.write_shadow_indices == [0, 1, 2, 3]
    assert int(state._parallel_real_len[0]) == 4


def test_refresh_mid_cycle_invalid_buffer_rebuilds_full_shadow(monkeypatch):
    """Invalid mid-cycle buffer recomputes full shadow (not Norefresh suffix)."""
    monkeypatch.setenv("DMTD_REFRESH_BACKEND", "merged")
    state = _refresh_state()
    state._next_computed[0] = 6
    state._parallel_real_len[0] = 0
    state._cycle_base[0] = -1

    plan = RefreshCyclePlanner(state).plan(
        _batch(start=6, query_len=1, is_prefilling=False)
    )

    assert plan.positions == [0, 1, 2, 3, 4, 5, 6, 7]
    assert plan.write_shadow_indices == [4, 5, 6, 7]
    assert int(state._parallel_real_len[0]) == 4
