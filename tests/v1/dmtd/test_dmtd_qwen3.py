# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from vllm.model_executor.models.config import DMTDQwen3ForCausalLMConfig
from vllm.transformers_utils.configs.dmtd_qwen3 import DMTDQwen3Config
from vllm.v1.worker.gpu.attn_utils import get_num_metadata_builders
from vllm.v1.worker.gpu.model_states.dmtd_qwen3 import (
    DMTDQwen3ModelState,
    NoRefreshCyclePlanner,
    RefreshCyclePlanner,
    VanillaCyclePlanner,
    _ParallelRowLayout,
)

TAU = 4


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


def _state(
    *,
    block_attention: str = "causal",
    history_mode: str = "shadow",
    max_num_reqs: int = 4,
    prompt_lens: list[int] | None = None,
) -> DMTDQwen3ModelState:
    state = object.__new__(DMTDQwen3ModelState)
    state.tau = TAU
    state.max_model_len = 64
    state.max_num_reqs = max_num_reqs
    state.block_attention = block_attention
    state.history_mode = history_mode
    state.mask_token_id = 127
    state._prompt_len = np.zeros(max_num_reqs, dtype=np.int64)
    if prompt_lens:
        state._prompt_len[: len(prompt_lens)] = prompt_lens
    state._cycle_base = np.full(max_num_reqs, -1, dtype=np.int64)
    state._next_computed = np.full(max_num_reqs, -1, dtype=np.int64)
    state._buffer_valid = np.zeros((max_num_reqs, TAU), dtype=np.bool_)
    state._parallel_real_len = np.zeros(max_num_reqs, dtype=np.int64)
    state.layout = _layout(max_num_reqs)
    state.DECODE_ROWS_PER_REQ = _decode_rows_per_req(block_attention, history_mode)
    state._parallel_graph_managers = {}
    if block_attention == "none":
        state._planner = VanillaCyclePlanner(state)
    elif history_mode == "real":
        state._planner = RefreshCyclePlanner(state)
    else:
        state._planner = NoRefreshCyclePlanner(state)
    return state


# Small stand-in for the real row layout: the absolute sizes do not matter, only
# that each group's base is fixed and independent of the other groups' lengths.
_MAX_PADDED_TOKENS = 16
_MAX_GROUP_ROWS = 32


def _layout(max_num_reqs: int) -> _ParallelRowLayout:
    return _ParallelRowLayout(
        prefill_base=0,
        causal_base=_MAX_PADDED_TOKENS,
        noncausal_base=_MAX_PADDED_TOKENS + _MAX_GROUP_ROWS,
        parallel_scratch_row=_MAX_PADDED_TOKENS + 2 * _MAX_GROUP_ROWS,
        cycle_scratch_row=max_num_reqs * TAU,
        tau=TAU,
        write_capacity=max_num_reqs * TAU,
    )


def _decode_rows_per_req(block_attention: str, history_mode: str) -> dict[str, int]:
    """Mirror of `DMTDQwen3ModelState.__init__`'s per-variant row counts, for the
    tests that build a state without running `__init__`."""
    if block_attention == "none":
        return {"causal": TAU if history_mode == "real" else 1}
    if history_mode == "real":
        if block_attention == "bidirectional":
            return {"causal": TAU, "noncausal": TAU}
        return {"causal": 2 * TAU}
    if block_attention == "bidirectional":
        return {"noncausal": TAU}
    return {"causal": TAU}


def _idle_writes(layout: _ParallelRowLayout, num_real: int) -> tuple[list, list]:
    """The scratch padding `resolve_gather_indices` appends after `num_real`
    real cycle-buffer writes, so tests can assert the real prefix separately."""
    src = [
        layout.parallel_scratch_row for _ in range(num_real, layout.write_capacity)
    ]
    buf = [
        layout.cycle_scratch_row + (i % TAU)
        for i in range(num_real, layout.write_capacity)
    ]
    return src, buf


def _batch(*reqs: tuple[int, int, bool]) -> SimpleNamespace:
    """Build a fake ``InputBatch`` from ``(start, query_len, is_prefilling)`` tuples,
    one per request, in ``batch_idx`` order (``idx_mapping_np`` is the identity)."""
    num_reqs = len(reqs)
    query_lens = np.array([q for _, q, _ in reqs], dtype=np.int32)
    query_start_loc_np = np.zeros(num_reqs + 1, dtype=np.int32)
    np.cumsum(query_lens, out=query_start_loc_np[1:])
    return SimpleNamespace(
        num_reqs=num_reqs,
        num_tokens=int(query_lens.sum()),
        num_tokens_after_padding=int(query_lens.sum()),
        idx_mapping_np=np.arange(num_reqs, dtype=np.int32),
        num_computed_tokens_np=np.array([s for s, _, _ in reqs], dtype=np.int32),
        num_scheduled_tokens=query_lens,
        query_start_loc_np=query_start_loc_np,
        is_prefilling_np=np.array([p for _, _, p in reqs], dtype=np.bool_),
    )


# --------------------------------------------------------------------------
# Config validation (layer split / cycle length are checkpoint-configured).
# --------------------------------------------------------------------------


def test_config_rejects_mismatched_layer_split():
    with pytest.raises(ValueError, match="num_hidden_layers"):
        DMTDQwen3Config(
            vocab_size=128,
            num_hidden_layers=5,
            num_parallel_layers=2,
            num_sequential_layers=2,
            mtp_horizon=4,
            mask_token_id=127,
        )


def test_config_rejects_non_positive_layer_counts():
    with pytest.raises(ValueError, match="must both be positive"):
        DMTDQwen3Config(
            vocab_size=128,
            num_hidden_layers=4,
            num_parallel_layers=4,
            num_sequential_layers=0,
            mtp_horizon=4,
            mask_token_id=127,
        )


def test_config_accepts_non_default_layer_split_and_cycle_length():
    """The layer split and cycle length are checkpoint-configured, not tied
    to the specific 28/8/36/tau=4 Qwen3-4B layout."""
    config = DMTDQwen3Config(
        vocab_size=128,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=12,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        num_parallel_layers=8,
        num_sequential_layers=4,
        mtp_horizon=3,
        mask_token_id=127,
    )
    assert config.num_parallel_layers == 8
    assert config.num_sequential_layers == 4
    assert config.mtp_horizon == 3

    vllm_config = _mock_vllm_config(config)
    # Must not raise: the vLLM-side hook no longer re-asserts a fixed
    # 28/8/36/tau=4 layout on top of DMTDQwen3Config's own validation.
    DMTDQwen3ForCausalLMConfig.verify_and_update_config(vllm_config)


def test_config_accepts_real_causal_and_default_shadow_causal():
    refresh = _tiny_config(dmtd_history_mode="real", dmtd_block_attention="causal")
    assert refresh.dmtd_history_mode == "real"
    assert refresh.dmtd_block_attention == "causal"

    norefresh = _tiny_config()
    assert norefresh.dmtd_history_mode == "shadow"
    assert norefresh.dmtd_block_attention == "causal"


def _mock_vllm_config(hf_config) -> MagicMock:
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
    return vllm_config


def test_model_config_enables_non_causal_only_for_bidirectional_parallel():
    """Bidirectional block attention needs `causal=False` calls (see the
    `noncausal` group), so it must flip on `use_non_causal`; it no longer
    forces any specific attention backend (FlexAttention is gone)."""
    vllm_config = _mock_vllm_config(_tiny_config(dmtd_block_attention="bidirectional"))
    DMTDQwen3ForCausalLMConfig.verify_and_update_config(vllm_config)
    assert vllm_config.attention_config.use_non_causal is True
    assert vllm_config.attention_config.backend is None


def test_model_config_leaves_non_causal_off_for_causal_parallel():
    vllm_config = _mock_vllm_config(_tiny_config(dmtd_block_attention="causal"))
    DMTDQwen3ForCausalLMConfig.verify_and_update_config(vllm_config)
    assert vllm_config.attention_config.use_non_causal is False


# --------------------------------------------------------------------------
# Parallel-layer attention metadata: one builder per group.
# --------------------------------------------------------------------------


def test_each_parallel_group_uses_its_own_metadata_builder():
    """A step can hold all three groups' metadata live at once, so each has to
    read from a builder of its own -- sharing one would have the later build
    overwrite the earlier one's persistent scratch."""
    indices = DMTDQwen3ModelState.GROUP_BUILDER_IDX
    assert set(indices) == set(DMTDQwen3ModelState.PARALLEL_GROUPS)
    # Index 0 stays with the sequential layers.
    assert 0 not in indices.values()
    assert len(set(indices.values())) == len(indices)


def test_dmtd_allocates_a_metadata_builder_for_every_group_index():
    """`attn_utils` has to allocate as many builders as the model state indexes
    into, or `get_metadata_builder` asserts at runtime."""
    needed = max(DMTDQwen3ModelState.GROUP_BUILDER_IDX.values()) + 1
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(architecture="DMTDQwen3ForCausalLM")
    )

    assert get_num_metadata_builders(vllm_config) >= needed
    # Other architectures are untouched.
    other = SimpleNamespace(model_config=SimpleNamespace(architecture="Qwen3ForCausalLM"))
    assert get_num_metadata_builders(other) == 1


# --------------------------------------------------------------------------
# Prefill: never touches the cycle machinery.
# --------------------------------------------------------------------------


def test_prefill_request_never_enters_causal_or_noncausal_group():
    state = _state()
    plan = state._plan_cycle(_batch((0, 5, True)))

    assert plan.causal.positions == []
    assert plan.noncausal.positions == []
    assert plan.prefill.positions == [0, 1, 2, 3, 4]
    assert plan.prefill_target_indices == [0, 1, 2, 3, 4]
    assert plan.prefill_orig_token_indices == [0, 1, 2, 3, 4]
    # Prefill never writes into the cycle_hidden buffer or the direct path.
    assert plan.write_req_slots == []
    assert plan.direct_group == [-1] * 5


def test_prefill_spans_multiple_requests_in_one_group():
    state = _state()
    plan = state._plan_cycle(_batch((0, 5, True), (0, 3, True)))

    assert plan.prefill.req_batch_indices == [0, 1]
    assert plan.prefill.positions == [0, 1, 2, 3, 4, 0, 1, 2]
    assert plan.prefill_target_indices == [0, 1, 2, 3, 4, 5, 6, 7]


# --------------------------------------------------------------------------
# Decode: cycle phase is anchored to this request's own prompt_len.
# --------------------------------------------------------------------------


def test_decode_cycle_head_is_always_local_position_zero():
    """The first generated token is always a fresh cycle head, regardless of
    how long the prompt was -- there is no 'prompt ends mid-cycle' case."""
    state = _state(prompt_lens=[5])
    # First decode step: absolute position 5, i.e. local_pos = 5 - 5 = 0.
    plan = state._plan_cycle(_batch((5, 1, False)))

    assert plan.causal.positions == [5, 6, 7, 8]
    assert plan.causal.token_abs_positions == [5, -1, -1, -1]
    assert state._cycle_base[0] == 0  # local cycle base, not absolute 5


def test_decode_mid_cycle_reuses_buffered_hidden():
    state = _state(prompt_lens=[5])
    state._plan_cycle(_batch((5, 1, False)))  # opens the cycle

    plan = state._plan_cycle(_batch((6, 1, False)))
    assert plan.causal.positions == []
    assert plan.noncausal.positions == []
    assert plan.direct_group == [-1]
    assert plan.seq_phases == [1]  # local_pos = 6 - 5 = 1


def test_different_prompt_lengths_give_different_absolute_cycle_heads():
    """Two requests with different prompt lengths open their cycles at
    different absolute positions, but both are local phase 0."""
    state = _state(prompt_lens=[5, 8])
    plan = state._plan_cycle(_batch((5, 1, False), (8, 1, False)))

    assert plan.causal.req_batch_indices == [0, 1]
    assert plan.causal.positions == [5, 6, 7, 8, 8, 9, 10, 11]
    assert plan.causal.seq_lens == [9, 12]


# --------------------------------------------------------------------------
# Bidirectional Norefresh: cycle head goes to the noncausal group.
# --------------------------------------------------------------------------


def test_bidirectional_norefresh_cycle_head_uses_noncausal_group():
    """A single-token cycle-open step never needs `direct_group`/`direct_local`
    for its own token: `req_positions` is exactly one whole cycle, so every
    phase (including this token's own phase 0) gets written into
    `cycle_hidden` before the model's `buffered` read -- the model always
    writes `cycle_hidden` before reading it back, so the plain buffered path
    already returns the value that was just computed this same step."""
    state = _state(block_attention="bidirectional", prompt_lens=[0])
    plan = state._plan_cycle(_batch((0, 1, False)))

    assert plan.causal.positions == []
    assert plan.noncausal.positions == [0, 1, 2, 3]
    assert plan.noncausal.token_abs_positions == [0, -1, -1, -1]
    assert plan.direct_group == [-1]
    assert plan.seq_req_slots == [0]
    assert plan.seq_phases == [0]

    direct_indices, write_shadow_indices = plan.resolve_offsets()
    assert direct_indices == [-1]
    assert write_shadow_indices == [0, 1, 2, 3]


def test_bidirectional_norefresh_prefill_like_multi_token_expands_full_cycle():
    """A decode-phase request scheduled with >1 new token this step (e.g.
    lookahead slack) still expands to the complete touched cycle block."""
    state = _state(block_attention="bidirectional", prompt_lens=[0])
    plan = state._plan_cycle(_batch((0, 5, False)))

    assert plan.noncausal.positions == list(range(8))
    assert plan.noncausal.token_abs_positions == [0, -1, -1, -1, 4, -1, -1, -1]
    assert plan.direct_group == [1, 1, 1, 1, 1]
    assert plan.direct_local == [0, 1, 2, 3, 4]


# --------------------------------------------------------------------------
# Refresh: causal block attention merges refresh+shadow into one causal row;
# bidirectional splits them into causal (refresh) + noncausal (shadow).
# --------------------------------------------------------------------------


def test_refresh_first_cycle_has_no_room_for_a_refresh_block():
    """With nothing before position 0 there is no refresh block to emit, so this
    one cycle per request is narrower than the steady state (and stays eager)."""
    state = _state(history_mode="real", prompt_lens=[0])
    plan = state._plan_cycle(_batch((0, 1, False)))

    assert plan.causal.positions == [0, 1, 2, 3]
    assert plan.causal.token_abs_positions == [0, -1, -1, -1]
    assert int(state._parallel_real_len[0]) == 0
    assert state._cycle_base[0] == 0


def test_refresh_steady_causal_merges_refresh_and_shadow_into_one_row():
    state = _state(history_mode="real", block_attention="causal", prompt_lens=[0])
    state._next_computed[0] = 4
    state._parallel_real_len[0] = 0

    plan = state._plan_cycle(_batch((4, 1, False)))

    assert plan.causal.req_batch_indices == [0]  # one merged row, not two
    assert plan.causal.positions == [0, 1, 2, 3, 4, 5, 6, 7]
    assert plan.causal.token_abs_positions == [0, 1, 2, 3, 4, -1, -1, -1]
    assert plan.noncausal.positions == []
    assert int(state._parallel_real_len[0]) == 4


def test_refresh_steady_bidirectional_splits_refresh_and_shadow_groups():
    state = _state(
        history_mode="real", block_attention="bidirectional", prompt_lens=[0]
    )
    state._next_computed[0] = 4
    state._parallel_real_len[0] = 0

    plan = state._plan_cycle(_batch((4, 1, False)))

    assert plan.causal.positions == [0, 1, 2, 3]  # refresh: plain causal
    assert plan.causal.token_abs_positions == [0, 1, 2, 3]
    assert plan.noncausal.positions == [4, 5, 6, 7]  # shadow: bidirectional
    assert plan.noncausal.token_abs_positions == [4, -1, -1, -1]
    assert int(state._parallel_real_len[0]) == 4
    # write_shadow_indices must point past the causal group's own length.
    direct_indices, write_shadow_indices = plan.resolve_offsets()
    assert write_shadow_indices == [4, 5, 6, 7]
    assert direct_indices == [4]


def test_refresh_first_cycle_after_a_prompt_still_emits_a_refresh_block():
    """Once there are `tau` positions behind the cycle head, the first cycle
    emits a refresh block too, so every cycle head is the same 2*tau shape.

    Those positions already hold real parallel-layer KV from prefill, and the
    parallel layers are a plain causal forward over real tokens, so re-running
    them writes back identical values."""
    state = _state(history_mode="real", block_attention="causal", prompt_lens=[8])
    state._plan_cycle(_batch((0, 8, True)))

    plan = state._plan_cycle(_batch((8, 1, False)))

    assert plan.causal.positions == [4, 5, 6, 7, 8, 9, 10, 11]
    # The refresh rows carry their real token ids; only the shadow tail is MASK.
    assert plan.causal.token_abs_positions == [4, 5, 6, 7, 8, -1, -1, -1]
    assert len(plan.causal.query_lens) == 1
    assert plan.causal.query_lens == [2 * TAU]


def test_refresh_mid_cycle_reuses_buffer_without_rerunning_p28():
    """Regression: contiguous decode must not look like a schedule gap.

    If ``_next_computed`` is not advanced each step, every token invalidates
    the cycle buffer and re-runs a full P28 shadow forward -- making Refresh
    as slow as the vanilla baseline even though Causal-Refresh is a single
    merged8 P28 call per cycle head.
    """
    state = _state(history_mode="real", block_attention="causal", prompt_lens=[8])
    # Prefill advances the contiguity cursor to the first decode position.
    state._plan_cycle(_batch((0, 8, True)))
    assert int(state._next_computed[0]) == 8

    head = state._plan_cycle(_batch((8, 1, False)))
    # refresh(last 4 prompt positions) + shadow(current cycle), one merged row.
    assert head.causal.positions == [4, 5, 6, 7, 8, 9, 10, 11]
    assert int(state._next_computed[0]) == 9

    for start in (9, 10, 11):
        mid = state._plan_cycle(_batch((start, 1, False)))
        assert mid.causal.positions == [], f"unexpected P28 work at pos={start}"
        assert mid.noncausal.positions == []
        assert int(state._next_computed[0]) == start + 1

    # Next cycle head: merged refresh(prev real 4) + shadow(current 4).
    steady = state._plan_cycle(_batch((12, 1, False)))
    assert steady.causal.positions == [8, 9, 10, 11, 12, 13, 14, 15]
    assert int(state._parallel_real_len[0]) == 4


# --------------------------------------------------------------------------
# Batching invariance: planning one request must be unaffected by whichever
# other requests happen to share the same step (regression test for the
# historical "batched attention corrupts non-head slots" bug -- now avoided
# entirely by using vLLM's native per-request-isolated causal/non-causal
# attention instead of a hand-rolled FlexAttention mask).
# --------------------------------------------------------------------------


@pytest.mark.parametrize("block_attention", ["causal", "bidirectional", "none"])
@pytest.mark.parametrize("history_mode", ["shadow", "real"])
def test_batching_does_not_change_a_requests_own_plan(block_attention, history_mode):
    def plan_for(batch_reqs, prompt_lens):
        state = _state(
            block_attention=block_attention,
            history_mode=history_mode,
            prompt_lens=prompt_lens,
        )
        return state._plan_cycle(_batch(*batch_reqs))

    # Request A alone: fresh cycle head right after a 3-token prompt.
    alone_a = plan_for([(3, 1, False)], [3])
    # Request B alone: fresh cycle head right after a 7-token prompt.
    alone_b = plan_for([(7, 1, False)], [7])
    # Both scheduled together in the same step.
    together = plan_for([(3, 1, False), (7, 1, False)], [3, 7])

    def positions_for(plan, batch_idx):
        group = plan.noncausal if block_attention == "bidirectional" else plan.causal
        out = []
        offset = 0
        for idx, qlen in zip(group.req_batch_indices, group.query_lens):
            if idx == batch_idx:
                out.append(group.positions[offset : offset + qlen])
            offset += qlen
        return out

    assert positions_for(together, 0) == positions_for(alone_a, 0)
    assert positions_for(together, 1) == positions_for(alone_b, 0)


# --------------------------------------------------------------------------
# CUDA graph support. The sequential-layer forward is shape-invariant and so
# always replayable; the parallel-layer groups get their own graphs, which
# requires every cycle-head step to be a uniform batch and every capture-time
# plan to be side-effect free.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("block_attention", ["causal", "bidirectional", "none"])
@pytest.mark.parametrize("history_mode", ["shadow", "real"])
def test_decode_cycle_heads_are_uniform_batches(block_attention, history_mode):
    """Every request at a cycle head must contribute exactly
    `DECODE_ROWS_PER_REQ[kind]` rows, or the group is not the uniform batch its
    captured graph was built for and falls back to eager."""
    prompt_len = 8
    state = _state(
        block_attention=block_attention,
        history_mode=history_mode,
        prompt_lens=[prompt_len, prompt_len],
        max_num_reqs=2,
    )
    rows_per_req = state.DECODE_ROWS_PER_REQ
    state._plan_cycle(_batch((0, prompt_len, True), (0, prompt_len, True)))

    heads_seen = 0
    for start in range(prompt_len, prompt_len + 3 * TAU):
        plan = state._plan_cycle(_batch((start, 1, False), (start, 1, False)))
        for kind, expected in rows_per_req.items():
            group = plan.group(kind)
            if not group.positions:
                continue
            heads_seen += 1
            assert group.query_lens == [expected, expected], (
                f"start={start} {kind}: {group.query_lens} rows per request, "
                f"expected a uniform {expected}"
            )
    assert heads_seen, "the rollout never reached a cycle head"


def test_first_cycle_refresh_rows_do_not_write_kv():
    """The first cycle's refresh rows exist only to keep the shape uniform, so
    their KV write is suppressed -- recomputing a position reproduces it only to
    within bf16 rounding, and those slots already hold real KV from prefill."""
    state = _state(history_mode="real", block_attention="causal", prompt_lens=[8])
    state._plan_cycle(_batch((0, 8, True)))

    first = state._plan_cycle(_batch((8, 1, False)))
    # refresh rows [4..7] are inert, shadow rows [8..11] write.
    assert first.causal.writes_kv == [False] * TAU + [True] * TAU

    for start in (9, 10, 11):
        state._plan_cycle(_batch((start, 1, False)))
    # A steady cycle really does need to overwrite the previous shadow KV.
    steady = state._plan_cycle(_batch((12, 1, False)))
    assert steady.causal.writes_kv == [True] * (2 * TAU)


def test_head_plan_is_uniform_and_side_effect_free():
    """Capture builds its plan from the shape alone, so it must not read or
    write live per-request cycle state -- the warmup pass before capture really
    executes, against the same request slots live requests occupy."""
    state = _state(history_mode="real", prompt_lens=[8])
    state._plan_cycle(_batch((0, 8, True)))
    state._plan_cycle(_batch((8, 1, False)))
    before = (
        state._cycle_base.copy(),
        state._next_computed.copy(),
        state._buffer_valid.copy(),
        state._parallel_real_len.copy(),
    )

    plan = state.head_plan("causal", 2)

    assert plan.causal.query_lens == [2 * TAU, 2 * TAU]
    assert len(plan.causal.positions) == 2 * 2 * TAU
    for expected, actual in zip(
        before,
        (
            state._cycle_base,
            state._next_computed,
            state._buffer_valid,
            state._parallel_real_len,
        ),
    ):
        assert np.array_equal(expected, actual)


def test_requires_eager_step_vetoes_prefill_only():
    """Cycle-head decode steps are replayable now that the parallel layers run
    outside the captured forward; prefill is not, because its hiddens bypass the
    cycle buffer entirely."""
    state = _state(prompt_lens=[8])
    req_states = SimpleNamespace(
        req_id_to_index={"a": 0},
        prefill_len=SimpleNamespace(np=np.array([8], dtype=np.int64)),
        num_computed_prefill_tokens=np.array([4], dtype=np.int64),
        num_computed_tokens_np=np.array([4], dtype=np.int64),
    )
    mid_prefill = SimpleNamespace(num_scheduled_tokens={"a": 4})
    assert state.requires_eager_step(mid_prefill, req_states)

    req_states.num_computed_prefill_tokens = np.array([8], dtype=np.int64)
    req_states.num_computed_tokens_np = np.array([8], dtype=np.int64)
    decode = SimpleNamespace(num_scheduled_tokens={"a": 1})
    assert not state.requires_eager_step(decode, req_states)

    # An unknown request is never guessed at.
    assert state.requires_eager_step(
        SimpleNamespace(num_scheduled_tokens={"ghost": 1}), req_states
    )


@pytest.mark.parametrize("block_attention", ["causal", "bidirectional", "none"])
@pytest.mark.parametrize("history_mode", ["shadow", "real"])
def test_needs_parallel_work_agrees_with_planner(block_attention, history_mode):
    """The eager-veto predicate is a hand-mirrored copy of each planner's
    early-return condition. If they ever disagree, a step needing parallel-layer
    work would replay a graph that does not contain it, so pin them together
    over a whole multi-cycle rollout."""
    prompt_len = 6
    state = _state(
        block_attention=block_attention,
        history_mode=history_mode,
        prompt_lens=[prompt_len],
    )
    # Prefill first, so the contiguity cursor starts where decode does.
    state._plan_cycle(_batch((0, prompt_len, True)))

    for start in range(prompt_len, prompt_len + 3 * TAU):
        predicted = state._planner.needs_parallel_work(
            req_slot=0, start=start, query_len=1
        )
        plan = state._plan_cycle(_batch((start, 1, False)))
        actual_rows = len(plan.causal.positions) + len(plan.noncausal.positions)
        assert predicted == (actual_rows > 0), (
            f"start={start}: predicate said {predicted} but the planner "
            f"produced {actual_rows} parallel-layer rows"
        )


@pytest.mark.parametrize("block_attention", ["causal", "bidirectional"])
@pytest.mark.parametrize("history_mode", ["shadow", "real"])
def test_needs_parallel_work_is_true_for_a_schedule_gap(block_attention, history_mode):
    """A rewound/skipped position invalidates the cycle buffer, so the step
    that resumes must be forced eager."""
    state = _state(
        block_attention=block_attention, history_mode=history_mode, prompt_lens=[4]
    )
    state._plan_cycle(_batch((0, 4, True)))
    state._plan_cycle(_batch((4, 1, False)))  # cycle head fills the buffer
    assert not state._planner.needs_parallel_work(req_slot=0, start=5, query_len=1)
    # Jumping past position 5 is a gap: nothing buffered that phase's hidden.
    assert state._planner.needs_parallel_work(req_slot=0, start=9, query_len=1)


def test_blank_plan_has_no_parallel_work_and_no_side_effects():
    state = _state(history_mode="real", prompt_lens=[8])
    state._plan_cycle(_batch((0, 8, True)))
    state._plan_cycle(_batch((8, 1, False)))
    before = (
        state._cycle_base.copy(),
        state._next_computed.copy(),
        state._buffer_valid.copy(),
        state._parallel_real_len.copy(),
    )

    plan = state._blank_plan(4)

    assert plan.prefill.positions == []
    assert plan.causal.positions == []
    assert plan.noncausal.positions == []
    assert plan.prefill_target_indices == [-1] * 4
    assert plan.direct_group == [-1] * 4
    after = (
        state._cycle_base,
        state._next_computed,
        state._buffer_valid,
        state._parallel_real_len,
    )
    for expected, actual in zip(before, after):
        assert np.array_equal(expected, actual)


def test_blank_plan_resolves_every_token_to_the_cycle_buffer():
    state = _state(prompt_lens=[4])
    plan = state._blank_plan(3)
    plan.seq_req_slots = [0, 1, 2]
    plan.seq_phases = [1, 2, 3]

    src, buf, add_embed, write_src, write_buf = plan.resolve_gather_indices(
        state.layout, 3
    )

    assert src == [-1, -1, -1]  # nothing comes from the parallel table
    assert buf == [0 * TAU + 1, 1 * TAU + 2, 2 * TAU + 3]
    assert add_embed == [True, True, True]
    # No real writes, so the whole (fixed-length) write vector is scratch.
    idle_src, idle_buf = _idle_writes(state.layout, 0)
    assert write_src == idle_src
    assert write_buf == idle_buf


def test_gather_indices_use_fixed_per_group_row_bases():
    """`src_index` addresses the persistent parallel table, whose per-group row
    bases are fixed so that a captured graph's output address never depends on
    what the other groups happen to hold this step."""
    state = _state(
        history_mode="real", block_attention="bidirectional", prompt_lens=[0]
    )
    state._next_computed[0] = 4
    state._parallel_real_len[0] = 0
    plan = state._plan_cycle(_batch((4, 1, False)))

    # Refresh block went to `causal` (4 rows), shadow block to `noncausal`.
    assert plan.causal.positions == [0, 1, 2, 3]
    assert plan.noncausal.positions == [4, 5, 6, 7]

    layout = state.layout
    src, _, add_embed, write_src, write_buf = plan.resolve_gather_indices(layout, 1)

    # The token reads noncausal row 0, at that group's own fixed base -- not at
    # an offset that depends on the 4 causal rows.
    assert src == [layout.noncausal_base]
    assert add_embed == [True]
    idle_src, idle_buf = _idle_writes(layout, 4)
    assert write_src == [layout.noncausal_base + i for i in range(4)] + idle_src
    assert write_buf == [0, 1, 2, 3] + idle_buf


def test_write_vectors_have_a_step_independent_length():
    """A captured graph copies a fixed number of cycle-buffer rows, so the write
    vectors are always `write_capacity` long whatever the step holds."""
    state = _state(prompt_lens=[4])
    layout = state.layout

    mid_cycle = state._blank_plan(1)
    head = state._plan_cycle(_batch((4, 1, False)))

    for plan in (mid_cycle, head):
        _, _, _, write_src, write_buf = plan.resolve_gather_indices(layout, 1)
        assert len(write_src) == len(write_buf) == layout.write_capacity
        # Every entry addresses a real row of both tables.
        assert all(0 <= row <= layout.parallel_scratch_row for row in write_src)
        assert all(0 <= row < (state.max_num_reqs + 1) * TAU for row in write_buf)


def test_plan_covers_padded_tokens():
    """CUDA graph padding makes the batch longer than the scheduled token
    count; every per-token vector still has to cover the padding rows."""
    state = _state(prompt_lens=[4])
    batch = _batch((4, 1, False))
    batch.num_tokens_after_padding = 4

    plan = state._plan_cycle(batch)

    assert len(plan.prefill_target_indices) == 4
    assert len(plan.direct_group) == 4
    src, buf, add_embed, _, _ = plan.resolve_gather_indices(state.layout, 4)
    assert len(src) == len(buf) == len(add_embed) == 4
    # The scheduled token is a cycle head; the three padding rows fall back to
    # a cycle-buffer read aimed at the scratch slot, so they cannot disturb a
    # live request's buffered hidden.
    assert src[1:] == [-1, -1, -1]
    assert buf[1:] == [state.layout.cycle_scratch_row] * 3


# --------------------------------------------------------------------------
# Original DMTD (`dmtd_block_attention="none"`): no MASK lookahead block, so
# only the cycle head gets a parallel-layer hidden and the rest of the cycle
# enters S8 on its token embedding alone.
# --------------------------------------------------------------------------


def _rollout(state, prompt_len, num_tokens):
    """Prefill, then one single-token decode step per generated token."""
    state._plan_cycle(_batch((0, prompt_len, True)))
    return [
        state._plan_cycle(_batch((prompt_len + i, 1, False)))
        for i in range(num_tokens)
    ]


def test_vanilla_config_naming_of_the_released_checkpoint():
    """The original release describes its split with the paper's layer-group
    names and carries neither knob, which is exactly the `none` + `real`
    variant -- so its own config.json has to load unmodified."""
    config = _tiny_config(
        num_encoding_layers=0,
        num_thinking_layers=28,
        num_decoding_layers=8,
    )

    assert config.num_parallel_layers == 28
    assert config.num_sequential_layers == 8
    assert config.dmtd_block_attention == "none"
    assert config.dmtd_history_mode == "real"
    # An explicit knob still wins over the inferred default.
    assert (
        _tiny_config(num_thinking_layers=28, dmtd_history_mode="shadow")
        .dmtd_history_mode
        == "shadow"
    )


def test_config_rejects_encoding_layers():
    """The paper's third layer group has no counterpart here: this
    implementation is only P28 + S8."""
    with pytest.raises(ValueError, match="num_encoding_layers"):
        _tiny_config(num_encoding_layers=2, num_thinking_layers=26)


def test_vanilla_leaves_non_causal_off():
    vllm_config = _mock_vllm_config(_tiny_config(dmtd_block_attention="none"))
    DMTDQwen3ForCausalLMConfig.verify_and_update_config(vllm_config)
    assert vllm_config.attention_config.use_non_causal is False


@pytest.mark.parametrize("history_mode", ["shadow", "real"])
def test_vanilla_runs_the_parallel_layers_only_at_cycle_heads(history_mode):
    """`m_i = 1 if local_pos % tau == 0 else 0`: the cadence the reference
    implementation writes as `1000 1000 ...`."""
    prompt_len = 6
    state = _state(
        block_attention="none", history_mode=history_mode, prompt_lens=[prompt_len]
    )

    plans = _rollout(state, prompt_len, 3 * TAU)

    ran_parallel = [bool(plan.causal.positions) for plan in plans]
    assert ran_parallel == [i % TAU == 0 for i in range(3 * TAU)]
    # Nothing ever reaches the noncausal group: there is no shadow block.
    assert all(not plan.noncausal.positions for plan in plans)


@pytest.mark.parametrize("history_mode", ["shadow", "real"])
def test_vanilla_mid_cycle_tokens_read_a_zero_row_and_add_their_embedding(
    history_mode,
):
    """A non-head position's S8 input is `embed(x_i) + 0`. The zero comes from
    the cycle buffer, which this variant never writes, so the model's
    unconditional gather stays shape-invariant."""
    prompt_len = 6
    state = _state(
        block_attention="none", history_mode=history_mode, prompt_lens=[prompt_len]
    )

    plans = _rollout(state, prompt_len, TAU)

    for phase, plan in enumerate(plans):
        src, buf, add_embed, write_src, write_buf = plan.resolve_gather_indices(
            state.layout, 1
        )
        assert add_embed == [True]
        if phase == 0:
            assert src[0] >= 0  # the head reads its own parallel-layer row
        else:
            assert src == [-1]  # ... everything else reads the buffer
            assert buf == [0 * TAU + phase]
        # No cycle-buffer write on any step, in either mode: the buffer stays
        # at its initial zeros, which is what makes the read above a `+ 0`.
        idle_src, idle_buf = _idle_writes(state.layout, 0)
        assert (write_src, write_buf) == (idle_src, idle_buf)
        assert plan.write_req_slots == []


def test_vanilla_refresh_refills_the_skipped_positions_at_the_next_head():
    """Cyclical refilling: the `tau - 1` positions the cycle skipped, plus the
    head itself, recomputed as one causal block of real tokens."""
    prompt_len = 6
    state = _state(
        block_attention="none", history_mode="real", prompt_lens=[prompt_len]
    )

    plans = _rollout(state, prompt_len, 2 * TAU + 1)

    first, steady = plans[0], plans[TAU]
    # First head has nothing to refill, so its extra rows are the last prompt
    # positions, kept only to keep the block a uniform `tau` and inert.
    assert first.causal.positions == [3, 4, 5, 6]
    assert first.causal.writes_kv == [False, False, False, True]
    # Steady head: the previous cycle's skipped positions [7, 8, 9] are holes in
    # the parallel-layer cache, so they are filled here along with head 10.
    assert steady.causal.positions == [7, 8, 9, 10]
    assert steady.causal.writes_kv == [True] * TAU
    # Real token ids, never MASK, and the window is a gap-free prefix.
    assert steady.causal.token_abs_positions == [7, 8, 9, 10]
    assert steady.causal.cache_positions == [7, 8, 9, 10]
    assert steady.causal.seq_lens == [11]
    # The head token reads the block's last row, not one of the refill rows.
    src, _, _, _, _ = steady.resolve_gather_indices(state.layout, 1)
    assert src == [state.layout.causal_base + TAU - 1]


def test_vanilla_norefresh_packs_heads_onto_a_lagging_parallel_cache():
    """Without refilling, the skipped positions are never written -- so they
    must not reserve a slot either, or the head would attend to whatever
    occupies them. Heads pack onto the end of the parallel-layer cache instead,
    which is the paged equivalent of the reference implementation's lagging
    per-layer-group cache."""
    prompt_len = 6
    state = _state(
        block_attention="none", history_mode="shadow", prompt_lens=[prompt_len]
    )

    plans = _rollout(state, prompt_len, 3 * TAU)

    heads = [plans[i * TAU] for i in range(3)]
    # One row per head, at its true position for RoPE ...
    assert [head.causal.positions for head in heads] == [[6], [10], [14]]
    # ... but packed contiguously in the cache, so each head's window is a
    # gap-free prefix of prompt + every earlier head.
    assert [head.causal.cache_positions for head in heads] == [[6], [7], [8]]
    assert [head.causal.seq_lens for head in heads] == [[7], [8], [9]]
    assert [head.causal.writes_kv for head in heads] == [[True]] * 3


def test_vanilla_refresh_leaves_no_hole_in_the_parallel_cache():
    """The invariant cyclical refilling exists for: by the time a head runs, the
    parallel-layer cache is a gap-free prefix, so every generated position is
    written exactly once and the head's own window is complete."""
    prompt_len = 6
    state = _state(
        block_attention="none", history_mode="real", prompt_lens=[prompt_len]
    )

    written: list[int] = []
    for plan in _rollout(state, prompt_len, 4 * TAU):
        group = plan.causal
        if not group.positions:
            continue
        written.extend(
            pos for pos, writes in zip(group.positions, group.writes_kv) if writes
        )
        # The block always ends at the head and reaches back to the first hole.
        assert group.positions[-1] + 1 == group.seq_lens[0]
        assert group.positions[-1] - group.positions[0] + 1 == len(group.positions)

    heads = list(range(prompt_len, prompt_len + 4 * TAU, TAU))
    # Every generated position up to the last head, once each: no hole, no
    # position written twice (a rewrite would perturb real KV at the ULP level).
    assert written == list(range(prompt_len, heads[-1] + 1))
