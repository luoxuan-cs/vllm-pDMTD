# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""V2 ModelState and cycle planners for DMTD Qwen3 variants."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from vllm.v1.worker.gpu.attn_utils import (
    build_attn_metadata,
    build_slot_mappings_by_layer,
)
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.model_states.default import DefaultModelState
from vllm.v1.worker.gpu.states import RequestState


@dataclass
class _ParallelBatch:
    req_batch_indices: list[int] = field(default_factory=list)
    query_lens: list[int] = field(default_factory=list)
    seq_lens: list[int] = field(default_factory=list)
    positions: list[int] = field(default_factory=list)
    # Absolute sequence positions for token IDs (-1 => MASK).
    token_abs_positions: list[int] = field(default_factory=list)
    # Back-compat alias for Norefresh unit tests (input-batch indices).
    real_input_indices: list[int] = field(default_factory=list)
    write_req_slots: list[int] = field(default_factory=list)
    write_phases: list[int] = field(default_factory=list)
    write_shadow_indices: list[int] = field(default_factory=list)
    # Sequential-layer wiring relative to the scheduled real tokens.
    direct_indices: list[int] = field(default_factory=list)
    seq_req_slots: list[int] = field(default_factory=list)
    seq_phases: list[int] = field(default_factory=list)
    # Optional second pass (refresh) for two_pass backend.
    refresh_req_batch_indices: list[int] = field(default_factory=list)
    refresh_query_lens: list[int] = field(default_factory=list)
    refresh_seq_lens: list[int] = field(default_factory=list)
    refresh_positions: list[int] = field(default_factory=list)
    refresh_token_abs_positions: list[int] = field(default_factory=list)


# Back-compat alias used by existing Norefresh unit tests.
_ShadowBatch = _ParallelBatch


def dmtd_bidirectional_shadow_history_mask(
    b: torch.Tensor,
    h: torch.Tensor,
    q_idx: torch.Tensor,
    kv_idx: torch.Tensor,
) -> torch.Tensor:
    """Causal history plus fully visible shadow slots in every cycle."""
    del b, h
    same_cycle = q_idx // 4 == kv_idx // 4
    return same_cycle | (kv_idx <= q_idx)


def _make_dmtd_bidirectional_real_history_mask(
    shadow_cycle: int | torch.Tensor,
):
    """Causal real history plus a bidirectional current shadow cycle."""

    def mask(
        b: torch.Tensor,
        h: torch.Tensor,
        q_idx: torch.Tensor,
        kv_idx: torch.Tensor,
    ) -> torch.Tensor:
        del b, h
        q_cycle = q_idx // 4
        kv_cycle = kv_idx // 4
        current_shadow = (q_cycle == shadow_cycle) & (kv_cycle == shadow_cycle)
        return current_shadow | (kv_idx <= q_idx)

    return mask


def _refresh_backend() -> str:
    value = os.environ.get("DMTD_REFRESH_BACKEND", "merged").strip().lower()
    if value not in {"merged", "two_pass"}:
        raise ValueError(
            f"DMTD_REFRESH_BACKEND must be 'merged' or 'two_pass', got {value!r}."
        )
    return value


class NoRefreshCyclePlanner:
    """Persistent shadow-KV planner (Causal-Parallel-Norefresh)."""

    def __init__(self, state: DMTDQwen3ModelState) -> None:
        self.state = state

    def plan(self, input_batch: InputBatch) -> _ParallelBatch:
        state = self.state
        req_batch_indices: list[int] = []
        query_lens: list[int] = []
        seq_lens: list[int] = []
        shadow_positions: list[int] = []
        token_abs_positions: list[int] = []
        real_input_indices: list[int] = []
        direct_indices = [-1] * input_batch.num_tokens
        seq_req_slots = [0] * input_batch.num_tokens
        seq_phases = [0] * input_batch.num_tokens
        write_req_slots: list[int] = []
        write_phases: list[int] = []
        write_shadow_indices: list[int] = []

        for batch_idx in range(input_batch.num_reqs):
            req_slot = int(input_batch.idx_mapping_np[batch_idx])
            start = int(input_batch.num_computed_tokens_np[batch_idx])
            query_len = int(input_batch.num_scheduled_tokens[batch_idx])
            query_start = int(input_batch.query_start_loc_np[batch_idx])
            query_end = query_start + query_len

            if state._next_computed[req_slot] not in (-1, start):
                state._invalidate(req_slot)
            state._next_computed[req_slot] = start + query_len

            for local_idx, pos in enumerate(range(start, start + query_len)):
                token_idx = query_start + local_idx
                seq_req_slots[token_idx] = req_slot
                seq_phases[token_idx] = pos % state.tau

            is_prefill = bool(input_batch.is_prefilling_np[batch_idx])
            is_multi_token = query_len > 1
            req_positions: list[int] = []
            req_token_abs: list[int] = []
            req_real_indices: list[int] = []
            if is_prefill or is_multi_token:
                if state.block_attention == "bidirectional":
                    # Every hidden in a bidirectional shadow cycle depends on
                    # all tau slots. Expand every touched prefill cycle to its
                    # complete [head, M, ..., M] block.
                    parallel_start = start - start % state.tau
                    scheduled_end = start + query_len
                    parallel_end = min(
                        ((scheduled_end + state.tau - 1) // state.tau) * state.tau,
                        state.max_model_len,
                    )
                    req_positions = list(range(parallel_start, parallel_end))
                else:
                    parallel_start = start
                    req_positions = list(range(start, start + query_len))
                req_token_abs = [
                    pos if pos % state.tau == 0 else -1 for pos in req_positions
                ]
                req_real_indices = [
                    (
                        query_start + pos - start
                        if pos % state.tau == 0 and start <= pos < start + query_len
                        else -1
                    )
                    for pos in req_positions
                ]
                shadow_start = len(shadow_positions)
                direct_indices[query_start:query_end] = [
                    shadow_start + pos - parallel_start
                    for pos in range(start, start + query_len)
                ]
            else:
                assert query_len == 1
                pos = start
                cycle_base = pos - pos % state.tau
                phase = pos % state.tau
                has_hidden = (
                    state._cycle_base[req_slot] == cycle_base
                    and state._buffer_valid[req_slot, phase]
                )
                if has_hidden:
                    continue
                cycle_end = min(cycle_base + state.tau, state.max_model_len)
                req_positions = list(
                    range(
                        cycle_base if state.block_attention == "bidirectional" else pos,
                        cycle_end,
                    )
                )
                req_token_abs = [-1] * len(req_positions)
                req_real_indices = [-1] * len(req_positions)
                if state.block_attention == "bidirectional":
                    req_token_abs[0] = cycle_base
                    if phase == 0:
                        req_real_indices[0] = query_start
                elif phase == 0:
                    req_token_abs[0] = pos
                    req_real_indices[0] = query_start

            if not req_positions:
                continue
            req_batch_indices.append(batch_idx)
            query_lens.append(len(req_positions))
            seq_lens.append(req_positions[-1] + 1)
            req_shadow_start = len(shadow_positions)
            shadow_positions.extend(req_positions)
            token_abs_positions.extend(req_token_abs)
            real_input_indices.extend(req_real_indices)

            final_cycle_base = req_positions[-1] - req_positions[-1] % state.tau
            if state._cycle_base[req_slot] != final_cycle_base:
                state._invalidate_buffer(req_slot)
                state._cycle_base[req_slot] = final_cycle_base
            for offset, pos in enumerate(req_positions):
                if pos - pos % state.tau != final_cycle_base:
                    continue
                phase = pos % state.tau
                state._buffer_valid[req_slot, phase] = True
                write_req_slots.append(req_slot)
                write_phases.append(phase)
                write_shadow_indices.append(req_shadow_start + offset)

        return _ParallelBatch(
            req_batch_indices=req_batch_indices,
            query_lens=query_lens,
            seq_lens=seq_lens,
            positions=shadow_positions,
            token_abs_positions=token_abs_positions,
            real_input_indices=real_input_indices,
            direct_indices=direct_indices,
            seq_req_slots=seq_req_slots,
            seq_phases=seq_phases,
            write_req_slots=write_req_slots,
            write_phases=write_phases,
            write_shadow_indices=write_shadow_indices,
        )


class RefreshCyclePlanner:
    """Real-KV refresh planner (Causal-Parallel-Refresh)."""

    def __init__(self, state: DMTDQwen3ModelState) -> None:
        self.state = state
        self.backend = _refresh_backend()

    def plan(self, input_batch: InputBatch) -> _ParallelBatch:
        state = self.state
        batch = _ParallelBatch(
            direct_indices=[-1] * input_batch.num_tokens,
            seq_req_slots=[0] * input_batch.num_tokens,
            seq_phases=[0] * input_batch.num_tokens,
        )

        for batch_idx in range(input_batch.num_reqs):
            req_slot = int(input_batch.idx_mapping_np[batch_idx])
            start = int(input_batch.num_computed_tokens_np[batch_idx])
            query_len = int(input_batch.num_scheduled_tokens[batch_idx])
            query_start = int(input_batch.query_start_loc_np[batch_idx])

            if state._next_computed[req_slot] not in (-1, start):
                state._invalidate_buffer(req_slot)
                # Resume with Refresh lag: completed prior cycles only. At a
                # cycle head the previous cycle is not yet refreshed.
                cycle_base_resume = start - start % state.tau
                if start > 0 and start % state.tau == 0:
                    state._parallel_real_len[req_slot] = cycle_base_resume - state.tau
                else:
                    state._parallel_real_len[req_slot] = cycle_base_resume
            state._next_computed[req_slot] = start + query_len

            for local_idx, pos in enumerate(range(start, start + query_len)):
                token_idx = query_start + local_idx
                batch.seq_req_slots[token_idx] = req_slot
                batch.seq_phases[token_idx] = pos % state.tau

            cycle_base = start - start % state.tau
            needs_parallel = True
            if state._cycle_base[req_slot] == cycle_base:
                phases_needed = [(start + i) % state.tau for i in range(query_len)]
                if all(state._buffer_valid[req_slot, ph] for ph in phases_needed):
                    needs_parallel = False

            if not needs_parallel:
                continue

            self._plan_cycle_parallel(
                batch,
                batch_idx=batch_idx,
                req_slot=req_slot,
                cycle_base=cycle_base,
            )

            phase_to_shadow = {
                write_phase: shadow_idx
                for slot, write_phase, shadow_idx in zip(
                    batch.write_req_slots,
                    batch.write_phases,
                    batch.write_shadow_indices,
                )
                if slot == req_slot
            }
            for local_idx in range(query_len):
                phase = (start + local_idx) % state.tau
                shadow_idx = phase_to_shadow.get(phase)
                if shadow_idx is not None:
                    batch.direct_indices[query_start + local_idx] = shadow_idx

        return batch

    def _plan_cycle_parallel(
        self,
        batch: _ParallelBatch,
        *,
        batch_idx: int,
        req_slot: int,
        cycle_base: int,
    ) -> None:
        state = self.state
        tau = state.tau
        cycle_end = min(cycle_base + tau, state.max_model_len)
        shadow_positions = list(range(cycle_base, cycle_end))
        shadow_token_abs = [
            pos if pos == cycle_base else -1 for pos in shadow_positions
        ]

        # First generation cycle (c==0): shadow only; parallel_real_len stays 0.
        # Steady cycle (c>=tau): refresh previous real block then shadow.
        need_refresh = (
            cycle_base > 0 and int(state._parallel_real_len[req_slot]) < cycle_base
        )
        refresh_positions = (
            list(range(cycle_base - tau, cycle_base)) if need_refresh else []
        )
        refresh_token_abs = list(refresh_positions)

        if state._cycle_base[req_slot] != cycle_base:
            state._invalidate_buffer(req_slot)
            state._cycle_base[req_slot] = cycle_base
        state._buffer_valid[req_slot].fill(False)

        base = len(batch.positions)
        if self.backend == "two_pass" and refresh_positions:
            batch.refresh_req_batch_indices.append(batch_idx)
            batch.refresh_query_lens.append(len(refresh_positions))
            batch.refresh_seq_lens.append(refresh_positions[-1] + 1)
            batch.refresh_positions.extend(refresh_positions)
            batch.refresh_token_abs_positions.extend(refresh_token_abs)
            parallel_positions = shadow_positions
            parallel_token_abs = shadow_token_abs
            shadow_offset = base
        else:
            # merged (default) or first-cycle shadow-only: one parallel-layer batch.
            parallel_positions = refresh_positions + shadow_positions
            parallel_token_abs = refresh_token_abs + shadow_token_abs
            shadow_offset = base + len(refresh_positions)

        batch.req_batch_indices.append(batch_idx)
        batch.query_lens.append(len(parallel_positions))
        batch.seq_lens.append(shadow_positions[-1] + 1)
        batch.positions.extend(parallel_positions)
        batch.token_abs_positions.extend(parallel_token_abs)

        for phase in range(len(shadow_positions)):
            state._buffer_valid[req_slot, phase] = True
            batch.write_req_slots.append(req_slot)
            batch.write_phases.append(phase)
            batch.write_shadow_indices.append(shadow_offset + phase)

        if need_refresh:
            # After this parallel-layer step, logical persistent real cache reaches cycle_base.
            state._parallel_real_len[req_slot] = cycle_base


class DMTDQwen3ModelState(DefaultModelState):
    """Variant-aware DMTD state: shadow history or real refresh."""

    def __init__(self, vllm_config, model, encoder_cache, device) -> None:
        super().__init__(vllm_config, model, encoder_cache, device)
        config = model.config
        self.tau = int(config.mtp_horizon)
        self.mask_token_id = int(config.mask_token_id)
        self.num_parallel_layers = int(config.num_parallel_layers)
        self.history_mode = getattr(config, "dmtd_history_mode", "shadow")
        self.block_attention = getattr(config, "dmtd_block_attention", "causal")

        self._req_id_to_index: dict[str, int] = {}
        self._cycle_base = np.full(self.max_num_reqs, -1, dtype=np.int64)
        self._next_computed = np.full(self.max_num_reqs, -1, dtype=np.int64)
        self._buffer_valid = np.zeros((self.max_num_reqs, self.tau), dtype=np.bool_)
        self._parallel_real_len = np.zeros(self.max_num_reqs, dtype=np.int64)
        self.cycle_hidden = torch.zeros(
            self.max_num_reqs,
            self.tau,
            config.hidden_size,
            dtype=self.dtype,
            device=device,
        )
        self._parallel_batch: _ParallelBatch | None = None
        self._parallel_slot_mapping: torch.Tensor | None = None
        self._refresh_slot_mapping: torch.Tensor | None = None
        self._refresh_attn_metadata: dict[str, Any] | None = None
        self._parallel_segment_meta: list[dict[str, Any]] = []
        self._refresh_segment_meta: list[dict[str, Any]] = []

        layers = model.model.layers
        self._parallel_layer_names = [
            layer.self_attn.attn.layer_name
            for layer in layers[: self.num_parallel_layers]
        ]
        if self.history_mode == "real":
            self._planner: NoRefreshCyclePlanner | RefreshCyclePlanner = (
                RefreshCyclePlanner(self)
            )
        else:
            self._planner = NoRefreshCyclePlanner(self)

    def _invalidate_buffer(self, req_index: int) -> None:
        self._cycle_base[req_index] = -1
        self._buffer_valid[req_index].fill(False)

    def _invalidate(self, req_index: int) -> None:
        self._invalidate_buffer(req_index)
        self._parallel_real_len[req_index] = 0

    def add_request(self, req_index, new_req_data) -> None:
        super().add_request(req_index, new_req_data)
        self._req_id_to_index[new_req_data.req_id] = req_index
        self._invalidate(req_index)
        self._next_computed[req_index] = new_req_data.num_computed_tokens

    def remove_request(self, req_id: str) -> None:
        req_index = self._req_id_to_index.pop(req_id, None)
        if req_index is not None:
            self._invalidate(req_index)
            self._next_computed[req_index] = -1

    def get_mm_embeddings(
        self,
        scheduled_encoder_inputs: dict[str, list[int]],
        input_batch: InputBatch,
        req_states: RequestState,
    ) -> None:
        return None

    def _make_shadow_batch(self, input_batch: InputBatch) -> _ParallelBatch:
        """Back-compat entry used by Norefresh unit tests."""
        return NoRefreshCyclePlanner(self).plan(input_batch)

    def _make_refresh_batch(self, input_batch: InputBatch) -> _ParallelBatch:
        """Plan a Refresh parallel-layer batch (unit tests + explicit refresh planning)."""
        return RefreshCyclePlanner(self).plan(input_batch)

    def _make_parallel_batch(self, input_batch: InputBatch) -> _ParallelBatch:
        return self._planner.plan(input_batch)

    def _build_slot_mapping(
        self,
        req_batch_indices: list[int],
        query_lens: list[int],
        positions: list[int],
        block_tables: tuple[torch.Tensor, ...],
        kv_cache_config,
        attn_groups,
    ) -> tuple[tuple[torch.Tensor, ...], torch.Tensor]:
        if len(kv_cache_config.kv_cache_groups) != 1:
            raise ValueError("DMTDQwen3 requires one shared KV cache group.")
        device = self.device
        req_indices = torch.tensor(req_batch_indices, dtype=torch.long, device=device)
        query_lens_t = torch.tensor(query_lens, dtype=torch.long, device=device)
        token_req_indices = torch.repeat_interleave(req_indices, query_lens_t)
        positions_t = torch.tensor(positions, dtype=torch.long, device=device)
        block_size = attn_groups[0][0].get_metadata_builder(1).kv_cache_spec.block_size
        block_numbers = block_tables[0][
            token_req_indices, positions_t // block_size
        ].long()
        slots = block_numbers * block_size + positions_t % block_size
        selected_block_tables = tuple(
            table.index_select(0, req_indices) for table in block_tables
        )
        return selected_block_tables, slots.unsqueeze(0)

    def _build_parallel_metadata(
        self,
        req_batch_indices: list[int],
        query_lens: list[int],
        seq_lens: list[int],
        positions: list[int],
        block_tables,
        slot_mappings,
        attn_groups,
        kv_cache_config,
        for_capture: bool,
        *,
        enable_bidirectional_shadow: bool = True,
    ) -> dict[str, Any]:
        device = self.device
        query_start_cpu = torch.zeros(len(query_lens) + 1, dtype=torch.int32)
        query_start_cpu[1:] = torch.tensor(query_lens).cumsum(0)
        query_start_gpu = query_start_cpu.to(device)
        seq_lens_t = torch.tensor(seq_lens, dtype=torch.int32, device=device)
        positions_t = torch.tensor(positions, dtype=torch.long, device=device)
        metadata_by_layer = build_attn_metadata(
            attn_groups=attn_groups,
            num_reqs=len(query_lens),
            num_tokens=len(positions),
            query_start_loc_gpu=query_start_gpu,
            query_start_loc_cpu=query_start_cpu,
            max_query_len=max(query_lens),
            seq_lens=seq_lens_t,
            max_seq_len=max(seq_lens),
            block_tables=block_tables,
            slot_mappings=slot_mappings,
            kv_cache_config=kv_cache_config,
            seq_lens_cpu_upper_bound=torch.tensor(seq_lens, dtype=torch.int32),
            positions=positions_t,
            is_prefilling=torch.tensor(
                [True] * len(req_batch_indices),
                dtype=torch.bool,
            ),
            for_cudagraph_capture=for_capture,
            metadata_builder_idx=1,
        )
        if self.block_attention != "bidirectional" or not enable_bidirectional_shadow:
            return metadata_by_layer

        if self.history_mode == "shadow":
            logical_mask_mod = dmtd_bidirectional_shadow_history_mask
        else:
            # Keep the cycle boundary tensor-valued so FlexAttention can treat
            # it as runtime data instead of specializing a compiled mask on a
            # new Python integer every cycle.
            shadow_cycle = torch.tensor(
                positions[-1] // self.tau,
                dtype=torch.long,
                device=device,
            )
            logical_mask_mod = _make_dmtd_bidirectional_real_history_mask(shadow_cycle)

        seen: set[int] = set()
        for metadata in metadata_by_layer.values():
            if id(metadata) in seen:
                continue
            seen.add(id(metadata))
            if not hasattr(metadata, "logical_mask_mod"):
                raise ValueError(
                    "DMTDQwen3 bidirectional block attention requires "
                    "FLEX_ATTENTION metadata."
                )
            metadata.logical_mask_mod = logical_mask_mod
            metadata.mask_mod = metadata.get_mask_mod()
            metadata.block_mask = (
                metadata._build_block_mask_direct()
                if metadata.direct_build
                else metadata.build_block_mask()
            )
        return metadata_by_layer

    def _own_parallel_metadata(self, metadata_by_layer: dict[str, Any]) -> dict[str, Any]:
        """Detach retained parallel-layer metadata from reusable builder buffers."""
        owned_by_id: dict[int, Any] = {}
        result: dict[str, Any] = {}
        for layer_name in self._parallel_layer_names:
            metadata = metadata_by_layer[layer_name]
            metadata_id = id(metadata)
            if metadata_id not in owned_by_id:
                make_owned_copy = getattr(metadata, "make_owned_copy", None)
                if make_owned_copy is None:
                    if self.block_attention == "bidirectional":
                        raise ValueError(
                            "DMTDQwen3 bidirectional block attention requires "
                            "owned FLEX_ATTENTION metadata."
                        )
                    owned_by_id[metadata_id] = metadata
                else:
                    owned_by_id[metadata_id] = make_owned_copy()
            result[layer_name] = owned_by_id[metadata_id]
        return result

    def prepare_attn(
        self,
        input_batch,
        cudagraph_mode,
        block_tables,
        slot_mappings,
        attn_groups,
        kv_cache_config,
        for_capture=False,
    ) -> dict[str, Any]:
        standard_metadata = super().prepare_attn(
            input_batch,
            cudagraph_mode,
            block_tables,
            slot_mappings,
            attn_groups,
            kv_cache_config,
            for_capture,
        )
        parallel_batch = self._make_parallel_batch(input_batch)
        self._parallel_batch = parallel_batch
        self._parallel_slot_mapping = None
        self._refresh_slot_mapping = None
        self._refresh_attn_metadata = None
        self._parallel_segment_meta: list[dict[str, Any]] = []
        self._refresh_segment_meta: list[dict[str, Any]] = []
        if not parallel_batch.positions:
            return standard_metadata

        # Build per-request parallel-layer metadata. Batched FlashAttention over multiple
        # independent DMTD shadow blocks has been observed to corrupt non-head
        # cycle slots under Refresh; sequential per-request FA is the
        # correctness path (still one model forward for the sequential layers).
        offset = 0
        for batch_idx, qlen, seq_len in zip(
            parallel_batch.req_batch_indices, parallel_batch.query_lens, parallel_batch.seq_lens
        ):
            seg_positions = parallel_batch.positions[offset : offset + qlen]
            seg_block_tables, seg_slot_mappings = self._build_slot_mapping(
                [batch_idx],
                [qlen],
                seg_positions,
                block_tables,
                kv_cache_config,
                attn_groups,
            )
            seg_metadata = self._build_parallel_metadata(
                [batch_idx],
                [qlen],
                [seq_len],
                seg_positions,
                seg_block_tables,
                seg_slot_mappings,
                attn_groups,
                kv_cache_config,
                for_capture,
            )
            seg_metadata = self._own_parallel_metadata(seg_metadata)
            seg_slots = build_slot_mappings_by_layer(
                seg_slot_mappings, kv_cache_config
            )[self._parallel_layer_names[0]]
            self._parallel_segment_meta.append(
                {
                    "start": offset,
                    "end": offset + qlen,
                    "slot_mapping": seg_slots,
                    "attn_metadata": {
                        name: seg_metadata[name] for name in self._parallel_layer_names
                    },
                }
            )
            offset += qlen

        # Install the first segment's metadata so a single-request path and
        # any code that inspects parallel-layer metadata still see valid FA
        # state; the model forward reinstalls each segment in order.
        for layer_name in self._parallel_layer_names:
            standard_metadata[layer_name] = self._parallel_segment_meta[0]["attn_metadata"][
                layer_name
            ]
        self._parallel_slot_mapping = self._parallel_segment_meta[0]["slot_mapping"]

        if parallel_batch.refresh_positions:
            refresh_offset = 0
            for batch_idx, qlen, seq_len in zip(
                parallel_batch.refresh_req_batch_indices,
                parallel_batch.refresh_query_lens,
                parallel_batch.refresh_seq_lens,
            ):
                seg_positions = parallel_batch.refresh_positions[
                    refresh_offset : refresh_offset + qlen
                ]
                seg_block_tables, seg_slot_mappings = self._build_slot_mapping(
                    [batch_idx],
                    [qlen],
                    seg_positions,
                    block_tables,
                    kv_cache_config,
                    attn_groups,
                )
                seg_metadata = self._build_parallel_metadata(
                    [batch_idx],
                    [qlen],
                    [seq_len],
                    seg_positions,
                    seg_block_tables,
                    seg_slot_mappings,
                    attn_groups,
                    kv_cache_config,
                    for_capture,
                    enable_bidirectional_shadow=False,
                )
                seg_metadata = self._own_parallel_metadata(seg_metadata)
                seg_slots = build_slot_mappings_by_layer(
                    seg_slot_mappings, kv_cache_config
                )[self._parallel_layer_names[0]]
                self._refresh_segment_meta.append(
                    {
                        "start": refresh_offset,
                        "end": refresh_offset + qlen,
                        "slot_mapping": seg_slots,
                        "attn_metadata": {
                            name: seg_metadata[name]
                            for name in self._parallel_layer_names
                        },
                    }
                )
                refresh_offset += qlen
            self._refresh_attn_metadata = self._refresh_segment_meta[0]["attn_metadata"]
            self._refresh_slot_mapping = self._refresh_segment_meta[0]["slot_mapping"]
        return standard_metadata

    def prepare_inputs(
        self, input_batch: InputBatch, req_states: RequestState
    ) -> dict[str, Any]:
        model_inputs = super().prepare_inputs(input_batch, req_states)
        parallel_batch = self._parallel_batch
        if parallel_batch is None:
            parallel_batch = self._make_parallel_batch(input_batch)
            self._parallel_slot_mapping = None
            self._refresh_slot_mapping = None
            self._refresh_attn_metadata = None

        device = self.device

        def expand_req_slots(
            req_batch_indices: list[int], query_lens: list[int]
        ) -> list[int]:
            slots: list[int] = []
            for batch_idx, qlen in zip(req_batch_indices, query_lens):
                req_slot = int(input_batch.idx_mapping_np[batch_idx])
                slots.extend([req_slot] * qlen)
            return slots

        def fill_ids(abs_positions: list[int], req_slots: list[int]) -> torch.Tensor:
            ids = torch.full(
                (len(abs_positions),),
                self.mask_token_id,
                dtype=input_batch.input_ids.dtype,
                device=device,
            )
            if not abs_positions:
                return ids
            all_ids = req_states.all_token_ids.gpu
            for i, (abs_pos, req_slot) in enumerate(zip(abs_positions, req_slots)):
                if abs_pos < 0:
                    continue
                ids[i] = all_ids[req_slot, abs_pos]
            return ids

        # Prefer absolute positions from all_token_ids; fall back to scheduled
        # input_ids indices for Norefresh back-compat when abs list is empty.
        if parallel_batch.token_abs_positions:
            main_slots = expand_req_slots(parallel_batch.req_batch_indices, parallel_batch.query_lens)
            shadow_ids = fill_ids(parallel_batch.token_abs_positions, main_slots)
        else:
            shadow_ids = torch.full(
                (len(parallel_batch.positions),),
                self.mask_token_id,
                dtype=input_batch.input_ids.dtype,
                device=device,
            )
            if parallel_batch.real_input_indices:
                real_indices = torch.tensor(
                    parallel_batch.real_input_indices, dtype=torch.long, device=device
                )
                real = real_indices >= 0
                shadow_ids[real] = input_batch.input_ids[real_indices[real]]

        refresh_ids = torch.empty(0, dtype=input_batch.input_ids.dtype, device=device)
        refresh_positions = torch.empty(0, dtype=torch.long, device=device)
        if parallel_batch.refresh_positions:
            refresh_slots = expand_req_slots(
                parallel_batch.refresh_req_batch_indices, parallel_batch.refresh_query_lens
            )
            refresh_ids = fill_ids(parallel_batch.refresh_token_abs_positions, refresh_slots)
            refresh_positions = torch.tensor(
                parallel_batch.refresh_positions, dtype=torch.long, device=device
            )

        def as_long(values: list[int]) -> torch.Tensor:
            return torch.tensor(values, dtype=torch.long, device=device)

        model_inputs.update(
            dmtd_shadow_input_ids=shadow_ids,
            dmtd_shadow_positions=as_long(parallel_batch.positions),
            dmtd_shadow_slot_mapping=self._parallel_slot_mapping,
            dmtd_parallel_segments=self._parallel_segment_meta,
            dmtd_refresh_input_ids=refresh_ids,
            dmtd_refresh_positions=refresh_positions,
            dmtd_refresh_slot_mapping=self._refresh_slot_mapping,
            dmtd_refresh_attn_metadata=self._refresh_attn_metadata,
            dmtd_refresh_segments=self._refresh_segment_meta,
            dmtd_direct_indices=as_long(parallel_batch.direct_indices),
            dmtd_seq_req_slots=as_long(parallel_batch.seq_req_slots),
            dmtd_seq_phases=as_long(parallel_batch.seq_phases),
            dmtd_cycle_hidden=self.cycle_hidden,
            dmtd_write_req_slots=as_long(parallel_batch.write_req_slots),
            dmtd_write_phases=as_long(parallel_batch.write_phases),
            dmtd_write_shadow_indices=as_long(parallel_batch.write_shadow_indices),
            dmtd_history_mode=self.history_mode,
        )
        self._parallel_batch = None
        self._parallel_slot_mapping = None
        self._refresh_slot_mapping = None
        self._refresh_attn_metadata = None
        self._parallel_segment_meta = []
        self._refresh_segment_meta = []
        return model_inputs
