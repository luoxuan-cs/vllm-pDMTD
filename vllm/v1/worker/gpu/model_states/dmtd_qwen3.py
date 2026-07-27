# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""V2 ModelState and cycle planners for DMTD Qwen3 variants.

Semantics (see parallel-eval/models/README.md for the training-side spec):

- Prefill (``pos < request.prompt_len``) never touches the DMTD cycle
  machinery at all: prompt tokens flow through the full 36-layer stack as one
  continuous, ordinary causal decoder pass (no MASK substitution, no
  parallel/sequential recombination). This is handled by the ``prefill``
  group below and consumed directly by the model as P28's raw output.
- Decode (``pos >= request.prompt_len``) uses the DMTD cycle mechanism, with
  the cycle phase anchored to *this request's own* prompt length rather than
  absolute sequence position: ``local_pos = pos - prompt_len``. The first
  generated token of every request is therefore always exactly a fresh cycle
  head (``local_pos == 0``), so there is no "prompt ends mid-cycle" case to
  special-case.
- All parallel-layer (P28) work for a scheduler step is batched across
  *every* request that needs it this step, grouped by required attention
  treatment (``causal`` vs ``noncausal``), instead of looped one request at a
  time. Batching independent requests together is safe for both causal and
  non-causal calls: each request's own ``block_table``/``seq_lens`` already
  isolates its KV history from every other request's, so ``causal=False``
  can only add visibility *within* one request's own tokens, never leak
  across requests.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from vllm.config.compilation import CUDAGraphMode
from vllm.forward_context import set_forward_context
from vllm.logger import init_logger
from vllm.v1.attention.backends.utils import PAD_SLOT_ID
from vllm.v1.worker.gpu.attn_utils import (
    build_attn_metadata,
    build_slot_mappings_by_layer,
)
from vllm.v1.worker.gpu.buffer_utils import async_copy_to_gpu
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.model_states.default import DefaultModelState
from vllm.v1.worker.gpu.model_states.interface import ModelSpecificAttnMetadata
from vllm.v1.worker.gpu.states import RequestState

logger = init_logger(__name__)


@dataclass
class _ParallelIsPrefillingAttnMetadata(ModelSpecificAttnMetadata):
    """Forces every grouped parallel-layer forward to build prefill-shaped
    attention metadata, since each group always processes fresh MASK/refresh/
    prompt tokens rather than a single decode step."""

    is_prefilling: torch.Tensor

    def get_extra_common_attn_kwargs(
        self,
        kv_cache_group_id: int,
        num_reqs: int,
    ) -> dict[str, Any]:
        return {"is_prefilling": self.is_prefilling[:num_reqs]}


@dataclass
class _GroupPlan:
    """One batched attention call spanning however many requests need this
    exact treatment (``causal`` bool) this step. Multiple independent
    requests contribute their own rows; ``causal`` is shared by the whole
    call (it only affects intra-request visibility, never leaks across the
    per-request KV histories isolated by ``block_table``/``seq_lens``)."""

    causal: bool
    req_batch_indices: list[int] = field(default_factory=list)
    query_lens: list[int] = field(default_factory=list)
    seq_lens: list[int] = field(default_factory=list)
    positions: list[int] = field(default_factory=list)
    # -1 => fill with the MASK token; else an absolute position to read the
    # real token id for from this request's full token history.
    token_abs_positions: list[int] = field(default_factory=list)
    # Whether each row's K/V is written back to the cache. False for rows that
    # exist only to keep the group's shape uniform: recomputing a position is
    # semantically a no-op but not bit-exact (a different query length makes
    # FlashAttention tile differently, and bf16 reductions are order-sensitive),
    # so a row that has nothing to refresh must leave the cache alone.
    writes_kv: list[bool] = field(default_factory=list)

    def add(
        self,
        batch_idx: int,
        positions: list[int],
        token_abs_positions: list[int],
        seq_len: int,
        writes_kv: list[bool] | None = None,
    ) -> int:
        """Append one request's contribution; returns its base offset within
        this group's flattened ``positions``."""
        if not positions:
            return -1
        base = len(self.positions)
        self.req_batch_indices.append(batch_idx)
        self.query_lens.append(len(positions))
        self.seq_lens.append(seq_len)
        self.positions.extend(positions)
        self.token_abs_positions.extend(token_abs_positions)
        self.writes_kv.extend(
            [True] * len(positions) if writes_kv is None else writes_kv
        )
        return base


@dataclass(frozen=True)
class _ParallelRowLayout:
    """Fixed row ranges carved out of the persistent parallel-output table.

    Each group writes to a range whose base never moves, which is what lets a
    group's forward be captured as a CUDA graph: the captured kernels bake in
    the output address they saw. Group bases must therefore not depend on how
    many rows the *other* groups happen to hold this step.

    The two scratch rows absorb padding. Index vectors are staged at a fixed
    length so a captured graph copies the same number of rows on every replay,
    and the entries that correspond to no real work are pointed here instead of
    being left out.
    """

    prefill_base: int
    causal_base: int
    noncausal_base: int
    parallel_scratch_row: int
    cycle_scratch_row: int
    tau: int
    write_capacity: int

    @property
    def num_parallel_rows(self) -> int:
        return self.parallel_scratch_row + 1

    def group_base(self, group: int) -> int:
        """Row base for a decode group tag (0=causal, 1=noncausal)."""
        return self.causal_base if group == 0 else self.noncausal_base


@dataclass
class _CyclePlan:
    """Full per-step plan: three attention groups plus the index wiring the
    model needs to assemble S8's input from their outputs."""

    prefill: _GroupPlan = field(default_factory=lambda: _GroupPlan(causal=True))
    causal: _GroupPlan = field(default_factory=lambda: _GroupPlan(causal=True))
    noncausal: _GroupPlan = field(default_factory=lambda: _GroupPlan(causal=False))

    # Prefill-group wiring: for token i in the scheduled batch, index into
    # `prefill`'s output (via `orig_token_indices`, see below), or -1.
    prefill_target_indices: list[int] = field(default_factory=list)
    # Original (scheduled-batch) token index each prefill-group row came
    # from, in the same flattened order as `prefill.positions` -- lets
    # `prepare_inputs` read real token ids directly out of this step's own
    # `input_batch.input_ids` instead of the historical all_token_ids lookup
    # the decode groups need.
    prefill_orig_token_indices: list[int] = field(default_factory=list)

    # Decode-group (S8) wiring, aligned with the scheduled-batch token order.
    # `direct_group`/`direct_local`: which group (0=causal, 1=noncausal, -1=none)
    # and the LOCAL offset within that group's own positions; resolved to a
    # single global index into cat(causal.out, noncausal.out) once the whole
    # plan is built (see `resolve_offsets`).
    direct_group: list[int] = field(default_factory=list)
    direct_local: list[int] = field(default_factory=list)
    seq_req_slots: list[int] = field(default_factory=list)
    seq_phases: list[int] = field(default_factory=list)

    # cycle_hidden buffer write wiring (decode positions only), same
    # local-offset + group-tag scheme as `direct_group`/`direct_local`.
    write_req_slots: list[int] = field(default_factory=list)
    write_phases: list[int] = field(default_factory=list)
    write_group: list[int] = field(default_factory=list)
    write_local: list[int] = field(default_factory=list)

    def group(self, kind: str) -> _GroupPlan:
        """The group named by `kind` (see `DMTDQwen3ModelState.PARALLEL_GROUPS`)."""
        return getattr(self, kind)

    def resolve_offsets(self) -> tuple[list[int], list[int]]:
        """Convert (group, local-offset) pairs into global indices into
        cat(causal.out, noncausal.out). Returns (direct_indices, write_shadow_indices)."""
        causal_len = len(self.causal.positions)

        def resolve(groups: list[int], locals_: list[int]) -> list[int]:
            out = []
            for grp, local in zip(groups, locals_):
                if grp < 0:
                    out.append(-1)
                elif grp == 0:
                    out.append(local)
                else:
                    out.append(causal_len + local)
            return out

        return (
            resolve(self.direct_group, self.direct_local),
            resolve(self.write_group, self.write_local),
        )

    def resolve_gather_indices(
        self, layout: _ParallelRowLayout, num_tokens: int
    ) -> tuple[list[int], list[int], list[bool], list[int], list[int]]:
        """Flatten the whole plan into index vectors the model can consume with
        plain ``index_select``/``index_copy_``, i.e. with no boolean masking and
        no device-side reads.

        All parallel-layer output lives in one persistent table with a fixed row
        range per group (see `_ParallelRowLayout`), so a row index stays valid
        no matter what the other groups hold this step. The cycle buffer is
        addressed as a flat ``((max_num_reqs + 1) * tau, hidden)`` table.

        Returns ``(src_index, buf_index, add_embed, write_src_index,
        write_buf_index)``. The first three are aligned to the scheduled token
        order; the two ``write_*`` vectors are always
        ``layout.write_capacity`` long, padded with scratch rows so that a
        captured graph performs the same copy on every replay.

        - ``src_index[i] >= 0``: take token i's S8 input from the parallel table
          at that row; ``-1`` means take it from the cycle buffer instead.
        - ``buf_index[i]``: flat cycle-buffer row for token i (only read when
          ``src_index[i] < 0``, but always in range so the gather is safe).
        - ``add_embed[i]``: whether ``embed(token)`` is added on top. False only
          for prefill positions, whose parallel output already carries the
          embedding through P28's own residual stream.
        """
        tau = layout.tau

        src_index = [-1] * num_tokens
        buf_index = [0] * num_tokens
        add_embed = [True] * num_tokens
        for i in range(num_tokens):
            prefill_row = self.prefill_target_indices[i]
            if prefill_row >= 0:
                src_index[i] = layout.prefill_base + prefill_row
                add_embed[i] = False
                continue
            group = self.direct_group[i]
            if group >= 0:
                src_index[i] = layout.group_base(group) + self.direct_local[i]
            else:
                buf_index[i] = self.seq_req_slots[i] * tau + self.seq_phases[i]

        num_writes = len(self.write_group)
        if num_writes > layout.write_capacity:
            raise ValueError(
                f"DMTD planned {num_writes} cycle-buffer writes but only "
                f"{layout.write_capacity} were reserved."
            )
        write_src_index = [
            layout.group_base(group) + local
            for group, local in zip(self.write_group, self.write_local)
        ]
        write_buf_index = [
            slot * tau + phase
            for slot, phase in zip(self.write_req_slots, self.write_phases)
        ]
        # Pad to the captured length. Surplus copies read one scratch source row
        # and land on distinct scratch destination rows, so they are both
        # harmless and free of duplicate `index_copy_` destinations.
        for pad in range(num_writes, layout.write_capacity):
            write_src_index.append(layout.parallel_scratch_row)
            write_buf_index.append(layout.cycle_scratch_row + (pad % tau))
        return src_index, buf_index, add_embed, write_src_index, write_buf_index


class _StagedIndexVectors:
    """Fixed-address int64 staging for the plan's per-step index vectors.

    All vectors share one pinned host region and one device region, carved into
    fixed per-name slices. Two properties matter:

    - One H2D copy per step instead of one per vector (the previous code issued
      a dozen separate ``torch.tensor(list, device=cuda)`` copies).
    - The device addresses never move, which is what CUDA graph capture
      requires: a captured graph reads whatever address it saw at capture time,
      so any per-step reallocation would make replay read freed memory.
    """

    def __init__(self, capacities: dict[str, int], device: torch.device) -> None:
        self._slices: dict[str, tuple[int, int]] = {}
        total = 0
        for name, capacity in capacities.items():
            self._slices[name] = (total, capacity)
            total += capacity
        self._host = torch.zeros(total, dtype=torch.int64, pin_memory=True)
        self._host_np = self._host.numpy()
        self._device = torch.zeros(total, dtype=torch.int64, device=device)
        self._lengths: dict[str, int] = dict.fromkeys(capacities, 0)

    def stage(self, name: str, values: list[int] | list[bool]) -> int:
        start, capacity = self._slices[name]
        length = len(values)
        if length > capacity:
            raise ValueError(
                f"DMTD staging vector {name!r} needs {length} entries but only "
                f"{capacity} were reserved."
            )
        if length:
            self._host_np[start : start + length] = values
        self._lengths[name] = length
        return length

    def flush(self) -> None:
        async_copy_to_gpu(self._host, out=self._device)

    def view(self, name: str) -> torch.Tensor:
        start, _ = self._slices[name]
        return self._device[start : start + self._lengths[name]]

    def host_np(self, name: str) -> np.ndarray:
        start, capacity = self._slices[name]
        return self._host_np[start : start + capacity]


@dataclass(frozen=True)
class _GroupShape:
    """The shape one parallel group runs at this step.

    `num_reqs` and `num_rows` are what the attention call is *built* for, which
    is a padded bucket when the group's forward will be replayed from a CUDA
    graph (tensor shapes and grid sizes are baked into a captured graph, so they
    have to be identical on every replay) and the real counts otherwise.
    """

    num_reqs: int
    num_rows: int
    graphed: bool


class _ParallelGroupBuffers:
    """Persistent attention inputs for one parallel-layer group.

    Every tensor a captured attention call reads has to live at an address that
    never moves, so these are allocated once at their worst case and refilled in
    place. Requests are numbered within the group (only those that need parallel
    work this step take part), which keeps the group's request count -- and so
    its graph -- one-dimensional.

    Rows and requests past what the group actually holds are padded to be inert,
    following the recipe the DFlash drafter uses: ``PAD_SLOT_ID`` blocks the KV
    write, ``seq_len == 0`` blocks the attention read, and ``query_start_loc``
    stays non-decreasing.
    """

    def __init__(self, *, max_rows: int, max_reqs: int, device: torch.device) -> None:
        self.max_rows = max_rows
        self.max_reqs = max_reqs
        # query_start_loc (max_reqs + 1 entries) and seq_lens (max_reqs) share
        # one pinned region so refilling both costs a single H2D copy.
        self._host = torch.zeros(2 * max_reqs + 1, dtype=torch.int32, pin_memory=True)
        self._host_np = self._host.numpy()
        self._device = torch.zeros(2 * max_reqs + 1, dtype=torch.int32, device=device)
        self.query_start_loc = self._device[: max_reqs + 1]
        self.seq_lens = self._device[max_reqs + 1 :]
        self.query_start_loc_cpu = self._host[: max_reqs + 1]
        self.seq_lens_cpu = self._host[max_reqs + 1 :]
        self.slot_mapping = torch.full(
            (1, max_rows), PAD_SLOT_ID, dtype=torch.int64, device=device
        )
        # Every parallel-layer row is a fresh MASK / refresh / prompt token, so
        # each group always builds prefill-shaped metadata.
        self.is_prefilling = torch.ones(max_reqs, dtype=torch.bool)
        self._block_table: torch.Tensor | None = None
        self._padded_slot_rows = max_rows

    def stage_requests(
        self, *, query_lens: list[int], seq_lens: list[int], shape: _GroupShape
    ) -> None:
        """Refill query_start_loc / seq_lens for this group's requests.

        Surplus rows are attributed to the padded requests that follow the real
        ones; their `seq_len` of 0 is what makes those rows inert.
        """
        num_real = len(query_lens)
        qsl = self._host_np[: self.max_reqs + 1]
        qsl[0] = 0
        np.cumsum(query_lens, out=qsl[1 : num_real + 1])
        surplus = shape.num_rows - int(qsl[num_real])
        if surplus and shape.num_reqs > num_real:
            # Spread the surplus evenly so every padded request holds the same
            # number of rows, which is what keeps the batch uniform.
            per_req, extra = divmod(surplus, shape.num_reqs - num_real)
            for i in range(num_real, shape.num_reqs):
                rows = per_req + (1 if i - num_real < extra else 0)
                qsl[i + 1] = qsl[i] + rows
        elif surplus:
            qsl[num_real] += surplus
        qsl[shape.num_reqs + 1 :] = qsl[shape.num_reqs]
        lens = self._host_np[self.max_reqs + 1 :]
        lens[:num_real] = seq_lens
        lens[num_real:] = 0
        async_copy_to_gpu(self._host, out=self._device)

    def fill_block_table(
        self, source: torch.Tensor, req_indices: torch.Tensor, num_reqs: int
    ) -> torch.Tensor:
        """Gather this group's requests' block tables into a fixed buffer.

        Returns the first `num_reqs` rows, which includes the padded requests --
        their rows are never read because their `seq_len` is 0.
        """
        if self._block_table is None or self._block_table.shape[1] != source.shape[1]:
            self._block_table = torch.zeros(
                (self.max_reqs, source.shape[1]),
                dtype=source.dtype,
                device=source.device,
            )
        num_real = int(req_indices.numel())
        if num_real:
            torch.index_select(
                source, 0, req_indices, out=self._block_table[:num_real]
            )
        return self._block_table[:num_reqs]

    def fill_slot_mapping(
        self,
        *,
        positions: torch.Tensor,
        row_group_reqs: torch.Tensor,
        writes_kv: torch.Tensor,
        block_table: torch.Tensor,
        block_size: int,
        num_real_rows: int,
        padded_rows: int,
    ) -> torch.Tensor:
        """Compute this group's slot mapping in place, then pad it out.

        Only the first `num_real_rows` can get a real slot, and only those whose
        `writes_kv` is set; everything else gets ``PAD_SLOT_ID`` so the KV-cache
        write skips it. Padding is rewritten from `num_real_rows` on every step
        rather than from wherever the last step stopped, because a stale *valid*
        slot left in this shared buffer would corrupt the KV cache.
        """
        out = self.slot_mapping[0]
        if num_real_rows:
            num_cols = block_table.shape[1]
            real_positions = positions[:num_real_rows]
            block_numbers = block_table.view(-1).index_select(
                0,
                row_group_reqs[:num_real_rows] * num_cols
                + (real_positions // block_size).clamp(max=num_cols - 1),
            )
            torch.where(
                writes_kv[:num_real_rows] > 0,
                block_numbers * block_size + real_positions % block_size,
                out.new_full((), PAD_SLOT_ID),
                out=out[:num_real_rows],
            )
        pad_end = max(padded_rows, self._padded_slot_rows)
        if pad_end > num_real_rows:
            out[num_real_rows:pad_end].fill_(PAD_SLOT_ID)
        self._padded_slot_rows = num_real_rows
        return self.slot_mapping[:, :padded_rows]


class NoRefreshCyclePlanner:
    """Persistent shadow-KV planner (Causal/Bidirectional-Parallel-Norefresh)."""

    def __init__(self, state: DMTDQwen3ModelState) -> None:
        self.state = state

    def needs_parallel_work(
        self, *, req_slot: int, start: int, query_len: int
    ) -> bool:
        """Mirror of `plan_request`'s early return, without touching any state.

        Must stay in sync with the buffer-hit condition below; a regression test
        asserts the two agree.
        """
        if query_len > 1:
            return True
        state = self.state
        local_pos = start - int(state._prompt_len[req_slot])
        cycle_base_local = local_pos - local_pos % state.tau
        phase = local_pos % state.tau
        return not (
            state._cycle_base[req_slot] == cycle_base_local
            and state._buffer_valid[req_slot, phase]
        )

    def plan_request(
        self,
        plan: _CyclePlan,
        *,
        batch_idx: int,
        req_slot: int,
        start: int,
        query_len: int,
        query_start: int,
    ) -> None:
        state = self.state
        prompt_len = int(state._prompt_len[req_slot])
        tau = state.tau

        for local_idx, pos in enumerate(range(start, start + query_len)):
            token_idx = query_start + local_idx
            plan.seq_req_slots[token_idx] = req_slot
            plan.seq_phases[token_idx] = (pos - prompt_len) % tau

        is_multi_token = query_len > 1
        group = plan.noncausal if state.block_attention == "bidirectional" else plan.causal

        req_positions: list[int]
        req_token_abs: list[int]
        if is_multi_token:
            local_start = start - prompt_len
            if state.block_attention == "bidirectional":
                # Every hidden in a bidirectional shadow cycle depends on all
                # tau slots. Expand every touched cycle to its complete
                # [head, M, ..., M] block.
                parallel_start_local = local_start - local_start % tau
                scheduled_end_local = local_start + query_len
                parallel_end_local = min(
                    ((scheduled_end_local + tau - 1) // tau) * tau,
                    state.max_model_len - prompt_len,
                )
                req_positions = [
                    prompt_len + p for p in range(parallel_start_local, parallel_end_local)
                ]
            else:
                req_positions = list(
                    range(start, min(start + query_len, state.max_model_len))
                )
            if not req_positions:
                return
            req_token_abs = [
                pos if (pos - prompt_len) % tau == 0 else -1 for pos in req_positions
            ]
            base = group.add(batch_idx, req_positions, req_token_abs, req_positions[-1] + 1)
            parallel_start = req_positions[0]
            group_tag = 1 if group is plan.noncausal else 0
            req_positions_end = req_positions[-1] + 1
            for local_idx, pos in enumerate(range(start, start + query_len)):
                # `req_positions` may have been clamped to max_model_len (e.g.
                # a memory-profiling dummy run can request an out-of-range
                # query_len); scheduled tokens beyond that clamp never had a
                # parallel-layer hidden computed for them, so leave their
                # wiring at the default "buffered" (-1) rather than pointing
                # past the end of this group's output.
                if not (parallel_start <= pos < req_positions_end):
                    continue
                token_idx = query_start + local_idx
                plan.direct_group[token_idx] = group_tag
                plan.direct_local[token_idx] = base + (pos - parallel_start)
        else:
            assert query_len == 1
            pos = start
            local_pos = pos - prompt_len
            cycle_base_local = local_pos - local_pos % tau
            phase = local_pos % tau
            has_hidden = (
                state._cycle_base[req_slot] == cycle_base_local
                and state._buffer_valid[req_slot, phase]
            )
            if has_hidden:
                return
            cycle_base = prompt_len + cycle_base_local
            cycle_end = min(cycle_base + tau, state.max_model_len)
            # Always the whole cycle, from its head, which keeps every cycle-head
            # step a uniform `tau`-row batch. At a cycle head (the normal case)
            # `pos` *is* the head, so this is the only option anyway. Resuming
            # mid-cycle after a schedule gap it also re-runs the earlier phases,
            # whose shadow KV is already in the cache; those rows keep their
            # write suppressed, since recomputing a position reproduces it only
            # up to bf16 rounding, not bit-exactly.
            req_positions = list(range(cycle_base, cycle_end))
            req_token_abs = [-1] * len(req_positions)
            if req_positions:
                req_token_abs[0] = cycle_base
            req_writes_kv = [p >= pos for p in req_positions]

            if not req_positions:
                return
            base = group.add(
                batch_idx,
                req_positions,
                req_token_abs,
                req_positions[-1] + 1,
                req_writes_kv,
            )

        final_cycle_base_local = (
            req_positions[-1] - prompt_len - (req_positions[-1] - prompt_len) % tau
        )
        if state._cycle_base[req_slot] != final_cycle_base_local:
            state._invalidate_buffer(req_slot)
            state._cycle_base[req_slot] = final_cycle_base_local
        group_tag = 1 if group is plan.noncausal else 0
        for offset, pos in enumerate(req_positions):
            local_pos = pos - prompt_len
            if local_pos - local_pos % tau != final_cycle_base_local:
                continue
            phase = local_pos % tau
            state._buffer_valid[req_slot, phase] = True
            plan.write_req_slots.append(req_slot)
            plan.write_phases.append(phase)
            plan.write_group.append(group_tag)
            plan.write_local.append(base + offset)


class RefreshCyclePlanner:
    """Real-KV refresh planner (Causal/Bidirectional-Parallel-Refresh)."""

    def __init__(self, state: DMTDQwen3ModelState) -> None:
        self.state = state

    def needs_parallel_work(
        self, *, req_slot: int, start: int, query_len: int
    ) -> bool:
        """Mirror of `plan_request`'s early return, without touching any state.

        Must stay in sync with the contiguity-gap and buffer-hit conditions
        below; a regression test asserts the two agree.
        """
        state = self.state
        tau = state.tau
        expected = int(state._next_computed[req_slot])
        if expected != -1 and start != expected:
            # A schedule gap invalidates the cycle buffer, forcing a rebuild.
            return True
        local_start = start - int(state._prompt_len[req_slot])
        cycle_base_local = local_start - local_start % tau
        if state._cycle_base[req_slot] != cycle_base_local:
            return True
        return not all(
            state._buffer_valid[req_slot, (local_start + i) % tau]
            for i in range(query_len)
        )

    def plan_request(
        self,
        plan: _CyclePlan,
        *,
        batch_idx: int,
        req_slot: int,
        start: int,
        query_len: int,
        query_start: int,
    ) -> None:
        state = self.state
        prompt_len = int(state._prompt_len[req_slot])
        tau = state.tau

        # Contiguity cursor: advanced to start+query_len at the end of every
        # successful plan_request. A mismatch means the scheduler skipped or
        # rewound tokens (preemption / recompute), so the cycle buffer and
        # Refresh lag may be stale. Historically this cursor was never
        # advanced, so *every* decode step looked like a gap and re-ran P28.
        expected = int(state._next_computed[req_slot])
        if expected != -1 and start != expected:
            state._invalidate_buffer(req_slot)
            # Resume with Refresh lag: completed prior cycles only. At a
            # cycle head the previous cycle is not yet refreshed.
            local_start = start - prompt_len
            cycle_base_resume_local = local_start - local_start % tau
            if local_start > 0 and local_start % tau == 0:
                state._parallel_real_len[req_slot] = cycle_base_resume_local - tau
            else:
                state._parallel_real_len[req_slot] = cycle_base_resume_local

        for local_idx, pos in enumerate(range(start, start + query_len)):
            token_idx = query_start + local_idx
            plan.seq_req_slots[token_idx] = req_slot
            plan.seq_phases[token_idx] = (pos - prompt_len) % tau

        local_start = start - prompt_len
        cycle_base_local = local_start - local_start % tau
        needs_parallel = True
        if state._cycle_base[req_slot] == cycle_base_local:
            phases_needed = [(local_start + i) % tau for i in range(query_len)]
            if all(state._buffer_valid[req_slot, ph] for ph in phases_needed):
                needs_parallel = False

        if needs_parallel:
            phase_to_write = self._plan_cycle_parallel(
                plan,
                batch_idx=batch_idx,
                req_slot=req_slot,
                prompt_len=prompt_len,
                cycle_base_local=cycle_base_local,
            )

            for local_idx in range(query_len):
                phase = (local_start + local_idx) % tau
                write = phase_to_write.get(phase)
                if write is not None:
                    token_idx = query_start + local_idx
                    plan.direct_group[token_idx], plan.direct_local[token_idx] = write

        state._next_computed[req_slot] = start + query_len

    def _plan_cycle_parallel(
        self,
        plan: _CyclePlan,
        *,
        batch_idx: int,
        req_slot: int,
        prompt_len: int,
        cycle_base_local: int,
    ) -> dict[int, tuple[int, int]]:
        state = self.state
        tau = state.tau
        cycle_base = prompt_len + cycle_base_local
        cycle_end = min(cycle_base + tau, state.max_model_len)
        shadow_positions = list(range(cycle_base, cycle_end))
        shadow_token_abs = [
            pos if pos == cycle_base else -1 for pos in shadow_positions
        ]

        # A refresh block is emitted on every cycle, so that a cycle head is
        # always the same `2*tau` (causal) or `tau + tau` (bidirectional) shape
        # and can therefore replay a captured graph.
        #
        # On a steady cycle it does its real job: overwrite the previous cycle's
        # shadow KV with real KV. When there is nothing to refresh -- the first
        # cycle -- the rows are still emitted, over the `tau` real positions
        # behind the cycle head, but with their KV write suppressed. Those slots
        # already hold real KV from prefill, and rewriting them would perturb it
        # at the ULP level rather than reproduce it.
        need_refresh = (
            cycle_base_local > 0
            and int(state._parallel_real_len[req_slot]) < cycle_base_local
        )
        refresh_start = cycle_base - tau
        refresh_positions = (
            list(range(refresh_start, cycle_base)) if refresh_start >= 0 else []
        )
        refresh_token_abs = list(refresh_positions)
        refresh_writes_kv = [need_refresh] * len(refresh_positions)

        if state._cycle_base[req_slot] != cycle_base_local:
            state._invalidate_buffer(req_slot)
            state._cycle_base[req_slot] = cycle_base_local
        state._buffer_valid[req_slot].fill(False)

        if state.block_attention == "bidirectional":
            # Refresh (causal continuation of already-known real tokens) and
            # the current cycle's shadow block (bidirectional within itself)
            # need different `causal` flags, so they cannot share one row --
            # refresh always goes to the causal group, shadow always to the
            # noncausal group.
            if refresh_positions:
                plan.causal.add(
                    batch_idx,
                    refresh_positions,
                    refresh_token_abs,
                    cycle_base,
                    refresh_writes_kv,
                )
            shadow_base = plan.noncausal.add(
                batch_idx, shadow_positions, shadow_token_abs, cycle_end
            )
            shadow_group = 1
        else:
            # Causal block attention: cross-cycle-causal + within-cycle-causal
            # is exactly one continuous causal sequence, so refresh + shadow
            # share a single row in the causal group.
            combined_positions = refresh_positions + shadow_positions
            combined_token_abs = refresh_token_abs + shadow_token_abs
            base = plan.causal.add(
                batch_idx,
                combined_positions,
                combined_token_abs,
                cycle_end,
                refresh_writes_kv + [True] * len(shadow_positions),
            )
            shadow_base = base + len(refresh_positions)
            shadow_group = 0

        if need_refresh:
            # After this parallel-layer step, logical persistent real cache
            # reaches cycle_base_local.
            state._parallel_real_len[req_slot] = cycle_base_local

        phase_to_write: dict[int, tuple[int, int]] = {}
        for phase in range(len(shadow_positions)):
            state._buffer_valid[req_slot, phase] = True
            plan.write_req_slots.append(req_slot)
            plan.write_phases.append(phase)
            plan.write_group.append(shadow_group)
            plan.write_local.append(shadow_base + phase)
            phase_to_write[phase] = (shadow_group, shadow_base + phase)
        return phase_to_write


class DMTDQwen3ModelState(DefaultModelState):
    """Variant-aware DMTD state: shadow history or real refresh."""

    PARALLEL_GROUPS = ("prefill", "causal", "noncausal")
    # Metadata builder per group. Index 0 belongs to the sequential layers; a
    # step holds all four sets live at once, so none may share a builder.
    # `attn_utils.get_num_metadata_builders` must allocate at least as many.
    GROUP_BUILDER_IDX = {"prefill": 1, "causal": 2, "noncausal": 3}

    def __init__(self, vllm_config, model, encoder_cache, device) -> None:
        super().__init__(vllm_config, model, encoder_cache, device)
        config = model.config
        self.tau = int(config.mtp_horizon)
        self.mask_token_id = int(config.mask_token_id)
        self.num_parallel_layers = int(config.num_parallel_layers)
        self.num_hidden_layers = int(config.num_hidden_layers)
        self.history_mode = getattr(config, "dmtd_history_mode", "shadow")
        self.block_attention = getattr(config, "dmtd_block_attention", "causal")

        self._req_id_to_index: dict[str, int] = {}
        self._prompt_len = np.zeros(self.max_num_reqs, dtype=np.int64)
        self._cycle_base = np.full(self.max_num_reqs, -1, dtype=np.int64)
        self._next_computed = np.full(self.max_num_reqs, -1, dtype=np.int64)
        self._buffer_valid = np.zeros((self.max_num_reqs, self.tau), dtype=np.bool_)
        self._parallel_real_len = np.zeros(self.max_num_reqs, dtype=np.int64)
        # One extra request slot beyond `max_num_reqs` is scratch: padded
        # cycle-buffer writes land there instead of corrupting a live request.
        self.cycle_hidden = torch.zeros(
            self.max_num_reqs + 1,
            self.tau,
            config.hidden_size,
            dtype=self.dtype,
            device=device,
        )
        self._cycle_plan: _CyclePlan | None = None
        self._prefill_slot_mapping: torch.Tensor | None = None
        self._causal_slot_mapping: torch.Tensor | None = None
        self._noncausal_slot_mapping: torch.Tensor | None = None
        self._prefill_attn_metadata: dict[str, Any] | None = None
        self._causal_attn_metadata: dict[str, Any] | None = None
        self._noncausal_attn_metadata: dict[str, Any] | None = None

        # Per-token vectors can be as long as a padded batch; CUDA graph
        # padding rounds up to a capture size, which may exceed the scheduler's
        # own token budget in principle.
        max_padded_tokens = max(
            self.max_num_tokens,
            vllm_config.compilation_config.max_cudagraph_capture_size or 0,
        )
        # A decode request contributes at most 2*tau parallel-layer rows
        # (Refresh steady cycle = refresh block + shadow block); a multi-token
        # (dummy/profile) batch can contribute up to its whole query length.
        max_group_rows = max_padded_tokens + self.max_num_reqs * 2 * self.tau
        self._max_padded_tokens = max_padded_tokens
        self._max_group_rows = max_group_rows
        # Persistent parallel-output table, one fixed row range per group.
        self.layout = _ParallelRowLayout(
            prefill_base=0,
            causal_base=max_padded_tokens,
            noncausal_base=max_padded_tokens + max_group_rows,
            parallel_scratch_row=max_padded_tokens + 2 * max_group_rows,
            cycle_scratch_row=self.max_num_reqs * self.tau,
            tau=self.tau,
            write_capacity=self.max_num_reqs * self.tau,
        )
        self.parallel_hidden = torch.zeros(
            self.layout.num_parallel_rows,
            config.hidden_size,
            dtype=self.dtype,
            device=device,
        )
        self._vectors = _StagedIndexVectors(
            {
                "prefill_positions": max_padded_tokens,
                "prefill_src_tokens": max_padded_tokens,
                "prefill_group_req": max_padded_tokens,
                "prefill_writes_kv": max_padded_tokens,
                "prefill_req_idx": self.max_num_reqs,
                "causal_positions": max_group_rows,
                "causal_abs": max_group_rows,
                "causal_slots": max_group_rows,
                "causal_group_req": max_group_rows,
                "causal_writes_kv": max_group_rows,
                "causal_req_idx": self.max_num_reqs,
                "noncausal_positions": max_group_rows,
                "noncausal_abs": max_group_rows,
                "noncausal_slots": max_group_rows,
                "noncausal_group_req": max_group_rows,
                "noncausal_writes_kv": max_group_rows,
                "noncausal_req_idx": self.max_num_reqs,
                "src_index": max_padded_tokens,
                "buf_index": max_padded_tokens,
                "add_embed": max_padded_tokens,
                "write_src_index": self.max_num_reqs * self.tau,
                "write_buf_index": self.max_num_reqs * self.tau,
            },
            device=device,
        )
        self._group_buffers = {
            "prefill": _ParallelGroupBuffers(
                max_rows=max_padded_tokens, max_reqs=self.max_num_reqs, device=device
            ),
            "causal": _ParallelGroupBuffers(
                max_rows=max_group_rows, max_reqs=self.max_num_reqs, device=device
            ),
            "noncausal": _ParallelGroupBuffers(
                max_rows=max_group_rows, max_reqs=self.max_num_reqs, device=device
            ),
        }
        self._group_shapes = {
            kind: _GroupShape(0, 0, False) for kind in self.PARALLEL_GROUPS
        }
        # Rows a single request contributes to each group on a decode cycle head.
        # Fixed per variant, which is what makes a cycle-head step a uniform
        # batch and so capturable -- the same property DFlash gets from freezing
        # `num_query_per_req` at config time.
        if self.history_mode == "real":
            if self.block_attention == "bidirectional":
                # Refresh is a causal continuation of real tokens, the shadow
                # block is bidirectional within itself, so they need separate
                # attention calls.
                self.DECODE_ROWS_PER_REQ = {"causal": self.tau, "noncausal": self.tau}
            else:
                # One continuous causal sequence: refresh and shadow share a row.
                self.DECODE_ROWS_PER_REQ = {"causal": 2 * self.tau}
        elif self.block_attention == "bidirectional":
            self.DECODE_ROWS_PER_REQ = {"noncausal": self.tau}
        else:
            self.DECODE_ROWS_PER_REQ = {"causal": self.tau}
        # Populated by `capture_extra_graphs`; until then every group runs eager.
        self._parallel_graph_managers: dict[str, Any] = {}
        # Token-id buffers for the parallel groups. Kept separate from the
        # int64 index staging because the model needs the embedding dtype, and
        # kept persistent for the same capture-stability reason.
        token_id_dtype = torch.int32
        self._prefill_ids = torch.zeros(
            max_padded_tokens, dtype=token_id_dtype, device=device
        )
        self._causal_ids = torch.zeros(
            max_group_rows, dtype=token_id_dtype, device=device
        )
        self._noncausal_ids = torch.zeros(
            max_group_rows, dtype=token_id_dtype, device=device
        )

        self._dmtd_model = model.model
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

    def capture_extra_graphs(self, block_tables, attn_groups, kv_cache_config) -> None:
        """Capture a CUDA graph per decode cycle-head shape, per group.

        The sequential-layer forward is already covered by the runner's own
        graphs and is shape-invariant, so these are the only graphs DMTD needs to
        own. Each is captured against a synthetic head plan, which keeps live
        per-request cycle state untouched -- important because the warmup pass
        before capture really executes.
        """
        from vllm.v1.worker.gpu.model_states.dmtd_cudagraph import (
            DMTDParallelCudaGraphManager,
        )

        cudagraph_mode = self.vllm_config.compilation_config.cudagraph_mode
        if cudagraph_mode.decode_mode() != CUDAGraphMode.FULL:
            return
        if os.environ.get("VLLM_DMTD_DISABLE_PARALLEL_CUDAGRAPH") == "1":
            # Benchmark switch: leaves the parallel layers eager while the
            # sequential-layer graphs stay on, to isolate their contribution.
            logger.info("DMTD parallel-layer CUDA graphs disabled by env var")
            return

        for kind, rows_per_req in self.DECODE_ROWS_PER_REQ.items():
            manager = DMTDParallelCudaGraphManager(
                self.vllm_config,
                self.device,
                cudagraph_mode,
                kind=kind,
                rows_per_req=rows_per_req,
            )
            if not manager.needs_capture():
                continue

            def prepare(num_reqs: int, kind=kind, rows_per_req=rows_per_req):
                plan = self.head_plan(kind, num_reqs)
                group = plan.group(kind)
                shape = _GroupShape(num_reqs, num_reqs * rows_per_req, True)
                self._stage_capture_group(plan, kind, shape)
                metadata, slot_mapping = self._prepare_one_group(
                    group,
                    kind,
                    block_tables.get_dummy_block_tables(num_reqs),
                    kv_cache_config,
                    attn_groups,
                    for_capture=True,
                    shape=shape,
                )
                assert metadata is not None and slot_mapping is not None
                ids = self._group_ids(kind)[: shape.num_rows]
                ids.fill_(self.mask_token_id)
                positions = self._vectors.view(f"{kind}_positions")
                return lambda: self._run_parallel_group_eager(
                    input_ids=ids,
                    positions=positions,
                    row_base=self._group_row_base(kind),
                    num_rows=shape.num_rows,
                    attn_metadata=metadata,
                    slot_mapping=slot_mapping,
                )

            manager.capture(
                prepare,
                progress_bar_desc=f"Capturing DMTD {kind} parallel-layer graphs",
            )
            self._parallel_graph_managers[kind] = manager

    def _stage_capture_group(
        self, plan: _CyclePlan, kind: str, shape: _GroupShape
    ) -> None:
        """Stage one group's row vectors for a capture run.

        Unlike `_stage_group_vectors` this touches only the group being captured
        and uses no `InputBatch`, since capture has no real batch. The slot
        mapping it feeds is all `PAD_SLOT_ID`, so no KV cache entry is written by
        either the warmup pass or any replay that does not first restage.
        """
        group = plan.group(kind)
        vectors = self._vectors
        self._group_shapes[kind] = shape
        vectors.stage(f"{kind}_positions", list(group.positions))
        vectors.stage(
            f"{kind}_group_req",
            [
                local
                for local, query_len in enumerate(group.query_lens)
                for _ in range(query_len)
            ],
        )
        vectors.stage(f"{kind}_req_idx", list(group.req_batch_indices))
        vectors.stage(f"{kind}_writes_kv", list(group.writes_kv))
        if kind != "prefill":
            vectors.stage(f"{kind}_abs", list(group.token_abs_positions))
            vectors.stage(f"{kind}_slots", [0] * shape.num_rows)
        vectors.flush()

    def _group_ids(self, kind: str) -> torch.Tensor:
        if kind == "prefill":
            return self._prefill_ids
        if kind == "causal":
            return self._causal_ids
        return self._noncausal_ids

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
        prompt_token_ids = new_req_data.prompt_token_ids
        self._prompt_len[req_index] = (
            len(prompt_token_ids) if prompt_token_ids is not None else 0
        )

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

    def requires_eager_step(
        self, scheduler_output: Any, req_states: RequestState
    ) -> bool:
        """Veto CUDA graph replay on a step that is still prefilling.

        The runner's graph covers the sequential layers plus the assembly that
        feeds them, and that part is shape-invariant: it reads one row per
        scheduled token out of two persistent tables, whether or not the parallel
        layers ran this step. A cycle-head step is therefore just as replayable
        as a mid-cycle one -- its parallel-layer work happens before the forward,
        under graphs this state owns (see `capture_extra_graphs`).

        Prefill is different: the token count is not a uniform decode shape, so
        the runner's dispatcher would not offer a decode graph for it anyway.
        Vetoing here keeps that explicit.
        """
        prefill_len = req_states.prefill_len.np
        computed_prefill = req_states.num_computed_prefill_tokens
        for req_id in scheduler_output.num_scheduled_tokens:
            req_slot = req_states.req_id_to_index.get(req_id)
            if req_slot is None:
                # Unknown request: fall back to eager rather than guess.
                return True
            if computed_prefill[req_slot] < prefill_len[req_slot]:
                return True
        return False

    def _blank_plan(self, num_tokens: int) -> _CyclePlan:
        """A plan with no parallel-layer work at all: every token reads its S8
        input from the cycle buffer.

        This is exactly the shape of a mid-cycle decode step, and it is the only
        shape CUDA graphs are captured for (see `requires_eager_step`). Building
        it takes no request state into account, so it is also side-effect free --
        important because capture drives `prepare_attn` with a synthetic batch
        whose request slots are the same ones live requests occupy.
        """
        plan = _CyclePlan()
        plan.prefill_target_indices = [-1] * num_tokens
        plan.direct_group = [-1] * num_tokens
        plan.direct_local = [-1] * num_tokens
        # The scratch request slot, so any token the caller does not overwrite
        # (CUDA graph padding rows) reads scratch rather than a live request's
        # buffered hidden. `plan_request` assigns a real slot to every token it
        # schedules, so only padding keeps this default.
        plan.seq_req_slots = [self.max_num_reqs] * num_tokens
        plan.seq_phases = [0] * num_tokens
        return plan

    def head_plan(self, kind: str, num_reqs: int) -> _CyclePlan:
        """A synthetic decode cycle-head plan for `kind`, for graph capture.

        Built from the shape alone -- `num_reqs` requests each contributing
        `DECODE_ROWS_PER_REQ[kind]` rows -- so it consults and mutates no live
        request state, the same property `_blank_plan` has and for the same
        reason: capture drives the planning path with a synthetic batch whose
        request slots are the ones live requests occupy.
        """
        rows_per_req = self.DECODE_ROWS_PER_REQ[kind]
        plan = self._blank_plan(num_reqs)
        group = plan.group(kind)
        for batch_idx in range(num_reqs):
            # Positions are irrelevant to the captured shape and are overwritten
            # on every replay; they only have to be in range for the rotary
            # embedding, and paired with PAD_SLOT_ID they touch no KV cache.
            group.add(
                batch_idx,
                list(range(rows_per_req)),
                [-1] * rows_per_req,
                rows_per_req,
            )
        return plan

    def _plan_cycle(self, input_batch: InputBatch) -> _CyclePlan:
        # Sized to the padded token count: with CUDA graphs the batch is padded
        # up to a captured size, and every per-token vector handed to the model
        # has to cover those padding rows too. Padding rows keep the defaults
        # here, so they resolve to a harmless cycle-buffer read.
        num_tokens = input_batch.num_tokens_after_padding
        plan = self._blank_plan(num_tokens)

        for batch_idx in range(input_batch.num_reqs):
            req_slot = int(input_batch.idx_mapping_np[batch_idx])
            start = int(input_batch.num_computed_tokens_np[batch_idx])
            query_len = int(input_batch.num_scheduled_tokens[batch_idx])
            query_start = int(input_batch.query_start_loc_np[batch_idx])
            query_end = query_start + query_len

            if bool(input_batch.is_prefilling_np[batch_idx]):
                base = plan.prefill.add(
                    batch_idx,
                    list(range(start, start + query_len)),
                    [-1] * query_len,
                    start + query_len,
                )
                plan.prefill_target_indices[query_start:query_end] = list(
                    range(base, base + query_len)
                )
                plan.prefill_orig_token_indices.extend(range(query_start, query_end))
                # Keep the Refresh contiguity cursor aligned across prefill so
                # the first decode step is not misread as a schedule gap.
                self._next_computed[req_slot] = start + query_len
                continue

            self._planner.plan_request(
                plan,
                batch_idx=batch_idx,
                req_slot=req_slot,
                start=start,
                query_len=query_len,
                query_start=query_start,
            )

        return plan

    def group_shape(self, group: _GroupPlan, kind: str) -> _GroupShape:
        """Decide the shape `group` runs at, and whether it can replay a graph.

        A group is eligible only when every participating request contributes
        exactly `DECODE_ROWS_PER_REQ[kind]` rows, i.e. when it is the uniform
        batch a captured graph was built for. That holds for every decode cycle
        head; prefill and multi-token decode groups stay eager.
        """
        num_rows = len(group.positions)
        if num_rows == 0:
            return _GroupShape(0, 0, False)
        num_reqs = len(group.req_batch_indices)
        rows_per_req = self.DECODE_ROWS_PER_REQ.get(kind)
        manager = self._parallel_graph_managers.get(kind)
        if (
            manager is not None
            and rows_per_req is not None
            and all(q == rows_per_req for q in group.query_lens)
        ):
            padded_reqs = manager.padded_num_reqs(num_reqs)
            if padded_reqs is not None:
                return _GroupShape(padded_reqs, padded_reqs * rows_per_req, True)
        return _GroupShape(num_reqs, num_rows, False)

    def _stage_group_vectors(self, plan: _CyclePlan, input_batch: InputBatch) -> None:
        """Stage every parallel group's per-row index vectors on the device.

        Runs before the metadata is built, because the slot mapping is derived on
        device from `positions` and the per-row request index.
        """
        vectors = self._vectors
        idx_mapping = input_batch.idx_mapping_np

        def expand(group: _GroupPlan, values: np.ndarray | None, pad_to: int) -> list:
            """One entry per row, carrying that row's group-local request index
            (or what `values` maps its batch index to), padded out to `pad_to`."""
            out: list[int] = []
            for local, (batch_idx, query_len) in enumerate(
                zip(group.req_batch_indices, group.query_lens)
            ):
                value = local if values is None else int(values[batch_idx])
                out.extend([value] * query_len)
            # Surplus rows belong to padded requests, whose seq_len is 0, so
            # these values only have to stay in range.
            out.extend([0] * (pad_to - len(out)))
            return out

        for kind in self.PARALLEL_GROUPS:
            group = plan.group(kind)
            shape = self.group_shape(group, kind)
            self._group_shapes[kind] = shape
            pad_to = shape.num_rows
            # Position 0 for the surplus rows: paired with PAD_SLOT_ID and
            # seq_len 0 they read and write nothing, so only the rotary
            # embedding sees the value and it just has to be in range.
            vectors.stage(
                f"{kind}_positions",
                group.positions + [0] * (pad_to - len(group.positions)),
            )
            vectors.stage(f"{kind}_group_req", expand(group, None, pad_to))
            vectors.stage(f"{kind}_req_idx", list(group.req_batch_indices))
            # Padding rows never write; their slot is PAD_SLOT_ID regardless.
            vectors.stage(
                f"{kind}_writes_kv",
                group.writes_kv + [False] * (pad_to - len(group.writes_kv)),
            )
            if kind != "prefill":
                abs_positions = group.token_abs_positions
                # -1 fills with MASK, which keeps the surplus ids valid.
                vectors.stage(
                    f"{kind}_abs",
                    abs_positions + [-1] * (pad_to - len(abs_positions)),
                )
                vectors.stage(f"{kind}_slots", expand(group, idx_mapping, pad_to))
        vectors.stage("prefill_src_tokens", plan.prefill_orig_token_indices)
        vectors.flush()

    def _prepare_one_group(
        self,
        group: _GroupPlan,
        kind: str,
        block_tables,
        kv_cache_config,
        attn_groups,
        for_capture: bool,
        shape: _GroupShape,
    ) -> tuple[dict[str, Any] | None, dict[str, torch.Tensor] | None]:
        """Build one group's attention state into its persistent buffers."""
        if not group.positions:
            return None, None
        if len(kv_cache_config.kv_cache_groups) != 1:
            raise ValueError("DMTDQwen3 requires one shared KV cache group.")

        buffers = self._group_buffers[kind]
        builder_idx = self.GROUP_BUILDER_IDX[kind]
        num_rows = len(group.positions)
        num_reqs = shape.num_reqs
        if shape.num_rows < num_rows or num_reqs < len(group.req_batch_indices):
            raise ValueError(
                f"DMTD {kind} group has {num_rows} rows over "
                f"{len(group.req_batch_indices)} requests, which does not fit "
                f"the padded shape {shape}."
            )

        buffers.stage_requests(
            query_lens=group.query_lens, seq_lens=group.seq_lens, shape=shape
        )
        vectors = self._vectors
        block_table = buffers.fill_block_table(
            block_tables[0], vectors.view(f"{kind}_req_idx"), num_reqs
        )

        positions = vectors.view(f"{kind}_positions")
        block_size = (
            attn_groups[0][0].get_metadata_builder(builder_idx).kv_cache_spec.block_size
        )
        slot_mappings = buffers.fill_slot_mapping(
            positions=positions,
            row_group_reqs=vectors.view(f"{kind}_group_req"),
            writes_kv=vectors.view(f"{kind}_writes_kv"),
            block_table=block_table,
            block_size=block_size,
            # A capture run must not touch the KV cache, so every row is
            # PAD_SLOT_ID -- the same thing `get_dummy_slot_mappings` does for
            # the runner's own capture.
            num_real_rows=0 if for_capture else num_rows,
            padded_rows=shape.num_rows,
        )

        # Capture with the worst-case sequence length so the graph stays valid at
        # any replay, mirroring DefaultModelState.prepare_attn.
        max_seq_len = self.max_model_len if for_capture else max(group.seq_lens)
        metadata_by_layer = build_attn_metadata(
            attn_groups=attn_groups,
            num_reqs=num_reqs,
            num_tokens=shape.num_rows,
            query_start_loc_gpu=buffers.query_start_loc[: num_reqs + 1],
            query_start_loc_cpu=buffers.query_start_loc_cpu[: num_reqs + 1],
            max_query_len=max(group.query_lens),
            seq_lens=buffers.seq_lens,
            max_seq_len=max_seq_len,
            block_tables=(block_table,),
            slot_mappings=slot_mappings,
            kv_cache_config=kv_cache_config,
            seq_lens_cpu_upper_bound=buffers.seq_lens_cpu,
            positions=positions,
            model_specific_attn_metadata=_ParallelIsPrefillingAttnMetadata(
                is_prefilling=buffers.is_prefilling,
            ),
            for_cudagraph_capture=for_capture or shape.graphed,
            causal=group.causal,
            metadata_builder_idx=builder_idx,
        )
        # No `make_owned_copy` here: each group owns a distinct metadata builder,
        # so the three sets built per step never alias each other's persistent
        # scratch, and keeping the builder's own buffers is what a captured graph
        # needs anyway.
        attn_metadata = {
            name: metadata_by_layer[name] for name in self._parallel_layer_names
        }
        return attn_metadata, build_slot_mappings_by_layer(
            slot_mappings, kv_cache_config
        )

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
        # CUDA graphs are only ever captured for the mid-cycle decode shape, so
        # capture must not consult (or mutate) live per-request cycle state.
        plan = (
            self._blank_plan(input_batch.num_tokens_after_padding)
            if for_capture
            else self._plan_cycle(input_batch)
        )
        self._cycle_plan = plan
        if cudagraph_mode == CUDAGraphMode.FULL and not for_capture:
            # The runner's graph covers the sequential layers and the assembly
            # in front of them, which reads from persistent tables and so is
            # shape-invariant. Prefill rows are the one thing it cannot absorb,
            # because their hiddens bypass the cycle buffer entirely.
            if plan.prefill.positions:
                raise RuntimeError(
                    "DMTD planned prefill parallel-layer work on a step "
                    f"dispatched to a full CUDA graph "
                    f"({len(plan.prefill.positions)} rows); requires_eager_step "
                    "and the cycle planners have diverged."
                )
        # `prepare_inputs` stages the group index vectors, but the slot mapping
        # below is derived from them on device, so they have to be on the GPU
        # before the metadata is built.
        self._stage_group_vectors(plan, input_batch)
        for kind in self.PARALLEL_GROUPS:
            metadata, slot_mapping = self._prepare_one_group(
                plan.group(kind),
                kind,
                block_tables,
                kv_cache_config,
                attn_groups,
                for_capture,
                shape=self._group_shapes[kind],
            )
            setattr(self, f"_{kind}_attn_metadata", metadata)
            setattr(self, f"_{kind}_slot_mapping", slot_mapping)
        # `standard_metadata` still carries batch-wide entries for the parallel
        # layers, but nothing reads them: those layers now run only under their
        # own group's forward context, never under the one built from this.
        return standard_metadata

    def _gather_prefill_ids(
        self,
        input_batch: InputBatch,
        src_token_indices: torch.Tensor,
        padded_rows: int,
    ) -> torch.Tensor:
        """Prefill rows are real tokens already present in this step's flattened
        ``input_ids``, so they are a straight gather by scheduled-token index."""
        num_rows = src_token_indices.numel()
        out = self._prefill_ids[:padded_rows]
        if num_rows:
            torch.index_select(
                input_batch.input_ids, 0, src_token_indices, out=out[:num_rows]
            )
        if padded_rows > num_rows:
            # Any valid id works for a row whose output nothing reads; MASK
            # keeps the padding deterministic.
            out[num_rows:].fill_(self.mask_token_id)
        return out

    def _gather_group_ids(
        self,
        req_states: RequestState,
        out_buffer: torch.Tensor,
        abs_positions: torch.Tensor,
        req_slots: torch.Tensor,
    ) -> torch.Tensor:
        """Fill one decode group's token ids: the real token at each row's
        absolute position, or MASK where the plan marked the row with -1.

        Done as a single flat gather plus a ``where`` so that no host-side loop
        and no per-row kernel launch is needed, and so the token values come
        from the GPU-resident history (a prerequisite for async scheduling,
        where the newest sampled token is not yet known on the host).
        """
        num_rows = abs_positions.numel()
        out = out_buffer[:num_rows]
        if not num_rows:
            return out
        all_ids = req_states.all_token_ids.gpu
        flat = req_slots * all_ids.shape[1] + abs_positions.clamp(min=0)
        gathered = all_ids.view(-1).index_select(0, flat)
        torch.where(
            abs_positions >= 0,
            gathered,
            gathered.new_full((), self.mask_token_id),
            out=out,
        )
        return out

    def _run_parallel_group_eager(
        self,
        *,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        row_base: int,
        num_rows: int,
        attn_metadata: dict[str, Any],
        slot_mapping: dict[str, torch.Tensor],
    ) -> None:
        """Run the parallel layers for one group, into its fixed row range.

        The group gets its own forward context rather than borrowing the
        sequential layers': its row count, attention metadata and `causal` flag
        are all its own, and this runs before the main forward is entered.
        """
        with set_forward_context(
            attn_metadata,
            self.vllm_config,
            num_tokens=num_rows,
            slot_mapping=slot_mapping,
        ):
            self._dmtd_model.run_parallel_group(
                input_ids,
                positions,
                out=self.parallel_hidden[row_base : row_base + num_rows],
            )

    def _group_row_base(self, kind: str) -> int:
        if kind == "prefill":
            return self.layout.prefill_base
        if kind == "causal":
            return self.layout.causal_base
        return self.layout.noncausal_base

    def _run_parallel_work(self, group_ids: dict[str, torch.Tensor]) -> None:
        """Run every parallel group this step needs, in group order.

        Each group either replays its own captured graph or runs eager; both
        write into the same fixed row range of the parallel-output table, so the
        sequential-layer forward that follows cannot tell the difference.
        """
        for kind in self.PARALLEL_GROUPS:
            shape = self._group_shapes[kind]
            if shape.num_rows == 0:
                continue
            metadata = getattr(self, f"_{kind}_attn_metadata")
            slot_mapping = getattr(self, f"_{kind}_slot_mapping")
            if metadata is None or slot_mapping is None:
                # Only reachable on a dummy run that skipped attention setup
                # entirely, where the output is discarded anyway.
                continue
            manager = self._parallel_graph_managers.get(kind)
            if shape.graphed and manager is not None:
                manager.replay(shape.num_reqs)
                continue
            self._run_parallel_group_eager(
                input_ids=group_ids[kind][: shape.num_rows],
                positions=self._vectors.view(f"{kind}_positions"),
                row_base=self._group_row_base(kind),
                num_rows=shape.num_rows,
                attn_metadata=metadata,
                slot_mapping=slot_mapping,
            )

    def prepare_inputs(
        self, input_batch: InputBatch, req_states: RequestState
    ) -> dict[str, Any]:
        model_inputs = super().prepare_inputs(input_batch, req_states)
        plan = self._cycle_plan
        if plan is None:
            # `prepare_attn` was skipped, which only happens on a dummy run with
            # no attention metadata at all. Plan no parallel work rather than
            # planning rows that then cannot be run.
            plan = self._blank_plan(input_batch.num_tokens_after_padding)
            self._prefill_slot_mapping = None
            self._causal_slot_mapping = None
            self._noncausal_slot_mapping = None
            self._prefill_attn_metadata = None
            self._causal_attn_metadata = None
            self._noncausal_attn_metadata = None
            self._stage_group_vectors(plan, input_batch)

        num_tokens = input_batch.num_tokens_after_padding
        if num_tokens > self._max_padded_tokens:
            raise ValueError(
                f"DMTD staging buffers hold {self._max_padded_tokens} tokens but "
                f"this step has {num_tokens} (after padding)."
            )

        (
            src_index,
            buf_index,
            add_embed,
            write_src_index,
            write_buf_index,
        ) = plan.resolve_gather_indices(self.layout, num_tokens)

        vectors = self._vectors
        vectors.stage("src_index", src_index)
        vectors.stage("buf_index", buf_index)
        vectors.stage("add_embed", add_embed)
        vectors.stage("write_src_index", write_src_index)
        vectors.stage("write_buf_index", write_buf_index)
        vectors.flush()

        group_ids = {
            "prefill": self._gather_prefill_ids(
                input_batch,
                vectors.view("prefill_src_tokens"),
                self._group_shapes["prefill"].num_rows,
            ),
            "causal": self._gather_group_ids(
                req_states,
                self._causal_ids,
                vectors.view("causal_abs"),
                vectors.view("causal_slots"),
            ),
            "noncausal": self._gather_group_ids(
                req_states,
                self._noncausal_ids,
                vectors.view("noncausal_abs"),
                vectors.view("noncausal_slots"),
            ),
        }

        # Run the parallel layers now, before the sequential-layer forward reads
        # their output. Keeping them out of that forward is what makes it
        # shape-invariant, and therefore capturable for every decode step.
        self._run_parallel_work(group_ids)

        model_inputs.update(
            dmtd_src_index=vectors.view("src_index"),
            dmtd_buf_index=vectors.view("buf_index"),
            dmtd_add_embed=vectors.view("add_embed"),
            dmtd_cycle_hidden=self.cycle_hidden,
            dmtd_parallel_hidden=self.parallel_hidden,
            dmtd_write_src_index=vectors.view("write_src_index"),
            dmtd_write_buf_index=vectors.view("write_buf_index"),
        )
        self._cycle_plan = None
        self._prefill_slot_mapping = None
        self._causal_slot_mapping = None
        self._noncausal_slot_mapping = None
        self._prefill_attn_metadata = None
        self._causal_attn_metadata = None
        self._noncausal_attn_metadata = None
        return model_inputs

    def prepare_dummy_inputs(self, num_reqs: int, num_tokens: int) -> dict[str, Any]:
        """Model kwargs for a dummy/capture run.

        Must expose exactly the keys `prepare_inputs` does, backed by the same
        persistent buffers, because a captured graph reads whatever addresses it
        saw here. The shape produced is the mid-cycle decode one (no
        parallel-layer groups), matching what `prepare_attn(for_capture=True)`
        plans.
        """
        model_inputs = super().prepare_dummy_inputs(num_reqs, num_tokens)
        if num_tokens > self._max_padded_tokens:
            raise ValueError(
                f"DMTD staging buffers hold {self._max_padded_tokens} tokens but "
                f"a dummy run asked for {num_tokens}."
            )

        vectors = self._vectors
        for kind in self.PARALLEL_GROUPS:
            self._group_shapes[kind] = _GroupShape(0, 0, False)
            vectors.stage(f"{kind}_positions", [])
            vectors.stage(f"{kind}_group_req", [])
            vectors.stage(f"{kind}_req_idx", [])
            vectors.stage(f"{kind}_writes_kv", [])
            if kind != "prefill":
                vectors.stage(f"{kind}_abs", [])
                vectors.stage(f"{kind}_slots", [])
        vectors.stage("prefill_src_tokens", [])
        vectors.stage("src_index", [-1] * num_tokens)
        # Values are irrelevant (a captured pass produces no usable output) but
        # must stay in range: the warmup pass before capture really executes,
        # and it must not touch a live request's buffered hidden either, hence
        # the scratch rows.
        vectors.stage(
            "buf_index",
            [self.layout.cycle_scratch_row + (i % self.tau) for i in range(num_tokens)],
        )
        vectors.stage("add_embed", [1] * num_tokens)
        self._stage_idle_writes()
        vectors.flush()

        model_inputs.update(
            dmtd_src_index=vectors.view("src_index"),
            dmtd_buf_index=vectors.view("buf_index"),
            dmtd_add_embed=vectors.view("add_embed"),
            dmtd_cycle_hidden=self.cycle_hidden,
            dmtd_parallel_hidden=self.parallel_hidden,
            dmtd_write_src_index=vectors.view("write_src_index"),
            dmtd_write_buf_index=vectors.view("write_buf_index"),
        )
        return model_inputs

    def _stage_idle_writes(self) -> None:
        """Stage a full-length cycle-buffer write vector that writes only scratch.

        The write vectors are always `write_capacity` long so that a captured
        graph copies a fixed number of rows; a dummy run has no real writes, so
        every entry is scratch.
        """
        layout = self.layout
        self._vectors.stage(
            "write_src_index", [layout.parallel_scratch_row] * layout.write_capacity
        )
        self._vectors.stage(
            "write_buf_index",
            [
                layout.cycle_scratch_row + (i % self.tau)
                for i in range(layout.write_capacity)
            ],
        )
