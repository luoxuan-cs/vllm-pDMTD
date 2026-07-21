# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Reference helpers for Causal-Parallel-Norefresh / Refresh tests.

Norefresh: the checkpoint's cached Transformers generation path is not a valid
oracle. Recomputing the complete causal shadow sequence is equivalent to
incremental shadow-KV execution and is intentionally slow but unambiguous.

Refresh: persistent parallel-layer KV holds only refreshed real tokens
(one-cycle lag). Current-cycle shadow slots are temporary for attention and
never enter the logical parallel-layer history. See
parallel-eval/models/README.md §7.

For Causal-Parallel-Refresh, teacher-forced ``forward(use_cache=False)`` over
the full real sequence is a useful 4B auxiliary oracle: training-time real/shadow
visibility matches incremental refresh+shadow for causal block attention. Stock
``generate(use_cache=True)`` is still invalid. Prefer ``RefreshCycleOracle`` for
cache/visibility invariants; use full-recompute logits for token parity.
"""

from dataclasses import dataclass, field
from typing import Protocol

import torch


class FullForwardModel(Protocol):
    def __call__(self, **kwargs): ...


@dataclass
class FullRecomputeStep:
    input_ids: torch.Tensor
    shadow_input_ids: torch.Tensor
    position_ids: torch.Tensor
    logits: torch.Tensor
    next_token_ids: torch.Tensor


def make_shadow_input_ids(
    input_ids: torch.Tensor,
    position_ids: torch.Tensor,
    *,
    cycle_length: int,
    mask_token_id: int,
) -> torch.Tensor:
    """Replace non-cycle-head tokens with the model's shadow MASK token."""
    if input_ids.shape != position_ids.shape:
        raise ValueError(
            "input_ids and position_ids must have the same shape, got "
            f"{tuple(input_ids.shape)} and {tuple(position_ids.shape)}"
        )
    if cycle_length < 1:
        raise ValueError(f"cycle_length must be positive, got {cycle_length}")
    mask = torch.full_like(input_ids, mask_token_id)
    return torch.where(position_ids.remainder(cycle_length).eq(0), input_ids, mask)


@torch.inference_mode()
def full_recompute_greedy(
    model: FullForwardModel,
    input_ids: torch.Tensor,
    *,
    max_new_tokens: int,
    cycle_length: int,
    mask_token_id: int,
    eos_token_ids: set[int] | None = None,
    pad_bidirectional_cycle: bool = False,
) -> tuple[torch.Tensor, list[FullRecomputeStep]]:
    """Generate greedily by recomputing the complete sequence every step.

    Bidirectional shadow attention needs the complete current cycle even when
    only its real prefix is known. ``pad_bidirectional_cycle`` appends MASK IDs
    through the cycle boundary, runs the full teacher-forced forward, and reads
    logits at the last real position. The sequential layers remain causal, so
    padded future sequential-layer positions cannot affect that result.
    """
    if input_ids.ndim != 2:
        raise ValueError(f"input_ids must be rank 2, got rank {input_ids.ndim}")
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be non-negative")

    tokens = input_ids.clone()
    batch_size = tokens.shape[0]
    finished = torch.zeros(batch_size, dtype=torch.bool, device=tokens.device)
    traces: list[FullRecomputeStep] = []
    eos = eos_token_ids or set()

    for _ in range(max_new_tokens):
        forward_tokens = tokens
        if pad_bidirectional_cycle:
            pad_len = (-tokens.shape[1]) % cycle_length
            if pad_len:
                forward_tokens = torch.cat(
                    (
                        tokens,
                        torch.full(
                            (batch_size, pad_len),
                            mask_token_id,
                            dtype=tokens.dtype,
                            device=tokens.device,
                        ),
                    ),
                    dim=1,
                )
        positions = torch.arange(
            forward_tokens.shape[1], device=tokens.device
        ).expand(
            batch_size, -1
        )
        outputs = model(
            input_ids=forward_tokens,
            position_ids=positions,
            use_cache=False,
            return_dict=True,
        )
        logits = outputs.logits[:, tokens.shape[1] - 1].float()
        next_token_ids = logits.argmax(dim=-1)
        if eos:
            next_token_ids = torch.where(
                finished,
                torch.full_like(next_token_ids, min(eos)),
                next_token_ids,
            )

        traces.append(
            FullRecomputeStep(
                input_ids=tokens.clone(),
                shadow_input_ids=make_shadow_input_ids(
                    forward_tokens,
                    positions,
                    cycle_length=cycle_length,
                    mask_token_id=mask_token_id,
                ),
                position_ids=positions.clone(),
                logits=logits.clone(),
                next_token_ids=next_token_ids.clone(),
            )
        )
        tokens = torch.cat((tokens, next_token_ids[:, None]), dim=1)
        if eos:
            is_eos = torch.zeros_like(finished)
            for token_id in eos:
                is_eos.logical_or_(next_token_ids.eq(token_id))
            finished.logical_or_(is_eos)
            if bool(finished.all()):
                break

    return tokens, traces


def expected_shadow_cache_length(
    num_real_tokens: int,
    *,
    cycle_length: int,
    has_cycle_lookahead: bool,
) -> int:
    """Return shadow slots written after a prefill/decode boundary."""
    if num_real_tokens < 0:
        raise ValueError("num_real_tokens must be non-negative")
    if not has_cycle_lookahead or num_real_tokens == 0:
        return num_real_tokens
    return ((num_real_tokens + cycle_length - 1) // cycle_length) * cycle_length


def expected_parallel_real_cache_length(
    num_real_tokens: int,
    cycle_length: int,
    *,
    before_refresh: bool = False,
) -> int:
    """Logical parallel-layer real-KV length for Causal-Parallel-Refresh.

    Persistent parallel-layer cache holds only completed cycles that have
    already been refreshed. Mid-cycle real tokens are not yet in that cache.

    After cycle-head refresh (the usual post-step view)::

        floor(num_real_tokens / cycle_length) * cycle_length

    Examples with ``cycle_length=4`` after each real token is known and any
    due refresh has run::

        num_real:  0 1 2 3 4 5 6 7 8 ...
        parallel_len:   0 0 0 0 4 4 4 4 8 ...

    Lag at a cycle head **before** refresh of the previous cycle: the
    just-finished cycle is still absent. Pass ``before_refresh=True`` when
    ``num_real_tokens`` is a positive multiple of ``cycle_length`` to model
    that instant (e.g. 8 real tokens before refreshing 4..7 → length 4).
    Mid-cycle, ``before_refresh`` has no effect.
    """
    if num_real_tokens < 0:
        raise ValueError("num_real_tokens must be non-negative")
    if cycle_length < 1:
        raise ValueError(f"cycle_length must be positive, got {cycle_length}")

    completed = (num_real_tokens // cycle_length) * cycle_length
    if (
        before_refresh
        and num_real_tokens > 0
        and num_real_tokens % cycle_length == 0
    ):
        return completed - cycle_length
    return completed


@dataclass
class RefreshStepTrace:
    """Control-flow record for one Refresh real-token sequential-layer step
    (or cycle open)."""

    position: int
    cycle_base: int
    ran_refresh: bool
    refresh_input_ids: list[int]
    refresh_positions: list[int]
    shadow_input_ids: list[int]
    shadow_positions: list[int]
    sequential_input_ids: list[int]
    sequential_positions: list[int]
    # Logical persistent parallel-layer real length after this step's
    # parallel-layer work.
    parallel_real_len: int
    # cycle_hidden slots written this cycle (always 4 after a parallel-layer
    # shadow pass).
    cycle_hidden_slots: int = 4
    notes: str = ""


@dataclass
class RefreshCycleOracle:
    """Explicit Causal-Parallel-Refresh cycle state machine (control-flow only).

    Simulates logical parallel-layer real cache length, temporary shadow
    block layout, first-cycle 4-slot-only parallel-layer forward,
    steady-state refresh-previous-4 then shadow ``[head, M, M, M]``, and
    per-token sequential-layer step as ``embed(real)+shadow``.

    This does not run model weights; traces are for scheduler/state invariants
    and for matching a future tiny FP32 / teacher-forced reference.
    """

    cycle_length: int = 4
    mask_token_id: int = 151660
    parallel_real_len: int = 0
    cycle_base: int = -1
    cycle_hidden_valid: list[bool] = field(default_factory=lambda: [False] * 4)
    real_token_ids: list[int] = field(default_factory=list)
    traces: list[RefreshStepTrace] = field(default_factory=list)
    finished: bool = False

    def reset(self) -> None:
        self.parallel_real_len = 0
        self.cycle_base = -1
        self.cycle_hidden_valid = [False] * self.cycle_length
        self.real_token_ids = []
        self.traces = []
        self.finished = False

    def _shadow_block_ids(self, head_token_id: int) -> list[int]:
        return [
            head_token_id,
            *[self.mask_token_id] * (self.cycle_length - 1),
        ]

    def _shadow_block_positions(self, cycle_base: int) -> list[int]:
        return list(range(cycle_base, cycle_base + self.cycle_length))

    def open_cycle(self, head_token_id: int) -> RefreshStepTrace:
        """Run the parallel layers at a cycle head: optional refresh of the
        previous cycle + shadow4."""
        if self.finished:
            raise RuntimeError("oracle already finished (EOS)")
        position = len(self.real_token_ids)
        if position % self.cycle_length != 0:
            raise ValueError(
                f"open_cycle requires a cycle head position, got {position}"
            )

        cycle_base = position
        ran_refresh = position > 0
        refresh_ids: list[int] = []
        refresh_pos: list[int] = []
        if ran_refresh:
            # Lag: previous cycle's real tokens are known but not yet in the
            # persistent parallel-layer cache.
            assert self.parallel_real_len == expected_parallel_real_cache_length(
                position, self.cycle_length, before_refresh=True
            )
            prev_base = position - self.cycle_length
            refresh_ids = list(self.real_token_ids[prev_base:position])
            refresh_pos = list(range(prev_base, position))
            self.parallel_real_len = position

        shadow_ids = self._shadow_block_ids(head_token_id)
        shadow_pos = self._shadow_block_positions(cycle_base)
        self.cycle_base = cycle_base
        self.cycle_hidden_valid = [True] * self.cycle_length

        # First cycle: parallel-layer forward is shadow-only (4 slots).
        # Steady: refresh4 + shadow4.
        notes = (
            "first cycle: shadow4 only"
            if not ran_refresh
            else "steady: refresh previous 4 then shadow [head,M,M,M]"
        )
        trace = RefreshStepTrace(
            position=position,
            cycle_base=cycle_base,
            ran_refresh=ran_refresh,
            refresh_input_ids=refresh_ids,
            refresh_positions=refresh_pos,
            shadow_input_ids=shadow_ids,
            shadow_positions=shadow_pos,
            sequential_input_ids=[],
            sequential_positions=[],
            parallel_real_len=self.parallel_real_len,
            cycle_hidden_slots=self.cycle_length,
            notes=notes,
        )
        self.traces.append(trace)
        return trace

    def sequential_step(
        self,
        real_token_id: int,
        *,
        eos_token_ids: set[int] | None = None,
    ) -> RefreshStepTrace:
        """Consume one real token through the sequential layers using the
        corresponding shadow slot.

        Call ``open_cycle`` first when ``len(real_token_ids) % cycle_length == 0``.
        Appends ``real_token_id`` to history. Does not advance ``parallel_real_len``
        (refresh happens only at the next cycle head).
        """
        if self.finished:
            raise RuntimeError("oracle already finished (EOS)")
        position = len(self.real_token_ids)
        cycle_base = position - position % self.cycle_length
        phase = position % self.cycle_length

        if position % self.cycle_length == 0:
            # Auto-open if caller did not; keeps step-by-step APIs ergonomic.
            if self.cycle_base != cycle_base or not self.cycle_hidden_valid[0]:
                self.open_cycle(real_token_id)

        if self.cycle_base != cycle_base or not self.cycle_hidden_valid[phase]:
            raise RuntimeError(
                f"missing cycle_hidden for position {position} "
                f"(cycle_base={self.cycle_base}, valid={self.cycle_hidden_valid})"
            )

        # Abstract sequential-layer step: embed(real) + cycle_hidden[phase];
        # no weights.
        self.real_token_ids.append(real_token_id)
        n = len(self.real_token_ids)
        # At a completed-cycle boundary the previous cycle is still lagged
        # until the next open_cycle refresh.
        at_boundary = n > 0 and n % self.cycle_length == 0
        assert self.parallel_real_len == expected_parallel_real_cache_length(
            n, self.cycle_length, before_refresh=at_boundary
        )

        parallel_after = self.parallel_real_len
        trace = RefreshStepTrace(
            position=position,
            cycle_base=cycle_base,
            ran_refresh=False,
            refresh_input_ids=[],
            refresh_positions=[],
            shadow_input_ids=[],
            shadow_positions=[],
            sequential_input_ids=[real_token_id],
            sequential_positions=[position],
            parallel_real_len=parallel_after,
            cycle_hidden_slots=self.cycle_length,
            notes="sequential-layer: embed(real)+shadow_hidden[phase]",
        )
        self.traces.append(trace)

        eos = eos_token_ids or set()
        if real_token_id in eos:
            self.finished = True
            # Incomplete cycle is left unrefreshed: parallel_real_len unchanged.
            for i in range(phase + 1, self.cycle_length):
                self.cycle_hidden_valid[i] = False
        return trace

    def run_tokens(
        self,
        token_ids: list[int],
        *,
        eos_token_ids: set[int] | None = None,
    ) -> list[RefreshStepTrace]:
        """Teacher-forced walk over known real tokens (prompt or rollout)."""
        out: list[RefreshStepTrace] = []
        for token_id in token_ids:
            if self.finished:
                break
            out.append(self.sequential_step(token_id, eos_token_ids=eos_token_ids))
        return out

    def parallel_real_len_sequence(self, num_real_tokens: int) -> list[int]:
        """Expected logical parallel-layer lengths for ``num_real_tokens`` in
        ``0..N``."""
        return [
            expected_parallel_real_cache_length(n, self.cycle_length)
            for n in range(num_real_tokens + 1)
        ]
