# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from .oracle import (
    expected_shadow_cache_length,
    full_recompute_greedy,
    make_shadow_input_ids,
)


class PositionLogitModel:
    """Tiny deterministic model used to test oracle control flow."""

    def __init__(self, vocab_size: int):
        self.vocab_size = vocab_size

    def __call__(self, *, input_ids, position_ids, use_cache, return_dict):
        assert use_cache is False
        assert return_dict is True
        next_ids = (input_ids + position_ids + 1).remainder(self.vocab_size)
        logits = torch.full(
            (*input_ids.shape, self.vocab_size),
            -1000.0,
            device=input_ids.device,
        )
        logits.scatter_(2, next_ids.unsqueeze(-1), 0.0)
        return SimpleNamespace(logits=logits)


def test_make_shadow_input_ids_keeps_only_cycle_heads():
    input_ids = torch.tensor([[10, 11, 12, 13, 14, 15]])
    positions = torch.arange(6).unsqueeze(0)

    shadow = make_shadow_input_ids(
        input_ids,
        positions,
        cycle_length=4,
        mask_token_id=99,
    )

    assert shadow.tolist() == [[10, 99, 99, 99, 14, 99]]


def test_full_recompute_greedy_replays_complete_history():
    tokens, traces = full_recompute_greedy(
        PositionLogitModel(vocab_size=32),
        torch.tensor([[3, 4]]),
        max_new_tokens=3,
        cycle_length=4,
        mask_token_id=31,
    )

    assert tokens.tolist() == [[3, 4, 6, 9, 13]]
    assert [trace.input_ids.shape[1] for trace in traces] == [2, 3, 4]
    assert traces[-1].shadow_input_ids.tolist() == [[3, 31, 31, 31]]


def test_bidirectional_oracle_pads_current_cycle_but_reads_real_position():
    tokens, traces = full_recompute_greedy(
        PositionLogitModel(vocab_size=32),
        torch.tensor([[3]]),
        max_new_tokens=1,
        cycle_length=4,
        mask_token_id=31,
        pad_bidirectional_cycle=True,
    )

    assert tokens.tolist() == [[3, 4]]
    assert traces[0].input_ids.tolist() == [[3]]
    assert traces[0].shadow_input_ids.tolist() == [[3, 31, 31, 31]]


@pytest.mark.parametrize(
    ("num_real_tokens", "lookahead", "expected"),
    [
        (0, False, 0),
        (3, False, 3),
        (1, True, 4),
        (4, True, 4),
        (5, True, 8),
    ],
)
def test_expected_shadow_cache_length(num_real_tokens, lookahead, expected):
    assert (
        expected_shadow_cache_length(
            num_real_tokens,
            cycle_length=4,
            has_cycle_lookahead=lookahead,
        )
        == expected
    )
