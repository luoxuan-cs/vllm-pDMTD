# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from .oracle import RefreshCycleOracle, expected_parallel_real_cache_length


@pytest.mark.parametrize(
    ("num_real_tokens", "before_refresh", "expected"),
    [
        (0, False, 0),
        (1, False, 0),
        (3, False, 0),
        (4, False, 4),
        (4, True, 0),
        (5, False, 4),
        (7, False, 4),
        (8, False, 8),
        (8, True, 4),
        (9, False, 8),
    ],
)
def test_expected_parallel_real_cache_length(num_real_tokens, before_refresh, expected):
    assert (
        expected_parallel_real_cache_length(
            num_real_tokens, 4, before_refresh=before_refresh
        )
        == expected
    )


def test_cache_length_sequence_matches_readme_lag():
    oracle = RefreshCycleOracle(cycle_length=4, mask_token_id=99)
    assert oracle.parallel_real_len_sequence(8) == [0, 0, 0, 0, 4, 4, 4, 4, 8]


def test_first_cycle_no_refresh():
    oracle = RefreshCycleOracle(cycle_length=4, mask_token_id=99)
    open_trace = oracle.open_cycle(10)

    assert open_trace.ran_refresh is False
    assert open_trace.refresh_input_ids == []
    assert open_trace.shadow_input_ids == [10, 99, 99, 99]
    assert open_trace.shadow_positions == [0, 1, 2, 3]
    assert open_trace.parallel_real_len == 0
    assert "first cycle" in open_trace.notes

    oracle.sequential_step(10)
    oracle.sequential_step(11)
    oracle.sequential_step(12)
    oracle.sequential_step(13)
    assert oracle.parallel_real_len == 0
    assert oracle.real_token_ids == [10, 11, 12, 13]


def test_position_4_refreshes_0_to_3_then_shadow():
    oracle = RefreshCycleOracle(cycle_length=4, mask_token_id=99)
    oracle.run_tokens([10, 11, 12, 13])
    assert oracle.parallel_real_len == 0

    open_trace = oracle.open_cycle(14)
    assert open_trace.ran_refresh is True
    assert open_trace.refresh_input_ids == [10, 11, 12, 13]
    assert open_trace.refresh_positions == [0, 1, 2, 3]
    assert open_trace.shadow_input_ids == [14, 99, 99, 99]
    assert open_trace.shadow_positions == [4, 5, 6, 7]
    assert open_trace.parallel_real_len == 4
    assert "steady" in open_trace.notes

    oracle.sequential_step(14)
    assert oracle.parallel_real_len == 4


def test_run_tokens_cache_length_progression():
    """Oracle state vs closed-form length across the first two cycles.

    After the sequential layers finish a cycle (n % 4 == 0) the previous cycle is still lagged
    until the next ``open_cycle``. The closed-form
    ``floor(n/4)*4`` is the post-refresh view used once that head runs.
    """
    oracle = RefreshCycleOracle(cycle_length=4, mask_token_id=99)
    tokens = [100 + i for i in range(9)]
    oracle_lengths = [0]
    for token_id in tokens:
        oracle.sequential_step(token_id)
        oracle_lengths.append(oracle.parallel_real_len)

    # n=0..9 oracle lengths after each append (lag at n=4 and n=8).
    assert oracle_lengths == [0, 0, 0, 0, 0, 4, 4, 4, 4, 8]
    # Closed-form post-refresh view for n=0..8.
    assert oracle.parallel_real_len_sequence(8) == [0, 0, 0, 0, 4, 4, 4, 4, 8]
    # Explicit lag at cycle heads before refresh.
    assert expected_parallel_real_cache_length(4, 4, before_refresh=True) == 0
    assert expected_parallel_real_cache_length(8, 4, before_refresh=True) == 4


def test_mid_cycle_eos_leaves_incomplete_cycle_unrefreshed():
    oracle = RefreshCycleOracle(cycle_length=4, mask_token_id=99)
    oracle.run_tokens([10, 11, 12, 13, 14, 15], eos_token_ids={15})

    assert oracle.finished is True
    assert oracle.real_token_ids == [10, 11, 12, 13, 14, 15]
    # Steady refresh at position 4 brought 0..3 into cache; tokens 4..5 are
    # in the incomplete cycle and must not be refreshed.
    assert oracle.parallel_real_len == 4
    assert oracle.cycle_hidden_valid == [True, True, False, False]
    assert expected_parallel_real_cache_length(6, 4) == 4
