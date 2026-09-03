from pathlib import Path

import pytest

from forcesmolvla.rft.offline_transitions import (
    GAMMA,
    macro_transition_specs,
)


ROOT = Path(__file__).parents[1]


@pytest.mark.parametrize("synthetic_terminal", [3, 6, 9, 12])
def test_synthetic_macro_fixture_has_no_self_loop_or_off_by_one(synthetic_terminal):
    rows = macro_transition_specs(synthetic_terminal)

    assert all(
        row.anchor_frame_index < row.next_frame_index <= synthetic_terminal
        for row in rows
    )
    assert sum(row.terminated for row in rows) == 1
    assert rows[-1].next_frame_index == synthetic_terminal
    assert rows[-1].discount == 0.0
    for index, row in enumerate(rows):
        assert row.next_frame_index - row.anchor_frame_index == 3
        assert row.behavior_mask == (True, True, True)
        assert row.discount == GAMMA * row.bootstrap_mask
        next_return = 0.0 if index + 1 == len(rows) else rows[index + 1].mc_return
        assert row.mc_return == pytest.approx(row.reward + row.discount * next_return)


@pytest.mark.parametrize("unaligned_terminal", [1, 2, 4, 8, 10])
def test_synthetic_macro_fixture_preserves_partial_terminal_mask(unaligned_terminal):
    rows = macro_transition_specs(unaligned_terminal)
    final = rows[-1]
    executed = unaligned_terminal - final.anchor_frame_index
    assert final.next_frame_index == unaligned_terminal
    assert final.behavior_mask == tuple(slot < executed for slot in range(3))
    assert final.terminated is True
    assert final.bootstrap_mask == 0.0
    assert final.discount == 0.0
