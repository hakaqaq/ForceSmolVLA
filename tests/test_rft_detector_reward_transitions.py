from pathlib import Path

import pytest

from forcesmolvla.rft.detector_reward_transitions import (
    causal_detection_trace,
    detector_macro_transitions,
    self_check,
)
from forcesmolvla.rft.manual_reward_transitions import load_training_transitions as load_manual


ROOT = Path(__file__).parents[1]


def test_detector_trigger_is_current_fifth_frame_not_streak_start():
    trace = causal_detection_trace(range(10), [0.1] * 3 + [0.9] * 7, [True] * 10)
    assert trace.streak_start_frame == 3
    assert trace.trigger_frame == 7
    assert trace.trigger_frame != trace.streak_start_frame


def test_detector_resets_on_gap_and_invalid_frame():
    assert causal_detection_trace([0, 1, 3, 4, 5, 6, 7], [0.9] * 7, [True] * 7).trigger_frame == 7
    assert causal_detection_trace(range(7), [0.9] * 7, [True, True, False, True, True, True, True]).trigger_frame is None


@pytest.mark.parametrize("terminal,steps", [(4, [3, 1]), (5, [3, 2]), (6, [3, 3])])
def test_detector_macro_transition_terminal_and_partial_mask(terminal, steps):
    rows = detector_macro_transitions(terminal)
    assert [row.executed_steps for row in rows] == steps
    assert sum(row.reward == 1.0 for row in rows) == 1
    assert sum(row.terminated for row in rows) == 1
    assert all(row.anchor_frame < row.next_frame <= terminal for row in rows)


def test_manual_g1_loader_is_fail_closed():
    with pytest.raises(RuntimeError, match="HISTORICAL_MANUAL_AUDIT"):
        load_manual(ROOT / "artifacts/development/stage2/g1_manual_reward_transition_view.v1")


def test_new_sources_have_no_manual_boundary_dependency():
    paths = [
        ROOT / "src/forcesmolvla/rft/detector_reward_transitions.py",
        ROOT / "configs/stage2_g1_frozen_detector_transition_view.development.json",
    ]
    forbidden = ("first_confident_complete_frame", "task2_reward_frame_labels", "reviewed.json")
    for path in paths:
        text = path.read_text(encoding="utf-8")
        assert not any(token in text for token in forbidden)
    self_check()
