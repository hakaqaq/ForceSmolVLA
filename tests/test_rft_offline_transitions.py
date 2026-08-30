import json
from pathlib import Path
import subprocess
import sys

import pytest

from forcesmolvla.rft.offline_transitions import (
    GAMMA,
    macro_transition_specs,
    validate_outcome_labels,
    validate_reward_spec,
)


ROOT = Path(__file__).parents[1]


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_real_g1_inputs_are_explicitly_unapproved_and_generation_is_blocked():
    labels = _json(ROOT / "labels/task2_episode_outcomes.v1.json")
    reward = _json(ROOT / "configs/stage2_reward_spec.development.yaml")

    validate_reward_spec(reward)
    assert reward["real_g1_generation_permitted"] is False
    assert labels["approval_status"] == "unapproved"
    assert labels["generation_permitted"] is False
    assert labels["episodes"] == []
    assert labels["terminal_inference"] == "forbidden"
    assert "last_valid_converted_frame" in labels["forbidden_terminal_rules"]


def test_unapproved_placeholder_cannot_pass_external_reward_label_validation():
    with pytest.raises(RuntimeError, match="G1_EXTERNAL_REWARD_LABELS_NOT_FROZEN"):
        validate_outcome_labels(
            _json(ROOT / "labels/task2_episode_outcomes.v1.json"),
            conversion_episodes=[],
            episode_lengths={},
        )


def test_real_builder_fails_closed_before_creating_output(tmp_path):
    output = tmp_path / "forbidden-real-g1-output"
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools/materialize_offline_demo_replay.py"),
            "--reward-labels",
            str(ROOT / "labels/task2_episode_outcomes.v1.json"),
            "--output-root",
            str(output),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "G1_REAL_BUILD_BLOCKED" in result.stderr
    assert not output.exists()


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
        assert row.discount == GAMMA * row.bootstrap_mask
        next_return = 0.0 if index + 1 == len(rows) else rows[index + 1].mc_return
        assert row.mc_return == pytest.approx(row.reward + row.discount * next_return)


@pytest.mark.parametrize("unaligned_terminal", [1, 2, 4, 8, 10])
def test_synthetic_macro_fixture_rejects_partial_terminal_actions(unaligned_terminal):
    with pytest.raises(ValueError, match="K=3|10 Hz"):
        macro_transition_specs(unaligned_terminal)
