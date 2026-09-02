from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from forcesmolvla.rft.online.production_bridge import (
    ProductionBridgeError,
    _formal_online_r_outcome,
    _pre_intervention_policy_boundary_decisions,
)
from forcesmolvla.rft.online import replay_training
from forcesmolvla.rft.online.replay_training import build_ack_macros


def _policy_row(
    decision: int,
    *,
    current_ns: int,
    ack_ns: int,
    action: float,
    chunk: str,
    generation: tuple[int, int, int] = (0, 0, 0),
    invalidated: bool = False,
) -> dict:
    policy_epoch, reset_generation, takeover_generation = generation
    def observation(timestamp_ns: int) -> dict:
        return {
            "materialized_timestamp_monotonic_ns": timestamp_ns,
            "clock_domain_id": "upper-host-monotonic",
            "state7_absolute": [0.0] * 7,
            "wrench6_calibrated_tcp": [0.0] * 6,
            "camera_external": {"blob_reference": "external.jpg"},
            "camera_wrist": {"blob_reference": "wrist.jpg"},
        }

    return {
        "identity": {
            "episode_id": "episode",
            "decision_id": decision,
            "source_ack_id": f"ack-{decision}",
            "transition_uid": f"uid-{decision}",
        },
        "generation": {
            "policy_epoch": policy_epoch,
            "reset_generation": reset_generation,
            "takeover_generation": takeover_generation,
        },
        "policy_lineage": {
            "proposal": {"invalidated_by_takeover": invalidated},
            "selection": {
                "sequence": decision,
                "chunk_id": chunk,
                "action_index": decision,
            },
        },
        "action_authority": {
            "accepted_absolute_action7": [action] * 6 + [0.0],
            "pose_ack": {
                "accepted": True,
                "upper_receive_monotonic_ns": ack_ns,
            },
            "gripper_terminal_provenance": {
                "origin_action_goal_id": f"gripper-{decision}"
            },
            "safety_arbitration": {"workspace_clipped": False},
        },
        "observation": observation(current_ns),
        "next_observation": observation(current_ns + 100_000_000),
        "outcome": {
            "reward": 0.0,
            "terminated": False,
            "truncated": False,
            "bootstrap_mask": 1.0,
            "discount": 0.99,
        },
    }


def test_production_ack_macros_use_strict_30hz_causal_zoh_and_100ms_horizon() -> None:
    rows = [
        _policy_row(
            0,
            current_ns=900_000_000,
            ack_ns=999_000_000,
            action=1.0,
            chunk="chunk-a",
        ),
        _policy_row(
            1,
            current_ns=1_000_000_000,
            ack_ns=1_050_000_000,
            action=2.0,
            chunk="chunk-b",
        ),
        _policy_row(
            2,
            current_ns=1_100_000_000,
            ack_ns=1_150_000_000,
            action=3.0,
            chunk="chunk-c",
        ),
        _policy_row(
            99,
            current_ns=1_000_000_000,
            ack_ns=1_020_000_000,
            action=99.0,
            chunk="invalidated-suffix",
            invalidated=True,
        ),
        _policy_row(
            3,
            current_ns=1_200_000_000,
            ack_ns=1_199_000_000,
            action=4.0,
            chunk="new-generation",
            generation=(1, 0, 1),
        ),
    ]

    macros = build_ack_macros(rows)
    assert all(
        item.transition["identity"]["decision_id"] != 99 for item in macros
    )
    macro = next(
        item
        for item in macros
        if item.transition["identity"]["decision_id"] == 1
    )
    assert macro.behavior.grid_monotonic_ns == (
        1_000_000_000,
        1_033_333_333,
        1_066_666_667,
    )
    assert macro.next_grid_monotonic_ns - macro.behavior.grid_monotonic_ns[0] == 100_000_000
    assert macro.behavior.ack_ids == ("ack-0", "ack-0", "ack-1")
    assert [item["chunk_id"] for item in macro.ack_provenance] == [
        "chunk-a",
        "chunk-a",
        "chunk-b",
    ]
    assert all(
        item["receive_monotonic_ns"] <= tick
        for item, tick in zip(
            macro.ack_provenance, macro.behavior.grid_monotonic_ns, strict=True
        )
    )
    assert not any(
        item["chunk_id"] in {"invalidated-suffix", "new-generation"}
        for item in macro.ack_provenance
    )


def test_each_takeover_marks_only_last_executed_old_generation_policy_row() -> None:
    sources = [
        {"receive_monotonic_ns": 900, "policy_epoch": 0, "reset_generation": 0, "takeover_generation": 0, "selection": {"sequence": 4}},
        {"receive_monotonic_ns": 950, "policy_epoch": 0, "reset_generation": 0, "takeover_generation": 0, "selection": {"sequence": 5}},
        {"receive_monotonic_ns": 1200, "policy_epoch": 1, "reset_generation": 0, "takeover_generation": 1, "selection": {"sequence": 6}},
        {"receive_monotonic_ns": 1250, "policy_epoch": 1, "reset_generation": 0, "takeover_generation": 1, "selection": {"sequence": 7}},
        {"receive_monotonic_ns": 1400, "policy_epoch": 2, "reset_generation": 0, "takeover_generation": 2, "selection": {"sequence": 8}},
    ]
    starts = [
        {"receive_monotonic_ns": 1000, "policy_epoch": 1, "reset_generation": 0, "takeover_generation": 1},
        {"receive_monotonic_ns": 1300, "policy_epoch": 2, "reset_generation": 0, "takeover_generation": 2},
        {"receive_monotonic_ns": 1500, "policy_epoch": 9, "reset_generation": 0, "takeover_generation": 9},
    ]

    assert _pre_intervention_policy_boundary_decisions(sources, starts) == {
        (0, 0, 0, 5),
        (1, 0, 1, 7),
    }
    truncated = _formal_online_r_outcome(
        terminal=False, truncated=True, terminal_observation_id="terminal"
    )
    assert truncated == {
        "reward": 0.0,
        "terminated": False,
        "truncated": True,
        "done": True,
        "bootstrap_mask": 0.0,
        "discount": 0.0,
        "operator_task_outcome": "success",
        "detector_outcome": "success",
        "terminal_observation_id": "terminal",
    }
    normal = _formal_online_r_outcome(
        terminal=False, truncated=False, terminal_observation_id="terminal"
    )
    assert normal["bootstrap_mask"] == 1.0 and normal["discount"] == 0.99
    with pytest.raises(ProductionBridgeError, match="TERMINAL_AND_TRUNCATED"):
        _formal_online_r_outcome(
            terminal=True, truncated=True, terminal_observation_id="terminal"
        )


def test_replay_materializer_and_batch_preserve_truncated(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rows = [
        _policy_row(
            0,
            current_ns=900_000_000,
            ack_ns=999_000_000,
            action=0.1,
            chunk="chunk-a",
        ),
        _policy_row(
            1,
            current_ns=1_000_000_000,
            ack_ns=1_099_000_000,
            action=0.2,
            chunk="chunk-b",
        ),
    ]
    rows[1]["outcome"].update(
        {"truncated": True, "bootstrap_mask": 0.0, "discount": 0.0}
    )
    macro = next(
        item
        for item in build_ack_macros(rows)
        if item.transition["identity"]["decision_id"] == 1
    )

    class IdentityNormalizer:
        @staticmethod
        def apply(value):
            return np.asarray(value)

    monkeypatch.setattr(
        replay_training,
        "_decode_path",
        lambda _path: np.zeros((3, 2, 2), dtype=np.uint8),
    )
    monkeypatch.setattr(
        "forcesmolvla.rft.batch.build_actor_batch",
        lambda _actor, samples, _device, include_action: {
            "sample_count": len(samples), "include_action": include_action
        },
    )
    normalizer = SimpleNamespace(
        state7=IdentityNormalizer(),
        wrench6=IdentityNormalizer(),
        delta_action7=IdentityNormalizer(),
    )
    replay = replay_training.FormalReplay(
        [macro], {"episode": tmp_path}, normalizer
    )
    sample = replay.materialize(0)
    assert sample["terminated"] is False
    assert sample["truncated"] is True
    assert sample["bootstrap"] is False
    assert sample["discount"] == 0.0

    batch = replay_training.build_batch(
        [sample], object(), torch.zeros(1), torch.device("cpu")
    )
    assert batch["truncated"].tolist() == [True]
    assert batch["bootstrap"].tolist() == [False]
