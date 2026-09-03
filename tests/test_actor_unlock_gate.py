from __future__ import annotations

import json
from pathlib import Path

import pytest

from forcesmolvla.rft.online.actor_unlock import (
    ActorUnlockPolicy,
    actor_unlock_is_ready,
)


def test_actor_stays_locked_without_manual_approval(tmp_path: Path) -> None:
    manifest = tmp_path / "actor_update_readiness.json"
    assert not actor_unlock_is_ready(
        manifest, actor_q_valid_ack_rows=1000, critic_only_updates=1000
    )


def test_actor_unlock_requires_common_thresholds_and_same_state_audit(
    tmp_path: Path,
) -> None:
    manifest = tmp_path / "actor_update_readiness.json"
    manifest.write_text(
        json.dumps(
            {
                "readiness_mode": "manual_approval",
                "approved": True,
                "actor_q_valid_ack_rows": 100,
                "critic_only_updates": 256,
                "same_state_ranking_audit": "same_state_critic_audit.json",
            }
        ),
        encoding="utf-8",
    )
    assert not actor_unlock_is_ready(
        manifest, actor_q_valid_ack_rows=99, critic_only_updates=256
    )
    assert actor_unlock_is_ready(
        manifest, actor_q_valid_ack_rows=100, critic_only_updates=256
    )


def test_automatic_readiness_does_not_require_manual_approval(
    tmp_path: Path,
) -> None:
    manifest = tmp_path / "actor_update_readiness.json"
    manifest.write_text(
        json.dumps(
            {
                "readiness_mode": "automatic_readiness",
                "actor_q_valid_ack_rows": 100,
                "critic_only_updates": 256,
                "same_state_ranking_audit": "same_state_critic_audit.json",
                "same_state_comparison_count": 20,
                "human_gt_policy_fraction": 0.60,
            }
        ),
        encoding="utf-8",
    )

    assert actor_unlock_is_ready(
        manifest,
        actor_q_valid_ack_rows=100,
        critic_only_updates=256,
        policy=ActorUnlockPolicy(mode="automatic_readiness"),
    )


def test_actor_readiness_modes_are_not_interchangeable(tmp_path: Path) -> None:
    manifest = tmp_path / "actor_update_readiness.json"
    manifest.write_text(
        json.dumps(
            {
                "readiness_mode": "automatic_readiness",
                "actor_q_valid_ack_rows": 100,
                "critic_only_updates": 256,
                "same_state_ranking_audit": "same_state_critic_audit.json",
                "same_state_comparison_count": 20,
                "human_gt_policy_fraction": 0.60,
            }
        ),
        encoding="utf-8",
    )

    assert not actor_unlock_is_ready(
        manifest, actor_q_valid_ack_rows=100, critic_only_updates=256
    )
    with pytest.raises(ValueError, match="FORCERFT_ACTOR_READINESS_POLICY_INVALID"):
        ActorUnlockPolicy(mode="invalid")  # type: ignore[arg-type]
