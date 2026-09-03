from __future__ import annotations

import json
from pathlib import Path

from forcesmolvla.rft.online.actor_unlock import actor_unlock_is_approved


def test_actor_stays_locked_without_manual_approval(tmp_path: Path) -> None:
    approval = tmp_path / "actor_q_unlock.json"
    assert not actor_unlock_is_approved(
        approval, actor_q_valid_ack_rows=1000, critic_only_updates=1000
    )


def test_actor_unlock_requires_common_thresholds_and_same_state_audit(
    tmp_path: Path,
) -> None:
    approval = tmp_path / "actor_q_unlock.json"
    approval.write_text(
        json.dumps(
            {
                "approved": True,
                "actor_q_valid_ack_rows": 100,
                "critic_only_updates": 256,
                "same_state_ranking_audit": "same_state_critic_audit.json",
            }
        ),
        encoding="utf-8",
    )
    assert not actor_unlock_is_approved(
        approval, actor_q_valid_ack_rows=99, critic_only_updates=256
    )
    assert actor_unlock_is_approved(
        approval, actor_q_valid_ack_rows=100, critic_only_updates=256
    )

