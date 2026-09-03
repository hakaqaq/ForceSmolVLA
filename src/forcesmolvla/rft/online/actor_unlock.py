"""Manual gate between online Critic-only warm-up and Actor updates."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path


@dataclass(frozen=True)
class ActorUnlockPolicy:
    minimum_actor_q_valid_ack_rows: int = 100
    minimum_critic_only_updates: int = 256


def actor_unlock_is_approved(
    approval_path: Path,
    *,
    actor_q_valid_ack_rows: int,
    critic_only_updates: int,
    policy: ActorUnlockPolicy = ActorUnlockPolicy(),
) -> bool:
    if (
        actor_q_valid_ack_rows < policy.minimum_actor_q_valid_ack_rows
        or critic_only_updates < policy.minimum_critic_only_updates
        or not approval_path.is_file()
    ):
        return False
    try:
        payload = json.loads(approval_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return bool(
        isinstance(payload, dict)
        and payload.get("approved") is True
        and int(payload.get("actor_q_valid_ack_rows", -1))
        >= policy.minimum_actor_q_valid_ack_rows
        and int(payload.get("actor_q_valid_ack_rows", -1))
        <= actor_q_valid_ack_rows
        and int(payload.get("critic_only_updates", -1))
        >= policy.minimum_critic_only_updates
        and int(payload.get("critic_only_updates", -1)) <= critic_only_updates
        and bool(str(payload.get("same_state_ranking_audit", "")).strip())
    )

