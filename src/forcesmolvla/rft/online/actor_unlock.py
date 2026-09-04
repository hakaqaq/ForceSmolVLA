"""Readiness gate between online Critic-only warm-up and Actor updates."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Literal


ActorReadinessMode = Literal[
    "offline_critic_ready",
    "manual_approval",
    "automatic_readiness",
]


@dataclass(frozen=True)
class ActorUnlockPolicy:
    minimum_actor_q_valid_ack_rows: int = 100
    minimum_critic_only_updates: int = 0
    mode: ActorReadinessMode = "offline_critic_ready"
    minimum_same_state_comparisons: int = 20
    minimum_human_gt_policy_fraction: float = 0.60

    def __post_init__(self) -> None:
        if (
            self.minimum_actor_q_valid_ack_rows < 0
            or self.minimum_critic_only_updates < 0
            or self.minimum_same_state_comparisons < 1
            or not 0.0 <= self.minimum_human_gt_policy_fraction <= 1.0
            or self.mode not in {
                "offline_critic_ready",
                "manual_approval",
                "automatic_readiness",
            }
        ):
            raise ValueError("FORCERFT_ACTOR_READINESS_POLICY_INVALID")


def actor_unlock_is_ready(
    readiness_manifest: Path,
    *,
    actor_q_valid_ack_rows: int,
    critic_only_updates: int,
    policy: ActorUnlockPolicy = ActorUnlockPolicy(),
) -> bool:
    if (
        actor_q_valid_ack_rows < policy.minimum_actor_q_valid_ack_rows
        or critic_only_updates < policy.minimum_critic_only_updates
    ):
        return False
    if policy.mode == "offline_critic_ready":
        return True
    if not readiness_manifest.is_file():
        return False
    try:
        payload = json.loads(readiness_manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if not isinstance(payload, dict):
        return False
    if payload.get("readiness_mode") != policy.mode:
        return False
    if policy.mode == "manual_approval":
        return bool(
            payload.get("approved") is True
            and str(payload.get("same_state_ranking_audit", "")).strip()
        )
    try:
        comparison_count = int(payload.get("comparison_count", -1))
        human_gt_policy = float(payload.get("human_gt_policy_fraction", -1.0))
    except (TypeError, ValueError):
        return False
    return bool(
        payload.get("same_observation_required") is True
        and comparison_count >= policy.minimum_same_state_comparisons
        and human_gt_policy >= policy.minimum_human_gt_policy_fraction
    )
