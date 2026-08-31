"""Frozen online runtime contracts and cross-file consistency checks."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping

from jsonschema import Draft202012Validator
import yaml


ROOT = Path(__file__).parents[4]
CONFIG_PATHS = {
    "trainability": ROOT / "configs/stage3_trainability_contract.v1.development.json",
    "transition": ROOT / "configs/stage3_transition_contract.v1.development.json",
    "replay": ROOT / "configs/stage3_replay_contract.v1.development.yaml",
    "reward_terminal": ROOT / "configs/stage3_reward_terminal_contract.v1.development.json",
    "online_hil": ROOT / "configs/stage3_online_hil.v1.development.yaml",
    "publication": ROOT / "configs/stage3_policy_publication.v1.development.json",
}
SCHEMA_PATHS = {
    "transition": ROOT / "schemas/stage3_ack_transition.v1.schema.json",
    "checkpoint": ROOT / "schemas/stage3_online_checkpoint.v1.schema.json",
}


@dataclass(frozen=True)
class CriticReadiness:
    critic_ready: bool = False
    actor_q_guidance_enabled: bool = False
    unlock_requires_explicit_approval: bool = True

    def validate(self) -> "CriticReadiness":
        if self.actor_q_guidance_enabled and not self.critic_ready:
            raise ValueError("ONLINE_REPLAY_ACTOR_Q_ENABLED_BEFORE_CRITIC_READY")
        if not self.unlock_requires_explicit_approval:
            raise ValueError("ONLINE_REPLAY_CRITIC_UNLOCK_MUST_REQUIRE_EXPLICIT_APPROVAL")
        return self


def _load(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    value = json.loads(text) if path.suffix == ".json" else yaml.safe_load(text)
    if not isinstance(value, dict):
        raise ValueError(f"ONLINE_REPLAY_CONTRACT_ROOT_NOT_MAPPING:{path.name}")
    return value


def load_online_contracts() -> dict[str, dict[str, Any]]:
    return {name: _load(path) for name, path in CONFIG_PATHS.items()}


def validate_online_contracts(
    contracts: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    values = dict(contracts or load_online_contracts())
    if set(values) != set(CONFIG_PATHS):
        raise ValueError("ONLINE_REPLAY_CONTRACT_SET_INCOMPLETE")
    trainability = values["trainability"]
    transition = values["transition"]
    replay = values["replay"]
    reward = values["reward_terminal"]
    runtime = values["online_hil"]

    if trainability["parent_binding"] != "PENDING":
        raise ValueError("ONLINE_REPLAY_G0_PARENT_MUST_REMAIN_PENDING_IN_G1_G2")
    if trainability["cross_stage_optimizer_rebuilt"] != "NOT_RUN":
        raise ValueError("ONLINE_REPLAY_CROSS_STAGE_OPTIMIZER_MUST_NOT_RUN_IN_G1_G2")
    CriticReadiness(**trainability["critic_readiness"]).validate()
    CriticReadiness(**runtime["critic_readiness"]).validate()
    temporal = transition["temporal"]
    if (
        temporal["data_grid_hz"], temporal["policy_hz"], temporal["flow_horizon"],
        temporal["critic_slots"], temporal["critic_action_features"],
        temporal["macro_duration_ms"],
    ) != (30, 10, 50, 3, 7, 100):
        raise ValueError("ONLINE_REPLAY_TEMPORAL_CONTRACT_DRIFT")
    if transition["temporal_parity"]["status"] != "BLOCKED":
        raise ValueError("ONLINE_REPLAY_RECORDED_TEMPORAL_PARITY_MUST_REMAIN_BLOCKED")
    if replay["intervention"]["canonical_payload_copies"] != 1:
        raise ValueError("ONLINE_REPLAY_REPLAY_CANONICAL_PAYLOAD_NOT_DEDUPLICATED")
    if reward["reward_gate"]["reward_bearing_online_update_authorized"]:
        raise ValueError("ONLINE_REPLAY_REWARD_BEARING_UPDATE_NOT_AUTHORIZED")
    if any(mode["authorized"] for mode in runtime["runtime_modes"].values()):
        raise ValueError("ONLINE_REPLAY_RUNTIME_MODE_UNEXPECTEDLY_AUTHORIZED")
    for schema in SCHEMA_PATHS.values():
        Draft202012Validator.check_schema(_load(schema))
    return {
        "G0_FINAL_PARENT_BINDING": "PENDING",
        "CROSS_STAGE_OPTIMIZER_REBUILT": "NOT_RUN",
        "temporal_parity": "BLOCKED",
        "critic_ready": False,
        "actor_q_guidance_enabled": False,
        "robot_execution_authorized": False,
    }


def apply_online_trainability(policy):
    """Delegate to the already accepted Frozen-VLM implementation, unchanged."""

    from forcesmolvla.rft.frozen_vlm_trainability import (
        apply_frozen_vlm_trainability,
    )

    manifest = apply_frozen_vlm_trainability(policy)
    if not manifest.frozen_names or not manifest.trainable_names:
        raise RuntimeError("ONLINE_REPLAY_TRAINABILITY_EMPTY_OWNERSHIP")
    return manifest
