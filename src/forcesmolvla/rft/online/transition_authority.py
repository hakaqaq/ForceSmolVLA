"""Pure CPU construction and validation for ACK-authoritative transitions."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Callable, Mapping, Sequence

from jsonschema import Draft202012Validator
import numpy as np

from forcesmolvla.action_delta import ActionDeltaProcessor
from forcesmolvla.temporal import select_latest_causal


ROOT = Path(__file__).parents[4]
SCHEMA_PATH = ROOT / "schemas/stage3_ack_transition.v1.schema.json"
SLOT_OWNERS = {
    "policy", "human_intervention", "human_release_hold", "safety_hold",
    "offline_demonstration",
}


class TransitionContractError(ValueError):
    pass


@dataclass(frozen=True)
class AcceptedAck:
    ack_id: str
    receive_monotonic_ns: int
    accepted_absolute_action7: tuple[float, ...]
    gripper_command_id: str
    gripper_ack_command_id: str
    slot_owner: str
    accepted_action_source: str
    intervention: bool
    accepted: bool = True
    workspace_clipped: bool = False

    def validate(self) -> "AcceptedAck":
        if not self.ack_id or not self.gripper_command_id or not self.gripper_ack_command_id:
            raise TransitionContractError("ONLINE_REPLAY_ACK_IDENTITY_MISSING")
        if self.gripper_command_id != self.gripper_ack_command_id:
            raise TransitionContractError("ONLINE_REPLAY_GRIPPER_COMMAND_ID_MISMATCH")
        if self.receive_monotonic_ns <= 0:
            raise TransitionContractError("ONLINE_REPLAY_ACK_TIMESTAMP_INVALID")
        if not self.accepted:
            raise TransitionContractError("ONLINE_REPLAY_ACK_REJECTED")
        if self.slot_owner not in SLOT_OWNERS:
            raise TransitionContractError("ONLINE_REPLAY_ACK_SLOT_OWNER_INVALID")
        values = np.asarray(self.accepted_absolute_action7, dtype=np.float64)
        if values.shape != (7,) or not np.isfinite(values).all():
            raise TransitionContractError("ONLINE_REPLAY_ACK_ACTION7_INVALID")
        if self.slot_owner == "human_intervention" and not (
            self.accepted_action_source == "human" and self.intervention
        ):
            raise TransitionContractError("ONLINE_REPLAY_EXPERT_OWNER_NOT_ACK_CONFIRMED")
        if self.slot_owner != "human_intervention" and self.intervention:
            raise TransitionContractError("ONLINE_REPLAY_INTERVENTION_OWNER_MISMATCH")
        return self


@dataclass(frozen=True)
class AckMacro:
    grid_monotonic_ns: tuple[int, int, int]
    ack_ids: tuple[str, str, str]
    gripper_command_ids: tuple[str, str, str]
    gripper_ack_command_ids: tuple[str, str, str]
    accepted_absolute_action_k7: np.ndarray
    slot_owner: tuple[str, str, str]
    workspace_clip_flags: tuple[bool, bool, bool]


REVISION_BOUND_EVENTS = {
    "request",
    "result",
    "chunk",
    "proposal",
    "ack_ledger",
    "current_observation",
    "next_observation",
    "transition",
}


def validate_episode_revision_bindings(
    event_bindings: Mapping[str, Mapping],
    *,
    policy_revision_id: str,
    model_sha256: str,
    policy_epoch: int,
) -> dict[str, dict]:
    """Fail closed so a caller can quarantine any cross-revision episode row."""

    if set(event_bindings) != REVISION_BOUND_EVENTS:
        raise TransitionContractError("ONLINE_REPLAY_REVISION_EVENT_BINDING_SET_INVALID")
    expected = {
        "policy_revision_id": policy_revision_id,
        "model_sha256": model_sha256,
        "policy_epoch": policy_epoch,
    }
    if (
        not policy_revision_id
        or len(model_sha256) != 64
        or any(char not in "0123456789abcdef" for char in model_sha256)
        or not isinstance(policy_epoch, int)
        or isinstance(policy_epoch, bool)
        or policy_epoch < 0
    ):
        raise TransitionContractError("ONLINE_REPLAY_EPISODE_REVISION_PIN_INVALID")
    value = {
        name: deepcopy(dict(event_bindings[name])) for name in sorted(event_bindings)
    }
    for name, binding in value.items():
        if set(binding) != set(expected) or binding != expected:
            raise TransitionContractError(
                f"ONLINE_REPLAY_CROSS_REVISION_{name.upper()}_QUARANTINE"
            )
    return value


def canonical_json_bytes(value: Mapping) -> bytes:
    try:
        return json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise TransitionContractError("ONLINE_REPLAY_CANONICAL_PAYLOAD_NOT_JSON_FINITE") from error


def canonical_payload_sha256(payload: Mapping) -> str:
    value = deepcopy(dict(payload))
    value.pop("integrity", None)
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def compute_transition_uid(payload: Mapping) -> str:
    try:
        identity = payload["identity"]
        observation = payload["observation"]
        next_observation = payload["next_observation"]
        behavior = payload["behavior_ack"]
        proposal = payload["policy_proposal"]
        stable = {
            "schema_version": payload["schema_version"],
            "run_id": identity["run_id"],
            "session_id": identity["session_id"],
            "episode_id": identity["episode_id"],
            "macro_index": identity["macro_index"],
            "observation_id": observation["observation_id"],
            "next_observation_id": None if next_observation is None else next_observation["observation_id"],
            "accepted_ack_ids": behavior["ack_ids"],
            "active_policy_revision": proposal["revision_id"],
        }
    except (KeyError, TypeError) as error:
        raise TransitionContractError("ONLINE_REPLAY_UID_FIELD_MISSING") from error
    return hashlib.sha256(canonical_json_bytes(stable)).hexdigest()


def validate_macro_grid(grid_monotonic_ns: Sequence[int]) -> tuple[int, int, int]:
    grid = np.asarray(grid_monotonic_ns, dtype=np.int64)
    if grid.shape != (3,) or np.any(grid <= 0) or np.any(np.diff(grid) <= 0):
        raise TransitionContractError("ONLINE_REPLAY_MACRO_GRID_SHAPE_OR_ORDER")
    indices = (grid * 30 + 500_000_000) // 1_000_000_000
    expected = ((indices * 1_000_000_000 + 15) // 30).astype(np.int64)
    if not np.array_equal(grid, expected) or not np.array_equal(np.diff(indices), [1, 1]):
        raise TransitionContractError("ONLINE_REPLAY_MACRO_GRID_PHASE_MISMATCH")
    return tuple(int(value) for value in grid)


def causal_zoh_ack_macro(
    acknowledgements: Sequence[AcceptedAck],
    grid_monotonic_ns: Sequence[int],
    *,
    max_ack_age_ms: float,
) -> AckMacro:
    grid = validate_macro_grid(grid_monotonic_ns)
    records = tuple(ack.validate() for ack in acknowledgements)
    stamps = np.asarray([ack.receive_monotonic_ns for ack in records], dtype=np.int64)
    if len(stamps) == 0 or (len(stamps) > 1 and np.any(np.diff(stamps) <= 0)):
        raise TransitionContractError("ONLINE_REPLAY_ACK_TIMESTAMPS_NOT_STRICTLY_INCREASING")
    selected = select_latest_causal(
        stamps, np.asarray(grid, dtype=np.int64), max_age_ms=max_ack_age_ms,
    )
    if not selected.valid.all():
        raise TransitionContractError("ONLINE_REPLAY_ACK_MISSING_OR_STALE")
    chosen = tuple(records[int(index)] for index in selected.source_indices)
    if any(ack.receive_monotonic_ns > tick for ack, tick in zip(chosen, grid, strict=True)):
        raise AssertionError("ONLINE_REPLAY_FUTURE_ACK_SELECTED")
    return AckMacro(
        grid_monotonic_ns=grid,
        ack_ids=tuple(ack.ack_id for ack in chosen),
        gripper_command_ids=tuple(ack.gripper_command_id for ack in chosen),
        gripper_ack_command_ids=tuple(ack.gripper_ack_command_id for ack in chosen),
        accepted_absolute_action_k7=np.asarray(
            [ack.accepted_absolute_action7 for ack in chosen], dtype=np.float64,
        ),
        slot_owner=tuple(ack.slot_owner for ack in chosen),
        workspace_clip_flags=tuple(ack.workspace_clipped for ack in chosen),
    )


def normalized_ack_behavior_action(
    macro: AckMacro,
    *,
    anchor_state7: Sequence[float],
    normalize_delta7: Callable[[np.ndarray], np.ndarray],
) -> np.ndarray:
    absolute = np.asarray(macro.accepted_absolute_action_k7, dtype=np.float64)
    state = np.asarray(anchor_state7, dtype=np.float64)
    if absolute.shape != (3, 7) or state.shape != (7,):
        raise TransitionContractError("ONLINE_REPLAY_ACK_ACTION_OR_STATE_SHAPE")
    delta = ActionDeltaProcessor.to_delta(absolute, state)
    normalized = np.asarray(normalize_delta7(delta), dtype=np.float64)
    if normalized.shape != (3, 7) or not np.isfinite(normalized).all():
        raise TransitionContractError("ONLINE_REPLAY_NORMALIZED_ACK_ACTION_INVALID")
    return normalized


def validate_reward_terminal(payload: Mapping, *, gamma_decision: float = 0.99) -> None:
    reward = payload.get("reward")
    terminated = payload.get("terminated")
    truncated = payload.get("truncated")
    bootstrap = payload.get("bootstrap")
    discount = payload.get("discount")
    quarantined = bool(payload.get("quarantined", False))
    next_observation_valid = bool(payload.get("next_observation_valid", False))
    if not isinstance(reward, (int, float)) or not math.isfinite(float(reward)):
        raise TransitionContractError("ONLINE_REPLAY_REWARD_NONFINITE")
    if quarantined:
        if not truncated or bootstrap is not None or discount is not None:
            raise TransitionContractError("ONLINE_REPLAY_QUARANTINE_OUTCOME_INVALID")
        return
    if not isinstance(terminated, bool) or not isinstance(truncated, bool) or not isinstance(bootstrap, bool):
        raise TransitionContractError("ONLINE_REPLAY_OUTCOME_BOOLEAN_INVALID")
    if terminated and truncated:
        raise TransitionContractError("ONLINE_REPLAY_TERMINATED_AND_TRUNCATED")
    if terminated and bootstrap:
        raise TransitionContractError("ONLINE_REPLAY_TERMINATED_BOOTSTRAP_NONZERO")
    expected = gamma_decision if bootstrap else 0.0
    if not isinstance(discount, (int, float)) or float(discount) != expected:
        raise TransitionContractError("ONLINE_REPLAY_DISCOUNT_BOOTSTRAP_MISMATCH")
    if not terminated and not next_observation_valid:
        raise TransitionContractError("ONLINE_REPLAY_NONTERMINAL_NEXT_OBSERVATION_INVALID")


def _schema() -> dict:
    return json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))


def validate_ack_transition(payload: Mapping) -> dict:
    value = deepcopy(dict(payload))
    errors = sorted(
        Draft202012Validator(_schema()).iter_errors(value),
        key=lambda error: tuple(str(item) for item in error.absolute_path),
    )
    if errors:
        path = ".".join(str(item) for item in errors[0].absolute_path)
        raise TransitionContractError(f"ONLINE_REPLAY_TRANSITION_SCHEMA:{path}:{errors[0].message}")
    if value["identity"]["transition_uid"] != compute_transition_uid(value):
        raise TransitionContractError("ONLINE_REPLAY_TRANSITION_UID_MISMATCH")
    if value["integrity"]["canonical_payload_sha256"] != canonical_payload_sha256(value):
        raise TransitionContractError("ONLINE_REPLAY_TRANSITION_DIGEST_MISMATCH")
    eligibility = value["eligibility"]
    if eligibility["quarantined"]:
        if any(eligibility[name] for name in ("critic_td_valid", "actor_q_valid", "expert_fm_available")):
            raise TransitionContractError("ONLINE_REPLAY_QUARANTINED_ROW_MARKED_TRAINABLE")
        if not eligibility["quarantine_reason"]:
            raise TransitionContractError("ONLINE_REPLAY_QUARANTINE_REASON_MISSING")
    elif eligibility["quarantine_reason"] is not None:
        raise TransitionContractError("ONLINE_REPLAY_NONQUARANTINED_REASON_PRESENT")
    observation = value["observation"]
    next_observation = value["next_observation"]
    if observation["episode_id"] != value["identity"]["episode_id"]:
        raise TransitionContractError("ONLINE_REPLAY_OBSERVATION_EPISODE_MISMATCH")
    if next_observation is not None and next_observation["episode_id"] != observation["episode_id"]:
        raise TransitionContractError("ONLINE_REPLAY_NEXT_OBSERVATION_CROSSES_EPISODE")
    if next_observation is not None and next_observation["timestamp_monotonic_ns"] <= observation["timestamp_monotonic_ns"]:
        raise TransitionContractError("ONLINE_REPLAY_NEXT_OBSERVATION_NOT_CAUSAL")
    behavior = value["behavior_ack"]
    for requested, acknowledged in zip(
        behavior["gripper_command_ids"], behavior["gripper_ack_command_ids"], strict=True,
    ):
        if requested != acknowledged:
            raise TransitionContractError("ONLINE_REPLAY_GRIPPER_COMMAND_ID_MISMATCH")
    owner_rules = {
        "policy": ("policy", False),
        "human_intervention": ("human", True),
        "human_release_hold": ("human", False),
        "safety_hold": ("safety", False),
        "offline_demonstration": ("offline", False),
    }
    for owner, source, intervention in zip(
        behavior["slot_owner"], behavior["accepted_action_source"],
        behavior["intervention_flags"], strict=True,
    ):
        if (source, intervention) != owner_rules[owner]:
            raise TransitionContractError("ONLINE_REPLAY_ACK_SLOT_OWNERSHIP_MISMATCH")
    fm = value["fm_target"]
    valid = np.asarray(fm["action_valid_mask_h50"], dtype=np.bool_)
    expert_slots = np.asarray(fm["expert_slot_mask_h50"], dtype=np.bool_)
    expert_features = np.asarray(fm["expert_feature_mask_h50x7"], dtype=np.bool_)
    expected_valid = np.arange(50) < int(valid.sum())
    if not np.array_equal(valid, expected_valid):
        raise TransitionContractError("ONLINE_REPLAY_ACTION_VALID_MASK_NOT_PREFIX")
    if not np.array_equal(expert_features, np.repeat(expert_slots[:, None], 7, axis=1)):
        raise TransitionContractError("ONLINE_REPLAY_EXPERT_OWNERSHIP_MUST_BE_SLOT_LEVEL")
    if np.any(expert_slots & ~valid):
        raise TransitionContractError("ONLINE_REPLAY_EXPERT_MASK_OUTSIDE_ACTION_VALID")
    outcome = dict(value["outcome"])
    outcome.update({
        "quarantined": eligibility["quarantined"],
        "next_observation_valid": next_observation is not None and next_observation["valid"],
    })
    validate_reward_terminal(outcome)
    if eligibility["critic_td_valid"] and not all(observation[name]["valid"] for name in ("camera1", "camera2")):
        raise TransitionContractError("ONLINE_REPLAY_TD_ROW_HAS_INVALID_CAMERA")
    return value


def finalize_ack_transition(payload_without_integrity: Mapping) -> dict:
    value = deepcopy(dict(payload_without_integrity))
    value.setdefault("identity", {})["transition_uid"] = compute_transition_uid(value)
    value["integrity"] = {"canonical_payload_sha256": canonical_payload_sha256(value)}
    return validate_ack_transition(value)
