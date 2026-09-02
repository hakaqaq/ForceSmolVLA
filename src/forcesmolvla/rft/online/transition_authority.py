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
from forcesmolvla.rft.critic_action_adapter_v2 import (
    CRITIC_ACTION_CONTRACT,
    CriticActionContract,
    build_critic_transition_grid,
)
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
    source_command_id: str = ""
    source_dispatch_sequence: int = -1
    source_model_index: int = -1
    episode_id: str = ""
    policy_revision: str = ""
    takeover_generation: int = 0
    reset_generation: int = 0
    chunk_id: str = ""
    chunk_compatibility_key: str = ""
    clock_domain: str = ""
    controller_authority: str = ""

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
        if self.source_dispatch_sequence < -1 or self.source_model_index < -1:
            raise TransitionContractError("ONLINE_REPLAY_ACK_SOURCE_INDEX_INVALID")
        if not all(
            (
                self.source_command_id,
                self.episode_id,
                self.policy_revision,
                self.chunk_id,
                self.chunk_compatibility_key,
                self.clock_domain,
                self.controller_authority,
            )
        ):
            raise TransitionContractError("ONLINE_REPLAY_ACK_LINEAGE_MISSING")
        if self.accepted_action_source in {"policy", "human"}:
            if self.source_dispatch_sequence < 0:
                raise TransitionContractError(
                    "ONLINE_REPLAY_ACK_DISPATCH_SEQUENCE_MISSING"
                )
            if self.accepted_action_source == "policy" and self.source_model_index < 0:
                raise TransitionContractError(
                    "ONLINE_REPLAY_ACK_MODEL_INDEX_MISSING"
                )
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
    behavior_mask: tuple[bool, bool, bool] = (True, True, True)
    source_command_ids: tuple[str, str, str] = ("", "", "")
    source_dispatch_sequences: tuple[int, int, int] = (-1, -1, -1)
    source_model_indices: tuple[int, int, int] = (-1, -1, -1)
    chunk_ids: tuple[str, str, str] = ("", "", "")
    contract_version: str = CRITIC_ACTION_CONTRACT.version
    next_timestamp_ns: int = 0
    macro_duration_ns: int = CRITIC_ACTION_CONTRACT.macro_duration_ns


def _lineage(ack: AcceptedAck) -> tuple[object, ...]:
    return (
        ack.episode_id,
        ack.accepted_action_source,
        ack.takeover_generation,
        ack.reset_generation,
        ack.policy_revision,
        ack.clock_domain,
        ack.controller_authority,
        ack.chunk_compatibility_key,
    )


def build_ack_behavior_macro(
    *,
    accepted_ack_stream: Sequence[AcceptedAck],
    anchor_timestamp_ns: int,
    action_source: str,
    contract: CriticActionContract = CRITIC_ACTION_CONTRACT,
    max_ack_age_ms: float,
    boundary_timestamp_ns: int | None = None,
    required_anchor_ack_id: str | None = None,
) -> AckMacro:
    """Build one fail-closed ACK-authoritative behavior macro."""

    contract.validate()
    if action_source not in {"policy", "human", "offline_demonstration"}:
        raise TransitionContractError("ONLINE_REPLAY_ACTION_SOURCE_INVALID")
    grid, nominal_next = build_critic_transition_grid(
        anchor_timestamp_ns, contract=contract
    )
    next_timestamp = nominal_next if boundary_timestamp_ns is None else int(
        boundary_timestamp_ns
    )
    if not grid[0] < next_timestamp <= nominal_next:
        raise TransitionContractError("ONLINE_REPLAY_PARTIAL_BOUNDARY_INVALID")
    mask = tuple(tick < next_timestamp for tick in grid)
    if not any(mask) or mask != tuple(index < sum(mask) for index in range(3)):
        raise TransitionContractError("ONLINE_REPLAY_BEHAVIOR_MASK_NOT_PREFIX")

    records = tuple(ack.validate() for ack in accepted_ack_stream)
    if not records:
        raise TransitionContractError("ONLINE_REPLAY_ACK_MISSING_OR_STALE")
    stamps = np.asarray(
        [ack.receive_monotonic_ns for ack in records], dtype=np.int64
    )
    if len(stamps) > 1 and np.any(np.diff(stamps) <= 0):
        raise TransitionContractError(
            "ONLINE_REPLAY_ACK_TIMESTAMPS_NOT_STRICTLY_INCREASING"
        )
    valid_grid = np.asarray(
        [tick for tick, valid in zip(grid, mask, strict=True) if valid],
        dtype=np.int64,
    )
    selected = select_latest_causal(
        stamps, valid_grid, max_age_ms=max_ack_age_ms
    )
    if not selected.valid.all():
        raise TransitionContractError("ONLINE_REPLAY_ACK_MISSING_OR_STALE")
    chosen_valid = tuple(records[int(index)] for index in selected.source_indices)
    if any(
        ack.receive_monotonic_ns > tick
        for ack, tick in zip(chosen_valid, valid_grid, strict=True)
    ):
        raise AssertionError("ONLINE_REPLAY_FUTURE_ACK_SELECTED")
    if any(ack.accepted_action_source != action_source for ack in chosen_valid):
        raise TransitionContractError("ONLINE_REPLAY_ACTION_SOURCE_BOUNDARY")
    expected_lineage = _lineage(chosen_valid[0])
    if any(_lineage(ack) != expected_lineage for ack in chosen_valid[1:]):
        raise TransitionContractError("ONLINE_REPLAY_ACK_LINEAGE_BOUNDARY")
    if required_anchor_ack_id is not None:
        if chosen_valid[0].ack_id != required_anchor_ack_id:
            raise TransitionContractError(
                "ONLINE_REPLAY_COMMAND_EFFECTIVE_ANCHOR_MISMATCH"
            )
        if any(ack.ack_id != required_anchor_ack_id for ack in chosen_valid):
            raise TransitionContractError(
                "ONLINE_REPLAY_COMMAND_EFFECTIVE_PHASE_CHANGED"
            )

    chosen: list[AcceptedAck | None] = list(chosen_valid)
    chosen.extend([None] * (contract.critic_slots - len(chosen)))
    absolute = np.zeros((contract.critic_slots, contract.action_dim), dtype=np.float64)
    for slot, ack in enumerate(chosen):
        if ack is not None:
            absolute[slot] = ack.accepted_absolute_action7

    def values(name: str, default):
        return tuple(default if ack is None else getattr(ack, name) for ack in chosen)

    return AckMacro(
        grid_monotonic_ns=grid,
        ack_ids=values("ack_id", ""),
        gripper_command_ids=values("gripper_command_id", ""),
        gripper_ack_command_ids=values("gripper_ack_command_id", ""),
        accepted_absolute_action_k7=absolute,
        slot_owner=values("slot_owner", ""),
        workspace_clip_flags=values("workspace_clipped", False),
        behavior_mask=mask,
        source_command_ids=values("source_command_id", ""),
        source_dispatch_sequences=values("source_dispatch_sequence", -1),
        source_model_indices=values("source_model_index", -1),
        chunk_ids=values("chunk_id", ""),
        contract_version=contract.version,
        next_timestamp_ns=next_timestamp,
        macro_duration_ns=next_timestamp - grid[0],
    )


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
    expected, _ = build_critic_transition_grid(
        int(grid[0]), contract=CRITIC_ACTION_CONTRACT
    )
    if tuple(int(value) for value in grid) != expected:
        raise TransitionContractError("ONLINE_REPLAY_MACRO_GRID_PHASE_MISMATCH")
    return tuple(int(value) for value in grid)


def causal_zoh_ack_macro(
    acknowledgements: Sequence[AcceptedAck],
    grid_monotonic_ns: Sequence[int],
    *,
    max_ack_age_ms: float,
) -> AckMacro:
    grid = validate_macro_grid(grid_monotonic_ns)
    source = acknowledgements[0].accepted_action_source if acknowledgements else ""
    return build_ack_behavior_macro(
        accepted_ack_stream=acknowledgements,
        anchor_timestamp_ns=grid[0],
        action_source=(
            "offline_demonstration" if source == "offline" else source
        ),
        contract=CRITIC_ACTION_CONTRACT,
        max_ack_age_ms=max_ack_age_ms,
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
    mask = np.asarray(macro.behavior_mask, dtype=np.bool_)
    normalized[~mask] = 0.0
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
    if bootstrap != (not (terminated or truncated)):
        raise TransitionContractError("ONLINE_REPLAY_OUTCOME_BOOTSTRAP_CONTRACT")
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
