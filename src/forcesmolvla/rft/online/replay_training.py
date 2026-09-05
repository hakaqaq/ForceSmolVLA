#!/usr/bin/env python3
"""Replay materialization and batch primitives shared by ForceRFT training."""

from __future__ import annotations

from dataclasses import dataclass, replace
from copy import deepcopy
from functools import lru_cache
import json
from pathlib import Path
import random
import sys
from typing import Any, Iterable, Mapping

import numpy as np
import torch
import yaml

from forcesmolvla.action_delta import ActionDeltaProcessor
from forcesmolvla.rft.online.action_representation import (
    ABSOLUTE_ACTION_ROTATION_REPRESENTATION,
    legacy_absolute_action7_to_rpy_xyz,
)
from forcesmolvla.rft.online.transition_authority import (
    AcceptedAck,
    AckMacro,
    ActorQEligibility,
    DISPATCH_DECISION_CRITIC_CONTRACT_VERSION,
    ONLINE_SEMANTICS_VERSION,
    TransitionContractError,
    build_ack_behavior_macro,
    derive_actor_q_eligibility,
    normalized_ack_behavior_action,
    normalized_behavior_residual,
)
from forcesmolvla.rft.critic_action_adapter_v2 import (
    CRITIC_ACTION_CONTRACT,
    build_critic_transition_grid,
)


ROOT = Path(__file__).resolve().parents[4]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(ROOT / "tools"))

TASK_ID = "task2"
FORMAL_R_ROOT = ROOT / "outputs/task2/online"
COMMON_TRAINING_CONFIG = ROOT / "configs/forcerft/online_ack_residual_actor_critic.yaml"
TASK_PROFILE_ROOT = ROOT / "configs/forcerft/tasks"
DATASET = ROOT / "datasets/task2_lerobotv3"
REWARD_TRANSITION_ROOT = ROOT / "datasets/task2_forcerft_offline_reward_transitions"
RANDOM_SEED = 4404
TASK = "Pick up the purple ring and place it onto the red peg."
ACK_RESIDUAL_TRANSITION_SCHEMA_VERSION = "forcesmolvla_ack_residual_transition.v2"
LEGACY_ACK_RESIDUAL_TRANSITION_SCHEMA_VERSIONS = frozenset(
    {"forcesmolvla_stage3_production_bridge_transition.v1"}
)
SUPPORTED_ACK_RESIDUAL_TRANSITION_SCHEMA_VERSIONS = frozenset(
    {
        ACK_RESIDUAL_TRANSITION_SCHEMA_VERSION,
        *LEGACY_ACK_RESIDUAL_TRANSITION_SCHEMA_VERSIONS,
    }
)


def load_common_actor_critic_config(task_id: str | None = None) -> dict[str, Any]:
    """Combine one immutable algorithm contract with one path-only task profile."""

    selected_task = TASK_ID if task_id is None else task_id
    common = yaml.safe_load(COMMON_TRAINING_CONFIG.read_text(encoding="utf-8"))
    profile = yaml.safe_load(
        (TASK_PROFILE_ROOT / f"{selected_task}.yaml").read_text(encoding="utf-8")
    )
    allowed_profile_keys = {
        "task_id",
        "dataset_root",
        "output_root",
        "offline_replay_root",
        "online_replay_root",
        "reward_calibration_path",
        "normalizer_path",
        "task_prompt",
        "workspace_configuration",
    }
    if set(profile) != allowed_profile_keys:
        raise ValueError("FORCERFT_TASK_PROFILE_FIELDS_INVALID")
    config = deepcopy(common)
    config["task"] = {
        "task_id": profile["task_id"],
        "output_root": profile["output_root"],
        "prompt": profile["task_prompt"],
    }
    config["paths"] = {
        "lerobot_v3_root": profile["dataset_root"],
        "online_replay_root": profile["online_replay_root"],
        "normalizer": profile["normalizer_path"],
        "reward_calibration": profile["reward_calibration_path"],
        "workspace_configuration": profile["workspace_configuration"],
    }
    return config


def algorithm_hyperparameters(config: Mapping[str, Any]) -> dict[str, Any]:
    """Return the task-independent algorithm portion for direct parity checks."""

    return {
        name: deepcopy(config[name])
        for name in (
            "environment",
            "batching",
            "ack_critic_warmup",
            "residual_actor_critic_training",
            "optimizer",
            "objective",
            "wrist_wrench_residual_actor",
            "ack_residual_twin_q",
        )
    }


def configure_task_paths(
    *,
    task_id: str,
    dataset_root: Path | None = None,
    reward_transition_root: Path | None = None,
    output_root: Path | None = None,
) -> None:
    """Configure task-scoped data and output roots before creating replay objects."""

    from forcesmolvla.training_runtime import (
        resolve_task_dataset_root,
        resolve_task_output_root,
        resolve_task_reward_transition_root,
    )

    global TASK_ID, FORMAL_R_ROOT, DATASET, REWARD_TRANSITION_ROOT, TASK
    TASK_ID = task_id
    DATASET = resolve_task_dataset_root(
        ROOT, task_id=task_id, dataset_root=dataset_root
    )
    REWARD_TRANSITION_ROOT = resolve_task_reward_transition_root(
        ROOT,
        task_id=task_id,
        reward_transition_root=reward_transition_root,
    )
    FORMAL_R_ROOT = resolve_task_output_root(
        ROOT, task_id=task_id, output_root=output_root
    ) / "online"
    conversion_path = DATASET / "conversion_manifest.json"
    if conversion_path.is_file():
        conversion = json.loads(conversion_path.read_text(encoding="utf-8"))
        tasks = {str(item["task"]) for item in conversion.get("episodes", ())}
        if len(tasks) == 1:
            TASK = tasks.pop()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _generation(row: Mapping[str, Any]) -> tuple[int, int, int]:
    value = row["generation"]
    return (
        int(value["policy_epoch"]),
        int(value["takeover_generation"]),
        int(value["reset_generation"]),
    )


@dataclass(frozen=True)
class ProductionAckMacro:
    transition: Mapping[str, Any]
    behavior: AckMacro
    next_grid_monotonic_ns: int
    ack_provenance: tuple[Mapping[str, Any], ...]
    actor_q_eligibility: ActorQEligibility


def _rational_grid_from_transition(
    row: Mapping[str, Any],
) -> tuple[tuple[int, int, int], int]:
    anchor = int(row["observation"]["materialized_timestamp_monotonic_ns"])
    grid, nominal_next = build_critic_transition_grid(
        anchor, contract=CRITIC_ACTION_CONTRACT
    )
    actual_next = int(
        row["next_observation"]["materialized_timestamp_monotonic_ns"]
    )
    outcome = row["outcome"]
    boundary = bool(outcome["terminated"] or outcome.get("truncated", False))
    if (
        actual_next <= anchor
        or actual_next > nominal_next
        or (not boundary and actual_next != nominal_next)
    ):
        raise RuntimeError("FORCERFT_ONLINE_REPLAY_TRANSITION_HORIZON_INVALID")
    persisted = row.get("critic_action_contract")
    if persisted is not None and (
        persisted.get("contract_version") != CRITIC_ACTION_CONTRACT.version
        or tuple(persisted.get("grid_timestamp_ns", ())) != grid
        or int(persisted.get("next_timestamp_ns", -1)) != actual_next
    ):
        raise RuntimeError("FORCERFT_ONLINE_REPLAY_PERSISTED_CONTRACT_MISMATCH")
    return grid, actual_next


def _accepted_ack(row: Mapping[str, Any]) -> AcceptedAck:
    authority = row["action_authority"]
    pose_ack = authority["pose_ack"]
    gripper_origin = authority["gripper_terminal_provenance"]
    source = str(
        row.get("action_source")
        or authority.get("executed_action_source", "policy")
    )
    command_id = str(
        gripper_origin.get(
            "origin_action_goal_id",
            authority.get("gripper", {}).get("command_id", ""),
        )
    )
    if not command_id:
        command_id = str(
            authority.get("gripper", {}).get(
                "source_command_id", row["identity"]["source_ack_id"]
            )
        )
    selection = row.get("policy_lineage", {}).get("selection", {})
    persisted = row.get("critic_action_contract", {})
    revision = str(
        row.get("policy_lineage", {}).get(
            "revision", persisted.get("policy_revision", "human-controller")
        )
    )
    generation = row["generation"]
    receive_ns = int(
        pose_ack.get(
            "upper_receive_monotonic_ns", pose_ack.get("receive_monotonic_ns", 0)
        )
    )
    accepted_action = np.asarray(
        authority["accepted_absolute_action7"], dtype=np.float64
    )
    representation = row.get("absolute_action_rotation_representation")
    if representation is None and source == "human":
        accepted_action = legacy_absolute_action7_to_rpy_xyz(accepted_action)
    elif representation not in {None, ABSOLUTE_ACTION_ROTATION_REPRESENTATION}:
        raise RuntimeError("FORCERFT_ONLINE_ACTION_ROTATION_REPRESENTATION_INVALID")
    return AcceptedAck(
        ack_id=str(row["identity"]["source_ack_id"]),
        receive_monotonic_ns=receive_ns,
        accepted_absolute_action7=tuple(float(value) for value in accepted_action),
        gripper_command_id=command_id,
        gripper_ack_command_id=command_id,
        slot_owner="human_intervention" if source == "human" else "policy",
        accepted_action_source=source,
        intervention=source == "human",
        accepted=bool(pose_ack["accepted"]),
        workspace_clipped=bool(
            authority.get("safety_arbitration", {}).get("workspace_clipped", False)
        ),
        source_command_id=str(
            pose_ack.get(
                "command_id",
                pose_ack.get("request_stamp_ns", row["identity"]["source_ack_id"]),
            )
        ),
        source_dispatch_sequence=int(
            selection.get("sequence", row["identity"]["decision_id"])
        ),
        source_model_index=int(selection.get("action_index", -1)),
        episode_id=str(row["identity"]["episode_id"]),
        policy_revision=revision,
        takeover_generation=int(generation["takeover_generation"]),
        reset_generation=int(generation["reset_generation"]),
        chunk_id=str(selection.get("chunk_id", persisted.get("chunk_id", "human"))),
        chunk_compatibility_key=str(
            persisted.get(
                "chunk_compatibility_key",
                f"{row['identity']['episode_id']}:{source}:"
                f"{generation['takeover_generation']}:{generation['reset_generation']}:"
                f"{revision}",
            )
        ),
        clock_domain=str(
            persisted.get(
                "clock_domain", row["observation"].get("clock_domain_id", "")
            )
        ),
        controller_authority=str(
            persisted.get("controller_authority", "fr3-reference-controller")
        ),
    )


def build_ack_macros(
    rows: Iterable[Mapping[str, Any]],
    *,
    max_ack_age_ms: float = CRITIC_ACTION_CONTRACT.max_ack_age_ms,
) -> tuple[ProductionAckMacro, ...]:
    """Build 100 ms K=3 behavior macros from real ACKs on the 30 Hz grid."""

    macros: list[ProductionAckMacro] = []
    legacy_rows: list[Mapping[str, Any]] = []
    for row in rows:
        if row.get("policy_lineage", {}).get("proposal", {}).get(
            "invalidated_by_takeover"
        ) is True:
            continue
        if row.get("online_semantics_version") == ONLINE_SEMANTICS_VERSION:
            persisted = row.get("critic_action_contract", {})
            if (
                persisted.get("contract_version")
                != DISPATCH_DECISION_CRITIC_CONTRACT_VERSION
            ):
                raise RuntimeError(
                    "FORCERFT_ONLINE_REPLAY_DISPATCH_CONTRACT_MISMATCH"
                )
            accepted = np.asarray(
                row.get("accepted_absolute_action_k7"), dtype=np.float64
            )
            mask = tuple(bool(value) for value in persisted.get("behavior_mask", ()))
            if (
                accepted.shape != (3, 7)
                or not np.isfinite(accepted).all()
                or mask != (True, True, True)
            ):
                raise RuntimeError(
                    "FORCERFT_ONLINE_REPLAY_DISPATCH_ACTION_INVALID"
                )
            behavior = AckMacro(
                grid_monotonic_ns=tuple(
                    int(value) for value in persisted["grid_timestamp_ns"]
                ),
                ack_ids=tuple(str(value) for value in persisted["source_ack_ids"]),
                gripper_command_ids=tuple(
                    str(value)
                    for value in persisted.get(
                        "gripper_command_ids",
                        [
                            row["action_authority"]["gripper_terminal_provenance"][
                                "origin_action_goal_id"
                            ]
                        ]
                        * 3,
                    )
                ),
                gripper_ack_command_ids=tuple(
                    str(value)
                    for value in persisted.get(
                        "gripper_ack_command_ids",
                        [
                            row["action_authority"]["gripper_terminal_provenance"][
                                "origin_action_goal_id"
                            ]
                        ]
                        * 3,
                    )
                ),
                accepted_absolute_action_k7=accepted,
                slot_owner=(
                    ("policy",) * 3
                    if row["action_source"] == "policy"
                    else ("human_intervention",) * 3
                ),
                workspace_clip_flags=tuple(
                    bool(row["action_authority"]["safety_arbitration"].get(
                        "workspace_clipped", False
                    ))
                    for _ in range(3)
                ),
                behavior_mask=mask,
                source_command_ids=tuple(
                    str(value) for value in persisted["source_command_ids"]
                ),
                source_dispatch_sequences=tuple(
                    int(value)
                    for value in persisted["source_dispatch_sequences"]
                ),
                source_model_indices=tuple(
                    int(value) for value in persisted["source_model_indices"]
                ),
                chunk_ids=tuple(
                    str(value) for value in persisted["source_chunk_ids"]
                ),
                controller_authorities=tuple(
                    str(persisted["controller_authority"]) for _ in range(3)
                ),
                contract_version=DISPATCH_DECISION_CRITIC_CONTRACT_VERSION,
                next_timestamp_ns=int(persisted["next_timestamp_ns"]),
                macro_duration_ns=int(persisted["macro_duration_ns"]),
            )
            eligibility = derive_actor_q_eligibility(
                macro=behavior,
                action_source=str(row["action_source"]),
                quarantined=False,
                observation_valid=True,
                next_observation_valid=True,
            )
            macros.append(
                ProductionAckMacro(
                    transition=row,
                    behavior=behavior,
                    next_grid_monotonic_ns=behavior.next_timestamp_ns,
                    ack_provenance=tuple(
                        {
                            "ack_id": behavior.ack_ids[index],
                            "source_command_id": behavior.source_command_ids[index],
                            "dispatch_sequence": behavior.source_dispatch_sequences[index],
                            "action_index": behavior.source_model_indices[index],
                            "chunk_id": behavior.chunk_ids[index],
                        }
                        for index in range(3)
                    ),
                    actor_q_eligibility=eligibility,
                )
            )
            continue
        legacy_rows.append(row)

    groups: dict[tuple[object, ...], list[Mapping[str, Any]]] = {}
    for row in legacy_rows:
        ack = _accepted_ack(row)
        key = (
            ack.episode_id,
            ack.accepted_action_source,
            ack.takeover_generation,
            ack.reset_generation,
            ack.policy_revision,
            ack.clock_domain,
            ack.controller_authority,
            ack.chunk_compatibility_key,
        )
        groups.setdefault(key, []).append(row)

    for group_rows in groups.values():
        ordered = sorted(
            group_rows, key=lambda row: _accepted_ack(row).receive_monotonic_ns
        )
        acknowledgements = tuple(_accepted_ack(row) for row in ordered)
        by_ack_id = {
            ack.ack_id: row for ack, row in zip(acknowledgements, ordered, strict=True)
        }
        for row in sorted(
            group_rows,
            key=lambda item: int(
                item["observation"]["materialized_timestamp_monotonic_ns"]
            ),
        ):
            grid, next_grid = _rational_grid_from_transition(row)
            outcome = row["outcome"]
            boundary = (
                next_grid
                if outcome["terminated"] or outcome.get("truncated", False)
                else None
            )
            action_source = _accepted_ack(row).accepted_action_source
            try:
                behavior = build_ack_behavior_macro(
                    accepted_ack_stream=acknowledgements,
                    anchor_timestamp_ns=grid[0],
                    action_source=action_source,
                    contract=CRITIC_ACTION_CONTRACT,
                    max_ack_age_ms=max_ack_age_ms,
                    boundary_timestamp_ns=boundary,
                    required_anchor_ack_id=(
                        str(row["identity"]["source_ack_id"])
                        if action_source == "policy"
                        and row.get("critic_action_contract", {}).get(
                            "anchor_semantics"
                        ) == "controller-accepted-command-effective"
                        else None
                    ),
                )
            except TransitionContractError as error:
                if str(error) in {
                    "ONLINE_REPLAY_ACK_MISSING_OR_STALE",
                    "ONLINE_REPLAY_COMMAND_EFFECTIVE_PHASE_CHANGED",
                }:
                    continue
                raise
            persisted = row.get("critic_action_contract")
            if persisted is not None:
                expected = {
                    "contract_version": behavior.contract_version,
                    "action_source": action_source,
                    "grid_timestamp_ns": list(behavior.grid_monotonic_ns),
                    "next_timestamp_ns": behavior.next_timestamp_ns,
                    "source_command_ids": list(behavior.source_command_ids),
                    "source_ack_ids": list(behavior.ack_ids),
                    "source_dispatch_sequences": list(
                        behavior.source_dispatch_sequences
                    ),
                    "source_model_indices": list(behavior.source_model_indices),
                    "source_chunk_ids": list(behavior.chunk_ids),
                    "behavior_mask": list(behavior.behavior_mask),
                    "macro_duration_ns": behavior.macro_duration_ns,
                    "discount": float(row["outcome"]["discount"]),
                }
                if any(persisted.get(key) != value for key, value in expected.items()):
                    raise RuntimeError(
                        "FORCERFT_ONLINE_REPLAY_PERSISTED_ACK_PROVENANCE_MISMATCH"
                    )
            provenance = tuple(
                {
                    "ack_id": ack_id,
                    "receive_monotonic_ns": (
                        -1 if not ack_id else _accepted_ack(by_ack_id[ack_id]).receive_monotonic_ns
                    ),
                    "source_command_id": behavior.source_command_ids[slot],
                    "chunk_id": behavior.chunk_ids[slot],
                    "action_index": behavior.source_model_indices[slot],
                    "dispatch_sequence": behavior.source_dispatch_sequences[slot],
                    "generation": (
                        {} if not ack_id else dict(by_ack_id[ack_id]["generation"])
                    ),
                }
                for slot, ack_id in enumerate(behavior.ack_ids)
            )
            eligibility = derive_actor_q_eligibility(
                macro=behavior,
                action_source=action_source,
                quarantined=bool(row.get("eligibility", {}).get("quarantined", False)),
                observation_valid=bool(row["observation"].get("valid", True)),
                next_observation_valid=bool(
                    row["next_observation"].get("valid", True)
                ),
            )
            macros.append(
                ProductionAckMacro(
                    row, behavior, next_grid, provenance, eligibility
                )
            )
    return tuple(macros)


def _sealed_episode_ids(root: Path) -> set[str]:
    sealed = set()
    for path in sorted((root / "episodes").glob("*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        if record.get("status") == "SEALED_COMMITTED":
            sealed.add(str(record.get("episode_id", "")))
    return sealed


def _transition_eligibility(
    row: dict[str, Any], source: str | None
) -> dict[str, Any]:
    eligibility = row["eligibility"]
    if "critic_td_valid" in eligibility:
        eligibility.setdefault("td_eligible", eligibility["critic_td_valid"])
    if "td_eligible" in eligibility and "fm_eligible" in eligibility:
        return eligibility
    outcome = row.get("outcome", {})
    outcome_pair = (
        outcome.get("operator_task_outcome"),
        outcome.get("detector_outcome"),
    )
    # Before failure admission existed, every admitted row was confirmed-success.
    # Materialize the new independent fields for those exact-resume artifacts only.
    if outcome_pair == ("success", "success"):
        eligibility.setdefault("td_eligible", True)
        eligibility.setdefault("fm_eligible", source == "human")
    return eligibility


def _validated_formal_row(
    envelope: Mapping[str, Any],
    *,
    episode_id: str,
    admission_relative: str,
) -> tuple[dict[str, Any], str]:
    require(
        envelope.get("episode_sealed") is True
        and envelope.get("admission_record") == admission_relative,
        "FORCERFT_ONLINE_REPLAY_EPISODE_AUTHORITY",
    )
    row = envelope["payload"]
    require(
        str(row["identity"]["episode_id"]) == episode_id,
        "FORCERFT_ONLINE_REPLAY_EPISODE_ID_MISMATCH",
    )
    require(
        row.get("schema_version") == ACK_RESIDUAL_TRANSITION_SCHEMA_VERSION
        and row.get("online_semantics_version") == ONLINE_SEMANTICS_VERSION,
        "FORCERFT_ONLINE_REPLAY_SEMANTICS_MISMATCH",
    )
    source = row.get(
        "action_source",
        row.get("action_authority", {}).get("executed_action_source"),
    )
    eligibility = _transition_eligibility(row, source)
    critic_td_valid = eligibility.get(
        "critic_td_valid", eligibility.get("td_eligible")
    )
    require(
        row["classification"] == "recorded_live_policy_execution_smoke"
        and source in {"policy", "human"}
        and row["action_authority"]["executed_action_source"] == source
        and eligibility.get("formal_replay") is True
        and eligibility.get("formal_training_replay_eligible") is True
        and eligibility.get("real_online_r") is True
        and eligibility.get("replay_membership") == "R_online"
        and critic_td_valid is True,
        "FORCERFT_ONLINE_REPLAY_MEMBERSHIP",
    )
    eligibility["critic_td_valid"] = True
    row["action_source"] = source
    row.setdefault("expert", source == "human")
    row.setdefault("intervention", source == "human")
    if source == "human":
        require(row["intervention"] is True, "FORCERFT_ONLINE_HUMAN_INVALID")
    else:
        require(
            row["expert"] is False and row["intervention"] is False,
            "FORCERFT_ONLINE_POLICY_REPLAY_SEMANTICS_INVALID",
        )
        row.setdefault("action_target", [[0.0] * 7 for _ in range(50)])
        row.setdefault("action_valid_mask", [[False] * 7 for _ in range(50)])
    return row, str(source)


def load_formal_online_episode(root: Path, admission_id: str) -> tuple[
    list[dict[str, Any]],
    tuple[ProductionAckMacro, ...],
    dict[str, Path],
    list[dict[str, Any]],
]:
    require(
        bool(admission_id)
        and Path(admission_id).name == admission_id
        and not admission_id.endswith(".json"),
        "FORCERFT_ONLINE_REPLAY_ADMISSION_ID_INVALID",
    )
    admission_relative = f"admissions/{admission_id}.json"
    admission_path = root / admission_relative
    seal_path = root / "episodes" / f"{admission_id}.json"
    require(
        admission_path.is_file() and seal_path.is_file(),
        "FORCERFT_ONLINE_REPLAY_ADMISSION_MISSING",
    )
    admission = json.loads(admission_path.read_text(encoding="utf-8"))
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    episode_id = str(admission.get("episode_id", ""))
    require(
        seal.get("status") == "SEALED_COMMITTED"
        and seal.get("admission_record") == admission_relative
        and str(seal.get("episode_id", "")) == episode_id
        and str(admission.get("admission_id", admission_id)) == admission_id
        and str(seal.get("admission_id", admission_id)) == admission_id,
        "FORCERFT_ONLINE_REPLAY_EPISODE_SEAL_INVALID",
    )
    require(
        admission.get("policy_execution_smoke_bridge") == "PASS",
        "FORCERFT_ONLINE_REPLAY_BRIDGE_NOT_PASS",
    )
    require(
        admission.get("source_episode_semantics")
        == {"formal_replay": False, "real_online_r": False},
        "FORCERFT_ONLINE_REPLAY_SOURCE_SEMANTICS",
    )
    require(
        (
            admission.get("operator_task_outcome"),
            admission.get("detector_outcome"),
        )
        in {("success", "success"), ("failure", "miss")},
        "FORCERFT_ONLINE_REPLAY_OUTCOME_CONFLICT",
    )
    expected = int(admission["accepted_unique_r_transition_count"])
    transition_uids = tuple(str(value) for value in seal["transition_uids"])
    admission_uids = tuple(
        str(item["transition_uid"]) for item in admission["transitions"]
    )
    require(
        len(transition_uids) == expected
        and len(set(transition_uids)) == expected
        and set(transition_uids) == set(admission_uids),
        "FORCERFT_ONLINE_REPLAY_ADMISSION_UID_MISMATCH",
    )

    policy_rows: list[dict[str, Any]] = []
    human_rows: list[dict[str, Any]] = []
    for uid in transition_uids:
        path = root / "replay" / f"{uid}.json"
        require(path.is_file(), "FORCERFT_ONLINE_REPLAY_TRANSITION_MISSING")
        envelope = json.loads(path.read_text(encoding="utf-8"))
        row, source = _validated_formal_row(
            envelope,
            episode_id=episode_id,
            admission_relative=admission_relative,
        )
        require(
            str(row["identity"]["transition_uid"]) == uid,
            "FORCERFT_ONLINE_REPLAY_TRANSITION_UID_MISMATCH",
        )
        if source == "human":
            human_rows.append(row)
        else:
            policy_rows.append(row)
    all_rows = [*policy_rows, *human_rows]
    require(len(all_rows) == expected, "FORCERFT_ONLINE_REPLAY_ADMISSION_COUNT")
    require(
        any(row["outcome"]["terminated"] for row in all_rows),
        "FORCERFT_ONLINE_REPLAY_MACRO_TERMINAL_MISSING",
    )
    macros = build_ack_macros(policy_rows)
    return (
        policy_rows,
        macros,
        {episode_id: Path(admission["source_episode"])},
        human_rows,
    )


def load_formal_online_r(root: Path) -> tuple[
    list[dict[str, Any]],
    tuple[ProductionAckMacro, ...],
    dict[str, Path],
    list[dict[str, Any]],
]:
    admission_ids = tuple(
        path.stem
        for path in sorted((root / "episodes").glob("*.json"))
        if json.loads(path.read_text(encoding="utf-8")).get("status")
        == "SEALED_COMMITTED"
    )
    require(admission_ids, "FORCERFT_ONLINE_REPLAY_ADMISSION_RECORD_COUNT")
    policy_rows: list[dict[str, Any]] = []
    macros: list[ProductionAckMacro] = []
    source_episodes: dict[str, Path] = {}
    human_rows: list[dict[str, Any]] = []
    for admission_id in admission_ids:
        policy, episode_macros, sources, human = load_formal_online_episode(
            root, admission_id
        )
        require(
            not set(source_episodes).intersection(sources),
            "FORCERFT_ONLINE_REPLAY_ADMISSION_EPISODE_DUPLICATE",
        )
        policy_rows.extend(policy)
        macros.extend(episode_macros)
        source_episodes.update(sources)
        human_rows.extend(human)
    all_rows = [*policy_rows, *human_rows]
    require(len(all_rows) >= 100, "FORCERFT_ONLINE_REPLAY_TRAINING_STARTS")
    require(
        len({row["identity"]["transition_uid"] for row in all_rows})
        == len(all_rows),
        "FORCERFT_ONLINE_REPLAY_UID_DUPLICATE",
    )
    require(
        macros
        and (
            any(macro.transition["outcome"]["terminated"] for macro in macros)
            or any(row["outcome"]["terminated"] for row in human_rows)
        ),
        "FORCERFT_ONLINE_REPLAY_MACRO_TERMINAL_MISSING",
    )
    return policy_rows, tuple(macros), source_episodes, human_rows


def count_sealed_critic_td_valid_transitions(root: Path) -> int:
    count = 0
    sealed_episodes = _sealed_episode_ids(root)
    for path in sorted((root / "replay").glob("*.json")):
        envelope = json.loads(path.read_text(encoding="utf-8"))
        if envelope.get("episode_sealed") is not True:
            continue
        row = envelope.get("payload", {})
        if str(row.get("identity", {}).get("episode_id", "")) not in sealed_episodes:
            continue
        source = row.get(
            "action_source",
            row.get("action_authority", {}).get("executed_action_source"),
        )
        eligibility = _transition_eligibility(row, source)
        count += bool(
            row.get("classification")
            == "recorded_live_policy_execution_smoke"
            and row.get("schema_version")
            == ACK_RESIDUAL_TRANSITION_SCHEMA_VERSION
            and row.get("online_semantics_version")
            == ONLINE_SEMANTICS_VERSION
            and source in {"policy", "human"}
            and row.get("action_authority", {}).get("executed_action_source")
            == source
            and eligibility.get("formal_replay") is True
            and eligibility.get("formal_training_replay_eligible") is True
            and eligibility.get("real_online_r") is True
            and eligibility.get("replay_membership") == "R_online"
            and eligibility.get("td_eligible") is True
        )
    return count


def count_sealed_autonomous_policy_transitions(root: Path) -> int:
    """Compatibility alias; warm-up is now based on all valid online ACK rows."""

    return count_sealed_critic_td_valid_transitions(root)


@lru_cache(maxsize=512)
def _decode_path(path: str) -> np.ndarray:
    from PIL import Image

    with Image.open(path) as image:
        value = np.asarray(image.convert("RGB"), dtype=np.uint8)
    require(value.shape == (480, 640, 3), "FORCERFT_ONLINE_REPLAY_IMAGE_SHAPE")
    return np.ascontiguousarray(value.transpose(2, 0, 1))


class FormalReplay:
    def __init__(self, macros, source_episodes: Mapping[str, Path], normalizer) -> None:
        self.macros = tuple(macros)
        self.source_episodes = dict(source_episodes)
        self.normalizer = normalizer

    def _sample(
        self, observation: Mapping[str, Any], identity: str, episode_id: str
    ) -> dict[str, Any]:
        source_episode = self.source_episodes[episode_id]
        return {
            "camera1": _decode_path(str(source_episode / observation["camera_external"]["blob_reference"])),
            "camera2": _decode_path(str(source_episode / observation["camera_wrist"]["blob_reference"])),
            "state7": self.normalizer.state7.apply(np.asarray(observation["state7_absolute"], dtype=np.float64)).astype(np.float32),
            "wrench6": self.normalizer.wrench6.apply(np.asarray(observation["wrench6_calibrated_tcp"], dtype=np.float64)).astype(np.float32),
            "task": TASK,
            "sample_identity": identity,
        }

    def materialize(self, index: int) -> dict[str, Any]:
        macro = self.macros[index]
        row = macro.transition
        state = np.asarray(row["observation"]["state7_absolute"], dtype=np.float64)
        absolute = macro.behavior.accepted_absolute_action_k7.copy()
        for slot in range(3):
            width = float(absolute[slot, 6])
            require(np.isclose(width, 0.0, atol=1e-6) or np.isclose(width, 0.085, atol=1e-6), "FORCERFT_ONLINE_REPLAY_GRIPPER_ENDPOINT")
            absolute[slot, 6] = 0.0 if width < 0.0425 else 0.085
        behavior = replace(
            macro.behavior, accepted_absolute_action_k7=absolute
        )
        action = normalized_ack_behavior_action(
            behavior,
            anchor_state7=state,
            normalize_delta7=self.normalizer.delta_action7.apply,
        ).astype(np.float32)
        uid = str(row["identity"]["transition_uid"])
        episode_id = str(row["identity"]["episode_id"])
        return {
            "current": self._sample(row["observation"], f"R:{uid}:current", episode_id),
            "next": self._sample(row["next_observation"], f"R:{uid}:next", episode_id),
            "behavior_action": action,
            "behavior_mask": np.asarray(
                macro.behavior.behavior_mask, dtype=np.bool_
            ),
            "behavior_provenance": macro.ack_provenance,
            "critic_action_contract_version": macro.behavior.contract_version,
            "reward": float(row["outcome"]["reward"]),
            "terminated": bool(row["outcome"]["terminated"]),
            "truncated": bool(row["outcome"]["truncated"]),
            "bootstrap": bool(row["outcome"]["bootstrap_mask"]),
            "discount": float(row["outcome"]["discount"]),
            "identity": f"R:{uid}",
            "expert": False,
            "action_source": "policy",
            "td_eligible": True,
            "fm_eligible": False,
            "actor_q_valid": macro.actor_q_eligibility.valid,
            "actor_q_eligibility_reason": macro.actor_q_eligibility.reason,
        }


class HumanCorrectionReplay:
    def __init__(self, rows, source_episodes: Mapping[str, Path], normalizer) -> None:
        self.macros = build_ack_macros(rows)
        self.rows = tuple(macro.transition for macro in self.macros)
        self.source_episodes = dict(source_episodes)
        self.normalizer = normalizer

    def _sample(
        self, observation: Mapping[str, Any], identity: str, episode_id: str
    ) -> dict[str, Any]:
        source_episode = self.source_episodes[episode_id]
        return {
            "camera1": _decode_path(
                str(source_episode / observation["camera_external"]["blob_reference"])
            ),
            "camera2": _decode_path(
                str(source_episode / observation["camera_wrist"]["blob_reference"])
            ),
            "state7": self.normalizer.state7.apply(
                np.asarray(observation["state7_absolute"], dtype=np.float64)
            ).astype(np.float32),
            "wrench6": self.normalizer.wrench6.apply(
                np.asarray(
                    observation["wrench6_calibrated_tcp"], dtype=np.float64
                )
            ).astype(np.float32),
            "task": TASK,
            "sample_identity": identity,
        }

    def materialize(self, index: int) -> dict[str, Any]:
        from forcesmolvla.action_delta import ActionDeltaProcessor

        macro = self.macros[index]
        row = macro.transition
        target = np.asarray(row["human_action_target_h50"], dtype=np.float64)
        representation = row.get("absolute_action_rotation_representation")
        if representation is None:
            target = legacy_absolute_action7_to_rpy_xyz(target)
        elif representation != ABSOLUTE_ACTION_ROTATION_REPRESENTATION:
            raise RuntimeError(
                "FORCERFT_ONLINE_ACTION_ROTATION_REPRESENTATION_INVALID"
            )
        feature_mask = np.asarray(
            row["human_action_valid_mask_h50"], dtype=np.bool_
        )
        state = np.asarray(row["observation"]["state7_absolute"], dtype=np.float64)
        absolute = np.where(feature_mask, target, state[None, :])
        action_target = self.normalizer.delta_action7.apply(
            ActionDeltaProcessor.to_delta(absolute, state)
        ).astype(np.float32)
        action_target[~feature_mask] = 0.0
        human_behavior_action_k3 = normalized_ack_behavior_action(
            macro.behavior,
            anchor_state7=state,
            normalize_delta7=self.normalizer.delta_action7.apply,
        ).astype(np.float32)
        human_behavior_mask_k3 = np.asarray(
            macro.behavior.behavior_mask, dtype=np.bool_
        )
        uid = str(row["identity"]["transition_uid"])
        episode_id = str(row["identity"]["episode_id"])
        fm_eligible = bool(row["eligibility"]["fm_eligible"])
        return {
            "current": self._sample(
                row["observation"], f"H:{uid}:current", episode_id
            ),
            "next": self._sample(
                row["next_observation"], f"H:{uid}:next", episode_id
            ),
            "behavior_action": human_behavior_action_k3,
            "behavior_mask": human_behavior_mask_k3,
            "reward": float(row["outcome"]["reward"]),
            "terminated": bool(row["outcome"]["terminated"]),
            "truncated": bool(row["outcome"]["truncated"]),
            "bootstrap": bool(row["outcome"]["bootstrap_mask"]),
            "discount": float(row["outcome"]["discount"]),
            "identity": f"H:{uid}",
            "expert": fm_eligible,
            "action_source": "human",
            "td_eligible": True,
            "fm_eligible": fm_eligible,
            "actor_q_valid": macro.actor_q_eligibility.valid,
            "actor_q_eligibility_reason": macro.actor_q_eligibility.reason,
            "action_target": action_target,
            "action_valid_mask": feature_mask,
            "human_action_target_h50": action_target,
            "human_action_valid_mask_h50": feature_mask,
            "human_behavior_action_k3": human_behavior_action_k3,
            "human_behavior_mask_k3": human_behavior_mask_k3,
            "critic_action_contract_version": macro.behavior.contract_version,
        }


@dataclass(frozen=True)
class ResidualTransitionBatch:
    state7: torch.Tensor
    wrench6: torch.Tensor
    wrench_delta6: torch.Tensor
    base_action_k6: torch.Tensor
    behavior_residual_k6: torch.Tensor
    action_mask: torch.Tensor
    next_state7: torch.Tensor
    next_wrench6: torch.Tensor
    next_wrench_delta6: torch.Tensor
    next_base_action_k6: torch.Tensor
    next_action_mask: torch.Tensor
    next_base_valid: torch.Tensor
    reward: torch.Tensor
    terminated: torch.Tensor
    truncated: torch.Tensor
    actor_q_valid: torch.Tensor
    human_residual_target6: torch.Tensor
    human_residual_valid: torch.Tensor


class OnlineResidualReplay:
    """Materialize only normalized low-dimensional tensors from sealed ACK rows."""

    def __init__(self, macros: Iterable[ProductionAckMacro], normalizer) -> None:
        self.normalizer = normalizer
        self.next_base_missing_rows = 0
        self.quarantined_current_schema_rows = 0
        self.rows = tuple(self._materialize_all(tuple(macros)))

    def append_macros(
        self, macros: Iterable[ProductionAckMacro]
    ) -> dict[str, int]:
        additions = tuple(macros)
        if not additions:
            return {}
        existing_episode_ids = {row["episode_id"] for row in self.rows}
        new_episode_ids = {
            str(macro.transition["identity"]["episode_id"])
            for macro in additions
        }
        require(
            not existing_episode_ids.intersection(new_episode_ids),
            "FORCERFT_ONLINE_REPLAY_EPISODE_ALREADY_LOADED",
        )
        materialized = tuple(self._materialize_all(additions))
        self.rows = (*self.rows, *materialized)
        return {
            episode_id: sum(
                int(row["episode_id"] == episode_id) for row in materialized
            )
            for episode_id in new_episode_ids
        }

    @staticmethod
    def _raw_state(observation: Mapping[str, Any]) -> np.ndarray:
        return np.asarray(
            observation.get("state7_absolute", observation.get("state7")),
            dtype=np.float64,
        )

    @staticmethod
    def _raw_wrench(observation: Mapping[str, Any]) -> np.ndarray:
        return np.asarray(
            observation.get(
                "wrench6_calibrated_tcp", observation.get("wrench6")
            ),
            dtype=np.float64,
        )

    def _normalized_observation(
        self, observation: Mapping[str, Any]
    ) -> tuple[np.ndarray, np.ndarray]:
        state = self.normalizer.state7.apply(self._raw_state(observation)).astype(
            np.float32
        )
        wrench = self.normalizer.wrench6.apply(
            self._raw_wrench(observation)
        ).astype(np.float32)
        require(
            state.shape == (7,)
            and wrench.shape == (6,)
            and np.isfinite(state).all()
            and np.isfinite(wrench).all(),
            "FORCERFT_ONLINE_LOW_DIM_OBSERVATION_INVALID",
        )
        return state, wrench

    def _decision_features(
        self, context: Mapping[str, Any]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        require(
            context.get("online_semantics_version") == ONLINE_SEMANTICS_VERSION
            and context.get("valid_for_residual_training") is True,
            "FORCERFT_ONLINE_REPLAY_DECISION_CONTEXT_INVALID",
        )
        raw_state = np.asarray(context.get("state7_absolute"), dtype=np.float64)
        raw_wrench = np.asarray(
            context.get("wrench6_calibrated_tcp"), dtype=np.float64
        )
        raw_wrench_delta = np.asarray(
            context.get("wrench_delta6_calibrated_tcp_100ms"),
            dtype=np.float64,
        )
        base_absolute = np.asarray(
            context.get("base_absolute_action7"), dtype=np.float64
        )
        require(
            raw_state.shape == (7,)
            and raw_wrench.shape == (6,)
            and raw_wrench_delta.shape == (6,)
            and base_absolute.shape == (7,)
            and np.isfinite(raw_state).all()
            and np.isfinite(raw_wrench).all()
            and np.isfinite(raw_wrench_delta).all()
            and np.isfinite(base_absolute).all(),
            "FORCERFT_ONLINE_REPLAY_DECISION_FEATURE_INVALID",
        )
        state = self.normalizer.state7.apply(raw_state).astype(np.float32)
        wrench = self.normalizer.wrench6.apply(raw_wrench).astype(np.float32)
        wrench_std = np.asarray(self.normalizer.wrench6.std, dtype=np.float64)
        require(
            wrench_std.shape == (6,)
            and np.isfinite(wrench_std).all()
            and np.all(wrench_std > 0.0),
            "FORCERFT_ONLINE_REPLAY_WRENCH_NORMALIZER_INVALID",
        )
        wrench_delta = (raw_wrench_delta / wrench_std).astype(np.float32)
        base, _accepted, _residual = normalized_behavior_residual(
            base_absolute_k7=base_absolute[None, :],
            accepted_absolute_k7=base_absolute[None, :],
            decision_state7=raw_state,
            normalize_delta7=self.normalizer.delta_action7.apply,
            valid_mask=np.ones(1, dtype=np.bool_),
        )
        expected = {
            "normalized_state7": state,
            "normalized_wrench6": wrench,
            "normalized_wrench_delta6": wrench_delta,
            "base_normalized_action6": base[0, :6],
        }
        for name, value in expected.items():
            persisted = context.get(name)
            if persisted is not None:
                candidate = np.asarray(persisted, dtype=np.float32)
                require(
                    candidate.shape == value.shape
                    and np.allclose(candidate, value, rtol=1.0e-5, atol=1.0e-6),
                    "FORCERFT_ONLINE_REPLAY_NORMALIZER_CONTEXT_MISMATCH",
                )
        return state, wrench, wrench_delta, base[0]

    def _materialize_all(
        self, macros: tuple[ProductionAckMacro, ...]
    ) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        ordered = sorted(
            macros,
            key=lambda macro: int(
                macro.transition.get("residual_decision_context", {}).get(
                    "decision_monotonic_ns", 0
                )
            ),
        )
        for macro in ordered:
            row = macro.transition
            context = row.get("residual_decision_context")
            raw_base = (
                context.get("base_absolute_action7")
                if isinstance(context, Mapping)
                else None
            )
            try:
                base_evidence = np.asarray(raw_base, dtype=np.float64)
            except (TypeError, ValueError):
                base_evidence = np.empty(0, dtype=np.float64)
            if (
                row.get("online_semantics_version") != ONLINE_SEMANTICS_VERSION
                or not isinstance(context, Mapping)
                or context.get("valid_for_residual_training") is not True
                or base_evidence.shape != (7,)
                or not np.isfinite(base_evidence).all()
            ):
                self.quarantined_current_schema_rows += 1
                continue
            state, wrench, wrench_delta, base_one = self._decision_features(
                context
            )
            base_absolute = np.asarray(
                row.get("base_absolute_action_k7"), dtype=np.float64
            )
            accepted_absolute = np.asarray(
                row.get("accepted_absolute_action_k7"), dtype=np.float64
            )
            mask = np.asarray(macro.behavior.behavior_mask, dtype=np.bool_)
            require(
                base_absolute.shape == (3, 7)
                and np.allclose(base_absolute, base_absolute[0])
                and np.allclose(base_absolute[0], context["base_absolute_action7"]),
                "FORCERFT_ONLINE_REPLAY_FROZEN_BASE_EVIDENCE_INVALID",
            )
            base, _accepted, residual = normalized_behavior_residual(
                base_absolute_k7=base_absolute,
                accepted_absolute_k7=accepted_absolute,
                decision_state7=np.asarray(context["state7_absolute"], dtype=np.float64),
                normalize_delta7=self.normalizer.delta_action7.apply,
                valid_mask=mask,
            )
            require(
                np.allclose(base[0], base_one, rtol=1.0e-5, atol=1.0e-6),
                "FORCERFT_ONLINE_REPLAY_BASE_CONTEXT_MISMATCH",
            )
            outcome = row["outcome"]
            terminal_boundary = bool(
                outcome["terminated"] or outcome["truncated"]
            )
            next_context = row.get("next_residual_decision_context")
            if next_context is None:
                if not terminal_boundary:
                    self.next_base_missing_rows += 1
                    continue
                next_state, next_wrench = self._normalized_observation(
                    row["next_observation"]
                )
                next_wrench_delta = np.zeros(6, dtype=np.float32)
                next_base = base.copy()
                next_base_valid = False
            else:
                require(
                    not terminal_boundary and isinstance(next_context, Mapping),
                    "FORCERFT_ONLINE_REPLAY_DECISION_SUCCESSOR_INVALID",
                )
                (
                    next_state,
                    next_wrench,
                    next_wrench_delta,
                    next_base_one,
                ) = self._decision_features(next_context)
                next_base = np.repeat(next_base_one[None, :], 3, axis=0)
                next_base_valid = True
            human_valid = bool(
                row.get("action_source") == "human"
                and row.get("human_residual_valid") is True
            )
            result.append(
                {
                    "state7": state,
                    "wrench6": wrench,
                    "wrench_delta6": wrench_delta,
                    "wrench_delta_interval_ns": int(
                        context.get("wrench_delta_interval_ns", 0)
                    ),
                    "base_action_k6": base[..., :6],
                    "behavior_residual_k6": residual,
                    "action_mask": mask,
                    "next_state7": next_state,
                    "next_wrench6": next_wrench,
                    "next_wrench_delta6": next_wrench_delta,
                    "next_wrench_delta_interval_ns": (
                        0
                        if next_context is None
                        else int(next_context.get("wrench_delta_interval_ns", 0))
                    ),
                    "next_base_action_k6": next_base[..., :6],
                    "next_action_mask": np.ones(3, dtype=np.bool_),
                    "next_base_valid": next_base_valid,
                    "reward": float(outcome["reward"]),
                    "terminated": bool(outcome["terminated"]),
                    "truncated": bool(outcome["truncated"]),
                    "actor_q_valid": bool(
                        row.get("action_source") == "policy"
                        and macro.actor_q_eligibility.valid
                    ),
                    "human_residual_target6": (
                        residual[0].copy()
                        if human_valid
                        else np.zeros(6, dtype=np.float32)
                    ),
                    "human_residual_valid": human_valid,
                    "action_source": str(row["action_source"]),
                    "episode_id": str(row["identity"]["episode_id"]),
                }
            )
        return result

    def _batch(self, indices: torch.Tensor, device: torch.device) -> ResidualTransitionBatch:
        rows = [self.rows[int(index)] for index in indices]

        def tensor(name: str, dtype=torch.float32):
            return torch.as_tensor(
                np.stack([row[name] for row in rows]), dtype=dtype, device=device
            )

        return ResidualTransitionBatch(
            state7=tensor("state7"),
            wrench6=tensor("wrench6"),
            wrench_delta6=tensor("wrench_delta6"),
            base_action_k6=tensor("base_action_k6"),
            behavior_residual_k6=tensor("behavior_residual_k6"),
            action_mask=tensor("action_mask", torch.bool),
            next_state7=tensor("next_state7"),
            next_wrench6=tensor("next_wrench6"),
            next_wrench_delta6=tensor("next_wrench_delta6"),
            next_base_action_k6=tensor("next_base_action_k6"),
            next_action_mask=tensor("next_action_mask", torch.bool),
            next_base_valid=torch.tensor(
                [row["next_base_valid"] for row in rows],
                dtype=torch.bool,
                device=device,
            ),
            reward=torch.tensor(
                [row["reward"] for row in rows], dtype=torch.float32, device=device
            ),
            terminated=torch.tensor(
                [row["terminated"] for row in rows], dtype=torch.bool, device=device
            ),
            truncated=torch.tensor(
                [row["truncated"] for row in rows], dtype=torch.bool, device=device
            ),
            actor_q_valid=torch.tensor(
                [row["actor_q_valid"] for row in rows],
                dtype=torch.bool,
                device=device,
            ),
            human_residual_target6=tensor("human_residual_target6"),
            human_residual_valid=torch.tensor(
                [row["human_residual_valid"] for row in rows],
                dtype=torch.bool,
                device=device,
            ),
        )

    def sample(
        self,
        batch_size: int,
        *,
        device: torch.device,
        seed: int,
        human_only: bool = False,
        policy_only: bool = False,
        actor_q_valid_only: bool = False,
    ) -> ResidualTransitionBatch | None:
        require(
            not (human_only and policy_only),
            "FORCERFT_REPLAY_SOURCE_FILTER_CONFLICT",
        )
        population = [
            index
            for index, row in enumerate(self.rows)
            if (not human_only or row["human_residual_valid"])
            and (not policy_only or row["action_source"] == "policy")
            and (not actor_q_valid_only or row["actor_q_valid"])
        ]
        if not population:
            return None
        generator = random.Random(int(seed))
        by_episode: dict[str, list[int]] = {}
        for index in population:
            by_episode.setdefault(self.rows[index]["episode_id"], []).append(index)
        groups = list(by_episode.values())
        group_order = list(range(len(groups)))
        generator.shuffle(group_order)
        indices: list[int] = []
        if len(population) >= batch_size:
            counts = [0] * len(groups)
            selected_count = 0
            while selected_count < batch_size:
                for group_index in group_order:
                    if counts[group_index] < len(groups[group_index]):
                        counts[group_index] += 1
                        selected_count += 1
                        if selected_count == batch_size:
                            break
            for group_index, count in enumerate(counts):
                indices.extend(generator.sample(groups[group_index], count))
            generator.shuffle(indices)
        else:
            for _ in range(batch_size):
                indices.append(generator.choice(generator.choice(groups)))
        indices = torch.tensor(indices, dtype=torch.long)
        return self._batch(indices, device)

    @property
    def critic_td_valid_rows(self) -> int:
        return len(self.rows)

    @property
    def actor_q_valid_rows(self) -> int:
        return sum(int(row["actor_q_valid"]) for row in self.rows)

    @property
    def human_residual_valid_rows(self) -> int:
        return sum(int(row["human_residual_valid"]) for row in self.rows)

    @property
    def nonzero_behavior_residual_rows(self) -> int:
        return sum(
            int(np.any(np.abs(row["behavior_residual_k6"]) > 1.0e-8))
            for row in self.rows
        )

    @property
    def critic_rows_per_episode(self) -> tuple[int, ...]:
        counts: dict[str, int] = {}
        for row in self.rows:
            counts[row["episode_id"]] = counts.get(row["episode_id"], 0) + 1
        return tuple(counts.values())


def _resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def _critic_observation(samples: list[dict[str, Any]], feature: torch.Tensor, device: torch.device):
    from forcesmolvla.rft.losses import CriticObservation

    return CriticObservation(
        torch.from_numpy(np.stack([item["camera1"] for item in samples])).to(device),
        torch.from_numpy(np.stack([item["camera2"] for item in samples])).to(device),
        feature[None, :].expand(len(samples), -1).clone(),
        torch.from_numpy(np.stack([item["state7"] for item in samples])).to(device),
        torch.from_numpy(np.stack([item["wrench6"] for item in samples])).to(device),
    ).validate()


def build_batch(rows: list[dict[str, Any]], actor, feature: torch.Tensor, device: torch.device) -> dict[str, Any]:
    from forcesmolvla.rft.batch import build_actor_batch

    require(
        all(row.get("td_eligible") is True for row in rows),
        "FORCERFT_ONLINE_REPLAY_BATCH_NOT_TD_ELIGIBLE",
    )
    rows = sorted(rows, key=lambda row: row["terminated"])
    current = [row["current"] for row in rows]
    following = [row["next"] for row in rows]
    return {
        "current_observation": _critic_observation(current, feature, device),
        "next_observation": _critic_observation(following, feature, device),
        "next_actor_batch": build_actor_batch(actor, following, device, include_action=False),
        "behavior_action": torch.from_numpy(np.stack([row["behavior_action"] for row in rows])).to(device),
        "behavior_mask": torch.from_numpy(
            np.stack(
                [
                    row.get("behavior_mask", np.ones(3, dtype=np.bool_))
                    for row in rows
                ]
            )
        ).to(device),
        "reward": torch.tensor([row["reward"] for row in rows], dtype=torch.float32, device=device),
        "terminated": torch.tensor([row["terminated"] for row in rows], dtype=torch.bool, device=device),
        "truncated": torch.tensor([row["truncated"] for row in rows], dtype=torch.bool, device=device),
        "bootstrap": torch.tensor([row["bootstrap"] for row in rows], dtype=torch.bool, device=device),
        "discount": torch.tensor([row["discount"] for row in rows], dtype=torch.float32, device=device),
        "td_eligible": torch.tensor([row["td_eligible"] for row in rows], dtype=torch.bool, device=device),
        "fm_eligible": torch.tensor([row["fm_eligible"] for row in rows], dtype=torch.bool, device=device),
        "identities": tuple(row["identity"] for row in rows),
    }
