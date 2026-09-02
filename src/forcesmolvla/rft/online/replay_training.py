#!/usr/bin/env python3
"""Replay materialization and batch primitives shared by ForceRFT training."""

from __future__ import annotations

from dataclasses import dataclass, replace
from functools import lru_cache
from io import BytesIO
import json
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping

import numpy as np
import torch
import yaml

from forcesmolvla.rft.online.transition_authority import (
    AcceptedAck,
    AckMacro,
    TransitionContractError,
    build_ack_behavior_macro,
    normalized_ack_behavior_action,
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

FORMAL_R_ROOT = ROOT / "outputs/task2/online"
TRAINING_CONFIG = ROOT / "configs/forcerft_actor_critic_training.development.yaml"
DATASET = ROOT / "datasets/task2_lerobotv3"
REWARD_TRANSITION_ROOT = ROOT / "artifacts/development/stage2/g1_frozen_detector_transition_view.v1"
SEED = 4404
TASK = "Pick up the purple ring and place it onto the red peg."


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
    return AcceptedAck(
        ack_id=str(row["identity"]["source_ack_id"]),
        receive_monotonic_ns=receive_ns,
        accepted_absolute_action7=tuple(
            float(value) for value in authority["accepted_absolute_action7"]
        ),
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

    groups: dict[tuple[object, ...], list[Mapping[str, Any]]] = {}
    for row in rows:
        if row.get("policy_lineage", {}).get("proposal", {}).get(
            "invalidated_by_takeover"
        ) is True:
            continue
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

    macros: list[ProductionAckMacro] = []
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
            macros.append(ProductionAckMacro(row, behavior, next_grid, provenance))
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


def load_formal_online_r(root: Path) -> tuple[
    list[dict[str, Any]],
    tuple[ProductionAckMacro, ...],
    dict[str, Path],
    list[dict[str, Any]],
]:
    admission_files = tuple(sorted((root / "admissions").glob("*.json")))
    require(admission_files, "FORCERFT_ONLINE_REPLAY_ADMISSION_RECORD_COUNT")
    sealed_episodes = _sealed_episode_ids(root)
    expected = 0
    source_episodes: dict[str, Path] = {}
    for path in admission_files:
        admission = json.loads(path.read_text(encoding="utf-8"))
        if str(admission.get("episode_id", "")) not in sealed_episodes:
            continue
        require(admission.get("policy_execution_smoke_bridge") == "PASS", "FORCERFT_ONLINE_REPLAY_BRIDGE_NOT_PASS")
        require(admission.get("source_episode_semantics") == {"formal_replay": False, "real_online_r": False}, "FORCERFT_ONLINE_REPLAY_SOURCE_SEMANTICS")
        require(
            (
                admission.get("operator_task_outcome"),
                admission.get("detector_outcome"),
            )
            in {("success", "success"), ("failure", "miss")},
            "FORCERFT_ONLINE_REPLAY_OUTCOME_CONFLICT",
        )
        episode_id = str(admission["episode_id"])
        require(episode_id not in source_episodes, "FORCERFT_ONLINE_REPLAY_ADMISSION_EPISODE_DUPLICATE")
        source_episodes[episode_id] = Path(admission["source_episode"])
        expected += int(admission["accepted_unique_r_transition_count"])

    policy_rows: list[dict[str, Any]] = []
    human_rows: list[dict[str, Any]] = []
    for path in sorted((root / "replay").glob("*.json")):
        envelope = json.loads(path.read_text(encoding="utf-8"))
        if envelope.get("episode_sealed") is not True:
            continue
        row = envelope["payload"]
        if str(row["identity"]["episode_id"]) not in source_episodes:
            continue
        source = row.get(
            "action_source",
            row.get("action_authority", {}).get("executed_action_source"),
        )
        eligibility = _transition_eligibility(row, source)
        fm_eligible = source == "human" and (
            row["outcome"].get("operator_task_outcome"),
            row["outcome"].get("detector_outcome"),
        ) == ("success", "success")
        require(
            row["classification"] == "recorded_live_policy_execution_smoke"
            and source in {"policy", "human"}
            and row["action_authority"]["executed_action_source"] == source
            and eligibility.get("formal_replay") is True
            and eligibility.get("formal_training_replay_eligible") is True
            and eligibility.get("real_online_r") is True
            and eligibility.get("replay_membership") == "R_online"
            and eligibility.get("td_eligible") is True
            and eligibility.get("fm_eligible") is fm_eligible,
            "FORCERFT_ONLINE_REPLAY_MEMBERSHIP",
        )
        row["action_source"] = source
        row.setdefault("expert", fm_eligible)
        row.setdefault("intervention", source == "human")
        if source == "human":
            target = np.asarray(
                row.get("human_action_target_h50"), dtype=np.float64
            )
            mask = np.asarray(
                row.get("human_action_valid_mask_h50"), dtype=np.bool_
            )
            require(
                row["expert"] is fm_eligible
                and row["intervention"] is True
                and target.shape == (50, 7)
                and mask.shape == (50, 7)
                and bool(mask.any())
                and np.all(np.isfinite(target)),
                "FORCERFT_ONLINE_HUMAN_EXPERT_TARGET_INVALID",
            )
            human_rows.append(row)
        else:
            require(
                row["expert"] is False and row["intervention"] is False,
                "FORCERFT_ONLINE_POLICY_REPLAY_SEMANTICS_INVALID",
            )
            row.setdefault("action_target", [[0.0] * 7 for _ in range(50)])
            row.setdefault(
                "action_valid_mask", [[False] * 7 for _ in range(50)]
            )
            policy_rows.append(row)
    all_rows = [*policy_rows, *human_rows]
    require(len(all_rows) == expected, "FORCERFT_ONLINE_REPLAY_ADMISSION_COUNT")
    require(
        len(policy_rows) >= 100,
        "FORCERFT_ONLINE_REPLAY_TRAINING_STARTS",
    )
    require(len({row["identity"]["transition_uid"] for row in all_rows}) == len(all_rows), "FORCERFT_ONLINE_REPLAY_UID_DUPLICATE")
    macros = build_ack_macros(policy_rows)
    require(
        macros
        and (
            any(macro.transition["outcome"]["terminated"] for macro in macros)
            or any(row["outcome"]["terminated"] for row in human_rows)
        ),
        "FORCERFT_ONLINE_REPLAY_MACRO_TERMINAL_MISSING",
    )
    return policy_rows, macros, source_episodes, human_rows


def count_sealed_autonomous_policy_transitions(root: Path) -> int:
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
        count += source == "policy" and eligibility.get("td_eligible") is True
    return count


@lru_cache(maxsize=512)
def _decode_path(path: str) -> np.ndarray:
    from PIL import Image

    with Image.open(path) as image:
        value = np.asarray(image.convert("RGB"), dtype=np.uint8)
    require(value.shape == (480, 640, 3), "FORCERFT_ONLINE_REPLAY_IMAGE_SHAPE")
    return np.ascontiguousarray(value.transpose(2, 0, 1))


def _decode_bytes(payload: bytes) -> np.ndarray:
    from PIL import Image

    with Image.open(BytesIO(payload)) as image:
        value = np.asarray(image.convert("RGB"), dtype=np.uint8)
    require(value.shape == (480, 640, 3), "FORCERFT_ONLINE_REPLAY_DEMO_IMAGE_SHAPE")
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
            "action_target": action_target,
            "action_valid_mask": feature_mask,
            "human_action_target_h50": action_target,
            "human_action_valid_mask_h50": feature_mask,
            "human_behavior_action_k3": human_behavior_action_k3,
            "human_behavior_mask_k3": human_behavior_mask_k3,
            "critic_action_contract_version": macro.behavior.contract_version,
        }


class DemoReplay:
    """Read the already converted online-training demonstration replay."""

    COLUMNS = (
        "observation.images.camera1",
        "observation.images.camera2",
        "observation.state",
        "observation.wrench",
    )

    def __init__(self, normalizer) -> None:
        from forcesmolvla.rft.losses import load_authorized_reward_train_transitions

        self.rows = load_authorized_reward_train_transitions(
            REWARD_TRANSITION_ROOT
        ).to_pylist()
        self.population = tuple(
            index for index, row in enumerate(self.rows)
            if any(row["executed_action_mask"])
        )
        require(self.population, "FORCERFT_ONLINE_REPLAY_DEMO_POPULATION_EMPTY")
        conversion = json.loads((DATASET / "conversion_manifest.json").read_text(encoding="utf-8"))
        self.tasks = {item["raw_episode_id"]: item["task"] for item in conversion["episodes"]}
        self.normalizer = normalizer
        self.raw: dict[tuple[str, int], dict[str, Any]] = {}

    def prefetch(self, schedule: Iterable[Iterable[int]]) -> None:
        import pyarrow.parquet as pq

        requested: dict[str, set[int]] = {}
        for batch in schedule:
            for index in batch:
                row = self.rows[index]
                for key in ("observation_row_reference", "next_observation_row_reference"):
                    reference = row[key]
                    requested.setdefault(reference["data_relative_path"], set()).add(int(reference["row_index"]))
        for position, (relative, indices) in enumerate(sorted(requested.items()), start=1):
            table = pq.read_table(DATASET / relative, columns=list(self.COLUMNS))
            for index in indices:
                self.raw[(relative, index)] = table.slice(index, 1).to_pylist()[0]
            del table
            if position % 10 == 0 or position == len(requested):
                print(f"[warmup] prefetched demonstration files {position}/{len(requested)}", file=sys.stderr, flush=True)

    def _sample(self, reference: Mapping[str, Any], identity: str, task: str) -> dict[str, Any]:
        source = self.raw[(reference["data_relative_path"], int(reference["row_index"]))]
        return {
            "camera1": _decode_bytes(source["observation.images.camera1"]["bytes"]),
            "camera2": _decode_bytes(source["observation.images.camera2"]["bytes"]),
            "state7": self.normalizer.state7.apply(np.asarray(source["observation.state"], dtype=np.float64)).astype(np.float32),
            "wrench6": self.normalizer.wrench6.apply(np.asarray(source["observation.wrench"], dtype=np.float64)).astype(np.float32),
            "task": task,
            "sample_identity": identity,
        }

    def materialize(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        identity = f"D:{row['episode_id']}:{row['transition_index']}"
        behavior_mask = np.asarray(row["executed_action_mask"], dtype=np.bool_)
        executed = np.asarray(
            row["normalized_delta_action_exec_flat"], dtype=np.float32
        ).reshape(-1, 7)
        require(
            executed.shape == (int(behavior_mask.sum()), 7),
            "FORCERFT_ONLINE_REPLAY_DEMO_ACTION_SHAPE",
        )
        action = np.zeros((3, 7), dtype=np.float32)
        action[behavior_mask] = executed
        return {
            "current": self._sample(row["observation_row_reference"], identity + ":current", self.tasks[row["episode_id"]]),
            "next": self._sample(row["next_observation_row_reference"], identity + ":next", self.tasks[row["episode_id"]]),
            "behavior_action": action,
            "behavior_mask": behavior_mask,
            "critic_action_contract_version": row.get(
                "critic_action_contract_version",
                CRITIC_ACTION_CONTRACT.version,
            ),
            "reward": float(row["reward"]),
            "terminated": bool(row["terminated"]),
            "truncated": bool(row.get("truncated", False)),
            "bootstrap": bool(row["bootstrap_mask"]),
            "discount": float(row["discount"]),
            "identity": identity,
            "expert": True,
            "action_source": "offline_demonstration",
            "td_eligible": True,
            "fm_eligible": True,
        }


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
