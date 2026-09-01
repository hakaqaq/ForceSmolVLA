#!/usr/bin/env python3
"""Replay materialization and batch primitives shared by ForceRFT training."""

from __future__ import annotations

from dataclasses import dataclass
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
    causal_zoh_ack_macro,
    normalized_ack_behavior_action,
    validate_macro_grid,
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
    index = (anchor * 30 + 500_000_000) // 1_000_000_000
    ticks = tuple(
        int(((index + offset) * 1_000_000_000 + 15) // 30)
        for offset in range(4)
    )
    grid = validate_macro_grid(ticks[:3])
    if anchor != grid[0] or int(
        row["next_observation"]["materialized_timestamp_monotonic_ns"]
    ) != ticks[3]:
        raise RuntimeError("FORCERFT_ONLINE_REPLAY_TRANSITION_HORIZON_NOT_100MS")
    return grid, ticks[3]


def _accepted_ack(row: Mapping[str, Any]) -> AcceptedAck:
    authority = row["action_authority"]
    pose_ack = authority["pose_ack"]
    gripper_origin = authority["gripper_terminal_provenance"]
    command_id = str(gripper_origin["origin_action_goal_id"])
    return AcceptedAck(
        ack_id=str(row["identity"]["source_ack_id"]),
        receive_monotonic_ns=int(pose_ack["upper_receive_monotonic_ns"]),
        accepted_absolute_action7=tuple(
            float(value) for value in authority["accepted_absolute_action7"]
        ),
        gripper_command_id=command_id,
        gripper_ack_command_id=command_id,
        slot_owner="policy",
        accepted_action_source="policy",
        intervention=False,
        accepted=bool(pose_ack["accepted"]),
        workspace_clipped=bool(
            authority.get("safety_arbitration", {}).get("workspace_clipped", False)
        ),
    )


def build_ack_macros(
    rows: Iterable[Mapping[str, Any]], *, max_ack_age_ms: float = 100.0
) -> tuple[ProductionAckMacro, ...]:
    """Build 100 ms K=3 behavior macros from real ACKs on the 30 Hz grid."""

    groups: dict[tuple[str, tuple[int, int, int]], list[Mapping[str, Any]]] = {}
    for row in rows:
        if row.get("policy_lineage", {}).get("proposal", {}).get(
            "invalidated_by_takeover"
        ) is True:
            continue
        key = (str(row["identity"].get("episode_id", "single")), _generation(row))
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
            try:
                behavior = causal_zoh_ack_macro(
                    acknowledgements, grid, max_ack_age_ms=max_ack_age_ms
                )
            except TransitionContractError as error:
                if str(error) == "ONLINE_REPLAY_ACK_MISSING_OR_STALE":
                    continue
                raise
            provenance = tuple(
                {
                    "ack_id": ack_id,
                    "receive_monotonic_ns": int(
                        by_ack_id[ack_id]["action_authority"]["pose_ack"][
                            "upper_receive_monotonic_ns"
                        ]
                    ),
                    "chunk_id": str(
                        by_ack_id[ack_id]["policy_lineage"]["selection"]["chunk_id"]
                    ),
                    "action_index": int(
                        by_ack_id[ack_id]["policy_lineage"]["selection"]["action_index"]
                    ),
                    "generation": dict(by_ack_id[ack_id]["generation"]),
                }
                for ack_id in behavior.ack_ids
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
        require(
            row["classification"] == "recorded_live_policy_execution_smoke"
            and source in {"policy", "human"}
            and row["action_authority"]["executed_action_source"] == source
            and row["eligibility"] == {
                "formal_replay": True,
                "formal_training_replay_eligible": True,
                "real_online_r": True,
                "replay_membership": "R_online",
            },
            "FORCERFT_ONLINE_REPLAY_MEMBERSHIP",
        )
        row["action_source"] = source
        row.setdefault("expert", source == "human")
        row.setdefault("intervention", source == "human")
        if source == "human":
            target = np.asarray(row.get("human_action_target"), dtype=np.float64)
            mask = np.asarray(
                row.get("human_action_valid_mask"), dtype=np.bool_
            )
            require(
                row["expert"] is True
                and row["intervention"] is True
                and target.shape == (50, 7)
                and mask.shape == (50, 7)
                and bool(mask.any())
                and np.all(np.isfinite(target)),
                "FORCERFT_ONLINE_HUMAN_EXPERT_TARGET_INVALID",
            )
            row["action_target"] = target.tolist()
            row["action_valid_mask"] = mask.tolist()
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
        count += source == "policy"
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
        behavior = AckMacro(
            grid_monotonic_ns=macro.behavior.grid_monotonic_ns,
            ack_ids=macro.behavior.ack_ids,
            gripper_command_ids=macro.behavior.gripper_command_ids,
            gripper_ack_command_ids=macro.behavior.gripper_ack_command_ids,
            accepted_absolute_action_k7=absolute,
            slot_owner=macro.behavior.slot_owner,
            workspace_clip_flags=macro.behavior.workspace_clip_flags,
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
            "behavior_provenance": macro.ack_provenance,
            "reward": float(row["outcome"]["reward"]),
            "terminated": bool(row["outcome"]["terminated"]),
            "truncated": bool(row["outcome"]["truncated"]),
            "bootstrap": bool(row["outcome"]["bootstrap_mask"]),
            "discount": float(row["outcome"]["discount"]),
            "identity": f"R:{uid}",
        }


class HumanCorrectionReplay:
    def __init__(self, rows, source_episodes: Mapping[str, Path], normalizer) -> None:
        self.rows = tuple(rows)
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

        row = self.rows[index]
        target = np.asarray(row["action_target"], dtype=np.float64)
        feature_mask = np.asarray(row["action_valid_mask"], dtype=np.bool_)
        state = np.asarray(row["observation"]["state7_absolute"], dtype=np.float64)
        absolute = np.where(feature_mask, target, state[None, :])
        action_target = self.normalizer.delta_action7.apply(
            ActionDeltaProcessor.to_delta(absolute, state)
        ).astype(np.float32)
        action_target[~feature_mask] = 0.0
        behavior_mask = feature_mask[:3].any(axis=1)
        require(bool(behavior_mask.any()), "FORCERFT_ONLINE_HUMAN_TD_ACTION_EMPTY")
        uid = str(row["identity"]["transition_uid"])
        episode_id = str(row["identity"]["episode_id"])
        return {
            "current": self._sample(
                row["observation"], f"H:{uid}:current", episode_id
            ),
            "next": self._sample(
                row["next_observation"], f"H:{uid}:next", episode_id
            ),
            "behavior_action": action_target[:3],
            "behavior_mask": behavior_mask,
            "reward": float(row["outcome"]["reward"]),
            "terminated": bool(row["outcome"]["terminated"]),
            "truncated": bool(row["outcome"]["truncated"]),
            "bootstrap": bool(row["outcome"]["bootstrap_mask"]),
            "discount": float(row["outcome"]["discount"]),
            "identity": f"H:{uid}",
            "expert": True,
            "action_source": "human",
            "action_target": action_target,
            "action_valid_mask": feature_mask,
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
            if all(row["executed_action_mask"])
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
        action = np.asarray(row["normalized_delta_action_exec_flat"], dtype=np.float32).reshape(3, 7)
        require(action.shape == (3, 7), "FORCERFT_ONLINE_REPLAY_DEMO_ACTION_SHAPE")
        return {
            "current": self._sample(row["observation_row_reference"], identity + ":current", self.tasks[row["episode_id"]]),
            "next": self._sample(row["next_observation_row_reference"], identity + ":next", self.tasks[row["episode_id"]]),
            "behavior_action": action,
            "reward": float(row["reward"]),
            "terminated": bool(row["terminated"]),
            "truncated": bool(row.get("truncated", False)),
            "bootstrap": bool(row["bootstrap_mask"]),
            "discount": float(row["discount"]),
            "identity": identity,
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
        "identities": tuple(row["identity"] for row in rows),
    }
