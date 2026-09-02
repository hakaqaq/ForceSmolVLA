"""Offline demonstration transition primitives with external reward labels."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import struct
from typing import Callable, Iterator, Literal

import numpy as np

from forcesmolvla.action_delta import ActionDeltaProcessor
from forcesmolvla.rft.critic_action_adapter_v2 import CRITIC_ACTION_CONTRACT
from forcesmolvla.training_data import prepare_training_sample


HORIZON = 50
EXECUTED_ACTION_SLOTS = 3
ANCHOR_STRIDE = 3
GAMMA = 0.99
PROVENANCE_KEYS = (
    "provenance.state_pose_age_ms",
    "provenance.camera1_age_ms",
    "provenance.camera2_age_ms",
    "provenance.intercamera_skew_ms",
    "provenance.pose_age_ms",
    "provenance.action_ack_age_ms",
    "provenance.pose_source_stamp_ns",
    "provenance.wrench_raw_source_stamp_ns",
    "provenance.wrench_filter_output_stamp_ns",
    "provenance.validity_bits",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def dataset_tree_sha256(
    root: Path, *, progress: Callable[[int, int, str], None] | None = None
) -> dict:
    root = root.resolve()
    files = sorted(path for path in root.rglob("*") if path.is_file())
    records = []
    total_size = 0
    for index, path in enumerate(files, start=1):
        relative = path.relative_to(root).as_posix()
        size = path.stat().st_size
        records.append(
            {"relative_path": relative, "file_size": size, "sha256": sha256_file(path)}
        )
        total_size += size
        if progress is not None:
            progress(index, len(files), relative)
    return {
        "algorithm": "sha256(canonical_json(sorted(relative_path,file_size,sha256)))",
        "sha256": canonical_sha256(records),
        "file_count": len(records),
        "total_file_size": total_size,
    }


def validate_reward_spec(payload: dict) -> None:
    expected = {
        "artifact_status": "development_only",
        "formal_reward_spec": "unapproved",
        "real_g1_generation_permitted": False,
        "terminal_and_reward_source": "external_frozen_episode_frame_reward_sidecar",
        "terminal_inference": "forbidden",
        "gamma_per_policy_decision": GAMMA,
        "mc_return_definition": "deferred_until_external_reward_labels_are_frozen",
    }
    if payload.get("schema_version") != "1.1" or any(
        payload.get(key) != value for key, value in expected.items()
    ):
        raise RuntimeError("G1_REWARD_SPEC_SEMANTICS_DRIFT")


def validate_action_contract(payload: dict) -> None:
    if (
        payload.get("schema_version") != "1.0"
        or payload.get("artifact_status") != "development_only"
        or payload.get("formal_eligible") is not False
        or payload.get("critic_action_input_dim") != 7
        or payload.get("critic_action_slots") != EXECUTED_ACTION_SLOTS
        or payload.get("critic_action_shape") != [EXECUTED_ACTION_SLOTS, 7]
        or payload.get("critic_duration_mode") != "fixed_k"
        or payload.get("partial_action_interface") is not False
        or payload.get("critic_receives_terminal_derived_duration_or_mask") is not False
        or payload.get("all_critic_action_slots_valid") is not True
        or payload.get("actor_q_guided_action_dims") != list(range(6))
        or payload.get("actor_q_gradient_mask_per_slot") != [1, 1, 1, 1, 1, 1, 0]
        or payload.get("gripper_objective") != "flow_matching_only"
        or payload.get("gripper_q_gradient") is not False
        or payload.get("public_inference_contract_unchanged") is not True
    ):
        raise RuntimeError("G1_ACTION_CONTRACT_SEMANTICS_DRIFT")


def validate_outcome_labels(
    payload: dict,
    *,
    conversion_episodes: list[dict],
    episode_lengths: dict[int, int],
) -> list[dict]:
    if (
        payload.get("schema_version") != "1.0"
        or payload.get("artifact_status") != "frozen_development_input"
        or payload.get("approval_status") != "frozen"
        or payload.get("label_scope") != "episode_frame_reward"
        or payload.get("terminal_inference") != "forbidden"
        or payload.get("terminal_source") != "external_annotation"
    ):
        raise RuntimeError("OFFLINE_DEMO_REPLAY_EXTERNAL_REWARD_LABELS_NOT_FROZEN")
    labels = payload.get("episodes")
    if not isinstance(labels, list) or len(labels) != 47:
        raise RuntimeError("G1_OUTCOME_LABEL_COUNT_INVALID")
    expected = {
        (entry["raw_episode_id"], int(entry["output_episode_index"])): entry
        for entry in conversion_episodes
    }
    observed = set()
    for label in labels:
        key = (label.get("raw_episode_id"), label.get("output_episode_index"))
        if key in observed or key not in expected:
            raise RuntimeError("G1_OUTCOME_LABEL_DUPLICATE_OR_UNKNOWN_EPISODE")
        observed.add(key)
        conversion = expected[key]
        output_index = int(label["output_episode_index"])
        if (
            label.get("split") != conversion.get("split")
            or not isinstance(label.get("terminal_frame_index"), int)
            or not 0 < label["terminal_frame_index"] < episode_lengths[output_index]
            or not isinstance(label.get("frame_labels"), list)
            or not label["frame_labels"]
        ):
            raise RuntimeError("G1_OUTCOME_LABEL_EPISODE_BINDING_DRIFT")
        terminal_labels = [
            item
            for item in label["frame_labels"]
            if item.get("terminated") is True
        ]
        if (
            len(terminal_labels) != 1
            or terminal_labels[0].get("frame_index")
            != label["terminal_frame_index"]
        ):
            raise RuntimeError("G1_EXTERNAL_TERMINAL_LABEL_INVALID")
    if observed != set(expected):
        raise RuntimeError("G1_OUTCOME_LABEL_EPISODE_COVERAGE_INCOMPLETE")
    return sorted(labels, key=lambda item: item["output_episode_index"])


@dataclass(frozen=True)
class MacroTransitionSpec:
    anchor_frame_index: int
    next_frame_index: int
    reward: float
    terminated: bool
    bootstrap_mask: float
    discount: float
    mc_return: float
    behavior_mask: tuple[bool, bool, bool]


def macro_transition_specs(terminal_frame_index: int) -> tuple[MacroTransitionSpec, ...]:
    """Synthetic-only endpoint-reward fixture; not a task2 label inference rule."""
    if terminal_frame_index < 1:
        raise ValueError("terminal boundary must admit one executed action")
    raw = []
    for anchor in range(0, terminal_frame_index, ANCHOR_STRIDE):
        next_frame = min(anchor + EXECUTED_ACTION_SLOTS, terminal_frame_index)
        executed_steps = next_frame - anchor
        behavior_mask = tuple(
            slot < executed_steps for slot in range(EXECUTED_ACTION_SLOTS)
        )
        terminated = next_frame == terminal_frame_index
        bootstrap = 0.0 if terminated else 1.0
        raw.append(
            {
                "anchor": anchor,
                "next": next_frame,
                "reward": 1.0 if terminated else 0.0,
                "terminated": terminated,
                "bootstrap": bootstrap,
                "discount": GAMMA * bootstrap,
                "behavior_mask": behavior_mask,
            }
        )
    returns = [0.0] * len(raw)
    following = 0.0
    for index in range(len(raw) - 1, -1, -1):
        following = raw[index]["reward"] + raw[index]["discount"] * following
        returns[index] = following
    result = tuple(
        MacroTransitionSpec(
            anchor_frame_index=item["anchor"],
            next_frame_index=item["next"],
            reward=item["reward"],
            terminated=item["terminated"],
            bootstrap_mask=item["bootstrap"],
            discount=item["discount"],
            mc_return=returns[index],
            behavior_mask=item["behavior_mask"],
        )
        for index, item in enumerate(raw)
    )
    if sum(item.terminated for item in result) != 1 or result[-1].reward != 1.0:
        raise AssertionError("each episode must have exactly one terminal transition")
    return result


@dataclass(frozen=True)
class PreparedOfflineTransition:
    row: dict
    absolute_action_chunk: np.ndarray
    delta_action_chunk: np.ndarray
    normalized_action_chunk: np.ndarray
    action_valid_mask: np.ndarray
    action_feature_mask: np.ndarray


def iter_episode_transitions(
    *,
    arrays: dict[str, np.ndarray],
    outcome: dict,
    normalizer,
    source_data_relative_path: str,
    task: str,
) -> Iterator[PreparedOfflineTransition]:
    episode_index = int(outcome["output_episode_index"])
    terminal = int(outcome["terminal_frame_index"])
    actions = np.asarray(arrays["action"], dtype=np.float64)
    states = np.asarray(arrays["observation.state"], dtype=np.float64)
    wrenches = np.asarray(arrays["observation.wrench"], dtype=np.float64)
    frame_indices = np.asarray(arrays["frame_index"], dtype=np.int64)
    episode_indices = np.asarray(arrays["episode_index"], dtype=np.int64)
    global_indices = np.asarray(arrays["index"], dtype=np.int64)
    if (
        actions.shape != (terminal + 1, 7)
        or states.shape != actions.shape
        or wrenches.shape != (terminal + 1, 6)
        or not np.array_equal(frame_indices, np.arange(terminal + 1))
        or not np.all(episode_indices == episode_index)
    ):
        raise RuntimeError("G1_EPISODE_ARRAY_CONTRACT_INVALID")

    for spec in macro_transition_specs(terminal):
        anchor = spec.anchor_frame_index
        offsets = np.arange(HORIZON)
        query = np.minimum(anchor + offsets, terminal)
        action_is_pad = anchor + offsets > terminal
        absolute_chunk = actions[query]
        sample = {
            "observation.state": states[anchor],
            "observation.wrench": wrenches[anchor],
            "action": absolute_chunk,
            "action_is_pad": action_is_pad,
            "episode_index": episode_index,
            "frame_index": anchor,
            "task": task,
            "observation.images.camera1": None,
            "observation.images.camera2": None,
        }
        for key in PROVENANCE_KEYS:
            sample[key] = arrays[key][anchor]
        prepared = prepare_training_sample(sample, normalizer)
        delta_chunk = ActionDeltaProcessor.to_delta(absolute_chunk, states[anchor])
        normalized = prepared["delta_action7"]
        valid = prepared["action_valid_mask"]
        feature = valid[:, None] & np.ones((1, 7), dtype=np.bool_)
        critic_action = np.asarray(normalized[:EXECUTED_ACTION_SLOTS], dtype=np.float32)
        behavior_mask = np.asarray(spec.behavior_mask, dtype=np.bool_)
        critic_action[~behavior_mask] = 0.0
        actor_q_mask = np.broadcast_to(
            np.arange(7)[None, :] < 6, (EXECUTED_ACTION_SLOTS, 7)
        ) & behavior_mask[:, None]
        row = {
            "raw_episode_id": outcome["raw_episode_id"],
            "output_episode_index": episode_index,
            "split": outcome["split"],
            "source_data_relative_path": source_data_relative_path,
            "anchor_row_index": anchor,
            "next_row_index": spec.next_frame_index,
            "anchor_frame_index": anchor,
            "next_frame_index": spec.next_frame_index,
            "terminal_frame_index": terminal,
            "anchor_global_index": int(global_indices[anchor]),
            "next_global_index": int(global_indices[spec.next_frame_index]),
            "critic_action_k7": critic_action.reshape(-1).tolist(),
            "executed_action_mask": behavior_mask.tolist(),
            "critic_action_contract_version": CRITIC_ACTION_CONTRACT.version,
            "critic_macro_duration_ns": int(
                round(
                    (spec.next_frame_index - anchor)
                    * 1_000_000_000
                    / CRITIC_ACTION_CONTRACT.model_grid_hz
                )
            ),
            "actor_q_gradient_mask_k7": actor_q_mask.reshape(-1).tolist(),
            "reward": spec.reward,
            "terminated": spec.terminated,
            "truncated": False,
            "timeout": False,
            "bootstrap_mask": spec.bootstrap_mask,
            "discount": spec.discount,
            "mc_return": spec.mc_return,
        }
        yield PreparedOfflineTransition(
            row=row,
            absolute_action_chunk=np.asarray(absolute_chunk, dtype=np.float64),
            delta_action_chunk=np.asarray(delta_chunk, dtype=np.float64),
            normalized_action_chunk=np.asarray(normalized, dtype=np.float32),
            action_valid_mask=np.asarray(valid, dtype=np.bool_),
            action_feature_mask=np.asarray(feature, dtype=np.bool_),
        )


class OrderedTensorDigest:
    def __init__(self) -> None:
        self._digest = hashlib.sha256()
        self.count = 0

    def update(self, identity: str, value: np.ndarray) -> None:
        array = np.ascontiguousarray(value)
        encoded = identity.encode("utf-8")
        dtype = array.dtype.str.encode("ascii")
        self._digest.update(struct.pack("<I", len(encoded)))
        self._digest.update(encoded)
        self._digest.update(struct.pack("<I", len(dtype)))
        self._digest.update(dtype)
        self._digest.update(struct.pack("<I", array.ndim))
        self._digest.update(struct.pack(f"<{array.ndim}q", *array.shape))
        self._digest.update(array.view(np.uint8))
        self.count += 1

    def record(self) -> dict:
        return {"ordered_tensor_count": self.count, "sha256": self._digest.hexdigest()}


def load_transition_split(
    root: Path, split: Literal["train", "val", "test"]
):
    if split not in {"train", "val", "test"}:
        raise ValueError(f"unsupported transition split: {split!r}")
    import pyarrow.parquet as pq

    table = pq.read_table(root / "transition_index.parquet", filters=[("split", "=", split)])
    if set(table.column("split").to_pylist()) != {split}:
        raise RuntimeError("G1_TRANSITION_SPLIT_FILTER_LEAK")
    return table
