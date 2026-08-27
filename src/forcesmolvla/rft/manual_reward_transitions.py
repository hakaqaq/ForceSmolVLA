"""Manual-boundary G1 transition view; no detector or classifier dependency."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Literal
import hashlib

import numpy as np

from forcesmolvla.action_delta import ActionDeltaProcessor
from forcesmolvla.training_data import prepare_training_sample
from forcesmolvla.rft.offline_transitions import PROVENANCE_KEYS


HORIZON = 50
K = 3
ANCHOR_STRIDE = 3
GAMMA = 0.99
REWARD_SOURCE = "human_reviewed_completion_boundary_v2"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def tensor_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(str(array.shape).encode("ascii"))
    digest.update(array.view(np.uint8))
    return digest.hexdigest()


@dataclass(frozen=True)
class ManualMacroTransition:
    anchor_frame: int
    next_frame: int
    terminal_frame: int
    executed_steps: int
    executed_action_mask: tuple[bool, bool, bool]
    reward: float
    terminated: bool
    bootstrap_mask: int
    discount: float
    mc_return: float


def manual_macro_transitions(terminal_frame: int) -> tuple[ManualMacroTransition, ...]:
    if not isinstance(terminal_frame, int) or isinstance(terminal_frame, bool) or terminal_frame <= 0:
        raise ValueError("terminal_frame must be a positive integer")
    raw = []
    for anchor in range(0, terminal_frame, ANCHOR_STRIDE):
        next_frame = min(anchor + K, terminal_frame)
        executed_steps = next_frame - anchor
        terminated = next_frame == terminal_frame
        raw.append({
            "anchor": anchor,
            "next": next_frame,
            "steps": executed_steps,
            "mask": tuple(slot < executed_steps for slot in range(K)),
            "reward": 1.0 if terminated else 0.0,
            "terminated": terminated,
            "bootstrap": 0 if terminated else 1,
            "discount": 0.0 if terminated else GAMMA,
        })
    following = 0.0
    returns = [0.0] * len(raw)
    for index in range(len(raw) - 1, -1, -1):
        following = raw[index]["reward"] + raw[index]["discount"] * following
        returns[index] = following
    result = tuple(
        ManualMacroTransition(
            anchor_frame=item["anchor"],
            next_frame=item["next"],
            terminal_frame=terminal_frame,
            executed_steps=item["steps"],
            executed_action_mask=item["mask"],
            reward=item["reward"],
            terminated=item["terminated"],
            bootstrap_mask=item["bootstrap"],
            discount=item["discount"],
            mc_return=returns[index],
        )
        for index, item in enumerate(raw)
    )
    _require(sum(item.reward == 1.0 for item in result) == 1, "G1_REWARD_COUNT_INVALID")
    _require(sum(item.terminated for item in result) == 1, "G1_TERMINAL_COUNT_INVALID")
    _require(result[-1].next_frame == terminal_frame, "G1_TERMINAL_NOT_REACHED")
    _require(all(item.anchor_frame < item.next_frame <= terminal_frame for item in result), "G1_SELF_LOOP_OR_POST_TERMINAL")
    return result


def validate_reviewed_completion_boundaries(
    reviewed: dict,
    *,
    reviewed_sha256: str,
    conversion_episodes: list[dict],
    split_manifest: dict,
    episode_lengths: dict[int, int],
) -> list[dict]:
    _require(reviewed_sha256 == "ecda7d480f6a4c49dbe63a31b7e3172b30a5470437510522b1da2217eae77a9c", "G1_REVIEWED_LABEL_SHA_DRIFT")
    _require(reviewed.get("schema_version") == "force_rft_task2_reward_frame_labels.v2", "G1_REVIEW_SCHEMA_DRIFT")
    episodes = reviewed.get("episodes")
    _require(reviewed.get("episode_count") == 47 and isinstance(episodes, list) and len(episodes) == 47, "G1_REVIEW_EPISODE_COUNT_INVALID")
    conversion_by_id = {item["raw_episode_id"]: item for item in conversion_episodes}
    expected_ids = set().union(*(set(split_manifest[name]) for name in ("train", "val", "test")))
    _require(set(conversion_by_id) == expected_ids and len(expected_ids) == 47, "G1_CONVERSION_SPLIT_COVERAGE_INVALID")
    split_alias = {"train": "train", "val": "val", "validation": "val", "test": "test"}
    observed = set()
    labels = []
    for episode in episodes:
        episode_id = episode.get("episode_id")
        _require(episode_id in conversion_by_id and episode_id not in observed, "G1_REVIEW_EPISODE_DUPLICATE_OR_UNKNOWN")
        observed.add(episode_id)
        conversion = conversion_by_id[episode_id]
        output_index = episode.get("output_episode_index")
        split = episode.get("split")
        terminal = episode.get("first_confident_complete_frame")
        _require(
            episode.get("manual_review_status") == "human_reviewed"
            and episode.get("completion_visible") is True
            and episode.get("completion_stable") is True
            and episode.get("positive_available") is True
            and episode.get("task_outcome_context") == "success",
            "G1_EPISODE_NOT_HUMAN_REVIEWED_SUCCESS",
        )
        _require(
            isinstance(output_index, int)
            and output_index == int(conversion["output_episode_index"])
            and split in split_alias
            and conversion["split"] == split_alias[split]
            and episode_id in split_manifest[split_alias[split]],
            "G1_EPISODE_SPLIT_OR_INDEX_DRIFT",
        )
        _require(
            isinstance(terminal, int)
            and not isinstance(terminal, bool)
            and 0 < terminal < episode_lengths[output_index]
            and episode.get("last_confident_incomplete_frame") == terminal - 1,
            "G1_MANUAL_TERMINAL_BOUNDARY_INVALID",
        )
        labels.append({
            "episode_id": episode_id,
            "output_episode_index": output_index,
            "split": split,
            "terminal_frame": terminal,
            "reward_source": REWARD_SOURCE,
            "human_label_sha256": reviewed_sha256,
        })
    _require(observed == expected_ids, "G1_REVIEW_COVERAGE_INCOMPLETE")
    return sorted(labels, key=lambda item: item["output_episode_index"])


@dataclass(frozen=True)
class PreparedManualTransition:
    row: dict
    absolute_action_chunk: np.ndarray
    delta_action_chunk: np.ndarray
    normalized_action_chunk: np.ndarray
    action_valid_mask: np.ndarray
    executed_normalized_action: np.ndarray


def iter_manual_episode_transitions(
    *,
    arrays: dict[str, np.ndarray],
    label: dict,
    normalizer,
    source_data_relative_path: str,
    task: str,
) -> Iterator[PreparedManualTransition]:
    episode_id = label["episode_id"]
    episode_index = int(label["output_episode_index"])
    terminal = int(label["terminal_frame"])
    actions = np.asarray(arrays["action"], dtype=np.float64)
    states = np.asarray(arrays["observation.state"], dtype=np.float64)
    wrenches = np.asarray(arrays["observation.wrench"], dtype=np.float64)
    frame_indices = np.asarray(arrays["frame_index"], dtype=np.int64)
    episode_indices = np.asarray(arrays["episode_index"], dtype=np.int64)
    global_indices = np.asarray(arrays["index"], dtype=np.int64)
    frame_count = len(actions)
    _require(
        actions.shape == states.shape == (frame_count, 7)
        and wrenches.shape == (frame_count, 6)
        and np.array_equal(frame_indices, np.arange(frame_count))
        and np.all(episode_indices == episode_index)
        and 0 < terminal < frame_count,
        "G1_EPISODE_ARRAY_CONTRACT_INVALID",
    )

    for spec in manual_macro_transitions(terminal):
        anchor = spec.anchor_frame
        offsets = np.arange(HORIZON)
        source_indices = np.minimum(anchor + offsets, frame_count - 1)
        action_is_pad = anchor + offsets >= frame_count
        absolute_chunk = actions[source_indices]
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
        normalized_expected = normalizer.delta_action7.apply(delta_chunk).astype(np.float32)
        valid_expected = (~action_is_pad).astype(np.bool_)
        normalized = np.asarray(prepared["delta_action7"], dtype=np.float32)
        valid = np.asarray(prepared["action_valid_mask"], dtype=np.bool_)
        _require(np.array_equal(normalized, normalized_expected), "G1_STAGE1_NORMALIZATION_ELEMENTWISE_MISMATCH")
        _require(np.array_equal(valid, valid_expected), "G1_STAGE1_ACTION_MASK_ELEMENTWISE_MISMATCH")
        executed = np.asarray(normalized[: spec.executed_steps], dtype=np.float32)
        _require(executed.shape == (spec.executed_steps, 7), "G1_EXECUTED_ACTION_SLICE_INVALID")
        _require(spec.next_frame - anchor == len(executed), "G1_ACTION_NEXT_OFF_BY_ONE")

        def row_reference(frame: int) -> dict:
            return {
                "dataset_root_id": "task2_lerobotv3",
                "data_relative_path": source_data_relative_path,
                "row_index": int(frame),
                "episode_id": episode_id,
                "frame_index": int(frame),
                "global_index": int(global_indices[frame]),
            }

        action_reference = {
            "dataset_root_id": "task2_lerobotv3",
            "data_relative_path": source_data_relative_path,
            "anchor_row_index": int(anchor),
            "source_frame_start_inclusive": int(anchor),
            "source_frame_stop_exclusive": int(min(anchor + HORIZON, frame_count)),
            "stage1_horizon": HORIZON,
            "executed_slice_start": 0,
            "executed_slice_stop_exclusive": spec.executed_steps,
            "absolute_action_chunk_sha256": tensor_sha256(absolute_chunk),
            "delta_action_chunk_sha256": tensor_sha256(delta_chunk),
            "normalized_action_chunk_sha256": tensor_sha256(normalized),
            "action_valid_mask_sha256": tensor_sha256(valid),
        }
        row = {
            "episode_id": episode_id,
            "output_episode_index": episode_index,
            "split": label["split"],
            "anchor_frame": anchor,
            "next_frame": spec.next_frame,
            "terminal_frame": terminal,
            "executed_steps": spec.executed_steps,
            "executed_action_mask": list(spec.executed_action_mask),
            "normalized_delta_action_exec_flat": executed.reshape(-1).tolist(),
            "stage1_action_valid_mask_h50": valid.tolist(),
            "reward": spec.reward,
            "terminated": spec.terminated,
            "bootstrap_mask": spec.bootstrap_mask,
            "discount": spec.discount,
            "mc_return": spec.mc_return,
            "reward_source": REWARD_SOURCE,
            "human_label_sha256": label["human_label_sha256"],
            "observation_row_reference": row_reference(anchor),
            "next_observation_row_reference": row_reference(spec.next_frame),
            "action_chunk_reference": action_reference,
            "online_reward_detector_ready": False,
            "detector_candidate_status": "rejected",
            "detector_prediction_used_for_reward": False,
        }
        yield PreparedManualTransition(
            row=row,
            absolute_action_chunk=np.asarray(absolute_chunk, dtype=np.float64),
            delta_action_chunk=np.asarray(delta_chunk, dtype=np.float64),
            normalized_action_chunk=normalized,
            action_valid_mask=valid,
            executed_normalized_action=executed,
        )


def load_training_transitions(root: Path):
    """Manual-boundary G1 is retained for audit and is never a training source."""
    del root
    raise RuntimeError("STAGE2_REJECTED_HISTORICAL_MANUAL_AUDIT_G1")


def load_transition_split_for_training(root: Path, split: Literal["train"] = "train"):
    del root, split
    raise RuntimeError("STAGE2_REJECTED_HISTORICAL_MANUAL_AUDIT_G1")


def self_check() -> None:
    expected = {
        1: [(0, 1, 1, (True, False, False))],
        2: [(0, 2, 2, (True, True, False))],
        3: [(0, 3, 3, (True, True, True))],
        4: [(0, 3, 3, (True, True, True)), (3, 4, 1, (True, False, False))],
        5: [(0, 3, 3, (True, True, True)), (3, 5, 2, (True, True, False))],
    }
    for terminal, values in expected.items():
        rows = manual_macro_transitions(terminal)
        assert [(r.anchor_frame, r.next_frame, r.executed_steps, r.executed_action_mask) for r in rows] == values
        assert rows[-1].reward == 1.0 and rows[-1].terminated and rows[-1].discount == 0.0
        assert all(r.anchor_frame < r.next_frame <= terminal for r in rows)
