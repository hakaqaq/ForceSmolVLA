"""Frozen-detector G1 reward and macro-transition semantics."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Iterator, Literal, Sequence

import numpy as np

from forcesmolvla.action_delta import ActionDeltaProcessor
from forcesmolvla.rft.offline_transitions import PROVENANCE_KEYS
from forcesmolvla.training_data import prepare_training_sample


HORIZON = 50
K = 3
ANCHOR_STRIDE = 3
GAMMA = 0.99
TAU = 0.83
REQUIRED_CONSECUTIVE_FRAMES = 5
REWARD_SOURCE = "frozen_classifier_detector"
CHECKPOINT_SHA256 = "6b4e366baa55993d150cb3dd86e67a1d708e58d836b123a0c433190835021510"


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
class DetectionTrace:
    consecutive_counts: tuple[int, ...]
    threshold_positive: tuple[bool, ...]
    latched: tuple[bool, ...]
    trigger_frame: int | None
    streak_start_frame: int | None


def causal_detection_trace(
    frames: Sequence[int],
    probabilities: Sequence[float],
    validity: Sequence[bool],
    *,
    tau: float = TAU,
    required: int = REQUIRED_CONSECUTIVE_FRAMES,
) -> DetectionTrace:
    """Latch on the current frame where the valid consecutive streak first reaches M."""

    if not 0.0 < tau < 1.0 or required < 1:
        raise ValueError("invalid detector parameters")
    if not len(frames) == len(probabilities) == len(validity):
        raise ValueError("detector input length mismatch")
    last_frame: int | None = None
    streak = 0
    streak_start: int | None = None
    trigger: int | None = None
    trigger_start: int | None = None
    counts: list[int] = []
    positives: list[bool] = []
    latched: list[bool] = []
    for raw_frame, raw_probability, raw_valid in zip(frames, probabilities, validity):
        frame = int(raw_frame)
        probability = float(raw_probability)
        valid = bool(raw_valid)
        usable = valid and frame >= 0 and np.isfinite(probability) and 0.0 <= probability <= 1.0
        if not usable:
            last_frame = None
            streak = 0
            streak_start = None
            positive = False
        else:
            if last_frame is None or frame != last_frame + 1:
                streak = 0
                streak_start = None
            last_frame = frame
            positive = probability >= tau
            if trigger is None:
                if positive:
                    if streak == 0:
                        streak_start = frame
                    streak += 1
                    if streak >= required:
                        trigger = frame
                        trigger_start = streak_start
                else:
                    streak = 0
                    streak_start = None
        positives.append(positive)
        counts.append(streak)
        latched.append(trigger is not None)
    return DetectionTrace(
        consecutive_counts=tuple(counts),
        threshold_positive=tuple(positives),
        latched=tuple(latched),
        trigger_frame=trigger,
        streak_start_frame=trigger_start,
    )


@dataclass(frozen=True)
class DetectorMacroTransition:
    anchor_frame: int
    next_frame: int
    detector_terminal_frame: int
    executed_steps: int
    executed_action_mask: tuple[bool, bool, bool]
    reward: float
    terminated: bool
    bootstrap_mask: int
    discount: float
    mc_return: float


def detector_macro_transitions(detector_terminal_frame: int) -> tuple[DetectorMacroTransition, ...]:
    if (
        not isinstance(detector_terminal_frame, int)
        or isinstance(detector_terminal_frame, bool)
        or detector_terminal_frame <= 0
    ):
        raise ValueError("detector_terminal_frame must be a positive integer")
    raw: list[dict] = []
    for anchor in range(0, detector_terminal_frame, ANCHOR_STRIDE):
        next_frame = min(anchor + K, detector_terminal_frame)
        executed_steps = next_frame - anchor
        terminated = next_frame == detector_terminal_frame
        raw.append(
            {
                "anchor": anchor,
                "next": next_frame,
                "steps": executed_steps,
                "mask": tuple(slot < executed_steps for slot in range(K)),
                "reward": 1.0 if terminated else 0.0,
                "terminated": terminated,
                "bootstrap": 0 if terminated else 1,
                "discount": 0.0 if terminated else GAMMA,
            }
        )
    following = 0.0
    returns = [0.0] * len(raw)
    for index in range(len(raw) - 1, -1, -1):
        following = raw[index]["reward"] + raw[index]["discount"] * following
        returns[index] = following
    result = tuple(
        DetectorMacroTransition(
            anchor_frame=item["anchor"],
            next_frame=item["next"],
            detector_terminal_frame=detector_terminal_frame,
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
    _require(sum(item.reward == 1.0 for item in result) == 1, "DETECTOR_G1_REWARD_COUNT_INVALID")
    _require(sum(item.terminated for item in result) == 1, "DETECTOR_G1_TERMINAL_COUNT_INVALID")
    _require(result[-1].next_frame == detector_terminal_frame, "DETECTOR_G1_TERMINAL_NOT_REACHED")
    _require(
        all(item.anchor_frame < item.next_frame <= detector_terminal_frame for item in result),
        "DETECTOR_G1_SELF_LOOP_OR_POST_TERMINAL",
    )
    return result


@dataclass(frozen=True)
class PreparedDetectorTransition:
    row: dict
    absolute_action_chunk: np.ndarray
    delta_action_chunk: np.ndarray
    normalized_action_chunk: np.ndarray
    action_valid_mask: np.ndarray
    executed_normalized_action: np.ndarray


def iter_detector_episode_transitions(
    *,
    arrays: dict[str, np.ndarray],
    episode: dict,
    detector_terminal_frame: int,
    detector_streak_start_frame: int,
    detector_probability_at_trigger: float,
    normalizer,
    source_data_relative_path: str,
    task: str,
) -> Iterator[PreparedDetectorTransition]:
    episode_id = episode["episode_id"]
    episode_index = int(episode["output_episode_index"])
    split = episode["split"]
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
        and 0 < detector_terminal_frame < frame_count,
        "DETECTOR_G1_EPISODE_ARRAY_CONTRACT_INVALID",
    )

    for spec in detector_macro_transitions(detector_terminal_frame):
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
        _require(
            np.array_equal(normalized, normalized_expected),
            "DETECTOR_G1_STAGE1_NORMALIZATION_ELEMENTWISE_MISMATCH",
        )
        _require(
            np.array_equal(valid, valid_expected),
            "DETECTOR_G1_STAGE1_ACTION_MASK_ELEMENTWISE_MISMATCH",
        )
        executed = np.asarray(normalized[: spec.executed_steps], dtype=np.float32)
        _require(executed.shape == (spec.executed_steps, 7), "DETECTOR_G1_EXECUTED_ACTION_SLICE_INVALID")
        _require(spec.next_frame - anchor == len(executed), "DETECTOR_G1_ACTION_NEXT_OFF_BY_ONE")

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
            "split": split,
            "anchor_frame": anchor,
            "next_frame": spec.next_frame,
            "detector_terminal_frame": detector_terminal_frame,
            "detector_streak_start_frame": detector_streak_start_frame,
            "detector_probability_at_trigger": detector_probability_at_trigger,
            "detector_probability_threshold": TAU,
            "detector_required_consecutive_frames": REQUIRED_CONSECUTIVE_FRAMES,
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
            "classifier_checkpoint_sha256": CHECKPOINT_SHA256,
            "observation_row_reference": row_reference(anchor),
            "next_observation_row_reference": row_reference(spec.next_frame),
            "action_chunk_reference": action_reference,
            "detector_prediction_used_for_reward": True,
            "manual_boundary_used": False,
            "reward_model_training_overlap": split == "train",
            "claim_scope": "development",
        }
        yield PreparedDetectorTransition(
            row=row,
            absolute_action_chunk=np.asarray(absolute_chunk, dtype=np.float64),
            delta_action_chunk=np.asarray(delta_chunk, dtype=np.float64),
            normalized_action_chunk=normalized,
            action_valid_mask=valid,
            executed_normalized_action=executed,
        )


def load_training_transitions(root: Path):
    """Only the detector-based G1 root and train rows are accepted."""

    import json
    import pyarrow.parquet as pq

    root = Path(root).resolve()
    manifest = json.loads((root / "g1_manifest.json").read_text(encoding="utf-8"))
    _require(
        manifest.get("schema_version") == "forcesmolvla_g1_frozen_detector_transition_view.v1"
        and manifest.get("artifact_role") == "development_frozen_detector_reward_source"
        and manifest.get("training_authorized") is True,
        "STAGE2_REJECTED_NON_DETECTOR_OR_UNAUTHORIZED_G1",
    )
    table = pq.read_table(root / "transition_index.parquet", filters=[("split", "=", "train")])
    _require(table.num_rows > 0 and set(table.column("split").to_pylist()) == {"train"}, "DETECTOR_G1_TRAIN_SPLIT_LEAK")
    return table


def load_transition_split_for_training(root: Path, split: Literal["train"] = "train"):
    if split != "train":
        raise ValueError("Stage-2 downstream loader permits detector G1 train transitions only")
    return load_training_transitions(root)


def self_check() -> None:
    trace = causal_detection_trace(
        list(range(7)),
        [0.1, 0.83, 0.9, 0.95, 0.97, 0.99, 0.2],
        [True] * 7,
    )
    assert trace.streak_start_frame == 1 and trace.trigger_frame == 5
    assert trace.consecutive_counts[:6] == (0, 1, 2, 3, 4, 5)
    gap = causal_detection_trace([0, 1, 3, 4, 5, 6, 7], [0.9] * 7, [True] * 7)
    assert gap.trigger_frame == 7 and gap.streak_start_frame == 3
    invalid = causal_detection_trace(list(range(6)), [0.9] * 6, [True, True, False, True, True, True])
    assert invalid.trigger_frame is None
    for terminal, expected_steps in ((4, [3, 1]), (5, [3, 2]), (6, [3, 3])):
        rows = detector_macro_transitions(terminal)
        assert [row.executed_steps for row in rows] == expected_steps
        assert rows[-1].reward == 1.0 and rows[-1].terminated and rows[-1].discount == 0.0
