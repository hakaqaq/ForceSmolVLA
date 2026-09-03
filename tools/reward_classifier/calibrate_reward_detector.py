#!/usr/bin/env python3
"""Calibrate a task-scoped causal reward detector on validation episodes."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from decimal import Decimal
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
IMAGE_SHAPE = (480, 640, 3)
CAMERA_KEYS = ("d435_third_person", "d405_wrist")
CLASS_NAMES = ("positive", "ordinary_negative", "hard_negative", "ambiguous")
TAUS = tuple(Decimal(i) / Decimal(100) for i in range(50, 100)) + (
    Decimal("0.995"),
    Decimal("0.999"),
)
CONSECUTIVE_COUNTS = (1, 2, 3, 4, 5, 6, 8, 10, 12, 15)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def load_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"JSON_OBJECT_REQUIRED:{path}")
    return value


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        temporary = Path(stream.name)
    os.replace(temporary, path)


def import_path(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    require(spec is not None and spec.loader is not None, f"IMPORT_FAILED:{path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def causal_trigger(frames, probabilities, tau: float, required: int) -> int | None:
    last = None
    streak = 0
    for frame, probability in zip(frames, probabilities, strict=True):
        frame = int(frame)
        if last is None or frame != last + 1:
            streak = 0
        last = frame
        streak = streak + 1 if float(probability) >= tau else 0
        if streak >= required:
            return frame
    return None


def longest_run(values) -> int:
    best = current = 0
    for value in values:
        current = current + 1 if bool(value) else 0
        best = max(best, current)
    return best


def evaluate(episodes: list[dict], tau: float, required: int) -> dict:
    results = []
    delays = []
    ordinary_false = ordinary_total = 0
    hard_false = hard_total = 0
    positive_true = positive_total = 0
    for episode in episodes:
        frames = episode["frames"]
        probabilities = episode["probabilities"]
        classes = episode["classes"]
        completion = episode["completion"]
        trigger = causal_trigger(frames, probabilities, tau, required)
        delay = None if trigger is None else trigger - completion
        if delay is not None:
            delays.append(delay)
        threshold_positive = probabilities >= tau
        ordinary = classes == 1
        hard = classes == 2
        positive = classes == 0
        ordinary_false += int(np.count_nonzero(threshold_positive & ordinary))
        ordinary_total += int(np.count_nonzero(ordinary))
        hard_false += int(np.count_nonzero(threshold_positive & hard))
        hard_total += int(np.count_nonzero(hard))
        positive_true += int(np.count_nonzero(threshold_positive & positive))
        positive_total += int(np.count_nonzero(positive))
        results.append(
            {
                "episode_id": episode["episode_id"],
                "first_confident_complete_frame": completion,
                "trigger_frame": trigger,
                "detection_delay_frames": delay,
                "longest_pre_completion_positive_run": longest_run(
                    threshold_positive[frames < completion]
                ),
            }
        )
    early = sum(item["detection_delay_frames"] is not None and item["detection_delay_frames"] < 0 for item in results)
    missed = sum(item["trigger_frame"] is None for item in results)
    return {
        "probability_threshold": tau,
        "required_consecutive_frames": required,
        "feasible": early == 0 and missed == 0,
        "early_trigger_episode_count": early,
        "missed_episode_count": missed,
        "max_detection_delay_frames": None if not delays else max(delays),
        "median_detection_delay_frames": None if not delays else float(np.median(delays)),
        "ordinary_negative_frame_fpr": ordinary_false / ordinary_total,
        "hard_negative_frame_fpr": hard_false / hard_total,
        "positive_frame_recall": positive_true / positive_total,
        "episodes": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--approve", action="store_true")
    args = parser.parse_args()

    from forcesmolvla.training_runtime import (
        resolve_task_dataset_root,
        resolve_task_output_root,
    )

    dataset_root = resolve_task_dataset_root(
        ROOT, task_id=args.task_id, dataset_root=args.dataset_root
    )
    output_root = resolve_task_output_root(
        ROOT, task_id=args.task_id, output_root=args.output_root
    )
    labels_path = ROOT / "labels" / f"{args.task_id}_reward_frame_labels.json"
    training_config = (
        ROOT / "configs/tasks" / args.task_id / "reward_classifier_training.json"
    )
    classifier_root = output_root / "reward_classifier"
    checkpoint = classifier_root / "checkpoints/best/best_checkpoint.msgpack"
    training_report_path = classifier_root / "reward_classifier_training_report.json"
    calibration_path = classifier_root / "detector_calibration.json"
    transition_config_path = (
        ROOT / "configs/tasks" / args.task_id / "forcerft_offline_reward_transitions.json"
    )
    require(not calibration_path.exists(), f"CALIBRATION_EXISTS:{calibration_path}")
    if args.approve:
        require(not transition_config_path.exists(), f"CONFIG_EXISTS:{transition_config_path}")

    training_report = load_json(training_report_path)
    require(
        training_report.get("status") == "complete"
        and training_report.get("task_id") == args.task_id
        and training_report["fixed_seed_reproducibility"]["exact_reproducibility_pass"],
        "REWARD_CLASSIFIER_TRAINING_NOT_COMPLETE",
    )
    best_step = int(training_report["primary_training_run"]["best_optimizer_update"])

    training_tool = import_path(
        "task_reward_classifier_training",
        ROOT / "tools/reward_classifier/train_reward_classifier.py",
    )
    training_tool.configure_task_inputs(
        task_id=args.task_id,
        dataset_root=dataset_root,
        reviewed_labels=labels_path,
    )
    cache_dir = args.cache_dir.resolve()
    cache_manifest = training_tool.verify_cache(cache_dir)
    require(
        cache_manifest["frozen_bindings"]["reviewed_labels"]["sha256"]
        == training_tool.sha256_file(labels_path)
        and cache_manifest["frozen_bindings"]["config"]["sha256"]
        == training_tool.sha256_file(training_config),
        "CALIBRATION_CACHE_INPUT_MISMATCH",
    )

    training_tool.install_type_only_octo_shim()
    sys.path.insert(0, "/home/rlc123/conrft/serl_launcher")
    from flax import serialization
    import jax
    import jax.numpy as jnp
    from serl_launcher.networks.reward_classifier import create_classifier

    require(jax.default_backend() == "gpu", "CALIBRATION_GPU_REQUIRED")
    camera1 = np.load(cache_dir / "camera1.npy", mmap_mode="r", allow_pickle=False)
    camera2 = np.load(cache_dir / "camera2.npy", mmap_mode="r", allow_pickle=False)
    validation_indices = np.load(cache_dir / "validation_indices.npy", allow_pickle=False)
    validation_classes = np.load(cache_dir / "validation_class_codes.npy", allow_pickle=False)
    episode_codes = np.load(cache_dir / "cache_episode_codes.npy", allow_pickle=False)
    frame_indices = np.load(cache_dir / "cache_frame_indices.npy", allow_pickle=False)
    safe_tree, _ = training_tool.npz_encoder_tree()
    sample = {key: jnp.zeros((1, 1, *IMAGE_SHAPE), dtype=jnp.uint8) for key in CAMERA_KEYS}
    with training_tool.trusted_safe_npz_pickle_bridge(safe_tree) as bridge:
        target = create_classifier(
            jax.random.PRNGKey(0), sample, list(CAMERA_KEYS),
            pretrained_encoder_path=str(bridge), n_way=2,
        )
    state = serialization.from_bytes(target, checkpoint.read_bytes())
    require(int(state.step) == best_step, "CALIBRATION_CHECKPOINT_STEP_MISMATCH")

    @jax.jit
    def infer(observations):
        return state.apply_fn({"params": state.params}, observations, train=False)

    logits = []
    for start in range(0, len(validation_indices), 128):
        selected = validation_indices[start : start + 128]
        observations = {
            CAMERA_KEYS[0]: jnp.asarray(np.asarray(camera1[selected]))[:, None],
            CAMERA_KEYS[1]: jnp.asarray(np.asarray(camera2[selected]))[:, None],
        }
        logits.append(np.asarray(jax.block_until_ready(infer(observations))).reshape(-1))
    logits = np.concatenate(logits).astype(np.float64)
    probabilities = np.where(
        logits >= 0,
        1.0 / (1.0 + np.exp(-logits)),
        np.exp(logits) / (1.0 + np.exp(logits)),
    )

    reviewed = load_json(labels_path)
    completion_by_episode = {
        item["episode_id"]: int(item["first_confident_complete_frame"])
        for item in reviewed["episodes"]
        if item["split"] in {"val", "validation"}
    }
    cache_episode_ids = cache_manifest["episode_ids"]
    selected_episode_codes = episode_codes[validation_indices]
    selected_frames = frame_indices[validation_indices]
    episodes = []
    for episode_id, completion in sorted(completion_by_episode.items()):
        code = cache_episode_ids.index(episode_id)
        mask = selected_episode_codes == code
        order = np.argsort(selected_frames[mask])
        frames = selected_frames[mask][order]
        require(np.array_equal(frames, np.arange(len(frames))), f"VALIDATION_FRAME_GAP:{episode_id}")
        episodes.append(
            {
                "episode_id": episode_id,
                "completion": completion,
                "frames": frames,
                "classes": validation_classes[mask][order],
                "probabilities": probabilities[mask][order],
            }
        )
    require(len(episodes) == len(completion_by_episode) > 0, "VALIDATION_EPISODE_COVERAGE_INVALID")

    candidates = [
        evaluate(episodes, float(tau), required)
        for tau in TAUS
        for required in CONSECUTIVE_COUNTS
    ]
    feasible = [item for item in candidates if item["feasible"]]
    require(feasible, "NO_FEASIBLE_CAUSAL_DETECTOR")
    feasible.sort(
        key=lambda item: (
            item["max_detection_delay_frames"],
            item["median_detection_delay_frames"],
            item["hard_negative_frame_fpr"],
            -item["probability_threshold"],
            -item["required_consecutive_frames"],
        )
    )
    selected = feasible[0]
    calibration = {
        "schema": "forcesmolvla.reward_detector_calibration",
        "status": "approved" if args.approve else "candidate",
        "task_id": args.task_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "scope": "validation_only",
        "validation_episode_count": len(episodes),
        "validation_frame_count": len(validation_indices),
        "candidate_grid_size": len(candidates),
        "feasible_candidate_count": len(feasible),
        "selected": selected,
        "classifier_checkpoint": str(checkpoint.relative_to(ROOT)),
        "classifier_train_state_step": best_step,
        "test_frames_evaluated": 0,
    }

    detector_spec = {
        "detector_id": f"{args.task_id}_reward_classifier",
        "probability_threshold": selected["probability_threshold"],
        "required_consecutive_frames": selected["required_consecutive_frames"],
        "detector_input_rate_hz": 30,
        "trigger_timestamp": "current_confirming_frame",
        "trigger_backfilled_to_streak_start": False,
        "latch_after_trigger": True,
        "reward_mode": "sparse_binary_terminal",
        "post_trigger_frames": "excluded",
    }
    transition_config = {
        "schema": "forcesmolvla.forcerft_offline_reward_transition_materialization",
        "status": "final",
        "task_id": args.task_id,
        "classifier_train_state_step": best_step,
        "detector_spec": detector_spec,
        "temporal_contract": {
            "data_rate_hz": 30, "policy_rate_hz": 10, "critic_steps": 3,
            "actor_horizon": 50, "anchor_stride": 3,
            "next_frame": "min(anchor_frame + critic_steps, detector_trigger_frame)",
            "executed_steps": "next_frame - anchor_frame",
            "partial_terminal_action_allowed": True, "terminal_self_loop": False,
        },
        "reward_contract": {
            "reward_source": "frozen_classifier_detector",
            "terminal_source": "causal_current_confirming_frame",
            "probability_as_continuous_reward": False,
            "manual_boundary_allowed": False,
            "episode_end_fallback_allowed": False,
            "detector_miss_policy": "exclude_without_fallback",
        },
        "action_contract": {
            "source": "recorded_absolute_action7",
            "delta_owner": "forcesmolvla.action_delta.ActionDeltaProcessor",
            "normalizer": "frozen_actor_action_normalizer", "normalizer_refit": False,
            "action_horizon": 50, "executed_slots": "first_executed_steps", "features": 7,
        },
        "inputs": {
            "classifier_checkpoint": str(checkpoint.relative_to(ROOT)),
            "safe_resnet10_npz": "assets/reward_classifier/resnet10_parameters.npz",
            "safe_resnet10_manifest": "assets/reward_classifier/resnet10_manifest.json",
            "actor_checkpoint": str((output_root / "sft/checkpoints/forcesmolvla_sft_step_010000").relative_to(ROOT)),
            "classifier_training_source": "tools/reward_classifier/train_reward_classifier.py",
            "adapter_source": "tools/reward_classifier/conrft_lerobot_v3_adapter.py",
            "detector_calibration": str(calibration_path.relative_to(ROOT)),
        },
        "required_runtime_audit": {
            "manual_label_files_opened": 0, "manual_boundary_fields_consumed": 0,
            "manual_terminal_fallback_count": 0, "classifier_optimizer_updates": 0,
            "detector_parameter_search_count": 0,
        },
    }
    atomic_json(calibration_path, calibration)
    if args.approve:
        atomic_json(transition_config_path, transition_config)
    print(json.dumps({
        "status": calibration["status"], "task_id": args.task_id,
        "probability_threshold": selected["probability_threshold"],
        "required_consecutive_frames": selected["required_consecutive_frames"],
        "max_detection_delay_frames": selected["max_detection_delay_frames"],
        "hard_negative_frame_fpr": selected["hard_negative_frame_fpr"],
        "config": str(transition_config_path) if args.approve else None,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
