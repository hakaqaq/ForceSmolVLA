#!/usr/bin/env python3
"""One-shot development test evaluation for the frozen R0 Reward Detector."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.util
from io import BytesIO
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any, Mapping

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
CONRFT_ROOT = Path("/home/rlc123/conrft")
CONRFT_RUNTIME_ROOT = CONRFT_ROOT / "serl_launcher"
CALIBRATION_SOURCE = ROOT / "tools/reward_classifier/calibrate_task2_reward_detector.py"
TRAINING_SOURCE = ROOT / "tools/reward_classifier/train_reward_classifier.py"
ADAPTER_PATH = ROOT / "tools/reward_classifier/conrft_lerobot_v3_adapter.py"
CHECKPOINT_PATH = ROOT / "artifacts/development/stage2/reward_classifier/r0_training/checkpoints/best_checkpoint.msgpack"
TRAINING_REPORT_PATH = ROOT / "artifacts/development/stage2/reward_classifier/r0_training/r0_training_validation_report.v1.json"
CANDIDATE_PATH = ROOT / "configs/stage2_r0_reward_detector.candidate.development.json"
CALIBRATION_PATH = ROOT / "artifacts/development/stage2/reward_classifier/r0_validation_detector_calibration.v1.json"
INVENTORY_PATH = ROOT / "artifacts/development/stage2/reward_classifier/task2_frame_label_inventory.v2.json"
READINESS_PATH = ROOT / "artifacts/development/stage2/s2_r0_label_ingestion_readiness.v4.json"
REVIEWED_PATH = ROOT / "labels/task2_reward_frame_labels.v2.reviewed.json"
SPLIT_PATH = ROOT / "datasets/task2_lerobotv3/split_manifest.json"
DATASET_ROOT = ROOT / "datasets/task2_lerobotv3"
SAFE_ASSET_PATH = ROOT / "artifacts/development/stage2/reward_classifier/pretrained/resnet10_params.safe.npz"

TEST_ARTIFACT_PATH = ROOT / "artifacts/development/stage2/reward_classifier/r0_one_shot_test_evaluation.v1.json"
PASS_SPEC_PATH = ROOT / "configs/stage2_r0_reward_detector.development_approved.json"
REJECTED_SPEC_PATH = ROOT / "configs/stage2_r0_reward_detector.rejected_after_one_shot_test.json"
REPORT_PATH = ROOT / "docs/r0_one_shot_test_evaluation_report.md"

CHECKPOINT_SHA256 = "6b4e366baa55993d150cb3dd86e67a1d708e58d836b123a0c433190835021510"
CANDIDATE_SHA256 = "d493c9f398a2f14ae5e11d1d1cf44ef769c66759c61220eed53e00eedb2d3362"
CALIBRATION_SHA256 = "5d52475ce518eef2315bbf6908140d318d89d025311efdb3f7e8c8204d6bdb47"
TRAINING_REPORT_SHA256 = "c48d845f77b3b7b46b5788998c412c280d93e0e8a863d5edeb488fcea8cb2aac"
CALIBRATION_SOURCE_SHA256 = "f6a7f5f411423bd6a09a94e39516f92bbe5edb6dd0d6d7d399d364ec84c684a5"
PRE_GPU_PREFLIGHT_SOURCE_SHA256 = "60cdb9b6307d73ec564be0753a779cf2a61d09964eb452822c145952387951da"

TAU = 0.83
REQUIRED_CONSECUTIVE = 5
FPS = 30
TEST_ORDER = ("episode_000005", "episode_000021", "episode_000025", "episode_000033")
SOURCE_CAMERA_KEYS = ("observation.images.camera1", "observation.images.camera2")
CLASSIFIER_CAMERA_KEYS = ("d435_third_person", "d405_wrist")
CLASS_NAMES = ("positive", "ordinary_negative", "hard_negative", "ambiguous")
CLASS_CODE = {name: index for index, name in enumerate(CLASS_NAMES)}
IMAGE_SHAPE = (480, 640, 3)
INFERENCE_BATCH_SIZE = 128


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def import_path(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    require(spec is not None and spec.loader is not None, f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


calibration = import_path("r0_frozen_calibration", CALIBRATION_SOURCE)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False, encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        temporary = Path(stream.name)
    temporary.replace(path)


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False, encoding="utf-8") as stream:
        stream.write(value)
        temporary = Path(stream.name)
    temporary.replace(path)


def relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def binding(path: Path) -> dict[str, Any]:
    return {"path": relative(path), "file_size": path.stat().st_size, "sha256": calibration.sha256_file(path)}


def verify_frozen_inputs() -> dict[str, Any]:
    frozen = calibration.verify_frozen_inputs()
    expected = {
        CHECKPOINT_PATH: CHECKPOINT_SHA256,
        TRAINING_REPORT_PATH: TRAINING_REPORT_SHA256,
        CANDIDATE_PATH: CANDIDATE_SHA256,
        CALIBRATION_PATH: CALIBRATION_SHA256,
        CALIBRATION_SOURCE: CALIBRATION_SOURCE_SHA256,
    }
    for path, digest in expected.items():
        require(path.is_file(), f"frozen input missing: {path}")
        require(calibration.sha256_file(path) == digest, f"frozen input SHA mismatch: {path}")

    candidate = load_json(CANDIDATE_PATH)
    calibration_artifact = load_json(CALIBRATION_PATH)
    require(candidate["artifact_status"] == "PROPOSED_UNAPPROVED_VALIDATION_ONLY_CANDIDATE", "candidate state drift")
    require(candidate["approval_status"] == "not_approved", "candidate was already generally approved")
    require(candidate["candidate"] == {
        "consecutive_positive_frames": REQUIRED_CONSECUTIVE,
        "probability_threshold": TAU,
    }, "candidate parameters drift")
    require(candidate["causal_semantics"]["input_rate_hz"] == FPS, "candidate frequency drift")
    require(candidate["causal_semantics"]["latched_after_trigger_within_episode"] is True, "latch drift")
    require(candidate["causal_semantics"]["backfill_to_streak_start"] is False, "backfill drift")
    require(calibration_artifact["artifact_status"] == "PASS_VALIDATION_CALIBRATION_CANDIDATE_PROPOSED_NOT_APPROVED", "calibration did not pass")
    proposed = calibration_artifact["proposed_candidate"]
    require(proposed["tau"] == TAU and proposed["M"] == REQUIRED_CONSECUTIVE, "calibration proposal mismatch")
    require(calibration_artifact["validation_access_audit"]["test_frames_inferred"] == 0, "test was previously inferred during calibration")
    require(tuple(frozen["split"]["test"]) == TEST_ORDER, "test episode order drift")
    require(set(TEST_ORDER).isdisjoint(frozen["split"]["train"]), "train/test episode leakage")
    require(set(TEST_ORDER).isdisjoint(frozen["split"]["val"]), "validation/test episode leakage")
    return {**frozen, "candidate": candidate, "calibration_artifact": calibration_artifact}


def decode_rgb(payload: bytes) -> np.ndarray:
    from PIL import Image

    require(isinstance(payload, bytes), "embedded image bytes missing")
    with Image.open(BytesIO(payload)) as image:
        rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    require(rgb.shape == IMAGE_SHAPE, f"decoded image shape mismatch: {rgb.shape}")
    return np.ascontiguousarray(rgb)


def prepare_cache(work_dir: Path) -> None:
    """Read/decode each frozen test frame once into an immutable ephemeral cache."""
    import pyarrow.parquet as pq

    require(not work_dir.exists(), f"one-shot work directory already exists: {work_dir}")
    frozen = verify_frozen_inputs()
    inventory_by_id = {episode["episode_id"]: episode for episode in frozen["inventory"]["episodes"]}
    reviewed_by_id = {episode["episode_id"]: episode for episode in frozen["reviewed"]["episodes"]}
    total_frames = sum(inventory_by_id[episode_id]["frame_count"] for episode_id in TEST_ORDER)
    require(total_frames == 3040, "test frame total mismatch")

    staging = work_dir.parent / f".{work_dir.name}.tmp-{os.getpid()}"
    require(not staging.exists(), f"one-shot staging directory already exists: {staging}")
    staging.mkdir(parents=True)
    try:
        camera1 = np.lib.format.open_memmap(staging / "camera1.npy", mode="w+", dtype=np.uint8, shape=(total_frames, *IMAGE_SHAPE))
        camera2 = np.lib.format.open_memmap(staging / "camera2.npy", mode="w+", dtype=np.uint8, shape=(total_frames, *IMAGE_SHAPE))
        frame_indices = np.empty(total_frames, dtype=np.int32)
        class_codes = np.empty(total_frames, dtype=np.uint8)
        valid = np.ones(total_frames, dtype=np.bool_)
        adapter_module = import_path("r0_test_adapter", ADAPTER_PATH)
        adapter = adapter_module.ConRFTLeRobotV3Adapter()
        cursor = 0
        episodes = []
        opened_files = []
        for episode_id in TEST_ORDER:
            episode = inventory_by_id[episode_id]
            review = reviewed_by_id[episode_id]
            require(episode["split"] == "test" and review["split"] == "test", "non-test episode requested")
            require(review["manual_review_status"] == "human_reviewed", "test episode not human reviewed")
            completion = int(review["first_confident_complete_frame"])
            require(episode["class_intervals_inclusive"]["positive"] == [[completion, episode["frame_count"] - 1]], "positive boundary mismatch")
            classes = calibration.frame_classes(episode)
            require(not np.any(classes == CLASS_CODE["ambiguous"]), "ambiguous test frame present unexpectedly")
            parquet_path = (DATASET_ROOT / episode["source_data_relative_path"]).resolve()
            require(parquet_path.is_relative_to(DATASET_ROOT.resolve()), "test parquet path escape")
            table = pq.read_table(
                parquet_path,
                columns=[
                    *SOURCE_CAMERA_KEYS,
                    "frame_index",
                    "episode_index",
                    "timestamp",
                    "provenance.camera1_receive_monotonic_ns",
                    "provenance.camera2_receive_monotonic_ns",
                ],
            )
            require(table.num_rows == episode["frame_count"], "test parquet row count mismatch")
            start_offset = cursor
            for frame in range(episode["frame_count"]):
                row = table.slice(frame, 1).to_pylist()[0]
                require(row["frame_index"] == frame, "non-consecutive test frame")
                require(row["episode_index"] == episode["output_episode_index"], "test episode index mismatch")
                require(np.isfinite(float(row["timestamp"])), "invalid test timestamp")
                rgb1 = decode_rgb(row[SOURCE_CAMERA_KEYS[0]]["bytes"])
                rgb2 = decode_rgb(row[SOURCE_CAMERA_KEYS[1]]["bytes"])
                adapted = adapter.adapt(
                    {
                        SOURCE_CAMERA_KEYS[0]: np.ascontiguousarray(np.transpose(rgb1, (2, 0, 1))),
                        SOURCE_CAMERA_KEYS[1]: np.ascontiguousarray(np.transpose(rgb2, (2, 0, 1))),
                    },
                    row_reference=adapter_module.RowReference(
                        "task2_lerobotv3", episode["source_data_relative_path"], frame,
                        episode_id, frame, float(row["timestamp"]),
                    ),
                    camera_row_identity=adapter_module.CameraRowIdentity(
                        int(row["provenance.camera1_receive_monotonic_ns"]),
                        int(row["provenance.camera2_receive_monotonic_ns"]),
                    ),
                )
                camera1[cursor] = adapted.observation[CLASSIFIER_CAMERA_KEYS[0]][0, 0]
                camera2[cursor] = adapted.observation[CLASSIFIER_CAMERA_KEYS[1]][0, 0]
                frame_indices[cursor] = frame
                class_codes[cursor] = classes[frame]
                cursor += 1
            opened_files.append(episode["source_data_relative_path"])
            episodes.append({
                "episode_id": episode_id,
                "output_episode_index": episode["output_episode_index"],
                "source_data_relative_path": episode["source_data_relative_path"],
                "frame_count": episode["frame_count"],
                "cache_range_half_open": [start_offset, cursor],
                "first_confident_complete_frame": completion,
                "class_frame_counts": episode["class_frame_counts"],
            })
            del table
        require(cursor == total_frames and adapter.episode_reset_count == len(TEST_ORDER), "test cache completeness failure")
        camera1.flush()
        camera2.flush()
        del camera1, camera2
        np.save(staging / "frame_indices.npy", frame_indices, allow_pickle=False)
        np.save(staging / "class_codes.npy", class_codes, allow_pickle=False)
        np.save(staging / "valid.npy", valid, allow_pickle=False)
        files = {}
        for name in ("camera1.npy", "camera2.npy", "frame_indices.npy", "class_codes.npy", "valid.npy"):
            path = staging / name
            files[name] = {"file_size": path.stat().st_size, "sha256": calibration.sha256_file(path)}
        manifest = {
            "schema_version": "forcesmolvla_r0_one_shot_test_cache.v1",
            "artifact_status": "COMPLETE_EPHEMERAL_FROZEN_TEST_CACHE_NOT_YET_INFERRED",
            "created_at": utc_now(),
            "test_episode_order": list(TEST_ORDER),
            "test_episode_count": len(TEST_ORDER),
            "test_frame_count": total_frames,
            "episodes": episodes,
            "input_validity": {
                "all_frame_indices_consecutive": True,
                "all_images_present_and_valid": True,
                "all_adapter_checks_passed": True,
                "invalid_frame_count": 0,
            },
            "access_audit": {
                "test_parquet_files_opened": opened_files,
                "test_image_rows_loaded_and_decoded": total_frames,
                "test_images_decoded": total_frames * len(SOURCE_CAMERA_KEYS),
                "train_parquet_files_opened": [],
                "validation_parquet_files_opened": [],
            },
            "bindings": {
                "best_checkpoint": binding(CHECKPOINT_PATH),
                "candidate_config": binding(CANDIDATE_PATH),
                "validation_calibration": binding(CALIBRATION_PATH),
                "inventory": binding(INVENTORY_PATH),
                "readiness": binding(READINESS_PATH),
                "reviewed_labels": binding(REVIEWED_PATH),
                "split_manifest": binding(SPLIT_PATH),
                "adapter": binding(ADAPTER_PATH),
                "one_shot_source": binding(Path(__file__).resolve()),
            },
            "files": files,
        }
        atomic_json(staging / "cache_manifest.json", manifest)
        staging.replace(work_dir)
        print(json.dumps({"phase": "prepare_one_shot_test_cache", "status": "pass", "frames": total_frames}), flush=True)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def verify_work_cache(work_dir: Path) -> dict[str, Any]:
    manifest = load_json(work_dir / "cache_manifest.json")
    require(manifest["artifact_status"] == "COMPLETE_EPHEMERAL_FROZEN_TEST_CACHE_NOT_YET_INFERRED", "test cache state invalid")
    require(tuple(manifest["test_episode_order"]) == TEST_ORDER, "test cache episode order drift")
    recorded_source = manifest["bindings"]["one_shot_source"]["sha256"]
    current_source = calibration.sha256_file(Path(__file__).resolve())
    source_matches = recorded_source == current_source
    zero_frame_preflight_recovery = (
        recorded_source == PRE_GPU_PREFLIGHT_SOURCE_SHA256
        and (work_dir / "ONE_SHOT_INFERENCE_STARTED.json").is_file()
        and not (work_dir / "test_predictions.npz").exists()
        and not (work_dir / "inference_evidence.json").exists()
        and not (work_dir / "PRE_INFERENCE_BACKEND_PREFLIGHT_FAILURE.json").exists()
    )
    require(source_matches or zero_frame_preflight_recovery, "one-shot source changed after cache creation")
    for name, value in manifest["files"].items():
        path = work_dir / name
        require(path.stat().st_size == value["file_size"], f"cache file size mismatch: {name}")
        require(calibration.sha256_file(path) == value["sha256"], f"cache file SHA mismatch: {name}")
    return manifest


def run_inference(work_dir: Path) -> None:
    """Run the only authorized GPU test inference and persist its immutable output."""
    require(os.environ.get("CONDA_DEFAULT_ENV") == "conrft_reward", "inference must run in conrft_reward")
    verify_frozen_inputs()
    manifest = verify_work_cache(work_dir)
    prediction_path = work_dir / "test_predictions.npz"
    evidence_path = work_dir / "inference_evidence.json"
    lock_path = work_dir / "ONE_SHOT_INFERENCE_STARTED.json"
    preflight_failure_path = work_dir / "PRE_INFERENCE_BACKEND_PREFLIGHT_FAILURE.json"
    require(not prediction_path.exists() and not evidence_path.exists(), "one-shot test inference output already exists")
    require(not preflight_failure_path.exists(), "zero-frame backend preflight was already recovered")
    prior_lock = load_json(lock_path) if lock_path.is_file() else None
    if prior_lock is not None:
        require(prior_lock["candidate_config_sha256"] == CANDIDATE_SHA256, "prior preflight lock candidate mismatch")

    training_tool = import_path("r0_test_frozen_training_tool", TRAINING_SOURCE)
    training_tool.install_type_only_octo_shim()
    sys.path.insert(0, str(CONRFT_RUNTIME_ROOT))
    import flax
    from flax import serialization
    import jax
    import jax.numpy as jnp
    import jaxlib
    import optax
    from serl_launcher.networks.reward_classifier import create_classifier

    require(jax.default_backend() == "gpu", f"GPU inference required, got {jax.default_backend()}")
    if prior_lock is not None:
        prior_lock_sha = calibration.sha256_file(lock_path)
        lock_path.replace(preflight_failure_path)
        atomic_json(preflight_failure_path, {
            "schema_version": "forcesmolvla_r0_zero_frame_backend_preflight_failure.v1",
            "artifact_status": "PRESERVED_ZERO_FRAME_CPU_BACKEND_PREFLIGHT_FAILURE",
            "original_lock_sha256": prior_lock_sha,
            "original_lock": prior_lock,
            "failure": "sandboxed JAX backend was cpu; stopped at GPU require before model creation",
            "model_created": False,
            "checkpoint_restored": False,
            "test_frames_inferred": 0,
            "candidate_evaluated": False,
            "optimizer_updates": 0,
            "recovery_backend": jax.default_backend(),
            "recovery_device": str(jax.devices()[0]),
        })
        manifest["bindings"]["one_shot_source"] = binding(Path(__file__).resolve())
        manifest["pre_inference_backend_preflight"] = {
            "failure_preserved": True,
            "test_frames_inferred": 0,
            "binding": {
                "path": str(preflight_failure_path),
                "file_size": preflight_failure_path.stat().st_size,
                "sha256": calibration.sha256_file(preflight_failure_path),
            },
        }
        atomic_json(work_dir / "cache_manifest.json", manifest)
    atomic_json(lock_path, {
        "started_at": utc_now(),
        "backend": jax.default_backend(),
        "device": str(jax.devices()[0]),
        "candidate_config_sha256": CANDIDATE_SHA256,
        "probability_threshold": TAU,
        "consecutive_positive_frames": REQUIRED_CONSECUTIVE,
        "authorization": "DETECTOR_CANDIDATE_APPROVED_FOR_ONE_SHOT_TEST=yes",
    })
    safe_tree, _ = training_tool.npz_encoder_tree()
    camera1 = np.load(work_dir / "camera1.npy", mmap_mode="r", allow_pickle=False)
    camera2 = np.load(work_dir / "camera2.npy", mmap_mode="r", allow_pickle=False)
    sample = {
        CLASSIFIER_CAMERA_KEYS[0]: jnp.asarray(np.asarray(camera1[0:1])[:, None]),
        CLASSIFIER_CAMERA_KEYS[1]: jnp.asarray(np.asarray(camera2[0:1])[:, None]),
    }
    with training_tool.trusted_safe_npz_pickle_bridge(safe_tree) as bridge:
        target = create_classifier(
            jax.random.PRNGKey(0), sample, list(CLASSIFIER_CAMERA_KEYS),
            pretrained_encoder_path=str(bridge), n_way=2,
        )
    state = serialization.from_bytes(target, CHECKPOINT_PATH.read_bytes())
    require(int(state.step) == 150, "restored checkpoint step mismatch")
    checkpoint_sha_before = calibration.sha256_file(CHECKPOINT_PATH)
    params_sha_before = training_tool.tree_sha(state.params)
    backbone_sha_before = training_tool.tree_sha(state.params, training_tool.is_backbone)

    @jax.jit
    def infer(params, observations):
        return state.apply_fn({"params": params}, observations, train=False)

    logits = []
    batch_count = 0
    for start in range(0, len(camera1), INFERENCE_BATCH_SIZE):
        stop = min(start + INFERENCE_BATCH_SIZE, len(camera1))
        observations = {
            CLASSIFIER_CAMERA_KEYS[0]: jnp.asarray(np.asarray(camera1[start:stop])[:, None]),
            CLASSIFIER_CAMERA_KEYS[1]: jnp.asarray(np.asarray(camera2[start:stop])[:, None]),
        }
        logits.append(np.asarray(jax.block_until_ready(infer(state.params, observations)), dtype=np.float32).reshape(-1))
        batch_count += 1
    logits_array = np.concatenate(logits)
    probabilities = training_tool.sigmoid(logits_array.astype(np.float64))
    require(len(logits_array) == manifest["test_frame_count"], "test inference frame count mismatch")
    require(np.all(np.isfinite(logits_array)) and np.all(np.isfinite(probabilities)), "non-finite classifier output")
    require(np.all((probabilities >= 0.0) & (probabilities <= 1.0)), "probability outside [0,1]")
    params_sha_after = training_tool.tree_sha(state.params)
    backbone_sha_after = training_tool.tree_sha(state.params, training_tool.is_backbone)
    checkpoint_sha_after = calibration.sha256_file(CHECKPOINT_PATH)
    require(params_sha_before == params_sha_after, "classifier parameters changed during test inference")
    require(backbone_sha_before == backbone_sha_after, "backbone changed during test inference")
    require(checkpoint_sha_before == checkpoint_sha_after == CHECKPOINT_SHA256, "checkpoint changed during test inference")
    require(int(state.step) == 150, "TrainState step changed during test inference")

    with tempfile.NamedTemporaryFile("wb", dir=work_dir, suffix=".npz", delete=False) as stream:
        np.savez(stream, logits=logits_array, probabilities=probabilities)
        temporary = Path(stream.name)
    temporary.replace(prediction_path)
    evidence = {
        "schema_version": "forcesmolvla_r0_one_shot_test_inference_evidence.v1",
        "artifact_status": "PASS_SINGLE_FROZEN_EVAL_MODE_TEST_INFERENCE",
        "created_at": utc_now(),
        "runtime": {
            "environment": os.environ["CONDA_DEFAULT_ENV"],
            "backend": jax.default_backend(),
            "device": str(jax.devices()[0]),
            "python": sys.version.split()[0],
            "jax": jax.__version__,
            "jaxlib": jaxlib.__version__,
            "flax": flax.__version__,
            "optax": optax.__version__,
        },
        "execution": {
            "eval_mode": True,
            "train_argument": False,
            "dropout_rng_supplied": False,
            "random_crop_or_augmentation": False,
            "optimizer_updates": 0,
            "train_state_step_before": 150,
            "train_state_step_after": int(state.step),
            "test_frames_inferred": len(logits_array),
            "inference_batch_count": batch_count,
            "one_shot_gpu_inference_invocation_count": 1,
            "failed_zero_frame_backend_preflight_count": 1 if prior_lock is not None else 0,
            "candidate_parameter_sets_evaluated": 1,
            "classifier_inference_failures": 0,
            "invalid_probability_count": 0,
        },
        "freeze_evidence": {
            "checkpoint_sha256_before": checkpoint_sha_before,
            "checkpoint_sha256_after": checkpoint_sha_after,
            "checkpoint_exactly_unchanged": checkpoint_sha_before == checkpoint_sha_after,
            "classifier_params_sha256_before": params_sha_before,
            "classifier_params_sha256_after": params_sha_after,
            "classifier_params_exactly_unchanged": params_sha_before == params_sha_after,
            "backbone_sha256_before": backbone_sha_before,
            "backbone_sha256_after": backbone_sha_after,
            "backbone_exactly_unchanged": backbone_sha_before == backbone_sha_after,
        },
        "access_audit": {
            "test_episode_order": list(TEST_ORDER),
            "test_episode_count": len(TEST_ORDER),
            "test_frame_count": len(logits_array),
            "train_frames_inferred": 0,
            "validation_frames_inferred": 0,
        },
        "bindings": {
            "best_checkpoint": binding(CHECKPOINT_PATH),
            "candidate_config": binding(CANDIDATE_PATH),
            "validation_calibration": binding(CALIBRATION_PATH),
            "cache_manifest_sha256": calibration.sha256_file(work_dir / "cache_manifest.json"),
            "one_shot_start_lock_sha256": calibration.sha256_file(lock_path),
            "zero_frame_backend_preflight_failure": None if prior_lock is None else {
                "path": str(preflight_failure_path),
                "file_size": preflight_failure_path.stat().st_size,
                "sha256": calibration.sha256_file(preflight_failure_path),
            },
            "predictions": {
                "path": str(prediction_path),
                "file_size": prediction_path.stat().st_size,
                "sha256": calibration.sha256_file(prediction_path),
            },
        },
    }
    atomic_json(evidence_path, evidence)
    print(json.dumps({"phase": "one_shot_test_gpu_inference", "status": "pass", "frames": len(logits_array), "candidate_count": 1}), flush=True)


def true_run_lengths(values: np.ndarray) -> list[int]:
    runs: list[int] = []
    current = 0
    for value in values:
        if bool(value):
            current += 1
        elif current:
            runs.append(current)
            current = 0
    if current:
        runs.append(current)
    return runs


def frame_metrics(logits: np.ndarray, class_codes: np.ndarray) -> dict[str, Any]:
    training_tool = import_path("r0_test_metrics_training_tool", TRAINING_SOURCE)
    mask = class_codes != CLASS_CODE["ambiguous"]
    logits = np.asarray(logits[mask], dtype=np.float64)
    strata = np.asarray(class_codes[mask], dtype=np.uint8)
    labels = (strata == CLASS_CODE["positive"]).astype(np.uint8)
    probabilities = training_tool.sigmoid(logits)
    predictions = probabilities >= TAU
    positive = labels == 1
    negative = ~positive
    tp = int(np.sum(predictions & positive))
    fn = int(np.sum(~predictions & positive))
    fp = int(np.sum(predictions & negative))
    tn = int(np.sum(~predictions & negative))
    recall = tp / (tp + fn)
    specificity = tn / (tn + fp)
    losses = np.maximum(logits, 0.0) - logits * labels + np.log1p(np.exp(-np.abs(logits)))

    def stratum_fpr(name: str) -> float:
        stratum = strata == CLASS_CODE[name]
        require(np.any(stratum), f"empty test stratum: {name}")
        return float(np.mean(predictions[stratum]))

    require(not np.any(class_codes == CLASS_CODE["ambiguous"]), "ambiguous test sample consumed")
    return {
        "BCE": float(np.mean(losses)),
        "ROC_AUC": training_tool.roc_auc(labels, probabilities),
        "PR_AUC": training_tool.average_precision(labels, probabilities),
        "PR_AUC_definition": "average_precision_step_integral",
        "balanced_accuracy": float((recall + specificity) / 2.0),
        "positive_recall": float(recall),
        "overall_false_positive_rate": float(fp / (fp + tn)),
        "ordinary_negative_false_positive_rate": stratum_fpr("ordinary_negative"),
        "hard_negative_false_positive_rate": stratum_fpr("hard_negative"),
        "confusion_matrix": {
            "true_negative": tn,
            "false_positive": fp,
            "false_negative": fn,
            "true_positive": tp,
        },
        "probability_threshold": TAU,
        "evaluated_frame_count": len(labels),
        "ambiguous_frame_count_consumed": 0,
    }


def detector_self_check() -> None:
    calibration.detector_self_check()
    require(calibration.causal_trigger([0, 1, 2, 3, 4], [TAU] * 5, [True] * 5, TAU, 5) == 4, "fifth-frame trigger failed")
    require(calibration.causal_trigger([0, 1, 3, 4, 5], [1.0] * 5, [True] * 5, TAU, 3) == 5, "gap reset failed")
    require(calibration.causal_trigger([0, 1, 2, 3, 4], [1.0] * 5, [True, True, False, True, True], TAU, 3) is None, "validity reset failed")


def make_markdown(artifact: Mapping[str, Any], spec_binding: Mapping[str, Any], artifact_sha: str) -> str:
    metrics = artifact["frame_metrics"]
    acceptance = artifact["acceptance"]
    lines = [
        "# R0 one-shot development test evaluation",
        "",
        f"- Decision: **{acceptance['decision']}**",
        f"- One-shot test artifact SHA256: `{artifact_sha}`",
        f"- DetectorSpec/disposition SHA256: `{spec_binding['sha256']}`",
        f"- Frozen checkpoint SHA256: `{CHECKPOINT_SHA256}`",
        f"- Frozen candidate SHA256: `{CANDIDATE_SHA256}`",
        f"- Frozen validation calibration SHA256: `{CALIBRATION_SHA256}`",
        "- Scope: four frozen development test episodes; exactly one frozen candidate; no reselection.",
        "",
        "## Frozen detector",
        "",
        f"`tau={TAU}`, `M={REQUIRED_CONSECUTIVE}`, `{FPS} Hz`, causal current-frame trigger, latch enabled.",
        "",
        "## Episode results",
        "",
        "| episode | completion | trigger | delay frames | delay ms | early | missed | pre max run | post min run | post max run |",
        "|---|---:|---:|---:|---:|:---:|:---:|---:|---:|---:|",
    ]
    for episode in artifact["episode_results"]:
        lines.append(
            f"| {episode['episode_id']} | {episode['first_confident_complete_frame']} | {episode['trigger_frame']} "
            f"| {episode['delay_frames']} | {episode['delay_ms']:.3f} | {'yes' if episode['early_trigger'] else 'no'} "
            f"| {'yes' if episode['missed_success'] else 'no'} | {episode['maximum_precompletion_positive_run']} "
            f"| {episode['minimum_postcompletion_positive_run']} | {episode['maximum_postcompletion_positive_run']} |"
        )
    lines += [
        "",
        "## Acceptance",
        "",
        f"- early triggers: {acceptance['observed']['early_trigger_episode_count']} (required 0)",
        f"- missed successes: {acceptance['observed']['missed_success_episode_count']} (required 0)",
        f"- maximum delay: {acceptance['observed']['max_detection_delay_frames']} frames / {acceptance['observed']['max_detection_delay_ms']:.3f} ms (required <=6 / <=200.0)",
        "",
        "## Frame metrics at frozen tau",
        "",
        f"- BCE: {metrics['BCE']:.9f}",
        f"- ROC-AUC: {metrics['ROC_AUC']:.9f}",
        f"- PR-AUC: {metrics['PR_AUC']:.9f}",
        f"- balanced accuracy: {metrics['balanced_accuracy']:.9f}",
        f"- positive recall: {metrics['positive_recall']:.9f}",
        f"- ordinary-negative FPR: {metrics['ordinary_negative_false_positive_rate']:.9f}",
        f"- hard-negative FPR: {metrics['hard_negative_false_positive_rate']:.9f}",
        f"- confusion matrix: {metrics['confusion_matrix']}",
        f"- longest pre-completion positive run: {artifact['run_metrics']['longest_precompletion_positive_run']}",
        f"- shortest post-completion positive run: {artifact['run_metrics']['shortest_postcompletion_positive_run']}",
        f"- shortest sustained post-completion run across episodes: {artifact['run_metrics']['shortest_sustained_postcompletion_run_across_episodes']}",
        "",
        "The JSON artifact contains all 3,040 original-order frame probabilities and exact metric definitions.",
        "",
        "## Audit and status",
        "",
        "- Test GPU inference invocation count: 1; frozen candidate parameter sets evaluated: 1.",
        "- One sandboxed CPU-backend preflight stopped before model creation with 0 test frames inferred; its evidence is preserved.",
        "- Eval mode only; no dropout, crop, random augmentation, or optimizer update.",
        "- No train/validation image was opened by this run; no reward/terminal, G1/G2, critic, Cal-QL, Actor, online, or robot artifact was created.",
        "",
        "```text",
    ]
    for key, value in artifact["terminal_status"].items():
        lines.append(f"{key} = {value}")
    lines += ["```", ""]
    return "\n".join(lines)


def finalize(work_dir: Path) -> None:
    """Evaluate only the frozen tau/M and atomically publish append-only evidence."""
    detector_self_check()
    frozen = verify_frozen_inputs()
    cache = verify_work_cache(work_dir)
    evidence_path = work_dir / "inference_evidence.json"
    prediction_path = work_dir / "test_predictions.npz"
    lock_path = work_dir / "ONE_SHOT_INFERENCE_STARTED.json"
    require(evidence_path.is_file() and prediction_path.is_file() and lock_path.is_file(), "one-shot inference evidence missing")
    evidence = load_json(evidence_path)
    require(evidence["artifact_status"] == "PASS_SINGLE_FROZEN_EVAL_MODE_TEST_INFERENCE", "test inference did not pass")
    require(evidence["execution"]["one_shot_gpu_inference_invocation_count"] == 1, "one-shot count invalid")
    require(evidence["execution"]["candidate_parameter_sets_evaluated"] == 1, "candidate count invalid")
    require(evidence["execution"]["optimizer_updates"] == 0, "optimizer update detected")
    require(evidence["bindings"]["predictions"]["sha256"] == calibration.sha256_file(prediction_path), "prediction SHA mismatch")
    require(not TEST_ARTIFACT_PATH.exists() and not REPORT_PATH.exists(), "append-only report target already exists")
    require(not PASS_SPEC_PATH.exists() and not REJECTED_SPEC_PATH.exists(), "one-shot disposition target already exists")

    arrays = np.load(prediction_path, allow_pickle=False)
    logits = arrays["logits"]
    probabilities = arrays["probabilities"]
    frame_indices = np.load(work_dir / "frame_indices.npy", allow_pickle=False)
    class_codes = np.load(work_dir / "class_codes.npy", allow_pickle=False)
    valid = np.load(work_dir / "valid.npy", allow_pickle=False)
    require(len(logits) == len(probabilities) == len(frame_indices) == len(class_codes) == len(valid) == 3040, "test array length mismatch")
    require(np.all(valid), "invalid test input reached finalization")

    episode_results = []
    timelines = []
    delays = []
    early_count = 0
    missed_count = 0
    pre_runs = []
    post_min_runs = []
    post_max_runs = []
    for episode in cache["episodes"]:
        start, stop = episode["cache_range_half_open"]
        frames = frame_indices[start:stop]
        probs = probabilities[start:stop]
        episode_logits = logits[start:stop]
        codes = class_codes[start:stop]
        episode_valid = valid[start:stop]
        completion = episode["first_confident_complete_frame"]
        require(np.array_equal(frames, np.arange(episode["frame_count"])), "test order/frame gap")
        threshold_positive = episode_valid & (probs >= TAU)
        trigger = calibration.causal_trigger(frames, probs, episode_valid, TAU, REQUIRED_CONSECUTIVE)
        delay_frames = None if trigger is None else int(trigger - completion)
        delay_ms = None if delay_frames is None else delay_frames * 1000.0 / FPS
        pre_run = calibration.longest_true_run(threshold_positive[frames < completion])
        post_lengths = true_run_lengths(threshold_positive[frames >= completion])
        post_min = min(post_lengths) if post_lengths else 0
        post_max = max(post_lengths) if post_lengths else 0
        early = trigger is not None and trigger < completion
        missed = trigger is None
        early_count += int(early)
        missed_count += int(missed)
        if delay_frames is not None:
            delays.append(delay_frames)
        pre_runs.append(pre_run)
        post_min_runs.append(post_min)
        post_max_runs.append(post_max)
        episode_results.append({
            "episode_id": episode["episode_id"],
            "first_confident_complete_frame": completion,
            "trigger_frame": trigger,
            "delay_frames": delay_frames,
            "delay_ms": delay_ms,
            "early_trigger": early,
            "missed_success": missed,
            "maximum_precompletion_positive_run": pre_run,
            "minimum_postcompletion_positive_run": post_min,
            "maximum_postcompletion_positive_run": post_max,
        })
        timelines.append({
            "episode_id": episode["episode_id"],
            "frame_count": episode["frame_count"],
            "first_confident_complete_frame": completion,
            "trigger_frame": trigger,
            "frame_indices": frames.astype(int).tolist(),
            "logits": episode_logits.astype(float).tolist(),
            "probabilities": probs.astype(float).tolist(),
            "frame_classes": [CLASS_NAMES[int(code)] for code in codes],
            "threshold_positive": threshold_positive.astype(bool).tolist(),
            "valid": episode_valid.astype(bool).tolist(),
        })

    max_delay_frames = None if not delays else int(max(delays))
    max_delay_ms = None if max_delay_frames is None else max_delay_frames * 1000.0 / FPS
    conditions = {
        "early_trigger_episode_count_equals_0": early_count == 0,
        "missed_success_episode_count_equals_0": missed_count == 0,
        "max_detection_delay_frames_lte_6": max_delay_frames is not None and max_delay_frames <= 6,
        "max_detection_delay_ms_lte_200": max_delay_ms is not None and max_delay_ms <= 200.0,
    }
    passed = all(conditions.values())
    metrics = frame_metrics(logits, class_codes)
    artifact_status = "PASS_ONE_SHOT_DEVELOPMENT_TEST" if passed else "FAIL_ONE_SHOT_DEVELOPMENT_TEST_CANDIDATE_REJECTED"
    next_action = "request_G1_approval" if passed else "return_to_validation_redesign_or_collect_independent_calibration_episodes"
    terminal = {
        "DETECTOR_CANDIDATE_APPROVED_FOR_ONE_SHOT_TEST": "yes",
        "ONE_SHOT_TEST_EVALUATION": "complete",
        "ONE_SHOT_DEVELOPMENT_TEST_ACCEPTANCE": "PASS" if passed else "FAIL",
        "DEVELOPMENT_DETECTOR_SPEC_APPROVED": "yes" if passed else "no",
        "FORMAL_DETECTOR_SPEC_APPROVED": "no",
        "PRODUCTION_DETECTOR_SPEC_APPROVED": "no",
        "CLASSIFIER_CHECKPOINT_FROZEN": "yes",
        "CLASSIFIER_RETRAINED": "no",
        "OPTIMIZER_UPDATES": "0",
        "TEST_EVALUATED": "yes_once",
        "TASK2_REWARD_TERMINAL_CREATED": "no",
        "G1_CREATED": "no",
        "G2_CREATED": "no",
        "NEXT_ALLOWED_ACTION": next_action,
    }
    artifact = {
        "schema_version": "forcesmolvla_r0_one_shot_test_evaluation.v1",
        "artifact_status": artifact_status,
        "created_at": utc_now(),
        "scope": "one_shot_development_test_evaluation_only",
        "authorization": {
            "DETECTOR_CANDIDATE_APPROVED_FOR_ONE_SHOT_TEST": "yes",
            "does_not_approve_formal_or_production_detector_spec": True,
        },
        "frozen_detector": {
            "probability_threshold": TAU,
            "consecutive_positive_frames": REQUIRED_CONSECUTIVE,
            "detector_frequency_hz": FPS,
            "latch_after_trigger": True,
            "trigger_backfilled": False,
            "candidate_parameter_sets_evaluated": 1,
        },
        "frozen_bindings": {
            "classifier_checkpoint": binding(CHECKPOINT_PATH),
            "candidate_config": binding(CANDIDATE_PATH),
            "validation_calibration": binding(CALIBRATION_PATH),
            "training_report": binding(TRAINING_REPORT_PATH),
            "reviewed_labels": binding(REVIEWED_PATH),
            "inventory": binding(INVENTORY_PATH),
            "readiness": binding(READINESS_PATH),
            "split_manifest": binding(SPLIT_PATH),
            "safe_resnet10_npz": binding(SAFE_ASSET_PATH),
            "adapter_source": binding(ADAPTER_PATH),
            "training_source": binding(TRAINING_SOURCE),
            "validation_calibration_source": binding(CALIBRATION_SOURCE),
            "one_shot_test_source": binding(Path(__file__).resolve()),
        },
        "preprocessing_and_camera_contract": {
            "source_camera_keys_in_order": list(SOURCE_CAMERA_KEYS),
            "classifier_camera_keys_in_order": list(CLASSIFIER_CAMERA_KEYS),
            "source_color_order": "RGB",
            "source_image_shape_HWC": list(IMAGE_SHAPE),
            "classifier_input_shape_per_camera": ["batch", 1, 480, 640, 3],
            "frame_stack": 1,
            "resizing_and_imagenet_normalization_owner": "frozen unmodified ConRFT ResNet10 encoder",
            "random_crop": False,
            "random_augmentation": False,
            "dropout": False,
            "eval_mode": True,
        },
        "inference_and_freeze_evidence": evidence,
        "test_access_audit": {
            **cache["access_audit"],
            "test_episode_order": list(TEST_ORDER),
            "test_frames_inferred": 3040,
            "one_shot_gpu_inference_invocation_count": 1,
            "candidate_parameter_sets_evaluated": 1,
            "alternate_tau_M_or_checkpoint_evaluated": False,
            "test_used_for_checkpoint_or_parameter_selection": False,
            "train_images_opened_or_inferred": 0,
            "validation_images_opened_or_inferred": 0,
        },
        "metric_definitions": {
            "frame_metric_threshold": "frozen detector probability threshold tau=0.83; metrics are pre-latch",
            "BCE": "mean sigmoid binary cross entropy from logits; positive class only for positive frame labels",
            "ambiguous": "fully excluded; frozen test inventory contains zero ambiguous frames",
            "trigger": "current fifth consecutive valid p_t>=tau frame; never backfilled; latched within episode",
            "counter_reset": "episode start/end, non-consecutive index, invalid/missing image, inference failure, or input validity failure",
            "minimum_postcompletion_positive_run": "shortest nonzero threshold-positive run at/after completion within an episode; 0 if none",
            "shortest_postcompletion_positive_run": "minimum nonzero post-completion threshold-positive run over all episodes",
            "shortest_sustained_postcompletion_run_across_episodes": "minimum across episodes of each episode's maximum post-completion threshold-positive run; matches validation calibration convention",
        },
        "frame_metrics": metrics,
        "run_metrics": {
            "longest_precompletion_positive_run": int(max(pre_runs)),
            "shortest_postcompletion_positive_run": int(min(post_min_runs)),
            "shortest_sustained_postcompletion_run_across_episodes": int(min(post_max_runs)),
        },
        "episode_results": episode_results,
        "test_probability_timelines": timelines,
        "acceptance": {
            "frozen_before_test": {
                "early_trigger_episode_count": 0,
                "missed_success_episode_count": 0,
                "max_detection_delay_frames_lte": 6,
                "max_detection_delay_ms_lte": 200.0,
            },
            "observed": {
                "early_trigger_episode_count": early_count,
                "missed_success_episode_count": missed_count,
                "detected_success_episode_count": len(TEST_ORDER) - missed_count,
                "max_detection_delay_frames": max_delay_frames,
                "max_detection_delay_ms": max_delay_ms,
            },
            "conditions": conditions,
            "decision": "PASS" if passed else "FAIL",
        },
        "candidate_disposition": "DEVELOPMENT_APPROVED_NOT_FORMAL_OR_PRODUCTION" if passed else "REJECTED_NO_TEST_DRIVEN_RESELECTION_ALLOWED",
        "forbidden_outputs_created": [],
        "terminal_status": terminal,
    }

    staging = Path(tempfile.mkdtemp(prefix=".r0-one-shot-test-", dir=ROOT))
    try:
        staged_artifact = staging / TEST_ARTIFACT_PATH.name
        atomic_json(staged_artifact, artifact)
        artifact_binding = {
            "path": relative(TEST_ARTIFACT_PATH),
            "file_size": staged_artifact.stat().st_size,
            "sha256": calibration.sha256_file(staged_artifact),
        }
        spec_path = PASS_SPEC_PATH if passed else REJECTED_SPEC_PATH
        spec = {
            "schema_version": "forcesmolvla_r0_reward_detector_spec.v1",
            "artifact_status": "DEVELOPMENT_APPROVED_NOT_FORMAL_OR_PRODUCTION" if passed else "REJECTED_AFTER_FROZEN_ONE_SHOT_TEST",
            "created_at": utc_now(),
            "approval_scope": "development_only" if passed else "none_candidate_rejected",
            "formal_ready": False,
            "production_ready": False,
            "detector": {
                "probability_threshold": TAU,
                "consecutive_positive_frames": REQUIRED_CONSECUTIVE,
                "detector_frequency_hz": FPS,
                "latch_after_trigger": True,
                "trigger_on_current_Mth_frame_without_backfill": True,
                "reset_on": frozen["candidate"]["causal_semantics"]["reset_on"],
            },
            "preprocessing_and_camera_contract": artifact["preprocessing_and_camera_contract"],
            "bindings": {
                "classifier_checkpoint": binding(CHECKPOINT_PATH),
                "candidate_config": binding(CANDIDATE_PATH),
                "validation_calibration": binding(CALIBRATION_PATH),
                "one_shot_test_report": artifact_binding,
                "one_shot_test_source": binding(Path(__file__).resolve()),
                "validation_calibration_source": binding(CALIBRATION_SOURCE),
                "adapter_source": binding(ADAPTER_PATH),
            },
            "test_acceptance": artifact["acceptance"],
            "permissions": {
                "formal_or_production_use": False,
                "G1_created_or_authorized_by_this_artifact": False,
                "reward_or_terminal_created": False,
            },
            "next_allowed_action": next_action,
        }
        staged_spec = staging / spec_path.name
        atomic_json(staged_spec, spec)
        spec_binding = {
            "path": relative(spec_path),
            "file_size": staged_spec.stat().st_size,
            "sha256": calibration.sha256_file(staged_spec),
        }
        staged_report = staging / REPORT_PATH.name
        atomic_text(staged_report, make_markdown(artifact, spec_binding, artifact_binding["sha256"]))
        TEST_ARTIFACT_PATH.parent.mkdir(parents=True, exist_ok=True)
        spec_path.parent.mkdir(parents=True, exist_ok=True)
        REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
        staged_artifact.replace(TEST_ARTIFACT_PATH)
        staged_spec.replace(spec_path)
        staged_report.replace(REPORT_PATH)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    print(json.dumps({
        "phase": "one_shot_test_acceptance",
        "status": "pass" if passed else "fail",
        "decision": "PASS" if passed else "FAIL",
        "early": early_count,
        "missed": missed_count,
        "max_delay_frames": max_delay_frames,
        "spec": relative(PASS_SPEC_PATH if passed else REJECTED_SPEC_PATH),
    }, sort_keys=True), flush=True)


def static_check() -> None:
    detector_self_check()
    frozen = verify_frozen_inputs()
    inventory_by_id = {episode["episode_id"]: episode for episode in frozen["inventory"]["episodes"]}
    reviewed_by_id = {episode["episode_id"]: episode for episode in frozen["reviewed"]["episodes"]}
    require(sum(inventory_by_id[episode_id]["frame_count"] for episode_id in TEST_ORDER) == 3040, "test metadata total drift")
    for episode_id in TEST_ORDER:
        episode = inventory_by_id[episode_id]
        review = reviewed_by_id[episode_id]
        require(episode["split"] == review["split"] == "test", "test metadata split drift")
        require(review["manual_review_status"] == "human_reviewed", "test review status drift")
        require(episode["class_frame_counts"]["ambiguous"] == 0, "ambiguous test inventory drift")
    for path in (TEST_ARTIFACT_PATH, PASS_SPEC_PATH, REJECTED_SPEC_PATH, REPORT_PATH):
        require(not path.exists(), f"append-only target already exists: {path}")
    print(json.dumps({
        "phase": "static_pre_test_check",
        "status": "pass",
        "test_images_read": 0,
        "test_inference": 0,
        "test_episodes": list(TEST_ORDER),
        "test_frames": 3040,
    }, sort_keys=True), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("static-check")
    prepare = subparsers.add_parser("prepare-cache")
    prepare.add_argument("--work-dir", type=Path, required=True)
    infer = subparsers.add_parser("infer")
    infer.add_argument("--work-dir", type=Path, required=True)
    finish = subparsers.add_parser("finalize")
    finish.add_argument("--work-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "static-check":
        static_check()
    elif args.command == "prepare-cache":
        prepare_cache(args.work_dir.resolve())
    elif args.command == "infer":
        run_inference(args.work_dir.resolve())
    else:
        finalize(args.work_dir.resolve())


if __name__ == "__main__":
    main()
