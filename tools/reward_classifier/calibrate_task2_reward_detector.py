#!/usr/bin/env python3
"""Validation-only causal Reward Detector calibration for the frozen R0 model."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
from decimal import Decimal
import hashlib
import importlib.util
from io import BytesIO
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from typing import Any, Iterator, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
CONRFT_ROOT = Path("/home/rlc123/conrft")
CONRFT_RUNTIME_ROOT = CONRFT_ROOT / "serl_launcher"
TRAINING_TOOL_PATH = ROOT / "tools/reward_classifier/train_reward_classifier.py"
ADAPTER_PATH = ROOT / "tools/reward_classifier/conrft_lerobot_v3_adapter.py"
CHECKPOINT_PATH = ROOT / "outputs/task2/reward_classifier/checkpoints/best/best_checkpoint.msgpack"
TRAINING_REPORT_PATH = ROOT / "artifacts/development/stage2/reward_classifier/r0_training/r0_training_validation_report.v1.json"
INVENTORY_PATH = ROOT / "artifacts/development/stage2/reward_classifier/task2_frame_label_inventory.v2.json"
READINESS_PATH = ROOT / "artifacts/development/stage2/s2_r0_label_ingestion_readiness.v4.json"
REVIEWED_PATH = ROOT / "labels/task2_reward_frame_labels.v2.reviewed.json"
SPLIT_PATH = ROOT / "datasets/task2_lerobotv3/split_manifest.json"
DATASET_ROOT = ROOT / "datasets/task2_lerobotv3"
SAFE_ASSET_PATH = ROOT / "artifacts/development/stage2/reward_classifier/pretrained/resnet10_params.safe.npz"

CANDIDATE_PATH = ROOT / "configs/stage2_r0_reward_detector.candidate.development.json"
CALIBRATION_PATH = ROOT / "artifacts/development/stage2/reward_classifier/r0_validation_detector_calibration.v1.json"
REPORT_PATH = ROOT / "docs/r0_validation_detector_calibration_report.md"

BEST_CHECKPOINT_SHA256 = "6b4e366baa55993d150cb3dd86e67a1d708e58d836b123a0c433190835021510"
TRAINING_REPORT_SHA256 = "c48d845f77b3b7b46b5788998c412c280d93e0e8a863d5edeb488fcea8cb2aac"
INVENTORY_SHA256 = "8839793f0e5d5c6d866b41e32bcb7fa576cd984a9faf5507719a1735be611a65"
READINESS_SHA256 = "64ae61e7d83c7be49451f4716c0e95921c2e9dbd062a553cec8f7fccdcc690aa"
REVIEWED_SHA256 = "ecda7d480f6a4c49dbe63a31b7e3172b30a5470437510522b1da2217eae77a9c"
SAFE_ASSET_SHA256 = "16052142a3ef841a12fb1d2a03965951e8fbf0dda3d89b995244419be7e1f9a5"
CONRFT_COMMIT = "a779fde7fa5db5a469960a8490c100f35b41b49e"

SOURCE_CAMERA_KEYS = ("observation.images.camera1", "observation.images.camera2")
CLASSIFIER_CAMERA_KEYS = ("d435_third_person", "d405_wrist")
IMAGE_SHAPE = (480, 640, 3)
CLASS_NAMES = ("positive", "ordinary_negative", "hard_negative", "ambiguous")
CLASS_CODE = {name: index for index, name in enumerate(CLASS_NAMES)}
M_VALUES = (1, 2, 3, 4, 5, 6, 8, 10, 12, 15)
TAU_DECIMALS = tuple(Decimal(index) / Decimal(100) for index in range(50, 100)) + (
    Decimal("0.995"),
    Decimal("0.999"),
)
FPS = 30
INFERENCE_BATCH_SIZE = 128


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


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
    return {"path": relative(path), "file_size": path.stat().st_size, "sha256": sha256_file(path)}


def import_path(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    require(spec is not None and spec.loader is not None, f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def git(*arguments: str) -> str:
    return subprocess.run(
        ["git", "-C", str(CONRFT_ROOT), *arguments],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def verify_frozen_inputs() -> dict[str, Any]:
    expected = {
        CHECKPOINT_PATH: BEST_CHECKPOINT_SHA256,
        TRAINING_REPORT_PATH: TRAINING_REPORT_SHA256,
        INVENTORY_PATH: INVENTORY_SHA256,
        READINESS_PATH: READINESS_SHA256,
        REVIEWED_PATH: REVIEWED_SHA256,
        SAFE_ASSET_PATH: SAFE_ASSET_SHA256,
    }
    for path, digest in expected.items():
        require(path.is_file(), f"frozen input missing: {path}")
        require(sha256_file(path) == digest, f"frozen input SHA mismatch: {path}")
    require(git("rev-parse", "HEAD") == CONRFT_COMMIT, "ConRFT commit mismatch")
    require(git("status", "--porcelain") == "", "ConRFT worktree modified")

    training = load_json(TRAINING_REPORT_PATH)
    inventory = load_json(INVENTORY_PATH)
    reviewed = load_json(REVIEWED_PATH)
    readiness = load_json(READINESS_PATH)
    split = load_json(SPLIT_PATH)
    require(training["artifact_status"] == "PASS_R0_DEVELOPMENT_CLASSIFIER_TRAINING_COMPLETE", "training report did not pass")
    require(training["primary_training_run"]["best_checkpoint_sha256"] == BEST_CHECKPOINT_SHA256, "best checkpoint/report mismatch")
    require(training["primary_training_run"]["best_optimizer_update"] == 150, "best checkpoint update mismatch")
    require(training["terminal_status"]["DETECTOR_THRESHOLD_APPROVED"] == "no", "detector already approved")
    require(training["terminal_status"]["TEST_EVALUATED"] == "no", "test was previously evaluated")
    require(inventory["validation"]["schema_valid"] is True, "label schema invalid")
    require(inventory["validation"]["intervals_valid"] is True, "label intervals invalid")
    require(inventory["validation"]["episode_leakage"] is False, "episode leakage")
    require(inventory["validation"]["row_leakage"] is False, "row leakage")
    require(readiness["readiness"]["DEVELOPMENT_R0_TRAINING_DATA_READY"] == "yes", "readiness invalid")

    val_order = split["val"]
    require(val_order == [
        "episode_000007", "episode_000012", "episode_000017", "episode_000037", "episode_000039"
    ], "validation episode order drift")
    split_sets = {name: set(split[name]) for name in ("train", "val", "test")}
    require(not (split_sets["train"] & split_sets["val"] | split_sets["train"] & split_sets["test"] | split_sets["val"] & split_sets["test"]), "split overlap")
    return {
        "training": training,
        "inventory": inventory,
        "reviewed": reviewed,
        "readiness": readiness,
        "split": split,
        "validation_order": val_order,
    }


def frame_classes(episode: Mapping[str, Any]) -> np.ndarray:
    result = np.full(episode["frame_count"], 255, dtype=np.uint8)
    for class_name in CLASS_NAMES:
        for start, stop in episode["class_intervals_inclusive"][class_name]:
            require(0 <= start <= stop < len(result), "class interval outside episode")
            require(np.all(result[start : stop + 1] == 255), "overlapping class intervals")
            result[start : stop + 1] = CLASS_CODE[class_name]
    require(np.all(result != 255), "unlabeled validation frame")
    return result


def decode_rgb(payload: bytes) -> np.ndarray:
    from PIL import Image

    require(isinstance(payload, bytes), "embedded image bytes missing")
    with Image.open(BytesIO(payload)) as image:
        rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    require(rgb.shape == IMAGE_SHAPE, f"decoded image shape mismatch: {rgb.shape}")
    return np.ascontiguousarray(rgb)


def prepare_cache(work_dir: Path) -> None:
    import pyarrow.parquet as pq

    require(not work_dir.exists(), f"work directory already exists: {work_dir}")
    frozen = verify_frozen_inputs()
    inventory_by_id = {episode["episode_id"]: episode for episode in frozen["inventory"]["episodes"]}
    reviewed_by_id = {episode["episode_id"]: episode for episode in frozen["reviewed"]["episodes"]}
    validation_order = frozen["validation_order"]
    total_frames = sum(inventory_by_id[episode_id]["frame_count"] for episode_id in validation_order)
    require(total_frames == 3775, "validation frame total mismatch")

    staging = work_dir.parent / f".{work_dir.name}.tmp-{os.getpid()}"
    require(not staging.exists(), f"work staging already exists: {staging}")
    staging.mkdir(parents=True)
    try:
        camera1 = np.lib.format.open_memmap(staging / "camera1.npy", mode="w+", dtype=np.uint8, shape=(total_frames, *IMAGE_SHAPE))
        camera2 = np.lib.format.open_memmap(staging / "camera2.npy", mode="w+", dtype=np.uint8, shape=(total_frames, *IMAGE_SHAPE))
        frame_indices = np.empty(total_frames, dtype=np.int32)
        episode_codes = np.empty(total_frames, dtype=np.uint8)
        class_codes = np.empty(total_frames, dtype=np.uint8)
        valid = np.ones(total_frames, dtype=np.bool_)
        adapter_module = import_path("r0_calibration_adapter", ADAPTER_PATH)
        adapter = adapter_module.ConRFTLeRobotV3Adapter()
        cursor = 0
        episodes = []
        opened_files = []
        for episode_code, episode_id in enumerate(validation_order):
            episode = inventory_by_id[episode_id]
            review = reviewed_by_id[episode_id]
            require(episode["split"] == "validation", "non-validation episode requested")
            require(review["manual_review_status"] == "human_reviewed", "validation episode not human reviewed")
            completion = int(review["first_confident_complete_frame"])
            require(episode["class_intervals_inclusive"]["positive"] == [[completion, episode["frame_count"] - 1]], "positive boundary mismatch")
            classes = frame_classes(episode)
            parquet_path = (DATASET_ROOT / episode["source_data_relative_path"]).resolve()
            require(parquet_path.is_relative_to(DATASET_ROOT.resolve()), "validation parquet path escape")
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
            require(table.num_rows == episode["frame_count"], "validation parquet row count mismatch")
            start_offset = cursor
            for frame in range(episode["frame_count"]):
                row = table.slice(frame, 1).to_pylist()[0]
                require(row["frame_index"] == frame, "non-consecutive validation frame")
                require(row["episode_index"] == episode["output_episode_index"], "validation episode index mismatch")
                require(np.isfinite(float(row["timestamp"])), "invalid validation timestamp")
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
                episode_codes[cursor] = episode_code
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
        require(cursor == total_frames and adapter.episode_reset_count == 5, "validation cache completeness failure")
        camera1.flush()
        camera2.flush()
        del camera1, camera2
        np.save(staging / "frame_indices.npy", frame_indices, allow_pickle=False)
        np.save(staging / "episode_codes.npy", episode_codes, allow_pickle=False)
        np.save(staging / "class_codes.npy", class_codes, allow_pickle=False)
        np.save(staging / "valid.npy", valid, allow_pickle=False)
        files = {}
        for name in ("camera1.npy", "camera2.npy", "frame_indices.npy", "episode_codes.npy", "class_codes.npy", "valid.npy"):
            path = staging / name
            files[name] = {"file_size": path.stat().st_size, "sha256": sha256_file(path)}
        manifest = {
            "schema_version": "forcesmolvla_r0_validation_only_cache.v1",
            "artifact_status": "COMPLETE_EPHEMERAL_VALIDATION_ONLY_CACHE",
            "created_at": utc_now(),
            "validation_episode_order": validation_order,
            "validation_episode_count": 5,
            "validation_frame_count": total_frames,
            "episodes": episodes,
            "input_validity": {
                "all_frame_indices_consecutive": True,
                "all_images_present_and_valid": True,
                "all_adapter_checks_passed": True,
                "invalid_frame_count": 0,
            },
            "access_audit": {
                "validation_parquet_files_opened": opened_files,
                "validation_image_rows_loaded_and_decoded": total_frames,
                "train_parquet_files_opened": [],
                "test_parquet_files_opened": [],
                "test_image_rows_loaded": 0,
                "test_images_decoded": 0,
            },
            "bindings": {
                "best_checkpoint": binding(CHECKPOINT_PATH),
                "training_report": binding(TRAINING_REPORT_PATH),
                "inventory": binding(INVENTORY_PATH),
                "readiness": binding(READINESS_PATH),
                "reviewed_labels": binding(REVIEWED_PATH),
                "split_manifest": binding(SPLIT_PATH),
                "adapter": binding(ADAPTER_PATH),
            },
            "files": files,
        }
        atomic_json(staging / "cache_manifest.json", manifest)
        staging.replace(work_dir)
        print(json.dumps({"phase": "prepare_validation_cache", "status": "pass", "frames": total_frames, "test_images": 0}), flush=True)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def verify_work_cache(work_dir: Path) -> dict[str, Any]:
    manifest = load_json(work_dir / "cache_manifest.json")
    require(manifest["artifact_status"] == "COMPLETE_EPHEMERAL_VALIDATION_ONLY_CACHE", "validation cache incomplete")
    require(manifest["access_audit"]["test_image_rows_loaded"] == 0, "test image entered cache")
    require(manifest["access_audit"]["test_parquet_files_opened"] == [], "test parquet entered cache")
    for name, value in manifest["files"].items():
        path = work_dir / name
        require(path.stat().st_size == value["file_size"], f"cache file size mismatch: {name}")
        require(sha256_file(path) == value["sha256"], f"cache file SHA mismatch: {name}")
    return manifest


def run_inference(work_dir: Path) -> None:
    require(os.environ.get("CONDA_DEFAULT_ENV") == "conrft_reward", "inference must run in conrft_reward")
    frozen = verify_frozen_inputs()
    manifest = verify_work_cache(work_dir)
    prediction_path = work_dir / "validation_predictions.npz"
    evidence_path = work_dir / "inference_evidence.json"
    require(not prediction_path.exists() and not evidence_path.exists(), "inference output already exists")

    training_tool = import_path("r0_frozen_training_tool", TRAINING_TOOL_PATH)
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
    checkpoint_sha_before = sha256_file(CHECKPOINT_PATH)
    params_sha_before = training_tool.tree_sha(state.params)
    backbone_sha_before = training_tool.tree_sha(state.params, training_tool.is_backbone)

    @jax.jit
    def infer(params, observations):
        return state.apply_fn({"params": params}, observations, train=False)

    logits = []
    for start in range(0, len(camera1), INFERENCE_BATCH_SIZE):
        stop = min(start + INFERENCE_BATCH_SIZE, len(camera1))
        observations = {
            CLASSIFIER_CAMERA_KEYS[0]: jnp.asarray(np.asarray(camera1[start:stop])[:, None]),
            CLASSIFIER_CAMERA_KEYS[1]: jnp.asarray(np.asarray(camera2[start:stop])[:, None]),
        }
        batch_logits = infer(state.params, observations)
        logits.append(np.asarray(jax.block_until_ready(batch_logits), dtype=np.float32).reshape(-1))
    logits_array = np.concatenate(logits)
    probabilities = np.asarray(1.0 / (1.0 + np.exp(-logits_array.astype(np.float64))), dtype=np.float64)
    require(len(logits_array) == manifest["validation_frame_count"], "inference frame count mismatch")
    require(np.all(np.isfinite(logits_array)) and np.all(np.isfinite(probabilities)), "non-finite classifier output")
    require(np.all((probabilities >= 0.0) & (probabilities <= 1.0)), "probability outside [0,1]")
    params_sha_after = training_tool.tree_sha(state.params)
    backbone_sha_after = training_tool.tree_sha(state.params, training_tool.is_backbone)
    checkpoint_sha_after = sha256_file(CHECKPOINT_PATH)
    require(params_sha_before == params_sha_after, "classifier parameters changed during inference")
    require(backbone_sha_before == backbone_sha_after, "backbone changed during inference")
    require(checkpoint_sha_before == checkpoint_sha_after == BEST_CHECKPOINT_SHA256, "checkpoint changed during inference")
    require(int(state.step) == 150, "TrainState step changed during inference")

    with tempfile.NamedTemporaryFile("wb", dir=work_dir, suffix=".npz", delete=False) as stream:
        np.savez(stream, logits=logits_array, probabilities=probabilities)
        temporary = Path(stream.name)
    temporary.replace(prediction_path)
    evidence = {
        "schema_version": "forcesmolvla_r0_validation_inference_evidence.v1",
        "artifact_status": "PASS_FROZEN_EVAL_MODE_VALIDATION_INFERENCE",
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
            "random_augmentation": False,
            "optimizer_created_by_restored_TrainState_but_not_called": True,
            "optimizer_updates": 0,
            "train_state_step_before": 150,
            "train_state_step_after": int(state.step),
            "validation_frames_inferred": len(logits_array),
            "classifier_inference_failures": 0,
            "invalid_probability_count": 0,
            "test_frames_inferred": 0,
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
            "validation_episode_count": 5,
            "validation_frame_count": len(logits_array),
            "test_parquet_files_opened": [],
            "test_image_rows_loaded": 0,
            "test_images_decoded": 0,
            "test_frames_inferred": 0,
        },
        "bindings": {
            "best_checkpoint": binding(CHECKPOINT_PATH),
            "training_report": binding(TRAINING_REPORT_PATH),
            "cache_manifest_sha256": sha256_file(work_dir / "cache_manifest.json"),
            "predictions": {
                "path": str(prediction_path),
                "file_size": prediction_path.stat().st_size,
                "sha256": sha256_file(prediction_path),
            },
        },
    }
    atomic_json(evidence_path, evidence)
    print(json.dumps({"phase": "validation_gpu_inference", "status": "pass", "frames": len(logits_array), "test_inference": 0}), flush=True)


def causal_trigger(
    frames: Sequence[int], probabilities: Sequence[float], validity: Sequence[bool], tau: float, required: int
) -> int | None:
    require(0.0 < tau < 1.0 and required >= 1, "invalid detector candidate")
    require(len(frames) == len(probabilities) == len(validity), "detector input length mismatch")
    last_frame: int | None = None
    streak = 0
    trigger: int | None = None
    for frame, probability, valid in zip(frames, probabilities, validity):
        if not valid or frame < 0 or not np.isfinite(probability) or not 0.0 <= probability <= 1.0:
            last_frame = None
            streak = 0
            continue
        if last_frame is None or frame != last_frame + 1:
            streak = 0
        last_frame = int(frame)
        if trigger is not None:  # latched for the rest of this episode
            continue
        streak = streak + 1 if probability >= tau else 0
        if streak >= required:
            trigger = int(frame)  # causal confirmation now; never backfilled
    return trigger


def longest_true_run(values: np.ndarray) -> int:
    longest = 0
    current = 0
    for value in values:
        current = current + 1 if bool(value) else 0
        longest = max(longest, current)
    return longest


def detector_self_check() -> None:
    assert causal_trigger([0, 1, 2], [0.9, 0.9, 0.9], [True] * 3, 0.8, 3) == 2
    assert causal_trigger([0, 2, 3], [0.9, 0.9, 0.9], [True] * 3, 0.8, 2) == 3
    assert causal_trigger([0, 1, 2, 3], [0.9] * 4, [True, False, True, True], 0.8, 2) == 3
    assert causal_trigger([0, 1, 2, 3], [0.9, 0.9, 0.0, 0.0], [True] * 4, 0.8, 2) == 1
    assert causal_trigger([0], [float("nan")], [True], 0.8, 1) is None


def evaluate_candidate(episodes: list[dict[str, Any]], tau: float, required: int) -> dict[str, Any]:
    episode_results = []
    all_codes = []
    all_positive = []
    longest_pre = 0
    post_longest_by_episode = []
    delays = []
    early = 0
    missed = 0
    for episode in episodes:
        frames = np.asarray(episode["frame_indices"], dtype=np.int32)
        probabilities = np.asarray(episode["probabilities"], dtype=np.float64)
        validity = np.asarray(episode["valid"], dtype=np.bool_)
        codes = np.asarray(episode["class_codes"], dtype=np.uint8)
        completion = episode["first_confident_complete_frame"]
        threshold_positive = validity & (probabilities >= tau)
        trigger = causal_trigger(frames, probabilities, validity, tau, required)
        delay_frames = None if trigger is None else trigger - completion
        delay_ms = None if delay_frames is None else delay_frames * 1000.0 / FPS
        pre_run = longest_true_run(threshold_positive[frames < completion])
        post_run = longest_true_run(threshold_positive[frames >= completion])
        longest_pre = max(longest_pre, pre_run)
        post_longest_by_episode.append(post_run)
        if trigger is None:
            missed += 1
        else:
            delays.append(delay_frames)
            early += int(trigger < completion)
        episode_results.append({
            "episode_id": episode["episode_id"],
            "first_confident_complete_frame": completion,
            "trigger_frame": trigger,
            "detection_delay_frames": delay_frames,
            "detection_delay_ms": delay_ms,
            "longest_pre_completion_consecutive_positive_length": pre_run,
            "longest_post_completion_consecutive_positive_length": post_run,
        })
        metric_mask = validity & (codes != CLASS_CODE["ambiguous"])
        all_codes.append(codes[metric_mask])
        all_positive.append(threshold_positive[metric_mask])
    codes = np.concatenate(all_codes)
    positives = np.concatenate(all_positive)

    def rate(class_name: str) -> float:
        mask = codes == CLASS_CODE[class_name]
        require(np.any(mask), f"empty validation class: {class_name}")
        return float(np.mean(positives[mask]))

    detected = len(episodes) - missed
    median_frames = None if not delays else float(np.median(np.asarray(delays, dtype=np.float64)))
    max_frames = None if not delays else int(max(delays))
    return {
        "tau": tau,
        "M": required,
        "early_trigger_episode_count": early,
        "missed_success_episode_count": missed,
        "detected_success_episode_count": detected,
        "episodes": episode_results,
        "median_detection_delay_frames": median_frames,
        "max_detection_delay_frames": max_frames,
        "median_detection_delay_ms": None if median_frames is None else median_frames * 1000.0 / FPS,
        "max_detection_delay_ms": None if max_frames is None else max_frames * 1000.0 / FPS,
        "ordinary_negative_frame_FPR": rate("ordinary_negative"),
        "hard_negative_frame_FPR": rate("hard_negative"),
        "completion_positive_frame_recall": rate("positive"),
        "longest_pre_completion_consecutive_positive_length": longest_pre,
        "shortest_post_completion_consecutive_positive_length": int(min(post_longest_by_episode)),
        "feasible": early == 0 and missed == 0,
    }


def pareto_flags(feasible: list[dict[str, Any]]) -> set[tuple[float, int]]:
    objectives = lambda candidate: (
        candidate["max_detection_delay_frames"],
        candidate["median_detection_delay_frames"],
        candidate["hard_negative_frame_FPR"],
    )
    result = set()
    for candidate in feasible:
        current = objectives(candidate)
        dominated = any(
            all(left <= right for left, right in zip(objectives(other), current))
            and any(left < right for left, right in zip(objectives(other), current))
            for other in feasible
            if other is not candidate
        )
        if not dominated:
            result.add((candidate["tau"], candidate["M"]))
    return result


def make_markdown(
    artifact: Mapping[str, Any], candidate: Mapping[str, Any], calibration_sha: str, candidate_sha: str
) -> str:
    selected = artifact["proposed_candidate"]
    lines = [
        "# R0 validation-only Reward Detector calibration",
        "",
        f"- Calibration artifact SHA256: `{calibration_sha}`",
        f"- Candidate config SHA256: `{candidate_sha}`",
        f"- Frozen classifier checkpoint SHA256: `{BEST_CHECKPOINT_SHA256}`",
        f"- Frozen training report SHA256: `{TRAINING_REPORT_SHA256}`",
        "- Scope: five validation episodes only; test image load/decode/inference = 0.",
        "- Candidate status: proposed and explicitly not approved.",
        "",
        "## Result",
        "",
    ]
    if selected is None:
        lines += ["No candidate satisfied zero early triggers and zero missed successes.", ""]
    else:
        lines += [
            f"Preferred candidate: `tau={selected['tau']}`, `M={selected['M']}`. This is not an approved DetectorSpec.",
            "",
            f"- max delay: {selected['max_detection_delay_frames']} frames / {selected['max_detection_delay_ms']:.3f} ms",
            f"- median delay: {selected['median_detection_delay_frames']} frames / {selected['median_detection_delay_ms']:.3f} ms",
            f"- ordinary-negative frame FPR: {selected['ordinary_negative_frame_FPR']:.9f}",
            f"- hard-negative frame FPR: {selected['hard_negative_frame_FPR']:.9f}",
            f"- completion-positive frame recall: {selected['completion_positive_frame_recall']:.9f}",
            "",
        ]
    lines += [
        "Selection followed the frozen lexicographic rule: minimum maximum delay, then median delay, hard-negative FPR, higher tau, then larger M.",
        "Frame FPR/recall are threshold-level metrics before the consecutive-frame latch. `shortest_post_completion` is the minimum, across episodes, of each episode's longest post-completion positive run.",
        "",
        "## Validation episode triggers",
        "",
        "| episode | frames | first confident complete | trigger | delay frames | delay ms |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    trigger_lookup = {}
    if selected is not None:
        for episode in selected["episodes"]:
            trigger_lookup[episode["episode_id"]] = episode["trigger_frame"]
            lines.append(
                f"| {episode['episode_id']} | {next(x['frame_count'] for x in artifact['validation_probability_timelines'] if x['episode_id'] == episode['episode_id'])} "
                f"| {episode['first_confident_complete_frame']} | {episode['trigger_frame']} | {episode['detection_delay_frames']} | {episode['detection_delay_ms']:.3f} |"
            )
    lines += ["", "## All feasible candidates and Pareto status", "", "| tau | M | max delay f | median delay f | hard FPR | ordinary FPR | positive recall | pre max run | post min run | Pareto |", "|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|"]
    for row in artifact["feasible_candidate_table"]:
        lines.append(
            f"| {row['tau']} | {row['M']} | {row['max_detection_delay_frames']} | {row['median_detection_delay_frames']} "
            f"| {row['hard_negative_frame_FPR']:.9f} | {row['ordinary_negative_frame_FPR']:.9f} "
            f"| {row['completion_positive_frame_recall']:.9f} | {row['longest_pre_completion_consecutive_positive_length']} "
            f"| {row['shortest_post_completion_consecutive_positive_length']} | {'yes' if row['pareto_optimal'] else 'no'} |"
        )
    lines += ["", "## Complete 30 Hz probability timelines", "", "Each block is in original frame order. `manual_phase=completion_positive` begins at the human `first_confident_complete_frame`; `preferred_trigger=yes` marks the causal current-frame trigger and is never backfilled.", ""]
    for episode in artifact["validation_probability_timelines"]:
        trigger = trigger_lookup.get(episode["episode_id"])
        completion = episode["first_confident_complete_frame"]
        lines += [
            "<details>",
            f"<summary>{episode['episode_id']}: {episode['frame_count']} frames, completion={completion}, trigger={trigger}</summary>",
            "",
            "```csv",
            "frame_index,probability,manual_phase,manual_frame_class,preferred_trigger",
        ]
        for frame, probability, frame_class in zip(
            episode["frame_indices"], episode["probabilities"], episode["frame_classes"]
        ):
            phase = "completion_positive" if frame >= completion else "pre_completion"
            lines.append(f"{frame},{probability:.10f},{phase},{frame_class},{'yes' if frame == trigger else 'no'}")
        lines += ["```", "", "</details>", ""]
    lines += [
        "## Boundary and access audit",
        "",
        "- Classifier checkpoint and parameter SHA were unchanged; TrainState remained at step 150 with zero optimizer updates.",
        "- Eval mode only: no dropout RNG and no random augmentation.",
        "- Counter reset logic covers episode boundaries, frame gaps, invalid/missing input, non-finite inference output, and failed validity checks.",
        "- Test Parquet/image load, decode, inference, selection, and tuning counts are all zero.",
        "- No reward/terminal, G1, G2, Twin-Q, Cal-QL, or Actor artifact was created.",
        "",
        "## Status",
        "",
        "```text",
        "CLASSIFIER_CHECKPOINT_FROZEN = yes",
        "VALIDATION_CALIBRATION_COMPLETE = yes",
        f"DETECTOR_CANDIDATE_CREATED = {'yes' if selected is not None else 'no'}",
        "DETECTOR_THRESHOLD_APPROVED = no",
        "TEST_EVALUATED = no",
        "TASK2_REWARD_TERMINAL_CREATED = no",
        "REWARD_TRANSITION_CREATED = no",
        "TWIN_Q_CREATED = no",
        "NEXT_ALLOWED_ACTION = request_detector_candidate_approval",
        "```",
        "",
    ]
    return "\n".join(lines)


def finalize(work_dir: Path) -> None:
    detector_self_check()
    frozen = verify_frozen_inputs()
    cache = verify_work_cache(work_dir)
    evidence_path = work_dir / "inference_evidence.json"
    prediction_path = work_dir / "validation_predictions.npz"
    require(evidence_path.is_file() and prediction_path.is_file(), "validation inference evidence missing")
    evidence = load_json(evidence_path)
    require(evidence["artifact_status"] == "PASS_FROZEN_EVAL_MODE_VALIDATION_INFERENCE", "validation inference did not pass")
    require(evidence["execution"]["optimizer_updates"] == 0, "classifier was updated")
    require(evidence["access_audit"]["test_frames_inferred"] == 0, "test inference detected")
    require(evidence["bindings"]["predictions"]["sha256"] == sha256_file(prediction_path), "prediction SHA mismatch")
    for path in (CANDIDATE_PATH, CALIBRATION_PATH, REPORT_PATH):
        require(not path.exists(), f"append-only target already exists: {path}")

    arrays = np.load(prediction_path, allow_pickle=False)
    logits = arrays["logits"]
    probabilities = arrays["probabilities"]
    frame_indices = np.load(work_dir / "frame_indices.npy", allow_pickle=False)
    class_codes = np.load(work_dir / "class_codes.npy", allow_pickle=False)
    valid = np.load(work_dir / "valid.npy", allow_pickle=False)
    require(len(logits) == len(probabilities) == len(frame_indices) == len(class_codes) == len(valid) == 3775, "calibration array length mismatch")

    episodes = []
    timelines = []
    for episode in cache["episodes"]:
        start, stop = episode["cache_range_half_open"]
        frames = frame_indices[start:stop]
        probs = probabilities[start:stop]
        episode_logits = logits[start:stop]
        codes = class_codes[start:stop]
        episode_valid = valid[start:stop]
        require(np.array_equal(frames, np.arange(episode["frame_count"])), "validation order/frame gap")
        require(np.all(episode_valid), "invalid validation input")
        values = {
            "episode_id": episode["episode_id"],
            "frame_count": episode["frame_count"],
            "first_confident_complete_frame": episode["first_confident_complete_frame"],
            "frame_indices": frames.astype(int).tolist(),
            "probabilities": probs.astype(float).tolist(),
            "logits": episode_logits.astype(float).tolist(),
            "class_codes": codes.astype(int).tolist(),
            "frame_classes": [CLASS_NAMES[int(code)] for code in codes],
            "valid": episode_valid.astype(bool).tolist(),
        }
        episodes.append(values)
        timelines.append({
            "episode_id": values["episode_id"],
            "frame_count": values["frame_count"],
            "first_confident_complete_frame": values["first_confident_complete_frame"],
            "frame_indices": values["frame_indices"],
            "probabilities": values["probabilities"],
            "logits": values["logits"],
            "frame_classes": values["frame_classes"],
            "invalid_frame_count": 0,
            "probability_summary": {
                "minimum": float(np.min(probs)),
                "maximum": float(np.max(probs)),
                "mean": float(np.mean(probs)),
                "at_first_confident_complete_frame": float(probs[episode["first_confident_complete_frame"]]),
            },
        })

    candidates = [
        evaluate_candidate(episodes, float(tau), required)
        for tau in TAU_DECIMALS
        for required in M_VALUES
    ]
    require(len(candidates) == 520, "fixed candidate grid size mismatch")
    feasible = [candidate for candidate in candidates if candidate["feasible"]]
    pareto = pareto_flags(feasible)
    for candidate in feasible:
        candidate["pareto_optimal"] = (candidate["tau"], candidate["M"]) in pareto
    feasible.sort(key=lambda candidate: (
        candidate["max_detection_delay_frames"],
        candidate["median_detection_delay_frames"],
        candidate["hard_negative_frame_FPR"],
        -candidate["tau"],
        -candidate["M"],
    ))
    selected = feasible[0] if feasible else None
    selected_summary = None if selected is None else {key: value for key, value in selected.items() if key != "pareto_optimal"}
    if selected_summary is not None:
        selected_summary["pareto_optimal"] = selected["pareto_optimal"]

    timeline_sha = canonical_sha([
        {"episode_id": episode["episode_id"], "frames": episode["frame_indices"], "probabilities": episode["probabilities"]}
        for episode in episodes
    ])
    candidate_config = {
        "schema_version": "forcesmolvla_r0_reward_detector_candidate.v1",
        "artifact_status": "PROPOSED_UNAPPROVED_VALIDATION_ONLY_CANDIDATE" if selected else "NO_FEASIBLE_VALIDATION_CANDIDATE",
        "approval_status": "not_approved",
        "created_at": utc_now(),
        "candidate": None if selected is None else {"probability_threshold": selected["tau"], "consecutive_positive_frames": selected["M"]},
        "causal_semantics": {
            "positive_t": "sigmoid(logit_t) >= probability_threshold",
            "trigger": "current frame at which the consecutive-positive streak first reaches M",
            "backfill_to_streak_start": False,
            "latched_after_trigger_within_episode": True,
            "reset_on": [
                "episode_start_or_end", "non_consecutive_frame_index", "invalid_or_missing_image",
                "classifier_inference_failure", "any_input_validity_failure",
            ],
            "input_rate_hz": FPS,
        },
        "selection": {
            "feasibility": {"early_trigger_episode_count": 0, "missed_success_episode_count": 0},
            "lexicographic_order": [
                "minimum_max_detection_delay", "minimum_median_detection_delay",
                "minimum_hard_negative_frame_FPR", "higher_tau", "larger_M",
            ],
            "selected_metrics": selected_summary,
        },
        "validation_probability_timeline_sha256": timeline_sha,
        "bindings": {
            "best_checkpoint": binding(CHECKPOINT_PATH),
            "training_report": binding(TRAINING_REPORT_PATH),
            "reviewed_labels": binding(REVIEWED_PATH),
            "inventory": binding(INVENTORY_PATH),
            "calibration_source": binding(Path(__file__).resolve()),
        },
        "prohibitions": {
            "detector_threshold_approved": False,
            "test_evaluated": False,
            "reward_or_terminal_created": False,
            "REWARD_TRANSITION_created": False,
            "TWIN_Q_created": False,
        },
    }

    staging = Path(tempfile.mkdtemp(prefix=".r0-validation-calibration-", dir=ROOT))
    try:
        staged_candidate = staging / CANDIDATE_PATH.name
        atomic_json(staged_candidate, candidate_config)
        candidate_binding = {
            "path": relative(CANDIDATE_PATH),
            "file_size": staged_candidate.stat().st_size,
            "sha256": sha256_file(staged_candidate),
        }
        feasible_table = [
            {key: value for key, value in candidate.items() if key != "episodes"}
            for candidate in feasible
        ]
        artifact = {
            "schema_version": "forcesmolvla_r0_validation_detector_calibration.v1",
            "artifact_status": "PASS_VALIDATION_CALIBRATION_CANDIDATE_PROPOSED_NOT_APPROVED" if selected else "VALIDATION_CALIBRATION_NO_FEASIBLE_CANDIDATE",
            "created_at": utc_now(),
            "scope": "validation_only_reward_detector_calibration",
            "frozen_bindings": {
                "best_checkpoint": binding(CHECKPOINT_PATH),
                "training_report": binding(TRAINING_REPORT_PATH),
                "reviewed_labels": binding(REVIEWED_PATH),
                "inventory": binding(INVENTORY_PATH),
                "readiness": binding(READINESS_PATH),
                "split_manifest": binding(SPLIT_PATH),
                "safe_resnet10_npz": binding(SAFE_ASSET_PATH),
                "calibration_source": binding(Path(__file__).resolve()),
                "candidate_config": candidate_binding,
            },
            "classifier_freeze_and_eval_evidence": evidence,
            "validation_access_audit": {
                **cache["access_audit"],
                "validation_frames_inferred": 3775,
                "validation_episode_order": cache["validation_episode_order"],
                "validation_random_augmentation": False,
                "validation_dropout": False,
                "test_checkpoint_selection_participation": False,
                "test_tuning_participation": False,
                "test_frames_inferred": 0,
            },
            "search_space": {
                "tau": [float(value) for value in TAU_DECIMALS],
                "M": list(M_VALUES),
                "candidate_count": len(candidates),
                "expanded_or_modified_from_user_spec": False,
            },
            "metric_definitions": {
                "early_trigger": "trigger_frame < first_confident_complete_frame",
                "detection_delay_frames": "trigger_frame - first_confident_complete_frame",
                "detection_delay_ms": "detection_delay_frames / 30 * 1000",
                "frame_FPR_and_recall": "threshold-level p_t>=tau metrics before the consecutive-frame latch; ambiguous excluded",
                "longest_pre_completion_consecutive_positive_length": "maximum threshold-positive run over frames before completion, then maximum across episodes",
                "shortest_post_completion_consecutive_positive_length": "per episode longest threshold-positive run at/after completion, then minimum across episodes",
                "trigger_not_backfilled": True,
            },
            "validation_probability_timeline_sha256": timeline_sha,
            "validation_probability_timelines": timelines,
            "all_candidates": candidates,
            "feasible_candidate_count": len(feasible),
            "pareto_candidate_count": len(pareto),
            "feasible_candidate_table": feasible_table,
            "proposed_candidate": selected_summary,
            "proposal_only_not_approved": True,
            "forbidden_outputs_created": [],
            "terminal_status": {
                "CLASSIFIER_CHECKPOINT_FROZEN": "yes",
                "VALIDATION_CALIBRATION_COMPLETE": "yes",
                "DETECTOR_CANDIDATE_CREATED": "yes" if selected else "no",
                "DETECTOR_THRESHOLD_APPROVED": "no",
                "TEST_EVALUATED": "no",
                "TASK2_REWARD_TERMINAL_CREATED": "no",
                "REWARD_TRANSITION_CREATED": "no",
                "TWIN_Q_CREATED": "no",
                "NEXT_ALLOWED_ACTION": "request_detector_candidate_approval" if selected else "report_validation_constraint_conflict",
            },
        }
        staged_calibration = staging / CALIBRATION_PATH.name
        atomic_json(staged_calibration, artifact)
        calibration_sha = sha256_file(staged_calibration)
        report = make_markdown(
            artifact, candidate_config, calibration_sha, candidate_binding["sha256"]
        )
        staged_report = staging / REPORT_PATH.name
        atomic_text(staged_report, report)
        CANDIDATE_PATH.parent.mkdir(parents=True, exist_ok=True)
        CALIBRATION_PATH.parent.mkdir(parents=True, exist_ok=True)
        REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
        staged_candidate.replace(CANDIDATE_PATH)
        staged_calibration.replace(CALIBRATION_PATH)
        staged_report.replace(REPORT_PATH)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    print(json.dumps({
        "phase": "fixed_grid_calibration", "status": "pass", "candidate_count": 520,
        "feasible_candidate_count": len(feasible),
        "proposed": None if selected is None else {"tau": selected["tau"], "M": selected["M"]},
        "test_inference": 0,
    }, sort_keys=True), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare-cache")
    prepare.add_argument("--work-dir", type=Path, required=True)
    infer = subparsers.add_parser("infer")
    infer.add_argument("--work-dir", type=Path, required=True)
    finish = subparsers.add_parser("finalize")
    finish.add_argument("--work-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "prepare-cache":
        prepare_cache(args.work_dir.resolve())
    elif args.command == "infer":
        run_inference(args.work_dir.resolve())
    else:
        finalize(args.work_dir.resolve())


if __name__ == "__main__":
    main()
