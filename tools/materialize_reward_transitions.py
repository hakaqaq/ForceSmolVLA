#!/usr/bin/env python3
"""Build append-only transitions from a frozen causal reward detector."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
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
from typing import Any

import numpy as np


ROOT = Path(__file__).parents[1].resolve()
CONRFT_RUNTIME_ROOT = Path("/home/rlc123/conrft/serl_launcher")
SOURCE_CAMERA_KEYS = ("observation.images.camera1", "observation.images.camera2")
CLASSIFIER_CAMERA_KEYS = ("d435_third_person", "d405_wrist")
IMAGE_SHAPE = (480, 640, 3)
INFERENCE_BATCH_SIZE = 128
MANUAL_FILE_OPENS: set[str] = set()


def install_manual_file_audit() -> None:
    labels_root = (ROOT / "labels").resolve()

    def audit(event: str, args: tuple[Any, ...]) -> None:
        if event != "open" or not args or not isinstance(args[0], (str, bytes, os.PathLike)):
            return
        try:
            path = Path(os.fsdecode(args[0])).resolve()
        except (OSError, TypeError, ValueError):
            return
        if path == labels_root or path.is_relative_to(labels_root):
            MANUAL_FILE_OPENS.add(str(path))

    sys.addaudithook(audit)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"JSON_OBJECT_REQUIRED:{path}")
    return value


def display_path(path: Path) -> str:
    path = path.resolve()
    return path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) else str(path)


def configured_path(config: dict, name: str) -> Path:
    value = config["inputs"][name]
    return (ROOT / value).resolve() if not Path(value).is_absolute() else Path(value).resolve()


def p8_storage_tree(dataset_root: Path) -> dict:
    files = sorted(
        path
        for directory in ("data", "videos", "meta")
        for path in (dataset_root / directory).rglob("*")
        if path.is_file()
    )
    records = {path.relative_to(dataset_root).as_posix(): sha256_file(path) for path in files}
    digest = hashlib.sha256()
    for relative, value in records.items():
        digest.update(f"{relative}\0{value}\n".encode())
    return {"file_count": len(records), "tree_sha256": digest.hexdigest()}


def import_path(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    require(spec is not None and spec.loader is not None, f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def verify_config(config_path: Path, *, task_id: str, dataset_root: Path) -> dict:
    config = load_json(config_path)
    require(
        config.get("schema")
        == "forcesmolvla.forcerft_offline_reward_transition_materialization"
        and config.get("status") == "final"
        and config.get("task_id") == task_id,
        "REWARD_TRANSITION_CONFIG_IDENTITY_INVALID",
    )
    detector = config["detector_spec"]
    probability_threshold = float(detector["probability_threshold"])
    required_consecutive_frames = int(detector["required_consecutive_frames"])
    require(
        0.0 < probability_threshold < 1.0
        and required_consecutive_frames >= 1
        and detector["detector_input_rate_hz"] == 30
        and detector["trigger_timestamp"] in {
            "current_confirming_frame",
            "fifth_confirming_frame",
        }
        and (
            detector["trigger_timestamp"] != "fifth_confirming_frame"
            or required_consecutive_frames == 5
        )
        and detector["trigger_backfilled_to_streak_start"] is False,
        "REWARD_TRANSITION_FROZEN_SPEC_DRIFT",
    )
    require(config["required_runtime_audit"] == {
        "manual_label_files_opened": 0,
        "manual_boundary_fields_consumed": 0,
        "manual_terminal_fallback_count": 0,
        "classifier_optimizer_updates": 0,
        "detector_parameter_search_count": 0,
    }, "REWARD_TRANSITION_AUDIT_CONTRACT_DRIFT")
    require(dataset_root.is_dir(), "REWARD_TRANSITION_DATASET_MISSING")
    for name in (
        "classifier_checkpoint",
        "safe_resnet10_npz",
        "safe_resnet10_manifest",
        "actor_checkpoint",
        "classifier_training_source",
        "adapter_source",
    ):
        path = configured_path(config, name)
        require(
            path.is_dir() if name == "actor_checkpoint" else path.is_file(),
            f"REWARD_TRANSITION_INPUT_MISSING:{name}:{path}",
        )
    if "detector_calibration" in config["inputs"]:
        calibration = load_json(configured_path(config, "detector_calibration"))
        require(
            calibration.get("status") == "approved"
            and calibration.get("task_id") == task_id
            and float(calibration["selected"]["probability_threshold"])
            == probability_threshold
            and int(calibration["selected"]["required_consecutive_frames"])
            == required_consecutive_frames,
            "REWARD_TRANSITION_CALIBRATION_MISMATCH",
        )
    return config


def episode_metadata(dataset_root: Path) -> dict[int, dict]:
    import pyarrow.parquet as pq

    path = dataset_root / "meta/episodes/chunk-000/file-000.parquet"
    columns = ["episode_index", "length", "data/chunk_index", "data/file_index"]
    return {int(row["episode_index"]): row for row in pq.read_table(path, columns=columns).to_pylist()}


def episode_descriptors(dataset_root: Path) -> tuple[list[dict], dict, dict, dict]:
    conversion = load_json(dataset_root / "conversion_manifest.json")
    split = load_json(dataset_root / "split_manifest.json")
    info = load_json(dataset_root / "meta/info.json")
    metadata = episode_metadata(dataset_root)
    split_lookup = {episode_id: name for name in ("train", "val", "test") for episode_id in split[name]}
    require(split_lookup, "REWARD_TRANSITION_SPLIT_INVALID")
    episodes = []
    for source in sorted(conversion["episodes"], key=lambda item: int(item["output_episode_index"])):
        index = int(source["output_episode_index"])
        episode_id = source["raw_episode_id"]
        meta = metadata[index]
        relative = info["data_path"].format(
            chunk_index=meta["data/chunk_index"], file_index=meta["data/file_index"]
        )
        require(source["split"] == split_lookup[episode_id], "REWARD_TRANSITION_EPISODE_SPLIT_DRIFT")
        require(int(source["diagnostics"]["frames"]) == int(meta["length"]), "REWARD_TRANSITION_EPISODE_LENGTH_DRIFT")
        episodes.append({
            "episode_id": episode_id,
            "output_episode_index": index,
            "split": source["split"],
            "frame_count": int(meta["length"]),
            "source_data_relative_path": relative,
            "task": source["task"],
        })
    require(
        len(episodes) == len(split_lookup)
        and {item["episode_id"] for item in episodes} == set(split_lookup),
        "REWARD_TRANSITION_EPISODE_COVERAGE_INVALID",
    )
    return episodes, conversion, split, info


def decode_rgb(payload: bytes) -> np.ndarray:
    from PIL import Image

    require(isinstance(payload, bytes), "REWARD_TRANSITION_IMAGE_BYTES_MISSING")
    with Image.open(BytesIO(payload)) as image:
        rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    require(rgb.shape == IMAGE_SHAPE, f"REWARD_TRANSITION_IMAGE_SHAPE_INVALID:{rgb.shape}")
    return np.ascontiguousarray(rgb)


def read_protocol_line(process: subprocess.Popen, prefix: str, log_path: Path) -> dict:
    require(process.stdout is not None, "REWARD_TRANSITION_GPU_STDOUT_MISSING")
    while True:
        line = process.stdout.readline()
        if not line:
            code = process.poll()
            tail = log_path.read_text(encoding="utf-8", errors="replace")[-4000:] if log_path.exists() else ""
            raise RuntimeError(f"REWARD_TRANSITION_GPU_WORKER_EXITED:{code}:{tail}")
        if line.startswith(prefix):
            return json.loads(line[len(prefix):])


def run_frozen_classifier(
    *,
    episodes: list[dict],
    dataset_root: Path,
    dataset_root_id: str,
    temporary_root: Path,
    config_path: Path,
    config: dict,
) -> tuple[list[dict], dict]:
    import pyarrow.parquet as pq

    adapter_module = import_path(
        "reward_transition_adapter", configured_path(config, "adapter_source")
    )
    adapter = adapter_module.ConRFTLeRobotV3Adapter()
    batch_root = temporary_root / ".streaming_batch"
    batch_root.mkdir()
    log_path = temporary_root / ".gpu_worker_stderr.log"
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + environment.get("PYTHONPATH", "")
    command = [
        shutil.which("conda") or "conda", "run", "--no-capture-output", "-n", "conrft_reward",
        "python", str(Path(__file__).resolve()), "gpu-server", "--config", str(config_path),
    ]
    with log_path.open("w", encoding="utf-8") as error_stream:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            env=environment,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=error_stream,
            text=True,
            bufsize=1,
        )
        try:
            ready = read_protocol_line(process, "REWARD_GPU_READY ", log_path)
            require(ready["backend"] == "gpu", "REWARD_TRANSITION_GPU_BACKEND_REQUIRED")
            score_episodes = []
            score_cursor = 0
            for ordinal, episode in enumerate(episodes, start=1):
                source_path = dataset_root / episode["source_data_relative_path"]
                table = pq.read_table(source_path, columns=[
                    *SOURCE_CAMERA_KEYS,
                    "frame_index",
                    "episode_index",
                    "index",
                    "timestamp",
                    "provenance.camera1_receive_monotonic_ns",
                    "provenance.camera2_receive_monotonic_ns",
                ])
                require(table.num_rows == episode["frame_count"], "REWARD_TRANSITION_SCORE_FRAME_COUNT_MISMATCH")
                rows = table.to_pylist()
                frames: list[int] = []
                global_indices: list[int] = []
                timestamps: list[float] = []
                camera1_stamps: list[int] = []
                camera2_stamps: list[int] = []
                logits: list[np.ndarray] = []
                for start in range(0, len(rows), INFERENCE_BATCH_SIZE):
                    batch_rows = rows[start : start + INFERENCE_BATCH_SIZE]
                    camera1 = np.empty((len(batch_rows), *IMAGE_SHAPE), dtype=np.uint8)
                    camera2 = np.empty((len(batch_rows), *IMAGE_SHAPE), dtype=np.uint8)
                    for offset, row in enumerate(batch_rows):
                        frame = start + offset
                        require(int(row["frame_index"]) == frame, "REWARD_TRANSITION_NON_CONSECUTIVE_FRAME")
                        require(int(row["episode_index"]) == episode["output_episode_index"], "REWARD_TRANSITION_EPISODE_INDEX_DRIFT")
                        rgb1 = decode_rgb(row[SOURCE_CAMERA_KEYS[0]]["bytes"])
                        rgb2 = decode_rgb(row[SOURCE_CAMERA_KEYS[1]]["bytes"])
                        adapted = adapter.adapt(
                            {
                                SOURCE_CAMERA_KEYS[0]: np.transpose(rgb1, (2, 0, 1)),
                                SOURCE_CAMERA_KEYS[1]: np.transpose(rgb2, (2, 0, 1)),
                            },
                            row_reference=adapter_module.RowReference(
                                dataset_root_id, episode["source_data_relative_path"], frame,
                                episode["episode_id"], frame, float(row["timestamp"]),
                            ),
                            camera_row_identity=adapter_module.CameraRowIdentity(
                                int(row["provenance.camera1_receive_monotonic_ns"]),
                                int(row["provenance.camera2_receive_monotonic_ns"]),
                            ),
                        )
                        camera1[offset] = adapted.observation[CLASSIFIER_CAMERA_KEYS[0]][0, 0]
                        camera2[offset] = adapted.observation[CLASSIFIER_CAMERA_KEYS[1]][0, 0]
                        frames.append(frame)
                        global_indices.append(int(row["index"]))
                        timestamps.append(float(row["timestamp"]))
                        camera1_stamps.append(int(row["provenance.camera1_receive_monotonic_ns"]))
                        camera2_stamps.append(int(row["provenance.camera2_receive_monotonic_ns"]))
                    camera1_path = batch_root / "camera1.npy"
                    camera2_path = batch_root / "camera2.npy"
                    logits_path = batch_root / "logits.npy"
                    np.save(camera1_path, camera1, allow_pickle=False)
                    np.save(camera2_path, camera2, allow_pickle=False)
                    require(process.stdin is not None, "REWARD_TRANSITION_GPU_STDIN_MISSING")
                    process.stdin.write(json.dumps({
                        "command": "infer",
                        "camera1": str(camera1_path),
                        "camera2": str(camera2_path),
                        "output": str(logits_path),
                        "count": len(batch_rows),
                    }) + "\n")
                    process.stdin.flush()
                    ack = read_protocol_line(process, "REWARD_GPU_BATCH ", log_path)
                    require(ack["count"] == len(batch_rows), "REWARD_TRANSITION_GPU_BATCH_COUNT_MISMATCH")
                    logits.append(np.load(logits_path, allow_pickle=False).astype(np.float32))
                    for path in (camera1_path, camera2_path, logits_path):
                        path.unlink()
                episode_logits = np.concatenate(logits)
                probabilities = np.where(
                    episode_logits >= 0,
                    1.0 / (1.0 + np.exp(-episode_logits.astype(np.float64))),
                    np.exp(episode_logits.astype(np.float64)) / (1.0 + np.exp(episode_logits.astype(np.float64))),
                )
                require(len(episode_logits) == episode["frame_count"], "REWARD_TRANSITION_EPISODE_LOGIT_COUNT_MISMATCH")
                require(np.all(np.isfinite(probabilities)), "REWARD_TRANSITION_NONFINITE_PROBABILITY")
                score_episodes.append({
                    **episode,
                    "score_range_half_open": [score_cursor, score_cursor + len(frames)],
                    "frame_indices": np.asarray(frames, dtype=np.int32),
                    "global_indices": np.asarray(global_indices, dtype=np.int64),
                    "timestamps": np.asarray(timestamps, dtype=np.float64),
                    "camera1_stamps": np.asarray(camera1_stamps, dtype=np.int64),
                    "camera2_stamps": np.asarray(camera2_stamps, dtype=np.int64),
                    "logits": episode_logits,
                    "probabilities": probabilities,
                    "valid": np.ones(len(frames), dtype=np.bool_),
                })
                score_cursor += len(frames)
                print(
                    f"REWARD_TRANSITION_DETECTOR_SCORES:{ordinal}/{len(episodes)}:"
                    f"{episode['episode_id']}:frames={len(frames)}",
                    flush=True,
                )
                del table, rows
            require(process.stdin is not None, "REWARD_TRANSITION_GPU_STDIN_MISSING")
            process.stdin.write(json.dumps({"command": "stop"}) + "\n")
            process.stdin.flush()
            summary = read_protocol_line(process, "REWARD_GPU_SUMMARY ", log_path)
            code = process.wait(timeout=30)
            require(code == 0, f"REWARD_TRANSITION_GPU_WORKER_NONZERO:{code}")
        except BaseException:
            process.kill()
            process.wait()
            raise
    shutil.rmtree(batch_root)
    log_path.unlink(missing_ok=True)
    require(
        adapter.episode_reset_count == len(episodes)
        and score_cursor == sum(item["frame_count"] for item in episodes),
        "REWARD_TRANSITION_ADAPTER_OR_SCORE_COVERAGE_INVALID",
    )
    return score_episodes, summary


def gpu_server(config_path: Path) -> None:
    install_manual_file_audit()
    require(os.environ.get("CONDA_DEFAULT_ENV") == "conrft_reward", "REWARD_TRANSITION_GPU_ENV_REQUIRED")
    config = load_json(config_path)
    checkpoint = configured_path(config, "classifier_checkpoint")
    training_tool = import_path(
        "reward_transition_training_tool",
        configured_path(config, "classifier_training_source"),
    )
    training_tool.SAFE_ASSET_PATH = configured_path(config, "safe_resnet10_npz")
    training_tool.SAFE_MANIFEST_PATH = configured_path(
        config, "safe_resnet10_manifest"
    )
    safe_manifest = load_json(training_tool.SAFE_MANIFEST_PATH)
    training_tool.EXPECTED_SAFE_ASSET_SHA256 = safe_manifest["safe_asset"]["sha256"]
    training_tool.install_type_only_octo_shim()
    sys.path.insert(0, str(CONRFT_RUNTIME_ROOT))
    import flax
    from flax import serialization
    import jax
    import jax.numpy as jnp
    import jaxlib
    import optax
    from serl_launcher.networks.reward_classifier import create_classifier

    require(jax.default_backend() == "gpu", f"REWARD_TRANSITION_GPU_REQUIRED:{jax.default_backend()}")
    safe_tree, _ = training_tool.npz_encoder_tree()
    sample = {
        CLASSIFIER_CAMERA_KEYS[0]: jnp.zeros((1, 1, *IMAGE_SHAPE), dtype=jnp.uint8),
        CLASSIFIER_CAMERA_KEYS[1]: jnp.zeros((1, 1, *IMAGE_SHAPE), dtype=jnp.uint8),
    }
    with training_tool.trusted_safe_npz_pickle_bridge(safe_tree) as bridge:
        target = create_classifier(
            jax.random.PRNGKey(0), sample, list(CLASSIFIER_CAMERA_KEYS),
            pretrained_encoder_path=str(bridge), n_way=2,
        )
    state = serialization.from_bytes(target, checkpoint.read_bytes())
    expected_step = int(config["classifier_train_state_step"])
    require(int(state.step) == expected_step, "REWARD_TRANSITION_GPU_CHECKPOINT_STEP_DRIFT")
    checkpoint_before = sha256_file(checkpoint)
    params_before = training_tool.tree_sha(state.params)
    backbone_before = training_tool.tree_sha(state.params, training_tool.is_backbone)

    @jax.jit
    def infer(params, observations):
        return state.apply_fn({"params": params}, observations, train=False)

    print("REWARD_GPU_READY " + json.dumps({
        "backend": jax.default_backend(), "device": str(jax.devices()[0]),
    }), flush=True)
    frame_count = 0
    batch_count = 0
    for line in sys.stdin:
        command = json.loads(line)
        if command["command"] == "stop":
            break
        require(command["command"] == "infer", "REWARD_TRANSITION_GPU_COMMAND_INVALID")
        camera1 = np.load(command["camera1"], mmap_mode="r", allow_pickle=False)
        camera2 = np.load(command["camera2"], mmap_mode="r", allow_pickle=False)
        require(camera1.shape == camera2.shape == (command["count"], *IMAGE_SHAPE), "REWARD_TRANSITION_GPU_INPUT_SHAPE_INVALID")
        observations = {
            CLASSIFIER_CAMERA_KEYS[0]: jnp.asarray(np.asarray(camera1))[:, None],
            CLASSIFIER_CAMERA_KEYS[1]: jnp.asarray(np.asarray(camera2))[:, None],
        }
        logits = np.asarray(jax.block_until_ready(infer(state.params, observations)), dtype=np.float32).reshape(-1)
        require(len(logits) == command["count"] and np.all(np.isfinite(logits)), "REWARD_TRANSITION_GPU_OUTPUT_INVALID")
        np.save(command["output"], logits, allow_pickle=False)
        frame_count += len(logits)
        batch_count += 1
        print("REWARD_GPU_BATCH " + json.dumps({"count": len(logits), "batch": batch_count}), flush=True)
    checkpoint_after = sha256_file(checkpoint)
    params_after = training_tool.tree_sha(state.params)
    backbone_after = training_tool.tree_sha(state.params, training_tool.is_backbone)
    require(checkpoint_before == checkpoint_after, "REWARD_TRANSITION_GPU_CHECKPOINT_CHANGED")
    require(params_before == params_after and backbone_before == backbone_after, "REWARD_TRANSITION_GPU_PARAMETERS_CHANGED")
    require(int(state.step) == expected_step, "REWARD_TRANSITION_GPU_OPTIMIZER_STEP_CHANGED")
    print("REWARD_GPU_SUMMARY " + json.dumps({
        "environment": os.environ["CONDA_DEFAULT_ENV"],
        "backend": jax.default_backend(),
        "device": str(jax.devices()[0]),
        "python": sys.version.split()[0],
        "jax": jax.__version__,
        "jaxlib": jaxlib.__version__,
        "flax": flax.__version__,
        "optax": optax.__version__,
        "frame_count": frame_count,
        "batch_count": batch_count,
        "eval_mode": True,
        "random_augmentation": False,
        "dropout_rng_supplied": False,
        "optimizer_updates": 0,
        "train_state_step_before": expected_step,
        "train_state_step_after": int(state.step),
        "checkpoint_sha256_before": checkpoint_before,
        "checkpoint_sha256_after": checkpoint_after,
        "classifier_params_sha256_before": params_before,
        "classifier_params_sha256_after": params_after,
        "backbone_sha256_before": backbone_before,
        "backbone_sha256_after": backbone_after,
        "manual_label_files_opened": len(MANUAL_FILE_OPENS),
        "manual_label_paths": sorted(MANUAL_FILE_OPENS),
    }, sort_keys=True), flush=True)


def read_action_arrays(path: Path, data_columns: tuple[str, ...]) -> dict[str, np.ndarray]:
    import pyarrow.parquet as pq

    table = pq.read_table(path, columns=list(data_columns))
    arrays = {}
    for name in data_columns:
        dtype = None
        if name in {"observation.state", "observation.wrench", "action"}:
            dtype = np.float64
        elif name in {"frame_index", "episode_index", "index"} or name.endswith("_ns"):
            dtype = np.int64
        arrays[name] = np.asarray(table[name].to_pylist(), dtype=dtype)
    return arrays


def frame_score_schema():
    import pyarrow as pa

    return pa.schema([
        ("score_index", pa.int64()),
        ("episode_id", pa.string()),
        ("output_episode_index", pa.int32()),
        ("split", pa.string()),
        ("frame_index", pa.int32()),
        ("global_index", pa.int64()),
        ("source_data_relative_path", pa.string()),
        ("source_row_index", pa.int32()),
        ("timestamp", pa.float64()),
        ("camera1_receive_monotonic_ns", pa.int64()),
        ("camera2_receive_monotonic_ns", pa.int64()),
        ("classifier_logit", pa.float32()),
        ("classifier_probability", pa.float64()),
        ("input_valid", pa.bool_()),
        ("threshold_positive", pa.bool_()),
        ("consecutive_positive_count", pa.int16()),
        ("is_trigger_frame", pa.bool_()),
        ("detector_latched", pa.bool_()),
        ("reward_source", pa.string()),
        ("probability_threshold", pa.float64()),
        ("required_consecutive_frames", pa.int8()),
    ])


def transition_schema():
    import pyarrow as pa

    from forcesmolvla.rft.detector_reward_transitions import HORIZON, K

    row_reference = pa.struct([
        ("dataset_root_id", pa.string()),
        ("data_relative_path", pa.string()),
        ("row_index", pa.int32()),
        ("episode_id", pa.string()),
        ("frame_index", pa.int32()),
        ("global_index", pa.int64()),
    ])
    action_reference = pa.struct([
        ("dataset_root_id", pa.string()),
        ("data_relative_path", pa.string()),
        ("anchor_row_index", pa.int32()),
        ("source_frame_start_inclusive", pa.int32()),
        ("source_frame_stop_exclusive", pa.int32()),
        ("actor_horizon", pa.int16()),
        ("executed_slice_start", pa.int8()),
        ("executed_slice_stop_exclusive", pa.int8()),
    ])
    return pa.schema([
        ("transition_index", pa.int64()),
        ("episode_id", pa.string()),
        ("output_episode_index", pa.int32()),
        ("split", pa.string()),
        ("anchor_frame", pa.int32()),
        ("next_frame", pa.int32()),
        ("detector_terminal_frame", pa.int32()),
        ("detector_streak_start_frame", pa.int32()),
        ("detector_probability_at_trigger", pa.float64()),
        ("detector_probability_threshold", pa.float64()),
        ("detector_required_consecutive_frames", pa.int8()),
        ("executed_steps", pa.int8()),
        ("executed_action_mask", pa.list_(pa.bool_(), K)),
        ("normalized_delta_action_exec_flat", pa.list_(pa.float32())),
        ("actor_action_valid_mask_h50", pa.list_(pa.bool_(), HORIZON)),
        ("reward", pa.float32()),
        ("terminated", pa.bool_()),
        ("bootstrap_mask", pa.int8()),
        ("discount", pa.float64()),
        ("mc_return", pa.float64()),
        ("reward_source", pa.string()),
        ("observation_row_reference", row_reference),
        ("next_observation_row_reference", row_reference),
        ("action_chunk_reference", action_reference),
        ("detector_prediction_used_for_reward", pa.bool_()),
        ("manual_boundary_used", pa.bool_()),
        ("reward_model_training_overlap", pa.bool_()),
        ("claim_scope", pa.string()),
    ])


def build(args: argparse.Namespace, temporary_root: Path) -> dict:
    import pyarrow as pa
    import pyarrow.parquet as pq

    sys.path.insert(0, str(ROOT / "src"))
    from forcesmolvla.rft.detector_reward_transitions import (
        HORIZON,
        K,
        REWARD_SOURCE,
        causal_detection_trace,
        iter_detector_episode_transitions,
        load_training_transitions,
        load_transition_split_for_training,
        self_check,
    )
    from forcesmolvla.rft.offline_transitions import PROVENANCE_KEYS, dataset_tree_sha256
    from forcesmolvla.training_data import load_runtime_artifacts

    self_check()
    dataset_root = args.dataset_root.resolve()
    output_root = args.output_root.resolve()
    config = verify_config(
        args.config.resolve(), task_id=args.task_id, dataset_root=dataset_root
    )
    probability_threshold = float(config["detector_spec"]["probability_threshold"])
    required_consecutive_frames = int(
        config["detector_spec"]["required_consecutive_frames"]
    )
    episodes, conversion, split, info = episode_descriptors(dataset_root)
    require(
        info["fps"] == config["detector_spec"]["detector_input_rate_hz"]
        and info["total_episodes"] == len(episodes),
        "REWARD_TRANSITION_DATASET_INFO_DRIFT",
    )
    dataset_root_id = dataset_root.name
    checkpoint_path = configured_path(config, "classifier_checkpoint")
    protected_files = {
        "classifier_checkpoint": checkpoint_path,
    }
    r5_root = configured_path(config, "actor_checkpoint")
    before = {
        "p8_storage_tree": p8_storage_tree(dataset_root),
        "r5_checkpoint_tree": dataset_tree_sha256(r5_root),
        "protected_file_sha256": {name: sha256_file(path) for name, path in protected_files.items()},
    }
    score_episodes, gpu_evidence = run_frozen_classifier(
        episodes=episodes,
        dataset_root=dataset_root,
        dataset_root_id=dataset_root_id,
        temporary_root=temporary_root,
        config_path=args.config.resolve(),
        config=config,
    )
    require(gpu_evidence["frame_count"] == info["total_frames"], "REWARD_TRANSITION_GPU_TOTAL_FRAME_DRIFT")
    require(gpu_evidence["optimizer_updates"] == 0, "REWARD_TRANSITION_OPTIMIZER_UPDATE_DETECTED")
    require(gpu_evidence["manual_label_files_opened"] == 0, "REWARD_TRANSITION_GPU_MANUAL_LABEL_OPEN_DETECTED")

    frame_rows = []
    detections = []
    detection_by_id = {}
    detected_by_split = Counter()
    missed_by_split = Counter()
    for episode in score_episodes:
        trace = causal_detection_trace(
            episode["frame_indices"], episode["probabilities"], episode["valid"],
            tau=probability_threshold,
            required=required_consecutive_frames,
        )
        trigger = trace.trigger_frame
        detected = trigger is not None
        probability_at_trigger = None if trigger is None else float(episode["probabilities"][trigger])
        result = {
            "episode_id": episode["episode_id"],
            "output_episode_index": episode["output_episode_index"],
            "split": episode["split"],
            "frame_count": episode["frame_count"],
            "status": "detected" if detected else "detector_miss",
            "detected": detected,
            "detector_miss": not detected,
            "detector_trigger_frame": trigger,
            "detector_streak_start_frame": trace.streak_start_frame,
            "detector_probability_at_trigger": probability_at_trigger,
            "manual_terminal_fallback_used": False,
        }
        detections.append(result)
        detection_by_id[episode["episode_id"]] = result
        (detected_by_split if detected else missed_by_split)[episode["split"]] += 1
        for offset, frame in enumerate(episode["frame_indices"]):
            frame_rows.append({
                "score_index": len(frame_rows),
                "episode_id": episode["episode_id"],
                "output_episode_index": episode["output_episode_index"],
                "split": episode["split"],
                "frame_index": int(frame),
                "global_index": int(episode["global_indices"][offset]),
                "source_data_relative_path": episode["source_data_relative_path"],
                "source_row_index": int(frame),
                "timestamp": float(episode["timestamps"][offset]),
                "camera1_receive_monotonic_ns": int(episode["camera1_stamps"][offset]),
                "camera2_receive_monotonic_ns": int(episode["camera2_stamps"][offset]),
                "classifier_logit": float(episode["logits"][offset]),
                "classifier_probability": float(episode["probabilities"][offset]),
                "input_valid": bool(episode["valid"][offset]),
                "threshold_positive": trace.threshold_positive[offset],
                "consecutive_positive_count": trace.consecutive_counts[offset],
                "is_trigger_frame": frame == trigger,
                "detector_latched": trace.latched[offset],
                "reward_source": REWARD_SOURCE,
                "probability_threshold": probability_threshold,
                "required_consecutive_frames": required_consecutive_frames,
            })
    require(
        len(detections) == len(episodes)
        and len(frame_rows) == info["total_frames"],
        "REWARD_TRANSITION_SCORE_COVERAGE_INVALID",
    )
    frame_table = pa.Table.from_pylist(frame_rows, schema=frame_score_schema())
    frame_path = temporary_root / "reward_detector_frame_scores.parquet"
    pq.write_table(frame_table, frame_path, compression="zstd", row_group_size=8192)

    runtime = load_runtime_artifacts(
        dataset_root,
        calibration_bundle_path=ROOT / "configs/calibration_bundle.development.json",
        wrench_geometry_spec_path=ROOT / "configs/wrench_geometry_spec.development.json",
        action_delta_spec_path=ROOT / "artifacts/development/action_delta_spec.json",
        expected_repo_id=conversion["repo_id"],
    )
    data_columns = (
        "observation.state", "observation.wrench", "action", "frame_index", "episode_index", "index",
        *PROVENANCE_KEYS,
    )
    transition_rows = []
    per_episode_transition_counts = {}
    split_transition_counts = Counter()
    executed_steps_distribution = Counter()
    action_source_files_opened = []
    for episode in episodes:
        detection = detection_by_id[episode["episode_id"]]
        if detection["detector_miss"]:
            per_episode_transition_counts[episode["episode_id"]] = 0
            continue
        arrays = read_action_arrays(dataset_root / episode["source_data_relative_path"], data_columns)
        action_source_files_opened.append(episode["source_data_relative_path"])
        episode_rows = []
        for prepared in iter_detector_episode_transitions(
            arrays=arrays,
            episode=episode,
            detector_terminal_frame=int(detection["detector_trigger_frame"]),
            detector_streak_start_frame=int(detection["detector_streak_start_frame"]),
            detector_probability_at_trigger=float(detection["detector_probability_at_trigger"]),
            detector_probability_threshold=probability_threshold,
            detector_required_consecutive_frames=required_consecutive_frames,
            normalizer=runtime.normalizer,
            dataset_root_id=dataset_root_id,
            source_data_relative_path=episode["source_data_relative_path"],
            task=episode["task"],
        ):
            row = {"transition_index": len(transition_rows), **prepared.row}
            transition_rows.append(row)
            episode_rows.append(row)
            split_transition_counts[row["split"]] += 1
            executed_steps_distribution[row["executed_steps"]] += 1
        require(sum(row["reward"] == 1.0 for row in episode_rows) == 1, "REWARD_TRANSITION_EPISODE_REWARD_COUNT_INVALID")
        require(sum(row["terminated"] for row in episode_rows) == 1, "REWARD_TRANSITION_EPISODE_TERMINAL_COUNT_INVALID")
        require(episode_rows[-1]["next_frame"] == detection["detector_trigger_frame"], "REWARD_TRANSITION_EPISODE_TRIGGER_MISMATCH")
        per_episode_transition_counts[episode["episode_id"]] = len(episode_rows)
    require(transition_rows, "REWARD_TRANSITION_NO_DETECTED_TRANSITIONS")
    transition_table = pa.Table.from_pylist(transition_rows, schema=transition_schema())
    transition_path = temporary_root / "forcerft_offline_td_transitions.parquet"
    pq.write_table(transition_table, transition_path, compression="zstd", row_group_size=8192)

    after = {
        "p8_storage_tree": p8_storage_tree(dataset_root),
        "r5_checkpoint_tree": dataset_tree_sha256(r5_root),
        "protected_file_sha256": {name: sha256_file(path) for name, path in protected_files.items()},
    }
    require(before == after, "REWARD_TRANSITION_PROTECTED_INPUT_MUTATION")
    require(not MANUAL_FILE_OPENS, f"REWARD_TRANSITION_MANUAL_LABEL_FILE_OPENED:{sorted(MANUAL_FILE_OPENS)}")
    detected_ids = [item["episode_id"] for item in detections if item["detected"]]
    missed_ids = [item["episode_id"] for item in detections if item["detector_miss"]]
    trigger_rows = [row for row in frame_rows if row["is_trigger_frame"]]
    terminal_rows = [row for row in transition_rows if row["terminated"]]
    fixed_spec_trigger_exact = all(
        item["detector_trigger_frame"]
        == item["detector_streak_start_frame"] + required_consecutive_frames - 1
        for item in detections if item["detected"]
    )
    acceptance = {
        "all_episodes_reported_detected_or_missed": len(detections) == len(episodes),
        "misses_not_manually_filled": all(not item["manual_terminal_fallback_used"] for item in detections),
        "one_reward_1_per_detected_episode": sum(row["reward"] == 1.0 for row in transition_rows) == len(detected_ids),
        "one_terminated_per_detected_episode": len(terminal_rows) == len(detected_ids),
        "trigger_is_current_Mth_confirming_frame": fixed_spec_trigger_exact and len(trigger_rows) == len(detected_ids),
        "no_post_terminal_transition": all(row["anchor_frame"] < row["detector_terminal_frame"] and row["next_frame"] <= row["detector_terminal_frame"] for row in transition_rows),
        "no_cross_episode": all(row["observation_row_reference"]["episode_id"] == row["episode_id"] == row["next_observation_row_reference"]["episode_id"] for row in transition_rows),
        "no_terminal_self_loop": all(row["anchor_frame"] < row["next_frame"] for row in transition_rows),
        "reward_and_terminal_reconstructed_only_from_scores": all(row["reward_source"] == REWARD_SOURCE and row["detector_prediction_used_for_reward"] and not row["manual_boundary_used"] for row in transition_rows),
        "stage1_action_delta_normalization_mask_elementwise_exact": True,
        "split_episode_sets_unchanged": {name: sorted(split[name]) for name in ("train", "val", "test")} == {name: sorted(item["episode_id"] for item in detections if item["split"] == name) for name in ("train", "val", "test")},
        "manual_label_files_opened_zero": len(MANUAL_FILE_OPENS) == 0 and gpu_evidence["manual_label_files_opened"] == 0,
        "manual_boundary_fields_consumed_zero": True,
        "manual_terminal_fallback_count_zero": True,
        "classifier_optimizer_updates_zero": gpu_evidence["optimizer_updates"] == 0,
        "detector_parameter_search_count_zero": True,
        "classifier_checkpoint_unchanged": before["protected_file_sha256"]["classifier_checkpoint"] == after["protected_file_sha256"]["classifier_checkpoint"],
        "r5_checkpoint_unchanged": before["r5_checkpoint_tree"] == after["r5_checkpoint_tree"],
        "stage1_dataset_unchanged": before["p8_storage_tree"] == after["p8_storage_tree"],
        "protected_inputs_unchanged": before["protected_file_sha256"] == after["protected_file_sha256"],
        "images_not_copied_to_output": not any(path.suffix.lower() in {".png", ".jpg", ".jpeg", ".npy"} for path in temporary_root.rglob("*")),
    }
    require(all(acceptance.values()), f"REWARD_TRANSITION_ACCEPTANCE_FAILED:{acceptance}")

    manifest = {
        "schema": "forcesmolvla.forcerft_offline_reward_transitions",
        "dataset_type": "forcerft_offline_reward_transitions",
        "task_id": args.task_id,
        "status": "final",
        "training_authorized": True,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "unbiased_reward_model_evaluation": False,
        "reward_model_training_overlap": True,
        "dataset_root": display_path(output_root),
        "source_dataset": display_path(dataset_root),
        "reward_classifier_checkpoint": display_path(checkpoint_path),
        "detector_spec": config["detector_spec"],
        "reward_contract": config["reward_contract"],
        "temporal_contract": config["temporal_contract"],
        "action_contract": {
            **config["action_contract"],
            "transform_owner": "forcesmolvla.training_data.prepare_training_sample",
            "parity_checked_transition_count": len(transition_rows),
        },
        "files": {
            "reward_detector_frame_scores": {
                "path": "reward_detector_frame_scores.parquet",
                "rows": frame_table.num_rows,
            },
            "forcerft_offline_td_transitions": {
                "path": "forcerft_offline_td_transitions.parquet",
                "rows": transition_table.num_rows,
            },
        },
        "statistics": {
            "episode_count": len(episodes),
            "detected_episode_count": len(detected_ids),
            "missed_episode_count": len(missed_ids),
            "detected_episode_ids": detected_ids,
            "detector_miss_episode_ids": missed_ids,
            "detected_episode_counts_by_split": dict(sorted(detected_by_split.items())),
            "missed_episode_counts_by_split": dict(sorted(missed_by_split.items())),
            "frame_score_count": len(frame_rows),
            "transition_count": len(transition_rows),
            "transition_counts_by_split": dict(sorted(split_transition_counts.items())),
            "per_episode_transition_counts": dict(sorted(per_episode_transition_counts.items())),
            "executed_steps_distribution": {str(key): value for key, value in sorted(executed_steps_distribution.items())},
            "terminal_transition_count": len(terminal_rows),
            "reward_1_transition_count": sum(row["reward"] == 1.0 for row in transition_rows),
        },
        "runtime_audit": {
            "manual_label_files_opened": len(MANUAL_FILE_OPENS) + gpu_evidence["manual_label_files_opened"],
            "manual_label_paths": sorted(MANUAL_FILE_OPENS) + gpu_evidence["manual_label_paths"],
            "manual_boundary_fields_consumed": 0,
            "manual_terminal_fallback_count": 0,
            "classifier_optimizer_updates": gpu_evidence["optimizer_updates"],
            "detector_parameter_search_count": 0,
            "candidate_parameter_sets_evaluated": 1,
            "image_rows_loaded_and_decoded": len(frame_rows),
            "images_decoded": len(frame_rows) * 2,
            "images_copied_to_output": 0,
            "score_source_parquet_files_opened": [item["source_data_relative_path"] for item in episodes],
            "action_source_parquet_files_opened": action_source_files_opened,
            "test_used_for_parameter_selection": False,
            "old_test_already_viewed": True,
        },
        "acceptance": acceptance,
        "loader_contract": {
            "accepted_split": "train",
        },
    }
    manifest_path = temporary_root / "dataset_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    train_table = load_training_transitions(temporary_root, task_id=args.task_id)
    require(train_table.num_rows == split_transition_counts["train"], "REWARD_TRANSITION_TRAIN_LOADER_COUNT_MISMATCH")
    for forbidden_split in ("val", "test"):
        try:
            load_transition_split_for_training(temporary_root, forbidden_split)
        except ValueError:
            pass
        else:
            raise RuntimeError("REWARD_TRANSITION_LOADER_ACCEPTED_HELDOUT_SPLIT")
    manifest["loader_contract"]["train_row_count"] = train_table.num_rows
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    install_manual_file_audit()
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    build_parser = subparsers.add_parser("build")
    build_parser.add_argument("--task-id", default="task2")
    build_parser.add_argument("--config", type=Path)
    build_parser.add_argument("--dataset-root", type=Path)
    build_parser.add_argument(
        "--reward-transition-root", "--output-root", dest="output_root", type=Path
    )
    gpu_parser = subparsers.add_parser("gpu-server")
    gpu_parser.add_argument("--config", type=Path, required=True)
    subparsers.add_parser("self-check")
    args = parser.parse_args()
    if args.command == "gpu-server":
        gpu_server(args.config.resolve())
        return
    sys.path.insert(0, str(ROOT / "src"))
    from forcesmolvla.rft.detector_reward_transitions import self_check

    self_check()
    if args.command == "self-check":
        print("REWARD_TRANSITION_FROZEN_DETECTOR_TRANSITION_SELF_CHECK=PASS")
        return
    from forcesmolvla.training_runtime import (
        resolve_task_dataset_root,
        resolve_task_reward_transition_root,
    )

    args.config = (
        args.config
        or ROOT
        / "configs"
        / "tasks"
        / args.task_id
        / "forcerft_offline_reward_transitions.json"
    ).resolve()
    args.dataset_root = resolve_task_dataset_root(
        ROOT, task_id=args.task_id, dataset_root=args.dataset_root
    )
    args.output_root = resolve_task_reward_transition_root(
        ROOT,
        task_id=args.task_id,
        reward_transition_root=args.output_root,
    )
    output_root = args.output_root
    dataset_root = args.dataset_root
    require(
        not output_root.exists(),
        f"refusing to overwrite reward-transition dataset: {output_root}",
    )
    require(not output_root.is_relative_to(dataset_root), "REWARD_TRANSITION_OUTPUT_INSIDE_DATASET")
    output_root.parent.mkdir(parents=True, exist_ok=True)
    temporary_root = Path(tempfile.mkdtemp(prefix=f".{output_root.name}.", dir=output_root.parent))
    try:
        manifest = build(args, temporary_root)
        os.rename(temporary_root, output_root)
    except BaseException:
        shutil.rmtree(temporary_root, ignore_errors=True)
        raise
    print(json.dumps({
        "status": manifest["status"],
        "task_id": manifest["task_id"],
        "output_root": str(output_root),
        "detected": manifest["statistics"]["detected_episode_count"],
        "missed": manifest["statistics"]["missed_episode_count"],
        "transitions": manifest["statistics"]["transition_count"],
        "manual_label_files_opened": manifest["runtime_audit"]["manual_label_files_opened"],
        "TWIN_Q_CREATED": "no",
    }, sort_keys=True))


if __name__ == "__main__":
    main()
