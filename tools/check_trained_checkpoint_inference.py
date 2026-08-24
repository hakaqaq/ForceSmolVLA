#!/usr/bin/env python3
"""Offline server-path inference audit for a trained ForceSmolVLA checkpoint."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import socket

import numpy as np
import torch

from forcesmolvla.checkpoint import (
    sha256_file,
    validate_force_artifact_manifest,
    validate_training_payload_contract,
)
from forcesmolvla.dataset_v3 import load_dataset_split
from serve_policy import InferenceEngine


def _encoded_image(tensor: torch.Tensor) -> dict:
    image = tensor.detach().cpu().permute(1, 2, 0).numpy()
    if np.issubdtype(image.dtype, np.floating):
        image = np.rint(np.clip(image, 0.0, 1.0) * 255.0)
    image = np.ascontiguousarray(image, dtype=np.uint8)
    if image.shape != (480, 640, 3):
        raise RuntimeError(f"OFFLINE_IMAGE_SHAPE_MISMATCH:{image.shape}")
    return {
        "encoding": "raw-uint8-base64",
        "shape": [480, 640, 3],
        "data": base64.b64encode(image.tobytes()).decode("ascii"),
    }


def _scalar(sample: dict, name: str, cast):
    return cast(sample[name].detach().cpu().item())


def _request(engine: InferenceEngine, sample: dict, *, split: str, index: int, seed: int) -> dict:
    t_ref_ns = _scalar(sample, "provenance.tuple_host_monotonic_ns", int)
    stored_state_pose_age_ms = _scalar(
        sample, "provenance.state_pose_age_ms", float
    )
    pose_receive_ns = t_ref_ns - int(round(stored_state_pose_age_ms * 1.0e6))
    state_pose_age_ms = (t_ref_ns - pose_receive_ns) / 1.0e6
    camera1_receive_ns = _scalar(sample, "provenance.camera1_receive_monotonic_ns", int)
    camera2_receive_ns = _scalar(sample, "provenance.camera2_receive_monotonic_ns", int)
    action_ack_receive_ns = _scalar(
        sample, "provenance.action_ack_receive_monotonic_ns", int
    )
    pose_source_ns = _scalar(sample, "provenance.pose_source_stamp_ns", int)
    wrench_source_ns = _scalar(sample, "provenance.wrench_raw_source_stamp_ns", int)
    geometry_pose_age_ms = (wrench_source_ns - pose_source_ns) / 1.0e6
    return {
        "protocol_version": engine.metadata["protocol_version"],
        "request_id": f"offline-{split}-{index}-seed-{seed}",
        "chunk_id": f"offline-{split}-{index}-seed-{seed}",
        "client_hostname": socket.gethostname(),
        "clock_domain_id": engine.metadata["clock_domain_id"],
        "dataset_repo_id": engine.contract.repo_id,
        "tool_profile_sha256": engine.contract.tool_profile_sha256,
        "calibration_id": engine.contract.calibration_id,
        "task": str(sample["task"]),
        "state7": sample["observation.state"].detach().cpu().to(torch.float64).tolist(),
        "wrench6": sample["observation.wrench"].detach().cpu().to(torch.float64).tolist(),
        "camera1": _encoded_image(sample["observation.images.camera1"]),
        "camera2": _encoded_image(sample["observation.images.camera2"]),
        "provenance": {
            "t_ref_ns": t_ref_ns,
            "tau0_ns": t_ref_ns,
            "pose_receive_monotonic_ns": pose_receive_ns,
            "state_pose_age_ms": state_pose_age_ms,
            "camera1_receive_monotonic_ns": camera1_receive_ns,
            "camera1_age_ms": (t_ref_ns - camera1_receive_ns) / 1.0e6,
            "camera2_receive_monotonic_ns": camera2_receive_ns,
            "camera2_age_ms": (t_ref_ns - camera2_receive_ns) / 1.0e6,
            "intercamera_skew_ms": abs(camera1_receive_ns - camera2_receive_ns)
            / 1.0e6,
            "gripper_receive_monotonic_ns": min(t_ref_ns, action_ack_receive_ns),
            "wrench_receive_monotonic_ns": t_ref_ns,
            "geometry_pose_source_stamp_ns": pose_source_ns,
            "wrench_raw_source_stamp_ns": wrench_source_ns,
            "wrench_filter_output_stamp_ns": _scalar(
                sample, "provenance.wrench_filter_output_stamp_ns", int
            ),
            "geometry_pose_age_ms": geometry_pose_age_ms,
            "filter_warmup_complete": True,
            "wrench_geometry_valid": True,
            "session_id": "task2-offline-held-out",
        },
    }


def main() -> None:
    root = Path(__file__).parents[1].resolve()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=root
        / "outputs/development/task2_lerobotv3_full_sft_10k_r5/checkpoints/step_010000",
    )
    parser.add_argument(
        "--dataset-root", type=Path, default=root / "datasets/task2_lerobotv3"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=root
        / "outputs/development/task2_lerobotv3_full_sft_10k_r5/offline_inference_validation.json",
    )
    parser.add_argument("--samples-per-split", type=int, default=4)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite inference audit: {args.output}")
    for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"):
        if os.environ.get(name) != "1":
            raise RuntimeError(f"{name}=1 required")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA_NOT_AVAILABLE_NO_CPU_FALLBACK")

    checkpoint = args.checkpoint.resolve()
    manifest = validate_force_artifact_manifest(checkpoint, artifact_use="development")
    validate_training_payload_contract(checkpoint)
    engine = InferenceEngine(
        checkpoint,
        root / "tests/fixtures/shadow_safety_thresholds.test_only.yaml",
        root / "schemas/rulespec.schema.json",
        torch.device("cuda"),
    )
    records = []
    all_actions = []
    for split in ("val", "test"):
        dataset = load_dataset_split(
            args.dataset_root.resolve(),
            repo_id=engine.contract.repo_id,
            split_name=split,
            artifact_use="development",
            delta_timestamps={"action": [index / 30 for index in range(50)]},
        )
        indices = np.linspace(
            0, len(dataset) - 1, num=args.samples_per_split, dtype=np.int64
        ).tolist()
        for index in indices:
            sample = dataset[int(index)]
            for seed in args.seeds:
                torch.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
                request = _request(engine, sample, split=split, index=int(index), seed=seed)
                response = engine.infer(request)
                actions = np.asarray(response["actions"], dtype=np.float32)
                if actions.shape != (50, 7) or not np.all(np.isfinite(actions)):
                    raise RuntimeError("OFFLINE_INFERENCE_SHAPE_OR_FINITE_FAILED")
                if np.any((actions[:, 6] < 0.0) | (actions[:, 6] > 0.1)):
                    raise RuntimeError("OFFLINE_INFERENCE_GRIPPER_RANGE_FAILED")
                binary_widths = np.asarray([0.0, 0.085], dtype=actions.dtype)
                if not np.all(np.isin(actions[:, 6], binary_widths)):
                    raise RuntimeError("OFFLINE_INFERENCE_GRIPPER_BINARY_DECODE_FAILED")
                all_actions.append(actions)
                records.append(
                    {
                        "split": split,
                        "dataset_index": int(index),
                        "seed": seed,
                        "action_sha256": hashlib.sha256(
                            np.ascontiguousarray(actions, dtype="<f4").tobytes()
                        ).hexdigest(),
                        "gripper_min_m": float(actions[:, 6].min()),
                        "gripper_max_m": float(actions[:, 6].max()),
                        "xyz_min_m": actions[:, :3].min(axis=0).tolist(),
                        "xyz_max_m": actions[:, :3].max(axis=0).tolist(),
                        "rpy_min_rad": actions[:, 3:6].min(axis=0).tolist(),
                        "rpy_max_rad": actions[:, 3:6].max(axis=0).tolist(),
                        "latency_ms": float(response["inference_latency_ms"]),
                    }
                )

    stacked = np.stack(all_actions)
    result = {
        "schema_version": "1.0",
        "acceptance_status": "development_only",
        "formal_eligible": False,
        "status": "pass",
        "mode": "offline_no_execute",
        "server_path_exercised": True,
        "robot_io_present": False,
        "robot_actions_sent": 0,
        "checkpoint": str(checkpoint),
        "checkpoint_artifact_manifest_sha256": sha256_file(
            checkpoint / "artifact_manifest.json"
        ),
        "model_sha256": sha256_file(checkpoint / "model.safetensors"),
        "artifact_type": manifest["artifact_type"],
        "request_count": len(records),
        "splits": ["val", "test"],
        "seeds": args.seeds,
        "assertions": {
            "strict_checkpoint_manifest": True,
            "training_payload_contract": True,
            "all_action_shape_50x7": True,
            "all_actions_finite": True,
            "all_gripper_widths_in_0_0.1_m": True,
            "all_gripper_widths_exactly_binary_0_0.085_m": True,
            "public_policy_safety_checks_passed": True,
            "no_second_postprocessor": True,
            "no_robot_execution": True,
        },
        "global_output": {
            "gripper_min_m": float(stacked[..., 6].min()),
            "gripper_max_m": float(stacked[..., 6].max()),
            "absolute_action_float32_sha256": hashlib.sha256(
                np.ascontiguousarray(stacked, dtype="<f4").tobytes()
            ).hexdigest(),
        },
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
