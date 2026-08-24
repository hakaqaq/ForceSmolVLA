"""Strict model-side input preparation for ForceSmolVLA inference."""

from __future__ import annotations

import base64
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import socket
from typing import Any

import numpy as np
import torch

from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS

from .configuration_forcesmolvla import CAMERA1, CAMERA2
from .context import ChunkContext
from .training_data import RuntimeArtifactBundle


PROTOCOL_VERSION = "forcesmolvla-http-v1"
CLOCK_DOMAIN = "upper_host_monotonic_ns"
IMAGE_SHAPE = (480, 640, 3)
HORIZON = 50


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class CheckpointInferenceContract:
    repo_id: str
    conversion_manifest_sha256: str
    tool_profile_sha256: str
    calibration_id: str
    max_pose_age_ms: float
    state_pose_max_age_ms: float
    camera_max_age_ms: float
    max_intercamera_skew_ms: float
    max_wrench_source_gap_ms: float
    filter_warmup_samples: int
    calibration_bundle: dict[str, Any]
    wrench_geometry_spec: dict[str, Any]
    converter_runtime_spec: dict[str, Any]


def load_checkpoint_inference_contract(checkpoint_dir: Path) -> CheckpointInferenceContract:
    checkpoint_dir = Path(checkpoint_dir).resolve()
    manifests = checkpoint_dir / "manifests"
    conversion_path = manifests / "conversion_manifest.json"
    calibration_path = manifests / "calibration_bundle.development.json"
    geometry_path = manifests / "wrench_geometry_spec.development.json"
    runtime_paths = tuple(manifests.glob("converter_runtime_spec.*.development.json"))
    if len(runtime_paths) != 1:
        raise RuntimeError(
            f"CHECKPOINT_CONVERTER_RUNTIME_SPEC_COUNT_MISMATCH:{len(runtime_paths)}"
        )
    conversion = json.loads(conversion_path.read_text(encoding="utf-8"))
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    geometry = json.loads(geometry_path.read_text(encoding="utf-8"))
    runtime = json.loads(runtime_paths[0].read_text(encoding="utf-8"))
    for name, payload in (
        ("calibration", calibration),
        ("geometry", geometry),
        ("runtime", runtime),
    ):
        if payload.get("artifact_status") != "development_only":
            raise RuntimeError(f"CHECKPOINT_{name.upper()}_STATUS_MISMATCH")
    if calibration.get("formal_ready") is not False or runtime.get("formal_ready") is not False:
        raise RuntimeError("CHECKPOINT_DEVELOPMENT_ARTIFACT_FORMAL_STATUS_MISMATCH")
    tool_hash = calibration["static_transform_tcp_sensor"]["tool_profile_sha256"]
    calibration_id = str(calibration["calibration_id"])
    if (
        geometry["calibration"]["tool_profile_sha256"] != tool_hash
        or geometry["calibration"]["calibration_id"] != calibration_id
    ):
        raise RuntimeError("CHECKPOINT_GEOMETRY_CALIBRATION_BINDING_MISMATCH")
    declared_ids = set(conversion.get("calibration_id_by_index", {}).values())
    if declared_ids != {calibration_id}:
        raise RuntimeError("CHECKPOINT_CONVERSION_CALIBRATION_ID_MISMATCH")
    if geometry["pose_selection"]["method"] != "causal_zoh_latest":
        raise RuntimeError("CHECKPOINT_NONCAUSAL_WRENCH_GEOMETRY_FORBIDDEN")
    if geometry["pose_selection"]["future_interpolation"] != "forbidden":
        raise RuntimeError("CHECKPOINT_FUTURE_WRENCH_INTERPOLATION_FORBIDDEN")
    if runtime["wrench_filter"]["future_interpolation"] != "forbidden":
        raise RuntimeError("CHECKPOINT_FUTURE_FILTER_INTERPOLATION_FORBIDDEN")
    if runtime["cameras"]["order"] != [
        "camera1:D435-third-person",
        "camera2:D405-wrist",
    ]:
        raise RuntimeError("CHECKPOINT_CAMERA_ORDER_MISMATCH")
    if not np.isclose(
        float(geometry["pose_selection"]["max_pose_age_ms"]),
        float(runtime["pose"]["max_age_ms"]),
        rtol=0.0,
        atol=0.0,
    ):
        raise RuntimeError("CHECKPOINT_POSE_AGE_CONTRACT_MISMATCH")
    geometry_transform = geometry["static_transform_tcp_sensor"]
    calibration_transform = calibration["static_transform_tcp_sensor"]
    for field in ("translation_m", "quaternion_xyzw"):
        if not np.array_equal(
            np.asarray(geometry_transform[field], dtype=np.float64),
            np.asarray(calibration_transform[field], dtype=np.float64),
        ):
            raise RuntimeError("CHECKPOINT_STATIC_TRANSFORM_MISMATCH")
    repo_id = conversion.get("repo_id")
    if not isinstance(repo_id, str) or not repo_id:
        raise RuntimeError("CHECKPOINT_CONVERSION_REPO_ID_MISSING")
    return CheckpointInferenceContract(
        repo_id=repo_id,
        conversion_manifest_sha256=sha256_file(conversion_path),
        tool_profile_sha256=str(tool_hash),
        calibration_id=calibration_id,
        max_pose_age_ms=float(runtime["pose"]["max_age_ms"]),
        state_pose_max_age_ms=float(runtime["pose"]["max_age_ms"]),
        camera_max_age_ms=float(runtime["cameras"]["max_age_ms"]),
        max_intercamera_skew_ms=float(runtime["cameras"]["max_intercamera_skew_ms"]),
        max_wrench_source_gap_ms=float(runtime["wrench_filter"]["max_source_gap_ms"]),
        filter_warmup_samples=int(runtime["wrench_filter"]["warmup_samples"]),
        calibration_bundle=calibration,
        wrench_geometry_spec=geometry,
        converter_runtime_spec=runtime,
    )


def decode_rgb_image(payload: Any, *, label: str) -> np.ndarray:
    if not isinstance(payload, dict) or set(payload) != {"encoding", "shape", "data"}:
        raise ValueError(f"{label} must be an exact encoded RGB image object")
    if payload["encoding"] != "raw-uint8-base64" or tuple(payload["shape"]) != IMAGE_SHAPE:
        raise ValueError(f"{label} must be raw RGB uint8 with shape {IMAGE_SHAPE}")
    try:
        raw = base64.b64decode(payload["data"], validate=True)
    except Exception as error:
        raise ValueError(f"{label} contains invalid base64") from error
    expected_bytes = int(np.prod(IMAGE_SHAPE))
    if len(raw) != expected_bytes:
        raise ValueError(f"{label} byte length mismatch")
    return np.frombuffer(raw, dtype=np.uint8).reshape(IMAGE_SHAPE).copy()


def _finite_vector(value: Any, width: int, label: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (width,) or not np.all(np.isfinite(array)):
        raise ValueError(f"{label} must be finite shape ({width},)")
    return array


def validate_inference_request(
    request: dict[str, Any],
    contract: CheckpointInferenceContract,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    required = {
        "protocol_version",
        "request_id",
        "chunk_id",
        "client_hostname",
        "clock_domain_id",
        "dataset_repo_id",
        "tool_profile_sha256",
        "calibration_id",
        "task",
        "state7",
        "wrench6",
        "camera1",
        "camera2",
        "provenance",
    }
    if set(request) != required:
        raise ValueError(
            f"inference request fields mismatch: missing={sorted(required - set(request))} "
            f"extra={sorted(set(request) - required)}"
        )
    if request["protocol_version"] != PROTOCOL_VERSION:
        raise ValueError("INFERENCE_PROTOCOL_VERSION_MISMATCH")
    for field in ("request_id", "chunk_id", "task"):
        if not isinstance(request[field], str) or not request[field].strip():
            raise ValueError(f"{field} must be a nonempty string")
    if request["client_hostname"] != socket.gethostname():
        raise RuntimeError("INFERENCE_CLIENT_SERVER_HOST_MISMATCH")
    if request["clock_domain_id"] != CLOCK_DOMAIN:
        raise RuntimeError("INFERENCE_CLOCK_DOMAIN_MISMATCH")
    if request["dataset_repo_id"] != contract.repo_id:
        raise RuntimeError("INFERENCE_DATASET_REPO_ID_MISMATCH")
    if request["tool_profile_sha256"] != contract.tool_profile_sha256:
        raise RuntimeError("INFERENCE_TOOL_PROFILE_HASH_MISMATCH")
    if request["calibration_id"] != contract.calibration_id:
        raise RuntimeError("INFERENCE_CALIBRATION_ID_MISMATCH")

    state7 = _finite_vector(request["state7"], 7, "state7")
    wrench6 = _finite_vector(request["wrench6"], 6, "wrench6")
    if not 0.0 <= state7[6] <= 0.1:
        raise ValueError("state7 gripper width must be in [0,0.1] m")
    camera1 = decode_rgb_image(request["camera1"], label="camera1")
    camera2 = decode_rgb_image(request["camera2"], label="camera2")
    provenance = request["provenance"]
    required_provenance = {
        "t_ref_ns",
        "tau0_ns",
        "pose_receive_monotonic_ns",
        "state_pose_age_ms",
        "camera1_receive_monotonic_ns",
        "camera1_age_ms",
        "camera2_receive_monotonic_ns",
        "camera2_age_ms",
        "intercamera_skew_ms",
        "gripper_receive_monotonic_ns",
        "wrench_receive_monotonic_ns",
        "geometry_pose_source_stamp_ns",
        "wrench_raw_source_stamp_ns",
        "wrench_filter_output_stamp_ns",
        "geometry_pose_age_ms",
        "filter_warmup_complete",
        "wrench_geometry_valid",
        "session_id",
    }
    if not isinstance(provenance, dict) or set(provenance) != required_provenance:
        raise ValueError("inference provenance fields mismatch")
    t_ref_ns = int(provenance["t_ref_ns"])
    tau0_ns = int(provenance["tau0_ns"])
    if t_ref_ns <= 0 or tau0_ns != t_ref_ns:
        raise ValueError("t_ref_ns/tau0_ns must be one positive atomic snapshot timestamp")
    receive_fields = (
        "pose_receive_monotonic_ns",
        "camera1_receive_monotonic_ns",
        "camera2_receive_monotonic_ns",
        "gripper_receive_monotonic_ns",
        "wrench_receive_monotonic_ns",
    )
    if any(int(provenance[name]) <= 0 or int(provenance[name]) > t_ref_ns for name in receive_fields):
        raise RuntimeError("INFERENCE_NONCAUSAL_RECEIVE_TIMESTAMP")
    age_limits = (
        ("state_pose_age_ms", contract.state_pose_max_age_ms),
        ("geometry_pose_age_ms", contract.max_pose_age_ms),
        ("camera1_age_ms", contract.camera_max_age_ms),
        ("camera2_age_ms", contract.camera_max_age_ms),
        ("intercamera_skew_ms", contract.max_intercamera_skew_ms),
    )
    for name, maximum in age_limits:
        age = float(provenance[name])
        if not np.isfinite(age) or age < 0.0 or age > maximum:
            raise RuntimeError(f"INFERENCE_{name.upper()}_EXCEEDED")
    receive_age_pairs = (
        ("pose_receive_monotonic_ns", "state_pose_age_ms"),
        ("camera1_receive_monotonic_ns", "camera1_age_ms"),
        ("camera2_receive_monotonic_ns", "camera2_age_ms"),
    )
    for receive_name, age_name in receive_age_pairs:
        computed_ms = (t_ref_ns - int(provenance[receive_name])) / 1.0e6
        if not np.isclose(
            computed_ms,
            float(provenance[age_name]),
            rtol=0.0,
            atol=1.0e-6,
        ):
            raise RuntimeError(f"INFERENCE_{age_name.upper()}_ARITHMETIC_MISMATCH")
    computed_skew_ms = abs(
        int(provenance["camera1_receive_monotonic_ns"])
        - int(provenance["camera2_receive_monotonic_ns"])
    ) / 1.0e6
    if not np.isclose(
        computed_skew_ms,
        float(provenance["intercamera_skew_ms"]),
        rtol=0.0,
        atol=1.0e-6,
    ):
        raise RuntimeError("INFERENCE_INTERCAMERA_SKEW_ARITHMETIC_MISMATCH")
    pose_source = int(provenance["geometry_pose_source_stamp_ns"])
    wrench_source = int(provenance["wrench_raw_source_stamp_ns"])
    filter_source = int(provenance["wrench_filter_output_stamp_ns"])
    if pose_source <= 0 or pose_source > wrench_source or filter_source != wrench_source:
        raise RuntimeError("INFERENCE_WRENCH_CAUSAL_GEOMETRY_INVALID")
    source_age_ms = (wrench_source - pose_source) / 1.0e6
    if not np.isclose(
        source_age_ms,
        float(provenance["geometry_pose_age_ms"]),
        rtol=0.0,
        atol=1.0e-6,
    ):
        raise RuntimeError("INFERENCE_WRENCH_POSE_AGE_ARITHMETIC_MISMATCH")
    if provenance["filter_warmup_complete"] is not True:
        raise RuntimeError("INFERENCE_WRENCH_FILTER_WARMUP_INCOMPLETE")
    if provenance["wrench_geometry_valid"] is not True:
        raise RuntimeError("INFERENCE_WRENCH_GEOMETRY_INVALID")
    if not isinstance(provenance["session_id"], str) or not provenance["session_id"]:
        raise ValueError("session_id must be a nonempty string")
    return state7, wrench6, camera1, camera2


def prepare_policy_inputs(
    policy,
    request: dict[str, Any],
    runtime_artifacts: RuntimeArtifactBundle,
    contract: CheckpointInferenceContract,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], ChunkContext]:
    state7, wrench6, camera1, camera2 = validate_inference_request(request, contract)
    normalizer = runtime_artifacts.normalizer
    state = normalizer.state7.apply(state7).astype(np.float32)
    wrench = normalizer.wrench6.apply(wrench6).astype(np.float32)
    tokenizer = policy.model.vlm_with_expert.processor.tokenizer
    tokenizer.padding_side = "right"
    tokenizer.truncation_side = "right"
    encoded = tokenizer(
        [request["task"].strip() + "\n"],
        padding="max_length",
        truncation=True,
        max_length=48,
        return_tensors="pt",
    )
    image_tensors = []
    for image in (camera1, camera2):
        tensor = torch.from_numpy(image).permute(2, 0, 1).contiguous()
        image_tensors.append(tensor.to(device=device, dtype=torch.float32).div_(255.0))
    raw_snapshot = torch.from_numpy(state7.astype(np.float32)).unsqueeze(0).to(device)
    batch = {
        CAMERA1: image_tensors[0].unsqueeze(0),
        CAMERA2: image_tensors[1].unsqueeze(0),
        "observation.state": torch.from_numpy(state).unsqueeze(0).to(device),
        "observation.wrench": torch.from_numpy(wrench).unsqueeze(0).to(device),
        OBS_LANGUAGE_TOKENS: encoded["input_ids"].to(device),
        OBS_LANGUAGE_ATTENTION_MASK: encoded["attention_mask"].to(
            device=device, dtype=torch.bool
        ),
        "raw_state_snapshot": raw_snapshot,
    }
    provenance = request["provenance"]
    valid = torch.ones(1, HORIZON, dtype=torch.bool, device=device)
    context = ChunkContext(
        policy_generation=policy._context_generation,
        raw_state_snapshot=raw_snapshot,
        t_ref_ns=torch.tensor([int(provenance["t_ref_ns"])], dtype=torch.int64),
        tau0_ns=torch.tensor([int(provenance["tau0_ns"])], dtype=torch.int64),
        clock_domain_id=(CLOCK_DOMAIN,),
        episode_id=("live-inference",),
        session_id=(str(provenance["session_id"]),),
        sample_id=(str(request["request_id"]),),
        chunk_id=(str(request["chunk_id"]),),
        action_valid_mask=valid,
        suffix_valid_mask=valid.clone(),
        calibration_bundle_hash=(runtime_artifacts.calibration_bundle_sha256,),
        wrench_geometry_spec_hash=(runtime_artifacts.wrench_geometry_spec_sha256,),
        normalizer_hash=(runtime_artifacts.normalizer_manifest_sha256,),
        calibration_mapping_hash_or_none=(contract.tool_profile_sha256,),
        wrench_geometry_valid=torch.ones(1, dtype=torch.bool, device=device),
        runtime_artifact_compatible=torch.ones(1, dtype=torch.bool, device=device),
        selected_provenance=(dict(provenance),),
    )
    context.validate(
        batch_size=1,
        horizon=HORIZON,
        policy_generation=policy._context_generation,
    )
    return batch, context
