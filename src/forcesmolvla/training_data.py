"""Fail-closed v4.1 training sample preparation."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .action_delta import ActionDeltaProcessor
from .normalizer import (
    CartesianNormalizerBundle,
    DELTA_ACTION_FIT_CONTRACT,
    FrozenFeatureNormalizer,
    NORMALIZER_SCHEMA_VERSION,
    NormalizationLedger,
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(payload: dict) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


@dataclass(frozen=True)
class RuntimeArtifactBundle:
    normalizer: CartesianNormalizerBundle
    normalizer_manifest_sha256: str
    calibration_bundle_sha256: str
    wrench_geometry_spec_sha256: str
    split_sha256: str
    action_delta_spec_sha256: str
    action_delta_source_sha256: str

    def validate_action_contract(self) -> None:
        for name, digest in (
            ("action_delta_spec", self.action_delta_spec_sha256),
            ("action_delta_source", self.action_delta_source_sha256),
        ):
            if len(digest) != 64 or any(value not in "0123456789abcdef" for value in digest):
                raise RuntimeError(f"RUNTIME_{name.upper()}_HASH_INVALID")
        source = Path(__import__("forcesmolvla.action_delta", fromlist=["__file__"]).__file__)
        if _sha256_file(source) != self.action_delta_source_sha256:
            raise RuntimeError("RUNTIME_ACTION_DELTA_SOURCE_HASH_MISMATCH")

    def validate_context_hashes(self, context) -> None:
        expected = (
            ("normalizer", context.normalizer_hash, self.normalizer_manifest_sha256),
            ("calibration", context.calibration_bundle_hash, self.calibration_bundle_sha256),
            ("geometry", context.wrench_geometry_spec_hash, self.wrench_geometry_spec_sha256),
        )
        for name, actual, digest in expected:
            if any(value != digest for value in actual):
                raise RuntimeError(f"RUNTIME_{name.upper()}_HASH_MISMATCH")


def load_normalizer_manifest(path: Path) -> CartesianNormalizerBundle:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != NORMALIZER_SCHEMA_VERSION:
        raise ValueError("legacy or unknown normalizer schema")
    if payload.get("owner") != "forcesmolvla.CartesianNormalizerBundle":
        raise ValueError("unexpected normalizer owner")
    if payload.get("inherited_lerobot_normalizers") != "Identity/disconnected":
        raise ValueError("inherited LeRobot normalizers must be disconnected")
    if payload.get("fit_contract", {}).get("delta_action7") != DELTA_ACTION_FIT_CONTRACT:
        raise ValueError("delta-action normalizer fit contract mismatch")

    normalizers = {}
    for name, width in (("state7", 7), ("wrench6", 6), ("delta_action7", 7)):
        feature = payload["features"][name]
        normalizer = FrozenFeatureNormalizer(
            name=name,
            mean=np.asarray(feature["mean"], dtype=np.float64),
            std=np.asarray(feature["std"], dtype=np.float64),
            fit_episode_ids=tuple(feature["fit_episode_ids"]),
        )
        if len(normalizer.mean) != width:
            raise ValueError(f"{name} normalizer width mismatch")
        normalizers[name] = normalizer

    bundle = CartesianNormalizerBundle(
        **normalizers,
        split_sha256=payload["split_sha256"],
        calibration_bundle_sha256=payload["calibration_bundle_sha256"],
        wrench_geometry_spec_sha256=payload["wrench_geometry_spec_sha256"],
        action_target_population=payload.get("action_target_population"),
    )
    population = payload.get("action_target_population", {})
    if (
        population.get("status") != "pass"
        or population.get("semantic_name") != "action_target7"
        or population.get("horizon") != 50
        or population.get("split_sha256") != bundle.split_sha256
        or population.get("valid_pair_count", 0) <= 0
    ):
        raise ValueError("action-target population binding is missing or invalid")
    if bundle.manifest() != payload:
        raise ValueError("normalizer manifest hash or contents mismatch")
    return bundle


def load_normalizer_bundle(dataset_root: Path) -> CartesianNormalizerBundle:
    return load_normalizer_manifest(dataset_root / "normalizer_manifest.json")


def load_runtime_artifacts(
    dataset_root: Path,
    *,
    calibration_bundle_path: Path,
    wrench_geometry_spec_path: Path,
    action_delta_spec_path: Path,
    expected_repo_id: str,
) -> RuntimeArtifactBundle:
    """Load and recompute every disk binding used by inference and training."""

    normalizer = load_normalizer_bundle(dataset_root)
    split = json.loads((dataset_root / "split_manifest.json").read_text(encoding="utf-8"))
    conversion = json.loads(
        (dataset_root / "conversion_manifest.json").read_text(encoding="utf-8")
    )
    split_sha256 = _canonical_sha256(split)
    calibration_sha256 = _sha256_file(calibration_bundle_path)
    geometry_sha256 = _sha256_file(wrench_geometry_spec_path)
    action_delta_spec = json.loads(action_delta_spec_path.read_text(encoding="utf-8"))
    action_delta_spec_sha256 = _sha256_file(action_delta_spec_path)
    action_delta_source_sha256 = str(action_delta_spec.get("source_sha256", ""))
    if (
        action_delta_spec.get("status") != "development_only"
        or action_delta_spec.get("schema_version") != "1.0"
        or action_delta_spec.get("representation") != "Cartesian7D_xyz_rpy_gripper"
        or action_delta_spec.get("second_lerobot_postprocessor_allowed") is not False
    ):
        raise RuntimeError("ACTION_DELTA_SPEC_CONTRACT_DRIFT")
    normalizer_path = dataset_root / "normalizer_manifest.json"
    normalizer_sha256 = _sha256_file(normalizer_path)
    manifest = normalizer.manifest()
    fit_ids = tuple(split.get("train", ()))
    if conversion.get("repo_id") != expected_repo_id:
        raise ValueError("conversion manifest repo_id mismatch")
    if conversion.get("split_sha256") != split_sha256 or normalizer.split_sha256 != split_sha256:
        raise ValueError("split artifact hash mismatch")
    if conversion.get("normalizer_stats_sha256") != manifest["normalizer_stats_sha256"]:
        raise ValueError("conversion/normalizer statistics hash mismatch")
    if normalizer.calibration_bundle_sha256 != calibration_sha256:
        raise ValueError("calibration bundle hash mismatch")
    if normalizer.wrench_geometry_spec_sha256 != geometry_sha256:
        raise ValueError("wrench geometry spec hash mismatch")
    for feature in (normalizer.state7, normalizer.wrench6, normalizer.delta_action7):
        if feature.fit_episode_ids != fit_ids:
            raise ValueError(f"{feature.name} was not fit on the exact train split")
    bundle = RuntimeArtifactBundle(
        normalizer=normalizer,
        normalizer_manifest_sha256=normalizer_sha256,
        calibration_bundle_sha256=calibration_sha256,
        wrench_geometry_spec_sha256=geometry_sha256,
        split_sha256=split_sha256,
        action_delta_spec_sha256=action_delta_spec_sha256,
        action_delta_source_sha256=action_delta_source_sha256,
    )
    bundle.validate_action_contract()
    return bundle


def load_checkpoint_runtime_artifacts(checkpoint_dir: Path) -> RuntimeArtifactBundle:
    """Load the self-contained inference artifacts embedded in a strict checkpoint."""

    checkpoint_dir = Path(checkpoint_dir).resolve()
    manifests = checkpoint_dir / "manifests"
    normalizer_path = manifests / "normalizer_manifest.json"
    split_path = manifests / "split_manifest.json"
    conversion_path = manifests / "conversion_manifest.json"
    calibration_path = manifests / "calibration_bundle.development.json"
    geometry_path = manifests / "wrench_geometry_spec.development.json"
    action_delta_path = manifests / "action_delta_spec.json"
    required = (
        normalizer_path,
        split_path,
        conversion_path,
        calibration_path,
        geometry_path,
        action_delta_path,
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"checkpoint runtime artifacts are incomplete: {missing}")

    normalizer = load_normalizer_manifest(normalizer_path)
    split = json.loads(split_path.read_text(encoding="utf-8"))
    conversion = json.loads(conversion_path.read_text(encoding="utf-8"))
    action_delta_spec = json.loads(action_delta_path.read_text(encoding="utf-8"))
    split_sha256 = _canonical_sha256(split)
    calibration_sha256 = _sha256_file(calibration_path)
    geometry_sha256 = _sha256_file(geometry_path)
    action_delta_spec_sha256 = _sha256_file(action_delta_path)
    action_delta_source_sha256 = str(action_delta_spec.get("source_sha256", ""))
    normalizer_sha256 = _sha256_file(normalizer_path)
    repo_id = conversion.get("repo_id")
    if not isinstance(repo_id, str) or not repo_id:
        raise RuntimeError("CHECKPOINT_CONVERSION_REPO_ID_MISSING")
    if (
        action_delta_spec.get("status") != "development_only"
        or action_delta_spec.get("schema_version") != "1.0"
        or action_delta_spec.get("representation") != "Cartesian7D_xyz_rpy_gripper"
        or action_delta_spec.get("second_lerobot_postprocessor_allowed") is not False
    ):
        raise RuntimeError("ACTION_DELTA_SPEC_CONTRACT_DRIFT")
    if conversion.get("split_sha256") != split_sha256:
        raise RuntimeError("CHECKPOINT_CONVERSION_SPLIT_HASH_MISMATCH")
    if normalizer.split_sha256 != split_sha256:
        raise RuntimeError("CHECKPOINT_NORMALIZER_SPLIT_HASH_MISMATCH")
    if normalizer.calibration_bundle_sha256 != calibration_sha256:
        raise RuntimeError("CHECKPOINT_CALIBRATION_HASH_MISMATCH")
    if normalizer.wrench_geometry_spec_sha256 != geometry_sha256:
        raise RuntimeError("CHECKPOINT_WRENCH_GEOMETRY_HASH_MISMATCH")
    if conversion.get("normalizer_stats_sha256") != normalizer.manifest()[
        "normalizer_stats_sha256"
    ]:
        raise RuntimeError("CHECKPOINT_NORMALIZER_STATISTICS_HASH_MISMATCH")
    fit_ids = tuple(split.get("train", ()))
    for feature in (normalizer.state7, normalizer.wrench6, normalizer.delta_action7):
        if feature.fit_episode_ids != fit_ids:
            raise RuntimeError("CHECKPOINT_NORMALIZER_TRAIN_SPLIT_MISMATCH")

    bundle = RuntimeArtifactBundle(
        normalizer=normalizer,
        normalizer_manifest_sha256=normalizer_sha256,
        calibration_bundle_sha256=calibration_sha256,
        wrench_geometry_spec_sha256=geometry_sha256,
        split_sha256=split_sha256,
        action_delta_spec_sha256=action_delta_spec_sha256,
        action_delta_source_sha256=action_delta_source_sha256,
    )
    bundle.validate_action_contract()
    return bundle


def prepare_training_sample(sample: dict, bundle: CartesianNormalizerBundle) -> dict:
    raw_state = np.asarray(sample["observation.state"], dtype=np.float64)
    raw_wrench = np.asarray(sample["observation.wrench"], dtype=np.float64)
    absolute_action = np.asarray(sample["action"], dtype=np.float64)
    action_is_pad = np.asarray(sample["action_is_pad"], dtype=np.bool_)
    if raw_state.shape != (7,) or raw_wrench.shape != (6,):
        raise ValueError("expected state7 and wrench6")
    if absolute_action.shape != (50, 7) or action_is_pad.shape != (50,):
        raise ValueError("expected a 50-step action7 chunk and padding mask")
    action_valid_mask = ~action_is_pad
    valid_count = int(action_valid_mask.sum())
    if not np.array_equal(action_valid_mask, np.arange(50) < valid_count):
        raise ValueError("action validity must be physically right-padded")
    if not np.all(np.isfinite(raw_state)) or not np.all(np.isfinite(raw_wrench)) or not np.all(
        np.isfinite(absolute_action)
    ):
        raise ValueError("training sample contains nonfinite values")
    validity_bits = int(np.asarray(sample["provenance.validity_bits"]).reshape(-1)[0])
    if validity_bits != 0xFF:
        raise ValueError(f"training tuple is not fully eligible: validity_bits={validity_bits:#x}")
    for key in (
        "provenance.state_pose_age_ms",
        "provenance.camera1_age_ms",
        "provenance.camera2_age_ms",
        "provenance.intercamera_skew_ms",
        "provenance.pose_age_ms",
        "provenance.action_ack_age_ms",
    ):
        age = float(np.asarray(sample[key]).reshape(-1)[0])
        if not np.isfinite(age) or age < 0:
            raise ValueError(f"training tuple has invalid noncausal age: {key}")
    pose_stamp = int(np.asarray(sample["provenance.pose_source_stamp_ns"]).reshape(-1)[0])
    wrench_stamp = int(
        np.asarray(sample["provenance.wrench_raw_source_stamp_ns"]).reshape(-1)[0]
    )
    filter_stamp = int(
        np.asarray(sample["provenance.wrench_filter_output_stamp_ns"]).reshape(-1)[0]
    )
    if pose_stamp > wrench_stamp or filter_stamp < wrench_stamp:
        raise ValueError("training tuple violates causal wrench timestamp ordering")

    delta_action = ActionDeltaProcessor.to_delta(absolute_action, raw_state)
    batch_id = f"episode={int(sample['episode_index'])}/frame={int(sample['frame_index'])}"
    ledger = NormalizationLedger()
    state, wrench, action = bundle.normalize_once(
        batch_id=batch_id,
        state7=raw_state,
        wrench6=raw_wrench,
        delta_action7=delta_action,
        ledger=ledger,
    )
    if ledger.counts != {"state7": 1, "wrench6": 1, "delta_action7": 1}:
        raise AssertionError("normalizers were not applied exactly once")

    digest = hashlib.sha256()
    for value in (raw_state, raw_wrench, absolute_action, action_is_pad):
        digest.update(np.ascontiguousarray(value).view(np.uint8))
    return {
        "batch_id": batch_id,
        "batch_sha256": digest.hexdigest(),
        "state7": state.astype(np.float32),
        "wrench6": wrench.astype(np.float32),
        "delta_action7": action.astype(np.float32),
        "action_valid_mask": action_valid_mask,
        "task": str(sample["task"]),
        "camera1": sample["observation.images.camera1"],
        "camera2": sample["observation.images.camera2"],
    }
