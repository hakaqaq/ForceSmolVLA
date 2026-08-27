#!/usr/bin/env python3
"""Inventory ConRFT ResNet-10 weights and freeze an allow_pickle=False copy.

Run only inside the pinned ``conrft_reward`` environment.  This is an asset
conversion utility: it does not construct a classifier TrainState, optimizer,
or checkpoint.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import pickle
import pickletools
import subprocess
import sys
import tempfile
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from flax.traverse_util import flatten_dict


ROOT = Path(__file__).resolve().parents[2]
CONRFT = Path("/home/rlc123/conrft")
COMPLETE_SHA256 = "175745d43d30233eb01b5369465d1c24c11b8ee71ccb734cc1c1bca13e07f57b"
SOURCE_URL = "https://github.com/rail-berkeley/serl/releases/download/resnet10/resnet10_params.pkl"
SOURCE_COMMIT = "a779fde7fa5db5a469960a8490c100f35b41b49e"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def array_sha256(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    return hashlib.sha256(contiguous.tobytes(order="C")).hexdigest()


def tree_content_sha(records: list[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for record in records:
        digest.update(json.dumps(record["parameter_path"], separators=(",", ":")).encode())
        digest.update(b"\0")
        digest.update(record["dtype"].encode())
        digest.update(b"\0")
        digest.update(json.dumps(record["shape"], separators=(",", ":")).encode())
        digest.update(b"\0")
        digest.update(record["array_sha256"].encode())
        digest.update(b"\n")
    return digest.hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False, encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
        temporary = Path(stream.name)
    temporary.replace(path)


def local_pickle_status(path: Path) -> dict[str, Any]:
    status = "complete_pickle"
    error = None
    try:
        for _ in pickletools.genops(path.read_bytes()):
            pass
    except Exception as exc:  # diagnostic only; never unpickle this repository copy
        status = "truncated_committed_pickle"
        error = f"{type(exc).__name__}: {exc}"
    return {
        "path": str(path),
        "file_size": path.stat().st_size,
        "sha256": sha256(path),
        "git_lfs_pointer": path.read_bytes()[:128].startswith(b"version https://git-lfs.github.com/spec/v1"),
        "classification": status,
        "pickletools_error": error,
    }


def git(*args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(CONRFT), *args], check=True, capture_output=True, text=True
    ).stdout.strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    pretrained_dir = ROOT / "artifacts/development/stage2/reward_classifier/pretrained"
    parser.add_argument("--pickle", type=Path, default=pretrained_dir / "resnet10_params.pkl")
    parser.add_argument("--safe-npz", type=Path, default=pretrained_dir / "resnet10_params.safe.npz")
    parser.add_argument(
        "--manifest", type=Path, default=pretrained_dir / "resnet10_asset_manifest.v4.json"
    )
    args = parser.parse_args()

    if os.environ.get("CONDA_DEFAULT_ENV") != "conrft_reward":
        raise RuntimeError("must run in isolated conrft_reward environment")
    if git("rev-parse", "HEAD") != SOURCE_COMMIT or git("status", "--porcelain"):
        raise RuntimeError("ConRFT fixed commit/clean-worktree prerequisite failed")
    if sha256(args.pickle) != COMPLETE_SHA256:
        raise RuntimeError("complete ResNet-10 asset SHA mismatch")

    sys.path.insert(0, str(CONRFT / "serl_launcher"))
    from serl_launcher.vision.resnet_v1 import resnetv1_configs

    with args.pickle.open("rb") as stream:
        encoder_params = pickle.load(stream)

    flat = flatten_dict(encoder_params)
    arrays: dict[str, np.ndarray] = {}
    records: list[dict[str, Any]] = []
    for index, (path, value) in enumerate(sorted(flat.items(), key=lambda item: item[0])):
        array = np.asarray(value)
        safe_key = f"array_{index:04d}"
        arrays[safe_key] = array
        records.append(
            {
                "safe_npz_key": safe_key,
                "parameter_path": list(path),
                "shape": list(array.shape),
                "dtype": str(array.dtype),
                "element_count": int(array.size),
                "byte_count": int(array.nbytes),
                "array_sha256": array_sha256(array),
            }
        )

    module = resnetv1_configs["resnetv1-10-frozen"](pre_pooling=True)
    expected = module.init(
        jax.random.PRNGKey(0), jnp.zeros((1, 128, 128, 3), dtype=jnp.uint8), train=False
    )["params"]
    expected_flat = flatten_dict(expected)
    asset_shapes = {path: tuple(np.asarray(value).shape) for path, value in flat.items()}
    expected_shapes = {path: tuple(np.asarray(value).shape) for path, value in expected_flat.items()}
    matched = sorted(path for path in expected_shapes if asset_shapes.get(path) == expected_shapes[path])
    missing = sorted(path for path in expected_shapes if path not in asset_shapes)
    unexpected = sorted(path for path in asset_shapes if path not in expected_shapes)
    shape_mismatches = sorted(
        path
        for path in expected_shapes.keys() & asset_shapes.keys()
        if expected_shapes[path] != asset_shapes[path]
    )
    # The release asset is the ImageNet checkpoint and intentionally includes
    # its 1000-way output head.  ConRFT's loader only copies keys present in the
    # frozen backbone, so the head is provenance-bound but not classifier input.
    allowed_unused = sorted(path for path in unexpected if path[0] == "output_head")
    forbidden_unexpected = sorted(set(unexpected) - set(allowed_unused))
    if missing or forbidden_unexpected or shape_mismatches:
        raise RuntimeError("ResNet-10 parameter key/shape coverage is incomplete")

    args.safe_npz.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("wb", dir=args.safe_npz.parent, suffix=".npz", delete=False) as stream:
        np.savez(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
        temporary_npz = Path(stream.name)
    temporary_npz.replace(args.safe_npz)

    verified: list[dict[str, Any]] = []
    with np.load(args.safe_npz, allow_pickle=False) as archive:
        if sorted(archive.files) != sorted(arrays):
            raise RuntimeError("safe NPZ key inventory mismatch")
        for record in records:
            recovered = archive[record["safe_npz_key"]]
            if (
                list(recovered.shape) != record["shape"]
                or str(recovered.dtype) != record["dtype"]
                or array_sha256(recovered) != record["array_sha256"]
            ):
                raise RuntimeError(f"safe NPZ array mismatch: {record['parameter_path']}")
            verified.append(record)

    birth_ns = args.pickle.stat().st_ctime_ns
    birth_iso = datetime.fromtimestamp(birth_ns / 1e9, tz=timezone.utc).isoformat()
    train_utils = CONRFT / "serl_launcher/serl_launcher/utils/train_utils.py"
    reward_classifier = CONRFT / "serl_launcher/serl_launcher/networks/reward_classifier.py"
    resnet_source = CONRFT / "serl_launcher/serl_launcher/vision/resnet_v1.py"
    manifest = {
        "schema_version": "force_rft_conrft_resnet10_asset.v4",
        "status": "PASS_FROZEN_SAFE_COPY_READY",
        "source_url": SOURCE_URL,
        "source_repository": git("remote", "get-url", "origin"),
        "source_commit": SOURCE_COMMIT,
        "download_timestamp": birth_iso,
        "download_timestamp_evidence": "frozen_asset_filesystem_ctime_ns",
        "unsafe_pickle_asset": {
            "relative_path": args.pickle.relative_to(ROOT).as_posix(),
            "file_size": args.pickle.stat().st_size,
            "sha256": sha256(args.pickle),
            "controlled_load_environment": "conrft_reward",
            "controlled_load_backend": jax.default_backend(),
        },
        "safe_asset": {
            "relative_path": args.safe_npz.relative_to(ROOT).as_posix(),
            "format": "numpy_npz_arrays_only",
            "load_contract": "numpy.load(..., allow_pickle=False)",
            "file_size": args.safe_npz.stat().st_size,
            "sha256": sha256(args.safe_npz),
            "conversion_semantics": "lossless array tree flattening; paths bound by parameter_inventory",
            "round_trip_verified": True,
        },
        "parameter_key_count": len(records),
        "parameter_element_count": sum(record["element_count"] for record in records),
        "parameter_byte_count": sum(record["byte_count"] for record in records),
        "parameter_shapes": {
            "/".join(record["parameter_path"]): {
                "shape": record["shape"], "dtype": record["dtype"]
            }
            for record in records
        },
        "parameter_inventory": records,
        "parameter_tree_content_sha256_before": tree_content_sha(records),
        "parameter_tree_content_sha256_after": tree_content_sha(verified),
        "expected_parameter_coverage": {
            "model": "ConRFT resnetv1-10-frozen",
            "expected_key_count": len(expected_shapes),
            "matched_key_and_shape_count": len(matched),
            "coverage_fraction": len(matched) / len(expected_shapes),
            "missing_paths": [list(path) for path in missing],
            "asset_only_paths": [list(path) for path in unexpected],
            "allowed_unused_imagenet_output_head_paths": [list(path) for path in allowed_unused],
            "forbidden_unexpected_paths": [list(path) for path in forbidden_unexpected],
            "shape_mismatches": [list(path) for path in shape_mismatches],
            "backbone_exact": True,
            "imagenet_output_head_consumed_by_conrft": False,
        },
        "loader_source": {
            "path": str(train_utils),
            "sha256": sha256(train_utils),
            "function": "load_resnet10_params",
            "classifier_loader_path": str(reward_classifier),
            "classifier_loader_sha256": sha256(reward_classifier),
            "resnet_definition_path": str(resnet_source),
            "resnet_definition_sha256": sha256(resnet_source),
        },
        "repository_copy": local_pickle_status(
            CONRFT / "examples/experiments/resnet10_params.pkl"
        ),
        "future_load_policy": "load frozen safe_asset only; mutable cache download forbidden",
        "conversion_tool": {
            "path": "tools/reward_classifier/freeze_conrft_resnet10_asset.py",
            "sha256": sha256(ROOT / "tools/reward_classifier/freeze_conrft_resnet10_asset.py"),
        },
        "classifier_checkpoint_created": False,
        "optimizer_created": False,
        "optimizer_update_count": 0,
    }
    if manifest["parameter_tree_content_sha256_before"] != manifest["parameter_tree_content_sha256_after"]:
        raise RuntimeError("safe conversion semantic checksum mismatch")
    atomic_json(args.manifest, manifest)
    print(
        json.dumps(
            {
                "manifest": str(args.manifest),
                "manifest_sha256": sha256(args.manifest),
                "safe_npz_sha256": sha256(args.safe_npz),
                "parameter_key_count": len(records),
                "coverage": 1.0,
            }
        )
    )


if __name__ == "__main__":
    main()
