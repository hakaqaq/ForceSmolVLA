#!/usr/bin/env python3
"""Fail-closed R0 development reward-classifier training and evidence.

``prepare-cache`` runs in the project environment (PyArrow/Pillow available),
reads only train/validation image rows, and creates an ephemeral native-resolution
RGB mmap. ``train`` runs only in the frozen conrft_reward environment on GPU.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import importlib.util
from io import BytesIO
import json
import os
from pathlib import Path
import pickle
import shutil
import subprocess
import sys
import tempfile
import types
from typing import Any, Iterable, Iterator, Mapping

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
CONRFT_ROOT = Path("/home/rlc123/conrft")
CONRFT_RUNTIME_ROOT = CONRFT_ROOT / "serl_launcher"
CONFIG_DEFAULT = ROOT / "configs/stage2_r0_reward_classifier_training.development.json"
DATASET_ROOT = ROOT / "datasets/task2_lerobotv3"
READINESS_PATH = ROOT / "artifacts/development/stage2/s2_r0_label_ingestion_readiness.v4.json"
INVENTORY_PATH = ROOT / "artifacts/development/stage2/reward_classifier/task2_frame_label_inventory.v2.json"
REVIEWED_PATH = ROOT / "labels/task2_reward_frame_labels.v2.reviewed.json"
SAFE_ASSET_PATH = ROOT / "artifacts/development/stage2/reward_classifier/pretrained/resnet10_params.safe.npz"
SAFE_MANIFEST_PATH = ROOT / "artifacts/development/stage2/reward_classifier/pretrained/resnet10_asset_manifest.v4.json"
ADAPTER_PATH = ROOT / "tools/reward_classifier/conrft_lerobot_v3_adapter.py"
SPLIT_PATH = DATASET_ROOT / "split_manifest.json"

EXPECTED_REVIEWED_SHA256 = "ecda7d480f6a4c49dbe63a31b7e3172b30a5470437510522b1da2217eae77a9c"
EXPECTED_READINESS_SHA256 = "64ae61e7d83c7be49451f4716c0e95921c2e9dbd062a553cec8f7fccdcc690aa"
EXPECTED_INVENTORY_SHA256 = "8839793f0e5d5c6d866b41e32bcb7fa576cd984a9faf5507719a1735be611a65"
EXPECTED_SAFE_ASSET_SHA256 = "16052142a3ef841a12fb1d2a03965951e8fbf0dda3d89b995244419be7e1f9a5"
EXPECTED_DATASET_STORAGE_SHA256 = "f9935b6479dc851e49444669065d20b8aef8cb3ad382f77f53391f701a55a58d"
EXPECTED_CONRFT_COMMIT = "a779fde7fa5db5a469960a8490c100f35b41b49e"

CAMERA_SOURCE_KEYS = ("observation.images.camera1", "observation.images.camera2")
CAMERA_CLASSIFIER_KEYS = ("d435_third_person", "d405_wrist")
CLASS_NAMES = ("positive", "ordinary_negative", "hard_negative", "ambiguous")
CLASS_CODE = {name: index for index, name in enumerate(CLASS_NAMES)}
TRAIN_COUNTS = {"positive": 128, "ordinary_negative": 64, "hard_negative": 64}
SEED = 0
OPTIMIZER_UPDATES = 150
VALIDATION_INTERVAL = 10
OVERFIT_UPDATES = 30
VALIDATION_BATCH_SIZE = 128
IMAGE_SHAPE = (480, 640, 3)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", dir=path.parent, delete=False, encoding="utf-8"
    ) as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        temporary = Path(stream.name)
    temporary.replace(path)


def atomic_bytes(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("wb", dir=path.parent, delete=False) as stream:
        stream.write(value)
        stream.flush()
        os.fsync(stream.fileno())
        temporary = Path(stream.name)
    temporary.replace(path)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def binding(path: Path) -> dict[str, Any]:
    return {
        "path": relative(path),
        "file_size": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(CONRFT_ROOT), *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def verify_frozen_inputs(config_path: Path, *, hash_dataset: bool) -> dict[str, Any]:
    expected = {
        REVIEWED_PATH: EXPECTED_REVIEWED_SHA256,
        READINESS_PATH: EXPECTED_READINESS_SHA256,
        INVENTORY_PATH: EXPECTED_INVENTORY_SHA256,
        SAFE_ASSET_PATH: EXPECTED_SAFE_ASSET_SHA256,
    }
    for path, digest in expected.items():
        require(path.is_file(), f"required input missing: {path}")
        require(sha256_file(path) == digest, f"frozen input SHA mismatch: {path}")

    config = load_json(config_path)
    require(config["schema_version"] == "forcesmolvla_r0_reward_classifier_training.v1", "config schema mismatch")
    require(config["optimizer"]["optimizer_updates"] == OPTIMIZER_UPDATES, "optimizer update count mismatch")
    require(config["optimizer"]["learning_rate"] == 1e-4, "learning rate mismatch")
    require(config["sampling"] == {
        "batch_size": 256,
        "hard_negative_per_update": 64,
        "ordinary_negative_per_update": 64,
        "positive_per_update": 128,
        "replacement": True,
        "scope": "train_split_only",
    }, "batch sampling contract mismatch")
    require(config["reproducibility"]["fixed_seed"] == SEED, "seed mismatch")
    require(config["training_augmentation"] == {
        "implementation": "ConRFT_batched_random_crop",
        "num_batch_dims": 2,
        "padding": 4,
    }, "augmentation contract mismatch")

    readiness = load_json(READINESS_PATH)
    inventory = load_json(INVENTORY_PATH)
    reviewed = load_json(REVIEWED_PATH)
    split_manifest = load_json(SPLIT_PATH)
    require(readiness["artifact_status"] == "PASS_DEVELOPMENT_R0_TRAINING_DATA_READY", "readiness did not pass")
    require(readiness["readiness"]["DEVELOPMENT_R0_TRAINING_DATA_READY"] == "yes", "training data is not ready")
    require(readiness["bindings"]["reviewed_labels"]["sha256"] == EXPECTED_REVIEWED_SHA256, "readiness label binding mismatch")
    require(readiness["bindings"]["frame_label_inventory"]["sha256"] == EXPECTED_INVENTORY_SHA256, "readiness inventory binding mismatch")
    require(inventory["artifact_status"] == "PASS_APPEND_ONLY_VALIDATED_LABEL_INGESTION", "inventory did not pass")
    require(inventory["validation"]["schema_valid"] is True, "label schema invalid")
    require(inventory["validation"]["intervals_valid"] is True, "label intervals invalid")
    require(inventory["validation"]["overlapping_frame_count"] == 0, "overlapping labels")
    require(inventory["validation"]["unlabeled_frame_count"] == 0, "unlabeled frames")
    require(inventory["leakage_checks"]["episode_leakage"] is False, "episode leakage")
    require(inventory["leakage_checks"]["row_leakage"] is False, "row leakage")
    require(len(reviewed["episodes"]) == 47, "reviewed episode count mismatch")
    require(all(e["manual_review_status"] == "human_reviewed" for e in reviewed["episodes"]), "not all episodes are human reviewed")

    split_sets = {name: set(split_manifest[name]) for name in ("train", "val", "test")}
    require([len(split_sets[name]) for name in ("train", "val", "test")] == [38, 5, 4], "split sizes mismatch")
    require(not (split_sets["train"] & split_sets["val"] | split_sets["train"] & split_sets["test"] | split_sets["val"] & split_sets["test"]), "split episode overlap")
    inv_sets = {
        "train": {e["episode_id"] for e in inventory["episodes"] if e["split"] == "train"},
        "val": {e["episode_id"] for e in inventory["episodes"] if e["split"] == "validation"},
        "test": {e["episode_id"] for e in inventory["episodes"] if e["split"] == "test"},
    }
    require(inv_sets == split_sets, "inventory/split-manifest mismatch")

    for entry in readiness["bindings"].values():
        if "path" not in entry:
            continue
        path = ROOT / entry["path"]
        require(path.is_file(), f"readiness binding missing: {path}")
        require(sha256_file(path) == entry["sha256"], f"readiness binding SHA mismatch: {path}")

    storage_sha = readiness["bindings"]["p8_dataset_storage"]["tree_sha256"]
    require(storage_sha == EXPECTED_DATASET_STORAGE_SHA256, "readiness dataset SHA mismatch")
    if hash_dataset:
        hashes: dict[str, str] = {}
        for directory in ("data", "videos", "meta"):
            for path in sorted((DATASET_ROOT / directory).rglob("*")):
                if path.is_file():
                    hashes[path.relative_to(DATASET_ROOT).as_posix()] = sha256_file(path)
        digest = hashlib.sha256()
        for name, value in hashes.items():
            digest.update(f"{name}\0{value}\n".encode())
        require(len(hashes) == 51, "dataset storage file count mismatch")
        require(digest.hexdigest() == EXPECTED_DATASET_STORAGE_SHA256, "dataset storage SHA mismatch")

    require(_git("rev-parse", "HEAD") == EXPECTED_CONRFT_COMMIT, "ConRFT commit mismatch")
    require(_git("status", "--porcelain") == "", "ConRFT worktree is modified")
    return {
        "config": config,
        "readiness": readiness,
        "inventory": inventory,
        "reviewed": reviewed,
        "split_manifest": split_manifest,
        "dataset_storage_sha256": storage_sha,
    }


def import_adapter():
    spec = importlib.util.spec_from_file_location("r0_conrft_lerobot_v3_adapter", ADAPTER_PATH)
    require(spec is not None and spec.loader is not None, "cannot import adapter")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def frame_pools(inventory: Mapping[str, Any], split: str) -> dict[str, list[tuple[str, int]]]:
    pools = {name: [] for name in CLASS_NAMES}
    for episode in inventory["episodes"]:
        if episode["split"] != split:
            continue
        for name in CLASS_NAMES:
            for start, stop in episode["class_intervals_inclusive"][name]:
                pools[name].extend((episode["episode_id"], frame) for frame in range(start, stop + 1))
    return pools


def build_schedule(inventory: Mapping[str, Any]) -> tuple[list[list[tuple[str, int, str]]], dict[str, Any]]:
    pools = frame_pools(inventory, "train")
    require({name: len(pools[name]) for name in CLASS_NAMES} == {
        "positive": 1712,
        "ordinary_negative": 19890,
        "hard_negative": 10025,
        "ambiguous": 197,
    }, "train pool inventory mismatch")
    rng = np.random.default_rng(SEED)
    schedule: list[list[tuple[str, int, str]]] = []
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    for _ in range(OPTIMIZER_UPDATES):
        batch: list[tuple[str, int, str]] = []
        for class_name, count in TRAIN_COUNTS.items():
            selected = rng.integers(0, len(pools[class_name]), size=count)
            batch.extend((*pools[class_name][int(index)], class_name) for index in selected)
        batch = [batch[int(index)] for index in rng.permutation(len(batch))]
        require(Counter(item[2] for item in batch) == Counter(TRAIN_COUNTS), "generated batch composition mismatch")
        require(all(item[2] != "ambiguous" for item in batch), "ambiguous frame scheduled")
        schedule.append(batch)
        for episode_id, _, class_name in batch:
            counts[episode_id][class_name] += 1
    stats = {
        episode_id: {name: int(counter[name]) for name in CLASS_NAMES}
        for episode_id, counter in sorted(counts.items())
    }
    return schedule, stats


def _decode_rgb(payload: bytes) -> np.ndarray:
    from PIL import Image

    with Image.open(BytesIO(payload)) as image:
        rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    require(rgb.shape == IMAGE_SHAPE, f"decoded image shape mismatch: {rgb.shape}")
    return np.ascontiguousarray(rgb)


def prepare_cache(cache_dir: Path, config_path: Path) -> None:
    import pyarrow.parquet as pq

    require(not cache_dir.exists(), f"cache directory already exists: {cache_dir}")
    verified = verify_frozen_inputs(config_path, hash_dataset=True)
    inventory = verified["inventory"]
    split_manifest = verified["split_manifest"]
    schedule, sampling_stats = build_schedule(inventory)

    train_required = {(episode_id, frame) for batch in schedule for episode_id, frame, _ in batch}
    val_pools = frame_pools(inventory, "validation")
    require(len(val_pools["ambiguous"]) == 0, "validation contains ambiguous frames")
    validation_rows = [
        (*row, class_name)
        for episode in inventory["episodes"]
        if episode["split"] == "validation"
        for class_name in ("ordinary_negative", "hard_negative", "positive")
        for row in [
            (episode["episode_id"], frame)
            for start, stop in episode["class_intervals_inclusive"][class_name]
            for frame in range(start, stop + 1)
        ]
    ]
    validation_rows.sort(key=lambda item: (item[0], item[1]))
    require(len(validation_rows) == 3775, "validation frame count mismatch")
    validation_keys = {(episode_id, frame) for episode_id, frame, _ in validation_rows}
    require(train_required.isdisjoint(validation_keys), "train/validation cache overlap")
    test_episodes = set(split_manifest["test"])
    require(not any(episode_id in test_episodes for episode_id, _ in train_required | validation_keys), "test row entered cache")

    train_rows = sorted(
        (episode_id, frame, next(name for name in CLASS_NAMES if any(
            start <= frame <= stop
            for episode in inventory["episodes"] if episode["episode_id"] == episode_id
            for start, stop in episode["class_intervals_inclusive"][name]
        )))
        for episode_id, frame in train_required
    )
    all_rows = train_rows + validation_rows
    key_to_cache = {(episode_id, frame): index for index, (episode_id, frame, _) in enumerate(all_rows)}
    schedule_indices = np.asarray(
        [[key_to_cache[(episode_id, frame)] for episode_id, frame, _ in batch] for batch in schedule],
        dtype=np.int32,
    )
    schedule_class_codes = np.asarray(
        [[CLASS_CODE[class_name] for _, _, class_name in batch] for batch in schedule],
        dtype=np.uint8,
    )
    require(schedule_indices.shape == (150, 256), "schedule shape mismatch")
    require(np.all(np.count_nonzero(schedule_class_codes == CLASS_CODE["positive"], axis=1) == 128), "positive batch count mismatch")
    require(np.all(np.count_nonzero(schedule_class_codes == CLASS_CODE["ordinary_negative"], axis=1) == 64), "ordinary batch count mismatch")
    require(np.all(np.count_nonzero(schedule_class_codes == CLASS_CODE["hard_negative"], axis=1) == 64), "hard batch count mismatch")
    require(not np.any(schedule_class_codes == CLASS_CODE["ambiguous"]), "ambiguous schedule consumption")

    staging = cache_dir.parent / f".{cache_dir.name}.tmp-{os.getpid()}"
    require(not staging.exists(), f"cache staging exists: {staging}")
    staging.mkdir(parents=True)
    try:
        camera1 = np.lib.format.open_memmap(staging / "camera1.npy", mode="w+", dtype=np.uint8, shape=(len(all_rows), *IMAGE_SHAPE))
        camera2 = np.lib.format.open_memmap(staging / "camera2.npy", mode="w+", dtype=np.uint8, shape=(len(all_rows), *IMAGE_SHAPE))
        adapter_module = import_adapter()
        adapter = adapter_module.ConRFTLeRobotV3Adapter()
        episodes = {episode["episode_id"]: episode for episode in inventory["episodes"]}
        rows_by_episode: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for cache_index, (episode_id, frame, _) in enumerate(all_rows):
            rows_by_episode[episode_id].append((frame, cache_index))

        source_rows_loaded = {"train": 0, "validation": 0, "test": 0}
        source_parquet_opened = {"train": [], "validation": [], "test": []}
        for episode_id in sorted(rows_by_episode):
            episode = episodes[episode_id]
            split = episode["split"]
            require(split in ("train", "validation"), "forbidden split requested during cache build")
            parquet_path = (DATASET_ROOT / episode["source_data_relative_path"]).resolve()
            require(parquet_path.is_relative_to(DATASET_ROOT.resolve()), "parquet path escape")
            table = pq.read_table(
                parquet_path,
                columns=[
                    *CAMERA_SOURCE_KEYS,
                    "frame_index",
                    "episode_index",
                    "timestamp",
                    "provenance.camera1_receive_monotonic_ns",
                    "provenance.camera2_receive_monotonic_ns",
                ],
            )
            require(table.num_rows == episode["frame_count"], "episode parquet row count mismatch")
            source_parquet_opened[split].append(episode["source_data_relative_path"])
            for frame, cache_index in rows_by_episode[episode_id]:
                row = table.slice(frame, 1).to_pylist()[0]
                require(row["frame_index"] == frame, "frame index mismatch")
                require(row["episode_index"] == episode["output_episode_index"], "episode index mismatch")
                rgb1 = _decode_rgb(row[CAMERA_SOURCE_KEYS[0]]["bytes"])
                rgb2 = _decode_rgb(row[CAMERA_SOURCE_KEYS[1]]["bytes"])
                adapted = adapter.adapt(
                    {
                        CAMERA_SOURCE_KEYS[0]: np.ascontiguousarray(np.transpose(rgb1, (2, 0, 1))),
                        CAMERA_SOURCE_KEYS[1]: np.ascontiguousarray(np.transpose(rgb2, (2, 0, 1))),
                    },
                    row_reference=adapter_module.RowReference(
                        "task2_lerobotv3",
                        episode["source_data_relative_path"],
                        frame,
                        episode_id,
                        frame,
                        float(row["timestamp"]),
                    ),
                    camera_row_identity=adapter_module.CameraRowIdentity(
                        int(row["provenance.camera1_receive_monotonic_ns"]),
                        int(row["provenance.camera2_receive_monotonic_ns"]),
                    ),
                )
                camera1[cache_index] = adapted.observation[CAMERA_CLASSIFIER_KEYS[0]][0, 0]
                camera2[cache_index] = adapted.observation[CAMERA_CLASSIFIER_KEYS[1]][0, 0]
                source_rows_loaded[split] += 1
            del table
        camera1.flush()
        camera2.flush()
        del camera1, camera2

        np.save(staging / "schedule_indices.npy", schedule_indices, allow_pickle=False)
        np.save(staging / "schedule_class_codes.npy", schedule_class_codes, allow_pickle=False)
        validation_indices = np.asarray(
            [key_to_cache[(episode_id, frame)] for episode_id, frame, _ in validation_rows], dtype=np.int32
        )
        validation_class_codes = np.asarray([CLASS_CODE[name] for _, _, name in validation_rows], dtype=np.uint8)
        np.save(staging / "validation_indices.npy", validation_indices, allow_pickle=False)
        np.save(staging / "validation_class_codes.npy", validation_class_codes, allow_pickle=False)
        cache_episode_ids = sorted({episode_id for episode_id, _, _ in all_rows})
        episode_code = {episode_id: index for index, episode_id in enumerate(cache_episode_ids)}
        np.save(staging / "cache_episode_codes.npy", np.asarray([episode_code[e] for e, _, _ in all_rows], dtype=np.uint8), allow_pickle=False)
        np.save(staging / "cache_frame_indices.npy", np.asarray([frame for _, frame, _ in all_rows], dtype=np.int32), allow_pickle=False)

        file_bindings = {}
        for name in (
            "camera1.npy", "camera2.npy", "schedule_indices.npy", "schedule_class_codes.npy",
            "validation_indices.npy", "validation_class_codes.npy", "cache_episode_codes.npy", "cache_frame_indices.npy",
        ):
            path = staging / name
            file_bindings[name] = {"file_size": path.stat().st_size, "sha256": sha256_file(path)}
        manifest = {
            "schema_version": "forcesmolvla_r0_ephemeral_native_rgb_cache.v1",
            "artifact_status": "COMPLETE_EPHEMERAL_TRAIN_VALIDATION_CACHE",
            "created_at": utc_now(),
            "cache_frame_count": len(all_rows),
            "native_rgb_shape": list(IMAGE_SHAPE),
            "train_unique_scheduled_frame_count": len(train_rows),
            "validation_complete_frame_count": len(validation_rows),
            "test_frame_count": 0,
            "ambiguous_frame_count": 0,
            "episode_ids": cache_episode_ids,
            "episode_reset_count": adapter.episode_reset_count,
            "schedule_sha256": canonical_sha(schedule),
            "baseline_sampling_by_episode_and_class": sampling_stats,
            "source_access_audit": {
                "image_rows_loaded": source_rows_loaded,
                "parquet_files_opened": source_parquet_opened,
                "test_parquet_files_opened": 0,
            },
            "frozen_bindings": {
                "config": binding(config_path),
                "inventory": binding(INVENTORY_PATH),
                "readiness": binding(READINESS_PATH),
                "reviewed_labels": binding(REVIEWED_PATH),
                "dataset_storage_sha256": EXPECTED_DATASET_STORAGE_SHA256,
                "adapter": binding(ADAPTER_PATH),
            },
            "files": file_bindings,
        }
        atomic_json(staging / "cache_manifest.json", manifest)
        staging.replace(cache_dir)
        print(json.dumps({
            "status": "cache_ready",
            "cache_dir": str(cache_dir),
            "train_unique_frames": len(train_rows),
            "validation_frames": len(validation_rows),
            "test_frames": 0,
            "ambiguous_frames": 0,
        }, sort_keys=True), flush=True)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def install_type_only_octo_shim() -> None:
    from typing import Any as TypingAny, Sequence as TypingSequence

    octo = types.ModuleType("octo")
    model = types.ModuleType("octo.model")
    octo_module = types.ModuleType("octo.model.octo_module")
    utils = types.ModuleType("octo.utils")
    typing_module = types.ModuleType("octo.utils.typing")

    class OctoTransformer:
        def __init__(self, *_: object, **__: object) -> None:
            raise RuntimeError("Octo instantiation is forbidden during R0 classifier training")

    octo_module.OctoTransformer = OctoTransformer
    typing_module.Config = TypingAny
    typing_module.Data = TypingAny
    typing_module.Params = TypingAny
    typing_module.PRNGKey = TypingAny
    typing_module.Sequence = TypingSequence
    octo.model = model
    octo.utils = utils
    model.octo_module = octo_module
    utils.typing = typing_module
    for name, module in {
        "octo": octo,
        "octo.model": model,
        "octo.model.octo_module": octo_module,
        "octo.utils": utils,
        "octo.utils.typing": typing_module,
    }.items():
        module.__dict__["__forcesmolvla_type_only_shim__"] = True
        sys.modules[name] = module


def verify_cache(cache_dir: Path) -> dict[str, Any]:
    manifest_path = cache_dir / "cache_manifest.json"
    require(manifest_path.is_file(), "cache manifest missing")
    manifest = load_json(manifest_path)
    require(manifest["artifact_status"] == "COMPLETE_EPHEMERAL_TRAIN_VALIDATION_CACHE", "cache incomplete")
    require(manifest["test_frame_count"] == 0 and manifest["ambiguous_frame_count"] == 0, "forbidden cache rows")
    for name, entry in manifest["files"].items():
        path = cache_dir / name
        require(path.stat().st_size == entry["file_size"], f"cache size mismatch: {name}")
        require(sha256_file(path) == entry["sha256"], f"cache SHA mismatch: {name}")
    return manifest


def npz_encoder_tree():
    from flax import traverse_util

    manifest = load_json(SAFE_MANIFEST_PATH)
    require(manifest["status"] == "PASS_FROZEN_SAFE_COPY_READY", "safe asset manifest status mismatch")
    require(manifest["safe_asset"]["sha256"] == EXPECTED_SAFE_ASSET_SHA256, "safe asset manifest SHA mismatch")
    flat: dict[tuple[str, ...], np.ndarray] = {}
    with np.load(SAFE_ASSET_PATH, allow_pickle=False) as archive:
        for record in manifest["parameter_inventory"]:
            value = archive[record["safe_npz_key"]]
            require(list(value.shape) == record["shape"] and str(value.dtype) == record["dtype"], "safe parameter metadata mismatch")
            require(hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest() == record["array_sha256"], "safe parameter SHA mismatch")
            flat[tuple(record["parameter_path"])] = value
    return traverse_util.unflatten_dict(flat), manifest


@contextmanager
def trusted_safe_npz_pickle_bridge(tree: Any) -> Iterator[Path]:
    """Give unmodified create_classifier() a trusted in-process NPZ-derived tree."""

    with tempfile.NamedTemporaryFile("wb", suffix=".trusted-safe-npz-bridge.pkl", delete=False) as stream:
        pickle.dump(tree, stream, protocol=4)
        path = Path(stream.name)
    try:
        yield path
    finally:
        path.unlink(missing_ok=True)


def flatten_named(tree: Any) -> list[tuple[tuple[str, ...], Any]]:
    from flax import traverse_util

    return sorted(traverse_util.flatten_dict(tree).items(), key=lambda item: item[0])


def tree_sha(tree: Any, predicate=lambda _: True) -> str:
    digest = hashlib.sha256()
    for path, leaf in flatten_named(tree):
        if not predicate(path):
            continue
        array = np.ascontiguousarray(np.asarray(leaf))
        digest.update("/".join(path).encode())
        digest.update(b"\0")
        digest.update(str(array.dtype).encode())
        digest.update(b"\0")
        digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode())
        digest.update(b"\0")
        digest.update(array.tobytes())
        digest.update(b"\n")
    return digest.hexdigest()


def leaves_norm(tree: Any, predicate=lambda _: True) -> float:
    total = 0.0
    for path, leaf in flatten_named(tree):
        if predicate(path):
            value = np.asarray(leaf, dtype=np.float64)
            total += float(np.sum(value * value))
    return float(np.sqrt(total))


def difference_norm(before: Any, after: Any, predicate=lambda _: True) -> float:
    before_flat = dict(flatten_named(before))
    after_flat = dict(flatten_named(after))
    require(before_flat.keys() == after_flat.keys(), "parameter tree keys changed")
    total = 0.0
    for path in before_flat:
        if predicate(path):
            delta = np.asarray(after_flat[path], dtype=np.float64) - np.asarray(before_flat[path], dtype=np.float64)
            total += float(np.sum(delta * delta))
    return float(np.sqrt(total))


def is_backbone(path: tuple[str, ...]) -> bool:
    return "pretrained_encoder" in path


def is_spatial(path: tuple[str, ...]) -> bool:
    return "SpatialLearnedEmbeddings_0" in path


def is_head(path: tuple[str, ...]) -> bool:
    return path[0] in ("Dense_0", "Dense_1", "LayerNorm_0")


def is_trainable(path: tuple[str, ...]) -> bool:
    return not is_backbone(path)


def sigmoid(values: np.ndarray) -> np.ndarray:
    positive = values >= 0
    result = np.empty_like(values, dtype=np.float64)
    result[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exp_values = np.exp(values[~positive])
    result[~positive] = exp_values / (1.0 + exp_values)
    return result


def roc_auc(labels: np.ndarray, probabilities: np.ndarray) -> float:
    order = np.argsort(probabilities, kind="mergesort")
    sorted_scores = probabilities[order]
    ranks = np.empty(len(labels), dtype=np.float64)
    start = 0
    while start < len(labels):
        stop = start + 1
        while stop < len(labels) and sorted_scores[stop] == sorted_scores[start]:
            stop += 1
        ranks[order[start:stop]] = (start + 1 + stop) / 2.0
        start = stop
    positives = labels == 1
    n_pos = int(np.sum(positives))
    n_neg = len(labels) - n_pos
    require(n_pos > 0 and n_neg > 0, "ROC-AUC requires both classes")
    return float((np.sum(ranks[positives]) - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def average_precision(labels: np.ndarray, probabilities: np.ndarray) -> float:
    order = np.argsort(-probabilities, kind="mergesort")
    y = labels[order]
    positives = int(np.sum(y))
    require(positives > 0, "PR-AUC requires positives")
    cumulative_tp = np.cumsum(y)
    precision = cumulative_tp / np.arange(1, len(y) + 1)
    return float(np.sum(precision[y == 1]) / positives)


def validation_metrics(logits: np.ndarray, labels: np.ndarray, strata: np.ndarray) -> dict[str, Any]:
    logits = np.asarray(logits, dtype=np.float64).reshape(-1)
    labels = np.asarray(labels, dtype=np.uint8).reshape(-1)
    strata = np.asarray(strata, dtype=np.uint8).reshape(-1)
    require(len(logits) == len(labels) == len(strata) == 3775, "validation row count mismatch")
    require(not np.any(strata == CLASS_CODE["ambiguous"]), "ambiguous validation metric consumption")
    losses = np.maximum(logits, 0.0) - logits * labels + np.log1p(np.exp(-np.abs(logits)))
    probabilities = sigmoid(logits)
    predictions = probabilities >= 0.5
    positive = labels == 1
    negative = ~positive
    tp = int(np.sum(predictions & positive))
    fn = int(np.sum(~predictions & positive))
    fp = int(np.sum(predictions & negative))
    tn = int(np.sum(~predictions & negative))
    recall = tp / (tp + fn)
    specificity = tn / (tn + fp)

    def stratum_fpr(name: str) -> float:
        mask = strata == CLASS_CODE[name]
        require(np.any(mask), f"validation stratum empty: {name}")
        return float(np.mean(predictions[mask]))

    return {
        "BCE": float(np.mean(losses)),
        "ROC_AUC": roc_auc(labels, probabilities),
        "PR_AUC": average_precision(labels, probabilities),
        "PR_AUC_definition": "average_precision_step_integral",
        "balanced_accuracy": float((recall + specificity) / 2.0),
        "positive_recall": float(recall),
        "overall_false_positive_rate": float(fp / (fp + tn)),
        "ordinary_negative_false_positive_rate": stratum_fpr("ordinary_negative"),
        "hard_negative_false_positive_rate": stratum_fpr("hard_negative"),
        "confusion_matrix": {"true_negative": tn, "false_positive": fp, "false_negative": fn, "true_positive": tp},
        "diagnostic_probability_threshold": 0.5,
        "detector_threshold_approved": False,
        "evaluated_frame_count": len(labels),
    }


def source_bindings(config_path: Path) -> dict[str, Any]:
    sources = {
        "training_source": Path(__file__).resolve(),
        "resolved_training_config": config_path,
        "adapter": ADAPTER_PATH,
        "readiness": READINESS_PATH,
        "inventory": INVENTORY_PATH,
        "reviewed_labels": REVIEWED_PATH,
        "safe_resnet10_npz": SAFE_ASSET_PATH,
        "safe_resnet10_manifest": SAFE_MANIFEST_PATH,
        "split_manifest": SPLIT_PATH,
        "conrft_reward_classifier": CONRFT_RUNTIME_ROOT / "serl_launcher/networks/reward_classifier.py",
        "conrft_resnet_v1": CONRFT_RUNTIME_ROOT / "serl_launcher/vision/resnet_v1.py",
        "conrft_encoding": CONRFT_RUNTIME_ROOT / "serl_launcher/common/encoding.py",
        "conrft_data_augmentations": CONRFT_RUNTIME_ROOT / "serl_launcher/vision/data_augmentations.py",
    }
    result = {}
    for name, path in sources.items():
        result[name] = {
            "path": relative(path) if path.resolve().is_relative_to(ROOT.resolve()) else str(path),
            "file_size": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    return result


def run_training(cache_dir: Path, output_dir: Path, config_path: Path) -> None:
    require(os.environ.get("CONDA_DEFAULT_ENV") == "conrft_reward", "training must run in conrft_reward")
    require(not output_dir.exists(), f"output directory already exists: {output_dir}")
    verified = verify_frozen_inputs(config_path, hash_dataset=False)
    cache_manifest = verify_cache(cache_dir)
    require(cache_manifest["frozen_bindings"]["config"]["sha256"] == sha256_file(config_path), "cache/config mismatch")
    require(cache_manifest["frozen_bindings"]["dataset_storage_sha256"] == EXPECTED_DATASET_STORAGE_SHA256, "cache/dataset mismatch")

    os.environ.setdefault("TF_CUDNN_DETERMINISTIC", "1")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    install_type_only_octo_shim()
    sys.path.insert(0, str(CONRFT_RUNTIME_ROOT))

    import flax
    from flax import serialization
    import jax
    import jax.numpy as jnp
    import jaxlib
    import optax
    from serl_launcher.networks.reward_classifier import create_classifier
    from serl_launcher.vision.data_augmentations import batched_random_crop

    require(jax.default_backend() == "gpu", f"real GPU required, got {jax.default_backend()}")
    require(jax.__version__ == "0.4.20" and jaxlib.__version__.startswith("0.4.20"), "JAX version drift")
    require(flax.__version__ == "0.8.0" and optax.__version__ == "0.1.5", "Flax/Optax version drift")

    safe_tree, safe_manifest = npz_encoder_tree()
    camera1 = np.load(cache_dir / "camera1.npy", mmap_mode="r", allow_pickle=False)
    camera2 = np.load(cache_dir / "camera2.npy", mmap_mode="r", allow_pickle=False)
    schedule = np.load(cache_dir / "schedule_indices.npy", allow_pickle=False)
    schedule_codes = np.load(cache_dir / "schedule_class_codes.npy", allow_pickle=False)
    val_indices = np.load(cache_dir / "validation_indices.npy", allow_pickle=False)
    val_codes = np.load(cache_dir / "validation_class_codes.npy", allow_pickle=False)
    require(camera1.shape == camera2.shape == (cache_manifest["cache_frame_count"], *IMAGE_SHAPE), "cache camera shape mismatch")
    require(schedule.shape == (150, 256), "schedule shape mismatch")
    require(len(val_indices) == 3775, "validation cache mismatch")
    val_labels = (val_codes == CLASS_CODE["positive"]).astype(np.uint8)

    sample = {
        CAMERA_CLASSIFIER_KEYS[0]: jnp.asarray(camera1[0:1, None]),
        CAMERA_CLASSIFIER_KEYS[1]: jnp.asarray(camera2[0:1, None]),
    }

    def new_classifier():
        rng = jax.random.PRNGKey(SEED)
        rng, initialization_key = jax.random.split(rng)
        with trusted_safe_npz_pickle_bridge(safe_tree) as bridge:
            state = create_classifier(
                initialization_key,
                sample,
                list(CAMERA_CLASSIFIER_KEYS),
                pretrained_encoder_path=str(bridge),
                n_way=2,
            )
        require(int(state.step) == 0, "classifier initial step mismatch")
        # ConRFT passes one named pretrained_encoder module instance to both
        # camera encoders. Flax therefore stores one shared parameter subtree
        # (under the d405 scope in this frozen source), not two copies.
        backbone_owners = [
            camera_key
            for camera_key in CAMERA_CLASSIFIER_KEYS
            if "pretrained_encoder" in state.params["encoder_def"][f"encoder_{camera_key}"]
        ]
        require(backbone_owners == ["d405_wrist"], "ConRFT shared backbone scope drift")
        model_tree = state.params["encoder_def"]["encoder_d405_wrist"]["pretrained_encoder"]
        for path, expected in flatten_named(safe_tree):
            if path[0] == "output_head":
                continue
            cursor = model_tree
            for component in path:
                cursor = cursor[component]
            require(np.array_equal(np.asarray(cursor), np.asarray(expected)), f"safe backbone mismatch: {'/'.join(path)}")
        return state, rng

    def host_batch(indices: np.ndarray) -> tuple[dict[str, Any], Any]:
        indices = np.asarray(indices, dtype=np.int64)
        obs = {
            CAMERA_CLASSIFIER_KEYS[0]: jnp.asarray(np.asarray(camera1[indices])[:, None]),
            CAMERA_CLASSIFIER_KEYS[1]: jnp.asarray(np.asarray(camera2[indices])[:, None]),
        }
        labels = jnp.asarray((np.asarray([CLASS_CODE["positive"] == code for code in schedule_codes_current], dtype=np.float32))[:, None])
        return obs, labels

    @jax.jit
    def update_step(state, observations, labels, augmentation_key, dropout_key):
        augmented = {
            key: batched_random_crop(value, augmentation_key, padding=4, num_batch_dims=2)
            for key, value in observations.items()
        }

        def loss_fn(params):
            logits = state.apply_fn(
                {"params": params}, augmented, rngs={"dropout": dropout_key}, train=True
            )
            return optax.sigmoid_binary_cross_entropy(logits, labels).mean()

        loss, grads = jax.value_and_grad(loss_fn)(state.params)
        return state.apply_gradients(grads=grads), loss

    @jax.jit
    def evidence_update_step(state, observations, labels, augmentation_key, dropout_key):
        augmented = {
            key: batched_random_crop(value, augmentation_key, padding=4, num_batch_dims=2)
            for key, value in observations.items()
        }

        def loss_fn(params):
            logits = state.apply_fn(
                {"params": params}, augmented, rngs={"dropout": dropout_key}, train=True
            )
            return optax.sigmoid_binary_cross_entropy(logits, labels).mean()

        loss, grads = jax.value_and_grad(loss_fn)(state.params)
        return state.apply_gradients(grads=grads), loss, grads

    @jax.jit
    def inference(params, observations):
        return create_apply_fn({"params": params}, observations, train=False)

    # Bind the unmodified BinaryClassifier apply function once for JIT inference.
    initial_for_apply, _ = new_classifier()
    create_apply_fn = initial_for_apply.apply_fn

    def get_batch(update_index: int) -> tuple[dict[str, Any], Any]:
        nonlocal schedule_codes_current
        schedule_codes_current = schedule_codes[update_index]
        return host_batch(schedule[update_index])

    def evaluate(state) -> dict[str, Any]:
        logits: list[np.ndarray] = []
        for start in range(0, len(val_indices), VALIDATION_BATCH_SIZE):
            selected = val_indices[start : start + VALIDATION_BATCH_SIZE]
            obs = {
                CAMERA_CLASSIFIER_KEYS[0]: jnp.asarray(np.asarray(camera1[selected])[:, None]),
                CAMERA_CLASSIFIER_KEYS[1]: jnp.asarray(np.asarray(camera2[selected])[:, None]),
            }
            value = inference(state.params, obs)
            logits.append(np.asarray(jax.block_until_ready(value)).reshape(-1))
        return validation_metrics(np.concatenate(logits), val_labels, val_codes)

    def deterministic_batch_bce(state, update_index: int) -> float:
        obs, labels = get_batch(update_index)
        logits = inference(state.params, obs)
        logits_np = np.asarray(jax.block_until_ready(logits), dtype=np.float64).reshape(-1)
        labels_np = np.asarray(labels, dtype=np.float64).reshape(-1)
        return float(np.mean(np.maximum(logits_np, 0.0) - logits_np * labels_np + np.log1p(np.exp(-np.abs(logits_np)))))

    # Real-data, real-GPU optimizer smoke: exactly one update.
    schedule_codes_current = schedule_codes[0]
    smoke_state, smoke_rng = new_classifier()
    smoke_before = smoke_state.params
    backbone_sha_before = tree_sha(smoke_before, is_backbone)
    smoke_obs, smoke_labels = get_batch(0)
    smoke_rng, aug_key = jax.random.split(smoke_rng)
    smoke_rng, dropout_key = jax.random.split(smoke_rng)
    smoke_state, smoke_loss, smoke_grads = evidence_update_step(
        smoke_state, smoke_obs, smoke_labels, aug_key, dropout_key
    )
    jax.block_until_ready(smoke_loss)
    backbone_sha_after = tree_sha(smoke_state.params, is_backbone)
    smoke_evidence = {
        "status": "pass",
        "real_gpu": True,
        "optimizer_updates": 1,
        "forward_backward_optimizer_updates_per_batch": 1,
        "loss": float(smoke_loss),
        "train_state_step_before": 0,
        "train_state_step_after": int(smoke_state.step),
        "backbone_parameter_sha256_before": backbone_sha_before,
        "backbone_parameter_sha256_after": backbone_sha_after,
        "backbone_sha_exactly_unchanged": backbone_sha_before == backbone_sha_after,
        "backbone_gradient_norm": leaves_norm(smoke_grads, is_backbone),
        "spatial_embedding_gradient_norm": leaves_norm(smoke_grads, is_spatial),
        "spatial_embedding_update_norm": difference_norm(smoke_before, smoke_state.params, is_spatial),
        "classifier_head_gradient_norm": leaves_norm(smoke_grads, is_head),
        "classifier_head_update_norm": difference_norm(smoke_before, smoke_state.params, is_head),
        "all_trainable_gradient_norm": leaves_norm(smoke_grads, is_trainable),
        "all_trainable_update_norm": difference_norm(smoke_before, smoke_state.params, is_trainable),
        "batch_composition": TRAIN_COUNTS,
        "ambiguous_consumed": 0,
    }
    require(smoke_evidence["train_state_step_after"] == 1, "smoke update count failed")
    require(smoke_evidence["backbone_sha_exactly_unchanged"], "backbone changed in smoke")
    require(smoke_evidence["backbone_gradient_norm"] == 0.0, "backbone gradient is nonzero")
    require(smoke_evidence["spatial_embedding_gradient_norm"] > 0.0 and smoke_evidence["spatial_embedding_update_norm"] > 0.0, "spatial embedding did not train")
    require(smoke_evidence["classifier_head_gradient_norm"] > 0.0 and smoke_evidence["classifier_head_update_norm"] > 0.0, "classifier head did not train")
    print(json.dumps({"phase": "gpu_optimizer_smoke", "status": "pass", "loss": float(smoke_loss)}), flush=True)

    # Fixed real minibatch overfit diagnostic, independent of the baseline.
    overfit_state, overfit_rng = new_classifier()
    overfit_backbone_before = tree_sha(overfit_state.params, is_backbone)
    overfit_eval_before = deterministic_batch_bce(overfit_state, 0)
    overfit_losses = []
    fixed_obs, fixed_labels = get_batch(0)
    for _ in range(OVERFIT_UPDATES):
        overfit_rng, aug_key = jax.random.split(overfit_rng)
        overfit_rng, dropout_key = jax.random.split(overfit_rng)
        overfit_state, loss = update_step(overfit_state, fixed_obs, fixed_labels, aug_key, dropout_key)
        overfit_losses.append(float(jax.block_until_ready(loss)))
    overfit_eval_after = deterministic_batch_bce(overfit_state, 0)
    overfit_backbone_after = tree_sha(overfit_state.params, is_backbone)
    overfit_evidence = {
        "status": "pass",
        "fixed_minibatch": True,
        "optimizer_updates": OVERFIT_UPDATES,
        "batch_composition": TRAIN_COUNTS,
        "deterministic_BCE_before": overfit_eval_before,
        "deterministic_BCE_after": overfit_eval_after,
        "deterministic_BCE_decreased": overfit_eval_after < overfit_eval_before,
        "train_losses": overfit_losses,
        "backbone_parameter_sha256_before": overfit_backbone_before,
        "backbone_parameter_sha256_after": overfit_backbone_after,
        "backbone_sha_exactly_unchanged": overfit_backbone_before == overfit_backbone_after,
        "ambiguous_consumed": 0,
    }
    require(overfit_evidence["deterministic_BCE_decreased"], "fixed minibatch overfit did not reduce BCE")
    require(overfit_evidence["backbone_sha_exactly_unchanged"], "backbone changed in overfit")
    print(json.dumps({"phase": "fixed_minibatch_overfit", "status": "pass", "BCE_before": overfit_eval_before, "BCE_after": overfit_eval_after}), flush=True)

    def baseline(run_name: str, save_primary: bool, checkpoint_dir: Path | None = None) -> tuple[dict[str, Any], bytes, bytes, bytes]:
        state, rng = new_classifier()
        initial_backbone_sha = tree_sha(state.params, is_backbone)
        initial_trainable_sha = tree_sha(state.params, is_trainable)
        trace = []
        train_losses = []
        best_state_bytes: bytes | None = None
        best_metrics: dict[str, Any] | None = None
        best_update: int | None = None
        for update in range(1, OPTIMIZER_UPDATES + 1):
            obs, labels = get_batch(update - 1)
            rng, aug_key = jax.random.split(rng)
            rng, dropout_key = jax.random.split(rng)
            state, loss = update_step(state, obs, labels, aug_key, dropout_key)
            loss_value = float(jax.block_until_ready(loss))
            train_losses.append(loss_value)
            require(int(state.step) == update, "one-update-per-batch contract failed")
            if update % VALIDATION_INTERVAL == 0:
                metrics = evaluate(state)
                trace.append({"optimizer_update": update, **metrics})
                better = best_metrics is None or metrics["BCE"] < best_metrics["BCE"] or (
                    metrics["BCE"] == best_metrics["BCE"] and metrics["PR_AUC"] > best_metrics["PR_AUC"]
                )
                if better:
                    best_metrics = metrics
                    best_update = update
                    best_state_bytes = serialization.to_bytes(state)
                print(json.dumps({"phase": run_name, "optimizer_update": update, "validation_BCE": metrics["BCE"], "validation_PR_AUC": metrics["PR_AUC"]}), flush=True)
        require(best_state_bytes is not None and best_metrics is not None and best_update is not None, "best checkpoint not selected")
        final_backbone_sha = tree_sha(state.params, is_backbone)
        final_trainable_sha = tree_sha(state.params, is_trainable)
        last_state_bytes = serialization.to_bytes(state)
        resume_payload = {
            "train_state": serialization.to_state_dict(state),
            "jax_rng": np.asarray(rng),
            "next_optimizer_update": OPTIMIZER_UPDATES + 1,
            "completed_optimizer_updates": OPTIMIZER_UPDATES,
            "schedule_cursor": OPTIMIZER_UPDATES,
            "schedule_sha256": cache_manifest["schedule_sha256"],
            "resolved_config_sha256": sha256_file(config_path),
            "seed": SEED,
        }
        resume_bytes = serialization.msgpack_serialize(resume_payload)
        if save_primary:
            require(checkpoint_dir is not None, "primary checkpoint directory missing")
            atomic_bytes(checkpoint_dir / "best_checkpoint.msgpack", best_state_bytes)
            atomic_bytes(checkpoint_dir / "last_checkpoint.msgpack", last_state_bytes)
            atomic_bytes(checkpoint_dir / "resume_state.msgpack", resume_bytes)
        result = {
            "run_name": run_name,
            "seed": SEED,
            "optimizer_updates": OPTIMIZER_UPDATES,
            "optimizer_updates_are_epochs": False,
            "train_loss_by_update": train_losses,
            "validation_trace": trace,
            "validation_evaluation_updates": list(range(10, 151, 10)),
            "full_validation_frames_per_evaluation": 3775,
            "best_optimizer_update": best_update,
            "best_validation_metrics": best_metrics,
            "selection_rule": "minimum_BCE_then_maximum_PR_AUC_on_exact_BCE_tie",
            "initial_backbone_parameter_sha256": initial_backbone_sha,
            "final_backbone_parameter_sha256": final_backbone_sha,
            "backbone_sha_exactly_unchanged": initial_backbone_sha == final_backbone_sha,
            "initial_trainable_parameter_sha256": initial_trainable_sha,
            "final_trainable_parameter_sha256": final_trainable_sha,
            "trainable_parameter_sha_changed": initial_trainable_sha != final_trainable_sha,
            "best_checkpoint_sha256": hashlib.sha256(best_state_bytes).hexdigest(),
            "last_checkpoint_sha256": hashlib.sha256(last_state_bytes).hexdigest(),
            "resume_state_sha256": hashlib.sha256(resume_bytes).hexdigest(),
            "ambiguous_consumed": 0,
            "test_evaluated": False,
        }
        require(result["backbone_sha_exactly_unchanged"], f"backbone changed in {run_name}")
        require(result["trainable_parameter_sha_changed"], f"trainable parameters unchanged in {run_name}")
        return result, best_state_bytes, last_state_bytes, resume_bytes

    staging = output_dir.parent / f".{output_dir.name}.tmp-{os.getpid()}"
    require(not staging.exists(), f"output staging exists: {staging}")
    checkpoint_dir = staging / "checkpoints" / "best"
    checkpoint_dir.mkdir(parents=True)
    try:
        primary, primary_best, primary_last, primary_resume = baseline("seed0_primary", True, checkpoint_dir)
        repeat, repeat_best, repeat_last, repeat_resume = baseline("seed0_repeat", False)
        reproducibility = {
            "seed": SEED,
            "full_150_update_repeat_executed": True,
            "sampling_schedule_exactly_reused": True,
            "validation_trace_exact_match": primary["validation_trace"] == repeat["validation_trace"],
            "train_loss_trace_exact_match": primary["train_loss_by_update"] == repeat["train_loss_by_update"],
            "best_checkpoint_bytes_exact_match": primary_best == repeat_best,
            "last_checkpoint_bytes_exact_match": primary_last == repeat_last,
            "resume_state_bytes_exact_match": primary_resume == repeat_resume,
            "primary": {
                "best_checkpoint_sha256": primary["best_checkpoint_sha256"],
                "last_checkpoint_sha256": primary["last_checkpoint_sha256"],
                "resume_state_sha256": primary["resume_state_sha256"],
            },
            "repeat": {
                "best_checkpoint_sha256": repeat["best_checkpoint_sha256"],
                "last_checkpoint_sha256": repeat["last_checkpoint_sha256"],
                "resume_state_sha256": repeat["resume_state_sha256"],
            },
        }
        reproducibility["exact_reproducibility_pass"] = all(
            reproducibility[key]
            for key in (
                "validation_trace_exact_match", "train_loss_trace_exact_match",
                "best_checkpoint_bytes_exact_match", "last_checkpoint_bytes_exact_match",
                "resume_state_bytes_exact_match",
            )
        )

        checkpoint_bindings = {
            name: {
                "path": f"{relative(output_dir)}/checkpoints/best/{name}",
                "file_size": (checkpoint_dir / name).stat().st_size,
                "sha256": sha256_file(checkpoint_dir / name),
            }
            for name in ("best_checkpoint.msgpack", "last_checkpoint.msgpack", "resume_state.msgpack")
        }
        restored_best = serialization.from_bytes(
            initial_for_apply, (checkpoint_dir / "best_checkpoint.msgpack").read_bytes()
        )
        restored_last = serialization.from_bytes(
            initial_for_apply, (checkpoint_dir / "last_checkpoint.msgpack").read_bytes()
        )
        restored_resume_payload = serialization.msgpack_restore(
            (checkpoint_dir / "resume_state.msgpack").read_bytes()
        )
        restored_resume_state = serialization.from_state_dict(
            initial_for_apply, restored_resume_payload["train_state"]
        )
        last_leaves, last_tree = jax.tree_util.tree_flatten(restored_last)
        resume_leaves, resume_tree = jax.tree_util.tree_flatten(restored_resume_state)
        checkpoint_restore_audit = {
            "status": "pass",
            "best_checkpoint_restored_step": int(restored_best.step),
            "last_checkpoint_restored_step": int(restored_last.step),
            "resume_state_restored_step": int(restored_resume_state.step),
            "resume_next_optimizer_update": int(restored_resume_payload["next_optimizer_update"]),
            "resume_completed_optimizer_updates": int(restored_resume_payload["completed_optimizer_updates"]),
            "resume_schedule_cursor": int(restored_resume_payload["schedule_cursor"]),
            "resume_rng_shape": list(restored_resume_payload["jax_rng"].shape),
            "last_and_resume_train_state_tree_exact": last_tree == resume_tree and all(
                np.array_equal(np.asarray(left), np.asarray(right))
                for left, right in zip(last_leaves, resume_leaves)
            ),
        }
        require(
            checkpoint_restore_audit["best_checkpoint_restored_step"] == OPTIMIZER_UPDATES
            and checkpoint_restore_audit["last_checkpoint_restored_step"] == OPTIMIZER_UPDATES
            and checkpoint_restore_audit["resume_state_restored_step"] == OPTIMIZER_UPDATES
            and checkpoint_restore_audit["resume_next_optimizer_update"] == OPTIMIZER_UPDATES + 1
            and checkpoint_restore_audit["last_and_resume_train_state_tree_exact"],
            "checkpoint/resume restore audit failed",
        )
        access = cache_manifest["source_access_audit"]
        report = {
            "schema_version": "forcesmolvla_r0_reward_classifier_training_report.v1",
            "artifact_status": "PASS_R0_DEVELOPMENT_CLASSIFIER_TRAINING_COMPLETE",
            "completed_at": utc_now(),
            "authorization": "user_approved_R0_development_reward_classifier_training",
            "resolved_training_config": binding(config_path),
            "frozen_input_bindings": {
                "reviewed_labels": binding(REVIEWED_PATH),
                "readiness": binding(READINESS_PATH),
                "inventory": binding(INVENTORY_PATH),
                "dataset_storage_tree_sha256": EXPECTED_DATASET_STORAGE_SHA256,
                "safe_resnet10_npz": binding(SAFE_ASSET_PATH),
                "safe_resnet10_manifest": binding(SAFE_MANIFEST_PATH),
                "ConRFT_git_commit": EXPECTED_CONRFT_COMMIT,
                "ConRFT_worktree_clean": True,
            },
            "runtime": {
                "environment": os.environ["CONDA_DEFAULT_ENV"],
                "python": sys.version.split()[0],
                "jax": jax.__version__,
                "jaxlib": jaxlib.__version__,
                "flax": flax.__version__,
                "optax": optax.__version__,
                "numpy": np.__version__,
                "backend": jax.default_backend(),
                "device": str(jax.devices()[0]),
                "determinism_environment": {
                    "TF_CUDNN_DETERMINISTIC": os.environ["TF_CUDNN_DETERMINISTIC"],
                    "CUBLAS_WORKSPACE_CONFIG": os.environ["CUBLAS_WORKSPACE_CONFIG"],
                },
            },
            "implementation": {
                "classifier": "ConRFT BinaryClassifier via unmodified create_classifier()",
                "safe_asset_loading": "allow_pickle=False NPZ reconstructed in memory; temporary trusted pickle bridge only because frozen create_classifier() accepts a pickle path",
                "safe_asset_output_head_consumed": False,
                "pretrained_backbone_frozen_by": "ConRFT resnetv1-10-frozen pre-pooling stop_gradient",
                "shared_backbone_parameter_scope": "encoder_def/encoder_d405_wrist/pretrained_encoder shared by both camera encoder calls in frozen ConRFT source",
                "objective": "optax.sigmoid_binary_cross_entropy",
                "optimizer": "Adam learning_rate=1e-4",
                "train_random_crop": "native ConRFT batched_random_crop padding=4 num_batch_dims=2",
                "validation_random_augmentation": False,
                "batch_size": 256,
                "gradient_accumulation": False,
                "one_forward_backward_optimizer_update_per_training_batch": True,
                "frame_stack": 1,
                "sampling_relationship": {
                    "ConRFT_compatible": "50_percent_positive_and_50_percent_negative",
                    "ForceRFT_development_extension": "negative_half_split_64_ordinary_and_64_hard",
                    "unmodified_ConRFT_sampling_claim": False,
                },
            },
            "gpu_optimizer_smoke": smoke_evidence,
            "fixed_minibatch_overfit": overfit_evidence,
            "primary_training_run": primary,
            "fixed_seed_reproducibility": reproducibility,
            "checkpoint_artifacts": checkpoint_bindings,
            "checkpoint_restore_audit": checkpoint_restore_audit,
            "actual_sampling": {
                "primary_baseline_total_occurrences": {
                    "positive": 150 * 128,
                    "ordinary_negative": 150 * 64,
                    "hard_negative": 150 * 64,
                    "ambiguous": 0,
                },
                "repeat_baseline_total_occurrences": {
                    "positive": 150 * 128,
                    "ordinary_negative": 150 * 64,
                    "hard_negative": 150 * 64,
                    "ambiguous": 0,
                },
                "primary_and_repeat_per_episode": cache_manifest["baseline_sampling_by_episode_and_class"],
                "unique_train_frames_materialized": cache_manifest["train_unique_scheduled_frame_count"],
                "ambiguous_zero_consumption": True,
            },
            "split_access_audit": {
                "source_image_rows_loaded_for_ephemeral_cache": access["image_rows_loaded"],
                "source_parquet_files_opened": access["parquet_files_opened"],
                "train_gradient_row_occurrences": {
                    "smoke": 256,
                    "fixed_minibatch_overfit": OVERFIT_UPDATES * 256,
                    "primary_baseline": OPTIMIZER_UPDATES * 256,
                    "fixed_seed_repeat": OPTIMIZER_UPDATES * 256,
                },
                "validation_inference_row_occurrences": {
                    "primary_baseline": 15 * 3775,
                    "fixed_seed_repeat": 15 * 3775,
                },
                "validation_gradient_row_occurrences": 0,
                "test_image_rows_loaded": 0,
                "test_storage_files_read_only_as_opaque_bytes_for_required_P8_tree_SHA_verification": True,
                "test_parquet_rows_or_schemas_loaded": 0,
                "test_images_decoded": 0,
                "test_inference_row_occurrences": 0,
                "test_checkpoint_selection_participation": False,
                "test_inventory_counts_read_as_metadata_only": verified["inventory"]["class_statistics"]["test"],
                "episode_leakage": "none",
                "row_leakage": "none",
            },
            "label_semantics": {
                "positive": 1,
                "ordinary_negative": 0,
                "hard_negative": 0,
                "ambiguous": "fully_excluded",
                "binary_classes_only": True,
            },
            "validation_threshold_disposition": {
                "threshold_0_5_metrics_are_training_diagnostics_only": True,
                "detector_threshold_approved": False,
                "calibration_executed": False,
            },
            "next_validation_only_recommendation": "Request approval for validation-only DetectorSpec calibration of probability threshold and consecutive-positive frames; keep test images sealed and do not generate reward/terminal during that calibration request.",
            "terminal_status": {
                "R0_CLASSIFIER_TRAINING": "complete",
                "BEST_CLASSIFIER_SELECTED": "yes",
                "DETECTOR_THRESHOLD_APPROVED": "no",
                "TEST_EVALUATED": "no",
                "TASK2_REWARD_TERMINAL_CREATED": "no",
                "REWARD_TRANSITION_CREATED": "no",
                "TWIN_Q_CREATED": "no",
                "NEXT_ALLOWED_ACTION": "request_validation_only_detector_calibration",
            },
        }
        report_path = staging / "r0_training_validation_report.v1.json"
        atomic_json(report_path, report)

        sources = source_bindings(config_path)
        artifacts = {"training_report": {
            "path": f"{relative(output_dir)}/r0_training_validation_report.v1.json",
            "file_size": report_path.stat().st_size,
            "sha256": sha256_file(report_path),
        }, **{f"checkpoint_{name.removesuffix('.msgpack')}": value for name, value in checkpoint_bindings.items()}}
        manifest = {
            "schema_version": "forcesmolvla_r0_reward_classifier_source_artifact_manifest.v1",
            "artifact_status": "PASS_BOUND_SOURCE_AND_TRAINING_ARTIFACTS",
            "created_at": utc_now(),
            "self_included": False,
            "sources": sources,
            "artifacts": artifacts,
            "dataset_storage_tree_sha256": EXPECTED_DATASET_STORAGE_SHA256,
            "ConRFT_git_commit": EXPECTED_CONRFT_COMMIT,
            "prohibited_artifacts_created": [],
        }
        atomic_json(staging / "source_artifact_manifest.v1.json", manifest)
        staging.replace(output_dir)
        print(json.dumps({
            "status": "training_complete",
            "output_dir": str(output_dir),
            "best_update": primary["best_optimizer_update"],
            "best_validation_BCE": primary["best_validation_metrics"]["BCE"],
            "best_validation_PR_AUC": primary["best_validation_metrics"]["PR_AUC"],
            "exact_reproducibility_pass": reproducibility["exact_reproducibility_pass"],
        }, sort_keys=True), flush=True)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=CONFIG_DEFAULT)
    parser.add_argument("--task-id", default="task2")
    parser.add_argument("--output-root", type=Path)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare-cache")
    prepare.add_argument("--cache-dir", type=Path, required=True)
    train = subparsers.add_parser("train")
    train.add_argument("--cache-dir", type=Path, required=True)
    train.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    if args.command == "prepare-cache":
        prepare_cache(args.cache_dir.resolve(), config_path)
    else:
        from forcesmolvla.training_runtime import resolve_task_output_root

        root = Path(__file__).resolve().parents[2]
        output_dir = (
            args.output_dir.resolve()
            if args.output_dir is not None
            else resolve_task_output_root(
                root, task_id=args.task_id, output_root=args.output_root
            )
            / "reward_classifier"
        )
        run_training(args.cache_dir.resolve(), output_dir, config_path)


if __name__ == "__main__":
    main()
