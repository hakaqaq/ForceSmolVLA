"""Pinned LeRobot v3 storage contract and writer construction."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal


STATE_NAMES = ["tcp_x", "tcp_y", "tcp_z", "tcp_roll", "tcp_pitch", "tcp_yaw", "gripper_width_m"]
WRENCH_NAMES = ["force_x_base_n", "force_y_base_n", "force_z_base_n", "moment_x_tcp_base_nm", "moment_y_tcp_base_nm", "moment_z_tcp_base_nm"]
ACTION_NAMES = [
    "target_tcp_x",
    "target_tcp_y",
    "target_tcp_z",
    "target_tcp_roll",
    "target_tcp_pitch",
    "target_tcp_yaw",
    "target_gripper_width_m",
]


def _read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError) as error:
        raise ValueError(f"required dataset manifest is missing or invalid: {path}") from error


def split_episode_indices(root: Path, split_name: Literal["train", "val", "test"]) -> list[int]:
    """Resolve raw episode IDs through the conversion manifest; never trust info.json splits."""
    if split_name not in {"train", "val", "test"}:
        raise ValueError(f"unsupported split: {split_name!r}")
    split = _read_json(root / "split_manifest.json")
    conversion = _read_json(root / "conversion_manifest.json")
    groups = {name: tuple(split.get(name, ())) for name in ("train", "val", "test")}
    sets = {name: set(values) for name, values in groups.items()}
    if any(len(sets[name]) != len(groups[name]) for name in groups):
        raise ValueError("split manifest contains duplicate episode IDs")
    if sets["train"] & sets["val"] or sets["train"] & sets["test"] or sets["val"] & sets["test"]:
        raise ValueError("split manifest is not episode-disjoint")
    episodes = tuple(conversion.get("episodes", ()))
    raw_ids = tuple(entry["raw_episode_id"] for entry in episodes)
    output_ids = tuple(int(entry["output_episode_index"]) for entry in episodes)
    if len(set(raw_ids)) != len(raw_ids):
        raise ValueError("conversion manifest contains duplicate raw episode IDs")
    if len(set(output_ids)) != len(output_ids):
        raise ValueError("conversion manifest contains duplicate output episode indices")
    mapping = dict(zip(raw_ids, output_ids, strict=True))
    if set(mapping) != sets["train"] | sets["val"] | sets["test"]:
        raise ValueError("split manifest does not exactly cover converted episodes")
    if conversion.get("split") != split:
        raise ValueError("conversion and split manifests disagree")
    return sorted(mapping[episode_id] for episode_id in groups[split_name])


def load_dataset_split(
    root: Path,
    *,
    repo_id: str,
    split_name: Literal["train", "val", "test"],
    artifact_use: Literal["development", "formal"],
    **kwargs,
):
    """Load one explicit v4.1 split and fail closed on development/formal status."""
    conversion = _read_json(root / "conversion_manifest.json")
    if conversion.get("repo_id") != repo_id:
        raise ValueError("requested repo_id does not match conversion manifest")
    if artifact_use == "formal":
        if conversion.get("artifact_status") != "approved" or conversion.get("formal_ready") is not True:
            raise ValueError("formal loading requires an approved, formal-ready conversion manifest")
    elif artifact_use == "development":
        if conversion.get("artifact_status") != "development_only" or conversion.get("formal_ready") is not False:
            raise ValueError("development loading requires a development-only conversion manifest")
    else:
        raise ValueError(f"unsupported artifact use: {artifact_use!r}")
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    return LeRobotDataset(
        repo_id=repo_id,
        root=root,
        episodes=split_episode_indices(root, split_name),
        download_videos=False,
        **kwargs,
    )


def storage_features(height: int = 480, width: int = 640) -> dict[str, dict]:
    return {
        "observation.images.camera1": {"dtype": "image", "shape": (3, height, width), "names": ["channel", "height", "width"]},
        "observation.images.camera2": {"dtype": "image", "shape": (3, height, width), "names": ["channel", "height", "width"]},
        "observation.state": {"dtype": "float32", "shape": (7,), "names": STATE_NAMES},
        "observation.wrench": {"dtype": "float32", "shape": (6,), "names": WRENCH_NAMES},
        "action": {"dtype": "float32", "shape": (7,), "names": ACTION_NAMES},
        "provenance.tuple_host_monotonic_ns": {"dtype": "int64", "shape": (1,), "names": ["ns"]},
        "provenance.state_pose_source_stamp_ns": {"dtype": "int64", "shape": (1,), "names": ["ns"]},
        "provenance.state_pose_age_ms": {"dtype": "float32", "shape": (1,), "names": ["ms"]},
        "provenance.camera1_receive_monotonic_ns": {"dtype": "int64", "shape": (1,), "names": ["ns"]},
        "provenance.camera1_age_ms": {"dtype": "float32", "shape": (1,), "names": ["ms"]},
        "provenance.camera2_receive_monotonic_ns": {"dtype": "int64", "shape": (1,), "names": ["ns"]},
        "provenance.camera2_age_ms": {"dtype": "float32", "shape": (1,), "names": ["ms"]},
        "provenance.intercamera_skew_ms": {"dtype": "float32", "shape": (1,), "names": ["ms"]},
        "provenance.gripper_source_stamp_ns": {"dtype": "int64", "shape": (1,), "names": ["ns"]},
        "provenance.pose_source_stamp_ns": {"dtype": "int64", "shape": (1,), "names": ["ns"]},
        "provenance.pose_age_ms": {"dtype": "float32", "shape": (1,), "names": ["ms"]},
        "provenance.wrench_raw_source_stamp_ns": {"dtype": "int64", "shape": (1,), "names": ["ns"]},
        "provenance.wrench_filter_output_stamp_ns": {"dtype": "int64", "shape": (1,), "names": ["ns"]},
        "provenance.action_ack_receive_monotonic_ns": {"dtype": "int64", "shape": (1,), "names": ["ns"]},
        "provenance.action_ack_age_ms": {"dtype": "float32", "shape": (1,), "names": ["ms"]},
        "provenance.calibration_index": {"dtype": "int64", "shape": (1,), "names": ["index"]},
        "provenance.validity_bits": {"dtype": "int64", "shape": (1,), "names": ["bitset"]},
    }


def create_dataset(
    root: Path,
    *,
    repo_id: str,
    fps: int = 30,
    height: int = 480,
    width: int = 640,
    image_writer_threads: int = 16,
):
    if root.exists():
        raise FileExistsError(f"refusing to overwrite existing dataset root: {root}")
    if fps != 30:
        raise ValueError("v4.1 storage fps must be 30")
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    return LeRobotDataset.create(
        repo_id=repo_id,
        fps=fps,
        features=storage_features(height, width),
        root=root,
        robot_type="fr3_available_sensor_v4_1",
        use_videos=False,
        image_writer_processes=0,
        image_writer_threads=image_writer_threads,
    )
