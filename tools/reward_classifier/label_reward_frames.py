#!/usr/bin/env python3
"""Build and serve a task-scoped reward-frame labeling workspace."""

from __future__ import annotations

import argparse
from http.server import ThreadingHTTPServer
import json
from pathlib import Path
import sys
import tempfile
from typing import Any

import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from forcesmolvla.training_runtime import resolve_task_dataset_root  # noqa: E402
from serve_task2_label_ui import FrameStore, make_handler  # noqa: E402


HUMAN_FIELDS = {
    "last_confident_incomplete_frame": None,
    "first_confident_complete_frame": None,
    "hard_negative_intervals": [],
    "ordinary_negative_intervals": [],
    "ambiguous_intervals": [],
    "completion_visible": None,
    "completion_stable": None,
    "positive_available": None,
    "confidence": None,
}


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", dir=path.parent, delete=False, encoding="utf-8"
    ) as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        temporary = Path(stream.name)
    temporary.replace(path)


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def build_workspace(
    *,
    task_id: str,
    dataset_root: Path,
    workspace: Path,
    train_episodes: int,
    val_episodes: int,
) -> tuple[Path, Path]:
    """Create a reusable, metadata-only review workspace."""

    index_path = workspace / "review_index.json"
    template_path = workspace / "label_template.json"
    protocol_path = workspace / "labeling_protocol.md"
    if index_path.is_file() and template_path.is_file() and protocol_path.is_file():
        index = _load(index_path)
        if (
            index.get("task_id") != task_id
            or Path(str(index.get("dataset_root_absolute_path", ""))).resolve()
            != dataset_root.resolve()
        ):
            raise RuntimeError("existing reward-label workspace belongs to another task or dataset")
        template = _load(template_path)
        changed = False
        for episode in template.get("episodes", []):
            for field in ("reviewer_id", "review_timestamp", "notes"):
                if field in episode:
                    episode.pop(field)
                    changed = True
        if changed:
            _write_json(template_path, template)
        return template_path, protocol_path
    if workspace.exists() and any(workspace.iterdir()):
        raise RuntimeError(f"incomplete reward-label workspace already exists: {workspace}")
    if train_episodes < 1 or val_episodes < 1:
        raise ValueError("train_episodes and val_episodes must both be positive")

    info = _load(dataset_root / "meta/info.json")
    conversion = _load(dataset_root / "conversion_manifest.json")
    split = _load(dataset_root / "split_manifest.json")
    source_episodes = conversion.get("episodes")
    if not isinstance(source_episodes, list) or not source_episodes:
        raise RuntimeError("conversion manifest has no episodes")
    tasks = {str(item.get("task", "")).strip() for item in source_episodes}
    if len(tasks) != 1 or not next(iter(tasks)):
        raise RuntimeError("dataset must contain exactly one non-empty task prompt")
    task_prompt = tasks.pop()
    split_by_id = {
        episode_id: name
        for name in ("train", "val", "test")
        for episode_id in split.get(name, [])
    }
    requested = {"train": train_episodes, "val": val_episodes}
    selected: list[dict[str, Any]] = []
    for name in ("train", "val"):
        candidates = [
            item
            for item in source_episodes
            if split_by_id.get(item.get("raw_episode_id")) == name
        ]
        if len(candidates) < requested[name]:
            raise RuntimeError(
                f"{task_id} only has {len(candidates)} {name} episodes; requested {requested[name]}"
            )
        selected.extend(candidates[: requested[name]])

    review_episodes = []
    label_episodes = []
    total_frames = 0
    chunks_size = int(info["chunks_size"])
    for episode in selected:
        episode_id = str(episode["raw_episode_id"])
        output_index = int(episode["output_episode_index"])
        name = split_by_id[episode_id]
        chunk_index, file_index = divmod(output_index, chunks_size)
        relative_parquet = str(info["data_path"]).format(
            chunk_index=chunk_index,
            file_index=file_index,
            episode_chunk=chunk_index,
        )
        table = pq.read_table(
            dataset_root / relative_parquet,
            columns=["timestamp", "frame_index", "episode_index", "index"],
        )
        values = table.to_pydict()
        frames = list(map(int, values["frame_index"]))
        if frames != list(range(len(frames))) or set(values["episode_index"]) != {
            output_index
        }:
            raise RuntimeError(f"invalid episode row mapping: {episode_id}")
        frame_count = len(frames)
        if frame_count != int(episode.get("frames", frame_count)):
            raise RuntimeError(f"frame count mismatch: {episode_id}")
        review_episodes.append(
            {
                "episode_id": episode_id,
                "output_episode_index": output_index,
                "split": name,
                "task_text_from_conversion_manifest": task_prompt,
                "frame_count": frame_count,
                "fps": int(info["fps"]),
                "parquet_relative_path": relative_parquet,
                "frame_indices": frames,
                "timestamps_seconds": values["timestamp"],
                "dataset_global_indices": list(map(int, values["index"])),
                "source_row_reference_format": (
                    f"{dataset_root.name}/{relative_parquet}#row={{frame_index}}"
                ),
                "candidate_10hz_frame_indices": list(range(0, frame_count, 3)),
                "detector_calibration_frame_indices": "all_30hz_frames",
            }
        )
        label_episodes.append(
            {
                "episode_id": episode_id,
                "output_episode_index": output_index,
                "split": name,
                "task_outcome_context": "success",
                "manual_review_status": "unreviewed",
                **HUMAN_FIELDS,
            }
        )
        total_frames += frame_count

    workspace.mkdir(parents=True, exist_ok=True)
    _write_json(
        index_path,
        {
            "schema": "forcesmolvla.reward_frame_review_index",
            "status": "ready_for_manual_review",
            "task_id": task_id,
            "task_prompt": task_prompt,
            "dataset_root_id": dataset_root.name,
            "dataset_root_absolute_path": str(dataset_root.resolve()),
            "fps": int(info["fps"]),
            "episode_count": len(review_episodes),
            "frame_count": total_frames,
            "camera_order": [
                "observation.images.camera1",
                "observation.images.camera2",
            ],
            "episodes": review_episodes,
        },
    )
    _write_json(
        template_path,
        {
            "schema": "forcesmolvla.reward_frame_labels",
            "status": "manual_review_in_progress",
            "task_id": task_id,
            "canonical_task_prompt": task_prompt,
            "programmatic_labels_generated": False,
            "episode_count": len(label_episodes),
            "episodes": label_episodes,
        },
    )
    protocol_path.write_text(
        "# Reward frame labeling protocol\n\n"
        f"Task: `{task_prompt}`\n\n"
        "Positive means the task is visibly complete, the object has been released, "
        "and the completed state remains stable. Frames before completion must be "
        "fully partitioned into ordinary-negative, hard-negative, or ambiguous "
        "inclusive intervals. Do not infer completion from the episode ending.\n",
        encoding="utf-8",
    )
    return template_path, protocol_path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--workspace", type=Path)
    parser.add_argument("--labels", type=Path)
    parser.add_argument("--train-episodes", type=int, default=16)
    parser.add_argument("--val-episodes", type=int, default=4)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    if args.host not in {"127.0.0.1", "localhost"} or not 0 < args.port < 65536:
        parser.error("label UI is restricted to a valid localhost port")
    args.dataset_root = resolve_task_dataset_root(
        ROOT, task_id=args.task_id, dataset_root=args.dataset_root
    )
    args.workspace = (
        ROOT / "artifacts" / args.task_id / "reward_labeling"
        if args.workspace is None
        else args.workspace
    ).resolve()
    args.labels = (
        ROOT / "labels" / f"{args.task_id}_reward_frame_labels.json"
        if args.labels is None
        else args.labels
    ).resolve()
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    template, protocol = build_workspace(
        task_id=args.task_id,
        dataset_root=args.dataset_root,
        workspace=args.workspace,
        train_episodes=args.train_episodes,
        val_episodes=args.val_episodes,
    )
    label_source = args.labels if args.labels.is_file() else template
    index = _load(args.workspace / "review_index.json")
    store = FrameStore(args.dataset_root, index)
    if args.check:
        first = index["episodes"][0]
        for camera in ("camera1", "camera2"):
            data, content_type = store.image(first["episode_id"], 0, camera)
            if not data or content_type not in {"image/png", "image/jpeg"}:
                raise RuntimeError("frame smoke failed")
        print(
            json.dumps(
                {
                    "status": "pass",
                    "task_id": args.task_id,
                    "episodes": index["episode_count"],
                    "labels": str(args.labels),
                }
            )
        )
        return 0
    ui = ROOT / "tools/reward_classifier/task2_label_ui.html"
    server = ThreadingHTTPServer(
        (args.host, args.port),
        make_handler(
            args.workspace,
            store,
            ui_path=ui,
            template_path=label_source,
            protocol_path=protocol,
        ),
    )
    print(f"{args.task_id} reward label UI: http://{args.host}:{args.port}")
    print(f"Export labels to: {args.labels}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
