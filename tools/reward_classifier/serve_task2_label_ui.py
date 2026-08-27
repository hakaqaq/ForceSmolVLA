#!/usr/bin/env python3
"""Serve the task2 dual-camera manual review UI on localhost.

The server has GET routes only and reads embedded image bytes from the frozen
LeRobot-v3 parquet rows.  Human edits stay in the browser and are exported as a
download; neither the dataset nor project files are writable through this UI.
"""

from __future__ import annotations

import argparse
from functools import lru_cache
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BUNDLE = ROOT / "artifacts/development/stage2/task2_reward_review_bundle_v1"
DEFAULT_DATASET = ROOT / "datasets/task2_lerobotv3"
DEFAULT_UI = ROOT / "tools/reward_classifier/task2_label_ui.html"
DEFAULT_LABEL_TEMPLATE = ROOT / "labels/task2_reward_frame_labels.v2.template.json"
DEFAULT_PROTOCOL = ROOT / "docs/task2_reward_labeling_protocol.v2.md"
CANONICAL_TASK_PROMPT = "Pick up the purple ring and place it onto the red peg."
CAMERA_KEYS = {
    "camera1": "observation.images.camera1",
    "camera2": "observation.images.camera2",
}


def load_index(bundle: Path) -> dict[str, Any]:
    index = json.loads((bundle / "review_index.json").read_text())
    if (
        index.get("artifact_status") != "MANUAL_REVIEW_MATERIALS_ONLY"
        or index.get("camera_order")
        != ["observation.images.camera1", "observation.images.camera2"]
        or index.get("fps") != 30
    ):
        raise RuntimeError("review bundle contract mismatch")
    return index


def load_label_contract(template_path: Path, protocol_path: Path) -> dict[str, Any]:
    template = json.loads(template_path.read_text())
    protocol = protocol_path.read_text()
    if (
        template.get("schema_version") != "force_rft_task2_reward_frame_labels.v2"
        or template.get("artifact_status") != "BLANK_MANUAL_LABEL_TEMPLATE_V2"
        or template.get("canonical_task_prompt") != CANONICAL_TASK_PROMPT
        or template.get("programmatic_labels_generated") is not False
        or CANONICAL_TASK_PROMPT not in protocol
        or "红色 peg 明确穿过紫色 ring 的中心孔" not in protocol
    ):
        raise RuntimeError("active v2 labeling contract mismatch")
    return template


class FrameStore:
    def __init__(self, dataset: Path, index: dict[str, Any]):
        self.dataset = dataset.resolve()
        self.episodes = {episode["episode_id"]: episode for episode in index["episodes"]}

    @lru_cache(maxsize=2)
    def _images(self, episode_id: str) -> dict[str, list[dict[str, Any]]]:
        episode = self.episodes[episode_id]
        path = (self.dataset / episode["parquet_relative_path"]).resolve()
        if not path.is_relative_to(self.dataset):
            raise RuntimeError("parquet path escapes dataset root")
        table = pq.read_table(path, columns=list(CAMERA_KEYS.values()))
        return {key: table.column(column).to_pylist() for key, column in CAMERA_KEYS.items()}

    def image(self, episode_id: str, frame_index: int, camera: str) -> tuple[bytes, str]:
        if episode_id not in self.episodes or camera not in CAMERA_KEYS:
            raise KeyError("unknown episode or camera")
        episode = self.episodes[episode_id]
        if not 0 <= frame_index < episode["frame_count"]:
            raise IndexError("frame outside episode")
        value = self._images(episode_id)[camera][frame_index]
        data = value.get("bytes") if isinstance(value, dict) else None
        if not isinstance(data, bytes):
            raise RuntimeError("embedded image bytes missing")
        if data.startswith(b"\x89PNG\r\n\x1a\n"):
            return data, "image/png"
        if data.startswith(b"\xff\xd8"):
            return data, "image/jpeg"
        raise RuntimeError("unsupported embedded image encoding")


def make_handler(
    bundle: Path,
    store: FrameStore,
    *,
    ui_path: Path,
    template_path: Path,
    protocol_path: Path,
):
    class Handler(BaseHTTPRequestHandler):
        def send_bytes(self, body: bytes, content_type: str, status: HTTPStatus = HTTPStatus.OK) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def send_json(self, value: Any, status: HTTPStatus = HTTPStatus.OK) -> None:
            self.send_bytes(
                json.dumps(value, separators=(",", ":")).encode(),
                "application/json; charset=utf-8",
                status,
            )

        def do_POST(self) -> None:  # noqa: N802
            self.send_json({"error": "read-only server"}, HTTPStatus.METHOD_NOT_ALLOWED)

        def do_PUT(self) -> None:  # noqa: N802
            self.do_POST()

        def do_DELETE(self) -> None:  # noqa: N802
            self.do_POST()

        def do_GET(self) -> None:  # noqa: N802
            request = urlparse(self.path)
            try:
                if request.path == "/":
                    self.send_bytes(ui_path.read_bytes(), "text/html; charset=utf-8")
                elif request.path == "/review_index.json":
                    self.send_bytes((bundle / "review_index.json").read_bytes(), "application/json")
                elif request.path == "/label_template.json":
                    self.send_bytes(template_path.read_bytes(), "application/json")
                elif request.path == "/labeling_protocol.md":
                    self.send_bytes(protocol_path.read_bytes(), "text/markdown; charset=utf-8")
                elif request.path == "/api/frame":
                    query = parse_qs(request.query)
                    episode_id = query["episode_id"][0]
                    frame_index = int(query["frame_index"][0])
                    camera = query["camera"][0]
                    data, content_type = store.image(episode_id, frame_index, camera)
                    self.send_bytes(data, content_type)
                else:
                    self.send_json({"error": "not found"}, HTTPStatus.NOT_FOUND)
            except (KeyError, IndexError, ValueError) as exc:
                self.send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
            except Exception as exc:
                self.send_json({"error": str(exc)}, HTTPStatus.INTERNAL_SERVER_ERROR)

        def log_message(self, format: str, *args: object) -> None:
            print(f"{self.address_string()} {format % args}")

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--ui", type=Path, default=DEFAULT_UI)
    parser.add_argument("--label-template", type=Path, default=DEFAULT_LABEL_TEMPLATE)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.host not in {"127.0.0.1", "localhost"}:
        raise RuntimeError("review server is restricted to localhost")
    index = load_index(args.bundle)
    label_contract = load_label_contract(args.label_template, args.protocol)
    store = FrameStore(args.dataset_root, index)
    if args.check:
        first = index["episodes"][0]
        for camera in CAMERA_KEYS:
            data, content_type = store.image(first["episode_id"], 0, camera)
            if not data or content_type not in {"image/png", "image/jpeg"}:
                raise RuntimeError("frame smoke failed")
        print(
            json.dumps(
                {
                    "status": "PASS",
                    "episodes": len(index["episodes"]),
                    "label_schema": label_contract["schema_version"],
                    "protocol": args.protocol.name,
                    "writes": False,
                }
            )
        )
        return
    server = ThreadingHTTPServer(
        (args.host, args.port),
        make_handler(
            args.bundle,
            store,
            ui_path=args.ui,
            template_path=args.label_template,
            protocol_path=args.protocol,
        ),
    )
    print(f"Read-only task2 label UI: http://{args.host}:{args.port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
