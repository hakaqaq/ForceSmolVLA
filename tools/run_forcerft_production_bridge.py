#!/usr/bin/env python3
"""Run the CPU-only ForceRFT production bridge for one recorder episode."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import tempfile

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/online_replay_production_bridge.v1.development.yaml"
REWARD_CLASSIFIER_TOOL = ROOT / "tools/reward_classifier/train_reward_classifier.py"
CONRFT_RUNTIME_ROOT = Path("/home/rlc123/conrft/serl_launcher")
CAMERA_KEYS = ("d435_third_person", "d405_wrist")
IMAGE_SHAPE = (480, 640, 3)
INFERENCE_BATCH_SIZE = 128
_OBSERVATION_QUALITY_REJECTION_PREFIXES = tuple(
    "BRIDGE_POLICY_EXECUTION_OBSERVATION_MATERIALIZATION_FAILED:ValueError:" + reason
    for reason in (
        "CAMERA_AGE_EXCEEDED:",
        "CLOCK_MAP_CALLBACK_DELAY_P99_EXCEEDED",
        "EPISODE_COMMON_INTERVAL_TOO_SHORT",
        "INTERCAMERA_SKEW_EXCEEDED",
        "STATE_POSE_AGE_EXCEEDED",
        "WRENCH_SOURCE_GAP_EXCEEDED",
        "WRENCH_VALID_SAMPLES_INSUFFICIENT_FOR_FILTER_WARMUP",
    )
)
_EPISODE_LOCAL_REJECTION_REASONS = frozenset(
    {
        "BRIDGE_POLICY_EXECUTION_ACTION_ACK_COVERAGE_MISMATCH",
    }
)


def _is_episode_quality_rejection(reason: str) -> bool:
    return (
        reason in _EPISODE_LOCAL_REJECTION_REASONS
        or reason.startswith(_OBSERVATION_QUALITY_REJECTION_PREFIXES)
    )


def _import_path(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class _LoadedFrozenRewardDetector:
    """One loaded/JIT-compiled detector reused by a local worker process."""

    def __init__(self, checkpoint: Path, expected_train_state_step: int) -> None:
        self.checkpoint = checkpoint.resolve()
        self.expected_train_state_step = int(expected_train_state_step)
        training_tool = _import_path(
            "reward_classifier_tool", REWARD_CLASSIFIER_TOOL
        )
        training_tool.install_type_only_octo_shim()
        sys.path.insert(0, str(CONRFT_RUNTIME_ROOT))
        from flax import serialization
        import jax
        import jax.numpy as jnp
        from serl_launcher.networks.reward_classifier import create_classifier

        if jax.default_backend() != "gpu":
            raise RuntimeError(
                f"BRIDGE_FROZEN_DETECTOR_GPU_REQUIRED:{jax.default_backend()}"
            )
        safe_tree, _ = training_tool.npz_encoder_tree()
        sample = {
            key: jnp.zeros((1, 1, *IMAGE_SHAPE), dtype=jnp.uint8)
            for key in CAMERA_KEYS
        }
        with training_tool.trusted_safe_npz_pickle_bridge(safe_tree) as bridge:
            target = create_classifier(
                jax.random.PRNGKey(0), sample, list(CAMERA_KEYS),
                pretrained_encoder_path=str(bridge), n_way=2,
            )
        state = serialization.from_bytes(target, self.checkpoint.read_bytes())
        if int(state.step) != self.expected_train_state_step:
            raise RuntimeError("BRIDGE_REWARD_DETECTOR_CHECKPOINT_STEP_DRIFT")

        @jax.jit
        def infer(observations):
            return state.apply_fn({"params": state.params}, observations, train=False)

        self.jax, self.jnp, self.infer = jax, jnp, infer

    def run(self, request_path: Path, output_path: Path) -> None:
        request = json.loads(request_path.read_text(encoding="utf-8"))
        batches = request.get("batches", [])
        if not batches:
            raise RuntimeError("BRIDGE_DETECTOR_BATCHES_MISSING")
        if (
            Path(request["checkpoint"]).expanduser().resolve() != self.checkpoint
            or int(request["expected_train_state_step"])
            != self.expected_train_state_step
        ):
            raise RuntimeError("BRIDGE_REWARD_DETECTOR_WORKER_IDENTITY_MISMATCH")
        logits: list[np.ndarray] = []
        frame_count = 0
        for batch in batches:
            arrays = [
                np.load(batch[name], mmap_mode="r", allow_pickle=False)
                for name in ("camera1", "camera2")
            ]
            count = int(batch["count"])
            if any(value.shape != (count, *IMAGE_SHAPE) for value in arrays):
                raise RuntimeError("BRIDGE_FROZEN_DETECTOR_IMAGE_BATCH_INVALID")
            observations = {
                key: self.jnp.asarray(np.asarray(value))[:, None]
                for key, value in zip(CAMERA_KEYS, arrays, strict=True)
            }
            logits.append(np.asarray(
                self.jax.block_until_ready(self.infer(observations)),
                dtype=np.float32,
            ).reshape(-1))
            frame_count += count
        values = np.concatenate(logits).astype(np.float64)
        probabilities = np.where(
            values >= 0, 1.0 / (1.0 + np.exp(-values)),
            np.exp(values) / (1.0 + np.exp(values)),
        )
        if len(probabilities) != frame_count or not np.all(np.isfinite(probabilities)):
            raise RuntimeError("BRIDGE_FROZEN_DETECTOR_OUTPUT_INVALID")
        np.save(output_path, probabilities, allow_pickle=False)
        print(json.dumps({
            "backend": self.jax.default_backend(), "frames": len(probabilities),
            "optimizer_updates": 0,
            "train_state_step": self.expected_train_state_step,
        }, sort_keys=True), flush=True)


def _detector_worker(request_path: Path, output_path: Path) -> None:
    request = json.loads(request_path.read_text(encoding="utf-8"))
    _LoadedFrozenRewardDetector(
        Path(request["checkpoint"]), int(request["expected_train_state_step"])
    ).run(request_path, output_path)


def _serve_detector_worker(
    socket_path: Path, checkpoint: Path, expected_train_state_step: int
) -> None:
    worker = _LoadedFrozenRewardDetector(checkpoint, expected_train_state_step)
    socket_path.unlink(missing_ok=True)
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as listener:
        listener.bind(str(socket_path))
        listener.listen(1)
        print(json.dumps({
            "status": "BRIDGE_REWARD_DETECTOR_WORKER_READY",
            "socket": str(socket_path),
        }, sort_keys=True), flush=True)
        try:
            while True:
                connection, _ = listener.accept()
                with connection, connection.makefile("rwb") as stream:
                    payload = json.loads(stream.readline())
                    if payload.get("shutdown") is True:
                        stream.write(b'{"ok":true}\n')
                        stream.flush()
                        return
                    try:
                        worker.run(Path(payload["request"]), Path(payload["output"]))
                        response = {"ok": True}
                    except Exception as error:
                        response = {
                            "ok": False,
                            "error": f"{type(error).__name__}:{error}",
                        }
                    stream.write((json.dumps(response) + "\n").encode("utf-8"))
                    stream.flush()
        finally:
            socket_path.unlink(missing_ok=True)


def _request_detector_worker(
    socket_path: Path, request_path: Path, output_path: Path
) -> None:
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
        connection.settimeout(300.0)
        connection.connect(str(socket_path))
        with connection.makefile("rwb") as stream:
            stream.write((json.dumps({
                "request": str(request_path), "output": str(output_path),
            }) + "\n").encode("utf-8"))
            stream.flush()
            response = json.loads(stream.readline())
    if response.get("ok") is not True:
        raise RuntimeError(
            f"BRIDGE_REWARD_DETECTOR_WORKER_FAILED:{response.get('error')}"
        )


class OneShotFrozenRewardDetector:
    def __init__(
        self, checkpoint: Path, detector_id: str, expected_train_state_step: int,
        *, worker_socket: Path | None = None,
    ) -> None:
        self.checkpoint = checkpoint
        self.detector_id = detector_id
        self.expected_train_state_step = int(expected_train_state_step)
        self.worker_socket = worker_socket

    def __call__(self, prepared):
        from forcesmolvla.rft.online.production_bridge import FrozenDetectorScores
        from PIL import Image

        with tempfile.TemporaryDirectory(prefix="online-frozen-detector-") as directory:
            root = Path(directory)
            request = root / "request.json"
            output = root / "probabilities.npy"
            batches = []
            for start in range(0, len(prepared.camera1_paths), INFERENCE_BATCH_SIZE):
                stop = min(start + INFERENCE_BATCH_SIZE, len(prepared.camera1_paths))
                paths_by_camera = (
                    prepared.camera1_paths[start:stop],
                    prepared.camera2_paths[start:stop],
                )
                batch = {"count": stop - start}
                for name, paths in zip(
                    ("camera1", "camera2"), paths_by_camera, strict=True
                ):
                    decoded = []
                    for path in paths:
                        with Image.open(path) as image:
                            rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
                        if rgb.shape != IMAGE_SHAPE:
                            raise RuntimeError(
                                "BRIDGE_FROZEN_DETECTOR_IMAGE_SHAPE_INVALID"
                            )
                        decoded.append(rgb)
                    batch_path = root / f"{name}_{start:06d}.npy"
                    np.save(
                        batch_path,
                        np.ascontiguousarray(decoded),
                        allow_pickle=False,
                    )
                    batch[name] = str(batch_path)
                batches.append(batch)
            request.write_text(
                json.dumps(
                    {
                        "batches": batches,
                        "checkpoint": str(self.checkpoint),
                        "expected_train_state_step": self.expected_train_state_step,
                    }
                ),
                encoding="utf-8",
            )
            if self.worker_socket is not None:
                _request_detector_worker(self.worker_socket, request, output)
            else:
                environment = os.environ.copy()
                environment["PYTHONPATH"] = str(ROOT / "src")
                subprocess.run(
                    [
                        shutil.which("conda") or "conda", "run",
                        "--no-capture-output", "-n", "conrft_reward", "python",
                        str(Path(__file__).resolve()),
                        "--detector-worker-request", str(request),
                        "--detector-worker-output", str(output),
                    ],
                    cwd=ROOT, env=environment, check=True,
                )
            probabilities = np.load(output, allow_pickle=False)
        return FrozenDetectorScores(
            probabilities=tuple(float(value) for value in probabilities),
            validity=(True,) * len(probabilities),
            detector_id=self.detector_id,
        )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-id", default="task2")
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--reward-transition-config", type=Path)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--episode", type=Path)
    parser.add_argument("--state-root", type=Path)
    parser.add_argument(
        "--deployed-actor-checkpoint",
        type=Path,
        help="Actor package that produced the policy-execution episode.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--admit-formal-online-r",
        action="store_true",
        help="Admit one bridge-PASS policy-execution smoke episode into formal R.",
    )
    parser.add_argument(
        "--operator-task-outcome",
        choices=("success", "failure"),
        help="Required operator semantic label for an integrated capture episode.",
    )
    parser.add_argument("--detector-worker-request", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--detector-worker-output", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--detector-worker-socket", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--serve-detector-worker", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args(argv)


def _resolve_actor_checkpoint(
    args: argparse.Namespace, *, output_root: Path
) -> Path:
    if args.deployed_actor_checkpoint is not None:
        return args.deployed_actor_checkpoint.resolve()
    if args.admit_formal_online_r:
        raise SystemExit(
            "--deployed-actor-checkpoint is required for formal online-R admission"
        )
    return (
        output_root / "sft/checkpoints/forcesmolvla_sft_step_010000"
    ).resolve()


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.detector_worker_request is not None:
        if args.detector_worker_output is None:
            raise SystemExit("--detector-worker-output is required")
        _detector_worker(args.detector_worker_request, args.detector_worker_output)
        return 0
    if args.serve_detector_worker:
        if args.detector_worker_socket is None or args.output_root is None:
            raise SystemExit(
                "--detector-worker-socket and --output-root are required"
            )
        reward_transition_config = (
            args.reward_transition_config
            or ROOT / "configs/tasks" / args.task_id
            / "forcerft_offline_reward_transitions.json"
        ).resolve()
        reward_transition_spec = json.loads(
            reward_transition_config.read_text(encoding="utf-8")
        )
        _serve_detector_worker(
            args.detector_worker_socket.resolve(),
            args.output_root.resolve()
            / "reward_classifier/checkpoints/best/best_checkpoint.msgpack",
            int(reward_transition_spec["classifier_train_state_step"]),
        )
        return 0
    from forcesmolvla.rft.online.production_bridge import (
        ProductionBridge,
        ProductionBridgeError,
        frozen_episode_materializer,
        load_bridge_config,
    )
    from forcesmolvla.training_runtime import resolve_task_output_root

    output_root = resolve_task_output_root(
        ROOT, task_id=args.task_id, output_root=args.output_root
    )
    reward_transition_config = (
        args.reward_transition_config
        or ROOT
        / "configs/tasks"
        / args.task_id
        / "forcerft_offline_reward_transitions.json"
    ).resolve()
    reward_transition_spec = json.loads(
        reward_transition_config.read_text(encoding="utf-8")
    )
    detector_id = str(reward_transition_spec["detector_spec"]["detector_id"])
    detector_train_state_step = int(
        reward_transition_spec["classifier_train_state_step"]
    )
    reward_detector_checkpoint = (
        output_root
        / "reward_classifier/checkpoints/best/best_checkpoint.msgpack"
    )
    actor_checkpoint = _resolve_actor_checkpoint(args, output_root=output_root)

    config, raw = load_bridge_config(args.config)
    episode = args.episode or Path(raw["recorded_offline_fixture"]["episode_dir"])
    if args.dry_run and args.admit_formal_online_r:
        raise SystemExit("--dry-run and --admit-formal-online-r are mutually exclusive")
    if not args.dry_run and args.state_root is None:
        raise SystemExit("--state-root is required unless --dry-run is used")
    state_root = args.state_root or Path("/tmp/forcesmolvla_stage3_bridge_dry_run")
    policy_execution_smoke = (
        (args.dry_run or args.admit_formal_online_r)
        and (
            episode.parent.parent
            / "integrated_capture"
            / episode.name
            / "streams/policy_execute_episode_seal.json"
        ).is_file()
    )
    bridge = ProductionBridge(
        config=config,
        state_root=state_root,
        parent_binding_path=actor_checkpoint,
        episode_materializer=(
            None
            if policy_execution_smoke and args.dry_run
            else frozen_episode_materializer(
                OneShotFrozenRewardDetector(
                    reward_detector_checkpoint,
                    detector_id,
                    detector_train_state_step,
                    worker_socket=args.detector_worker_socket,
                ),
                parent_binding_path=actor_checkpoint,
                detector_config_path=reward_transition_config,
            )
        ),
    )
    if args.admit_formal_online_r:
        if args.operator_task_outcome not in {"success", "failure"}:
            raise SystemExit("--operator-task-outcome is required for formal online-R admission")
        try:
            report = bridge.admit_policy_execution_smoke(
                episode,
                operator_task_outcome=args.operator_task_outcome,
            )
        except ProductionBridgeError as error:
            reason = str(error)
            if _is_episode_quality_rejection(reason):
                print(json.dumps({
                    "status": "FORMAL_ONLINE_R_REJECTED",
                    "reason": reason,
                }, sort_keys=True, indent=2))
                return 0
            raise
    else:
        report = bridge.process_episode(
            episode,
            dry_run=args.dry_run,
            operator_task_outcome=args.operator_task_outcome,
        )
    print(json.dumps(report.to_dict(), sort_keys=True, indent=2))
    return 0 if report.status in {
        "DRY_RUN_READY",
        "SEALED_COMMITTED",
        "ACTIVE_STAGED",
        "FORMAL_ONLINE_R_ADMITTED",
    } else 2


if __name__ == "__main__":
    raise SystemExit(main())
