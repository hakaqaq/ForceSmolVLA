#!/usr/bin/env python3
"""Run the CPU-only Stage-3 filesystem shadow bridge for one recorder episode."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/stage3_production_bridge.v1.development.yaml"
REWARD_DETECTOR_CHECKPOINT = (
    ROOT
    / "artifacts/development/stage2/reward_classifier/r0_training/checkpoints/best_checkpoint.msgpack"
)
REWARD_CLASSIFIER_TOOL = ROOT / "tools/reward_classifier/train_reward_classifier.py"
CONRFT_RUNTIME_ROOT = Path("/home/rlc123/conrft/serl_launcher")
CAMERA_KEYS = ("d435_third_person", "d405_wrist")
IMAGE_SHAPE = (480, 640, 3)
INFERENCE_BATCH_SIZE = 128


def _import_path(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _detector_worker(request_path: Path, output_path: Path) -> None:
    """Run one frozen reward-detector inference job and exit."""

    request = json.loads(request_path.read_text(encoding="utf-8"))
    batches = request.get("batches", [])
    if not batches:
        raise RuntimeError("BRIDGE_DETECTOR_BATCHES_MISSING")
    training_tool = _import_path(
        "stage3_reward_classifier_tool", REWARD_CLASSIFIER_TOOL
    )
    training_tool.install_type_only_octo_shim()
    sys.path.insert(0, str(CONRFT_RUNTIME_ROOT))
    from flax import serialization
    import jax
    import jax.numpy as jnp
    from serl_launcher.networks.reward_classifier import create_classifier

    if jax.default_backend() != "gpu":
        raise RuntimeError(f"BRIDGE_FROZEN_G1_GPU_REQUIRED:{jax.default_backend()}")
    safe_tree, _ = training_tool.npz_encoder_tree()
    sample = {
        key: jnp.zeros((1, 1, *IMAGE_SHAPE), dtype=jnp.uint8)
        for key in CAMERA_KEYS
    }
    with training_tool.trusted_safe_npz_pickle_bridge(safe_tree) as bridge:
        target = create_classifier(
            jax.random.PRNGKey(0),
            sample,
            list(CAMERA_KEYS),
            pretrained_encoder_path=str(bridge),
            n_way=2,
        )
    state = serialization.from_bytes(
        target, REWARD_DETECTOR_CHECKPOINT.read_bytes()
    )
    if int(state.step) != 150:
        raise RuntimeError("BRIDGE_REWARD_DETECTOR_CHECKPOINT_STEP_DRIFT")

    @jax.jit
    def infer(observations):
        return state.apply_fn({"params": state.params}, observations, train=False)

    logits: list[np.ndarray] = []
    frame_count = 0
    for batch in batches:
        arrays = [
            np.load(batch[name], mmap_mode="r", allow_pickle=False)
            for name in ("camera1", "camera2")
        ]
        count = int(batch["count"])
        if any(value.shape != (count, *IMAGE_SHAPE) for value in arrays):
            raise RuntimeError("BRIDGE_FROZEN_G1_IMAGE_BATCH_INVALID")
        observations = {
            key: jnp.asarray(np.asarray(value))[:, None]
            for key, value in zip(CAMERA_KEYS, arrays, strict=True)
        }
        logits.append(
            np.asarray(jax.block_until_ready(infer(observations)), dtype=np.float32)
            .reshape(-1)
        )
        frame_count += count
    values = np.concatenate(logits).astype(np.float64)
    probabilities = np.where(
        values >= 0,
        1.0 / (1.0 + np.exp(-values)),
        np.exp(values) / (1.0 + np.exp(values)),
    )
    if len(probabilities) != frame_count or not np.all(np.isfinite(probabilities)):
        raise RuntimeError("BRIDGE_FROZEN_G1_OUTPUT_INVALID")
    np.save(output_path, probabilities, allow_pickle=False)
    print(
        json.dumps(
            {
                "backend": jax.default_backend(),
                "frames": len(probabilities),
                "optimizer_updates": 0,
                "train_state_step": int(state.step),
            },
            sort_keys=True,
        )
    )


class OneShotFrozenRewardDetector:
    def __call__(self, prepared):
        from forcesmolvla.rft.stage3.production_bridge import FrozenDetectorScores
        from PIL import Image

        with tempfile.TemporaryDirectory(prefix="stage3-frozen-g1-") as directory:
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
                                "BRIDGE_FROZEN_G1_IMAGE_SHAPE_INVALID"
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
                json.dumps({"batches": batches}),
                encoding="utf-8",
            )
            environment = os.environ.copy()
            environment["PYTHONPATH"] = str(ROOT / "src")
            subprocess.run(
                [
                    shutil.which("conda") or "conda",
                    "run",
                    "--no-capture-output",
                    "-n",
                    "conrft_reward",
                    "python",
                    str(Path(__file__).resolve()),
                    "--detector-worker-request",
                    str(request),
                    "--detector-worker-output",
                    str(output),
                ],
                cwd=ROOT,
                env=environment,
                check=True,
            )
            probabilities = np.load(output, allow_pickle=False)
        return FrozenDetectorScores(
            probabilities=tuple(float(value) for value in probabilities),
            validity=(True,) * len(probabilities),
        )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--episode", type=Path)
    parser.add_argument("--state-root", type=Path)
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
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.detector_worker_request is not None:
        if args.detector_worker_output is None:
            raise SystemExit("--detector-worker-output is required")
        _detector_worker(args.detector_worker_request, args.detector_worker_output)
        return 0
    from forcesmolvla.rft.stage3.production_bridge import (
        Stage3ProductionBridge,
        frozen_episode_materializer,
        load_bridge_config,
    )

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
    bridge = Stage3ProductionBridge(
        config=config,
        state_root=state_root,
        episode_materializer=(
            None
            if policy_execution_smoke
            else frozen_episode_materializer(OneShotFrozenRewardDetector())
        ),
    )
    if args.admit_formal_online_r:
        if args.operator_task_outcome != "success":
            raise SystemExit(
                "--operator-task-outcome success is required for formal online-R admission"
            )
        report = bridge.admit_policy_execution_smoke(
            episode,
            operator_task_outcome=args.operator_task_outcome,
        )
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
