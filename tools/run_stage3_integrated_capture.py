#!/usr/bin/env python3
"""Single-controller integrated native capture and policy-lineage entry."""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from forcesmolvla.rft.stage3.integrated_capture import (  # noqa: E402
    IntegratedCaptureError,
    build_capture_contract,
    capture_mode_semantics,
    run_integrated_capture,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate or launch one integrated recorder-owned capture process.",
    )
    parser.add_argument("--mode", choices=("shadow", "policy-execute"), required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--episode-time", type=float, default=60.0)
    parser.add_argument("--tool-profile", default="onrobot_robotiq")
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--episode-id", required=True)
    parser.add_argument("--policy-revision", required=True)
    parser.add_argument("--policy-epoch", type=int, default=0)
    parser.add_argument("--reset-generation", type=int, default=0)
    parser.add_argument("--takeover-generation", type=int, default=0)
    parser.add_argument("--deployment-binding", type=Path)
    parser.add_argument(
        "--launch",
        action="store_true",
        help="Invoke one bound integrated backend; omitted means contract validation only.",
    )
    parser.add_argument(
        "--backend",
        help="Python module:attribute implementing the integrated backend protocol.",
    )
    return parser.parse_args(argv)


def _backend(specification: str | None) -> Any:
    if not specification or ":" not in specification:
        raise IntegratedCaptureError("INTEGRATED_CAPTURE_BACKEND_BINDING_REQUIRED")
    module_name, attribute = specification.split(":", 1)
    if not module_name or not attribute:
        raise IntegratedCaptureError("INTEGRATED_CAPTURE_BACKEND_BINDING_INVALID")
    try:
        candidate = getattr(importlib.import_module(module_name), attribute)
        return candidate() if isinstance(candidate, type) else candidate
    except (AttributeError, ImportError, TypeError) as error:
        raise IntegratedCaptureError("INTEGRATED_CAPTURE_BACKEND_LOAD_FAILED") from error


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    requested_mode = capture_mode_semantics(args.mode)
    try:
        if args.episodes != 1:
            raise IntegratedCaptureError("INTEGRATED_CAPTURE_ONE_EPISODE_PER_SEAL_REQUIRED")
        if args.episode_time <= 0:
            raise IntegratedCaptureError("INTEGRATED_CAPTURE_EPISODE_TIME_INVALID")
        contract = build_capture_contract(
            mode=args.mode,
            session_id=args.session_id,
            episode_id=args.episode_id,
            policy_revision=args.policy_revision,
            policy_epoch=args.policy_epoch,
            reset_generation=args.reset_generation,
            takeover_generation=args.takeover_generation,
            deployment_binding=args.deployment_binding,
        )
        recorder_arguments = {
            "root": str(args.root.resolve()),
            "task": args.task,
            "episodes": args.episodes,
            "episode_time": args.episode_time,
            "tool_profile": args.tool_profile,
        }
        if args.launch:
            result = run_integrated_capture(
                contract=contract,
                backend=_backend(args.backend),
                recorder_arguments=recorder_arguments,
            )
            payload = {
                "status": "CAPTURE_SEALED",
                "contract": contract.to_dict(),
                "episode_seal": dict(result),
            }
        else:
            if args.backend is not None:
                raise IntegratedCaptureError("INTEGRATED_CAPTURE_BACKEND_WITHOUT_LAUNCH")
            payload = {
                "status": "VALIDATED_NOT_LAUNCHED",
                "contract": contract.to_dict(),
                "recorder_arguments": recorder_arguments,
                "robot_or_ros_started": False,
            }
    except (OSError, TypeError, ValueError, IntegratedCaptureError) as error:
        print(json.dumps({
            "status": "BLOCKED",
            "reason": str(error),
            "requested_mode": args.mode,
            "requested_mode_semantics": requested_mode,
            "robot_or_ros_started": False,
        }, indent=2, sort_keys=True))
        return 2
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
