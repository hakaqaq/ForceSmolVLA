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
DEFAULT_SHADOW_BACKEND = (
    "forcesmolvla.rft.stage3.integrated_shadow_backend:IntegratedShadowBackend"
)

from forcesmolvla.rft.stage3.integrated_capture import (  # noqa: E402
    IntegratedCaptureError,
    build_capture_contract,
    capture_mode_semantics,
    run_integrated_capture,
    validate_development_policy_package,
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
        "--allow-development-policy-execution-smoke",
        action="store_true",
        help=(
            "explicitly unlock one approved development policy-execution "
            "episode; has no effect without --mode policy-execute"
        ),
    )
    parser.add_argument("--policy-host", default="127.0.0.1")
    parser.add_argument("--policy-port", type=int, default=8000)
    parser.add_argument(
        "--deployment-profile",
        type=Path,
        default=ROOT / "configs/deployment.active.development.json",
    )
    parser.add_argument("--inference-timeout", type=float, default=30.0)
    parser.add_argument("--shadow-inference-period", type=float, default=0.1)
    parser.add_argument("--backend-start-timeout", type=float, default=180.0)
    parser.add_argument("--policy-replan-steps", type=int, default=8)
    parser.add_argument("--policy-queue-low-watermark", type=int, default=4)
    parser.add_argument("--max-force-n", type=float, default=25.0)
    parser.add_argument("--max-torque-nm", type=float, default=2.0)
    parser.add_argument(
        "--async-learner",
        action="store_true",
        help="require the unified Stage-3 inference/Learner runtime",
    )
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


def _policy_execution_deployment(
    profile_path: Path,
    deployment_binding: Path | None,
    policy_revision: str,
) -> Path:
    """Resolve profile-selected package and binding before any backend is launched."""

    try:
        profile = json.loads(profile_path.resolve().read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise IntegratedCaptureError(
            "POLICY_EXECUTE_APPROVED_DEVELOPMENT_PROFILE_UNREADABLE"
        ) from error
    if (
        not isinstance(profile, dict)
        or profile.get("schema_version")
        != "forcesmolvla-deployment-profile-v1"
        or profile.get("artifact_status") != "development_only"
    ):
        raise IntegratedCaptureError(
            "POLICY_EXECUTE_APPROVED_DEVELOPMENT_PROFILE_REQUIRED"
        )
    try:
        profile_binding = Path(str(profile["deployment_binding"]))
        checkpoint = Path(str(profile["checkpoint"]))
    except (KeyError, TypeError, ValueError) as error:
        raise IntegratedCaptureError(
            "POLICY_EXECUTE_APPROVED_DEVELOPMENT_PROFILE_REQUIRED"
        ) from error
    if not profile_binding.is_absolute():
        profile_binding = ROOT / profile_binding
    if not checkpoint.is_absolute():
        checkpoint = ROOT / checkpoint
    selected_binding = (
        profile_binding.resolve()
        if deployment_binding is None
        else deployment_binding.expanduser().resolve()
    )
    if selected_binding != profile_binding.resolve():
        raise IntegratedCaptureError(
            "POLICY_EXECUTE_APPROVED_DEVELOPMENT_BINDING_MISMATCH"
        )
    validate_development_policy_package(checkpoint, policy_revision)
    return selected_binding


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    requested_mode = capture_mode_semantics(args.mode)
    try:
        if args.episodes != 1:
            raise IntegratedCaptureError("INTEGRATED_CAPTURE_ONE_EPISODE_PER_SEAL_REQUIRED")
        if args.episode_time <= 0:
            raise IntegratedCaptureError("INTEGRATED_CAPTURE_EPISODE_TIME_INVALID")
        if (
            args.policy_host not in {"127.0.0.1", "localhost"}
            or args.policy_port <= 0
            or args.inference_timeout <= 0
            or args.shadow_inference_period <= 0
            or args.backend_start_timeout <= 0
            or args.max_force_n <= 0
            or args.max_torque_nm <= 0
        ):
            raise IntegratedCaptureError("INTEGRATED_CAPTURE_BACKEND_ARGUMENTS_INVALID")
        if not 0 < args.policy_queue_low_watermark < args.policy_replan_steps <= 50:
            raise IntegratedCaptureError("POLICY_EXECUTE_RUNTIME_LIMITS_INVALID")
        if args.async_learner and args.mode != "policy-execute":
            raise IntegratedCaptureError("ASYNC_LEARNER_REQUIRES_POLICY_EXECUTE")
        deployment_binding = args.deployment_binding
        if args.mode == "policy-execute":
            if not args.allow_development_policy_execution_smoke:
                raise IntegratedCaptureError("POLICY_EXECUTE_EXPLICIT_FLAG_REQUIRED")
            deployment_binding = _policy_execution_deployment(
                args.deployment_profile,
                deployment_binding,
                args.policy_revision,
            )
        contract = build_capture_contract(
            mode=args.mode,
            session_id=args.session_id,
            episode_id=args.episode_id,
            policy_revision=args.policy_revision,
            policy_epoch=args.policy_epoch,
            reset_generation=args.reset_generation,
            takeover_generation=args.takeover_generation,
            deployment_binding=deployment_binding,
            allow_development_policy_execution_smoke=(
                args.allow_development_policy_execution_smoke
            ),
        )
        recorder_arguments = {
            "root": str(args.root.resolve()),
            "task": args.task,
            "episodes": args.episodes,
            "episode_time": args.episode_time,
            "tool_profile": args.tool_profile,
            "policy_host": args.policy_host,
            "policy_port": args.policy_port,
            "deployment_profile": str(args.deployment_profile.resolve()),
            "inference_timeout": args.inference_timeout,
            "shadow_inference_period": args.shadow_inference_period,
            "backend_start_timeout": args.backend_start_timeout,
            "policy_replan_steps": args.policy_replan_steps,
            "policy_queue_low_watermark": args.policy_queue_low_watermark,
            "max_force_n": args.max_force_n,
            "max_torque_nm": args.max_torque_nm,
            "async_learner": args.async_learner,
        }
        if args.launch:
            result = run_integrated_capture(
                contract=contract,
                backend=_backend(args.backend or DEFAULT_SHADOW_BACKEND),
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
