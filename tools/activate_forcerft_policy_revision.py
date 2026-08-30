#!/usr/bin/env python3
"""Activate or query a published Stage-3 development policy revision."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from forcesmolvla.rft.stage3.publication import (  # noqa: E402
    InMemoryRevisionStateMachine,
    QuiescentBoundary,
    RevisionRecord,
    RevisionState,
    load_revision_registry,
    save_revision_registry,
)


class PolicyActivationError(RuntimeError):
    pass


def _json(path: Path, code: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, UnicodeDecodeError) as error:
        raise PolicyActivationError(code) from error
    if not isinstance(value, dict):
        raise PolicyActivationError(code)
    return value


def _activation_boundary(witness_path: Path) -> QuiescentBoundary:
    witness = _json(witness_path, "STAGE3_REAL_HOME_WITNESS_REQUIRED")
    quiescent = witness.get("quiescent")
    home = witness.get("home_result")
    if (
        witness.get("kind") != "reset_home_quiescent"
        or witness.get("source") != "recorded_home_backend"
        or witness.get("robot_home") is not True
        or int(witness.get("reset_generation", -1)) < 1
        or int(witness.get("completed_monotonic_ns", 0)) <= 0
        or not isinstance(quiescent, Mapping)
        or not isinstance(home, Mapping)
        or home.get("home_completed") is not True
        or home.get("controller_idle") is not True
        or int(home.get("controller_owner_count", -1)) != 1
    ):
        raise PolicyActivationError("STAGE3_REAL_HOME_WITNESS_REQUIRED")
    boundary = QuiescentBoundary(
        active_episode=bool(quiescent.get("active_episode", True)),
        inflight_inference=int(quiescent.get("inflight_inference", -1)),
        queued_actions=int(quiescent.get("queued_actions", -1)),
        unconsumed_acks=int(quiescent.get("unconsumed_acks", -1)),
        robot_home=True,
        wal_sealed=quiescent.get("wal_sealed") is True,
        candidate_validation_complete=True,
    )
    try:
        boundary.validate_for_activation()
    except (RuntimeError, ValueError) as error:
        raise PolicyActivationError("STAGE3_RESET_HOME_NOT_QUIESCENT") from error
    return boundary


def _validate_published_candidate(
    package: Path, candidate_id: str, candidate_revision: str
) -> None:
    candidate = _json(
        package / "candidate.json", "STAGE3_PUBLISHED_CANDIDATE_INVALID"
    )
    manifest = _json(
        package / "artifact_manifest.json", "STAGE3_PUBLISHED_CANDIDATE_INVALID"
    )
    metadata = manifest.get("metadata")
    if (
        candidate.get("revision_id") != candidate_id
        or candidate.get("model_revision") != candidate_revision
        or candidate.get("state") != "published"
        or candidate.get("published") is not True
        or candidate.get("activated") is not False
        or not isinstance(metadata, Mapping)
        or metadata.get("candidate_revision_id") != candidate_id
        or metadata.get("model_revision") != candidate_revision
        or metadata.get("published") is not True
    ):
        raise PolicyActivationError("STAGE3_PUBLISHED_CANDIDATE_INVALID")


def revision_status(registry: Path) -> dict[str, Any]:
    machine = load_revision_registry(registry, fresh_process=False)
    state = machine.snapshot()
    return {
        "active_revision": state["active_revision_id"],
        "previous_revision": state["previous_revision_id"],
        "pending_revision": state["pending_revision_id"],
        "policy_epoch": state["policy_epoch"],
        "records": state["records"],
    }


def activate_published_candidate(
    *,
    registry: Path,
    home_witness: Path,
    candidate_package: Path,
    candidate_id: str,
    candidate_revision: str,
    current_active_revision: str,
) -> dict[str, Any]:
    boundary = _activation_boundary(home_witness)
    _validate_published_candidate(
        candidate_package.resolve(), candidate_id, candidate_revision
    )
    if registry.is_file():
        machine = load_revision_registry(registry, fresh_process=True)
        if machine.active_revision_id == candidate_id:
            status = revision_status(registry)
            status["candidate_activated"] = True
            return status
        if machine.active_revision_id != current_active_revision:
            raise PolicyActivationError("STAGE3_ACTIVE_REVISION_MISMATCH")
    else:
        machine = InMemoryRevisionStateMachine(
            RevisionRecord(
                current_active_revision,
                current_active_revision,
                RevisionState.ACTIVE,
            ),
            safe_reset_required=True,
        )
    machine.acknowledge_reset_boundary(boundary)
    try:
        candidate = machine.register_candidate(candidate_id, candidate_revision)
        if candidate.state is RevisionState.CANDIDATE:
            machine.stage(candidate_id)
        elif candidate.state is not RevisionState.PENDING:
            raise PolicyActivationError("STAGE3_CANDIDATE_NOT_STAGEABLE")
        activated = machine.activate_pending(boundary)
    except (KeyError, RuntimeError, ValueError) as error:
        if isinstance(error, PolicyActivationError):
            raise
        raise PolicyActivationError(f"STAGE3_REVISION_ACTIVATION_FAILED:{error}") from error
    if (
        activated.revision_id != candidate_id
        or activated.model_sha256 != candidate_revision
        or machine.previous_revision_id != current_active_revision
    ):
        raise PolicyActivationError("STAGE3_REVISION_ACTIVATION_STATE_INVALID")
    save_revision_registry(registry, machine)
    status = revision_status(registry)
    status["candidate_activated"] = True
    return status


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    activate = subparsers.add_parser("activate")
    activate.add_argument("--registry", type=Path, required=True)
    activate.add_argument("--home-witness", type=Path, required=True)
    activate.add_argument("--candidate-package", type=Path, required=True)
    activate.add_argument("--candidate-id", required=True)
    activate.add_argument("--candidate-revision", required=True)
    activate.add_argument("--current-active-revision", required=True)
    status = subparsers.add_parser("status")
    status.add_argument("--registry", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "activate":
        result = activate_published_candidate(
            registry=args.registry,
            home_witness=args.home_witness,
            candidate_package=args.candidate_package,
            candidate_id=args.candidate_id,
            candidate_revision=args.candidate_revision,
            current_active_revision=args.current_active_revision,
        )
    else:
        result = revision_status(args.registry)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
