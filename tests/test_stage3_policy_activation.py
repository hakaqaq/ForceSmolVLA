from __future__ import annotations

import json
from pathlib import Path

import pytest

from robot.deployment.reset_home_witness import write_reset_home_witness
from tools.activate_forcerft_policy_revision import (
    PolicyActivationError,
    activate_published_candidate,
    revision_status,
)


ACTIVE = "e24c1d6bb0a778921659514ac47c692b952178aa39af2601ccf0fc32bf94774d"
CANDIDATE_ID = "stage3-online-r-joint-cycle-000010-candidate"
CANDIDATE = "ab97aefb6a916a4f03e02d264e6c4b2f5c6462d2a7d6e1e9ebcd171d3a527c6b"


def _write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _seal(path: Path) -> Path:
    _write(
        path,
        {
            "episode_id": "episode_000000",
            "technical_seal": "complete",
            "sealed_monotonic_ns": 10,
            "reset_generation": 0,
            "policy_request_count": 4,
            "policy_result_count": 4,
            "policy_request_canceled_count": 0,
            "controller_process_count": 1,
            "deploy_controller_started": False,
        },
    )
    return path


def _home_result() -> dict:
    return {
        "home_completed": True,
        "controller_idle": True,
        "gateway_status": "completed: joint position target",
        "completed_monotonic_ns": 20,
        "controller_owner_count": 1,
        "max_joint_error_rad": 0.001,
        "max_joint_velocity_rad_s": 0.001,
        "home_joint_tolerance_rad": 0.01,
        "home_velocity_tolerance_rad_s": 0.02,
        "settle_time_s": 0.5,
        "home_implementation": (
            "record_franka_spacemouse_publisher."
            "FrankaRecordSpaceMousePublisher.move_to_recorded_home"
        ),
        "quiescent": {
            "active_episode": False,
            "inflight_inference": 0,
            "queued_actions": 0,
            "unconsumed_acks": 0,
            "wal_sealed": True,
        },
    }


def _candidate(path: Path) -> Path:
    _write(
        path / "candidate.json",
        {
            "revision_id": CANDIDATE_ID,
            "model_revision": CANDIDATE,
            "state": "published",
            "published": True,
            "activated": False,
        },
    )
    _write(
        path / "artifact_manifest.json",
        {
            "metadata": {
                "candidate_revision_id": CANDIDATE_ID,
                "model_revision": CANDIDATE,
                "published": True,
                "activated": False,
            }
        },
    )
    return path


def test_real_home_success_writes_witness(tmp_path: Path) -> None:
    output = tmp_path / "home.json"
    witness = write_reset_home_witness(
        output=output,
        previous_episode_seal=_seal(tmp_path / "seal.json"),
        home_backend=_home_result,
    )

    assert output.is_file()
    assert witness["source"] == "recorded_home_backend"
    assert witness["robot_home"] is True
    assert witness["reset_generation"] == 1
    assert witness["quiescent"]["active_episode"] is False
    assert witness["quiescent"]["inflight_inference"] == 0


def test_home_failure_does_not_write_witness(tmp_path: Path) -> None:
    output = tmp_path / "home.json"

    def fail() -> dict:
        raise RuntimeError("home failed")

    with pytest.raises(RuntimeError, match="home failed"):
        write_reset_home_witness(
            output=output,
            previous_episode_seal=_seal(tmp_path / "seal.json"),
            home_backend=fail,
        )
    assert not output.exists()


def test_activation_requires_real_witness_and_rejects_synthetic_boolean(
    tmp_path: Path,
) -> None:
    kwargs = {
        "registry": tmp_path / "registry.json",
        "candidate_package": _candidate(tmp_path / "candidate"),
        "candidate_id": CANDIDATE_ID,
        "candidate_revision": CANDIDATE,
        "current_active_revision": ACTIVE,
    }
    with pytest.raises(PolicyActivationError, match="REAL_HOME_WITNESS_REQUIRED"):
        activate_published_candidate(
            home_witness=tmp_path / "missing.json", **kwargs
        )

    synthetic = tmp_path / "synthetic.json"
    _write(
        synthetic,
        {
            "kind": "reset_home_quiescent",
            "source": "synthetic_boolean",
            "robot_home": True,
        },
    )
    with pytest.raises(PolicyActivationError, match="REAL_HOME_WITNESS_REQUIRED"):
        activate_published_candidate(home_witness=synthetic, **kwargs)
    assert not kwargs["registry"].exists()


def test_valid_witness_activates_candidate_and_preserves_previous(
    tmp_path: Path,
) -> None:
    witness = tmp_path / "home.json"
    write_reset_home_witness(
        output=witness,
        previous_episode_seal=_seal(tmp_path / "seal.json"),
        home_backend=_home_result,
    )
    registry = tmp_path / "registry.json"

    result = activate_published_candidate(
        registry=registry,
        home_witness=witness,
        candidate_package=_candidate(tmp_path / "candidate"),
        candidate_id=CANDIDATE_ID,
        candidate_revision=CANDIDATE,
        current_active_revision=ACTIVE,
    )

    assert result["candidate_activated"] is True
    assert result["active_revision"] == CANDIDATE_ID
    assert result["previous_revision"] == ACTIVE
    queried = revision_status(registry)
    assert queried["active_revision"] == CANDIDATE_ID
    assert queried["previous_revision"] == ACTIVE
    records = {row["revision_id"]: row for row in queried["records"]}
    assert records[CANDIDATE_ID]["state"] == "active"
    assert records[ACTIVE]["state"] == "previous"
