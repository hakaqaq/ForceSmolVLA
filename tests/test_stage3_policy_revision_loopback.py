from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import json
import os
from pathlib import Path
import stat
import subprocess
import sys

from jsonschema import Draft202012Validator
import pytest
import yaml


ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import run_stage3_policy_revision_loopback as g6p

from forcesmolvla.rft.stage3.protocol import (
    InferenceDisposition,
    PolicyEpochGate,
    TransportEnvelope,
)
from forcesmolvla.rft.stage3.publication import (
    InMemoryRevisionStateMachine,
    QuiescentBoundary,
    RevisionRecord,
    RevisionState,
    SimulatedPublicationCrash,
    canonical_sha256,
    export_immutable_revision,
    load_revision_registry,
    save_revision_registry,
    validate_immutable_revision,
)
from forcesmolvla.rft.stage3.transition import (
    REVISION_BOUND_EVENTS,
    TransitionContractError,
    validate_episode_revision_bindings,
)


SHA0 = "0" * 64
SHA1 = "1" * 64
CONFIG_PATH = ROOT / "configs/stage3_policy_revision_loopback.v1.development.yaml"


@pytest.fixture(scope="module")
def config() -> dict:
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def bindings(config: dict) -> dict:
    return g6p.build_revision_bindings(config)


def quiet(**overrides) -> QuiescentBoundary:
    values = {
        "active_episode": False,
        "inflight_inference": 0,
        "queued_actions": 0,
        "unconsumed_acks": 0,
        "robot_home": True,
        "wal_sealed": True,
        "candidate_validation_complete": True,
    }
    values.update(overrides)
    return QuiescentBoundary(**values)


def envelope(
    *,
    epoch: int = 0,
    revision: str = "r0",
    model: str = SHA0,
    request: str = "request-0",
    chunk: str = "chunk-0",
) -> TransportEnvelope:
    return TransportEnvelope(
        run_id="run",
        session_id="session",
        episode_id="episode",
        request_id=request,
        chunk_id=chunk,
        arbitration_epoch_at_request=epoch,
        policy_revision_id=revision,
        model_sha256=model,
        t_ref_monotonic_ns=1,
        observation_id="observation",
    )


def test_recursive_source_binding_covers_required_real_sources_and_blockers(
    bindings: dict, config: dict,
) -> None:
    source = bindings["recursive_source"]
    recorded = {entry["relative_path"] for entry in source["files"]}
    all_python = {
        path.relative_to(ROOT).as_posix()
        for path in (ROOT / "src/forcesmolvla").rglob("*.py")
    }
    assert all_python <= recorded
    assert {
        "configs/stage3_policy_revision_loopback.v1.development.yaml",
        "src/forcesmolvla/rft/stage3/protocol.py",
        "tools/run_stage3_policy_revision_loopback.py",
        "tools/serve_policy.py",
        "schemas/stage3_policy_revision.v1.schema.json",
        "environment-manifest/requirements.lock",
    } <= recorded
    for files in config["contract_files"].values():
        assert set(files) <= recorded
    assert bindings["vendor"]["commit"] == "30da8e687a6dfc617fcd94afc367ac7071c376ce"
    assert bindings["vendor"]["files"]
    production = source["production_source_binding"]
    assert production["complete"] is False
    assert any("stage3_policy_server" in blocker for blocker in production["blockers"])
    assert any("robot_reset_home_witness" in blocker for blocker in production["blockers"])
    assert any("durable_production_wal" in blocker for blocker in production["blockers"])


def test_immutable_content_addressed_export_is_atomic_idempotent_and_read_only(
    tmp_path: Path, bindings: dict,
) -> None:
    root = tmp_path / "revisions"
    first = export_immutable_revision(root, model_payload=b"tiny-model", bindings=bindings)
    validated = validate_immutable_revision(first.path, expected_bindings=bindings)
    assert first.revision_id == canonical_sha256({
        key: validated["manifest"][key]
        for key in (
            "schema_version", "artifact_kind", "synthetic_revision_payload",
            "model", "files", "files_tree_sha256", "bindings",
        )
    })
    assert first.created is True
    second = export_immutable_revision(root, model_payload=b"tiny-model", bindings=bindings)
    assert second.created is False
    assert second.revision_id == first.revision_id
    assert second.canonical_manifest_digest == first.canonical_manifest_digest
    assert (first.path / "COMPLETED.json").is_file()
    assert all(
        not (path.stat().st_mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))
        for path in [first.path, *first.path.rglob("*")]
    )


def test_crash_before_atomic_rename_and_registry_replace_never_publish_partial_state(
    tmp_path: Path, bindings: dict,
) -> None:
    revisions = tmp_path / "revisions"
    with pytest.raises(SimulatedPublicationCrash, match="BEFORE_ATOMIC_RENAME"):
        export_immutable_revision(
            revisions,
            model_payload=b"crashing-model",
            bindings=bindings,
            fault="before_atomic_rename",
        )
    assert list(revisions.iterdir())
    assert all(path.name.startswith(".revision-tmp-") for path in revisions.iterdir())

    machine = InMemoryRevisionStateMachine(
        RevisionRecord("r0", SHA0, RevisionState.ACTIVE, artifact_digest=SHA0)
    )
    registry = tmp_path / "registry/registry.json"
    save_revision_registry(registry, machine)
    changed = InMemoryRevisionStateMachine.from_snapshot(machine.snapshot())
    changed.invalidate_policy_epoch("human_takeover")
    with pytest.raises(SimulatedPublicationCrash, match="REGISTRY_CRASH"):
        save_revision_registry(registry, changed, fault_before_replace=True)
    assert load_revision_registry(registry, fresh_process=False).policy_epoch == 0


def test_lifecycle_serialization_pending_rollback_and_fresh_recovery_fail_closed(
    tmp_path: Path,
) -> None:
    machine = InMemoryRevisionStateMachine(
        RevisionRecord("r0", SHA0, RevisionState.ACTIVE, artifact_digest=SHA0)
    )
    machine.register_candidate("r1", SHA1, artifact_digest=SHA1)
    machine.stage("r1")
    machine.activate_pending(quiet())
    machine.register_candidate("r2", SHA0, artifact_digest=SHA0)
    machine.stage("r2")
    machine.rollback(quiet(), reason="test rollback")
    machine.register_candidate(
        "bad", SHA1, artifact_digest=SHA1, validation_complete=False,
    )
    machine.reject("bad", "validation failed")
    registry = tmp_path / "registry.json"
    save_revision_registry(registry, machine)
    recovered = load_revision_registry(registry, fresh_process=True)
    assert recovered.active_revision_id == "r0"
    assert recovered.pending_revision_id == "r2"
    assert recovered.previous_revision_id is None
    assert recovered.record("r1").state is RevisionState.ROLLED_BACK
    assert recovered.record("bad").state is RevisionState.REJECTED
    assert recovered.record("bad").rejection_reason == "validation failed"
    assert recovered.policy_epoch == 2
    assert recovered.publication_counters == machine.publication_counters
    assert recovered.safe_reset_required is True
    assert recovered.action_authorization_allowed is False
    with pytest.raises(RuntimeError, match="SAFE_RESET_REQUIRED"):
        recovered.activate_pending(quiet())
    with pytest.raises(RuntimeError, match="SAFE_RESET_REQUIRED"):
        recovered.begin_episode()


@pytest.mark.parametrize(
    "override",
    [
        {"active_episode": True},
        {"inflight_inference": 1},
        {"queued_actions": 1},
        {"unconsumed_acks": 1},
        {"robot_home": False},
        {"wal_sealed": False},
        {"candidate_validation_complete": False},
    ],
)
def test_each_quiescent_activation_condition_is_fail_closed(override: dict) -> None:
    machine = InMemoryRevisionStateMachine(
        RevisionRecord("r0", SHA0, RevisionState.ACTIVE)
    )
    machine.register_candidate("r1", SHA1)
    machine.stage("r1")
    before = machine.snapshot()
    queue_state = ["queued"] if override.get("queued_actions") else []
    transition_state = {"committed": False}
    with pytest.raises(RuntimeError, match="NOT_QUIESCENT"):
        machine.activate_pending(quiet(**override))
    assert machine.snapshot() == before
    assert queue_state == (["queued"] if override.get("queued_actions") else [])
    assert transition_state == {"committed": False}


def test_one_episode_pins_revision_model_and_epoch_while_new_candidate_is_pending() -> None:
    machine = InMemoryRevisionStateMachine(
        RevisionRecord("r0", SHA0, RevisionState.ACTIVE)
    )
    machine.begin_episode()
    pin = machine.episode_pin()
    machine.register_candidate("r1", SHA1)
    machine.stage("r1")
    assert machine.active_revision_id == "r0"
    assert machine.policy_epoch == 0
    machine.assert_episode_binding("r0", SHA0, 0)
    for wrong in (("r1", SHA0, 0), ("r0", SHA1, 0), ("r0", SHA0, 1)):
        with pytest.raises(RuntimeError, match="ONE_EPISODE_ONE_REVISION"):
            machine.assert_episode_binding(*wrong)


def test_policy_epoch_gate_stale_drops_old_model_request_chunk_revision_and_epoch() -> None:
    gate = PolicyEpochGate(active_revision_id="r0", active_model_sha256=SHA0)
    current = envelope()
    assert gate.pin_request(current) is InferenceDisposition.ACCEPT
    assert gate.classify_result(current) is InferenceDisposition.ACCEPT
    for old in (
        replace(current, policy_revision_id="old"),
        replace(current, model_sha256=SHA1),
        replace(current, request_id="old-request"),
        replace(current, chunk_id="old-chunk"),
        replace(current, arbitration_epoch_at_request=1),
    ):
        assert gate.classify_result(old) is InferenceDisposition.STALE_DROP
    assert gate.invalidate_queued_policy() == 1
    assert gate.has_pinned_request is False
    assert gate.classify_result(current) is InferenceDisposition.STALE_DROP
    assert gate.activate_revision("r1", SHA1) == 2


def test_cross_revision_observation_ack_or_transition_is_quarantinable() -> None:
    expected = {
        "policy_revision_id": "r0",
        "model_sha256": SHA0,
        "policy_epoch": 0,
    }
    events = {name: deepcopy(expected) for name in REVISION_BOUND_EVENTS}
    assert validate_episode_revision_bindings(events, **expected)
    for event in ("current_observation", "next_observation", "ack_ledger", "transition"):
        mismatched = deepcopy(events)
        mismatched[event]["policy_revision_id"] = "r1"
        with pytest.raises(TransitionContractError, match=f"CROSS_REVISION_{event.upper()}_QUARANTINE"):
            validate_episode_revision_bindings(mismatched, **expected)


def test_invalid_candidate_cannot_enter_pending() -> None:
    machine = InMemoryRevisionStateMachine(
        RevisionRecord("r0", SHA0, RevisionState.ACTIVE)
    )
    machine.register_candidate("bad", SHA1, validation_complete=False)
    with pytest.raises(RuntimeError, match="VALIDATION_INCOMPLETE"):
        machine.stage("bad")
    rejected = machine.reject("bad", "binding mismatch")
    assert rejected.state is RevisionState.REJECTED
    assert machine.pending_revision_id is None


def test_cli_runs_twice_with_identical_canonical_report_and_full_fault_evidence(
    tmp_path: Path,
) -> None:
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    reports = []
    for index in (1, 2):
        report_path = tmp_path / f"report-{index}.json"
        completed = subprocess.run(
            [
                sys.executable,
                str(ROOT / "tools/run_stage3_policy_revision_loopback.py"),
                "--output-root",
                str(tmp_path / f"run-{index}"),
                "--report",
                str(report_path),
            ],
            cwd=ROOT,
            env=env,
            text=True,
            capture_output=True,
            check=True,
        )
        assert "G6P_RESULT=PASS" in completed.stdout
        reports.append(json.loads(report_path.read_text(encoding="utf-8")))
    assert reports[0] == reports[1]
    assert reports[0]["canonical_report_sha256"] == reports[1]["canonical_report_sha256"]
    canonical = deepcopy(reports[0])
    recorded_digest = canonical.pop("canonical_report_sha256")
    assert recorded_digest == canonical_sha256(canonical)
    schema = json.loads(
        (ROOT / "schemas/stage3_policy_revision_loopback_report.v1.schema.json").read_text()
    )
    Draft202012Validator(schema).validate(reports[0])
    assert reports[0]["fault_injection"]["all_passed"] is True
    assert len(reports[0]["fault_injection"]["cases"]) >= 23
    assert reports[0]["fresh_process_recovery"]["safe_reset_required"] is True
    assert reports[0]["fresh_process_recovery"]["action_authorization_allowed"] is False
    assert reports[0]["PRODUCTION_SOURCE_BINDING_COMPLETE"] is False
    assert "/tmp/" not in json.dumps(reports[0])
    for index, report in ((1, reports[0]), (2, reports[1])):
        for name in ("stable", "activated_then_rolled_back", "pending"):
            revision_id = report["revision_artifacts"][name]["revision_id"]
            revision_path = tmp_path / f"run-{index}/revisions" / revision_id
            assert revision_path.is_dir()
            assert not revision_path.stat().st_mode & stat.S_IWUSR


def test_cli_has_no_ros_robot_or_network_server_imports() -> None:
    source = (ROOT / "tools/run_stage3_policy_revision_loopback.py").read_text(encoding="utf-8")
    import ast

    tree = ast.parse(source)
    banned = {"rclpy", "rospy", "roslib", "requests", "httpx", "socket", "http"}
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".", 1)[0])
    assert imported.isdisjoint(banned)
    assert "serve_policy" not in imported
    assert "deploy_forcesmolvla" not in imported


def test_checked_in_report_is_schema_valid_and_canonically_self_signed() -> None:
    report_path = (
        ROOT / "artifacts/development/stage3/stage3_policy_revision_loopback.v1.json"
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    schema = json.loads(
        (ROOT / "schemas/stage3_policy_revision_loopback_report.v1.schema.json").read_text()
    )
    Draft202012Validator(schema).validate(report)
    digest = report.pop("canonical_report_sha256")
    assert digest == canonical_sha256(report)
    assert report["G6P_RESULT"] == "PASS"
    assert report["G6P_LOOPBACK_ACTIVATION"] is True
    assert report["POLICY_REVISION_ACTIVATED"] is False
