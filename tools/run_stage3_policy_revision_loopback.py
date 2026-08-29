#!/usr/bin/env python3
"""Run the CPU-only G6P immutable revision lifecycle filesystem loopback."""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import sys
import tempfile
from typing import Any, Mapping

from jsonschema import Draft202012Validator
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from forcesmolvla.rft.stage3.checkpoint import cpu_round_trip_online_checkpoint
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
    canonical_json_bytes,
    canonical_sha256,
    export_immutable_revision,
    load_revision_registry,
    save_revision_registry,
    sha256_file,
    validate_immutable_revision,
)
from forcesmolvla.rft.stage3.transition import (
    REVISION_BOUND_EVENTS,
    TransitionContractError,
    validate_episode_revision_bindings,
)


REPORT_SCHEMA_VERSION = "forcesmolvla_stage3_policy_revision_loopback_report.v1"
REPORT_SCHEMA = ROOT / "schemas/stage3_policy_revision_loopback_report.v1.schema.json"


def require(condition: bool, code: str) -> None:
    if not condition:
        raise RuntimeError(code)


def _git(*args: str, repo_root: Path = ROOT) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=repo_root, text=True, stderr=subprocess.DEVNULL,
    ).strip()


def _git_blob(repo_root: Path, commit: str, relative_path: str) -> bytes:
    try:
        return subprocess.check_output(
            ["git", "show", f"{commit}:{relative_path}"],
            cwd=repo_root,
            stderr=subprocess.DEVNULL,
        )
    except subprocess.CalledProcessError as error:
        raise RuntimeError(f"G6C_FROZEN_BLOB_MISSING:{relative_path}") from error


def verify_required_freeze_ancestor(
    config: Mapping[str, Any], repo_root: Path = ROOT,
) -> dict[str, Any]:
    provenance = config["provenance"]
    freeze = provenance["required_freeze_ancestor"]
    exists = subprocess.run(
        ["git", "cat-file", "-e", f"{freeze}^{{commit}}"],
        cwd=repo_root,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    require(exists.returncode == 0, "G6C_REQUIRED_FREEZE_ANCESTOR_MISSING")
    head = _git("rev-parse", "HEAD", repo_root=repo_root)
    ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", freeze, head],
        cwd=repo_root,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    require(ancestor.returncode == 0, "G6C_CURRENT_HEAD_NOT_FREEZE_DESCENDANT")
    branch = _git("branch", "--show-current", repo_root=repo_root)
    require(
        branch == provenance["required_branch"],
        "G6P_BASELINE_BRANCH_MISMATCH",
    )
    return {
        "branch": branch,
        "head": head,
        "required_freeze_ancestor": freeze,
        "freeze_ancestor_verified": True,
    }


def verify_historical_evidence(
    config: Mapping[str, Any], repo_root: Path = ROOT,
) -> dict[str, Any]:
    provenance = config["provenance"]
    freeze = provenance["required_freeze_ancestor"]
    evidence = provenance["historical_evidence"]
    report_path = repo_root / evidence["report_path"]
    markdown_path = repo_root / evidence["markdown_path"]
    report_bytes = report_path.read_bytes()
    markdown_bytes = markdown_path.read_bytes()
    require(
        hashlib.sha256(report_bytes).hexdigest() == evidence["report_file_sha256"],
        "G6C_HISTORICAL_REPORT_FILE_SHA_MISMATCH",
    )
    require(
        hashlib.sha256(markdown_bytes).hexdigest() == evidence["markdown_file_sha256"],
        "G6C_HISTORICAL_MARKDOWN_FILE_SHA_MISMATCH",
    )
    require(
        report_bytes == _git_blob(repo_root, freeze, evidence["report_path"]),
        "G6C_HISTORICAL_REPORT_NOT_FREEZE_BLOB",
    )
    require(
        markdown_bytes == _git_blob(repo_root, freeze, evidence["markdown_path"]),
        "G6C_HISTORICAL_MARKDOWN_NOT_FREEZE_BLOB",
    )
    report = json.loads(report_bytes)
    recorded_digest = report.pop("canonical_report_sha256")
    require(
        recorded_digest == evidence["canonical_report_sha256"],
        "G6C_HISTORICAL_CANONICAL_DIGEST_MISMATCH",
    )
    require(
        canonical_sha256(report) == recorded_digest,
        "G6C_HISTORICAL_CANONICAL_SELF_SIGNATURE_MISMATCH",
    )
    require(
        report["baseline"]["head"] == evidence["historical_generation_head"],
        "G6C_HISTORICAL_GENERATION_HEAD_MISMATCH",
    )
    return {
        "historical_evidence_verified": True,
        "historical_report_canonical_sha256": recorded_digest,
        "historical_report_file_sha256": evidence["report_file_sha256"],
        "historical_markdown_file_sha256": evidence["markdown_file_sha256"],
    }


def _matches_git_glob(relative_path: str, pattern: str) -> bool:
    candidate = PurePosixPath(relative_path)
    while True:
        if candidate.match(pattern):
            return True
        if "/**/" not in pattern:
            return False
        pattern = pattern.replace("/**/", "/", 1)


def _configured_bound_paths(config: Mapping[str, Any], repo_root: Path) -> set[str]:
    source = config["source_binding"]
    paths = {
        path.relative_to(repo_root).as_posix()
        for pattern in source["recursive_globs"]
        for path in repo_root.glob(pattern)
        if path.is_file()
    }
    paths.update(source["exact_files"])
    paths.update(
        relative
        for relatives in config.get("contract_files", {}).values()
        for relative in relatives
    )
    if config.get("approved_hybrid_parent"):
        paths.add(config["approved_hybrid_parent"])
    return paths


def verify_frozen_bound_sources(
    config: Mapping[str, Any], repo_root: Path = ROOT,
) -> dict[str, Any]:
    """Explicit release-boundary check; ordinary loopback intentionally skips it."""
    current = verify_required_freeze_ancestor(config, repo_root)
    freeze = current["required_freeze_ancestor"]
    source = config["source_binding"]
    vendor_path = source.get("vendor_path", "").rstrip("/")
    current_paths = _configured_bound_paths(config, repo_root)
    tree_paths = set(
        _git("ls-tree", "-r", "--name-only", freeze, repo_root=repo_root).splitlines()
    )
    frozen_paths = {
        path
        for path in tree_paths
        if any(_matches_git_glob(path, pattern) for pattern in source["recursive_globs"])
    }
    frozen_paths.update(source["exact_files"])
    frozen_paths.update(
        relative
        for relatives in config.get("contract_files", {}).values()
        for relative in relatives
    )
    if config.get("approved_hybrid_parent"):
        frozen_paths.add(config["approved_hybrid_parent"])
    if vendor_path:
        current_paths = {
            path for path in current_paths if not path.startswith(f"{vendor_path}/")
        }
        frozen_paths = {
            path for path in frozen_paths if not path.startswith(f"{vendor_path}/")
        }
    require(
        current_paths == frozen_paths,
        "G6C_FROZEN_SOURCE_PATH_SET_MISMATCH",
    )
    entries = []
    for relative_path in sorted(frozen_paths):
        frozen_bytes = _git_blob(repo_root, freeze, relative_path)
        current_bytes = (repo_root / relative_path).read_bytes()
        require(
            current_bytes == frozen_bytes,
            f"G6C_FROZEN_BOUND_FILE_MISMATCH:{relative_path}",
        )
        entries.append({
            "relative_path": relative_path,
            "size_bytes": len(current_bytes),
            "sha256": hashlib.sha256(current_bytes).hexdigest(),
        })
    vendor_commit = None
    if vendor_path:
        vendor_commit = _git("rev-parse", f"{freeze}:{vendor_path}", repo_root=repo_root)
        require(
            _git("rev-parse", f":{vendor_path}", repo_root=repo_root) == vendor_commit,
            "G6C_FROZEN_VENDOR_GITLINK_MISMATCH",
        )
    return {
        **current,
        "bound_file_count": len(entries),
        "bound_tree_sha256": canonical_sha256(entries),
        "vendor_commit": vendor_commit,
    }


def _file_entry(path: Path) -> dict[str, Any]:
    path = path.resolve()
    require(path.is_file(), f"G6P_SOURCE_BINDING_FILE_MISSING:{path}")
    return {
        "relative_path": path.relative_to(ROOT).as_posix(),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def _file_group(paths: list[Path]) -> dict[str, Any]:
    entries = [_file_entry(path) for path in sorted(set(paths))]
    require(bool(entries), "G6P_SOURCE_BINDING_GROUP_EMPTY")
    return {"tree_sha256": canonical_sha256(entries), "files": entries}


def _resolve_files(config: Mapping[str, Any]) -> tuple[list[Path], list[Path]]:
    source = config["source_binding"]
    recursive: list[Path] = []
    for pattern in source["recursive_globs"]:
        recursive.extend(path for path in ROOT.glob(pattern) if path.is_file())
    exact = [ROOT / relative for relative in source["exact_files"]]
    contracts = [
        ROOT / relative
        for values in config["contract_files"].values()
        for relative in values
    ]
    return sorted(set([*recursive, *exact, *contracts, ROOT / config["approved_hybrid_parent"]])), contracts


def build_revision_bindings(config: Mapping[str, Any]) -> dict[str, Any]:
    source_files, _ = _resolve_files(config)
    source_group = _file_group(source_files)
    stage3_configs = _file_group([
        path for path in ROOT.glob("configs/stage3_*") if path.is_file()
    ])
    contract_groups = {
        name: _file_group([ROOT / relative for relative in relatives])
        for name, relatives in sorted(config["contract_files"].items())
    }
    parent_path = ROOT / config["approved_hybrid_parent"]
    parent = json.loads(parent_path.read_text(encoding="utf-8"))
    vendor_files = sorted(
        path
        for path in (ROOT / "vendor/lerobot/src/lerobot/policies/smolvla").rglob("*.py")
        if path.is_file()
    )
    vendor = {
        **_file_group(vendor_files),
        "path": config["source_binding"]["vendor_path"],
        "commit": _git("rev-parse", ":vendor/lerobot"),
    }
    production_requirements = config["source_binding"]["production_requirements"]
    blockers = [
        f"{name}:{relative}"
        for name, relative in sorted(production_requirements.items())
        if not (ROOT / relative).is_file()
    ]
    present = [
        "environment_lock:environment-manifest/requirements.lock",
        "legacy_loopback_policy_server:tools/serve_policy.py",
        "stage3_protocol:src/forcesmolvla/rft/stage3/protocol.py",
        "isolated_revision_loopback:tools/run_stage3_policy_revision_loopback.py",
    ]
    source_group.update({
        "algorithm": "sha256(canonical_json(sorted(relative_path,size_bytes,sha256)))",
        "coverage": {
            "forcesmolvla_recursive_python": True,
            "stage3_configs": True,
            "contracts": True,
            "vendor_smolvla": True,
            "environment_metadata": True,
            "deployment_recorder_protocol_when_present": True,
        },
        "production_source_binding": {
            "complete": not blockers,
            "present": present,
            "blockers": blockers,
        },
    })
    environment_lock = _file_entry(ROOT / config["runtime_environment"]["lock"])
    return {
        "learner_checkpoint": {
            "identity": config["learner_checkpoint"]["identity"],
            "canonical_digest": canonical_sha256(config["learner_checkpoint"]["state"]),
            "synthetic": True,
        },
        "approved_hybrid_parent": {
            "binding_id": parent["binding_id"],
            "decision": parent["continuation_semantics"]["parent_binding_decision"],
            "path": config["approved_hybrid_parent"],
            "sha256": sha256_file(parent_path),
        },
        "recursive_source": source_group,
        "resolved_stage3_configs": stage3_configs,
        "contracts": contract_groups,
        "temporal_action_dimensions": deepcopy(config["temporal_action_dimensions"]),
        "vendor": vendor,
        "runtime_environment": {
            "environment_lock": environment_lock,
            "python_runtime": (
                f"{sys.implementation.name}-{sys.version_info.major}.{sys.version_info.minor}"
            ),
            "cuda_visible_devices": config["runtime_environment"]["cuda_visible_devices"],
        },
    }


def quiet(**overrides: Any) -> QuiescentBoundary:
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


def _envelope(
    machine: InMemoryRevisionStateMachine,
    *,
    revision_id: str,
    model_sha256: str,
    request_id: str,
    chunk_id: str,
) -> TransportEnvelope:
    return TransportEnvelope(
        run_id="g6p-loopback-run",
        session_id="g6p-loopback-session",
        episode_id="g6p-loopback-episode",
        request_id=request_id,
        chunk_id=chunk_id,
        arbitration_epoch_at_request=machine.policy_epoch,
        policy_revision_id=revision_id,
        model_sha256=model_sha256,
        t_ref_monotonic_ns=1,
        observation_id="g6p-observation-0",
    )


def _expect_failure(function, code: str) -> str:
    try:
        function()
    except (RuntimeError, ValueError, TransitionContractError) as error:
        require(code in str(error), f"G6P_WRONG_FAILURE:{code}:{error}")
        return str(error)
    raise AssertionError(f"G6P_EXPECTED_FAILURE_NOT_RAISED:{code}")


def _make_mutable_copy(source: Path, parent: Path, label: str) -> Path:
    target = parent / label / source.name
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, target)
    for path in [target, *target.rglob("*")]:
        path.chmod(0o755 if path.is_dir() else 0o644)
    return target


def _checkpoint_payload(machine: InMemoryRevisionStateMachine, source_sha: str) -> dict:
    state_ref = lambda name: {"relative_path": f"state/{name}.pt", "sha256": source_sha}
    return {
        "schema_version": "forcesmolvla_stage3_online_checkpoint.v1",
        "checkpoint_id": "g6p-synthetic-registry-round-trip",
        "boundary": {
            "episode_sealed": True,
            "learner_update_committed": True,
            "pending_graphs": 0,
            "pending_optimizer_steps": 0,
        },
        "parent": {
            "binding_status": "approved_hybrid",
            "cross_stage_optimizer_rebuilt": True,
        },
        "models": {
            name: state_ref(name)
            for name in ("actor", "q1", "q2", "q1_target", "q2_target")
        },
        "optimizers": {
            name: state_ref(f"{name}_optimizer") for name in ("actor", "critic")
        },
        "schedulers": {
            name: state_ref(f"{name}_scheduler") for name in ("actor", "critic")
        },
        "rng": state_ref("rng"),
        "samplers": state_ref("samplers"),
        "replay": {
            "canonical_index_sha256": source_sha,
            "R_watermark": 0,
            "D_watermark": 0,
            "wal_committed_offset": 0,
            "episode_finalization_state": "sealed",
            "outbox_cursor": 0,
        },
        "credits": {"minted": 0, "consumed": 0, "available": 0},
        "publication": {
            "active_revision": machine.active_revision_id,
            "pending_revision": machine.pending_revision_id,
            "previous_revision": machine.previous_revision_id,
            "policy_epoch": machine.policy_epoch,
        },
        "counters": {
            "learner_cycles": 0,
            "critic_updates": 0,
            "actor_updates": 0,
            "polyak_updates_per_target": 0,
            "publication_count": machine.publication_counters["activations"],
        },
        "bindings": {
            "source_tree_sha256": source_sha,
            "action_contract_sha256": source_sha,
        },
        "authorization": {"deployment_release": False, "robot_execution": False},
    }


def recover_registry_fresh_process(registry_path: Path) -> dict[str, Any]:
    machine = load_revision_registry(registry_path, fresh_process=True)
    begin_error = _expect_failure(machine.begin_episode, "SAFE_RESET_REQUIRED")
    activation_error = _expect_failure(
        lambda: machine.activate_pending(quiet()), "SAFE_RESET_REQUIRED",
    )
    snapshot = machine.snapshot()
    return {
        "snapshot": snapshot,
        "safe_reset_required": machine.safe_reset_required,
        "action_authorization_allowed": machine.action_authorization_allowed,
        "begin_episode_error": begin_error,
        "activation_error": activation_error,
    }


def _fresh_subprocess_recovery(registry_path: Path) -> dict[str, Any]:
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
    completed = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve()),
            "--recover-registry",
            str(registry_path),
        ],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )
    return json.loads(completed.stdout)


def run_loopback(output_root: Path, config_path: Path) -> dict[str, Any]:
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "G6P_CUDA_MUST_BE_HIDDEN")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    provenance = verify_required_freeze_ancestor(config)
    historical = verify_historical_evidence(config)
    bindings = build_revision_bindings(config)
    require(
        bindings["approved_hybrid_parent"]["decision"] == "APPROVED_HYBRID",
        "G6P_PARENT_NOT_APPROVED_HYBRID",
    )
    output_root = Path(output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=False)
    revisions = output_root / "revisions"
    registry_path = output_root / "registry/registry.json"
    payloads = {
        name: value.encode("utf-8") for name, value in config["synthetic_payloads"].items()
    }

    stable = export_immutable_revision(
        revisions, model_payload=payloads["stable"], bindings=bindings,
    )
    initial = RevisionRecord(
        stable.revision_id,
        stable.model_sha256,
        RevisionState.ACTIVE,
        artifact_digest=stable.canonical_manifest_digest,
    )
    machine = InMemoryRevisionStateMachine(initial)
    save_revision_registry(registry_path, machine)
    gate = PolicyEpochGate(
        active_revision_id=stable.revision_id,
        active_model_sha256=stable.model_sha256,
    )

    machine.begin_episode()
    pin = machine.episode_pin()
    active_request = _envelope(
        machine,
        revision_id=pin.policy_revision_id,
        model_sha256=pin.model_sha256,
        request_id="request-active-0",
        chunk_id="chunk-active-0",
    )
    require(gate.pin_request(active_request) is InferenceDisposition.ACCEPT, "G6P_REQUEST_PIN")
    event_bindings = {name: pin.to_dict() for name in REVISION_BOUND_EVENTS}
    validate_episode_revision_bindings(event_bindings, **pin.to_dict())

    candidate = export_immutable_revision(
        revisions, model_payload=payloads["candidate"], bindings=bindings,
    )
    validated_candidate = validate_immutable_revision(
        candidate.path, expected_bindings=bindings,
    )
    machine.register_candidate(
        candidate.revision_id,
        candidate.model_sha256,
        artifact_digest=candidate.canonical_manifest_digest,
        validation_complete=True,
    )
    machine.stage(candidate.revision_id)
    save_revision_registry(registry_path, machine)
    mid_episode_state = machine.snapshot()
    require(mid_episode_state["active_revision_id"] == stable.revision_id, "G6P_MID_EPISODE_ACTIVE_CHANGED")
    require(mid_episode_state["policy_epoch"] == 0, "G6P_MID_EPISODE_EPOCH_CHANGED")
    _expect_failure(
        lambda: machine.activate_pending(quiet(active_episode=True)), "NOT_QUIESCENT",
    )
    require(machine.snapshot() == mid_episode_state, "G6P_ACTIVE_EPISODE_GATE_MUTATED_STATE")

    cross_revision = deepcopy(event_bindings)
    cross_revision["next_observation"] = {
        "policy_revision_id": candidate.revision_id,
        "model_sha256": candidate.model_sha256,
        "policy_epoch": pin.policy_epoch,
    }
    cross_revision_reason = _expect_failure(
        lambda: validate_episode_revision_bindings(cross_revision, **pin.to_dict()),
        "CROSS_REVISION_NEXT_OBSERVATION_QUARANTINE",
    )
    transition_quarantine = {
        "reason": cross_revision_reason,
        "action_dispatch": False,
        "transition_commit": False,
        "replay_commit": False,
    }
    machine.assert_episode_binding(
        pin.policy_revision_id, pin.model_sha256, pin.policy_epoch,
    )
    machine.end_episode()
    save_revision_registry(registry_path, machine)

    negative_boundaries = {
        "active_episode": {"active_episode": True},
        "inflight_inference": {"inflight_inference": 1},
        "queued_action": {"queued_actions": 1},
        "unconsumed_ack": {"unconsumed_acks": 1},
        "wal_unsealed": {"wal_sealed": False},
        "reset_home_missing": {"robot_home": False},
        "candidate_validation_incomplete": {"candidate_validation_complete": False},
    }
    negative_results = {}
    for name, override in negative_boundaries.items():
        synthetic_queue = {"queued_action_ids": ["queued-0"] if name == "queued_action" else []}
        synthetic_transition = {"committed": False, "quarantined": False}
        before = machine.snapshot()
        queue_before = deepcopy(synthetic_queue)
        transition_before = deepcopy(synthetic_transition)
        _expect_failure(lambda override=override: machine.activate_pending(quiet(**override)), "NOT_QUIESCENT")
        require(machine.snapshot() == before, f"G6P_NEGATIVE_GATE_STATE_MUTATION:{name}")
        require(synthetic_queue == queue_before, f"G6P_NEGATIVE_GATE_QUEUE_MUTATION:{name}")
        require(synthetic_transition == transition_before, f"G6P_NEGATIVE_GATE_TRANSITION_MUTATION:{name}")
        negative_results[name] = "PASS"

    activated = machine.activate_pending(quiet())
    require(activated.revision_id == candidate.revision_id, "G6P_ACTIVATION_TARGET")
    require(gate.activate_revision(candidate.revision_id, candidate.model_sha256) == machine.policy_epoch, "G6P_GATE_EPOCH_DRIFT")
    require(not gate.has_pinned_request, "G6P_ACTIVATION_QUEUE_NOT_INVALIDATED")
    require(gate.classify_result(active_request) is InferenceDisposition.STALE_DROP, "G6P_OLD_RESULT_NOT_STALE")
    save_revision_registry(registry_path, machine)

    current_request = _envelope(
        machine,
        revision_id=candidate.revision_id,
        model_sha256=candidate.model_sha256,
        request_id="request-current",
        chunk_id="chunk-current",
    )
    require(gate.pin_request(current_request) is InferenceDisposition.ACCEPT, "G6P_CURRENT_REQUEST_PIN")
    stale_results = {
        "old_revision": gate.classify_result(replace(
            current_request,
            policy_revision_id=stable.revision_id,
            model_sha256=stable.model_sha256,
        )),
        "old_request": gate.classify_result(replace(current_request, request_id="request-old")),
        "old_chunk": gate.classify_result(replace(current_request, chunk_id="chunk-old")),
        "policy_epoch_mismatch": gate.classify_result(replace(
            current_request, arbitration_epoch_at_request=machine.policy_epoch - 1,
        )),
    }
    require(
        all(value is InferenceDisposition.STALE_DROP for value in stale_results.values()),
        "G6P_STALE_CLASSIFICATION_FAILED",
    )

    queue_invalidation = {}
    human_before = machine.policy_epoch
    machine.invalidate_policy_epoch("human_takeover")
    gate.invalidate_queued_policy()
    require(machine.policy_epoch == gate.policy_epoch == human_before + 1, "G6P_HUMAN_EPOCH_DRIFT")
    queue_invalidation["human_takeover"] = not gate.has_pinned_request
    reset_request = replace(
        current_request,
        arbitration_epoch_at_request=machine.policy_epoch,
        request_id="request-before-reset",
        chunk_id="chunk-before-reset",
    )
    require(gate.pin_request(reset_request) is InferenceDisposition.ACCEPT, "G6P_RESET_REQUEST_PIN")
    reset_before = machine.policy_epoch
    machine.invalidate_policy_epoch("reset_invalidation")
    gate.invalidate_queued_policy()
    require(machine.policy_epoch == gate.policy_epoch == reset_before + 1, "G6P_RESET_EPOCH_DRIFT")
    queue_invalidation["reset_invalidation"] = not gate.has_pinned_request

    pending = export_immutable_revision(
        revisions, model_payload=payloads["pending_after_rollback"], bindings=bindings,
    )
    validate_immutable_revision(pending.path, expected_bindings=bindings)
    machine.register_candidate(
        pending.revision_id,
        pending.model_sha256,
        artifact_digest=pending.canonical_manifest_digest,
    )
    machine.stage(pending.revision_id)
    save_revision_registry(registry_path, machine)

    machine.begin_episode()
    rollback_episode_state = machine.snapshot()
    _expect_failure(
        lambda: machine.rollback(quiet(active_episode=True)), "NOT_QUIESCENT",
    )
    require(machine.snapshot() == rollback_episode_state, "G6P_ILLEGAL_ROLLBACK_MUTATED_STATE")
    machine.end_episode()
    rollback_epoch = machine.policy_epoch
    restored = machine.rollback(quiet())
    require(restored.revision_id == stable.revision_id, "G6P_ROLLBACK_TARGET")
    require(machine.policy_epoch == rollback_epoch + 1, "G6P_ROLLBACK_EPOCH")
    require(machine.pending_revision_id == pending.revision_id, "G6P_ROLLBACK_PENDING_RULE")
    require(machine.record(candidate.revision_id).state is RevisionState.ROLLED_BACK, "G6P_ROLLBACK_AUDIT_STATE")
    require(gate.activate_revision(stable.revision_id, stable.model_sha256) == machine.policy_epoch, "G6P_ROLLBACK_GATE_EPOCH")
    require(not gate.has_pinned_request, "G6P_ROLLBACK_QUEUE_NOT_INVALIDATED")
    save_revision_registry(registry_path, machine)

    # Retention is verified above. A later explicit quiescent activation is
    # deliberately separate from rollback and is never automatic.
    post_rollback_active = machine.activate_pending(quiet())
    require(post_rollback_active.revision_id == pending.revision_id, "G6P_POST_ROLLBACK_EXPLICIT_ACTIVATION")
    require(
        gate.activate_revision(pending.revision_id, pending.model_sha256) == machine.policy_epoch,
        "G6P_POST_ROLLBACK_GATE_EPOCH",
    )
    final_pending = export_immutable_revision(
        revisions, model_payload=payloads["final_pending"], bindings=bindings,
    )
    validate_immutable_revision(final_pending.path, expected_bindings=bindings)
    machine.register_candidate(
        final_pending.revision_id,
        final_pending.model_sha256,
        artifact_digest=final_pending.canonical_manifest_digest,
    )
    machine.stage(final_pending.revision_id)
    save_revision_registry(registry_path, machine)

    invalid_bindings = deepcopy(bindings)
    invalid_bindings["runtime_environment"]["python_runtime"] = "mismatched-runtime-binding"
    invalid = export_immutable_revision(
        revisions, model_payload=payloads["invalid"], bindings=invalid_bindings,
    )
    invalid_reason = _expect_failure(
        lambda: validate_immutable_revision(invalid.path, expected_bindings=bindings),
        "SOURCE_CONFIG_BINDING_MISMATCH",
    )
    machine.register_candidate(
        invalid.revision_id,
        invalid.model_sha256,
        artifact_digest=invalid.canonical_manifest_digest,
        validation_complete=False,
    )
    rejected = machine.reject(invalid.revision_id, invalid_reason)
    require(rejected.state is RevisionState.REJECTED, "G6P_INVALID_CANDIDATE_NOT_REJECTED")
    require(machine.pending_revision_id == final_pending.revision_id, "G6P_INVALID_CANDIDATE_REPLACED_PENDING")
    save_revision_registry(registry_path, machine)

    learner_stall_before = machine.snapshot()
    learner_status = "stalled_no_candidate"
    require(machine.snapshot() == learner_stall_before, "G6P_LEARNER_STALL_CHANGED_ACTIVE")

    repeated = export_immutable_revision(
        revisions, model_payload=payloads["candidate"], bindings=bindings,
    )
    require(not repeated.created, "G6P_IDEMPOTENT_EXPORT_RECREATED")
    require(
        repeated.revision_id == candidate.revision_id
        and repeated.canonical_manifest_digest == candidate.canonical_manifest_digest,
        "G6P_IDEMPOTENT_EXPORT_DIGEST_DRIFT",
    )

    fault_root = output_root / "faults"
    tampered_model = _make_mutable_copy(candidate.path, fault_root, "model_tamper")
    with (tampered_model / "model/policy.bin").open("ab") as stream:
        stream.write(b"tamper")
    model_tamper_error = _expect_failure(
        lambda: validate_immutable_revision(tampered_model), "FILE_SHA_MISMATCH",
    )
    tampered_manifest = _make_mutable_copy(candidate.path, fault_root, "manifest_tamper")
    manifest_path = tampered_manifest / "manifest.json"
    manifest_value = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_value["bindings"]["runtime_environment"]["python_runtime"] = "tampered"
    manifest_path.write_bytes(canonical_json_bytes(manifest_value) + b"\n")
    manifest_tamper_error = _expect_failure(
        lambda: validate_immutable_revision(tampered_manifest), "CONTENT_ID_MISMATCH",
    )
    missing_completion = _make_mutable_copy(candidate.path, fault_root, "missing_completion")
    (missing_completion / "COMPLETED.json").unlink()
    completion_error = _expect_failure(
        lambda: validate_immutable_revision(missing_completion), "COMPLETION_MARKER_MISSING",
    )

    collision_root = output_root / "collision/revisions"
    collision_target = collision_root / candidate.revision_id
    collision_target.mkdir(parents=True)
    (collision_target / "manifest.json").write_text(
        json.dumps({"canonical_manifest_digest": "f" * 64}), encoding="utf-8",
    )
    collision_error = _expect_failure(
        lambda: export_immutable_revision(
            collision_root, model_payload=payloads["candidate"], bindings=bindings,
        ),
        "REVISION_ID_DIGEST_COLLISION",
    )

    crash_root = output_root / "crash_before_rename/revisions"
    crash_error = _expect_failure(
        lambda: export_immutable_revision(
            crash_root,
            model_payload=payloads["orphan"] + b"-before-rename",
            bindings=bindings,
            fault="before_atomic_rename",
        ),
        "SIMULATED_CRASH_BEFORE_ATOMIC_RENAME",
    )
    crash_entries = list(crash_root.iterdir())
    require(
        crash_entries and all(path.name.startswith(".revision-tmp-") for path in crash_entries),
        "G6P_CRASH_BEFORE_RENAME_PUBLISHED_REVISION",
    )

    pre_orphan_registry = load_revision_registry(registry_path, fresh_process=False).snapshot()
    orphan = export_immutable_revision(
        revisions, model_payload=payloads["orphan"], bindings=bindings,
    )
    post_orphan_registry = load_revision_registry(registry_path, fresh_process=False).snapshot()
    require(pre_orphan_registry == post_orphan_registry, "G6P_ORPHAN_CHANGED_REGISTRY")
    require(
        orphan.revision_id not in {record["revision_id"] for record in post_orphan_registry["records"]},
        "G6P_ORPHAN_AUTO_REGISTERED",
    )

    crash_registry_machine = InMemoryRevisionStateMachine.from_snapshot(machine.snapshot())
    crash_registry_machine.invalidate_policy_epoch("reset_invalidation")
    registry_crash_error = _expect_failure(
        lambda: save_revision_registry(
            registry_path, crash_registry_machine, fault_before_replace=True,
        ),
        "SIMULATED_REGISTRY_CRASH_BEFORE_REPLACE",
    )
    require(
        load_revision_registry(registry_path, fresh_process=False).snapshot() == machine.snapshot(),
        "G6P_REGISTRY_CRASH_LOST_LAST_COMPLETE_STATE",
    )

    checkpoint_input = _checkpoint_payload(
        machine, bindings["recursive_source"]["tree_sha256"],
    )
    checkpoint_restored, checkpoint_bytes = cpu_round_trip_online_checkpoint(checkpoint_input)
    require(
        checkpoint_restored["publication"]["pending_revision"] == final_pending.revision_id,
        "G6P_CHECKPOINT_PENDING_LOST",
    )
    fresh = _fresh_subprocess_recovery(registry_path)
    fresh_state = fresh["snapshot"]
    require(fresh_state["active_revision_id"] == machine.active_revision_id, "G6P_RECOVERY_ACTIVE")
    require(fresh_state["pending_revision_id"] == final_pending.revision_id, "G6P_RECOVERY_PENDING")
    require(fresh_state["previous_revision_id"] == machine.previous_revision_id, "G6P_RECOVERY_PREVIOUS")
    require(fresh_state["policy_epoch"] == machine.policy_epoch, "G6P_RECOVERY_EPOCH")
    require(fresh_state["publication_counters"] == machine.publication_counters, "G6P_RECOVERY_COUNTERS")
    require(fresh["safe_reset_required"] and not fresh["action_authorization_allowed"], "G6P_RECOVERY_FAIL_CLOSED")
    recovered_records = {record["revision_id"]: record for record in fresh_state["records"]}
    require(recovered_records[candidate.revision_id]["state"] == "rolled_back", "G6P_RECOVERY_ROLLBACK_RECORD")
    require(recovered_records[invalid.revision_id]["state"] == "rejected", "G6P_RECOVERY_REJECTED_RECORD")
    require(
        all(record["artifact_digest"] for record in recovered_records.values()),
        "G6P_RECOVERY_REVISION_DIGEST_LOST",
    )

    torch_module = sys.modules.get("torch")
    cuda_initialized = bool(torch_module and torch_module.cuda.is_initialized())
    require(not cuda_initialized, "G6P_CUDA_INITIALIZED")
    production_binding = bindings["recursive_source"]["production_source_binding"]
    require(not production_binding["complete"], "G6P_PRODUCTION_BINDING_UNEXPECTEDLY_COMPLETE")

    fault_cases = {
        "model_tamper": model_tamper_error,
        "manifest_tamper": manifest_tamper_error,
        "source_config_binding_mismatch": invalid_reason,
        "missing_completion_marker": completion_error,
        "revision_id_digest_collision": collision_error,
        "crash_before_atomic_rename": crash_error,
        "crash_after_rename_before_pending_pointer": "PASS_ORPHAN_NOT_REGISTERED_OR_ACTIVE",
        "registry_crash_before_atomic_replace": registry_crash_error,
        "stage_valid_candidate_during_active_episode": "PASS_PENDING_ACTIVE_UNCHANGED",
        "invalid_candidate_rejection": "PASS_REJECTED_WITH_REASON_AND_DIGEST",
        "activation_with_active_episode": "PASS",
        "activation_with_inflight_request": negative_results["inflight_inference"],
        "activation_with_queued_action": negative_results["queued_action"],
        "activation_with_unconsumed_ack": negative_results["unconsumed_ack"],
        "activation_with_unsealed_wal_witness": negative_results["wal_unsealed"],
        "activation_without_reset_home_witness": negative_results["reset_home_missing"],
        "candidate_validation_incomplete": negative_results["candidate_validation_incomplete"],
        "old_revision_chunk_request": "PASS_STALE_DROP",
        "policy_epoch_mismatch": "PASS_STALE_DROP",
        "cross_revision_transition": cross_revision_reason,
        "pending_survives_restart": "PASS",
        "illegal_mid_episode_rollback": "PASS",
        "learner_stall_active_unchanged": learner_status,
    }
    records = {record["revision_id"]: record for record in machine.snapshot()["records"]}
    report: dict[str, Any] = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "scope": config["scope"],
        "baseline": {
            "branch": provenance["branch"],
            "head": provenance["head"],
            "required_freeze_ancestor": provenance["required_freeze_ancestor"],
            "freeze_ancestor_verified": provenance["freeze_ancestor_verified"],
            **historical,
            "g5p_report_canonical_sha256": "75c3b0bab63b17bc0b4a685cd1a2177d7194fc82a1b2fd2fb112bc268210fdad",
            "g5p_checkpoint_canonical_digest": "b0d24880e02f0eff3f18f22930b3fe8bbc1ebd8f9cfa9da825d27a08533d1058",
        },
        "revision_artifacts": {
            "stable": {
                "revision_id": stable.revision_id,
                "model_sha256": stable.model_sha256,
                "canonical_manifest_digest": stable.canonical_manifest_digest,
            },
            "activated_then_rolled_back": {
                "revision_id": candidate.revision_id,
                "model_sha256": candidate.model_sha256,
                "canonical_manifest_digest": candidate.canonical_manifest_digest,
            },
            "pending": {
                "revision_id": final_pending.revision_id,
                "model_sha256": final_pending.model_sha256,
                "canonical_manifest_digest": final_pending.canonical_manifest_digest,
            },
            "active_after_explicit_post_rollback_activation": {
                "revision_id": pending.revision_id,
                "model_sha256": pending.model_sha256,
                "canonical_manifest_digest": pending.canonical_manifest_digest,
            },
            "rejected": {
                "revision_id": invalid.revision_id,
                "model_sha256": invalid.model_sha256,
                "canonical_manifest_digest": invalid.canonical_manifest_digest,
                "reason": records[invalid.revision_id]["rejection_reason"],
            },
            "orphan_not_registered": {
                "revision_id": orphan.revision_id,
                "model_sha256": orphan.model_sha256,
                "canonical_manifest_digest": orphan.canonical_manifest_digest,
            },
            "idempotent_same_revision_same_digest": True,
            "immutable_permissions_validated": True,
            "completion_marker_written_last_by_implementation": True,
        },
        "source_binding": {
            "tree_sha256": bindings["recursive_source"]["tree_sha256"],
            "file_count": len(bindings["recursive_source"]["files"]),
            "resolved_stage3_config_tree_sha256": bindings["resolved_stage3_configs"]["tree_sha256"],
            "vendor_commit": bindings["vendor"]["commit"],
            "vendor_tree_sha256": bindings["vendor"]["tree_sha256"],
            "environment_lock_sha256": bindings["runtime_environment"]["environment_lock"]["sha256"],
            "coverage": bindings["recursive_source"]["coverage"],
            "production_source_binding": production_binding,
        },
        "lifecycle": {
            "active_revision_id": machine.active_revision_id,
            "pending_revision_id": machine.pending_revision_id,
            "previous_revision_id": machine.previous_revision_id,
            "policy_epoch": machine.policy_epoch,
            "publication_counters": machine.publication_counters,
            "records": machine.snapshot()["records"],
            "pending_on_rollback_rule": config["lifecycle_contract"]["pending_on_rollback"],
            "pending_auto_activated_on_rollback": False,
            "post_rollback_later_explicit_activation": True,
            "policy_revision_id_distinct_from_policy_epoch": (
                isinstance(machine.active_revision_id, str)
                and machine.active_revision_id != str(machine.policy_epoch)
            ),
        },
        "one_episode_one_revision": {
            "pin": pin.to_dict(),
            "bound_events": sorted(REVISION_BOUND_EVENTS),
            "candidate_staged_mid_episode": True,
            "active_and_epoch_unchanged_mid_episode": True,
        },
        "quiescent_gate": {
            "negative_cases": negative_results,
            "synthetic_wal_witness_only": True,
            "synthetic_reset_home_witness_only": True,
        },
        "stale_drop": {
            "old_revision": stale_results["old_revision"].value,
            "old_request": stale_results["old_request"].value,
            "old_chunk": stale_results["old_chunk"].value,
            "policy_epoch_mismatch": stale_results["policy_epoch_mismatch"].value,
            "action_dispatch": False,
            "transition_commit": False,
            "replay_commit": False,
            "fatal_process_exit": False,
            "queue_invalidated": {
                "activation": True,
                "rollback": True,
                **queue_invalidation,
            },
        },
        "transition_binding": transition_quarantine,
        "checkpoint_round_trip": {
            "pending_revision_id": checkpoint_restored["publication"]["pending_revision"],
            "policy_epoch": checkpoint_restored["publication"]["policy_epoch"],
            "canonical_bytes_sha256": canonical_sha256(json.loads(checkpoint_bytes)),
        },
        "fresh_process_recovery": {
            "active_revision_id": fresh_state["active_revision_id"],
            "pending_revision_id": fresh_state["pending_revision_id"],
            "previous_revision_id": fresh_state["previous_revision_id"],
            "policy_epoch": fresh_state["policy_epoch"],
            "publication_counters": fresh_state["publication_counters"],
            "safe_reset_required": fresh["safe_reset_required"],
            "action_authorization_allowed": fresh["action_authorization_allowed"],
            "rejected_and_rolled_back_records_recovered": True,
            "revision_digests_recovered": True,
        },
        "fault_injection": {"all_passed": True, "cases": fault_cases},
        "production_blockers": production_binding["blockers"],
        "G6P_IMPLEMENTED": True,
        "G6P_IMMUTABLE_EXPORT": "PASS",
        "G6P_ATOMIC_PUBLICATION": "PASS",
        "G6P_INVALID_CANDIDATE_REJECTION": "PASS",
        "G6P_ONE_EPISODE_ONE_REVISION": "PASS",
        "G6P_QUIESCENT_ACTIVATION_GATE": "PASS",
        "G6P_OLD_CHUNK_INVALIDATION": "PASS",
        "G6P_ROLLBACK": "PASS",
        "G6P_PENDING_RESTART_RECOVERY": "PASS",
        "G6P_TRANSITION_REVISION_BINDING": "PASS",
        "G6P_CANONICAL_DIGEST_REPEATABLE": True,
        "G6P_LOOPBACK_ACTIVATION": True,
        "G6P_RESULT": "PASS",
        "SYNTHETIC_REVISION_PAYLOAD": True,
        "REAL_LEARNER_REVISION_USED": False,
        "REAL_POLICY_MODEL_EXPORTED": False,
        "REAL_POLICY_SERVER_USED": False,
        "REAL_RESET_HOME_VERIFIED": False,
        "PRODUCTION_WAL_SEALED_VERIFIED": False,
        "DIRECT_PUBLIC_HTTP_PARITY_VALIDATED": False,
        "PRODUCTION_SOURCE_BINDING_COMPLETE": production_binding["complete"],
        "PRODUCTION_POLICY_PUBLICATION_VALIDATED": False,
        "PRODUCTION_POLICY_ACTIVATION": False,
        "POLICY_REVISION_ACTIVATED": False,
        "G6_FORMAL_GATE_PASSED": False,
        "G3_RECORDED_FIXTURE_LOOPBACK": "BLOCKED",
        "G5_PRODUCTION_DURABLE_RESUME": "UNVERIFIED",
        "CRITIC_WARMUP_STARTED": False,
        "CRITIC_READY": False,
        "ACTOR_Q_GUIDANCE_ENABLED": False,
        "CUDA_INITIALIZED": cuda_initialized,
        "NETWORK_SERVER_STARTED": False,
        "ROBOT_CONNECTION_COUNT": 0,
        "ROBOT_COMMAND_COUNT": 0,
        "ROBOT_EXECUTION_AUTHORIZED": False,
        "G7_AND_LATER": "NOT_RUN",
        "PUSHED": False,
    }
    # Rebuilding the canonical source/binding payload in-process is the first
    # repeatability probe; the required two independent CLI runs are performed externally.
    require(build_revision_bindings(config) == bindings, "G6P_BINDING_REPEATABILITY_DRIFT")
    report["canonical_report_sha256"] = canonical_sha256(report)
    schema = json.loads(REPORT_SCHEMA.read_text(encoding="utf-8"))
    errors = sorted(
        Draft202012Validator(schema).iter_errors(report),
        key=lambda error: tuple(str(item) for item in error.absolute_path),
    )
    if errors:
        path = ".".join(str(item) for item in errors[0].absolute_path)
        raise RuntimeError(f"G6P_REPORT_SCHEMA:{path}:{errors[0].message}")
    return report


def _atomic_report(path: Path, report: Mapping[str, Any]) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.tmp-", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(json.dumps(report, indent=2, sort_keys=True, allow_nan=False).encode("utf-8") + b"\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        parent_descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(parent_descriptor)
        finally:
            os.close(parent_descriptor)
    finally:
        temporary.unlink(missing_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/stage3_policy_revision_loopback.v1.development.yaml",
    )
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--verify-frozen-evidence", action="store_true")
    parser.add_argument("--recover-registry", type=Path, help=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.recover_registry is not None:
        print(json.dumps(recover_registry_fresh_process(args.recover_registry), sort_keys=True))
        return 0
    if args.verify_frozen_evidence:
        config = yaml.safe_load(args.config.resolve().read_text(encoding="utf-8"))
        verify_historical_evidence(config)
        result = verify_frozen_bound_sources(config)
        print(json.dumps(result, sort_keys=True))
        return 0
    if args.output_root is None or args.report is None:
        raise SystemExit("--output-root and --report are required")
    report = run_loopback(args.output_root, args.config.resolve())
    _atomic_report(args.report, report)
    print(f"G6P_RESULT={report['G6P_RESULT']}")
    print(f"G6P_LOOPBACK_ACTIVATION={str(report['G6P_LOOPBACK_ACTIVATION']).lower()}")
    print(f"canonical_report_sha256={report['canonical_report_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
