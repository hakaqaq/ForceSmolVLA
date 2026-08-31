"""CPU-only immutable policy revision artifacts and lifecycle state."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, replace
from enum import Enum
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Mapping

from jsonschema import Draft202012Validator


ROOT = Path(__file__).parents[4]
REVISION_SCHEMA_PATH = ROOT / "schemas/stage3_policy_revision.v1.schema.json"
REVISION_SCHEMA_VERSION = "forcesmolvla_stage3_policy_revision.v1"
REGISTRY_SCHEMA_VERSION = "forcesmolvla_stage3_policy_revision_registry.v1"
REVISION_MANIFEST = "manifest.json"
REVISION_COMPLETION = "COMPLETED.json"


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_copy(value: Any) -> Any:
    return json.loads(canonical_json_bytes(value))


def _write_exclusive(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def _write_json_exclusive(path: Path, value: Any) -> None:
    _write_exclusive(path, canonical_json_bytes(value) + b"\n")


def _fsync_directories(root: Path) -> None:
    directories = [path for path in root.rglob("*") if path.is_dir()]
    for path in sorted(directories, key=lambda item: len(item.parts), reverse=True) + [root]:
        descriptor = os.open(path, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def _artifact_entries(root: Path) -> list[dict[str, Any]]:
    excluded = {REVISION_MANIFEST, REVISION_COMPLETION}
    return [
        {
            "relative_path": path.relative_to(root).as_posix(),
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name not in excluded
    ]


def _make_tree_read_only(root: Path) -> None:
    for path in root.rglob("*"):
        if path.is_file():
            path.chmod(0o444)
    for path in sorted(
        (item for item in root.rglob("*") if item.is_dir()),
        key=lambda item: len(item.parts),
        reverse=True,
    ):
        path.chmod(0o555)
    root.chmod(0o555)


def _remove_tree(path: Path) -> None:
    if not path.exists():
        return
    for item in path.rglob("*"):
        try:
            item.chmod(0o755 if item.is_dir() else 0o644)
        except FileNotFoundError:
            pass
    path.chmod(0o755)
    shutil.rmtree(path)


def _revision_identity(manifest: Mapping[str, Any]) -> dict[str, Any]:
    return {
        name: deepcopy(manifest[name])
        for name in (
            "schema_version",
            "artifact_kind",
            "synthetic_revision_payload",
            "model",
            "files",
            "files_tree_sha256",
            "bindings",
        )
    }


@dataclass(frozen=True)
class RevisionArtifact:
    revision_id: str
    model_sha256: str
    canonical_manifest_digest: str
    path: Path
    created: bool


class SimulatedPublicationCrash(RuntimeError):
    pass


def validate_immutable_revision(
    revision_directory: Path,
    *,
    expected_bindings: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate content, completion, identity, bindings, and read-only publication."""

    revision_directory = Path(revision_directory).resolve()
    completion_path = revision_directory / REVISION_COMPLETION
    manifest_path = revision_directory / REVISION_MANIFEST
    if not completion_path.is_file():
        raise RuntimeError("ONLINE_REPLAY_REVISION_COMPLETION_MARKER_MISSING")
    if not manifest_path.is_file():
        raise RuntimeError("ONLINE_REPLAY_REVISION_MANIFEST_MISSING")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        completion = json.loads(completion_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise RuntimeError("ONLINE_REPLAY_REVISION_JSON_INVALID") from error
    schema = json.loads(REVISION_SCHEMA_PATH.read_text(encoding="utf-8"))
    errors = sorted(
        Draft202012Validator(schema).iter_errors(manifest),
        key=lambda error: tuple(str(item) for item in error.absolute_path),
    )
    if errors:
        path = ".".join(str(item) for item in errors[0].absolute_path)
        raise RuntimeError(f"ONLINE_REPLAY_REVISION_SCHEMA:{path}:{errors[0].message}")
    entries = _artifact_entries(revision_directory)
    if entries != manifest["files"]:
        raise RuntimeError("ONLINE_REPLAY_REVISION_FILE_SHA_MISMATCH")
    if canonical_sha256(entries) != manifest["files_tree_sha256"]:
        raise RuntimeError("ONLINE_REPLAY_REVISION_FILE_TREE_MISMATCH")
    model_entries = [
        entry for entry in entries if entry["relative_path"].startswith("model/")
    ]
    model = manifest["model"]
    if (
        model_entries != model["files"]
        or canonical_sha256(model_entries) != model["tree_sha256"]
        or model_entries[0]["sha256"] != model["payload_sha256"]
    ):
        raise RuntimeError("ONLINE_REPLAY_REVISION_MODEL_TREE_MISMATCH")
    revision_id = canonical_sha256(_revision_identity(manifest))
    if revision_id != manifest["revision_id"] or revision_directory.name != revision_id:
        raise RuntimeError("ONLINE_REPLAY_REVISION_CONTENT_ID_MISMATCH")
    manifest_without_digest = deepcopy(manifest)
    manifest_without_digest.pop("canonical_manifest_digest")
    manifest_digest = canonical_sha256(manifest_without_digest)
    if manifest_digest != manifest["canonical_manifest_digest"]:
        raise RuntimeError("ONLINE_REPLAY_REVISION_MANIFEST_DIGEST_MISMATCH")
    expected_completion = {
        "schema_version": REVISION_SCHEMA_VERSION,
        "revision_id": revision_id,
        "manifest_sha256": sha256_file(manifest_path),
        "canonical_manifest_digest": manifest_digest,
        "complete": True,
    }
    if completion != expected_completion:
        raise RuntimeError("ONLINE_REPLAY_REVISION_COMPLETION_MARKER_INVALID")
    if expected_bindings is not None and manifest["bindings"] != _json_copy(expected_bindings):
        raise RuntimeError("ONLINE_REPLAY_REVISION_SOURCE_CONFIG_BINDING_MISMATCH")
    writable = [
        path.relative_to(revision_directory).as_posix()
        for path in [revision_directory, *revision_directory.rglob("*")]
        if path.stat().st_mode & 0o222
    ]
    if writable:
        raise RuntimeError(f"ONLINE_REPLAY_REVISION_NOT_IMMUTABLE:{writable[0]}")
    return {"manifest": manifest, "completion": completion}


def export_immutable_revision(
    revision_root: Path,
    *,
    model_payload: bytes,
    bindings: Mapping[str, Any],
    fault: str | None = None,
) -> RevisionArtifact:
    """Publish a content-addressed revision by same-filesystem atomic rename."""

    if not isinstance(model_payload, bytes) or not model_payload:
        raise ValueError("ONLINE_REPLAY_REVISION_MODEL_PAYLOAD_EMPTY")
    revision_root = Path(revision_root).resolve()
    revision_root.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".revision-tmp-", dir=revision_root))
    keep_temporary = False
    try:
        _write_exclusive(temporary / "model/policy.bin", model_payload)
        _write_json_exclusive(temporary / "bindings.json", _json_copy(bindings))
        entries = _artifact_entries(temporary)
        model_entries = [
            entry for entry in entries if entry["relative_path"].startswith("model/")
        ]
        identity = {
            "schema_version": REVISION_SCHEMA_VERSION,
            "artifact_kind": "isolated_immutable_policy_revision",
            "synthetic_revision_payload": True,
            "model": {
                "format": "deterministic_tiny_synthetic_bytes",
                "payload_sha256": model_entries[0]["sha256"],
                "tree_sha256": canonical_sha256(model_entries),
                "files": model_entries,
            },
            "files": entries,
            "files_tree_sha256": canonical_sha256(entries),
            "bindings": _json_copy(bindings),
        }
        revision_id = canonical_sha256(identity)
        manifest = {**identity, "revision_id": revision_id}
        manifest["canonical_manifest_digest"] = canonical_sha256(manifest)
        _write_json_exclusive(temporary / REVISION_MANIFEST, manifest)
        _fsync_directories(temporary)
        completion = {
            "schema_version": REVISION_SCHEMA_VERSION,
            "revision_id": revision_id,
            "manifest_sha256": sha256_file(temporary / REVISION_MANIFEST),
            "canonical_manifest_digest": manifest["canonical_manifest_digest"],
            "complete": True,
        }
        # Completion is deliberately created last, after every payload and manifest.
        _write_json_exclusive(temporary / REVISION_COMPLETION, completion)
        _fsync_directories(temporary)
        _make_tree_read_only(temporary)
        target = revision_root / revision_id
        if target.exists():
            existing_path = target / REVISION_MANIFEST
            try:
                existing = json.loads(existing_path.read_text(encoding="utf-8"))
            except (FileNotFoundError, json.JSONDecodeError, UnicodeDecodeError) as error:
                raise RuntimeError("ONLINE_REPLAY_REVISION_ID_DIGEST_COLLISION") from error
            if existing.get("canonical_manifest_digest") != manifest["canonical_manifest_digest"]:
                raise RuntimeError("ONLINE_REPLAY_REVISION_ID_DIGEST_COLLISION")
            validate_immutable_revision(target, expected_bindings=bindings)
            _remove_tree(temporary)
            return RevisionArtifact(
                revision_id,
                manifest["model"]["payload_sha256"],
                manifest["canonical_manifest_digest"],
                target,
                False,
            )
        if fault == "before_atomic_rename":
            keep_temporary = True
            raise SimulatedPublicationCrash("ONLINE_REPLAY_SIMULATED_CRASH_BEFORE_ATOMIC_RENAME")
        if fault is not None:
            raise ValueError(f"ONLINE_REPLAY_REVISION_FAULT_UNKNOWN:{fault}")
        os.replace(temporary, target)
        descriptor = os.open(revision_root, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        validate_immutable_revision(target, expected_bindings=bindings)
        return RevisionArtifact(
            revision_id,
            manifest["model"]["payload_sha256"],
            manifest["canonical_manifest_digest"],
            target,
            True,
        )
    except BaseException:
        if not keep_temporary:
            _remove_tree(temporary)
        raise


class RevisionState(str, Enum):
    CANDIDATE = "candidate"
    PENDING = "pending"
    ACTIVE = "active"
    PREVIOUS = "previous"
    REJECTED = "rejected"
    ROLLED_BACK = "rolled_back"


@dataclass(frozen=True)
class RevisionRecord:
    revision_id: str
    model_sha256: str
    state: RevisionState
    rejection_reason: str | None = None
    artifact_digest: str | None = None
    validation_complete: bool = True

    def validate(self) -> "RevisionRecord":
        if not self.revision_id:
            raise ValueError("ONLINE_REPLAY_REVISION_ID_EMPTY")
        for name, value in (
            ("MODEL", self.model_sha256),
            ("ARTIFACT", self.artifact_digest),
        ):
            if value is not None and (
                len(value) != 64 or any(char not in "0123456789abcdef" for char in value)
            ):
                raise ValueError(f"ONLINE_REPLAY_REVISION_{name}_SHA_INVALID")
        if self.state in {RevisionState.REJECTED, RevisionState.ROLLED_BACK}:
            if not self.rejection_reason:
                raise ValueError("ONLINE_REPLAY_REVISION_DISPOSITION_REASON_MISSING")
        elif self.rejection_reason is not None:
            raise ValueError("ONLINE_REPLAY_REVISION_UNEXPECTED_DISPOSITION_REASON")
        if not isinstance(self.validation_complete, bool):
            raise ValueError("ONLINE_REPLAY_REVISION_VALIDATION_STATE_INVALID")
        return self

    def to_dict(self) -> dict[str, Any]:
        return {
            "revision_id": self.revision_id,
            "model_sha256": self.model_sha256,
            "artifact_digest": self.artifact_digest,
            "state": self.state.value,
            "validation_complete": self.validation_complete,
            "rejection_reason": self.rejection_reason,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "RevisionRecord":
        return cls(
            revision_id=value["revision_id"],
            model_sha256=value["model_sha256"],
            state=RevisionState(value["state"]),
            rejection_reason=value.get("rejection_reason"),
            artifact_digest=value.get("artifact_digest"),
            validation_complete=value["validation_complete"],
        ).validate()


@dataclass(frozen=True)
class EpisodeRevisionPin:
    policy_revision_id: str
    model_sha256: str
    policy_epoch: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy_revision_id": self.policy_revision_id,
            "model_sha256": self.model_sha256,
            "policy_epoch": self.policy_epoch,
        }


@dataclass(frozen=True)
class QuiescentBoundary:
    active_episode: bool
    inflight_inference: int
    queued_actions: int
    unconsumed_acks: int
    robot_home: bool
    wal_sealed: bool
    candidate_validation_complete: bool = True

    def validate_for_activation(self) -> None:
        if min(self.inflight_inference, self.queued_actions, self.unconsumed_acks) < 0:
            raise ValueError("ONLINE_REPLAY_QUIESCENT_COUNTER_NEGATIVE")
        if (
            self.active_episode
            or self.inflight_inference != 0
            or self.queued_actions != 0
            or self.unconsumed_acks != 0
            or not self.robot_home
            or not self.wal_sealed
            or not self.candidate_validation_complete
        ):
            raise RuntimeError("ONLINE_REPLAY_REVISION_ACTIVATION_NOT_QUIESCENT")


_COUNTER_NAMES = (
    "candidate_registrations",
    "pending_publications",
    "activations",
    "rejections",
    "rollbacks",
    "epoch_invalidations",
)


class InMemoryRevisionStateMachine:
    """The single lifecycle primitive; registry persistence serializes this state."""

    def __init__(
        self,
        active: RevisionRecord,
        *,
        initial_epoch: int = 0,
        safe_reset_required: bool = False,
    ) -> None:
        active.validate()
        if active.state is not RevisionState.ACTIVE:
            raise ValueError("ONLINE_REPLAY_INITIAL_REVISION_NOT_ACTIVE")
        if initial_epoch < 0:
            raise ValueError("ONLINE_REPLAY_POLICY_EPOCH_INITIAL_STATE_INVALID")
        self._records = {active.revision_id: active}
        self.active_revision_id = active.revision_id
        self.pending_revision_id: str | None = None
        self.previous_revision_id: str | None = None
        self.episode_revision_id: str | None = None
        self.episode_model_sha256: str | None = None
        self.episode_policy_epoch: int | None = None
        self.policy_epoch = int(initial_epoch)
        self.publication_counters = {name: 0 for name in _COUNTER_NAMES}
        self.safe_reset_required = bool(safe_reset_required)

    def register_candidate(
        self,
        revision_id: str,
        model_sha256: str,
        *,
        artifact_digest: str | None = None,
        validation_complete: bool = True,
    ) -> RevisionRecord:
        if revision_id in self._records:
            existing = self._records[revision_id]
            if (
                existing.model_sha256 != model_sha256
                or existing.artifact_digest != artifact_digest
            ):
                raise RuntimeError("ONLINE_REPLAY_REVISION_ID_SHA_COLLISION")
            return existing
        record = RevisionRecord(
            revision_id,
            model_sha256,
            RevisionState.CANDIDATE,
            artifact_digest=artifact_digest,
            validation_complete=validation_complete,
        ).validate()
        self._records[revision_id] = record
        self.publication_counters["candidate_registrations"] += 1
        return record

    def stage(self, revision_id: str) -> RevisionRecord:
        record = self._records[revision_id]
        if record.state is not RevisionState.CANDIDATE:
            raise RuntimeError("ONLINE_REPLAY_ONLY_CANDIDATE_CAN_BE_STAGED")
        if not record.validation_complete:
            raise RuntimeError("ONLINE_REPLAY_CANDIDATE_VALIDATION_INCOMPLETE")
        if self.pending_revision_id is not None:
            raise RuntimeError("ONLINE_REPLAY_PENDING_REVISION_ALREADY_EXISTS")
        staged = replace(record, state=RevisionState.PENDING)
        self._records[revision_id] = staged
        self.pending_revision_id = revision_id
        self.publication_counters["pending_publications"] += 1
        return staged

    def reject(self, revision_id: str, reason: str) -> RevisionRecord:
        if not reason:
            raise ValueError("ONLINE_REPLAY_REVISION_REJECTION_REASON_EMPTY")
        record = self._records[revision_id]
        if record.state not in {RevisionState.CANDIDATE, RevisionState.PENDING}:
            raise RuntimeError("ONLINE_REPLAY_ACTIVE_OR_PREVIOUS_REVISION_CANNOT_BE_REJECTED")
        rejected = replace(
            record, state=RevisionState.REJECTED, rejection_reason=reason,
        ).validate()
        self._records[revision_id] = rejected
        if self.pending_revision_id == revision_id:
            self.pending_revision_id = None
        self.publication_counters["rejections"] += 1
        return rejected

    def _require_recovered_reset(self) -> None:
        if self.safe_reset_required:
            raise RuntimeError("ONLINE_REPLAY_SAFE_RESET_REQUIRED_AFTER_RECOVERY")

    def activate_pending(self, boundary: QuiescentBoundary) -> RevisionRecord:
        self._require_recovered_reset()
        boundary.validate_for_activation()
        if self.episode_revision_id is not None:
            raise RuntimeError("ONLINE_REPLAY_REVISION_ACTIVATION_DURING_EPISODE")
        if self.pending_revision_id is None:
            raise RuntimeError("ONLINE_REPLAY_NO_PENDING_REVISION")
        current = self._records[self.active_revision_id]
        pending = self._records[self.pending_revision_id]
        if not pending.validation_complete:
            raise RuntimeError("ONLINE_REPLAY_CANDIDATE_VALIDATION_INCOMPLETE")
        self._records[current.revision_id] = replace(current, state=RevisionState.PREVIOUS)
        activated = replace(pending, state=RevisionState.ACTIVE)
        self._records[activated.revision_id] = activated
        self.previous_revision_id = current.revision_id
        self.active_revision_id = activated.revision_id
        self.pending_revision_id = None
        self.policy_epoch += 1
        self.publication_counters["activations"] += 1
        return activated

    def begin_episode(self) -> str:
        self._require_recovered_reset()
        if self.episode_revision_id is not None:
            raise RuntimeError("ONLINE_REPLAY_EPISODE_ALREADY_ACTIVE")
        active = self._records[self.active_revision_id]
        self.episode_revision_id = active.revision_id
        self.episode_model_sha256 = active.model_sha256
        self.episode_policy_epoch = self.policy_epoch
        return self.episode_revision_id

    def episode_pin(self) -> EpisodeRevisionPin:
        if (
            self.episode_revision_id is None
            or self.episode_model_sha256 is None
            or self.episode_policy_epoch is None
        ):
            raise RuntimeError("ONLINE_REPLAY_NO_ACTIVE_EPISODE")
        return EpisodeRevisionPin(
            self.episode_revision_id,
            self.episode_model_sha256,
            self.episode_policy_epoch,
        )

    def assert_episode_revision(self, revision_id: str) -> None:
        if self.episode_revision_id is None or revision_id != self.episode_revision_id:
            raise RuntimeError("ONLINE_REPLAY_ONE_EPISODE_ONE_REVISION_VIOLATION")

    def assert_episode_binding(
        self, revision_id: str, model_sha256: str, policy_epoch: int,
    ) -> None:
        if self.episode_pin() != EpisodeRevisionPin(revision_id, model_sha256, policy_epoch):
            raise RuntimeError("ONLINE_REPLAY_ONE_EPISODE_ONE_REVISION_VIOLATION")

    def end_episode(self) -> None:
        if self.episode_revision_id is None:
            raise RuntimeError("ONLINE_REPLAY_NO_ACTIVE_EPISODE")
        self.episode_revision_id = None
        self.episode_model_sha256 = None
        self.episode_policy_epoch = None

    def rollback(
        self,
        boundary: QuiescentBoundary,
        *,
        reason: str = "simulated rollback at quiescent reset boundary",
    ) -> RevisionRecord:
        self._require_recovered_reset()
        boundary.validate_for_activation()
        if self.episode_revision_id is not None:
            raise RuntimeError("ONLINE_REPLAY_ROLLBACK_DURING_EPISODE")
        if self.previous_revision_id is None:
            raise RuntimeError("ONLINE_REPLAY_NO_PREVIOUS_REVISION")
        current = self._records[self.active_revision_id]
        previous = self._records[self.previous_revision_id]
        if previous.state is not RevisionState.PREVIOUS:
            raise RuntimeError("ONLINE_REPLAY_ROLLBACK_TARGET_NOT_PREVIOUS_STABLE")
        self._records[current.revision_id] = replace(
            current, state=RevisionState.ROLLED_BACK, rejection_reason=reason,
        ).validate()
        restored = replace(previous, state=RevisionState.ACTIVE)
        self._records[restored.revision_id] = restored
        # A pending candidate is retained but never auto-activated on rollback.
        self.previous_revision_id = None
        self.active_revision_id = restored.revision_id
        self.policy_epoch += 1
        self.publication_counters["rollbacks"] += 1
        return restored

    def invalidate_policy_epoch(self, reason: str) -> int:
        if reason not in {"human_takeover", "reset_invalidation"}:
            raise ValueError("ONLINE_REPLAY_POLICY_EPOCH_INVALIDATION_REASON")
        self.policy_epoch += 1
        self.publication_counters["epoch_invalidations"] += 1
        return self.policy_epoch

    def acknowledge_reset_boundary(self, boundary: QuiescentBoundary) -> None:
        boundary.validate_for_activation()
        self.safe_reset_required = False

    def acknowledge_synthetic_reset_boundary(self, boundary: QuiescentBoundary) -> None:
        self.acknowledge_reset_boundary(boundary)

    @property
    def action_authorization_allowed(self) -> bool:
        return not self.safe_reset_required

    def record(self, revision_id: str) -> RevisionRecord:
        return self._records[revision_id]

    def snapshot(self) -> dict[str, Any]:
        episode = None
        if self.episode_revision_id is not None:
            episode = self.episode_pin().to_dict()
        return {
            "schema_version": REGISTRY_SCHEMA_VERSION,
            "records": [
                self._records[key].to_dict() for key in sorted(self._records)
            ],
            "active_revision_id": self.active_revision_id,
            "pending_revision_id": self.pending_revision_id,
            "previous_revision_id": self.previous_revision_id,
            "episode_pin": episode,
            "policy_epoch": self.policy_epoch,
            "publication_counters": dict(self.publication_counters),
            "safe_reset_required": self.safe_reset_required,
        }

    @classmethod
    def from_snapshot(
        cls, value: Mapping[str, Any], *, fresh_process: bool = False,
    ) -> "InMemoryRevisionStateMachine":
        snapshot = _json_copy(value)
        required = {
            "schema_version",
            "records",
            "active_revision_id",
            "pending_revision_id",
            "previous_revision_id",
            "episode_pin",
            "policy_epoch",
            "publication_counters",
            "safe_reset_required",
        }
        if set(snapshot) != required or snapshot["schema_version"] != REGISTRY_SCHEMA_VERSION:
            raise RuntimeError("ONLINE_REPLAY_REVISION_REGISTRY_SCHEMA_INVALID")
        records = [RevisionRecord.from_dict(record) for record in snapshot["records"]]
        record_map = {record.revision_id: record for record in records}
        if len(record_map) != len(records) or not records:
            raise RuntimeError("ONLINE_REPLAY_REVISION_REGISTRY_RECORDS_INVALID")
        active_id = snapshot["active_revision_id"]
        pending_id = snapshot["pending_revision_id"]
        previous_id = snapshot["previous_revision_id"]
        if active_id not in record_map or record_map[active_id].state is not RevisionState.ACTIVE:
            raise RuntimeError("ONLINE_REPLAY_REVISION_REGISTRY_ACTIVE_INVALID")
        if pending_id is not None and (
            pending_id not in record_map or record_map[pending_id].state is not RevisionState.PENDING
        ):
            raise RuntimeError("ONLINE_REPLAY_REVISION_REGISTRY_PENDING_INVALID")
        if previous_id is not None and (
            previous_id not in record_map or record_map[previous_id].state is not RevisionState.PREVIOUS
        ):
            raise RuntimeError("ONLINE_REPLAY_REVISION_REGISTRY_PREVIOUS_INVALID")
        counters = snapshot["publication_counters"]
        if set(counters) != set(_COUNTER_NAMES) or any(
            not isinstance(count, int) or count < 0 for count in counters.values()
        ):
            raise RuntimeError("ONLINE_REPLAY_REVISION_REGISTRY_COUNTERS_INVALID")
        policy_epoch = snapshot["policy_epoch"]
        if not isinstance(policy_epoch, int) or policy_epoch < 0:
            raise RuntimeError("ONLINE_REPLAY_REVISION_REGISTRY_EPOCH_INVALID")
        machine = cls.__new__(cls)
        machine._records = record_map
        machine.active_revision_id = active_id
        machine.pending_revision_id = pending_id
        machine.previous_revision_id = previous_id
        episode = snapshot["episode_pin"]
        if episode is None:
            machine.episode_revision_id = None
            machine.episode_model_sha256 = None
            machine.episode_policy_epoch = None
        else:
            pin = EpisodeRevisionPin(**episode)
            if pin.policy_revision_id not in record_map:
                raise RuntimeError("ONLINE_REPLAY_REVISION_REGISTRY_EPISODE_PIN_INVALID")
            machine.episode_revision_id = pin.policy_revision_id
            machine.episode_model_sha256 = pin.model_sha256
            machine.episode_policy_epoch = pin.policy_epoch
        machine.policy_epoch = policy_epoch
        machine.publication_counters = counters
        machine.safe_reset_required = bool(snapshot["safe_reset_required"] or fresh_process)
        return machine


def save_revision_registry(
    registry_path: Path,
    machine: InMemoryRevisionStateMachine,
    *,
    fault_before_replace: bool = False,
) -> dict[str, Any]:
    """Atomically persist a standalone lifecycle registry/pointer."""

    registry_path = Path(registry_path).resolve()
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    state = machine.snapshot()
    payload = {
        "schema_version": REGISTRY_SCHEMA_VERSION,
        "state": state,
        "canonical_registry_digest": canonical_sha256(state),
    }
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{registry_path.name}.tmp-", dir=registry_path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(canonical_json_bytes(payload) + b"\n")
            stream.flush()
            os.fsync(stream.fileno())
        if fault_before_replace:
            raise SimulatedPublicationCrash("ONLINE_REPLAY_SIMULATED_REGISTRY_CRASH_BEFORE_REPLACE")
        os.replace(temporary, registry_path)
        parent_descriptor = os.open(registry_path.parent, os.O_RDONLY)
        try:
            os.fsync(parent_descriptor)
        finally:
            os.close(parent_descriptor)
        return payload
    finally:
        if temporary.exists() and not fault_before_replace:
            temporary.unlink()


def load_revision_registry(
    registry_path: Path, *, fresh_process: bool = True,
) -> InMemoryRevisionStateMachine:
    registry_path = Path(registry_path).resolve()
    try:
        payload = json.loads(registry_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, UnicodeDecodeError) as error:
        raise RuntimeError("ONLINE_REPLAY_REVISION_REGISTRY_INCOMPLETE") from error
    if set(payload) != {"schema_version", "state", "canonical_registry_digest"}:
        raise RuntimeError("ONLINE_REPLAY_REVISION_REGISTRY_SCHEMA_INVALID")
    if payload["schema_version"] != REGISTRY_SCHEMA_VERSION:
        raise RuntimeError("ONLINE_REPLAY_REVISION_REGISTRY_SCHEMA_INVALID")
    if payload["canonical_registry_digest"] != canonical_sha256(payload["state"]):
        raise RuntimeError("ONLINE_REPLAY_REVISION_REGISTRY_DIGEST_MISMATCH")
    return InMemoryRevisionStateMachine.from_snapshot(
        payload["state"], fresh_process=fresh_process,
    )
