"""Fail-closed shared recorder/policy-lineage capture entry contracts."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol

from .policy_lineage import (
    POLICY_LINEAGE_SCHEMA,
    UPPER_CLOCK_DOMAIN,
    PolicyLineageAudit,
    PolicyLineageError,
    PolicyResultLineage,
)


INTEGRATED_CAPTURE_SCHEMA = "forcesmolvla-stage3-integrated-capture-v1"
RECORDER_CONTROL_CHAIN = "franky_native_hilserl_cartesian_impedance"
RECORDER_ENTRY = Path(
    "/home/rlc123/fr3_client_ws/scripts/record_franka_hilserl_impedance.py"
)
CAPTURE_MODE_SEMANTICS: dict[str, dict[str, Any]] = {
    "shadow": {
        "actual_action_source": "human",
        "policy_inference": True,
        "policy_execution": False,
        "formal_replay": False,
        "real_online_r": False,
        "activation_authorized": True,
    },
    "policy-execute": {
        "actual_action_source": "policy",
        "policy_inference": True,
        "policy_execution": True,
        "formal_replay": False,
        "real_online_r": False,
        "activation_authorized": False,
        "unlock_requires": [
            "future_explicit_authorization",
            "verified_deployment_binding",
        ],
    },
}


class IntegratedCaptureError(RuntimeError):
    """A capture-mode, shared-lineage, seal, or controller isolation failure."""


def capture_mode_semantics(mode: str) -> dict[str, Any]:
    try:
        return dict(CAPTURE_MODE_SEMANTICS[mode])
    except KeyError as error:
        raise IntegratedCaptureError("INTEGRATED_CAPTURE_MODE_INVALID") from error


def _nonempty(value: Any, reason: str) -> str:
    text = str(value).strip()
    if not text:
        raise IntegratedCaptureError(reason)
    return text


def _counter(value: Any, reason: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise IntegratedCaptureError(reason)
    return value


@dataclass(frozen=True)
class SharedCaptureIdentity:
    session_id: str
    episode_id: str
    clock_domain_id: str
    policy_revision: str
    policy_epoch: int
    reset_generation: int
    takeover_generation: int

    def validate(self) -> "SharedCaptureIdentity":
        _nonempty(self.session_id, "INTEGRATED_CAPTURE_SESSION_ID_MISSING")
        _nonempty(self.episode_id, "INTEGRATED_CAPTURE_EPISODE_ID_MISSING")
        _nonempty(self.policy_revision, "INTEGRATED_CAPTURE_POLICY_REVISION_MISSING")
        if self.clock_domain_id != UPPER_CLOCK_DOMAIN:
            raise IntegratedCaptureError("INTEGRATED_CAPTURE_CLOCK_DOMAIN_INVALID")
        _counter(self.policy_epoch, "INTEGRATED_CAPTURE_POLICY_EPOCH_INVALID")
        _counter(self.reset_generation, "INTEGRATED_CAPTURE_RESET_GENERATION_INVALID")
        _counter(self.takeover_generation, "INTEGRATED_CAPTURE_TAKEOVER_GENERATION_INVALID")
        return self


@dataclass(frozen=True)
class IntegratedCaptureContract:
    mode: str
    identity: SharedCaptureIdentity
    actual_action_source: str
    policy_inference: bool
    policy_execution: bool
    formal_replay: bool
    real_online_r: bool
    controller_owner: str
    controller_process_count: int
    recorder_controller: bool
    deploy_controller: bool
    control_chain_id: str
    recorder_entry: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": INTEGRATED_CAPTURE_SCHEMA,
            **asdict(self),
        }


def build_capture_contract(
    *,
    mode: str,
    session_id: str,
    episode_id: str,
    policy_revision: str,
    policy_epoch: int,
    reset_generation: int,
    takeover_generation: int,
    deployment_binding: Path | None = None,
) -> IntegratedCaptureContract:
    """Build the only currently authorized mode; policy execution stays unreachable."""

    semantics = capture_mode_semantics(mode)
    if mode == "policy-execute":
        del deployment_binding
        raise IntegratedCaptureError(
            "POLICY_EXECUTE_HARD_DISABLED:"
            "future_explicit_authorization_and_deployment_binding_required"
        )
    identity = SharedCaptureIdentity(
        session_id=session_id,
        episode_id=episode_id,
        clock_domain_id=UPPER_CLOCK_DOMAIN,
        policy_revision=policy_revision,
        policy_epoch=policy_epoch,
        reset_generation=reset_generation,
        takeover_generation=takeover_generation,
    ).validate()
    return IntegratedCaptureContract(
        mode="shadow",
        identity=identity,
        actual_action_source=str(semantics["actual_action_source"]),
        policy_inference=bool(semantics["policy_inference"]),
        policy_execution=bool(semantics["policy_execution"]),
        formal_replay=bool(semantics["formal_replay"]),
        real_online_r=bool(semantics["real_online_r"]),
        controller_owner="recorder",
        controller_process_count=1,
        recorder_controller=True,
        deploy_controller=False,
        control_chain_id=RECORDER_CONTROL_CHAIN,
        recorder_entry=str(RECORDER_ENTRY),
    )


class IntegratedCaptureLedger:
    """One identity/clock/observation store shared by recorder and policy lineage."""

    def __init__(self, contract: IntegratedCaptureContract) -> None:
        if contract.mode != "shadow" or contract.policy_execution:
            raise IntegratedCaptureError("INTEGRATED_CAPTURE_LEDGER_MODE_NOT_AUTHORIZED")
        self.contract = contract
        identity = contract.identity
        self._policy = PolicyLineageAudit(
            episode_id=identity.episode_id,
            policy_revision=identity.policy_revision,
            reset_generation=identity.reset_generation,
            clock_domain_id=identity.clock_domain_id,
        )
        self._observations: dict[str, dict[str, Any]] = {}
        self._requests: dict[str, dict[str, Any]] = {}
        self._results: dict[str, PolicyResultLineage] = {}
        self._actual_acks: dict[str, dict[str, Any]] = {}
        self._sealed: dict[str, Any] | None = None

    def _shared(self) -> dict[str, Any]:
        return asdict(self.contract.identity)

    def _ensure_open(self) -> None:
        if self._sealed is not None:
            raise IntegratedCaptureError("INTEGRATED_CAPTURE_EPISODE_ALREADY_SEALED")

    def record_observation(
        self,
        *,
        observation_id: str,
        t_ref_ns: int,
        stream_timestamps_ns: Mapping[str, int],
        stream_ids: Mapping[str, str],
    ) -> dict[str, Any]:
        self._ensure_open()
        identity = _nonempty(observation_id, "INTEGRATED_CAPTURE_OBSERVATION_ID_MISSING")
        if t_ref_ns <= 0:
            raise IntegratedCaptureError("INTEGRATED_CAPTURE_OBSERVATION_TIME_INVALID")
        required = {
            "measured_tcp_pose", "wrench_notch_sensor", "gripper_state",
            "external_camera", "wrist_camera",
        }
        if set(stream_timestamps_ns) != required or set(stream_ids) != required:
            raise IntegratedCaptureError("INTEGRATED_CAPTURE_OBSERVATION_STREAM_SET_INVALID")
        timestamps = {name: int(value) for name, value in stream_timestamps_ns.items()}
        if any(value <= 0 or value > t_ref_ns for value in timestamps.values()):
            raise IntegratedCaptureError("INTEGRATED_CAPTURE_OBSERVATION_STREAM_TIME_INVALID")
        streams = {
            name: _nonempty(value, "INTEGRATED_CAPTURE_OBSERVATION_STREAM_ID_MISSING")
            for name, value in stream_ids.items()
        }
        record = {
            "schema": INTEGRATED_CAPTURE_SCHEMA,
            **self._shared(),
            "observation_id": identity,
            "t_ref_ns": int(t_ref_ns),
            "stream_timestamps_ns": timestamps,
            "stream_ids": streams,
        }
        previous = self._observations.get(identity)
        if previous is not None and previous != record:
            raise IntegratedCaptureError("INTEGRATED_CAPTURE_OBSERVATION_ID_CONFLICT")
        self._observations[identity] = record
        return dict(record)

    def record_policy_request(
        self,
        request: Mapping[str, Any],
        *,
        observation_id: str,
        recorded_monotonic_ns: int,
    ) -> dict[str, Any]:
        self._ensure_open()
        observation = self._observations.get(observation_id)
        if observation is None:
            raise IntegratedCaptureError("INTEGRATED_CAPTURE_POLICY_REQUEST_OBSERVATION_MISSING")
        provenance = request.get("provenance")
        if (
            not isinstance(provenance, Mapping)
            or int(provenance.get("t_ref_ns", 0)) != observation["t_ref_ns"]
        ):
            raise IntegratedCaptureError(
                "INTEGRATED_CAPTURE_POLICY_REQUEST_OBSERVATION_TIME_MISMATCH"
            )
        try:
            lineage = self._policy.record_request(
                request,
                policy_epoch=self.contract.identity.policy_epoch,
                takeover_generation=self.contract.identity.takeover_generation,
                recorded_monotonic_ns=recorded_monotonic_ns,
            )
        except PolicyLineageError as error:
            raise IntegratedCaptureError(str(error)) from error
        record = {
            "schema": POLICY_LINEAGE_SCHEMA,
            **self._shared(),
            **lineage,
            "observation_id": observation_id,
        }
        self._requests[lineage["request_id"]] = record
        return dict(record)

    def record_policy_result(
        self,
        request: Mapping[str, Any],
        result: Mapping[str, Any],
        *,
        recorded_monotonic_ns: int,
    ) -> dict[str, Any]:
        self._ensure_open()
        try:
            lineage = self._policy.record_result(
                request, result, recorded_monotonic_ns=recorded_monotonic_ns,
            )
        except PolicyLineageError as error:
            raise IntegratedCaptureError(str(error)) from error
        request_record = self._requests.get(lineage.request_id)
        if request_record is None:
            raise IntegratedCaptureError("INTEGRATED_CAPTURE_POLICY_RESULT_REQUEST_MISSING")
        record = {
            "schema": POLICY_LINEAGE_SCHEMA,
            **self._shared(),
            **lineage.selection_fields(),
            "observation_id": request_record["observation_id"],
            "shadow_proposal": True,
            "executed": False,
        }
        self._results[lineage.result_id] = lineage
        return record

    def record_actual_action_ack(
        self,
        *,
        ack_id: str,
        observation_id: str,
        receive_monotonic_ns: int,
        actual_action_source: str,
        policy_result_id: str | None = None,
        proposal_id: str | None = None,
    ) -> dict[str, Any]:
        self._ensure_open()
        identity = _nonempty(ack_id, "INTEGRATED_CAPTURE_ACK_ID_MISSING")
        observation = self._observations.get(observation_id)
        if observation is None:
            raise IntegratedCaptureError("INTEGRATED_CAPTURE_ACK_OBSERVATION_MISSING")
        if (
            actual_action_source != "human"
            or actual_action_source != self.contract.actual_action_source
        ):
            raise IntegratedCaptureError("INTEGRATED_CAPTURE_SHADOW_ACTUAL_SOURCE_NOT_HUMAN")
        if policy_result_id is not None or proposal_id is not None:
            raise IntegratedCaptureError("SHADOW_PROPOSAL_CANNOT_BIND_HUMAN_ACTION_ACK")
        if receive_monotonic_ns < int(observation["t_ref_ns"]):
            raise IntegratedCaptureError("INTEGRATED_CAPTURE_ACK_TIME_INVALID")
        record = {
            "schema": INTEGRATED_CAPTURE_SCHEMA,
            **self._shared(),
            "ack_id": identity,
            "observation_id": observation_id,
            "receive_monotonic_ns": int(receive_monotonic_ns),
            "actual_action_source": "human",
            "policy_result_id": None,
            "proposal_id": None,
            "policy_executed_transition": False,
            "formal_replay": False,
            "real_online_r": False,
        }
        previous = self._actual_acks.get(identity)
        if previous is not None and previous != record:
            raise IntegratedCaptureError("INTEGRATED_CAPTURE_ACK_ID_CONFLICT")
        self._actual_acks[identity] = record
        return dict(record)

    def seal_episode(
        self,
        *,
        seal_id: str,
        sealed_monotonic_ns: int,
        terminal_observation_id: str,
    ) -> dict[str, Any]:
        self._ensure_open()
        if not (
            self._observations
            and self._requests
            and self._results
            and self._actual_acks
        ):
            raise IntegratedCaptureError("INTEGRATED_CAPTURE_EPISODE_INCOMPLETE")
        if terminal_observation_id not in self._observations:
            raise IntegratedCaptureError("INTEGRATED_CAPTURE_TERMINAL_OBSERVATION_MISSING")
        request_ids = set(self._requests)
        result_request_ids = {lineage.request_id for lineage in self._results.values()}
        if request_ids != result_request_ids:
            raise IntegratedCaptureError("INTEGRATED_CAPTURE_UNSEALED_POLICY_LINEAGE")
        latest = max(
            [row["t_ref_ns"] for row in self._observations.values()]
            + [row["receive_monotonic_ns"] for row in self._actual_acks.values()]
        )
        if sealed_monotonic_ns < latest:
            raise IntegratedCaptureError("INTEGRATED_CAPTURE_SEAL_TIME_INVALID")
        self._sealed = {
            "schema": INTEGRATED_CAPTURE_SCHEMA,
            **self._shared(),
            "seal_id": _nonempty(seal_id, "INTEGRATED_CAPTURE_SEAL_ID_MISSING"),
            "sealed_monotonic_ns": int(sealed_monotonic_ns),
            "terminal_observation_id": terminal_observation_id,
            "observation_count": len(self._observations),
            "policy_request_count": len(self._requests),
            "policy_result_count": len(self._results),
            "human_action_ack_count": len(self._actual_acks),
            "actual_action_source": "human",
            "policy_inference": True,
            "policy_execution": False,
            "shadow_proposals_executed": False,
            "formal_replay": False,
            "real_online_r": False,
        }
        return dict(self._sealed)


@dataclass(frozen=True)
class CaptureBackendCapabilities:
    controller_owner: str
    controller_process_count: int
    starts_recorder_controller: bool
    starts_deploy_controller: bool
    control_chain_id: str
    shares_observation_store: bool
    emits_episode_seal: bool


class IntegratedCaptureBackend(Protocol):
    capabilities: CaptureBackendCapabilities

    def capture(
        self,
        *,
        contract: IntegratedCaptureContract,
        ledger: IntegratedCaptureLedger,
        recorder_arguments: Mapping[str, Any],
    ) -> Mapping[str, Any]: ...


def run_integrated_capture(
    *,
    contract: IntegratedCaptureContract,
    backend: IntegratedCaptureBackend,
    recorder_arguments: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Invoke one integrated backend only after proving single-controller ownership."""

    expected = CaptureBackendCapabilities(
        controller_owner="recorder",
        controller_process_count=1,
        starts_recorder_controller=True,
        starts_deploy_controller=False,
        control_chain_id=RECORDER_CONTROL_CHAIN,
        shares_observation_store=True,
        emits_episode_seal=True,
    )
    if backend.capabilities != expected:
        raise IntegratedCaptureError("INTEGRATED_CAPTURE_BACKEND_CAPABILITIES_INVALID")
    ledger = IntegratedCaptureLedger(contract)
    result = backend.capture(
        contract=contract, ledger=ledger, recorder_arguments=dict(recorder_arguments),
    )
    if not isinstance(result, Mapping) or result.get("seal_id") is None:
        raise IntegratedCaptureError("INTEGRATED_CAPTURE_BACKEND_EPISODE_SEAL_MISSING")
    return result


__all__ = [
    "CaptureBackendCapabilities",
    "CAPTURE_MODE_SEMANTICS",
    "IntegratedCaptureBackend",
    "IntegratedCaptureContract",
    "IntegratedCaptureError",
    "IntegratedCaptureLedger",
    "RECORDER_CONTROL_CHAIN",
    "RECORDER_ENTRY",
    "SharedCaptureIdentity",
    "build_capture_contract",
    "capture_mode_semantics",
    "run_integrated_capture",
]
