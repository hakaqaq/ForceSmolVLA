"""Deterministic CPU-only synthetic Actor/Learner loopback for G3P.

The synthetic fixture is a tool test and can never satisfy the recorded-live
G3 gate.  Nothing in this module imports a robot, ROS, network, publisher, or
checkpoint-weight loader.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Mapping

from jsonschema import Draft202012Validator
import numpy as np
import torch

from forcesmolvla.action_delta import ActionDeltaProcessor
from forcesmolvla.normalizer import CartesianNormalizerBundle, NormalizationLedger
from forcesmolvla.training_data import load_normalizer_manifest

from .contracts import validate_stage3_contracts
from .learner import ProvisionalStage3Learner, TrainingStartsBlocked
from .protocol import InferenceDisposition, PolicyEpochGate, TransportEnvelope
from .publication import (
    InMemoryRevisionStateMachine,
    QuiescentBoundary,
    RevisionRecord,
    RevisionState,
)
from .replay import D_EXPERT, R_ONLINE, ReplayDigestCollisionError, Stage3Replay
from .transition import (
    AcceptedAck,
    AckMacro,
    TransitionContractError,
    canonical_json_bytes,
    canonical_payload_sha256,
    causal_zoh_ack_macro,
    finalize_ack_transition,
    normalized_ack_behavior_action,
)
from .update_credit import CreditsUnavailable, UpdateCreditLedger


ROOT = Path(__file__).parents[4]
REPORT_SCHEMA_PATH = ROOT / "schemas/stage3_recorded_loopback_report.v1.schema.json"
DEFAULT_RECORDED_FIXTURE_PATH = ROOT / "golden_fixtures/stage3_recorded_ack_fixture.v1.json"
NORMALIZER_MANIFEST_PATH = (
    ROOT
    / "artifacts/development/stage2/stage2b_cycle210_evaluation_smoke_checkpoint.v1"
    / "manifests/normalizer_manifest.json"
)
ACTION_CONTRACT_PATH = (
    NORMALIZER_MANIFEST_PATH.parent / "stage2_action_contract.v2.development.json"
)
CALIBRATION_PATH = NORMALIZER_MANIFEST_PATH.parent / "calibration_bundle.development.json"
WRENCH_CONTRACT_PATH = NORMALIZER_MANIFEST_PATH.parent / "wrench_geometry_spec.development.json"

SHA_PLACEHOLDER = "0" * 64
ANCHOR_STATE7 = np.asarray([0.55, 0.0, 0.2, 0.0, 0.0, 0.0, 0.085], dtype=np.float64)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_json(value: Mapping) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value, dtype="<f8")
    digest = hashlib.sha256()
    digest.update(str(tuple(array.shape)).encode("ascii"))
    digest.update(array.view(np.uint8))
    return digest.hexdigest()


def _module_state_sha256(module: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def rational_grid_for_macro(macro_index: int) -> tuple[int, int, int]:
    """Return three 30 Hz ticks with a fixed 10 Hz anchor phase."""

    if macro_index < 0:
        raise ValueError("G3P_MACRO_INDEX_NEGATIVE")
    anchor_index = 30 + 3 * macro_index
    indices = np.arange(anchor_index, anchor_index + 3, dtype=np.int64)
    grid = (indices * 1_000_000_000 + 15) // 30
    return tuple(int(value) for value in grid)


@dataclass(frozen=True)
class FakeActionProposal:
    """Fake implementation of the real Actor transport/action interface."""

    envelope: TransportEnvelope
    normalized_action_h50: np.ndarray
    absolute_action_h50: np.ndarray
    action_h50_sha256: str
    flow_noise_sha256: str


class FakeActor:
    """Produces deterministic Hx7 chunks without claiming SmolVLA parity."""

    def __init__(self, normalizer: CartesianNormalizerBundle, *, seed: int) -> None:
        self.normalizer = normalizer
        self.seed = int(seed)
        self.model_sha256 = _sha256_json(
            {
                "implementation": "deterministic_fake_actor_interface",
                "seed": self.seed,
                "H": 50,
                "features": 7,
                "smolvla_forward_validated": False,
            }
        )

    def propose(self, *, macro_index: int, policy_epoch: int) -> FakeActionProposal:
        horizon = np.arange(50, dtype=np.float64)[:, None]
        feature = np.arange(7, dtype=np.float64)[None, :]
        normalized = 0.01 * np.sin(
            (horizon + 1.0) * (feature + 1.0) / 17.0 + macro_index * 0.001,
        )
        normalized[:, 6] = (
            0.085 - self.normalizer.delta_action7.mean[6]
        ) / self.normalizer.delta_action7.std[6]
        delta = self.normalizer.delta_action7.inverse(normalized)
        absolute = ActionDeltaProcessor.from_delta(delta, ANCHOR_STATE7)
        episode_id = f"synthetic-episode-{macro_index:04d}"
        envelope = TransportEnvelope(
            run_id="g3p-synthetic-run",
            session_id="g3p-synthetic-session",
            episode_id=episode_id,
            request_id=f"request-{macro_index:04d}",
            chunk_id=f"chunk-{macro_index:04d}",
            arbitration_epoch_at_request=policy_epoch,
            policy_revision_id="fake-active-r0",
            model_sha256=self.model_sha256,
            t_ref_monotonic_ns=rational_grid_for_macro(macro_index)[0],
            observation_id=f"observation-{macro_index:04d}",
        ).validate()
        return FakeActionProposal(
            envelope=envelope,
            normalized_action_h50=normalized,
            absolute_action_h50=absolute,
            action_h50_sha256=_array_sha256(absolute),
            flow_noise_sha256=_array_sha256(np.zeros((50, 7), dtype=np.float64)),
        )


@dataclass(frozen=True)
class GatewayOutcome:
    quarantined: bool
    quarantine_reason: str | None
    macro: AckMacro | None
    normalized_action_k7: np.ndarray | None
    target_normalized_action_h50: np.ndarray | None


class FakeGateway:
    """ACK-authoritative 30 Hz fake gateway with no external side effects."""

    def __init__(
        self,
        normalizer: CartesianNormalizerBundle,
        normalization_ledger: NormalizationLedger,
    ) -> None:
        self.normalizer = normalizer
        self.normalization_ledger = normalization_ledger
        self.epoch_gate = PolicyEpochGate(active_revision_id="fake-active-r0")
        self.positive_ack_count = 0
        self.quarantine_count = 0

    def begin_human_takeover(self) -> int:
        return self.epoch_gate.invalidate_queued_policy()

    def _owner_chunk(
        self,
        proposal: FakeActionProposal,
        owner: str,
        *,
        macro_index: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        if owner == "policy":
            normalized = proposal.normalized_action_h50.copy()
        else:
            # Human/offline targets are independent of the invalidated policy
            # chunk; the proposal remains only as observation/request evidence.
            horizon = np.arange(50, dtype=np.float64)[:, None]
            feature = np.arange(7, dtype=np.float64)[None, :]
            phase = 0.25 if owner == "human_intervention" else 0.5
            normalized = 0.015 * np.cos(
                (horizon + 1.0) * (feature + 1.0) / 19.0
                + macro_index * 0.001
                + phase,
            )
            normalized[:, 6] = (
                0.085 - self.normalizer.delta_action7.mean[6]
            ) / self.normalizer.delta_action7.std[6]
        delta = self.normalizer.delta_action7.inverse(normalized)
        return normalized, ActionDeltaProcessor.from_delta(delta, ANCHOR_STATE7)

    def _quarantine(self, reason: str) -> GatewayOutcome:
        self.quarantine_count += 1
        return GatewayOutcome(True, reason, None, None, None)

    def execute_policy_macro(
        self,
        proposal: FakeActionProposal,
        *,
        macro_index: int,
        fault: str | None = None,
    ) -> GatewayOutcome:
        if self.epoch_gate.classify_result(proposal.envelope) is InferenceDisposition.STALE_DROP:
            return self._quarantine("STAGE3_STALE_POLICY_CHUNK")
        return self._execute(proposal, macro_index=macro_index, owner="policy", fault=fault)

    def execute_human_macro(
        self,
        proposal: FakeActionProposal,
        *,
        macro_index: int,
    ) -> GatewayOutcome:
        return self._execute(
            proposal, macro_index=macro_index, owner="human_intervention", fault=None,
        )

    def execute_offline_demonstration(
        self,
        proposal: FakeActionProposal,
        *,
        macro_index: int,
    ) -> GatewayOutcome:
        return self._execute(
            proposal, macro_index=macro_index, owner="offline_demonstration", fault=None,
        )

    def _execute(
        self,
        proposal: FakeActionProposal,
        *,
        macro_index: int,
        owner: str,
        fault: str | None,
    ) -> GatewayOutcome:
        normalized_h50, absolute_h50 = self._owner_chunk(
            proposal, owner, macro_index=macro_index,
        )
        grid = rational_grid_for_macro(macro_index)
        source = (
            "human" if owner == "human_intervention" else
            "offline" if owner == "offline_demonstration" else "policy"
        )
        acknowledgements = []
        for slot, tick in enumerate(grid):
            received = tick - 1
            if fault == "stale_ack":
                received = tick - 1_000_000_000
            command_id = f"gripper-{macro_index:04d}-{slot}"
            acknowledgements.append(
                AcceptedAck(
                    ack_id=f"ack-positive-{macro_index:04d}-{slot}",
                    receive_monotonic_ns=received,
                    accepted_absolute_action7=tuple(float(v) for v in absolute_h50[slot]),
                    gripper_command_id=command_id,
                    gripper_ack_command_id=command_id,
                    slot_owner=owner,
                    accepted_action_source=source,
                    intervention=owner == "human_intervention",
                    accepted=fault != "rejected_ack",
                )
            )
        selected_grid = grid[:2] if fault == "partial_macro" else grid
        if fault == "missing_ack":
            acknowledgements = []
        try:
            macro = causal_zoh_ack_macro(
                acknowledgements, selected_grid, max_ack_age_ms=50.0,
            )

            def normalize_once(delta7: np.ndarray) -> np.ndarray:
                self.normalization_ledger.claim(
                    f"accepted-macro-{macro_index:04d}", "delta_action7",
                )
                return self.normalizer.delta_action7.apply(delta7)

            normalized_k7 = normalized_ack_behavior_action(
                macro, anchor_state7=ANCHOR_STATE7, normalize_delta7=normalize_once,
            )
        except (TransitionContractError, ValueError, RuntimeError) as error:
            return self._quarantine(str(error))
        if any(not ack_id.startswith("ack-positive-") for ack_id in macro.ack_ids):
            return self._quarantine("G3P_ACK_ID_NOT_POSITIVE")
        if macro.gripper_command_ids != macro.gripper_ack_command_ids:
            return self._quarantine("G3P_GRIPPER_ACK_IDENTITY_MISMATCH")
        self.positive_ack_count += 3
        return GatewayOutcome(False, None, macro, normalized_k7, normalized_h50)


def _observation(macro_index: int, *, next_value: bool) -> dict:
    grid = rational_grid_for_macro(macro_index)
    timestamp = grid[-1] + 1 if next_value else grid[0] - 1
    identifier = (
        f"next-observation-{macro_index:04d}"
        if next_value else f"observation-{macro_index:04d}"
    )
    episode_id = f"synthetic-episode-{macro_index:04d}"
    camera_sha = hashlib.sha256(identifier.encode("utf-8")).hexdigest()
    camera = {
        "blob_reference": f"synthetic:{identifier}",
        "sha256": camera_sha,
        "timestamp_monotonic_ns": timestamp,
        "age_ms": 0.0,
        "valid": True,
    }
    state = ANCHOR_STATE7.copy()
    if next_value:
        state[0] += 0.0001
    return {
        "observation_id": identifier,
        "episode_id": episode_id,
        "timestamp_monotonic_ns": timestamp,
        "state7": state.tolist(),
        "wrench6": [0.01 * (macro_index % 3)] * 6,
        "camera1": camera,
        "camera2": camera,
        "valid": True,
    }


def _transition_bindings() -> dict[str, str]:
    return {
        "action_contract_sha256": _sha256_file(ACTION_CONTRACT_PATH),
        "normalizer_sha256": _sha256_file(NORMALIZER_MANIFEST_PATH),
        "calibration_sha256": _sha256_file(CALIBRATION_PATH),
        "wrench_contract_sha256": _sha256_file(WRENCH_CONTRACT_PATH),
        "rulespec_sha256": _sha256_file(
            ROOT / "configs/stage3_transition_contract.v1.development.json",
        ),
        "reward_contract_sha256": _sha256_file(
            ROOT / "configs/stage3_reward_terminal_contract.v1.development.json",
        ),
        "deployment_binding_sha256": _sha256_file(
            ROOT / "configs/stage3_online_hil.v1.development.yaml",
        ),
        "source_tree_sha256": _sha256_file(
            ROOT / "src/forcesmolvla/rft/stage3/transition.py",
        ),
    }


def build_sealed_transition(
    proposal: FakeActionProposal,
    gateway: GatewayOutcome,
    *,
    macro_index: int,
    task: str = "synthetic-tool-test",
) -> dict:
    """Seal and validate one trainable transition through the real G1 API."""

    if gateway.quarantined or gateway.macro is None:
        raise TransitionContractError("G3P_QUARANTINED_MACRO_CANNOT_ENTER_REPLAY")
    macro = gateway.macro
    owner = macro.slot_owner[0]
    expert = owner in {"human_intervention", "offline_demonstration"}
    source = (
        "human" if owner == "human_intervention" else
        "offline" if owner == "offline_demonstration" else "policy"
    )
    payload = {
        "schema_version": "forcesmolvla_stage3_ack_transition.v1",
        "identity": {
            "run_id": proposal.envelope.run_id,
            "session_id": proposal.envelope.session_id,
            "episode_id": proposal.envelope.episode_id,
            "macro_index": macro_index,
            "task": task,
        },
        "bindings": _transition_bindings(),
        "observation": _observation(macro_index, next_value=False),
        "next_observation": _observation(macro_index, next_value=True),
        "policy_proposal": {
            "revision_id": proposal.envelope.policy_revision_id,
            "model_sha256": proposal.envelope.model_sha256,
            "policy_epoch": proposal.envelope.arbitration_epoch_at_request,
            "request_id": proposal.envelope.request_id,
            "chunk_id": proposal.envelope.chunk_id,
            "action_h50_sha256": proposal.action_h50_sha256,
            "flow_noise_sha256": proposal.flow_noise_sha256,
        },
        "behavior_ack": {
            "K": 3,
            "ack_ids": list(macro.ack_ids),
            "gripper_command_ids": list(macro.gripper_command_ids),
            "gripper_ack_command_ids": list(macro.gripper_ack_command_ids),
            "accepted_absolute_action_k7": macro.accepted_absolute_action_k7.tolist(),
            "normalized_delta_action_k7": gateway.normalized_action_k7.tolist(),
            "slot_owner": list(macro.slot_owner),
            "accepted_action_source": [source] * 3,
            "intervention_flags": [owner == "human_intervention"] * 3,
            "workspace_clip_flags": list(macro.workspace_clip_flags),
        },
        "fm_target": {
            "target_action_h50": gateway.target_normalized_action_h50.tolist(),
            "action_valid_mask_h50": [True] * 50,
            "expert_slot_mask_h50": [expert] * 50,
            "expert_feature_mask_h50x7": [[expert] * 7 for _ in range(50)],
        },
        "outcome": {
            "reward": float((macro_index % 5) * 0.01),
            "reward_revision": "g3p-synthetic-reward-v1",
            "terminated": False,
            "truncated": False,
            "bootstrap": True,
            "discount": 0.99,
        },
        "eligibility": {
            "critic_td_valid": True,
            "actor_q_valid": True,
            "expert_fm_available": expert,
            "quarantined": False,
            "quarantine_reason": None,
        },
        "commit": {
            "episode_sealed": True,
            "execution_event_sequence": macro_index,
            "ack_watermark": macro_index,
        },
    }
    return finalize_ack_transition(payload)


def canonical_report_sha256(report: Mapping) -> str:
    value = deepcopy(dict(report))
    value.pop("canonical_report_sha256", None)
    # Freeze metadata describes the evidence file, not the canonical tool run.
    value.pop("evidence_freeze", None)
    return _sha256_json(value)


def validate_loopback_report(report: Mapping) -> dict:
    value = deepcopy(dict(report))
    schema = json.loads(REPORT_SCHEMA_PATH.read_text(encoding="utf-8"))
    errors = sorted(
        Draft202012Validator(schema).iter_errors(value),
        key=lambda error: tuple(str(part) for part in error.absolute_path),
    )
    if errors:
        path = ".".join(str(part) for part in errors[0].absolute_path)
        raise ValueError(f"G3P_REPORT_SCHEMA:{path}:{errors[0].message}")
    if value["canonical_report_sha256"] != canonical_report_sha256(value):
        raise ValueError("G3P_REPORT_DIGEST_MISMATCH")
    evidence_freeze = value.get("evidence_freeze")
    if evidence_freeze is not None and evidence_freeze["canonical_report_digest"] != value[
        "canonical_report_sha256"
    ]:
        raise ValueError("G3P_FREEZE_DIGEST_ALIAS_MISMATCH")
    if value["fixture_kind"] == "synthetic_tool_test" and value["formal_gate_passed"]:
        raise ValueError("G3P_SYNTHETIC_CANNOT_PASS_FORMAL_GATE")
    return value


def _stage_fake_revision(learner: ProvisionalStage3Learner) -> dict:
    source_state_sha = _module_state_sha256(learner.actor)
    config = {
        "fixture_kind": "synthetic_tool_test",
        "actor": "TinyActor",
        "critic": "TinyTwinQ",
        "critic_gradient_steps": learner.critic_gradient_steps,
        "actor_gradient_steps": learner.actor_gradient_steps,
        "optimizer": "test_only_sgd",
    }
    config_sha = _sha256_json(config)
    revision_digest = _sha256_json(
        {
            "source_state_sha256": source_state_sha,
            "config_sha256": config_sha,
            "source_digest_algorithm": "canonical-json-plus-tensor-bytes-v1",
        }
    )
    revision_id = f"fake-policy-{revision_digest[:16]}"
    initial = RevisionRecord(
        "fake-active-r0", _sha256_json({"fake_active": 0}), RevisionState.ACTIVE,
    )
    machine = InMemoryRevisionStateMachine(initial)
    machine.register_candidate(revision_id, source_state_sha)
    staged = machine.stage(revision_id)
    machine.begin_episode()
    episode_blocked = False
    try:
        machine.activate_pending(
            QuiescentBoundary(True, 0, 0, 0, True, True),
        )
    except RuntimeError as error:
        episode_blocked = "NOT_QUIESCENT" in str(error) or "DURING_EPISODE" in str(error)
    machine.end_episode()
    inflight_blocked = False
    try:
        machine.activate_pending(
            QuiescentBoundary(False, 1, 0, 0, True, True),
        )
    except RuntimeError as error:
        inflight_blocked = "NOT_QUIESCENT" in str(error)
    if machine.active_revision_id != initial.revision_id:
        raise AssertionError("G3P_FAKE_REVISION_WAS_ACTIVATED")
    return {
        "staged": staged.state is RevisionState.PENDING,
        "activated": False,
        "active_revision_id": machine.active_revision_id,
        "staged_revision_id": revision_id,
        "source_state_sha256": source_state_sha,
        "config_sha256": config_sha,
        "revision_digest_sha256": revision_digest,
        "episode_activation_blocked": episode_blocked,
        "inflight_activation_blocked": inflight_blocked,
        "publisher_connected": False,
        "deployment_directory_written": False,
    }


def run_synthetic_loopback(*, seed: int = 20260828) -> dict:
    """Run the complete G3P tool test and return a canonical report."""

    contracts = validate_stage3_contracts()
    normalizer = load_normalizer_manifest(NORMALIZER_MANIFEST_PATH)
    normalization_ledger = NormalizationLedger()
    actor = FakeActor(normalizer, seed=seed)
    gateway = FakeGateway(normalizer, normalization_ledger)
    credits = UpdateCreditLedger(
        credits_per_transition=1,
        credits_per_joint_cycle=100,
    )
    replay = Stage3Replay(max_online_transitions=1000, credit_ledger=credits)
    learner = ProvisionalStage3Learner(
        replay=replay,
        credit_ledger=credits,
        delta_action_mean7=torch.tensor(normalizer.delta_action7.mean),
        delta_action_std7=torch.tensor(normalizer.delta_action7.std),
        seed=seed,
    )

    offline_index = 1000
    offline_proposal = actor.propose(
        macro_index=offline_index, policy_epoch=gateway.epoch_gate.policy_epoch,
    )
    offline_outcome = gateway.execute_offline_demonstration(
        offline_proposal, macro_index=offline_index,
    )
    replay.commit(
        build_sealed_transition(offline_proposal, offline_outcome, macro_index=offline_index),
        origin="offline_demonstration",
    )

    training_blocked_at_99 = False
    intervention_uid = None
    stale_takeover_outcome = None
    last_payload = None
    for macro_index in range(100):
        proposal = actor.propose(
            macro_index=macro_index, policy_epoch=gateway.epoch_gate.policy_epoch,
        )
        if macro_index == 42:
            gateway.begin_human_takeover()
            stale_takeover_outcome = gateway.execute_policy_macro(
                proposal, macro_index=macro_index,
            )
            accepted = gateway.execute_human_macro(proposal, macro_index=macro_index)
        else:
            accepted = gateway.execute_policy_macro(proposal, macro_index=macro_index)
        payload = build_sealed_transition(proposal, accepted, macro_index=macro_index)
        commit = replay.commit(payload, origin="online")
        if macro_index == 42:
            intervention_uid = commit.transition_uid
        if macro_index == 98:
            try:
                learner.assert_training_ready()
            except TrainingStartsBlocked:
                training_blocked_at_99 = True
        last_payload = payload

    learner.assert_training_ready()
    if stale_takeover_outcome is None or not stale_takeover_outcome.quarantined:
        raise AssertionError("G3P_TAKEOVER_DID_NOT_INVALIDATE_POLICY_CHUNK")
    if intervention_uid is None or intervention_uid not in replay.membership_uids(D_EXPERT):
        raise AssertionError("G3P_INTERVENTION_NOT_DUAL_MEMBERSHIP")

    duplicate = replay.commit(last_payload, origin="online")
    conflict = deepcopy(last_payload)
    conflict["outcome"]["reward"] += 1.0
    conflict["integrity"]["canonical_payload_sha256"] = canonical_payload_sha256(conflict)
    conflicting_digest_rejected = False
    try:
        replay.commit(conflict, origin="online")
    except ReplayDigestCollisionError:
        conflicting_digest_rejected = True

    fault_outcomes = {}
    for offset, fault in enumerate(
        ("partial_macro", "missing_ack", "rejected_ack", "stale_ack"), start=2000,
    ):
        proposal = actor.propose(
            macro_index=offset, policy_epoch=gateway.epoch_gate.policy_epoch,
        )
        fault_outcomes[fault] = gateway.execute_policy_macro(
            proposal, macro_index=offset, fault=fault,
        )
    if not all(value.quarantined for value in fault_outcomes.values()):
        raise AssertionError("G3P_ACK_FAULT_NOT_QUARANTINED")

    learner_result = learner.run_joint_cycle()
    zero_credit_backpressure = False
    try:
        credits.consume_joint_cycle()
    except CreditsUnavailable:
        zero_credit_backpressure = True
    publication = _stage_fake_revision(learner)
    replay_audit = replay.audit()
    normalizer_count = normalization_ledger.counts.get("delta_action7", 0)
    successful_macros = 101
    if normalizer_count != successful_macros:
        raise AssertionError("G3P_FROZEN_NORMALIZER_NOT_EXACTLY_ONCE")

    report = {
        "schema_version": "forcesmolvla_stage3_recorded_loopback_report.v1",
        "fixture_kind": "synthetic_tool_test",
        "fixture_path": None,
        "tool_status": "PASS",
        "blocked_reason": None,
        "formal_gate_passed": False,
        "robot_execution_authorized": False,
        "seed": seed,
        "source_bindings": {
            "normalizer_manifest_path": str(NORMALIZER_MANIFEST_PATH.relative_to(ROOT)),
            "normalizer_manifest_sha256": _sha256_file(NORMALIZER_MANIFEST_PATH),
            "action_contract_sha256": _sha256_file(ACTION_CONTRACT_PATH),
            "transition_source_sha256": _sha256_file(
                ROOT / "src/forcesmolvla/rft/stage3/transition.py",
            ),
            "replay_source_sha256": _sha256_file(
                ROOT / "src/forcesmolvla/rft/stage3/replay.py",
            ),
            "loss_source_sha256": _sha256_file(
                ROOT / "src/forcesmolvla/rft/stage3/losses.py",
            ),
            "publication_source_sha256": _sha256_file(
                ROOT / "src/forcesmolvla/rft/stage3/publication.py",
            ),
        },
        "collection": {
            "fake_actor_interface": True,
            "smolvla_forward_validated": False,
            "action_horizon": 50,
            "accepted_slots_per_decision": 3,
            "data_grid_hz": 30,
            "policy_anchor_hz": 10,
            "positive_ack_count": gateway.positive_ack_count,
            "gripper_command_ack_identity": True,
            "normalizer_application_count": normalizer_count,
            "successful_macro_count": successful_macros,
            "episode_sealed_before_commit": True,
        },
        "training_gate": {
            "training_starts_unique_R": 100,
            "blocked_at_99": training_blocked_at_99,
            "unlocked_at_100": len(replay.membership_uids(R_ONLINE)) == 100,
            "test_only_credit_configuration": {
                "credits_per_transition": 1,
                "credits_per_joint_cycle": 100,
                "production_policy": False,
            },
            "credits_after_cycle": credits.snapshot().available,
            "zero_credit_backpressure": zero_credit_backpressure,
        },
        "replay": {
            **replay_audit,
            "mixed_replay_ratio": "50_50",
            "intervention_dual_membership": intervention_uid in replay.membership_uids(R_ONLINE)
            and intervention_uid in replay.membership_uids(D_EXPERT),
            "independent_offline_demonstration": True,
        },
        "learner": learner_result,
        "fault_injection": {
            "duplicate_same_uid_same_digest_noop": duplicate.idempotent_noop,
            "conflicting_digest_fail_closed": conflicting_digest_rejected,
            "human_takeover_invalidated_stale_policy": stale_takeover_outcome.quarantined,
            "partial_macro_quarantined": fault_outcomes["partial_macro"].quarantined,
            "missing_ack_quarantined": fault_outcomes["missing_ack"].quarantined,
            "rejected_ack_quarantined": fault_outcomes["rejected_ack"].quarantined,
            "stale_ack_quarantined": fault_outcomes["stale_ack"].quarantined,
            "quarantined_fault_replay_commit_count": 0,
        },
        "policy_revision": publication,
        "deferred": {
            "G0_FINAL_PARENT_BINDING": contracts["G0_FINAL_PARENT_BINDING"],
            "CROSS_STAGE_OPTIMIZER_REBUILT": contracts["CROSS_STAGE_OPTIMIZER_REBUILT"],
            "G3_RECORDED_FIXTURE_LOOPBACK": "BLOCKED",
            "G4_AND_LATER": "NOT_RUN",
            "critic_ready": contracts["critic_ready"],
            "actor_q_guidance_enabled": contracts["actor_q_guidance_enabled"],
            "real_durable_production_WAL": "NOT_IMPLEMENTED",
            "real_publisher_server": "NOT_IMPLEMENTED",
            "online_HIL": "NOT_IMPLEMENTED",
        },
        "canonical_report_sha256": SHA_PLACEHOLDER,
    }
    report["canonical_report_sha256"] = canonical_report_sha256(report)
    return validate_loopback_report(report)


def recorded_fixture_blocked_report(fixture_path: Path | None = None) -> dict:
    """Return schema-valid BLOCKED for the non-authorized recorded-live mode."""

    path = fixture_path or DEFAULT_RECORDED_FIXTURE_PATH
    reason = (
        "RECORDED_LIVE_FIXTURE_MISSING"
        if not path.is_file()
        else "RECORDED_LIVE_GATE_NOT_AUTHORIZED_IN_G3P"
    )
    report = {
        "schema_version": "forcesmolvla_stage3_recorded_loopback_report.v1",
        "fixture_kind": "recorded_live",
        "fixture_path": str(path),
        "tool_status": "BLOCKED",
        "blocked_reason": reason,
        "formal_gate_passed": False,
        "robot_execution_authorized": False,
        "seed": None,
        "source_bindings": {},
        "collection": None,
        "training_gate": None,
        "replay": None,
        "learner": None,
        "fault_injection": None,
        "policy_revision": None,
        "deferred": {
            "G0_FINAL_PARENT_BINDING": "PENDING",
            "CROSS_STAGE_OPTIMIZER_REBUILT": "NOT_RUN",
            "G3_RECORDED_FIXTURE_LOOPBACK": "BLOCKED",
            "G4_AND_LATER": "NOT_RUN",
            "critic_ready": False,
            "actor_q_guidance_enabled": False,
            "real_durable_production_WAL": "NOT_IMPLEMENTED",
            "real_publisher_server": "NOT_IMPLEMENTED",
            "online_HIL": "NOT_IMPLEMENTED",
        },
        "canonical_report_sha256": SHA_PLACEHOLDER,
    }
    report["canonical_report_sha256"] = canonical_report_sha256(report)
    return validate_loopback_report(report)
