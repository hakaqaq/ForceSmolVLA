"""Append-only Stage-3 G1/G2 contracts and CPU primitives."""

from .batch import MixedReplaySampler, build_expert_feature_mask
from .checkpoint import cpu_round_trip_online_checkpoint, validate_online_checkpoint_metadata
from .contracts import apply_stage3_trainability, load_stage3_contracts, validate_stage3_contracts
from .losses import (
    compute_expert_only_flow_matching_loss,
    compute_min_twin_q_guidance_from_values,
    compute_online_twin_q_td_loss,
    compute_stage3_actor_objective,
    compute_stage3_min_twin_q_actor_loss,
)
from .learner import ProvisionalStage3Learner, Stage3LossAPI, TrainingStartsBlocked
from .loopback import (
    FakeActor,
    FakeGateway,
    canonical_report_sha256,
    recorded_fixture_blocked_report,
    run_synthetic_loopback,
    validate_loopback_report,
)
from .parent import (
    ParentBindingError,
    load_parent_binding,
    preflight_parent_binding,
    validate_parent_binding_schema,
    validate_parent_binding_semantics,
)
from .protocol import InferenceDisposition, PolicyEpochGate, TransportEnvelope
from .publication import (
    InMemoryRevisionStateMachine,
    QuiescentBoundary,
    RevisionRecord,
    RevisionState,
)
from .replay import D_EXPERT, R_ONLINE, Stage3Replay
from .transition import (
    AcceptedAck,
    causal_zoh_ack_macro,
    finalize_ack_transition,
    validate_ack_transition,
    validate_reward_terminal,
)
from .update_credit import CreditsUnavailable, UpdateCreditLedger


__all__ = [
    "AcceptedAck",
    "CreditsUnavailable",
    "D_EXPERT",
    "InferenceDisposition",
    "InMemoryRevisionStateMachine",
    "FakeActor",
    "FakeGateway",
    "MixedReplaySampler",
    "ParentBindingError",
    "PolicyEpochGate",
    "ProvisionalStage3Learner",
    "QuiescentBoundary",
    "R_ONLINE",
    "RevisionRecord",
    "RevisionState",
    "Stage3Replay",
    "Stage3LossAPI",
    "TrainingStartsBlocked",
    "TransportEnvelope",
    "UpdateCreditLedger",
    "apply_stage3_trainability",
    "build_expert_feature_mask",
    "causal_zoh_ack_macro",
    "canonical_report_sha256",
    "compute_expert_only_flow_matching_loss",
    "compute_min_twin_q_guidance_from_values",
    "compute_online_twin_q_td_loss",
    "compute_stage3_actor_objective",
    "compute_stage3_min_twin_q_actor_loss",
    "cpu_round_trip_online_checkpoint",
    "finalize_ack_transition",
    "load_stage3_contracts",
    "load_parent_binding",
    "preflight_parent_binding",
    "recorded_fixture_blocked_report",
    "run_synthetic_loopback",
    "validate_ack_transition",
    "validate_online_checkpoint_metadata",
    "validate_reward_terminal",
    "validate_loopback_report",
    "validate_parent_binding_schema",
    "validate_parent_binding_semantics",
    "validate_stage3_contracts",
]
