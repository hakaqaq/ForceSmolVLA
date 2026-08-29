"""Append-only Stage-3 G1/G2 contracts and CPU primitives."""

from importlib import import_module
from typing import Any


_MODULE_EXPORTS = {
    "batch": ("MixedReplaySampler", "build_expert_feature_mask"),
    "checkpoint": (
        "cpu_round_trip_online_checkpoint",
        "validate_online_checkpoint_metadata",
    ),
    "contracts": (
        "apply_stage3_trainability",
        "load_stage3_contracts",
        "validate_stage3_contracts",
    ),
    "losses": (
        "compute_expert_only_flow_matching_loss",
        "compute_min_twin_q_guidance_from_values",
        "compute_online_twin_q_td_loss",
        "compute_stage3_actor_objective",
        "compute_stage3_min_twin_q_actor_loss",
    ),
    "learner": (
        "ProvisionalStage3Learner",
        "Stage3LossAPI",
        "TrainingStartsBlocked",
    ),
    "loopback": (
        "FakeActor",
        "FakeGateway",
        "canonical_report_sha256",
        "recorded_fixture_blocked_report",
        "run_synthetic_loopback",
        "validate_loopback_report",
    ),
    "parent": (
        "ParentBindingError",
        "load_parent_binding",
        "preflight_parent_binding",
        "validate_parent_binding_schema",
        "validate_parent_binding_semantics",
    ),
    "protocol": ("InferenceDisposition", "PolicyEpochGate", "TransportEnvelope"),
    "publication": (
        "EpisodeRevisionPin",
        "InMemoryRevisionStateMachine",
        "QuiescentBoundary",
        "RevisionArtifact",
        "RevisionRecord",
        "RevisionState",
        "export_immutable_revision",
        "load_revision_registry",
        "save_revision_registry",
        "validate_immutable_revision",
    ),
    "replay": ("D_EXPERT", "R_ONLINE", "Stage3Replay"),
    "transition": (
        "AcceptedAck",
        "causal_zoh_ack_macro",
        "finalize_ack_transition",
        "validate_ack_transition",
        "validate_episode_revision_bindings",
        "validate_reward_terminal",
    ),
    "update_credit": ("CreditsUnavailable", "UpdateCreditLedger"),
}
_EXPORTS = {
    name: module_name
    for module_name, names in _MODULE_EXPORTS.items()
    for name in names
}
__all__ = list(_EXPORTS)


def __getattr__(name: str) -> Any:
    try:
        module_name = _EXPORTS[name]
    except KeyError as error:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from error
    value = getattr(import_module(f"{__name__}.{module_name}"), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted((*globals(), *__all__))
