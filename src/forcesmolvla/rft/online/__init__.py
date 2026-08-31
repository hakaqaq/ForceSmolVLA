"""Append-only online-replay training contracts and CPU primitives."""

from importlib import import_module
from typing import Any


_MODULE_EXPORTS = {
    "training_batch": ("MixedReplaySampler", "build_expert_feature_mask"),
    "learner_checkpoint": (
        "cpu_round_trip_online_checkpoint",
        "validate_online_checkpoint_metadata",
    ),
    "training_contracts": (
        "apply_online_trainability",
        "load_online_contracts",
        "validate_online_contracts",
    ),
    "training_losses": (
        "compute_expert_only_flow_matching_loss",
        "compute_min_twin_q_guidance_from_values",
        "compute_online_twin_q_td_loss",
        "compute_online_actor_objective",
        "compute_online_min_twin_q_actor_loss",
    ),
    "learner": (
        "OnlineLearner",
        "OnlineLossAPI",
        "TrainingStartsBlocked",
    ),
    "policy_protocol": ("InferenceDisposition", "PolicyEpochGate", "TransportEnvelope"),
    "policy_revision": (
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
    "replay": ("D_EXPERT", "R_ONLINE", "OnlineReplay"),
    "transition_authority": (
        "AcceptedAck",
        "causal_zoh_ack_macro",
        "finalize_ack_transition",
        "validate_ack_transition",
        "validate_episode_revision_bindings",
        "validate_reward_terminal",
    ),
    "sample_credit": ("CreditsUnavailable", "UpdateCreditLedger"),
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
