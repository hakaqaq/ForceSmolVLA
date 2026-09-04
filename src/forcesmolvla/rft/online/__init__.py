"""Append-only online-replay training contracts and CPU primitives."""

from importlib import import_module
from typing import Any


_MODULE_EXPORTS = {
    "training_losses": (
        "ResidualActorLoss",
        "residual_actor_loss",
        "residual_critic_loss",
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
