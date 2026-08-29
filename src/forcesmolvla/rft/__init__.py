"""Stage-2 offline RFT sidecars; the v4.2 Actor core remains unchanged."""

from importlib import import_module
from typing import Any


_EXPORTS = {
    "critic_action_for_q_guidance": "flow_sampling",
    "sample_normalized_action_chunk_with_grad": "flow_sampling",
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
