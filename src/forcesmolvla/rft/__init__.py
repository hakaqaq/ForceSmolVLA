"""Stage-2 offline RFT sidecars; the v4.2 Actor core remains unchanged."""

from .flow_sampling import (
    critic_action_for_q_guidance,
    sample_normalized_action_chunk_with_grad,
)

__all__ = ["critic_action_for_q_guidance", "sample_normalized_action_chunk_with_grad"]
