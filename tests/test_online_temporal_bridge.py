from __future__ import annotations

import numpy as np
import pytest

from forcesmolvla.rft.online.transition_authority import (
    AcceptedAck,
    TransitionContractError,
    causal_zoh_ack_macro,
    normalized_ack_behavior_action,
    validate_macro_grid,
)


GRID = (133_333_333, 166_666_666, 200_000_000)


def ack(identifier: str, timestamp: int, value: float = 0.0) -> AcceptedAck:
    return AcceptedAck(
        ack_id=identifier,
        receive_monotonic_ns=timestamp,
        accepted_absolute_action7=(value, 0.0, 0.0, 0.0, 0.0, 0.0, 0.085),
        gripper_command_id=f"gripper-{identifier}",
        gripper_ack_command_id=f"gripper-{identifier}",
        slot_owner="policy",
        accepted_action_source="policy",
        intervention=False,
        source_command_id=f"pose-{identifier}",
        source_dispatch_sequence=0,
        source_model_index=0,
        episode_id="episode",
        policy_revision="revision",
        chunk_id="chunk",
        chunk_compatibility_key="generation-0",
        clock_domain="upper-host-monotonic",
        controller_authority="fr3-reference-controller",
    )


def test_same_ack_can_causally_zoh_to_three_30hz_slots() -> None:
    macro = causal_zoh_ack_macro([ack("a", 100_000_000)], GRID, max_ack_age_ms=101.0)
    assert macro.ack_ids == ("a", "a", "a")
    np.testing.assert_array_equal(macro.accepted_absolute_action_k7[:, 6], [0.085] * 3)
    normalized = normalized_ack_behavior_action(
        macro,
        anchor_state7=np.zeros(7),
        normalize_delta7=lambda value: value,
    )
    assert normalized.shape == (3, 7)
    assert np.isfinite(normalized).all()


def test_future_missing_rejected_and_300ms_interpretation_fails() -> None:
    with pytest.raises(TransitionContractError, match="MISSING_OR_STALE"):
        causal_zoh_ack_macro([ack("future", 150_000_000)], GRID, max_ack_age_ms=101.0)
    with pytest.raises(TransitionContractError, match="GRID_PHASE"):
        validate_macro_grid((100_000_000, 200_000_000, 300_000_000))


def test_out_of_order_or_rejected_ack_fails_closed() -> None:
    with pytest.raises(TransitionContractError, match="STRICTLY_INCREASING"):
        causal_zoh_ack_macro(
            [ack("late", 120_000_000), ack("early", 100_000_000)],
            GRID,
            max_ack_age_ms=101.0,
        )
    rejected = AcceptedAck(**{**ack("x", 100_000_000).__dict__, "accepted": False})
    with pytest.raises(TransitionContractError, match="ACK_REJECTED"):
        causal_zoh_ack_macro([rejected], GRID, max_ack_age_ms=101.0)
