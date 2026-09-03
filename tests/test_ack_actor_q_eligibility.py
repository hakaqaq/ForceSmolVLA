from __future__ import annotations

from dataclasses import replace

import numpy as np

from forcesmolvla.rft.critic_action_adapter_v2 import CRITIC_ACTION_CONTRACT
from forcesmolvla.rft.online.transition_authority import (
    AckMacro,
    derive_actor_q_eligibility,
)


def _macro() -> AckMacro:
    return AckMacro(
        grid_monotonic_ns=(1, 2, 3),
        ack_ids=("ack", "ack", "ack"),
        gripper_command_ids=("gripper", "gripper", "gripper"),
        gripper_ack_command_ids=("gripper", "gripper", "gripper"),
        accepted_absolute_action_k7=np.zeros((3, 7), dtype=np.float64),
        slot_owner=("policy", "policy", "policy"),
        workspace_clip_flags=(False, False, False),
        source_command_ids=("command", "command", "command"),
        source_dispatch_sequences=(7, 7, 7),
        source_model_indices=(0, 0, 0),
        chunk_ids=("chunk", "chunk", "chunk"),
    )


def _eligible(macro: AckMacro, source: str = "policy"):
    return derive_actor_q_eligibility(
        macro=macro, action_source=source, quarantined=False
    )


def test_only_full_held_ack_macro_is_actor_q_eligible() -> None:
    assert _eligible(_macro()).valid
    assert not _eligible(
        replace(_macro(), source_command_ids=("a", "a", "b"))
    ).valid
    assert not _eligible(
        replace(_macro(), behavior_mask=(True, True, False))
    ).valid
    assert not _eligible(
        replace(_macro(), workspace_clip_flags=(False, True, False))
    ).valid


def test_offline_demonstration_is_never_actor_q_eligible() -> None:
    result = _eligible(_macro(), "offline_demonstration")
    assert not result.valid
    assert result.reason == "offline_demonstration_not_ack_deployment_semantics"
    assert result.contract_version == "ack_actor_q_v1"


def test_contract_mismatch_fails_closed() -> None:
    result = _eligible(
        replace(_macro(), contract_version=CRITIC_ACTION_CONTRACT.version + "-old")
    )
    assert not result.valid and result.reason == "critic_action_contract_mismatch"
