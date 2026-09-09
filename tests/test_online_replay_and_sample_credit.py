from __future__ import annotations

from copy import deepcopy

import pytest
import torch

from forcesmolvla.rft.online.training_batch import MixedReplaySampler, build_expert_feature_mask
from forcesmolvla.rft.online.replay import (
    D_EXPERT,
    R_ONLINE,
    ReplayDigestCollisionError,
    OnlineReplay,
)
from forcesmolvla.rft.online.transition_authority import (
    canonical_payload_sha256,
    finalize_ack_transition,
)
from forcesmolvla.rft.online.sample_credit import (
    CreditsUnavailable,
    TdCycleCreditLedger,
    UpdateCreditLedger,
)
from test_online_transition_authority import transition_payload


def test_R_D_membership_payload_dedupe_uid_and_credit_rules() -> None:
    ledger = UpdateCreditLedger(credits_per_transition=1, credits_per_joint_cycle=1)
    replay = OnlineReplay(max_online_transitions=10, credit_ledger=ledger)
    autonomous = finalize_ack_transition(transition_payload(macro_index=0))
    intervention = finalize_ack_transition(
        transition_payload(owner="human_intervention", macro_index=1)
    )
    offline = finalize_ack_transition(
        transition_payload(owner="offline_demonstration", macro_index=2)
    )
    first = replay.commit(autonomous, origin="online")
    second = replay.commit(intervention, origin="online")
    third = replay.commit(offline, origin="offline_demonstration")
    assert first.added_memberships == (R_ONLINE,) and first.credit_minted
    assert second.added_memberships == (R_ONLINE, D_EXPERT) and second.credit_minted
    assert third.added_memberships == (D_EXPERT,) and not third.credit_minted
    assert replay.canonical_payload_count == 3
    assert replay.audit()["dual_membership_count"] == 1
    duplicate = replay.commit(intervention, origin="online")
    assert duplicate.idempotent_noop and not duplicate.credit_minted
    assert ledger.snapshot().minted == 2

    collision = deepcopy(intervention)
    collision["outcome"]["reward"] = 1.0
    collision["integrity"]["canonical_payload_sha256"] = canonical_payload_sha256(collision)
    with pytest.raises(ReplayDigestCollisionError, match="COLLISION"):
        replay.commit(collision, origin="online")


def test_credits_block_at_zero_and_round_trip_exactly() -> None:
    ledger = UpdateCreditLedger(credits_per_transition=2, credits_per_joint_cycle=2)
    with pytest.raises(CreditsUnavailable, match="BLOCKED_NO_CREDITS"):
        ledger.consume_joint_cycle()
    assert ledger.mint_for_unique_online_transition("uid")
    assert not ledger.mint_for_unique_online_transition("uid")
    ledger.consume_joint_cycle()
    assert ledger.snapshot().available == 0
    restored = UpdateCreditLedger.from_state_dict(ledger.state_dict())
    assert restored.state_dict() == ledger.state_dict()


def test_cumulative_td_cycle_credit_rounds_only_after_global_total() -> None:
    ledger = TdCycleCreditLedger(new_td_rows_per_cycle=8)
    ledger.register_admission(
        admission_id="first",
        episode_id="000/episode",
        td_uids={f"a:{index}" for index in range(3)},
    )
    assert ledger.snapshot(completed_cycles=0).allowed_cycles == 0
    ledger.register_admission(
        admission_id="second",
        episode_id="001/episode",
        td_uids={f"b:{index}" for index in range(5)},
    )
    snapshot = ledger.snapshot(completed_cycles=0)
    assert snapshot.unique_td_rows == 8
    assert snapshot.allowed_cycles == snapshot.available_cycles == 1
    assert not ledger.register_admission(
        admission_id="second",
        episode_id="001/episode",
        td_uids={f"b:{index}" for index in range(5)},
    )
    restored = TdCycleCreditLedger.from_state_dict(ledger.state_dict())
    assert restored.state_dict() == ledger.state_dict()


def test_one_thousand_td_rows_allow_exactly_125_reserved_cycles() -> None:
    ledger = TdCycleCreditLedger(new_td_rows_per_cycle=8)
    for episode, count in (("a", 334), ("b", 333), ("c", 333)):
        ledger.register_admission(
            admission_id=episode,
            episode_id=f"{episode}/episode",
            td_uids={f"{episode}:{index}" for index in range(count)},
        )
    assert ledger.startup_ready(minimum_td_rows=1000, minimum_episodes=3)
    for completed in range(125):
        assert ledger.reserve_cycle(completed_cycles=completed)
        assert ledger.snapshot(completed_cycles=completed).in_flight_cycle == completed + 1
        ledger.complete_cycle(completed_cycle=completed + 1)
    exhausted = ledger.snapshot(completed_cycles=125)
    assert exhausted.allowed_cycles == 125
    assert exhausted.available_cycles == 0
    assert not ledger.reserve_cycle(completed_cycles=125)


def test_mixed_sampler_origin_and_expert_mask_prevent_R_self_imitation() -> None:
    replay = OnlineReplay(max_online_transitions=10)
    intervention = finalize_ack_transition(
        transition_payload(owner="human_intervention", macro_index=3)
    )
    replay.commit(intervention, origin="online")
    sampler = MixedReplaySampler(replay, seed=0)
    batch = sampler.sample(R_count=2, D_count=2)
    assert [sample.origin_pool for sample in batch.samples] == [R_ONLINE] * 2 + [D_EXPERT] * 2

    valid = torch.ones(4, 50, dtype=torch.bool)
    owners = [["human_intervention"] * 50 for _ in range(4)]
    mask = build_expert_feature_mask(
        valid, owners, [sample.origin_pool for sample in batch.samples]
    )
    assert torch.count_nonzero(mask[:2]) == 0
    assert torch.all(mask[2:])
    owners[2] = ["human_release_hold"] * 50
    owners[3] = ["safety_hold"] * 50
    held = build_expert_feature_mask(valid, owners, [R_ONLINE, R_ONLINE, D_EXPERT, D_EXPERT])
    assert torch.count_nonzero(held[2]) == 0
    assert torch.count_nonzero(held[3]) == 0
