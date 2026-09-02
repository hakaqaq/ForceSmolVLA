from __future__ import annotations

import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from forcesmolvla.rft import critic_action_adapter_v2 as adapter
from forcesmolvla.rft.online import temporal_parity, transition_authority
from forcesmolvla.rft.online.replay_training import HumanCorrectionReplay
from forcesmolvla.rft.online.training_losses import compute_online_twin_q_td_loss


ROOT = Path(__file__).parents[1]


def _symbol(module, name: str):
    assert hasattr(module, name), f"missing contract symbol: {name}"
    return getattr(module, name)


def _contract():
    cls = _symbol(adapter, "CriticActionContract")
    contract = _symbol(adapter, "CRITIC_ACTION_CONTRACT")
    assert isinstance(contract, cls)
    return contract


def _ack(
    ack_id: str,
    timestamp_ns: int,
    value: float,
    *,
    source: str = "human",
    episode: str = "episode",
    takeover: int = 1,
    reset: int = 0,
    revision: str = "revision",
    chunk: str = "chunk",
    compatibility: str = "authority-generation-1",
):
    AcceptedAck = transition_authority.AcceptedAck
    return AcceptedAck(
        ack_id=ack_id,
        receive_monotonic_ns=timestamp_ns,
        accepted_absolute_action7=(value,) * 6 + (0.085,),
        gripper_command_id=f"gripper-{ack_id}",
        gripper_ack_command_id=f"gripper-{ack_id}",
        slot_owner="human_intervention" if source == "human" else "policy",
        accepted_action_source=source,
        intervention=source == "human",
        source_command_id=f"pose-{ack_id}",
        source_dispatch_sequence=int(ack_id.rsplit("-", 1)[-1]),
        source_model_index=0,
        episode_id=episode,
        policy_revision=revision,
        takeover_generation=takeover,
        reset_generation=reset,
        chunk_id=chunk,
        chunk_compatibility_key=compatibility,
        clock_domain="upper-host-monotonic",
        controller_authority="fr3-reference-controller",
    )


def _macro(acks, *, source="human", boundary=None):
    build = _symbol(transition_authority, "build_ack_behavior_macro")
    return build(
        accepted_ack_stream=acks,
        anchor_timestamp_ns=1_000_000_000,
        action_source=source,
        contract=_contract(),
        max_ack_age_ms=_contract().max_ack_age_ms,
        boundary_timestamp_ns=boundary,
    )


def test_human_fm_target_is_not_reused_as_td_action() -> None:
    source = inspect.getsource(HumanCorrectionReplay.materialize)
    assert "action_target[:3]" not in source
    assert 'row["action_target"]' not in source
    assert 'row["human_action_target_h50"]' in source
    assert "human_behavior_action_k3" in source


def test_human_td_macro_uses_strict_30hz_grid() -> None:
    macro = _macro([_ack("ack-0", 999_000_000, 1.0)])
    assert macro.grid_monotonic_ns == (
        1_000_000_000,
        1_033_333_333,
        1_066_666_667,
    )


def test_human_td_macro_uses_causal_ack_zoh() -> None:
    macro = _macro([
        _ack("ack-0", 999_000_000, 1.0),
        _ack("ack-1", 1_050_000_000, 2.0),
    ])
    assert macro.ack_ids == ("ack-0", "ack-0", "ack-1")
    assert macro.source_command_ids == ("pose-ack-0", "pose-ack-0", "pose-ack-1")


def test_human_td_next_observation_is_exactly_100ms() -> None:
    ticks, next_tick = _symbol(adapter, "build_critic_transition_grid")(
        1_000_000_000, contract=_contract()
    )
    assert next_tick - ticks[0] == 100_000_000


def test_human_td_discount_matches_100ms_contract() -> None:
    contract = _contract()
    assert contract.macro_duration_ns == 100_000_000
    assert contract.gamma == 0.99


def test_human_td_preserves_pose_and_gripper_ack_identity() -> None:
    macro = _macro([_ack("ack-0", 999_000_000, 1.0)])
    assert macro.ack_ids == ("ack-0",) * 3
    assert macro.gripper_command_ids == macro.gripper_ack_command_ids


def test_policy_human_demo_share_critic_action_contract() -> None:
    contract = _contract()
    assert contract.action_sources == ("policy", "human", "offline_demonstration")
    assert transition_authority.CRITIC_ACTION_CONTRACT is contract


def test_actor_and_bootstrap_share_candidate_contract() -> None:
    actor = _symbol(adapter, "command_effective_candidate_action")
    bootstrap = _symbol(adapter, "bootstrap_command_effective_candidate_action")
    assert bootstrap is actor


def test_actor_candidate_respects_command_effective_phase() -> None:
    index_map = _symbol(adapter, "command_effective_execution_index_map")(
        contract=_contract(), anchor_timestamp_ns=1_000_000_000
    )
    assert index_map == (0, 0, 0)
    assert index_map != (0, 1, 2)


def test_policy_behavior_rejects_mid_macro_command_refresh() -> None:
    build = _symbol(transition_authority, "build_ack_behavior_macro")
    with pytest.raises(
        transition_authority.TransitionContractError,
        match="COMMAND_EFFECTIVE_PHASE_CHANGED",
    ):
        build(
            accepted_ack_stream=(
                _ack("ack-0", 999_000_000, 1.0, source="policy"),
                _ack("ack-1", 1_050_000_000, 2.0, source="policy"),
            ),
            anchor_timestamp_ns=1_000_000_000,
            action_source="policy",
            contract=_contract(),
            max_ack_age_ms=_contract().max_ack_age_ms,
            required_anchor_ack_id="ack-0",
        )


def test_previous_zoh_slots_receive_no_actor_gradient() -> None:
    project = _symbol(adapter, "command_effective_candidate_action")
    chunk = torch.randn(1, 50, 7, requires_grad=True)
    action = project(
        chunk,
        contract=_contract(),
        anchor_timestamp_ns=1_000_000_000,
        delta_action_mean7=torch.zeros(7),
        delta_action_std7=torch.ones(7),
    )
    action[..., :6].sum().backward()
    assert chunk.grad is not None
    assert torch.count_nonzero(chunk.grad[:, 1:, :6]) == 0


def test_gripper_receives_no_q_gradient() -> None:
    project = _symbol(adapter, "command_effective_candidate_action")
    chunk = torch.randn(1, 50, 7, requires_grad=True)
    project(
        chunk,
        contract=_contract(),
        anchor_timestamp_ns=1_000_000_000,
        delta_action_mean7=torch.zeros(7),
        delta_action_std7=torch.ones(7),
    ).sum().backward()
    assert torch.count_nonzero(chunk.grad[..., 6]) == 0


def test_tcp6_candidate_slots_receive_expected_gradient() -> None:
    project = _symbol(adapter, "command_effective_candidate_action")
    chunk = torch.randn(2, 50, 7, requires_grad=True)
    project(
        chunk,
        contract=_contract(),
        anchor_timestamp_ns=1_000_000_000,
        delta_action_mean7=torch.zeros(7),
        delta_action_std7=torch.ones(7),
    )[..., :6].sum().backward()
    assert torch.all(chunk.grad[:, 0, :6] == 3)


@pytest.mark.parametrize(
    ("field", "override"),
    [
        ("accepted_action_source", "policy"),
        ("takeover_generation", 2),
        ("reset_generation", 2),
        ("policy_revision", "other-revision"),
        ("chunk_compatibility_key", "other-authority"),
    ],
)
def test_macro_rejects_lineage_boundary(field: str, override) -> None:
    first = _ack("ack-0", 999_000_000, 1.0)
    values = dict(first.__dict__)
    values.update(
        ack_id="ack-1",
        receive_monotonic_ns=1_050_000_000,
        source_command_id="pose-ack-1",
        source_dispatch_sequence=1,
        **{field: override},
    )
    second = transition_authority.AcceptedAck(**values)
    with pytest.raises(transition_authority.TransitionContractError):
        _macro([first, second])


def test_macro_never_crosses_action_source() -> None:
    test_macro_rejects_lineage_boundary("accepted_action_source", "policy")


def test_macro_never_crosses_takeover_generation() -> None:
    test_macro_rejects_lineage_boundary("takeover_generation", 2)


def test_macro_never_crosses_reset_generation() -> None:
    test_macro_rejects_lineage_boundary("reset_generation", 2)


def test_macro_never_crosses_policy_revision() -> None:
    test_macro_rejects_lineage_boundary("policy_revision", "other-revision")


def test_macro_never_crosses_incompatible_chunk_boundary() -> None:
    test_macro_rejects_lineage_boundary("chunk_compatibility_key", "other-authority")


def test_terminal_partial_macro_mask_is_preserved() -> None:
    macro = _macro(
        [_ack("ack-0", 999_000_000, 1.0)], boundary=1_066_666_667
    )
    assert macro.behavior_mask == (True, True, False)
    assert np.count_nonzero(macro.accepted_absolute_action_k7[2]) == 0


class _MaskCritic(nn.Module):
    def forward(self, _observation, action, mask):
        return (action * mask[..., None]).sum(dim=(1, 2))


def test_critic_consumes_behavior_mask() -> None:
    critic = _MaskCritic()
    action = torch.ones(1, 3, 7)
    mask = torch.tensor([[True, True, False]])
    value = critic(None, action, mask)
    action[:, 2] = 10_000
    assert torch.equal(value, critic(None, action, mask))


def test_partial_macro_discount_and_bootstrap_are_correct() -> None:
    class _Never(nn.Module):
        def forward(self, *_args):
            raise AssertionError("partial terminal called target")

    critic = _MaskCritic()
    result = compute_online_twin_q_td_loss(
        q1=critic,
        q2=critic,
        q1_target=_Never(),
        q2_target=_Never(),
        observation=torch.zeros(1, 1),
        next_observation=torch.zeros(1, 1),
        ack_behavior_action_k7=torch.ones(1, 3, 7),
        behavior_mask=torch.tensor([[True, True, False]]),
        reward=torch.tensor([1.0]),
        discount=torch.tensor([0.0]),
        terminated=torch.tensor([True]),
        truncated=torch.tensor([False]),
        bootstrap_mask=torch.tensor([False]),
        next_policy_action_fn=lambda _observation: (_ for _ in ()).throw(
            AssertionError("partial terminal called Actor")
        ),
    )
    assert result.next_actor_calls == result.target_q1_calls == result.target_q2_calls == 0
    assert result.target.tolist() == [1.0]


def _recorded_report():
    runner = _symbol(temporal_parity, "run_p0a_recorded_live_parity")
    return runner()


def test_recorded_live_policy_ack_parity() -> None:
    report = _recorded_report()
    assert report["paths"]["policy_ack_behavior"] is True


def test_recorded_live_human_ack_parity() -> None:
    report = _recorded_report()
    assert report["paths"]["human_ack_behavior"] is True


def test_recorded_live_gripper_identity_parity() -> None:
    report = _recorded_report()
    assert report["comparisons"]["pose_and_gripper_ack_identity"] is True


def test_recorded_live_gate_rejects_synthetic_fixture(tmp_path: Path) -> None:
    gate = _symbol(temporal_parity, "p0a_formal_gate")
    synthetic = {
        "fixture_kind": "synthetic_unit_test",
        "synthetic": True,
        "recorded_live_evidence": False,
        "paths": {name: True for name in (
            "policy_ack_behavior", "human_ack_behavior", "offline_demo_behavior",
            "actor_candidate", "bootstrap_candidate",
        )},
    }
    assert gate(synthetic) == "BLOCKED"
