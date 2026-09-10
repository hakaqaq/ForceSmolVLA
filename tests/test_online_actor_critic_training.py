from __future__ import annotations

import inspect
from pathlib import Path
import sys
import threading
from types import SimpleNamespace
import warnings

import numpy as np
import pytest
import torch


ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import serve_forcerft_residual_actor_critic as learner_server  # noqa: E402

from forceprior.rft.online.residual_actor_critic_runtime import (
    InferencePriorityCoordinator,
    ResidualActorCriticSchedule,
)
from forceprior.rft.online.controller_acceptance import (
    CandidateGuardBatch,
    ControllerAcceptanceBatch,
    HILSERL_ACCEPTANCE_MAPPING_KIND,
    map_residual_to_controller_ack,
    policy_candidate_guard_valid,
)
from forceprior.rft.online.replay_training import (
    ACK_RESIDUAL_TRANSITION_SCHEMA_VERSION,
    LEGACY_ACK_RESIDUAL_TRANSITION_SCHEMA_VERSIONS,
    OnlineResidualReplay,
    ProductionAckMacro,
    algorithm_hyperparameters,
    load_common_actor_critic_config,
)
from forceprior.rft.online.transition_authority import (
    AckMacro,
    ActorQEligibility,
    DISPATCH_DECISION_CRITIC_CONTRACT_VERSION,
    ONLINE_SEMANTICS_VERSION,
    normalized_behavior_residual,
)
from forceprior.rft.critic import (
    RESIDUAL_ACTION_OFFSET,
    RESIDUAL_ACTION_WIDTH,
    build_twin_q,
)
from forceprior.rft.online.training_losses import (
    residual_actor_loss,
    residual_critic_loss,
)
from forceprior.rft.online.sample_credit import TdCycleCreditLedger
from forceprior.rft.residual_actor import (
    RESIDUAL_BOUND_MODE_SCALAR,
    make_residual_actor_pair,
    resolve_residual_cap6,
)


class ConstantQ(torch.nn.Module):
    def __init__(self, value: float) -> None:
        super().__init__()
        self.value = torch.nn.Parameter(torch.tensor(value))
        self.batch_sizes: list[int] = []
        self.residuals: list[torch.Tensor] = []
        self.grippers: list[torch.Tensor] = []

    def forward(
        self, state, wrench, wrench_delta, base, gripper, proposal
    ):
        del wrench, wrench_delta, base
        self.batch_sizes.append(len(state))
        self.residuals.append(proposal.detach().clone())
        self.grippers.append(gripper.detach().clone())
        return self.value.expand(len(state)) + proposal.mean(1) * 0.0


class TargetActor(torch.nn.Module):
    def __init__(self, value: float = 0.0) -> None:
        super().__init__()
        self.calls = 0
        self.value = float(value)

    def forward(self, **kwargs):
        self.calls += 1
        return torch.full(
            (len(kwargs["normalized_state7"]), 6), self.value
        )


class ScalarResidualActor(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.value = torch.nn.Parameter(torch.tensor(0.0))
        self.max_normalized_residual = 0.5
        self.batch_sizes: list[int] = []

    def forward(self, **kwargs):
        batch = len(kwargs["normalized_state7"])
        self.batch_sizes.append(batch)
        return self.value.expand(batch, 6)


class IdentityTransform:
    def __init__(self, width: int) -> None:
        self.mean = np.zeros(width, dtype=np.float64)
        self.std = np.ones(width, dtype=np.float64)

    def apply(self, value):
        return np.asarray(value)

    def inverse(self, value):
        return np.asarray(value)


def acceptance_mapping(
    *,
    decision_state7: object | None = None,
    valid: bool = True,
    step_dt_s: float = 1.0,
    filter_time_constant_s: float = 1.0e-3,
) -> dict[str, object]:
    state = np.asarray(
        [0.0] * 7 if decision_state7 is None else decision_state7,
        dtype=np.float64,
    )
    position = state[:3].tolist()
    return {
        "mapping_kind": (
            HILSERL_ACCEPTANCE_MAPPING_KIND if valid else "unavailable"
        ),
        "unavailable_reason": None if valid else "test_mapping_unavailable",
        "decision_state7": state.tolist(),
        "upper_execution_position_m": position,
        "upper_execution_quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
        "adapter_position_m": position,
        "adapter_quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
        "translation_action_scale_m": [1.0] * 3,
        "rotation_action_scale_rad": [1.0] * 3,
        "workspace_min_m": [-1.0] * 3,
        "workspace_max_m": [1.0] * 3,
        "filter_position_before_m": position,
        "filter_quaternion_before_xyzw": [0.0, 0.0, 0.0, 1.0],
        "actual_position_m": position,
        "actual_quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
        "step_dt_s": step_dt_s,
        "filter_time_constant_s": filter_time_constant_s,
        "translation_clip_positive_m": [1.0] * 3,
        "translation_clip_negative_m": [1.0] * 3,
        "rotation_clip_positive": [1.0] * 3,
        "rotation_clip_negative": [1.0] * 3,
        "policy_single_action_guard": {
            "workspace_min_xyz_m": [-1.0] * 3,
            "workspace_max_xyz_m": [1.0] * 3,
            "orientation_min_rpy_rad": [-np.pi] * 3,
            "orientation_max_rpy_rad": [np.pi] * 3,
            "gimbal_margin_rad": np.deg2rad(2.0),
            "gripper_width_m": float(state[6]),
            "gripper_min_width_m": 0.0,
            "gripper_max_width_m": 0.1,
            "continuity_max_xyz_m": 0.08,
            "continuity_max_rotation_rad": np.deg2rad(25.0),
            "continuity_max_gripper_delta_m": 0.1,
        },
    }


def acceptance_batch(
    batch_size: int, *, valid: bool = True
) -> ControllerAcceptanceBatch:
    def repeat(values: object) -> torch.Tensor:
        return torch.as_tensor(values, dtype=torch.float32).repeat(batch_size, 1)

    return ControllerAcceptanceBatch(
        valid=torch.full((batch_size,), valid, dtype=torch.bool),
        control_source=torch.zeros(batch_size, 1),
        decision_state7=repeat([[0.0] * 7]),
        upper_position3=repeat([[0.0] * 3]),
        upper_quaternion4=repeat([[0.0, 0.0, 0.0, 1.0]]),
        adapter_position3=repeat([[0.0] * 3]),
        adapter_quaternion4=repeat([[0.0, 0.0, 0.0, 1.0]]),
        translation_scale3=repeat([[1.0] * 3]),
        rotation_scale3=repeat([[1.0] * 3]),
        workspace_min3=repeat([[-1.0] * 3]),
        workspace_max3=repeat([[1.0] * 3]),
        filter_position_before3=repeat([[0.0] * 3]),
        filter_quaternion_before4=repeat([[0.0, 0.0, 0.0, 1.0]]),
        actual_position3=repeat([[0.0] * 3]),
        actual_quaternion4=repeat([[0.0, 0.0, 0.0, 1.0]]),
        step_dt_s=repeat([[1.0]]),
        filter_time_constant_s=repeat([[1.0e-3]]),
        translation_clip_positive3=repeat([[1.0] * 3]),
        translation_clip_negative3=repeat([[1.0] * 3]),
        rotation_clip_positive3=repeat([[1.0] * 3]),
        rotation_clip_negative3=repeat([[1.0] * 3]),
        policy_workspace_min3=repeat([[-1.0] * 3]),
        policy_workspace_max3=repeat([[1.0] * 3]),
        policy_orientation_min3=repeat([[-np.pi] * 3]),
        policy_orientation_max3=repeat([[np.pi] * 3]),
        policy_gimbal_margin_rad=repeat([[np.deg2rad(2.0)]]),
        policy_gripper_width_m=repeat([[0.0]]),
        policy_gripper_min_m=repeat([[0.0]]),
        policy_gripper_max_m=repeat([[0.1]]),
        policy_continuity_max_xyz_m=repeat([[0.08]]),
        policy_continuity_max_rotation_rad=repeat([[np.deg2rad(25.0)]]),
        policy_continuity_max_gripper_delta_m=repeat([[0.1]]),
        delta_action_mean6=repeat([[0.0] * 6]),
        delta_action_std6=repeat([[1.0] * 6]),
    )


def candidate_guard_batch(
    batch_size: int, *, valid: bool = True
) -> CandidateGuardBatch:
    return acceptance_batch(batch_size, valid=valid).candidate_guard()


def decision_context(
    *,
    timestamp_ns: int,
    base_absolute: object,
    acceptance_valid: bool = True,
) -> dict[str, object]:
    state = [0.0] * 7
    return {
        "online_semantics_version": ONLINE_SEMANTICS_VERSION,
        "valid_for_residual_training": True,
        "invalid_reason": None,
        "decision_monotonic_ns": timestamp_ns,
        "state7_absolute": state,
        "wrench6_calibrated_tcp": [0.0] * 6,
        "wrench_delta6_calibrated_tcp_100ms": [0.0] * 6,
        "wrench_delta_interval_ns": 0,
        "base_absolute_action7": base_absolute,
        "candidate_acceptance_mapping": acceptance_mapping(
            decision_state7=state,
            valid=acceptance_valid,
        ),
    }


def human_replay(
    *,
    terminated: bool = True,
    include_successor: bool = True,
    acceptance_valid: bool = True,
) -> OnlineResidualReplay:
    observation = {
        "state7_absolute": [0.0] * 7,
        "wrench6_calibrated_tcp": [0.0] * 6,
        "materialized_timestamp_monotonic_ns": 1_000_000_000,
    }
    next_observation = {
        **observation,
        "materialized_timestamp_monotonic_ns": 1_100_000_000,
    }
    accepted = np.repeat(
        np.asarray([[0.2, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0]]),
        3,
        axis=0,
    )
    behavior = AckMacro(
        grid_monotonic_ns=(1_000_000_000, 1_033_333_333, 1_066_666_667),
        ack_ids=("a", "a", "a"),
        gripper_command_ids=("g", "g", "g"),
        gripper_ack_command_ids=("g", "g", "g"),
        accepted_absolute_action_k7=accepted,
        slot_owner=("human_intervention",) * 3,
        workspace_clip_flags=(False,) * 3,
        contract_version=DISPATCH_DECISION_CRITIC_CONTRACT_VERSION,
    )
    base = [0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    transition = {
        "schema_version": ACK_RESIDUAL_TRANSITION_SCHEMA_VERSION,
        "online_semantics_version": ONLINE_SEMANTICS_VERSION,
        "identity": {"episode_id": "human-episode"},
        "action_source": "human",
        "observation": observation,
        "next_observation": next_observation,
        "outcome": {
            "reward": 1.0,
            "terminated": terminated,
            "truncated": False,
        },
        "eligibility": {"actor_q_valid": True, "critic_td_valid": False},
        "human_residual_valid": True,
        "pre_takeover_base_absolute_action7": base,
        "base_absolute_action_k7": np.repeat(
            np.asarray(base)[None, :], 3, axis=0
        ).tolist(),
        "accepted_absolute_action_k7": accepted.tolist(),
        "residual_decision_context": decision_context(
            timestamp_ns=1_000_000_000,
            base_absolute=base,
            acceptance_valid=acceptance_valid,
        ),
        "next_residual_decision_context": (
            None
            if terminated or not include_successor
            else decision_context(
                timestamp_ns=1_100_000_000,
                base_absolute=base,
                acceptance_valid=acceptance_valid,
            )
        ),
        "next_action_source": (
            None if terminated or not include_successor else "human"
        ),
        "next_accepted_absolute_action7": (
            None if terminated or not include_successor else accepted[0].tolist()
        ),
    }
    macro = ProductionAckMacro(
        transition=transition,
        behavior=behavior,
        next_grid_monotonic_ns=1_100_000_000,
        ack_provenance=(),
        actor_q_eligibility=ActorQEligibility(True, "valid"),
    )
    normalizer = SimpleNamespace(
        state7=IdentityTransform(7),
        wrench6=IdentityTransform(6),
        delta_action7=IdentityTransform(7),
    )
    return OnlineResidualReplay((macro,), normalizer)


def policy_replay(
    *,
    schema_version: str,
    base_action: object,
    selection_sequence: int = 7,
    selection_decision_ns: int = 1_000_000_000,
    selection_revision: str = "test-revision",
    proposal_present: bool = True,
    terminated: bool = True,
) -> OnlineResidualReplay:
    observation = {
        "state7_absolute": [0.0] * 7,
        "wrench6_calibrated_tcp": [0.0] * 6,
        "materialized_timestamp_monotonic_ns": 1_000_000_000,
    }
    accepted = np.repeat(
        np.asarray([[0.2, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0]]), 3, axis=0
    )
    transition = {
        "schema_version": schema_version,
        "online_semantics_version": (
            ONLINE_SEMANTICS_VERSION
            if schema_version == ACK_RESIDUAL_TRANSITION_SCHEMA_VERSION
            else None
        ),
        "identity": {"episode_id": "policy-episode", "decision_id": 7},
        "action_source": "policy",
        "observation": observation,
        "next_observation": {
            **observation,
            "materialized_timestamp_monotonic_ns": 1_100_000_000,
        },
        "outcome": {
            "reward": 0.0,
            "terminated": terminated,
            "truncated": False,
        },
        "eligibility": {"actor_q_valid": True, "critic_td_valid": True},
        "accepted_absolute_action_k7": accepted.tolist(),
        "next_residual_decision_context": None,
    }
    if base_action is not None:
        base_absolute = np.asarray(base_action, dtype=np.float64)[0].tolist()
        proposal = np.asarray([0.01] * 6, dtype=np.float64)
        normalized_base = np.repeat(
            np.asarray(base_absolute)[None, :], 3, axis=0
        )
        composed = normalized_base.copy()
        composed[:, :6] += proposal
        transition["base_absolute_action_k7"] = np.repeat(
            np.asarray(base_absolute)[None, :], 3, axis=0
        ).tolist()
        transition["base_normalized_action_k7"] = normalized_base.tolist()
        transition["applied_residual_tcp6"] = proposal.tolist()
        transition["composed_normalized_action_k7"] = composed.tolist()
        transition["policy_lineage"] = {
            "revision": "test-revision",
            "selection": {
                "sequence": selection_sequence,
                "policy_revision": selection_revision,
                "applied_residual_tcp6": proposal.tolist(),
                "base_normalized_action7": normalized_base[0].tolist(),
                "base_absolute_action7": base_absolute,
                "composed_normalized_action7": composed[0].tolist(),
                "residual_decision_context": {
                    "decision_monotonic_ns": selection_decision_ns
                },
            }
        }
        context = decision_context(
            timestamp_ns=1_000_000_000,
            base_absolute=base_absolute,
        )
        context["base_normalized_action6"] = normalized_base[0, :6].tolist()
        transition["residual_decision_context"] = context
        if not proposal_present:
            transition.pop("applied_residual_tcp6")
    else:
        transition["controller_normalized_action_k7"] = accepted.tolist()
        transition["composed_normalized_action_k7"] = accepted.tolist()
    macro = ProductionAckMacro(
        transition=transition,
        behavior=AckMacro(
            grid_monotonic_ns=(1_000_000_000, 1_033_333_333, 1_066_666_667),
            ack_ids=("a", "a", "a"),
            gripper_command_ids=("g", "g", "g"),
            gripper_ack_command_ids=("g", "g", "g"),
            accepted_absolute_action_k7=accepted,
            slot_owner=("policy",) * 3,
            workspace_clip_flags=(False,) * 3,
            contract_version=DISPATCH_DECISION_CRITIC_CONTRACT_VERSION,
        ),
        next_grid_monotonic_ns=1_100_000_000,
        ack_provenance=(),
        actor_q_eligibility=ActorQEligibility(True, "valid"),
    )
    normalizer = SimpleNamespace(
        state7=IdentityTransform(7),
        wrench6=IdentityTransform(6),
        delta_action7=IdentityTransform(7),
    )
    return OnlineResidualReplay((macro,), normalizer)


def batch(batch_size: int = 2) -> SimpleNamespace:
    zeros7 = torch.zeros(batch_size, 7)
    zeros6 = torch.zeros(batch_size, 6)
    zeros_k6 = torch.zeros(batch_size, 3, 6)
    mask = torch.ones(batch_size, 3, dtype=torch.bool)
    return SimpleNamespace(
        state7=zeros7,
        wrench6=zeros6,
        wrench_delta6=zeros6,
        base_action6=zeros6,
        base_gripper=torch.zeros(batch_size, 1),
        behavior_proposal6=zeros6,
        candidate_guard=candidate_guard_batch(batch_size),
        next_state7=zeros7,
        next_wrench6=zeros6,
        next_wrench_delta6=zeros6,
        next_base_action6=zeros6,
        next_base_gripper=torch.zeros(batch_size, 1),
        next_base_valid=torch.ones(batch_size, dtype=torch.bool),
        next_candidate_guard=candidate_guard_batch(batch_size),
        reward=torch.ones(batch_size),
        terminated=torch.tensor([False, True][:batch_size]),
        truncated=torch.zeros(batch_size, dtype=torch.bool),
        actor_q_valid=torch.ones(batch_size, dtype=torch.bool),
        human_residual_target6=zeros6,
        human_residual_valid=torch.zeros(batch_size, dtype=torch.bool),
    )


def test_residual_cap6_uses_frozen_normalizer_and_zero_actor_is_exact() -> None:
    std = np.asarray([0.02, 0.005, 0.01, 0.1, 0.01, 0.001, 1.0])
    normalizer = SimpleNamespace(
        delta_action7=SimpleNamespace(std=std)
    )
    config = {
        "max_normalized_residual": 0.1,
        "residual_bound_mode": RESIDUAL_BOUND_MODE_SCALAR,
        "max_translation_residual_per_axis_m": 0.001,
        "max_rpy_residual_per_axis_rad": np.deg2rad(0.5),
    }
    cap6 = resolve_residual_cap6(normalizer, config)
    actor, target = make_residual_actor_pair(
        hidden_dim=16,
        max_normalized_residual=0.1,
        residual_cap6=cap6,
        residual_bound_mode=RESIDUAL_BOUND_MODE_SCALAR,
    )
    inputs = {
        "normalized_state7": torch.randn(5, 7),
        "normalized_wrench6": torch.randn(5, 6),
        "normalized_wrench_delta6": torch.randn(5, 6),
        "base_action6": torch.randn(5, 6),
    }
    assert torch.equal(actor(**inputs), torch.zeros(5, 6))
    assert torch.equal(actor.residual_cap6, target.residual_cap6)
    assert torch.equal(cap6, cap6[0].expand_as(cap6))
    physical = cap6.numpy() * std[:6]
    assert np.all(physical[:3] <= 0.001 + 1e-9)
    assert np.all(physical[3:] <= np.deg2rad(0.5) + 1e-9)
    actor.layers[-1].bias.data.fill_(100.0)
    assert torch.allclose(actor(**inputs), cap6.expand(5, -1))


def test_residual_critic_td_target_is_proposal_space_and_bootstrap_safe() -> None:
    q1, q2 = ConstantQ(0.0), ConstantQ(1.0)
    q1_target, q2_target = ConstantQ(2.0), ConstantQ(3.0)
    target_actor = TargetActor()
    loss = residual_critic_loss(
        q1, q2, q1_target, q2_target, target_actor, batch(), gamma=0.5
    )
    assert torch.isclose(loss, torch.tensor(1.5))
    assert target_actor.calls == 1
    assert q1.batch_sizes == q2.batch_sizes == [2]
    assert q1_target.batch_sizes == q2_target.batch_sizes == [1]
    assert "base_actor" not in inspect.signature(residual_critic_loss).parameters
    assert not any(
        "camera" in name
        for name in inspect.signature(residual_critic_loss).parameters
    )


def test_current_and_target_q_use_proposals_with_independent_guard() -> None:
    actor = ScalarResidualActor()
    q1, q2 = ConstantQ(0.0), ConstantQ(1.0)
    policy = batch(1)
    policy.candidate_guard.valid[:] = False
    losses = residual_actor_loss(
        q1,
        q2,
        actor,
        policy,
        None,
        actor_q_weight=1.0,
        residual_l2_weight=0.01,
        human_residual_weight=1.0,
    )
    assert losses.actor_q_valid_count == 0
    assert losses.actor_q_guard_unknown_count == 1
    # The saved behavior proposal remains independently reportable.
    assert q1.batch_sizes == q2.batch_sizes == [1]

    policy.candidate_guard.valid[:] = True
    policy.base_gripper[:] = 0.75
    losses = residual_actor_loss(
        q1,
        q2,
        actor,
        policy,
        None,
        actor_q_weight=1.0,
        residual_l2_weight=0.01,
        human_residual_weight=1.0,
    )
    assert losses.actor_q_valid_count == 1
    assert torch.equal(q1.grippers[-1], torch.full((1, 1), 0.75))

    critic_batch = batch(1)
    critic_batch.next_base_gripper[:] = -0.4
    target_q1, target_q2 = ConstantQ(2.0), ConstantQ(3.0)
    details = residual_critic_loss(
        q1,
        q2,
        target_q1,
        target_q2,
        TargetActor(0.02),
        critic_batch,
        gamma=0.5,
        return_details=True,
    )
    assert details.td_valid_count == 1
    assert details.target_candidate_guard_rejected_count == 0
    assert details.target_candidate_guard_unknown_count == 0
    assert torch.allclose(
        target_q1.residuals[-1],
        torch.full((1, 6), 0.02),
        atol=1.0e-5,
    )
    assert torch.equal(target_q1.grippers[-1], torch.full((1, 1), -0.4))

    # The lower controller would clamp this target to x=0.796, but deployment's
    # earlier single-action check rejects x=0.797 instead of dispatching it.
    actor.value.data.fill_(0.002)
    policy = batch(1)
    for context in (policy.candidate_guard, policy.next_candidate_guard):
        context.decision_state7[:, 0] = 0.795
        context.upper_position3[:, 0] = 0.795
        context.policy_workspace_max3[:, 0] = 0.796
    rejected = residual_actor_loss(
        ConstantQ(0.0),
        ConstantQ(1.0),
        actor,
        policy,
        None,
        actor_q_weight=1.0,
        residual_l2_weight=0.01,
        human_residual_weight=1.0,
    )
    assert rejected.actor_q_valid_count == 0
    assert rejected.actor_q_guard_rejected_count == 1

    target_q1, target_q2 = ConstantQ(2.0), ConstantQ(3.0)
    rejected_target = residual_critic_loss(
        ConstantQ(0.0),
        ConstantQ(1.0),
        target_q1,
        target_q2,
        TargetActor(0.002),
        policy,
        gamma=0.5,
        return_details=True,
    )
    assert rejected_target.td_valid_count == 0
    assert rejected_target.target_candidate_guard_rejected_count == 1
    assert target_q1.batch_sizes == target_q2.batch_sizes == []


def test_candidate_guard_uses_decision_anchor_and_normalizes_proposal_once() -> None:
    context = candidate_guard_batch(1)
    request_pose_x = 0.45
    decision_pose_x = 0.50
    measured_ack_pose_x = 0.51
    assert len({request_pose_x, decision_pose_x, measured_ack_pose_x}) == 3
    context.decision_state7[:, 0] = decision_pose_x
    context.upper_position3[:, 0] = decision_pose_x
    context.delta_action_mean6[:, 0] = 0.03
    context.delta_action_std6[:, 0] = 0.02
    context.policy_workspace_max3[:, 0] = 0.55
    base = torch.zeros(1, 6)
    base[:, 0] = 0.5  # (0.04 m - mean 0.03 m) / std 0.02 m
    proposal = torch.zeros(1, 6)
    proposal[:, 0] = 0.1

    physical_correction = proposal[0, 0] * context.delta_action_std6[0, 0]
    assert torch.isclose(physical_correction, torch.tensor(0.002))
    assert policy_candidate_guard_valid(proposal, base, context).item() is True


def test_unknown_successor_guard_is_not_relabelled_as_terminal() -> None:
    critic_batch = batch(1)
    critic_batch.next_candidate_guard.valid[:] = False
    q1, q2 = ConstantQ(0.0), ConstantQ(1.0)
    q1_target, q2_target = ConstantQ(2.0), ConstantQ(3.0)
    target_actor = TargetActor(0.01)
    details = residual_critic_loss(
        q1,
        q2,
        q1_target,
        q2_target,
        target_actor,
        critic_batch,
        gamma=0.99,
        return_details=True,
    )
    assert details.td_valid_count == 0
    assert details.target_candidate_guard_unknown_count == 1
    assert critic_batch.terminated.item() is False
    assert q1.batch_sizes == q2.batch_sizes == []
    assert q1_target.batch_sizes == q2_target.batch_sizes == []

    critic_batch.terminated[:] = True
    residual_critic_loss(
        q1,
        q2,
        q1_target,
        q2_target,
        target_actor,
        critic_batch,
        gamma=0.99,
    )
    assert target_actor.calls == 1
    assert q1_target.batch_sizes == q2_target.batch_sizes == []


def test_filter_leash_mapping_is_differentiable_and_not_a_recorded_point() -> None:
    context = acceptance_batch(1)
    proposal = torch.full((1, 6), 2.0e-6, requires_grad=True)
    mapped = map_residual_to_controller_ack(
        proposal,
        torch.zeros(1, 6),
        context,
    )
    assert mapped.valid.tolist() == [True]
    assert torch.allclose(
        mapped.residual_k6[:, 0], proposal, rtol=1.0e-4, atol=1.0e-7
    )
    mapped.residual_k6.sum().backward()
    assert proposal.grad is not None
    assert torch.isfinite(proposal.grad).all()
    assert torch.count_nonzero(proposal.grad) == 6


def test_same_visible_context_and_proposal_can_have_different_accepted_actions() -> None:
    context = acceptance_batch(2)
    context.step_dt_s[:] = 0.01
    context.filter_time_constant_s[:] = 1.0
    context.filter_position_before3[1, 0] = 0.05
    proposal = torch.zeros(2, 6)
    proposal[:, 0] = 0.01
    mapped = map_residual_to_controller_ack(
        proposal,
        torch.zeros(2, 6),
        context,
    )
    assert mapped.valid.tolist() == [True, True]
    assert torch.equal(proposal[0], proposal[1])
    assert not torch.equal(mapped.residual_k6[0], mapped.residual_k6[1])


def test_nonterminal_human_successor_is_bc_only() -> None:
    replay = human_replay(terminated=False)
    assert replay.critic_td_valid_rows == 0
    assert replay.human_residual_valid_rows == 1
    critic_batch = replay.sample(
        1,
        device=torch.device("cpu"),
        seed=1,
        td_mappable_only=True,
    )
    assert critic_batch is None


def test_recorded_rows_are_distinct_from_currently_usable_td_rows() -> None:
    replay = human_replay(terminated=False, acceptance_valid=False)
    assert replay.recorded_transition_rows == 1
    assert replay.critic_td_valid_rows == 0
    assert replay.human_residual_valid_rows == 1
    assert list(
        replay.iter_td_batches(8, device=torch.device("cpu"), seed=3)
    ) == []


def test_recorded_policy_proposal_is_not_replaced_by_ack_minus_base() -> None:
    replay = policy_replay(
        schema_version=ACK_RESIDUAL_TRANSITION_SCHEMA_VERSION,
        base_action=[[0.0] * 7 for _ in range(3)],
    )
    row = replay.rows[0]
    assert np.allclose(row["behavior_proposal6"], 0.01)
    assert np.allclose(
        row["accepted_residual_k6"][:, [0, 3]],
        [[0.2, 0.1]] * 3,
    )
    critic_batch = replay.sample(
        1, device=torch.device("cpu"), seed=5, td_mappable_only=True
    )
    assert critic_batch is not None
    q1, q2 = ConstantQ(0.0), ConstantQ(1.0)
    residual_critic_loss(
        q1,
        q2,
        ConstantQ(2.0),
        ConstantQ(3.0),
        TargetActor(0.04),
        critic_batch,
        gamma=0.99,
    )
    assert torch.allclose(q1.residuals[-1], torch.full((1, 6), 0.01))


def test_critic_context_uses_base_gripper_without_control_source() -> None:
    critic_batch = batch(2)
    critic_batch.terminated[:] = True
    critic_batch.base_gripper[:] = torch.tensor([[0.085], [0.0]])
    q1, q2 = ConstantQ(0.0), ConstantQ(1.0)
    residual_critic_loss(
        q1, q2, ConstantQ(2.0), ConstantQ(3.0), TargetActor(), critic_batch, 0.99
    )
    assert torch.equal(q1.grippers[-1], torch.tensor([[0.085], [0.0]]))


def test_human_imitation_projects_only_bc_target() -> None:
    actor = ScalarResidualActor()
    human = batch(2)
    human.human_residual_valid[:] = True
    human.human_residual_target6[:] = 0.8
    human.behavior_proposal6[:] = 0.8
    raw_target = human.human_residual_target6.clone()
    raw_behavior = human.behavior_proposal6.clone()
    losses = residual_actor_loss(
        ConstantQ(0.0),
        ConstantQ(1.0),
        actor,
        None,
        human,
        actor_q_weight=1.0,
        residual_l2_weight=0.01,
        human_residual_weight=1.0,
    )
    assert losses.human_residual_projected_count == 2
    assert losses.human_residual_valid_count == 2
    assert torch.isclose(losses.human, torch.tensor(0.25))
    assert torch.equal(human.human_residual_target6, raw_target)
    assert torch.equal(human.behavior_proposal6, raw_behavior)


def test_actor_q_mask_and_invalid_human_residual_are_skipped() -> None:
    q1, q2 = ConstantQ(0.0), ConstantQ(1.0)
    actor = ScalarResidualActor()
    policy = batch(2)
    policy.actor_q_valid = torch.tensor([False, True])
    human = batch(2)
    human.human_residual_valid = torch.tensor([False, False])
    losses = residual_actor_loss(
        q1,
        q2,
        actor,
        policy,
        human,
        actor_q_weight=1.0,
        residual_l2_weight=0.01,
        human_residual_weight=1.0,
    )
    assert losses.actor_q_valid_count == 1
    assert losses.human_residual_valid_count == 0
    # Candidate, actual behavior, and zero-proposal diagnostics each use the
    # one policy-eligible row.
    assert q1.batch_sizes == q2.batch_sizes == [1, 1, 1]
    assert actor.batch_sizes == [2]
    assert torch.equal(losses.human, torch.zeros_like(losses.human))


def test_human_bc_updates_actor_but_never_critic_action_columns() -> None:
    replay = human_replay()
    row = replay.rows[0]
    assert row["human_residual_valid"] is True
    assert np.count_nonzero(row["accepted_residual_k6"]) > 0
    assert np.count_nonzero(row["human_residual_target6"]) > 0

    q1, q2, q1_target, q2_target = build_twin_q(hidden_dim=16, seed=13)
    target_actor = TargetActor()
    optimizer = torch.optim.Adam((*q1.parameters(), *q2.parameters()), lr=3e-4)
    before = q1.layers[0].weight[
        :, RESIDUAL_ACTION_OFFSET : RESIDUAL_ACTION_OFFSET + RESIDUAL_ACTION_WIDTH
    ].detach().clone()
    critic_batch = replay.sample(
        8, device=torch.device("cpu"), seed=1, td_mappable_only=True
    )
    assert critic_batch is None
    after = q1.layers[0].weight[
        :, RESIDUAL_ACTION_OFFSET : RESIDUAL_ACTION_OFFSET + RESIDUAL_ACTION_WIDTH
    ].detach()
    assert torch.equal(before, after)
    actor = ScalarResidualActor()
    human_batch = replay.sample(
        8, device=torch.device("cpu"), seed=2, human_only=True
    )
    assert human_batch is not None
    loss = residual_actor_loss(
        q1,
        q2,
        actor,
        None,
        human_batch,
        actor_q_weight=0.1,
        residual_l2_weight=0.1,
        human_residual_weight=1.0,
    )
    loss.total.backward()
    assert actor.value.grad is not None and actor.value.grad.abs() > 0


def test_human_context_supplies_l2_when_no_policy_context_exists() -> None:
    replay = human_replay()
    human_batch = replay.sample(
        8, device=torch.device("cpu"), seed=2, human_only=True
    )
    assert human_batch is not None
    actor = ScalarResidualActor()
    with torch.no_grad():
        actor.value.fill_(0.25)
    loss = residual_actor_loss(
        ConstantQ(0.0),
        ConstantQ(0.0),
        actor,
        None,
        human_batch,
        actor_q_weight=0.1,
        residual_l2_weight=0.1,
        human_residual_weight=1.0,
    )
    assert torch.isclose(loss.residual, torch.tensor(0.25**2))
    assert torch.isclose(loss.output_norm, torch.tensor(6 * 0.25**2).sqrt())


def test_nonzero_policy_proposal_can_train_critic_action_columns() -> None:
    replay = policy_replay(
        schema_version=ACK_RESIDUAL_TRANSITION_SCHEMA_VERSION,
        base_action=[[0.0] * 7 for _ in range(3)],
    )
    critic_batch = replay.sample(
        8, device=torch.device("cpu"), seed=1, td_mappable_only=True
    )
    assert critic_batch is not None
    # A non-zero proposal only identifies its Q dependence when the TD error is
    # non-zero.  The terminal fixture otherwise has both reward and initial Q
    # exactly zero, which correctly produces no optimizer update at all.
    critic_batch.reward.fill_(1.0)
    q1, q2, q1_target, q2_target = build_twin_q(hidden_dim=16, seed=13)
    optimizer = torch.optim.Adam((*q1.parameters(), *q2.parameters()), lr=3e-4)
    before = q1.layers[0].weight[
        :, RESIDUAL_ACTION_OFFSET : RESIDUAL_ACTION_OFFSET + RESIDUAL_ACTION_WIDTH
    ].detach().clone()
    optimizer.zero_grad(set_to_none=True)
    residual_critic_loss(
        q1, q2, q1_target, q2_target, TargetActor(), critic_batch, gamma=0.99
    ).backward()
    optimizer.step()
    after = q1.layers[0].weight[
        :, RESIDUAL_ACTION_OFFSET : RESIDUAL_ACTION_OFFSET + RESIDUAL_ACTION_WIDTH
    ].detach()
    assert not torch.equal(before, after)


def test_zero_policy_proposals_leave_zero_initialized_action_columns_zero() -> None:
    replay = policy_replay(
        schema_version=ACK_RESIDUAL_TRANSITION_SCHEMA_VERSION,
        base_action=[[0.0] * 7 for _ in range(3)],
    )
    replay.rows[0]["behavior_proposal6"][:] = 0.0
    critic_batch = replay.sample(
        8, device=torch.device("cpu"), seed=1, td_mappable_only=True
    )
    assert critic_batch is not None
    q1, q2, q1_target, q2_target = build_twin_q(hidden_dim=16, seed=13)
    optimizer = torch.optim.Adam((*q1.parameters(), *q2.parameters()), lr=3e-4)
    optimizer.zero_grad(set_to_none=True)
    residual_critic_loss(
        q1, q2, q1_target, q2_target, TargetActor(), critic_batch, gamma=0.99
    ).backward()
    optimizer.step()
    for q in (q1, q2):
        columns = q.layers[0].weight[
            :, RESIDUAL_ACTION_OFFSET : RESIDUAL_ACTION_OFFSET + RESIDUAL_ACTION_WIDTH
        ]
        assert torch.count_nonzero(columns) == 0


def test_zero_initialized_q_has_no_initial_value_gradient_on_zero_actor() -> None:
    actor, _target = make_residual_actor_pair(hidden_dim=16)
    q1, q2, _q1_target, _q2_target = build_twin_q(hidden_dim=16, seed=9)
    losses = residual_actor_loss(
        q1,
        q2,
        actor,
        batch(4),
        None,
        actor_q_weight=0.1,
        residual_l2_weight=0.1,
        human_residual_weight=1.0,
    )
    losses.total.backward()
    assert torch.count_nonzero(actor.layers[-1].weight.grad) == 0
    assert torch.count_nonzero(actor.layers[-1].bias.grad) == 0


def test_same_decision_anchor_removes_motion_from_behavior_residual() -> None:
    class Affine:
        def __init__(self) -> None:
            self.mean = np.asarray([0.3] * 7)
            self.std = np.asarray([2.0] * 7)

        def apply(self, value):
            return (np.asarray(value) - self.mean) / self.std

    decision_state = np.asarray([0.002, 0.0, 0.0, 0.1, -0.2, 0.3, 0.085])
    base = np.repeat(
        np.asarray([[0.010, 0.0, 0.0, 0.1, -0.2, 0.3, 0.085]]),
        3,
        axis=0,
    )
    base_normalized, accepted_normalized, residual = normalized_behavior_residual(
        base_absolute_k7=base,
        accepted_absolute_k7=base.copy(),
        decision_state7=decision_state,
        normalize_delta7=Affine().apply,
        valid_mask=np.ones(3, dtype=np.bool_),
    )
    assert np.array_equal(base_normalized, accepted_normalized)
    assert np.count_nonzero(residual) == 0

    controller_accepted = base.copy()
    controller_accepted[:, 0] += 0.004
    _base, _accepted, controller_residual = normalized_behavior_residual(
        base_absolute_k7=base,
        accepted_absolute_k7=controller_accepted,
        decision_state7=decision_state,
        normalize_delta7=Affine().apply,
        valid_mask=np.ones(3, dtype=np.bool_),
    )
    assert np.allclose(controller_residual[:, 0], 0.002)


def test_dispatch_actor_context_is_the_replay_context_and_hold_has_no_fake_step() -> None:
    class RecordingActor(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.marker = torch.nn.Parameter(torch.zeros(()))
            self.inputs = None

        def forward(self, **kwargs):
            self.inputs = {
                name: value.detach().cpu().clone() for name, value in kwargs.items()
            }
            return torch.full((1, 6), 0.01)

    class Safety:
        workspace_min_xyz_m = np.asarray([-1.0] * 3)
        workspace_max_xyz_m = np.asarray([1.0] * 3)
        orientation_min_rpy_rad = np.asarray([-np.pi] * 3)
        orientation_max_rpy_rad = np.asarray([np.pi] * 3)
        gimbal_margin_rad = np.deg2rad(2.0)
        gripper_min_width_m = 0.0
        gripper_max_width_m = 0.1
        continuity_max_xyz_m = 0.08
        continuity_max_rotation_rad = np.deg2rad(25.0)
        continuity_max_gripper_delta_m = 0.1

        @staticmethod
        def validate_chunk(*_args):
            return None

    normalizer = SimpleNamespace(
        state7=IdentityTransform(7),
        wrench6=IdentityTransform(6),
        delta_action7=IdentityTransform(7),
    )
    engine = object.__new__(learner_server.serve_policy.InferenceEngine)
    engine.residual_actor = RecordingActor()
    engine._residual_lock = threading.Lock()
    engine.runtime_artifacts = SimpleNamespace(
        normalizer=normalizer,
        normalizer_manifest_sha256="existing-manifest-id",
    )
    engine.policy = SimpleNamespace(_action_safety_profile=Safety())
    state = [0.1, 0.0, 0.2, 0.0, 0.0, 0.0, 0.085]
    chunk = [
        [0.11, 0.0, 0.2, 0.0, 0.0, 0.0, 0.085],
        [0.12, 0.0, 0.2, 0.0, 0.0, 0.0, 0.085],
        [0.13, 0.0, 0.2, 0.0, 0.0, 0.0, 0.085],
    ]
    response = engine.residual_decision(
        {
            "decision_monotonic_ns": 1_000_000_000,
            "state7": state,
            "wrench6": [1.0] * 6,
            "wrench_delta6": [5.0] * 6,
            "base_absolute_action7": chunk[2],
        }
    )
    assert response["policy_single_action_guard"]["continuity_max_xyz_m"] == 0.08
    assert np.allclose(
        engine.residual_actor.inputs["base_action6"][0].numpy(),
        [0.03, 0.0, 0.0, 0.0, 0.0, 0.0],
    )
    assert np.array_equal(
        engine.residual_actor.inputs["normalized_wrench_delta6"][0].numpy(),
        np.full(6, 5.0, dtype=np.float32),
    )

    current_context = {
        "online_semantics_version": ONLINE_SEMANTICS_VERSION,
        "valid_for_residual_training": True,
        "invalid_reason": None,
        "decision_monotonic_ns": 1_000_000_000,
        "state7_absolute": state,
        "wrench6_calibrated_tcp": [1.0] * 6,
        "wrench_delta6_calibrated_tcp_100ms": [5.0] * 6,
        "wrench_delta_interval_ns": 80_000_000,
        "base_absolute_action7": chunk[2],
        "normalized_state7": response["normalized_state7"],
        "normalized_wrench6": response["normalized_wrench6"],
        "normalized_wrench_delta6": response["normalized_wrench_delta6"],
        "base_normalized_action6": response["base_normalized_action7"][:6],
        "candidate_acceptance_mapping": acceptance_mapping(
            decision_state7=state
        ),
    }
    next_base = [0.14, 0.0, 0.2, 0.0, 0.0, 0.0, 0.085]
    next_context = {
        **current_context,
        "decision_monotonic_ns": 1_400_000_000,
        "wrench6_calibrated_tcp": [2.0] * 6,
        "wrench_delta6_calibrated_tcp_100ms": [7.0] * 6,
        "wrench_delta_interval_ns": 400_000_000,
        "base_absolute_action7": next_base,
        "normalized_wrench6": [2.0] * 6,
        "normalized_wrench_delta6": [7.0] * 6,
        "base_normalized_action6": [0.04, 0.0, 0.0, 0.0, 0.0, 0.0],
        "candidate_acceptance_mapping": acceptance_mapping(
            decision_state7=state
        ),
        "previous_policy_dispatch_sequence": 9,
    }
    accepted = np.repeat(
        np.asarray(response["composed_absolute_action7"])[None, :], 3, axis=0
    )
    behavior = AckMacro(
        grid_monotonic_ns=(1_000_000_000, 1_033_333_333, 1_066_666_667),
        ack_ids=("ack",) * 3,
        gripper_command_ids=("gripper",) * 3,
        gripper_ack_command_ids=("gripper",) * 3,
        accepted_absolute_action_k7=accepted,
        slot_owner=("policy",) * 3,
        workspace_clip_flags=(False,) * 3,
        source_command_ids=("command",) * 3,
        source_dispatch_sequences=(9,) * 3,
        source_model_indices=(2,) * 3,
        chunk_ids=("chunk",) * 3,
        controller_authorities=("controller",) * 3,
        contract_version=DISPATCH_DECISION_CRITIC_CONTRACT_VERSION,
        next_timestamp_ns=1_400_000_000,
        macro_duration_ns=400_000_000,
    )
    transition = {
        "schema_version": ACK_RESIDUAL_TRANSITION_SCHEMA_VERSION,
        "online_semantics_version": ONLINE_SEMANTICS_VERSION,
        "identity": {"episode_id": "dispatch-episode", "decision_id": 9},
        "action_source": "policy",
        "observation": {
            "state7_absolute": state,
            "wrench6_calibrated_tcp": [1.0] * 6,
        },
        "next_observation": {
            "state7_absolute": state,
            "wrench6_calibrated_tcp": [2.0] * 6,
        },
        "outcome": {"reward": 0.0, "terminated": False, "truncated": False},
        "eligibility": {"critic_td_valid": True, "actor_q_valid": True},
        "base_absolute_action_k7": np.repeat(
            np.asarray(chunk[2])[None, :], 3, axis=0
        ).tolist(),
        "base_normalized_action_k7": np.repeat(
            np.asarray(response["base_normalized_action7"])[None, :], 3, axis=0
        ).tolist(),
        "applied_residual_tcp6": response["applied_residual_tcp6"],
        "composed_normalized_action_k7": np.repeat(
            np.asarray(response["composed_normalized_action7"])[None, :],
            3,
            axis=0,
        ).tolist(),
        "policy_lineage": {
            "revision": "test-revision",
            "selection": {
                "sequence": 9,
                "policy_revision": "test-revision",
                "applied_residual_tcp6": response["applied_residual_tcp6"],
                "base_normalized_action7": response["base_normalized_action7"],
                "base_absolute_action7": chunk[2],
                "composed_normalized_action7": response[
                    "composed_normalized_action7"
                ],
                "residual_decision_context": {
                    "decision_monotonic_ns": 1_000_000_000
                },
            }
        },
        "accepted_absolute_action_k7": accepted.tolist(),
        "residual_decision_context": current_context,
        "next_residual_decision_context": next_context,
        "next_action_source": "policy",
        "next_accepted_absolute_action7": next_base,
        "next_applied_residual_tcp6": [0.0] * 6,
        "human_residual_valid": False,
    }
    replay = OnlineResidualReplay(
        (
            ProductionAckMacro(
                transition=transition,
                behavior=behavior,
                next_grid_monotonic_ns=1_400_000_000,
                ack_provenance=(),
                actor_q_eligibility=ActorQEligibility(True, "valid"),
            ),
        ),
        normalizer,
    )
    assert replay.critic_td_valid_rows == 1
    row = replay.rows[0]
    assert np.array_equal(row["wrench_delta6"], np.full(6, 5.0))
    assert np.array_equal(row["next_wrench_delta6"], np.full(6, 7.0))
    assert np.allclose(row["behavior_proposal6"], 0.01)
    assert np.allclose(row["accepted_residual_k6"], 0.01)
    assert row["next_base_valid"] is True
    assert np.allclose(row["base_gripper"], [0.085])
    assert row["next_candidate_guard"]["valid"] is True
    assert np.allclose(row["next_base_gripper"], [0.085])

    missing_next_proposal = dict(transition)
    missing_next_proposal.pop("next_applied_residual_tcp6")
    invalid = OnlineResidualReplay(
        (
            ProductionAckMacro(
                transition=missing_next_proposal,
                behavior=behavior,
                next_grid_monotonic_ns=1_400_000_000,
                ack_provenance=(),
                actor_q_eligibility=ActorQEligibility(True, "valid"),
            ),
        ),
        normalizer,
    )
    assert invalid.recorded_transition_rows == 1
    assert invalid.critic_td_valid_rows == 0


def test_policy_value_sampling_excludes_human_and_missing_next_base() -> None:
    replay = human_replay()
    assert replay.sample(
        1,
        device=torch.device("cpu"),
        seed=0,
        policy_only=True,
        actor_q_valid_only=True,
    ) is None
    assert replay.sample(
        1, device=torch.device("cpu"), seed=0, human_only=True
    ) is not None

    missing_next_base = human_replay(terminated=False, include_successor=False)
    assert missing_next_base.recorded_transition_rows == 1
    assert missing_next_base.critic_td_valid_rows == 0
    assert missing_next_base.human_residual_valid_rows == 1
    assert missing_next_base.sample(
        1, device=torch.device("cpu"), seed=0, human_only=True
    ) is not None
    assert missing_next_base.next_base_missing_rows == 1


def test_nonterminal_policy_without_direct_successor_is_kept_but_not_td() -> None:
    replay = policy_replay(
        schema_version=ACK_RESIDUAL_TRANSITION_SCHEMA_VERSION,
        base_action=[[0.0] * 7 for _ in range(3)],
        terminated=False,
    )
    assert replay.recorded_transition_rows == 1
    assert replay.critic_td_valid_rows == 0
    assert replay.actor_q_valid_rows == 1
    assert replay.next_base_missing_rows == 1


def test_missing_or_legacy_policy_base_is_not_a_valid_dispatch_row() -> None:
    legacy_schema = next(iter(LEGACY_ACK_RESIDUAL_TRANSITION_SCHEMA_VERSIONS))
    legacy = policy_replay(schema_version=legacy_schema, base_action=None)
    assert legacy.critic_td_valid_rows == 0
    assert legacy.nonzero_behavior_residual_rows == 0
    assert legacy.quarantined_current_schema_rows == 1

    missing_current = policy_replay(
        schema_version=ACK_RESIDUAL_TRANSITION_SCHEMA_VERSION,
        base_action=None,
    )
    assert missing_current.rows == ()
    assert missing_current.quarantined_current_schema_rows == 1

    corrupted_current = policy_replay(
        schema_version=ACK_RESIDUAL_TRANSITION_SCHEMA_VERSION,
        base_action=[[float("nan")] * 7 for _ in range(3)],
    )
    assert corrupted_current.rows == ()
    assert corrupted_current.quarantined_current_schema_rows == 1

    current = policy_replay(
        schema_version=ACK_RESIDUAL_TRANSITION_SCHEMA_VERSION,
        base_action=[[0.0] * 7 for _ in range(3)],
    )
    assert current.critic_td_valid_rows == 1
    assert current.nonzero_behavior_residual_rows == 1


@pytest.mark.parametrize(
    "overrides",
    (
        {"selection_sequence": 8},
        {"selection_decision_ns": 1_000_000_001},
        {"selection_revision": "wrong-revision"},
        {"proposal_present": False},
    ),
)
def test_wrong_policy_proposal_lineage_is_not_td_valid(overrides) -> None:
    replay = policy_replay(
        schema_version=ACK_RESIDUAL_TRANSITION_SCHEMA_VERSION,
        base_action=[[0.0] * 7 for _ in range(3)],
        **overrides,
    )
    assert replay.recorded_transition_rows == 1
    assert replay.critic_td_valid_rows == 0
    assert replay.rows[0]["policy_proposal_valid"] is False


def test_replay_sampling_is_without_replacement_when_population_is_large_enough() -> None:
    replay = policy_replay(
        schema_version=ACK_RESIDUAL_TRANSITION_SCHEMA_VERSION,
        base_action=[[0.0] * 7 for _ in range(3)],
    )
    prototype = replay.rows[0]
    replay.rows = tuple(
        {**prototype, "state7": np.full(7, index, dtype=np.float32)}
        for index in range(8)
    )
    sampled = replay.sample(8, device=torch.device("cpu"), seed=7)
    assert sampled is not None
    assert len(set(sampled.state7[:, 0].tolist())) == 8

    replay.rows = tuple(
        {
            **prototype,
            "episode_id": episode_id,
            "state7": np.full(7, value, dtype=np.float32),
        }
        for episode_id, value, count in (
            ("short", 0.0, 20),
            ("long", 1.0, 100),
        )
        for _ in range(count)
    )
    balanced = replay.sample(10, device=torch.device("cpu"), seed=7)
    assert balanced is not None
    assert balanced.state7[:, 0].tolist().count(0.0) == 5
    assert balanced.state7[:, 0].tolist().count(1.0) == 5


def test_replay_sampling_copies_read_only_normalizer_stats() -> None:
    replay = policy_replay(
        schema_version=ACK_RESIDUAL_TRANSITION_SCHEMA_VERSION,
        base_action=[[0.0] * 7 for _ in range(3)],
    )
    replay.normalizer.delta_action7.mean.setflags(write=False)
    replay.normalizer.delta_action7.std.setflags(write=False)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        sampled = replay.sample(1, device=torch.device("cpu"), seed=7)

    assert sampled is not None
    assert not any("not writable" in str(item.message) for item in caught)


def test_online_schedule_is_continuous_2q_1actor_with_100_1000_cadence() -> None:
    policy = ResidualActorCriticSchedule()
    assert policy.scheduling_mode == "continuous_async"
    assert policy.twin_q_updates_per_cycle == 2
    assert policy.residual_actor_updates_per_cycle == 1
    assert not policy.candidate_due(99)
    assert policy.candidate_due(100)
    assert not policy.checkpoint_due(999)
    assert policy.checkpoint_due(1000)
    assert not hasattr(policy, "max_cycles_per_admitted_episode")
    assert not hasattr(policy, "admitted_rows_per_cycle")


def test_human_bc_rows_do_not_count_toward_policy_td_startup_or_credit() -> None:
    policy = ResidualActorCriticSchedule()
    ledger = TdCycleCreditLedger(new_td_rows_per_cycle=8)
    ledger.register_admission(
        admission_id="episode-a",
        episode_id="episode-a",
        td_uids={f"policy:{index}" for index in range(700)},
    )
    snapshot = ledger.snapshot(completed_cycles=0)
    assert snapshot.unique_td_rows == 700
    assert snapshot.allowed_cycles == 87
    assert not policy.training_ready(
        snapshot.unique_td_rows, snapshot.distinct_td_episodes
    )
    # Three hundred legal human BC rows are deliberately absent from td_uids.
    ledger.register_admission(
        admission_id="episode-b-human-only",
        episode_id="episode-b-human-only",
        td_uids=set(),
    )
    assert ledger.snapshot(completed_cycles=0) == snapshot

    ledger.register_admission(
        admission_id="episode-b",
        episode_id="episode-b",
        td_uids={f"policy-b:{index}" for index in range(150)},
    )
    ledger.register_admission(
        admission_id="episode-c",
        episode_id="episode-c",
        td_uids={f"policy-c:{index}" for index in range(150)},
    )
    ready = ledger.snapshot(completed_cycles=0)
    assert policy.training_ready(ready.unique_td_rows, ready.distinct_td_episodes)
    assert ready.unique_td_rows == 1000
    assert ready.allowed_cycles == 125


def test_task_profiles_cannot_override_algorithm_parameters() -> None:
    task2 = load_common_actor_critic_config("task2")
    task3 = load_common_actor_critic_config("task3")
    assert algorithm_hyperparameters(task2) == algorithm_hyperparameters(task3)
    assert task2["task"] != task3["task"]
    assert task2["residual_actor_critic_training"] == {
        "scheduling_mode": "continuous_async",
        "new_td_rows_per_cycle": 8,
        "twin_q_updates_per_cycle": 2,
        "residual_actor_updates_per_cycle": 1,
        "residual_candidate_interval_cycles": 100,
        "training_checkpoint_interval_cycles": 1000,
        "retained_training_checkpoint_count": 10,
        "checkpoint_on_warmup_complete": True,
        "checkpoint_on_candidate_activation": False,
    }


def tiny_continuous_learner(*, learner_state: str, warmup_updates: int = 0):
    learner = learner_server.ResidualActorCriticLearner.__new__(
        learner_server.ResidualActorCriticLearner
    )
    actor = torch.nn.Linear(2, 2)
    learner.replay_root = Path("/unused")
    learner.replay = None
    learner.training_policy = ResidualActorCriticSchedule(
        checkpoint_on_warmup_complete=False,
    )
    learner._loaded_episode_keys = set()
    learner._admission_progress = {}
    learner._expected_admission_id = None
    learner._state_lock = threading.RLock()
    learner.sampled_session_ids = set()
    learner.sampled_episode_ids = set()
    learner.latest_replay_refresh_ms = 0.0
    learner.latest_critic_update_ms = 0.0
    learner.latest_actor_update_ms = 0.0
    learner.latest_cycle_ms = 0.0
    learner.human_supervision_diagnostics = {}
    learner.credit_ledger = TdCycleCreditLedger(new_td_rows_per_cycle=8)
    for episode, count in (("a", 334), ("b", 333), ("c", 333)):
        learner.credit_ledger.register_admission(
            admission_id=episode,
            episode_id=f"{episode}/episode",
            td_uids={f"{episode}:{index}" for index in range(count)},
        )
    learner.learner = {
        "residual_actor": actor,
        "runtime": {
            "learner_state": learner_state,
            "ack_critic_warmup_complete": learner_state == "residual_actor_critic_training",
            "ack_critic_warmup_steps": warmup_updates,
            "residual_actor_critic_cycles": 0,
            "partial_cycle_q_updates": 0,
            "counters": {
                "twin_q_optimizer_steps": warmup_updates,
                "residual_actor_optimizer_steps": 0,
                "residual_actor_update_attempts": 0,
                "residual_actor_updates_skipped_no_gradient": 0,
                "twin_q_target_update_steps": warmup_updates,
                "critic_sample_draws": 0,
                "policy_sample_draws": 0,
                "human_sample_draws": 0,
            },
            "replay": {
                "critic_td_valid_rows": 0,
                "actor_q_valid_rows": 0,
                "human_residual_valid_rows": 0,
                "training_credit_ledger": learner.credit_ledger.state_dict(),
            },
            "scheduling": {
                "mode": "continuous_async",
                "candidate_export_generation": "test-generation",
                "last_publish_attempt_cycle": 0,
                "last_published_cycle": 0,
            },
        },
    }
    return learner


def set_test_credit(
    learner: learner_server.ResidualActorCriticLearner,
    counts: tuple[int, ...],
) -> None:
    learner.credit_ledger = TdCycleCreditLedger(new_td_rows_per_cycle=8)
    for episode_index, count in enumerate(counts):
        admission = f"credit-{episode_index}"
        learner.credit_ledger.register_admission(
            admission_id=admission,
            episode_id=f"{admission}/episode",
            td_uids={f"{admission}:{index}" for index in range(count)},
        )
    learner.learner["runtime"]["replay"][
        "training_credit_ledger"
    ] = learner.credit_ledger.state_dict()


def test_counter_snapshot_waits_for_atomic_actor_counter_update() -> None:
    learner = tiny_continuous_learner(
        learner_state="residual_actor_critic_training", warmup_updates=256
    )
    counters = learner.learner["runtime"]["counters"]
    started = threading.Event()
    finished = threading.Event()
    observed: list[dict[str, object]] = []

    def read_snapshot() -> None:
        started.set()
        observed.append(learner.counter_snapshot())
        finished.set()

    with learner._state_lock:
        counters["residual_actor_update_attempts"] = 1
        thread = threading.Thread(target=read_snapshot)
        thread.start()
        assert started.wait(1.0)
        assert not finished.wait(0.05)
        counters["residual_actor_updates_skipped_no_gradient"] = 1
    thread.join(1.0)
    assert finished.is_set()
    assert observed[0]["residual_actor_update_attempts"] == 1
    assert observed[0]["residual_actor_optimizer_steps"] == 0
    assert observed[0]["residual_actor_updates_skipped_no_gradient"] == 1


def test_replay_refresh_loads_only_newly_sealed_episodes(monkeypatch) -> None:
    learner = tiny_continuous_learner(learner_state="ack_replay_collection")
    learner.credit_ledger = TdCycleCreditLedger(new_td_rows_per_cycle=8)
    learner.learner["runtime"]["replay"][
        "training_credit_ledger"
    ] = learner.credit_ledger.state_dict()
    learner.normalizer = SimpleNamespace(
        delta_action7=SimpleNamespace(std=np.ones(7))
    )
    learner.current_session_id = None
    learner.unique_r_count = 0
    learner.r_macro_count = 0
    learner.next_base_missing_rows = 0
    learner.quarantined_current_schema_rows = 0
    learner.nonzero_behavior_residual_rows = 0
    learner.nonzero_policy_proposal_rows = 0
    learner.nonzero_accepted_residual_rows = 0
    signatures = [["a"]]
    monkeypatch.setattr(
        learner, "_episode_signature", lambda: tuple(signatures[0])
    )

    class FakeReplay:
        def __init__(self, _macros, _normalizer) -> None:
            self.counts: list[int] = []
            self.rows: list[dict] = []
            self.next_base_missing_rows = 0
            self.quarantined_current_schema_rows = 0
            self.nonzero_behavior_residual_rows = 0
            self.nonzero_policy_proposal_rows = 0
            self.nonzero_accepted_residual_rows = 0
            self.candidate_guard_unknown_rows = 0

        def append_macros(self, macros):
            macro = tuple(macros)[0]
            episode_id = macro.transition["identity"]["episode_id"]
            count = int(macro.transition["materialized_count"])
            self.counts.append(count)
            admission_id = str(episode_id).split("/", 1)[0]
            self.rows.extend(
                {
                    "transition_uid": f"{admission_id}:{index}",
                    "episode_id": episode_id,
                    "critic_td_valid": True,
                    "human_residual_valid": False,
                }
                for index in range(count)
            )
            return {episode_id: count}

        @property
        def critic_rows_per_episode(self):
            return tuple(self.counts)

        @property
        def critic_td_valid_rows(self):
            return sum(self.counts)

        @property
        def recorded_transition_rows(self):
            return sum(self.counts)

        def critic_td_rows_for_episode(self, episode_id):
            admission_id = str(episode_id).split("/", 1)[0]
            return {"a": 99, "b": 1, "c": 400, "d": 1}[admission_id]

        actor_q_valid_rows = property(lambda self: sum(self.counts))
        human_residual_valid_rows = property(lambda _self: 0)

    calls: list[str] = []

    def load_episode(_root, admission_id):
        calls.append(admission_id)
        episode_id = f"{admission_id}/episode"
        row = {
            "identity": {
                "episode_id": episode_id,
                "session_id": "current" if admission_id == "d" else "old",
            },
            "materialized_count": {"a": 99, "b": 1, "c": 400, "d": 1}[
                admission_id
            ],
        }
        macro = SimpleNamespace(transition=row)
        return [row], (macro,), {episode_id: Path("episode")}, []

    monkeypatch.setattr(learner_server.warmup, "OnlineResidualReplay", FakeReplay)
    monkeypatch.setattr(
        learner_server.warmup, "load_formal_online_episode", load_episode
    )
    monkeypatch.setattr(
        learner_server.warmup,
        "load_formal_online_r",
        lambda _root: (_ for _ in ()).throw(AssertionError("full reload")),
    )
    monkeypatch.setattr(
        learner_server.warmup, "build_ack_macros", lambda _rows: ()
    )

    learner._refresh_replay()
    assert calls == ["a"]
    assert learner._admission_progress["a"]["cycle_count_when_observed"] == 0
    signatures[0].append("b")
    learner._refresh_replay()
    assert learner._admission_progress["b"]["cycle_count_when_observed"] == 0
    learner.learner["runtime"]["residual_actor_critic_cycles"] = 1
    signatures[0].append("c")
    learner.notify_admission("c")
    learner._refresh_replay()
    assert calls == ["a", "b", "c"]
    assert learner.learner["runtime"]["replay"]["loaded_episode_keys"] == [
        "a",
        "b",
        "c",
    ]
    assert learner._admission_progress["c"] == {
        "episode_key": "c",
        "recorded_transition_rows": 400,
        "admitted_rows_for_latest_episode": 400,
        "policy_rows_for_latest_episode": 1,
        "human_rows_for_latest_episode": 0,
        "human_bc_rows_for_latest_episode": 0,
        "cycle_count_when_observed": 1,
    }
    learner.current_session_id = "current"
    signatures[0].append("d")
    with pytest.raises(
        RuntimeError, match="CURRENT_EPISODE_ALREADY_IN_REPLAY"
    ):
        learner._refresh_replay()
    assert "d" not in learner._loaded_episode_keys


def test_replay_signature_ignores_uncommitted_and_rejected_records(
    tmp_path: Path,
) -> None:
    learner = learner_server.ResidualActorCriticLearner.__new__(
        learner_server.ResidualActorCriticLearner
    )
    learner.replay_root = tmp_path
    (tmp_path / "admissions").mkdir()
    (tmp_path / "rejected").mkdir()
    (tmp_path / "admissions/uncommitted.json").write_text("{}")
    (tmp_path / "rejected/rejected.json").write_text("{}")
    assert learner._episode_signature() == ()


def test_collecting_does_not_update_actor_or_critic(monkeypatch) -> None:
    learner = tiny_continuous_learner(learner_state="ack_replay_collection")
    set_test_credit(learner, (395,))
    actor_before = {
        name: value.detach().clone()
        for name, value in learner.residual_actor.state_dict().items()
    }
    monkeypatch.setattr(
        learner,
        "_refresh_replay",
        lambda: SimpleNamespace(critic_td_valid_rows=99),
    )
    result = learner(object())
    assert result["learner_state"] == "ack_replay_collection"
    assert result["learner_critic_steps"] == result["learner_actor_steps"] == 0
    assert learner.learner["runtime"]["counters"] == {
        "twin_q_optimizer_steps": 0,
        "residual_actor_optimizer_steps": 0,
        "residual_actor_update_attempts": 0,
        "residual_actor_updates_skipped_no_gradient": 0,
        "twin_q_target_update_steps": 0,
        "critic_sample_draws": 0,
        "policy_sample_draws": 0,
        "human_sample_draws": 0,
    }
    assert all(
        torch.equal(actor_before[name], value)
        for name, value in learner.residual_actor.state_dict().items()
    )


def test_1000_rows_three_episodes_runs_critic_warmup_then_starts_training(
    monkeypatch,
) -> None:
    learner = tiny_continuous_learner(learner_state="ack_replay_collection")
    learner.training_policy = ResidualActorCriticSchedule(
        checkpoint_on_warmup_complete=True,
        checkpoint_on_candidate_activation=False,
    )
    replay = SimpleNamespace(
        critic_td_valid_rows=1000, critic_rows_per_episode=(334, 333, 333)
    )
    monkeypatch.setattr(
        learner_server.warmup,
        "count_sealed_critic_td_valid_transitions",
        lambda _root: 1000,
    )
    monkeypatch.setattr(learner, "_refresh_replay", lambda: replay)
    actor_before = {
        name: value.detach().clone()
        for name, value in learner.residual_actor.state_dict().items()
    }
    calls = []

    def critic_update(_coordinator, _replay, *, warmup):
        assert warmup is True
        calls.append(1)
        runtime = learner.learner["runtime"]
        runtime["ack_critic_warmup_steps"] += 1
        runtime["counters"]["twin_q_optimizer_steps"] += 1
        runtime["counters"]["twin_q_target_update_steps"] += 1
        return 0.25

    monkeypatch.setattr(learner, "_critic_update", critic_update)
    checkpoint_calls = []
    monkeypatch.setattr(
        learner,
        "save_checkpoint",
        lambda: checkpoint_calls.append(1) or Path("warmup-checkpoint"),
    )
    result = learner(object())
    assert len(calls) == 256
    assert result["ack_critic_warmup_steps"] == 256
    assert result["learner_actor_steps"] == 0
    assert learner.learner["runtime"]["learner_state"] == "residual_actor_critic_training"
    assert learner.learner["runtime"]["ack_critic_warmup_complete"] is True
    assert checkpoint_calls == [1]
    assert result["latest_checkpoint_path"] == "warmup-checkpoint"
    assert all(
        torch.equal(actor_before[name], value)
        for name, value in learner.residual_actor.state_dict().items()
    )


def test_no_currently_guard_eligible_td_row_waits_without_advancing_critic() -> None:
    learner = actor_update_test_learner()
    q1_target, q2_target = build_twin_q(hidden_dim=16, seed=31)[2:]
    learner.learner["q1_target"] = q1_target
    learner.learner["q2_target"] = q2_target
    learner.learner["critic_optimizer"] = torch.optim.Adam(
        (
            *learner.learner["q1"].parameters(),
            *learner.learner["q2"].parameters(),
        ),
        lr=3.0e-4,
    )
    learner.learner["config"]["optimizer"]["twin_q"] = {
        "grad_clip_norm": 10.0
    }
    learner.learner["config"]["objective"]["command_macro_discount"] = 0.99
    learner.latest_target_candidate_guard_rejected_count = 0
    learner.latest_target_candidate_guard_unknown_count = 0
    learner.latest_critic_td_available_count = 0
    critic_batch = batch(1)
    critic_batch.next_candidate_guard.valid[:] = False

    class SparseReplay:
        @staticmethod
        def iter_td_batches(*_args, **_kwargs):
            yield critic_batch

    counters_before = dict(learner.learner["runtime"]["counters"])
    q_before = {
        name: value.detach().clone()
        for name, value in learner.learner["q1"].state_dict().items()
    }
    result = learner._critic_update(
        InferencePriorityCoordinator(), SparseReplay(), warmup=False
    )
    assert result is None
    assert learner.latest_critic_td_available_count == 0
    assert learner.latest_target_candidate_guard_unknown_count == 1
    assert learner.learner["runtime"]["counters"] == counters_before
    assert all(
        torch.equal(q_before[name], value)
        for name, value in learner.learner["q1"].state_dict().items()
    )


@pytest.mark.parametrize("warmup", [True, False], ids=["warmup", "joint"])
def test_nonfinite_critic_gradient_does_not_update_parameters_or_counters(
    monkeypatch, warmup: bool
) -> None:
    learner = actor_update_test_learner()
    q1_target, q2_target = build_twin_q(hidden_dim=16, seed=31)[2:]
    learner.learner["q1_target"] = q1_target
    learner.learner["q2_target"] = q2_target
    learner.learner["critic_optimizer"] = torch.optim.Adam(
        (
            *learner.learner["q1"].parameters(),
            *learner.learner["q2"].parameters(),
        ),
        lr=3.0e-4,
    )
    learner.learner["config"]["optimizer"]["twin_q"] = {
        "grad_clip_norm": 10.0
    }
    learner.learner["config"]["objective"]["command_macro_discount"] = 0.99
    critic_batch = batch(1)
    critic_batch.session_ids = ("session",)
    critic_batch.episode_ids = ("episode",)

    class FiniteForwardNonfiniteBackward(torch.autograd.Function):
        @staticmethod
        def forward(ctx, parameter):
            return parameter.new_tensor(1.0)

        @staticmethod
        def backward(ctx, grad_output):
            parameter = next(learner.learner["q1"].parameters())
            return torch.full_like(parameter, float("inf"))

    def nonfinite_gradient_loss(*_args, **_kwargs):
        parameter = next(learner.learner["q1"].parameters())
        loss = FiniteForwardNonfiniteBackward.apply(parameter)
        assert torch.isfinite(loss)
        return SimpleNamespace(
            total=loss,
            td_valid_count=1,
            target_candidate_guard_rejected_count=0,
            target_candidate_guard_unknown_count=0,
        )

    class Replay:
        @staticmethod
        def iter_td_batches(*_args, **_kwargs):
            yield critic_batch

    monkeypatch.setattr(
        learner_server, "residual_critic_loss", nonfinite_gradient_loss
    )
    module_names = ("q1", "q2", "q1_target", "q2_target")
    parameters_before = {
        module_name: {
            name: value.detach().clone()
            for name, value in learner.learner[module_name].state_dict().items()
        }
        for module_name in module_names
    }
    runtime = learner.learner["runtime"]
    counters_before = dict(runtime["counters"])
    warmup_steps_before = int(runtime["ack_critic_warmup_steps"])
    partial_steps_before = int(runtime["partial_cycle_q_updates"])

    with pytest.raises(RuntimeError, match="FORCERFT_CRITIC_GRADIENT_NONFINITE"):
        learner._critic_update(
            InferencePriorityCoordinator(), Replay(), warmup=warmup
        )

    for module_name in module_names:
        assert all(
            torch.equal(parameters_before[module_name][name], value)
            for name, value in learner.learner[module_name].state_dict().items()
        )
    assert learner.learner["critic_optimizer"].state == {}
    for counter_name in (
        "twin_q_optimizer_steps",
        "twin_q_target_update_steps",
        "residual_actor_optimizer_steps",
        "residual_actor_update_attempts",
        "residual_actor_updates_skipped_no_gradient",
    ):
        assert runtime["counters"][counter_name] == counters_before[counter_name]
    assert runtime["ack_critic_warmup_steps"] == warmup_steps_before
    assert runtime["partial_cycle_q_updates"] == partial_steps_before


def test_unavailable_td_does_not_complete_joint_cycle(monkeypatch) -> None:
    learner = tiny_continuous_learner(
        learner_state="residual_actor_critic_training", warmup_updates=256
    )
    replay = SimpleNamespace(
        recorded_transition_rows=100,
        critic_td_valid_rows=100,
    )
    monkeypatch.setattr(learner, "_refresh_replay", lambda: replay)
    monkeypatch.setattr(
        learner,
        "_critic_update",
        lambda _coordinator, _replay, *, warmup: None,
    )
    monkeypatch.setattr(
        learner,
        "_actor_update",
        lambda *_args: (_ for _ in ()).throw(
            AssertionError("Actor must not run without its two Critic updates")
        ),
    )

    result = learner(InferencePriorityCoordinator())
    assert result["waiting_for_mappable_td"] is True
    assert result["learner_critic_steps"] == 0
    assert result["learner_actor_steps"] == 0
    assert learner.learner["runtime"]["residual_actor_critic_cycles"] == 0


def test_partial_q_cycle_resumes_without_claiming_or_repeating_completed_q(
    monkeypatch,
) -> None:
    learner = tiny_continuous_learner(
        learner_state="residual_actor_critic_training", warmup_updates=256
    )
    replay = SimpleNamespace(critic_td_valid_rows=100)
    monkeypatch.setattr(learner, "_refresh_replay", lambda: replay)
    outcomes = iter((0.5, None, 0.4))

    def critic_update(_coordinator, _replay, *, warmup):
        assert warmup is False
        value = next(outcomes)
        if value is not None:
            counters = learner.learner["runtime"]["counters"]
            counters["twin_q_optimizer_steps"] += 1
            counters["twin_q_target_update_steps"] += 1
            learner.learner["runtime"]["partial_cycle_q_updates"] += 1
        return value

    def actor_update(_coordinator, _replay):
        counters = learner.learner["runtime"]["counters"]
        counters["residual_actor_update_attempts"] += 1
        counters["residual_actor_updates_skipped_no_gradient"] += 1
        return {
            "total": 0.1,
            "value": 0.0,
            "applied": False,
            "skip_reason": "no_effective_gradient",
            "grad_norm": 0.0,
            "support_available": False,
        }

    monkeypatch.setattr(learner, "_critic_update", critic_update)
    monkeypatch.setattr(learner, "_actor_update", actor_update)
    first = learner(InferencePriorityCoordinator())
    assert first["waiting_for_mappable_td"] is True
    assert first["partial_cycle_q_updates"] == 1
    assert learner.learner["runtime"]["residual_actor_critic_cycles"] == 0

    second = learner(InferencePriorityCoordinator())
    assert second["residual_actor_critic_cycle"] == 1
    assert second["learner_critic_steps"] == 1
    assert second["learner_polyak_steps"] == 1
    assert learner.learner["runtime"]["partial_cycle_q_updates"] == 0
    assert learner.learner["runtime"]["counters"]["twin_q_optimizer_steps"] == 258


@pytest.mark.parametrize("actor_applied", [True, False], ids=["applied", "skipped"])
def test_partial_two_resume_completes_cycle_and_publishes_once(
    monkeypatch, tmp_path: Path, actor_applied: bool
) -> None:
    learner = actor_update_test_learner()
    runtime = learner.learner["runtime"]
    runtime["residual_actor_critic_cycles"] = 99
    runtime["partial_cycle_q_updates"] = 2
    counters = runtime["counters"]
    counters["twin_q_optimizer_steps"] = 456
    counters["twin_q_target_update_steps"] = 456
    counters["residual_actor_update_attempts"] = 99
    counters["residual_actor_optimizer_steps"] = 99 if actor_applied else 0
    counters["residual_actor_updates_skipped_no_gradient"] = (
        0 if actor_applied else 99
    )
    learner.checkpoint_root = tmp_path / "training_checkpoints"
    monkeypatch.setattr(
        learner,
        "_refresh_replay",
        lambda: SimpleNamespace(critic_td_valid_rows=100),
    )
    monkeypatch.setattr(
        learner,
        "_critic_update",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("completed Q updates must not be repeated")
        ),
    )
    actor_calls = []
    monkeypatch.setattr(
        learner,
        "_actor_update",
        lambda *_args: actor_calls.append(1)
        or {
            "total": 0.1,
            "value": -0.2,
            "applied": actor_applied,
            "skip_reason": None if actor_applied else "no_effective_gradient",
            "grad_norm": 1.0 if actor_applied else 0.0,
            "support_available": actor_applied,
        },
    )

    result = learner(InferencePriorityCoordinator())

    assert result["residual_actor_critic_cycle"] == 100
    assert result["learner_critic_steps"] == 0
    assert result["learner_polyak_steps"] == 0
    assert result["latest_critic_td_loss"] is None
    assert result["learner_actor_update_attempts"] == 1
    assert result["learner_actor_steps"] == int(actor_applied)
    assert result["actor_update_applied"] is actor_applied
    assert actor_calls == [1]
    assert runtime["partial_cycle_q_updates"] == 0
    assert counters["twin_q_optimizer_steps"] == 456
    assert counters["twin_q_target_update_steps"] == 456
    assert counters["residual_actor_update_attempts"] == 100
    assert counters["residual_actor_optimizer_steps"] == (
        100 if actor_applied else 0
    )
    assert counters["residual_actor_updates_skipped_no_gradient"] == (
        0 if actor_applied else 100
    )

    async_runtime = learner_server.AsyncResidualActorCriticRuntime.__new__(
        learner_server.AsyncResidualActorCriticRuntime
    )
    async_runtime.learner_job = learner
    async_runtime.coordinator = InferencePriorityCoordinator()
    async_runtime._policy = learner.training_policy
    staged = []
    async_runtime._stage_actor_candidate = staged.append

    assert async_runtime._process_completed_cycle_events(result) == (True, False)
    assert async_runtime._process_completed_cycle_events(result) == (False, False)
    assert len(staged) == 1
    assert runtime["scheduling"]["publication_event_count"] == 1
    assert runtime["scheduling"]["last_published_cycle"] == 100
    assert len(list(tmp_path.glob("**/cycle_000100.json"))) == 1


def test_residual_training_cycle_is_exactly_two_critic_and_one_actor(
    monkeypatch,
) -> None:
    learner = tiny_continuous_learner(
        learner_state="residual_actor_critic_training", warmup_updates=256
    )
    replay = SimpleNamespace(
        critic_td_valid_rows=100, critic_rows_per_episode=(100,)
    )
    monkeypatch.setattr(
        learner_server.warmup,
        "count_sealed_critic_td_valid_transitions",
        lambda _root: 100,
    )
    monkeypatch.setattr(learner, "_refresh_replay", lambda: replay)
    critic_calls = []
    actor_calls = []
    monkeypatch.setattr(
        learner,
        "_critic_update",
        lambda _coordinator, _replay, *, warmup: critic_calls.append(warmup)
        or 0.5,
    )
    monkeypatch.setattr(
        learner,
        "_actor_update",
        lambda _coordinator, _replay: actor_calls.append(1)
        or {
            "total": 0.1,
            "value": -0.2,
            "applied": True,
            "skip_reason": None,
            "grad_norm": 1.0,
            "support_available": True,
        },
    )
    result = learner(object())
    assert critic_calls == [False, False]
    assert actor_calls == [1]
    assert result["learner_critic_steps"] == 2
    assert result["learner_actor_steps"] == 1
    assert result["residual_actor_critic_cycle"] == 1


def actor_update_test_learner() -> learner_server.ResidualActorCriticLearner:
    learner = learner_server.ResidualActorCriticLearner.__new__(
        learner_server.ResidualActorCriticLearner
    )
    actor, actor_target = make_residual_actor_pair(
        hidden_dim=16,
        max_normalized_residual=0.1,
        residual_cap6=[0.1] * 6,
        residual_bound_mode=RESIDUAL_BOUND_MODE_SCALAR,
    )
    q1, q2, _q1_target, _q2_target = build_twin_q(hidden_dim=16, seed=17)
    learner.device = torch.device("cpu")
    learner.training_policy = ResidualActorCriticSchedule(
        residual_policy_value_batch_size=8,
        human_residual_imitation_batch_size=8,
        training_checkpoint_interval_cycles=1_000,
        checkpoint_on_warmup_complete=False,
    )
    learner.latest_residual_actor_output_norm = 0.0
    learner.latest_actor_update_ms = 0.0
    learner.latest_critic_update_ms = 0.0
    learner.latest_cycle_ms = 0.0
    learner.latest_replay_refresh_ms = 0.0
    learner.nonzero_behavior_residual_rows = 0
    learner._state_lock = threading.RLock()
    learner.sampled_session_ids = set()
    learner.sampled_episode_ids = set()
    learner._expected_admission_id = None
    learner._admission_progress = {}
    learner._loaded_episode_keys = set()
    learner.replay = None
    learner.learner = {
        "residual_actor": actor,
        "residual_actor_target": actor_target,
        "q1": q1,
        "q2": q2,
        "residual_actor_optimizer": torch.optim.Adam(
            actor.parameters(), lr=1.0e-4
        ),
        "config": {
            "environment": {"random_seed": 4404},
            "wrist_wrench_residual_actor": {
                "residual_bound_mode": RESIDUAL_BOUND_MODE_SCALAR
            },
            "optimizer": {
                "residual_actor": {"grad_clip_norm": 1.0},
                "twin_q_polyak_tau": 0.005,
            },
            "objective": {
                "value_objective_weight": 1.0,
                "residual_magnitude_penalty_weight": 0.01,
                "human_residual_imitation_weight": 1.0,
            },
        },
        "runtime": {
            "learner_state": "residual_actor_critic_training",
            "ack_critic_warmup_complete": True,
            "ack_critic_warmup_steps": 256,
            "residual_actor_critic_cycles": 0,
            "partial_cycle_q_updates": 0,
            "active_residual_policy_revision": (
                "task3-residual-policy-step-000000"
            ),
            "online_adaptation_id": "task3-ack-residual-gradient-test",
            "counters": {
                "twin_q_optimizer_steps": 256,
                "residual_actor_optimizer_steps": 0,
                "residual_actor_update_attempts": 0,
                "residual_actor_updates_skipped_no_gradient": 0,
                "twin_q_target_update_steps": 256,
                "critic_sample_draws": 0,
                "policy_sample_draws": 0,
                "human_sample_draws": 0,
            },
            "replay": {
                "critic_td_valid_rows": 100,
                "actor_q_valid_rows": 100,
                "human_residual_valid_rows": 0,
                "per_episode_critic_row_counts": {},
            },
            "scheduling": {
                "mode": "continuous_async",
                "candidate_export_generation": "test-generation",
                "last_publish_attempt_cycle": 0,
                "last_published_cycle": 0,
            },
        },
    }
    learner.human_supervision_diagnostics = {}
    set_test_credit(learner, (534, 533, 533))
    return learner


def test_single_replay_population_continues_beyond_old_ten_cycle_limit(
    monkeypatch,
) -> None:
    accepted = [[0.2, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0]] * 3
    replay = policy_replay(
        schema_version=ACK_RESIDUAL_TRANSITION_SCHEMA_VERSION,
        base_action=accepted,
    )
    replay.rows = replay.rows * 100
    learner = actor_update_test_learner()
    monkeypatch.setattr(learner, "_refresh_replay", lambda: replay)

    def critic_update(_coordinator, _replay, *, warmup):
        assert warmup is False
        counters = learner.learner["runtime"]["counters"]
        counters["twin_q_optimizer_steps"] += 1
        counters["twin_q_target_update_steps"] += 1
        return 0.25

    monkeypatch.setattr(learner, "_critic_update", critic_update)
    actor_before = {
        name: value.detach().clone()
        for name, value in learner.learner["residual_actor"].state_dict().items()
    }
    target_before = {
        name: value.detach().clone()
        for name, value in learner.learner[
            "residual_actor_target"
        ].state_dict().items()
    }
    coordinator = InferencePriorityCoordinator()
    for _ in range(157):
        result = learner(coordinator)
        assert result["actor_update_attempted"] is True
        assert result["actor_update_applied"] is False
        assert result["actor_update_skip_reason"] == "no_effective_gradient"
        assert result["actor_grad_norm"] == 0.0

    runtime = learner.learner["runtime"]
    assert runtime["residual_actor_critic_cycles"] == 157
    assert runtime["counters"] == {
        "twin_q_optimizer_steps": 570,
        "residual_actor_optimizer_steps": 0,
        "residual_actor_update_attempts": 157,
        "residual_actor_updates_skipped_no_gradient": 157,
            "twin_q_target_update_steps": 570,
            "critic_sample_draws": 0,
            "policy_sample_draws": 1256,
            "human_sample_draws": 0,
        }
    assert learner.learner["residual_actor_optimizer"].state == {}
    assert learner.training_policy.candidate_due(100)
    assert all(
        torch.equal(actor_before[name], value)
        for name, value in learner.learner[
            "residual_actor"
        ].state_dict().items()
    )
    assert all(
        torch.equal(target_before[name], value)
        for name, value in learner.learner[
            "residual_actor_target"
        ].state_dict().items()
    )


def test_candidate_cadence_uses_cycle_not_effective_actor_updates(
    tmp_path: Path,
) -> None:
    learner = actor_update_test_learner()
    counters = learner.learner["runtime"]["counters"]
    counters["residual_actor_update_attempts"] = 157
    counters["residual_actor_updates_skipped_no_gradient"] = 157
    learner.checkpoint_root = tmp_path / "training_checkpoints"
    replay = human_replay()
    coordinator = InferencePriorityCoordinator()

    for expected_step in range(1, 11):
        metrics = learner._actor_update(coordinator, replay)
        assert metrics["applied"] is True
        assert metrics["grad_norm"] > 0.0
        counters["residual_actor_update_attempts"] += 1
        counters["residual_actor_optimizer_steps"] += 1
        assert counters["residual_actor_optimizer_steps"] == expected_step
        assert not learner.training_policy.candidate_due(expected_step)

    assert counters["residual_actor_update_attempts"] == 167
    assert counters["residual_actor_updates_skipped_no_gradient"] == 157
    assert learner.learner["residual_actor_optimizer"].state
    candidate = learner.export_actor_candidate(100)
    assert candidate is not None
    assert "cycle-000100" in candidate["revision_id"]
    assert candidate["residual_actor_optimizer_steps"] == 10
    assert (candidate["checkpoint"] / "residual_actor.pt").is_file()
    candidate_state = torch.load(
        candidate["checkpoint"] / "candidate_state.pt",
        map_location="cpu",
        weights_only=False,
    )
    assert candidate_state["checkpoint_kind"] == learner_server.CANDIDATE_CHECKPOINT_KIND
    assert candidate_state["online_semantics_version"] == ONLINE_SEMANTICS_VERSION
    assert len(candidate_state["actor_content_sha256"]) == 64
