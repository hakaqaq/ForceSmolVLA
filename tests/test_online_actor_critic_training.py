from __future__ import annotations

import inspect
from pathlib import Path
import sys
import threading
from types import SimpleNamespace

import numpy as np
import torch


ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import serve_forcerft_residual_actor_critic as learner_server  # noqa: E402

from forcesmolvla.rft.online.residual_actor_critic_runtime import (
    InferencePriorityCoordinator,
    ResidualActorCriticSchedule,
)
from forcesmolvla.rft.online.replay_training import (
    ACK_RESIDUAL_TRANSITION_SCHEMA_VERSION,
    LEGACY_ACK_RESIDUAL_TRANSITION_SCHEMA_VERSIONS,
    OnlineResidualReplay,
    ProductionAckMacro,
    algorithm_hyperparameters,
    load_common_actor_critic_config,
)
from forcesmolvla.rft.online.transition_authority import (
    AckMacro,
    ActorQEligibility,
    DISPATCH_DECISION_CRITIC_CONTRACT_VERSION,
    ONLINE_SEMANTICS_VERSION,
    normalized_behavior_residual,
)
from forcesmolvla.rft.critic import (
    RESIDUAL_ACTION_OFFSET,
    RESIDUAL_ACTION_WIDTH,
    build_twin_q,
)
from forcesmolvla.rft.online.training_losses import (
    residual_actor_loss,
    residual_critic_loss,
)
from forcesmolvla.rft.residual_actor import make_residual_actor_pair


class ConstantQ(torch.nn.Module):
    def __init__(self, value: float) -> None:
        super().__init__()
        self.value = torch.nn.Parameter(torch.tensor(value))
        self.batch_sizes: list[int] = []
        self.residuals: list[torch.Tensor] = []
        self.sources: list[torch.Tensor] = []
        self.grippers: list[torch.Tensor] = []

    def forward(
        self, state, wrench, wrench_delta, base, residual, mask, source, gripper
    ):
        del wrench, wrench_delta, base, mask
        self.batch_sizes.append(len(state))
        self.residuals.append(residual.detach().clone())
        self.sources.append(source.detach().clone())
        self.grippers.append(gripper.detach().clone())
        return self.value.expand(len(state)) + residual.flatten(1).mean(1) * 0.0


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


def decision_context(
    *, timestamp_ns: int, base_absolute: object
) -> dict[str, object]:
    return {
        "online_semantics_version": ONLINE_SEMANTICS_VERSION,
        "valid_for_residual_training": True,
        "invalid_reason": None,
        "decision_monotonic_ns": timestamp_ns,
        "state7_absolute": [0.0] * 7,
        "wrench6_calibrated_tcp": [0.0] * 6,
        "wrench_delta6_calibrated_tcp_100ms": [0.0] * 6,
        "wrench_delta_interval_ns": 0,
        "base_absolute_action7": base_absolute,
        "candidate_acceptance_mapping": {
            "mapping_kind": "recorded_ack_point_only",
            "identity_valid": False,
        },
    }


def human_replay(*, terminated: bool = True) -> OnlineResidualReplay:
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
        "eligibility": {"actor_q_valid": True},
        "human_residual_valid": True,
        "pre_takeover_base_absolute_action7": base,
        "base_absolute_action_k7": np.repeat(
            np.asarray(base)[None, :], 3, axis=0
        ).tolist(),
        "accepted_absolute_action_k7": accepted.tolist(),
        "residual_decision_context": decision_context(
            timestamp_ns=1_000_000_000, base_absolute=base
        ),
        "next_residual_decision_context": None,
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


def policy_replay(*, schema_version: str, base_action: object) -> OnlineResidualReplay:
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
        "identity": {"episode_id": "policy-episode"},
        "action_source": "policy",
        "observation": observation,
        "next_observation": {
            **observation,
            "materialized_timestamp_monotonic_ns": 1_100_000_000,
        },
        "outcome": {"reward": 0.0, "terminated": True, "truncated": False},
        "eligibility": {"actor_q_valid": True},
        "accepted_absolute_action_k7": accepted.tolist(),
        "next_residual_decision_context": None,
    }
    if base_action is not None:
        base_absolute = np.asarray(base_action, dtype=np.float64)[0].tolist()
        transition["base_absolute_action_k7"] = np.repeat(
            np.asarray(base_absolute)[None, :], 3, axis=0
        ).tolist()
        transition["residual_decision_context"] = decision_context(
            timestamp_ns=1_000_000_000,
            base_absolute=base_absolute,
        )
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
        base_action_k6=zeros_k6,
        behavior_residual_k6=zeros_k6,
        action_mask=mask,
        control_source=torch.zeros(batch_size, 1),
        gripper_command=torch.zeros(batch_size, 1),
        candidate_acceptance_identity_valid=torch.ones(
            batch_size, dtype=torch.bool
        ),
        next_state7=zeros7,
        next_wrench6=zeros6,
        next_wrench_delta6=zeros6,
        next_base_action_k6=zeros_k6,
        next_action_mask=mask,
        next_base_valid=torch.ones(batch_size, dtype=torch.bool),
        next_control_source=torch.zeros(batch_size, 1),
        next_gripper_command=torch.zeros(batch_size, 1),
        next_candidate_acceptance_identity_valid=torch.zeros(
            batch_size, dtype=torch.bool
        ),
        next_recorded_proposal_residual6=zeros6,
        next_recorded_behavior_residual_k6=zeros_k6,
        next_recorded_point_valid=torch.ones(batch_size, dtype=torch.bool),
        reward=torch.ones(batch_size),
        terminated=torch.tensor([False, True][:batch_size]),
        truncated=torch.zeros(batch_size, dtype=torch.bool),
        actor_q_valid=torch.ones(batch_size, dtype=torch.bool),
        human_residual_target6=zeros6,
        human_residual_valid=torch.zeros(batch_size, dtype=torch.bool),
    )


def test_residual_critic_td_target_is_ack_only_and_bootstrap_safe() -> None:
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


def test_current_and_target_q_use_only_proven_accepted_candidates() -> None:
    actor = ScalarResidualActor()
    q1, q2 = ConstantQ(0.0), ConstantQ(1.0)
    policy = batch(1)
    policy.candidate_acceptance_identity_valid[:] = False
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
    assert losses.actor_q_mapping_unavailable_count == 1
    assert q1.batch_sizes == q2.batch_sizes == []

    policy.candidate_acceptance_identity_valid[:] = True
    policy.control_source[:] = 0.0
    policy.gripper_command[:] = 0.75
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
    assert torch.equal(q1.sources[-1], torch.zeros(1, 1))
    assert torch.equal(q1.grippers[-1], torch.full((1, 1), 0.75))

    critic_batch = batch(1)
    critic_batch.next_recorded_behavior_residual_k6[:] = 0.25
    critic_batch.next_gripper_command[:] = -0.4
    target_q1, target_q2 = ConstantQ(2.0), ConstantQ(3.0)
    details = residual_critic_loss(
        q1,
        q2,
        target_q1,
        target_q2,
        TargetActor(),
        critic_batch,
        gamma=0.5,
        return_details=True,
    )
    assert details.td_valid_count == 1
    assert details.target_candidate_unavailable_count == 0
    assert torch.equal(
        target_q1.residuals[-1], torch.full((1, 3, 6), 0.25)
    )
    assert torch.equal(target_q1.grippers[-1], torch.full((1, 1), -0.4))


def test_unmappable_successor_is_not_relabelled_as_terminal() -> None:
    critic_batch = batch(1)
    critic_batch.next_candidate_acceptance_identity_valid[:] = False
    q1, q2 = ConstantQ(0.0), ConstantQ(1.0)
    q1_target, q2_target = ConstantQ(2.0), ConstantQ(3.0)
    # A real ACK exists, but only for proposal zero.  A different target-Actor
    # candidate cannot reuse that historical acceptance point.
    target_actor = TargetActor(0.1)
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
    assert details.target_candidate_unavailable_count == 1
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


def test_critic_conditions_policy_and_human_on_accepted_gripper() -> None:
    critic_batch = batch(2)
    critic_batch.terminated[:] = True
    critic_batch.control_source[:] = torch.tensor([[0.0], [1.0]])
    critic_batch.gripper_command[:] = torch.tensor([[0.85], [-0.25]])
    q1, q2 = ConstantQ(0.0), ConstantQ(1.0)
    residual_critic_loss(
        q1, q2, ConstantQ(2.0), ConstantQ(3.0), TargetActor(), critic_batch, 0.99
    )
    assert torch.equal(q1.sources[-1], torch.tensor([[0.0], [1.0]]))
    assert torch.equal(q1.grippers[-1], torch.tensor([[0.85], [-0.25]]))


def test_human_imitation_projects_only_bc_target() -> None:
    actor = ScalarResidualActor()
    human = batch(2)
    human.human_residual_valid[:] = True
    human.human_residual_target6[:] = 0.8
    human.behavior_residual_k6[:] = 0.8
    raw_target = human.human_residual_target6.clone()
    raw_behavior = human.behavior_residual_k6.clone()
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
    assert torch.equal(human.behavior_residual_k6, raw_behavior)


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
    assert q1.batch_sizes == q2.batch_sizes == [1]
    assert actor.batch_sizes == [2]
    assert torch.equal(losses.human, torch.zeros_like(losses.human))


def test_valid_human_residual_reaches_critic_and_unlocks_action_columns() -> None:
    replay = human_replay()
    row = replay.rows[0]
    assert row["human_residual_valid"] is True
    assert np.count_nonzero(row["behavior_residual_k6"]) > 0
    assert np.count_nonzero(row["human_residual_target6"]) > 0

    q1, q2, q1_target, q2_target = build_twin_q(hidden_dim=16, seed=13)
    target_actor = TargetActor()
    optimizer = torch.optim.Adam((*q1.parameters(), *q2.parameters()), lr=3e-4)
    before = q1.layers[0].weight[
        :, RESIDUAL_ACTION_OFFSET : RESIDUAL_ACTION_OFFSET + RESIDUAL_ACTION_WIDTH
    ].detach().clone()
    critic_batch = replay.sample(8, device=torch.device("cpu"), seed=1)
    assert critic_batch is not None
    optimizer.zero_grad(set_to_none=True)
    residual_critic_loss(
        q1, q2, q1_target, q2_target, target_actor, critic_batch, gamma=0.99
    ).backward()
    optimizer.step()
    after = q1.layers[0].weight[
        :, RESIDUAL_ACTION_OFFSET : RESIDUAL_ACTION_OFFSET + RESIDUAL_ACTION_WIDTH
    ].detach()
    assert not torch.equal(before, after)


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
        "identity": {"episode_id": "dispatch-episode"},
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
        "base_absolute_action_k7": np.repeat(
            np.asarray(chunk[2])[None, :], 3, axis=0
        ).tolist(),
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
    assert np.allclose(row["behavior_residual_k6"], 0.01)
    assert row["next_base_valid"] is True
    assert np.array_equal(row["control_source"], [0.0])
    assert np.allclose(row["gripper_command"], [0.085])
    assert row["next_recorded_point_valid"] is True
    assert np.array_equal(row["next_control_source"], [0.0])
    assert np.allclose(row["next_gripper_command"], [0.085])


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

    missing_next_base = human_replay(terminated=False)
    assert missing_next_base.rows == ()
    assert missing_next_base.next_base_missing_rows == 1


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


def test_online_schedule_is_2q_1actor_and_episode_bounded() -> None:
    policy = ResidualActorCriticSchedule()
    assert policy.twin_q_updates_per_cycle == 2
    assert policy.residual_actor_updates_per_cycle == 1
    assert policy.cycles_for_admission(100) == 2
    assert policy.cycles_for_admission(400) == 7
    assert policy.cycles_for_admission(641) == 10
    assert policy.cycles_for_observed_admission(
        new_critic_td_valid_rows=99,
        total_critic_td_valid_rows=99,
    ) == 0
    assert policy.cycles_for_observed_admission(
        new_critic_td_valid_rows=1,
        total_critic_td_valid_rows=100,
    ) == 1
    assert policy.residual_actor_critic_cycle_budget((100, 400, 641)) == 19
    assert not policy.candidate_due(9)
    assert policy.candidate_due(10)
    assert ResidualActorCriticSchedule(
        admitted_rows_per_cycle=32,
        max_cycles_per_admitted_episode=20,
    ).cycles_for_admission(400) == 13


def test_task_profiles_cannot_override_algorithm_parameters() -> None:
    task2 = load_common_actor_critic_config("task2")
    task3 = load_common_actor_critic_config("task3")
    assert algorithm_hyperparameters(task2) == algorithm_hyperparameters(task3)
    assert task2["task"] != task3["task"]
    assert task2["residual_actor_critic_training"] == {
        "admitted_rows_per_cycle": 64,
        "twin_q_updates_per_cycle": 2,
        "residual_actor_updates_per_cycle": 1,
        "max_cycles_per_admitted_episode": 10,
        "residual_candidate_interval_actor_steps": 10,
        "training_checkpoint_interval_cycles": 20,
        "retained_training_checkpoint_count": 10,
        "checkpoint_on_warmup_complete": True,
        "checkpoint_on_candidate_activation": True,
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
        checkpoint_on_candidate_activation=False,
    )
    learner._loaded_episode_keys = set()
    learner._admission_progress = {}
    learner._expected_admission_id = None
    learner._joint_cycle_budget = 0
    learner.latest_replay_refresh_ms = 0.0
    learner.latest_critic_update_ms = 0.0
    learner.latest_actor_update_ms = 0.0
    learner.latest_cycle_ms = 0.0
    learner.learner = {
        "residual_actor": actor,
        "runtime": {
            "learner_state": learner_state,
            "ack_critic_warmup_complete": learner_state == "residual_actor_critic_training",
            "ack_critic_warmup_steps": warmup_updates,
            "residual_actor_critic_cycles": 0,
            "counters": {
                "twin_q_optimizer_steps": warmup_updates,
                "residual_actor_optimizer_steps": 0,
                "residual_actor_update_attempts": 0,
                "residual_actor_updates_skipped_no_gradient": 0,
                "twin_q_target_update_steps": warmup_updates,
            },
            "replay": {
                "critic_td_valid_rows": 0,
                "actor_q_valid_rows": 0,
                "human_residual_valid_rows": 0,
            },
        },
    }
    return learner


def test_replay_refresh_loads_only_newly_sealed_episodes(monkeypatch) -> None:
    learner = tiny_continuous_learner(learner_state="ack_replay_collection")
    learner.normalizer = object()
    learner.current_session_id = None
    learner.unique_r_count = 0
    learner.r_macro_count = 0
    learner.next_base_missing_rows = 0
    learner.quarantined_current_schema_rows = 0
    learner.nonzero_behavior_residual_rows = 0
    signatures = [["a"]]
    monkeypatch.setattr(
        learner, "_episode_signature", lambda: tuple(signatures[0])
    )

    class FakeReplay:
        def __init__(self, _macros, _normalizer) -> None:
            self.counts: list[int] = []
            self.next_base_missing_rows = 0
            self.quarantined_current_schema_rows = 0
            self.nonzero_behavior_residual_rows = 0

        def append_macros(self, macros):
            macro = tuple(macros)[0]
            episode_id = macro.transition["identity"]["episode_id"]
            count = int(macro.transition["materialized_count"])
            self.counts.append(count)
            return {episode_id: count}

        @property
        def critic_rows_per_episode(self):
            return tuple(self.counts)

        @property
        def critic_td_valid_rows(self):
            return sum(self.counts)

        actor_q_valid_rows = property(lambda self: sum(self.counts))
        human_residual_valid_rows = property(lambda _self: 0)

    calls: list[str] = []

    def load_episode(_root, admission_id):
        calls.append(admission_id)
        episode_id = f"{admission_id}/episode"
        row = {
            "identity": {"episode_id": episode_id, "session_id": "old"},
            "materialized_count": {"a": 99, "b": 1, "c": 400}[admission_id],
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
    assert learner.admission_budget_status("a")["computed_cycle_budget"] == 0
    signatures[0].append("b")
    learner._refresh_replay()
    assert learner.admission_budget_status("b")["computed_cycle_budget"] == 1
    learner.learner["runtime"]["residual_actor_critic_cycles"] = 1
    signatures[0].append("c")
    learner.expect_admission("c")
    learner._refresh_replay()
    assert calls == ["a", "b", "c"]
    assert learner.learner["runtime"]["replay"]["loaded_episode_keys"] == [
        "a",
        "b",
        "c",
    ]
    assert learner.admission_budget_status("c") == {
        "episode_key": "c",
        "admitted_rows_for_latest_episode": 400,
        "computed_cycle_budget": 7,
        "cycle_count_at_admission_start": 1,
        "target_cycle_count_after_admission": 8,
        "completed_cycle_count_for_latest_admission": 0,
        "remaining_cycle_budget": 7,
    }


def test_collecting_does_not_update_actor_or_critic(monkeypatch) -> None:
    learner = tiny_continuous_learner(learner_state="ack_replay_collection")
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
    }
    assert all(
        torch.equal(actor_before[name], value)
        for name, value in learner.residual_actor.state_dict().items()
    )


def test_100_rows_runs_exactly_256_critic_warmup_then_starts_residual_training(
    monkeypatch,
) -> None:
    learner = tiny_continuous_learner(learner_state="ack_replay_collection")
    learner.training_policy = ResidualActorCriticSchedule(
        checkpoint_on_warmup_complete=True,
        checkpoint_on_candidate_activation=False,
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
    learner._joint_cycle_budget = 1
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
    actor, actor_target = make_residual_actor_pair(hidden_dim=16)
    q1, q2, _q1_target, _q2_target = build_twin_q(hidden_dim=16, seed=17)
    learner.device = torch.device("cpu")
    learner.training_policy = ResidualActorCriticSchedule(
        residual_policy_value_batch_size=8,
        human_residual_imitation_batch_size=8,
        training_checkpoint_interval_cycles=1_000,
        checkpoint_on_warmup_complete=False,
        checkpoint_on_candidate_activation=False,
    )
    learner.latest_residual_actor_output_norm = 0.0
    learner.latest_actor_update_ms = 0.0
    learner.latest_critic_update_ms = 0.0
    learner.latest_cycle_ms = 0.0
    learner.latest_replay_refresh_ms = 0.0
    learner.nonzero_behavior_residual_rows = 0
    learner._joint_cycle_budget = 0
    learner._expected_admission_id = None
    learner._admission_progress = {}
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
            },
            "replay": {
                "critic_td_valid_rows": 100,
                "actor_q_valid_rows": 100,
                "human_residual_valid_rows": 0,
            },
        },
    }
    return learner


def test_157_zero_residual_cycles_do_not_advance_actor_optimizer_or_candidate(
    monkeypatch,
) -> None:
    accepted = [[0.2, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0]] * 3
    replay = policy_replay(
        schema_version=ACK_RESIDUAL_TRANSITION_SCHEMA_VERSION,
        base_action=accepted,
    )
    replay.rows = replay.rows * 100
    learner = actor_update_test_learner()
    learner._joint_cycle_budget = 157
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
    }
    assert learner.learner["residual_actor_optimizer"].state == {}
    assert not learner.training_policy.candidate_due(0)
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


def test_candidate_waits_for_ten_effective_human_residual_actor_updates(
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
        assert counters["residual_actor_optimizer_steps"] == expected_step
        assert learner.training_policy.candidate_due(expected_step) is (
            expected_step == 10
        )

    assert counters["residual_actor_update_attempts"] == 167
    assert counters["residual_actor_updates_skipped_no_gradient"] == 157
    assert learner.learner["residual_actor_optimizer"].state
    candidate = learner.export_actor_candidate(10)
    assert candidate is not None
    assert candidate["revision_id"].endswith("000010")
    assert (candidate["checkpoint"] / "residual_actor.pt").is_file()
    candidate_state = torch.load(
        candidate["checkpoint"] / "candidate_state.pt",
        map_location="cpu",
        weights_only=False,
    )
    assert candidate_state == {
        "checkpoint_kind": learner_server.CANDIDATE_CHECKPOINT_KIND,
        "online_semantics_version": ONLINE_SEMANTICS_VERSION,
    }
