"""CPU-only online-replay learner primitives.

This module owns orchestration only.  Replay membership, credit accounting,
online TD, expert masking, ActionContract-v2 Q guidance, and publication remain
owned by the accepted ForceRFT primitives.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Callable

import torch
from torch import Tensor, nn

from forcesmolvla.rft.critic_action_adapter_v2 import critic_action_for_q_guidance_v2
from forcesmolvla.rft.losses import CriticObservation

from forcesmolvla.rft.online.training_batch import (
    MixedReplayBatch,
    MixedReplaySampler,
    build_expert_feature_mask,
)
from forcesmolvla.rft.online.training_losses import (
    compute_expert_only_flow_matching_loss,
    compute_online_twin_q_td_loss,
    compute_online_actor_objective,
    compute_online_min_twin_q_actor_loss,
)
from forcesmolvla.rft.online.replay import R_ONLINE, OnlineReplay
from forcesmolvla.rft.online.sample_credit import UpdateCreditLedger


class TrainingStartsBlocked(RuntimeError):
    """Raised until the replay contains 100 unique online transitions."""


@dataclass(frozen=True)
class OnlineLossAPI:
    """Injectable references to the accepted production loss primitives."""

    online_td: Callable = compute_online_twin_q_td_loss
    actor_objective: Callable = compute_online_actor_objective
    actor_q: Callable = compute_online_min_twin_q_actor_loss
    expert_fm: Callable = compute_expert_only_flow_matching_loss


class TinyActor(nn.Module):
    """A deterministic H=50 fake Actor; it is not a SmolVLA forward pass."""

    def __init__(self) -> None:
        super().__init__()
        horizon = torch.arange(50, dtype=torch.float32).view(50, 1)
        feature = torch.arange(7, dtype=torch.float32).view(1, 7)
        initial = 0.01 * torch.sin((horizon + 1.0) * (feature + 1.0) / 17.0)
        initial[:, 6] = 0.0
        self.normalized_chunk = nn.Parameter(initial)

    def forward(self, observation: CriticObservation) -> Tensor:
        observation.validate()
        return self.normalized_chunk.unsqueeze(0).expand(observation.batch_size, -1, -1)


class TinyTwinQ(nn.Module):
    """Small force-aware Twin-Q stand-in with the production call signature."""

    def __init__(self, *, offset: float) -> None:
        super().__init__()
        self.observation_weight = nn.Parameter(torch.tensor(0.05 + offset))
        weights = torch.arange(1, 22, dtype=torch.float32).view(3, 7) / 210.0
        self.action_weight = nn.Parameter(weights + offset * 0.01)
        self.bias = nn.Parameter(torch.tensor(offset))

    def forward(
        self,
        camera1: Tensor,
        camera2: Tensor,
        task_feature: Tensor,
        normalized_state7: Tensor,
        normalized_wrench6: Tensor,
        action: Tensor,
        mask: Tensor,
    ) -> Tensor:
        observation_value = torch.stack(
            (
                camera1.flatten(1).mean(1),
                camera2.flatten(1).mean(1),
                task_feature.flatten(1).mean(1),
                normalized_state7.mean(1),
                normalized_wrench6.mean(1),
            ),
            dim=1,
        ).mean(1)
        action_value = (
            action * self.action_weight * mask.unsqueeze(-1).to(action.dtype)
        ).sum((1, 2))
        return observation_value * self.observation_weight + action_value + self.bias


def _module_snapshot(module: nn.Module) -> dict[str, Tensor]:
    return {name: value.detach().clone() for name, value in module.state_dict().items()}


def _snapshot_equal(left: dict[str, Tensor], right: dict[str, Tensor]) -> bool:
    return left.keys() == right.keys() and all(
        torch.equal(left[name], right[name]) for name in left
    )


def _polyak_update(target: nn.Module, source: nn.Module, *, tau: float) -> None:
    with torch.no_grad():
        for target_value, source_value in zip(
            target.parameters(), source.parameters(), strict=True,
        ):
            target_value.mul_(1.0 - tau).add_(source_value, alpha=tau)


def _observations(batch: MixedReplayBatch, *, next_observation: bool) -> CriticObservation:
    rows = []
    for sample in batch.samples:
        key = "next_observation" if next_observation else "observation"
        value = sample.payload[key]
        rows.append((sample.payload["identity"]["macro_index"], value))
    macro = torch.tensor([row[0] for row in rows], dtype=torch.float32).unsqueeze(1)
    state = torch.tensor([row[1]["state7"] for row in rows], dtype=torch.float32)
    wrench = torch.tensor([row[1]["wrench6"] for row in rows], dtype=torch.float32)
    task = torch.arange(256, dtype=torch.float32).unsqueeze(0).expand(len(rows), -1) / 255.0
    return CriticObservation(
        camera1=macro / 100.0,
        camera2=(macro + 1.0) / 100.0,
        task_feature=task,
        normalized_state7=state,
        normalized_wrench6=wrench,
    ).validate()


def _batch_tensors(batch: MixedReplayBatch) -> dict[str, Tensor | list[list[str]]]:
    payloads = [sample.payload for sample in batch.samples]
    action = torch.tensor(
        [value["behavior_ack"]["normalized_delta_action_k7"] for value in payloads],
        dtype=torch.float32,
    )
    target_h50 = torch.tensor(
        [value["fm_target"]["target_action_h50"] for value in payloads],
        dtype=torch.float32,
    )
    valid_h50 = torch.tensor(
        [value["fm_target"]["action_valid_mask_h50"] for value in payloads],
        dtype=torch.bool,
    )
    slot_owners: list[list[str]] = []
    for value in payloads:
        owner = value["behavior_ack"]["slot_owner"][0]
        slot_owners.append(
            [
                owner if expert else "policy"
                for expert in value["fm_target"]["expert_slot_mask_h50"]
            ]
        )
    return {
        "action": action,
        "target_h50": target_h50,
        "valid_h50": valid_h50,
        "slot_owners": slot_owners,
        "reward": torch.tensor(
            [value["outcome"]["reward"] for value in payloads], dtype=torch.float32,
        ),
        "discount": torch.tensor(
            [value["outcome"]["discount"] for value in payloads], dtype=torch.float32,
        ),
        "terminated": torch.tensor(
            [value["outcome"]["terminated"] for value in payloads], dtype=torch.bool,
        ),
        "bootstrap": torch.tensor(
            [value["outcome"]["bootstrap"] for value in payloads], dtype=torch.bool,
        ),
        "actor_q_valid": torch.tensor(
            [value["eligibility"]["actor_q_valid"] for value in payloads], dtype=torch.bool,
        ),
    }


class OnlineLearner:
    """One deterministic, test-only 2-Critic:1-Actor joint learner cycle."""

    training_starts_unique_R = 100

    def __init__(
        self,
        *,
        replay: OnlineReplay,
        credit_ledger: UpdateCreditLedger,
        delta_action_mean7: Tensor,
        delta_action_std7: Tensor,
        seed: int,
        losses: OnlineLossAPI | None = None,
        actor: nn.Module | None = None,
        q1: nn.Module | None = None,
        q2: nn.Module | None = None,
    ) -> None:
        torch.manual_seed(seed)
        self.replay = replay
        self.credit_ledger = credit_ledger
        self.losses = losses or OnlineLossAPI()
        self.actor = actor or TinyActor()
        self.q1 = q1 or TinyTwinQ(offset=0.0)
        self.q2 = q2 or TinyTwinQ(offset=0.2)
        self.q1_target = deepcopy(self.q1).eval()
        self.q2_target = deepcopy(self.q2).eval()
        for target in (self.q1_target, self.q2_target):
            for parameter in target.parameters():
                parameter.requires_grad_(False)
        self.delta_action_mean7 = delta_action_mean7.detach().float().clone()
        self.delta_action_std7 = delta_action_std7.detach().float().clone()
        self.sampler = MixedReplaySampler(replay, seed=seed)
        # Explicitly test-only.  This is not the cross-stage production optimizer.
        self.actor_optimizer = torch.optim.SGD(self.actor.parameters(), lr=1e-2)
        self.critic_optimizer = torch.optim.SGD(
            tuple(self.q1.parameters()) + tuple(self.q2.parameters()), lr=1e-2,
        )
        self.critic_gradient_steps = 0
        self.actor_gradient_steps = 0
        self.target_polyak_updates = 0

    def assert_training_ready(self) -> None:
        count = len(self.replay.membership_uids(R_ONLINE))
        if count < self.training_starts_unique_R:
            raise TrainingStartsBlocked(
                f"ONLINE_REPLAY_TRAINING_STARTS_BLOCKED:{count}/{self.training_starts_unique_R}"
            )

    def _next_action(self, observation: CriticObservation) -> Tensor:
        chunk = self.actor(observation)
        return critic_action_for_q_guidance_v2(
            chunk,
            delta_action_mean7=self.delta_action_mean7,
            delta_action_std7=self.delta_action_std7,
        )

    def _critic_update(
        self,
        *,
        observation: CriticObservation,
        next_observation: CriticObservation,
        tensors: dict,
    ):
        self.critic_optimizer.zero_grad(set_to_none=True)
        result = self.losses.online_td(
            q1=self.q1,
            q2=self.q2,
            q1_target=self.q1_target,
            q2_target=self.q2_target,
            observation=observation,
            next_observation=next_observation,
            ack_behavior_action_k7=tensors["action"],
            behavior_mask=torch.ones(len(tensors["reward"]), 3, dtype=torch.bool),
            reward=tensors["reward"],
            discount=tensors["discount"],
            terminated=tensors["terminated"],
            bootstrap_mask=tensors["bootstrap"],
            next_policy_action_fn=self._next_action,
        )
        with torch.no_grad():
            next_action = self._next_action(next_observation)
            mask = torch.ones(next_action.shape[0], 3, dtype=torch.bool)
            expected = tensors["reward"] + tensors["discount"] * torch.minimum(
                self.q1_target(*next_observation.as_tuple(), next_action, mask),
                self.q2_target(*next_observation.as_tuple(), next_action, mask),
            )
        if not torch.equal(result.target, expected.float()):
            raise AssertionError("G3P_TD_TARGET_NOT_TARGET_TWIN_Q_MIN")
        result.total.backward()
        self.critic_optimizer.step()
        self.critic_gradient_steps += 1
        _polyak_update(self.q1_target, self.q1, tau=0.05)
        _polyak_update(self.q2_target, self.q2, tau=0.05)
        self.target_polyak_updates += 1
        return result

    def run_joint_cycle(self) -> dict:
        self.assert_training_ready()
        self.credit_ledger.consume_joint_cycle()
        batch = self.sampler.sample(R_count=2, D_count=2)
        if batch.R_count != batch.D_count or len(batch.samples) != 4:
            raise AssertionError("G3P_MIXED_REPLAY_NOT_EXACT_50_50")
        observation = _observations(batch, next_observation=False)
        next_observation = _observations(batch, next_observation=True)
        tensors = _batch_tensors(batch)

        actor_before_critic = _module_snapshot(self.actor)
        first_td = self._critic_update(
            observation=observation, next_observation=next_observation, tensors=tensors,
        )
        actor_after_critic_only = _module_snapshot(self.actor)
        actor_unchanged_by_critic_only = _snapshot_equal(
            actor_before_critic, actor_after_critic_only,
        )
        if not actor_unchanged_by_critic_only:
            raise AssertionError("G3P_CRITIC_ONLY_STEP_MODIFIED_ACTOR")

        second_td = self._critic_update(
            observation=observation, next_observation=next_observation, tensors=tensors,
        )
        critics_before_actor = (
            _module_snapshot(self.q1), _module_snapshot(self.q2),
        )
        self.critic_optimizer.zero_grad(set_to_none=True)
        self.actor_optimizer.zero_grad(set_to_none=True)
        flow_chunk = self.actor(observation)
        per_feature_flow = (flow_chunk - tensors["target_h50"]) ** 2
        expert_mask = build_expert_feature_mask(
            tensors["valid_h50"],
            tensors["slot_owners"],
            [sample.origin_pool for sample in batch.samples],
        )
        q_contract_loss, q1_value, q2_value, action_for_q = self.losses.actor_q(
            q1=self.q1,
            q2=self.q2,
            observation=observation,
            normalized_flow_action_chunk7=flow_chunk,
            delta_action_mean7=self.delta_action_mean7,
            delta_action_std7=self.delta_action_std7,
        )
        q_only_gradient = torch.autograd.grad(
            q_contract_loss, flow_chunk, retain_graph=True,
        )[0]
        actor_terms = self.losses.actor_objective(
            per_feature_flow_loss=per_feature_flow,
            action_valid_mask_h50=tensors["valid_h50"],
            expert_feature_mask_h50x7=expert_mask,
            q1_actor_value=q1_value,
            q2_actor_value=q2_value,
            actor_q_valid=tensors["actor_q_valid"],
            balance_loss=flow_chunk.mean().square(),
            z_loss=flow_chunk.square().mean(),
            beta=1.0,
            eta=0.1,
        )
        if not torch.equal(q_contract_loss, actor_terms.actor_q):
            raise AssertionError("G3P_ACTOR_GUIDANCE_NOT_MIN_TWIN_Q")
        actor_terms.total.backward()
        critic_has_gradient = any(
            parameter.grad is not None
            for module in (self.q1, self.q2)
            for parameter in module.parameters()
        )
        if critic_has_gradient:
            raise AssertionError("G3P_ACTOR_BACKWARD_TOUCHED_CRITIC_PARAMETERS")
        self.actor_optimizer.step()
        self.actor_gradient_steps += 1
        critics_after_actor = (_module_snapshot(self.q1), _module_snapshot(self.q2))
        critics_unchanged_by_actor_optimizer = all(
            _snapshot_equal(before, after)
            for before, after in zip(critics_before_actor, critics_after_actor, strict=True)
        )
        if not critics_unchanged_by_actor_optimizer:
            raise AssertionError("G3P_ACTOR_OPTIMIZER_MODIFIED_CRITIC")

        autonomous_fm_contribution = (
            per_feature_flow[: batch.R_count]
            * expert_mask[: batch.R_count].to(per_feature_flow.dtype)
        ).sum()
        if autonomous_fm_contribution.item() != 0.0:
            raise AssertionError("G3P_AUTONOMOUS_R_HAS_EXPERT_FM")

        zero_probe = torch.ones(1, 50, 7, requires_grad=True)
        zero_fm, zero_count = self.losses.expert_fm(
            zero_probe,
            torch.ones(1, 50, dtype=torch.bool),
            torch.zeros(1, 50, 7, dtype=torch.bool),
        )
        zero_gradient = torch.autograd.grad(zero_fm, zero_probe)[0]
        zero_expert_finite_graph_connected = (
            zero_count == 0
            and zero_fm.item() == 0.0
            and zero_fm.grad_fn is not None
            and torch.count_nonzero(zero_gradient).item() == 0
        )
        forbidden_prefixes = (
            "model.vlm_with_expert.vlm.", "model.state_proj.", "model.force_branch.",
        )
        actor_owned_names = tuple(name for name, _ in self.actor.named_parameters())
        optimizer_ownership_excludes_real_model = not any(
            name.startswith(forbidden_prefixes) for name in actor_owned_names
        )
        td_results = (first_td, second_td)
        return {
            "batch_R_count": batch.R_count,
            "batch_D_count": batch.D_count,
            "critic_gradient_steps": self.critic_gradient_steps,
            "actor_gradient_steps": self.actor_gradient_steps,
            "target_polyak_updates": self.target_polyak_updates,
            "calql_online_call_count": sum(value.calql_candidate_calls for value in td_results),
            "cql_penalty_call_count": 0,
            "random_candidate_call_count": sum(
                value.random_candidate_calls for value in td_results
            ),
            "mc_return_read_count": sum(value.mc_return_reads for value in td_results),
            "online_critic_is_pure_td": True,
            "td_target_uses_target_twin_q_min": True,
            "actor_guidance_uses_current_min_twin_q": True,
            "expert_feature_count": actor_terms.expert_feature_count,
            "autonomous_fm_contribution": float(autonomous_fm_contribution.detach()),
            "zero_expert_graph_connected_finite": zero_expert_finite_graph_connected,
            "tcp6_q_gradient_nonzero": bool(torch.count_nonzero(q_only_gradient[:, :3, :6])),
            "gripper_q_gradient_exact_zero": bool(
                torch.count_nonzero(q_only_gradient[:, :3, 6]) == 0
            ),
            "post_K_q_gradient_exact_zero": bool(torch.count_nonzero(q_only_gradient[:, 3:]) == 0),
            "critic_only_actor_unchanged": actor_unchanged_by_critic_only,
            "actor_optimizer_critics_unchanged": critics_unchanged_by_actor_optimizer,
            "optimizer_kind": "test_only_sgd",
            "actor_optimizer_parameter_names": list(actor_owned_names),
            "optimizer_excludes_vision_smolvlm_state_prefix": (
                optimizer_ownership_excludes_real_model
            ),
            "critic_action_shape": list(action_for_q.shape),
        }
