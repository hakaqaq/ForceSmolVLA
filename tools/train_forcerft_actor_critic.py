#!/usr/bin/env python3
"""Train the canonical demo-only Frozen-VLM ForceRFT Actor/Critic."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from copy import deepcopy
import json
import math
import os
from pathlib import Path
import random
import shutil
import sys
import tempfile
from typing import Any, Iterable, Mapping

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for path in (SRC, ROOT / "tools"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from forcesmolvla.rft.online import replay_training as warmup  # noqa: E402


TASK_ID = "task2"
TASK_OUTPUT_ROOT = ROOT / "outputs/task2"
SFT_CHECKPOINT = (
    TASK_OUTPUT_ROOT
    / "sft/checkpoints/forcesmolvla_sft_step_010000"
)
CRITIC_CHECKPOINT = (
    TASK_OUTPUT_ROOT
    / "offline/checkpoints/offline_twin_q_critic_warmup_step_000256"
)
JOINT_CHECKPOINT = (
    TASK_OUTPUT_ROOT
    / "offline/checkpoints/offline_actor_critic_cycle_000210"
)
ACTOR_CHECKPOINT_ID = "offline-actor-critic-cycle-000210"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def make_schedules(
    r_rng: random.Random,
    d_rng: random.Random,
    *,
    r_population_size: int,
    d_population: tuple[int, ...],
    fm_population: tuple[int, ...],
    cycles: int,
) -> tuple[list[list[int]], list[list[int]], list[list[int]], list[list[int]]]:
    require(bool(fm_population), "ONLINE_REPLAY_JOINT_NO_FM_POPULATION")
    fm_indices = frozenset(fm_population)
    critic_r: list[list[int]] = []
    critic_d: list[list[int]] = []
    actor_r: list[list[int]] = []
    actor_d: list[list[int]] = []
    for _cycle in range(cycles):
        for _substep in range(2):
            critic_r.append(r_rng.sample(range(r_population_size), 32))
            critic_d.append(d_rng.sample(d_population, 32))
        actor_r.append(r_rng.sample(range(r_population_size), 12))
        actor_batch = d_rng.sample(d_population, 12)
        if fm_indices.isdisjoint(actor_batch):
            actor_batch[-1] = d_rng.choice(fm_population)
        actor_d.append(actor_batch)
    return critic_r, critic_d, actor_r, actor_d


def assert_optimizer_ownership(
    actor_optimizer: torch.optim.Optimizer,
    critic_optimizer: torch.optim.Optimizer,
    *,
    frozen_parameters: Iterable[torch.nn.Parameter],
) -> None:
    actor_ids = [id(p) for group in actor_optimizer.param_groups for p in group["params"]]
    critic_ids = [id(p) for group in critic_optimizer.param_groups for p in group["params"]]
    frozen_ids = {id(p) for p in frozen_parameters}
    require(len(actor_ids) == len(set(actor_ids)), "ONLINE_REPLAY_JOINT_ACTOR_OPTIMIZER_DUPLICATE")
    require(len(critic_ids) == len(set(critic_ids)), "ONLINE_REPLAY_JOINT_CRITIC_OPTIMIZER_DUPLICATE")
    require(not (set(actor_ids) & set(critic_ids)), "ONLINE_REPLAY_JOINT_OPTIMIZER_OVERLAP")
    require(not (set(actor_ids) & frozen_ids), "ONLINE_REPLAY_JOINT_FROZEN_PARAMETER_IN_ACTOR_OPTIMIZER")


class JointDemoReplay(warmup.DemoReplay):
    """One expert pool: offline demonstrations plus online human corrections."""

    def __init__(
        self,
        normalizer,
        human_rows=(),
        source_episodes: Mapping[str, Path] | None = None,
    ) -> None:
        super().__init__(normalizer)
        self.offline_population = tuple(self.population)
        self.offline_count = len(self.rows)
        conversion = json.loads((warmup.DATASET / "conversion_manifest.json").read_text(encoding="utf-8"))
        self.frame_counts = {item["raw_episode_id"]: int(item["frames"]) for item in conversion["episodes"]}
        self.actions: dict[tuple[str, int], list[float]] = {}
        self.set_human_rows(human_rows, source_episodes or {})

    def set_human_rows(
        self, human_rows, source_episodes: Mapping[str, Path]
    ) -> None:
        self.human_replay = warmup.HumanCorrectionReplay(
            human_rows, source_episodes, self.normalizer
        )
        self.population = (
            *self.offline_population,
            *range(
                self.offline_count,
                self.offline_count + len(self.human_replay.rows),
            ),
        )

    @property
    def fm_population(self) -> tuple[int, ...]:
        return (
            *self.offline_population,
            *(
                self.offline_count + index
                for index, row in enumerate(self.human_replay.rows)
                if row["eligibility"]["fm_eligible"] is True
            ),
        )

    def prefetch_joint(
        self,
        critic_batches: Iterable[Iterable[int]],
        actor_batches: Iterable[Iterable[int]],
    ) -> None:
        import pyarrow.parquet as pq

        observation_requested: dict[str, set[int]] = {}
        action_requested: dict[str, set[int]] = {}
        all_batches = [*critic_batches, *actor_batches]
        for batch in all_batches:
            for index in batch:
                if index >= self.offline_count:
                    continue
                row = self.rows[index]
                for key in ("observation_row_reference", "next_observation_row_reference"):
                    ref = row[key]
                    observation_requested.setdefault(ref["data_relative_path"], set()).add(int(ref["row_index"]))
        for batch in actor_batches:
            for index in batch:
                if index >= self.offline_count:
                    continue
                row = self.rows[index]
                ref = row["observation_row_reference"]
                anchor = int(ref["row_index"])
                last = self.frame_counts[row["episode_id"]] - 1
                action_requested.setdefault(ref["data_relative_path"], set()).update(
                    min(anchor + offset, last) for offset in range(50)
                )
        files = sorted(set(observation_requested) | set(action_requested))
        for position, relative in enumerate(files, start=1):
            if relative in observation_requested:
                table = pq.read_table(warmup.DATASET / relative, columns=list(self.COLUMNS))
                for index in observation_requested[relative]:
                    self.raw[(relative, index)] = table.slice(index, 1).to_pylist()[0]
                del table
            if relative in action_requested:
                table = pq.read_table(warmup.DATASET / relative, columns=["action"])
                for index in action_requested[relative]:
                    self.actions[(relative, index)] = table.slice(index, 1).to_pylist()[0]["action"]
                del table
            if position % 10 == 0 or position == len(files):
                print(f"[joint] prefetched demonstration files {position}/{len(files)}", file=sys.stderr, flush=True)

    def materialize(self, index: int) -> dict[str, Any]:
        if index >= self.offline_count:
            return self.human_replay.materialize(index - self.offline_count)
        result = super().materialize(index)
        result["expert"] = True
        result["action_source"] = "offline_demonstration"
        return result

    def materialize_actor(self, index: int) -> dict[str, Any]:
        from forcesmolvla.action_delta import ActionDeltaProcessor

        if index >= self.offline_count:
            result = self.human_replay.materialize(index - self.offline_count)
            feature_mask = result["action_valid_mask"]
            result["current"]["delta_action7"] = result["action_target"]
            result["current"]["action_valid_mask"] = feature_mask.any(axis=1)
            result["expert_feature_mask"] = (
                feature_mask
                if result["fm_eligible"]
                else np.zeros_like(feature_mask)
            )
            return result
        result = self.materialize(index)
        row = self.rows[index]
        ref = row["observation_row_reference"]
        anchor = int(ref["row_index"])
        last = self.frame_counts[row["episode_id"]] - 1
        source_indices = np.minimum(anchor + np.arange(50), last)
        absolute = np.asarray(
            [self.actions[(ref["data_relative_path"], int(source))] for source in source_indices],
            dtype=np.float64,
        )
        current = self.raw[(ref["data_relative_path"], anchor)]
        delta = ActionDeltaProcessor.to_delta(
            absolute, np.asarray(current["observation.state"], dtype=np.float64)
        )
        result["current"]["delta_action7"] = self.normalizer.delta_action7.apply(delta).astype(np.float32)
        result["current"]["action_valid_mask"] = (anchor + np.arange(50) <= last)
        result["expert"] = True
        result["expert_feature_mask"] = np.repeat(
            result["current"]["action_valid_mask"][:, None], 7, axis=1
        )
        return result


def _online_actor_row(replay: warmup.FormalReplay, index: int) -> dict[str, Any]:
    row = replay.materialize(index)
    row["current"]["delta_action7"] = np.zeros((50, 7), dtype=np.float32)
    # The target is never used by FM because this row is not expert. Keeping the
    # topology valid lets the same Actor batch serve Q-guidance without imitation.
    row["current"]["action_valid_mask"] = np.ones(50, dtype=np.bool_)
    row["expert"] = False
    row["action_source"] = "policy"
    row["expert_feature_mask"] = np.zeros((50, 7), dtype=np.bool_)
    return row


def build_actor_training_batch(
    rows: list[dict[str, Any]], actor, feature: torch.Tensor, device: torch.device,
) -> dict[str, Any]:
    from forcesmolvla.rft.batch import build_actor_batch

    samples = [row["current"] for row in rows]
    action_sources = tuple(row["action_source"] for row in rows)
    return {
        "current_observation": warmup._critic_observation(samples, feature, device),
        "current_actor_batch": build_actor_batch(actor, samples, device, include_action=True),
        "expert_rows": torch.tensor([row["expert"] for row in rows], dtype=torch.bool, device=device),
        "expert_feature_mask": torch.from_numpy(
            np.stack([row["expert_feature_mask"] for row in rows])
        ).to(device),
        "action_sources": action_sources,
        "policy_row_mask": torch.tensor(
            [source == "policy" for source in action_sources],
            dtype=torch.bool,
            device=device,
        ),
        "behavior_action": torch.from_numpy(
            np.stack([row["behavior_action"] for row in rows])
        ).to(device),
        "behavior_mask": torch.from_numpy(
            np.stack([row["behavior_mask"] for row in rows])
        ).to(device),
        "terminated": torch.tensor(
            [row["terminated"] for row in rows], dtype=torch.bool, device=device
        ),
        "truncated": torch.tensor(
            [row["truncated"] for row in rows], dtype=torch.bool, device=device
        ),
        "td_eligible": torch.tensor(
            [row["td_eligible"] for row in rows], dtype=torch.bool, device=device
        ),
        "fm_eligible": torch.tensor(
            [row["fm_eligible"] for row in rows], dtype=torch.bool, device=device
        ),
        "identities": tuple(row["identity"] for row in rows),
    }


def _range_update(current: list[float], value: torch.Tensor) -> None:
    value = value.detach().float()
    require(bool(torch.isfinite(value).all()), "ONLINE_REPLAY_JOINT_NONFINITE_Q")
    current[0] = min(current[0], float(value.min().cpu()))
    current[1] = max(current[1], float(value.max().cpu()))


def _gradient_norm(parameters: Iterable[torch.nn.Parameter]) -> float:
    total = sum(
        float(parameter.grad.detach().float().square().sum().cpu())
        for parameter in parameters if parameter.grad is not None
    )
    return math.sqrt(total)


def critic_step(
    *,
    step: int,
    actor,
    q1,
    q2,
    q1_target,
    q2_target,
    optimizer,
    batch,
    flow,
    noise_generator,
    delta_mean,
    delta_std,
    microbatch_size: int | None = None,
    microbatch_slot=None,
) -> dict[str, Any]:
    from forcesmolvla.rft.critic_action_adapter_v2 import (
        bootstrap_command_effective_candidate_action,
    )
    from forcesmolvla.rft.online.training_losses import compute_online_twin_q_td_loss
    from forcesmolvla.rft.throughput_v2 import fast_polyak_update, index_actor_batch

    device = batch["reward"].device
    batch_size = int(batch["reward"].numel())
    microbatch_size = batch_size if microbatch_size is None else int(microbatch_size)
    require(
        1 <= microbatch_size <= batch_size and batch_size % microbatch_size == 0,
        "ONLINE_REPLAY_JOINT_CRITIC_MICROBATCH",
    )
    slot = microbatch_slot or (lambda _kind: nullcontext())
    optimizer.zero_grad(set_to_none=True)
    actor.eval()
    loss = 0.0
    q1_values: list[torch.Tensor] = []
    q2_values: list[torch.Tensor] = []
    for start in range(0, batch_size, microbatch_size):
        with slot("critic_microbatch"):
            positions = list(range(start, start + microbatch_size))
            index = torch.tensor(positions, dtype=torch.long, device=device)
            local_terminated = batch["terminated"][index]
            local_truncated = batch["truncated"][index]
            local_bootstrap = batch["bootstrap"][index]
            local_next_actor = index_actor_batch(
                batch["next_actor_batch"], positions
            )
            bootstrap_positions = torch.nonzero(
                local_bootstrap, as_tuple=False
            ).flatten().tolist()
            next_actor_batch = index_actor_batch(
                local_next_actor, bootstrap_positions
            )

            def next_action(_observation) -> torch.Tensor:
                count = len(bootstrap_positions)
                noise = torch.randn(
                    count, 50, 7,
                    dtype=torch.float32, device=device, generator=noise_generator,
                )
                with torch.no_grad(), torch.autocast(
                    device_type="cuda", dtype=torch.bfloat16
                ):
                    chunk = flow.sample(
                        actor, next_actor_batch, noise,
                        call_id=(
                            f"online-actor-critic-critic-{step:03d}"
                            f"-micro={start}:{start + microbatch_size}"
                        ),
                        purpose="td_next",
                    )
                return bootstrap_command_effective_candidate_action(
                    chunk,
                    delta_action_mean7=delta_mean,
                    delta_action_std7=delta_std,
                ).detach().float()

            result = compute_online_twin_q_td_loss(
                q1=q1,
                q2=q2,
                q1_target=q1_target,
                q2_target=q2_target,
                observation=batch["current_observation"].index(index),
                next_observation=batch["next_observation"].index(index),
                ack_behavior_action_k7=batch["behavior_action"][index],
                behavior_mask=batch["behavior_mask"][index],
                reward=batch["reward"][index],
                discount=batch["discount"][index],
                terminated=local_terminated,
                truncated=local_truncated,
                bootstrap_mask=local_bootstrap,
                next_policy_action_fn=next_action,
            )
            require(
                result.calql_candidate_calls
                == result.random_candidate_calls
                == result.mc_return_reads
                == 0,
                "ONLINE_REPLAY_JOINT_CRITIC_NOT_PURE_TD",
            )
            weight = microbatch_size / batch_size
            (result.total * weight).backward()
            loss += float(result.total.detach().cpu()) * weight
            q1_values.append(result.q1_value.detach())
            q2_values.append(result.q2_value.detach())

    parameters = [
        parameter for module in (q1, q2) for parameter in module.parameters()
        if parameter.requires_grad
    ]
    with slot("critic_optimizer"):
        require(
            all(parameter.grad is None for parameter in actor.parameters())
            and all(parameter.grad is None for target in (q1_target, q2_target) for parameter in target.parameters())
            and all(parameter.grad is None or bool(torch.isfinite(parameter.grad).all()) for parameter in parameters),
            "ONLINE_REPLAY_JOINT_CRITIC_GRADIENT_OWNERSHIP",
        )
        torch.nn.utils.clip_grad_norm_(parameters, 10.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        fast_polyak_update(q1, q1_target, tau=0.005, target_name="q1_target")
        fast_polyak_update(q2, q2_target, tau=0.005, target_name="q2_target")
        q1_value = torch.cat(q1_values)
        q2_value = torch.cat(q2_values)
    return {
        "loss": loss,
        "q1": q1_value,
        "q2": q2_value,
    }


def actor_step(
    *,
    cycle: int,
    actor,
    q1,
    q2,
    q1_target,
    q2_target,
    optimizer,
    scheduler,
    batch,
    flow,
    delta_mean,
    delta_std,
    config,
    microbatch_slot=None,
) -> dict[str, Any]:
    from forcesmolvla.force_token import RouterState
    from forcesmolvla.rft.frozen_vlm_trainability import FROZEN_PREFIXES, frozen_prefix_flow_matching_terms
    from forcesmolvla.rft.online.training_losses import (
        compute_online_actor_objective,
        compute_online_min_twin_q_actor_loss,
        compute_policy_behavior_anchor_loss,
    )
    from forcesmolvla.rft.critic_action_adapter_v2 import (
        command_effective_execution_index_map,
    )
    from forcesmolvla.rft.throughput_v2 import index_actor_batch
    from forcesmolvla.router_training import collect_pass_a_statistics, microbatch_two_pass_terms

    device = batch["expert_rows"].device
    batch_size = len(batch["identities"])
    microbatch = int(config["batching"]["flow_inference_subbatch"])
    require(batch_size == 24 and microbatch == 4, "ONLINE_REPLAY_JOINT_ACTOR_BATCH")
    parameters = [parameter for parameter in actor.parameters() if parameter.requires_grad]
    slot = microbatch_slot or (lambda _kind: nullcontext())
    with slot("actor_setup"):
        optimizer.zero_grad(set_to_none=True)
        for critic in (q1, q2, q1_target, q2_target):
            critic.zero_grad(set_to_none=True)
        valid = batch["current_actor_batch"]["action_valid_mask"].bool()
        expert_rows = batch["expert_rows"]
        expert_feature_mask = batch["expert_feature_mask"].bool()
        total_expert_features = int(
            (expert_feature_mask & valid.unsqueeze(-1)).sum()
        )
        require(total_expert_features > 0, "ONLINE_REPLAY_JOINT_NO_EXPERT_FM")
        anchor_eligible = (
            batch["policy_row_mask"]
            & ~batch["terminated"]
            & ~batch["truncated"]
            & batch["behavior_mask"].any(dim=1)
        )
        total_anchor_rows = int(anchor_eligible.sum())
        human_fm_present = any(
            source == "human" and bool(batch["fm_eligible"][index])
            for index, source in enumerate(batch["action_sources"])
        )
    fm_total = q_total = anchor_total = 0.0
    q1_values: list[torch.Tensor] = []
    q2_values: list[torch.Tensor] = []
    tcp_q_gradient_square = 0.0
    gripper_q_gradient_max = 0.0
    expert_gripper_fm_square = 0.0
    online_fm_gradient_max = 0.0
    human_fm_gradient_square = 0.0
    actor.train(True)

    for start in range(0, batch_size, microbatch):
        with slot("actor_microbatch"):
            positions = list(range(start, start + microbatch))
            actor_micro = index_actor_batch(batch["current_actor_batch"], positions)
            index = torch.tensor(positions, dtype=torch.long, device=device)
            observation = batch["current_observation"].index(index)
            local_valid = valid[start : start + microbatch]
            local_expert = expert_rows[start : start + microbatch]
            expert_mask = expert_feature_mask[start : start + microbatch]
            local_sources = batch["action_sources"][start : start + microbatch]
            local_human = torch.tensor(
                [source == "human" for source in local_sources],
                dtype=torch.bool,
                device=device,
            )
            local_policy = batch["policy_row_mask"][index]
            local_terminated = batch["terminated"][index]
            local_truncated = batch["truncated"][index]
            local_behavior_mask = batch["behavior_mask"][index]
            fm_noise = torch.randn(microbatch, 50, 7, dtype=torch.float32, device=device)
            fm_time = torch.rand(microbatch, dtype=torch.float32, device=device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                flow_losses, feature_mask, router_state, _contract = frozen_prefix_flow_matching_terms(
                    actor,
                    actor_micro,
                    noise=fm_noise,
                    time=fm_time,
                    call_id=f"online-actor-critic-cycle={cycle}-fm={start}",
                )
            detached_router = RouterState(
                logits_fp32=router_state.logits_fp32.detach(),
                probabilities_fp32=router_state.probabilities_fp32.detach(),
                route_ids=router_state.route_ids.detach(),
                valid_mask=router_state.valid_mask.detach(),
            )
            auxiliary = microbatch_two_pass_terms(
                flow_losses, router_state,
                collect_pass_a_statistics([detached_router], [feature_mask]),
            )
            flow7 = flow_losses[..., :7]

            q_noise = torch.randn(microbatch, 50, 7, dtype=torch.float32, device=device)
            actor.eval()
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                chunk = flow.sample(
                    actor,
                    actor_micro,
                    q_noise,
                    call_id=f"online-actor-critic-cycle={cycle}-actor-q={start}",
                    purpose="actor_guidance",
                )
                chunk.retain_grad()
                q_contract_loss, q1_value, q2_value, q_action = compute_online_min_twin_q_actor_loss(
                    q1=q1,
                    q2=q2,
                    observation=observation,
                    normalized_flow_action_chunk7=chunk,
                    execution_index_map=(
                        command_effective_execution_index_map()
                    ),
                    delta_action_mean7=delta_mean,
                    delta_action_std7=delta_std,
                )
            actor.train(True)
            policy_anchor = compute_policy_behavior_anchor_loss(
                q_action,
                batch["behavior_action"][index],
                local_behavior_mask,
                local_policy,
                local_terminated,
                local_truncated,
            )
            local_anchor_rows = int(
                (
                    local_policy
                    & ~local_terminated
                    & ~local_truncated
                    & local_behavior_mask.any(dim=1)
                ).sum()
            )
            anchor_weight = (
                local_anchor_rows / total_anchor_rows
                if total_anchor_rows
                else 0.0
            )
            expert_count = int(
                (expert_mask & local_valid.unsqueeze(-1)).sum()
            )
            fm_weight = expert_count / total_expert_features
            q_weight = microbatch / batch_size
            terms = compute_online_actor_objective(
                per_feature_flow_loss=flow7,
                action_valid_mask_h50=local_valid,
                expert_feature_mask_h50x7=expert_mask,
                q1_actor_value=q1_value,
                q2_actor_value=q2_value,
                actor_q_valid=torch.ones(microbatch, dtype=torch.bool, device=device),
                policy_behavior_anchor_loss=policy_anchor,
                balance_loss=auxiliary.balance,
                z_loss=auxiliary.z,
                beta=float(config["loss"]["beta_expert_flow_matching"]) * fm_weight,
                eta=float(config["loss"]["eta_actor_q"]) * q_weight,
                lambda_policy_behavior_anchor=(
                    float(config["loss"]["lambda_policy_behavior_anchor"])
                    * anchor_weight
                ),
                balance_weight=float(config["loss"]["balance_weight"]) / (batch_size / microbatch),
                z_weight=float(config["loss"]["z_weight"]) / (batch_size / microbatch),
            )
            require(torch.equal(q_contract_loss, terms.actor_q), "ONLINE_REPLAY_JOINT_ACTOR_Q_NOT_MIN_TWIN")
            q_gradient = torch.autograd.grad(q_contract_loss, chunk, retain_graph=True)[0]
            execution_index_map = command_effective_execution_index_map()
            tcp_q_gradient_square += float(
                q_gradient[:, execution_index_map, :6].float().square().sum().cpu()
            )
            gripper_q_gradient_max = max(
                gripper_q_gradient_max,
                float(q_gradient[:, execution_index_map, 6].float().abs().max().cpu()),
            )
            fm_gradient = torch.autograd.grad(terms.expert_flow_matching, flow7, retain_graph=True)[0]
            if bool((~local_expert).any()):
                online_fm_gradient_max = max(
                    online_fm_gradient_max,
                    float(fm_gradient[~local_expert].abs().max().cpu()),
                )
            if bool(local_human.any()):
                human_fm_gradient_square += float(
                    fm_gradient[local_human].float().square().sum().cpu()
                )
            if bool(local_expert.any()):
                expert_gripper_fm_square += float(
                    fm_gradient[local_expert, :, 6].float().square().sum().cpu()
                )
            terms.total.backward()
            fm_total += fm_weight * float(terms.expert_flow_matching.detach().cpu())
            q_total += q_weight * float(terms.actor_q.detach().cpu())
            anchor_total += anchor_weight * float(
                terms.policy_behavior_anchor.detach().cpu()
            )
            q1_values.append(q1_value.detach())
            q2_values.append(q2_value.detach())

    with slot("actor_optimizer"):
        require(online_fm_gradient_max == 0.0, "ONLINE_REPLAY_JOINT_ONLINE_SELF_IMITATION")
        require(
            not human_fm_present or human_fm_gradient_square > 0.0,
            "ONLINE_REPLAY_JOINT_HUMAN_FM_MISSING",
        )
        require(expert_gripper_fm_square > 0.0, "ONLINE_REPLAY_JOINT_EXPERT_GRIPPER_FM_MISSING")
        require(gripper_q_gradient_max == 0.0 and tcp_q_gradient_square > 0.0, "ONLINE_REPLAY_JOINT_Q_GRADIENT_SEMANTICS")
        require(
            all(parameter.grad is None for critic in (q1, q2, q1_target, q2_target) for parameter in critic.parameters()),
            "ONLINE_REPLAY_JOINT_ACTOR_BACKWARD_TOUCHED_CRITIC",
        )
        frozen_gradient_max = max(
            (
                float(parameter.grad.detach().abs().max().cpu())
                for name, parameter in actor.named_parameters()
                if name.startswith(FROZEN_PREFIXES) and parameter.grad is not None
            ),
            default=0.0,
        )
        actor_gradient = _gradient_norm(parameters)
        require(
            frozen_gradient_max == 0.0
            and math.isfinite(actor_gradient)
            and all(parameter.grad is None or bool(torch.isfinite(parameter.grad).all()) for parameter in parameters),
            "ONLINE_REPLAY_JOINT_ACTOR_GRADIENT_INVALID",
        )
        torch.nn.utils.clip_grad_norm_(parameters, 10.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        q1_value = torch.cat(q1_values)
        q2_value = torch.cat(q2_values)
    actor.eval()
    return {
        "fm_loss": fm_total,
        "actor_q_loss": q_total,
        "policy_behavior_anchor_loss": anchor_total,
        "q1": q1_value,
        "q2": q2_value,
        "actor_gradient_norm": actor_gradient,
        "tcp6_q_gradient_norm": math.sqrt(tcp_q_gradient_square),
        "gripper_q_gradient_max": gripper_q_gradient_max,
        "frozen_vlm_gradient_max": frozen_gradient_max,
    }


def save_joint_checkpoint(
    path: Path,
    *,
    actor,
    modules: Mapping[str, torch.nn.Module],
    critic_optimizer,
    actor_optimizer,
    actor_scheduler,
    runtime_state: Mapping[str, Any],
    parent_binding: Mapping[str, Any] | None,
    source_checkpoint: Path,
    total_joint_cycles: int,
    actor_checkpoint_id: str,
    actor_parent_path: Path | None = None,
    parent_binding_id: str | None = None,
    critic_scheduler=None,
    checkpoint_kind: str = "online_replay_actor_critic_training",
    actor_directory: str = "candidate_policy",
) -> None:
    from forcesmolvla.checkpoint import export_development_actor_checkpoint
    from forcesmolvla.rft.critic_action_adapter_v2 import CRITIC_ACTION_CONTRACT

    require(not path.exists(), "ONLINE_REPLAY_JOINT_CHECKPOINT_EXISTS")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=path.name + ".tmp-", dir=path.parent))
    try:
        (temporary / "models").mkdir()
        (temporary / "optimizers").mkdir()
        (temporary / "state").mkdir()
        (temporary / "artifacts").mkdir()
        candidate = temporary / actor_directory
        if actor_parent_path is None:
            require(parent_binding is not None, "FORCERFT_ACTOR_PARENT_REQUIRED")
            actor_parent_path = Path(
                parent_binding["actor_parent"]["architecture_binding"]["container_path"]
            )
        binding_id = parent_binding_id or (
            str(parent_binding["binding_id"]) if parent_binding is not None else ""
        )
        require(bool(binding_id), "FORCERFT_PARENT_BINDING_ID_REQUIRED")
        export_development_actor_checkpoint(
            policy=actor,
            destination=candidate,
            runtime_parent=actor_parent_path,
            source_joint_checkpoint=path,
            candidate_revision_id=actor_checkpoint_id,
            parent_binding_id=binding_id,
            published=False,
        )
        for name, module in modules.items():
            torch.save(module.state_dict(), temporary / "models" / f"{name}_state.pt")
        torch.save(critic_optimizer.state_dict(), temporary / "optimizers/critic_optimizer_state.pt")
        torch.save(actor_optimizer.state_dict(), temporary / "optimizers/actor_optimizer_state.pt")
        torch.save(actor_scheduler.state_dict(), temporary / "optimizers/actor_scheduler_state.pt")
        if critic_scheduler is not None:
            torch.save(
                critic_scheduler.state_dict(),
                temporary / "optimizers/critic_scheduler_state.pt",
            )
        torch.save(dict(runtime_state), temporary / "state/runtime_state.pt")
        artifacts = runtime_state.get("runtime_artifacts", {})
        normalizer_source = Path(str(artifacts.get("normalizer", "")))
        action_contract_source = Path(str(artifacts.get("action_contract", "")))
        if checkpoint_kind == "offline_actor_critic_exact_resume":
            require(normalizer_source.is_file(), "FORCERFT_NORMALIZER_MISSING")
            require(action_contract_source.is_file(), "FORCERFT_ACTION_CONTRACT_MISSING")
        if normalizer_source.is_file():
            shutil.copy2(normalizer_source, temporary / "artifacts/normalizer_manifest.json")
        if action_contract_source.is_file():
            shutil.copy2(action_contract_source, temporary / "artifacts/action_delta_spec.json")
        metadata = {
            "kind": checkpoint_kind,
            "complete": True,
            "source_checkpoint": str(source_checkpoint),
            "joint_cycles": total_joint_cycles,
            "critic_ready": True,
            "actor_q_guidance_enabled": True,
            "critic_action_contract_version": CRITIC_ACTION_CONTRACT.version,
            "parent_binding_id": binding_id,
            "actor_directory": actor_directory,
            "critic_optimizer_restored": True,
            "actor_optimizer_restored": True,
            "actor_checkpoint": {
                "checkpoint_id": actor_checkpoint_id,
                "path": actor_directory,
            },
        }
        (temporary / "metadata.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(temporary, path)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def load_joint_checkpoint_once(
    path: Path,
    *,
    actor,
    modules: Mapping[str, torch.nn.Module],
    critic_optimizer,
    actor_optimizer,
    actor_scheduler,
    device: torch.device,
    critic_scheduler=None,
) -> dict[str, Any]:
    from safetensors.torch import load_file
    from forcesmolvla.rft.online.sample_credit import UpdateCreditLedger

    metadata = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
    if metadata.get("kind") == "online_actor_critic_exact_resume":
        from forcesmolvla.rft.critic_action_adapter_v2 import (
            CRITIC_ACTION_CONTRACT,
        )

        require(
            metadata.get("critic_action_contract_version")
            == CRITIC_ACTION_CONTRACT.version,
            "FORCERFT_LEGACY_ONLINE_ACTION_SEMANTICS_INCOMPATIBLE",
        )
    require(
        metadata.get("complete") is True
        and metadata.get("critic_ready") is True
        and metadata.get("actor_q_guidance_enabled") is True,
        "ONLINE_REPLAY_JOINT_CHECKPOINT_INCOMPLETE",
    )
    actor_directory = str(metadata.get("actor_directory", "candidate_policy"))
    actor_state = load_file(str(path / actor_directory / "model.safetensors"), device="cpu")
    actor.load_state_dict(actor_state, strict=True)
    for name, module in modules.items():
        state = torch.load(path / "models" / f"{name}_state.pt", map_location=device, weights_only=True)
        module.load_state_dict(state, strict=True)
    critic_optimizer.load_state_dict(torch.load(
        path / "optimizers/critic_optimizer_state.pt", map_location=device, weights_only=True
    ))
    actor_optimizer.load_state_dict(torch.load(
        path / "optimizers/actor_optimizer_state.pt", map_location=device, weights_only=True
    ))
    actor_scheduler.load_state_dict(torch.load(
        path / "optimizers/actor_scheduler_state.pt", map_location="cpu", weights_only=True
    ))
    critic_scheduler_path = path / "optimizers/critic_scheduler_state.pt"
    if critic_scheduler is not None:
        require(critic_scheduler_path.is_file(), "FORCERFT_CRITIC_SCHEDULER_MISSING")
        critic_scheduler.load_state_dict(
            torch.load(critic_scheduler_path, map_location="cpu", weights_only=True)
        )
    runtime = torch.load(path / "state/runtime_state.pt", map_location="cpu", weights_only=False)
    credits = UpdateCreditLedger.from_state_dict(runtime["sample_credit"])
    require(credits.snapshot().available == runtime["sample_credit"]["minted"] - runtime["sample_credit"]["consumed"], "ONLINE_REPLAY_JOINT_CREDIT_RESTORE")
    for name in ("r_rng", "d_rng"):
        probe = random.Random()
        probe.setstate(runtime["sampler_state"][name])
    return runtime


def load_resume_modules(
    checkpoint: Path,
    actor_package: Path,
    device: torch.device,
):
    from forcesmolvla.modeling_forcesmolvla import ForceSmolVLAPolicy
    from forcesmolvla.rft.critic import build_twin_q

    metadata_path = checkpoint / "metadata.json"
    require(metadata_path.is_file(), "FORCERFT_EXACT_RESUME_METADATA_MISSING")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("kind") == "online_actor_critic_exact_resume":
        from forcesmolvla.rft.critic_action_adapter_v2 import (
            CRITIC_ACTION_CONTRACT,
        )

        require(
            metadata.get("critic_action_contract_version")
            == CRITIC_ACTION_CONTRACT.version,
            "FORCERFT_LEGACY_ONLINE_ACTION_SEMANTICS_INCOMPATIBLE",
        )
    actor_directory = str(metadata.get("actor_directory", ""))
    require(
        metadata.get("kind")
        in {"offline_actor_critic_exact_resume", "online_actor_critic_exact_resume"}
        and actor_directory == "actor"
        and actor_package.resolve() == (checkpoint / "actor").resolve(),
        "FORCERFT_EXACT_RESUME_ACTOR_PACKAGE_MISMATCH",
    )
    binding = {
        "normalizer_binding": {
            "absolute_path": str(checkpoint / "artifacts/normalizer_manifest.json")
        }
    }
    config = warmup.yaml.safe_load(
        warmup.TRAINING_CONFIG.read_text(encoding="utf-8")
    )
    actor = ForceSmolVLAPolicy.from_pretrained(
        actor_package,
        local_files_only=True,
        force_download=False,
        strict=True,
        artifact_use="development",
    ).to(device)
    data = config["data"]
    q1, q2, q1_target, q2_target, _conversion = build_twin_q(
        warmup._resolve(data["critic_backbone_npz"]),
        warmup._resolve(data["critic_backbone_manifest"]),
        seed=0,
    )
    q1.train(True)
    q2.train(True)
    q1_target.make_permanent_eval_target()
    q2_target.make_permanent_eval_target()
    q1, q2, q1_target, q2_target = (
        module.to(device) for module in (q1, q2, q1_target, q2_target)
    )
    return actor, q1, q2, q1_target, q2_target, binding, config


def load_offline_training_parents(
    *,
    actor_checkpoint: Path,
    critic_checkpoint: Path,
    device: torch.device,
):
    """Restore the SFT Actor and the completed offline Twin-Q warmup."""

    from forcesmolvla.modeling_forcesmolvla import ForceSmolVLAPolicy
    from forcesmolvla.rft.critic import build_twin_q
    from forcesmolvla.rft.frozen_vlm_trainability import (
        apply_frozen_vlm_trainability,
        build_frozen_vlm_actor_optimizer,
    )

    config = warmup.yaml.safe_load(warmup.TRAINING_CONFIG.read_text(encoding="utf-8"))
    actor = ForceSmolVLAPolicy.from_pretrained(
        actor_checkpoint,
        local_files_only=True,
        force_download=False,
        strict=True,
        artifact_use="development",
    ).to(device)
    apply_frozen_vlm_trainability(actor)
    data = config["data"]
    q1, q2, q1_target, q2_target, _conversion = build_twin_q(
        warmup._resolve(data["critic_backbone_npz"]),
        warmup._resolve(data["critic_backbone_manifest"]),
        seed=0,
    )
    modules = {
        "q1": q1,
        "q2": q2,
        "q1_target": q1_target,
        "q2_target": q2_target,
    }
    for name, module in modules.items():
        module.load_state_dict(
            torch.load(
                critic_checkpoint / "models" / f"{name}_state.pt",
                map_location="cpu",
                weights_only=True,
            ),
            strict=True,
        )
    q1.train(True)
    q2.train(True)
    q1_target.make_permanent_eval_target()
    q2_target.make_permanent_eval_target()
    q1, q2, q1_target, q2_target = (
        module.to(device) for module in (q1, q2, q1_target, q2_target)
    )
    modules = {
        "q1": q1,
        "q2": q2,
        "q1_target": q1_target,
        "q2_target": q2_target,
    }
    critic_optimizer = torch.optim.Adam(
        [
            parameter
            for module in (q1, q2)
            for parameter in module.parameters()
            if parameter.requires_grad
        ],
        lr=3e-4,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.0,
    )
    critic_optimizer.load_state_dict(
        torch.load(
            critic_checkpoint / "optimizers/critic_optimizer_state.pt",
            map_location=device,
            weights_only=True,
        )
    )
    critic_scheduler = torch.optim.lr_scheduler.LambdaLR(
        critic_optimizer, lambda _step: 1.0
    )
    critic_scheduler.load_state_dict(
        torch.load(
            critic_checkpoint / "schedulers/critic_scheduler_state.pt",
            map_location="cpu",
            weights_only=True,
        )
    )
    actor_optimizer, actor_scheduler, actor_ownership = (
        build_frozen_vlm_actor_optimizer(
            actor, lr=float(config["optimizer"]["actor"]["lr"])
        )
    )
    return (
        actor,
        q1,
        q2,
        q1_target,
        q2_target,
        modules,
        critic_optimizer,
        critic_scheduler,
        actor_optimizer,
        actor_scheduler,
        actor_ownership,
        config,
    )


def _restore_critic_parent_rng(
    critic_checkpoint: Path, device: torch.device
) -> tuple[torch.Generator, random.Random, random.Random]:
    rng = torch.load(
        critic_checkpoint / "state/rng_states.pt",
        map_location="cpu",
        weights_only=False,
    )
    random.setstate(rng["python_random_state"])
    np.random.set_state(rng["numpy_random_state"])
    torch.set_rng_state(rng["torch_cpu_rng_state"])
    if device.type == "cuda":
        torch.cuda.set_rng_state_all(rng["torch_cuda_rng_states"])
    noise = torch.Generator(device=device)
    noise.set_state(rng["named_generator_states"]["td_next_action_flow_noise"])
    d_rng = random.Random()
    d_rng.setstate(random.getstate())
    r_rng = random.Random(4405)
    return noise, r_rng, d_rng


def make_offline_demo_schedules(
    rng: random.Random,
    population: tuple[int, ...],
    *,
    cycles: int,
) -> tuple[list[list[int]], list[list[int]]]:
    critic = [rng.sample(population, 64) for _ in range(cycles * 2)]
    actor = [rng.sample(population, 24) for _ in range(cycles)]
    return critic, actor


def run_offline_joint_training(
    *,
    cycles: int,
    actor_checkpoint: Path,
    critic_checkpoint: Path,
    checkpoint: Path,
) -> dict[str, Any]:
    """Run the sole demo-only Frozen-VLM Actor/Critic training phase."""

    from forcesmolvla.rft.critic import frozen_task_feature
    from forcesmolvla.rft.frozen_vlm_trainability import FROZEN_PREFIXES
    from forcesmolvla.rft.online.sample_credit import UpdateCreditLedger
    from forcesmolvla.rft.throughput_v2 import FrozenPrefixFlowCounter
    from forcesmolvla.training_data import load_normalizer_manifest

    require(cycles == 210, "FORCERFT_OFFLINE_JOINT_CYCLES_MUST_BE_210")
    require(torch.cuda.is_available(), "FORCERFT_OFFLINE_JOINT_CUDA_UNAVAILABLE")
    require(not checkpoint.exists(), "FORCERFT_OFFLINE_JOINT_CHECKPOINT_EXISTS")
    device = torch.device("cuda:0")
    (
        actor,
        q1,
        q2,
        q1_target,
        q2_target,
        modules,
        critic_optimizer,
        critic_scheduler,
        actor_optimizer,
        actor_scheduler,
        actor_ownership,
        config,
    ) = load_offline_training_parents(
        actor_checkpoint=actor_checkpoint,
        critic_checkpoint=critic_checkpoint,
        device=device,
    )
    frozen = [
        parameter
        for name, parameter in actor.named_parameters()
        if name.startswith(FROZEN_PREFIXES)
    ]
    assert_optimizer_ownership(
        actor_optimizer, critic_optimizer, frozen_parameters=frozen
    )
    require(
        actor_ownership["frozen_parameter_in_optimizer"] == 0,
        "FORCERFT_OFFLINE_FROZEN_PARAMETER_IN_OPTIMIZER",
    )
    critic_noise, r_rng, d_rng = _restore_critic_parent_rng(
        critic_checkpoint, device
    )
    normalizer_path = warmup.DATASET / "normalizer_manifest.json"
    normalizer = load_normalizer_manifest(normalizer_path)
    replay = JointDemoReplay(normalizer)
    critic_schedule, actor_schedule = make_offline_demo_schedules(
        d_rng, replay.population, cycles=cycles
    )
    replay.prefetch_joint(critic_schedule, actor_schedule)
    feature = torch.from_numpy(frozen_task_feature()).to(
        device=device, dtype=torch.float32
    )
    delta_mean = torch.tensor(
        normalizer.delta_action7.mean, dtype=torch.float32, device=device
    )
    delta_std = torch.tensor(
        normalizer.delta_action7.std, dtype=torch.float32, device=device
    )
    flow = FrozenPrefixFlowCounter(
        inference_batch_size=int(config["batching"]["flow_inference_subbatch"])
    )
    td_losses: list[float] = []
    fm_losses: list[float] = []
    actor_q_losses: list[float] = []
    for cycle in range(cycles):
        for substep in range(2):
            rows = [
                replay.materialize(index)
                for index in critic_schedule[cycle * 2 + substep]
            ]
            batch = warmup.build_batch(rows, actor, feature, device)
            record = critic_step(
                step=cycle * 2 + substep,
                actor=actor,
                q1=q1,
                q2=q2,
                q1_target=q1_target,
                q2_target=q2_target,
                optimizer=critic_optimizer,
                batch=batch,
                flow=flow,
                noise_generator=critic_noise,
                delta_mean=delta_mean,
                delta_std=delta_std,
                microbatch_size=4,
            )
            critic_scheduler.step()
            td_losses.append(float(record["loss"]))
            del batch, rows
        actor_rows = [
            replay.materialize_actor(index) for index in actor_schedule[cycle]
        ]
        actor_batch = build_actor_training_batch(
            actor_rows, actor, feature, device
        )
        actor_record = actor_step(
            cycle=cycle,
            actor=actor,
            q1=q1,
            q2=q2,
            q1_target=q1_target,
            q2_target=q2_target,
            optimizer=actor_optimizer,
            scheduler=actor_scheduler,
            batch=actor_batch,
            flow=flow,
            delta_mean=delta_mean,
            delta_std=delta_std,
            config=config,
        )
        fm_losses.append(float(actor_record["fm_loss"]))
        actor_q_losses.append(float(actor_record["actor_q_loss"]))
        del actor_batch, actor_rows
        if (cycle + 1) % 10 == 0 or cycle + 1 == cycles:
            print(
                f"[offline-joint] completed cycle {cycle + 1}/{cycles}",
                file=sys.stderr,
                flush=True,
            )
    runtime_state = {
        "online_joint_cycles": 0,
        "source_checkpoint": str(critic_checkpoint),
        "actor_parent_checkpoint": str(actor_checkpoint),
        "flags": {
            "offline_demo_only": True,
            "vlm_frozen": True,
            "critic_ready": True,
            "actor_q_guidance_enabled": True,
        },
        "counters": {
            "joint_cycles": cycles,
            "critic_optimizer_steps": cycles * 2,
            "actor_optimizer_steps": cycles,
            "target_polyak_steps": cycles * 2,
            "critic_parent_optimizer_steps": 256,
        },
        "replay": {
            "offline_demo_root": str(warmup.REWARD_TRANSITION_ROOT),
            "offline_demo_population": len(replay.population),
            "offline_cursor": cycles,
            "current_episode_sampled": False,
        },
        "sample_credit": UpdateCreditLedger(
            credits_per_transition=1, credits_per_joint_cycle=1
        ).state_dict(),
        "sampler_state": {
            "cycle": cycles,
            "r_rng": r_rng.getstate(),
            "d_rng": d_rng.getstate(),
        },
        "rng_state": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch_cpu": torch.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state_all(),
            "critic_noise_generator": critic_noise.get_state().cpu(),
        },
        "optimizer_ownership": {
            "overlap": 0,
            "frozen_vlm_or_state_prefix_in_actor_optimizer": 0,
            "critic_optimizer_restored_from_warmup": True,
            "actor_optimizer_fresh": True,
        },
        "runtime_artifacts": {
            "normalizer": str(normalizer_path),
            "action_contract": str(
                actor_checkpoint
                / "manifests/action_delta_spec.json"
            ),
        },
        "actor_checkpoint": {"checkpoint_id": ACTOR_CHECKPOINT_ID},
        "step_metrics": {
            "critic_td_loss": td_losses,
            "actor_fm_loss": fm_losses,
            "actor_min_twin_q_loss": actor_q_losses,
        },
    }
    save_joint_checkpoint(
        checkpoint,
        actor=actor,
        modules=modules,
        critic_optimizer=critic_optimizer,
        actor_optimizer=actor_optimizer,
        actor_scheduler=actor_scheduler,
        critic_scheduler=critic_scheduler,
        runtime_state=runtime_state,
        parent_binding=None,
        actor_parent_path=actor_checkpoint,
        parent_binding_id="task2-offline-sft-and-twin-q",
        source_checkpoint=critic_checkpoint,
        total_joint_cycles=cycles,
        actor_checkpoint_id=ACTOR_CHECKPOINT_ID,
        checkpoint_kind="offline_actor_critic_exact_resume",
        actor_directory="actor",
    )
    restored = load_joint_checkpoint_once(
        checkpoint,
        actor=actor,
        modules=modules,
        critic_optimizer=critic_optimizer,
        actor_optimizer=actor_optimizer,
        actor_scheduler=actor_scheduler,
        critic_scheduler=critic_scheduler,
        device=device,
    )
    require(
        restored["counters"] == runtime_state["counters"],
        "FORCERFT_OFFLINE_EXACT_RESUME_INVALID",
    )
    return {
        "OFFLINE_JOINT_CYCLES": cycles,
        "OFFLINE_CRITIC_OPTIMIZER_STEPS": cycles * 2,
        "OFFLINE_ACTOR_OPTIMIZER_STEPS": cycles,
        "FULL_EXACT_RESUME_LOAD": "PASS",
        "OFFLINE_ACTOR_CRITIC_CHECKPOINT_PATH": str(checkpoint),
    }


def strict_load_offline_checkpoint(
    *, actor_checkpoint: Path, critic_checkpoint: Path, checkpoint: Path
) -> dict[str, Any]:
    """Strictly restore every offline training state without an optimizer step."""

    require(torch.cuda.is_available(), "FORCERFT_OFFLINE_JOINT_CUDA_UNAVAILABLE")
    device = torch.device("cuda:0")
    (
        actor, _q1, _q2, _q1_target, _q2_target, modules,
        critic_optimizer, critic_scheduler, actor_optimizer, actor_scheduler,
        _ownership, _config,
    ) = load_offline_training_parents(
        actor_checkpoint=actor_checkpoint,
        critic_checkpoint=critic_checkpoint,
        device=device,
    )
    runtime = load_joint_checkpoint_once(
        checkpoint,
        actor=actor, modules=modules,
        critic_optimizer=critic_optimizer,
        actor_optimizer=actor_optimizer,
        actor_scheduler=actor_scheduler,
        critic_scheduler=critic_scheduler,
        device=device,
    )
    counters = runtime["counters"]
    require(
        int(counters["joint_cycles"]) == 210
        and int(counters["critic_optimizer_steps"]) == 420
        and int(counters["actor_optimizer_steps"]) == 210
        and int(counters["target_polyak_steps"]) == 420
        and critic_optimizer.state and actor_optimizer.state
        and actor_scheduler.last_epoch == 210
        and (checkpoint / "artifacts/normalizer_manifest.json").is_file()
        and (checkpoint / "artifacts/action_delta_spec.json").is_file(),
        "FORCERFT_OFFLINE_EXACT_RESUME_INVALID",
    )
    return {
        "FULL_EXACT_RESUME_LOAD": "PASS",
        "OFFLINE_JOINT_CYCLES": 210,
        "OFFLINE_CRITIC_OPTIMIZER_STEPS": 420,
        "OFFLINE_ACTOR_OPTIMIZER_STEPS": 210,
        "OPTIMIZER_STEP_COUNT": 0,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-id", default=TASK_ID)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--offline-joint-cycles", type=int, default=210)
    parser.add_argument("--actor-checkpoint", type=Path)
    parser.add_argument("--critic-checkpoint", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--strict-load-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    from forcesmolvla.training_runtime import resolve_task_output_root

    args = parse_args()
    output_root = resolve_task_output_root(
        ROOT, task_id=args.task_id, output_root=args.output_root
    )
    actor_checkpoint = args.actor_checkpoint or (
        output_root / "sft/checkpoints/forcesmolvla_sft_step_010000"
    )
    critic_checkpoint = args.critic_checkpoint or (
        output_root
        / "offline/checkpoints/offline_twin_q_critic_warmup_step_000256"
    )
    checkpoint = args.checkpoint or (
        output_root / "offline/checkpoints/offline_actor_critic_cycle_000210"
    )
    print(
        json.dumps(
            (
                strict_load_offline_checkpoint(
                    actor_checkpoint=actor_checkpoint,
                    critic_checkpoint=critic_checkpoint,
                    checkpoint=checkpoint,
                )
                if args.strict_load_only
                else run_offline_joint_training(
                    cycles=args.offline_joint_cycles,
                    actor_checkpoint=actor_checkpoint,
                    critic_checkpoint=critic_checkpoint,
                    checkpoint=checkpoint,
                )
            ),
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
