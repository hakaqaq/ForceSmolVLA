#!/usr/bin/env python3
"""Continue online-replay warmup with Actor/Critic joint cycles."""

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

import train_forcerft_critic_warmup as warmup  # noqa: E402


RESUME_CHECKPOINT = (
    warmup.FORMAL_R_ROOT / "checkpoints/online_actor_critic_cycle_000010"
)
RESUME_ACTOR_PACKAGE = (
    ROOT
    / "artifacts/development/stage3/published"
    / "online_actor_critic_cycle_000010_actor_export.v1"
)
JOINT_CHECKPOINT = warmup.FORMAL_R_ROOT / "checkpoints/online_actor_critic_cycle_000020"
CANDIDATE_REVISION_ID = "stage3-online-r-joint-cycle-000020-candidate"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def make_schedules(
    r_rng: random.Random,
    d_rng: random.Random,
    *,
    r_population_size: int,
    d_population: tuple[int, ...],
    cycles: int,
) -> tuple[list[list[int]], list[list[int]], list[list[int]], list[list[int]]]:
    critic_r: list[list[int]] = []
    critic_d: list[list[int]] = []
    actor_r: list[list[int]] = []
    actor_d: list[list[int]] = []
    for _cycle in range(cycles):
        for _substep in range(2):
            critic_r.append(r_rng.sample(range(r_population_size), 32))
            critic_d.append(d_rng.sample(d_population, 32))
        actor_r.append(r_rng.sample(range(r_population_size), 12))
        actor_d.append(d_rng.sample(d_population, 12))
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
    """Adds only the stored H=50 action targets needed by expert FM."""

    def __init__(self, normalizer) -> None:
        super().__init__(normalizer)
        conversion = json.loads((warmup.DATASET / "conversion_manifest.json").read_text(encoding="utf-8"))
        self.frame_counts = {item["raw_episode_id"]: int(item["frames"]) for item in conversion["episodes"]}
        self.actions: dict[tuple[str, int], list[float]] = {}

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
                row = self.rows[index]
                for key in ("observation_row_reference", "next_observation_row_reference"):
                    ref = row[key]
                    observation_requested.setdefault(ref["data_relative_path"], set()).add(int(ref["row_index"]))
        for batch in actor_batches:
            for index in batch:
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

    def materialize_actor(self, index: int) -> dict[str, Any]:
        from forcesmolvla.action_delta import ActionDeltaProcessor

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
        return result


def _online_actor_row(replay: warmup.FormalReplay, index: int) -> dict[str, Any]:
    row = replay.materialize(index)
    row["current"]["delta_action7"] = np.zeros((50, 7), dtype=np.float32)
    # The target is never used by FM because this row is not expert. Keeping the
    # topology valid lets the same Actor batch serve Q-guidance without imitation.
    row["current"]["action_valid_mask"] = np.ones(50, dtype=np.bool_)
    row["expert"] = False
    return row


def build_actor_training_batch(
    rows: list[dict[str, Any]], actor, feature: torch.Tensor, device: torch.device,
) -> dict[str, Any]:
    from forcesmolvla.rft.batch import build_actor_batch

    samples = [row["current"] for row in rows]
    return {
        "current_observation": warmup._critic_observation(samples, feature, device),
        "current_actor_batch": build_actor_batch(actor, samples, device, include_action=True),
        "expert_rows": torch.tensor([row["expert"] for row in rows], dtype=torch.bool, device=device),
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
    from forcesmolvla.rft.critic_action_adapter_v2 import critic_action_for_q_guidance_v2
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
            local_next_actor = index_actor_batch(
                batch["next_actor_batch"], positions
            )
            nonterminal_positions = torch.nonzero(
                ~local_terminated, as_tuple=False
            ).flatten().tolist()
            next_actor_batch = index_actor_batch(
                local_next_actor, nonterminal_positions
            )

            def next_action(_observation) -> torch.Tensor:
                count = len(nonterminal_positions)
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
                return critic_action_for_q_guidance_v2(
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
                bootstrap_mask=batch["bootstrap"][index],
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
        total_expert_features = int((valid & expert_rows[:, None]).sum()) * 7
        require(total_expert_features > 0, "ONLINE_REPLAY_JOINT_NO_EXPERT_FM")
    fm_total = q_total = 0.0
    q1_values: list[torch.Tensor] = []
    q2_values: list[torch.Tensor] = []
    tcp_q_gradient_square = 0.0
    gripper_q_gradient_max = 0.0
    expert_gripper_fm_square = 0.0
    online_fm_gradient_max = 0.0
    actor.train(True)

    for start in range(0, batch_size, microbatch):
        with slot("actor_microbatch"):
            positions = list(range(start, start + microbatch))
            actor_micro = index_actor_batch(batch["current_actor_batch"], positions)
            index = torch.tensor(positions, dtype=torch.long, device=device)
            observation = batch["current_observation"].index(index)
            local_valid = valid[start : start + microbatch]
            local_expert = expert_rows[start : start + microbatch]
            expert_mask = local_expert[:, None, None].expand(-1, 50, 7)
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
                q_contract_loss, q1_value, q2_value, _q_action = compute_online_min_twin_q_actor_loss(
                    q1=q1,
                    q2=q2,
                    observation=observation,
                    normalized_flow_action_chunk7=chunk,
                    delta_action_mean7=delta_mean,
                    delta_action_std7=delta_std,
                )
            actor.train(True)
            expert_count = int((local_valid & local_expert[:, None]).sum()) * 7
            fm_weight = expert_count / total_expert_features
            q_weight = microbatch / batch_size
            terms = compute_online_actor_objective(
                per_feature_flow_loss=flow7,
                action_valid_mask_h50=local_valid,
                expert_feature_mask_h50x7=expert_mask,
                q1_actor_value=q1_value,
                q2_actor_value=q2_value,
                actor_q_valid=torch.ones(microbatch, dtype=torch.bool, device=device),
                balance_loss=auxiliary.balance,
                z_loss=auxiliary.z,
                beta=float(config["loss"]["beta_expert_flow_matching"]) * fm_weight,
                eta=float(config["loss"]["eta_actor_q"]) * q_weight,
                balance_weight=float(config["loss"]["balance_weight"]) / (batch_size / microbatch),
                z_weight=float(config["loss"]["z_weight"]) / (batch_size / microbatch),
            )
            require(torch.equal(q_contract_loss, terms.actor_q), "ONLINE_REPLAY_JOINT_ACTOR_Q_NOT_MIN_TWIN")
            q_gradient = torch.autograd.grad(q_contract_loss, chunk, retain_graph=True)[0]
            tcp_q_gradient_square += float(q_gradient[:, :3, :6].float().square().sum().cpu())
            gripper_q_gradient_max = max(
                gripper_q_gradient_max,
                float(q_gradient[:, :3, 6].float().abs().max().cpu()),
            )
            fm_gradient = torch.autograd.grad(terms.expert_flow_matching, flow7, retain_graph=True)[0]
            if bool((~local_expert).any()):
                online_fm_gradient_max = max(
                    online_fm_gradient_max,
                    float(fm_gradient[~local_expert].abs().max().cpu()),
                )
            if bool(local_expert.any()):
                expert_gripper_fm_square += float(
                    fm_gradient[local_expert, :, 6].float().square().sum().cpu()
                )
            terms.total.backward()
            fm_total += fm_weight * float(terms.expert_flow_matching.detach().cpu())
            q_total += q_weight * float(terms.actor_q.detach().cpu())
            q1_values.append(q1_value.detach())
            q2_values.append(q2_value.detach())

    with slot("actor_optimizer"):
        require(online_fm_gradient_max == 0.0, "ONLINE_REPLAY_JOINT_ONLINE_SELF_IMITATION")
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
    parent_binding: Mapping[str, Any],
    source_checkpoint: Path,
    total_joint_cycles: int,
    candidate_revision_id: str,
) -> None:
    from forcesmolvla.checkpoint import export_development_actor_checkpoint

    require(not path.exists(), "ONLINE_REPLAY_JOINT_CHECKPOINT_EXISTS")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=path.name + ".tmp-", dir=path.parent))
    try:
        (temporary / "models").mkdir()
        (temporary / "optimizers").mkdir()
        (temporary / "state").mkdir()
        candidate = temporary / "candidate_policy"
        export_development_actor_checkpoint(
            policy=actor,
            destination=candidate,
            runtime_parent=Path(
                parent_binding["actor_parent"]["architecture_binding"]["container_path"]
            ),
            source_joint_checkpoint=path,
            candidate_revision_id=candidate_revision_id,
            parent_binding_id=parent_binding["binding_id"],
            published=False,
        )
        for name, module in modules.items():
            torch.save(module.state_dict(), temporary / "models" / f"{name}_state.pt")
        torch.save(critic_optimizer.state_dict(), temporary / "optimizers/critic_optimizer_state.pt")
        torch.save(actor_optimizer.state_dict(), temporary / "optimizers/actor_optimizer_state.pt")
        torch.save(actor_scheduler.state_dict(), temporary / "optimizers/actor_scheduler_state.pt")
        torch.save(dict(runtime_state), temporary / "state/runtime_state.pt")
        metadata = {
            "kind": "online_replay_actor_critic_training",
            "complete": True,
            "source_checkpoint": str(source_checkpoint),
            "joint_cycles": total_joint_cycles,
            "critic_ready": True,
            "actor_q_guidance_enabled": True,
            "parent_binding_id": parent_binding["binding_id"],
            "critic_optimizer_restored": True,
            "actor_optimizer_restored": True,
            "candidate_policy_revision": {
                "revision_id": candidate_revision_id,
                "path": "candidate_policy",
                "state": "candidate",
                "activated": False,
                "published": False,
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
) -> dict[str, Any]:
    from safetensors.torch import load_file
    from forcesmolvla.rft.online.sample_credit import UpdateCreditLedger

    metadata = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
    require(
        metadata.get("complete") is True
        and metadata.get("critic_ready") is True
        and metadata.get("actor_q_guidance_enabled") is True,
        "ONLINE_REPLAY_JOINT_CHECKPOINT_INCOMPLETE",
    )
    actor_state = load_file(str(path / "candidate_policy/model.safetensors"), device="cpu")
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
    *,
    allow_checkpoint_candidate: bool = False,
):
    from forcesmolvla.modeling_forcesmolvla import ForceSmolVLAPolicy
    from forcesmolvla.rft.critic import build_twin_q

    binding = json.loads(warmup.PARENT_BINDING.read_text(encoding="utf-8"))
    config = warmup.yaml.safe_load(
        warmup.TRAINING_CONFIG.read_text(encoding="utf-8")
    )
    candidate = json.loads(
        (actor_package / "candidate.json").read_text(encoding="utf-8")
    )
    source_matches = (
        Path(candidate["source_joint_checkpoint"]).resolve() == checkpoint.resolve()
    )
    published = (
        candidate.get("state") == "published"
        and candidate.get("published") is True
    )
    internal_checkpoint_candidate = (
        allow_checkpoint_candidate
        and actor_package.resolve() == (checkpoint / "candidate_policy").resolve()
        and candidate.get("state") == "candidate"
        and candidate.get("published") is False
        and candidate.get("activated") is False
    )
    require(
        source_matches and (published or internal_checkpoint_candidate),
        "ONLINE_REPLAY_JOINT_RESUME_ACTOR_PACKAGE_MISMATCH",
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


def run(
    *,
    cycles: int,
    checkpoint: Path,
    resume_checkpoint: Path,
    resume_actor_package: Path,
    candidate_revision_id: str,
) -> dict[str, Any]:
    from forcesmolvla.rft.critic import frozen_task_feature
    from forcesmolvla.rft.frozen_vlm_trainability import FROZEN_PREFIXES, apply_frozen_vlm_trainability, build_frozen_vlm_actor_optimizer
    from forcesmolvla.rft.online.sample_credit import UpdateCreditLedger
    from forcesmolvla.rft.throughput_v2 import FrozenPrefixFlowCounter
    from forcesmolvla.training_data import load_normalizer_manifest

    require(cycles == 10, "ONLINE_REPLAY_JOINT_REQUIRES_10_CYCLES")
    require(torch.cuda.is_available(), "ONLINE_REPLAY_JOINT_CUDA_UNAVAILABLE")
    device = torch.device("cuda:0")
    all_r, r_macros, source_episodes = warmup.load_formal_online_r(
        warmup.FORMAL_R_ROOT
    )
    actor, q1, q2, q1_target, q2_target, binding, config = load_resume_modules(
        resume_checkpoint, resume_actor_package, device
    )
    trainability = apply_frozen_vlm_trainability(actor)
    critic_parameters = [
        parameter for module in (q1, q2) for parameter in module.parameters()
        if parameter.requires_grad
    ]
    critic_optimizer = torch.optim.Adam(
        critic_parameters, lr=3e-4, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0,
    )
    modules = {"q1": q1, "q2": q2, "q1_target": q1_target, "q2_target": q2_target}
    actor_optimizer, actor_scheduler, actor_ownership = build_frozen_vlm_actor_optimizer(
        actor, lr=float(config["optimizer"]["actor"]["lr"]),
    )
    resume_runtime = load_joint_checkpoint_once(
        resume_checkpoint,
        actor=actor,
        modules=modules,
        critic_optimizer=critic_optimizer,
        actor_optimizer=actor_optimizer,
        actor_scheduler=actor_scheduler,
        device=device,
    )
    previous = resume_runtime["counters"]
    require(
        previous["joint_cycles"] == 10
        and previous["critic_optimizer_steps"] == 20
        and previous["actor_optimizer_steps"] == 10
        and previous["target_polyak_steps"] == 20
        and critic_optimizer.state
        and actor_optimizer.state
        and actor_scheduler.last_epoch == 10,
        "ONLINE_REPLAY_JOINT_EXACT_RESUME_INVALID",
    )
    credits = UpdateCreditLedger.from_state_dict(resume_runtime["sample_credit"])
    new_r_transition_count = sum(
        credits.mint_for_unique_online_transition(
            row["identity"]["transition_uid"]
        )
        for row in all_r
    )
    require(
        credits.snapshot().credited_transition_count == len(all_r),
        "ONLINE_REPLAY_JOINT_REPLAY_CREDIT_MISMATCH",
    )

    random.setstate(resume_runtime["rng_state"]["python"])
    np.random.set_state(resume_runtime["rng_state"]["numpy"])
    torch.set_rng_state(resume_runtime["rng_state"]["torch_cpu"])
    torch.cuda.set_rng_state_all(resume_runtime["rng_state"]["torch_cuda"])
    critic_noise = torch.Generator(device=device)
    critic_noise.set_state(
        resume_runtime["rng_state"]["critic_noise_generator"]
    )
    r_rng = random.Random()
    d_rng = random.Random()
    r_rng.setstate(resume_runtime["sampler_state"]["r_rng"])
    d_rng.setstate(resume_runtime["sampler_state"]["d_rng"])

    frozen_parameters = [
        parameter for name, parameter in actor.named_parameters()
        if name.startswith(FROZEN_PREFIXES)
    ]
    assert_optimizer_ownership(
        actor_optimizer, critic_optimizer, frozen_parameters=frozen_parameters,
    )
    require(
        actor_ownership["frozen_parameter_in_optimizer"] == 0
        and trainability.trainable_actor_parameter_tensors == actor_ownership["parameter_tensor_count"],
        "ONLINE_REPLAY_JOINT_ACTOR_OPTIMIZER_OWNERSHIP",
    )

    normalizer = load_normalizer_manifest(Path(binding["normalizer_binding"]["absolute_path"]))
    r_replay = warmup.FormalReplay(r_macros, source_episodes, normalizer)
    d_replay = JointDemoReplay(normalizer)
    critic_r_schedule, critic_d_schedule, actor_r_schedule, actor_d_schedule = make_schedules(
        r_rng,
        d_rng,
        r_population_size=len(r_macros),
        d_population=d_replay.population,
        cycles=cycles,
    )
    d_replay.prefetch_joint(critic_d_schedule, actor_d_schedule)
    feature = torch.from_numpy(frozen_task_feature()).to(device=device, dtype=torch.float32)
    delta_mean = torch.tensor(normalizer.delta_action7.mean, dtype=torch.float32, device=device)
    delta_std = torch.tensor(normalizer.delta_action7.std, dtype=torch.float32, device=device)
    flow = FrozenPrefixFlowCounter(inference_batch_size=int(config["batching"]["flow_inference_subbatch"]))

    td_losses: list[float] = []
    fm_losses: list[float] = []
    actor_q_losses: list[float] = []
    actor_gradient_range = [math.inf, -math.inf]
    tcp_q_gradient_range = [math.inf, -math.inf]
    q1_range = [math.inf, -math.inf]
    q2_range = [math.inf, -math.inf]
    gripper_q_gradient_max = 0.0
    frozen_vlm_gradient_max = 0.0
    nonfinite_count = 0
    oom_count = 0
    critic_steps = actor_steps = target_steps = 0
    cycle_offset = int(previous["joint_cycles"])
    critic_step_offset = int(previous["critic_optimizer_steps"])

    for cycle in range(cycles):
        credits.consume_joint_cycle()
        for substep in range(2):
            schedule_index = cycle * 2 + substep
            rows = [r_replay.materialize(index) for index in critic_r_schedule[schedule_index]]
            rows.extend(d_replay.materialize(index) for index in critic_d_schedule[schedule_index])
            batch = warmup.build_batch(rows, actor, feature, device)
            try:
                record = critic_step(
                    step=critic_step_offset + critic_steps,
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
                )
            except torch.cuda.OutOfMemoryError:
                oom_count += 1
                raise
            except FloatingPointError:
                nonfinite_count += 1
                raise
            td_losses.append(record["loss"])
            _range_update(q1_range, record["q1"])
            _range_update(q2_range, record["q2"])
            critic_steps += 1
            target_steps += 1

        actor_rows = [_online_actor_row(r_replay, index) for index in actor_r_schedule[cycle]]
        actor_rows.extend(d_replay.materialize_actor(index) for index in actor_d_schedule[cycle])
        actor_batch = build_actor_training_batch(actor_rows, actor, feature, device)
        try:
            actor_record = actor_step(
                cycle=cycle_offset + cycle,
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
        except torch.cuda.OutOfMemoryError:
            oom_count += 1
            raise
        except FloatingPointError:
            nonfinite_count += 1
            raise
        fm_losses.append(actor_record["fm_loss"])
        actor_q_losses.append(actor_record["actor_q_loss"])
        _range_update(q1_range, actor_record["q1"])
        _range_update(q2_range, actor_record["q2"])
        actor_gradient_range[0] = min(actor_gradient_range[0], actor_record["actor_gradient_norm"])
        actor_gradient_range[1] = max(actor_gradient_range[1], actor_record["actor_gradient_norm"])
        tcp_q_gradient_range[0] = min(tcp_q_gradient_range[0], actor_record["tcp6_q_gradient_norm"])
        tcp_q_gradient_range[1] = max(tcp_q_gradient_range[1], actor_record["tcp6_q_gradient_norm"])
        gripper_q_gradient_max = max(gripper_q_gradient_max, actor_record["gripper_q_gradient_max"])
        frozen_vlm_gradient_max = max(frozen_vlm_gradient_max, actor_record["frozen_vlm_gradient_max"])
        actor_steps += 1
        print(f"[joint] completed cycle {cycle + 1}/{cycles}", file=sys.stderr, flush=True)

    require(
        critic_steps == 20
        and actor_steps == 10
        and target_steps == 20
        and nonfinite_count == oom_count == 0
        and gripper_q_gradient_max == 0.0
        and frozen_vlm_gradient_max == 0.0,
        "ONLINE_REPLAY_JOINT_COMPLETION_CONTRACT",
    )
    total_joint_cycles = cycle_offset + cycles
    total_critic_steps = critic_step_offset + critic_steps
    total_actor_steps = int(previous["actor_optimizer_steps"]) + actor_steps
    total_target_steps = int(previous["target_polyak_steps"]) + target_steps
    runtime_state = {
        "source_checkpoint": str(resume_checkpoint),
        "flags": {"critic_ready": True, "actor_q_guidance_enabled": True},
        "counters": {
            "joint_cycles": total_joint_cycles,
            "critic_optimizer_steps": total_critic_steps,
            "actor_optimizer_steps": total_actor_steps,
            "target_polyak_steps": total_target_steps,
        },
        "replay": {
            "formal_r_root": str(warmup.FORMAL_R_ROOT),
            "unique_r_transition_count": len(all_r),
            "new_r_transition_count": new_r_transition_count,
            "eligible_ack_macro_count": len(r_macros),
            "mix": {"R": 32, "D": 32},
        },
        "sample_credit": credits.state_dict(),
        "sampler_state": {
            "cycle": total_joint_cycles,
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
            "critic_optimizer_restored_from_joint_checkpoint": True,
            "actor_optimizer_restored_from_joint_checkpoint": True,
        },
        "candidate_policy_revision": {
            "revision_id": candidate_revision_id,
            "state": "candidate",
            "activated": False,
            "published": False,
        },
        "step_metrics": {"critic_td_loss": list(td_losses)},
    }
    save_joint_checkpoint(
        checkpoint,
        actor=actor,
        modules=modules,
        critic_optimizer=critic_optimizer,
        actor_optimizer=actor_optimizer,
        actor_scheduler=actor_scheduler,
        runtime_state=runtime_state,
        parent_binding=binding,
        source_checkpoint=resume_checkpoint,
        total_joint_cycles=total_joint_cycles,
        candidate_revision_id=candidate_revision_id,
    )
    restored = load_joint_checkpoint_once(
        checkpoint,
        actor=actor,
        modules=modules,
        critic_optimizer=critic_optimizer,
        actor_optimizer=actor_optimizer,
        actor_scheduler=actor_scheduler,
        device=device,
    )
    require(
        restored["counters"] == runtime_state["counters"],
        "ONLINE_REPLAY_JOINT_CHECKPOINT_LOAD",
    )
    return {
        "NEW_R_TRANSITION_COUNT": new_r_transition_count,
        "TOTAL_R_TRANSITION_COUNT": len(all_r),
        "JOINT_CYCLES_COMPLETED": cycles,
        "CRITIC_OPTIMIZER_TOTAL_STEPS": total_critic_steps,
        "ACTOR_OPTIMIZER_TOTAL_STEPS": total_actor_steps,
        "TARGET_POLYAK_TOTAL_STEPS": total_target_steps,
        "TD_LOSS_FIRST_LAST": [td_losses[0], td_losses[-1]],
        "FM_LOSS_FIRST_LAST": [fm_losses[0], fm_losses[-1]],
        "ACTOR_MIN_TWIN_Q_LOSS_FIRST_LAST": [actor_q_losses[0], actor_q_losses[-1]],
        "Q1_MIN_MAX": q1_range,
        "Q2_MIN_MAX": q2_range,
        "ACTOR_GRADIENT_NORM_MIN_MAX": actor_gradient_range,
        "TCP6_Q_GRADIENT_NORM_MIN_MAX": tcp_q_gradient_range,
        "GRIPPER_Q_GRADIENT_MAX": gripper_q_gradient_max,
        "FROZEN_VLM_GRADIENT_MAX": frozen_vlm_gradient_max,
        "NONFINITE_COUNT": nonfinite_count,
        "OOM_COUNT": oom_count,
        "SAMPLE_CREDITS_REMAINING": credits.snapshot().available,
        "ONLINE_REPLAY_JOINT_CHECKPOINT_PATH": str(checkpoint),
        "CRITIC_READY": True,
        "ACTOR_Q_GUIDANCE_ENABLED": True,
        "NEW_CANDIDATE_ID": candidate_revision_id,
        "CANDIDATE_POLICY_REVISION_PATH": str(checkpoint / "candidate_policy"),
        "CANDIDATE_POLICY_REVISION_STATE": "candidate",
        "REVISION_ACTIVATED": False,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cycles", type=int, default=10)
    parser.add_argument("--checkpoint", type=Path, default=JOINT_CHECKPOINT)
    parser.add_argument("--resume-checkpoint", type=Path, default=RESUME_CHECKPOINT)
    parser.add_argument(
        "--resume-actor-package", type=Path, default=RESUME_ACTOR_PACKAGE
    )
    parser.add_argument("--candidate-id", default=CANDIDATE_REVISION_ID)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    print(
        json.dumps(
            run(
                cycles=args.cycles,
                checkpoint=args.checkpoint,
                resume_checkpoint=args.resume_checkpoint,
                resume_actor_package=args.resume_actor_package,
                candidate_revision_id=args.candidate_id,
            ),
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
