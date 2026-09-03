"""Pure, zero-update ForceRFT Actor/Critic loss primitives.

This module deliberately owns no optimizer, target-Actor, Polyak update, or
proposal distribution.  Callers must pass every Cal-QL candidate explicitly.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import math
from pathlib import Path
from typing import Callable, Iterator

import torch
from torch import Tensor, nn

from forcesmolvla.rft.critic import (
    ACTION_DIM,
    ACTION_SLOTS,
    DEFAULT_REWARD_TRANSITION_ROOT,
)
from forcesmolvla.rft.flow_sampling import (
    critic_action_for_q_guidance,
    sample_normalized_action_chunk_with_grad,
)


AUTHORIZED_G4_COLUMNS = (
    "transition_index",
    "episode_id",
    "split",
    "anchor_frame",
    "next_frame",
    "executed_steps",
    "executed_action_mask",
    "normalized_delta_action_exec_flat",
    "reward",
    "terminated",
    "bootstrap_mask",
    "discount",
    "mc_return",
    "reward_source",
    "observation_row_reference",
    "next_observation_row_reference",
)


@dataclass(frozen=True)
class CriticObservation:
    """The exact force-aware observation accepted by the G2 critic."""

    camera1: Tensor
    camera2: Tensor
    task_feature: Tensor
    normalized_state7: Tensor
    normalized_wrench6: Tensor

    @property
    def batch_size(self) -> int:
        return int(self.camera1.shape[0])

    def validate(self) -> "CriticObservation":
        batch = self.batch_size
        if batch < 1 or any(
            value.shape[0] != batch
            for value in (
                self.camera2,
                self.task_feature,
                self.normalized_state7,
                self.normalized_wrench6,
            )
        ):
            raise ValueError("G4_CRITIC_OBSERVATION_BATCH_MISMATCH")
        return self

    def index(self, index: Tensor) -> "CriticObservation":
        self.validate()
        if index.device != self.camera1.device:
            raise ValueError("G4_OBSERVATION_INDEX_DEVICE_MISMATCH")
        return CriticObservation(*(value[index] for value in self.as_tuple())).validate()

    def repeat_candidates(self, count: int) -> "CriticObservation":
        self.validate()
        if count < 1:
            raise ValueError("G4_CANDIDATE_REPEAT_COUNT_INVALID")
        return CriticObservation(
            *(value.repeat_interleave(count, dim=0) for value in self.as_tuple())
        ).validate()

    def as_tuple(self) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        return (
            self.camera1,
            self.camera2,
            self.task_feature,
            self.normalized_state7,
            self.normalized_wrench6,
        )


@dataclass(frozen=True)
class TwinQLossTerms:
    total: Tensor
    q1: Tensor
    q2: Tensor
    td1: Tensor
    td2: Tensor
    calql1: Tensor
    calql2: Tensor


@dataclass(frozen=True)
class ActorLossTerms:
    total: Tensor
    flow_matching: Tensor
    actor_q: Tensor
    balance: Tensor
    z: Tensor


def _finite_fp32(value: Tensor, name: str, shape: tuple[int, ...] | None = None) -> Tensor:
    if not isinstance(value, Tensor) or not value.is_floating_point():
        raise TypeError(f"G4_{name}_MUST_BE_FLOATING_TENSOR")
    if shape is not None and tuple(value.shape) != shape:
        raise ValueError(f"G4_{name}_SHAPE_INVALID")
    value = value.float()
    if not torch.isfinite(value).all():
        raise FloatingPointError(f"G4_{name}_NONFINITE")
    return value


def _bool_vector(value: Tensor, name: str, batch: int) -> Tensor:
    if value.dtype != torch.bool or tuple(value.shape) != (batch,):
        raise ValueError(f"G4_{name}_MUST_BE_BOOL_VECTOR")
    return value


def derive_loss_masks(executed_action_mask: Tensor, terminated: Tensor) -> dict[str, Tensor]:
    """Apply the frozen K=3 ownership rules without outcome-derived policy masks."""

    if executed_action_mask.dtype != torch.bool or executed_action_mask.ndim != 2:
        raise ValueError("G4_EXECUTED_ACTION_MASK_INVALID")
    batch, slots = executed_action_mask.shape
    if slots != ACTION_SLOTS:
        raise ValueError("G4_EXECUTED_ACTION_MASK_MUST_BE_K3")
    terminated = _bool_vector(terminated, "TERMINATED", batch)
    counts = executed_action_mask.sum(dim=-1)
    expected = torch.arange(ACTION_SLOTS, device=executed_action_mask.device)[None] < counts[:, None]
    if torch.any(counts == 0) or not torch.equal(executed_action_mask, expected):
        raise ValueError("G4_BEHAVIOR_MASK_MUST_BE_NONEMPTY_PREFIX")
    full = executed_action_mask.all(dim=-1)
    return {
        "full_macro_valid": full,
        "calql_valid": full & ~terminated,
        "actor_q_valid": full,
    }


def validate_discount_contract(
    discount: Tensor, terminated: Tensor, bootstrap_mask: Tensor
) -> None:
    """Treat redundant terminal/bootstrap fields as assertions, never multipliers."""

    batch = int(discount.numel())
    discount = _finite_fp32(discount, "DISCOUNT", (batch,))
    terminated = _bool_vector(terminated, "TERMINATED", batch)
    if tuple(bootstrap_mask.shape) != (batch,):
        raise ValueError("G4_BOOTSTRAP_MASK_SHAPE_INVALID")
    bootstrap = bootstrap_mask.to(dtype=torch.float32)
    if not torch.all((bootstrap == 0.0) | (bootstrap == 1.0)):
        raise ValueError("G4_BOOTSTRAP_MASK_NOT_BINARY")
    expected = torch.tensor(0.99, dtype=torch.float32, device=discount.device) * bootstrap
    if not torch.equal(discount, expected):
        raise ValueError("G4_DISCOUNT_IS_NOT_0P99_TIMES_BOOTSTRAP")
    if torch.any(terminated & ((discount != 0.0) | (bootstrap != 0.0))):
        raise ValueError("G4_TERMINAL_DISCOUNT_OR_BOOTSTRAP_NONZERO")
    if not torch.equal(terminated, bootstrap == 0.0):
        raise ValueError("G4_TERMINATED_BOOTSTRAP_INCONSISTENT")


def compute_td_target(
    reward: Tensor,
    discount: Tensor,
    terminated: Tensor,
    bootstrap_mask: Tensor,
    next_q1_nonterminal: Tensor,
    next_q2_nonterminal: Tensor,
) -> Tensor:
    """Compute ``r + discount * min(target Q1,Q2)`` in fp32.

    The next-Q tensors contain nonterminal rows only.  Consequently terminal
    rows cannot accidentally trigger Actor or target-Critic evaluation here.
    """

    batch = int(reward.numel())
    reward = _finite_fp32(reward, "REWARD", (batch,))
    discount = _finite_fp32(discount, "DISCOUNT", (batch,))
    terminated = _bool_vector(terminated, "TERMINATED", batch)
    validate_discount_contract(discount, terminated, bootstrap_mask)
    count = int((~terminated).sum())
    q1 = _finite_fp32(next_q1_nonterminal, "NEXT_Q1", (count,))
    q2 = _finite_fp32(next_q2_nonterminal, "NEXT_Q2", (count,))
    target = reward.clone()
    if count:
        target[~terminated] = reward[~terminated] + discount[~terminated] * torch.minimum(q1, q2)
    if not torch.isfinite(target).all():
        raise FloatingPointError("G4_TD_TARGET_NONFINITE")
    return target.detach()


def _slice_actor_batch(batch: dict, index: Tensor) -> dict:
    size = int(index.numel())
    selected = int(index.sum())
    result = {}
    cpu_index = index.detach().cpu().tolist()
    for name, value in batch.items():
        if isinstance(value, Tensor) and value.ndim and value.shape[0] == size:
            result[name] = value[index]
        elif isinstance(value, (list, tuple)) and len(value) == size:
            chosen = [item for item, keep in zip(value, cpu_index, strict=True) if keep]
            result[name] = type(value)(chosen)
        else:
            result[name] = value
    if selected and not result:
        raise ValueError("G4_EMPTY_SLICED_ACTOR_BATCH")
    return result


@contextmanager
def _temporary_eval(module: nn.Module) -> Iterator[None]:
    training = module.training
    module.eval()
    try:
        yield
    finally:
        module.train(training)


@contextmanager
def critics_as_action_differentiators(*critics: nn.Module) -> Iterator[None]:
    """Freeze critic parameters while preserving dQ/dAction, then restore state."""

    snapshots = [
        (critic, critic.training, [parameter.requires_grad for parameter in critic.parameters()])
        for critic in critics
    ]
    for critic, _training, _flags in snapshots:
        critic.eval()
        for parameter in critic.parameters():
            parameter.requires_grad_(False)
    try:
        yield
    finally:
        for critic, training, flags in snapshots:
            for parameter, flag in zip(critic.parameters(), flags, strict=True):
                parameter.requires_grad_(flag)
            critic.train(training)


def compute_td_target_from_current_actor(
    *,
    reward: Tensor,
    discount: Tensor,
    terminated: Tensor,
    bootstrap_mask: Tensor,
    next_observation: CriticObservation,
    next_actor_batch: dict,
    next_noise7: Tensor,
    actor: nn.Module,
    q1_target: nn.Module,
    q2_target: nn.Module,
    delta_action_mean7: Tensor,
    delta_action_std7: Tensor,
    call_id: str,
    sample_action_fn: Callable = sample_normalized_action_chunk_with_grad,
) -> Tensor:
    """Filter terminals first, then use the current Actor and target Twin-Q."""

    batch = int(reward.numel())
    terminated = _bool_vector(terminated, "TERMINATED", batch)
    next_observation.validate()
    if next_observation.batch_size != batch or tuple(next_noise7.shape) != (batch, 50, 7):
        raise ValueError("G4_NEXT_INPUT_BATCH_OR_NOISE_SHAPE_INVALID")
    if any(target.training for target in (q1_target, q2_target)) or any(
        parameter.requires_grad
        for target in (q1_target, q2_target)
        for parameter in target.parameters()
    ):
        raise RuntimeError("G4_TARGET_CRITICS_MUST_BE_PERMANENT_EVAL_NO_GRAD")
    nonterminal = ~terminated
    count = int(nonterminal.sum())
    if count == 0:
        empty = reward.new_empty((0,), dtype=torch.float32)
        return compute_td_target(
            reward, discount, terminated, bootstrap_mask, empty, empty
        )

    actor_batch = _slice_actor_batch(next_actor_batch, nonterminal)
    observation = next_observation.index(nonterminal)
    noise = next_noise7[nonterminal].float()
    with _temporary_eval(actor), torch.no_grad(), torch.autocast(
        device_type=noise.device.type,
        dtype=torch.bfloat16,
        enabled=noise.device.type == "cuda",
    ):
        chunk = sample_action_fn(
            actor,
            actor_batch,
            noise,
            call_id=call_id,
            purpose="td_next",
        )
        action = critic_action_for_q_guidance(
            chunk,
            delta_action_mean7=delta_action_mean7,
            delta_action_std7=delta_action_std7,
        ).detach().float()
    policy_mask = torch.ones(count, ACTION_SLOTS, dtype=torch.bool, device=action.device)
    with torch.no_grad():
        q1 = q1_target(*observation.as_tuple(), action, policy_mask)
        q2 = q2_target(*observation.as_tuple(), action, policy_mask)
    return compute_td_target(reward, discount, terminated, bootstrap_mask, q1, q2)


def compute_behavior_q(
    critic: nn.Module,
    observation: CriticObservation,
    behavior_action: Tensor,
    executed_action_mask: Tensor,
) -> Tensor:
    observation.validate()
    batch = observation.batch_size
    action = _finite_fp32(behavior_action, "BEHAVIOR_ACTION", (batch, ACTION_SLOTS, ACTION_DIM))
    if executed_action_mask.dtype != torch.bool or tuple(executed_action_mask.shape) != (batch, ACTION_SLOTS):
        raise ValueError("G4_BEHAVIOR_MASK_INVALID")
    return _finite_fp32(
        critic(*observation.as_tuple(), action, executed_action_mask),
        "DATASET_Q",
        (batch,),
    )


def validate_legal_gripper_endpoints(candidates: Tensor, endpoints: Tensor) -> None:
    if tuple(endpoints.shape) != (2,) or endpoints.dtype != torch.float32:
        raise ValueError("G4_NORMALIZED_GRIPPER_ENDPOINTS_INVALID")
    values = candidates[..., 6].float()
    if torch.any((values != endpoints[0]) & (values != endpoints[1])):
        raise ValueError("G4_CANDIDATE_GRIPPER_NOT_LEGAL_DISCRETE_ENDPOINT")


def evaluate_calql_candidates(
    critic: nn.Module,
    current_observation: CriticObservation,
    random_candidates: Tensor,
    policy_current_candidates: Tensor,
    policy_next_candidates: Tensor,
    normalized_gripper_endpoints: Tensor,
) -> Tensor:
    """Evaluate all three explicit candidate sets at the current observation."""

    current_observation.validate()
    batch = current_observation.batch_size
    shape = tuple(random_candidates.shape)
    if len(shape) != 4 or shape[0] != batch or shape[2:] != (ACTION_SLOTS, ACTION_DIM) or shape[1] < 1:
        raise ValueError("G4_RANDOM_CANDIDATE_SHAPE_INVALID")
    if tuple(policy_current_candidates.shape) != shape or tuple(policy_next_candidates.shape) != shape:
        raise ValueError("G4_POLICY_CANDIDATE_SHAPE_MISMATCH")
    candidates = []
    for name, value in (
        ("RANDOM_CANDIDATES", random_candidates),
        ("POLICY_CURRENT_CANDIDATES", policy_current_candidates),
        ("POLICY_NEXT_CANDIDATES", policy_next_candidates),
    ):
        value = _finite_fp32(value, name, shape).detach()
        validate_legal_gripper_endpoints(value, normalized_gripper_endpoints)
        candidates.append(value)
    joined = torch.cat(candidates, dim=1)
    candidate_count = joined.shape[1]
    observation = current_observation.repeat_candidates(candidate_count)
    action = joined.reshape(batch * candidate_count, ACTION_SLOTS, ACTION_DIM)
    mask = torch.ones(
        batch * candidate_count,
        ACTION_SLOTS,
        dtype=torch.bool,
        device=action.device,
    )
    q = critic(*observation.as_tuple(), action, mask).float().reshape(batch, candidate_count)
    return _finite_fp32(q, "CANDIDATE_Q", (batch, candidate_count))


def compute_calql_penalty(
    q_dataset: Tensor,
    q_candidates: Tensor,
    mc_return: Tensor,
    calql_valid: Tensor,
    *,
    temperature: float,
    clip_min: float,
    clip_max: float,
) -> Tensor:
    """Cal-QL-style finite-candidate conservative penalty.

    This is not importance-corrected exact CQL: there is no proposal-density
    correction, and the MC lower bound is applied to candidate Q values only.
    """

    batch = int(q_dataset.numel())
    q_dataset = _finite_fp32(q_dataset, "DATASET_Q", (batch,))
    if q_candidates.ndim != 2 or q_candidates.shape[0] != batch or q_candidates.shape[1] < 3:
        raise ValueError("G4_CANDIDATE_Q_SHAPE_INVALID")
    q_candidates = _finite_fp32(q_candidates, "CANDIDATE_Q", tuple(q_candidates.shape))
    if q_candidates.shape[1] % 3:
        raise ValueError("G4_CANDIDATE_Q_COUNT_MUST_EQUAL_3M")
    mc_return = _finite_fp32(mc_return, "MC_RETURN", (batch,))
    calql_valid = _bool_vector(calql_valid, "CALQL_VALID", batch)
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("G4_CALQL_TEMPERATURE_INVALID")
    if not all(math.isfinite(value) for value in (clip_min, clip_max)) or clip_min > clip_max:
        raise ValueError("G4_CALQL_CLIP_INVALID")
    calibrated_candidates = torch.maximum(q_candidates, mc_return[:, None])
    values = torch.cat((q_dataset[:, None], calibrated_candidates), dim=1)
    item_count = values.shape[1]
    lse = torch.tensor(temperature, dtype=torch.float32, device=values.device) * (
        torch.logsumexp(values / temperature, dim=1)
        - torch.tensor(math.log(item_count), dtype=torch.float32, device=values.device)
    )
    delta = torch.clamp(lse - q_dataset, min=clip_min, max=clip_max)
    result = delta[calql_valid].mean() if torch.any(calql_valid) else q_dataset.sum() * 0.0
    return _finite_fp32(result.reshape(()), "CALQL_PENALTY", ())


def compute_twin_q_critic_loss(
    *,
    q1_dataset: Tensor,
    q2_dataset: Tensor,
    td_target: Tensor,
    q1_candidates: Tensor,
    q2_candidates: Tensor,
    mc_return: Tensor,
    calql_valid: Tensor,
    alpha_calql: float,
    temperature: float,
    clip_min: float,
    clip_max: float,
) -> TwinQLossTerms:
    """Combine all-row TD regression with valid-row finite-candidate Cal-QL."""

    batch = int(q1_dataset.numel())
    q1_dataset = _finite_fp32(q1_dataset, "Q1_DATASET", (batch,))
    q2_dataset = _finite_fp32(q2_dataset, "Q2_DATASET", (batch,))
    td_target = _finite_fp32(td_target, "TD_TARGET", (batch,)).detach()
    if not math.isfinite(alpha_calql) or alpha_calql < 0:
        raise ValueError("G4_CALQL_ALPHA_INVALID")
    td1 = torch.mean(torch.square(q1_dataset - td_target))
    td2 = torch.mean(torch.square(q2_dataset - td_target))
    calql1 = compute_calql_penalty(
        q1_dataset,
        q1_candidates,
        mc_return,
        calql_valid,
        temperature=temperature,
        clip_min=clip_min,
        clip_max=clip_max,
    )
    calql2 = compute_calql_penalty(
        q2_dataset,
        q2_candidates,
        mc_return,
        calql_valid,
        temperature=temperature,
        clip_min=clip_min,
        clip_max=clip_max,
    )
    q1 = td1 + float(alpha_calql) * calql1
    q2 = td2 + float(alpha_calql) * calql2
    total = (q1 + q2) / 2.0
    for name, value in (
        ("CRITIC_TOTAL", total),
        ("CRITIC_Q1", q1),
        ("CRITIC_Q2", q2),
        ("TD1", td1),
        ("TD2", td2),
    ):
        _finite_fp32(value.reshape(()), name, ())
    return TwinQLossTerms(total, q1, q2, td1, td2, calql1, calql2)


def build_actor_q_action(
    actor_action_chunk7: Tensor,
    *,
    delta_action_mean7: Tensor,
    delta_action_std7: Tensor,
) -> Tensor:
    """Apply frozen C_Q: differentiable TCP6 plus detached binary gripper."""

    return critic_action_for_q_guidance(
        actor_action_chunk7,
        delta_action_mean7=delta_action_mean7,
        delta_action_std7=delta_action_std7,
    ).float()


def compute_actor_q_loss(
    *,
    q1: nn.Module,
    q2: nn.Module,
    current_observation: CriticObservation,
    actor_action_chunk7: Tensor,
    actor_q_valid: Tensor,
    delta_action_mean7: Tensor,
    delta_action_std7: Tensor,
) -> Tensor:
    """Return ``-mean((Q1+Q2)/2)`` while keeping only dQ/dAction."""

    current_observation.validate()
    batch = current_observation.batch_size
    actor_q_valid = _bool_vector(actor_q_valid, "ACTOR_Q_VALID", batch)
    action = build_actor_q_action(
        actor_action_chunk7,
        delta_action_mean7=delta_action_mean7,
        delta_action_std7=delta_action_std7,
    )
    if not torch.any(actor_q_valid):
        return _finite_fp32((action.sum() * 0.0).reshape(()), "ACTOR_Q_LOSS", ())
    observation = current_observation.index(actor_q_valid)
    action = action[actor_q_valid]
    mask = torch.ones(action.shape[0], ACTION_SLOTS, dtype=torch.bool, device=action.device)
    with critics_as_action_differentiators(q1, q2):
        q_mean = (
            q1(*observation.as_tuple(), action, mask)
            + q2(*observation.as_tuple(), action, mask)
        ).float() / 2.0
        loss = -q_mean.mean()
    return _finite_fp32(loss.reshape(()), "ACTOR_Q_LOSS", ())


def compute_offline_actor_objective(
    *,
    flow_matching_loss: Tensor,
    actor_q_loss: Tensor,
    balance_loss: Tensor,
    z_loss: Tensor,
    beta: float,
    eta: float,
) -> ActorLossTerms:
    """Combine each Actor term exactly once; actor_q_loss already has its minus."""

    if not all(math.isfinite(value) and value >= 0 for value in (beta, eta)):
        raise ValueError("G4_ACTOR_BETA_OR_ETA_INVALID")
    flow = _finite_fp32(flow_matching_loss.reshape(()), "FLOW_MATCHING_LOSS", ())
    actor_q = _finite_fp32(actor_q_loss.reshape(()), "ACTOR_Q_LOSS", ())
    balance = _finite_fp32(balance_loss.reshape(()), "BALANCE_LOSS", ())
    z = _finite_fp32(z_loss.reshape(()), "Z_LOSS", ())
    total = float(beta) * flow + float(eta) * actor_q + 0.01 * balance + 0.001 * z
    return ActorLossTerms(
        _finite_fp32(total.reshape(()), "ACTOR_TOTAL", ()), flow, actor_q, balance, z
    )


def validate_mc_return_recurrence(rows: list[dict], *, tolerance: float = 1e-12) -> dict:
    """Validate automatic detector returns episode-by-episode in stored order."""

    if not rows:
        raise ValueError("G4_MC_RETURN_ROWS_EMPTY")
    maximum_error = 0.0
    terminal_rows = 0
    for index, row in enumerate(rows):
        if row.get("reward_source") != "frozen_classifier_detector":
            raise ValueError("G4_MC_RETURN_NOT_FROM_AUTOMATIC_DETECTOR")
        same_next_episode = index + 1 < len(rows) and rows[index + 1]["episode_id"] == row["episode_id"]
        next_return = float(rows[index + 1]["mc_return"]) if same_next_episode else 0.0
        expected = float(row["reward"]) + float(row["discount"]) * next_return
        error = abs(float(row["mc_return"]) - expected)
        maximum_error = max(maximum_error, error)
        if bool(row["terminated"]):
            terminal_rows += 1
            if same_next_episode or float(row["discount"]) != 0.0:
                raise ValueError("G4_MC_RETURN_TERMINAL_ORDER_INVALID")
        if error > tolerance:
            raise ValueError(f"G4_MC_RETURN_RECURRENCE_FAILED:{error}")
    return {
        "row_count": len(rows),
        "terminal_row_count": terminal_rows,
        "maximum_absolute_error": maximum_error,
        "tolerance": tolerance,
    }


def load_authorized_reward_train_transitions(
    root: Path = DEFAULT_REWARD_TRANSITION_ROOT,
    *,
    task_id: str | None = None,
):
    """Open an authorized task reward-transition dataset."""

    root = Path(root).resolve()
    from forcesmolvla.rft.detector_reward_transitions import load_training_transitions

    table = load_training_transitions(root, task_id=task_id)
    missing = set(AUTHORIZED_G4_COLUMNS) - set(table.column_names)
    if missing:
        raise RuntimeError(f"G4_REWARD_TRANSITION_COLUMNS_MISSING:{sorted(missing)}")
    result = table.select(AUTHORIZED_G4_COLUMNS)
    if set(result.column("split").to_pylist()) != {"train"}:
        raise RuntimeError("G4_HELDOUT_TRANSITION_LEAK")
    return result
