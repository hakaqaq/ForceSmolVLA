"""CPU mixed-pool sampling and expert ownership masks."""

from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Sequence

import torch
from torch import Tensor

from .replay import D_EXPERT, R_ONLINE, Stage3Replay


EXPERT_OWNERS = {"human_intervention", "offline_demonstration"}


@dataclass(frozen=True)
class ReplaySample:
    transition_uid: str
    origin_pool: str
    payload: dict


@dataclass(frozen=True)
class MixedReplayBatch:
    samples: tuple[ReplaySample, ...]
    R_count: int
    D_count: int


class MixedReplaySampler:
    def __init__(self, replay: Stage3Replay, *, seed: int) -> None:
        self.replay = replay
        self._rng = random.Random(seed)

    def sample(self, *, R_count: int, D_count: int) -> MixedReplayBatch:
        if R_count <= 0 or D_count <= 0:
            raise ValueError("STAGE3_MIXED_BATCH_COUNTS_MUST_BE_POSITIVE")
        selected: list[ReplaySample] = []
        for pool, count in ((R_ONLINE, R_count), (D_EXPERT, D_count)):
            population = self.replay.membership_uids(pool)
            if not population:
                raise RuntimeError(f"STAGE3_REPLAY_POOL_EMPTY:{pool}")
            for uid in self._rng.choices(population, k=count):
                selected.append(ReplaySample(uid, pool, self.replay.get_payload(uid)))
        return MixedReplayBatch(tuple(selected), R_count=R_count, D_count=D_count)

    def state_dict(self) -> dict:
        return {"python_random_state": self._rng.getstate()}

    def load_state_dict(self, state: dict) -> None:
        if set(state) != {"python_random_state"}:
            raise ValueError("STAGE3_MIXED_SAMPLER_STATE_INVALID")
        self._rng.setstate(state["python_random_state"])


def build_expert_feature_mask(
    action_valid_mask_h50: Tensor,
    slot_owners: Sequence[Sequence[str]],
    origin_pools: Sequence[str],
) -> Tensor:
    if action_valid_mask_h50.dtype != torch.bool or action_valid_mask_h50.ndim != 2:
        raise ValueError("STAGE3_ACTION_VALID_MASK_MUST_BE_BOOL_BH")
    batch, horizon = action_valid_mask_h50.shape
    if len(slot_owners) != batch or len(origin_pools) != batch:
        raise ValueError("STAGE3_EXPERT_MASK_BATCH_MISMATCH")
    expert = torch.zeros(batch, horizon, dtype=torch.bool, device=action_valid_mask_h50.device)
    for row, (owners, pool) in enumerate(zip(slot_owners, origin_pools, strict=True)):
        if len(owners) != horizon:
            raise ValueError("STAGE3_SLOT_OWNER_HORIZON_MISMATCH")
        if pool not in {R_ONLINE, D_EXPERT}:
            raise ValueError("STAGE3_SAMPLE_ORIGIN_POOL_INVALID")
        if pool == D_EXPERT:
            expert[row] = torch.tensor(
                [owner in EXPERT_OWNERS for owner in owners],
                dtype=torch.bool,
                device=expert.device,
            )
    expert &= action_valid_mask_h50
    return expert.unsqueeze(-1).expand(batch, horizon, 7).clone()
