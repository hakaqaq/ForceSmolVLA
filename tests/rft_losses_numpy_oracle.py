"""Independent NumPy golden formulas for Stage-2 G4 tests and preflight."""

from __future__ import annotations

import math

import numpy as np


def td_target(reward, discount, terminated, next_q1_nonterminal, next_q2_nonterminal):
    reward = np.asarray(reward, dtype=np.float32)
    discount = np.asarray(discount, dtype=np.float32)
    terminated = np.asarray(terminated, dtype=np.bool_)
    result = reward.copy()
    result[~terminated] = reward[~terminated] + discount[~terminated] * np.minimum(
        np.asarray(next_q1_nonterminal, dtype=np.float32),
        np.asarray(next_q2_nonterminal, dtype=np.float32),
    )
    return result


def calql_penalty(
    q_dataset,
    q_candidates,
    mc_return,
    valid,
    *,
    temperature,
    clip_min,
    clip_max,
):
    q_dataset = np.asarray(q_dataset, dtype=np.float32)
    q_candidates = np.asarray(q_candidates, dtype=np.float32)
    mc_return = np.asarray(mc_return, dtype=np.float32)
    calibrated = np.maximum(q_candidates, mc_return[:, None])
    values = np.concatenate((q_dataset[:, None], calibrated), axis=1)
    scaled = values / np.float32(temperature)
    maximum = scaled.max(axis=1, keepdims=True)
    logsumexp = maximum[:, 0] + np.log(np.exp(scaled - maximum).sum(axis=1))
    lse = np.float32(temperature) * (
        logsumexp - np.float32(math.log(values.shape[1]))
    )
    delta = np.clip(lse - q_dataset, clip_min, clip_max)
    selected = delta[np.asarray(valid, dtype=np.bool_)]
    return np.float32(selected.mean()) if selected.size else np.float32(0.0)


def twin_q_loss(
    q1_dataset,
    q2_dataset,
    target,
    q1_candidates,
    q2_candidates,
    mc_return,
    valid,
    *,
    alpha,
    temperature,
    clip_min,
    clip_max,
):
    q1_dataset = np.asarray(q1_dataset, dtype=np.float32)
    q2_dataset = np.asarray(q2_dataset, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)
    td1 = np.mean(np.square(q1_dataset - target), dtype=np.float32)
    td2 = np.mean(np.square(q2_dataset - target), dtype=np.float32)
    c1 = calql_penalty(
        q1_dataset,
        q1_candidates,
        mc_return,
        valid,
        temperature=temperature,
        clip_min=clip_min,
        clip_max=clip_max,
    )
    c2 = calql_penalty(
        q2_dataset,
        q2_candidates,
        mc_return,
        valid,
        temperature=temperature,
        clip_min=clip_min,
        clip_max=clip_max,
    )
    q1 = td1 + np.float32(alpha) * c1
    q2 = td2 + np.float32(alpha) * c2
    return {
        "total": np.float32((q1 + q2) / np.float32(2.0)),
        "q1": q1,
        "q2": q2,
        "td1": td1,
        "td2": td2,
        "calql1": c1,
        "calql2": c2,
    }

