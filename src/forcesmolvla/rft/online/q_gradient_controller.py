"""Adaptive parameter-gradient scaling for production Actor Q guidance."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Mapping

import torch
from torch import Tensor


@dataclass(frozen=True)
class QGradientDecision:
    eta: float
    preservation_grad_norm: float
    q_grad_norm_raw: float
    q_grad_norm_weighted: float
    applied_ratio: float
    cosine: float
    skipped_reason: str | None
    hard_cap_applied: bool
    audited: bool


def gradient_pair_statistics(
    preservation_grads: Iterable[Tensor | None],
    q_grads: Iterable[Tensor | None],
    *,
    epsilon: float = 1.0e-8,
) -> tuple[float, float, float]:
    preservation_square = 0.0
    q_square = 0.0
    dot = 0.0
    for preservation, q_grad in zip(preservation_grads, q_grads, strict=True):
        if preservation is not None:
            preservation = preservation.detach().float()
            preservation_square += float(preservation.square().sum())
        if q_grad is not None:
            q_grad = q_grad.detach().float()
            q_square += float(q_grad.square().sum())
        if preservation is not None and q_grad is not None:
            dot += float((preservation * q_grad).sum())
    preservation_norm = math.sqrt(preservation_square)
    q_norm = math.sqrt(q_square)
    cosine = dot / (preservation_norm * q_norm + float(epsilon))
    return preservation_norm, q_norm, cosine


class QGradientRatioController:
    def __init__(
        self,
        *,
        target_ratio: float = 0.03,
        hard_max_ratio: float = 0.10,
        ema_decay: float = 0.95,
        eta_min: float = 0.0,
        eta_max: float = 0.10,
        epsilon: float = 1.0e-8,
        calibration_interval: int = 10,
    ) -> None:
        if not (
            0.0 <= target_ratio <= hard_max_ratio
            and 0.0 <= eta_min <= eta_max
            and 0.0 <= ema_decay < 1.0
            and epsilon > 0.0
            and calibration_interval >= 1
        ):
            raise ValueError("FORCERFT_Q_GRADIENT_CONTROLLER_CONFIG_INVALID")
        self.target_ratio = float(target_ratio)
        self.hard_max_ratio = float(hard_max_ratio)
        self.ema_decay = float(ema_decay)
        self.eta_min = float(eta_min)
        self.eta_max = float(eta_max)
        self.epsilon = float(epsilon)
        self.calibration_interval = int(calibration_interval)
        self.controller_step = 0
        self.ema_q_grad_norm = 0.0
        self.ema_preservation_grad_norm = 0.0
        self.current_eta = 0.0

    def should_audit(self) -> bool:
        return self.controller_step % self.calibration_interval == 0

    def update(
        self,
        preservation_grads: Iterable[Tensor | None],
        q_grads: Iterable[Tensor | None],
        *,
        actor_q_valid_count: int,
    ) -> QGradientDecision:
        preservation = tuple(preservation_grads)
        q_gradient = tuple(q_grads)
        preservation_norm, q_norm, cosine = gradient_pair_statistics(
            preservation, q_gradient, epsilon=self.epsilon
        )
        self.controller_step += 1
        reason: str | None = None
        hard_cap_applied = False
        finite = all(math.isfinite(value) for value in (preservation_norm, q_norm, cosine))
        if actor_q_valid_count <= 0:
            reason = "no_actor_q_valid_rows"
        elif not finite:
            reason = "nonfinite_gradient"
        elif preservation_norm == 0.0:
            reason = "zero_preservation_gradient"
        elif q_norm == 0.0:
            reason = "zero_q_gradient"

        applied_eta = 0.0
        if reason is None:
            decay = self.ema_decay
            self.ema_preservation_grad_norm = (
                decay * self.ema_preservation_grad_norm
                + (1.0 - decay) * preservation_norm
            )
            self.ema_q_grad_norm = (
                decay * self.ema_q_grad_norm + (1.0 - decay) * q_norm
            )
            eta = self.target_ratio * self.ema_preservation_grad_norm / (
                self.ema_q_grad_norm + self.epsilon
            )
            eta = min(max(eta, self.eta_min), self.eta_max)
            applied_ratio = eta * q_norm / (preservation_norm + self.epsilon)
            if applied_ratio > self.hard_max_ratio:
                eta *= self.hard_max_ratio / applied_ratio
                applied_ratio = self.hard_max_ratio
                hard_cap_applied = True
            self.current_eta = eta
            applied_eta = eta
        else:
            applied_ratio = 0.0
            if reason != "no_actor_q_valid_rows":
                self.current_eta = 0.0

        return QGradientDecision(
            eta=applied_eta,
            preservation_grad_norm=preservation_norm,
            q_grad_norm_raw=q_norm,
            q_grad_norm_weighted=applied_eta * q_norm,
            applied_ratio=applied_ratio,
            cosine=cosine,
            skipped_reason=reason,
            hard_cap_applied=hard_cap_applied,
            audited=True,
        )

    def hold(self, *, actor_q_valid_count: int) -> QGradientDecision:
        """Advance one Actor update while reusing the last calibrated eta."""

        self.controller_step += 1
        eta = self.current_eta if actor_q_valid_count > 0 else 0.0
        ratio = eta * self.ema_q_grad_norm / (
            self.ema_preservation_grad_norm + self.epsilon
        )
        return QGradientDecision(
            eta=eta,
            preservation_grad_norm=self.ema_preservation_grad_norm,
            q_grad_norm_raw=self.ema_q_grad_norm,
            q_grad_norm_weighted=eta * self.ema_q_grad_norm,
            applied_ratio=min(ratio, self.hard_max_ratio),
            cosine=0.0,
            skipped_reason=(
                None if actor_q_valid_count > 0 else "no_actor_q_valid_rows"
            ),
            hard_cap_applied=False,
            audited=False,
        )

    def state_dict(self) -> dict[str, float | int]:
        return {
            "controller_step": self.controller_step,
            "ema_q_grad_norm": self.ema_q_grad_norm,
            "ema_preservation_grad_norm": self.ema_preservation_grad_norm,
            "current_eta": self.current_eta,
            "target_ratio": self.target_ratio,
            "hard_max_ratio": self.hard_max_ratio,
            "eta_min": self.eta_min,
            "eta_max": self.eta_max,
            "ema_decay": self.ema_decay,
            "epsilon": self.epsilon,
            "calibration_interval": self.calibration_interval,
        }

    def load_state_dict(self, state: Mapping[str, float | int]) -> None:
        for name in (
            "target_ratio",
            "hard_max_ratio",
            "eta_min",
            "eta_max",
            "ema_decay",
            "epsilon",
            "calibration_interval",
        ):
            if float(state[name]) != getattr(self, name):
                raise ValueError("FORCERFT_Q_GRADIENT_CONTROLLER_RESUME_CONFIG_MISMATCH")
        self.controller_step = int(state["controller_step"])
        self.ema_q_grad_norm = float(state["ema_q_grad_norm"])
        self.ema_preservation_grad_norm = float(
            state["ema_preservation_grad_norm"]
        )
        self.current_eta = float(state["current_eta"])
