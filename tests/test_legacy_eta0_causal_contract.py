from __future__ import annotations

from pathlib import Path
import sys

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from train_forcerft_actor_critic import validate_exact_eta0_ablation  # noqa: E402


def _baseline() -> dict:
    return {
        "offline_training": {
            "critic_updates_per_cycle": 2,
            "actor_updates_per_cycle": 1,
            "target_polyak_updates_per_cycle": 2,
        },
        "optimizer": {"actor": {"lr": 1.0e-5}},
        "loss": {
            "beta_expert_flow_matching": 1.0,
            "eta_actor_q": 0.0,
            "lambda_policy_behavior_anchor": 0.1,
        },
    }


def test_eta0_config_changes_only_q_weight() -> None:
    experiment = yaml.safe_load(
        (ROOT / "configs/experiments/forcerft_legacy_eta0_causal.yaml").read_text()
    )
    assert experiment["loss"]["eta_actor_q"] == 0.0
    assert experiment["optimizer"]["actor"]["lr"] == 1.0e-5
    assert experiment["offline_training"]["joint_cycles"] == 210


def test_eta0_contract_rejects_lr_or_schedule_changes() -> None:
    validate_exact_eta0_ablation(
        _baseline(),
        cycles=210,
        eta_actor_q_override=0.0,
        actor_lr_override=None,
    )
    with pytest.raises(RuntimeError, match="CYCLES_MUST_BE_210"):
        validate_exact_eta0_ablation(
            _baseline(), cycles=10, eta_actor_q_override=0.0,
            actor_lr_override=None,
        )
    with pytest.raises(RuntimeError, match="LR_OVERRIDE_FORBIDDEN"):
        validate_exact_eta0_ablation(
            _baseline(), cycles=210, eta_actor_q_override=0.0,
            actor_lr_override=1.0e-6,
        )
