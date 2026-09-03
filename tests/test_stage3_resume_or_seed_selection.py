from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from forcesmolvla.rft.critic_action_adapter_v2 import CRITIC_ACTION_CONTRACT
from forcesmolvla.rft.online.actor_learner_runtime import (
    AsyncRuntimeError,
    online_checkpoint_path,
    select_resume_or_seed_checkpoint,
)


FILES = (
    "actor/model.safetensors",
    "actor/config.json",
    "actor/artifact_manifest.json",
    "models/q1_state.pt",
    "models/q2_state.pt",
    "models/q1_target_state.pt",
    "models/q2_target_state.pt",
    "optimizers/actor_optimizer_state.pt",
    "optimizers/critic_optimizer_state.pt",
    "optimizers/actor_scheduler_state.pt",
    "optimizers/critic_scheduler_state.pt",
    "state/runtime_state.pt",
    "artifacts/normalizer_manifest.json",
    "artifacts/action_delta_spec.json",
)


def _checkpoint(path: Path, kind: str, *, cycle: int = 0) -> Path:
    path.mkdir(parents=True)
    metadata = {"complete": True, "kind": kind, "actor_directory": "actor"}
    if kind == "online_actor_critic_exact_resume":
        metadata["critic_action_contract_version"] = CRITIC_ACTION_CONTRACT.version
    if kind == "stage3_safe_seed_v1":
        metadata.update(
            actor_equal_to_sft=True,
            actor_updates_enabled=False,
            actor_q_guidance_enabled=False,
            critic_updates_enabled=True,
            legacy_actor210_parent=False,
            critic_action_contract_version=CRITIC_ACTION_CONTRACT.version,
        )
    (path / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    for relative in FILES:
        target = path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.touch()
    torch.save(
        {
            "online_joint_cycles": cycle if kind.startswith("online_") else 0,
            "counters": {
                "joint_cycles": cycle,
                "critic_optimizer_steps": cycle * 2,
                "actor_optimizer_steps": cycle,
                "target_polyak_steps": cycle * 2,
            },
        },
        path / "state/runtime_state.pt",
    )
    torch.save(
        {"last_epoch": cycle}, path / "optimizers/actor_scheduler_state.pt"
    )
    return path


def test_online_checkpoint_wins_over_explicit_safe_seed(tmp_path: Path) -> None:
    seed = _checkpoint(tmp_path / "seed", "stage3_safe_seed_v1")
    online = _checkpoint(
        online_checkpoint_path(tmp_path / "online/checkpoints", 5),
        "online_actor_critic_exact_resume",
        cycle=5,
    )
    selected = select_resume_or_seed_checkpoint(
        tmp_path, configured_seed_bundle=seed
    )
    assert selected.path == online.resolve()
    assert selected.kind == "online_actor_critic_exact_resume"


def test_explicit_safe_seed_is_required_when_online_is_absent(tmp_path: Path) -> None:
    seed = _checkpoint(tmp_path / "seed", "stage3_safe_seed_v1")
    selected = select_resume_or_seed_checkpoint(
        tmp_path, configured_seed_bundle=seed
    )
    assert selected.path == seed.resolve()
    assert selected.kind == "stage3_safe_seed_v1"

    with pytest.raises(AsyncRuntimeError, match="RESUME_OR_SAFE_SEED_REQUIRED"):
        select_resume_or_seed_checkpoint(tmp_path, configured_seed_bundle=None)


def test_legacy_actor210_requires_explicit_flag(tmp_path: Path) -> None:
    legacy = _checkpoint(
        tmp_path / "offline/checkpoints/offline_actor_critic_cycle_000210",
        "offline_actor_critic_exact_resume",
        cycle=210,
    )
    with pytest.raises(AsyncRuntimeError, match="RESUME_OR_SAFE_SEED_REQUIRED"):
        select_resume_or_seed_checkpoint(tmp_path, configured_seed_bundle=None)
    selected = select_resume_or_seed_checkpoint(
        tmp_path,
        configured_seed_bundle=None,
        allow_legacy_offline_fallback=True,
    )
    assert selected.path == legacy.resolve()
    assert selected.kind == "legacy_offline_actor_critic_ablation"


def test_wrong_seed_kind_fails_closed(tmp_path: Path) -> None:
    wrong = _checkpoint(tmp_path / "wrong", "offline_actor_critic_exact_resume")
    with pytest.raises(AsyncRuntimeError, match="SAFE_SEED_MISSING_OR_INCOMPLETE"):
        select_resume_or_seed_checkpoint(
            tmp_path, configured_seed_bundle=wrong
        )
