from __future__ import annotations

import inspect
from pathlib import Path

import pytest

from forcesmolvla.rft.online import residual_actor_critic_runtime as runtime
from forcesmolvla.rft.online.residual_actor_critic_runtime import (
    AsyncRuntimeError,
    training_checkpoint_path,
    select_resume_or_bootstrap_checkpoint,
)


def test_latest_final_online_checkpoint_wins_over_seed(
    tmp_path: Path, monkeypatch,
) -> None:
    seed = tmp_path / "seed"
    seed.mkdir()
    root = tmp_path / "online_ack_residual/training_checkpoints"
    first = training_checkpoint_path(root, 5)
    latest = training_checkpoint_path(root, 10)
    first.mkdir(parents=True)
    latest.mkdir()
    monkeypatch.setattr(
        runtime,
        "exact_resume_checkpoint_is_recoverable",
        lambda path, *, expected_kind: path in {seed, first, latest},
    )
    selected = select_resume_or_bootstrap_checkpoint(
        tmp_path, configured_bootstrap_checkpoint=seed
    )
    assert selected.path == latest.resolve()
    assert selected.kind == "residual_actor_critic_training"


def test_explicit_final_seed_is_required_without_online(
    tmp_path: Path, monkeypatch,
) -> None:
    seed = tmp_path / "seed"
    seed.mkdir()
    monkeypatch.setattr(
        runtime,
        "exact_resume_checkpoint_is_recoverable",
        lambda path, *, expected_kind: path == seed.resolve()
        and expected_kind == "online_residual_bootstrap",
    )
    selected = select_resume_or_bootstrap_checkpoint(
        tmp_path, configured_bootstrap_checkpoint=seed
    )
    assert selected.path == seed.resolve() and selected.kind == "online_residual_bootstrap"
    with pytest.raises(
        AsyncRuntimeError, match="RESUME_OR_ONLINE_RESIDUAL_BOOTSTRAP_REQUIRED"
    ):
        select_resume_or_bootstrap_checkpoint(
            tmp_path, configured_bootstrap_checkpoint=None
        )


def test_legacy_offline_fallback_is_not_an_api_or_selection_path(
    tmp_path: Path,
) -> None:
    assert "allow_legacy_offline_fallback" not in inspect.signature(
        select_resume_or_bootstrap_checkpoint
    ).parameters
    legacy = tmp_path / "offline/checkpoints/offline_actor_critic_cycle_000210"
    legacy.mkdir(parents=True)
    with pytest.raises(
        AsyncRuntimeError, match="RESUME_OR_ONLINE_RESIDUAL_BOOTSTRAP_REQUIRED"
    ):
        select_resume_or_bootstrap_checkpoint(
            tmp_path, configured_bootstrap_checkpoint=None
        )
