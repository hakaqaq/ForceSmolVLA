from __future__ import annotations

import inspect
from pathlib import Path

import pytest

from forcesmolvla.rft.online import actor_learner_runtime as runtime
from forcesmolvla.rft.online.actor_learner_runtime import (
    AsyncRuntimeError,
    online_checkpoint_path,
    select_resume_or_seed_checkpoint,
)


def test_latest_final_online_checkpoint_wins_over_seed(
    tmp_path: Path, monkeypatch,
) -> None:
    seed = tmp_path / "seed"
    seed.mkdir()
    first = online_checkpoint_path(tmp_path / "online/checkpoints", 5)
    latest = online_checkpoint_path(tmp_path / "online/checkpoints", 10)
    first.mkdir(parents=True)
    latest.mkdir()
    monkeypatch.setattr(
        runtime,
        "exact_resume_checkpoint_is_recoverable",
        lambda path, *, expected_kind: path in {seed, first, latest},
    )
    selected = select_resume_or_seed_checkpoint(
        tmp_path, configured_seed_bundle=seed
    )
    assert selected.path == latest.resolve()
    assert selected.kind == "online_residual_actor_critic"


def test_explicit_final_seed_is_required_without_online(
    tmp_path: Path, monkeypatch,
) -> None:
    seed = tmp_path / "seed"
    seed.mkdir()
    monkeypatch.setattr(
        runtime,
        "exact_resume_checkpoint_is_recoverable",
        lambda path, *, expected_kind: path == seed.resolve()
        and expected_kind == "stage3_seed",
    )
    selected = select_resume_or_seed_checkpoint(
        tmp_path, configured_seed_bundle=seed
    )
    assert selected.path == seed.resolve() and selected.kind == "stage3_seed"
    with pytest.raises(AsyncRuntimeError, match="RESUME_OR_SAFE_SEED_REQUIRED"):
        select_resume_or_seed_checkpoint(
            tmp_path, configured_seed_bundle=None
        )


def test_legacy_offline_fallback_is_not_an_api_or_selection_path(
    tmp_path: Path,
) -> None:
    assert "allow_legacy_offline_fallback" not in inspect.signature(
        select_resume_or_seed_checkpoint
    ).parameters
    legacy = tmp_path / "offline/checkpoints/offline_actor_critic_cycle_000210"
    legacy.mkdir(parents=True)
    with pytest.raises(AsyncRuntimeError, match="RESUME_OR_SAFE_SEED_REQUIRED"):
        select_resume_or_seed_checkpoint(
            tmp_path, configured_seed_bundle=None
        )
