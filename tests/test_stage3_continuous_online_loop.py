from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import json
import sys

import pytest


ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import run_forcerft_online_loop as loop  # noqa: E402


ACTIVE_ID = "active-cycle22"
ACTIVE_MODEL = "model-cycle22"


def _write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _registry(path: Path) -> Path:
    _write(path, {
        "state": {
            "active_revision_id": ACTIVE_ID,
            "policy_epoch": 2,
            "records": [{
                "revision_id": ACTIVE_ID,
                "model_sha256": ACTIVE_MODEL,
                "state": "active",
            }],
        }
    })
    return path


def _checkpoint(path: Path, revision: str, cycle: int) -> Path:
    _write(path / "metadata.json", {
        "complete": True,
        "joint_cycles": cycle,
        "candidate_policy_revision": {"revision_id": revision},
    })
    return path


def test_reads_active_deployment_and_latest_checkpoint_without_cycle_constants(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(loop, "ROOT", tmp_path)
    active = loop.read_active_revision(_registry(tmp_path / "registry.json"))
    package = tmp_path / "published/active"
    binding = tmp_path / "live/binding.json"
    _write(package / "candidate.json", {
        "revision_id": ACTIVE_ID,
        "model_revision": ACTIVE_MODEL,
        "state": "published",
        "published": True,
    })
    _write(binding, {})
    _write(tmp_path / "configs/deployment.dynamic.development.json", {
        "artifact_status": "development_only",
        "checkpoint": str(package.relative_to(tmp_path)),
        "deployment_binding": str(binding.relative_to(tmp_path)),
        "deployment_binding_sha256": "trusted-binding",
    })
    formal = tmp_path / "formal"
    expected = _checkpoint(formal / "checkpoints/arbitrary-name", ACTIVE_ID, 22)

    deployment = loop.discover_active_deployment(active)
    checkpoint = loop.discover_checkpoint_for_revision(formal, active.revision_id)

    assert active.policy_epoch == 2
    assert deployment.profile.name == "deployment.dynamic.development.json"
    assert deployment.trusted_binding == "trusted-binding"
    assert checkpoint == expected.resolve()
    plan = loop._episode_plan(
        SimpleNamespace(
            registry=tmp_path / "registry.json",
            formal_r_root=formal,
            root_prefix=tmp_path / "datasets/continuous",
        ),
        1,
    )
    assert plan[4] == expected.resolve()
    assert plan[5].name.startswith("stage3_real_async_joint_cycle_000023_pending_")
    assert plan[6].startswith("stage3-online-r-real-async-joint-cycle-000023-pending-")


def test_bootstrap_admits_007_before_publish_and_home_activation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    episode = tmp_path / "capture/episodes/episode_000000"
    episode.mkdir(parents=True)
    checkpoint = _checkpoint(tmp_path / "cycle23", "pending-cycle23", 23)
    args = SimpleNamespace(
        bootstrap_episode=episode,
        bootstrap_checkpoint=checkpoint,
    )
    calls: list[str] = []
    candidate = loop.Candidate(
        "pending-cycle23", "model-cycle23", checkpoint,
        tmp_path / "package", tmp_path / "profile", tmp_path / "binding",
    )
    monkeypatch.setattr(loop, "_admit", lambda *_args: calls.append("admit"))
    monkeypatch.setattr(
        loop, "_publish",
        lambda *_args: calls.append("publish") or candidate,
    )
    monkeypatch.setattr(
        loop, "_activate",
        lambda *_args, **_kwargs: calls.append("activate"),
    )

    loop._bootstrap(args)

    assert calls == ["admit", "publish", "activate"]


def test_capture_seal_blocks_current_episode_sampling(tmp_path: Path) -> None:
    root = tmp_path / "capture"
    pending = _checkpoint(tmp_path / "pending", "pending-cycle23", 23)
    resume = tmp_path / "cycle22"
    active = loop.ActiveRevision(ACTIVE_ID, ACTIVE_MODEL, 2)
    seal_path = loop._episode_seal(root)
    seal = {
        "technical_seal": "complete",
        "active_actor_revision": ACTIVE_ID,
        "active_actor_model_revision": ACTIVE_MODEL,
        "learner_resume_checkpoint": str(resume),
        "current_episode_sampled_by_learner": False,
        "learner_critic_steps": 2,
        "learner_actor_steps": 1,
        "pending_checkpoint_path": str(pending),
        "pending_candidate_id": "pending-cycle23",
        "pending_candidate_published": False,
        "pending_candidate_activated": False,
    }
    _write(seal_path, seal)

    loop._validate_capture(
        root=root,
        active=active,
        resume=resume,
        pending=pending,
        pending_id="pending-cycle23",
    )
    seal["current_episode_sampled_by_learner"] = True
    _write(seal_path, seal)
    with pytest.raises(loop.ContinuousLoopError, match="CAPTURE_SEAL_INVALID"):
        loop._validate_capture(
            root=root,
            active=active,
            resume=resume,
            pending=pending,
            pending_id="pending-cycle23",
        )


def test_success_finishes_only_in_bridge_admit_publish_home_activate_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    candidate = loop.Candidate(
        "candidate", "model", tmp_path / "checkpoint", tmp_path / "package",
        tmp_path / "profile", tmp_path / "binding",
    )
    monkeypatch.setattr(loop, "_bridge", lambda *_a: calls.append("bridge"))
    monkeypatch.setattr(loop, "_admit", lambda *_a: calls.append("admit"))
    monkeypatch.setattr(
        loop, "_publish", lambda *_a: calls.append("publish") or candidate,
    )
    monkeypatch.setattr(
        loop, "_activate", lambda *_a, **_k: calls.append("activate"),
    )

    loop._finish_episode(
        SimpleNamespace(),
        episode=tmp_path / "episode",
        episode_seal=tmp_path / "seal",
        pending=tmp_path / "pending",
        outcome="success",
    )

    assert calls == ["bridge", "admit", "publish", "activate"]


def test_failure_stops_loop_without_skipping_to_next_episode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[int] = []
    monkeypatch.setattr(loop, "_bootstrap", lambda _args: None)

    def fail(_args, index: int) -> None:
        calls.append(index)
        raise loop.ContinuousLoopError("episode failed")

    monkeypatch.setattr(loop, "_run_episode", fail)
    with pytest.raises(loop.ContinuousLoopError, match="episode failed"):
        loop.run_loop(SimpleNamespace(max_episodes=2))
    assert calls == [1]
