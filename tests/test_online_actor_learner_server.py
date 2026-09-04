from __future__ import annotations

from pathlib import Path
import sys
import threading

import pytest
import torch


ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from serve_forcerft_actor_learner import (  # noqa: E402
    AsyncPolicyLearnerRuntime,
    ContinuousLearner,
    _select_deployed_actor_for_resume,
)
from forcesmolvla.rft.online.actor_learner_runtime import (  # noqa: E402
    OnlineTrainingPolicy,
)
from forcesmolvla.rft.online.policy_revision import (  # noqa: E402
    InMemoryRevisionStateMachine,
    RevisionRecord,
    RevisionState,
)


BASE_MODEL_ID = "a" * 64


class FakeEngine:
    def __init__(self) -> None:
        self.metadata = {
            "service_role": "model_inference_only",
            "model_sha256": BASE_MODEL_ID,
        }
        self._lock = threading.Lock()
        self.policy = torch.nn.Linear(1, 1)
        self.residual_actor = torch.nn.Linear(1, 1)
        torch.nn.init.zeros_(self.residual_actor.weight)
        torch.nn.init.zeros_(self.residual_actor.bias)
        self.reset_count = 0

    def reset_residual_episode_context(self) -> None:
        self.reset_count += 1

    def infer(self, request):
        return {"request_id": request["request_id"], "actions": [[0.0] * 7] * 50}


class FakeLearner:
    def __init__(self) -> None:
        self.training_policy = OnlineTrainingPolicy()
        self.save_calls = 0
        self.learner = {
            "runtime": {
                "phase": "joint",
                "critic_burnin_complete": True,
                "critic_burnin_updates": 256,
                "active_residual_revision": "task3-residual-step-000000",
            }
        }

    def set_current_session(self, _session_id: str) -> None:
        pass

    def clear_current_session(self) -> None:
        pass

    def mark_active_residual_revision(self, revision_id: str) -> None:
        self.learner["runtime"]["active_residual_revision"] = revision_id

    def save_checkpoint(self):
        self.save_calls += 1
        return None

    def __call__(self, _coordinator):
        return {"waiting_for_replay": True, "phase": "joint"}


def runtime(tmp_path: Path) -> AsyncPolicyLearnerRuntime:
    revision = "task3-residual-step-000000"
    machine = InMemoryRevisionStateMachine(
        RevisionRecord(revision, BASE_MODEL_ID, RevisionState.ACTIVE)
    )
    return AsyncPolicyLearnerRuntime(
        engine=FakeEngine(),
        machine=machine,
        session_id="session-1",
        episode_id="episode-1",
        active_revision_id=revision,
        active_model_revision=BASE_MODEL_ID,
        active_actor_checkpoint=tmp_path / "initial-residual",
        learner_resume_checkpoint=tmp_path / "seed",
        online_checkpoint_root=tmp_path / "online/checkpoints",
        learner_job=FakeLearner(),
        active_actor_online_cycle=0,
    )


def identity() -> dict[str, str]:
    return {
        "session_id": "session-1",
        "episode_id": "episode-1",
        "policy_revision": BASE_MODEL_ID,
    }


def test_resume_keeps_fixed_base_and_restores_active_residual(tmp_path: Path) -> None:
    resume = tmp_path / "online/checkpoints/online_actor_critic_cycle_000010"
    (resume / "state").mkdir(parents=True)
    (resume / "models").mkdir()
    base = tmp_path / "fixed-base"
    torch.save(
        {
            "base_actor_checkpoint": str(base),
            "active_residual_revision": "task3-residual-step-000010",
            "online_joint_cycles": 10,
        },
        resume / "state/runtime_state.pt",
    )
    torch.save({}, resume / "models/residual_actor.pt")
    candidate = tmp_path / "online/actor_candidates/online_actor_step_000010"
    candidate.mkdir(parents=True)
    torch.save({}, candidate / "residual_actor.pt")
    selected = _select_deployed_actor_for_resume(resume_checkpoint=resume)
    assert selected == (
        base.resolve(),
        (candidate / "residual_actor.pt").resolve(),
        "task3-residual-step-000010",
        10,
    )


def test_candidate_contains_only_residual_actor_state(tmp_path: Path) -> None:
    learner = ContinuousLearner.__new__(ContinuousLearner)
    learner.checkpoint_root = tmp_path / "online/checkpoints"
    learner.learner = {
        "residual_actor": torch.nn.Linear(2, 1),
        "runtime": {"active_residual_revision": "task3-residual-step-000000"},
    }
    candidate = learner.export_actor_candidate(10)
    files = {
        path.relative_to(candidate["checkpoint"]).as_posix()
        for path in candidate["checkpoint"].rglob("*")
        if path.is_file()
    }
    assert candidate["revision_id"] == "task3-residual-step-000010"
    assert files == {"residual_actor.pt"}


def test_step_10_candidate_activates_only_after_episode_boundary(
    tmp_path: Path,
) -> None:
    service = runtime(tmp_path)
    base_before = {
        name: value.detach().clone()
        for name, value in service.engine.policy.state_dict().items()
    }
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    replacement = torch.nn.Linear(1, 1)
    torch.nn.init.constant_(replacement.weight, 2.0)
    torch.nn.init.constant_(replacement.bias, 3.0)
    torch.save(replacement.state_dict(), candidate / "residual_actor.pt")

    service.start_episode(identity())
    service._stage_actor_candidate(
        {
            "revision_id": "task3-residual-step-000010",
            "checkpoint": candidate,
            "online_joint_cycle": 10,
        }
    )
    assert torch.count_nonzero(service.engine.residual_actor.weight) == 0
    assert service.active_revision_id.endswith("000000")
    service.end_episode(identity())
    assert torch.equal(service.engine.residual_actor.weight, replacement.weight)
    assert torch.equal(service.engine.residual_actor.bias, replacement.bias)
    assert service.active_revision_id.endswith("000010")
    assert service.engine.reset_count == 1
    assert all(
        torch.equal(base_before[name], value)
        for name, value in service.engine.policy.state_dict().items()
    )
    assert service.active_model_revision == BASE_MODEL_ID


def test_runtime_identity_and_graceful_checkpoint(tmp_path: Path) -> None:
    service = runtime(tmp_path)
    with pytest.raises(RuntimeError, match="CAPTURE_IDENTITY_MISMATCH"):
        service.start_episode({**identity(), "episode_id": "wrong"})
    service.start_episode(identity())
    with pytest.raises(RuntimeError, match="INFERENCE_SESSION_MISMATCH"):
        service.infer(
            {"request_id": "bad", "provenance": {"session_id": "wrong"}}
        )
    service.abort_episode(identity())
    first = service.quiesce_and_save({})
    second = service.quiesce_and_save({})
    assert first["quiesced"] and second["quiesced"]
    assert service.learner_job.save_calls == 1
