from __future__ import annotations

from pathlib import Path
import sys
import threading

import pytest
import torch


ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from serve_forcerft_residual_actor_critic import (  # noqa: E402
    AsyncResidualActorCriticRuntime,
    ResidualActorCriticLearner,
    _select_deployed_actor_for_resume,
)
from forcesmolvla.rft.online.residual_actor_critic_runtime import (  # noqa: E402
    ResidualActorCriticSchedule,
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
        self.training_policy = ResidualActorCriticSchedule()
        self.save_calls = 0
        self.learner = {
            "runtime": {
                "learner_state": "residual_actor_critic_training",
                "ack_critic_warmup_complete": True,
                "ack_critic_warmup_steps": 256,
                "active_residual_policy_revision": "task3-residual-policy-step-000000",
                "online_adaptation_id": "task3-ack-residual-test",
                "counters": {
                    "residual_actor_optimizer_steps": 0,
                    "residual_actor_update_attempts": 0,
                    "residual_actor_updates_skipped_no_gradient": 0,
                },
            }
        }

    def set_current_session(self, _session_id: str) -> None:
        pass

    def clear_current_session(self) -> None:
        pass

    def mark_active_residual_policy_revision(self, revision_id: str) -> None:
        self.learner["runtime"]["active_residual_policy_revision"] = revision_id

    def save_checkpoint(self):
        self.save_calls += 1
        return None

    def __call__(self, _coordinator):
        return {"waiting_for_replay": True, "learner_state": "residual_actor_critic_training"}


class DrainLearner(FakeLearner):
    def __init__(
        self,
        *,
        cycle_budget: int = 7,
        actor_updates_applied: bool = True,
    ) -> None:
        super().__init__()
        self.cycle_budget = cycle_budget
        self.actor_updates_applied = actor_updates_applied
        self.completed_cycles = 0
        self.expected_admission_id: str | None = None
        self.latest_replay_refresh_ms = 4.0
        self.latest_critic_update_ms = 2.0
        self.latest_actor_update_ms = 1.0
        self.latest_cycle_ms = 5.0

    def expect_admission(self, admission_id: str) -> None:
        self.expected_admission_id = admission_id

    def admission_budget_status(self, admission_id: str):
        if admission_id != self.expected_admission_id:
            return None
        return {
            "episode_key": admission_id,
            "admitted_rows_for_latest_episode": 400,
            "computed_cycle_budget": self.cycle_budget,
            "cycle_count_at_admission_start": 0,
            "target_cycle_count_after_admission": self.cycle_budget,
            "completed_cycle_count_for_latest_admission": self.completed_cycles,
            "remaining_cycle_budget": self.cycle_budget - self.completed_cycles,
        }

    def __call__(self, _coordinator):
        if self.expected_admission_id is None or self.completed_cycles >= self.cycle_budget:
            counters = self.learner["runtime"]["counters"]
            return {
                "waiting_for_replay": True,
                "learner_state": "residual_actor_critic_training",
                "residual_actor_critic_cycle": self.completed_cycles,
                "residual_actor_optimizer_steps": counters[
                    "residual_actor_optimizer_steps"
                ],
            }
        self.completed_cycles += 1
        counters = self.learner["runtime"]["counters"]
        counters["residual_actor_update_attempts"] += 1
        if self.actor_updates_applied:
            counters["residual_actor_optimizer_steps"] += 1
        else:
            counters["residual_actor_updates_skipped_no_gradient"] += 1
        return {
            "waiting_for_replay": False,
            "learner_state": "residual_actor_critic_training",
            "residual_actor_critic_cycle": self.completed_cycles,
            "residual_actor_optimizer_steps": counters[
                "residual_actor_optimizer_steps"
            ],
            "residual_actor_update_attempts": counters[
                "residual_actor_update_attempts"
            ],
            "residual_actor_updates_skipped_no_gradient": counters[
                "residual_actor_updates_skipped_no_gradient"
            ],
            "learner_actor_steps": int(self.actor_updates_applied),
            "actor_update_attempted": True,
            "actor_update_applied": self.actor_updates_applied,
            "actor_update_skip_reason": (
                None
                if self.actor_updates_applied
                else "no_effective_gradient"
            ),
        }


class FailingDrainLearner(DrainLearner):
    def __call__(self, _coordinator):
        raise RuntimeError("synthetic learner failure")


class RecoveryDrainLearner(DrainLearner):
    def __init__(self) -> None:
        super().__init__(cycle_budget=7)
        self.completed_cycles = 3
        self.learner["runtime"]["residual_actor_critic_cycles"] = 3
        counters = self.learner["runtime"]["counters"]
        counters["residual_actor_optimizer_steps"] = 3
        counters["residual_actor_update_attempts"] = 3

    def outstanding_budget_status(self):
        return {
            "total_entitled_cycle_budget": self.cycle_budget,
            "completed_cycle_count": self.completed_cycles,
            "remaining_cycle_budget": self.cycle_budget - self.completed_cycles,
            "recovery_budget_drain_required": (
                self.completed_cycles < self.cycle_budget
            ),
        }

    def recovery_preflight(self):
        return self.outstanding_budget_status()

    def __call__(self, _coordinator):
        if self.completed_cycles >= self.cycle_budget:
            return {
                "waiting_for_replay": True,
                "learner_state": "residual_actor_critic_training",
                "residual_actor_critic_cycle": self.completed_cycles,
                "residual_actor_optimizer_steps": self.completed_cycles,
            }
        self.completed_cycles += 1
        self.learner["runtime"][
            "residual_actor_critic_cycles"
        ] = self.completed_cycles
        counters = self.learner["runtime"]["counters"]
        counters["residual_actor_optimizer_steps"] += 1
        counters["residual_actor_update_attempts"] += 1
        return {
            "waiting_for_replay": False,
            "learner_state": "residual_actor_critic_training",
            "residual_actor_critic_cycle": self.completed_cycles,
            "residual_actor_optimizer_steps": self.completed_cycles,
            "learner_actor_steps": 1,
        }


def runtime(tmp_path: Path) -> AsyncResidualActorCriticRuntime:
    revision = "task3-residual-policy-step-000000"
    machine = InMemoryRevisionStateMachine(
        RevisionRecord(revision, BASE_MODEL_ID, RevisionState.ACTIVE)
    )
    return AsyncResidualActorCriticRuntime(
        engine=FakeEngine(),
        machine=machine,
        session_id="session-1",
        episode_id="episode-1",
        active_revision_id=revision,
        active_model_revision=BASE_MODEL_ID,
        active_actor_checkpoint=tmp_path / "initial-residual",
        learner_resume_checkpoint=tmp_path / "seed",
        online_checkpoint_root=tmp_path
        / "online_ack_residual/training_checkpoints",
        learner_job=FakeLearner(),
        active_actor_online_cycle=0,
    )


def drain_runtime(tmp_path: Path, learner: FakeLearner) -> AsyncResidualActorCriticRuntime:
    revision = "task3-residual-policy-step-000000"
    machine = InMemoryRevisionStateMachine(
        RevisionRecord(revision, BASE_MODEL_ID, RevisionState.ACTIVE)
    )
    return AsyncResidualActorCriticRuntime(
        engine=FakeEngine(),
        machine=machine,
        session_id="session-1",
        episode_id="episode-1",
        active_revision_id=revision,
        active_model_revision=BASE_MODEL_ID,
        active_actor_checkpoint=tmp_path / "initial-residual",
        learner_resume_checkpoint=tmp_path / "seed",
        online_checkpoint_root=tmp_path
        / "online_ack_residual/training_checkpoints",
        learner_job=learner,
        active_actor_online_cycle=0,
    )


def identity() -> dict[str, str]:
    return {
        "session_id": "session-1",
        "episode_id": "episode-1",
        "policy_revision": BASE_MODEL_ID,
    }


def test_resume_keeps_fixed_base_and_restores_active_residual(tmp_path: Path) -> None:
    resume = (
        tmp_path
        / "online_ack_residual/training_checkpoints"
        / "residual_actor_critic_cycle_000010"
    )
    (resume / "state").mkdir(parents=True)
    (resume / "models").mkdir()
    base = tmp_path / "fixed-base"
    torch.save(
        {
            "frozen_base_policy_checkpoint": str(base),
            "active_residual_policy_revision": "task3-residual-policy-step-000010",
            "residual_actor_critic_cycles": 10,
            "online_adaptation_id": "task3-ack-residual-test",
        },
        resume / "state/runtime_state.pt",
    )
    torch.save({}, resume / "models/residual_actor.pt")
    candidate = (
        tmp_path
        / "online_ack_residual/policy_candidates/task3-ack-residual-test"
        / "residual_actor_step_000010"
    )
    candidate.mkdir(parents=True)
    torch.save({}, candidate / "residual_actor.pt")
    selected = _select_deployed_actor_for_resume(resume_checkpoint=resume)
    assert selected == (
        base.resolve(),
        (candidate / "residual_actor.pt").resolve(),
        "task3-residual-policy-step-000010",
        10,
    )


def test_candidate_contains_only_residual_actor_state(tmp_path: Path) -> None:
    learner = ResidualActorCriticLearner.__new__(ResidualActorCriticLearner)
    learner.checkpoint_root = (
        tmp_path / "online_ack_residual/training_checkpoints"
    )
    learner.learner = {
        "residual_actor": torch.nn.Linear(2, 1),
        "runtime": {
            "active_residual_policy_revision": "task3-residual-policy-step-000000",
            "online_adaptation_id": "task3-ack-residual-test",
        },
    }
    candidate = learner.export_actor_candidate(10)
    files = {
        path.relative_to(candidate["checkpoint"]).as_posix()
        for path in candidate["checkpoint"].rglob("*")
        if path.is_file()
    }
    assert candidate["revision_id"] == "task3-residual-policy-step-000010"
    assert files == {"residual_actor.pt"}
    with pytest.raises(RuntimeError, match="CANDIDATE_PATH_COLLISION"):
        learner.export_actor_candidate(10)


def test_unchanged_residual_actor_does_not_publish_candidate(tmp_path: Path) -> None:
    learner = ResidualActorCriticLearner.__new__(ResidualActorCriticLearner)
    learner.checkpoint_root = tmp_path / "online_ack_residual/training_checkpoints"
    actor = torch.nn.Linear(2, 1)
    active = torch.nn.Linear(2, 1)
    active.load_state_dict(actor.state_dict())
    learner.learner = {
        "residual_actor": actor,
        "runtime": {
            "active_residual_policy_revision": "task3-residual-policy-step-000000",
            "online_adaptation_id": "task3-ack-residual-test",
        },
    }
    assert learner.export_actor_candidate(
        10, active_residual_actor=active
    ) is None
    assert not (learner.checkpoint_root.parent / "policy_candidates").exists()


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
            "revision_id": "task3-residual-policy-step-000010",
            "checkpoint": candidate,
            "residual_actor_critic_cycle": 10,
        }
    )
    assert torch.count_nonzero(service.engine.residual_actor.weight) == 0
    assert service.active_revision_id.endswith("000000")
    service.end_episode(identity())
    assert torch.equal(service.engine.residual_actor.weight, replacement.weight)
    assert torch.equal(service.engine.residual_actor.bias, replacement.bias)
    assert service.active_revision_id.endswith("000010")
    assert service.engine.reset_count == 1
    assert service.learner_job.save_calls == 1
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


def test_prepare_episode_waits_for_admission_specific_budget_drain(
    tmp_path: Path,
) -> None:
    learner = DrainLearner(cycle_budget=7)
    service = drain_runtime(tmp_path, learner)
    try:
        service.start_episode(identity())
        service.end_episode(identity())
        with pytest.raises(RuntimeError, match="BEFORE_ADMISSION_DRAIN"):
            service.prepare_episode(
                {"session_id": "session-2", "episode_id": "episode-2"}
            )
        result = service.drain_admission_budget(
            {
                "session_id": "session-1",
                "episode_id": "episode-1",
                "admission_id": "admission-1",
                "timeout_seconds": 2.0,
            }
        )
        assert result["status"] == "TRAINING_BUDGET_DRAINED"
        assert result["computed_cycle_budget"] == 7
        assert result["completed_cycle_count"] == 7
        assert result["remaining_cycle_budget"] == 0
        assert result["twin_q_updates"] == 14
        assert result["residual_actor_updates"] == 7
        assert result["replay_refresh_ms"] == 4.0
        prepared = service.prepare_episode(
            {"session_id": "session-2", "episode_id": "episode-2"}
        )
        assert prepared["runtime_session_id"] == "session-2"
        assert prepared["runtime_episode_id"] == "episode-2"
    finally:
        service.stop()


def test_restart_recovers_and_drains_remaining_episode_budget(
    tmp_path: Path,
) -> None:
    service = drain_runtime(tmp_path, RecoveryDrainLearner())
    try:
        status = service.status()
        assert status["recovery_budget_drain_required"] is True
        assert status["total_entitled_cycle_budget"] == 7
        assert status["outstanding_training_cycle_budget"] == 4
        with pytest.raises(RuntimeError, match="BEFORE_ADMISSION_DRAIN"):
            service.prepare_episode(
                {"session_id": "session-2", "episode_id": "episode-2"}
            )

        result = service.drain_outstanding_budget(
            {"timeout_seconds": 2.0}
        )
        assert result["status"] == "OUTSTANDING_TRAINING_BUDGET_DRAINED"
        assert result["drained_cycle_count"] == 4
        assert result["remaining_cycle_budget"] == 0
        assert result["twin_q_updates"] == 8
        assert result["residual_actor_updates"] == 4

        prepared = service.prepare_episode(
            {"session_id": "session-2", "episode_id": "episode-2"}
        )
        assert prepared["runtime_session_id"] == "session-2"
        assert prepared["runtime_episode_id"] == "episode-2"
    finally:
        service.stop()


def test_zero_gradient_drain_reports_attempts_without_actor_updates(
    tmp_path: Path,
) -> None:
    service = drain_runtime(
        tmp_path,
        DrainLearner(cycle_budget=2, actor_updates_applied=False),
    )
    try:
        service.start_episode(identity())
        service.end_episode(identity())
        result = service.drain_admission_budget(
            {
                "session_id": "session-1",
                "episode_id": "episode-1",
                "admission_id": "admission-1",
                "timeout_seconds": 2.0,
            }
        )
        assert result["residual_actor_update_attempts"] == 2
        assert result["residual_actor_updates"] == 0
        assert result["residual_actor_updates_skipped_no_gradient"] == 2
        status = service.status()
        assert status["residual_actor_optimizer_steps"] == 0
        assert status["residual_actor_update_attempts"] == 2
        assert status["residual_actor_updates_skipped_no_gradient"] == 2
        assert status["actor_candidate_count"] == 0
    finally:
        service.stop()


def test_budget_drain_timeout_keeps_next_episode_blocked(tmp_path: Path) -> None:
    learner = DrainLearner(cycle_budget=1)
    learner.admission_budget_status = lambda _admission_id: None
    service = drain_runtime(tmp_path, learner)
    try:
        service.start_episode(identity())
        service.end_episode(identity())
        with pytest.raises(RuntimeError, match="TRAINING_DRAIN_TIMEOUT"):
            service.drain_admission_budget(
                {
                    "session_id": "session-1",
                    "episode_id": "episode-1",
                    "admission_id": "missing-admission",
                    "timeout_seconds": 0.01,
                }
            )
        assert service.status()["admission_resolution_required"] is True
        with pytest.raises(RuntimeError, match="BEFORE_ADMISSION_DRAIN"):
            service.prepare_episode(
                {"session_id": "session-2", "episode_id": "episode-2"}
            )
    finally:
        service.stop()


def test_budget_drain_learner_failure_keeps_next_episode_blocked(
    tmp_path: Path,
) -> None:
    service = drain_runtime(tmp_path, FailingDrainLearner(cycle_budget=1))
    try:
        service.start_episode(identity())
        service.end_episode(identity())
        with pytest.raises(RuntimeError, match="DRAIN_LEARNER_FAILED"):
            service.drain_admission_budget(
                {
                    "session_id": "session-1",
                    "episode_id": "episode-1",
                    "admission_id": "admission-1",
                    "timeout_seconds": 1.0,
                }
            )
        assert service.status()["learner_worker_state"] == "failed"
        with pytest.raises(RuntimeError, match="BEFORE_ADMISSION_DRAIN"):
            service.prepare_episode(
                {"session_id": "session-2", "episode_id": "episode-2"}
            )
    finally:
        service.stop()
