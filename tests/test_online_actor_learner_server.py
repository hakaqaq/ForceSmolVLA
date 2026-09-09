from __future__ import annotations

from pathlib import Path
import json
import sys
import threading

import pytest
import torch


ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import serve_forcerft_residual_actor_critic as learner_server  # noqa: E402
from serve_forcerft_residual_actor_critic import (  # noqa: E402
    AsyncResidualActorCriticRuntime,
    ResidualActorCriticLearner,
    _select_deployed_actor_for_resume,
)
from forcesmolvla.rft.online.residual_actor_critic_runtime import (  # noqa: E402
    ONLINE_ADAPTATION_DIRECTORY_NAME,
    ResidualActorCriticSchedule,
)
from forcesmolvla.rft.online.residual_actor_critic_checkpoint import (  # noqa: E402
    CANDIDATE_CHECKPOINT_KIND,
)
from forcesmolvla.rft.online.policy_revision import (  # noqa: E402
    InMemoryRevisionStateMachine,
    RevisionRecord,
    RevisionState,
)
from forcesmolvla.rft.online.transition_authority import (  # noqa: E402
    ONLINE_SEMANTICS_VERSION,
)


BASE_MODEL_ID = "a" * 64


class FakeEngine:
    def __init__(self) -> None:
        self.metadata = {
            "service_role": "model_inference_only",
            "model_sha256": BASE_MODEL_ID,
        }
        self._lock = threading.Lock()
        self._residual_lock = threading.Lock()
        self.policy = torch.nn.Linear(1, 1)
        self.residual_actor = torch.nn.Linear(1, 1)
        torch.nn.init.zeros_(self.residual_actor.weight)
        torch.nn.init.zeros_(self.residual_actor.bias)
        self.reset_count = 0

    def reset_residual_episode_context(self) -> None:
        self.reset_count += 1

    def infer(self, request):
        return {"request_id": request["request_id"], "actions": [[0.0] * 7] * 50}

    def residual_decision(self, request):
        return {
            "decision_monotonic_ns": request["decision_monotonic_ns"],
            "base_absolute_action7": request.get(
                "base_absolute_action7", [0.0] * 7
            ),
        }


class FakeLearner:
    def __init__(self) -> None:
        self.training_policy = ResidualActorCriticSchedule()
        self.save_calls = 0
        self.replay_root = Path("/unused")
        self.learner = {
            "runtime": {
                "learner_state": "residual_actor_critic_training",
                "ack_critic_warmup_complete": True,
                "ack_critic_warmup_steps": 256,
                "residual_actor_critic_cycles": 0,
                "partial_cycle_q_updates": 0,
                "active_residual_policy_revision": "task3-residual-policy-step-000000",
                "online_adaptation_id": "task3-ack-residual-test",
                "counters": {
                    "twin_q_optimizer_steps": 256,
                    "residual_actor_optimizer_steps": 0,
                    "residual_actor_update_attempts": 0,
                    "residual_actor_updates_skipped_no_gradient": 0,
                    "twin_q_target_update_steps": 256,
                },
                "replay": {
                    "critic_td_valid_rows": 100,
                    "actor_q_valid_rows": 100,
                    "human_residual_valid_rows": 0,
                },
                "scheduling": {
                    "mode": "continuous_async",
                    "last_publish_attempt_cycle": 0,
                    "last_published_cycle": 0,
                    "publication_event_count": 0,
                    "last_periodic_checkpoint_cycle": 0,
                    "periodic_checkpoint_event_count": 0,
                    "active_publication_cycle": None,
                    "active_actor_optimizer_step": 0,
                    "active_actor_checkpoint": "/unused/actor",
                    "active_policy_epoch": 0,
                    "active_policy_epoch_status": "known",
                    "pending_publication": None,
                },
            }
        }

    def set_current_session(self, _session_id: str) -> None:
        pass

    def clear_current_session(self) -> None:
        pass

    def mark_active_residual_policy_revision(self, revision_id: str, **_metadata) -> None:
        self.learner["runtime"]["active_residual_policy_revision"] = revision_id

    def sampling_provenance(self, _session_id: str):
        return {
            "current_episode_sampled": False,
            "current_episode_replay_membership": False,
            "sampled_session_ids": [],
        }

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

    def notify_admission(self, admission_id: str) -> None:
        self.expected_admission_id = admission_id

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
        self.learner["runtime"]["residual_actor_critic_cycles"] = self.completed_cycles
        counters = self.learner["runtime"]["counters"]
        counters["twin_q_optimizer_steps"] += 2
        counters["twin_q_target_update_steps"] += 2
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


class SchedulingEventLearner(FakeLearner):
    def __init__(self, root: Path) -> None:
        super().__init__()
        self.root = root
        self.events: list[tuple[str, int]] = []

    def export_actor_candidate(self, cycle: int, _coordinator=None):
        checkpoint = self.root / f"candidate-{cycle}"
        checkpoint.mkdir()
        torch.save(torch.nn.Linear(1, 1).state_dict(), checkpoint / "residual_actor.pt")
        torch.save(
            {
                "checkpoint_kind": CANDIDATE_CHECKPOINT_KIND,
                "online_semantics_version": ONLINE_SEMANTICS_VERSION,
            },
            checkpoint / "candidate_state.pt",
        )
        scheduling = self.learner["runtime"]["scheduling"]
        scheduling["last_publish_attempt_cycle"] = cycle
        scheduling["last_published_cycle"] = cycle
        scheduling["publication_event_count"] += 1
        self.events.append(("publish", cycle))
        return {
            "revision_id": f"task3-residual-policy-cycle-{cycle:06d}-gtest",
            "checkpoint": checkpoint,
            "residual_actor_critic_cycle": cycle,
            "residual_actor_optimizer_steps": 0,
        }

    def save_checkpoint(self):
        cycle = self.learner["runtime"]["residual_actor_critic_cycles"]
        self.events.append(("checkpoint", cycle))
        return self.root / f"checkpoint-{cycle}"


class EpisodeOverlapLearner(FakeLearner):
    def __init__(self) -> None:
        super().__init__()
        self.gate = threading.Event()
        self.completed = False

    def __call__(self, coordinator):
        if not self.gate.is_set() or self.completed:
            return {
                "waiting_for_replay": True,
                "learner_state": "residual_actor_critic_training",
            }
        for kind in ("critic", "critic", "actor"):
            with coordinator.learner_step_slot(
                kind, initial_estimate_s=0.01, coverage_reserve_s=0.01
            ):
                pass
        runtime = self.learner["runtime"]
        runtime["residual_actor_critic_cycles"] = 1
        runtime["counters"]["twin_q_optimizer_steps"] += 2
        runtime["counters"]["twin_q_target_update_steps"] += 2
        runtime["counters"]["residual_actor_update_attempts"] += 1
        runtime["counters"]["residual_actor_updates_skipped_no_gradient"] += 1
        self.completed = True
        return {
            "waiting_for_replay": False,
            "learner_state": "residual_actor_critic_training",
            "residual_actor_critic_cycle": 1,
            "learner_actor_steps": 0,
            "actor_update_attempted": True,
            "actor_update_applied": False,
            "actor_update_skip_reason": "no_effective_gradient",
        }


class RecoveryDrainLearner(DrainLearner):
    def __init__(self) -> None:
        super().__init__(cycle_budget=7)
        self.completed_cycles = 3
        self.learner["runtime"]["residual_actor_critic_cycles"] = 3
        counters = self.learner["runtime"]["counters"]
        counters["residual_actor_optimizer_steps"] = 3
        counters["residual_actor_update_attempts"] = 3

    def recovery_preflight(self):
        return {
            "scheduling_mode": "continuous_async",
            "completed_cycle_count": self.completed_cycles,
            "recovery_budget_drain_required": False,
        }

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
    learner = FakeLearner()
    learner.replay_root = tmp_path / "formal_replay"
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
        / ONLINE_ADAPTATION_DIRECTORY_NAME
        / "training_checkpoints",
        learner_job=learner,
        active_actor_online_cycle=0,
    )


def drain_runtime(tmp_path: Path, learner: FakeLearner) -> AsyncResidualActorCriticRuntime:
    revision = "task3-residual-policy-step-000000"
    machine = InMemoryRevisionStateMachine(
        RevisionRecord(revision, BASE_MODEL_ID, RevisionState.ACTIVE)
    )
    learner.replay_root = tmp_path / "formal_replay"
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
        / ONLINE_ADAPTATION_DIRECTORY_NAME
        / "training_checkpoints",
        learner_job=learner,
        active_actor_online_cycle=0,
    )


def identity() -> dict[str, str]:
    return {
        "session_id": "session-1",
        "episode_id": "episode-1",
        "policy_revision": BASE_MODEL_ID,
    }


def test_metadata_reads_active_policy_epoch_status_from_scheduling(
    tmp_path: Path,
) -> None:
    service = runtime(tmp_path)
    try:
        assert service.metadata["active_policy_epoch_status"] == "known"
    finally:
        service.stop()


def write_committed_admission(root: Path, admission_id: str = "001__episode-1") -> None:
    (root / "episodes").mkdir(parents=True, exist_ok=True)
    (root / "admissions").mkdir()
    episode_id = "1/episode-1"
    (root / "episodes" / f"{admission_id}.json").write_text(
        json.dumps({
            "status": "SEALED_COMMITTED",
            "admission_id": admission_id,
            "episode_id": episode_id,
            "admission_record": f"admissions/{admission_id}.json",
        })
    )
    (root / "admissions" / f"{admission_id}.json").write_text(
        json.dumps({
            "admission_id": admission_id,
            "episode_id": episode_id,
            "episode_sealed": True,
        })
    )


def test_resume_keeps_fixed_base_and_restores_active_residual(tmp_path: Path) -> None:
    resume = (
        tmp_path
        / ONLINE_ADAPTATION_DIRECTORY_NAME
        / "training_checkpoints"
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
        / ONLINE_ADAPTATION_DIRECTORY_NAME
        / "policy_candidates/task3-ack-residual-test"
        / "residual_actor_step_000010"
    )
    candidate.mkdir(parents=True)
    torch.save({}, candidate / "residual_actor.pt")
    torch.save(
        {
            "checkpoint_kind": CANDIDATE_CHECKPOINT_KIND,
            "online_semantics_version": ONLINE_SEMANTICS_VERSION,
        },
        candidate / "candidate_state.pt",
    )
    selected = _select_deployed_actor_for_resume(resume_checkpoint=resume)
    assert selected == (
        base.resolve(),
        candidate.resolve(),
        "task3-residual-policy-step-000010",
        None,
        10,
        0,
        "legacy_unknown",
    )


def test_candidate_contains_only_residual_actor_state(tmp_path: Path) -> None:
    learner = ResidualActorCriticLearner.__new__(ResidualActorCriticLearner)
    learner.checkpoint_root = (
        tmp_path / ONLINE_ADAPTATION_DIRECTORY_NAME / "training_checkpoints"
    )
    learner._state_lock = threading.RLock()
    learner.learner = {
        "residual_actor": torch.nn.Linear(2, 1),
        "runtime": {
            "active_residual_policy_revision": "task3-residual-policy-step-000000",
            "online_adaptation_id": "task3-ack-residual-test",
            "counters": {"residual_actor_optimizer_steps": 7},
            "scheduling": {
                "candidate_export_generation": "123-test",
                "last_publish_attempt_cycle": 0,
                "last_published_cycle": 0,
                "publication_event_count": 0,
            },
        },
    }
    candidate = learner.export_actor_candidate(100)
    files = {
        path.relative_to(candidate["checkpoint"]).as_posix()
        for path in candidate["checkpoint"].rglob("*")
        if path.is_file()
    }
    assert candidate["revision_id"] == "task3-residual-policy-cycle-000100-g123"
    assert files == {"residual_actor.pt", "candidate_state.pt"}
    repeated = learner.export_actor_candidate(200)
    assert repeated["checkpoint"] == candidate["checkpoint"]
    assert repeated["revision_id"] != candidate["revision_id"]
    assert learner.learner["runtime"]["scheduling"]["publication_event_count"] == 2


def test_unchanged_residual_actor_still_records_cycle_publication(tmp_path: Path) -> None:
    learner = ResidualActorCriticLearner.__new__(ResidualActorCriticLearner)
    learner.checkpoint_root = (
        tmp_path / ONLINE_ADAPTATION_DIRECTORY_NAME / "training_checkpoints"
    )
    learner._state_lock = threading.RLock()
    actor = torch.nn.Linear(2, 1)
    active = torch.nn.Linear(2, 1)
    active.load_state_dict(actor.state_dict())
    learner.learner = {
        "residual_actor": actor,
        "runtime": {
            "active_residual_policy_revision": "task3-residual-policy-step-000000",
            "online_adaptation_id": "task3-ack-residual-test",
            "counters": {"residual_actor_optimizer_steps": 0},
            "scheduling": {
                "candidate_export_generation": "456-test",
                "last_publish_attempt_cycle": 0,
                "last_published_cycle": 0,
                "publication_event_count": 0,
            },
        },
    }
    candidate = learner.export_actor_candidate(100)
    assert candidate["residual_actor_optimizer_steps"] == 0
    assert candidate["checkpoint"].is_dir()
    assert learner.learner["runtime"]["scheduling"]["last_published_cycle"] == 100


def test_cycle_publication_and_checkpoint_events_use_100_1000_cadence(
    tmp_path: Path,
) -> None:
    learner = SchedulingEventLearner(tmp_path)
    service = drain_runtime(tmp_path, learner)
    try:
        for cycle in (99, 100, 100, 200, 999, 1000, 1999, 2000):
            learner.learner["runtime"]["residual_actor_critic_cycles"] = cycle
            service._process_completed_cycle_events(
                {"residual_actor_critic_cycle": cycle}
            )
        assert learner.events == [
            ("publish", 100),
            ("publish", 200),
            ("publish", 1000),
            ("checkpoint", 1000),
            ("publish", 2000),
            ("checkpoint", 2000),
        ]
        assert service.machine.pending_revision_id == (
            "task3-residual-policy-cycle-002000-gtest"
        )
        scheduling = learner.learner["runtime"]["scheduling"]
        assert scheduling["last_published_cycle"] == 2000
        assert scheduling["last_periodic_checkpoint_cycle"] == 2000
        assert scheduling["publication_event_count"] == 4
        assert scheduling["periodic_checkpoint_event_count"] == 2
    finally:
        service.stop()


def test_cycle_100_candidate_activates_only_after_episode_boundary(
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
    torch.save(
        {
            "checkpoint_kind": CANDIDATE_CHECKPOINT_KIND,
            "online_semantics_version": ONLINE_SEMANTICS_VERSION,
        },
        candidate / "candidate_state.pt",
    )

    service.start_episode(identity())
    service._stage_actor_candidate(
        {
            "revision_id": "task3-residual-policy-cycle-000100-gtest",
            "checkpoint": candidate,
            "residual_actor_critic_cycle": 100,
            "residual_actor_optimizer_steps": 63,
        }
    )
    assert torch.count_nonzero(service.engine.residual_actor.weight) == 0
    assert service.active_revision_id.endswith("000000")
    service.end_episode(identity())
    assert torch.equal(service.engine.residual_actor.weight, replacement.weight)
    assert torch.equal(service.engine.residual_actor.bias, replacement.bias)
    assert "cycle-000100" in service.active_revision_id
    assert service.engine.reset_count == 1
    assert service.learner_job.save_calls == 0
    assert all(
        torch.equal(base_before[name], value)
        for name, value in service.engine.policy.state_dict().items()
    )
    assert service.active_model_revision == BASE_MODEL_ID


def test_resume_restores_pending_candidate_without_auto_activation(
    tmp_path: Path,
) -> None:
    candidate = tmp_path / "restored-candidate"
    candidate.mkdir()
    replacement = torch.nn.Linear(1, 1)
    torch.nn.init.constant_(replacement.weight, 4.0)
    torch.save(replacement.state_dict(), candidate / "residual_actor.pt")
    torch.save(
        {
            "checkpoint_kind": CANDIDATE_CHECKPOINT_KIND,
            "online_semantics_version": ONLINE_SEMANTICS_VERSION,
        },
        candidate / "candidate_state.pt",
    )
    learner = FakeLearner()
    learner.learner["runtime"]["scheduling"]["pending_publication"] = {
        "revision_id": "task3-residual-policy-cycle-000100-grestore",
        "checkpoint": str(candidate),
        "residual_actor_critic_cycle": 100,
        "residual_actor_optimizer_steps": 77,
    }
    service = drain_runtime(tmp_path, learner)
    try:
        assert service.active_revision_id.endswith("step-000000")
        assert service.machine.pending_revision_id.endswith("grestore")
        service.prepare_episode(
            {"session_id": "session-2", "episode_id": "episode-2"}
        )
        assert service.active_revision_id.endswith("grestore")
        assert torch.equal(
            service.engine.residual_actor.weight, replacement.weight
        )
        assert service.status()["active_actor_online_cycle"] == 100
    finally:
        service.stop()


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
    with pytest.raises(RuntimeError, match="RUNTIME_QUIESCED"):
        service.prepare_episode(
            {"session_id": "session-2", "episode_id": "episode-2"}
        )


def test_same_generation_checkpoint_save_is_idempotent_but_binding_change_is_not(
    tmp_path: Path, monkeypatch,
) -> None:
    module = torch.nn.Linear(1, 1)
    actor_optimizer = torch.optim.Adam(module.parameters(), lr=1e-4)
    critic_optimizer = torch.optim.Adam(module.parameters(), lr=3e-4)
    learner = ResidualActorCriticLearner.__new__(ResidualActorCriticLearner)
    learner._state_lock = threading.RLock()
    learner._last_saved_checkpoint_signature = None
    learner._last_saved_checkpoint_path = None
    learner.checkpoint_root = tmp_path / "training_checkpoints"
    learner.training_policy = ResidualActorCriticSchedule()
    learner.learner = {
        "residual_actor": module,
        "residual_actor_target": module,
        "q1": module,
        "q2": module,
        "q1_target": module,
        "q2_target": module,
        "residual_actor_optimizer": actor_optimizer,
        "critic_optimizer": critic_optimizer,
        "config": {},
        "runtime": {
            "residual_actor_critic_cycles": 1000,
            "active_residual_policy_revision": "actor-cycle-900",
            "scheduling": {"last_saved_checkpoint_cycle": 0},
        },
    }
    saved = []

    def fake_save(path, **kwargs):
        path.mkdir(parents=True, exist_ok=True)
        saved.append(kwargs["runtime_state"])
        return path

    monkeypatch.setattr(
        learner_server, "save_residual_actor_critic_checkpoint", fake_save
    )
    monkeypatch.setattr(
        learner_server,
        "exact_resume_checkpoint_is_recoverable",
        lambda *_args, **_kwargs: True,
    )
    monkeypatch.setattr(
        learner_server,
        "retain_latest_training_checkpoints",
        lambda *_args, **_kwargs: (),
    )

    first = learner.save_checkpoint()
    second = learner.save_checkpoint()
    assert first == second
    assert len(saved) == 1

    learner.learner["runtime"]["active_residual_policy_revision"] = (
        "actor-cycle-1000"
    )
    learner.save_checkpoint()
    assert len(saved) == 2


def test_episode_active_allows_real_critic_and_actor_attempt_overlap(
    tmp_path: Path,
) -> None:
    learner = EpisodeOverlapLearner()
    service = drain_runtime(tmp_path, learner)
    try:
        service.start_episode(identity())
        learner.gate.set()
        service._wake_learner.set()
        with service._lock:
            assert service._lock.wait_for(lambda: learner.completed, timeout=2.0)
        status = service.status()
        assert status["episode_active"] is True
        assert status["completed_learner_cycles"] == 1
        assert status["total_twin_q_optimizer_steps"] == 258
        assert status["residual_actor_update_attempts"] == 1
        assert status["actor_and_learner_concurrently_alive"] is True
        assert status["capture_window"]["delta"][
            "completed_learner_cycles"
        ] == 1
        service.abort_episode(identity())
    finally:
        service.stop()


def test_residual_decision_echoes_control_generation_not_model_epoch(
    tmp_path: Path,
) -> None:
    service = runtime(tmp_path)
    service.start_episode(identity())
    service.machine.invalidate_policy_epoch("human_takeover")

    result = service.residual_decision(
        {
            "session_id": "session-1",
            "decision_monotonic_ns": 123,
            "control_policy_epoch": 7,
            "control_takeover_generation": 4,
            "base_absolute_action7": [0.0] * 7,
        }
    )

    assert result["control_policy_epoch"] == 7
    assert result["control_takeover_generation"] == 4
    assert result["residual_policy_epoch"] == 1
    assert result["active_residual_policy_revision"].endswith("000000")
    service.abort_episode(identity())


def test_committed_admission_notification_releases_next_capture_without_drain(
    tmp_path: Path,
) -> None:
    learner = DrainLearner(cycle_budget=7)
    service = drain_runtime(tmp_path, learner)
    write_committed_admission(learner.replay_root)
    try:
        service.start_episode(identity())
        service.end_episode(identity())
        with pytest.raises(RuntimeError, match="BEFORE_ADMISSION_RESOLUTION"):
            service.prepare_episode(
                {"session_id": "session-2", "episode_id": "episode-2"}
            )
        result = service.notify_admission_committed(
            {
                "session_id": "session-1",
                "episode_id": "episode-1",
                "admission_id": "001__episode-1",
            }
        )
        assert result["status"] == "FORMAL_ADMISSION_REGISTERED"
        prepared = service.prepare_episode(
            {"session_id": "session-2", "episode_id": "episode-2"}
        )
        assert prepared["runtime_session_id"] == "session-2"
        assert prepared["runtime_episode_id"] == "episode-2"
        with pytest.raises(RuntimeError, match="DRAIN_NOT_APPLICABLE"):
            service.drain_admission_budget({})
    finally:
        service.stop()


def test_restart_has_no_legacy_budget_debt_barrier(
    tmp_path: Path,
) -> None:
    service = drain_runtime(tmp_path, RecoveryDrainLearner())
    try:
        status = service.status()
        assert status["recovery_budget_drain_required"] is False
        prepared = service.prepare_episode(
            {"session_id": "session-2", "episode_id": "episode-2"}
        )
        assert prepared["runtime_session_id"] == "session-2"
        assert prepared["runtime_episode_id"] == "episode-2"
        with pytest.raises(RuntimeError, match="DRAIN_NOT_APPLICABLE"):
            service.drain_outstanding_budget({})
    finally:
        service.stop()


def test_zero_gradient_cycles_count_attempts_without_actor_updates(
    tmp_path: Path,
) -> None:
    learner = DrainLearner(cycle_budget=2, actor_updates_applied=False)
    service = drain_runtime(tmp_path, learner)
    write_committed_admission(service.learner_job.replay_root)
    try:
        service.start_episode(identity())
        service.end_episode(identity())
        service.notify_admission_committed(
            {
                "session_id": "session-1",
                "episode_id": "episode-1",
                "admission_id": "001__episode-1",
            }
        )
        with service._lock:
            assert service._lock.wait_for(
                lambda: learner.completed_cycles == 2, timeout=2.0
            )
        status = service.status()
        assert status["residual_actor_optimizer_steps"] == 0
        assert status["residual_actor_update_attempts"] == 2
        assert status["residual_actor_updates_skipped_no_gradient"] == 2
        assert status["actor_candidate_count"] == 0
    finally:
        service.stop()


def test_uncommitted_admission_does_not_release_next_episode(tmp_path: Path) -> None:
    service = drain_runtime(tmp_path, DrainLearner(cycle_budget=1))
    try:
        service.start_episode(identity())
        service.end_episode(identity())
        with pytest.raises(RuntimeError, match="COMMITTED_MANIFEST_MISSING"):
            service.notify_admission_committed({
                "session_id": "session-1",
                "episode_id": "episode-1",
                "admission_id": "missing-admission",
            })
        assert service.status()["admission_resolution_required"] is True
    finally:
        service.stop()


def test_background_learner_failure_is_reported(
    tmp_path: Path,
) -> None:
    service = drain_runtime(tmp_path, FailingDrainLearner(cycle_budget=1))
    try:
        with service._lock:
            assert service._lock.wait_for(
                lambda: service._learner_worker_state == "failed", timeout=1.0
            )
        assert service.status()["learner_worker_state"] == "failed"
    finally:
        service.stop()
