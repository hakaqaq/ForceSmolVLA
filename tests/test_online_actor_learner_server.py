from __future__ import annotations

import io
from pathlib import Path
import sys
import threading
import time
from types import SimpleNamespace

import pytest
import torch


ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import serve_forcerft_actor_learner as server  # noqa: E402
from serve_forcerft_actor_learner import (  # noqa: E402
    AsyncPolicyLearnerRuntime,
    ContinuousLearner,
    RequestHandler,
)
from forcesmolvla.rft.online.actor_learner_runtime import (
    InferencePriorityCoordinator,
    reconcile_post_checkpoint_replay,
)
from forcesmolvla.rft.online.sample_credit import UpdateCreditLedger


class FakeMachine:
    active_revision_id = "active-cycle10"
    policy_epoch = 1

    def __init__(self) -> None:
        self.active = False

    def begin_episode(self) -> str:
        assert not self.active
        self.active = True
        return self.active_revision_id

    def episode_pin(self):
        return type("Pin", (), {"model_sha256": "model-cycle10", "policy_epoch": 1})()

    def assert_episode_binding(self, revision, model, epoch) -> None:
        assert (revision, model, epoch) == (
            self.active_revision_id, "model-cycle10", 1,
        )

    def end_episode(self) -> None:
        assert self.active
        self.active = False


class FakeEngine:
    metadata = {
        "service_role": "model_inference_only",
        "model_sha256": "model-cycle10",
    }

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.policy = FakePolicy()

    def infer(self, request):
        time.sleep(0.02)
        return {"request_id": request["request_id"], "actions": [[0.0] * 7] * 50}


class FakePolicy:
    def __init__(self) -> None:
        self.value = 0

    def state_dict(self):
        return {"value": self.value}

    def load_state_dict(self, state, strict=True):
        assert strict is True
        self.value = state["value"]

    def eval(self):
        return self


class FakeLearner:
    def __init__(self) -> None:
        self.cycle = 0
        self.save_calls = 0

    def set_current_session(self, _session_id: str) -> None:
        pass

    def clear_current_session(self) -> None:
        pass

    def save_checkpoint(self):
        self.save_calls += 1
        return None

    def __call__(self, coordinator):
        with coordinator.learner_step_slot("critic", initial_estimate_s=0.0):
            time.sleep(0.005)
        self.cycle += 1
        actor = FakePolicy()
        actor.value = self.cycle
        return {
            "learner_critic_steps": 2,
            "learner_actor_steps": 1,
            "learner_polyak_steps": 2,
            "current_episode_sampled": False,
            "nonfinite_count": 0,
            "oom_count": 0,
            "online_joint_cycle": self.cycle,
            "learner_actor": actor,
            "latest_checkpoint_path": None,
        }


class FailingLearner(FakeLearner):
    def __call__(self, coordinator):
        raise RuntimeError("learner failed after a partial cycle")


def _runtime(
    *, learner: FakeLearner | None = None, checkpoint_root: Path | None = None
) -> AsyncPolicyLearnerRuntime:
    return AsyncPolicyLearnerRuntime(
        engine=FakeEngine(),
        machine=FakeMachine(),
        session_id="session-1",
        episode_id="episode_000000",
        active_revision_id="active-cycle10",
        active_model_revision="model-cycle10",
        learner_resume_checkpoint=Path("/tmp/cycle20"),
        online_checkpoint_root=checkpoint_root or Path("/tmp/online-checkpoints"),
        learner_job=learner or FakeLearner(),
    )


def _identity() -> dict[str, str]:
    return {
        "session_id": "session-1",
        "episode_id": "episode_000000",
        "policy_revision": "model-cycle10",
    }


def test_resume_reconciles_append_only_replay_and_credits_once() -> None:
    credits = UpdateCreditLedger(
        credits_per_transition=1, credits_per_joint_cycle=1,
    )
    checkpoint_uids = [f"checkpoint-{index:03d}" for index in range(618)]
    post_checkpoint_uids = [f"episode-006-{index:03d}" for index in range(202)]
    for uid in checkpoint_uids:
        assert credits.mint_for_unique_online_transition(uid)
    for _ in range(122):
        credits.consume_joint_cycle()
    rows = [
        {
            "identity": {
                "transition_uid": uid,
                "episode_id": (
                    "episode_006" if uid.startswith("episode-006-")
                    else "checkpoint_episode"
                ),
            }
        }
        for uid in checkpoint_uids + post_checkpoint_uids
    ]

    assert reconcile_post_checkpoint_replay(credits, rows) == 202
    assert credits.snapshot().credited_transition_count == 820
    assert credits.snapshot().available == 698
    assert {
        row["identity"]["transition_uid"]
        for row in rows if row["identity"]["episode_id"] == "episode_006"
    } == set(post_checkpoint_uids)
    assert reconcile_post_checkpoint_replay(credits, rows) == 0
    assert credits.snapshot().available == 698


def test_less_than_100_online_rows_runs_no_optimizer_step(tmp_path: Path) -> None:
    learner = ContinuousLearner(
        device=torch.device("cpu"),
        resume_checkpoint=tmp_path / "resume",
        checkpoint_root=tmp_path / "checkpoints",
        replay_root=tmp_path / "online",
        current_session_id=None,
    )
    result = learner(InferencePriorityCoordinator())
    assert result["waiting_for_replay"] is True
    assert result["learner_critic_steps"] == 0
    assert result["learner_actor_steps"] == 0
    assert result["learner_polyak_steps"] == 0
    assert learner.learner is None


def test_training_start_prepares_learner_on_configured_device(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    device = torch.device("cpu")
    learner = ContinuousLearner(
        device=device,
        resume_checkpoint=tmp_path / "resume",
        checkpoint_root=tmp_path / "checkpoints",
        replay_root=tmp_path / "online",
        current_session_id=None,
    )
    observed: list[torch.device] = []
    monkeypatch.setattr(
        server.warmup,
        "count_sealed_autonomous_policy_transitions",
        lambda _root: 100,
    )
    monkeypatch.setattr(
        server.warmup,
        "load_formal_online_r",
        lambda _root: ([], [], {}, []),
    )
    monkeypatch.setattr(
        server,
        "prepare_learner",
        lambda configured_device, *_args, **_kwargs: observed.append(
            configured_device
        ) or {},
    )

    assert learner._ensure_learner() is True
    assert observed == [device]


def test_refreshed_schedule_is_prefetched_before_next_cycle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    schedules = ([[1], [2]], [[3], [4]], [[5]], [[6]])
    calls = []

    class DemoReplay:
        population = (0, 1, 2)

        def prefetch_joint(self, critic, actor) -> None:
            calls.append((critic, actor))

    learner = {
        "r_rng": object(),
        "d_rng": object(),
        "r_replay": SimpleNamespace(macros=(0, 1, 2)),
        "d_replay": DemoReplay(),
    }
    monkeypatch.setattr(server.joint, "make_schedules", lambda *_args, **_kwargs: schedules)

    server._refresh_training_schedules(learner)

    assert learner["critic_r"] == schedules[0]
    assert learner["actor_d"] == schedules[3]
    assert calls == [(schedules[1], schedules[3])]


def test_cycle_five_broadcast_is_memory_only_and_stop_saves_once(
    tmp_path: Path,
) -> None:
    learner = FakeLearner()
    checkpoint_root = tmp_path / "online-checkpoints"
    runtime = _runtime(learner=learner, checkpoint_root=checkpoint_root)
    runtime.start_episode(_identity())
    runtime.infer({
        "request_id": "request-1",
        "provenance": {"session_id": "session-1"},
    })
    deadline = time.monotonic() + 2.0
    while runtime.status()["actor_parameter_broadcast_count"] < 1:
        assert time.monotonic() < deadline
        time.sleep(0.005)
    assert runtime.engine.policy.value >= 5
    assert not checkpoint_root.exists()
    runtime.abort_episode(_identity())
    runtime.stop()
    assert learner.save_calls == 1


def test_failed_learner_is_not_checkpointed_on_stop() -> None:
    learner = FailingLearner()
    runtime = _runtime(learner=learner)
    runtime.start_episode(_identity())
    runtime.infer({
        "request_id": "request-1",
        "provenance": {"session_id": "session-1"},
    })
    deadline = time.monotonic() + 2.0
    while runtime.status()["learner_state"] != "failed":
        assert time.monotonic() < deadline
        time.sleep(0.005)
    runtime.abort_episode(_identity())
    runtime.stop()
    assert learner.save_calls == 0


def test_runtime_pins_actor_and_runs_persistent_learner() -> None:
    runtime = _runtime()
    runtime.start_episode(_identity())
    result = runtime.infer({
        "request_id": "request-1",
        "provenance": {"session_id": "session-1"},
    })
    assert result["request_id"] == "request-1"
    deadline = time.monotonic() + 2.0
    while runtime.status()["learner_actor_steps"] < 1:
        assert time.monotonic() < deadline
        time.sleep(0.005)
    runtime.end_episode(_identity())
    status = runtime.status()
    assert status["actor_and_learner_concurrently_alive"] is True
    assert status["learner_critic_steps"] == 2
    assert status["learner_actor_steps"] == 1
    assert status["current_episode_sampled"] is False
    assert status["server_persistent"] is True
    runtime.stop()


def test_runtime_rejects_capture_and_inference_identity_mismatch() -> None:
    runtime = _runtime()
    with pytest.raises(RuntimeError, match="CAPTURE_IDENTITY_MISMATCH"):
        runtime.start_episode({**_identity(), "episode_id": "wrong"})
    runtime.start_episode(_identity())
    with pytest.raises(RuntimeError, match="INFERENCE_SESSION_MISMATCH"):
        runtime.infer({
            "request_id": "request-1",
            "provenance": {"session_id": "wrong"},
        })
    runtime.abort_episode(_identity())
    runtime.stop()


def test_http_runtime_endpoints_share_the_inference_runtime() -> None:
    runtime = _runtime()
    handler = object.__new__(RequestHandler)
    handler.server = SimpleNamespace(engine=runtime)
    responses: list[tuple[int, dict]] = []
    handler._write_json = lambda code, payload: responses.append((code, payload))
    try:
        import json

        body = json.dumps(_identity()).encode()
        handler.path = "/runtime/episode-start"
        handler.headers = {"Content-Length": str(len(body))}
        handler.rfile = io.BytesIO(body)
        handler.do_POST()
        assert responses.pop(0)[0] == 200

        infer = json.dumps({
            "request_id": "request-http",
            "provenance": {"session_id": "session-1"},
        }).encode()
        handler.path = "/infer"
        handler.headers = {"Content-Length": str(len(infer))}
        handler.rfile = io.BytesIO(infer)
        handler.do_POST()
        assert responses.pop(0) == (200, {
            "request_id": "request-http", "actions": [[0.0] * 7] * 50,
        })
    finally:
        runtime.abort_episode(_identity())
        runtime.stop()
