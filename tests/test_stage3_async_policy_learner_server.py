from __future__ import annotations

import http.client
from http.server import ThreadingHTTPServer
from pathlib import Path
import sys
import threading
import time

import pytest


ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from serve_forcerft_actor_learner import (  # noqa: E402
    AsyncPolicyLearnerRuntime,
    RequestHandler,
)
from run_stage3_async_actor_learner import reconcile_post_checkpoint_replay  # noqa: E402
from forcesmolvla.rft.stage3.update_credit import UpdateCreditLedger


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

    def infer(self, request):
        time.sleep(0.02)
        return {"request_id": request["request_id"], "actions": [[0.0] * 7] * 50}


def _learner_job(coordinator):
    with coordinator.learner_step_slot("critic", initial_estimate_s=0.0):
        time.sleep(0.005)
    return {
        "learner_critic_steps": 2,
        "learner_actor_steps": 1,
        "learner_polyak_steps": 2,
        "current_episode_sampled": False,
        "nonfinite_count": 0,
        "oom_count": 0,
        "pending_checkpoint_path": "/tmp/pending",
        "pending_candidate_id": "pending-cycle21",
        "pending_candidate_published": False,
        "pending_candidate_activated": False,
    }


def _runtime() -> AsyncPolicyLearnerRuntime:
    return AsyncPolicyLearnerRuntime(
        engine=FakeEngine(),
        machine=FakeMachine(),
        session_id="session-1",
        episode_id="episode_000000",
        active_revision_id="active-cycle10",
        active_model_revision="model-cycle10",
        learner_resume_checkpoint=Path("/tmp/cycle20"),
        pending_checkpoint=Path("/tmp/pending"),
        pending_candidate_id="pending-cycle21",
        learner_job=_learner_job,
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


def test_runtime_pins_actor_and_runs_one_learner_cycle() -> None:
    runtime = _runtime()
    runtime.start_episode(_identity())
    result = runtime.infer({
        "request_id": "request-1",
        "provenance": {"session_id": "session-1"},
    })
    assert result["request_id"] == "request-1"
    deadline = time.monotonic() + 2.0
    while runtime.status()["learner_state"] != "complete":
        assert time.monotonic() < deadline
        time.sleep(0.005)
    runtime.end_episode(_identity())
    status = runtime.status()
    assert status["actor_and_learner_concurrently_alive"] is True
    assert status["learner_critic_steps"] == 2
    assert status["learner_actor_steps"] == 1
    assert status["current_episode_sampled"] is False
    assert status["pending_candidate_published"] is False
    assert status["pending_candidate_activated"] is False


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


def test_http_runtime_endpoints_share_the_inference_runtime() -> None:
    runtime = _runtime()
    server = ThreadingHTTPServer(("127.0.0.1", 0), RequestHandler)
    server.engine = runtime
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    connection = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=2)
    try:
        import json

        body = json.dumps(_identity())
        connection.request(
            "POST", "/runtime/episode-start", body,
            {"Content-Type": "application/json"},
        )
        assert connection.getresponse().status == 200
        connection.close()
        connection = http.client.HTTPConnection(
            "127.0.0.1", server.server_port, timeout=2
        )
        infer = json.dumps({
            "request_id": "request-http",
            "provenance": {"session_id": "session-1"},
        })
        connection.request(
            "POST", "/infer", infer, {"Content-Type": "application/json"},
        )
        response = connection.getresponse()
        assert response.status == 200
        assert json.loads(response.read())["request_id"] == "request-http"
    finally:
        connection.close()
        runtime.abort_episode(_identity())
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
