from __future__ import annotations

import json
from pathlib import Path
import threading
import time

import numpy as np
import pytest
import torch

from forcesmolvla.rft.critic_action_adapter_v2 import CRITIC_ACTION_CONTRACT

from forcesmolvla.rft.online.actor_learner_runtime import (
    AsyncRuntimeError,
    EpisodePin,
    H50ActionCache,
    InferenceRequest,
    InferencePriorityCoordinator,
    OnlineTrainingPolicy,
    PinnedEpisode,
    TakeoverWindow,
    prepare_learner,
    run_concurrent_window,
    run_timed_actor,
    online_checkpoint_path,
    retain_latest_online_checkpoints,
    select_exact_resume_checkpoint,
)


EXACT_RESUME_FILES = (
    "actor/model.safetensors", "actor/config.json", "actor/artifact_manifest.json",
    "models/q1_state.pt", "models/q2_state.pt", "models/q1_target_state.pt",
    "models/q2_target_state.pt", "optimizers/actor_optimizer_state.pt",
    "optimizers/critic_optimizer_state.pt", "optimizers/actor_scheduler_state.pt",
    "optimizers/critic_scheduler_state.pt", "state/runtime_state.pt",
    "artifacts/normalizer_manifest.json", "artifacts/action_delta_spec.json",
)


def _exact_checkpoint(path: Path, kind: str, *, compatible: bool = True) -> None:
    path.mkdir(parents=True)
    metadata = {
        "complete": True, "kind": kind, "actor_directory": "actor",
    }
    if kind == "online_actor_critic_exact_resume" and compatible:
        metadata["critic_action_contract_version"] = (
            CRITIC_ACTION_CONTRACT.version
        )
    (path / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    for relative in EXACT_RESUME_FILES:
        target = path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.touch()
    online_cycles = (
        int(path.name.rsplit("_", 1)[1])
        if kind == "online_actor_critic_exact_resume"
        else 0
    )
    joint_cycles = 210 + online_cycles
    torch.save({
        "online_joint_cycles": online_cycles,
        "counters": {
            "joint_cycles": joint_cycles,
            "critic_optimizer_steps": joint_cycles * 2,
            "actor_optimizer_steps": joint_cycles,
            "target_polyak_steps": joint_cycles * 2,
        },
    }, path / "state/runtime_state.pt")
    torch.save(
        {"last_epoch": joint_cycles},
        path / "optimizers/actor_scheduler_state.pt",
    )


class FakeRegistry:
    def __init__(self) -> None:
        self.active = "active"
        self.model = "model"
        self.epoch = 3
        self.episode = False
        self.pinned = None

    def begin_episode(self) -> str:
        assert not self.episode
        self.episode = True
        self.pinned = (self.active, self.model, self.epoch)
        return self.active

    def episode_pin(self):
        return type("Pin", (), {
            "model_sha256": self.model,
            "policy_epoch": self.epoch,
        })()

    def assert_episode_binding(self, revision, model, epoch) -> None:
        assert (revision, model, epoch) == self.pinned

    def end_episode(self) -> None:
        assert self.episode
        self.episode = False
        self.pinned = None


def test_fixed_online_training_schedule_and_checkpoint_retention(tmp_path) -> None:
    policy = OnlineTrainingPolicy()
    assert not policy.training_ready(99)
    assert policy.training_ready(100)
    assert policy.broadcast_due(5) and not policy.broadcast_due(6)
    assert policy.checkpoint_due(50) and not policy.checkpoint_due(55)
    for cycle in (50, 100, 107):
        online_checkpoint_path(tmp_path, cycle).mkdir()
    retained = retain_latest_online_checkpoints(tmp_path)
    assert [path.name for path in retained] == [
        "online_actor_critic_cycle_000100",
        "online_actor_critic_cycle_000107",
    ]
    assert not online_checkpoint_path(tmp_path, 50).exists()


def test_resume_selection_prefers_latest_recoverable_online_then_offline(tmp_path) -> None:
    offline = tmp_path / "offline/checkpoints/offline_actor_critic_cycle_000210"
    _exact_checkpoint(offline, "offline_actor_critic_exact_resume")
    assert select_exact_resume_checkpoint(tmp_path) == offline.resolve()

    checkpoint_root = tmp_path / "online/checkpoints"
    cycle_50 = online_checkpoint_path(checkpoint_root, 50)
    cycle_75_legacy = online_checkpoint_path(checkpoint_root, 75)
    cycle_100 = online_checkpoint_path(checkpoint_root, 100)
    cycle_107_incomplete = online_checkpoint_path(checkpoint_root, 107)
    _exact_checkpoint(cycle_50, "online_actor_critic_exact_resume")
    _exact_checkpoint(
        cycle_75_legacy, "online_actor_critic_exact_resume", compatible=False
    )
    _exact_checkpoint(cycle_100, "online_actor_critic_exact_resume")
    torch.save(
        {"last_epoch": 999},
        cycle_100 / "optimizers/actor_scheduler_state.pt",
    )
    cycle_107_incomplete.mkdir(parents=True)
    assert select_exact_resume_checkpoint(tmp_path) == cycle_50.resolve()


def test_prepare_learner_uses_exact_resume_loader_signature(tmp_path) -> None:
    checkpoint = tmp_path / "offline_actor_critic_cycle_000210"
    checkpoint.mkdir()
    (checkpoint / "metadata.json").write_text(
        json.dumps({"actor_directory": "actor"}), encoding="utf-8"
    )

    class LoaderReached(RuntimeError):
        pass

    class JointApi:
        @staticmethod
        def load_resume_modules(checkpoint_path, actor_path, device):
            assert checkpoint_path == checkpoint
            assert actor_path == checkpoint / "actor"
            assert device == torch.device("cpu")
            raise LoaderReached

    with pytest.raises(LoaderReached):
        prepare_learner(
            torch.device("cpu"),
            [],
            [],
            {},
            resume_checkpoint=checkpoint,
            warmup_api=object(),
            joint_api=JointApi(),
            task="Pick up the purple ring and place it onto the red peg.",
        )


def test_episode_revision_is_pinned_for_whole_window() -> None:
    registry = FakeRegistry()
    pin = EpisodePin("active", "model", 3)
    with PinnedEpisode(registry, pin) as actual:
        assert actual == pin
        registry.active = "candidate"
        registry.assert_episode_binding("active", "model", 3)
    assert registry.episode is False


def test_inference_waiter_preempts_next_learner_step() -> None:
    coordinator = InferencePriorityCoordinator()
    order: list[str] = []
    first_learner_acquired = threading.Event()
    release_first = threading.Event()

    def first_learner() -> None:
        with coordinator.learner_step_slot():
            first_learner_acquired.set()
            release_first.wait(timeout=2)

    def inference() -> None:
        with coordinator.inference_slot():
            order.append("inference")
            time.sleep(0.01)

    def second_learner() -> None:
        with coordinator.learner_step_slot():
            order.append("learner")

    threads = [threading.Thread(target=first_learner)]
    threads[0].start()
    assert first_learner_acquired.wait(timeout=2)
    threads.extend([threading.Thread(target=inference), threading.Thread(target=second_learner)])
    threads[1].start()
    deadline = time.monotonic() + 2
    while not coordinator.inference_pending and time.monotonic() < deadline:
        time.sleep(0.001)
    assert coordinator.inference_pending
    threads[2].start()
    release_first.set()
    for thread in threads:
        thread.join(timeout=2)
        assert not thread.is_alive()
    assert order == ["inference", "learner"]


def test_h50_cache_enforces_revision_and_low_watermark() -> None:
    cache = H50ActionCache(replan_steps=8, low_watermark=4)
    pin = EpisodePin("r1", "model", 2)
    chunk = np.arange(350, dtype=np.float32).reshape(50, 7)
    cache.adopt(chunk, anchor_ns=1_000_000_000, revision_id="r1", policy_epoch=2)
    for step in range(4):
        action = cache.dispatch(
            timestamp_ns=1_000_000_000 + step * 100_000_000,
            pin=pin,
        )
        assert action is not None and action.shape == (7,)
    assert cache.needs_replan
    assert cache.dispatch(
        timestamp_ns=1_400_000_000,
        pin=EpisodePin("r2", "model2", 2),
    ) is None
    assert cache.stale_count == 1


def test_fake_actor_and_learner_are_concurrently_alive() -> None:
    coordinator = InferencePriorityCoordinator()
    pin = EpisodePin("r1", "model", 0)

    def actor(barrier: threading.Barrier):
        initial = np.zeros((50, 7), dtype=np.float32)
        return run_timed_actor(
            timestamps_ns=[0, 100_000_000, 200_000_000, 300_000_000],
            samples=[0, 1, 2, 3],
            infer=lambda _sample, _request: np.zeros((50, 7), dtype=np.float32),
            coordinator=coordinator,
            pin=pin,
            start_barrier=barrier,
            realtime_scale=0.01,
            initial_chunk=initial,
            initial_latency_ms=1.0,
        )

    def learner(barrier: threading.Barrier):
        with coordinator.worker_alive("learner"):
            barrier.wait()
            with coordinator.learner_step_slot():
                time.sleep(0.002)
        return {"steps": 1}

    actor_result, learner_result = run_concurrent_window(
        actor=actor, learner=learner, coordinator=coordinator,
    )
    assert coordinator.concurrently_alive
    assert actor_result["request_count"] >= 2
    assert actor_result["queue_underrun_count"] == 0
    assert actor_result["stale_action_count"] == 0
    assert learner_result == {"steps": 1}


def test_pin_rejects_wrong_active_revision() -> None:
    registry = FakeRegistry()
    with pytest.raises(AsyncRuntimeError, match="ACTIVE_REVISION_MISMATCH"):
        with PinnedEpisode(registry, EpisodePin("wrong", "model", 3)):
            pass
    assert registry.episode is False


def test_learner_waits_for_action_coverage() -> None:
    coordinator = InferencePriorityCoordinator()
    acquired = threading.Event()
    coordinator.begin_actor_window(0.05)

    def learner() -> None:
        with coordinator.learner_step_slot(
            "micro", initial_estimate_s=0.20, coverage_reserve_s=0.05
        ):
            acquired.set()

    thread = threading.Thread(target=learner)
    thread.start()
    time.sleep(0.02)
    assert not acquired.is_set()
    coordinator.update_action_coverage(0.40)
    assert acquired.wait(timeout=1)
    thread.join(timeout=1)
    coordinator.end_actor_window()
    assert coordinator.slowest_learner_microstep is not None


def test_takeover_rejects_old_result_and_prefills_fresh_generation() -> None:
    coordinator = InferencePriorityCoordinator()
    pin = EpisodePin("r1", "model", 4)
    requests: list[InferenceRequest] = []
    release_old = threading.Event()

    def infer(_sample, request: InferenceRequest):
        requests.append(request)
        if request.takeover_generation == 0:
            release_old.wait(timeout=1)
        return np.full((50, 7), request.request_index, dtype=np.float32)

    def actor(barrier: threading.Barrier):
        return run_timed_actor(
            timestamps_ns=[0, 100_000_000, 200_000_000, 300_000_000],
            samples=[0, 1, 2, 3],
            infer=infer,
            coordinator=coordinator,
            pin=pin,
            start_barrier=barrier,
            realtime_scale=0.1,
            initial_chunk=np.zeros((50, 7), dtype=np.float32),
            initial_latency_ms=1.0,
            takeover_windows=[TakeoverWindow(
                start_ns=50_000_000,
                resume_ns=150_000_000,
                takeover_generation=1,
            )],
        )

    def learner(barrier: threading.Barrier):
        steps = 0
        with coordinator.worker_alive("learner"):
            barrier.wait()
            while steps < 3:
                with coordinator.learner_step_slot(
                    "fake", initial_estimate_s=0.0, coverage_reserve_s=0.0
                ):
                    steps += 1
                    time.sleep(0.01)
                    if steps == 2:
                        release_old.set()
        return {"steps": steps}

    actor_result, learner_result = run_concurrent_window(
        actor=actor, learner=learner, coordinator=coordinator,
    )
    old, fresh = requests[:2]
    assert old.takeover_generation == 0
    assert fresh.takeover_generation == 1
    assert fresh.t_ref_ns == 200_000_000
    assert len({
        old.request_id, old.result_id, old.chunk_id, old.proposal_id,
        fresh.request_id, fresh.result_id, fresh.chunk_id, fresh.proposal_id,
    }) == 8
    assert actor_result["queue_underrun_count"] == 0
    assert actor_result["stale_action_count"] == 0
    assert actor_result["old_result_post_takeover_adopt_count"] == 0
    assert actor_result["old_result_post_takeover_reject_count"] == 1
    assert actor_result["post_takeover_fresh_request_count"] >= 1
    assert actor_result["post_takeover_fresh_chunk_adopt_count"] >= 1
    assert learner_result["steps"] == 3
