#!/usr/bin/env python3
"""Serve the active online-replay Actor and one exact-resume Learner cycle."""

from __future__ import annotations

import argparse
from contextlib import AbstractContextManager
from http.server import ThreadingHTTPServer
import json
from pathlib import Path
import random
import sys
import threading
import time
from typing import Any, Callable, Mapping

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for path in (SRC, ROOT / "tools"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import train_forcerft_critic_warmup as warmup  # noqa: E402
import train_forcerft_actor_critic as joint  # noqa: E402
import serve_policy  # noqa: E402
from forcesmolvla.rft.online.actor_learner_runtime import (  # noqa: E402
    EpisodePin,
    InferencePriorityCoordinator,
    PinnedEpisode,
    prepare_learner,
)
from forcesmolvla.rft.online.policy_revision import load_revision_registry  # noqa: E402


DEFAULT_RESUME_CHECKPOINT = (
    warmup.FORMAL_R_ROOT / "checkpoints/stage3_joint_cycle_000020"
)
DEFAULT_PENDING_CHECKPOINT = (
    warmup.FORMAL_R_ROOT
    / "checkpoints/stage3_real_async_joint_cycle_000021_pending_20260830_001"
)
DEFAULT_PENDING_CANDIDATE_ID = (
    "stage3-online-r-real-async-joint-cycle-000021-candidate"
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _replay_snapshot() -> tuple[tuple[str, int, int], ...]:
    return tuple(sorted(
        (path.name, path.stat().st_size, path.stat().st_mtime_ns)
        for path in (warmup.FORMAL_R_ROOT / "replay").glob("*.json")
    ))


class OneCycleLearner:
    """One restored joint cycle over the replay snapshot loaded at startup."""

    def __init__(
        self,
        *,
        device: torch.device,
        resume_checkpoint: Path,
        pending_checkpoint: Path,
        pending_candidate_id: str,
        current_session_id: str,
    ) -> None:
        require(not pending_checkpoint.exists(), "ONLINE_REPLAY_ASYNC_PENDING_CHECKPOINT_EXISTS")
        self.device = device
        self.resume_checkpoint = resume_checkpoint.resolve()
        self.pending_checkpoint = pending_checkpoint.resolve()
        self.pending_candidate_id = pending_candidate_id
        self.current_session_id = current_session_id
        self.replay_before = _replay_snapshot()
        all_r, r_macros, source_episodes = warmup.load_formal_online_r(
            warmup.FORMAL_R_ROOT
        )
        require(
            not any(
                row["identity"].get("session_id") == current_session_id
                for row in all_r
            ),
            "ONLINE_REPLAY_ASYNC_CURRENT_EPISODE_ALREADY_IN_REPLAY",
        )
        self.unique_r_count = len(all_r)
        self.r_macro_count = len(r_macros)
        self.learner = prepare_learner(
            device,
            all_r,
            r_macros,
            source_episodes,
            resume_checkpoint=self.resume_checkpoint,
            warmup_api=warmup,
            joint_api=joint,
        )

    def __call__(
        self, coordinator: InferencePriorityCoordinator
    ) -> dict[str, Any]:
        learner = self.learner
        previous = learner["runtime"]["counters"]
        critic_offset = int(previous["critic_optimizer_steps"])
        cycle = int(previous["joint_cycles"])
        selected_identities: list[str] = []
        td_losses: list[float] = []
        nonfinite_count = 0
        oom_count = 0

        def learner_slot(kind: str):
            return coordinator.learner_step_slot(kind)

        learner["credits"].consume_joint_cycle()
        for substep in range(2):
            rows = [
                learner["r_replay"].materialize(index)
                for index in learner["critic_r"][substep]
            ]
            rows.extend(
                learner["d_replay"].materialize(index)
                for index in learner["critic_d"][substep]
            )
            selected_identities.extend(str(row["identity"]) for row in rows)
            try:
                with learner_slot("critic_batch_prepare"):
                    batch = warmup.build_batch(
                        rows,
                        learner["actor"],
                        learner["feature"],
                        self.device,
                    )
                record = joint.critic_step(
                    step=critic_offset + substep,
                    actor=learner["actor"],
                    q1=learner["q1"],
                    q2=learner["q2"],
                    q1_target=learner["q1_target"],
                    q2_target=learner["q2_target"],
                    optimizer=learner["critic_optimizer"],
                    batch=batch,
                    flow=learner["flow"],
                    noise_generator=learner["critic_noise"],
                    delta_mean=learner["delta_mean"],
                    delta_std=learner["delta_std"],
                    microbatch_size=4,
                    microbatch_slot=learner_slot,
                )
                del batch
            except torch.cuda.OutOfMemoryError:
                oom_count += 1
                raise
            except FloatingPointError:
                nonfinite_count += 1
                raise
            td_losses.append(float(record["loss"]))

        rows = [
            joint._online_actor_row(learner["r_replay"], index)
            for index in learner["actor_r"][0]
        ]
        rows.extend(
            learner["d_replay"].materialize_actor(index)
            for index in learner["actor_d"][0]
        )
        selected_identities.extend(str(row["identity"]) for row in rows)
        try:
            with learner_slot("actor_batch_prepare"):
                batch = joint.build_actor_training_batch(
                    rows,
                    learner["actor"],
                    learner["feature"],
                    self.device,
                )
            actor_record = joint.actor_step(
                cycle=cycle,
                actor=learner["actor"],
                q1=learner["q1"],
                q2=learner["q2"],
                q1_target=learner["q1_target"],
                q2_target=learner["q2_target"],
                optimizer=learner["actor_optimizer"],
                scheduler=learner["actor_scheduler"],
                batch=batch,
                flow=learner["flow"],
                delta_mean=learner["delta_mean"],
                delta_std=learner["delta_std"],
                config=learner["config"],
                microbatch_slot=learner_slot,
            )
            del batch
        except torch.cuda.OutOfMemoryError:
            oom_count += 1
            raise
        except FloatingPointError:
            nonfinite_count += 1
            raise

        current_episode_sampled = any(
            self.current_session_id in identity
            for identity in selected_identities
        )
        replay_after = _replay_snapshot()
        require(
            not current_episode_sampled
            and replay_after == self.replay_before
            and nonfinite_count == 0
            and oom_count == 0,
            "ONLINE_REPLAY_ASYNC_LEARNER_COMPLETION_CONTRACT",
        )
        counters = {
            "joint_cycles": cycle + 1,
            "critic_optimizer_steps": critic_offset + 2,
            "actor_optimizer_steps": int(previous["actor_optimizer_steps"]) + 1,
            "target_polyak_steps": int(previous["target_polyak_steps"]) + 2,
        }
        runtime_state = {
            "source_checkpoint": str(self.resume_checkpoint),
            "flags": {"critic_ready": True, "actor_q_guidance_enabled": True},
            "counters": counters,
            "replay": {
                "formal_r_root": str(warmup.FORMAL_R_ROOT),
                "unique_r_transition_count": self.unique_r_count,
                "new_r_transition_count": learner["new_r_transition_count"],
                "eligible_ack_macro_count": self.r_macro_count,
                "mix": {"R": 32, "D": 32},
                "current_episode_sampled": False,
            },
            "sample_credit": learner["credits"].state_dict(),
            "sampler_state": {
                "cycle": cycle + 1,
                "r_rng": learner["r_rng"].getstate(),
                "d_rng": learner["d_rng"].getstate(),
            },
            "rng_state": {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch_cpu": torch.get_rng_state(),
                "torch_cuda": torch.cuda.get_rng_state_all(),
                "critic_noise_generator": learner["critic_noise"].get_state().cpu(),
            },
            "optimizer_ownership": {
                "overlap": 0,
                "frozen_vlm_or_state_prefix_in_actor_optimizer": 0,
                "critic_optimizer_restored_from_joint_checkpoint": True,
                "actor_optimizer_restored_from_joint_checkpoint": True,
            },
            "candidate_policy_revision": {
                "revision_id": self.pending_candidate_id,
                "state": "candidate",
                "activated": False,
                "published": False,
                "coordinator_disposition": "pending_episode_boundary",
            },
            "step_metrics": {
                "critic_td_loss": td_losses,
                "actor_fm_loss": [float(actor_record["fm_loss"])],
                "actor_min_twin_q_loss": [float(actor_record["actor_q_loss"])],
            },
        }
        joint.save_joint_checkpoint(
            self.pending_checkpoint,
            actor=learner["actor"],
            modules=learner["modules"],
            critic_optimizer=learner["critic_optimizer"],
            actor_optimizer=learner["actor_optimizer"],
            actor_scheduler=learner["actor_scheduler"],
            runtime_state=runtime_state,
            parent_binding=learner["binding"],
            source_checkpoint=self.resume_checkpoint,
            total_joint_cycles=cycle + 1,
            candidate_revision_id=self.pending_candidate_id,
        )
        candidate = json.loads(
            (self.pending_checkpoint / "candidate_policy/candidate.json").read_text(
                encoding="utf-8"
            )
        )
        require(
            candidate.get("revision_id") == self.pending_candidate_id
            and candidate.get("state") == "candidate"
            and candidate.get("published") is False
            and candidate.get("activated") is False,
            "ONLINE_REPLAY_ASYNC_PENDING_CANDIDATE_STATE_INVALID",
        )
        return {
            "learner_critic_steps": 2,
            "learner_actor_steps": 1,
            "learner_polyak_steps": 2,
            "current_episode_sampled": False,
            "nonfinite_count": nonfinite_count,
            "oom_count": oom_count,
            "pending_checkpoint_path": str(self.pending_checkpoint),
            "pending_candidate_id": self.pending_candidate_id,
            "pending_candidate_published": False,
            "pending_candidate_activated": False,
        }


class AsyncPolicyLearnerRuntime:
    """One episode pin around HTTP inference and one background Learner cycle."""

    def __init__(
        self,
        *,
        engine: Any,
        machine: Any,
        session_id: str,
        episode_id: str,
        active_revision_id: str,
        active_model_revision: str,
        learner_resume_checkpoint: Path,
        pending_checkpoint: Path,
        pending_candidate_id: str,
        learner_job: Callable[[InferencePriorityCoordinator], Mapping[str, Any]],
        inference_stream: Any = None,
    ) -> None:
        self.engine = engine
        self.machine = machine
        self.session_id = session_id
        self.episode_id = episode_id
        self.active_revision_id = active_revision_id
        self.active_model_revision = active_model_revision
        self.learner_resume_checkpoint = learner_resume_checkpoint.resolve()
        self.pending_checkpoint = pending_checkpoint.resolve()
        self.pending_candidate_id = pending_candidate_id
        self.learner_job = learner_job
        self.inference_stream = inference_stream
        self.coordinator = InferencePriorityCoordinator()
        self._lock = threading.Lock()
        self._episode_active = False
        self._learner_started = False
        self._learner_state = "ready"
        self._learner_result: dict[str, Any] = {}
        self._learner_error: str | None = None
        self._inference_request_count = 0
        self._actor_alive: AbstractContextManager[Any] | None = None
        self._pin: PinnedEpisode | None = None

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            **self.engine.metadata,
            "online_actor_learner": True,
            "runtime_session_id": self.session_id,
            "runtime_episode_id": self.episode_id,
            "learner_started": self._learner_started,
            "learner_resume_checkpoint": str(self.learner_resume_checkpoint),
            "active_actor_revision": self.active_revision_id,
            "active_actor_model_revision": self.active_model_revision,
            "pending_checkpoint_path": str(self.pending_checkpoint),
            "pending_candidate_id": self.pending_candidate_id,
            "pending_candidate_published": False,
            "pending_candidate_activated": False,
        }

    def status(self) -> dict[str, Any]:
        with self._lock:
            result = dict(self._learner_result)
            return {
                "online_actor_learner": True,
                "runtime_session_id": self.session_id,
                "runtime_episode_id": self.episode_id,
                "episode_active": self._episode_active,
                "actor_revision_pinned": self._episode_active,
                "active_actor_revision": self.active_revision_id,
                "active_actor_model_revision": self.active_model_revision,
                "learner_started": self._learner_started,
                "learner_state": self._learner_state,
                "learner_resume_checkpoint": str(self.learner_resume_checkpoint),
                "learner_critic_steps": int(result.get("learner_critic_steps", 0)),
                "learner_actor_steps": int(result.get("learner_actor_steps", 0)),
                "learner_polyak_steps": int(result.get("learner_polyak_steps", 0)),
                "current_episode_sampled": bool(
                    result.get("current_episode_sampled", False)
                ),
                "pending_checkpoint_path": str(self.pending_checkpoint),
                "pending_candidate_id": self.pending_candidate_id,
                "pending_candidate_published": False,
                "pending_candidate_activated": False,
                "inference_request_count": self._inference_request_count,
                "actor_and_learner_concurrently_alive": (
                    self.coordinator.concurrently_alive
                ),
                "nonfinite_count": int(result.get("nonfinite_count", 0)),
                "oom_count": int(result.get("oom_count", 0)),
                "learner_error": self._learner_error,
            }

    def start_episode(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        self._validate_identity(payload)
        with self._lock:
            require(not self._episode_active, "ONLINE_REPLAY_ASYNC_EPISODE_ALREADY_ACTIVE")
            require(not self._learner_started, "ONLINE_REPLAY_ASYNC_SERVER_SINGLE_EPISODE_ONLY")
            self._pin = PinnedEpisode(
                self.machine,
                EpisodePin(
                    self.active_revision_id,
                    self.active_model_revision,
                    int(self.machine.policy_epoch),
                ),
            )
            self._pin.__enter__()
            self._actor_alive = self.coordinator.worker_alive("actor")
            self._actor_alive.__enter__()
            self.coordinator.begin_actor_window(0.8)
            self._episode_active = True
        return self.status()

    def _validate_identity(self, payload: Mapping[str, Any]) -> None:
        require(
            payload.get("session_id") == self.session_id
            and payload.get("episode_id") == self.episode_id
            and payload.get("policy_revision") == self.active_model_revision,
            "ONLINE_REPLAY_ASYNC_CAPTURE_IDENTITY_MISMATCH",
        )

    def _start_learner(self) -> None:
        with self._lock:
            if self._learner_started:
                return
            self._learner_started = True
            self._learner_state = "running"

        def run() -> None:
            try:
                with self.coordinator.worker_alive("learner"):
                    result = dict(self.learner_job(self.coordinator))
                with self._lock:
                    self._learner_result = result
                    self._learner_state = "complete"
            except Exception as error:
                with self._lock:
                    self._learner_error = f"{type(error).__name__}:{error}"
                    self._learner_state = "failed"

        threading.Thread(
            target=run, name="online-actor-learner", daemon=True
        ).start()

    def infer(self, request: dict[str, Any]) -> dict[str, Any]:
        provenance = request.get("provenance", {})
        require(self._episode_active, "ONLINE_REPLAY_ASYNC_EPISODE_INACTIVE")
        require(
            provenance.get("session_id") == self.session_id,
            "ONLINE_REPLAY_ASYNC_INFERENCE_SESSION_MISMATCH",
        )
        with self.coordinator.inference_slot():
            self._start_learner()
            if self.inference_stream is None:
                result = self.engine.infer(request)
            else:
                with torch.cuda.stream(self.inference_stream):
                    result = self.engine.infer(request)
                self.inference_stream.synchronize()
        with self._lock:
            self._inference_request_count += 1
        self.coordinator.update_action_coverage(0.8)
        return result

    def end_episode(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        self._validate_identity(payload)
        self._end_episode()
        return self.status()

    def abort_episode(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        self._validate_identity(payload)
        self._end_episode()
        return self.status()

    def _end_episode(self) -> None:
        with self._lock:
            require(self._episode_active, "ONLINE_REPLAY_ASYNC_EPISODE_INACTIVE")
            self.coordinator.end_actor_window()
            assert self._actor_alive is not None and self._pin is not None
            self._actor_alive.__exit__(None, None, None)
            self._pin.__exit__(None, None, None)
            self._actor_alive = None
            self._pin = None
            self._episode_active = False


class RequestHandler(serve_policy.RequestHandler):
    @property
    def runtime(self) -> AsyncPolicyLearnerRuntime:
        return self.server.engine  # type: ignore[attr-defined]

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/runtime/status":
            self._write_json(200, self.runtime.status())
            return
        super().do_GET()

    def do_POST(self) -> None:  # noqa: N802
        if self.path not in {
            "/runtime/episode-start",
            "/runtime/episode-end",
            "/runtime/episode-abort",
        }:
            super().do_POST()
            return
        try:
            length = int(self.headers.get("Content-Length", "-1"))
            if length <= 0 or length > serve_policy.MAX_REQUEST_BYTES:
                raise ValueError("ONLINE_REPLAY_ASYNC_RUNTIME_REQUEST_SIZE_INVALID")
            payload = json.loads(self.rfile.read(length))
            if not isinstance(payload, dict):
                raise ValueError("ONLINE_REPLAY_ASYNC_RUNTIME_REQUEST_MUST_BE_OBJECT")
            method = {
                "/runtime/episode-start": self.runtime.start_episode,
                "/runtime/episode-end": self.runtime.end_episode,
                "/runtime/episode-abort": self.runtime.abort_episode,
            }[self.path]
            self._write_json(200, method(payload))
        except Exception as error:
            self._write_json(422, {
                "error": type(error).__name__, "detail": str(error),
            })


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deployment-profile", type=Path, required=True)
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--episode-id", required=True)
    parser.add_argument("--learner-resume-checkpoint", type=Path, default=DEFAULT_RESUME_CHECKPOINT)
    parser.add_argument("--pending-checkpoint", type=Path, default=DEFAULT_PENDING_CHECKPOINT)
    parser.add_argument("--pending-candidate-id", default=DEFAULT_PENDING_CANDIDATE_ID)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--allow-development-robot-execution", action="store_true")
    parser.add_argument("--deployment-binding", type=Path)
    parser.add_argument("--trusted-deployment-binding-sha256")
    args = parser.parse_args(argv)
    if args.host not in {"127.0.0.1", "localhost"} or args.port <= 0:
        parser.error("only a valid loopback endpoint is allowed")
    if not args.allow_development_robot_execution:
        parser.error("--allow-development-robot-execution is required")
    return args


def build_runtime(args: argparse.Namespace) -> AsyncPolicyLearnerRuntime:
    require(torch.cuda.is_available(), "ONLINE_REPLAY_ASYNC_CUDA_UNAVAILABLE")
    profile = serve_policy.load_deployment_profile(args.deployment_profile, ROOT)
    binding_path = (
        Path(profile["deployment_binding"])
        if args.deployment_binding is None
        else args.deployment_binding.resolve()
    )
    trusted = (
        str(profile["deployment_binding_sha256"])
        if args.trusted_deployment_binding_sha256 is None
        else args.trusted_deployment_binding_sha256
    )
    device = torch.device("cuda:0")
    engine = serve_policy.InferenceEngine(
        Path(profile["checkpoint"]),
        Path(profile["rulespec"]),
        ROOT / "schemas/rulespec.schema.json",
        device,
        allow_development_robot_execution=True,
        deployment_binding_path=binding_path,
        trusted_deployment_binding_sha256=trusted,
    )
    machine = load_revision_registry(async_runner.REVISION_REGISTRY, fresh_process=False)
    active = machine.record(machine.active_revision_id)
    deployed = json.loads(
        (Path(profile["checkpoint"]) / "candidate.json").read_text(
            encoding="utf-8"
        )
    )
    require(
        deployed.get("state") == "published"
        and deployed.get("published") is True
        and active.revision_id == deployed.get("revision_id")
        and active.model_sha256 == engine.model_sha256
        and active.model_sha256 == deployed.get("model_revision"),
        "ONLINE_REPLAY_ASYNC_ACTIVE_DEPLOYMENT_MISMATCH",
    )
    learner = OneCycleLearner(
        device=device,
        resume_checkpoint=args.learner_resume_checkpoint,
        pending_checkpoint=args.pending_checkpoint,
        pending_candidate_id=args.pending_candidate_id,
        current_session_id=args.session_id,
    )
    return AsyncPolicyLearnerRuntime(
        engine=engine,
        machine=machine,
        session_id=args.session_id,
        episode_id=args.episode_id,
        active_revision_id=active.revision_id,
        active_model_revision=active.model_sha256,
        learner_resume_checkpoint=args.learner_resume_checkpoint,
        pending_checkpoint=args.pending_checkpoint,
        pending_candidate_id=args.pending_candidate_id,
        learner_job=learner,
        inference_stream=torch.cuda.Stream(device=device, priority=-1),
    )


def main() -> int:
    args = parse_args()
    runtime = build_runtime(args)
    server = ThreadingHTTPServer((args.host, args.port), RequestHandler)
    server.engine = runtime  # type: ignore[attr-defined]
    print(
        f"[model] active Actor revision={runtime.active_revision_id} "
        f"model={runtime.active_model_revision}",
        flush=True,
    )
    print(
        f"[learner] exact-resume={runtime.learner_resume_checkpoint} "
        f"pending={runtime.pending_checkpoint}",
        flush=True,
    )
    print(
        f"[server] listening on http://{args.host}:{args.port} "
        "robot_io=false execution=approved_binding_supervised_development",
        flush=True,
    )
    try:
        server.serve_forever(poll_interval=0.2)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
