#!/usr/bin/env python3
"""Serve one persistent ForceRFT Actor/Learner process across episodes."""

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
from typing import Any, Mapping

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for path in (SRC, ROOT / "tools"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from forcesmolvla.rft.online import replay_training as warmup  # noqa: E402
import train_forcerft_actor_critic as joint  # noqa: E402
import serve_policy  # noqa: E402
from forcesmolvla.rft.online.actor_learner_runtime import (  # noqa: E402
    OnlineTrainingPolicy,
    broadcast_actor_parameters,
    exact_resume_checkpoint_is_recoverable,
    online_checkpoint_path,
    retain_latest_online_checkpoints,
    EpisodePin,
    InferencePriorityCoordinator,
    PinnedEpisode,
    prepare_learner,
    reconcile_post_checkpoint_replay,
    select_resume_or_seed_checkpoint,
)
from forcesmolvla.rft.online.actor_unlock import (  # noqa: E402
    ActorUnlockPolicy,
    actor_unlock_is_approved,
)
from forcesmolvla.rft.online.policy_revision import (  # noqa: E402
    InMemoryRevisionStateMachine,
    RevisionRecord,
    RevisionState,
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _session_was_sampled(
    session_id: str | None, selected_identities: list[str]
) -> bool:
    return session_id is not None and any(
        session_id in identity for identity in selected_identities
    )


def _validate_cycle_completion(
    *, current_episode_sampled: bool, nonfinite_count: int, oom_count: int
) -> None:
    require(
        not current_episode_sampled
        and nonfinite_count == 0
        and oom_count == 0,
        "ONLINE_REPLAY_ASYNC_LEARNER_COMPLETION_CONTRACT",
    )


def _refresh_training_schedules(learner: dict[str, Any]) -> None:
    policy = learner.get("training_policy", OnlineTrainingPolicy())
    schedules = joint.make_schedules(
        learner["r_rng"],
        learner["d_rng"],
        r_population_size=len(learner["r_replay"].macros),
        d_population=learner["d_replay"].population,
        fm_population=learner["d_replay"].fm_population,
        cycles=1,
        critic_updates_per_cycle=policy.critic_updates_per_cycle,
        actor_updates_per_cycle=policy.actor_updates_per_cycle,
        demo_ratio=policy.demo_ratio,
        online_ratio=policy.online_ratio,
    )
    (
        learner["critic_r"],
        learner["critic_d"],
        learner["actor_r"],
        learner["actor_d"],
    ) = schedules
    learner["d_replay"].prefetch_joint(
        learner["critic_d"], learner["actor_d"]
    )


class ContinuousLearner:
    """Persistent Learner over sealed replay, independent of episode boundaries."""

    def __init__(
        self,
        *,
        device: torch.device,
        resume_checkpoint: Path,
        checkpoint_root: Path,
        replay_root: Path,
        current_session_id: str | None,
        task: str,
        sft_reference_checkpoint: Path | None = None,
        actor_unlock_approval: Path | None = None,
        training_policy: OnlineTrainingPolicy | None = None,
    ) -> None:
        self.device = device
        self.resume_checkpoint = resume_checkpoint.resolve()
        self.checkpoint_root = checkpoint_root.resolve()
        self.checkpoint_root.mkdir(parents=True, exist_ok=True)
        self.replay_root = replay_root.resolve()
        self.current_session_id = current_session_id
        self.task = task
        self.sft_reference_checkpoint = (
            None
            if sft_reference_checkpoint is None
            else sft_reference_checkpoint.resolve()
        )
        self.actor_unlock_approval = (
            self.checkpoint_root.parent / "approvals/actor_q_unlock.json"
            if actor_unlock_approval is None
            else actor_unlock_approval.resolve()
        )
        self.training_policy = training_policy or OnlineTrainingPolicy()
        self.learner: dict[str, Any] | None = None
        self.unique_r_count = 0
        self.r_macro_count = 0

    def set_current_session(self, session_id: str) -> None:
        self.current_session_id = session_id

    def clear_current_session(self) -> None:
        self.current_session_id = None

    def _ensure_learner(self) -> bool:
        if self.learner is not None:
            return True
        if (
            warmup.count_sealed_autonomous_policy_transitions(
                self.replay_root
            )
            < self.training_policy.training_starts
        ):
            return False
        all_r, r_macros, source_episodes, human_rows = warmup.load_formal_online_r(
            self.replay_root
        )
        if self.current_session_id is not None:
            require(
                not any(
                    row["identity"].get("session_id") == self.current_session_id
                    for row in [*all_r, *human_rows]
                ),
                "ONLINE_REPLAY_ASYNC_CURRENT_EPISODE_ALREADY_IN_REPLAY",
            )
        self.unique_r_count = len(all_r)
        self.r_macro_count = len(r_macros)
        self.learner = prepare_learner(
            self.device,
            all_r,
            r_macros,
            source_episodes,
            human_rows,
            resume_checkpoint=self.resume_checkpoint,
            warmup_api=warmup,
            joint_api=joint,
            task=self.task,
            sft_reference_checkpoint=self.sft_reference_checkpoint,
        )
        return True

    def __call__(
        self, coordinator: InferencePriorityCoordinator
    ) -> dict[str, Any]:
        current_session_id = self.current_session_id
        if not self._ensure_learner():
            return {
                "waiting_for_replay": True,
                "learner_critic_steps": 0,
                "learner_actor_steps": 0,
                "learner_polyak_steps": 0,
                "current_episode_sampled": False,
                "nonfinite_count": 0,
                "oom_count": 0,
            }
        assert self.learner is not None
        learner = self.learner
        all_r, r_macros, source_episodes, human_rows = warmup.load_formal_online_r(
            self.replay_root
        )
        if current_session_id is not None:
            require(
                not any(
                    row["identity"].get("session_id") == current_session_id
                    for row in [*all_r, *human_rows]
                ),
                "ONLINE_REPLAY_ASYNC_CURRENT_EPISODE_ALREADY_IN_REPLAY",
            )
        reconcile_post_checkpoint_replay(learner["credits"], all_r)
        if not learner["credits"].can_consume_joint_cycle():
            return {
                "waiting_for_replay": True,
                "learner_critic_steps": 0,
                "learner_actor_steps": 0,
                "learner_polyak_steps": 0,
                "current_episode_sampled": False,
                "nonfinite_count": 0,
                "oom_count": 0,
            }
        if len(r_macros) != len(learner["r_replay"].macros):
            learner["r_replay"] = warmup.FormalReplay(
                r_macros, source_episodes, learner["normalizer"]
            )
            learner["actor_q_valid_ack_rows"] = sum(
                int(macro.actor_q_eligibility.valid) for macro in r_macros
            )
        if len(human_rows) != len(learner["d_replay"].human_replay.rows):
            learner["d_replay"].set_human_rows(
                human_rows, source_episodes
            )
        self.unique_r_count = len(all_r)
        self.r_macro_count = len(r_macros)
        previous = learner["runtime"]["counters"]
        critic_offset = int(previous["critic_optimizer_steps"])
        base_cycle = int(previous["joint_cycles"])
        online_cycle = int(learner.get("online_joint_cycles", 0))
        selected_identities: list[str] = []
        td_losses: list[float] = []
        nonfinite_count = 0
        oom_count = 0

        def learner_slot(kind: str):
            return coordinator.learner_step_slot(kind)

        learner["credits"].consume_joint_cycle()
        for substep in range(self.training_policy.critic_updates_per_cycle):
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
            learner["critic_scheduler"].step()

        if not learner["actor_updates_enabled"]:
            learner["critic_only_updates"] += (
                self.training_policy.critic_updates_per_cycle
            )
            learner["actor_updates_enabled"] = actor_unlock_is_approved(
                self.actor_unlock_approval,
                actor_q_valid_ack_rows=learner["actor_q_valid_ack_rows"],
                critic_only_updates=learner["critic_only_updates"],
                policy=ActorUnlockPolicy(
                    minimum_actor_q_valid_ack_rows=int(
                        learner["config"]["actor_unlock"][
                            "minimum_actor_q_valid_ack_rows"
                        ]
                    ),
                    minimum_critic_only_updates=int(
                        learner["config"]["actor_unlock"][
                            "minimum_critic_only_updates"
                        ]
                    ),
                ),
            )

        actor_records: list[dict[str, Any]] = []
        if learner["actor_updates_enabled"]:
            for actor_substep, (r_indices, d_indices) in enumerate(
                zip(learner["actor_r"], learner["actor_d"], strict=True)
            ):
                rows = [
                    joint._online_actor_row(learner["r_replay"], index)
                    for index in r_indices
                ]
                rows.extend(
                    learner["d_replay"].materialize_actor(index)
                    for index in d_indices
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
                    actor_records.append(
                        joint.actor_step(
                            cycle=(
                                int(previous["actor_optimizer_steps"])
                                + actor_substep
                            ),
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
                            reference_actor=learner["reference_actor"],
                            q_gradient_controller=learner[
                                "q_gradient_controller"
                            ],
                        )
                    )
                    del batch
                except torch.cuda.OutOfMemoryError:
                    oom_count += 1
                    raise
                except FloatingPointError:
                    nonfinite_count += 1
                    raise

        current_episode_sampled = _session_was_sampled(
            current_session_id, selected_identities
        )
        # This cycle owns its in-memory replay snapshot. Append-only Online-R
        # growth becomes visible on the next cycle, as in ConRFT.
        _validate_cycle_completion(
            current_episode_sampled=current_episode_sampled,
            nonfinite_count=nonfinite_count,
            oom_count=oom_count,
        )
        counters = {
            "joint_cycles": base_cycle + 1,
            "critic_optimizer_steps": (
                critic_offset + self.training_policy.critic_updates_per_cycle
            ),
            "actor_optimizer_steps": int(previous["actor_optimizer_steps"])
            + len(actor_records),
            "target_polyak_steps": int(previous["target_polyak_steps"])
            + self.training_policy.target_polyak_updates_per_cycle,
        }
        runtime_state = {
            "online_joint_cycles": online_cycle + 1,
            "source_checkpoint": str(self.resume_checkpoint),
            "reference_actor_checkpoint": str(
                learner["reference_actor_checkpoint"]
            ),
            "q_gradient_controller": learner[
                "q_gradient_controller"
            ].state_dict(),
            "critic_only_updates": learner["critic_only_updates"],
            "flags": {
                "critic_ready": True,
                "critic_updates_enabled": True,
                "actor_updates_enabled": learner["actor_updates_enabled"],
                "actor_q_guidance_enabled": learner["actor_updates_enabled"],
            },
            "counters": counters,
            "replay": {
                "formal_r_root": str(self.replay_root),
                "unique_r_transition_count": self.unique_r_count,
                "new_r_transition_count": learner["new_r_transition_count"],
                "eligible_ack_macro_count": self.r_macro_count,
                "actor_q_valid_ack_rows": learner["actor_q_valid_ack_rows"],
                "mix": {"R": 32, "D": 32},
                "current_episode_sampled": False,
            },
            "sample_credit": learner["credits"].state_dict(),
            "sampler_state": {
                "cycle": online_cycle + 1,
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
            "runtime_artifacts": learner["runtime"].get("runtime_artifacts", {}),
            "step_metrics": {
                "critic_td_loss": td_losses,
                "actor_fm_loss": (
                    [float(record["fm_loss"]) for record in actor_records]
                ),
                "actor_min_twin_q_loss": (
                    [float(record["actor_q_loss"]) for record in actor_records]
                ),
            },
        }
        learner["runtime"] = runtime_state
        learner["online_joint_cycles"] = online_cycle + 1
        _refresh_training_schedules(learner)
        latest_checkpoint = None
        if self.training_policy.checkpoint_due(online_cycle + 1):
            latest_checkpoint = self.save_checkpoint()
        return {
            "learner_critic_steps": self.training_policy.critic_updates_per_cycle,
            "learner_actor_steps": len(actor_records),
            "learner_polyak_steps": self.training_policy.target_polyak_updates_per_cycle,
            "current_episode_sampled": False,
            "nonfinite_count": nonfinite_count,
            "oom_count": oom_count,
            "online_joint_cycle": online_cycle + 1,
            "learner_actor": learner["actor"],
            "actor_updates_enabled": learner["actor_updates_enabled"],
            "critic_only_updates": learner["critic_only_updates"],
            "latest_checkpoint_path": (
                None if latest_checkpoint is None else str(latest_checkpoint)
            ),
        }

    def save_checkpoint(self) -> Path | None:
        if self.learner is None:
            return None
        learner = self.learner
        completed = int(learner.get("online_joint_cycles", 0))
        if completed <= 0:
            return None
        target = online_checkpoint_path(self.checkpoint_root, completed)
        if not target.exists():
            joint.save_joint_checkpoint(
                target,
                actor=learner["actor"], modules=learner["modules"],
                critic_optimizer=learner["critic_optimizer"],
                actor_optimizer=learner["actor_optimizer"],
                actor_scheduler=learner["actor_scheduler"],
                critic_scheduler=learner["critic_scheduler"],
                runtime_state=learner["runtime"], parent_binding=None,
                actor_parent_path=self.resume_checkpoint / "actor",
                parent_binding_id=(
                    f"{learner['config']['task']['task_id']}"
                    "-online-actor-critic-exact-resume"
                ),
                source_checkpoint=self.resume_checkpoint,
                total_joint_cycles=int(learner["runtime"]["counters"]["joint_cycles"]),
                actor_checkpoint_id=f"online-actor-critic-cycle-{completed:06d}",
                checkpoint_kind="online_actor_critic_exact_resume",
                actor_directory="actor",
                metadata_overrides={
                    "critic_ready": bool(
                        learner["runtime"]["flags"]["actor_updates_enabled"]
                    ),
                    "critic_updates_enabled": True,
                    "actor_updates_enabled": bool(
                        learner["runtime"]["flags"]["actor_updates_enabled"]
                    ),
                    "actor_q_guidance_enabled": bool(
                        learner["runtime"]["flags"]["actor_updates_enabled"]
                    ),
                },
            )
        retain_latest_online_checkpoints(
            self.checkpoint_root, keep=self.training_policy.keep_latest_checkpoints
        )
        return target


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
        online_checkpoint_root: Path,
        learner_job: ContinuousLearner,
        inference_stream: Any = None,
    ) -> None:
        self.engine = engine
        self.machine = machine
        self.session_id = session_id
        self.episode_id = episode_id
        self.active_revision_id = active_revision_id
        self.active_model_revision = active_model_revision
        self.learner_resume_checkpoint = learner_resume_checkpoint.resolve()
        self.online_checkpoint_root = online_checkpoint_root.resolve()
        self.learner_job = learner_job
        self.inference_stream = inference_stream
        self.coordinator = InferencePriorityCoordinator()
        self._lock = threading.Lock()
        self._episode_active = False
        self._learner_started = False
        self._learner_state = "ready"
        self._learner_result: dict[str, Any] = {}
        self._learner_error: str | None = None
        self._stop_learner = threading.Event()
        self._broadcast_count = 0
        self._active_actor_online_cycle = (
            int(self.learner_resume_checkpoint.name.rsplit("_", 1)[1])
            if self.learner_resume_checkpoint.name.startswith(
                "online_actor_critic_cycle_"
            )
            else 0
        )
        self._policy = getattr(
            learner_job, "training_policy", OnlineTrainingPolicy()
        )
        self._learner_thread: threading.Thread | None = None
        self._inference_request_count = 0
        self._actor_alive: AbstractContextManager[Any] | None = None
        self._pin: PinnedEpisode | None = None

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            **self.engine.metadata,
            "online_actor_learner": True,
            "server_persistent": True,
            "current_episode_sampling": False,
            "training_starts": 100,
            "actor_parameter_broadcast_period": 5,
            "active_actor_online_cycle": self._active_actor_online_cycle,
            "checkpoint_period": 50,
            "keep_latest_checkpoints": 2,
            "save_checkpoint_on_graceful_exit": False,
            "save_checkpoint_on_operator_q": True,
            "runtime_session_id": self.session_id,
            "runtime_episode_id": self.episode_id,
            "learner_started": self._learner_started,
            "learner_resume_checkpoint": str(self.learner_resume_checkpoint),
            "active_actor_revision": self.active_revision_id,
            "active_actor_model_revision": self.active_model_revision,
            "policy_epoch": int(self.machine.policy_epoch),
            "online_checkpoint_root": str(self.online_checkpoint_root),
        }

    def status(self) -> dict[str, Any]:
        with self._lock:
            result = dict(self._learner_result)
            return {
                "online_actor_learner": True,
                "server_persistent": True,
                "current_episode_sampling": False,
                "runtime_session_id": self.session_id,
                "runtime_episode_id": self.episode_id,
                "episode_active": self._episode_active,
                "actor_revision_pinned": self._episode_active,
                "active_actor_revision": self.active_revision_id,
                "active_actor_model_revision": self.active_model_revision,
                "policy_epoch": int(self.machine.policy_epoch),
                "learner_started": self._learner_started,
                "learner_state": self._learner_state,
                "learner_resume_checkpoint": str(self.learner_resume_checkpoint),
                "learner_critic_steps": int(result.get("learner_critic_steps", 0)),
                "learner_actor_steps": int(result.get("learner_actor_steps", 0)),
                "learner_polyak_steps": int(result.get("learner_polyak_steps", 0)),
                "current_episode_sampled": bool(
                    result.get("current_episode_sampled", False)
                ),
                "online_checkpoint_root": str(self.online_checkpoint_root),
                "inference_request_count": self._inference_request_count,
                "actor_and_learner_concurrently_alive": (
                    self.coordinator.concurrently_alive
                ),
                "nonfinite_count": int(result.get("nonfinite_count", 0)),
                "oom_count": int(result.get("oom_count", 0)),
                "learner_error": self._learner_error,
                "latest_checkpoint_path": (
                    result.get("latest_checkpoint_path")
                ),
                "actor_parameter_broadcast_count": self._broadcast_count,
                "active_actor_online_cycle": self._active_actor_online_cycle,
            }

    def start_episode(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        self._validate_identity(payload)
        with self._lock:
            require(not self._episode_active, "ONLINE_REPLAY_ASYNC_EPISODE_ALREADY_ACTIVE")
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

    def prepare_episode(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        require(not self._episode_active, "ONLINE_REPLAY_ASYNC_EPISODE_ALREADY_ACTIVE")
        require(
            payload.get("policy_revision") == self.active_model_revision,
            "ONLINE_REPLAY_ASYNC_CAPTURE_IDENTITY_MISMATCH",
        )
        session_id = str(payload.get("session_id", ""))
        episode_id = str(payload.get("episode_id", ""))
        require(bool(session_id and episode_id), "ONLINE_REPLAY_ASYNC_CAPTURE_IDENTITY_MISMATCH")
        with self._lock:
            self.session_id = session_id
            self.episode_id = episode_id
            self.learner_job.set_current_session(session_id)
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
                    while not self._stop_learner.is_set():
                        result = dict(self.learner_job(self.coordinator))
                        if result.get("waiting_for_replay"):
                            with self._lock:
                                self._learner_state = "waiting_for_replay"
                            self._stop_learner.wait(0.25)
                            continue
                        cycle = int(result["online_joint_cycle"])
                        if (
                            int(result.get("learner_actor_steps", 0)) > 0
                            and self._policy.broadcast_due(cycle)
                        ):
                            with self.engine._lock:
                                broadcast_actor_parameters(
                                    result["learner_actor"], self.engine.policy
                                )
                            with self._lock:
                                self._broadcast_count += 1
                                self._active_actor_online_cycle = cycle
                                broadcast_count = self._broadcast_count
                            print(
                                "[model] deployed online Actor "
                                f"cycle={cycle:06d} broadcast={broadcast_count}",
                                flush=True,
                            )
                        result.pop("learner_actor", None)
                        with self._lock:
                            self._learner_result = result
                            self._learner_state = "running"
            except Exception as error:
                with self._lock:
                    self._learner_error = f"{type(error).__name__}:{error}"
                    self._learner_state = "failed"

        self._learner_thread = threading.Thread(
            target=run, name="online-actor-learner", daemon=True
        )
        self._learner_thread.start()

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

    def checkpoint_on_operator_q(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        self._validate_identity(payload)
        require(not self._episode_active, "ONLINE_REPLAY_ASYNC_EPISODE_ACTIVE")
        self.stop()
        with self._lock:
            require(self._learner_state != "failed", "ONLINE_REPLAY_ASYNC_LEARNER_FAILED")
        checkpoint = self.learner_job.save_checkpoint()
        return {
            **self.status(),
            "operator_q_checkpoint_path": (
                None if checkpoint is None else str(checkpoint)
            ),
        }

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
            self.learner_job.clear_current_session()

    def stop(self) -> None:
        self._stop_learner.set()
        if self._learner_thread is not None:
            self._learner_thread.join()


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
            "/runtime/prepare-episode",
            "/runtime/episode-start",
            "/runtime/episode-end",
            "/runtime/episode-abort",
            "/runtime/operator-q-checkpoint",
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
                "/runtime/prepare-episode": self.runtime.prepare_episode,
                "/runtime/episode-start": self.runtime.start_episode,
                "/runtime/episode-end": self.runtime.end_episode,
                "/runtime/episode-abort": self.runtime.abort_episode,
                "/runtime/operator-q-checkpoint": (
                    self.runtime.checkpoint_on_operator_q
                ),
            }[self.path]
            self._write_json(200, method(payload))
        except Exception as error:
            self._write_json(422, {
                "error": type(error).__name__, "detail": str(error),
            })


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-id", default="task2")
    parser.add_argument("--task", required=True)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--reward-transition-root", type=Path)
    parser.add_argument("--safety-config", type=Path)
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--episode-id", required=True)
    parser.add_argument("--learner-resume-checkpoint", type=Path)
    parser.add_argument("--stage3-seed-bundle", type=Path)
    parser.add_argument("--sft-reference-checkpoint", type=Path)
    parser.add_argument("--allow-legacy-offline-fallback", action="store_true")
    parser.add_argument(
        "--allow-development-policy-execution-smoke",
        action="store_true",
        help="explicitly enable the existing supervised HIL robot-execution path",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args(argv)
    if args.host not in {"127.0.0.1", "localhost"} or args.port <= 0:
        parser.error("only a valid loopback endpoint is allowed")
    return args


def build_runtime(args: argparse.Namespace) -> AsyncPolicyLearnerRuntime:
    from forcesmolvla.training_runtime import (
        resolve_task_dataset_root,
        resolve_task_output_root,
        resolve_task_reward_transition_root,
    )

    require(
        args.allow_development_policy_execution_smoke,
        "ONLINE_REPLAY_ASYNC_ROBOT_EXECUTION_FLAG_REQUIRED",
    )
    require(torch.cuda.is_available(), "ONLINE_REPLAY_ASYNC_CUDA_UNAVAILABLE")
    output_root = resolve_task_output_root(
        ROOT, task_id=args.task_id, output_root=args.output_root
    )
    dataset_root = resolve_task_dataset_root(
        ROOT, task_id=args.task_id, dataset_root=args.dataset_root
    )
    reward_transition_root = resolve_task_reward_transition_root(
        ROOT,
        task_id=args.task_id,
        reward_transition_root=args.reward_transition_root,
    )
    warmup.configure_task_paths(
        task_id=args.task_id,
        dataset_root=dataset_root,
        reward_transition_root=reward_transition_root,
        output_root=output_root,
    )
    require(warmup.TASK == args.task.strip(), "FORCERFT_TASK_PROMPT_MISMATCH")
    resume_checkpoint = (
        select_resume_or_seed_checkpoint(
            output_root,
            configured_seed_bundle=getattr(args, "stage3_seed_bundle", None),
            allow_legacy_offline_fallback=getattr(
                args, "allow_legacy_offline_fallback", False
            ),
        ).path
        if args.learner_resume_checkpoint is None
        else args.learner_resume_checkpoint.resolve()
    )
    checkpoint_metadata = json.loads(
        (resume_checkpoint / "metadata.json").read_text(encoding="utf-8")
    )
    checkpoint_kind = str(checkpoint_metadata.get("kind", ""))
    require(
        checkpoint_kind
        in {
            "stage3_safe_seed_v1",
            "legacy_offline_actor_critic_ablation",
            "offline_actor_critic_exact_resume",
            "online_actor_critic_exact_resume",
        }
        and (
            checkpoint_kind
            not in {
                "legacy_offline_actor_critic_ablation",
                "offline_actor_critic_exact_resume",
            }
            or getattr(args, "allow_legacy_offline_fallback", False)
        )
        and exact_resume_checkpoint_is_recoverable(
            resume_checkpoint, expected_kind=checkpoint_kind
        ),
        "FORCERFT_EXACT_RESUME_CHECKPOINT_INVALID",
    )
    actor_package = resume_checkpoint / str(checkpoint_metadata["actor_directory"])
    device = torch.device("cuda:0")
    safety_config = (
        ROOT / f"configs/live_action_safety.{args.task_id}.development.yaml"
        if args.safety_config is None
        else args.safety_config.resolve()
    )
    engine = serve_policy.InferenceEngine(
        actor_package,
        safety_config,
        ROOT / "schemas/rulespec.schema.json",
        device,
        allow_development_policy_execution_smoke=True,
    )
    engine.metadata.update({
        "robot_execution_allowed": True,
        "robot_execution_mode": "supervised_development",
        "development_execution_override": True,
        "gripper_max_age_ms": 300.0,
        "controller_ack_timeout_ms": 20.0,
    })
    actor_checkpoint = checkpoint_metadata.get("actor_checkpoint", {})
    active_revision_id = str(actor_checkpoint.get("checkpoint_id", ""))
    require(
        checkpoint_kind
        in {
            "stage3_safe_seed_v1",
            "legacy_offline_actor_critic_ablation",
            "offline_actor_critic_exact_resume",
            "online_actor_critic_exact_resume",
        }
        and active_revision_id,
        "FORCERFT_EXACT_RESUME_METADATA_INVALID",
    )
    machine = InMemoryRevisionStateMachine(
        RevisionRecord(
            active_revision_id,
            engine.model_sha256,
            RevisionState.ACTIVE,
        )
    )
    checkpoint_root = output_root / "online/checkpoints"
    common_config = warmup.load_common_actor_critic_config(args.task_id)
    online_config = common_config["online_training"]
    training_policy = OnlineTrainingPolicy(
        training_starts=int(online_config["training_starts"]),
        demo_ratio=float(online_config["demo_ratio"]),
        online_ratio=float(online_config["online_ratio"]),
        critic_updates_per_cycle=int(online_config["critic_updates_per_cycle"]),
        actor_updates_per_cycle=int(online_config["actor_updates_per_cycle"]),
        target_polyak_updates_per_cycle=int(
            online_config["target_polyak_updates_per_cycle"]
        ),
        actor_parameter_broadcast_period=int(
            online_config["actor_parameter_broadcast_period"]
        ),
        checkpoint_period=int(online_config["checkpoint_period"]),
        keep_latest_checkpoints=int(online_config["keep_latest_checkpoints"]),
    )
    sft_reference_checkpoint = (
        output_root / "sft/checkpoints/forcesmolvla_sft_step_010000"
        if args.sft_reference_checkpoint is None
        else args.sft_reference_checkpoint.resolve()
    )
    learner = ContinuousLearner(
        device=device,
        resume_checkpoint=resume_checkpoint,
        checkpoint_root=checkpoint_root,
        replay_root=output_root / "online",
        current_session_id=args.session_id,
        task=args.task.strip(),
        sft_reference_checkpoint=sft_reference_checkpoint,
        training_policy=training_policy,
    )
    return AsyncPolicyLearnerRuntime(
        engine=engine,
        machine=machine,
        session_id=args.session_id,
        episode_id=args.episode_id,
        active_revision_id=active_revision_id,
        active_model_revision=engine.model_sha256,
        learner_resume_checkpoint=resume_checkpoint,
        online_checkpoint_root=checkpoint_root,
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
        f"model={runtime.active_model_revision} "
        f"deployed_online_cycle={runtime.metadata['active_actor_online_cycle']}",
        flush=True,
    )
    print(
        f"[learner] exact-resume={runtime.learner_resume_checkpoint} "
        f"online_checkpoints={runtime.online_checkpoint_root}",
        flush=True,
    )
    print(
        f"[server] listening on http://{args.host}:{args.port} "
        "robot_io=false execution=supervised-development",
        flush=True,
    )
    try:
        server.serve_forever(poll_interval=0.2)
    except KeyboardInterrupt:
        pass
    finally:
        runtime.stop()
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
