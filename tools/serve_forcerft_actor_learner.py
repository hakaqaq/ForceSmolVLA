#!/usr/bin/env python3
"""Serve one persistent ForceRFT Actor/Learner process across episodes."""

from __future__ import annotations

import argparse
from contextlib import AbstractContextManager
from http.server import ThreadingHTTPServer
import json
from pathlib import Path
import sys
import threading
from typing import Any, Mapping

import torch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for path in (SRC, ROOT / "tools"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from forcesmolvla.rft.online import replay_training as warmup  # noqa: E402
import serve_policy  # noqa: E402
from forcesmolvla.rft.online.actor_learner_runtime import (  # noqa: E402
    OnlineTrainingPolicy,
    exact_resume_checkpoint_is_recoverable,
    online_checkpoint_path,
    retain_latest_online_checkpoints,
    EpisodePin,
    InferencePriorityCoordinator,
    PinnedEpisode,
    prepare_learner,
    select_resume_or_seed_checkpoint,
)
from forcesmolvla.rft.critic import polyak_update  # noqa: E402
from forcesmolvla.rft.residual_actor import WristWrenchResidualActor  # noqa: E402
from forcesmolvla.rft.online.learner_checkpoint import (  # noqa: E402
    save_residual_checkpoint,
)
from forcesmolvla.rft.online.training_losses import (  # noqa: E402
    residual_actor_loss,
    residual_critic_loss,
)
from forcesmolvla.rft.online.policy_revision import (  # noqa: E402
    InMemoryRevisionStateMachine,
    RevisionRecord,
    RevisionState,
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _load_residual_checkpoint(policy: Any, checkpoint: Path) -> None:
    path = checkpoint if checkpoint.is_file() else checkpoint / "residual_actor.pt"
    policy.load_state_dict(
        torch.load(path, map_location=next(policy.parameters()).device, weights_only=True),
        strict=True,
    )
    policy.eval()


def _select_deployed_actor_for_resume(
    *, resume_checkpoint: Path
) -> tuple[Path, Path, str, int]:
    """Resolve the fixed base package and the last episode-active residual only."""

    resume_checkpoint = resume_checkpoint.resolve()
    runtime = torch.load(
        resume_checkpoint / "state/runtime_state.pt",
        map_location="cpu",
        weights_only=False,
    )
    base = Path(runtime["base_actor_checkpoint"]).resolve()
    revision_id = str(runtime["active_residual_revision"])
    try:
        actor_step = int(revision_id.rsplit("-", 1)[1])
    except ValueError:
        actor_step = 0
    candidate = (
        resume_checkpoint.parent.parent
        / "actor_candidates"
        / f"online_actor_step_{actor_step:06d}"
    )
    residual = candidate / "residual_actor.pt"
    if actor_step > 0:
        require(
            residual.is_file(),
            "FORCERFT_ACTIVE_RESIDUAL_CANDIDATE_MISSING",
        )
    else:
        residual = resume_checkpoint / "models/residual_actor.pt"
    return base, residual.resolve(), revision_id, int(
        runtime.get("online_joint_cycles", 0)
    )

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


class ContinuousLearner:
    """Three-phase learner over sealed real ACK replay only."""

    def __init__(
        self,
        *,
        device: torch.device,
        resume_checkpoint: Path,
        checkpoint_root: Path,
        replay_root: Path,
        current_session_id: str | None,
        task: str,
        normalizer_path: Path | None = None,
        training_policy: OnlineTrainingPolicy | None = None,
    ) -> None:
        from forcesmolvla.training_data import load_normalizer_manifest

        self.device = device
        self.resume_checkpoint = resume_checkpoint.resolve()
        self.checkpoint_root = checkpoint_root.resolve()
        self.checkpoint_root.mkdir(parents=True, exist_ok=True)
        self.replay_root = replay_root.resolve()
        self.current_session_id = current_session_id
        self.task = task
        self.learner = prepare_learner(
            device,
            resume_checkpoint=self.resume_checkpoint,
        )
        self.training_policy = (
            self.learner["training_policy"]
            if training_policy is None
            else training_policy
        )
        path = (
            warmup.DATASET / "normalizer_manifest.json"
            if normalizer_path is None
            else Path(normalizer_path).resolve()
        )
        self.normalizer = load_normalizer_manifest(path)
        self.replay: warmup.OnlineResidualReplay | None = None
        self.unique_r_count = 0
        self.r_macro_count = 0
        self._replay_signature: tuple[str, ...] | None = None
        self._replay_snapshot: tuple[Any, ...] | None = None
        self._materialized_replay_signature: tuple[str, ...] | None = None

    @property
    def residual_actor(self) -> torch.nn.Module:
        return self.learner["residual_actor"]

    def set_current_session(self, session_id: str) -> None:
        self.current_session_id = session_id

    def clear_current_session(self) -> None:
        self.current_session_id = None

    def _load_replay_snapshot(self) -> tuple[Any, ...]:
        signature = self._episode_signature()
        if self._replay_snapshot is None or signature != self._replay_signature:
            self._replay_snapshot = warmup.load_formal_online_r(self.replay_root)
            self._replay_signature = signature
        return self._replay_snapshot

    def _episode_signature(self) -> tuple[str, ...]:
        return tuple(
            path.name
            for path in sorted((self.replay_root / "episodes").glob("*.json"))
        )

    def _refresh_replay(self) -> warmup.OnlineResidualReplay:
        signature = self._episode_signature()
        if (
            self.replay is not None
            and signature == self._materialized_replay_signature
        ):
            return self.replay
        all_r, policy_macros, _source_episodes, human_rows = (
            self._load_replay_snapshot()
        )
        if self.current_session_id is not None:
            require(
                not any(
                    row["identity"].get("session_id") == self.current_session_id
                    for row in [*all_r, *human_rows]
                ),
                "ONLINE_REPLAY_ASYNC_CURRENT_EPISODE_ALREADY_IN_REPLAY",
            )
        macros = (*policy_macros, *warmup.build_ack_macros(human_rows))
        self.replay = warmup.OnlineResidualReplay(macros, self.normalizer)
        self._materialized_replay_signature = signature
        self._replay_snapshot = None
        self.unique_r_count = len(all_r) + len(human_rows)
        self.r_macro_count = len(macros)
        runtime_replay = self.learner["runtime"]["replay"]
        runtime_replay.update(
            critic_td_valid_rows=self.replay.critic_td_valid_rows,
            actor_q_valid_rows=self.replay.actor_q_valid_rows,
            human_residual_valid_rows=self.replay.human_residual_valid_rows,
        )
        return self.replay

    def _critic_update(
        self,
        coordinator: InferencePriorityCoordinator,
        replay: warmup.OnlineResidualReplay,
        *,
        burnin: bool,
    ) -> float:
        learner = self.learner
        counters = learner["runtime"]["counters"]
        step = int(counters["critic_optimizer_steps"])
        batch = replay.sample(
            self.training_policy.critic_batch_size,
            device=self.device,
            seed=int(learner["config"]["environment"]["seed"]) + step,
        )
        require(batch is not None, "FORCERFT_CRITIC_REPLAY_EMPTY")
        with coordinator.learner_step_slot(
            "critic_burnin" if burnin else "critic"
        ):
            optimizer = learner["critic_optimizer"]
            optimizer.zero_grad(set_to_none=True)
            loss = residual_critic_loss(
                learner["q1"],
                learner["q2"],
                learner["q1_target"],
                learner["q2_target"],
                learner["residual_actor_target"],
                batch,
                float(learner["config"]["loss"]["gamma_macro"]),
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                (*learner["q1"].parameters(), *learner["q2"].parameters()),
                float(learner["config"]["optimizer"]["critic"]["grad_clip_norm"]),
            )
            optimizer.step()
            tau = float(learner["config"]["optimizer"]["polyak_tau"])
            polyak_update(learner["q1"], learner["q1_target"], tau)
            polyak_update(learner["q2"], learner["q2_target"], tau)
        counters["critic_optimizer_steps"] = step + 1
        counters["target_polyak_steps"] = int(counters["target_polyak_steps"]) + 1
        if burnin:
            learner["runtime"]["critic_burnin_updates"] = (
                int(learner["runtime"].get("critic_burnin_updates", 0)) + 1
            )
        return float(loss.detach())

    def _actor_update(
        self,
        coordinator: InferencePriorityCoordinator,
        replay: warmup.OnlineResidualReplay,
    ) -> dict[str, float]:
        learner = self.learner
        counters = learner["runtime"]["counters"]
        step = int(counters["actor_optimizer_steps"])
        seed = int(learner["config"]["environment"]["seed"]) + 1_000_000 + step
        policy_batch = replay.sample(
            self.training_policy.actor_policy_batch_size,
            device=self.device,
            seed=seed,
        )
        human_batch = replay.sample(
            self.training_policy.actor_human_batch_size,
            device=self.device,
            seed=seed + 1,
            human_only=True,
        )
        require(policy_batch is not None, "FORCERFT_ACTOR_REPLAY_EMPTY")
        critic_parameters = (
            *learner["q1"].parameters(),
            *learner["q2"].parameters(),
        )
        with coordinator.learner_step_slot(
            "actor", episode_idle_required=True
        ):
            for parameter in critic_parameters:
                parameter.requires_grad_(False)
            try:
                optimizer = learner["residual_actor_optimizer"]
                optimizer.zero_grad(set_to_none=True)
                losses = residual_actor_loss(
                    learner["q1"],
                    learner["q2"],
                    learner["residual_actor"],
                    policy_batch,
                    human_batch,
                    actor_q_weight=float(
                        learner["config"]["loss"]["actor_q_weight"]
                    ),
                    residual_l2_weight=float(
                        learner["config"]["loss"]["residual_l2_weight"]
                    ),
                    human_residual_weight=float(
                        learner["config"]["loss"]["human_residual_weight"]
                    ),
                )
                losses.total.backward()
                torch.nn.utils.clip_grad_norm_(
                    learner["residual_actor"].parameters(),
                    float(
                        learner["config"]["optimizer"]["actor"]["grad_clip_norm"]
                    ),
                )
                optimizer.step()
            finally:
                for parameter in critic_parameters:
                    parameter.requires_grad_(True)
            polyak_update(
                learner["residual_actor"],
                learner["residual_actor_target"],
                float(learner["config"]["optimizer"]["polyak_tau"]),
            )
        counters["actor_optimizer_steps"] = step + 1
        return {
            "total": float(losses.total.detach()),
            "value": float(losses.value.detach()),
            "residual": float(losses.residual.detach()),
            "human": float(losses.human.detach()),
        }

    def __call__(
        self, coordinator: InferencePriorityCoordinator
    ) -> dict[str, Any]:
        learner = self.learner
        runtime = learner["runtime"]
        signature = self._episode_signature()
        count = (
            self.replay.critic_td_valid_rows
            if self.replay is not None
            and signature == self._materialized_replay_signature
            else warmup.count_sealed_critic_td_valid_transitions(self.replay_root)
        )
        runtime["replay"]["critic_td_valid_rows"] = count
        if count < self.training_policy.training_starts:
            runtime["phase"] = "collecting"
            return {
                "waiting_for_replay": True,
                "phase": "collecting",
                "learner_critic_steps": 0,
                "learner_actor_steps": 0,
                "learner_polyak_steps": 0,
                "current_episode_sampled": False,
                "nonfinite_count": 0,
                "oom_count": 0,
            }

        replay = self._refresh_replay()
        if runtime["phase"] in {"collecting", "critic_burnin"}:
            runtime["phase"] = "critic_burnin"
            before = {
                name: value.detach().clone()
                for name, value in learner["residual_actor"].state_dict().items()
            }
            losses = []
            remaining = (
                self.training_policy.critic_burnin_updates
                - int(runtime.get("critic_burnin_updates", 0))
            )
            for _ in range(max(0, remaining)):
                losses.append(
                    self._critic_update(coordinator, replay, burnin=True)
                )
            require(
                all(
                    torch.equal(before[name], value)
                    for name, value in learner["residual_actor"].state_dict().items()
                ),
                "FORCERFT_BURNIN_MODIFIED_RESIDUAL_ACTOR",
            )
            runtime["phase"] = "joint"
            runtime["critic_burnin_complete"] = True
            return {
                "waiting_for_replay": False,
                "phase": "joint",
                "critic_burnin_complete": True,
                "critic_burnin_updates": int(runtime["critic_burnin_updates"]),
                "learner_critic_steps": len(losses),
                "learner_actor_steps": 0,
                "learner_polyak_steps": len(losses),
                "online_joint_cycle": int(runtime["online_joint_cycles"]),
                "actor_optimizer_steps": int(
                    runtime["counters"]["actor_optimizer_steps"]
                ),
                "latest_critic_td_loss": losses[-1] if losses else None,
                "current_episode_sampled": False,
                "nonfinite_count": 0,
                "oom_count": 0,
            }

        require(
            runtime["phase"] == "joint"
            and runtime["critic_burnin_complete"] is True,
            "FORCERFT_ONLINE_PHASE_INVALID",
        )
        cycle = int(runtime["online_joint_cycles"])
        budget = self.training_policy.joint_cycle_budget(
            replay.critic_rows_per_episode
        )
        if cycle >= budget:
            return {
                "waiting_for_replay": True,
                "phase": "joint",
                "episode_cycle_budget_exhausted": True,
                "episode_cycle_budget": budget,
                "learner_critic_steps": 0,
                "learner_actor_steps": 0,
                "learner_polyak_steps": 0,
                "current_episode_sampled": False,
                "nonfinite_count": 0,
                "oom_count": 0,
            }

        critic_losses = [
            self._critic_update(coordinator, replay, burnin=False)
            for _ in range(self.training_policy.critic_updates_per_cycle)
        ]
        actor_metrics = self._actor_update(coordinator, replay)
        runtime["online_joint_cycles"] = cycle + 1
        learner["online_joint_cycles"] = cycle + 1
        latest_checkpoint = None
        if self.training_policy.checkpoint_due(cycle + 1):
            latest_checkpoint = self.save_checkpoint()
        return {
            "waiting_for_replay": False,
            "phase": "joint",
            "learner_critic_steps": self.training_policy.critic_updates_per_cycle,
            "learner_actor_steps": 1,
            "learner_polyak_steps": self.training_policy.critic_updates_per_cycle,
            "current_episode_sampled": False,
            "nonfinite_count": 0,
            "oom_count": 0,
            "online_joint_cycle": cycle + 1,
            "actor_optimizer_steps": int(
                runtime["counters"]["actor_optimizer_steps"]
            ),
            "latest_critic_td_loss": critic_losses[-1],
            "latest_actor_loss": actor_metrics["total"],
            "latest_min_twin_q": -actor_metrics["value"],
            "latest_checkpoint_path": (
                None if latest_checkpoint is None else str(latest_checkpoint)
            ),
        }

    def export_actor_candidate(self, actor_optimizer_steps: int) -> dict[str, Any]:
        runtime = self.learner["runtime"]
        current = str(runtime["active_residual_revision"])
        task_id = current.split("-residual-step-", 1)[0]
        revision_id = f"{task_id}-residual-step-{actor_optimizer_steps:06d}"
        destination = (
            self.checkpoint_root.parent
            / "actor_candidates"
            / f"online_actor_step_{actor_optimizer_steps:06d}"
        )
        destination.mkdir(parents=True, exist_ok=True)
        path = destination / "residual_actor.pt"
        if not path.exists():
            torch.save(self.learner["residual_actor"].state_dict(), path)
        return {
            "revision_id": revision_id,
            "checkpoint": destination.resolve(),
            "actor_optimizer_steps": int(actor_optimizer_steps),
        }

    def mark_active_residual_revision(self, revision_id: str) -> None:
        self.learner["runtime"]["active_residual_revision"] = str(revision_id)

    def save_checkpoint(self) -> Path:
        learner = self.learner
        completed = int(learner["runtime"]["online_joint_cycles"])
        target = online_checkpoint_path(self.checkpoint_root, completed)
        save_residual_checkpoint(
            target,
            residual_actor=learner["residual_actor"],
            residual_actor_target=learner["residual_actor_target"],
            q1=learner["q1"],
            q2=learner["q2"],
            q1_target=learner["q1_target"],
            q2_target=learner["q2_target"],
            residual_actor_optimizer=learner["residual_actor_optimizer"],
            critic_optimizer=learner["critic_optimizer"],
            runtime_state=learner["runtime"],
            config=learner["config"],
        )
        retain_latest_online_checkpoints(
            self.checkpoint_root,
            keep=self.training_policy.keep_latest_checkpoints,
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
        active_actor_checkpoint: Path,
        learner_resume_checkpoint: Path,
        online_checkpoint_root: Path,
        learner_job: ContinuousLearner,
        active_actor_online_cycle: int | None = None,
        inference_stream: Any = None,
    ) -> None:
        self.engine = engine
        self.machine = machine
        self.session_id = session_id
        self.episode_id = episode_id
        self.active_revision_id = active_revision_id
        self.active_model_revision = active_model_revision
        self.active_actor_checkpoint = active_actor_checkpoint.resolve()
        self.base_actor_checkpoint = Path(
            engine.metadata.get("checkpoint", active_actor_checkpoint)
        ).resolve()
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
        self._candidate_count = 0
        self._candidate_checkpoints: dict[str, Path] = {}
        self._candidate_online_cycles: dict[str, int] = {}
        self._active_actor_online_cycle = (
            int(active_actor_online_cycle)
            if active_actor_online_cycle is not None
            else (
                int(self.active_actor_checkpoint.name.rsplit("_", 1)[1])
                if self.active_actor_checkpoint.name.startswith("online_actor_step_")
                else 0
            )
        )
        self._policy = getattr(
            learner_job, "training_policy", OnlineTrainingPolicy()
        )
        self._learner_thread: threading.Thread | None = None
        self._inference_request_count = 0
        self._actor_alive: AbstractContextManager[Any] | None = None
        self._pin: PinnedEpisode | None = None
        self._quiesced_checkpoint: Path | None = None
        self._quiesced = False

    @property
    def metadata(self) -> dict[str, Any]:
        learner_runtime = getattr(self.learner_job, "learner", {}).get(
            "runtime", {}
        )
        return {
            **self.engine.metadata,
            "online_actor_learner": True,
            "server_persistent": True,
            "current_episode_sampling": False,
            "training_starts": self._policy.training_starts,
            "actor_candidate_period": self._policy.actor_candidate_period,
            "max_joint_cycles_per_admitted_episode": (
                self._policy.max_joint_cycles_per_admitted_episode
            ),
            "active_actor_online_cycle": self._active_actor_online_cycle,
            "checkpoint_period": self._policy.checkpoint_period,
            "keep_latest_checkpoints": self._policy.keep_latest_checkpoints,
            "phase": learner_runtime.get("phase", "collecting"),
            "critic_burnin_complete": bool(
                learner_runtime.get("critic_burnin_complete", False)
            ),
            "save_checkpoint_on_graceful_exit": True,
            "save_checkpoint_on_operator_q": True,
            "runtime_session_id": self.session_id,
            "runtime_episode_id": self.episode_id,
            "learner_started": self._learner_started,
            "learner_resume_checkpoint": str(self.learner_resume_checkpoint),
            "active_actor_revision": self.active_revision_id,
            "active_actor_model_revision": self.active_model_revision,
            "active_actor_checkpoint": str(self.active_actor_checkpoint),
            "base_actor_checkpoint": str(self.base_actor_checkpoint),
            "pending_actor_revision": self.machine.pending_revision_id,
            "actor_candidate_count": self._candidate_count,
            "policy_epoch": int(self.machine.policy_epoch),
            "online_checkpoint_root": str(self.online_checkpoint_root),
        }

    def status(self) -> dict[str, Any]:
        with self._lock:
            result = dict(self._learner_result)
            learner_runtime = getattr(self.learner_job, "learner", {}).get(
                "runtime", {}
            )
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
                "active_actor_checkpoint": str(self.active_actor_checkpoint),
                "base_actor_checkpoint": str(self.base_actor_checkpoint),
                "pending_actor_revision": self.machine.pending_revision_id,
                "actor_candidate_count": self._candidate_count,
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
                "actor_activation_count": self._broadcast_count,
                # Compatibility for existing capture summaries.
                "actor_parameter_broadcast_count": self._broadcast_count,
                "active_actor_online_cycle": self._active_actor_online_cycle,
                "online_joint_cycle": int(result.get("online_joint_cycle", 0)),
                "actor_optimizer_steps": int(result.get("actor_optimizer_steps", 0)),
                "phase": result.get(
                    "phase", learner_runtime.get("phase", "collecting")
                ),
                "critic_burnin_complete": bool(
                    learner_runtime.get("critic_burnin_complete", False)
                ),
                "critic_burnin_updates": int(
                    learner_runtime.get("critic_burnin_updates", 0)
                ),
                "latest_critic_td_loss": result.get("latest_critic_td_loss"),
                "latest_actor_loss": result.get("latest_actor_loss"),
                "latest_min_twin_q": result.get("latest_min_twin_q"),
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
        session_id = str(payload.get("session_id", ""))
        episode_id = str(payload.get("episode_id", ""))
        require(bool(session_id and episode_id), "ONLINE_REPLAY_ASYNC_CAPTURE_IDENTITY_MISMATCH")
        with self._lock:
            self._activate_pending_actor_locked()
            self.engine.reset_residual_episode_context()
            self.session_id = session_id
            self.episode_id = episode_id
            self.learner_job.set_current_session(session_id)
        return self.status()

    def _stage_actor_candidate(self, candidate: Mapping[str, Any]) -> None:
        revision_id = str(candidate["revision_id"])
        checkpoint = Path(candidate["checkpoint"]).resolve()
        with self._lock:
            pending = self.machine.pending_revision_id
            if pending is not None:
                self.machine.reject(pending, "superseded by a newer Actor candidate")
            # Capture identity remains bound to the immutable frozen base Actor.
            # Residual revisions are tracked by revision_id, not a new digest.
            self.machine.register_candidate(revision_id, self.active_model_revision)
            self.machine.stage(revision_id)
            self._candidate_checkpoints[revision_id] = checkpoint
            self._candidate_online_cycles[revision_id] = int(
                candidate["online_joint_cycle"]
            )
            self._candidate_count += 1
        print(
            "[model] staged online Actor candidate "
            f"revision={revision_id} checkpoint={checkpoint}",
            flush=True,
        )

    def _activate_pending_actor_locked(self) -> None:
        pending = self.machine.pending_revision_id
        if pending is None:
            return
        checkpoint = self._candidate_checkpoints[pending]
        activated = self.machine.activate_pending_at_episode_boundary()
        with self.engine._lock:
            _load_residual_checkpoint(self.engine.residual_actor, checkpoint)
        self.engine.reset_residual_episode_context()
        self.active_revision_id = activated.revision_id
        self.active_actor_checkpoint = checkpoint
        self.learner_job.mark_active_residual_revision(activated.revision_id)
        self._active_actor_online_cycle = self._candidate_online_cycles.pop(
            activated.revision_id
        )
        self._broadcast_count += 1
        print(
            "[model] activated online Actor at episode boundary "
            f"revision={activated.revision_id} epoch={self.machine.policy_epoch}",
            flush=True,
        )

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
                        actor_optimizer_steps = int(
                            result.get("actor_optimizer_steps", 0)
                        )
                        if (
                            int(result.get("learner_actor_steps", 0)) > 0
                            and self._policy.candidate_due(actor_optimizer_steps)
                        ):
                            candidate = self.learner_job.export_actor_candidate(
                                actor_optimizer_steps
                            )
                            candidate["online_joint_cycle"] = cycle
                            self._stage_actor_candidate(candidate)
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

    def quiesce_and_save(self, _payload: Mapping[str, Any]) -> dict[str, Any]:
        require(not self._episode_active, "ONLINE_REPLAY_ASYNC_EPISODE_ACTIVE")
        self.stop()
        with self._lock:
            require(
                self._learner_state != "failed",
                "ONLINE_REPLAY_ASYNC_LEARNER_FAILED",
            )
        if not self._quiesced:
            self._quiesced_checkpoint = self.learner_job.save_checkpoint()
            self._quiesced = True
        return {
            **self.status(),
            "quiesced": True,
            "quiesced_checkpoint_path": (
                None
                if self._quiesced_checkpoint is None
                else str(self._quiesced_checkpoint)
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
            self._activate_pending_actor_locked()

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
            "/runtime/quiesce-and-save",
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
                "/runtime/quiesce-and-save": self.runtime.quiesce_and_save,
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
    parser.add_argument("--safety-config", type=Path)
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--episode-id", required=True)
    parser.add_argument("--learner-resume-checkpoint", type=Path)
    parser.add_argument("--stage3-seed-bundle", type=Path)
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
    warmup.configure_task_paths(
        task_id=args.task_id,
        dataset_root=dataset_root,
        output_root=output_root,
    )
    require(warmup.TASK == args.task.strip(), "FORCERFT_TASK_PROMPT_MISMATCH")
    if args.learner_resume_checkpoint is None:
        selected = select_resume_or_seed_checkpoint(
            output_root,
            configured_seed_bundle=getattr(args, "stage3_seed_bundle", None),
        )
        resume_checkpoint = selected.path
        checkpoint_kind = selected.kind
    else:
        resume_checkpoint = args.learner_resume_checkpoint.resolve()
        checkpoint_kind = (
            "stage3_seed"
            if resume_checkpoint.name.startswith("stage3_base_actor_residual_q_")
            else "online_residual_actor_critic"
        )
    require(
        exact_resume_checkpoint_is_recoverable(
            resume_checkpoint, expected_kind=checkpoint_kind
        ),
        "FORCERFT_EXACT_RESUME_CHECKPOINT_INVALID",
    )
    (
        base_actor_checkpoint,
        residual_checkpoint,
        active_revision_id,
        active_actor_online_cycle,
    ) = _select_deployed_actor_for_resume(
        resume_checkpoint=resume_checkpoint,
    )
    device = torch.device("cuda:0")
    safety_config = (
        ROOT / f"configs/live_action_safety.{args.task_id}.development.yaml"
        if args.safety_config is None
        else args.safety_config.resolve()
    )
    engine = serve_policy.InferenceEngine(
        base_actor_checkpoint,
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
    engine.policy.eval().requires_grad_(False)
    require(
        active_revision_id
        and not any(parameter.requires_grad for parameter in engine.policy.parameters()),
        "FORCERFT_BASE_ACTOR_NOT_FROZEN",
    )
    common_config = warmup.load_common_actor_critic_config(args.task_id)
    warmup_config = common_config["warmup"]
    batching_config = common_config["batching"]
    online_config = common_config["online_training"]
    try:
        active_actor_steps = int(active_revision_id.rsplit("-", 1)[1])
    except ValueError:
        active_actor_steps = 0
    initial_policy_epoch = active_actor_steps // int(
        online_config["actor_candidate_period"]
    )
    machine = InMemoryRevisionStateMachine(
        RevisionRecord(
            active_revision_id,
            engine.model_sha256,
            RevisionState.ACTIVE,
        ),
        initial_epoch=initial_policy_epoch,
    )
    checkpoint_root = output_root / "online/checkpoints"
    training_policy = OnlineTrainingPolicy(
        training_starts=int(warmup_config["training_starts"]),
        critic_burnin_updates=int(warmup_config["critic_burnin_updates"]),
        critic_batch_size=int(batching_config["critic_batch_size"]),
        actor_policy_batch_size=int(
            batching_config["actor_policy_batch_size"]
        ),
        actor_human_batch_size=int(
            batching_config["actor_human_batch_size"]
        ),
        critic_updates_per_cycle=int(online_config["critic_updates_per_cycle"]),
        actor_updates_per_cycle=int(online_config["actor_updates_per_cycle"]),
        max_joint_cycles_per_admitted_episode=int(
            online_config["max_joint_cycles_per_admitted_episode"]
        ),
        actor_candidate_period=int(
            online_config["actor_candidate_period"]
        ),
        checkpoint_period=int(online_config["checkpoint_period"]),
        keep_latest_checkpoints=int(online_config["keep_latest_checkpoints"]),
    )
    learner = ContinuousLearner(
        device=device,
        resume_checkpoint=resume_checkpoint,
        checkpoint_root=checkpoint_root,
        replay_root=output_root / "online",
        current_session_id=args.session_id,
        task=args.task.strip(),
        normalizer_path=dataset_root / "normalizer_manifest.json",
        training_policy=training_policy,
    )
    engine.residual_actor = WristWrenchResidualActor(
        hidden_dim=int(common_config["residual_actor"]["hidden_dim"]),
        max_normalized_residual=float(
            common_config["residual_actor"]["max_normalized_residual"]
        ),
    ).to(device)
    engine.residual_actor.eval().requires_grad_(False)
    if active_actor_steps > 0:
        _load_residual_checkpoint(engine.residual_actor, residual_checkpoint)
    return AsyncPolicyLearnerRuntime(
        engine=engine,
        machine=machine,
        session_id=args.session_id,
        episode_id=args.episode_id,
        active_revision_id=active_revision_id,
        active_model_revision=engine.model_sha256,
        active_actor_checkpoint=residual_checkpoint,
        learner_resume_checkpoint=resume_checkpoint,
        online_checkpoint_root=checkpoint_root,
        learner_job=learner,
        active_actor_online_cycle=active_actor_online_cycle,
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
        if runtime.status()["learner_state"] != "failed":
            checkpoint = runtime.learner_job.save_checkpoint()
            print(
                f"[learner] graceful-exit checkpoint={checkpoint or 'none'}",
                flush=True,
            )
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
