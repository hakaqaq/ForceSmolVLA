#!/usr/bin/env python3
"""Serve one persistent ForceRFT Actor/Learner process across episodes."""

from __future__ import annotations

import argparse
from copy import deepcopy
from contextlib import AbstractContextManager, nullcontext
import hashlib
from http.server import ThreadingHTTPServer
import json
import os
from pathlib import Path
import sys
import threading
import time
from typing import Any, Mapping

import torch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for path in (SRC, ROOT / "tools"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from forcesmolvla.rft.online import replay_training as warmup  # noqa: E402
import serve_policy  # noqa: E402
from forcesmolvla.rft.online.residual_actor_critic_runtime import (  # noqa: E402
    ONLINE_ADAPTATION_DIRECTORY_NAME,
    ResidualActorCriticSchedule,
    exact_resume_checkpoint_is_recoverable,
    training_checkpoint_path,
    retain_latest_training_checkpoints,
    EpisodePin,
    InferencePriorityCoordinator,
    PinnedEpisode,
    load_checkpoint_training_config,
    prepare_learner,
    require_exact_resume_algorithm_config,
    select_resume_or_bootstrap_checkpoint,
)
from forcesmolvla.rft.critic import (  # noqa: E402
    RESIDUAL_ACTION_OFFSET,
    RESIDUAL_ACTION_WIDTH,
    polyak_update,
)
from forcesmolvla.rft.residual_actor import WristWrenchResidualActor  # noqa: E402
from forcesmolvla.rft.online.residual_actor_critic_checkpoint import (  # noqa: E402
    CANDIDATE_CHECKPOINT_KIND,
    TRAINING_CHECKPOINT_KIND,
    save_residual_actor_critic_checkpoint,
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
from forcesmolvla.rft.online.transition_authority import (  # noqa: E402
    ONLINE_SEMANTICS_VERSION,
)
from forcesmolvla.rft.online.schedule_migration import (  # noqa: E402
    schedule_migration_required,
)


ACTOR_GRAD_EPSILON = 1.0e-12


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _residual_actor_state_sha256(state: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _load_residual_checkpoint(policy: Any, checkpoint: Path) -> None:
    path = checkpoint if checkpoint.is_file() else checkpoint / "residual_actor.pt"
    if checkpoint.is_dir():
        metadata_path = checkpoint / "candidate_state.pt"
        require(
            metadata_path.is_file(),
            "FORCERFT_RESIDUAL_CANDIDATE_METADATA_MISSING",
        )
        metadata = torch.load(
            metadata_path, map_location="cpu", weights_only=False
        )
        require(
            metadata.get("checkpoint_kind") == CANDIDATE_CHECKPOINT_KIND
            and metadata.get("online_semantics_version")
            == ONLINE_SEMANTICS_VERSION,
            "FORCERFT_RESIDUAL_CANDIDATE_SEMANTICS_MISMATCH",
        )
    policy.load_state_dict(
        torch.load(path, map_location=next(policy.parameters()).device, weights_only=True),
        strict=True,
    )
    policy.eval()


def _select_deployed_actor_for_resume(
    *, resume_checkpoint: Path
) -> tuple[Path, Path, str, int | None, int, int, str]:
    """Resolve the fixed base package and the last episode-active residual only."""

    resume_checkpoint = resume_checkpoint.resolve()
    runtime = torch.load(
        resume_checkpoint / "state/runtime_state.pt",
        map_location="cpu",
        weights_only=False,
    )
    base = Path(runtime["frozen_base_policy_checkpoint"]).resolve()
    revision_id = str(runtime["active_residual_policy_revision"])
    online_adaptation_id = str(runtime["online_adaptation_id"])
    scheduling = runtime.get("scheduling")
    if isinstance(scheduling, Mapping) and scheduling.get("mode") == "continuous_async":
        residual = Path(str(scheduling["active_actor_checkpoint"])).resolve()
        require(
            residual.is_file()
            or (
                (residual / "residual_actor.pt").is_file()
                and (residual / "candidate_state.pt").is_file()
            ),
            "FORCERFT_ACTIVE_RESIDUAL_CANDIDATE_MISSING",
        )
        publication_cycle = scheduling.get("active_publication_cycle")
        require(
            publication_cycle is None
            or isinstance(publication_cycle, int)
            and publication_cycle >= 0,
            "FORCERFT_ACTIVE_PUBLICATION_CYCLE_INVALID",
        )
        return (
            base,
            residual,
            revision_id,
            publication_cycle,
            int(scheduling["active_actor_optimizer_step"]),
            int(scheduling["active_policy_epoch"]),
            str(scheduling["active_policy_epoch_status"]),
        )
    try:
        actor_step = int(revision_id.rsplit("-", 1)[1])
    except ValueError:
        actor_step = 0
    candidate = (
        resume_checkpoint.parent.parent
        / "policy_candidates"
        / online_adaptation_id
        / f"residual_actor_step_{actor_step:06d}"
    )
    residual = candidate
    if actor_step > 0:
        require(
            (residual / "residual_actor.pt").is_file()
            and (residual / "candidate_state.pt").is_file(),
            "FORCERFT_ACTIVE_RESIDUAL_CANDIDATE_MISSING",
        )
    else:
        residual = resume_checkpoint / "models/residual_actor.pt"
    return (
        base,
        residual.resolve(),
        revision_id,
        None,
        actor_step,
        0,
        "legacy_unknown",
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


class ResidualActorCriticLearner:
    """Three-learner_state learner over sealed real ACK replay only."""

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
        # Exact resume is checkpoint-authoritative.  The repository YAML is
        # validated by build_runtime(), never used to override this schedule.
        self.training_policy = self.learner["training_policy"]
        path = (
            warmup.DATASET / "normalizer_manifest.json"
            if normalizer_path is None
            else Path(normalizer_path).resolve()
        )
        self.normalizer = load_normalizer_manifest(path)
        self.replay: warmup.OnlineResidualReplay | None = None
        self.unique_r_count = 0
        self.r_macro_count = 0
        self.next_base_missing_rows = 0
        self.quarantined_current_schema_rows = 0
        self.nonzero_behavior_residual_rows = 0
        self.latest_residual_actor_output_norm = 0.0
        self.latest_replay_refresh_ms = 0.0
        self.latest_critic_update_ms = 0.0
        self.latest_actor_update_ms = 0.0
        self.latest_cycle_ms = 0.0
        self.latest_target_candidate_unavailable_count = 0
        self.latest_critic_td_available_count = 0
        self.latest_actor_q_mapping_unavailable_count = 0
        self.latest_human_residual_projected_count = 0
        self.latest_human_residual_valid_count = 0
        self.sampled_session_ids: set[str] = set()
        self.sampled_episode_ids: set[str] = set()
        self._state_lock = threading.RLock()
        self._last_saved_checkpoint_signature: str | None = None
        self._last_saved_checkpoint_path: Path | None = None
        self._loaded_episode_keys: set[str] = set()
        self._admission_progress: dict[str, dict[str, Any]] = {}
        self._expected_admission_id: str | None = None
        checkpoint_replay = self.learner["runtime"].get("replay", {})
        self._checkpoint_loaded_episode_keys = set(
            checkpoint_replay.get("loaded_episode_keys", ())
        )
        self._checkpoint_per_episode_critic_row_counts = dict(
            checkpoint_replay.get("per_episode_critic_row_counts", {})
        )
        runtime = self.learner["runtime"]
        runtime.setdefault("partial_cycle_q_updates", 0)
        completed = int(runtime.get("residual_actor_critic_cycles", 0))
        scheduling = runtime.setdefault("scheduling", {})
        scheduling.setdefault("mode", "continuous_async")
        scheduling.setdefault(
            "last_publish_attempt_cycle",
            completed - completed % self.training_policy.residual_candidate_interval_cycles,
        )
        scheduling.setdefault(
            "last_published_cycle", scheduling["last_publish_attempt_cycle"]
        )
        scheduling.setdefault("publication_event_count", 0)
        scheduling.setdefault(
            "last_periodic_checkpoint_cycle",
            completed - completed % self.training_policy.training_checkpoint_interval_cycles,
        )
        scheduling.setdefault("periodic_checkpoint_event_count", 0)
        scheduling.setdefault("last_saved_checkpoint_cycle", None)
        scheduling.setdefault("active_publication_cycle", None)
        scheduling.setdefault(
            "active_actor_optimizer_step",
            int(runtime["counters"].get("residual_actor_optimizer_steps", 0)),
        )
        scheduling.setdefault("active_policy_epoch", 0)
        scheduling.setdefault("active_policy_epoch_status", "legacy_unknown")
        scheduling.setdefault("pending_publication", None)
        scheduling.setdefault(
            "retired_admission_cycle_budgets",
            dict(checkpoint_replay.get("admission_cycle_budgets", {})),
        )
        scheduling["candidate_export_generation"] = (
            f"{time.time_ns()}-{os.getpid()}"
        )
        require(
            scheduling["mode"] == "continuous_async"
            and 0 <= int(runtime["partial_cycle_q_updates"])
            <= self.training_policy.twin_q_updates_per_cycle,
            "FORCERFT_CONTINUOUS_SCHEDULING_STATE_INVALID",
        )

    @property
    def residual_actor(self) -> torch.nn.Module:
        return self.learner["residual_actor"]

    def set_current_session(self, session_id: str) -> None:
        with self._state_lock:
            require(
                self.replay is None
                or not any(
                    row.get("session_id") == session_id
                    for row in self.replay.rows
                ),
                "ONLINE_REPLAY_ASYNC_CURRENT_EPISODE_ALREADY_IN_REPLAY",
            )
            self.current_session_id = session_id

    def clear_current_session(self) -> None:
        with self._state_lock:
            self.current_session_id = None

    def sampling_provenance(self, session_id: str) -> dict[str, Any]:
        with self._state_lock:
            rows = () if self.replay is None else self.replay.rows
            return {
                "current_episode_sampled": session_id in self.sampled_session_ids,
                "current_episode_replay_membership": any(
                    row.get("session_id") == session_id for row in rows
                ),
                "sampled_session_ids": sorted(self.sampled_session_ids),
            }

    def counter_snapshot(self) -> dict[str, int]:
        with self._state_lock:
            runtime = self.learner["runtime"]
            counters = runtime["counters"]
            scheduling = runtime["scheduling"]
            return {
                "completed_learner_cycles": int(
                    runtime["residual_actor_critic_cycles"]
                ),
                "partial_cycle_q_updates": int(
                    runtime.get("partial_cycle_q_updates", 0)
                ),
                "total_twin_q_optimizer_steps": int(
                    counters["twin_q_optimizer_steps"]
                ),
                "warmup_twin_q_optimizer_steps": int(
                    runtime.get("ack_critic_warmup_steps", 0)
                ),
                "joint_twin_q_optimizer_steps": int(
                    counters["twin_q_optimizer_steps"]
                ) - int(runtime.get("ack_critic_warmup_steps", 0)),
                "residual_actor_optimizer_steps": int(
                    counters["residual_actor_optimizer_steps"]
                ),
                "residual_actor_update_attempts": int(
                    counters["residual_actor_update_attempts"]
                ),
                "residual_actor_updates_skipped_no_gradient": int(
                    counters["residual_actor_updates_skipped_no_gradient"]
                ),
                "actor_parameter_publication_events": int(
                    scheduling.get("publication_event_count", 0)
                ),
                "periodic_checkpoint_events": int(
                    scheduling.get("periodic_checkpoint_event_count", 0)
                ),
            }

    def _record_sampled_batch(
        self, batch: warmup.ResidualTransitionBatch | None
    ) -> None:
        if batch is None:
            return
        with self._state_lock:
            if not hasattr(self, "sampled_session_ids"):
                self.sampled_session_ids = set()
                self.sampled_episode_ids = set()
            self.sampled_session_ids.update(
                value for value in batch.session_ids if value
            )
            self.sampled_episode_ids.update(batch.episode_ids)

    def _episode_signature(self) -> tuple[str, ...]:
        return tuple(
            path.stem
            for path in sorted((self.replay_root / "episodes").glob("*.json"))
        )

    def notify_admission(self, admission_id: str) -> None:
        require(
            bool(admission_id)
            and Path(admission_id).name == admission_id
            and not admission_id.endswith(".json"),
            "FORCERFT_ADMISSION_NOTIFICATION_ID_INVALID",
        )
        with self._state_lock:
            self._expected_admission_id = admission_id

    def outstanding_budget_status(self) -> dict[str, Any]:
        completed = int(
            self.learner["runtime"]["residual_actor_critic_cycles"]
        )
        return {
            "scheduling_mode": "continuous_async",
            "completed_cycle_count": completed,
            "recovery_budget_drain_required": False,
        }

    def recovery_preflight(self) -> dict[str, Any]:
        """Validate checkpoint replay identity without rebuilding retired debt."""

        self._refresh_replay()
        require(
            self._checkpoint_loaded_episode_keys.issubset(
                self._loaded_episode_keys
            ),
            "FORCERFT_RECOVERY_CHECKPOINT_EPISODE_MISSING",
        )
        for admission_id, count in (
            self._checkpoint_per_episode_critic_row_counts.items()
        ):
            require(
                admission_id in self._admission_progress
                and int(
                    self._admission_progress[admission_id][
                        "admitted_rows_for_latest_episode"
                    ]
                )
                == int(count),
                "FORCERFT_RECOVERY_REPLAY_ROW_COUNT_MISMATCH",
            )
        return self.outstanding_budget_status()

    def _latest_budget_metrics(self) -> dict[str, Any]:
        admission_id = self._expected_admission_id
        if admission_id is None and self._admission_progress:
            admission_id = next(reversed(self._admission_progress))
        status = (
            None
            if admission_id is None
            else self._admission_progress.get(admission_id)
        )
        return {
            "latest_observed_admission_id": (
                None if status is None else admission_id
            ),
            "latest_admitted_episode_key": (
                None if status is None else status["episode_key"]
            ),
            "admitted_rows_for_latest_episode": (
                0 if status is None else status["admitted_rows_for_latest_episode"]
            ),
            "recorded_rows_for_latest_episode": (
                0 if status is None else status["recorded_transition_rows"]
            ),
            "cycle_count_when_admission_observed": (
                0 if status is None else status["cycle_count_when_observed"]
            ),
            "replay_refresh_ms": self.latest_replay_refresh_ms,
            "latest_critic_update_ms": self.latest_critic_update_ms,
            "latest_actor_update_ms": self.latest_actor_update_ms,
            "latest_cycle_ms": self.latest_cycle_ms,
        }

    def _refresh_replay(self) -> warmup.OnlineResidualReplay:
        started = time.perf_counter()
        signature = self._episode_signature()
        current_keys = set(signature)
        require(
            self._loaded_episode_keys.issubset(current_keys),
            "FORCERFT_INCREMENTAL_REPLAY_EPISODE_REMOVED",
        )
        if self.replay is None:
            self.replay = warmup.OnlineResidualReplay((), self.normalizer)
        added_keys = [
            key for key in signature if key not in self._loaded_episode_keys
        ]
        for admission_id in added_keys:
            policy_rows, policy_macros, source_episodes, human_rows = (
                warmup.load_formal_online_episode(
                    self.replay_root, admission_id
                )
            )
            macros = (*policy_macros, *warmup.build_ack_macros(human_rows))
            with self._state_lock:
                if self.current_session_id is not None:
                    require(
                        not any(
                            row["identity"].get("session_id")
                            == self.current_session_id
                            for row in [*policy_rows, *human_rows]
                        ),
                        "ONLINE_REPLAY_ASYNC_CURRENT_EPISODE_ALREADY_IN_REPLAY",
                    )
                added_counts = self.replay.append_macros(macros)
                require(
                    len(source_episodes) == 1,
                    "FORCERFT_INCREMENTAL_REPLAY_EPISODE_ID_INVALID",
                )
                episode_id = next(iter(source_episodes))
                recorded_rows = int(added_counts.get(episode_id, 0))
                admitted_rows = self.replay.critic_td_rows_for_episode(
                    episode_id
                )
                require(
                    recorded_rows > 0,
                    "FORCERFT_INCREMENTAL_REPLAY_NO_RECORDED_ROWS",
                )
                self._loaded_episode_keys.add(admission_id)
                self._admission_progress[admission_id] = {
                    "episode_key": admission_id,
                    "recorded_transition_rows": recorded_rows,
                    "admitted_rows_for_latest_episode": admitted_rows,
                    "cycle_count_when_observed": int(
                        self.learner["runtime"][
                            "residual_actor_critic_cycles"
                        ]
                    ),
                }
                self.unique_r_count += len(policy_rows) + len(human_rows)
                self.r_macro_count += len(macros)
        if added_keys:
            self.latest_replay_refresh_ms = (
                time.perf_counter() - started
            ) * 1000.0
        with self._state_lock:
            self.next_base_missing_rows = self.replay.next_base_missing_rows
            self.quarantined_current_schema_rows = (
                self.replay.quarantined_current_schema_rows
            )
            self.nonzero_behavior_residual_rows = (
                self.replay.nonzero_behavior_residual_rows
            )
            runtime_replay = self.learner["runtime"]["replay"]
            runtime_replay.update(
                recorded_transition_rows=self.replay.recorded_transition_rows,
                critic_td_valid_rows=self.replay.critic_td_valid_rows,
                actor_q_valid_rows=self.replay.actor_q_valid_rows,
                human_residual_valid_rows=self.replay.human_residual_valid_rows,
                loaded_episode_keys=sorted(self._loaded_episode_keys),
                per_episode_critic_row_counts={
                    admission_id: int(
                        progress["admitted_rows_for_latest_episode"]
                    )
                    for admission_id, progress in self._admission_progress.items()
                },
                replay_generation=len(self._loaded_episode_keys),
            )
        return self.replay

    def _critic_residual_column_norm(self) -> float:
        if not all(name in self.learner for name in ("q1", "q2")):
            return 0.0
        columns = []
        for name in ("q1", "q2"):
            first_layer = self.learner[name].layers[0]
            columns.append(
                first_layer.weight[
                    :,
                    RESIDUAL_ACTION_OFFSET : (
                        RESIDUAL_ACTION_OFFSET + RESIDUAL_ACTION_WIDTH
                    ),
                ].detach().flatten()
            )
        return float(torch.cat(columns).norm().cpu())

    def _critic_update(
        self,
        coordinator: InferencePriorityCoordinator,
        replay: warmup.OnlineResidualReplay,
        *,
        warmup: bool,
    ) -> float | None:
        started = time.perf_counter()
        learner = self.learner
        counters = learner["runtime"]["counters"]
        step = int(counters["twin_q_optimizer_steps"])
        with coordinator.learner_step_slot(
            "ack_critic_warmup" if warmup else "critic"
        ):
            optimizer = learner["critic_optimizer"]
            optimizer.zero_grad(set_to_none=True)
            loss_result = None
            unavailable_count = 0
            seed = int(learner["config"]["environment"]["random_seed"]) + step
            for batch in replay.iter_td_batches(
                self.training_policy.twin_q_batch_size,
                device=self.device,
                seed=seed,
            ):
                candidate = residual_critic_loss(
                    learner["q1"],
                    learner["q2"],
                    learner["q1_target"],
                    learner["q2_target"],
                    learner["residual_actor_target"],
                    batch,
                    float(
                        learner["config"]["objective"]["command_macro_discount"]
                    ),
                    return_details=True,
                )
                unavailable_count += int(
                    candidate.target_candidate_unavailable_count
                )
                if candidate.td_valid_count:
                    loss_result = candidate
                    self._record_sampled_batch(batch)
                    break
            if loss_result is None:
                self.latest_target_candidate_unavailable_count = unavailable_count
                self.latest_critic_td_available_count = 0
                self.latest_critic_update_ms = (
                    time.perf_counter() - started
                ) * 1000.0
                return None
            loss_result.total.backward()
            torch.nn.utils.clip_grad_norm_(
                (*learner["q1"].parameters(), *learner["q2"].parameters()),
                float(learner["config"]["optimizer"]["twin_q"]["grad_clip_norm"]),
            )
            optimizer.step()
            tau = float(learner["config"]["optimizer"]["twin_q_polyak_tau"])
            polyak_update(learner["q1"], learner["q1_target"], tau)
            polyak_update(learner["q2"], learner["q2_target"], tau)
        with self._state_lock:
            counters["twin_q_optimizer_steps"] = step + 1
            counters["twin_q_target_update_steps"] = int(
                counters["twin_q_target_update_steps"]
            ) + 1
            if warmup:
                learner["runtime"]["ack_critic_warmup_steps"] = (
                    int(learner["runtime"].get("ack_critic_warmup_steps", 0)) + 1
                )
            else:
                learner["runtime"]["partial_cycle_q_updates"] = int(
                    learner["runtime"].get("partial_cycle_q_updates", 0)
                ) + 1
        self.latest_target_candidate_unavailable_count = int(unavailable_count)
        self.latest_critic_td_available_count = int(loss_result.td_valid_count)
        value = float(loss_result.total.detach())
        self.latest_critic_update_ms = (
            time.perf_counter() - started
        ) * 1000.0
        return value

    def _actor_update(
        self,
        coordinator: InferencePriorityCoordinator,
        replay: warmup.OnlineResidualReplay,
    ) -> dict[str, Any]:
        started = time.perf_counter()
        learner = self.learner
        counters = learner["runtime"]["counters"]
        step = int(counters["residual_actor_optimizer_steps"])
        seed = (
            int(learner["config"]["environment"]["random_seed"])
            + 1_000_000
            + step
        )
        policy_batch = replay.sample(
            self.training_policy.residual_policy_value_batch_size,
            device=self.device,
            seed=seed,
            policy_only=True,
            actor_q_valid_only=True,
        )
        human_batch = replay.sample(
            self.training_policy.human_residual_imitation_batch_size,
            device=self.device,
            seed=seed + 1,
            human_only=True,
        )
        require(
            policy_batch is not None or human_batch is not None,
            "FORCERFT_ACTOR_REPLAY_EMPTY",
        )
        self._record_sampled_batch(policy_batch)
        self._record_sampled_batch(human_batch)
        critic_parameters = (
            *learner["q1"].parameters(),
            *learner["q2"].parameters(),
        )
        with coordinator.learner_step_slot("actor"):
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
                        learner["config"]["objective"]["value_objective_weight"]
                    ),
                    residual_l2_weight=float(
                        learner["config"]["objective"]["residual_magnitude_penalty_weight"]
                    ),
                    human_residual_weight=float(
                        learner["config"]["objective"]["human_residual_imitation_weight"]
                    ),
                )
                losses.total.backward()
                actor_grad_norm_tensor = torch.nn.utils.clip_grad_norm_(
                    learner["residual_actor"].parameters(),
                    float(
                        learner["config"]["optimizer"]["residual_actor"]["grad_clip_norm"]
                    ),
                )
                require(
                    bool(torch.isfinite(actor_grad_norm_tensor).item()),
                    "FORCERFT_ACTOR_GRADIENT_NONFINITE",
                )
                actor_grad_norm = float(actor_grad_norm_tensor.detach().cpu())
                support_available = bool(
                    replay.human_residual_valid_rows > 0
                    or self._critic_residual_column_norm() > ACTOR_GRAD_EPSILON
                )
                actor_update_applied = bool(
                    support_available and actor_grad_norm > ACTOR_GRAD_EPSILON
                )
                if actor_update_applied:
                    optimizer.step()
                else:
                    optimizer.zero_grad(set_to_none=True)
            finally:
                for parameter in critic_parameters:
                    parameter.requires_grad_(True)
            if actor_update_applied:
                polyak_update(
                    learner["residual_actor"],
                    learner["residual_actor_target"],
                    float(learner["config"]["optimizer"]["twin_q_polyak_tau"]),
                )
        self.latest_residual_actor_output_norm = float(losses.output_norm.detach())
        self.latest_actor_q_mapping_unavailable_count = int(
            losses.actor_q_mapping_unavailable_count
        )
        self.latest_human_residual_projected_count = int(
            losses.human_residual_projected_count
        )
        self.latest_human_residual_valid_count = int(
            losses.human_residual_valid_count
        )
        self.latest_actor_update_ms = (
            time.perf_counter() - started
        ) * 1000.0
        return {
            "total": float(losses.total.detach()),
            "value": float(losses.value.detach()),
            "residual": float(losses.residual.detach()),
            "human": float(losses.human.detach()),
            "output_norm": self.latest_residual_actor_output_norm,
            "attempted": True,
            "applied": actor_update_applied,
            "skip_reason": (
                None if actor_update_applied else "no_effective_gradient"
            ),
            "grad_norm": actor_grad_norm,
            "support_available": support_available,
            "actor_q_mapping_unavailable_count": int(
                losses.actor_q_mapping_unavailable_count
            ),
            "human_residual_projected_count": int(
                losses.human_residual_projected_count
            ),
            "human_residual_valid_count": int(
                losses.human_residual_valid_count
            ),
        }

    def _actor_counter_metrics(self) -> dict[str, int]:
        counters = self.learner["runtime"]["counters"]
        applied = int(counters["residual_actor_optimizer_steps"])
        attempts = int(counters.get("residual_actor_update_attempts", applied))
        skipped = int(
            counters.get(
                "residual_actor_updates_skipped_no_gradient",
                attempts - applied,
            )
        )
        require(
            attempts == applied + skipped,
            "FORCERFT_RESIDUAL_ACTOR_UPDATE_COUNTER_MISMATCH",
        )
        return {
            "residual_actor_optimizer_steps": applied,
            "residual_actor_update_attempts": attempts,
            "residual_actor_updates_skipped_no_gradient": skipped,
        }

    def __call__(
        self, coordinator: InferencePriorityCoordinator
    ) -> dict[str, Any]:
        learner = self.learner
        runtime = learner["runtime"]
        replay = self._refresh_replay()
        count = replay.critic_td_valid_rows
        runtime["replay"]["recorded_transition_rows"] = int(
            getattr(replay, "recorded_transition_rows", count)
        )
        runtime["replay"]["critic_td_valid_rows"] = count
        if count < self.training_policy.minimum_ack_transitions:
            runtime["learner_state"] = "ack_replay_collection"
            return {
                "waiting_for_replay": True,
                "learner_state": "ack_replay_collection",
                "learner_critic_steps": 0,
                "learner_actor_steps": 0,
                "learner_polyak_steps": 0,
                "current_episode_sampled": False,
                "nonfinite_count": 0,
                "oom_count": 0,
                **self._actor_counter_metrics(),
                **self._latest_budget_metrics(),
            }
        if runtime["learner_state"] in {"ack_replay_collection", "ack_critic_warmup"}:
            runtime["learner_state"] = "ack_critic_warmup"
            before = {
                name: value.detach().clone()
                for name, value in learner["residual_actor"].state_dict().items()
            }
            losses = []
            remaining = (
                self.training_policy.ack_critic_warmup_steps
                - int(runtime.get("ack_critic_warmup_steps", 0))
            )
            for _ in range(max(0, remaining)):
                value = self._critic_update(coordinator, replay, warmup=True)
                if value is None:
                    break
                losses.append(value)
            require(
                all(
                    torch.equal(before[name], value)
                    for name, value in learner["residual_actor"].state_dict().items()
                ),
                "FORCERFT_WARMUP_MODIFIED_RESIDUAL_ACTOR",
            )
            warmup_complete = int(runtime["ack_critic_warmup_steps"]) == (
                self.training_policy.ack_critic_warmup_steps
            )
            if not warmup_complete:
                return {
                    "waiting_for_replay": True,
                    "waiting_for_mappable_td": True,
                    "learner_state": "ack_critic_warmup",
                    "ack_critic_warmup_complete": False,
                    "ack_critic_warmup_steps": int(
                        runtime["ack_critic_warmup_steps"]
                    ),
                    "learner_critic_steps": len(losses),
                    "learner_actor_steps": 0,
                    "learner_polyak_steps": len(losses),
                    "current_episode_sampled": False,
                    "nonfinite_count": 0,
                    "oom_count": 0,
                    **self._actor_counter_metrics(),
                    **self._latest_budget_metrics(),
                }
            runtime["learner_state"] = "residual_actor_critic_training"
            runtime["ack_critic_warmup_complete"] = True
            latest_checkpoint = (
                self.save_checkpoint()
                if self.training_policy.checkpoint_on_warmup_complete
                else None
            )
            return {
                "waiting_for_replay": False,
                "learner_state": "residual_actor_critic_training",
                "ack_critic_warmup_complete": True,
                "ack_critic_warmup_steps": int(runtime["ack_critic_warmup_steps"]),
                "learner_critic_steps": len(losses),
                "learner_actor_steps": 0,
                "learner_polyak_steps": len(losses),
                "residual_actor_critic_cycle": int(runtime["residual_actor_critic_cycles"]),
                **self._actor_counter_metrics(),
                "latest_critic_td_loss": losses[-1] if losses else None,
                "nonzero_behavior_residual_rows": int(
                    getattr(self, "nonzero_behavior_residual_rows", 0)
                ),
                "human_residual_valid_rows": int(
                    getattr(replay, "human_residual_valid_rows", 0)
                ),
                "critic_residual_column_norm": self._critic_residual_column_norm(),
                "residual_actor_output_norm": float(
                    getattr(self, "latest_residual_actor_output_norm", 0.0)
                ),
                "current_episode_sampled": False,
                "nonfinite_count": 0,
                "oom_count": 0,
                "latest_checkpoint_path": (
                    None
                    if latest_checkpoint is None
                    else str(latest_checkpoint)
                ),
                **self._latest_budget_metrics(),
            }

        require(
            runtime["learner_state"] == "residual_actor_critic_training"
            and runtime["ack_critic_warmup_complete"] is True,
            "FORCERFT_ONLINE_PHASE_INVALID",
        )
        cycle = int(runtime["residual_actor_critic_cycles"])
        cycle_started = time.perf_counter()
        critic_losses = []
        partial_q_updates = int(runtime.get("partial_cycle_q_updates", 0))
        for _ in range(
            partial_q_updates, self.training_policy.twin_q_updates_per_cycle
        ):
            value = self._critic_update(coordinator, replay, warmup=False)
            if value is None:
                return {
                    "waiting_for_replay": True,
                    "waiting_for_mappable_td": True,
                    "learner_state": "residual_actor_critic_training",
                    "learner_critic_steps": len(critic_losses),
                    "learner_actor_steps": 0,
                    "learner_polyak_steps": len(critic_losses),
                    "partial_cycle_q_updates": int(
                        runtime["partial_cycle_q_updates"]
                    ),
                    "current_episode_sampled": False,
                    "nonfinite_count": 0,
                    "oom_count": 0,
                    **self._actor_counter_metrics(),
                    **self._latest_budget_metrics(),
                }
            critic_losses.append(value)
        actor_metrics = self._actor_update(coordinator, replay)
        with self._state_lock:
            counters = runtime["counters"]
            attempts = int(counters["residual_actor_update_attempts"]) + 1
            counters["residual_actor_update_attempts"] = attempts
            if actor_metrics["applied"]:
                counters["residual_actor_optimizer_steps"] = int(
                    counters["residual_actor_optimizer_steps"]
                ) + 1
            else:
                counters["residual_actor_updates_skipped_no_gradient"] = int(
                    counters["residual_actor_updates_skipped_no_gradient"]
                ) + 1
            runtime["residual_actor_critic_cycles"] = cycle + 1
            runtime["partial_cycle_q_updates"] = 0
            learner["residual_actor_critic_cycles"] = cycle + 1
        self.latest_cycle_ms = (time.perf_counter() - cycle_started) * 1000.0
        return {
            "waiting_for_replay": False,
            "learner_state": "residual_actor_critic_training",
            "learner_critic_steps": len(critic_losses),
            "learner_actor_steps": int(actor_metrics["applied"]),
            "learner_actor_update_attempts": 1,
            "learner_polyak_steps": len(critic_losses),
            "current_episode_sampled": False,
            "nonfinite_count": 0,
            "oom_count": 0,
            "residual_actor_critic_cycle": cycle + 1,
            **self._actor_counter_metrics(),
            "actor_update_attempted": True,
            "actor_update_applied": bool(actor_metrics["applied"]),
            "actor_update_skip_reason": actor_metrics["skip_reason"],
            "actor_grad_norm": actor_metrics["grad_norm"],
            "actor_support_available": actor_metrics["support_available"],
            "latest_critic_td_loss": (
                critic_losses[-1] if critic_losses else None
            ),
            "latest_actor_loss": actor_metrics["total"],
            "latest_min_twin_q": -actor_metrics["value"],
            "target_candidate_mapping_unavailable_count": int(
                getattr(self, "latest_target_candidate_unavailable_count", 0)
            ),
            "actor_q_mapping_unavailable_count": int(
                actor_metrics.get("actor_q_mapping_unavailable_count", 0)
            ),
            "human_residual_projected_count": int(
                actor_metrics.get("human_residual_projected_count", 0)
            ),
            "human_residual_projection_denominator": int(
                actor_metrics.get("human_residual_valid_count", 0)
            ),
            "nonzero_behavior_residual_rows": int(
                getattr(self, "nonzero_behavior_residual_rows", 0)
            ),
            "human_residual_valid_rows": int(
                getattr(replay, "human_residual_valid_rows", 0)
            ),
            "critic_residual_column_norm": self._critic_residual_column_norm(),
            "residual_actor_output_norm": actor_metrics.get(
                "output_norm",
                float(getattr(self, "latest_residual_actor_output_norm", 0.0)),
            ),
            "partial_cycle_q_updates": 0,
            **self._latest_budget_metrics(),
        }

    def export_actor_candidate(
        self,
        completed_cycle: int,
        coordinator: InferencePriorityCoordinator | None = None,
    ) -> dict[str, Any]:
        runtime = self.learner["runtime"]
        counters = runtime["counters"]
        residual_actor_optimizer_steps = int(
            counters["residual_actor_optimizer_steps"]
        )
        current = str(runtime["active_residual_policy_revision"])
        task_id = current.split("-residual-policy-", 1)[0]
        generation = str(
            runtime["scheduling"]["candidate_export_generation"]
        )
        revision_id = (
            f"{task_id}-residual-policy-cycle-{completed_cycle:06d}"
            f"-g{generation.split('-', 1)[0]}"
        )
        snapshot_slot = (
            nullcontext()
            if coordinator is None
            else coordinator.learner_step_slot(
                "actor_snapshot",
                initial_estimate_s=0.02,
                coverage_reserve_s=0.05,
                historical_estimate_cap_s=0.20,
            )
        )
        with snapshot_slot:
            with self._state_lock:
                state = {
                    name: tensor.detach().cpu().clone()
                    for name, tensor in self.learner[
                        "residual_actor"
                    ].state_dict().items()
                }
        actor_sha256 = _residual_actor_state_sha256(state)
        candidate_root = (
            self.checkpoint_root.parent
            / "policy_candidates"
            / str(runtime["online_adaptation_id"])
        )
        destination = candidate_root / "blobs" / actor_sha256
        if destination.exists():
            require(
                (destination / "residual_actor.pt").is_file()
                and (destination / "candidate_state.pt").is_file(),
                "FORCERFT_RESIDUAL_CANDIDATE_BLOB_INVALID",
            )
            stored_state = torch.load(
                destination / "residual_actor.pt",
                map_location="cpu",
                weights_only=True,
            )
            stored_metadata = torch.load(
                destination / "candidate_state.pt",
                map_location="cpu",
                weights_only=False,
            )
            require(
                isinstance(stored_state, Mapping)
                and _residual_actor_state_sha256(stored_state) == actor_sha256
                and stored_metadata.get("checkpoint_kind")
                == CANDIDATE_CHECKPOINT_KIND
                and stored_metadata.get("online_semantics_version")
                == ONLINE_SEMANTICS_VERSION
                and stored_metadata.get("actor_content_sha256")
                == actor_sha256,
                "FORCERFT_RESIDUAL_CANDIDATE_BLOB_INVALID",
            )
        else:
            temporary = destination.with_name(
                f".{destination.name}.writing-{os.getpid()}"
            )
            temporary.mkdir(parents=True, exist_ok=False)
            try:
                torch.save(state, temporary / "residual_actor.pt")
                torch.save(
                    {
                        "checkpoint_kind": CANDIDATE_CHECKPOINT_KIND,
                        "online_semantics_version": ONLINE_SEMANTICS_VERSION,
                        "actor_content_sha256": actor_sha256,
                    },
                    temporary / "candidate_state.pt",
                )
                destination.parent.mkdir(parents=True, exist_ok=True)
                os.replace(temporary, destination)
            except BaseException:
                if temporary.exists():
                    import shutil

                    shutil.rmtree(temporary)
                raise
        event = {
            "schema": "forcesmolvla-residual-publication-event-v1",
            "revision_id": revision_id,
            "checkpoint": str(destination.resolve()),
            "residual_actor_critic_cycle": int(completed_cycle),
            "residual_actor_optimizer_steps": int(residual_actor_optimizer_steps),
            "actor_content_sha256": actor_sha256,
            "candidate_export_generation": generation,
        }
        event_root = candidate_root / "publications" / generation
        event_root.mkdir(parents=True, exist_ok=True)
        event_path = event_root / f"cycle_{completed_cycle:06d}.json"
        temporary_event = event_path.with_suffix(f".writing-{os.getpid()}")
        temporary_event.write_text(
            json.dumps(event, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(temporary_event, event_path)
        with self._state_lock:
            scheduling = runtime["scheduling"]
            scheduling["last_publish_attempt_cycle"] = int(completed_cycle)
            scheduling["last_published_cycle"] = int(completed_cycle)
            scheduling["publication_event_count"] = int(
                scheduling.get("publication_event_count", 0)
            ) + 1
            scheduling["pending_publication"] = dict(event)
        return {**event, "checkpoint": destination.resolve()}

    def mark_active_residual_policy_revision(
        self,
        revision_id: str,
        *,
        publication_cycle: int,
        actor_optimizer_step: int,
        policy_epoch: int,
        actor_checkpoint: Path,
    ) -> None:
        with self._state_lock:
            runtime = self.learner["runtime"]
            runtime["active_residual_policy_revision"] = str(revision_id)
            scheduling = runtime["scheduling"]
            scheduling["active_publication_cycle"] = int(publication_cycle)
            scheduling["active_actor_optimizer_step"] = int(actor_optimizer_step)
            scheduling["active_actor_checkpoint"] = str(
                Path(actor_checkpoint).resolve()
            )
            scheduling["active_policy_epoch"] = int(policy_epoch)
            scheduling["active_policy_epoch_status"] = "known"
            scheduling["pending_publication"] = None

    def save_checkpoint(self) -> Path:
        with self._state_lock:
            learner = self.learner
            completed = int(learner["runtime"]["residual_actor_critic_cycles"])
            target = training_checkpoint_path(self.checkpoint_root, completed)
            learner["runtime"]["scheduling"][
                "last_saved_checkpoint_cycle"
            ] = completed
            runtime_state = deepcopy(learner["runtime"])
            signature = hashlib.sha256(
                json.dumps(
                    runtime_state,
                    default=str,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest()
            if (
                signature == self._last_saved_checkpoint_signature
                and target == self._last_saved_checkpoint_path
                and exact_resume_checkpoint_is_recoverable(
                    target, expected_kind=TRAINING_CHECKPOINT_KIND
                )
            ):
                return target
            runtime_state["checkpoint_kind"] = TRAINING_CHECKPOINT_KIND
            save_residual_actor_critic_checkpoint(
                target,
                residual_actor=learner["residual_actor"],
                residual_actor_target=learner["residual_actor_target"],
                q1=learner["q1"],
                q2=learner["q2"],
                q1_target=learner["q1_target"],
                q2_target=learner["q2_target"],
                residual_actor_optimizer=learner["residual_actor_optimizer"],
                critic_optimizer=learner["critic_optimizer"],
                runtime_state=runtime_state,
                config=learner["config"],
            )
            self._last_saved_checkpoint_signature = signature
            self._last_saved_checkpoint_path = target
        retain_latest_training_checkpoints(
            self.checkpoint_root,
            keep=self.training_policy.retained_training_checkpoint_count,
        )
        return target


class AsyncResidualActorCriticRuntime:
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
        learner_job: ResidualActorCriticLearner,
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
        self.frozen_base_policy_checkpoint = Path(
            engine.metadata.get("checkpoint", active_actor_checkpoint)
        ).resolve()
        self.learner_resume_checkpoint = learner_resume_checkpoint.resolve()
        self.online_checkpoint_root = online_checkpoint_root.resolve()
        self.learner_job = learner_job
        self.inference_stream = inference_stream
        self.coordinator = InferencePriorityCoordinator()
        self._lock = threading.Condition()
        self._episode_active = False
        self._learner_started = False
        self._learner_worker_state = "ready"
        self._learner_result: dict[str, Any] = {}
        self._learner_error: str | None = None
        self._stop_learner = threading.Event()
        self._wake_learner = threading.Event()
        self._broadcast_count = 0
        self._candidate_count = 0
        self._candidate_checkpoints: dict[str, Path] = {}
        self._candidate_online_cycles: dict[str, int] = {}
        self._candidate_actor_steps: dict[str, int] = {}
        self._active_actor_online_cycle = active_actor_online_cycle
        self._policy = getattr(
            learner_job, "training_policy", ResidualActorCriticSchedule()
        )
        self._learner_thread: threading.Thread | None = None
        self._inference_request_count = 0
        self._actor_alive: AbstractContextManager[Any] | None = None
        self._pin: PinnedEpisode | None = None
        self._quiesce_save_lock = threading.Lock()
        self._quiesced_checkpoint: Path | None = None
        self._quiesced = False
        self._admission_resolution_required = False
        self._last_registered_admission_id: str | None = None
        self._last_training_log_monotonic = 0.0
        self._capture_window: dict[str, Any] | None = None
        self._last_capture_window: dict[str, Any] | None = None
        recovery_preflight = getattr(
            self.learner_job, "recovery_preflight", None
        )
        recovery = (
            dict(recovery_preflight())
            if callable(recovery_preflight)
            else {"scheduling_mode": "continuous_async", "completed_cycle_count": 0,
                  "recovery_budget_drain_required": False}
        )
        self._recovery_preflight = recovery
        self._learner_result.update(recovery)
        self._restore_pending_publication()
        self._start_learner()

    def _restore_pending_publication(self) -> None:
        runtime = getattr(self.learner_job, "learner", {}).get("runtime", {})
        pending = runtime.get("scheduling", {}).get("pending_publication")
        if not isinstance(pending, Mapping):
            return
        revision_id = str(pending.get("revision_id", ""))
        checkpoint = Path(str(pending.get("checkpoint", ""))).resolve()
        require(
            revision_id
            and revision_id != self.active_revision_id
            and (checkpoint / "residual_actor.pt").is_file()
            and (checkpoint / "candidate_state.pt").is_file(),
            "FORCERFT_PENDING_PUBLICATION_DEPENDENCY_INVALID",
        )
        self.machine.register_candidate(revision_id, self.active_model_revision)
        self.machine.stage(revision_id)
        self._candidate_checkpoints[revision_id] = checkpoint
        self._candidate_online_cycles[revision_id] = int(
            pending["residual_actor_critic_cycle"]
        )
        self._candidate_actor_steps[revision_id] = int(
            pending["residual_actor_optimizer_steps"]
        )

    def _learner_counter_snapshot(self) -> dict[str, int]:
        snapshot = getattr(self.learner_job, "counter_snapshot", None)
        if callable(snapshot):
            return dict(snapshot())
        runtime = getattr(self.learner_job, "learner", {}).get("runtime", {})
        counters = runtime.get("counters", {})
        scheduling = runtime.get("scheduling", {})
        actor = self._residual_actor_update_counters()
        return {
            "completed_learner_cycles": int(
                runtime.get(
                    "residual_actor_critic_cycles",
                    self._learner_result.get("residual_actor_critic_cycle", 0),
                )
            ),
            "partial_cycle_q_updates": int(
                runtime.get("partial_cycle_q_updates", 0)
            ),
            "total_twin_q_optimizer_steps": int(
                counters.get("twin_q_optimizer_steps", 0)
            ),
            "warmup_twin_q_optimizer_steps": int(
                runtime.get("ack_critic_warmup_steps", 0)
            ),
            "joint_twin_q_optimizer_steps": int(
                counters.get("twin_q_optimizer_steps", 0)
            ) - int(runtime.get("ack_critic_warmup_steps", 0)),
            "actor_parameter_publication_events": int(
                scheduling.get("publication_event_count", 0)
            ),
            "periodic_checkpoint_events": int(
                scheduling.get("periodic_checkpoint_event_count", 0)
            ),
            **actor,
        }

    @staticmethod
    def _counter_delta(
        start: Mapping[str, int], end: Mapping[str, int]
    ) -> dict[str, int]:
        return {name: int(end[name]) - int(value) for name, value in start.items()}

    def _capture_window_snapshot(self) -> dict[str, Any] | None:
        window = self._capture_window or self._last_capture_window
        if window is None:
            return None
        if window.get("finalized") is True:
            return dict(window)
        end = self._learner_counter_snapshot()
        provenance_reader = getattr(
            self.learner_job, "sampling_provenance", None
        )
        provenance = (
            dict(provenance_reader(window["session_id"]))
            if callable(provenance_reader)
            else {
                "current_episode_sampled": False,
                "current_episode_replay_membership": False,
                "sampled_session_ids": [],
            }
        )
        return {
            **window,
            "end": end,
            "delta": self._counter_delta(window["start"], end),
            **provenance,
        }

    def _residual_actor_update_counters(self) -> dict[str, int]:
        counters = (
            getattr(self.learner_job, "learner", {})
            .get("runtime", {})
            .get("counters", {})
        )
        applied = int(
            counters.get(
                "residual_actor_optimizer_steps",
                self._learner_result.get("residual_actor_optimizer_steps", 0),
            )
        )
        attempts = int(
            counters.get(
                "residual_actor_update_attempts",
                self._learner_result.get(
                    "residual_actor_update_attempts", applied
                ),
            )
        )
        skipped = int(
            counters.get(
                "residual_actor_updates_skipped_no_gradient",
                self._learner_result.get(
                    "residual_actor_updates_skipped_no_gradient",
                    attempts - applied,
                ),
            )
        )
        require(
            attempts == applied + skipped,
            "FORCERFT_RESIDUAL_ACTOR_UPDATE_COUNTER_MISMATCH",
        )
        return {
            "residual_actor_optimizer_steps": applied,
            "residual_actor_update_attempts": attempts,
            "residual_actor_updates_skipped_no_gradient": skipped,
        }

    @property
    def metadata(self) -> dict[str, Any]:
        learner_runtime = getattr(self.learner_job, "learner", {}).get(
            "runtime", {}
        )
        return {
            **self.engine.metadata,
            "online_residual_actor_critic": True,
            "server_persistent": True,
            "current_episode_sampling": False,
            "minimum_ack_transitions": self._policy.minimum_ack_transitions,
            "scheduling_mode": self._policy.scheduling_mode,
            "residual_candidate_interval_cycles": (
                self._policy.residual_candidate_interval_cycles
            ),
            "active_actor_online_cycle": self._active_actor_online_cycle,
            "active_policy_epoch_status": scheduling.get(
                "active_policy_epoch_status", "legacy_unknown"
            ),
            "training_checkpoint_interval_cycles": self._policy.training_checkpoint_interval_cycles,
            "retained_training_checkpoint_count": self._policy.retained_training_checkpoint_count,
            "learner_state": learner_runtime.get("learner_state", "ack_replay_collection"),
            "ack_critic_warmup_complete": bool(
                learner_runtime.get("ack_critic_warmup_complete", False)
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
            "frozen_base_policy_checkpoint": str(self.frozen_base_policy_checkpoint),
            "pending_actor_revision": self.machine.pending_revision_id,
            "actor_candidate_count": self._candidate_count,
            "policy_epoch": int(self.machine.policy_epoch),
            "online_checkpoint_root": str(self.online_checkpoint_root),
            "recovery_budget_drain_required": False,
        }

    def status(self) -> dict[str, Any]:
        with self._lock:
            result = dict(self._learner_result)
            actor_counters = self._residual_actor_update_counters()
            learner_counters = self._learner_counter_snapshot()
            capture_window = self._capture_window_snapshot()
            learner_runtime = getattr(self.learner_job, "learner", {}).get(
                "runtime", {}
            )
            scheduling = learner_runtime.get("scheduling", {})
            return {
                "online_residual_actor_critic": True,
                "server_persistent": True,
                "current_episode_sampling": False,
                "runtime_session_id": self.session_id,
                "runtime_episode_id": self.episode_id,
                "episode_active": self._episode_active,
                "actor_revision_pinned": self._episode_active,
                "active_actor_revision": self.active_revision_id,
                "active_actor_model_revision": self.active_model_revision,
                "active_actor_checkpoint": str(self.active_actor_checkpoint),
                "frozen_base_policy_checkpoint": str(self.frozen_base_policy_checkpoint),
                "pending_actor_revision": self.machine.pending_revision_id,
                "actor_candidate_count": self._candidate_count,
                "policy_epoch": int(self.machine.policy_epoch),
                "active_policy_epoch_status": scheduling.get(
                    "active_policy_epoch_status", "legacy_unknown"
                ),
                "learner_started": self._learner_started,
                "learner_worker_state": self._learner_worker_state,
                "learner_resume_checkpoint": str(self.learner_resume_checkpoint),
                "learner_critic_steps": int(result.get("learner_critic_steps", 0)),
                "learner_actor_steps": int(result.get("learner_actor_steps", 0)),
                "learner_polyak_steps": int(result.get("learner_polyak_steps", 0)),
                "current_episode_sampled": bool(
                    capture_window is not None
                    and capture_window["current_episode_sampled"]
                ),
                "current_episode_replay_membership": bool(
                    capture_window is not None
                    and capture_window["current_episode_replay_membership"]
                ),
                "capture_window": capture_window,
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
                "active_actor_optimizer_step": int(
                    scheduling.get("active_actor_optimizer_step", 0)
                ),
                "last_publish_attempt_cycle": int(
                    scheduling.get("last_publish_attempt_cycle", 0)
                ),
                "last_published_cycle": scheduling.get(
                    "last_published_cycle"
                ),
                "last_periodic_checkpoint_cycle": int(
                    scheduling.get("last_periodic_checkpoint_cycle", 0)
                ),
                "last_checkpoint_cycle": scheduling.get(
                    "last_saved_checkpoint_cycle"
                ),
                "actor_parameter_publication_count": int(
                    scheduling.get("publication_event_count", 0)
                ),
                "residual_actor_critic_cycle": learner_counters[
                    "completed_learner_cycles"
                ],
                **learner_counters,
                **actor_counters,
                "actor_update_attempted": bool(
                    result.get("actor_update_attempted", False)
                ),
                "actor_update_applied": bool(
                    result.get("actor_update_applied", False)
                ),
                "actor_update_skip_reason": result.get(
                    "actor_update_skip_reason"
                ),
                "actor_grad_norm": result.get("actor_grad_norm"),
                "actor_support_available": bool(
                    result.get("actor_support_available", False)
                ),
                "learner_state": result.get(
                    "learner_state", learner_runtime.get("learner_state", "ack_replay_collection")
                ),
                "ack_critic_warmup_complete": bool(
                    learner_runtime.get("ack_critic_warmup_complete", False)
                ),
                "ack_critic_warmup_steps": int(
                    learner_runtime.get("ack_critic_warmup_steps", 0)
                ),
                "latest_critic_td_loss": result.get("latest_critic_td_loss"),
                "waiting_for_mappable_td": bool(
                    result.get("waiting_for_mappable_td", False)
                ),
                "recorded_transition_rows": int(
                    learner_runtime.get("replay", {}).get(
                        "recorded_transition_rows", 0
                    )
                ),
                "critic_td_valid_rows": int(
                    learner_runtime.get("replay", {}).get(
                        "critic_td_valid_rows", 0
                    )
                ),
                "latest_critic_td_available_count": int(
                    getattr(
                        self.learner_job,
                        "latest_critic_td_available_count",
                        0,
                    )
                ),
                "latest_actor_loss": result.get("latest_actor_loss"),
                "latest_min_twin_q": result.get("latest_min_twin_q"),
                "nonzero_behavior_residual_rows": int(
                    result.get(
                        "nonzero_behavior_residual_rows",
                        getattr(
                            self.learner_job,
                            "nonzero_behavior_residual_rows",
                            0,
                        ),
                    )
                ),
                "human_residual_valid_rows": int(
                    result.get(
                        "human_residual_valid_rows",
                        learner_runtime.get("replay", {}).get(
                            "human_residual_valid_rows", 0
                        ),
                    )
                ),
                "critic_residual_column_norm": result.get(
                    "critic_residual_column_norm"
                ),
                "residual_actor_output_norm": result.get(
                    "residual_actor_output_norm", 0.0
                ),
                "target_candidate_mapping_unavailable_count": int(
                    result.get(
                        "target_candidate_mapping_unavailable_count",
                        getattr(
                            self.learner_job,
                            "latest_target_candidate_unavailable_count",
                            0,
                        ),
                    )
                ),
                "actor_q_mapping_unavailable_count": int(
                    result.get(
                        "actor_q_mapping_unavailable_count",
                        getattr(
                            self.learner_job,
                            "latest_actor_q_mapping_unavailable_count",
                            0,
                        ),
                    )
                ),
                "human_residual_projected_count": int(
                    result.get(
                        "human_residual_projected_count",
                        getattr(
                            self.learner_job,
                            "latest_human_residual_projected_count",
                            0,
                        ),
                    )
                ),
                "human_residual_projection_denominator": int(
                    result.get(
                        "human_residual_projection_denominator",
                        getattr(
                            self.learner_job,
                            "latest_human_residual_valid_count",
                            0,
                        ),
                    )
                ),
                "quarantined_current_schema_rows": int(
                    getattr(
                        self.learner_job,
                        "quarantined_current_schema_rows",
                        0,
                    )
                ),
                "next_base_missing_rows": int(
                    getattr(self.learner_job, "next_base_missing_rows", 0)
                ),
                "latest_observed_admission_id": result.get(
                    "latest_observed_admission_id"
                ),
                "latest_admitted_episode_key": result.get(
                    "latest_admitted_episode_key"
                ),
                "admitted_rows_for_latest_episode": int(
                    result.get("admitted_rows_for_latest_episode", 0)
                ),
                "cycle_count_when_admission_observed": int(
                    result.get("cycle_count_when_admission_observed", 0)
                ),
                "replay_refresh_ms": float(
                    result.get("replay_refresh_ms", 0.0)
                ),
                "latest_critic_update_ms": float(
                    result.get("latest_critic_update_ms", 0.0)
                ),
                "latest_actor_update_ms": float(
                    result.get("latest_actor_update_ms", 0.0)
                ),
                "latest_cycle_ms": float(
                    result.get("latest_cycle_ms", 0.0)
                ),
                "admission_resolution_required": self._admission_resolution_required,
                "drain_in_progress": False,
                "recovery_budget_drain_required": False,
                **self.coordinator.status_snapshot(),
            }

    def start_episode(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        self._validate_identity(payload)
        with self._lock:
            require(
                not self._stop_learner.is_set() and not self._quiesced,
                "ONLINE_REPLAY_ASYNC_RUNTIME_QUIESCED",
            )
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
            self._capture_window = {
                "schema": "forcesmolvla-continuous-learner-capture-window-v1",
                "session_id": self.session_id,
                "episode_id": self.episode_id,
                "pinned_actor_revision": self.active_revision_id,
                "pinned_actor_model_revision": self.active_model_revision,
                "pinned_actor_publication_cycle": self._active_actor_online_cycle,
                "pinned_actor_policy_epoch": int(self.machine.policy_epoch),
                "start": self._learner_counter_snapshot(),
            }
            self._episode_active = True
        return self.status()

    def prepare_episode(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        session_id = str(payload.get("session_id", ""))
        episode_id = str(payload.get("episode_id", ""))
        require(bool(session_id and episode_id), "ONLINE_REPLAY_ASYNC_CAPTURE_IDENTITY_MISMATCH")
        with self._lock:
            require(
                not self._stop_learner.is_set() and not self._quiesced,
                "ONLINE_REPLAY_ASYNC_RUNTIME_QUIESCED",
            )
            require(
                not self._episode_active,
                "ONLINE_REPLAY_ASYNC_EPISODE_ALREADY_ACTIVE",
            )
            require(
                self._learner_worker_state != "failed",
                "ONLINE_REPLAY_ASYNC_LEARNER_FAILED",
            )
            require(
                not self._admission_resolution_required,
                "FORCERFT_PREPARE_EPISODE_BEFORE_ADMISSION_RESOLUTION",
            )
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
                self._candidate_checkpoints.pop(pending, None)
                self._candidate_online_cycles.pop(pending, None)
                self._candidate_actor_steps.pop(pending, None)
            # Capture identity remains bound to the immutable frozen base Actor.
            # Residual revisions are tracked by revision_id, not a new digest.
            self.machine.register_candidate(revision_id, self.active_model_revision)
            self.machine.stage(revision_id)
            self._candidate_checkpoints[revision_id] = checkpoint
            self._candidate_online_cycles[revision_id] = int(
                candidate["residual_actor_critic_cycle"]
            )
            self._candidate_actor_steps[revision_id] = int(
                candidate["residual_actor_optimizer_steps"]
            )
            self._candidate_count += 1
        print(
            "[residual-candidate] staged "
            f"revision={revision_id} checkpoint={checkpoint}",
            flush=True,
        )

    def _activate_pending_actor_locked(self) -> None:
        pending = self.machine.pending_revision_id
        if pending is None:
            return
        checkpoint = self._candidate_checkpoints[pending]
        activated = self.machine.activate_pending_at_episode_boundary()
        with self.engine._residual_lock:
            _load_residual_checkpoint(self.engine.residual_actor, checkpoint)
        self.engine.reset_residual_episode_context()
        self.engine.metadata["active_policy_epoch_status"] = "known"
        self.active_revision_id = activated.revision_id
        self.active_actor_checkpoint = checkpoint
        self._active_actor_online_cycle = self._candidate_online_cycles.pop(
            activated.revision_id
        )
        actor_optimizer_step = self._candidate_actor_steps.pop(
            activated.revision_id
        )
        self._broadcast_count += 1
        self.learner_job.mark_active_residual_policy_revision(
            activated.revision_id,
            publication_cycle=int(self._active_actor_online_cycle),
            actor_optimizer_step=actor_optimizer_step,
            policy_epoch=int(self.machine.policy_epoch),
            actor_checkpoint=checkpoint,
        )
        print(
            "[residual-activation] activated at episode boundary "
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

    def _process_completed_cycle_events(
        self, result: dict[str, Any]
    ) -> tuple[bool, bool]:
        cycle = int(result["residual_actor_critic_cycle"])
        scheduling = self.learner_job.learner["runtime"]["scheduling"]
        published = False
        checkpointed = False
        if self._policy.candidate_due(
            cycle,
            last_publish_attempt_cycle=int(
                scheduling["last_publish_attempt_cycle"]
            ),
        ):
            try:
                candidate = self.learner_job.export_actor_candidate(
                    cycle, self.coordinator
                )
            except RuntimeError as error:
                if (
                    not self._stop_learner.is_set()
                    or str(error) != "ONLINE_REPLAY_ASYNC_LEARNER_CANCELLED"
                ):
                    raise
                candidate = self.learner_job.export_actor_candidate(cycle)
            self._stage_actor_candidate(candidate)
            published = True
        if (
            self._policy.checkpoint_due(cycle)
            and cycle > int(scheduling["last_periodic_checkpoint_cycle"])
        ):
            scheduling["last_periodic_checkpoint_cycle"] = cycle
            scheduling["periodic_checkpoint_event_count"] = int(
                scheduling.get("periodic_checkpoint_event_count", 0)
            ) + 1
            checkpoint_path = self.learner_job.save_checkpoint()
            result["latest_checkpoint_path"] = str(checkpoint_path)
            checkpointed = True
            print(
                "[training-checkpoint] periodic "
                f"cycle={cycle} checkpoint={checkpoint_path}",
                flush=True,
            )
        return published, checkpointed

    def _start_learner(self) -> None:
        with self._lock:
            if self._learner_started:
                return
            self._learner_started = True
            self._learner_worker_state = "running"

        def run() -> None:
            try:
                with self.coordinator.worker_alive("learner"):
                    while not self._stop_learner.is_set():
                        result = dict(self.learner_job(self.coordinator))
                        if result.get("waiting_for_replay"):
                            with self._lock:
                                self._learner_result = result
                                self._learner_worker_state = "waiting_for_replay"
                                self._lock.notify_all()
                            self._wake_learner.wait(0.5)
                            self._wake_learner.clear()
                            continue
                        cycle = int(result["residual_actor_critic_cycle"])
                        published, checkpointed = (
                            self._process_completed_cycle_events(result)
                        )
                        label = (
                            "residual-training"
                            if result.get("actor_update_attempted")
                            else "critic-warmup"
                        )
                        now = time.monotonic()
                        if (
                            label == "critic-warmup"
                            or published
                            or checkpointed
                            or cycle % 25 == 0
                            or now - self._last_training_log_monotonic >= 5.0
                        ):
                            print(
                                f"[{label}] cycle={cycle} "
                                f"nonzero_behavior_residual_rows="
                                f"{result.get('nonzero_behavior_residual_rows', 0)} "
                                f"human_residual_valid_rows="
                                f"{result.get('human_residual_valid_rows', 0)} "
                                f"critic_residual_column_norm="
                                f"{result.get('critic_residual_column_norm')} "
                                f"actor_grad_norm="
                                f"{result.get('actor_grad_norm')} "
                                f"actor_update_applied="
                                f"{result.get('actor_update_applied')} "
                                f"actor_update_skip_reason="
                                f"{result.get('actor_update_skip_reason')} "
                                f"residual_actor_output_norm="
                                f"{result.get('residual_actor_output_norm', 0.0)}",
                                flush=True,
                            )
                            self._last_training_log_monotonic = now
                        result.pop("learner_actor", None)
                        with self._lock:
                            self._learner_result = result
                            self._learner_worker_state = "running"
                            self._lock.notify_all()
            except Exception as error:
                with self._lock:
                    if (
                        self._stop_learner.is_set()
                        and str(error) == "ONLINE_REPLAY_ASYNC_LEARNER_CANCELLED"
                    ):
                        self._learner_worker_state = "stopped"
                    else:
                        self._learner_error = f"{type(error).__name__}:{error}"
                        self._learner_worker_state = "failed"
                    self._lock.notify_all()
            finally:
                with self._lock:
                    if self._learner_worker_state != "failed":
                        self._learner_worker_state = "stopped"
                    self._lock.notify_all()

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

    def residual_decision(self, request: dict[str, Any]) -> dict[str, Any]:
        """Run the small CPU Actor without waiting for frozen-VLA inference."""

        require(self._episode_active, "ONLINE_REPLAY_ASYNC_EPISODE_INACTIVE")
        control_policy_epoch = request.get("control_policy_epoch")
        control_takeover_generation = request.get("control_takeover_generation")
        require(
            request.get("session_id") == self.session_id,
            "ONLINE_REPLAY_ASYNC_RESIDUAL_SESSION_MISMATCH",
        )
        require(
            not isinstance(control_policy_epoch, bool)
            and isinstance(control_policy_epoch, int)
            and control_policy_epoch >= 0
            and not isinstance(control_takeover_generation, bool)
            and isinstance(control_takeover_generation, int)
            and control_takeover_generation >= 0,
            "ONLINE_REPLAY_ASYNC_CONTROL_GENERATION_INVALID",
        )
        result = self.engine.residual_decision(request)
        result["active_residual_policy_revision"] = self.active_revision_id
        result["residual_policy_epoch"] = int(self.machine.policy_epoch)
        result["control_policy_epoch"] = int(control_policy_epoch)
        result["control_takeover_generation"] = int(
            control_takeover_generation
        )
        return result

    def end_episode(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        self._validate_identity(payload)
        self._end_episode(admission_required=True)
        return self.status()

    def abort_episode(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        self._validate_identity(payload)
        self._end_episode(admission_required=False)
        return self.status()

    def resolve_rejected_admission(
        self, payload: Mapping[str, Any]
    ) -> dict[str, Any]:
        require(
            payload.get("session_id") == self.session_id
            and payload.get("episode_id") == self.episode_id,
            "ONLINE_REPLAY_ASYNC_CAPTURE_IDENTITY_MISMATCH",
        )
        with self._lock:
            require(
                not self._episode_active
                and self._admission_resolution_required,
                "FORCERFT_REJECTED_ADMISSION_RESOLUTION_INVALID",
            )
            self._admission_resolution_required = False
            self._lock.notify_all()
        return self.status()

    def notify_admission_committed(
        self, payload: Mapping[str, Any]
    ) -> dict[str, Any]:
        require(
            payload.get("session_id") == self.session_id
            and payload.get("episode_id") == self.episode_id,
            "ONLINE_REPLAY_ASYNC_CAPTURE_IDENTITY_MISMATCH",
        )
        admission_id = str(payload.get("admission_id", ""))
        require(
            bool(admission_id)
            and Path(admission_id).name == admission_id
            and not admission_id.endswith(".json"),
            "FORCERFT_ADMISSION_NOTIFICATION_ID_INVALID",
        )
        replay_root = Path(self.learner_job.replay_root).resolve()
        manifest_path = replay_root / "episodes" / f"{admission_id}.json"
        require(
            manifest_path.is_file(),
            "FORCERFT_ADMISSION_COMMITTED_MANIFEST_MISSING",
        )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        formal_episode_id = str(manifest.get("episode_id", ""))
        require(
            manifest.get("status") == "SEALED_COMMITTED"
            and manifest.get("admission_id") == admission_id
            and formal_episode_id.endswith(f"/{self.episode_id}"),
            "FORCERFT_ADMISSION_COMMITTED_MANIFEST_INVALID",
        )
        capture_index = self.session_id.rsplit("_", 1)[-1].rsplit("-", 1)[-1]
        require(
            formal_episode_id.split("/", 1)[0] == capture_index,
            "FORCERFT_ADMISSION_SESSION_MANIFEST_MISMATCH",
        )
        admission_record = (
            replay_root / str(manifest.get("admission_record", ""))
        ).resolve()
        require(
            admission_record.is_relative_to(replay_root)
            and admission_record.is_file(),
            "FORCERFT_ADMISSION_RECORD_MISSING",
        )
        record = json.loads(admission_record.read_text(encoding="utf-8"))
        require(
            record.get("admission_id") == admission_id
            and record.get("episode_id") == formal_episode_id
            and record.get("episode_sealed") is True,
            "FORCERFT_ADMISSION_RECORD_INVALID",
        )
        with self._lock:
            if self._last_registered_admission_id == admission_id:
                return {
                    **self.status(),
                    "status": "FORMAL_ADMISSION_ALREADY_REGISTERED",
                    "admission_id": admission_id,
                }
            require(
                not self._episode_active and self._admission_resolution_required,
                "FORCERFT_ADMISSION_NOTIFICATION_STATE_INVALID",
            )
            self.learner_job.notify_admission(admission_id)
            self._last_registered_admission_id = admission_id
            self._admission_resolution_required = False
            self._wake_learner.set()
            self._lock.notify_all()
        return {
            **self.status(),
            "status": "FORMAL_ADMISSION_REGISTERED",
            "admission_id": admission_id,
        }

    def drain_admission_budget(
        self, payload: Mapping[str, Any]
    ) -> dict[str, Any]:
        del payload
        raise RuntimeError("FORCERFT_CONTINUOUS_ASYNC_DRAIN_NOT_APPLICABLE")

    def drain_outstanding_budget(
        self, payload: Mapping[str, Any]
    ) -> dict[str, Any]:
        del payload
        raise RuntimeError("FORCERFT_CONTINUOUS_ASYNC_DRAIN_NOT_APPLICABLE")

    def checkpoint_on_operator_q(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        self._validate_identity(payload)
        require(not self._episode_active, "ONLINE_REPLAY_ASYNC_EPISODE_ACTIVE")
        self.stop()
        with self._lock:
            require(
                self._learner_worker_state != "failed",
                "ONLINE_REPLAY_ASYNC_LEARNER_FAILED",
            )
        checkpoint = self._save_final_checkpoint_once()
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
                self._learner_worker_state != "failed",
                "ONLINE_REPLAY_ASYNC_LEARNER_FAILED",
            )
        checkpoint = self._save_final_checkpoint_once()
        return {
            **self.status(),
            "quiesced": True,
            "quiesced_checkpoint_path": (
                None
                if checkpoint is None
                else str(checkpoint)
            ),
        }

    def _save_final_checkpoint_once(self) -> Path | None:
        with self._quiesce_save_lock:
            if not self._quiesced:
                self._quiesced_checkpoint = self.learner_job.save_checkpoint()
                self._quiesced = True
            return self._quiesced_checkpoint

    def _end_episode(self, *, admission_required: bool) -> None:
        with self._lock:
            require(self._episode_active, "ONLINE_REPLAY_ASYNC_EPISODE_INACTIVE")
            self.coordinator.end_actor_window()
            assert self._actor_alive is not None and self._pin is not None
            self._actor_alive.__exit__(None, None, None)
            self._pin.__exit__(None, None, None)
            self._actor_alive = None
            self._pin = None
            self._episode_active = False
            self._admission_resolution_required = admission_required
            finalized = self._capture_window_snapshot()
            if finalized is not None:
                finalized["finalized"] = True
            self._last_capture_window = finalized
            self._capture_window = None
            self.learner_job.clear_current_session()
            self._activate_pending_actor_locked()
            self._lock.notify_all()

    def stop(self) -> None:
        if self._stop_learner.is_set() and (
            self._learner_thread is None or not self._learner_thread.is_alive()
        ):
            return
        self._stop_learner.set()
        self._wake_learner.set()
        self.coordinator.cancel_waiters()
        if self._learner_thread is not None:
            self._learner_thread.join(timeout=30.0)
            require(
                not self._learner_thread.is_alive(),
                "ONLINE_REPLAY_ASYNC_LEARNER_STOP_TIMEOUT",
            )


class RequestHandler(serve_policy.RequestHandler):
    @property
    def runtime(self) -> AsyncResidualActorCriticRuntime:
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
            "/runtime/drain-admission-budget",
            "/runtime/drain-outstanding-budget",
            "/runtime/notify-admission-committed",
            "/runtime/resolve-rejected-admission",
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
                "/runtime/drain-admission-budget": (
                    self.runtime.drain_admission_budget
                ),
                "/runtime/drain-outstanding-budget": (
                    self.runtime.drain_outstanding_budget
                ),
                "/runtime/notify-admission-committed": (
                    self.runtime.notify_admission_committed
                ),
                "/runtime/resolve-rejected-admission": (
                    self.runtime.resolve_rejected_admission
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
    parser.add_argument("--ack-replay-root", type=Path)
    parser.add_argument("--safety-config", type=Path)
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--episode-id", required=True)
    parser.add_argument("--learner-resume-checkpoint", type=Path)
    parser.add_argument("--online-residual-bootstrap-checkpoint", type=Path)
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


def build_runtime(args: argparse.Namespace) -> AsyncResidualActorCriticRuntime:
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
    args.ack_replay_root = (
        output_root
        / ONLINE_ADAPTATION_DIRECTORY_NAME
        / "formal_replay"
        if args.ack_replay_root is None
        else args.ack_replay_root.resolve()
    )
    warmup.configure_task_paths(
        task_id=args.task_id,
        dataset_root=dataset_root,
        output_root=output_root,
    )
    require(warmup.TASK == args.task.strip(), "FORCERFT_TASK_PROMPT_MISMATCH")
    if args.learner_resume_checkpoint is None:
        selected = select_resume_or_bootstrap_checkpoint(
            output_root,
            configured_bootstrap_checkpoint=getattr(
                args, "online_residual_bootstrap_checkpoint", None
            ),
        )
        resume_checkpoint = selected.path
        checkpoint_kind = selected.kind
    else:
        resume_checkpoint = args.learner_resume_checkpoint.resolve()
        checkpoint_state = torch.load(
            resume_checkpoint / "state/runtime_state.pt",
            map_location="cpu",
            weights_only=False,
        )
        checkpoint_kind = str(checkpoint_state.get("checkpoint_kind", ""))
    require(
        exact_resume_checkpoint_is_recoverable(
            resume_checkpoint, expected_kind=checkpoint_kind
        ),
        "FORCERFT_EXACT_RESUME_CHECKPOINT_INVALID",
    )
    current_config = warmup.load_common_actor_critic_config(args.task_id)
    checkpoint_config = load_checkpoint_training_config(resume_checkpoint)
    require(
        not schedule_migration_required(checkpoint_config, current_config),
        f"FORCERFT_SCHEDULE_MIGRATION_REQUIRED:{resume_checkpoint}",
    )
    require_exact_resume_algorithm_config(
        checkpoint_config=checkpoint_config,
        current_config=current_config,
    )
    (
        frozen_base_policy_checkpoint,
        residual_checkpoint,
        active_revision_id,
        active_actor_online_cycle,
        active_actor_steps,
        initial_policy_epoch,
        initial_policy_epoch_status,
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
        frozen_base_policy_checkpoint,
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
    from forcesmolvla.training_data import load_normalizer_manifest

    replay_normalizer = load_normalizer_manifest(
        dataset_root / "normalizer_manifest.json"
    )
    require(
        replay_normalizer.manifest()
        == engine.runtime_artifacts.normalizer.manifest(),
        "FORCERFT_EXACT_RESUME_NORMALIZER_MISMATCH",
    )
    require(
        active_revision_id
        and not any(parameter.requires_grad for parameter in engine.policy.parameters()),
        "FORCERFT_BASE_ACTOR_NOT_FROZEN",
    )
    machine = InMemoryRevisionStateMachine(
        RevisionRecord(
            active_revision_id,
            engine.model_sha256,
            RevisionState.ACTIVE,
        ),
        initial_epoch=initial_policy_epoch,
    )
    checkpoint_root = (
        output_root
        / ONLINE_ADAPTATION_DIRECTORY_NAME
        / "training_checkpoints"
    )
    learner = ResidualActorCriticLearner(
        device=device,
        resume_checkpoint=resume_checkpoint,
        checkpoint_root=checkpoint_root,
        replay_root=args.ack_replay_root,
        current_session_id=args.session_id,
        task=args.task.strip(),
        normalizer_path=dataset_root / "normalizer_manifest.json",
    )
    engine.residual_actor = WristWrenchResidualActor(
        hidden_dim=int(checkpoint_config["wrist_wrench_residual_actor"]["hidden_dim"]),
        max_normalized_residual=float(
            checkpoint_config["wrist_wrench_residual_actor"]["max_normalized_residual"]
        ),
    ).to("cpu")
    engine.residual_actor.eval().requires_grad_(False)
    engine.metadata["online_semantics_version"] = ONLINE_SEMANTICS_VERSION
    engine.metadata["active_policy_epoch_status"] = initial_policy_epoch_status
    _load_residual_checkpoint(engine.residual_actor, residual_checkpoint)
    return AsyncResidualActorCriticRuntime(
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
        f"[residual-activation] active revision={runtime.active_revision_id} "
        f"model={runtime.active_model_revision} "
        f"deployed_online_cycle={runtime.metadata['active_actor_online_cycle']}",
        flush=True,
    )
    print(
        f"[residual-training] exact-resume={runtime.learner_resume_checkpoint} "
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
        if (
            not runtime.status()["episode_active"]
            and runtime.status()["learner_worker_state"] != "failed"
        ):
            checkpoint = runtime.quiesce_and_save({}).get(
                "quiesced_checkpoint_path"
            )
            print(
                f"[training-checkpoint] graceful-exit={checkpoint or 'none'}",
                flush=True,
            )
        else:
            runtime.stop()
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
