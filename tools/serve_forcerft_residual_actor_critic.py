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


ACTOR_GRAD_EPSILON = 1.0e-12


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


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
) -> tuple[Path, Path, str, int]:
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
    return base, residual.resolve(), revision_id, int(
        runtime.get("residual_actor_critic_cycles", 0)
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
        self.latest_actor_q_mapping_unavailable_count = 0
        self.latest_human_residual_projected_count = 0
        self.latest_human_residual_valid_count = 0
        self._loaded_episode_keys: set[str] = set()
        self._admission_progress: dict[str, dict[str, Any]] = {}
        self._expected_admission_id: str | None = None
        self._joint_cycle_budget = 0
        checkpoint_replay = self.learner["runtime"].get("replay", {})
        self._checkpoint_loaded_episode_keys = set(
            checkpoint_replay.get("loaded_episode_keys", ())
        )
        self._checkpoint_per_episode_critic_row_counts = dict(
            checkpoint_replay.get("per_episode_critic_row_counts", {})
        )
        self._checkpoint_admission_cycle_budgets = dict(
            checkpoint_replay.get("admission_cycle_budgets", {})
        )

    @property
    def residual_actor(self) -> torch.nn.Module:
        return self.learner["residual_actor"]

    def set_current_session(self, session_id: str) -> None:
        self.current_session_id = session_id

    def clear_current_session(self) -> None:
        self.current_session_id = None

    def _episode_signature(self) -> tuple[str, ...]:
        return tuple(
            path.stem
            for path in sorted((self.replay_root / "episodes").glob("*.json"))
        )

    def expect_admission(self, admission_id: str) -> None:
        require(
            bool(admission_id)
            and Path(admission_id).name == admission_id
            and not admission_id.endswith(".json"),
            "FORCERFT_TRAINING_DRAIN_ADMISSION_ID_INVALID",
        )
        self._expected_admission_id = admission_id

    def admission_budget_status(self, admission_id: str) -> dict[str, Any] | None:
        progress = self._admission_progress.get(admission_id)
        if progress is None:
            return None
        cycle = int(self.learner["runtime"]["residual_actor_critic_cycles"])
        start = int(progress["cycle_count_at_admission_start"])
        target = int(progress["target_cycle_count_after_admission"])
        budget = int(progress["computed_cycle_budget"])
        completed = max(0, min(budget, cycle - start))
        return {
            **progress,
            "completed_cycle_count_for_latest_admission": completed,
            "remaining_cycle_budget": max(0, target - cycle),
        }

    def outstanding_budget_status(self) -> dict[str, Any]:
        completed = int(
            self.learner["runtime"]["residual_actor_critic_cycles"]
        )
        total = int(self._joint_cycle_budget)
        require(
            0 <= completed <= total,
            "FORCERFT_RECOVERY_CYCLE_BUDGET_INCONSISTENT",
        )
        remaining = total - completed
        return {
            "total_entitled_cycle_budget": total,
            "completed_cycle_count": completed,
            "remaining_cycle_budget": remaining,
            "recovery_budget_drain_required": remaining > 0,
        }

    def recovery_preflight(self) -> dict[str, Any]:
        """Rebuild sealed-replay debt before any new episode may start."""

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
        for admission_id, budget in (
            self._checkpoint_admission_cycle_budgets.items()
        ):
            require(
                admission_id in self._admission_progress
                and int(
                    self._admission_progress[admission_id][
                        "computed_cycle_budget"
                    ]
                )
                == int(budget),
                "FORCERFT_RECOVERY_REPLAY_BUDGET_MISMATCH",
            )
        return self.outstanding_budget_status()

    def _latest_budget_metrics(self) -> dict[str, Any]:
        admission_id = self._expected_admission_id
        if admission_id is None and self._admission_progress:
            admission_id = next(reversed(self._admission_progress))
        status = (
            None
            if admission_id is None
            else self.admission_budget_status(admission_id)
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
            "computed_cycle_budget": (
                0 if status is None else status["computed_cycle_budget"]
            ),
            "cycle_count_at_admission_start": (
                0 if status is None else status["cycle_count_at_admission_start"]
            ),
            "target_cycle_count_after_admission": (
                0 if status is None else status["target_cycle_count_after_admission"]
            ),
            "completed_cycle_count_for_latest_admission": (
                0
                if status is None
                else status["completed_cycle_count_for_latest_admission"]
            ),
            "remaining_cycle_budget": (
                0 if status is None else status["remaining_cycle_budget"]
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
            before_budget = self._joint_cycle_budget
            policy_rows, policy_macros, source_episodes, human_rows = (
                warmup.load_formal_online_episode(
                    self.replay_root, admission_id
                )
            )
            if self.current_session_id is not None:
                require(
                    not any(
                        row["identity"].get("session_id")
                        == self.current_session_id
                        for row in [*policy_rows, *human_rows]
                    ),
                    "ONLINE_REPLAY_ASYNC_CURRENT_EPISODE_ALREADY_IN_REPLAY",
                )
            macros = (*policy_macros, *warmup.build_ack_macros(human_rows))
            added_counts = self.replay.append_macros(macros)
            require(
                len(source_episodes) == 1,
                "FORCERFT_INCREMENTAL_REPLAY_EPISODE_ID_INVALID",
            )
            episode_id = next(iter(source_episodes))
            admitted_rows = int(added_counts.get(episode_id, 0))
            require(
                admitted_rows > 0,
                "FORCERFT_INCREMENTAL_REPLAY_NO_VALID_ROWS",
            )
            computed_budget = self.training_policy.cycles_for_observed_admission(
                new_critic_td_valid_rows=admitted_rows,
                total_critic_td_valid_rows=self.replay.critic_td_valid_rows,
            )
            after_budget = before_budget + computed_budget
            self._joint_cycle_budget = after_budget
            self._loaded_episode_keys.add(admission_id)
            self._admission_progress[admission_id] = {
                "episode_key": admission_id,
                "admitted_rows_for_latest_episode": admitted_rows,
                "computed_cycle_budget": computed_budget,
                "cycle_count_at_admission_start": before_budget,
                "target_cycle_count_after_admission": after_budget,
            }
            self.unique_r_count += len(policy_rows) + len(human_rows)
            self.r_macro_count += len(macros)
        if added_keys:
            self.latest_replay_refresh_ms = (
                time.perf_counter() - started
            ) * 1000.0
        self.next_base_missing_rows = self.replay.next_base_missing_rows
        self.quarantined_current_schema_rows = (
            self.replay.quarantined_current_schema_rows
        )
        self.nonzero_behavior_residual_rows = (
            self.replay.nonzero_behavior_residual_rows
        )
        runtime_replay = self.learner["runtime"]["replay"]
        runtime_replay.update(
            critic_td_valid_rows=self.replay.critic_td_valid_rows,
            actor_q_valid_rows=self.replay.actor_q_valid_rows,
            human_residual_valid_rows=self.replay.human_residual_valid_rows,
            loaded_episode_keys=sorted(self._loaded_episode_keys),
            per_episode_critic_row_counts={
                admission_id: int(progress["admitted_rows_for_latest_episode"])
                for admission_id, progress in self._admission_progress.items()
            },
            admission_cycle_budgets={
                admission_id: int(progress["computed_cycle_budget"])
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
    ) -> float:
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
            seed = int(learner["config"]["environment"]["random_seed"]) + step
            for sample_attempt in range(8):
                batch = replay.sample(
                    self.training_policy.twin_q_batch_size,
                    device=self.device,
                    seed=seed + sample_attempt * 1_000_003,
                )
                require(batch is not None, "FORCERFT_CRITIC_REPLAY_EMPTY")
                loss_result = residual_critic_loss(
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
                if loss_result.td_valid_count:
                    break
            require(
                loss_result is not None and loss_result.td_valid_count > 0,
                "FORCERFT_CRITIC_BATCH_HAS_NO_MAPPABLE_TD_ROW",
            )
            loss_result.total.backward()
            torch.nn.utils.clip_grad_norm_(
                (*learner["q1"].parameters(), *learner["q2"].parameters()),
                float(learner["config"]["optimizer"]["twin_q"]["grad_clip_norm"]),
            )
            optimizer.step()
            tau = float(learner["config"]["optimizer"]["twin_q_polyak_tau"])
            polyak_update(learner["q1"], learner["q1_target"], tau)
            polyak_update(learner["q2"], learner["q2_target"], tau)
        counters["twin_q_optimizer_steps"] = step + 1
        counters["twin_q_target_update_steps"] = int(counters["twin_q_target_update_steps"]) + 1
        if warmup:
            learner["runtime"]["ack_critic_warmup_steps"] = (
                int(learner["runtime"].get("ack_critic_warmup_steps", 0)) + 1
            )
        self.latest_target_candidate_unavailable_count = int(
            loss_result.target_candidate_unavailable_count
        )
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
        attempts = int(counters.get("residual_actor_update_attempts", step)) + 1
        counters["residual_actor_update_attempts"] = attempts
        if actor_update_applied:
            counters["residual_actor_optimizer_steps"] = step + 1
        else:
            counters["residual_actor_updates_skipped_no_gradient"] = int(
                counters.get(
                    "residual_actor_updates_skipped_no_gradient",
                    attempts - 1 - step,
                )
            ) + 1
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
                losses.append(
                    self._critic_update(coordinator, replay, warmup=True)
                )
            require(
                all(
                    torch.equal(before[name], value)
                    for name, value in learner["residual_actor"].state_dict().items()
                ),
                "FORCERFT_WARMUP_MODIFIED_RESIDUAL_ACTOR",
            )
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
        budget = self._joint_cycle_budget
        if cycle >= budget:
            return {
                "waiting_for_replay": True,
                "learner_state": "residual_actor_critic_training",
                "episode_cycle_budget_exhausted": True,
                "episode_cycle_budget": budget,
                "learner_critic_steps": 0,
                "learner_actor_steps": 0,
                "learner_polyak_steps": 0,
                "current_episode_sampled": False,
                "nonfinite_count": 0,
                "oom_count": 0,
                **self._actor_counter_metrics(),
                **self._latest_budget_metrics(),
            }

        cycle_started = time.perf_counter()
        critic_losses = [
            self._critic_update(coordinator, replay, warmup=False)
            for _ in range(self.training_policy.twin_q_updates_per_cycle)
        ]
        actor_metrics = self._actor_update(coordinator, replay)
        runtime["residual_actor_critic_cycles"] = cycle + 1
        learner["residual_actor_critic_cycles"] = cycle + 1
        self.latest_cycle_ms = (time.perf_counter() - cycle_started) * 1000.0
        latest_checkpoint = None
        if self.training_policy.checkpoint_due(cycle + 1):
            latest_checkpoint = self.save_checkpoint()
        return {
            "waiting_for_replay": False,
            "learner_state": "residual_actor_critic_training",
            "learner_critic_steps": self.training_policy.twin_q_updates_per_cycle,
            "learner_actor_steps": int(actor_metrics["applied"]),
            "learner_actor_update_attempts": 1,
            "learner_polyak_steps": self.training_policy.twin_q_updates_per_cycle,
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
            "latest_critic_td_loss": critic_losses[-1],
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
            "latest_checkpoint_path": (
                None if latest_checkpoint is None else str(latest_checkpoint)
            ),
            **self._latest_budget_metrics(),
        }

    def export_actor_candidate(
        self,
        residual_actor_optimizer_steps: int,
        *,
        active_residual_actor: torch.nn.Module | None = None,
    ) -> dict[str, Any] | None:
        if active_residual_actor is not None:
            candidate_state = self.learner["residual_actor"].state_dict()
            active_state = active_residual_actor.state_dict()
            if candidate_state.keys() == active_state.keys() and all(
                torch.equal(candidate_state[name].detach().cpu(), value.detach().cpu())
                for name, value in active_state.items()
            ):
                return None
        runtime = self.learner["runtime"]
        current = str(runtime["active_residual_policy_revision"])
        task_id = current.split("-residual-policy-step-", 1)[0]
        revision_id = f"{task_id}-residual-policy-step-{residual_actor_optimizer_steps:06d}"
        destination = (
            self.checkpoint_root.parent
            / "policy_candidates"
            / str(runtime["online_adaptation_id"])
            / f"residual_actor_step_{residual_actor_optimizer_steps:06d}"
        )
        try:
            destination.mkdir(parents=True, exist_ok=False)
        except FileExistsError as error:
            raise RuntimeError(
                "FORCERFT_RESIDUAL_CANDIDATE_PATH_COLLISION"
            ) from error
        path = destination / "residual_actor.pt"
        torch.save(self.learner["residual_actor"].state_dict(), path)
        torch.save(
            {
                "checkpoint_kind": CANDIDATE_CHECKPOINT_KIND,
                "online_semantics_version": ONLINE_SEMANTICS_VERSION,
            },
            destination / "candidate_state.pt",
        )
        return {
            "revision_id": revision_id,
            "checkpoint": destination.resolve(),
            "residual_actor_optimizer_steps": int(residual_actor_optimizer_steps),
        }

    def mark_active_residual_policy_revision(self, revision_id: str) -> None:
        self.learner["runtime"]["active_residual_policy_revision"] = str(revision_id)

    def save_checkpoint(self) -> Path:
        learner = self.learner
        completed = int(learner["runtime"]["residual_actor_critic_cycles"])
        target = training_checkpoint_path(self.checkpoint_root, completed)
        runtime_state = dict(learner["runtime"])
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
        self._broadcast_count = 0
        self._candidate_count = 0
        self._candidate_checkpoints: dict[str, Path] = {}
        self._candidate_online_cycles: dict[str, int] = {}
        self._active_actor_online_cycle = (
            int(active_actor_online_cycle)
            if active_actor_online_cycle is not None
            else (
                int(self.active_actor_checkpoint.name.rsplit("_", 1)[1])
                if self.active_actor_checkpoint.name.startswith("residual_actor_step_")
                else 0
            )
        )
        self._policy = getattr(
            learner_job, "training_policy", ResidualActorCriticSchedule()
        )
        self._learner_thread: threading.Thread | None = None
        self._inference_request_count = 0
        self._actor_alive: AbstractContextManager[Any] | None = None
        self._pin: PinnedEpisode | None = None
        self._quiesced_checkpoint: Path | None = None
        self._quiesced = False
        self._admission_resolution_required = False
        self._drain_in_progress = False
        self._latest_budget_drain_elapsed_ms = 0.0
        recovery_preflight = getattr(
            self.learner_job, "recovery_preflight", None
        )
        recovery = (
            dict(recovery_preflight())
            if callable(recovery_preflight)
            else {
                "total_entitled_cycle_budget": 0,
                "completed_cycle_count": 0,
                "remaining_cycle_budget": 0,
                "recovery_budget_drain_required": False,
            }
        )
        self._recovery_budget_drain_required = bool(
            recovery["recovery_budget_drain_required"]
        )
        self._recovery_preflight = recovery
        self._learner_result.update(recovery)

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
            "residual_candidate_interval_actor_steps": (
                self._policy.residual_candidate_interval_actor_steps
            ),
            "max_cycles_per_admitted_episode": (
                self._policy.max_cycles_per_admitted_episode
            ),
            "admitted_rows_per_cycle": self._policy.admitted_rows_per_cycle,
            "active_actor_online_cycle": self._active_actor_online_cycle,
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
            "recovery_budget_drain_required": (
                self._recovery_budget_drain_required
            ),
            "outstanding_training_cycle_budget": int(
                self._recovery_preflight["remaining_cycle_budget"]
            ),
        }

    def status(self) -> dict[str, Any]:
        with self._lock:
            result = dict(self._learner_result)
            actor_counters = self._residual_actor_update_counters()
            learner_runtime = getattr(self.learner_job, "learner", {}).get(
                "runtime", {}
            )
            outstanding_status = getattr(
                self.learner_job, "outstanding_budget_status", None
            )
            outstanding = (
                dict(outstanding_status())
                if callable(outstanding_status)
                else dict(self._recovery_preflight)
            )
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
                "learner_started": self._learner_started,
                "learner_worker_state": self._learner_worker_state,
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
                "residual_actor_critic_cycle": int(result.get("residual_actor_critic_cycle", 0)),
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
                "computed_cycle_budget": int(
                    result.get("computed_cycle_budget", 0)
                ),
                "completed_cycle_count_for_latest_admission": int(
                    result.get(
                        "completed_cycle_count_for_latest_admission", 0
                    )
                ),
                "remaining_cycle_budget": int(
                    result.get("remaining_cycle_budget", 0)
                ),
                "target_cycle_count_after_admission": int(
                    result.get("target_cycle_count_after_admission", 0)
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
                "budget_drain_elapsed_ms": self._latest_budget_drain_elapsed_ms,
                "admission_resolution_required": self._admission_resolution_required,
                "drain_in_progress": self._drain_in_progress,
                "recovery_budget_drain_required": (
                    self._recovery_budget_drain_required
                ),
                "total_entitled_cycle_budget": int(
                    outstanding["total_entitled_cycle_budget"]
                ),
                "outstanding_training_cycle_budget": int(
                    outstanding["remaining_cycle_budget"]
                ),
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
        session_id = str(payload.get("session_id", ""))
        episode_id = str(payload.get("episode_id", ""))
        require(bool(session_id and episode_id), "ONLINE_REPLAY_ASYNC_CAPTURE_IDENTITY_MISMATCH")
        with self._lock:
            require(
                not self._episode_active,
                "ONLINE_REPLAY_ASYNC_EPISODE_ALREADY_ACTIVE",
            )
            require(
                not self._admission_resolution_required
                and not self._recovery_budget_drain_required
                and not self._drain_in_progress,
                "FORCERFT_PREPARE_EPISODE_BEFORE_ADMISSION_DRAIN",
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
            # Capture identity remains bound to the immutable frozen base Actor.
            # Residual revisions are tracked by revision_id, not a new digest.
            self.machine.register_candidate(revision_id, self.active_model_revision)
            self.machine.stage(revision_id)
            self._candidate_checkpoints[revision_id] = checkpoint
            self._candidate_online_cycles[revision_id] = int(
                candidate["residual_actor_critic_cycle"]
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
        self.active_revision_id = activated.revision_id
        self.active_actor_checkpoint = checkpoint
        self.learner_job.mark_active_residual_policy_revision(activated.revision_id)
        self._active_actor_online_cycle = self._candidate_online_cycles.pop(
            activated.revision_id
        )
        self._broadcast_count += 1
        if self._policy.checkpoint_on_candidate_activation:
            checkpoint_path = self.learner_job.save_checkpoint()
            self._learner_result["latest_checkpoint_path"] = (
                None if checkpoint_path is None else str(checkpoint_path)
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
                            self._stop_learner.wait(0.25)
                            continue
                        cycle = int(result["residual_actor_critic_cycle"])
                        residual_actor_optimizer_steps = int(
                            result.get("residual_actor_optimizer_steps", 0)
                        )
                        if (
                            int(result.get("learner_actor_steps", 0)) > 0
                            and self._policy.candidate_due(residual_actor_optimizer_steps)
                        ):
                            with self.engine._residual_lock:
                                candidate = self.learner_job.export_actor_candidate(
                                    residual_actor_optimizer_steps,
                                    active_residual_actor=self.engine.residual_actor,
                                )
                            if candidate is None:
                                result["candidate_skipped_unchanged"] = True
                                print(
                                    "[residual-candidate] skipped unchanged "
                                    f"actor_step={residual_actor_optimizer_steps}",
                                    flush=True,
                                )
                            else:
                                candidate["residual_actor_critic_cycle"] = cycle
                                self._stage_actor_candidate(candidate)
                        label = (
                            "residual-training"
                            if result.get("actor_update_attempted")
                            else "critic-warmup"
                        )
                        print(
                            f"[{label}] "
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
                        result.pop("learner_actor", None)
                        with self._lock:
                            self._learner_result = result
                            self._learner_worker_state = "running"
                            self._lock.notify_all()
            except Exception as error:
                with self._lock:
                    self._learner_error = f"{type(error).__name__}:{error}"
                    self._learner_worker_state = "failed"
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
                and self._admission_resolution_required
                and not self._drain_in_progress,
                "FORCERFT_REJECTED_ADMISSION_RESOLUTION_INVALID",
            )
            self._admission_resolution_required = False
            self._lock.notify_all()
        return self.status()

    def drain_admission_budget(
        self, payload: Mapping[str, Any]
    ) -> dict[str, Any]:
        require(
            payload.get("session_id") == self.session_id
            and payload.get("episode_id") == self.episode_id,
            "ONLINE_REPLAY_ASYNC_CAPTURE_IDENTITY_MISMATCH",
        )
        admission_id = str(payload.get("admission_id", ""))
        timeout_seconds = float(payload.get("timeout_seconds", 60.0))
        require(
            bool(admission_id) and 0.0 < timeout_seconds <= 600.0,
            "FORCERFT_TRAINING_DRAIN_REQUEST_INVALID",
        )
        started = time.monotonic()
        deadline = started + timeout_seconds
        with self._lock:
            require(
                not self._episode_active
                and self._admission_resolution_required
                and not self._drain_in_progress,
                "FORCERFT_TRAINING_DRAIN_STATE_INVALID",
            )
            self._drain_in_progress = True
            actor_counters_before = self._residual_actor_update_counters()
        try:
            self.learner_job.expect_admission(admission_id)
            self._start_learner()
            with self._lock:
                while True:
                    require(
                        self._learner_error is None
                        and self._learner_worker_state != "failed",
                        "FORCERFT_TRAINING_DRAIN_LEARNER_FAILED",
                    )
                    progress = self.learner_job.admission_budget_status(
                        admission_id
                    )
                    drained = bool(
                        progress is not None
                        and progress["remaining_cycle_budget"] == 0
                        and self._learner_worker_state
                        in {"waiting_for_replay", "idle"}
                    )
                    if drained:
                        elapsed_ms = (time.monotonic() - started) * 1000.0
                        self._latest_budget_drain_elapsed_ms = elapsed_ms
                        self._admission_resolution_required = False
                        self._drain_in_progress = False
                        self._lock.notify_all()
                        budget = int(progress["computed_cycle_budget"])
                        actor_counters_after = (
                            self._residual_actor_update_counters()
                        )
                        actor_attempts = (
                            actor_counters_after[
                                "residual_actor_update_attempts"
                            ]
                            - actor_counters_before[
                                "residual_actor_update_attempts"
                            ]
                        )
                        actor_updates = (
                            actor_counters_after[
                                "residual_actor_optimizer_steps"
                            ]
                            - actor_counters_before[
                                "residual_actor_optimizer_steps"
                            ]
                        )
                        return {
                            "status": "TRAINING_BUDGET_DRAINED",
                            "admission_id": admission_id,
                            **progress,
                            "completed_cycle_count": int(
                                progress[
                                    "completed_cycle_count_for_latest_admission"
                                ]
                            ),
                            "twin_q_updates": (
                                budget * self._policy.twin_q_updates_per_cycle
                            ),
                            "residual_actor_update_attempts": actor_attempts,
                            "residual_actor_updates": actor_updates,
                            "residual_actor_updates_skipped_no_gradient": (
                                actor_attempts - actor_updates
                            ),
                            "warmup_steps_completed": int(
                                getattr(self.learner_job, "learner", {})
                                .get("runtime", {})
                                .get("ack_critic_warmup_steps", 0)
                            ),
                            "candidate_staged": (
                                self.machine.pending_revision_id is not None
                            ),
                            "budget_drain_elapsed_ms": elapsed_ms,
                            "replay_refresh_ms": float(
                                getattr(
                                    self.learner_job,
                                    "latest_replay_refresh_ms",
                                    0.0,
                                )
                            ),
                            "latest_critic_update_ms": float(
                                getattr(
                                    self.learner_job,
                                    "latest_critic_update_ms",
                                    0.0,
                                )
                            ),
                            "latest_actor_update_ms": float(
                                getattr(
                                    self.learner_job,
                                    "latest_actor_update_ms",
                                    0.0,
                                )
                            ),
                            "latest_cycle_ms": float(
                                getattr(
                                    self.learner_job,
                                    "latest_cycle_ms",
                                    0.0,
                                )
                            ),
                        }
                    remaining = deadline - time.monotonic()
                    require(
                        remaining > 0.0,
                        "FORCERFT_TRAINING_DRAIN_TIMEOUT",
                    )
                    self._lock.wait(timeout=remaining)
        except BaseException:
            with self._lock:
                self._drain_in_progress = False
                self._lock.notify_all()
            raise

    def drain_outstanding_budget(
        self, payload: Mapping[str, Any]
    ) -> dict[str, Any]:
        """Drain replay debt recovered before the current session existed."""

        timeout_seconds = float(payload.get("timeout_seconds", 60.0))
        require(
            0.0 < timeout_seconds <= 600.0,
            "FORCERFT_TRAINING_DRAIN_REQUEST_INVALID",
        )
        started = time.monotonic()
        deadline = started + timeout_seconds
        with self._lock:
            require(
                not self._episode_active
                and not self._admission_resolution_required
                and self._recovery_budget_drain_required
                and not self._drain_in_progress,
                "FORCERFT_OUTSTANDING_TRAINING_DRAIN_STATE_INVALID",
            )
            starting = self.learner_job.outstanding_budget_status()
            cycles_to_drain = int(starting["remaining_cycle_budget"])
            require(
                cycles_to_drain > 0,
                "FORCERFT_OUTSTANDING_TRAINING_DRAIN_STATE_INVALID",
            )
            self._drain_in_progress = True
            actor_counters_before = self._residual_actor_update_counters()
        try:
            self._start_learner()
            with self._lock:
                while True:
                    require(
                        self._learner_error is None
                        and self._learner_worker_state != "failed",
                        "FORCERFT_TRAINING_DRAIN_LEARNER_FAILED",
                    )
                    progress = self.learner_job.outstanding_budget_status()
                    drained = bool(
                        progress["remaining_cycle_budget"] == 0
                        and self._learner_worker_state
                        in {"waiting_for_replay", "idle"}
                    )
                    if drained:
                        elapsed_ms = (time.monotonic() - started) * 1000.0
                        self._latest_budget_drain_elapsed_ms = elapsed_ms
                        self._recovery_budget_drain_required = False
                        self._recovery_preflight = dict(progress)
                        self._drain_in_progress = False
                        self._lock.notify_all()
                        actor_counters_after = (
                            self._residual_actor_update_counters()
                        )
                        actor_attempts = (
                            actor_counters_after[
                                "residual_actor_update_attempts"
                            ]
                            - actor_counters_before[
                                "residual_actor_update_attempts"
                            ]
                        )
                        actor_updates = (
                            actor_counters_after[
                                "residual_actor_optimizer_steps"
                            ]
                            - actor_counters_before[
                                "residual_actor_optimizer_steps"
                            ]
                        )
                        return {
                            "status": "OUTSTANDING_TRAINING_BUDGET_DRAINED",
                            **progress,
                            "drained_cycle_count": cycles_to_drain,
                            "twin_q_updates": (
                                cycles_to_drain
                                * self._policy.twin_q_updates_per_cycle
                            ),
                            "residual_actor_update_attempts": actor_attempts,
                            "residual_actor_updates": actor_updates,
                            "residual_actor_updates_skipped_no_gradient": (
                                actor_attempts - actor_updates
                            ),
                            "budget_drain_elapsed_ms": elapsed_ms,
                        }
                    remaining = deadline - time.monotonic()
                    require(
                        remaining > 0.0,
                        "FORCERFT_TRAINING_DRAIN_TIMEOUT",
                    )
                    self._lock.wait(timeout=remaining)
        except BaseException:
            with self._lock:
                self._drain_in_progress = False
                self._lock.notify_all()
            raise

    def checkpoint_on_operator_q(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        self._validate_identity(payload)
        require(not self._episode_active, "ONLINE_REPLAY_ASYNC_EPISODE_ACTIVE")
        self.stop()
        with self._lock:
            require(
                self._learner_worker_state != "failed",
                "ONLINE_REPLAY_ASYNC_LEARNER_FAILED",
            )
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
                self._learner_worker_state != "failed",
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
            self.learner_job.clear_current_session()
            self._activate_pending_actor_locked()
            self._lock.notify_all()

    def stop(self) -> None:
        self._stop_learner.set()
        if self._learner_thread is not None:
            self._learner_thread.join()


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
    require_exact_resume_algorithm_config(
        checkpoint_config=checkpoint_config,
        current_config=current_config,
    )
    (
        frozen_base_policy_checkpoint,
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
    online_config = checkpoint_config["residual_actor_critic_training"]
    try:
        active_actor_steps = int(active_revision_id.rsplit("-", 1)[1])
    except ValueError:
        active_actor_steps = 0
    initial_policy_epoch = active_actor_steps // int(
        online_config["residual_candidate_interval_actor_steps"]
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
    if active_actor_steps > 0:
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
        runtime.stop()
        if runtime.status()["learner_worker_state"] != "failed":
            checkpoint = runtime.learner_job.save_checkpoint()
            print(
                f"[training-checkpoint] graceful-exit={checkpoint or 'none'}",
                flush=True,
            )
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
