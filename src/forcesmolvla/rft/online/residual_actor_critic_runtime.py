"""Coordination primitives for online ACK-residual Actor-Critic training."""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
import math
from pathlib import Path
import shutil
import threading
import time
from typing import Any, Callable, Iterator, Mapping, Sequence

import numpy as np
import torch


class AsyncRuntimeError(RuntimeError):
    pass


@dataclass(frozen=True)
class SelectedCheckpoint:
    """An explicitly classified online residual training parent."""

    path: Path
    kind: str


@dataclass(frozen=True)
class ResidualActorCriticSchedule:
    """Fixed scheduling contract for the persistent online Learner."""

    minimum_ack_transitions: int = 100
    ack_critic_warmup_steps: int = 256
    twin_q_batch_size: int = 128
    residual_policy_value_batch_size: int = 64
    human_residual_imitation_batch_size: int = 32
    admitted_rows_per_cycle: int = 64
    twin_q_updates_per_cycle: int = 2
    residual_actor_updates_per_cycle: int = 1
    max_cycles_per_admitted_episode: int = 10
    residual_candidate_interval_actor_steps: int = 10
    training_checkpoint_interval_cycles: int = 20
    retained_training_checkpoint_count: int = 10
    checkpoint_on_warmup_complete: bool = True
    checkpoint_on_candidate_activation: bool = True

    def __post_init__(self) -> None:
        require(
            self.minimum_ack_transitions >= 0
            and self.ack_critic_warmup_steps >= 1
            and self.twin_q_batch_size >= 1
            and self.residual_policy_value_batch_size >= 1
            and self.human_residual_imitation_batch_size >= 1
            and self.admitted_rows_per_cycle >= 1
            and self.twin_q_updates_per_cycle >= 1
            and self.residual_actor_updates_per_cycle == 1
            and self.max_cycles_per_admitted_episode >= 1
            and self.residual_candidate_interval_actor_steps >= 1
            and self.training_checkpoint_interval_cycles >= 1
            and self.retained_training_checkpoint_count >= 1
            and isinstance(self.checkpoint_on_warmup_complete, bool)
            and isinstance(self.checkpoint_on_candidate_activation, bool),
            "FORCERFT_RESIDUAL_ACTOR_CRITIC_SCHEDULE_INVALID",
        )

    def training_ready(self, online_transition_count: int) -> bool:
        return online_transition_count >= self.minimum_ack_transitions

    def candidate_due(self, completed_actor_steps: int) -> bool:
        return (
            completed_actor_steps > 0
            and completed_actor_steps % self.residual_candidate_interval_actor_steps == 0
        )

    def cycles_for_admission(self, new_critic_td_valid_rows: int) -> int:
        require(
            new_critic_td_valid_rows >= 0,
            "FORCERFT_ONLINE_ADMITTED_ROW_COUNT_INVALID",
        )
        if new_critic_td_valid_rows == 0:
            return 0
        return min(
            self.max_cycles_per_admitted_episode,
            max(
                1,
                math.ceil(
                    new_critic_td_valid_rows / self.admitted_rows_per_cycle
                ),
            ),
        )

    def cycles_for_observed_admission(
        self,
        *,
        new_critic_td_valid_rows: int,
        total_critic_td_valid_rows: int,
    ) -> int:
        """Apply the fixed warmup-only semantics for pre-threshold admissions.

        Rows admitted before the ACK threshold train the one-time Critic warmup
        but never accrue retroactive residual Actor-Critic cycle debt.  The
        admission that reaches the threshold receives only its own cycle budget.
        """

        require(
            total_critic_td_valid_rows >= new_critic_td_valid_rows >= 0,
            "FORCERFT_ONLINE_ADMITTED_ROW_COUNT_INVALID",
        )
        if total_critic_td_valid_rows < self.minimum_ack_transitions:
            return 0
        return self.cycles_for_admission(new_critic_td_valid_rows)

    def residual_actor_critic_cycle_budget(self, admitted_episode_rows: int | Sequence[int]) -> int:
        """Total deterministic budget, derivable again after checkpoint resume."""

        rows = (
            (admitted_episode_rows,)
            if isinstance(admitted_episode_rows, int)
            else admitted_episode_rows
        )
        return sum(self.cycles_for_admission(int(count)) for count in rows)

    def checkpoint_due(self, completed_cycle: int) -> bool:
        return (
            completed_cycle > 0
            and completed_cycle % self.training_checkpoint_interval_cycles == 0
        )


def training_checkpoint_path(checkpoint_root: Path, completed_cycle: int) -> Path:
    require(completed_cycle >= 0, "FORCERFT_ONLINE_CYCLE_INVALID")
    return checkpoint_root / f"residual_actor_critic_cycle_{completed_cycle:06d}"


def exact_resume_checkpoint_is_recoverable(
    checkpoint: Path, *, expected_kind: str
) -> bool:
    """Reject incomplete final-format bootstrap/resume directories."""

    from forcesmolvla.rft.online.residual_actor_critic_checkpoint import (
        residual_actor_critic_checkpoint_is_recoverable,
    )

    if expected_kind not in {"online_residual_bootstrap", "residual_actor_critic_training"}:
        return False
    return residual_actor_critic_checkpoint_is_recoverable(checkpoint)


def select_resume_or_bootstrap_checkpoint(
    output_root: Path,
    *,
    configured_bootstrap_checkpoint: Path | None,
) -> SelectedCheckpoint:
    """Choose online exact-resume, then the explicit bootstrap, else fail closed."""

    online_root = (
        output_root.resolve() / "online_ack_residual/training_checkpoints"
    )
    candidates: list[tuple[int, Path]] = []
    for path in online_root.glob("residual_actor_critic_cycle_*"):
        if not path.is_dir():
            continue
        try:
            cycle = int(path.name.rsplit("_", 1)[1])
        except ValueError:
            continue
        if exact_resume_checkpoint_is_recoverable(
            path, expected_kind="residual_actor_critic_training"
        ):
            candidates.append((cycle, path.resolve()))
    if candidates:
        return SelectedCheckpoint(
            path=max(candidates)[1],
            kind="residual_actor_critic_training",
        )

    if configured_bootstrap_checkpoint is not None:
        bootstrap = configured_bootstrap_checkpoint.resolve()
        require(
            exact_resume_checkpoint_is_recoverable(
                bootstrap, expected_kind="online_residual_bootstrap"
            ),
            "FORCERFT_ONLINE_RESIDUAL_BOOTSTRAP_MISSING_OR_INCOMPLETE",
        )
        return SelectedCheckpoint(
            path=bootstrap, kind="online_residual_bootstrap"
        )

    raise AsyncRuntimeError("FORCERFT_RESUME_OR_ONLINE_RESIDUAL_BOOTSTRAP_REQUIRED")


def retain_latest_training_checkpoints(checkpoint_root: Path, *, keep: int = 2) -> tuple[Path, ...]:
    """Keep the newest exact-resume directories after a successful save."""

    require(keep >= 1, "FORCERFT_ONLINE_CHECKPOINT_RETENTION_INVALID")
    checkpoints: list[tuple[int, Path]] = []
    for path in checkpoint_root.glob("residual_actor_critic_cycle_*"):
        if not path.is_dir():
            continue
        try:
            cycle = int(path.name.rsplit("_", 1)[1])
        except ValueError:
            continue
        checkpoints.append((cycle, path))
    checkpoints.sort()
    for _cycle, path in checkpoints[:-keep]:
        shutil.rmtree(path)
    return tuple(path for _cycle, path in checkpoints[-keep:])


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AsyncRuntimeError(message)


def load_checkpoint_training_config(checkpoint: Path) -> dict[str, Any]:
    """Load the authoritative algorithm configuration from one checkpoint."""

    import yaml

    checkpoint = Path(checkpoint).resolve()
    config = yaml.safe_load(
        (checkpoint / "state/config.yaml").read_text(encoding="utf-8")
    )
    require(isinstance(config, dict), "FORCERFT_CHECKPOINT_CONFIG_INVALID")
    return dict(config)


def require_exact_resume_algorithm_config(
    *,
    checkpoint_config: Mapping[str, Any],
    current_config: Mapping[str, Any],
) -> None:
    """Fail closed instead of mixing checkpoint state with current YAML values."""

    from forcesmolvla.rft.online.replay_training import algorithm_hyperparameters

    try:
        matches = algorithm_hyperparameters(
            checkpoint_config
        ) == algorithm_hyperparameters(current_config)
    except (KeyError, TypeError):
        matches = False
    require(matches, "FORCERFT_EXACT_RESUME_CONFIG_MISMATCH")


def actor_optimizer_state_is_valid_for_resume(
    actor_optimizer: torch.optim.Optimizer,
    residual_actor_optimizer_steps: int,
) -> bool:
    """A Critic-only checkpoint legitimately has no Actor optimizer state."""

    return residual_actor_optimizer_steps == 0 or bool(actor_optimizer.state)


def reconcile_post_checkpoint_replay(credits: Any, all_r: Sequence[dict]) -> int:
    """Mint credit exactly once for live replay UIDs admitted after checkpoint."""

    uids = [str(row["identity"]["transition_uid"]) for row in all_r]
    require(len(set(uids)) == len(uids), "ONLINE_REPLAY_ASYNC_LIVE_REPLAY_UID_DUPLICATE")
    minted = sum(credits.mint_for_unique_online_transition(uid) for uid in uids)
    snapshot = credits.snapshot()
    require(
        snapshot.credited_transition_count == len(uids)
        and 0 <= snapshot.consumed <= snapshot.minted
        and snapshot.available == snapshot.minted - snapshot.consumed,
        "ONLINE_REPLAY_ASYNC_REPLAY_OR_CREDIT_MISMATCH",
    )
    return minted


def prepare_learner(
    device: torch.device,
    *,
    resume_checkpoint: Path,
) -> dict[str, Any]:
    """Restore only the residual Actor, target Actor, and image-free Twin-Q."""

    from forcesmolvla.rft.critic import build_twin_q
    from forcesmolvla.rft.online.residual_actor_critic_checkpoint import (
        load_residual_actor_critic_checkpoint,
    )
    from forcesmolvla.rft.residual_actor import make_residual_actor_pair

    resume_checkpoint = Path(resume_checkpoint).resolve()
    config = load_checkpoint_training_config(resume_checkpoint)
    require(
        int(config["batching"]["command_macro_slots"]) == 3,
        "FORCERFT_COMMAND_MACRO_SLOTS_INVALID",
    )
    residual_actor, residual_actor_target = make_residual_actor_pair(
        hidden_dim=int(config["wrist_wrench_residual_actor"]["hidden_dim"]),
        max_normalized_residual=float(
            config["wrist_wrench_residual_actor"]["max_normalized_residual"]
        ),
    )
    q1, q2, q1_target, q2_target = build_twin_q(
        hidden_dim=int(config["ack_residual_twin_q"]["hidden_dim"]),
        seed=int(config["environment"]["random_seed"]) + 1,
    )
    for module in (
        residual_actor,
        residual_actor_target,
        q1,
        q2,
        q1_target,
        q2_target,
    ):
        module.to(device)
    residual_actor_optimizer = torch.optim.Adam(
        residual_actor.parameters(),
        lr=float(config["optimizer"]["residual_actor"]["lr"]),
    )
    critic_optimizer = torch.optim.Adam(
        (*q1.parameters(), *q2.parameters()),
        lr=float(config["optimizer"]["twin_q"]["lr"]),
    )
    runtime, loaded_config = load_residual_actor_critic_checkpoint(
        resume_checkpoint,
        residual_actor=residual_actor,
        residual_actor_target=residual_actor_target,
        q1=q1,
        q2=q2,
        q1_target=q1_target,
        q2_target=q2_target,
        residual_actor_optimizer=residual_actor_optimizer,
        critic_optimizer=critic_optimizer,
        device=device,
    )
    require(config == loaded_config, "FORCERFT_CHECKPOINT_CONFIG_DRIFT")
    counters = runtime["counters"]
    applied_actor_steps = int(counters["residual_actor_optimizer_steps"])
    actor_update_attempts = int(
        counters.setdefault(
            "residual_actor_update_attempts", applied_actor_steps
        )
    )
    skipped_actor_updates = int(
        counters.setdefault(
            "residual_actor_updates_skipped_no_gradient",
            actor_update_attempts - applied_actor_steps,
        )
    )
    require(
        actor_update_attempts
        == applied_actor_steps + skipped_actor_updates,
        "FORCERFT_RESIDUAL_ACTOR_UPDATE_COUNTER_MISMATCH",
    )
    completed_warmup = int(runtime.get("ack_critic_warmup_steps", 0))
    expected_warmup = int(config["ack_critic_warmup"]["optimizer_steps"])
    require(
        0 <= completed_warmup <= expected_warmup
        and int(runtime["counters"]["twin_q_optimizer_steps"])
        >= completed_warmup
        and (
            runtime["learner_state"] != "ack_replay_collection" or completed_warmup == 0
        )
        and (
            runtime["learner_state"] != "residual_actor_critic_training"
            or runtime["ack_critic_warmup_complete"] is True
            and completed_warmup == expected_warmup
        )
        and (
            runtime["ack_critic_warmup_complete"] is not True
            or runtime["learner_state"] == "residual_actor_critic_training"
            and completed_warmup == expected_warmup
        ),
        "FORCERFT_CHECKPOINT_LEARNER_STATE_COUNTER_MISMATCH",
    )
    residual_actor.train()
    q1.train()
    q2.train()
    for target in (residual_actor_target, q1_target, q2_target):
        target.eval().requires_grad_(False)
    warmup = config["ack_critic_warmup"]
    batching = config["batching"]
    online = config["residual_actor_critic_training"]
    policy = ResidualActorCriticSchedule(
        minimum_ack_transitions=int(warmup["minimum_ack_transitions"]),
        ack_critic_warmup_steps=int(warmup["optimizer_steps"]),
        twin_q_batch_size=int(batching["twin_q_batch_size"]),
        residual_policy_value_batch_size=int(batching["residual_policy_value_batch_size"]),
        human_residual_imitation_batch_size=int(batching["human_residual_imitation_batch_size"]),
        admitted_rows_per_cycle=int(online["admitted_rows_per_cycle"]),
        twin_q_updates_per_cycle=int(online["twin_q_updates_per_cycle"]),
        residual_actor_updates_per_cycle=int(online["residual_actor_updates_per_cycle"]),
        max_cycles_per_admitted_episode=int(
            online["max_cycles_per_admitted_episode"]
        ),
        residual_candidate_interval_actor_steps=int(
            online["residual_candidate_interval_actor_steps"]
        ),
        training_checkpoint_interval_cycles=int(online["training_checkpoint_interval_cycles"]),
        retained_training_checkpoint_count=int(online["retained_training_checkpoint_count"]),
        checkpoint_on_warmup_complete=bool(
            online["checkpoint_on_warmup_complete"]
        ),
        checkpoint_on_candidate_activation=bool(
            online["checkpoint_on_candidate_activation"]
        ),
    )
    return {
        "residual_actor": residual_actor,
        "residual_actor_target": residual_actor_target,
        "q1": q1,
        "q2": q2,
        "q1_target": q1_target,
        "q2_target": q2_target,
        "modules": {
            "residual_actor": residual_actor,
            "residual_actor_target": residual_actor_target,
            "q1": q1,
            "q2": q2,
            "q1_target": q1_target,
            "q2_target": q2_target,
        },
        "config": config,
        "residual_actor_optimizer": residual_actor_optimizer,
        "critic_optimizer": critic_optimizer,
        "runtime": runtime,
        "training_policy": policy,
        "resume_checkpoint": resume_checkpoint,
        "residual_actor_critic_cycles": int(runtime.get("residual_actor_critic_cycles", 0)),
    }


class InferencePriorityCoordinator:
    """Serialize GPU work and give queued inference priority over Learner work."""

    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._owner: str | None = None
        self._inference_waiters = 0
        self._alive = {"actor": 0, "learner": 0}
        self._actor_window_active = False
        self._coverage_deadline = 0.0
        self.learner_microstep_ms: list[tuple[str, float]] = []
        self.concurrently_alive = False

    @contextmanager
    def worker_alive(self, role: str) -> Iterator[None]:
        if role not in self._alive:
            raise ValueError("ONLINE_REPLAY_ASYNC_WORKER_ROLE_INVALID")
        with self._condition:
            self._alive[role] += 1
            self.concurrently_alive |= all(self._alive.values())
            self._condition.notify_all()
        try:
            yield
        finally:
            with self._condition:
                self._alive[role] -= 1
                self._condition.notify_all()

    @contextmanager
    def inference_slot(self, *, reserved: bool = False) -> Iterator[None]:
        with self._condition:
            if not reserved:
                self._inference_waiters += 1
                self._condition.notify_all()
            elif self._inference_waiters < 1:
                raise AsyncRuntimeError("ONLINE_REPLAY_ASYNC_INFERENCE_RESERVATION_MISSING")
            try:
                self._condition.wait_for(lambda: self._owner is None)
                self._owner = "inference"
            finally:
                self._inference_waiters -= 1
        try:
            yield
        finally:
            with self._condition:
                self._owner = None
                self._condition.notify_all()

    def reserve_inference(self) -> None:
        with self._condition:
            self._inference_waiters += 1
            self._condition.notify_all()

    def cancel_inference_reservation(self) -> None:
        with self._condition:
            if self._inference_waiters < 1:
                raise AsyncRuntimeError("ONLINE_REPLAY_ASYNC_INFERENCE_RESERVATION_MISSING")
            self._inference_waiters -= 1
            self._condition.notify_all()

    @contextmanager
    def learner_step_slot(
        self,
        kind: str = "learner",
        *,
        initial_estimate_s: float = 0.45,
        coverage_reserve_s: float = 0.10,
        episode_idle_required: bool = False,
    ) -> Iterator[None]:
        if initial_estimate_s < 0 or coverage_reserve_s < 0:
            raise ValueError("ONLINE_REPLAY_ASYNC_LEARNER_ESTIMATE_INVALID")
        with self._condition:
            prior = [
                milliseconds / 1000.0
                for name, milliseconds in self.learner_microstep_ms
                if name == kind
            ]
            estimate = max([initial_estimate_s, *prior])

            def ready() -> bool:
                coverage = self._coverage_deadline - time.monotonic()
                return (
                    self._owner is None
                    and self._inference_waiters == 0
                    and (
                        not episode_idle_required
                        or not self._actor_window_active
                    )
                    and (
                        not self._actor_window_active
                        or coverage >= estimate + coverage_reserve_s
                    )
                )

            self._condition.wait_for(
                ready
            )
            self._owner = "learner"
        started = time.perf_counter()
        try:
            yield
        finally:
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            with self._condition:
                self.learner_microstep_ms.append((kind, elapsed_ms))
                self._owner = None
                self._condition.notify_all()

    def wait_for_both_workers(self, timeout_s: float) -> None:
        with self._condition:
            if not self._condition.wait_for(
                lambda: all(self._alive.values()), timeout=timeout_s
            ):
                raise AsyncRuntimeError("ONLINE_REPLAY_ASYNC_WORKERS_NOT_CONCURRENT")

    @property
    def inference_pending(self) -> bool:
        with self._condition:
            return self._inference_waiters > 0

    def begin_actor_window(self, coverage_s: float) -> None:
        with self._condition:
            self._actor_window_active = True
            self._coverage_deadline = time.monotonic() + max(0.0, coverage_s)
            self._condition.notify_all()

    def update_action_coverage(self, coverage_s: float) -> None:
        with self._condition:
            if self._actor_window_active:
                self._coverage_deadline = time.monotonic() + max(0.0, coverage_s)
                self._condition.notify_all()

    def end_actor_window(self) -> None:
        with self._condition:
            self._actor_window_active = False
            self._coverage_deadline = 0.0
            self._condition.notify_all()

    @property
    def slowest_learner_microstep(self) -> tuple[str, float] | None:
        with self._condition:
            return max(self.learner_microstep_ms, key=lambda item: item[1], default=None)


@dataclass(frozen=True)
class EpisodePin:
    revision_id: str
    model_revision: str
    policy_epoch: int


@dataclass(frozen=True)
class TakeoverWindow:
    start_ns: int
    resume_ns: int
    takeover_generation: int

    def __post_init__(self) -> None:
        if (
            self.start_ns < 0
            or self.resume_ns <= self.start_ns
            or self.takeover_generation < 1
        ):
            raise ValueError("ONLINE_REPLAY_ASYNC_TAKEOVER_WINDOW_INVALID")


@dataclass(frozen=True)
class InferenceRequest:
    request_index: int
    request_id: str
    result_id: str
    chunk_id: str
    proposal_id: str
    t_ref_ns: int
    revision_id: str
    model_revision: str
    policy_epoch: int
    takeover_generation: int


class PinnedEpisode:
    """Read-only episode pin over the existing revision state machine."""

    def __init__(self, machine: Any, expected: EpisodePin) -> None:
        self.machine = machine
        self.expected = expected
        self.pin: EpisodePin | None = None

    def __enter__(self) -> EpisodePin:
        revision_id = self.machine.begin_episode()
        value = self.machine.episode_pin()
        self.pin = EpisodePin(
            revision_id=revision_id,
            model_revision=value.model_sha256,
            policy_epoch=value.policy_epoch,
        )
        if self.pin != self.expected:
            self.machine.end_episode()
            self.pin = None
            raise AsyncRuntimeError("ONLINE_REPLAY_ASYNC_ACTIVE_REVISION_MISMATCH")
        return self.expected

    def __exit__(self, exc_type, exc, traceback) -> None:
        if self.pin is not None:
            self.machine.assert_episode_binding(
                self.pin.revision_id,
                self.pin.model_revision,
                self.pin.policy_epoch,
            )
            self.machine.end_episode()


class H50ActionCache:
    """H=50 cache with the existing eight-step/low-watermark dispatch policy."""

    def __init__(self, *, replan_steps: int = 8, low_watermark: int = 4) -> None:
        if not 1 <= low_watermark < replan_steps <= 50:
            raise ValueError("ONLINE_REPLAY_ASYNC_ACTION_CACHE_CONFIG")
        self.replan_steps = replan_steps
        self.low_watermark = low_watermark
        self.chunk: np.ndarray | None = None
        self.anchor_ns = 0
        self.revision_id = ""
        self.policy_epoch = -1
        self.takeover_generation = -1
        self.dispatched = 0
        self.stale_count = 0

    @property
    def remaining(self) -> int:
        return max(0, self.replan_steps - self.dispatched) if self.chunk is not None else 0

    @property
    def needs_replan(self) -> bool:
        return self.chunk is None or self.remaining <= self.low_watermark

    def adopt(
        self,
        chunk: np.ndarray,
        *,
        anchor_ns: int,
        revision_id: str,
        policy_epoch: int,
        takeover_generation: int = 0,
    ) -> None:
        value = np.asarray(chunk)
        if value.shape != (50, 7) or not np.isfinite(value).all():
            raise AsyncRuntimeError("ONLINE_REPLAY_ASYNC_INFERENCE_ACTION_INVALID")
        self.chunk = value.copy()
        self.anchor_ns = int(anchor_ns)
        self.revision_id = revision_id
        self.policy_epoch = int(policy_epoch)
        self.takeover_generation = int(takeover_generation)
        self.dispatched = 0

    def flush(self) -> None:
        self.chunk = None
        self.anchor_ns = 0
        self.dispatched = 0

    def dispatch(
        self,
        *,
        timestamp_ns: int,
        pin: EpisodePin,
        takeover_generation: int = 0,
    ) -> np.ndarray | None:
        if self.chunk is None:
            return None
        if (
            self.revision_id != pin.revision_id
            or self.policy_epoch != pin.policy_epoch
            or self.takeover_generation != takeover_generation
        ):
            self.stale_count += 1
            self.flush()
            return None
        if self.dispatched >= self.replan_steps:
            return None
        elapsed_ns = max(0, int(timestamp_ns) - self.anchor_ns)
        index = math.ceil(elapsed_ns * 30 / 1_000_000_000)
        if index >= 50:
            self.stale_count += 1
            self.flush()
            return None
        self.dispatched += 1
        return self.chunk[index].copy()


def percentile_latency_ms(values: Sequence[float]) -> list[float]:
    if not values:
        return [0.0, 0.0, 0.0]
    return [
        float(np.percentile(values, 50)),
        float(np.percentile(values, 95)),
        float(max(values)),
    ]


def run_timed_actor(
    *,
    timestamps_ns: Sequence[int],
    samples: Sequence[Any],
    infer: Callable[[Any, InferenceRequest], np.ndarray],
    coordinator: InferencePriorityCoordinator,
    pin: EpisodePin,
    start_barrier: threading.Barrier,
    replan_steps: int = 8,
    low_watermark: int = 7,
    realtime_scale: float = 1.0,
    initial_chunk: np.ndarray | None = None,
    initial_latency_ms: float | None = None,
    takeover_windows: Sequence[TakeoverWindow] = (),
) -> dict[str, Any]:
    if len(timestamps_ns) != len(samples) or not timestamps_ns:
        raise ValueError("ONLINE_REPLAY_ASYNC_TIMING_FIXTURE_INVALID")
    if realtime_scale <= 0:
        raise ValueError("ONLINE_REPLAY_ASYNC_REALTIME_SCALE_INVALID")
    if any(right <= left for left, right in zip(timestamps_ns, timestamps_ns[1:])):
        raise ValueError("ONLINE_REPLAY_ASYNC_TIMING_NOT_MONOTONIC")
    windows = sorted(takeover_windows, key=lambda value: value.start_ns)
    if any(
        left.resume_ns > right.start_ns
        or right.takeover_generation != left.takeover_generation + 1
        for left, right in zip(windows, windows[1:])
    ):
        raise ValueError("ONLINE_REPLAY_ASYNC_TAKEOVER_WINDOWS_INVALID")

    cache = H50ActionCache(
        replan_steps=replan_steps, low_watermark=low_watermark,
    )
    request_count = int(initial_chunk is not None)
    underruns = 0
    latency_ms: list[float] = [] if initial_latency_ms is None else [initial_latency_ms]
    future: tuple[Future, InferenceRequest] | None = None
    retired: list[tuple[Future, InferenceRequest]] = []
    takeover_generation = 0
    takeover_active = False
    resume_wait = False
    old_result_post_takeover_adopt_count = 0
    old_result_post_takeover_reject_count = 0
    post_takeover_fresh_request_count = 0
    post_takeover_fresh_chunk_adopt_count = 0
    start_ns = timestamps_ns[0]
    wall_start = time.monotonic()

    def submit(
        pool: ThreadPoolExecutor,
        sample: Any,
        anchor_ns: int,
    ) -> tuple[Future, InferenceRequest]:
        nonlocal request_count, post_takeover_fresh_request_count
        request_count += 1
        identity = (
            f"{pin.revision_id}:epoch={pin.policy_epoch}:"
            f"takeover={takeover_generation}:request={request_count:06d}"
        )
        request = InferenceRequest(
            request_index=request_count,
            request_id=f"request:{identity}",
            result_id=f"result:{identity}",
            chunk_id=f"chunk:{identity}",
            proposal_id=f"proposal:{identity}",
            t_ref_ns=int(anchor_ns),
            revision_id=pin.revision_id,
            model_revision=pin.model_revision,
            policy_epoch=pin.policy_epoch,
            takeover_generation=takeover_generation,
        )
        if takeover_generation > 0:
            post_takeover_fresh_request_count += 1
        coordinator.reserve_inference()

        def call() -> tuple[np.ndarray, float]:
            started = time.perf_counter()
            with coordinator.inference_slot(reserved=True):
                result = infer(sample, request)
            return result, (time.perf_counter() - started) * 1000.0

        try:
            return pool.submit(call), request
        except BaseException:
            coordinator.cancel_inference_reservation()
            raise

    def collect(
        pending: tuple[Future, InferenceRequest],
        *,
        adopt: bool,
    ) -> bool:
        nonlocal old_result_post_takeover_adopt_count
        nonlocal old_result_post_takeover_reject_count
        nonlocal post_takeover_fresh_chunk_adopt_count
        completed, request = pending
        chunk, latency = completed.result()
        latency_ms.append(latency)
        generation_matches = (
            request.revision_id == pin.revision_id
            and request.model_revision == pin.model_revision
            and request.policy_epoch == pin.policy_epoch
            and request.takeover_generation == takeover_generation
        )
        if not generation_matches:
            old_result_post_takeover_reject_count += 1
            return False
        if not adopt:
            return False
        cache.adopt(
            chunk,
            anchor_ns=request.t_ref_ns,
            revision_id=request.revision_id,
            policy_epoch=request.policy_epoch,
            takeover_generation=request.takeover_generation,
        )
        if request.takeover_generation > 0:
            post_takeover_fresh_chunk_adopt_count += 1
        return True

    def sleep_until(timestamp_ns: int) -> None:
        target = wall_start + ((timestamp_ns - start_ns) / 1e9) * realtime_scale
        delay = target - time.monotonic()
        if delay > 0:
            time.sleep(delay)

    with coordinator.worker_alive("actor"), ThreadPoolExecutor(
        max_workers=1, thread_name_prefix="online-actor-inference"
    ) as pool:
        if initial_chunk is not None:
            cache.adopt(
                initial_chunk,
                anchor_ns=start_ns,
                revision_id=pin.revision_id,
                policy_epoch=pin.policy_epoch,
                takeover_generation=takeover_generation,
            )
        coordinator.begin_actor_window(cache.remaining / 10.0)
        try:
            start_barrier.wait()
            timeline = [
                (int(timestamp_ns), 1, "dispatch", sample)
                for timestamp_ns, sample in zip(
                    timestamps_ns, samples, strict=True
                )
            ]
            for window in windows:
                timeline.extend((
                    (window.start_ns, 0, "takeover", window),
                    (window.resume_ns, 0, "resume", window),
                ))
            for timestamp_ns, _priority, event, payload in sorted(timeline):
                sleep_until(timestamp_ns)
                if event == "takeover":
                    window = payload
                    if takeover_active or window.takeover_generation != takeover_generation + 1:
                        raise AsyncRuntimeError("ONLINE_REPLAY_ASYNC_TAKEOVER_GENERATION_INVALID")
                    cache.flush()
                    if future is not None:
                        retired.append(future)
                        future = None
                    takeover_generation = window.takeover_generation
                    takeover_active = True
                    resume_wait = False
                    coordinator.end_actor_window()
                    continue
                if event == "resume":
                    if not takeover_active or payload.takeover_generation != takeover_generation:
                        raise AsyncRuntimeError("ONLINE_REPLAY_ASYNC_TAKEOVER_RESUME_INVALID")
                    takeover_active = False
                    resume_wait = True
                    continue

                sample = payload
                for old in tuple(retired):
                    if old[0].done():
                        collect(old, adopt=False)
                        retired.remove(old)
                if takeover_active:
                    continue
                if resume_wait:
                    future = submit(pool, sample, timestamp_ns)
                    if not collect(future, adopt=True):
                        raise AsyncRuntimeError("ONLINE_REPLAY_ASYNC_RESUME_RESULT_REJECTED")
                    future = None
                    resume_wait = False
                    coordinator.begin_actor_window(cache.remaining / 10.0)
                elif future is not None and future[0].done():
                    collect(future, adopt=True)
                    future = None
                action = cache.dispatch(
                    timestamp_ns=timestamp_ns,
                    pin=pin,
                    takeover_generation=takeover_generation,
                )
                if action is None:
                    underruns += 1
                if cache.needs_replan and future is None:
                    future = submit(pool, sample, timestamp_ns)
                coordinator.update_action_coverage(cache.remaining / 10.0)
            if future is not None:
                collect(future, adopt=False)
            for old in retired:
                collect(old, adopt=False)
        finally:
            coordinator.end_actor_window()

    return {
        "request_count": request_count,
        "latency_p50_p95_max_ms": percentile_latency_ms(latency_ms),
        "queue_underrun_count": underruns,
        "stale_action_count": cache.stale_count,
        "old_result_post_takeover_adopt_count": (
            old_result_post_takeover_adopt_count
        ),
        "old_result_post_takeover_reject_count": (
            old_result_post_takeover_reject_count
        ),
        "post_takeover_fresh_request_count": (
            post_takeover_fresh_request_count
        ),
        "post_takeover_fresh_chunk_adopt_count": (
            post_takeover_fresh_chunk_adopt_count
        ),
        "takeover_generation": takeover_generation,
    }


def run_concurrent_window(
    *,
    actor: Callable[[threading.Barrier], dict[str, Any]],
    learner: Callable[[threading.Barrier], dict[str, Any]],
    coordinator: InferencePriorityCoordinator,
) -> tuple[dict[str, Any], dict[str, Any]]:
    barrier = threading.Barrier(2)
    with ThreadPoolExecutor(max_workers=2, thread_name_prefix="online-runtime") as pool:
        actor_future = pool.submit(actor, barrier)
        learner_future = pool.submit(learner, barrier)
        coordinator.wait_for_both_workers(timeout_s=30.0)
        actor_result = actor_future.result()
        learner_result = learner_future.result()
    if not coordinator.concurrently_alive:
        raise AsyncRuntimeError("ONLINE_REPLAY_ASYNC_WORKERS_NOT_CONCURRENT")
    return actor_result, learner_result
