#!/usr/bin/env python3
"""Run one no-robot Stage-3 Actor/Learner concurrency window on GPU."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import sys
import threading
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for path in (SRC, ROOT / "tools"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_stage3_critic_warmup as warmup  # noqa: E402
import run_stage3_joint_update as joint  # noqa: E402
from forcesmolvla.rft.stage3.async_runtime import (  # noqa: E402
    EpisodePin,
    InferenceRequest,
    InferencePriorityCoordinator,
    PinnedEpisode,
    TakeoverWindow,
    run_concurrent_window,
    run_timed_actor,
)


ACTIVE_REVISION_ID = "stage3-online-r-joint-cycle-000010-candidate"
ACTIVE_MODEL_REVISION = "ab97aefb6a916a4f03e02d264e6c4b2f5c6462d2a7d6e1e9ebcd171d3a527c6b"
ACTIVE_ACTOR_PACKAGE = (
    ROOT
    / "artifacts/development/stage3/published"
    / "stage3_joint_cycle_000010_candidate.v1"
)
REVISION_REGISTRY = (
    ROOT
    / "artifacts/development/stage3/runtime"
    / "stage3_policy_revision_registry.json"
)
RESUME_CHECKPOINT = warmup.FORMAL_R_ROOT / "checkpoints/stage3_async_joint_cycle_000021_pending"
PENDING_CHECKPOINT = warmup.FORMAL_R_ROOT / "checkpoints/stage3_async_joint_cycle_000022_pending"
PENDING_CANDIDATE_ID = "stage3-online-r-joint-cycle-000022-candidate"
FIXTURE_SESSION_ID = "stage3-joint-cycle000010-validation-20260830-002"
RUNTIME_EPISODE_ID = "stage3-async-no-robot-window"
FIXTURE_INTERVENTIONS = (
    Path("/home/rlc123/fr3_client_ws/datasets")
    / "task2_policy_execute_stage3_joint_cycle000010_validation_20260830_002"
    / "integrated_capture/episode_000000/streams/policy_execute_intervention.jsonl"
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _fixture_rows(all_r: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = sorted(
        (
            row for row in all_r
            if row["identity"].get("session_id") == FIXTURE_SESSION_ID
        ),
        key=lambda row: int(row["observation"]["source_t_ref_monotonic_ns"]),
    )
    require(len(rows) == 395, "STAGE3_ASYNC_FIXTURE_TRANSITION_COUNT")
    timestamps = [int(row["observation"]["source_t_ref_monotonic_ns"]) for row in rows]
    require(
        all(right > left for left, right in zip(timestamps, timestamps[1:])),
        "STAGE3_ASYNC_FIXTURE_TIME_ORDER",
    )
    return rows


def _takeover_windows() -> list[TakeoverWindow]:
    active: dict[str, Any] | None = None
    windows: list[TakeoverWindow] = []
    for line in FIXTURE_INTERVENTIONS.read_text(encoding="utf-8").splitlines():
        event = json.loads(line)
        if event["event"] == "intervention_start":
            require(active is None, "STAGE3_ASYNC_NESTED_TAKEOVER")
            active = event
        elif event["event"] == "intervention_end":
            require(active is not None, "STAGE3_ASYNC_TAKEOVER_END_WITHOUT_START")
            require(
                int(event["takeover_generation"])
                == int(active["takeover_generation"]),
                "STAGE3_ASYNC_TAKEOVER_GENERATION_MISMATCH",
            )
            windows.append(TakeoverWindow(
                start_ns=int(active["receive_monotonic_ns"]),
                resume_ns=int(event["receive_monotonic_ns"]),
                takeover_generation=int(event["takeover_generation"]),
            ))
            active = None
    require(active is None and windows, "STAGE3_ASYNC_TAKEOVER_WINDOW_MISSING")
    return windows


def _load_actor(device: torch.device):
    from forcesmolvla.modeling_forcesmolvla import ForceSmolVLAPolicy

    candidate = json.loads(
        (ACTIVE_ACTOR_PACKAGE / "candidate.json").read_text(encoding="utf-8")
    )
    require(
        candidate.get("revision_id") == ACTIVE_REVISION_ID
        and candidate.get("model_revision") == ACTIVE_MODEL_REVISION
        and candidate.get("state") == "published"
        and candidate.get("published") is True,
        "STAGE3_ASYNC_ACTIVE_ACTOR_PACKAGE_MISMATCH",
    )
    actor = ForceSmolVLAPolicy.from_pretrained(
        ACTIVE_ACTOR_PACKAGE,
        local_files_only=True,
        force_download=False,
        strict=True,
        artifact_use="development",
    ).to(device)
    actor.eval()
    for parameter in actor.parameters():
        parameter.requires_grad_(False)
    return actor


def reconcile_post_checkpoint_replay(credits, all_r) -> int:
    """Mint credit once for live replay UIDs admitted after the checkpoint."""

    uids = [str(row["identity"]["transition_uid"]) for row in all_r]
    require(len(set(uids)) == len(uids), "STAGE3_ASYNC_LIVE_REPLAY_UID_DUPLICATE")
    minted = sum(credits.mint_for_unique_online_transition(uid) for uid in uids)
    require(
        credits.snapshot().credited_transition_count == len(uids)
        and credits.snapshot().available > 0,
        "STAGE3_ASYNC_REPLAY_OR_CREDIT_MISMATCH",
    )
    return minted


def _prepare_learner(
    device: torch.device,
    all_r,
    r_macros,
    source_episodes,
    *,
    resume_checkpoint: Path = RESUME_CHECKPOINT,
):
    from forcesmolvla.rft.critic import frozen_task_feature
    from forcesmolvla.rft.frozen_vlm_trainability import (
        FROZEN_PREFIXES,
        apply_frozen_vlm_trainability,
        build_frozen_vlm_actor_optimizer,
    )
    from forcesmolvla.rft.stage3.update_credit import UpdateCreditLedger
    from forcesmolvla.rft.throughput_v2 import FrozenPrefixFlowCounter
    from forcesmolvla.training_data import load_normalizer_manifest

    actor_package = resume_checkpoint / "candidate_policy"
    actor, q1, q2, q1_target, q2_target, binding, config = joint.load_resume_modules(
        resume_checkpoint,
        actor_package,
        device,
        allow_checkpoint_candidate=True,
    )
    trainability = apply_frozen_vlm_trainability(actor)
    critic_parameters = [
        parameter for module in (q1, q2) for parameter in module.parameters()
        if parameter.requires_grad
    ]
    critic_optimizer = torch.optim.Adam(
        critic_parameters,
        lr=3e-4,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.0,
    )
    actor_optimizer, actor_scheduler, actor_ownership = (
        build_frozen_vlm_actor_optimizer(
            actor, lr=float(config["optimizer"]["actor"]["lr"]),
        )
    )
    modules = {
        "q1": q1,
        "q2": q2,
        "q1_target": q1_target,
        "q2_target": q2_target,
    }
    runtime = joint.load_joint_checkpoint_once(
        resume_checkpoint,
        actor=actor,
        modules=modules,
        critic_optimizer=critic_optimizer,
        actor_optimizer=actor_optimizer,
        actor_scheduler=actor_scheduler,
        device=device,
    )
    counters = runtime["counters"]
    joint_cycles = int(counters["joint_cycles"])
    require(
        counters == {
            "joint_cycles": joint_cycles,
            "critic_optimizer_steps": joint_cycles * 2,
            "actor_optimizer_steps": joint_cycles,
            "target_polyak_steps": joint_cycles * 2,
        }
        and critic_optimizer.state
        and actor_optimizer.state
        and actor_scheduler.last_epoch == joint_cycles,
        "STAGE3_ASYNC_LEARNER_EXACT_RESUME_INVALID",
    )
    credits = UpdateCreditLedger.from_state_dict(runtime["sample_credit"])
    new_r_transition_count = reconcile_post_checkpoint_replay(credits, all_r)

    random.setstate(runtime["rng_state"]["python"])
    np.random.set_state(runtime["rng_state"]["numpy"])
    torch.set_rng_state(runtime["rng_state"]["torch_cpu"])
    torch.cuda.set_rng_state_all(runtime["rng_state"]["torch_cuda"])
    critic_noise = torch.Generator(device=device)
    critic_noise.set_state(runtime["rng_state"]["critic_noise_generator"])
    r_rng, d_rng = random.Random(), random.Random()
    r_rng.setstate(runtime["sampler_state"]["r_rng"])
    d_rng.setstate(runtime["sampler_state"]["d_rng"])

    frozen = [
        parameter for name, parameter in actor.named_parameters()
        if name.startswith(FROZEN_PREFIXES)
    ]
    joint.assert_optimizer_ownership(
        actor_optimizer, critic_optimizer, frozen_parameters=frozen,
    )
    require(
        actor_ownership["frozen_parameter_in_optimizer"] == 0
        and trainability.trainable_actor_parameter_tensors
        == actor_ownership["parameter_tensor_count"],
        "STAGE3_ASYNC_OPTIMIZER_OWNERSHIP",
    )

    normalizer = load_normalizer_manifest(
        Path(binding["normalizer_binding"]["absolute_path"])
    )
    r_replay = warmup.FormalReplay(r_macros, source_episodes, normalizer)
    d_replay = joint.JointDemoReplay(normalizer)
    schedules = joint.make_schedules(
        r_rng,
        d_rng,
        r_population_size=len(r_macros),
        d_population=d_replay.population,
        cycles=1,
    )
    critic_r, critic_d, actor_r, actor_d = schedules
    d_replay.prefetch_joint(critic_d, actor_d)
    feature = torch.from_numpy(frozen_task_feature()).to(
        device=device, dtype=torch.float32
    )
    delta_mean = torch.tensor(
        normalizer.delta_action7.mean, dtype=torch.float32, device=device
    )
    delta_std = torch.tensor(
        normalizer.delta_action7.std, dtype=torch.float32, device=device
    )
    flow = FrozenPrefixFlowCounter(
        inference_batch_size=int(config["batching"]["flow_inference_subbatch"])
    )
    return {
        "actor": actor,
        "q1": q1,
        "q2": q2,
        "q1_target": q1_target,
        "q2_target": q2_target,
        "modules": modules,
        "binding": binding,
        "config": config,
        "critic_optimizer": critic_optimizer,
        "actor_optimizer": actor_optimizer,
        "actor_scheduler": actor_scheduler,
        "runtime": runtime,
        "credits": credits,
        "new_r_transition_count": new_r_transition_count,
        "critic_noise": critic_noise,
        "r_rng": r_rng,
        "d_rng": d_rng,
        "r_replay": r_replay,
        "d_replay": d_replay,
        "critic_r": critic_r,
        "critic_d": critic_d,
        "actor_r": actor_r,
        "actor_d": actor_d,
        "feature": feature,
        "delta_mean": delta_mean,
        "delta_std": delta_std,
        "flow": flow,
        "normalizer": normalizer,
        "resume_checkpoint": resume_checkpoint,
    }


def run(*, checkpoint: Path, realtime_scale: float) -> dict[str, Any]:
    from preflight_s2_g4_losses_gpu import actor_batch
    from forcesmolvla.rft.stage3.publication import load_revision_registry
    from forcesmolvla.rft.throughput_v2 import FrozenPrefixFlowCounter

    require(torch.cuda.is_available(), "STAGE3_ASYNC_CUDA_UNAVAILABLE")
    require(not checkpoint.exists(), "STAGE3_ASYNC_PENDING_CHECKPOINT_EXISTS")
    device = torch.device("cuda:0")
    replay_before = tuple(
        sorted(
            (path.name, path.stat().st_size, path.stat().st_mtime_ns)
            for path in (warmup.FORMAL_R_ROOT / "replay").glob("*.json")
        )
    )
    all_r, r_macros, source_episodes = warmup.load_formal_online_r(
        warmup.FORMAL_R_ROOT
    )
    fixture_rows = _fixture_rows(all_r)
    active_actor = _load_actor(device)
    learner = _prepare_learner(device, all_r, r_macros, source_episodes)
    fixture_replay = warmup.FormalReplay((), source_episodes, learner["normalizer"])
    actor_flow = FrozenPrefixFlowCounter(inference_batch_size=1)
    actor_noise = torch.Generator(device=device).manual_seed(840_021)
    inference_stream = torch.cuda.Stream(device=device, priority=-1)
    coordinator = InferencePriorityCoordinator()
    actor_window_active = threading.Event()
    learner_state: dict[str, Any] = {}
    oom_count = 0
    nonfinite_count = 0

    machine = load_revision_registry(REVISION_REGISTRY, fresh_process=False)
    expected_pin = EpisodePin(
        ACTIVE_REVISION_ID, ACTIVE_MODEL_REVISION, machine.policy_epoch,
    )

    def infer(row: dict[str, Any], request: InferenceRequest) -> np.ndarray:
        episode_id = str(row["identity"]["episode_id"])
        sample = fixture_replay._sample(
            row["observation"],
            f"{RUNTIME_EPISODE_ID}:{request.request_id}",
            episode_id,
        )
        with torch.cuda.stream(inference_stream):
            batch = actor_batch(
                active_actor, [sample], device, include_action=False
            )
            noise = torch.randn(
                1, 50, 7,
                dtype=torch.float32,
                device=device,
                generator=actor_noise,
            )
            with torch.no_grad(), torch.autocast(
                device_type="cuda", dtype=torch.bfloat16
            ):
                chunk = actor_flow.sample(
                    active_actor,
                    batch,
                    noise,
                    call_id=f"stage3-async-actor-{request.request_index:06d}",
                    purpose="td_next",
                )
        inference_stream.synchronize()
        return chunk[0].detach().cpu().numpy()

    timestamps = [
        int(row["observation"]["source_t_ref_monotonic_ns"])
        for row in fixture_rows
    ]
    warmup_started = torch.cuda.Event(enable_timing=True)
    warmup_finished = torch.cuda.Event(enable_timing=True)
    warmup_request = InferenceRequest(
        request_index=1,
        request_id="request:stage3-async-warmup:000001",
        result_id="result:stage3-async-warmup:000001",
        chunk_id="chunk:stage3-async-warmup:000001",
        proposal_id="proposal:stage3-async-warmup:000001",
        t_ref_ns=timestamps[0],
        revision_id=expected_pin.revision_id,
        model_revision=expected_pin.model_revision,
        policy_epoch=expected_pin.policy_epoch,
        takeover_generation=0,
    )
    with coordinator.inference_slot():
        warmup_started.record(inference_stream)
        warmup_chunk = infer(fixture_rows[0], warmup_request)
        warmup_finished.record(inference_stream)
        warmup_finished.synchronize()
    warmup_latency_ms = float(warmup_started.elapsed_time(warmup_finished))

    def actor_worker(barrier: threading.Barrier) -> dict[str, Any]:
        actor_window_active.set()
        try:
            return run_timed_actor(
                timestamps_ns=timestamps,
                samples=fixture_rows,
                infer=infer,
                coordinator=coordinator,
                pin=expected_pin,
                start_barrier=barrier,
                realtime_scale=realtime_scale,
                initial_chunk=warmup_chunk,
                initial_latency_ms=warmup_latency_ms,
                takeover_windows=_takeover_windows(),
            )
        finally:
            actor_window_active.clear()

    def learner_worker(barrier: threading.Barrier) -> dict[str, Any]:
        nonlocal oom_count, nonfinite_count
        critic_during_window = actor_during_window = 0
        td_losses: list[float] = []
        selected_identities: list[str] = []

        def learner_slot(kind: str):
            return coordinator.learner_step_slot(kind)

        with coordinator.worker_alive("learner"):
            barrier.wait()
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
                selected_identities.extend(row["identity"] for row in rows)
                try:
                    with learner_slot("critic_batch_prepare"):
                        batch = warmup.build_batch(
                            rows,
                            learner["actor"],
                            learner["feature"],
                            device,
                        )
                    record = joint.critic_step(
                        step=42 + substep,
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
                td_losses.append(record["loss"])
                if actor_window_active.is_set():
                    critic_during_window += 1

            rows = [
                joint._online_actor_row(learner["r_replay"], index)
                for index in learner["actor_r"][0]
            ]
            rows.extend(
                learner["d_replay"].materialize_actor(index)
                for index in learner["actor_d"][0]
            )
            selected_identities.extend(row["identity"] for row in rows)
            try:
                with learner_slot("actor_batch_prepare"):
                    batch = joint.build_actor_training_batch(
                        rows,
                        learner["actor"],
                        learner["feature"],
                        device,
                    )
                actor_record = joint.actor_step(
                    cycle=21,
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
            if actor_window_active.is_set():
                actor_during_window += 1

        current_sampled = any(
            RUNTIME_EPISODE_ID in identity for identity in selected_identities
        )
        learner_state.update(
            td_losses=td_losses,
            actor_record=actor_record,
            current_episode_sampled=current_sampled,
        )
        return {
            "critic_steps_during_actor_window": critic_during_window,
            "actor_steps_during_actor_window": actor_during_window,
            "current_episode_sampled": current_sampled,
        }

    with PinnedEpisode(machine, expected_pin):
        actor_result, learner_result = run_concurrent_window(
            actor=actor_worker,
            learner=learner_worker,
            coordinator=coordinator,
        )

    replay_after = tuple(
        sorted(
            (path.name, path.stat().st_size, path.stat().st_mtime_ns)
            for path in (warmup.FORMAL_R_ROOT / "replay").glob("*.json")
        )
    )
    final_machine = load_revision_registry(REVISION_REGISTRY, fresh_process=False)
    active_unchanged = (
        final_machine.active_revision_id == ACTIVE_REVISION_ID
        and final_machine.policy_epoch == expected_pin.policy_epoch
        and final_machine.pending_revision_id is None
    )
    slowest = coordinator.slowest_learner_microstep
    target_met = (
        coordinator.concurrently_alive
        and actor_result["queue_underrun_count"] == 0
        and actor_result["stale_action_count"] == 0
        and actor_result["old_result_post_takeover_adopt_count"] == 0
        and actor_result["post_takeover_fresh_request_count"] >= 1
        and actor_result["post_takeover_fresh_chunk_adopt_count"] >= 1
        and learner_result["critic_steps_during_actor_window"] >= 2
        and learner_result["actor_steps_during_actor_window"] >= 1
        and not learner_result["current_episode_sampled"]
        and replay_after == replay_before
        and active_unchanged
        and nonfinite_count == 0
        and oom_count == 0
    )
    if not target_met:
        raise RuntimeError(
            "STAGE3_ASYNC_RUNTIME_TARGET_NOT_MET:"
            + json.dumps({
                "queue_underrun_count": actor_result["queue_underrun_count"],
                "stale_action_count": actor_result["stale_action_count"],
                "old_result_post_takeover_adopt_count": actor_result[
                    "old_result_post_takeover_adopt_count"
                ],
                "post_takeover_fresh_request_count": actor_result[
                    "post_takeover_fresh_request_count"
                ],
                "post_takeover_fresh_chunk_adopt_count": actor_result[
                    "post_takeover_fresh_chunk_adopt_count"
                ],
                "inference_max_latency_ms": actor_result[
                    "latency_p50_p95_max_ms"
                ][2],
                "critic_steps_during_actor_window": learner_result[
                    "critic_steps_during_actor_window"
                ],
                "actor_steps_during_actor_window": learner_result[
                    "actor_steps_during_actor_window"
                ],
                "slowest_learner_microstep": slowest,
            }, sort_keys=True)
        )

    previous = learner["runtime"]["counters"]
    runtime_state = {
        "source_checkpoint": str(RESUME_CHECKPOINT),
        "flags": {"critic_ready": True, "actor_q_guidance_enabled": True},
        "counters": {
            "joint_cycles": int(previous["joint_cycles"]) + 1,
            "critic_optimizer_steps": int(previous["critic_optimizer_steps"]) + 2,
            "actor_optimizer_steps": int(previous["actor_optimizer_steps"]) + 1,
            "target_polyak_steps": int(previous["target_polyak_steps"]) + 2,
        },
        "replay": {
            "formal_r_root": str(warmup.FORMAL_R_ROOT),
            "unique_r_transition_count": len(all_r),
            "new_r_transition_count": 0,
            "eligible_ack_macro_count": len(r_macros),
            "mix": {"R": 32, "D": 32},
            "current_episode_sampled": False,
        },
        "sample_credit": learner["credits"].state_dict(),
        "sampler_state": {
            "cycle": int(previous["joint_cycles"]) + 1,
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
            "revision_id": PENDING_CANDIDATE_ID,
            "state": "candidate",
            "activated": False,
            "published": False,
            "coordinator_disposition": "pending_episode_boundary",
        },
        "step_metrics": {
            "critic_td_loss": learner_state["td_losses"],
            "actor_fm_loss": [learner_state["actor_record"]["fm_loss"]],
            "actor_min_twin_q_loss": [
                learner_state["actor_record"]["actor_q_loss"]
            ],
        },
    }
    joint.save_joint_checkpoint(
        checkpoint,
        actor=learner["actor"],
        modules=learner["modules"],
        critic_optimizer=learner["critic_optimizer"],
        actor_optimizer=learner["actor_optimizer"],
        actor_scheduler=learner["actor_scheduler"],
        runtime_state=runtime_state,
        parent_binding=learner["binding"],
        source_checkpoint=RESUME_CHECKPOINT,
        total_joint_cycles=22,
        candidate_revision_id=PENDING_CANDIDATE_ID,
    )
    require(
        replay_after == replay_before
        and not learner_result["current_episode_sampled"]
        and active_unchanged
        and nonfinite_count == 0
        and oom_count == 0,
        "STAGE3_ASYNC_COMPLETION_CONTRACT",
    )
    concurrent_training = target_met
    return {
        "ASYNC_ACTOR_LEARNER_IMPLEMENTED": True,
        "ACTOR_AND_LEARNER_CONCURRENTLY_ALIVE": coordinator.concurrently_alive,
        "ACTOR_REVISION_PINNED": True,
        "ACTOR_INFERENCE_REQUEST_COUNT": actor_result["request_count"],
        "ACTOR_INFERENCE_LATENCY_P50_P95_MAX_MS": actor_result[
            "latency_p50_p95_max_ms"
        ],
        "ACTION_QUEUE_UNDERRUN_COUNT": actor_result["queue_underrun_count"],
        "STALE_ACTION_COUNT": actor_result["stale_action_count"],
        "OLD_RESULT_POST_TAKEOVER_ADOPT_COUNT": actor_result[
            "old_result_post_takeover_adopt_count"
        ],
        "POST_TAKEOVER_FRESH_REQUEST_COUNT": actor_result[
            "post_takeover_fresh_request_count"
        ],
        "POST_TAKEOVER_FRESH_CHUNK_ADOPT_COUNT": actor_result[
            "post_takeover_fresh_chunk_adopt_count"
        ],
        "LEARNER_RESUME_CHECKPOINT": str(RESUME_CHECKPOINT),
        "LEARNER_CRITIC_STEPS_DURING_ACTOR_WINDOW": learner_result[
            "critic_steps_during_actor_window"
        ],
        "LEARNER_ACTOR_STEPS_DURING_ACTOR_WINDOW": learner_result[
            "actor_steps_during_actor_window"
        ],
        "CURRENT_EPISODE_SAMPLED_BY_LEARNER": False,
        "NONFINITE_COUNT": nonfinite_count,
        "OOM_COUNT": oom_count,
        "ACTIVE_REVISION_UNCHANGED": active_unchanged,
        "PENDING_CANDIDATE_ID": PENDING_CANDIDATE_ID,
        "CONCURRENT_COLLECTION_AND_TRAINING": concurrent_training,
        "LEARNER_SLOWEST_MICROSTEP": None if slowest is None else {
            "kind": slowest[0], "milliseconds": slowest[1],
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=PENDING_CHECKPOINT)
    parser.add_argument("--realtime-scale", type=float, default=1.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    print(json.dumps(run(
        checkpoint=args.checkpoint,
        realtime_scale=args.realtime_scale,
    ), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
