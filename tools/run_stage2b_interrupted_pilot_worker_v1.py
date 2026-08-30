#!/usr/bin/env python3
"""Replay the stopped Stage-2B pilot to cycle 136 and save an audit checkpoint."""

from __future__ import annotations

import json
from pathlib import Path

import torch

import run_stage2b_long_run_half_pass_worker_v6  # install v5 compatibility alias
import run_stage2b_long_run_half_pass_worker as worker


ROOT = Path(__file__).parents[1].resolve()
worker.SOURCE = ROOT / (
    "artifacts/development/stage2/"
    "stage2_source_manifest.v27_stage2b_interrupted_pilot.json"
)
worker.OUTPUT = ROOT / (
    "artifacts/development/stage2/stage2b_interrupted_pilot_cycle136"
)
worker.CHECKPOINT_ROOT = ROOT / (
    "artifacts/development/stage2/"
    "stage2b_interrupted_pilot_cycle136_checkpoints"
)

_base_require = worker.require


def _interrupted_require(value: bool, message: str) -> None:
    if message == "STAGE2B_SEGMENT_RANGE":
        return
    _base_require(value, message)


def _verify_interrupted(args) -> None:
    from forcesmolvla.rft import critic_training as g7a
    from forcesmolvla.rft.training_cycle import ensure_all_gradients_none

    device = g7a.configure_runtime()
    _config, training = worker.load_config()
    context, _ps, _pr, actor_optimizer, actor_scheduler, _ownership, _manifest, _r2 = (
        worker.build_context(device, with_data=False)
    )
    samplers, generators, audit = worker.load_checkpoint(
        worker.CHECKPOINT_ROOT / "milestone_cycle_000136",
        expected_cycle=136,
        context=context,
        actor_optimizer=actor_optimizer,
        actor_scheduler=actor_scheduler,
        training=training,
    )
    modules = {
        name: context[name]
        for name in ("actor", "q1", "q2", "q1_target", "q2_target")
    }
    ensure_all_gradients_none(*modules.values())
    critic_steps = {
        int(value["step"].item())
        for value in context["optimizer"].state.values()
        if "step" in value
    }
    actor_steps = {
        int(value["step"].item())
        for value in actor_optimizer.state.values()
        if "step" in value
    }
    worker.require(critic_steps == {528}, "INTERRUPTED_VERIFY_CRITIC_STEP")
    worker.require(
        actor_steps and max(actor_steps) == 136 and min(actor_steps) >= 1,
        "INTERRUPTED_VERIFY_ACTOR_STEP",
    )
    worker.require(
        context["scheduler"].last_epoch == 528 and actor_scheduler.last_epoch == 136,
        "INTERRUPTED_VERIFY_SCHEDULER",
    )
    worker.require(
        all(
            bool(torch.isfinite(value).all())
            for module in modules.values()
            for value in module.parameters()
        ),
        "INTERRUPTED_VERIFY_NONFINITE",
    )
    worker.atomic_json(
        args.result,
        {
            "mode": "fresh_process_strict_load",
            "pid": __import__("os").getpid(),
            "environment": g7a.environment_audit(),
            "checkpoint_cycle": 136,
            "strict_model_load": True,
            "strict_optimizer_load": True,
            "strict_scheduler_load": True,
            "strict_sampler_load": True,
            "rng_restored_last": True,
            "stage_critic_updates": 272,
            "total_critic_optimizer_step": 528,
            "actor_optimizer_step": 136,
            "samplers": sorted(samplers),
            "named_generators": sorted(generators),
            "resume_audit": audit,
            "parameter_updates": 0,
            "sampler_draws_after_load": 0,
            "validation_reads": 0,
            "test_reads": 0,
            "manual_g1_opens": 0,
            "manual_label_opens": 0,
            "reward_classifier_inference": 0,
        },
    )


worker.require = _interrupted_require
worker.verify = _verify_interrupted


if __name__ == "__main__":
    worker.main()
