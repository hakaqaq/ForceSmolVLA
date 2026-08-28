#!/usr/bin/env python3
"""Recreate a lost Stage-2B boundary report without any training update."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

import run_stage2b_long_run_half_pass_worker_v5  # installs append-only bindings
import run_stage2b_long_run_half_pass_worker as worker


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cycle", type=int, choices=(0, 105), required=True)
    parser.add_argument("--result", type=Path, required=True)
    args = parser.parse_args()
    worker.require(not args.result.exists(), "STAGE2B_RECOVERY_AUDIT_APPEND_ONLY")

    import preflight_s2_g5_single_cycle_gpu as g5
    import run_s2_g7a_worker as g7a
    from forcesmolvla.rft.canonical_state import canonical_digest
    from forcesmolvla.rft.frozen_vlm_trainability import frozen_state_digest
    from forcesmolvla.rft.training_cycle import ensure_all_gradients_none

    g5.install_open_audit()
    device = g7a.configure_runtime()
    _config, training = worker.load_config()
    (
        context, _parent_sampler_states, _parent_rng,
        actor_optimizer, actor_scheduler, _ownership, _trainability, _r2,
    ) = worker.build_context(device, with_data=True)
    checkpoint = worker.CHECKPOINT_ROOT / f"milestone_cycle_{args.cycle:06d}"
    samplers, generators, resume = worker.load_checkpoint(
        checkpoint,
        expected_cycle=args.cycle,
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
    frozen_reference = frozen_state_digest(context["actor"])
    fixed = torch.load(worker.FIXED, map_location=device, weights_only=False)
    worker.require(worker.sha(worker.FIXED) == "002235cfc18cf939652c7a1bbe27ca0e752cf2e25e89fa415465ccfb3e8777e2", "STAGE2B_FIXED_SHA")
    fixed_indices = list(context["data"].actor_population[:24])
    fixed_noise = torch.randn(
        1, 50, 7,
        generator=torch.Generator(device=device).manual_seed(19224),
        device=device,
    )
    rng_before = canonical_digest(g5.capture_rng_states(generators))
    sampler_before = {name: sampler.state_dict() for name, sampler in samplers.items()}
    boundary = worker.boundary_audit(
        cycle=args.cycle,
        context=context,
        frozen_reference=frozen_reference,
        fixed_indices=fixed_indices,
        fixed_noise=fixed_noise,
        device=device,
        generators=generators,
        g5=g5,
    )
    gradient = worker.gradient_scale_diagnostic(
        cycle=args.cycle,
        context=context,
        indices=fixed_indices,
        device=device,
        generators=generators,
        g5=g5,
    )
    validation = worker.validation(
        cycle=args.cycle,
        context=context,
        fixed=fixed,
        device=device,
        generators=generators,
        g5=g5,
    )
    ensure_all_gradients_none(*modules.values())
    worker.require(canonical_digest(g5.capture_rng_states(generators)) == rng_before, "STAGE2B_RECOVERY_AUDIT_CONSUMED_RNG")
    worker.require(
        {name: sampler.state_dict() for name, sampler in samplers.items()} == sampler_before,
        "STAGE2B_RECOVERY_AUDIT_CONSUMED_SAMPLER",
    )
    worker.require(frozen_state_digest(context["actor"]) == frozen_reference, "STAGE2B_RECOVERY_AUDIT_FROZEN_DRIFT")
    worker.atomic_json(args.result, {
        "mode": "zero_update_recovered_boundary_audit",
        "cycle": args.cycle,
        "pid": worker.os.getpid(),
        "environment": g7a.environment_audit(),
        "resume_audit": resume,
        "boundary_audit": boundary,
        "gradient_scale_diagnostic": gradient,
        "validation_diagnostic": validation,
        "optimizer_updates": 0,
        "polyak_updates": 0,
        "sampler_draws": 0,
        "training_rng_consumption": 0,
        "validation_transition_reads": 1205,
        "test_transition_reads": 0,
        "manual_g1_opens": len(g5.FORBIDDEN_OPENS["manual_g1"]),
        "manual_label_opens": len(g5.FORBIDDEN_OPENS["manual_labels"]),
        "reward_classifier_inference": 0,
        "reward_classifier_updates": 0,
    })


if __name__ == "__main__":
    main()
