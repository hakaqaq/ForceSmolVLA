#!/usr/bin/env python3
"""Build a clearly named Stage-3 seed from an SFT Actor and warm-up Twin-Q."""

from __future__ import annotations

import argparse
from pathlib import Path
import random
import sys

import torch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for path in (SRC, ROOT / "tools"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from forcesmolvla.rft.online import replay_training as warmup  # noqa: E402
from forcesmolvla.rft.online.q_gradient_controller import (  # noqa: E402
    QGradientRatioController,
)
from forcesmolvla.rft.online.sample_credit import UpdateCreditLedger  # noqa: E402
import train_forcerft_actor_critic as joint  # noqa: E402


SEED_DIRECTORY_NAME = "stage3_sft_actor_warmup_critic_cycle_000000"


def build_stage3_seed_bundle(
    *,
    task_id: str,
    output_root: Path,
    dataset_root: Path,
    reward_transition_root: Path,
    actor_checkpoint: Path,
    critic_checkpoint: Path,
    checkpoint: Path,
    normalizer: Path,
    action_contract: Path,
    common_online_config: Path,
    reward_detector_manifest: Path | None = None,
    reward_calibration_manifest: Path | None = None,
) -> Path:
    device = torch.device("cpu")
    warmup.configure_task_paths(
        task_id=task_id,
        dataset_root=dataset_root,
        reward_transition_root=reward_transition_root,
        output_root=output_root,
    )
    (
        actor,
        _q1,
        _q2,
        _q1_target,
        _q2_target,
        modules,
        critic_optimizer,
        critic_scheduler,
        actor_optimizer,
        actor_scheduler,
        _actor_ownership,
        _config,
    ) = joint.load_offline_training_parents(
        actor_checkpoint=actor_checkpoint,
        critic_checkpoint=critic_checkpoint,
        device=device,
        actor_lr_override=1.0e-6,
        production_config=True,
    )
    parent_rng = torch.load(
        critic_checkpoint / "state/rng_states.pt",
        map_location="cpu",
        weights_only=False,
    )
    r_rng = random.Random(4405)
    d_rng = random.Random(4406)
    controller = QGradientRatioController()
    runtime_state = {
        "online_joint_cycles": 0,
        "source_checkpoint": str(critic_checkpoint.resolve()),
        "actor_parent_checkpoint": str(actor_checkpoint.resolve()),
        "reference_actor_checkpoint": str(actor_checkpoint.resolve()),
        "critic_only_updates": 0,
        "flags": {
            "critic_ready": False,
            "critic_updates_enabled": True,
            "actor_updates_enabled": False,
            "actor_q_guidance_enabled": False,
        },
        "counters": {
            "joint_cycles": 0,
            "critic_optimizer_steps": 0,
            "actor_optimizer_steps": 0,
            "target_polyak_steps": 0,
            "critic_parent_optimizer_steps": 256,
        },
        "replay": {
            "formal_r_root": str((output_root / "online").resolve()),
            "unique_r_transition_count": 0,
            "new_r_transition_count": 0,
            "eligible_ack_macro_count": 0,
            "actor_q_valid_ack_rows": 0,
            "current_episode_sampled": False,
        },
        "sample_credit": UpdateCreditLedger(
            credits_per_transition=1,
            credits_per_joint_cycle=1,
        ).state_dict(),
        "sampler_state": {
            "cycle": 0,
            "r_rng": r_rng.getstate(),
            "d_rng": d_rng.getstate(),
        },
        "rng_state": {
            "python": parent_rng["python_random_state"],
            "numpy": parent_rng["numpy_random_state"],
            "torch_cpu": parent_rng["torch_cpu_rng_state"],
            "torch_cuda": parent_rng["torch_cuda_rng_states"],
            "critic_noise_generator": parent_rng["named_generator_states"][
                "td_next_action_flow_noise"
            ],
        },
        "q_gradient_controller": controller.state_dict(),
        "optimizer_ownership": {
            "overlap": 0,
            "actor_optimizer_fresh": True,
            "critic_optimizer_restored_from_warmup": True,
        },
        "runtime_artifacts": {
            "normalizer": str(normalizer.resolve()),
            "action_contract": str(action_contract.resolve()),
            "common_online_config": str(common_online_config.resolve()),
            "reward_detector_manifest": (
                ""
                if reward_detector_manifest is None
                else str(reward_detector_manifest.resolve())
            ),
            "reward_calibration_manifest": (
                ""
                if reward_calibration_manifest is None
                else str(reward_calibration_manifest.resolve())
            ),
        },
        "step_metrics": {
            "critic_td_loss": [],
            "actor_fm_loss": [],
            "actor_min_twin_q_loss": [],
        },
    }
    joint.save_joint_checkpoint(
        checkpoint,
        actor=actor,
        modules=modules,
        critic_optimizer=critic_optimizer,
        actor_optimizer=actor_optimizer,
        actor_scheduler=actor_scheduler,
        critic_scheduler=critic_scheduler,
        runtime_state=runtime_state,
        parent_binding=None,
        actor_parent_path=actor_checkpoint,
        parent_binding_id=f"{task_id}-stage3-sft-actor-warmup-critic",
        source_checkpoint=critic_checkpoint,
        total_joint_cycles=0,
        actor_checkpoint_id=f"{task_id}-stage3-sft-actor-warmup-critic-cycle-000000",
        checkpoint_kind="stage3_safe_seed_v1",
        actor_directory="actor",
        metadata_overrides={
            "actor_source_kind": "sft",
            "actor_equal_to_sft": True,
            "actor_optimizer_steps": 0,
            "actor_updates_enabled": False,
            "actor_q_guidance_enabled": False,
            "critic_updates_enabled": True,
            "reference_actor_source": "sft",
            "robot_execution_actor": "sft",
            "legacy_actor210_parent": False,
        },
    )
    return checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--reward-transition-root", type=Path, required=True)
    parser.add_argument("--actor-checkpoint", type=Path, required=True)
    parser.add_argument("--critic-checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--normalizer", type=Path)
    parser.add_argument("--action-contract", type=Path)
    parser.add_argument(
        "--common-online-config",
        type=Path,
        default=ROOT / "configs/forcerft/actor_critic_common.yaml",
    )
    parser.add_argument("--reward-detector-manifest", type=Path)
    parser.add_argument("--reward-calibration-manifest", type=Path)
    args = parser.parse_args()
    args.checkpoint = (
        args.output_root / "stage3_seed/checkpoints" / SEED_DIRECTORY_NAME
        if args.checkpoint is None
        else args.checkpoint
    )
    args.normalizer = (
        args.dataset_root / "normalizer_manifest.json"
        if args.normalizer is None
        else args.normalizer
    )
    args.action_contract = (
        args.actor_checkpoint / "manifests/action_delta_spec.json"
        if args.action_contract is None
        else args.action_contract
    )
    return args


def main() -> int:
    args = parse_args()
    result = build_stage3_seed_bundle(**vars(args))
    print(result.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
