#!/usr/bin/env python3
"""Build the online ACK-residual bootstrap checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from forcesmolvla.rft.critic import build_twin_q  # noqa: E402
from forcesmolvla.rft.online.residual_actor_critic_runtime import (  # noqa: E402
    ONLINE_ADAPTATION_DIRECTORY_NAME,
)
from forcesmolvla.rft.online.residual_actor_critic_checkpoint import (  # noqa: E402
    BOOTSTRAP_CHECKPOINT_KIND,
    save_residual_actor_critic_checkpoint,
)
from forcesmolvla.rft.online.transition_authority import (  # noqa: E402
    ONLINE_SEMANTICS_VERSION,
)
from forcesmolvla.rft.residual_actor import make_residual_actor_pair  # noqa: E402


BOOTSTRAP_DIRECTORY_NAME = "base_policy_zero_residual_random_twin_q"


def _load_base_actor(checkpoint: Path) -> torch.nn.Module:
    from forcesmolvla.modeling_forcesmolvla import ForceSmolVLAPolicy

    return ForceSmolVLAPolicy.from_pretrained(
        checkpoint,
        local_files_only=True,
        force_download=False,
        strict=True,
        artifact_use="development",
    )


def _normalizer_parameters_match(
    *, dataset_root: Path, frozen_base_policy_checkpoint: Path
) -> bool:
    from forcesmolvla.training_data import (
        load_checkpoint_runtime_artifacts,
        load_normalizer_manifest,
    )

    dataset_normalizer = load_normalizer_manifest(
        dataset_root / "normalizer_manifest.json"
    )
    base_runtime = load_checkpoint_runtime_artifacts(
        frozen_base_policy_checkpoint
    )
    return dataset_normalizer.manifest() == base_runtime.normalizer.manifest()


def build_online_residual_bootstrap(
    *,
    task_id: str,
    output_root: Path,
    dataset_root: Path,
    frozen_base_policy_checkpoint: Path,
    checkpoint: Path,
    online_residual_config: Path,
) -> Path:
    del output_root  # retained CLI path binding; no replay is read.
    frozen_base_policy_checkpoint = Path(frozen_base_policy_checkpoint).resolve()
    if not _normalizer_parameters_match(
        dataset_root=Path(dataset_root).resolve(),
        frozen_base_policy_checkpoint=frozen_base_policy_checkpoint,
    ):
        raise RuntimeError("FORCERFT_BOOTSTRAP_NORMALIZER_MISMATCH")
    config = yaml.safe_load(
        Path(online_residual_config).read_text(encoding="utf-8")
    )
    if int(config["batching"]["command_macro_slots"]) != 3:
        raise ValueError("FORCERFT_COMMAND_MACRO_SLOTS_INVALID")
    base_actor = _load_base_actor(frozen_base_policy_checkpoint).to("cpu")
    base_actor.eval().requires_grad_(False)
    if any(parameter.requires_grad for parameter in base_actor.parameters()):
        raise RuntimeError("FORCERFT_BASE_ACTOR_NOT_FROZEN")

    seed = int(config["environment"]["random_seed"])
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        residual_actor, residual_actor_target = make_residual_actor_pair(
            hidden_dim=int(config["wrist_wrench_residual_actor"]["hidden_dim"]),
            max_normalized_residual=float(
                config["wrist_wrench_residual_actor"]["max_normalized_residual"]
            ),
        )
        q1, q2, q1_target, q2_target = build_twin_q(
            hidden_dim=int(config["ack_residual_twin_q"]["hidden_dim"]), seed=seed + 1
        )
    residual_actor_optimizer = torch.optim.Adam(
        residual_actor.parameters(),
        lr=float(config["optimizer"]["residual_actor"]["lr"]),
    )
    critic_optimizer = torch.optim.Adam(
        (*q1.parameters(), *q2.parameters()),
        lr=float(config["optimizer"]["twin_q"]["lr"]),
    )
    runtime_state = {
        "checkpoint_kind": BOOTSTRAP_CHECKPOINT_KIND,
        "online_semantics_version": ONLINE_SEMANTICS_VERSION,
        "frozen_base_policy_checkpoint": str(frozen_base_policy_checkpoint),
        "residual_actor_critic_cycles": 0,
        "learner_state": "ack_replay_collection",
        "ack_critic_warmup_complete": False,
        "ack_critic_warmup_steps": 0,
        "active_residual_policy_revision": f"{task_id}-residual-policy-step-000000",
        "online_adaptation_id": f"{task_id}-ack-dispatch-residual-{time.time_ns()}",
        "counters": {
            "twin_q_optimizer_steps": 0,
            "residual_actor_optimizer_steps": 0,
            "residual_actor_update_attempts": 0,
            "residual_actor_updates_skipped_no_gradient": 0,
            "twin_q_target_update_steps": 0,
        },
        "replay": {
            "critic_td_valid_rows": 0,
            "actor_q_valid_rows": 0,
            "human_residual_valid_rows": 0,
            "loaded_episode_keys": [],
            "per_episode_critic_row_counts": {},
            "admission_cycle_budgets": {},
            "replay_generation": 0,
        },
    }
    return save_residual_actor_critic_checkpoint(
        checkpoint,
        residual_actor=residual_actor,
        residual_actor_target=residual_actor_target,
        q1=q1,
        q2=q2,
        q1_target=q1_target,
        q2_target=q2_target,
        residual_actor_optimizer=residual_actor_optimizer,
        critic_optimizer=critic_optimizer,
        runtime_state=runtime_state,
        config=config,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument(
        "--frozen-base-policy-checkpoint", type=Path, required=True
    )
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument(
        "--online-residual-config",
        type=Path,
        default=ROOT / "configs/forcerft/online_ack_residual_actor_critic.yaml",
    )
    args = parser.parse_args(argv)
    if args.checkpoint is None:
        args.checkpoint = (
            args.output_root
            / ONLINE_ADAPTATION_DIRECTORY_NAME
            / "bootstrap_checkpoints"
            / BOOTSTRAP_DIRECTORY_NAME
        )
    return args


def main() -> int:
    args = parse_args()
    result = build_online_residual_bootstrap(**vars(args))
    print(result.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
