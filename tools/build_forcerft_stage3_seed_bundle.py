#!/usr/bin/env python3
"""Build the final Stage-3 seed: frozen base path, zero residual, random Twin-Q."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from forcesmolvla.rft.critic import build_twin_q  # noqa: E402
from forcesmolvla.rft.online.learner_checkpoint import (  # noqa: E402
    save_residual_checkpoint,
)
from forcesmolvla.rft.residual_actor import make_residual_actor_pair  # noqa: E402


SEED_DIRECTORY_NAME = "stage3_base_actor_residual_q_cycle_000000"


def _load_base_actor(checkpoint: Path) -> torch.nn.Module:
    from forcesmolvla.modeling_forcesmolvla import ForceSmolVLAPolicy

    return ForceSmolVLAPolicy.from_pretrained(
        checkpoint,
        local_files_only=True,
        force_download=False,
        strict=True,
        artifact_use="development",
    )


def build_stage3_seed_bundle(
    *,
    task_id: str,
    output_root: Path,
    dataset_root: Path,
    base_actor_checkpoint: Path,
    checkpoint: Path,
    common_online_config: Path,
) -> Path:
    del output_root, dataset_root  # retained CLI path bindings; no demo replay is read.
    base_actor_checkpoint = Path(base_actor_checkpoint).resolve()
    config = yaml.safe_load(
        Path(common_online_config).read_text(encoding="utf-8")
    )
    base_actor = _load_base_actor(base_actor_checkpoint).to("cpu")
    base_actor.eval().requires_grad_(False)
    if any(parameter.requires_grad for parameter in base_actor.parameters()):
        raise RuntimeError("FORCERFT_BASE_ACTOR_NOT_FROZEN")

    seed = int(config["environment"]["seed"])
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        residual_actor, residual_actor_target = make_residual_actor_pair(
            hidden_dim=int(config["residual_actor"]["hidden_dim"]),
            max_normalized_residual=float(
                config["residual_actor"]["max_normalized_residual"]
            ),
        )
        q1, q2, q1_target, q2_target = build_twin_q(
            hidden_dim=int(config["critic"]["hidden_dim"]), seed=seed + 1
        )
    residual_actor_optimizer = torch.optim.Adam(
        residual_actor.parameters(), lr=float(config["optimizer"]["actor"]["lr"])
    )
    critic_optimizer = torch.optim.Adam(
        (*q1.parameters(), *q2.parameters()),
        lr=float(config["optimizer"]["critic"]["lr"]),
    )
    runtime_state = {
        "base_actor_checkpoint": str(base_actor_checkpoint),
        "online_joint_cycles": 0,
        "phase": "collecting",
        "critic_burnin_complete": False,
        "critic_burnin_updates": 0,
        "active_residual_revision": f"{task_id}-residual-step-000000",
        "counters": {
            "critic_optimizer_steps": 0,
            "actor_optimizer_steps": 0,
            "target_polyak_steps": 0,
        },
        "replay": {
            "critic_td_valid_rows": 0,
            "actor_q_valid_rows": 0,
            "human_residual_valid_rows": 0,
        },
    }
    return save_residual_checkpoint(
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
    parser.add_argument("--base-actor-checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument(
        "--common-online-config",
        type=Path,
        default=ROOT / "configs/forcerft/actor_critic_common.yaml",
    )
    args = parser.parse_args(argv)
    if args.checkpoint is None:
        args.checkpoint = (
            args.output_root / "stage3_seed/checkpoints" / SEED_DIRECTORY_NAME
        )
    return args


def main() -> int:
    args = parse_args()
    result = build_stage3_seed_bundle(**vars(args))
    print(result.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
