"""Compact checkpoints for the online residual Actor and Twin-Q."""

from __future__ import annotations

import os
from pathlib import Path
import shutil
from typing import Any, Mapping

import torch
import yaml


class OnlineCheckpointSchemaError(ValueError):
    pass


RESIDUAL_CHECKPOINT_FILES = (
    "models/residual_actor.pt",
    "models/residual_actor_target.pt",
    "models/q1.pt",
    "models/q2.pt",
    "models/q1_target.pt",
    "models/q2_target.pt",
    "optimizers/residual_actor_optimizer.pt",
    "optimizers/critic_optimizer.pt",
    "state/runtime_state.pt",
    "state/config.yaml",
)


def residual_checkpoint_is_recoverable(checkpoint: Path) -> bool:
    checkpoint = Path(checkpoint)
    if not checkpoint.is_dir() or not all(
        (checkpoint / relative).is_file()
        for relative in RESIDUAL_CHECKPOINT_FILES
    ):
        return False
    try:
        state = torch.load(
            checkpoint / "state/runtime_state.pt",
            map_location="cpu",
            weights_only=False,
        )
        counters = state["counters"]
        replay = state["replay"]
        return bool(
            state["phase"] in {"collecting", "critic_burnin", "joint"}
            and isinstance(state["critic_burnin_complete"], bool)
            and int(state.get("critic_burnin_updates", 0)) >= 0
            and int(state["online_joint_cycles"]) >= 0
            and isinstance(state["base_actor_checkpoint"], str)
            and bool(state["base_actor_checkpoint"])
            and isinstance(state["active_residual_revision"], str)
            and bool(state["active_residual_revision"])
            and all(
                int(counters[name]) >= 0
                for name in (
                    "critic_optimizer_steps",
                    "actor_optimizer_steps",
                    "target_polyak_steps",
                )
            )
            and all(
                int(replay[name]) >= 0
                for name in (
                    "critic_td_valid_rows",
                    "actor_q_valid_rows",
                    "human_residual_valid_rows",
                )
            )
        )
    except (KeyError, OSError, RuntimeError, TypeError, ValueError):
        return False


def save_residual_checkpoint(
    checkpoint: Path,
    *,
    residual_actor: torch.nn.Module,
    residual_actor_target: torch.nn.Module,
    q1: torch.nn.Module,
    q2: torch.nn.Module,
    q1_target: torch.nn.Module,
    q2_target: torch.nn.Module,
    residual_actor_optimizer: torch.optim.Optimizer,
    critic_optimizer: torch.optim.Optimizer,
    runtime_state: Mapping[str, Any],
    config: Mapping[str, Any],
) -> Path:
    """Atomically save only the final residual training state."""

    checkpoint = Path(checkpoint).resolve()
    if checkpoint.exists() and not residual_checkpoint_is_recoverable(checkpoint):
        raise OnlineCheckpointSchemaError("FORCERFT_CHECKPOINT_DESTINATION_EXISTS")
    temporary = checkpoint.with_name(f".{checkpoint.name}.writing-{os.getpid()}")
    if temporary.exists():
        shutil.rmtree(temporary)
    try:
        for directory in ("models", "optimizers", "state"):
            (temporary / directory).mkdir(parents=True, exist_ok=True)
        modules = {
            "residual_actor": residual_actor,
            "residual_actor_target": residual_actor_target,
            "q1": q1,
            "q2": q2,
            "q1_target": q1_target,
            "q2_target": q2_target,
        }
        for name, module in modules.items():
            torch.save(module.state_dict(), temporary / f"models/{name}.pt")
        torch.save(
            residual_actor_optimizer.state_dict(),
            temporary / "optimizers/residual_actor_optimizer.pt",
        )
        torch.save(
            critic_optimizer.state_dict(),
            temporary / "optimizers/critic_optimizer.pt",
        )
        torch.save(dict(runtime_state), temporary / "state/runtime_state.pt")
        (temporary / "state/config.yaml").write_text(
            yaml.safe_dump(dict(config), sort_keys=False), encoding="utf-8"
        )
        if checkpoint.exists():
            replaced = checkpoint.with_name(
                f".{checkpoint.name}.replaced-{os.getpid()}"
            )
            if replaced.exists():
                shutil.rmtree(replaced)
            os.replace(checkpoint, replaced)
            try:
                os.replace(temporary, checkpoint)
            except BaseException:
                os.replace(replaced, checkpoint)
                raise
            shutil.rmtree(replaced)
        else:
            os.replace(temporary, checkpoint)
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    if not residual_checkpoint_is_recoverable(checkpoint):
        raise OnlineCheckpointSchemaError("FORCERFT_CHECKPOINT_WRITE_INCOMPLETE")
    return checkpoint


def load_residual_checkpoint(
    checkpoint: Path,
    *,
    residual_actor: torch.nn.Module,
    residual_actor_target: torch.nn.Module,
    q1: torch.nn.Module,
    q2: torch.nn.Module,
    q1_target: torch.nn.Module,
    q2_target: torch.nn.Module,
    residual_actor_optimizer: torch.optim.Optimizer | None = None,
    critic_optimizer: torch.optim.Optimizer | None = None,
    device: torch.device | str = "cpu",
) -> tuple[dict[str, Any], dict[str, Any]]:
    if not residual_checkpoint_is_recoverable(checkpoint):
        raise OnlineCheckpointSchemaError("FORCERFT_CHECKPOINT_INCOMPLETE")
    checkpoint = Path(checkpoint)
    modules = {
        "residual_actor": residual_actor,
        "residual_actor_target": residual_actor_target,
        "q1": q1,
        "q2": q2,
        "q1_target": q1_target,
        "q2_target": q2_target,
    }
    for name, module in modules.items():
        module.load_state_dict(
            torch.load(
                checkpoint / f"models/{name}.pt",
                map_location=device,
                weights_only=True,
            ),
            strict=True,
        )
    if residual_actor_optimizer is not None:
        residual_actor_optimizer.load_state_dict(
            torch.load(
                checkpoint / "optimizers/residual_actor_optimizer.pt",
                map_location=device,
                weights_only=False,
            )
        )
    if critic_optimizer is not None:
        critic_optimizer.load_state_dict(
            torch.load(
                checkpoint / "optimizers/critic_optimizer.pt",
                map_location=device,
                weights_only=False,
            )
        )
    runtime = torch.load(
        checkpoint / "state/runtime_state.pt",
        map_location="cpu",
        weights_only=False,
    )
    config = yaml.safe_load(
        (checkpoint / "state/config.yaml").read_text(encoding="utf-8")
    )
    return dict(runtime), dict(config)
