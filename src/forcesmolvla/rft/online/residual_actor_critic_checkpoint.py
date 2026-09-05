"""Compact checkpoints for ACK-residual Actor-Critic training."""

from __future__ import annotations

import os
from pathlib import Path
import shutil
from typing import Any, Mapping

import torch
import yaml

from forcesmolvla.rft.online.transition_authority import ONLINE_SEMANTICS_VERSION


class OnlineCheckpointSchemaError(ValueError):
    pass


RESIDUAL_ACTOR_CRITIC_CHECKPOINT_FILES = (
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

BOOTSTRAP_CHECKPOINT_KIND = "online_residual_bootstrap"
TRAINING_CHECKPOINT_KIND = "residual_actor_critic_training"
CANDIDATE_CHECKPOINT_KIND = "residual_actor_candidate"
CHECKPOINT_KINDS = {BOOTSTRAP_CHECKPOINT_KIND, TRAINING_CHECKPOINT_KIND}


def _nonnegative_int(value: Any) -> bool:
    return not isinstance(value, bool) and isinstance(value, int) and value >= 0


def residual_actor_critic_checkpoint_is_recoverable(
    checkpoint: Path,
    *,
    expected_kind: str | None = None,
) -> bool:
    checkpoint = Path(checkpoint)
    if not checkpoint.is_dir() or not all(
        (checkpoint / relative).is_file()
        for relative in RESIDUAL_ACTOR_CRITIC_CHECKPOINT_FILES
    ):
        return False
    try:
        state = torch.load(
            checkpoint / "state/runtime_state.pt",
            map_location="cpu",
            weights_only=False,
        )
        counters = state["counters"]
        applied_actor_steps = counters["residual_actor_optimizer_steps"]
        actor_update_attempts = counters["residual_actor_update_attempts"]
        skipped_actor_updates = counters[
            "residual_actor_updates_skipped_no_gradient"
        ]
        replay = state["replay"]
        loaded_episode_keys = replay.get("loaded_episode_keys", [])
        per_episode_counts = replay.get("per_episode_critic_row_counts", {})
        admission_cycle_budgets = replay.get("admission_cycle_budgets", {})
        return bool(
            state.get("checkpoint_kind") in CHECKPOINT_KINDS
            and (expected_kind is None or state["checkpoint_kind"] == expected_kind)
            and state.get("online_semantics_version") == ONLINE_SEMANTICS_VERSION
            and state["learner_state"]
            in {
                "ack_replay_collection",
                "ack_critic_warmup",
                "residual_actor_critic_training",
            }
            and isinstance(state["ack_critic_warmup_complete"], bool)
            and _nonnegative_int(state.get("ack_critic_warmup_steps"))
            and _nonnegative_int(state["residual_actor_critic_cycles"])
            and isinstance(state["frozen_base_policy_checkpoint"], str)
            and bool(state["frozen_base_policy_checkpoint"])
            and isinstance(state["active_residual_policy_revision"], str)
            and bool(state["active_residual_policy_revision"])
            and isinstance(state["online_adaptation_id"], str)
            and bool(state["online_adaptation_id"])
            and all(
                _nonnegative_int(counters[name])
                for name in (
                    "twin_q_optimizer_steps",
                    "residual_actor_optimizer_steps",
                    "residual_actor_update_attempts",
                    "residual_actor_updates_skipped_no_gradient",
                    "twin_q_target_update_steps",
                )
            )
            and actor_update_attempts
            == applied_actor_steps + skipped_actor_updates
            and all(
                _nonnegative_int(replay[name])
                for name in (
                    "critic_td_valid_rows",
                    "actor_q_valid_rows",
                    "human_residual_valid_rows",
                )
            )
            and isinstance(loaded_episode_keys, list)
            and len(set(loaded_episode_keys)) == len(loaded_episode_keys)
            and all(
                isinstance(value, str) and value
                for value in loaded_episode_keys
            )
            and isinstance(per_episode_counts, dict)
            and all(
                isinstance(key, str)
                and key
                and _nonnegative_int(value)
                for key, value in per_episode_counts.items()
            )
            and isinstance(admission_cycle_budgets, dict)
            and all(
                isinstance(key, str)
                and key
                and _nonnegative_int(value)
                for key, value in admission_cycle_budgets.items()
            )
            and _nonnegative_int(replay.get("replay_generation", 0))
        )
    except (KeyError, OSError, RuntimeError, TypeError, ValueError):
        return False


def save_residual_actor_critic_checkpoint(
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
    expected_kind = runtime_state.get("checkpoint_kind")
    if expected_kind not in CHECKPOINT_KINDS:
        raise OnlineCheckpointSchemaError("FORCERFT_CHECKPOINT_KIND_INVALID")
    if runtime_state.get("online_semantics_version") != ONLINE_SEMANTICS_VERSION:
        raise OnlineCheckpointSchemaError("FORCERFT_CHECKPOINT_SEMANTICS_INVALID")
    if checkpoint.exists() and not residual_actor_critic_checkpoint_is_recoverable(
        checkpoint, expected_kind=expected_kind
    ):
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
    if not residual_actor_critic_checkpoint_is_recoverable(
        checkpoint, expected_kind=str(expected_kind)
    ):
        raise OnlineCheckpointSchemaError("FORCERFT_CHECKPOINT_WRITE_INCOMPLETE")
    return checkpoint


def load_residual_actor_critic_checkpoint(
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
    expected_kind: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if not residual_actor_critic_checkpoint_is_recoverable(
        checkpoint, expected_kind=expected_kind
    ):
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
