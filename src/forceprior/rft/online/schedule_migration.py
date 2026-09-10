"""Auditable one-way migration from the bounded legacy online schedule."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
from typing import Any, Mapping

import torch
import yaml

from forceprior.rft.online.replay_training import algorithm_hyperparameters
from forceprior.rft.online.residual_actor_critic_checkpoint import (
    CANDIDATE_CHECKPOINT_KIND,
    TRAINING_CHECKPOINT_KIND,
    residual_actor_critic_checkpoint_is_recoverable,
)
from forceprior.rft.online.transition_authority import ONLINE_SEMANTICS_VERSION


MIGRATION_VERSION = "forceprior-continuous-async-schedule-migration-v1"
LEGACY_KEYS = {
    "admitted_rows_per_cycle",
    "twin_q_updates_per_cycle",
    "residual_actor_updates_per_cycle",
    "max_cycles_per_admitted_episode",
    "residual_candidate_interval_actor_steps",
    "training_checkpoint_interval_cycles",
    "retained_training_checkpoint_count",
    "checkpoint_on_warmup_complete",
    "checkpoint_on_candidate_activation",
}
CONTINUOUS_KEYS = {
    "scheduling_mode",
    "twin_q_updates_per_cycle",
    "residual_actor_updates_per_cycle",
    "residual_candidate_interval_cycles",
    "training_checkpoint_interval_cycles",
    "retained_training_checkpoint_count",
    "checkpoint_on_warmup_complete",
    "checkpoint_on_candidate_activation",
}
UNCHANGED_SCHEDULE_KEYS = {
    "twin_q_updates_per_cycle",
    "residual_actor_updates_per_cycle",
    "retained_training_checkpoint_count",
    "checkpoint_on_warmup_complete",
}


class ScheduleMigrationError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ScheduleMigrationError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tree_hashes(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)): _sha256(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _config_diff(source: Mapping[str, Any], target: Mapping[str, Any]) -> dict[str, Any]:
    source_online = source["residual_actor_critic_training"]
    target_online = target["residual_actor_critic_training"]
    keys = sorted(set(source_online) | set(target_online))
    return {
        f"residual_actor_critic_training.{key}": {
            "source": source_online.get(key, "<absent>"),
            "target": target_online.get(key, "<absent>"),
        }
        for key in keys
        if source_online.get(key, "<absent>")
        != target_online.get(key, "<absent>")
    }


def require_schedule_only_migration(
    source_config: Mapping[str, Any], target_config: Mapping[str, Any]
) -> dict[str, Any]:
    source_algorithm = algorithm_hyperparameters(source_config)
    target_algorithm = algorithm_hyperparameters(target_config)
    source_online = source_algorithm.pop("residual_actor_critic_training")
    target_online = target_algorithm.pop("residual_actor_critic_training")
    _require(
        source_algorithm == target_algorithm,
        "FORCERFT_SCHEDULE_MIGRATION_NON_SCHEDULER_CONFIG_DRIFT",
    )
    _require(
        set(source_online) == LEGACY_KEYS
        and set(target_online) == CONTINUOUS_KEYS,
        "FORCERFT_SCHEDULE_MIGRATION_SCHEMA_INVALID",
    )
    _require(
        all(source_online[key] == target_online[key] for key in UNCHANGED_SCHEDULE_KEYS),
        "FORCERFT_SCHEDULE_MIGRATION_UNAPPROVED_SCHEDULER_DRIFT",
    )
    _require(
        target_online["scheduling_mode"] == "continuous_async"
        and int(target_online["residual_candidate_interval_cycles"]) == 100
        and int(target_online["training_checkpoint_interval_cycles"]) == 1000
        and target_online["checkpoint_on_candidate_activation"] is False,
        "FORCERFT_SCHEDULE_MIGRATION_TARGET_INVALID",
    )
    return _config_diff(source_config, target_config)


def schedule_migration_required(
    checkpoint_config: Mapping[str, Any], current_config: Mapping[str, Any]
) -> bool:
    try:
        require_schedule_only_migration(checkpoint_config, current_config)
    except (KeyError, ScheduleMigrationError, TypeError):
        return False
    return True


def _legacy_active_actor(
    source: Path, runtime: Mapping[str, Any]
) -> tuple[Path, int]:
    revision = str(runtime["active_residual_policy_revision"])
    try:
        actor_step = int(revision.rsplit("-", 1)[1])
    except ValueError as error:
        raise ScheduleMigrationError(
            "FORCERFT_SCHEDULE_MIGRATION_LEGACY_REVISION_INVALID"
        ) from error
    online_root = source.parent.parent
    if actor_step > 0:
        candidate = (
            online_root
            / "policy_candidates"
            / str(runtime["online_adaptation_id"])
            / f"residual_actor_step_{actor_step:06d}"
        )
        _require(
            (candidate / "residual_actor.pt").is_file()
            and (candidate / "candidate_state.pt").is_file(),
            "FORCERFT_SCHEDULE_MIGRATION_ACTIVE_ACTOR_MISSING",
        )
        return candidate / "residual_actor.pt", actor_step
    matches: list[Path] = []
    for state_path in online_root.glob("bootstrap_checkpoints/*/state/runtime_state.pt"):
        state = torch.load(state_path, map_location="cpu", weights_only=False)
        if (
            state.get("online_adaptation_id") == runtime.get("online_adaptation_id")
            and state.get("active_residual_policy_revision") == revision
        ):
            matches.append(state_path.parent.parent / "models/residual_actor.pt")
    _require(
        len(matches) == 1 and matches[0].is_file(),
        "FORCERFT_SCHEDULE_MIGRATION_ZERO_RESIDUAL_SOURCE_AMBIGUOUS",
    )
    return matches[0], 0


def migrate_schedule_checkpoint(
    source_checkpoint: Path,
    target_config_path: Path,
    destination: Path,
) -> Path:
    source = Path(source_checkpoint).resolve()
    target_config_path = Path(target_config_path).resolve()
    destination = Path(destination).resolve()
    _require(
        residual_actor_critic_checkpoint_is_recoverable(
            source, expected_kind=TRAINING_CHECKPOINT_KIND
        ),
        "FORCERFT_SCHEDULE_MIGRATION_SOURCE_INVALID",
    )
    _require(
        target_config_path.is_file() and not destination.exists(),
        "FORCERFT_SCHEDULE_MIGRATION_TARGET_EXISTS_OR_CONFIG_MISSING",
    )
    source_config = yaml.safe_load((source / "state/config.yaml").read_text())
    target_config = yaml.safe_load(target_config_path.read_text())
    differences = require_schedule_only_migration(source_config, target_config)
    runtime = torch.load(
        source / "state/runtime_state.pt", map_location="cpu", weights_only=False
    )
    _require(
        runtime.get("checkpoint_kind") == TRAINING_CHECKPOINT_KIND,
        "FORCERFT_SCHEDULE_MIGRATION_SOURCE_KIND_INVALID",
    )
    active_actor, active_actor_step = _legacy_active_actor(source, runtime)
    source_hashes = _tree_hashes(source)
    source_identity = hashlib.sha256(
        json.dumps(source_hashes, sort_keys=True).encode("utf-8")
    ).hexdigest()
    completed_cycle = int(runtime["residual_actor_critic_cycles"])
    generation = f"migration-{source_identity[:12]}-{time.time_ns()}"
    migrated_revision = (
        str(runtime["active_residual_policy_revision"]).split(
            "-residual-policy-", 1
        )[0]
        + f"-residual-policy-migrated-g{source_identity[:12]}"
    )
    temporary = destination.with_name(
        f".{destination.name}.writing-{os.getpid()}"
    )
    _require(not temporary.exists(), "FORCERFT_SCHEDULE_MIGRATION_TEMP_EXISTS")
    shutil.copytree(source, temporary)
    active_dependency = (
        destination.parent
        / "schedule_migration_dependencies"
        / source_identity
        / "active_actor"
    )
    try:
        if not active_dependency.exists():
            dependency_temporary = active_dependency.with_name(
                f".{active_dependency.name}.writing-{os.getpid()}"
            )
            dependency_temporary.mkdir(parents=True)
            shutil.copy2(
                active_actor, dependency_temporary / "residual_actor.pt"
            )
            torch.save(
                {
                    "checkpoint_kind": CANDIDATE_CHECKPOINT_KIND,
                    "online_semantics_version": ONLINE_SEMANTICS_VERSION,
                    "migration_source_sha256": source_identity,
                },
                dependency_temporary / "candidate_state.pt",
            )
            active_dependency.parent.mkdir(parents=True, exist_ok=True)
            os.replace(dependency_temporary, active_dependency)
        _require(
            (active_dependency / "residual_actor.pt").is_file()
            and (active_dependency / "candidate_state.pt").is_file(),
            "FORCERFT_SCHEDULE_MIGRATION_ACTIVE_DEPENDENCY_INVALID",
        )
        replay = dict(runtime["replay"])
        retired_budgets = dict(replay.pop("admission_cycle_budgets", {}))
        runtime["replay"] = replay
        partial_q_updates = (
            int(runtime["counters"]["twin_q_optimizer_steps"])
            - int(runtime.get("ack_critic_warmup_steps", 0))
            - completed_cycle * int(target_config[
                "residual_actor_critic_training"
            ]["twin_q_updates_per_cycle"])
        )
        _require(
            0 <= partial_q_updates <= int(target_config[
                "residual_actor_critic_training"
            ]["twin_q_updates_per_cycle"]),
            "FORCERFT_SCHEDULE_MIGRATION_PARTIAL_CYCLE_INVALID",
        )
        runtime["partial_cycle_q_updates"] = partial_q_updates
        runtime["active_residual_policy_revision"] = migrated_revision
        runtime["scheduling"] = {
            "mode": "continuous_async",
            "last_publish_attempt_cycle": 0,
            "last_published_cycle": None,
            "publication_event_count": 0,
            "last_periodic_checkpoint_cycle": 0,
            "periodic_checkpoint_event_count": 0,
            "last_saved_checkpoint_cycle": completed_cycle,
            "events_not_before_cycle": completed_cycle,
            "active_publication_cycle": None,
            "active_actor_optimizer_step": active_actor_step,
            "active_actor_checkpoint": str(active_dependency),
            "active_policy_epoch": 0,
            "active_policy_epoch_status": "rebased_after_legacy_migration",
            "pending_publication": None,
            "candidate_export_generation": generation,
            "retired_admission_cycle_budgets": retired_budgets,
        }
        runtime["scheduling_migration"] = {
            "version": MIGRATION_VERSION,
            "source_checkpoint": str(source),
            "source_checkpoint_sha256": source_identity,
            "migrated_at_cycle": completed_cycle,
            "legacy_active_revision": str(
                torch.load(
                    source / "state/runtime_state.pt",
                    map_location="cpu",
                    weights_only=False,
                )["active_residual_policy_revision"]
            ),
            "config_differences": differences,
        }
        torch.save(runtime, temporary / "state/runtime_state.pt")
        (temporary / "state/config.yaml").write_text(
            yaml.safe_dump(deepcopy(target_config), sort_keys=False), encoding="utf-8"
        )
        (temporary / "state/schedule_migration.json").write_text(
            json.dumps(runtime["scheduling_migration"], indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        os.replace(temporary, destination)
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    _require(
        residual_actor_critic_checkpoint_is_recoverable(
            destination, expected_kind=TRAINING_CHECKPOINT_KIND
        ),
        "FORCERFT_SCHEDULE_MIGRATION_OUTPUT_INVALID",
    )
    return destination
