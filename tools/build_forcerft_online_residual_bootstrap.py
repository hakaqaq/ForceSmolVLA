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

from forceprior.rft.critic import (  # noqa: E402
    CRITIC_ACTION_REPRESENTATION,
    CRITIC_CANDIDATE_FEASIBILITY,
    CRITIC_INPUT_SPEC,
    CRITIC_TD_SOURCE_MODE,
    build_twin_q,
    require_critic_input_config,
)
from forceprior.rft.online.residual_actor_critic_runtime import (  # noqa: E402
    ONLINE_ADAPTATION_DIRECTORY_NAME,
)
from forceprior.rft.online.residual_actor_critic_checkpoint import (  # noqa: E402
    BOOTSTRAP_CHECKPOINT_KIND,
    save_residual_actor_critic_checkpoint,
)
from forceprior.rft.online.transition_authority import (  # noqa: E402
    ONLINE_SEMANTICS_VERSION,
)
from forceprior.rft.residual_actor import (  # noqa: E402
    make_residual_actor_pair,
    resolve_residual_cap6,
)


BOOTSTRAP_DIRECTORY_NAME = "base_policy_zero_residual_filter_leash_random_twin_q"


def _load_base_actor(checkpoint: Path) -> torch.nn.Module:
    from forceprior.modeling_forceprior import ForcePriorPolicy

    return ForcePriorPolicy.from_pretrained(
        checkpoint,
        local_files_only=True,
        force_download=False,
        strict=True,
        artifact_use="development",
    )


def _normalizer_parameters_match(
    *, dataset_root: Path, frozen_base_policy_checkpoint: Path
) -> bool:
    from forceprior.training_data import (
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
    if checkpoint.exists():
        raise RuntimeError("FORCERFT_BOOTSTRAP_DESTINATION_EXISTS")
    frozen_base_policy_checkpoint = Path(frozen_base_policy_checkpoint).resolve()
    if not _normalizer_parameters_match(
        dataset_root=Path(dataset_root).resolve(),
        frozen_base_policy_checkpoint=frozen_base_policy_checkpoint,
    ):
        raise RuntimeError("FORCERFT_BOOTSTRAP_NORMALIZER_MISMATCH")
    config = yaml.safe_load(
        Path(online_residual_config).read_text(encoding="utf-8")
    )
    from forceprior.training_data import load_normalizer_manifest

    normalizer = load_normalizer_manifest(
        Path(dataset_root).resolve() / "normalizer_manifest.json"
    )
    residual_cap6 = resolve_residual_cap6(
        normalizer, config["wrist_wrench_residual_actor"]
    )
    if int(config["batching"]["command_macro_slots"]) != 3:
        raise ValueError("FORCERFT_COMMAND_MACRO_SLOTS_INVALID")
    require_critic_input_config(config["ack_residual_twin_q"])
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
            residual_cap6=residual_cap6,
            residual_bound_mode=str(
                config["wrist_wrench_residual_actor"]["residual_bound_mode"]
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
        "critic_input_spec": CRITIC_INPUT_SPEC,
        "critic_action_representation": CRITIC_ACTION_REPRESENTATION,
        "critic_td_source_mode": CRITIC_TD_SOURCE_MODE,
        "critic_candidate_feasibility": CRITIC_CANDIDATE_FEASIBILITY,
        "residual_bound_mode": config["wrist_wrench_residual_actor"][
            "residual_bound_mode"
        ],
        "frozen_base_policy_checkpoint": str(frozen_base_policy_checkpoint),
        "residual_actor_critic_cycles": 0,
        "partial_cycle_q_updates": 0,
        "learner_state": "ack_replay_collection",
        "ack_critic_warmup_complete": False,
        "ack_critic_warmup_steps": 0,
        "active_residual_policy_revision": f"{task_id}-residual-policy-step-000000",
        "online_adaptation_id": f"{task_id}-ack-filter-leash-residual-{time.time_ns()}",
        "scheduling": {
            "mode": "continuous_async",
            "last_publish_attempt_cycle": 0,
            "last_published_cycle": None,
            "publication_event_count": 0,
            "last_periodic_checkpoint_cycle": 0,
            "periodic_checkpoint_event_count": 0,
            "last_saved_checkpoint_cycle": None,
            "active_publication_cycle": None,
            "active_actor_optimizer_step": 0,
            "active_actor_checkpoint": str(
                checkpoint.resolve() / "models/residual_actor.pt"
            ),
            "active_policy_epoch": 0,
            "active_policy_epoch_status": "bootstrap",
            "pending_publication": None,
        },
        "counters": {
            "twin_q_optimizer_steps": 0,
            "residual_actor_optimizer_steps": 0,
            "residual_actor_update_attempts": 0,
            "residual_actor_updates_skipped_no_gradient": 0,
            "twin_q_target_update_steps": 0,
            "critic_sample_draws": 0,
            "policy_sample_draws": 0,
            "human_sample_draws": 0,
        },
        "replay": {
            "recorded_transition_rows": 0,
            "critic_td_valid_rows": 0,
            "actor_q_valid_rows": 0,
            "human_residual_valid_rows": 0,
            "nonzero_policy_proposal_rows": 0,
            "nonzero_accepted_residual_rows": 0,
            "loaded_episode_keys": [],
            "per_episode_critic_row_counts": {},
            "replay_generation": 0,
            "training_credit_ledger": {
                "schema": "forceprior-td-cycle-credit-ledger-v1",
                "new_td_rows_per_cycle": int(
                    config["residual_actor_critic_training"][
                        "new_td_rows_per_cycle"
                    ]
                ),
                "admissions": {},
                "in_flight_cycle": None,
            },
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
    config = yaml.safe_load(args.online_residual_config.read_text(encoding="utf-8"))
    from forceprior.training_data import load_normalizer_manifest

    normalizer = load_normalizer_manifest(
        args.dataset_root.resolve() / "normalizer_manifest.json"
    )
    cap6 = resolve_residual_cap6(
        normalizer, config["wrist_wrench_residual_actor"]
    ).numpy()
    physical = cap6 * normalizer.delta_action7.std[:6]
    print(
        "resolved residual cap6="
        f"{cap6.tolist()} translation_mm={(physical[:3] * 1000.0).tolist()} "
        f"rpy_deg={(physical[3:] * 180.0 / 3.141592653589793).tolist()}"
    )
    print(result.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
