from __future__ import annotations

from pathlib import Path
import json
import random
import sys

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from train_forcerft_critic_warmup import (  # noqa: E402
    build_ack_macros,
    load_formal_online_r,
    load_checkpoint_once,
    save_checkpoint,
)
from forcesmolvla.rft.online.sample_credit import UpdateCreditLedger  # noqa: E402


def _row(
    decision: int,
    *,
    generation: int,
    terminated: bool = False,
    episode_id: str | None = None,
) -> dict:
    return {
        "identity": {
            "decision_id": decision,
            **({"episode_id": episode_id} if episode_id is not None else {}),
        },
        "generation": {
            "policy_epoch": generation,
            "takeover_generation": generation,
            "reset_generation": 0,
        },
        "policy_lineage": {"selection": {"sequence": decision}},
        "outcome": {"terminated": terminated},
    }


def test_ack_macros_do_not_cross_override_or_takeover() -> None:
    rows = [
        _row(1, generation=0),
        _row(2, generation=0),
        _row(3, generation=0),
        _row(5, generation=1),
        _row(6, generation=1),
        _row(7, generation=1, terminated=True),
    ]

    macros = build_ack_macros(rows)

    assert [[item["identity"]["decision_id"] for item in macro] for macro in macros] == [
        [1, 2, 3],
        [5, 6, 7],
    ]
    assert macros[-1][-1]["outcome"]["terminated"] is True


def test_formal_replay_loads_multiple_admitted_episodes(tmp_path: Path) -> None:
    for episode_index in range(2):
        episode_id = f"session_{episode_index}/episode_000000"
        source = tmp_path / f"source_{episode_index}"
        admission = {
            "episode_id": episode_id,
            "source_episode": str(source),
            "policy_execution_smoke_bridge": "PASS",
            "source_episode_semantics": {
                "formal_replay": False,
                "real_online_r": False,
            },
            "accepted_unique_r_transition_count": 50,
        }
        path = tmp_path / "admissions" / f"episode_{episode_index}.json"
        path.parent.mkdir(exist_ok=True)
        path.write_text(json.dumps(admission), encoding="utf-8")
        for decision in range(50):
            row = _row(
                decision,
                generation=episode_index,
                terminated=decision == 49,
                episode_id=episode_id,
            )
            row["identity"]["transition_uid"] = f"{episode_index}-{decision}"
            row.update(
                {
                    "classification": "recorded_live_policy_execution_smoke",
                    "action_authority": {"executed_action_source": "policy"},
                    "eligibility": {
                        "formal_replay": True,
                        "formal_training_replay_eligible": True,
                        "real_online_r": True,
                        "replay_membership": "R_online",
                    },
                }
            )
            replay = tmp_path / "replay" / f"{episode_index}-{decision}.json"
            replay.parent.mkdir(exist_ok=True)
            replay.write_text(json.dumps({"payload": row}), encoding="utf-8")

    rows, macros, sources = load_formal_online_r(tmp_path)

    assert len(rows) == 100
    assert len(macros) == 96
    assert set(sources) == {
        "session_0/episode_000000",
        "session_1/episode_000000",
    }


def test_critic_only_checkpoint_round_trip(tmp_path: Path) -> None:
    modules = {
        "q1": torch.nn.Linear(3, 1),
        "q2": torch.nn.Linear(3, 1),
        "q1_target": torch.nn.Linear(3, 1),
        "q2_target": torch.nn.Linear(3, 1),
    }
    optimizer = torch.optim.Adam(
        tuple(modules["q1"].parameters()) + tuple(modules["q2"].parameters()),
        lr=3e-4,
    )
    (modules["q1"](torch.ones(1, 3)) + modules["q2"](torch.ones(1, 3))).sum().backward()
    optimizer.step()
    credits = UpdateCreditLedger(credits_per_transition=1, credits_per_joint_cycle=1)
    for uid in ("a", "b"):
        credits.mint_for_unique_online_transition(uid)
    credits.consume_joint_cycle()
    r_rng = random.Random(1)
    d_rng = random.Random(2)
    probe = torch.Generator(device="cpu").manual_seed(3)
    runtime = {
        "counters": {"critic_warmup_steps": 1},
        "sample_credit": credits.state_dict(),
        "sampler_state": {"r_rng": r_rng.getstate(), "d_rng": d_rng.getstate()},
        "rng_state": {"noise_generator_cpu_probe": probe.get_state()},
    }
    binding = {
        "binding_id": "approved_hybrid_cycle210_actor_g7a_r2_twin_q.v1",
        "actor_parent": {"absolute_path": "/parent/model.safetensors"},
    }
    checkpoint = tmp_path / "checkpoint"

    save_checkpoint(
        checkpoint,
        modules=modules,
        optimizer=optimizer,
        runtime_state=runtime,
        actor_parent=binding,
    )
    loaded = load_checkpoint_once(
        checkpoint,
        modules=modules,
        optimizer=optimizer,
        device=torch.device("cpu"),
    )

    assert loaded["counters"]["critic_warmup_steps"] == 1
    assert loaded["sample_credit"]["minted"] == 2
    assert loaded["sample_credit"]["consumed"] == 1
    metadata = __import__("json").loads((checkpoint / "metadata.json").read_text())
    assert metadata["actor_parent_binding"]["optimizer_created"] is False
