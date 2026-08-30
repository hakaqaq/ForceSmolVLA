from __future__ import annotations

from pathlib import Path
import random
import sys

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from run_stage3_critic_warmup import (  # noqa: E402
    build_ack_macros,
    load_checkpoint_once,
    save_checkpoint,
)
from forcesmolvla.rft.stage3.update_credit import UpdateCreditLedger  # noqa: E402


def _row(decision: int, *, generation: int, terminated: bool = False) -> dict:
    return {
        "identity": {"decision_id": decision},
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
