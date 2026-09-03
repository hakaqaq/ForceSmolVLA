from __future__ import annotations

from pathlib import Path
import sys

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from train_forcerft_actor_critic import (  # noqa: E402
    _snapshot_actor_state,
    assert_offline_initialization_keeps_actor_frozen,
)


def test_offline_critic_initialization_keeps_parameters_and_buffers() -> None:
    actor = torch.nn.Sequential(
        torch.nn.Linear(3, 4),
        torch.nn.BatchNorm1d(4),
    )
    initial = _snapshot_actor_state(actor)
    optimizer = torch.optim.AdamW(actor.parameters(), lr=1.0e-5)

    for parameter in actor.parameters():
        parameter.requires_grad_(False)

    assert_offline_initialization_keeps_actor_frozen(actor, initial, optimizer)
    assert optimizer.state == {}
