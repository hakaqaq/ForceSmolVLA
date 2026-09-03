from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from train_forcerft_actor_critic import build_actor_training_batch  # noqa: E402


def _row(source: str, valid: bool | None) -> dict:
    sample = {
        "camera1": np.zeros((3, 2, 2), dtype=np.uint8),
        "camera2": np.zeros((3, 2, 2), dtype=np.uint8),
        "state7": np.zeros(7, dtype=np.float32),
        "wrench6": np.zeros(6, dtype=np.float32),
        "task": "task",
        "sample_identity": source,
        "delta_action7": np.zeros((50, 7), dtype=np.float32),
        "action_valid_mask": np.ones(50, dtype=np.bool_),
    }
    row = {
        "current": sample,
        "expert": source == "offline_demonstration",
        "expert_feature_mask": np.zeros((50, 7), dtype=np.bool_),
        "action_source": source,
        "behavior_action": np.zeros((3, 7), dtype=np.float32),
        "behavior_mask": np.ones(3, dtype=np.bool_),
        "terminated": False,
        "truncated": False,
        "td_eligible": True,
        "fm_eligible": source == "offline_demonstration",
        "identity": source,
    }
    if valid is not None:
        row["actor_q_valid"] = valid
    return row


def test_batch_uses_authoritative_mask_and_offline_defaults_false(monkeypatch) -> None:
    monkeypatch.setattr(
        "forcesmolvla.rft.batch.build_actor_batch",
        lambda *_args, **_kwargs: {"action_valid_mask": torch.ones(2, 50)},
    )
    batch = build_actor_training_batch(
        [_row("policy", True), _row("offline_demonstration", None)],
        object(),
        torch.zeros(256),
        torch.device("cpu"),
    )
    assert batch["actor_q_valid"].tolist() == [True, False]
