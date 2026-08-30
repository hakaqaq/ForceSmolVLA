"""Canonical Actor batch construction for ForceRFT training and validation."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch


def build_actor_batch(
    policy: Any,
    samples: list[dict],
    device: torch.device,
    *,
    include_action: bool,
) -> dict:
    from forcesmolvla.configuration_forcesmolvla import CAMERA1, CAMERA2
    from lerobot.utils.constants import (
        ACTION,
        OBS_LANGUAGE_ATTENTION_MASK,
        OBS_LANGUAGE_TOKENS,
    )

    tokenizer = policy.model.vlm_with_expert.processor.tokenizer
    tokenizer.padding_side = "right"
    tokenizer.truncation_side = "right"
    encoded = tokenizer(
        [sample["task"] + "\n" for sample in samples],
        padding="max_length",
        truncation=True,
        max_length=48,
        return_tensors="pt",
    )
    result = {
        CAMERA1: torch.from_numpy(
            np.stack([sample["camera1"] for sample in samples])
        ).float().div_(255).to(device),
        CAMERA2: torch.from_numpy(
            np.stack([sample["camera2"] for sample in samples])
        ).float().div_(255).to(device),
        "observation.state": torch.from_numpy(
            np.stack([sample["state7"] for sample in samples])
        ).to(device),
        "observation.wrench": torch.from_numpy(
            np.stack([sample["wrench6"] for sample in samples])
        ).to(device),
        OBS_LANGUAGE_TOKENS: encoded["input_ids"].to(device),
        OBS_LANGUAGE_ATTENTION_MASK: encoded["attention_mask"].to(
            device=device, dtype=torch.bool
        ),
        "sample_identity": tuple(sample["sample_identity"] for sample in samples),
    }
    if include_action:
        result[ACTION] = torch.from_numpy(
            np.stack([sample["delta_action7"] for sample in samples])
        ).to(device)
        result["action_valid_mask"] = torch.from_numpy(
            np.stack([sample["action_valid_mask"] for sample in samples])
        ).to(device)
    return result
