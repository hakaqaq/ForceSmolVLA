#!/usr/bin/env python3
"""Offline-only Cartesian7D chunk inference/reset/cache smoke on frozen assets."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import socket
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path(__file__).parents[1])
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"):
        if os.environ.get(name) != "1":
            raise RuntimeError(f"{name}=1 required")
    socket.socket.connect = lambda self, address: (_ for _ in ()).throw(
        RuntimeError(f"NETWORK_ACCESS_FORBIDDEN: {address}")
    )

    import numpy as np
    import torch

    from forcesmolvla.checkpoint import load_offline_base_policy
    from forcesmolvla.configuration_forcesmolvla import CAMERA1, CAMERA2
    from forcesmolvla.context import ChunkContext
    from forcesmolvla.normalizer import CartesianNormalizerBundle, FrozenFeatureNormalizer
    from forcesmolvla.rules import load_and_validate_rulespec
    from forcesmolvla.shadow import sha256_file
    from forcesmolvla.training_data import RuntimeArtifactBundle
    from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA_NOT_AVAILABLE")
    root = args.project_root.resolve()
    with contextlib.redirect_stdout(sys.stderr):
        policy, load_report = load_offline_base_policy(
            root / "assets" / "base_checkpoint",
            root / "assets" / "smolvlm_constructor",
            device=args.device,
        )
    device = torch.device(args.device)
    digest = "a" * 64
    feature = lambda name, mean, std: FrozenFeatureNormalizer(  # noqa: E731
        name, np.asarray(mean), np.asarray(std), ("synthetic-e0",)
    )
    policy.bind_runtime_artifacts(
        RuntimeArtifactBundle(
            normalizer=CartesianNormalizerBundle(
                state7=feature("state7", np.zeros(7), np.ones(7)),
                wrench6=feature("wrench6", np.zeros(6), np.ones(6)),
                delta_action7=feature(
                    "delta_action7",
                    [0, 0, 0, 0, 0, 0, 0.05],
                    [0.001] * 7,
                ),
                split_sha256=digest,
                calibration_bundle_sha256=digest,
                wrench_geometry_spec_sha256=digest,
            ),
            normalizer_manifest_sha256=digest,
            calibration_bundle_sha256=digest,
            wrench_geometry_spec_sha256=digest,
            split_sha256=digest,
            action_delta_spec_sha256=digest,
            action_delta_source_sha256=sha256_file(
                root / "src/forcesmolvla/action_delta.py"
            ),
        )
    )
    action_rules_path = root / "tests/fixtures/shadow_safety_thresholds.test_only.yaml"
    policy.bind_action_safety_rules(
        load_and_validate_rulespec(
            action_rules_path, root / "schemas/rulespec.schema.json", formal=False
        ),
        rules_sha256=sha256_file(action_rules_path),
    )
    batch = {
        CAMERA1: torch.zeros(2, 3, 480, 640, device=device),
        CAMERA2: torch.ones(2, 3, 480, 640, device=device) * 0.5,
        "observation.state": torch.tensor(
            [[0.5, -0.1, 0.2, 0.1, -0.2, 0.3, 0.05],
             [0.4, 0.2, 0.1, -0.2, 0.1, -0.3, 0.04]],
            device=device,
        ),
        OBS_LANGUAGE_TOKENS: torch.arange(48, device=device).view(1, 48).expand(2, -1),
        OBS_LANGUAGE_ATTENTION_MASK: torch.tensor(
            [[True] * 48, [True] * 17 + [False] * 31], dtype=torch.bool, device=device
        ),
    }
    batch["raw_state_snapshot"] = batch["observation.state"].detach().clone()
    valid = torch.tensor(
        [[True] * 50, [True] * 47 + [False] * 3], dtype=torch.bool, device=device
    )

    def context(generation: int) -> ChunkContext:
        return ChunkContext(
            policy_generation=generation,
            raw_state_snapshot=batch["observation.state"].detach().cpu(),
            t_ref_ns=torch.tensor([1, 2], dtype=torch.int64),
            tau0_ns=torch.tensor([3, 4], dtype=torch.int64),
            clock_domain_id=("controller", "controller"),
            episode_id=("synthetic-e0", "synthetic-e1"),
            session_id=("synthetic", "synthetic"),
            sample_id=("sample0", "sample1"),
            chunk_id=("chunk0", "chunk1"),
            action_valid_mask=valid.cpu(),
            suffix_valid_mask=valid.cpu().clone(),
            calibration_bundle_hash=(digest, digest),
            wrench_geometry_spec_hash=(digest, digest),
            normalizer_hash=(digest, digest),
            calibration_mapping_hash_or_none=(None, None),
            wrench_geometry_valid=torch.ones(2, dtype=torch.bool),
            runtime_artifact_compatible=torch.ones(2, dtype=torch.bool),
            selected_provenance=({"fixture": 0}, {"fixture": 1}),
        )

    policy.eval()
    with torch.inference_mode(), torch.autocast(
        device_type=args.device, dtype=torch.bfloat16
    ):
        first = policy.predict_action_chunk(batch, context(0), noise=4107)
    try:
        policy.predict_action_chunk(batch, context(0), noise=4107)
    except RuntimeError as error:
        consumed_error = str(error)
    else:
        raise AssertionError("consumed context was accepted")
    stale = context(0)
    policy.reset()
    try:
        policy.predict_action_chunk(batch, stale, noise=4107)
    except RuntimeError as error:
        reset_error = str(error)
    else:
        raise AssertionError("reset-invalidated context was accepted")
    with torch.inference_mode(), torch.autocast(
        device_type=args.device, dtype=torch.bfloat16
    ):
        second = policy.predict_action_chunk(batch, context(1), noise=4107)
    try:
        policy.select_action(batch)
    except RuntimeError as error:
        select_action_error = str(error)
    else:
        raise AssertionError("select_action was not rejected")
    first_cpu = first.float().cpu()
    result = {
        "status": "pass",
        "artifact_status": "development_only",
        "device": args.device,
        "output_shape": list(first.shape),
        "deterministic_reset_replay_max_abs": float((first - second).abs().max().cpu()),
        "invalid_tail_max_abs": float(first_cpu[1, 47:].abs().max()),
        "output_float32_sha256": hashlib.sha256(
            np.asarray(first_cpu, dtype="<f4").tobytes()
        ).hexdigest(),
        "consumed_context_error": consumed_error,
        "reset_context_error": reset_error,
        "select_action_error": select_action_error,
        "base_loaded_tensors": load_report.loaded_tensor_count,
        "missing_keys": list(load_report.missing_keys),
        "unexpected_keys": list(load_report.unexpected_keys),
        "robot_actions_sent": 0,
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
