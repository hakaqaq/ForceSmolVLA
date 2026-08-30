"""Shared fixed-B=2 P8 checkpoint parity computation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from forcesmolvla.action_delta import ActionDeltaProcessor
from forcesmolvla.context import ChunkContext
from lerobot.policies.smolvla.modeling_smolvla import pad_vector
from lerobot.utils.constants import ACTION

from forcesmolvla.training_runtime import build_training_batch as _make_batch


def canonical_sha256(value) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def tensor_record(tensor: torch.Tensor) -> dict:
    value = tensor.detach().cpu().contiguous()
    byte_view = value.view(torch.uint8)
    digest = hashlib.sha256(byte_view.numpy().tobytes()).hexdigest()
    return {"dtype": str(value.dtype), "shape": list(value.shape), "sha256": digest}


def cache_record(cache: dict) -> dict:
    entries = []
    digest = hashlib.sha256()
    for layer in sorted(cache):
        for name in ("key_states", "value_states"):
            record = tensor_record(cache[layer][name])
            entry = {"layer": int(layer), "name": name, **record}
            entries.append(entry)
            digest.update(json.dumps(entry, sort_keys=True, separators=(",", ":")).encode())
    return {"entries": entries, "sha256": digest.hexdigest()}


def chunk_context_from_fixture(fixture: dict, *, policy_generation: int) -> ChunkContext:
    payload = fixture["chunk_context"]
    if payload["policy_generation"] != policy_generation:
        raise RuntimeError("P8_FIXED_CHUNK_CONTEXT_GENERATION_MISMATCH")
    return ChunkContext(
        policy_generation=policy_generation,
        raw_state_snapshot=torch.tensor(payload["raw_state_snapshot"], dtype=torch.float32),
        t_ref_ns=torch.tensor(payload["t_ref_ns"], dtype=torch.int64),
        tau0_ns=torch.tensor(payload["tau0_ns"], dtype=torch.int64),
        clock_domain_id=tuple(payload["clock_domain_id"]),
        episode_id=tuple(payload["episode_id"]),
        session_id=tuple(payload["session_id"]),
        sample_id=tuple(payload["sample_id"]),
        chunk_id=tuple(payload["chunk_id"]),
        action_valid_mask=torch.tensor(payload["action_valid_mask"], dtype=torch.bool),
        suffix_valid_mask=torch.tensor(payload["suffix_valid_mask"], dtype=torch.bool),
        calibration_bundle_hash=tuple(payload["calibration_bundle_hash"]),
        wrench_geometry_spec_hash=tuple(payload["wrench_geometry_spec_hash"]),
        normalizer_hash=tuple(payload["normalizer_hash"]),
        calibration_mapping_hash_or_none=tuple(payload["calibration_mapping_hash_or_none"]),
        wrench_geometry_valid=torch.tensor(payload["wrench_geometry_valid"], dtype=torch.bool),
        runtime_artifact_compatible=torch.tensor(
            payload["runtime_artifact_compatible"], dtype=torch.bool
        ),
        selected_provenance=tuple(payload["selected_provenance"]),
    )


def load_fixed_validation_inputs(policy, dataset_root: Path, fixture: dict, device):
    from forcesmolvla.dataset_v3 import load_dataset_split
    from forcesmolvla.rules import load_and_validate_rulespec
    from forcesmolvla.shadow import sha256_file
    from forcesmolvla.training_data import load_runtime_artifacts, prepare_training_sample

    root = Path(__file__).parents[1].resolve()

    conversion = json.loads(
        (dataset_root / "conversion_manifest.json").read_text(encoding="utf-8")
    )
    repo_id = conversion.get("repo_id")
    if not isinstance(repo_id, str) or not repo_id:
        raise RuntimeError("P8_DATASET_REPO_ID_MISSING")
    dataset = load_dataset_split(
        dataset_root,
        repo_id=repo_id,
        split_name="val",
        artifact_use="development",
        delta_timestamps={"action": [index / 30 for index in range(50)]},
    )
    raw_samples = []
    for expected in fixture["tuple_list"]:
        sample = dataset[int(expected["fixture_position"])]
        if (
            int(sample["episode_index"]) != int(expected["episode_index"])
            or int(sample["frame_index"]) != int(expected["frame_index"])
        ):
            raise RuntimeError("P8_FIXED_VALIDATION_TUPLE_MISMATCH")
        raw_samples.append(sample)
    runtime_artifacts = load_runtime_artifacts(
        dataset_root,
        calibration_bundle_path=root / "configs/calibration_bundle.development.json",
        wrench_geometry_spec_path=root / "configs/wrench_geometry_spec.development.json",
        action_delta_spec_path=root / "artifacts/development/action_delta_spec.json",
        expected_repo_id=repo_id,
    )
    action_rules_path = root / "tests/fixtures/shadow_safety_thresholds.test_only.yaml"
    action_rules = load_and_validate_rulespec(
        action_rules_path,
        root / "schemas/rulespec.schema.json",
        formal=False,
    )
    policy.bind_action_safety_rules(
        action_rules, rules_sha256=sha256_file(action_rules_path)
    )
    prepared = [
        prepare_training_sample(sample, runtime_artifacts.normalizer) for sample in raw_samples
    ]
    batch = _make_batch(policy, prepared, device)
    batch["raw_state_snapshot"] = torch.stack(
        [torch.as_tensor(sample["observation.state"], dtype=torch.float32) for sample in raw_samples]
    ).to(device)
    return batch, raw_samples, runtime_artifacts


def compute_fixed_parity(policy, batch, runtime_artifacts, fixture: dict) -> dict:
    if batch[ACTION].shape != (2, 50, 7):
        raise RuntimeError("P8_PARITY_REQUIRES_FIXED_B2_H50_ACTION7")
    device = batch[ACTION].device
    noise7 = torch.tensor(fixture["epsilon7"]["tensor"], dtype=torch.float32, device=device)
    timestep = torch.tensor(fixture["time"]["tensor"], dtype=torch.float32, device=device)
    chunk_context = chunk_context_from_fixture(
        fixture, policy_generation=policy._context_generation
    )
    if not torch.equal(
        batch["raw_state_snapshot"].detach().cpu(), chunk_context.raw_state_snapshot
    ):
        raise RuntimeError("P8_PARITY_RAW_STATE_FIXTURE_MISMATCH")

    policy.eval()
    policy.bind_runtime_artifacts(runtime_artifacts)
    with torch.inference_mode(), torch.autocast(
        device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
    ):
        images, image_masks = policy.prepare_images(batch)
        state = policy.prepare_state(batch)
        state = state * (torch.arange(32, device=device) < 7).view(1, 32).to(state.dtype)
        wrench = policy._prepare_wrench(batch, device=device)
        prefix = policy.model.encode_prefix(
            images,
            image_masks,
            batch["observation.language.tokens"],
            batch["observation.language.attention_mask"],
            state,
        )
        prefix.validate()
        actions32 = pad_vector(batch[ACTION], 32)
        noise32 = pad_vector(noise7, 32)
        feature_mask, suffix_valid = policy._action_masks(batch, horizon=50, device=device)
        expanded_time = timestep[:, None, None]
        x_t = (expanded_time * noise32 + (1.0 - expanded_time) * actions32)
        x_t = x_t * feature_mask.to(dtype=x_t.dtype)
        force_context = policy.model.build_force_context(
            prefix.prefix_out, prefix.prefix_valid_mask, wrench
        )
        velocity = policy.model.velocity_cached(
            prefix,
            x_t,
            timestep,
            action_feature_mask=feature_mask,
            suffix_valid_mask=suffix_valid,
            force_context=force_context,
        )
        normalized_delta7, absolute_tensor = policy._predict_action_chunks(
            batch, chunk_context=chunk_context, noise=noise7
        )

    prefix_components = {
        "prefix_out": tensor_record(prefix.prefix_out),
        "prefix_valid_mask": tensor_record(prefix.prefix_valid_mask),
        "prefix_segment_ids": tensor_record(prefix.prefix_segment_ids),
        "prefix_position_ids": tensor_record(prefix.prefix_position_ids),
        "layout": {
            "camera1": [0, 64],
            "camera2": [64, 128],
            "language": [128, 176],
            "state": [176, 177],
            "physical_length": 177,
        },
    }
    cache = cache_record(prefix.past_key_values)
    delta_numpy = normalized_delta7.detach().cpu().to(torch.float32).numpy().astype(np.float64)
    raw_state = batch["raw_state_snapshot"].detach().cpu().numpy().astype(np.float64)
    absolute7 = absolute_tensor.detach().cpu().to(torch.float32).numpy().astype(np.float64)
    unnormalized_delta7 = ActionDeltaProcessor.to_delta(absolute7, raw_state)
    unnormalized_tensor = torch.from_numpy(np.ascontiguousarray(unnormalized_delta7))
    result = {
        "batch_size": 2,
        "horizon": 50,
        "prefix_context": {
            "components": prefix_components,
            "sha256": canonical_sha256(prefix_components),
        },
        "prefix_cache": cache,
        "velocity": tensor_record(velocity),
        "normalized_delta7_chunk": tensor_record(normalized_delta7),
        "unnormalized_delta7_chunk": tensor_record(unnormalized_tensor),
        "absolute7_chunk": tensor_record(absolute_tensor.cpu()),
        "normalizer_call_count": {"state7": 2, "wrench6": 2, "delta_action7": 2},
    }
    result["parity_sha256"] = canonical_sha256(result)
    return result
