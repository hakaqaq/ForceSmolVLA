"""CPU-only fail-closed preflight for the approved hybrid Stage-3 parent."""

from __future__ import annotations

from collections import Counter
import gc
import hashlib
import json
import math
from pathlib import Path
import struct
from typing import Any, Mapping

from jsonschema import Draft202012Validator
import yaml


ROOT = Path(__file__).resolve().parents[4]
DEFAULT_CONFIG = ROOT / "configs/stage3_parent_binding.v1.development.json"
SCHEMA = ROOT / "schemas/stage3_parent_binding.v1.schema.json"


class ParentBindingError(RuntimeError):
    """A hybrid parent artifact or contract failed closed."""


def _require(condition: bool, code: str) -> None:
    if not condition:
        raise ParentBindingError(code)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tree_record(root: Path, known_hashes: Mapping[Path, str] | None = None) -> dict[str, Any]:
    root = Path(root).resolve()
    _require(root.is_dir(), f"STAGE3_PARENT_TREE_MISSING:{root}")
    known = {Path(path).resolve(): value for path, value in (known_hashes or {}).items()}
    digest = hashlib.sha256()
    total = 0
    files = sorted(path for path in root.rglob("*") if path.is_file())
    for path in files:
        value = known.get(path.resolve()) or sha256_file(path)
        digest.update(f"{path.relative_to(root).as_posix()}\0{value}\n".encode())
        total += path.stat().st_size
    return {
        "tree_sha256": digest.hexdigest(),
        "file_count": len(files),
        "total_file_size": total,
    }


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"STAGE3_PARENT_JSON_ROOT:{path}")
    return value


def _yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"STAGE3_PARENT_YAML_ROOT:{path}")
    return value


def _resolve(path: str) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def load_parent_binding(path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    return _json(Path(path))


def validate_parent_binding_schema(value: Mapping[str, Any]) -> dict[str, Any]:
    binding = json.loads(json.dumps(dict(value), allow_nan=False))
    schema = _json(SCHEMA)
    Draft202012Validator.check_schema(schema)
    errors = sorted(
        Draft202012Validator(schema).iter_errors(binding),
        key=lambda error: tuple(str(part) for part in error.absolute_path),
    )
    if errors:
        location = ".".join(str(part) for part in errors[0].absolute_path)
        raise ParentBindingError(
            f"STAGE3_PARENT_SCHEMA:{location}:{errors[0].message}"
        )
    return binding


def validate_parent_binding_semantics(value: Mapping[str, Any]) -> dict[str, Any]:
    binding = validate_parent_binding_schema(value)
    continuation = binding["continuation_semantics"]
    _require(binding["binding_type"] == "new_hybrid_stage3_bootstrap", "STAGE3_PARENT_TYPE")
    _require(
        continuation == {
            "parent_binding_decision": "APPROVED_HYBRID",
            "exact_phase2_continuation": False,
            "not_exact_phase2_cycle210_continuation": True,
            "cycle210_full_learner_checkpoint_available": False,
            "cycle210_full_learner_checkpoint_expected_path": continuation[
                "cycle210_full_learner_checkpoint_expected_path"
            ],
            "full_learner_resume": False,
            "actor_source_role": "cycle210_evaluation_checkpoint_stage3_initial_actor",
            "critic_source_role": "g7a_r2_online_and_target_twin_q",
        },
        "STAGE3_PARENT_CONTINUATION_SEMANTICS",
    )
    _require(binding["actor_parent"]["selected"], "STAGE3_PARENT_ACTOR_NOT_SELECTED")
    _require(binding["actor_parent"]["full_learner_resume"] is False, "STAGE3_PARENT_ACTOR_RESUME")
    for name in (
        "normalizer_binding", "action_contract_binding", "task_feature_binding",
        "calibration_binding", "runtime_contract_binding",
    ):
        _require(binding[name]["selected"] is True, f"STAGE3_PARENT_BINDING_NOT_SELECTED:{name}")
    _require(binding["critic_parent"]["source_id"] == "G7A-r2", "STAGE3_PARENT_CRITIC_SOURCE")
    _require(binding["target_critic_parent"]["source_id"] == "G7A-r2", "STAGE3_PARENT_TARGET_SOURCE")
    for group_name in ("critic_parent", "target_critic_parent"):
        _require(
            all(item["selected"] is True for item in binding[group_name]["artifacts"]),
            f"STAGE3_PARENT_GROUP_ARTIFACT_NOT_SELECTED:{group_name}",
        )
    _require(
        binding["actor_parent"]["architecture_binding"]["module"]
        == "forcesmolvla.modeling_forcesmolvla.ForceSmolVLAPolicy",
        "STAGE3_PARENT_ACTOR_ARCHITECTURE",
    )
    _require(
        binding["critic_parent"]["architecture_binding"]["module"]
        == binding["target_critic_parent"]["architecture_binding"]["module"]
        == "forcesmolvla.rft.critic.ForceAwareMacroCritic",
        "STAGE3_PARENT_CRITIC_ARCHITECTURE",
    )
    optimizer = binding["optimizer_policy"]
    for name in (
        "inherit_actor_optimizer", "inherit_critic_optimizer", "inherit_scheduler",
        "inherit_rng", "inherit_sampler", "instantiated_in_this_round",
    ):
        _require(optimizer[name] is False, f"STAGE3_PARENT_OPTIMIZER_POLICY:{name}")
    _require(optimizer["rebuild_spec_status"] == "FROZEN", "STAGE3_PARENT_REBUILD_SPEC")
    safety = binding["initial_safety_state"]
    _require(safety["initial_actor_update_enabled"] is False, "STAGE3_PARENT_ACTOR_UPDATE_LOCK")
    _require(safety["initial_actor_q_guidance_enabled"] is False, "STAGE3_PARENT_Q_GUIDANCE_LOCK")
    _require(safety["critic_warmup_required"] is True, "STAGE3_PARENT_WARMUP_REQUIRED")
    _require(safety["critic_ready"] is False, "STAGE3_PARENT_CRITIC_NOT_READY")
    _require(safety["unlock_requires_independent_critic_gate"] is True, "STAGE3_PARENT_UNLOCK_GATE")
    _require(not any(binding["authorization"].values()), "STAGE3_PARENT_AUTHORIZATION_EXPANDED")
    return binding


def _verify_artifact(record: Mapping[str, Any], cache: dict[Path, str]) -> dict[str, Any]:
    path = Path(record["absolute_path"])
    _require(path.is_file(), f"STAGE3_PARENT_ARTIFACT_MISSING:{record['logical_role']}")
    resolved = path.resolve()
    _require(str(resolved) == record["resolved_realpath"], f"STAGE3_PARENT_REALPATH:{record['logical_role']}")
    _require(path.stat().st_size == record["size_bytes"], f"STAGE3_PARENT_SIZE:{record['logical_role']}")
    digest = sha256_file(path)
    _require(digest == record["sha256"], f"STAGE3_PARENT_SHA256:{record['logical_role']}")
    cache[resolved] = digest
    return {
        "path": str(path),
        "resolved_realpath": str(resolved),
        "size_bytes": path.stat().st_size,
        "sha256": digest,
        "status": "PASS",
    }


def _verify_named_file(path: str, expected: str, label: str, cache: dict[Path, str]) -> Path:
    resolved = _resolve(path).resolve()
    _require(resolved.is_file(), f"STAGE3_PARENT_BOUND_FILE_MISSING:{label}")
    digest = sha256_file(resolved)
    _require(digest == expected, f"STAGE3_PARENT_BOUND_FILE_SHA:{label}")
    cache[resolved] = digest
    return resolved


def _safetensors_header(path: Path) -> dict[str, Any]:
    with path.open("rb") as stream:
        raw_length = stream.read(8)
        _require(len(raw_length) == 8, "STAGE3_PARENT_ACTOR_SAFETENSORS_LENGTH")
        header_length = struct.unpack("<Q", raw_length)[0]
        _require(0 < header_length < path.stat().st_size - 8, "STAGE3_PARENT_ACTOR_HEADER_LENGTH")
        header = json.loads(stream.read(header_length))
    _require(isinstance(header, dict), "STAGE3_PARENT_ACTOR_HEADER_ROOT")
    tensors = {name: item for name, item in header.items() if name != "__metadata__"}
    _require(len(tensors) == 574, "STAGE3_PARENT_ACTOR_TENSOR_COUNT")
    widths = {"F32": 4, "BF16": 2}
    ranges: list[tuple[int, int]] = []
    for name, item in tensors.items():
        _require(isinstance(item, dict), f"STAGE3_PARENT_ACTOR_TENSOR_HEADER:{name}")
        dtype = item.get("dtype")
        shape = item.get("shape")
        offsets = item.get("data_offsets")
        _require(dtype in widths, f"STAGE3_PARENT_ACTOR_DTYPE:{name}")
        _require(
            isinstance(shape, list) and all(isinstance(dim, int) and dim >= 0 for dim in shape),
            f"STAGE3_PARENT_ACTOR_SHAPE:{name}",
        )
        _require(
            isinstance(offsets, list) and len(offsets) == 2
            and all(isinstance(offset, int) for offset in offsets),
            f"STAGE3_PARENT_ACTOR_OFFSET:{name}",
        )
        start, end = offsets
        numel = math.prod(shape)
        _require(0 <= start <= end and end - start == numel * widths[dtype], f"STAGE3_PARENT_ACTOR_SPAN:{name}")
        ranges.append((start, end))
    ranges.sort()
    _require(ranges[0][0] == 0, "STAGE3_PARENT_ACTOR_FIRST_OFFSET")
    _require(all(left[1] == right[0] for left, right in zip(ranges, ranges[1:])), "STAGE3_PARENT_ACTOR_OFFSET_GAP")
    _require(ranges[-1][1] == path.stat().st_size - 8 - header_length, "STAGE3_PARENT_ACTOR_LAST_OFFSET")
    required = {
        "model.action_in_proj.weight": ("F32", [720, 32]),
        "model.action_out_proj.weight": ("F32", [32, 720]),
        "model.force_adapter.learned_action_slot": ("F32", [50, 720]),
        "model.force_branch.force_mlp.linear_in.weight": ("F32", [960, 6]),
        "model.state_proj.weight": ("F32", [960, 32]),
    }
    for name, (dtype, shape) in required.items():
        _require(name in tensors, f"STAGE3_PARENT_ACTOR_KEY:{name}")
        _require(tensors[name]["dtype"] == dtype and tensors[name]["shape"] == shape, f"STAGE3_PARENT_ACTOR_KEY_SPEC:{name}")
    return {
        "header_bytes": header_length,
        "tensor_count": len(tensors),
        "dtype_counts": dict(sorted(Counter(item["dtype"] for item in tensors.values()).items())),
        "required_key_shape_dtype_count": len(required),
        "tensor_payload_bytes": ranges[-1][1],
    }


def _actor_compatibility(binding: Mapping[str, Any]) -> dict[str, Any]:
    actor = binding["actor_parent"]
    architecture = actor["architecture_binding"]
    config = _json(Path(architecture["config_path"]))
    _require(config.get("type") == "force_smolvla", "STAGE3_PARENT_ACTOR_CONFIG_TYPE")
    _require(config.get("chunk_size") == 50 and config.get("num_steps") == 10, "STAGE3_PARENT_ACTOR_H50_N10")
    inputs = config.get("input_features", {})
    _require(inputs.get("observation.state", {}).get("shape") == [7], "STAGE3_PARENT_ACTOR_STATE7")
    _require(inputs.get("observation.wrench", {}).get("shape") == [6], "STAGE3_PARENT_ACTOR_WRENCH6")
    _require(config.get("output_features", {}).get("action", {}).get("shape") == [7], "STAGE3_PARENT_ACTOR_ACTION7")
    cameras = [name for name in inputs if name.startswith("observation.images.")]
    _require(len(cameras) == 2, "STAGE3_PARENT_ACTOR_CAMERAS")
    manifest = _json(Path(architecture["actor_export_manifest_path"]))
    metadata = manifest.get("metadata", {})
    coverage = metadata.get("actor_state_coverage", {})
    _require(coverage == {"coverage_fraction": 1.0, "full_actor_state": True, "loaded_tensor_count": 574, "source_tensor_count": 574}, "STAGE3_PARENT_ACTOR_EXPORT_COVERAGE")
    _require(metadata.get("strict_load") == {"missing_keys": 0, "unexpected_keys": 0}, "STAGE3_PARENT_ACTOR_PRIOR_STRICT_LOAD")
    _require(metadata.get("critic_exported") is False and metadata.get("optimizer_exported") is False, "STAGE3_PARENT_ACTOR_EXPORT_SCOPE")
    _require(metadata.get("scheduler_exported") is False and metadata.get("rng_exported") is False and metadata.get("sampler_exported") is False, "STAGE3_PARENT_ACTOR_EXPORT_NOT_LEARNER")
    _require(metadata.get("training_parent_allowed") is False, "STAGE3_PARENT_ACTOR_EXPORT_TRAINING_SCOPE")
    sources = metadata.get("source_bindings", {})
    _require(sources.get("cycle210_actor_state_sha256") == architecture["source_actor_state_sha256"], "STAGE3_PARENT_ACTOR_SOURCE_STATE")
    _require(manifest.get("payloads", {}).get("model.safetensors", {}).get("sha256") == actor["sha256"], "STAGE3_PARENT_ACTOR_MANIFEST_MODEL_SHA")
    return {
        "status": "PASS",
        "validation_level": actor["load_validation_level"],
        "safetensors": _safetensors_header(Path(actor["absolute_path"])),
        "config": {"H": 50, "N": 10, "state": 7, "wrench": 6, "action": 7, "cameras": 2},
        "prior_strict_export_coverage": "574/574",
        "full_learner_resume": False,
        "real_model_forward": "NOT_RUN",
    }


def _critic_compatibility(binding: Mapping[str, Any]) -> dict[str, Any]:
    import torch
    from forcesmolvla.rft.critic import (
        ForceAwareMacroCritic,
        FrozenConRFTResNet10,
        frozen_task_feature,
    )

    _require(not torch.cuda.is_initialized(), "STAGE3_PARENT_CUDA_ALREADY_INITIALIZED")
    module = ForceAwareMacroCritic(
        FrozenConRFTResNet10(), FrozenConRFTResNet10(), task_feature=frozen_task_feature()
    )
    expected = module.state_dict()
    expected_numel = sum(tensor.numel() for tensor in expected.values())
    results: dict[str, Any] = {}
    groups = (binding["critic_parent"], binding["target_critic_parent"])
    for group in groups:
        for artifact in group["artifacts"]:
            role = artifact["logical_role"]
            state = torch.load(
                artifact["absolute_path"], map_location="cpu", weights_only=True
            )
            validate_critic_state_against_expected(state, expected, role)
            incompatible = module.load_state_dict(state, strict=True)
            _require(not incompatible.missing_keys and not incompatible.unexpected_keys, f"STAGE3_PARENT_CRITIC_STRICT_LOAD:{role}")
            results[role] = {
                "status": "PASS",
                "key_count": len(state),
                "numel": sum(tensor.numel() for tensor in state.values()),
                "missing_keys": 0,
                "unexpected_keys": 0,
                "shape_mismatches": 0,
                "dtype_mismatches": 0,
                "map_location": "cpu",
                "weights_only": True,
                "strict": True,
            }
            del state
            gc.collect()
    del expected, module
    gc.collect()
    _require(not torch.cuda.is_initialized(), "STAGE3_PARENT_CUDA_INITIALIZED")
    return {
        "status": "PASS",
        "module": "forcesmolvla.rft.critic.ForceAwareMacroCritic",
        "expected_key_count": 130,
        "expected_numel": expected_numel,
        "artifacts": results,
        "target_fallback_from_online": False,
        "optimizer_instantiated": False,
        "optimizer_steps": 0,
        "polyak_updates": 0,
        "real_model_forward": "NOT_RUN",
        "cuda_initialized": False,
    }


def validate_critic_state_against_expected(
    state: Mapping[str, Any], expected: Mapping[str, Any], role: str
) -> None:
    """Apply the same strict key/shape/dtype policy used by the real preflight."""

    import torch

    _require(isinstance(state, Mapping), f"STAGE3_PARENT_CRITIC_STATE_ROOT:{role}")
    _require(
        all(isinstance(value, torch.Tensor) and value.device.type == "cpu" for value in state.values()),
        f"STAGE3_PARENT_CRITIC_TENSOR:{role}",
    )
    missing = sorted(set(expected) - set(state))
    unexpected = sorted(set(state) - set(expected))
    _require(not missing, f"STAGE3_PARENT_CRITIC_MISSING_KEYS:{role}:{missing[:3]}")
    _require(not unexpected, f"STAGE3_PARENT_CRITIC_UNEXPECTED_KEYS:{role}:{unexpected[:3]}")
    mismatched = sorted(
        name for name in expected
        if tuple(expected[name].shape) != tuple(state[name].shape)
        or expected[name].dtype != state[name].dtype
    )
    _require(not mismatched, f"STAGE3_PARENT_CRITIC_KEY_SPEC:{role}:{mismatched[:3]}")


def _cross_component_compatibility(binding: Mapping[str, Any]) -> dict[str, Any]:
    from forcesmolvla.rft.critic import frozen_task_feature_sha256

    actor_config = _json(Path(binding["actor_parent"]["architecture_binding"]["config_path"]))
    critic_config = _yaml(Path(binding["critic_parent"]["architecture_binding"]["config_path"]))
    normalizer = _json(Path(binding["normalizer_binding"]["absolute_path"]))
    action = _json(Path(binding["action_contract_binding"]["absolute_path"]))
    topology = _json(Path(binding["task_feature_binding"]["absolute_path"]))
    calibration = _json(Path(binding["calibration_binding"]["absolute_path"]))
    runtime = _json(Path(binding["runtime_contract_binding"]["absolute_path"]))
    transition = _json(ROOT / "configs/stage3_transition_contract.v1.development.json")
    g7a = _yaml(Path(binding["optimizer_policy"]["critic_candidate"]["source_path"]))
    evidence = binding["compatibility_evidence"]
    actor_export = _json(Path(binding["actor_parent"]["architecture_binding"]["actor_export_manifest_path"]))
    features = normalizer.get("features", {})
    for name, width in (("state7", 7), ("wrench6", 6), ("delta_action7", 7)):
        record = features.get(name, {})
        _require(len(record.get("mean", [])) == width and len(record.get("std", [])) == width, f"STAGE3_PARENT_NORMALIZER_SHAPE:{name}")
        _require(all(math.isfinite(value) for value in record["mean"] + record["std"]), f"STAGE3_PARENT_NORMALIZER_FINITE:{name}")
        _require(all(value > 0 for value in record["std"]), f"STAGE3_PARENT_NORMALIZER_STD:{name}")
    _require(normalizer.get("calibration_bundle_sha256") == binding["calibration_binding"]["sha256"], "STAGE3_PARENT_CALIBRATION_NORMALIZER_BINDING")
    export_payloads = actor_export.get("payloads", {})
    for name, artifact_name in (
        ("manifests/normalizer_manifest.json", "normalizer_binding"),
        ("manifests/stage2_action_contract.v2.development.json", "action_contract_binding"),
        ("manifests/calibration_bundle.development.json", "calibration_binding"),
        ("manifests/converter_runtime_spec.task2.development.json", "runtime_contract_binding"),
    ):
        _require(
            export_payloads.get(name, {}).get("sha256") == binding[artifact_name]["sha256"],
            f"STAGE3_PARENT_ACTOR_EXPORT_RUNTIME_BINDING:{artifact_name}",
        )
    _require(action.get("critic_action_shape") == [3, 7], "STAGE3_PARENT_ACTION_CONTRACT_SHAPE")
    _require(action.get("actor_q_guided_action_dims") == [0, 1, 2, 3, 4, 5], "STAGE3_PARENT_TCP6_CONTRACT")
    _require(action.get("gripper_q_gradient") is False and action.get("gripper", {}).get("stop_gradient") is True, "STAGE3_PARENT_GRIPPER_CONTRACT")
    _require(action.get("frozen_normalizer_manifest", {}).get("sha256") == binding["normalizer_binding"]["sha256"], "STAGE3_PARENT_ACTION_NORMALIZER_BINDING")
    task = topology.get("gpu_zero_update_preflight", {}).get("task_condition", {})
    _require(task.get("frozen_task_feature_dim") == 256, "STAGE3_PARENT_TASK_DIM")
    _require(task.get("frozen_task_feature_sha256") == binding["task_feature_binding"]["logical_object_sha256"], "STAGE3_PARENT_TASK_EVIDENCE_DIGEST")
    _require(frozen_task_feature_sha256() == binding["task_feature_binding"]["logical_object_sha256"], "STAGE3_PARENT_TASK_RECOMPUTED_DIGEST")
    _require(calibration.get("validated") is True and calibration.get("formal_ready") is False, "STAGE3_PARENT_CALIBRATION_STATUS")
    _require(len(calibration.get("sensor_bias6", [])) == 6 and len(calibration.get("wrench_sign6", [])) == 6, "STAGE3_PARENT_CALIBRATION_SHAPE")
    _require(runtime.get("controller_grid", {}).get("fps") == 30, "STAGE3_PARENT_RUNTIME_GRID")
    _require(runtime.get("formal_ready") is False and runtime.get("artifact_status") == "development_only", "STAGE3_PARENT_RUNTIME_STATUS")
    temporal = transition.get("temporal", {})
    _require(
        (temporal.get("data_grid_hz"), temporal.get("policy_hz"), temporal.get("flow_horizon"), temporal.get("critic_slots"), temporal.get("critic_action_features"))
        == (30, 10, 50, 3, 7),
        "STAGE3_PARENT_TEMPORAL_CONTRACT",
    )
    critic_interface = critic_config.get("critic_interface", {})
    _require(critic_interface.get("f_policy_hz") == 10 and critic_interface.get("action_shape") == [3, 7], "STAGE3_PARENT_CRITIC_CONFIG_ACTION")
    observation = critic_config.get("observation", {})
    _require(observation.get("normalized_state_features") == 7 and observation.get("normalized_wrench_features") == 6 and observation.get("frozen_task_feature_dim") == 256, "STAGE3_PARENT_CRITIC_CONFIG_OBSERVATION")
    _require(g7a.get("transition_contract", {}).get("critic_action_shape") == [3, 7] and g7a.get("transition_contract", {}).get("policy_rate_hz") == 10, "STAGE3_PARENT_G7A_TEMPORAL_ACTION")
    _require(actor_config.get("chunk_size") == 50 and actor_config.get("output_features", {}).get("action", {}).get("shape") == [7], "STAGE3_PARENT_ACTOR_ACTION_CONTRACT")
    actor_image_source = _resolve(evidence["actor_image_adapter_source"]["path"]).read_text(encoding="utf-8")
    critic_source = _resolve(evidence["critic_source"]["path"]).read_text(encoding="utf-8")
    _require("def build_actor_batch(" in actor_image_source and ".float().div_(255)" in actor_image_source, "STAGE3_PARENT_ACTOR_IMAGE_RANGE_SOURCE")
    _require("image.dtype != torch.uint8" in critic_source and "value / 255.0" in critic_source, "STAGE3_PARENT_CRITIC_IMAGE_RANGE_SOURCE")
    return {
        "status": "PASS",
        "temporal": {"data_grid_hz": 30, "policy_hz": 10, "H": 50, "K": 3, "action_features": 7},
        "normalizer_action_contract": "PASS",
        "actor_critic_features": {"state": 7, "wrench": 6, "action": 7, "task": 256, "cameras": 2},
        "image_contracts": {
            "actor": "float32 [0,1] before Actor preprocessing",
            "critic": "uint8 [0,255] accepted; internal float32 /255",
        },
        "task_feature": {"dimension": 256, "sha256": frozen_task_feature_sha256(), "evidence_level": "container_plus_recomputed_logical_tensor_digest"},
        "calibration_runtime_hashes": "PASS",
        "tcp6_q_gradient": True,
        "gripper_q_gradient": False,
        "calibration_formal_ready": False,
        "runtime_formal_ready": False,
    }


def _git_head() -> str:
    head = (ROOT / ".git/HEAD").read_text(encoding="utf-8").strip()
    if head.startswith("ref: "):
        ref = ROOT / ".git" / head[5:]
        if ref.is_file():
            return ref.read_text(encoding="utf-8").strip()
        packed = ROOT / ".git/packed-refs"
        if packed.is_file():
            suffix = " " + head[5:]
            for line in packed.read_text(encoding="utf-8").splitlines():
                if line.endswith(suffix):
                    return line.split(" ", 1)[0]
    return head


def preflight_parent_binding(config_path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    binding = validate_parent_binding_semantics(load_parent_binding(config_path))
    cache: dict[Path, str] = {}
    artifacts: dict[str, Any] = {}
    artifacts["actor"] = _verify_artifact(binding["actor_parent"], cache)
    for group_name in ("critic_parent", "target_critic_parent"):
        for artifact in binding[group_name]["artifacts"]:
            artifacts[artifact["logical_role"]] = _verify_artifact(artifact, cache)
    for name in ("normalizer_binding", "action_contract_binding", "task_feature_binding", "calibration_binding", "runtime_contract_binding"):
        artifacts[name] = _verify_artifact(binding[name], cache)

    evidence = binding["compatibility_evidence"]
    for label in ("critic_config", "critic_source", "stage3_transition_contract", "actor_image_adapter_source"):
        item = evidence[label]
        _verify_named_file(item["path"], item["sha256"], label, cache)
    checkpoint = evidence["g7a_r2_checkpoint"]
    checkpoint_manifest_path = _verify_named_file(checkpoint["checkpoint_manifest_path"], checkpoint["checkpoint_manifest_sha256"], "g7a_r2_checkpoint_manifest", cache)
    _verify_named_file(checkpoint["source_manifest_path"], checkpoint["source_manifest_sha256"], "g7a_r2_source_manifest", cache)
    adapters = evidence["temporal_action_adapter_binding"]
    for label, path_key, sha_key in (
        ("temporal", "temporal_path", "temporal_sha256"),
        ("action_delta", "action_delta_path", "action_delta_sha256"),
        ("critic_action_adapter", "critic_action_adapter_path", "critic_action_adapter_sha256"),
        ("processor_graph", "processor_graph_path", "processor_graph_sha256"),
    ):
        _verify_named_file(adapters[path_key], adapters[sha_key], label, cache)
    actor_arch = binding["actor_parent"]["architecture_binding"]
    _verify_named_file(actor_arch["config_path"], actor_arch["config_sha256"], "actor_config", cache)
    _verify_named_file(actor_arch["actor_export_manifest_path"], actor_arch["actor_export_manifest_sha256"], "actor_export_manifest", cache)
    critic_arch = binding["critic_parent"]["architecture_binding"]
    _verify_named_file(critic_arch["config_path"], critic_arch["config_sha256"], "critic_architecture_config", cache)
    _verify_named_file(binding["optimizer_policy"]["critic_candidate"]["source_path"], binding["optimizer_policy"]["critic_candidate"]["source_sha256"], "critic_optimizer_spec_source", cache)
    _verify_named_file("src/forcesmolvla/rft/frozen_vlm_trainability.py", binding["optimizer_policy"]["actor_candidate"]["source_sha256"], "actor_optimizer_spec_source", cache)

    checkpoint_manifest = _json(checkpoint_manifest_path)
    payload = dict(checkpoint_manifest)
    claimed_payload_sha = payload.pop("manifest_payload_sha256", None)
    actual_payload_sha = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    _require(claimed_payload_sha == checkpoint["checkpoint_manifest_payload_sha256"] == actual_payload_sha, "STAGE3_PARENT_G7A_MANIFEST_PAYLOAD")
    file_records = {item["relative_path"]: item["sha256"] for item in checkpoint_manifest.get("files", [])}
    for group_name in ("critic_parent", "target_critic_parent"):
        for artifact in binding[group_name]["artifacts"]:
            relative = Path(artifact["absolute_path"]).relative_to(Path(critic_arch["container_path"])).as_posix()
            _require(file_records.get(relative) == artifact["sha256"], f"STAGE3_PARENT_G7A_MANIFEST_ARTIFACT:{artifact['logical_role']}")
    protected_snapshot = _json(Path(critic_arch["container_path"]) / "manifests/protected_snapshot.json")
    _require(
        protected_snapshot.get("files", {}).get("action_contract_v2", {}).get("sha256")
        == binding["action_contract_binding"]["sha256"],
        "STAGE3_PARENT_G7A_ACTION_CONTRACT_BINDING",
    )
    _require(
        protected_snapshot.get("files", {}).get("action_adapter_v2", {}).get("sha256")
        == evidence["temporal_action_adapter_binding"]["critic_action_adapter_sha256"],
        "STAGE3_PARENT_G7A_ACTION_ADAPTER_BINDING",
    )
    _require(
        protected_snapshot.get("g5_protected", {}).get("files", {}).get("dataset_normalizer", {}).get("sha256")
        == binding["normalizer_binding"]["sha256"],
        "STAGE3_PARENT_G7A_NORMALIZER_BINDING",
    )

    actor_tree = tree_record(Path(actor_arch["container_path"]), cache)
    critic_tree = tree_record(Path(critic_arch["container_path"]), cache)
    for actual, architecture, label in ((actor_tree, actor_arch, "ACTOR"), (critic_tree, critic_arch, "CRITIC")):
        _require(actual["tree_sha256"] == architecture["container_tree_sha256"], f"STAGE3_PARENT_{label}_TREE_SHA")
        _require(actual["file_count"] == architecture["container_file_count"], f"STAGE3_PARENT_{label}_TREE_COUNT")
        _require(actual["total_file_size"] == architecture["container_total_file_size"], f"STAGE3_PARENT_{label}_TREE_SIZE")
    full_checkpoint = Path(binding["continuation_semantics"]["cycle210_full_learner_checkpoint_expected_path"])
    _require(not full_checkpoint.exists(), "STAGE3_PARENT_FULL_LEARNER_AVAILABILITY_DRIFT")

    actor_result = _actor_compatibility(binding)
    critic_result = _critic_compatibility(binding)
    cross_result = _cross_component_compatibility(binding)
    report: dict[str, Any] = {
        "schema_version": "forcesmolvla_stage3_parent_binding_preflight.v1",
        "binding_id": binding["binding_id"],
        "binding_type": binding["binding_type"],
        "parent_binding_decision": "APPROVED_HYBRID",
        "source_head": _git_head(),
        "binding_config_path": str(Path(config_path).resolve()),
        "binding_config_sha256": sha256_file(Path(config_path)),
        "tool_status": "PASS",
        "G0A_HYBRID_PARENT_BINDING": "PASS",
        "G0_FINAL_PARENT_BINDING": "BOUND_APPROVED_HYBRID",
        "PARENT_PAYLOAD_COMPLETE_FOR_HYBRID": True,
        "STRICT_PHASE2_CONTINUATION_AVAILABLE": False,
        "continuation_semantics": binding["continuation_semantics"],
        "artifacts": artifacts,
        "container_trees": {"actor": actor_tree, "g7a_r2": critic_tree},
        "ACTOR_METADATA_COMPATIBILITY": "PASS",
        "CRITIC_CPU_STATE_COMPATIBILITY": "PASS",
        "TARGET_CRITIC_CPU_STATE_COMPATIBILITY": "PASS",
        "CROSS_COMPONENT_CONTRACT_COMPATIBILITY": "PASS",
        "actor_preflight": actor_result,
        "critic_preflight": critic_result,
        "cross_component_preflight": cross_result,
        "optimizer": {
            "CROSS_STAGE_OPTIMIZER_REBUILD_SPEC": "FROZEN",
            "CROSS_STAGE_OPTIMIZER_REBUILT": "NOT_RUN",
            "instantiated": False,
            "optimizer_steps": 0,
            "polyak_updates": 0,
        },
        "safety": {
            "INITIAL_ACTOR_UPDATE_ENABLED": False,
            "CRITIC_WARMUP_REQUIRED": True,
            "CRITIC_READY": False,
            "ACTOR_Q_GUIDANCE_ENABLED": False,
        },
        "G0_FORMAL_GATE_PASSED": False,
        "G4P_GPU_NUMERICAL_PREFLIGHT_READY": True,
        "REAL_MODEL_FORWARD": "NOT_RUN",
        "G4_AND_LATER": "NOT_RUN",
        "CUDA_INITIALIZED": False,
        "ROBOT_CONNECTION_COUNT": 0,
        "ROBOT_COMMAND_COUNT": 0,
        "ROBOT_EXECUTION_AUTHORIZED": False,
        "unverified_items": binding["unverified_items"],
    }
    report["canonical_report_sha256"] = hashlib.sha256(
        json.dumps(report, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()
    return report


def render_parent_binding_markdown(report: Mapping[str, Any]) -> str:
    actor = report["artifacts"]["actor"]
    q1 = report["artifacts"]["online_q1"]
    q2 = report["artifacts"]["online_q2"]
    tq1 = report["artifacts"]["target_q1"]
    tq2 = report["artifacts"]["target_q2"]
    task = report["cross_component_preflight"]["task_feature"]
    lines = [
        "# Stage-3 G0A approved-hybrid parent binding preflight v1",
        "",
        "This report freezes the explicitly approved new hybrid Stage-3 bootstrap. It is not an exact Phase-2 cycle210 learner continuation and it does not authorize training, GPU model loading, publication, networking, ROS, or robot execution.",
        "",
        "## Decision and boundary",
        "",
        f"- `tool_status={report['tool_status']}`",
        f"- `G0A_HYBRID_PARENT_BINDING={report['G0A_HYBRID_PARENT_BINDING']}`",
        f"- `G0_FINAL_PARENT_BINDING={report['G0_FINAL_PARENT_BINDING']}`",
        "- `binding_type=new_hybrid_stage3_bootstrap`",
        "- `not_exact_phase2_cycle210_continuation=true`",
        "- `cycle210_full_learner_checkpoint_available=false`",
        "- `full_learner_resume=false`",
        "- `PARENT_PAYLOAD_COMPLETE_FOR_HYBRID=true`",
        "- `STRICT_PHASE2_CONTINUATION_AVAILABLE=false`",
        "",
        "The cycle210 evaluation export supplies only the Stage-3 initial Actor. It has no Critic, target, optimizer, scheduler, RNG, sampler, or learner cursor. G7A-r2 independently supplies Q1/Q2 and the stored target Q1/Q2. G7A-r5 remains present and explicitly unselected.",
        "",
        "## Selected payloads",
        "",
        "| Role | Path | SHA-256 | Validation |",
        "|---|---|---|---|",
        f"| Actor | `{actor['path']}` | `{actor['sha256']}` | safetensors header/key/shape/dtype plus prior 574/574 strict export evidence; no tensor load or forward |",
        f"| Q1 | `{q1['path']}` | `{q1['sha256']}` | CPU `weights_only=True`, strict key/shape/dtype |",
        f"| Q2 | `{q2['path']}` | `{q2['sha256']}` | CPU `weights_only=True`, strict key/shape/dtype |",
        f"| target Q1 | `{tq1['path']}` | `{tq1['sha256']}` | CPU `weights_only=True`, strict key/shape/dtype |",
        f"| target Q2 | `{tq2['path']}` | `{tq2['sha256']}` | CPU `weights_only=True`, strict key/shape/dtype |",
        "",
        f"Actor container tree: `{report['container_trees']['actor']['tree_sha256']}` ({report['container_trees']['actor']['file_count']} files, {report['container_trees']['actor']['total_file_size']} bytes).",
        "",
        f"G7A-r2 container tree: `{report['container_trees']['g7a_r2']['tree_sha256']}` ({report['container_trees']['g7a_r2']['file_count']} files, {report['container_trees']['g7a_r2']['total_file_size']} bytes).",
        "",
        "## Compatibility result",
        "",
        f"- `ACTOR_METADATA_COMPATIBILITY={report['ACTOR_METADATA_COMPATIBILITY']}`",
        f"- `CRITIC_CPU_STATE_COMPATIBILITY={report['CRITIC_CPU_STATE_COMPATIBILITY']}`",
        f"- `TARGET_CRITIC_CPU_STATE_COMPATIBILITY={report['TARGET_CRITIC_CPU_STATE_COMPATIBILITY']}`",
        f"- `CROSS_COMPONENT_CONTRACT_COMPATIBILITY={report['CROSS_COMPONENT_CONTRACT_COMPATIBILITY']}`",
        "- Actor `H=50`, Flow `N=10`; Critic `K=3`, action7; rational 30 Hz data grid and fixed 10 Hz policy phase match.",
        "- Actor images are float32 `[0,1]` before Actor preprocessing. Critic images use the distinct uint8 `[0,255]` path and are converted internally to float32 `/255`.",
        f"- Canonical task feature is 256D with logical tensor SHA-256 `{task['sha256']}`; evidence is `{task['evidence_level']}`.",
        "- State7, wrench6, normalizer, ActionContract-v2, calibration/runtime hashes, TCP6 Q-gradient, and gripper stop-gradient contracts match.",
        "- The calibration and runtime records remain development-only/formal-not-ready; this binding does not upgrade their formal status.",
        "",
        "## Optimizer and safety state",
        "",
        "Only the rebuild specification is frozen. Actor/Critic optimizers are fresh-by-policy but were not instantiated; no optimizer, scheduler, RNG, or sampler state is inherited. The G3P tiny CPU optimizer is not a cross-stage rebuild.",
        "",
        "- `CROSS_STAGE_OPTIMIZER_REBUILD_SPEC=FROZEN`",
        "- `CROSS_STAGE_OPTIMIZER_REBUILT=NOT_RUN`",
        "- `INITIAL_ACTOR_UPDATE_ENABLED=false`",
        "- `CRITIC_WARMUP_REQUIRED=true`",
        "- `CRITIC_READY=false`",
        "- `ACTOR_Q_GUIDANCE_ENABLED=false`",
        "",
        "Actor Q-guidance may be enabled only after a separately authorized Critic warmup/stability gate. This preflight does not implement or simulate that unlock.",
        "",
        "## Deferred validation",
        "",
    ]
    lines.extend(f"- {item}" for item in report["unverified_items"])
    lines.extend([
        "",
        "Therefore `G0_FORMAL_GATE_PASSED=false`, `REAL_MODEL_FORWARD=NOT_RUN`, and `G4_AND_LATER=NOT_RUN`. The next eligible activity is a separately authorized G4P GPU numerical preflight.",
        "",
        "## Safety footer",
        "",
        "```text",
        f"canonical_report_sha256={report['canonical_report_sha256']}",
        "CUDA_INITIALIZED=false",
        "ROBOT_CONNECTION_COUNT=0",
        "ROBOT_COMMAND_COUNT=0",
        "ROBOT_EXECUTION_AUTHORIZED=false",
        "```",
        "",
    ])
    return "\n".join(lines)
