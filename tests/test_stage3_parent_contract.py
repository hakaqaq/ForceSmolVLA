from __future__ import annotations

import ast
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path

import pytest
import torch

from forcesmolvla.rft.stage3 import parent as parent_module
from forcesmolvla.rft.stage3.parent import (
    DEFAULT_CONFIG,
    ParentBindingError,
    load_parent_binding,
    preflight_parent_binding,
    validate_critic_state_against_expected,
    validate_parent_binding_schema,
    validate_parent_binding_semantics,
)


ROOT = Path(__file__).parents[1]


@pytest.fixture(scope="module")
def binding() -> dict:
    return load_parent_binding()


@pytest.fixture(scope="module")
def real_preflight() -> dict:
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
    assert not torch.cuda.is_initialized()
    result = preflight_parent_binding()
    assert not torch.cuda.is_initialized()
    return result


def test_valid_approved_hybrid_binding_schema(binding: dict) -> None:
    assert validate_parent_binding_schema(binding) == binding
    assert validate_parent_binding_semantics(binding) == binding


def test_cycle210_evaluation_actor_is_selected_and_not_a_learner_resume(binding: dict) -> None:
    actor = binding["actor_parent"]
    assert actor["selected"] is True
    assert actor["sha256"] == "e24c1d6bb0a778921659514ac47c692b952178aa39af2601ccf0fc32bf94774d"
    assert actor["full_learner_resume"] is False
    assert "cycle210_evaluation_smoke_checkpoint.v1/model.safetensors" in actor["absolute_path"]


def test_g7a_r2_online_and_target_twin_q_are_selected(binding: dict) -> None:
    assert binding["critic_parent"]["source_id"] == "G7A-r2"
    assert binding["target_critic_parent"]["source_id"] == "G7A-r2"
    assert [item["logical_role"] for item in binding["critic_parent"]["artifacts"]] == ["online_q1", "online_q2"]
    assert [item["logical_role"] for item in binding["target_critic_parent"]["artifacts"]] == ["target_q1", "target_q2"]


def test_g7a_r5_is_retained_but_explicitly_unselected(binding: dict) -> None:
    candidate = binding["compatibility_evidence"]["unselected_parent_candidates"][0]
    assert candidate["logical_id"] == "G7A-r5-Actor"
    assert candidate["selected"] is False
    assert Path(candidate["path"]).is_dir()


def test_binding_is_not_exact_cycle210_continuation(binding: dict) -> None:
    semantics = binding["continuation_semantics"]
    assert semantics["exact_phase2_continuation"] is False
    assert semantics["not_exact_phase2_cycle210_continuation"] is True
    assert semantics["full_learner_resume"] is False


def test_missing_cycle210_full_learner_payload_is_explicit_and_not_masked(
    binding: dict, real_preflight: dict,
) -> None:
    semantics = binding["continuation_semantics"]
    assert semantics["cycle210_full_learner_checkpoint_available"] is False
    assert not Path(semantics["cycle210_full_learner_checkpoint_expected_path"]).exists()
    assert real_preflight["STRICT_PHASE2_CONTINUATION_AVAILABLE"] is False
    changed = deepcopy(binding)
    changed["continuation_semantics"]["cycle210_full_learner_checkpoint_available"] = True
    with pytest.raises(ParentBindingError, match="STAGE3_PARENT_SCHEMA"):
        validate_parent_binding_semantics(changed)


@pytest.mark.parametrize("role", ["Actor", "online_q1", "target_q1"])
def test_missing_parent_artifact_fails_closed(tmp_path: Path, binding: dict, role: str) -> None:
    if role == "Actor":
        record = deepcopy(binding["actor_parent"])
    elif role == "online_q1":
        record = deepcopy(binding["critic_parent"]["artifacts"][0])
    else:
        record = deepcopy(binding["target_critic_parent"]["artifacts"][0])
    missing = tmp_path / f"missing-{role}.bin"
    record["absolute_path"] = str(missing)
    record["resolved_realpath"] = str(missing)
    with pytest.raises(ParentBindingError, match="STAGE3_PARENT_ARTIFACT_MISSING"):
        parent_module._verify_artifact(record, {})


@pytest.mark.parametrize("role", ["Actor", "online_q2", "target_q2"])
def test_mismatched_parent_sha_fails_closed(tmp_path: Path, binding: dict, role: str) -> None:
    if role == "Actor":
        record = deepcopy(binding["actor_parent"])
    elif role == "online_q2":
        record = deepcopy(binding["critic_parent"]["artifacts"][1])
    else:
        record = deepcopy(binding["target_critic_parent"]["artifacts"][1])
    payload = tmp_path / f"payload-{role}.bin"
    payload.write_bytes(b"parent-payload")
    record.update({
        "absolute_path": str(payload),
        "resolved_realpath": str(payload.resolve()),
        "size_bytes": payload.stat().st_size,
        "sha256": "0" * 64,
    })
    with pytest.raises(ParentBindingError, match="STAGE3_PARENT_SHA256"):
        parent_module._verify_artifact(record, {})


@pytest.mark.parametrize(
    ("state", "pattern"),
    [
        ({}, "MISSING_KEYS"),
        ({"weight": torch.zeros(2, 3), "extra": torch.zeros(1)}, "UNEXPECTED_KEYS"),
        ({"weight": torch.zeros(3, 2)}, "KEY_SPEC"),
        ({"weight": torch.zeros(2, 3, dtype=torch.float64)}, "KEY_SPEC"),
    ],
    ids=["missing-key", "unexpected-key", "shape-mismatch", "dtype-mismatch"],
)
def test_critic_architecture_key_shape_dtype_mismatch_fails_closed(
    state: dict, pattern: str,
) -> None:
    expected = {"weight": torch.zeros(2, 3)}
    with pytest.raises(ParentBindingError, match=pattern):
        validate_critic_state_against_expected(state, expected, "test_q")


@pytest.mark.parametrize(
    ("binding_name", "mutation", "pattern"),
    [
        ("normalizer_binding", "normalizer", "NORMALIZER_SHAPE"),
        ("action_contract_binding", "action", "ACTION_CONTRACT_SHAPE"),
        ("task_feature_binding", "task", "TASK_EVIDENCE_DIGEST"),
        ("calibration_binding", "calibration", "CALIBRATION_STATUS"),
        ("runtime_contract_binding", "runtime", "RUNTIME_GRID"),
    ],
    ids=["normalizer", "action-contract", "task-feature", "calibration", "runtime"],
)
def test_cross_component_mismatch_fails_closed(
    tmp_path: Path,
    binding: dict,
    binding_name: str,
    mutation: str,
    pattern: str,
) -> None:
    changed = deepcopy(binding)
    source = Path(changed[binding_name]["absolute_path"])
    payload = json.loads(source.read_text(encoding="utf-8"))
    if mutation == "normalizer":
        payload["features"]["state7"]["mean"] = payload["features"]["state7"]["mean"][:-1]
    elif mutation == "action":
        payload["critic_action_shape"] = [4, 7]
    elif mutation == "task":
        payload["gpu_zero_update_preflight"]["task_condition"]["frozen_task_feature_sha256"] = "0" * 64
    elif mutation == "calibration":
        payload["validated"] = False
    else:
        payload["controller_grid"]["fps"] = 29
    altered = tmp_path / f"{mutation}.json"
    altered.write_text(json.dumps(payload), encoding="utf-8")
    changed[binding_name]["absolute_path"] = str(altered)
    with pytest.raises(ParentBindingError, match=pattern):
        parent_module._cross_component_compatibility(changed)


@pytest.mark.parametrize(
    "field",
    [
        "inherit_actor_optimizer",
        "inherit_critic_optimizer",
        "inherit_scheduler",
        "inherit_rng",
        "inherit_sampler",
        "instantiated_in_this_round",
    ],
)
def test_inherited_optimizer_rng_sampler_or_instantiation_is_rejected(
    binding: dict, field: str,
) -> None:
    changed = deepcopy(binding)
    changed["optimizer_policy"][field] = True
    with pytest.raises(ParentBindingError, match="STAGE3_PARENT_SCHEMA"):
        validate_parent_binding_semantics(changed)


def test_initial_actor_freeze_and_q_guidance_lock(binding: dict, real_preflight: dict) -> None:
    safety = binding["initial_safety_state"]
    assert safety == {
        "initial_actor_update_enabled": False,
        "initial_actor_q_guidance_enabled": False,
        "critic_warmup_required": True,
        "critic_ready": False,
        "unlock_requires_independent_critic_gate": True,
    }
    assert real_preflight["safety"] == {
        "INITIAL_ACTOR_UPDATE_ENABLED": False,
        "CRITIC_WARMUP_REQUIRED": True,
        "CRITIC_READY": False,
        "ACTOR_Q_GUIDANCE_ENABLED": False,
    }


def test_real_cpu_preflight_is_complete_for_hybrid_and_does_not_initialize_cuda(
    real_preflight: dict,
) -> None:
    assert real_preflight["tool_status"] == "PASS"
    assert real_preflight["PARENT_PAYLOAD_COMPLETE_FOR_HYBRID"] is True
    assert real_preflight["ACTOR_METADATA_COMPATIBILITY"] == "PASS"
    assert real_preflight["CRITIC_CPU_STATE_COMPATIBILITY"] == "PASS"
    assert real_preflight["TARGET_CRITIC_CPU_STATE_COMPATIBILITY"] == "PASS"
    assert real_preflight["CROSS_COMPONENT_CONTRACT_COMPATIBILITY"] == "PASS"
    assert real_preflight["CUDA_INITIALIZED"] is False
    assert real_preflight["REAL_MODEL_FORWARD"] == "NOT_RUN"
    assert real_preflight["optimizer"]["CROSS_STAGE_OPTIMIZER_REBUILT"] == "NOT_RUN"
    assert real_preflight["ROBOT_COMMAND_COUNT"] == 0


def test_parent_module_and_cli_have_no_ros_robot_serve_deploy_or_network_imports() -> None:
    banned = {
        "rclpy", "rospy", "roslib", "franka", "franka_msgs", "moveit",
        "requests", "httpx", "socket", "subprocess",
    }
    violations = []
    for path in (
        ROOT / "src/forcesmolvla/rft/stage3/parent.py",
        ROOT / "tools/preflight_stage3_parent.py",
    ):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module]
            for name in names:
                if name.split(".", 1)[0] in banned:
                    violations.append((path.name, name))
    assert violations == []
    source = (ROOT / "tools/preflight_stage3_parent.py").read_text(encoding="utf-8")
    assert "serve_policy" not in source and "deploy_forcesmolvla" not in source


def test_binding_config_file_sha_is_stable_and_schema_is_draft_2020_12() -> None:
    schema = json.loads((ROOT / "schemas/stage3_parent_binding.v1.schema.json").read_text())
    assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
    assert len(hashlib.sha256(DEFAULT_CONFIG.read_bytes()).hexdigest()) == 64
