from __future__ import annotations

import ast
from copy import deepcopy
import importlib
import json
import os
from pathlib import Path
import sys

from jsonschema import ValidationError
import pytest
import torch
import yaml


ROOT = Path(__file__).parents[1]
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))
gpu = importlib.import_module("preflight_stage3_gpu")


@pytest.fixture(scope="module")
def config() -> dict:
    return yaml.safe_load(
        (ROOT / "configs/stage3_gpu_preflight.v1.development.yaml").read_text(encoding="utf-8")
    )


def _minimal_report() -> dict:
    report = {
        "schema_version": "forcesmolvla_stage3_gpu_preflight_report.v1",
        "tool_status": "PASS",
        "preflight_only": True,
        "environment": {
            "cuda_device_index": 0,
            "gpu_name": "NVIDIA GeForce RTX 4090 D",
            "gpu_uuid": "GPU-test",
            "python_executable": "/test/python",
            "torch_version": "test",
            "torch_cuda_version": "test",
            "cudnn_version": 1,
            "initial_free_vram_mib": 1.0,
        },
        "parent_load": {
            "actor_strict_load": True,
            "critic_strict_load": True,
            "target_critic_strict_load": True,
            "sha_before_after_equal": True,
        },
        "data": {
            "critic_batch": {"R_count": 32, "D_count": 32},
            "actor_batch": {"R_count": 12, "D_count": 12},
            "normalizer_binding": {"sha256": "0" * 64},
            "task_feature_digest": "0" * 64,
            "writes_real_replay": False,
        },
        "optimizer_ownership": {
            "factory_validated": True,
            "actor_critic_parameter_id_intersection": 0,
            "frozen_parameters_in_optimizers": 0,
            "target_parameters_in_optimizers": 0,
            "fresh_initial_state_entries": 0,
            "each_trainable_parameter_exactly_one_owner": True,
        },
        "cycles": {
            "warmup_joint_cycles": 1,
            "measured_joint_cycles": 3,
            "critic_optimizer_steps": 8,
            "actor_optimizer_steps": 4,
            "target_polyak_steps": 8,
        },
        "numerics": {
            "all_finite": True,
            "frozen_hash_unchanged": True,
            "calql_online_call_count": 0,
            "gradient_ownership_passed": True,
        },
        "performance": {
            "load_only_vram_mib": 1.0,
            "warmup_peak_vram_mib": 1.0,
            "measured_peak_vram_mib": 1.0,
            "final_vram_mib": 0.0,
            "peak_cpu_rss_mib": 1.0,
            "oom_count": 0,
            "nonfinite_count": 0,
        },
        "safety": {
            "CRITIC_READY": False,
            "ACTOR_Q_GUIDANCE_ENABLED": False,
            "G0_FORMAL_GATE_PASSED": False,
            "G3_RECORDED_FIXTURE_LOOPBACK": "BLOCKED",
            "G5_AND_LATER": "NOT_RUN",
            "ROBOT_CONNECTION_COUNT": 0,
            "ROBOT_COMMAND_COUNT": 0,
        },
        "parent_checkpoint_mutated": False,
        "runtime_optimizer_state_persisted": False,
        "policy_revision_exported": False,
        "robot_execution_authorized": False,
        "evidence_freeze": {
            "G4P_RESULT": "PASS",
            "R_SOURCE": "synthetic_preflight_R_only",
            "REAL_ONLINE_R_USED": False,
            "PREFLIGHT_ACTOR_STEPS_DISPOSABLE": True,
            "PRODUCTION_ACTOR_STATE_MUTATED": False,
            "RUNTIME_OPTIMIZER_STATE_PERSISTED": False,
            "CRITIC_WARMUP_STARTED": False,
            "CRITIC_READY": False,
            "ACTOR_Q_GUIDANCE_ENABLED": False,
            "ETA_3_APPROVED": False,
            "GPU_COEXISTENCE_VALIDATED": False,
            "G5_AND_LATER": "NOT_RUN",
            "ROBOT_EXECUTION_AUTHORIZED": False,
        },
        "eta_gradient_diagnostic": {
            "source_fields": [
                "actor_updates[].gradient_geometry.weighted_q_norm",
                "actor_updates[].gradient_geometry.weighted_fm_norm",
            ],
            "preflight_eta": 0.1,
            "candidate_eta": 3.0,
            "candidate_approved": False,
            "per_cycle": [
                {
                    "cycle": cycle,
                    "weighted_q_norm": 0.1,
                    "weighted_fm_norm": 1.0,
                    "weighted_q_over_weighted_fm": 0.1,
                    "eta_3_linear_rescale_q_over_fm": 3.0,
                }
                for cycle in range(4)
            ],
            "statements": [
                "eta=3 remains a provisional numerical-preflight candidate.",
                "No eta calibration or Actor Q-guidance approval is granted by G4P.",
            ],
        },
    }
    report["canonical_report_sha256"] = gpu.canonical_report_sha256(report)
    return report


def test_config_freezes_exact_g4p_scope(config: dict) -> None:
    assert gpu.validate_gpu_preflight_config(config) == config
    assert config["parent_binding"] == {
        "path": "configs/stage3_parent_binding.v1.development.json",
        "binding_id": "approved_hybrid_cycle210_actor_g7a_r2_twin_q.v1",
        "binding_type": "new_hybrid_stage3_bootstrap",
        "actor_source": "cycle210_evaluation",
        "critic_source": "G7A-r2",
        "target_critic_source": "G7A-r2",
        "strict_load": True,
        "random_critic_fallback": False,
        "target_copy_fallback": False,
    }
    assert config["batching"] == {
        "critic_batch_size": 64,
        "critic_R_count": 32,
        "critic_D_count": 32,
        "actor_batch_size": 24,
        "actor_R_count": 12,
        "actor_D_count": 12,
        "flow_inference_subbatch": 4,
        "flow_horizon": 50,
        "flow_steps": 10,
        "critic_slots": 3,
        "action_features": 7,
    }


def test_config_rejects_batch_or_loss_semantic_reduction(config: dict) -> None:
    changed = deepcopy(config)
    changed["batching"]["actor_batch_size"] = 16
    with pytest.raises(gpu.G4PError, match="BATCH_OR_TOPOLOGY"):
        gpu.validate_gpu_preflight_config(changed)
    changed = deepcopy(config)
    changed["loss"]["calql_enabled"] = True
    with pytest.raises(gpu.G4PError, match="LOSS_SCOPE"):
        gpu.validate_gpu_preflight_config(changed)


def test_fixed_real_row_selection_is_nonoverlapping_and_pool_exact() -> None:
    rows = [
        {"executed_action_mask": [True, True, True], "terminated": False}
        for _ in range(100)
    ] + [{"executed_action_mask": [True, True, True], "terminated": True}]
    selected = gpu.select_fixed_indices(rows, seed=17)
    assert len(selected["critic_indices"]) == 64
    assert len(selected["actor_indices"]) == 24
    assert len(set(selected["critic_indices"] + selected["actor_indices"])) == 88
    assert selected["critic_indices"][-1] == 100
    assert selected["critic_origin_pool"].count("synthetic_preflight_R_only") == 32
    assert selected["critic_origin_pool"].count("offline_D") == 32
    assert selected["actor_origin_pool"].count("synthetic_preflight_R_only") == 12
    assert selected["actor_origin_pool"].count("offline_D") == 12


def test_report_schema_requires_safe_pass_evidence() -> None:
    report = _minimal_report()
    assert gpu.validate_report(report) == report
    changed = deepcopy(report)
    changed["runtime_optimizer_state_persisted"] = True
    changed["canonical_report_sha256"] = gpu.canonical_report_sha256(changed)
    with pytest.raises(ValidationError):
        gpu.validate_report(changed)
    changed = deepcopy(report)
    changed["cycles"]["actor_optimizer_steps"] = 3
    changed["canonical_report_sha256"] = gpu.canonical_report_sha256(changed)
    with pytest.raises(ValidationError):
        gpu.validate_report(changed)


def test_evidence_freeze_recomputes_each_cycle_gradient_ratio(config: dict) -> None:
    report = _minimal_report()
    report["data"]["critic_batch"]["R_source"] = "synthetic_preflight_R_only"
    report["data"]["actor_batch"]["R_source"] = "synthetic_preflight_R_only"
    report["actor_updates"] = [
        {
            "cycle": cycle,
            "gradient_geometry": {
                "weighted_q_norm": float(cycle + 1),
                "weighted_fm_norm": float(2 * (cycle + 1)),
            },
        }
        for cycle in range(4)
    ]
    frozen = gpu.freeze_g4p_evidence(report, config)
    assert frozen["evidence_freeze"]["ETA_3_APPROVED"] is False
    assert [
        item["weighted_q_over_weighted_fm"]
        for item in frozen["eta_gradient_diagnostic"]["per_cycle"]
    ] == [0.5] * 4
    assert [
        item["eta_3_linear_rescale_q_over_fm"]
        for item in frozen["eta_gradient_diagnostic"]["per_cycle"]
    ] == [15.0] * 4


def test_gpu_tool_uses_real_production_primitives_and_fail_closed_loading() -> None:
    source = (ROOT / "tools/preflight_stage3_gpu.py").read_text(encoding="utf-8")
    for symbol in (
        "apply_frozen_vlm_trainability",
        "frozen_prefix_flow_matching_terms",
        "compute_online_twin_q_td_loss",
        "compute_stage3_min_twin_q_actor_loss",
        "compute_stage3_actor_objective",
        "polyak_update_verified",
        "TrainData",
    ):
        assert symbol in source
    assert "strict=True" in source
    assert "weights_only=True" in source
    assert "map_location=\"cpu\"" in source
    assert "strict=False" not in source
    assert "copy.deepcopy(q1" not in source and "copy.deepcopy(q2" not in source


def test_gpu_tool_has_no_robot_ros_network_or_process_imports_and_import_is_cpu_only() -> None:
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
    assert not torch.cuda.is_initialized()
    path = ROOT / "tools/preflight_stage3_gpu.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    banned = {
        "rclpy", "rospy", "roslib", "franka", "franka_msgs", "moveit",
        "requests", "httpx", "socket", "subprocess",
    }
    violations = []
    for node in ast.walk(tree):
        names = []
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module:
            names = [node.module]
        violations.extend(name for name in names if name.split(".", 1)[0] in banned)
    assert violations == []
    source = path.read_text(encoding="utf-8")
    assert "serve_policy" not in source and "deploy_forcesmolvla" not in source
    assert "ROBOT_COMMAND_COUNT" in source and "ROBOT_CONNECTION_COUNT" in source
    assert not torch.cuda.is_initialized()


def test_schema_is_draft_2020_12_and_config_sha_is_recordable() -> None:
    schema = json.loads(
        (ROOT / "schemas/stage3_gpu_preflight_report.v1.schema.json").read_text(encoding="utf-8")
    )
    assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
    assert len(gpu.sha256_file(ROOT / "configs/stage3_gpu_preflight.v1.development.yaml")) == 64
