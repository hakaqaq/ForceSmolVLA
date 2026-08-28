from __future__ import annotations

from pathlib import Path

import torch

from forcesmolvla.rft.throughput_v2 import (
    canonical_observation_identity,
    concat_actor_batches,
    fast_polyak_update,
    index_actor_batch,
    lightweight_state_token,
)


ROOT = Path(__file__).parents[1]


def test_batch_index_and_concat_preserve_order() -> None:
    batch = {
        "x": torch.arange(12).reshape(3, 4),
        "sample_identity": ("a", "b", "c"),
        "constant": "task",
    }
    selected = index_actor_batch(batch, [2, 0])
    assert selected["sample_identity"] == ("c", "a")
    assert selected["x"].tolist() == [[8, 9, 10, 11], [0, 1, 2, 3]]
    joined = concat_actor_batches((selected, index_actor_batch(batch, [1])))
    assert joined["sample_identity"] == ("c", "a", "b")
    assert joined["x"].shape == (3, 4)


def test_grouped_flow_identity_only_removes_candidate_spelling() -> None:
    assert canonical_observation_identity("episode/frame=12/cql_current=0") == "episode/frame=12"
    assert canonical_observation_identity("episode/frame=12/cql_current=1") == "episode/frame=12"
    assert canonical_observation_identity("episode/next=12/cql_next=1") == "episode/frame=12"
    assert canonical_observation_identity("episode/frame=13") != "episode/frame=12"


def test_fast_polyak_has_exact_formula_and_target_ownership() -> None:
    online = torch.nn.Linear(3, 2).float().eval()
    target = torch.nn.Linear(3, 2).float().eval()
    target.load_state_dict(online.state_dict())
    for value in target.parameters():
        value.requires_grad_(False)
    before = {name: value.detach().clone() for name, value in target.named_parameters()}
    with torch.no_grad():
        for value in online.parameters():
            value.add_(0.25)
    report = fast_polyak_update(online, target, tau=0.005, target_name="target")
    assert report["boundary_exact_audit_required"] is True
    for name, value in target.named_parameters():
        expected = before[name].mul(0.995).add(dict(online.named_parameters())[name], alpha=0.005)
        assert torch.equal(value, expected)
        assert value.requires_grad is False
    assert target.training is False


def test_lightweight_token_detects_mutation_without_tensor_copy() -> None:
    module = torch.nn.Linear(2, 1)
    before = lightweight_state_token(module)
    with torch.no_grad():
        module.weight.add_(1)
    assert lightweight_state_token(module) != before


def test_throughput_config_is_benchmark_only_and_semantics_frozen() -> None:
    import yaml

    config = yaml.safe_load((ROOT / "configs/stage2_throughput_v2.development.yaml").read_text())
    assert config["authorization"] == "benchmark_only_no_training_checkpoint"
    assert config["fixed"] == {
        "actor_physical_batch_size": 24,
        "critic_physical_batch_size": 128,
        "critic_updates_per_actor_update": 2,
        "horizon": 50,
        "flow_euler_steps": 10,
        "calql_candidates_per_source": 2,
        "eta": 3.0,
        "beta": 1.0,
        "calql_alpha": 0.1,
        "polyak_tau": 0.005,
        "deterministic_algorithms": True,
        "actor_trainability": "frozen_vlm_force_action_trainable",
    }
    assert [item["id"] for item in config["benchmark"]["candidates"]] == [
        "baseline_current_implementation",
        "candidate_A_async_data_pipeline",
        "candidate_B_prefix_cache",
        "candidate_C_flow_subbatch_8",
        "candidate_D_flow_subbatch_16",
        "candidate_E_grouped_td_calql_flow",
    ]
    assert config["limits"] == {
        "long_run": False,
        "checkpoint_created": False,
        "policy_evaluation": False,
        "deployment": False,
        "robot_execution": False,
    }


def test_candidate_e_cannot_be_silently_aliased_to_candidate_d() -> None:
    import yaml

    config = yaml.safe_load((ROOT / "configs/stage2_throughput_v2.development.yaml").read_text())
    candidates = {item["id"]: item for item in config["benchmark"]["candidates"]}
    assert candidates["candidate_D_flow_subbatch_16"]["grouped_td_calql_flow"] is False
    assert candidates["candidate_E_grouped_td_calql_flow"]["grouped_td_calql_flow"] is True
    source = (ROOT / "tools/benchmark_stage2_throughput_v2_gpu.py").read_text()
    assert "grouped_td_calql_flow" in source


def test_original_artifacts_are_not_output_targets() -> None:
    source = (ROOT / "tools/benchmark_stage2_throughput_v2_gpu.py").read_text()
    assert "artifacts/development/stage2/throughput_v2" in source
    assert "g7b_joint_smoke_checkpoint" not in source
    assert "stage2b_long_run_half_pass_checkpoints" not in source
