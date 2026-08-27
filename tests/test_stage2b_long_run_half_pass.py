from pathlib import Path

import yaml

from forcesmolvla.rft.g7_long_run import counters_for_cycle


ROOT = Path(__file__).parents[1]


def test_half_pass_contract_is_bounded_and_uses_g7a_parent() -> None:
    config = yaml.safe_load(
        (ROOT / "configs/stage2b_long_run_half_pass.development.yaml").read_text()
    )
    assert config["authorization"] == "yes_for_210_cycles_only"
    assert config["parent"]["path"].endswith("g7a_r2_critic_warmup_checkpoint")
    assert config["parent"]["g7b_smoke_checkpoint_used_as_parent"] is False
    assert config["recipe"] == {
        "mode": "frozen-backbone_value-guided_force-action_refinement",
        "joint_cycles": 210,
        "critic_updates_per_cycle": 2,
        "actor_updates_per_cycle": 1,
        "expected_critic_updates": 420,
        "expected_actor_updates": 210,
        "expected_polyak_updates_per_target": 420,
        "actor_transition_exposure": 5040,
        "critic_transition_exposure": 53760,
        "actor_transition_passes": 0.5002481389578164,
        "projected_runtime_hours": 7.356627534725218,
        "projection_cycles_per_hour": 28.545688769581542,
    }
    assert config["batching"]["actor_physical_batch_size"] == 24
    assert config["batching"]["critic_physical_batch_size"] == 128
    assert config["loss"]["eta_actor_q"] == 3.0
    assert config["diagnostics"]["validation_cycles"] == [0, 105, 210]
    assert config["stop_after"]["additional_long_run_authorized"] is False
    assert counters_for_cycle(210)["critic_optimizer_updates"] == 420
    assert counters_for_cycle(210)["actor_optimizer_updates"] == 210
