from pathlib import Path

import yaml


ROOT = Path(__file__).parents[1]


def test_one_pass_authorization_is_bounded_and_keeps_half_pass_checkpoint() -> None:
    value = yaml.safe_load(
        (ROOT / "configs/stage2b_long_run_one_pass_throughput_v2.authorized.yaml").read_text()
    )
    assert value["authorization"] == "yes_for_420_total_cycles"
    assert value["recipe"]["target_cycles"] == 420
    assert value["recipe"]["continuation_start_cycle"] == 210
    assert value["recipe"]["continuation_cycles"] == 210
    assert value["recipe"]["actor_transition_exposure"] == 10080
    assert value["recipe"]["td_row_membership"] == 53760
    assert value["recipe"]["calql_row_membership"] == 53760
    assert value["boundaries"]["auto_continue_after_cycle210_safety_gate"] is True
    assert value["boundaries"]["auto_continue_beyond_cycle420"] is False
    assert value["boundaries"]["rollout_authorized"] is False
    assert value["boundaries"]["robot_execution_authorized"] is False
    assert value["recipe"]["flow_inference_subbatch"] == 4
    assert value["runtime"]["decoded_cache_max_bytes"] == 8 * 1024**3
