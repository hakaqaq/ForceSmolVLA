from pathlib import Path

import yaml


ROOT = Path(__file__).parents[1]


def test_authorized_half_pass_is_exactly_bounded() -> None:
    value = yaml.safe_load(
        (ROOT / "configs/stage2b_long_run_half_pass_throughput_v2.authorized.yaml").read_text()
    )
    assert value["authorization"] == "yes_for_210_cycles_only"
    assert value["recipe"] == {
        **value["recipe"],
        "target_cycles": 210,
        "actor_batch": 24,
        "critic_batch": 64,
        "flow_inference_subbatch": 4,
    }
    assert value["boundaries"]["auto_continue_to_1_pass"] is False
    assert not any(value["parent"][name] for name in (
        "old_cycle105_allowed", "old_interrupted_pilot_allowed",
        "exact_resume_temporary_checkpoint_allowed",
    ))
