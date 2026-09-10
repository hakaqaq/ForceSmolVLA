from pathlib import Path


ROOT = Path(__file__).parents[1]


def test_v42_freezes_inheritance_innovation_and_online_scope():
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    required = (
        "ForcePrior SFT",
        "Autonomous Proposal Critic Warm-Up",
        "Continuous Online Actor/Learner",
        "1,000 valid autonomous policy TD transitions",
        "2 Twin-Q + 1 wrist-wrench residual Actor",
    )
    for statement in required:
        assert statement in text


def test_readme_describes_development_online_scope_without_production_claim():
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    normalized = " ".join(text.split())
    assert "actual adapter, filter, leash, workspace limits, force/torque limits, and ACK chain remain active" in normalized
    assert "This probe is not an autonomous-success evaluation" in normalized
    assert "CPU tests validate scheduling and concurrency semantics only" in normalized
