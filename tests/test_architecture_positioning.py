from pathlib import Path


ROOT = Path(__file__).parents[1]


def test_v42_freezes_inheritance_innovation_and_online_scope():
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    required = (
        "ForcePrior SFT",
        "自主 proposal Critic warm-up",
        "持续在线 Actor/Learner",
        "1000 条有效自主 policy TD",
        "2 Twin-Q + 1 wrist-wrench residual Actor",
    )
    for statement in required:
        assert statement in text


def test_readme_describes_development_online_scope_without_production_claim():
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    normalized = " ".join(text.split())
    assert "实际 adapter、filter、leash、workspace、力/力矩限制与 ACK 链均保持" in normalized
    assert "它不是自主成功率评估" in normalized
    assert "CPU 测试只验证调度与并发语义" in normalized
