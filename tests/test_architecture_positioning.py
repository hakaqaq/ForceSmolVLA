from pathlib import Path


ROOT = Path(__file__).parents[1]


def test_v42_freezes_inheritance_innovation_and_online_scope():
    text = (ROOT / "ForceSmolVLA_Implementation_Spec_v4_2.md").read_text(encoding="utf-8")
    required = (
        "post-VLM prefix 不包含与未来 `H` 个动作位置天然对齐的表示",
        "Action Expert 内部仍具有 `H` 个 action suffix hidden",
        "guidance 分支不显式以当前 noisy action、flow timestep 和 Action Expert hidden 为查询",
        "继承自 ForceVLA 的思想：post-VLM force fusion 与稀疏 MoE force refinement",
        "本项目的新结构：Action-Query Force Residual Adapter",
        "冻结 VLM 的在线 Actor–Critic 仅是后续能力",
        "不接收、不拼接或修改 SmolVLA 原生 `past_key_values`",
    )
    for statement in required:
        assert statement in text


def test_readme_describes_development_online_scope_without_production_claim():
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    normalized = " ".join(text.split())
    assert "真机采集与持续 Actor/Learner 循环" in text
    assert "不等同于 formal detector validation" in normalized
    assert "Action Expert 内部仍有 H 个 action suffix hidden" in normalized
