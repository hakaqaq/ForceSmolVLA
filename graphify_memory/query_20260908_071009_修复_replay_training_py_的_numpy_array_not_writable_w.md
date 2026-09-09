---
type: "query"
date: "2026-09-08T07:10:09.637754+00:00"
question: "修复 replay_training.py 的 NumPy array not writable warning，并核对 human 相关在线计数的统计域"
contributor: "graphify"
outcome: "useful"
source_nodes: ["replay_training.py", "FrozenFeatureNormalizer", "integrated_capture.py", "production_bridge.py"]
---

# Q: 修复 replay_training.py 的 NumPy array not writable warning，并核对 human 相关在线计数的统计域

## Answer

Expanded query: [numpy, array, torch, tensor, mean, normalizer, replay, human, residual, rows, intervention, ack]. FrozenFeatureNormalizer 将 mean/std 设为只读，而 OnlineResidualReplay._batch 使用 torch.as_tensor 共享 NumPy 内存并触发 warning。最小修复是对 mean6/std6 使用 torch.tensor 显式复制，保留 dtype/device/expand 和训练语义；新增只读 normalizer 回归测试。policy-execute 的 human_action_ack_count 按模式定义为 0，policy_action_ack_count 是该模式的 actual ACK 数；intervention_count 是当前采集 episode 的接管事件数；human_override_replay_count 是当前 admission 生成的人类 transition 数；human_residual_valid_rows 是当前加载 replay population 的累计有效人类 residual row 数。未发现计数逻辑错误。

## Outcome

- Signal: useful

## Source Nodes

- replay_training.py
- FrozenFeatureNormalizer
- integrated_capture.py
- production_bridge.py