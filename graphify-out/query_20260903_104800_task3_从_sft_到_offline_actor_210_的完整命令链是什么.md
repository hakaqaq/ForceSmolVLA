---
type: "path"
date: "2026-09-03T10:48:00.149347+00:00"
question: "Task3 从 SFT 到 offline actor 210 的完整命令链是什么？"
contributor: "graphify"
outcome: "useful"
source_nodes: ["tools/train_forcesmolvla_sft.py", "tools/reward_classifier/train_reward_classifier.py", "tools/reward_classifier/calibrate_reward_detector.py", "tools/materialize_reward_transitions.py", "tools/train_twin_q_critic.py", "tools/train_forcerft_actor_critic.py"]
---

# Q: Task3 从 SFT 到 offline actor 210 的完整命令链是什么？

## Answer

Task3 链路：train_forcesmolvla_sft.py；label_reward_frames.py（16 train + 4 val）；train_reward_classifier.py prepare-cache；在 conrft_reward 环境 train；在 conrft_reward 环境 calibrate_reward_detector.py --approve；materialize_reward_transitions.py build；train_twin_q_critic.py --run；train_forcerft_actor_critic.py --offline-joint-cycles 210。当前路径为 outputs/task3、datasets/task3_lerobotv3、datasets/task3_forcerft_offline_reward_transitions，缓存为 .cache/task3_reward_classifier。

## Outcome

- Signal: useful

## Source Nodes

- tools/train_forcesmolvla_sft.py
- tools/reward_classifier/train_reward_classifier.py
- tools/reward_classifier/calibrate_reward_detector.py
- tools/materialize_reward_transitions.py
- tools/train_twin_q_critic.py
- tools/train_forcerft_actor_critic.py