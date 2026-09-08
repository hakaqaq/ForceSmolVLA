---
type: "query"
date: "2026-09-08T05:58:06.915321+00:00"
question: "run_forcerft_online_loop.py 这条命令在线采集的数据和保存的 checkpoint 都在哪个路径"
contributor: "graphify"
outcome: "useful"
source_nodes: ["learner_checkpoint.py", "OnlineReplay", "training_cycle_runtime.py"]
---

# Q: run_forcerft_online_loop.py 这条命令在线采集的数据和保存的 checkpoint 都在哪个路径

## Answer

Expanded from original query via vocab: [online, capture, output, root, dataset, episode, checkpoint, training, bootstrap, residual, learner]. 原始在线采集位于 /home/rlc123/ForceSmolVLA/datasets/task3_forcerft_online/，每次采集使用三位编号目录；当前最新完整采集为 039，原生 episode 在 039/episodes/episode_000000，集成策略审计流在 039/integrated_capture/episode_000000。训练接收后的 formal replay 位于 /home/rlc123/ForceSmolVLA/outputs/task3/online_ack_residual_filter_leash/formal_replay/。输入 bootstrap 位于命令指定的 bootstrap_checkpoints/base_policy_zero_residual_filter_leash_random_twin_q。在线生成的 exact-resume checkpoint 位于 training_checkpoints/residual_actor_critic_cycle_NNNNNN；当前存在 cycle_000000 与 cycle_000009，最新为 cycle_000009。下次启动优先恢复最新可恢复 training checkpoint，只有没有训练 checkpoint 时才使用 bootstrap。

## Outcome

- Signal: useful

## Source Nodes

- learner_checkpoint.py
- OnlineReplay
- training_cycle_runtime.py