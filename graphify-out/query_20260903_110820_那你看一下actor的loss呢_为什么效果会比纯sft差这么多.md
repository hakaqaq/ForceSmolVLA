---
type: "query"
date: "2026-09-03T11:08:20.662678+00:00"
question: "那你看一下Actor的loss呢，为什么效果会比纯sft差这么多"
contributor: "graphify"
outcome: "useful"
source_nodes: ["actor_step", "critic_step", "compute_online_actor_objective", "compute_online_twin_q_td_loss", "train_forcerft_actor_critic.py", "forcerft_actor_critic_training.development.yaml"]
---

# Q: 那你看一下Actor的loss呢，为什么效果会比纯sft差这么多

## Answer

Expanded via graph vocab: actor, loss, critic, reward, transition, offline, gradient, value, metrics, cycle, sft, training. Task3 Actor210 的 critic TD loss 下降，但 actor FM loss 与 actor min-Q loss 均近乎横盘；同帧验证 FM 与 SFT 持平。主要问题是 38/9117 的稀疏终奖、全成功示范无失败动作、联合阶段 pure TD 且无 CalQL/CQL/MC-return、离线示范行的 behavior anchor 实际为零。Critic 对非终止样本 MC return 明显低估，Actor 动作 minQ 仍低于示范动作，210 次更新未产生可测 Q 改善，却用 fresh AdamW、1e-5 学习率更新约1.554亿参数。因此奖励器不是首要证据，当前 Actor recipe 和 Critic action ranking 更可疑。

## Outcome

- Signal: useful

## Source Nodes

- actor_step
- critic_step
- compute_online_actor_objective
- compute_online_twin_q_td_loss
- train_forcerft_actor_critic.py
- forcerft_actor_critic_training.development.yaml