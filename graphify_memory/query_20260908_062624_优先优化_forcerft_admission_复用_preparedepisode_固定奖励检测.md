---
type: "query"
date: "2026-09-08T06:26:24.283691+00:00"
question: "优先优化 ForceRFT admission：复用 PreparedEpisode、固定奖励检测 batch 并预热、增加阶段计时、修正 training_started 日志"
contributor: "graphify"
outcome: "useful"
source_nodes: ["ProductionBridge", "OneShotFrozenRewardDetector", "prepare_episode"]
---

# Q: 优先优化 ForceRFT admission：复用 PreparedEpisode、固定奖励检测 batch 并预热、增加阶段计时、修正 training_started 日志

## Answer

Expanded from original query via vocab: [admission, prepare, episode, reward, detector, replay, transition, persist, write, batch, training, worker]. ProductionBridge formal admission now reuses the PreparedEpisode produced by integrated validation for frozen reward materialization and reports data_preparation, reward_detection, transition_build, persistence, and total timings. The persistent detector warms a fixed 128-frame JAX shape before READY; all batches infer at 128, tail images are decoded only once and padded in memory, and predictions are sliced to the valid count. Online logging now reads training_starts_reached. WAL/outbox/replay persistence semantics remain unchanged pending live timing. Targeted tests: 114 passed. Full CPU suite: 563 passed, 7 unrelated baseline failures.

## Outcome

- Signal: useful

## Source Nodes

- ProductionBridge
- OneShotFrozenRewardDetector
- prepare_episode