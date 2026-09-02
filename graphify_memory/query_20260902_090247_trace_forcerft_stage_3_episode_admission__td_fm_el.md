---
type: "query"
date: "2026-09-02T09:02:47.131261+00:00"
question: "Trace ForceRFT Stage-3 episode admission, TD/FM eligibility, and online Actor objective for the requested failure-replay and policy-anchor change"
contributor: "graphify"
outcome: "useful"
source_nodes: ["ProductionBridge", "admit_policy_execution_smoke", "load_formal_online_r", "build_batch", "actor_step", "compute_online_actor_objective"]
---

# Q: Trace ForceRFT Stage-3 episode admission, TD/FM eligibility, and online Actor objective for the requested failure-replay and policy-anchor change

## Answer

Expanded from original query via graph vocabulary: [admission, episode, outcome, replay, eligible, actor, loss, behavior, policy, terminal, materialize, bridge]. The path is ProductionBridge.admit_policy_execution_smoke -> formal online policy/human transition materialization -> replay_training.load_formal_online_r and build_batch -> JointDemoReplay/FormalReplay -> actor_step -> compute_online_actor_objective. Outcome/eligibility must be written at transition materialization, FM masking must be decided independently in replay materialization, and the existing Q action returned by compute_online_min_twin_q_actor_loss is the direct anchor input.

## Outcome

- Signal: useful

## Source Nodes

- ProductionBridge
- admit_policy_execution_smoke
- load_formal_online_r
- build_batch
- actor_step
- compute_online_actor_objective