---
type: "query"
date: "2026-09-02T10:40:07.592005+00:00"
question: "Remaining minimal fix: guarantee one FM-eligible row per Actor update"
contributor: "graphify"
outcome: "useful"
source_nodes: ["JointDemoReplay", "make_schedules()", "restore_rng_states_last()"]
---

# Q: Remaining minimal fix: guarantee one FM-eligible row per Actor update

## Answer

Expanded from the request via graph vocab: [joint, demo, replay, population, schedules, actor, eligible, failure, human, rng, restore, sampling]. DFS located JointDemoReplay and make_schedules in train_forcerft_actor_critic.py, with online call sites in actor_learner_runtime.py and serve_forcerft_actor_learner.py and RNG restore coverage. The minimal fix adds a read-only fm_population and repairs only all-non-FM Actor D-side draws using the restored d_rng; Critic sampling remains over full d_population.

## Outcome

- Signal: useful

## Source Nodes

- JointDemoReplay
- make_schedules()
- restore_rng_states_last()