---
type: "query"
date: "2026-09-01T13:03:58.289671+00:00"
question: "修复 ONLINE_REPLAY_ASYNC_LEARNER_COMPLETION_CONTRACT，并与 ConRFT 在线 append 行为一致"
contributor: "graphify"
outcome: "useful"
source_nodes: ["ContinuousLearner", "._start_learner()", "AsyncPolicyLearnerRuntime"]
---

# Q: 修复 ONLINE_REPLAY_ASYNC_LEARNER_COMPLETION_CONTRACT，并与 ConRFT 在线 append 行为一致

## Answer

Expanded async learner/replay/episode/training nodes. Root cause: ContinuousLearner compared the live replay directory before and after an optimizer cycle, so expected append-only Online-R admission during that cycle falsely violated completion. Minimal fix removes global replay-directory equality while keeping current-episode sampling, non-finite, and OOM completion guards. Each cycle continues to train from its owned in-memory replay snapshot; newly admitted rows are visible next cycle, matching ConRFT-style behavior. Target tests passed 20/20; online tests passed 198 with one unrelated external source-hash audit deselected.

## Outcome

- Signal: useful

## Source Nodes

- ContinuousLearner
- ._start_learner()
- AsyncPolicyLearnerRuntime