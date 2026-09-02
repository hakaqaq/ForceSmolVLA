---
type: "query"
date: "2026-09-01T12:57:34.824504+00:00"
question: "ONLINE_REPLAY_ASYNC_LEARNER_COMPLETION_CONTRACT 怎么又报错了"
contributor: "graphify"
outcome: "useful"
source_nodes: ["ContinuousLearner", "._start_learner()", ".end_episode()", "_run_episode()"]
---

# Q: ONLINE_REPLAY_ASYNC_LEARNER_COMPLETION_CONTRACT 怎么又报错了

## Answer

Expanded from original query via graph vocab: [async, learner, contract, replay, online, episode, end, abort, cycle, status, error, completed]. Root cause: ContinuousLearner snapshots every replay JSON before an optimizer cycle and requires the whole replay directory to be byte-metadata identical afterward. The persistent learner has one cycle credit per admitted transition and continues training while admission appends replay files. Episode _003 admission wrote its record at 20:52:33, appended 420 replay entries through 20:52:45.981, then published the committed episode marker at 20:52:45.991. A learner cycle overlapped those append-only writes, so replay_after != replay_before and raised ONLINE_REPLAY_ASYNC_LEARNER_COMPLETION_CONTRACT. This immutability assertion conflicts with ConRFT-style online insertion; the cycle already samples an in-memory replay snapshot, so new committed rows should be picked up on a later cycle instead of failing. The episode-abort 422 is secondary because episode-end had already made the episode inactive. The outer loop deleted unsealed _004. _001 and _003 remain admitted with 865 replay rows; no online checkpoint exists, so exact resume remains offline cycle 210.

## Outcome

- Signal: useful

## Source Nodes

- ContinuousLearner
- ._start_learner()
- .end_episode()
- _run_episode()