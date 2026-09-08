---
type: "query"
date: "2026-09-08T06:54:08.980985+00:00"
question: "优化 ForceRFT admission 持久化：保留文件格式和文件 fsync，按 WAL、outbox、replay 分阶段集中目录 fsync，并用同一封存 episode 对比"
contributor: "graphify"
outcome: "useful"
source_nodes: ["ProductionBridge", "_immutable_write", "_fsync_directory"]
---

# Q: 优化 ForceRFT admission 持久化：保留文件格式和文件 fsync，按 WAL、outbox、replay 分阶段集中目录 fsync，并用同一封存 episode 对比

## Answer

Expanded from original query via graph vocab: [admission, persistence, immutable, write, fsync, directory, wal, outbox, replay, transition, episode, seal]. ProductionBridge formal admission now writes all WAL entries then fsyncs wal once, all outbox entries then fsyncs outbox once, all replay entries then fsyncs replay once, and commits the episode seal last. _immutable_write keeps per-file fsync and defaults to its prior per-file directory sync behavior for all other callers. Recovery without a seal re-syncs every dependency directory even when files are idempotently present. Episode 043 with 436 transitions and 1310 byte-identical JSON files benchmarked at 13.613s median old vs 7.165s median batched, 1.90x and 47.4 percent reduction. Bridge tests 83 passed; full CPU suite 565 passed and the same 7 unrelated or evidence-blocked failures remained.

## Outcome

- Signal: useful

## Source Nodes

- ProductionBridge
- _immutable_write
- _fsync_directory