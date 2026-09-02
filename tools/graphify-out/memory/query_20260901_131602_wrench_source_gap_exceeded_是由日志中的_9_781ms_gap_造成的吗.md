---
type: "query"
date: "2026-09-01T13:16:02.591963+00:00"
question: "WRENCH_SOURCE_GAP_EXCEEDED 是由日志中的 9.781ms gap 造成的吗，在线循环应如何修复并与 ConRFT 行为一致"
contributor: "graphify"
outcome: "useful"
source_nodes: ["run_forcerft_production_bridge.py", ".prepare_episode()", "AsyncPolicyLearnerRuntime"]
---

# Q: WRENCH_SOURCE_GAP_EXCEEDED 是由日志中的 9.781ms gap 造成的吗，在线循环应如何修复并与 ConRFT 行为一致

## Answer

Expanded from graph vocabulary: [bridge, causal, contract, episode, lerobot, prepare, production, raw, runtime, source, warmup]. DFS traced the production bridge and episode preparation path. The saved episode contained exactly one wrench source gap above contract: 9.780736 ms at raw line 12194, matching the live warning. The frozen 9 ms contract correctly rejects this episode, so the minimal safe fix preserves the threshold and replay integrity but turns this exact materialization failure into structured FORMAL_ONLINE_R_REJECTED. The persistent online loop now skips that episode and continues to the next capture until max admitted episodes. Replaying _004 returned exit code 0 with rejection and replay remained 865; online regression passed 200 tests with one unrelated external source-hash audit deselected.

## Outcome

- Signal: useful

## Source Nodes

- run_forcerft_production_bridge.py
- .prepare_episode()
- AsyncPolicyLearnerRuntime