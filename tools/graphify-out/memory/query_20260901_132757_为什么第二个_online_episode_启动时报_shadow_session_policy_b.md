---
type: "query"
date: "2026-09-01T13:27:57.039042+00:00"
question: "为什么第二个 online episode 启动时报 SHADOW_SESSION_POLICY_BINDING_MISMATCH 和 recorder KeyboardInterrupt"
contributor: "graphify"
outcome: "corrected"
correction: "The mismatch was caused by a two-phase session.json write race, not stale Actor revision metadata."
source_nodes: ["AsyncPolicyLearnerRuntime", ".metadata()", ".prepare_episode()"]
---

# Q: 为什么第二个 online episode 启动时报 SHADOW_SESSION_POLICY_BINDING_MISMATCH 和 recorder KeyboardInterrupt

## Answer

Expanded from graph vocabulary: [active, actor, binding, episode, learner, metadata, model, policy, prepare, revision, runtime, session]. DFS first suggested tracing AsyncPolicyLearnerRuntime metadata/prepare bindings, but direct source and captured artifacts corrected the initial revision-drift hypothesis. The policy hashes and tool/task bindings matched. Root cause: record_franka_hilserl_impedance wraps the base recorder manifest writer; the base writer creates session.json first, then the HIL-SERL wrapper enriches controller and workspace and rewrites it. IntegratedCaptureBackend waited only for file existence and intermittently validated the base manifest between those writes, then stopped the recorder, producing the secondary KeyboardInterrupt. Fix waits until controller and workspace exist before applying strict binding validation. Regression: 37 targeted tests and 201 online tests passed (one unrelated external source-hash audit deselected). _006 is admitted with replay total 1325 and exact resume is online cycle 8.

## Outcome

- Signal: corrected
- Correction: The mismatch was caused by a two-phase session.json write race, not stale Actor revision metadata.

## Source Nodes

- AsyncPolicyLearnerRuntime
- .metadata()
- .prepare_episode()