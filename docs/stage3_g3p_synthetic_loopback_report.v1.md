# Stage-3 G3P provisional synthetic loopback evidence

## Scope and verdict

This evidence freezes the accepted CPU-only G3P synthetic Actor/Learner loopback.
It is a synthetic tool test, not the recorded-live G3 gate and not authorization
for a production, GPU, network, ROS, or robot path.

```text
fixture_kind=synthetic_tool_test
formal_gate_passed=false
recorded_live_fixture_available=false
robot_execution_authorized=false

training_starts_unique_R=100
mixed_replay_R_D_ratio=50_50
critic_gradient_steps=2
actor_gradient_steps=1
target_polyak_updates=2
calql_online_call_count=0
policy_revision_staged=true
policy_revision_activated=false

canonical_report_digest=63cb791dfb9f12ed461283772bbba80c8254fa04b6ad13405f197c8b999e9b8e
source_head_before_commit=53a0fcfbffaed0c7f042367cf5ad4b7fe39dcf6f
```

The schema-valid machine-readable evidence is
`artifacts/development/stage3/stage3_g3p_synthetic_loopback.v1.json`.
Its file SHA-256 before the freeze commit is
`74b3a940fd87b9a7c897133dceadf6ab4772704ce33b4ebd3c28ae41992c7148`.
The canonical digest excludes only the `evidence_freeze` metadata object, so the
accepted tool-run identity remains the reviewed `63cb...e9b8e` value.

## Reproduction

The isolated collection command completed with exit code 0 and collected 48
Stage-3 test nodes:

```bash
CUDA_VISIBLE_DEVICES='' \
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
PYTHONPATH=src:vendor/lerobot/src \
/home/rlc123/anaconda3/envs/forcesmolvla/bin/python \
-m pytest --collect-only -q tests/test_stage3_*.py
```

The isolated CPU test command completed with exit code 0:

```bash
CUDA_VISIBLE_DEVICES='' \
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
PYTHONPATH=src:vendor/lerobot/src \
/home/rlc123/anaconda3/envs/forcesmolvla/bin/python \
-m pytest -q tests/test_stage3_*.py
```

Result: `48 passed, 0 failed, 0 skipped`.

Two independent synthetic CLI executions with seed `20260828` produced
byte-identical schema-valid reports and the same canonical digest:

```text
63cb791dfb9f12ed461283772bbba80c8254fa04b6ad13405f197c8b999e9b8e
```

A CLI execution against an explicitly missing recorded-live fixture completed
with exit code 0 and generated a schema-valid report containing:

```text
tool_status=BLOCKED
blocked_reason=RECORDED_LIVE_FIXTURE_MISSING
formal_gate_passed=false
robot_execution_authorized=false
```

Exit code 0 means only that the CLI successfully generated a valid report.
Automation must inspect both `tool_status` and `formal_gate_passed`; it must not
interpret a BLOCKED report as a formal gate pass.

## Frozen semantic evidence

- The fake Actor produces deterministic `H=50` action chunks with request,
  policy revision, and observation bindings.
- The fake gateway uses the rational 30 Hz grid and fixed 10 Hz anchor phase,
  accepts `K=3` ACK-authoritative absolute7 slots, preserves gripper command/ACK
  identity, and applies the frozen delta-action normalizer exactly once.
- Human takeover invalidates the stale policy chunk. Partial macros and missing,
  rejected, or stale ACKs are quarantined and add zero replay commits.
- Replay unlocks only on the 100th unique online R transition, samples exact
  50/50 R/D, keeps intervention dual membership with one canonical payload, and
  fails closed on same-UID/different-digest conflicts.
- One joint cycle performs one critic-only update followed by one actor+critic
  update: two Critic gradient steps, one Actor gradient step, and two target-Q
  Polyak update rounds. Online Cal-QL, CQL penalty, random-candidate, and
  MC-return call counts are zero.
- A fake immutable policy revision is staged but never activated or published.

## Deferred production work

- The recorder-to-WAL/replay production bridge is not implemented.
- Durable production WAL/outbox is not implemented.
- `G0_FINAL_PARENT_BINDING=PENDING` remains unchanged.
- The tiny CPU test optimizer does not mean the cross-stage optimizer was
  rebuilt: `CROSS_STAGE_OPTIMIZER_REBUILT=NOT_RUN`.
- Reports under `/tmp` are transient reproduction outputs, not persistent
  acceptance evidence; only this document and the checked-in JSON are frozen.
- Recorded-live fixture parity and the formal G3 gate remain blocked.
- G4 and later phases, real publisher/server, GPU, checkpoint, ROS, and robot
  execution remain not run or unauthorized.
