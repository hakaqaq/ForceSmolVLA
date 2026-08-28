# Stage‑3 G6P isolated immutable policy-revision lifecycle loopback

## Scope and result

This report covers only:

```text
G6P = isolated immutable policy-revision lifecycle loopback
G7_AND_LATER = NOT_RUN
```

The implemented path is CPU-only synthetic filesystem evidence. It does not export a real
Actor, start a policy server, publish a production revision, verify a robot Home state, or
verify a production WAL/outbox.

- Result: `PASS`.
- Canonical report SHA256: `d597ef3631a580e4cc8e67e00d7dacf4190de14ba830760cfe5c2e7225e80fd6`.
- Canonical JSON artifact:
  `artifacts/development/stage3/stage3_policy_revision_loopback.v1.json`.
- Two independent final `/tmp` CLI runs produced byte-identical JSON and the same digest.

## Baseline gate

- Branch: `stage3-online-hil`.
- HEAD: `0698f7635f479a36794c349ce0e1e77a3a26bc2d`.
- Recomputed G5P report canonical SHA256:
  `75c3b0bab63b17bc0b4a685cd1a2177d7194fc82a1b2fd2fb112bc268210fdad`.
- Recorded G5P checkpoint canonical digest:
  `b0d24880e02f0eff3f18f22930b3fe8bbc1ebd8f9cfa9da825d27a08533d1058`.
- The protected 2.57 GB G5P learner checkpoint was not read, copied, moved, or deleted.
- Baseline Stage‑3 tests: `99 collected`, `99 passed`, `0 failed`, `0 skipped`.
- Final Stage‑3 tests after G6P: `117 collected`, `117 passed`, `0 failed`, `0 skipped`.
- Baseline worktree contained no changes outside the pre-existing untracked
  `graphify-out/` and `src/graphify-out/` trees.

The first raw pytest collection attempt was stopped by an auto-discovered system
`launch_testing` plugin before repository tests were collected. All authoritative baseline
and G6P runs used `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`, preventing ROS pytest plugins from
being imported by the test process.

## Real repository interface audit

The implementation was selected only after directly inspecting the current HEAD.

| File | Real symbol / finding |
|---|---|
| `src/forcesmolvla/rft/stage3/publication.py` | Existing `RevisionState`, `RevisionRecord`, `QuiescentBoundary`, and `InMemoryRevisionStateMachine` were the compatible lifecycle primitives. `register_candidate`, `stage`, `activate_pending`, episode pinning, `rollback`, and `record` existed but had no durable serialization or immutable filesystem exporter. |
| `src/forcesmolvla/rft/stage3/protocol.py` | `TransportEnvelope` already carried revision/model/request/chunk identities. `PolicyEpochGate.classify_result` already stale-dropped old epoch or revision, but did not pin active model/request/chunk. |
| `src/forcesmolvla/rft/stage3/transition.py` | `validate_ack_transition` and `finalize_ack_transition` bound the policy proposal revision/model/epoch/request/chunk. Current/next observation and ACK-ledger revision equality were not represented by a shared validator. |
| `src/forcesmolvla/rft/stage3/checkpoint.py` | `validate_online_checkpoint_metadata` and `cpu_round_trip_online_checkpoint` already round-tripped active/pending/previous/policy_epoch and publication count. The separate G5P exact-resume path intentionally rejects pending revision state and was left unchanged. |
| `configs/stage3_policy_publication.v1.development.json` | Existing in-memory-only G2 contract already required reset/Home, sealed WAL, no episode/request/queue/ACK, epoch invalidation, and rollback boundary semantics. |
| `tests/test_stage3_protocol_and_publication.py` | Existing tests fixed backward-compatible constructor/method expectations and confirmed the old G2 primitive must be extended rather than replaced. |

No second lifecycle state machine was added. The same
`InMemoryRevisionStateMachine` now owns lifecycle transitions, snapshot serialization,
recovery validation, publication counters, rollback audit state, and recovery reset gating.

## Immutable revision artifact

`export_immutable_revision` writes a deterministic tiny model payload and `bindings.json`
inside a same-filesystem temporary directory. It computes per-file SHA256 records, model
tree SHA256, and the canonical artifact identity. `revision_id` is the SHA256 of canonical
artifact content; it contains no timestamp, PID, path, or random UUID. The exporter then:

1. writes payload and binding files with file fsync;
2. writes the canonical manifest;
3. fsyncs the directory tree;
4. writes `COMPLETED.json` last;
5. fsyncs again;
6. removes write permission from the complete tree;
7. atomically renames the temporary directory to the revision ID; and
8. fsyncs the revision-store parent.

Same-ID/same-digest export is an idempotent no-op. A same-ID/different-digest target fails
closed as `STAGE3_REVISION_ID_DIGEST_COLLISION`. Lifecycle state is stored only in the
standalone atomically replaced registry, never by modifying an immutable revision manifest.

The final recovered registry contains:

| State | Revision ID |
|---|---|
| active | `8abaf0264bc54917741e52d185e5424a7e2c04d5127a64184b86deea404a3a2f` |
| pending | `425c85edb1cd2964fba646e0149b73026ec2a1ade3ed97b7d717ba3e223a6216` |
| previous | `a9eef400e873f072fc6930d9441cfbede5fe39b0cfe45a9212758189701f369e` |
| rolled_back | `c2434f987b7966c1507b4c4d944467772ee6f648905919a770ea9cef39122cee` |
| rejected | `153fbc9b21b78e1fbdb698adfe58d39b68c05ab77b40d63d8405ef5bd2873c7e` |

Publication counters recovered exactly: 4 candidate registrations, 3 pending
publications, 2 explicit simulated activations, 1 rejection, 1 rollback, and 2 explicit
epoch invalidations. Final `policy_epoch=5` is distinct from every revision ID.

## Lifecycle, quiescence, stale-drop, and rollback

- A candidate was exported and fully validated during an active episode, then entered
  pending while active revision/model/epoch remained pinned to the episode start.
- Request, result, chunk, proposal, ACK ledger, current observation, next observation,
  and transition bindings were validated against the same episode pin.
- A cross-revision next observation raised a quarantine disposition; action dispatch,
  transition commit, and replay commit all remained false.
- Each activation condition was negated independently: active episode, in-flight request,
  queued action, unconsumed ACK, unsealed synthetic WAL witness, missing synthetic
  reset/Home witness, and incomplete candidate validation. Every case preserved active,
  pending, epoch, queue, and transition state.
- Simulated activation incremented policy epoch and invalidated the pinned request/chunk.
  Old revision, model, request, chunk, and epoch results were normal stale-drops with no
  fatal exit.
- Human takeover and reset invalidation each incremented policy epoch and cleared the
  pinned policy queue.
- Mid-episode rollback failed closed. Quiescent rollback targeted the immutable previous
  stable revision, marked the replaced revision `rolled_back` with a reason, incremented
  epoch, and cleared the old request/chunk.
- The rollback contract is `retain_pending_no_auto_activation`. This was checked before a
  later, separate explicit activation at a new quiescent boundary. Rollback itself never
  auto-activated the pending candidate.
- Learner stall and invalid candidate validation left the stable active revision unchanged.

The `robot_home` and `wal_sealed` inputs are explicitly synthetic boolean witnesses. They
test gate logic only and are not evidence of a real robot Home reset or production WAL seal.

## Persistence and fault injection

The lifecycle registry is canonical JSON with an internal digest and atomic same-filesystem
replacement. A fresh CPU Python subprocess recovered active, pending, previous,
rejected/rolled-back records, revision artifact digests, policy epoch, and all publication
counters. Fresh recovery unconditionally sets `safe_reset_required=true`; episode start,
candidate activation, and action authorization fail closed until a new boundary is supplied.

The existing Stage‑3 online checkpoint metadata round-trip also retained the pending
revision and policy epoch. The G5P exact-resume checkpoint implementation was not modified
or used.

The following 23 cases passed:

- model tamper;
- manifest tamper;
- source/config/runtime binding mismatch;
- missing completion marker;
- revision ID/digest collision;
- crash before atomic revision rename;
- crash after revision rename before pending-pointer update;
- crash before atomic registry replacement;
- valid candidate staged during an active episode;
- invalid candidate rejection with durable reason and digest;
- activation with active episode;
- activation with in-flight request;
- activation with queued action;
- activation with unconsumed ACK;
- activation with unsealed synthetic WAL witness;
- activation without synthetic reset/Home witness;
- activation with incomplete validation;
- old revision/request/chunk stale-drop;
- policy epoch mismatch stale-drop;
- cross-revision transition quarantine;
- pending survives restart and checkpoint round-trip;
- illegal mid-episode rollback; and
- learner stall leaves active revision unchanged.

Temporary revision directories and orphaned complete immutable candidates are never scanned
into registry state and cannot become active. A failed registry replacement preserves the
last complete registry.

## Source binding and production blockers

The revision manifest recursively binds 91 files, including all
`src/forcesmolvla/**/*.py`, all Stage‑3 configs, required action/normalizer/calibration/
wrench/reward/RuleSpec/task-feature/trainability contracts, Stage‑3 protocol and schemas,
the isolated CLI, the present legacy server source, the environment lock, and vendor
SmolVLA Python sources.

- Recursive source tree SHA256:
  `5917ec1b8e951bb7364b0ab557b0881568d3c97ab1afc16bcd1946541ddcd3ff`.
- Resolved Stage‑3 config tree SHA256:
  `5729f8ef5a08be4b1dd3d4ffa0ad920fe778d71a4dd0393b6091154341baa336`.
- Vendor LeRobot commit: `30da8e687a6dfc617fcd94afc367ac7071c376ce`.
- Vendor SmolVLA tree SHA256:
  `a37faf6653470d8b55d6e97979e61afd0768f52bf0793bb603e8027c3fa56744`.
- Environment lock SHA256:
  `5b95fa25264637c9e114b9137015f0f7cf43b866b88d212e3d16b80b823cbd25`.

Production source binding is incomplete because these components do not exist:

- `tools/serve_policy_stage3.py`;
- `tools/export_policy_revision_stage3.py`;
- `tools/publish_policy_revision_stage3.py`;
- `robot/deployment/reset_home_witness.py`;
- `src/forcesmolvla/rft/stage3/wal.py`; and
- `src/forcesmolvla/rft/stage3/outbox.py`.

`tools/serve_policy.py` and `environment-manifest/requirements.lock` exist, but the old
loopback HTTP server is not a Stage‑3 revision publisher/server and was not started or
modified. No production server, robot runtime, deployment client, WAL, or outbox was added
to manufacture a complete result.

## Actual changed files and reasons

| File | Reason |
|---|---|
| `src/forcesmolvla/rft/stage3/publication.py` | Extend the existing single lifecycle primitive with content-addressed immutable export/validation, rolled-back audit state, episode revision/model/epoch pinning, counters, canonical snapshots, atomic registry, and fresh-recovery reset gate. |
| `src/forcesmolvla/rft/stage3/protocol.py` | Extend `PolicyEpochGate` with active model and request/chunk pinning so old model/request/chunk are normal stale-drops and invalidation clears queued identity. |
| `src/forcesmolvla/rft/stage3/transition.py` | Add the shared eight-event episode revision-binding validator used to quarantine cross-revision rows. |
| `src/forcesmolvla/rft/stage3/__init__.py` | Export the new functions from the existing Stage‑3 public API. |
| `configs/stage3_policy_revision_loopback.v1.development.yaml` | Freeze deterministic tiny payloads, binding inputs, source closure, dimensions, production component expectations, and rollback/recovery rules. |
| `schemas/stage3_policy_revision.v1.schema.json` | Validate immutable revision manifests and mandatory bindings. |
| `schemas/stage3_policy_revision_loopback_report.v1.schema.json` | Validate the report and fail-closed safety claims. |
| `tools/run_stage3_policy_revision_loopback.py` | Orchestrate the isolated filesystem lifecycle, subprocess recovery, fault injection, schema validation, and canonical report. |
| `tests/test_stage3_policy_revision_loopback.py` | Cover export, lifecycle, quiescence, stale-drop, rollback, recovery, source closure, CLI repeatability, safety imports, and checked-in artifact integrity. |
| `artifacts/development/stage3/stage3_policy_revision_loopback.v1.json` | Store deterministic canonical G6P evidence. |
| `docs/stage3_policy_revision_loopback_report.v1.md` | Record the real interface audit, evidence, blockers, changes, and stop boundary. |

No Phase‑1/2 source, old server, recorder, deployment, robot, or safety path was modified.

## Final markers and stop boundary

```text
G6P_IMPLEMENTED=true
G6P_IMMUTABLE_EXPORT=PASS
G6P_ATOMIC_PUBLICATION=PASS
G6P_INVALID_CANDIDATE_REJECTION=PASS
G6P_ONE_EPISODE_ONE_REVISION=PASS
G6P_QUIESCENT_ACTIVATION_GATE=PASS
G6P_OLD_CHUNK_INVALIDATION=PASS
G6P_ROLLBACK=PASS
G6P_PENDING_RESTART_RECOVERY=PASS
G6P_TRANSITION_REVISION_BINDING=PASS
G6P_CANONICAL_DIGEST_REPEATABLE=true
G6P_LOOPBACK_ACTIVATION=true
G6P_RESULT=PASS

SYNTHETIC_REVISION_PAYLOAD=true
REAL_LEARNER_REVISION_USED=false
REAL_POLICY_MODEL_EXPORTED=false
REAL_POLICY_SERVER_USED=false
REAL_RESET_HOME_VERIFIED=false
PRODUCTION_WAL_SEALED_VERIFIED=false
DIRECT_PUBLIC_HTTP_PARITY_VALIDATED=false
PRODUCTION_SOURCE_BINDING_COMPLETE=false
PRODUCTION_POLICY_PUBLICATION_VALIDATED=false
PRODUCTION_POLICY_ACTIVATION=false
POLICY_REVISION_ACTIVATED=false
G6_FORMAL_GATE_PASSED=false

G3_RECORDED_FIXTURE_LOOPBACK=BLOCKED
G5_PRODUCTION_DURABLE_RESUME=UNVERIFIED
CRITIC_WARMUP_STARTED=false
CRITIC_READY=false
ACTOR_Q_GUIDANCE_ENABLED=false

CUDA_INITIALIZED=false
NETWORK_SERVER_STARTED=false
ROBOT_CONNECTION_COUNT=0
ROBOT_COMMAND_COUNT=0
ROBOT_EXECUTION_AUTHORIZED=false
G7_AND_LATER=NOT_RUN
PUSHED=false
```

The work stops at G6P. Nothing was committed or pushed; no server, real policy export,
production revision activation, robot connection, or G7 work was started.
