# P9 pure-offline Shadow implementation and preflight report

> Historical notice（2026-08-20）：v4.2 修改了 action inverse/safety、P9 pass assertions 和 source binding；本报告不是当前 P9 acceptance，且 P8 未重新通过前禁止进入 P9。

Date: 2026-08-20 (Asia/Shanghai)  
Status: `development_only`  
Formal eligibility: `false`

## Entry gate

- P8 development gate remained pass: strict local checkpoint, 78 hash-bound payloads, fresh-process offline parity exact.
- The user-requested ForceVLA training wait completed naturally. `forcevla_task2_train` exited with status 0 after step 10000; `/home/rlc123/ForceVLA/checkpoints/forcevla_task2_lora/task2_seed42/10000` contains the finalized metadata/params/train-state payloads and no temporary checkpoint directory. No input, pause, or kill was sent to that training process.
- The user's conditional P9 approval therefore became effective before P9 implementation/preflight began.

## Implemented boundary

- Added a deterministic integer-time 30 Hz action / 10 Hz policy / 1 kHz controller simulator with `H=50`, `K=3` and 100 ms maximum final ZOH extension.
- Added pre-model compatibility/geometry/clock checks; workspace, orientation, delta, gripper, observation-age, transport, end-to-apply and slot-lateness candidate checks; and aggregate expiry, missed-tick and hold-overrun checks.
- Added latest-generation-wins arbitration. Only a dispatch-valid generation cancels not-yet-started entries; an invalid newer generation leaves the valid older plan intact; old generations arriving after a newer generation are stale-rejected.
- Metrics are derived only from `actual_dispatched_indices`. Intervals must be strictly monotonic, nonnegative and nonoverlapping.
- Each record contains the full normalized and absolute 7D chunk, clock map, all timing fields, full `ChunkContext`, camera IDs/timestamps/tensor hashes, prompt/token hashes, noise provenance/hash, raw/filter timestamps, calibration/geometry/normalizer hashes, checkpoint/config/artifact hashes and explicit RTC/native-queue/ROS/action-send state.
- Replay verifies the artifact and per-record hashes, recomputes candidate timing/safety and arbitration, and requires exact equality of the actual dispatch result.

No ROS client, live robot interface, Franky/RTC queue, native `select_action`, publisher, goal sender or robot action path was added. P9 only reads task1, a P8 development checkpoint and test fixtures, then writes JSON records.

## Threshold and clock separation

- `configs/shadow_safety_thresholds.development.yaml` remains unchanged with unapproved/null production candidates.
- Numerical test boundaries exist only in `tests/fixtures/shadow_safety_thresholds.test_only.yaml` with `mode=test_only` and approval/signature pending.
- The synthetic clock map exists only in `tests/fixtures/shadow_clock_map.test_only.json`, is marked synthetic/development-only and has no signature or approval.
- The production resolver rejects missing clock maps and both test-only artifacts. Missing, stale or clock-domain-mismatched maps produce `candidate_valid=false` in the tested candidate path.
- No test threshold was copied into the production/development candidate YAML or P9 resolved config.

## Test results

- Full suite: `104 passed`.
- P9 coverage includes exact tick/index calculation, latest-wins supersession, invalid-new-does-not-supersede, production missing/stale/mismatched clock fail-close, actual interval monotonicity, hold overrun, hash tamper rejection and exact record replay.
- A separate fresh Python process loaded the written task1 record and reproduced the dispatch result exactly.

## Real RTX 4090D task1 preflight

Input checkpoint: `outputs/development/p8_checkpoint_seed42_step000001`  
Input dataset: `datasets/task1_forcesmolvla_v4_1`  
Sample recorded: validation episode 4, frame 0  
Inference: two cameras, `B=2`, `H=50`, full ForceToken-MoE checkpoint, 10 flow steps

| Measurement | Result |
|---|---:|
| CUDA inference latency | 411.515 ms |
| Wall inference latency | 411.541 ms |
| Peak allocated | 1,552,972,288 bytes |
| Peak reserved | 1,677,721,600 bytes |
| Candidate chunk index | 13 |
| End-to-apply age | 434 ms |
| Slot lateness | 0.667 ms |
| Actual dispatched actions | 0 |
| Robot actions sent | 0 |

The real task1 candidate was correctly rejected with `SHADOW_END_TO_APPLY_EXCEEDED`: the test-only 150 ms boundary is below the measured 434 ms end-to-apply age. This result is not silently relaxed and is not a production threshold decision. The valid dispatch/supersession path is covered by the separate synthetic golden fixture.

## Hash-bound outputs

- P9 source binding: `403fffc0361895bc2c2945144106558503e9a6ce43d03fa42cf0e9bc0ab668e8`
- P9 resolved config: `77dfb12e9073f8a8859eb55953ca3385aa6ae5efa419effa979c557db123623a`
- P9 GPU preflight: `baf1f178a6af8768397070d24e1cb39908d847d2af2fb74df1b78bcd5cff503e`
- task1 record file: `416365e38dcb5a4e22a04e4366577e6db7024e068ff174c173cdae102f3ad7d9`
- record artifact: `98d783a3d9208dc3a9c995eb1e481708bf01d01bc879a2260216eecafd9e8009`
- replay file: `c6b743cb2a2b329a3869213cafa2e1f1eb2c1b5df52cac4bf7121dda722899c1`

## Gate conclusion and blockers

P9 algorithmic development gate: **pass**. P5 through P9 are implemented in the required sequence, and P9 record/replay is deterministic and read-only.

Production/formal Shadow remains fail-closed because:

- no production sensor-to-controller or GPU-to-controller clock map exists;
- safety thresholds outside test fixtures remain null/unapproved;
- detached-signature algorithm, trusted key, verifier and approver remain unresolved;
- the real full-model latency sample exceeds the test-only end-to-apply boundary;
- task1 evidence is within-session algorithmic development replay, not production Shadow or cross-session generalization evidence.

No P9 result authorizes a live robot connection or action dispatch.
