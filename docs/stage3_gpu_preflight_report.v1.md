# Stage-3 G4P isolated GPU numerical preflight v1

This is a disposable numerical preflight on the approved-hybrid parent. It is not Critic warmup, online training, a formal Stage-3 gate, policy publication, or robot authorization.

## Result and immutable boundary

- `tool_status=PASS`
- `preflight_only=true`
- `parent_checkpoint_mutated=false`
- `runtime_optimizer_state_persisted=false`
- `policy_revision_exported=false`
- `robot_execution_authorized=false`
- `CRITIC_READY=false`
- `ACTOR_Q_GUIDANCE_ENABLED=false`
- `CRITIC_WARMUP_STARTED=false`
- `G5_AND_LATER=NOT_RUN`
- `GPU_COEXISTENCE_VALIDATED=false`

## Environment

- CUDA device `0` / visible `0`: `NVIDIA GeForce RTX 4090 D` (`GPU-0f7fb0c9-3905-7cd0-7e0e-bad412e1cc66`).
- Python `/home/rlc123/anaconda3/envs/forcesmolvla/bin/python`; PyTorch `2.11.0+cu128`, CUDA `12.8`, cuDNN `91900`.
- Initial free VRAM `23029.6` MiB.

## Parent load

- Actor strict load: `True` with 574 tensors, 0 missing, 0 unexpected, 0 shape mismatches.
- Online Q1/Q2 and stored target Q1/Q2 were loaded with CPU `weights_only=True` and strict keys/shapes/dtypes before moving to GPU.
- G7A-r5, random Critic fallback, and target-from-online fallback were not used.
- Binding and every selected parent artifact SHA are identical before and after.

## Data and cycles

- Critic C64 is exactly 32 `synthetic_preflight_R_only` + 32 real offline D rows. Actor B24 is exactly 12 + 12; all 88 underlying real observation rows are non-overlapping.
- The R label is numerical-preflight-only. Images/state/wrench/action/reward/terminal values still come through the frozen real Phase-2 train pipeline and are not online evidence.
- H=50, N=10, K=3; every Actor flow inference is subbatched at 4.
- Updates: 8 Critic, 4 Actor, 8 paired target Polyak applications.

## Optimizer and gradient ownership

- Fresh initial optimizer state entries: `0`; Actor/Critic ID intersection: `0`.
- Frozen parameters in optimizers: `0`; target parameters in optimizers: `0`.
- `apply_frozen_vlm_trainability()` was called. Vision/SmolVLM/language embeddings and state-prefix projection stayed frozen/eval/detached; Force/Action modules stayed trainable.

## Numerical evidence

- All finite: `True`; frozen hashes unchanged: `True`.
- Cal-QL/CQL/random-candidate/MC-return online calls: `0/0/0/0`.
- Terminal probe next Actor/target-Q calls: `0/0/0`.
- Critic is pure TD with stored target Twin-Q min. Actor uses expert-only FM plus current min Twin-Q; autonomous FM and gripper Q-gradient are exactly zero, TCP6 Q-gradient and expert gripper FM-gradient are nonzero.
- Each Actor step also records a fixed 4-row/fixed-noise before/after TCP6 drift and binary gripper pattern-change probe.

## Evidence-freeze and eta diagnostic

- `G4P_RESULT=PASS`; `R_SOURCE=synthetic_preflight_R_only`; `REAL_ONLINE_R_USED=false`.
- The four Actor optimizer steps changed only the disposable preflight instance: `PREFLIGHT_ACTOR_STEPS_DISPOSABLE=true` and `PRODUCTION_ACTOR_STATE_MUTATED=false`.
- `RUNTIME_OPTIMIZER_STATE_PERSISTED=false`; `ETA_3_APPROVED=false`; no Critic warmup was started.
- Weighted Q/FM ratios below are recomputed independently from each cycle's stored weighted gradient norms. The eta=3 column is only a linear rescaling diagnostic.

| Cycle | Weighted Q norm | Weighted FM norm | Q / FM | Linear eta=3 diagnostic |
|---:|---:|---:|---:|---:|
| 0 | 0.000260606751847 | 0.827020623178 | 0.000315115179167 | 0.00945345537501 |
| 1 | 0.000296429636821 | 0.650257897602 | 0.000455864723695 | 0.0136759417109 |
| 2 | 0.000621424523775 | 0.832295002819 | 0.000746639739119 | 0.0223991921736 |
| 3 | 0.00129431570298 | 0.686809343072 | 0.00188453421031 | 0.0565360263094 |

eta=3 remains a provisional numerical-preflight candidate.
No eta calibration or Actor Q-guidance approval is granted by G4P.

| Cycle | Kind | Wall s | Peak allocated MiB | Cycles/hour |
|---:|---|---:|---:|---:|
| 0 | warmup | 15.519 | 6780.9 | 231.97 |
| 1 | measured | 15.237 | 7638.4 | 236.26 |
| 2 | measured | 15.131 | 7638.4 | 237.92 |
| 3 | measured | 15.064 | 7638.4 | 238.98 |

Load-only allocated VRAM: `1591.9` MiB; warm-up peak: `6780.9` MiB; measured peak: `7638.4` MiB; post-release allocated: `65.0` MiB; peak CPU RSS: `9903.8` MiB.

## Deferred production capability

- Critic formal warmup or stability gate
- runtime optimizer persistence or exact resume
- recorded-live G3 fixture loopback
- shadow learner, online training, policy export or activation
- GPU coexistence, ROS, networking or robot execution

```text
canonical_report_sha256=968fafebd030e961a63f19b420c361d1a8981d0e2747926c8e16aaf0d6474c11
G0_FORMAL_GATE_PASSED=false
G3_RECORDED_FIXTURE_LOOPBACK=BLOCKED
G5_AND_LATER=NOT_RUN
ROBOT_CONNECTION_COUNT=0
ROBOT_COMMAND_COUNT=0
ROBOT_EXECUTION_AUTHORIZED=false
```
