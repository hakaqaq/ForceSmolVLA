# Stage-3 G0A approved-hybrid parent binding preflight v1

This report freezes the explicitly approved new hybrid Stage-3 bootstrap. It is not an exact Phase-2 cycle210 learner continuation and it does not authorize training, GPU model loading, publication, networking, ROS, or robot execution.

## Authority precedence

The sole authoritative Stage-3 parent selection is the mutually consistent set `configs/stage3_parent_binding.v1.development.json`, `artifacts/development/stage3/stage3_parent_binding_preflight.v1.json`, and this report. The three `stage3_parent_inventory*` provenance files are historical read-only scans only; any historical `selected_option=B` in them is superseded and must not be interpreted as the current route. The only authorized route is `approved_hybrid`: cycle210 evaluation Actor plus G7A-r2 online and target Critics, with `binding_type=new_hybrid_stage3_bootstrap`.

## Decision and boundary

- `tool_status=PASS`
- `G0A_HYBRID_PARENT_BINDING=PASS`
- `G0_FINAL_PARENT_BINDING=BOUND_APPROVED_HYBRID`
- `binding_type=new_hybrid_stage3_bootstrap`
- `not_exact_phase2_cycle210_continuation=true`
- `cycle210_full_learner_checkpoint_available=false`
- `full_learner_resume=false`
- `PARENT_PAYLOAD_COMPLETE_FOR_HYBRID=true`
- `STRICT_PHASE2_CONTINUATION_AVAILABLE=false`

The cycle210 evaluation export supplies only the Stage-3 initial Actor. It has no Critic, target, optimizer, scheduler, RNG, sampler, or learner cursor. G7A-r2 independently supplies Q1/Q2 and the stored target Q1/Q2. G7A-r5 remains present and explicitly unselected.

## Selected payloads

| Role | Path | SHA-256 | Validation |
|---|---|---|---|
| Actor | `/home/rlc123/ForceSmolVLA/artifacts/development/stage2/stage2b_cycle210_evaluation_smoke_checkpoint.v1/model.safetensors` | `e24c1d6bb0a778921659514ac47c692b952178aa39af2601ccf0fc32bf94774d` | safetensors header/key/shape/dtype plus prior 574/574 strict export evidence; no tensor load or forward |
| Q1 | `/home/rlc123/ForceSmolVLA/artifacts/development/stage2/g7a_r2_critic_warmup_checkpoint/models/q1_state.pt` | `13f464ea2c07184dd6a564af6743872af2e2c06cbf21e23dfe29f20363bb4a66` | CPU `weights_only=True`, strict key/shape/dtype |
| Q2 | `/home/rlc123/ForceSmolVLA/artifacts/development/stage2/g7a_r2_critic_warmup_checkpoint/models/q2_state.pt` | `86022ddc6b78ea06919c0d02d7a7b7ee718d958df35f509d2a53ce9584ed53ec` | CPU `weights_only=True`, strict key/shape/dtype |
| target Q1 | `/home/rlc123/ForceSmolVLA/artifacts/development/stage2/g7a_r2_critic_warmup_checkpoint/models/q1_target_state.pt` | `da3e80a552c65be49e5f9d7f911cd1c3b2e466e4c1402a71181a5de4d24e62ea` | CPU `weights_only=True`, strict key/shape/dtype |
| target Q2 | `/home/rlc123/ForceSmolVLA/artifacts/development/stage2/g7a_r2_critic_warmup_checkpoint/models/q2_target_state.pt` | `a113fe238acbbf6921eb736a988c2dfd6b8074d01369786c860961ff6a1e6810` | CPU `weights_only=True`, strict key/shape/dtype |

Actor container tree: `0945ab6d984663b82c2546f2b70eb190e59907bbc87a44fa806fda0f729682b7` (54 files, 1425391406 bytes).

G7A-r2 container tree: `f8c08b9058d173211a7306d370a97a848bfc1f7569ac52e6cc88baacff0c0d40` (18 files, 296019846 bytes).

## Compatibility result

- `ACTOR_METADATA_COMPATIBILITY=PASS`
- `CRITIC_CPU_STATE_COMPATIBILITY=PASS`
- `TARGET_CRITIC_CPU_STATE_COMPATIBILITY=PASS`
- `CROSS_COMPONENT_CONTRACT_COMPATIBILITY=PASS`
- Actor `H=50`, Flow `N=10`; Critic `K=3`, action7; rational 30 Hz data grid and fixed 10 Hz policy phase match.
- Actor images are float32 `[0,1]` before Actor preprocessing. Critic images use the distinct uint8 `[0,255]` path and are converted internally to float32 `/255`.
- Canonical task feature is 256D with logical tensor SHA-256 `b40aa90369c4c355405ed7234dfb1a5935ee3dd05df73c7847132e2b34d44103`; evidence is `container_plus_recomputed_logical_tensor_digest`.
- State7, wrench6, normalizer, ActionContract-v2, calibration/runtime hashes, TCP6 Q-gradient, and gripper stop-gradient contracts match.
- The calibration and runtime records remain development-only/formal-not-ready; this binding does not upgrade their formal status.

## Optimizer and safety state

Only the rebuild specification is frozen. Actor/Critic optimizers are fresh-by-policy but were not instantiated; no optimizer, scheduler, RNG, or sampler state is inherited. The G3P tiny CPU optimizer is not a cross-stage rebuild.

- `CROSS_STAGE_OPTIMIZER_REBUILD_SPEC=FROZEN`
- `CROSS_STAGE_OPTIMIZER_REBUILT=NOT_RUN`
- `INITIAL_ACTOR_UPDATE_ENABLED=false`
- `CRITIC_WARMUP_REQUIRED=true`
- `CRITIC_READY=false`
- `ACTOR_Q_GUIDANCE_ENABLED=false`

Actor Q-guidance may be enabled only after a separately authorized Critic warmup/stability gate. This preflight does not implement or simulate that unlock.

## Deferred validation

- full Actor tensor load and real Actor forward on GPU
- real Critic numerical forward on GPU
- Stage-3 Actor optimizer instantiation and ownership audit
- Stage-3 Critic optimizer instantiation and ownership audit
- cross-stage optimizer rebuilt status remains NOT_RUN
- independent Critic warmup and stability gate
- recorded-live temporal parity and robot execution
- formal calibration/runtime approval fields

Therefore `G0_FORMAL_GATE_PASSED=false`, `REAL_MODEL_FORWARD=NOT_RUN`, and `G4_AND_LATER=NOT_RUN`. The next eligible activity is a separately authorized G4P GPU numerical preflight.

## Safety footer

```text
canonical_report_sha256=ba31c7b55cedd275cc3fb9e4665e5dc58a193024366b195e7aef87e0fbd02a8d
CUDA_INITIALIZED=false
ROBOT_CONNECTION_COUNT=0
ROBOT_COMMAND_COUNT=0
ROBOT_EXECUTION_AUTHORIZED=false
```
