# Stage-3 G3 handoff v1

Generated at `2026-08-28T07:02:07Z`. This document freezes the completed
Stage-3 G1/G2/G1A/G1B development work for a new Codex session. It does not
authorize or execute G3, training, online collection, policy publication, ROS,
or robot operation.

## 1. Repository identity

```text
repository absolute path = /home/rlc123/ForceSmolVLA
branch before handoff = main
handoff branch = stage3-online-hil
baseline HEAD = d00384330eed01ddd0dfe3bf435e4545fb679a71
ForceSmolVLA release version = v2.1.1
Python distribution version = 0.1.0
ConRFT frozen repository = /home/rlc123/conrft
ConRFT frozen reference commit = a779fde7fa5db5a469960a8490c100f35b41b49e
ConRFT worktree = clean
```

The release identity comes from `README.md`, `PHASE2_RELEASE.md`, the existing
parent inventory, and the `v2.1.1` release binding. The Python package metadata
still declares `0.1.0`; both identities are recorded rather than conflated.

Before the handoff files were created, `git status --short` reported only the
following pre-existing untracked Stage-3 work and parent-inventory evidence:

```text
?? artifacts/development/stage3/
?? configs/stage3_online_hil.v1.development.yaml
?? configs/stage3_policy_publication.v1.development.json
?? configs/stage3_replay_contract.v1.development.yaml
?? configs/stage3_reward_terminal_contract.v1.development.json
?? configs/stage3_trainability_contract.v1.development.json
?? configs/stage3_transition_contract.v1.development.json
?? docs/stage3_parent_inventory_report.v1.md
?? schemas/stage3_ack_transition.v1.schema.json
?? schemas/stage3_online_checkpoint.v1.schema.json
?? schemas/stage3_recorded_ack_fixture.v1.schema.json
?? schemas/stage3_temporal_parity_report.v1.schema.json
?? src/forcesmolvla/rft/stage3/
?? tests/test_stage3_ack_transition.py
?? tests/test_stage3_checkpoint.py
?? tests/test_stage3_contracts.py
?? tests/test_stage3_losses.py
?? tests/test_stage3_no_robot_imports.py
?? tests/test_stage3_protocol_and_publication.py
?? tests/test_stage3_recorded_ack_parity.py
?? tests/test_stage3_replay_and_credit.py
?? tests/test_stage3_temporal_bridge.py
?? tools/preflight_stage3_recorded_ack_parity.py
```

There were no tracked-file modifications. The exact expanded path list is in
`g3_handoff_manifest.v1.json.preexisting_uncommitted_paths`. The existing files
`stage3_parent_inventory.v1.json`, `stage3_parent_inventory_manifest.v1.json`,
and `stage3_parent_inventory_report.v1.md` are protected evidence and are not
part of the handoff commit.

## 2. Completed work

### G1 — contracts and schemas

The following append-only contracts are implemented and cross-checked:

| File | Frozen content |
|---|---|
| `configs/stage3_trainability_contract.v1.development.json` | Pending parent binding, no cross-stage optimizer, Frozen-VLM ownership, expert-only FM, min-Twin-Q guidance, TCP6-only Q gradient, readiness locked. |
| `configs/stage3_transition_contract.v1.development.json` | 30 Hz rational data grid, 10 Hz policy anchors, `H=50`, `K=3`, 100 ms ACK-authoritative macro action, canonical UID/digest, quarantine. |
| `configs/stage3_replay_contract.v1.development.yaml` | Canonical single payload, `R_online`/`D_expert`, intervention dual membership, 50:50 candidate sampling, `training_starts=100`, sample-credit rules. |
| `configs/stage3_reward_terminal_contract.v1.development.json` | Reward/terminated/truncated/bootstrap/discount consistency and reward-bearing online update lock. |
| `configs/stage3_online_hil.v1.development.yaml` | All live modes disabled, Critic readiness locked, provisional 2:1 learner recipe not authorized. |
| `configs/stage3_policy_publication.v1.development.json` | Revision identity/lifecycle, quiescent episode boundary, stale-result handling and rollback contract. |
| `schemas/stage3_ack_transition.v1.schema.json` | Canonical ACK transition schema including observation, proposal, accepted Kx7 action, expert Hx7 mask, outcome, eligibility, seal and integrity. |
| `schemas/stage3_online_checkpoint.v1.schema.json` | CPU metadata schema for models, rebuilt optimizers, RNG/samplers, replay, credits, publication and counters. |

`src/forcesmolvla/rft/stage3/contracts.py` provides
`CriticReadiness.validate()`, `load_stage3_contracts()`,
`validate_stage3_contracts()`, and `apply_stage3_trainability()`. The latter
delegates to the real Phase-2
`forcesmolvla.rft.frozen_vlm_trainability.apply_frozen_vlm_trainability()`;
it does not mirror or reimplement that ownership logic.

G1 tests:

- `tests/test_stage3_contracts.py::test_g1_contracts_are_cross_consistent_and_locked`
- `tests/test_stage3_contracts.py::test_stage3_trainability_reuses_frozen_vlm_contract`
- `tests/test_stage3_ack_transition.py::test_finalize_uid_digest_and_schema_are_stable`
- `tests/test_stage3_ack_transition.py::test_human_expert_requires_ack_source_and_intervention`
- `tests/test_stage3_ack_transition.py::test_reward_terminal_matrix_and_quarantine`
- `tests/test_stage3_checkpoint.py::test_checkpoint_schema_and_cpu_json_round_trip`
- `tests/test_stage3_checkpoint.py::test_checkpoint_fails_nonboundary_and_counter_or_credit_drift`

Conclusion: G1 contract files and schemas are implemented and CPU-tested, but
the recorded-live temporal gate remains blocked and no final parent is bound.

### G2 — pure CPU core

| File | Key symbols | Result |
|---|---|---|
| `src/forcesmolvla/rft/stage3/transition.py` | `AcceptedAck.validate`, `causal_zoh_ack_macro`, `normalized_ack_behavior_action`, `compute_transition_uid`, `canonical_payload_sha256`, `validate_ack_transition`, `validate_reward_terminal`, `finalize_ack_transition` | ACK-authoritative Kx7 construction, identity/integrity, reward outcome and fail-closed quarantine. |
| `src/forcesmolvla/rft/stage3/replay.py` | `memberships_for_transition`, `Stage3Replay.commit` | Intervention can belong to R and D while one canonical payload is stored; same UID/digest is idempotent and digest collision is fatal. |
| `src/forcesmolvla/rft/stage3/batch.py` | `MixedReplaySampler.sample`, `build_expert_feature_mask` | In-memory 50:50 R/D sampling and expert-only feature mask. |
| `src/forcesmolvla/rft/stage3/losses.py` | `compute_online_twin_q_td_loss`, `compute_expert_only_flow_matching_loss`, `compute_min_twin_q_guidance_from_values`, `compute_stage3_actor_objective`, `compute_stage3_min_twin_q_actor_loss` | Pure online TD has no Cal-QL/random/MC path; Actor FM is expert-only; Q guidance is `min(Q1,Q2)`; zero-expert batches are graph-connected exact zero. |
| `src/forcesmolvla/rft/stage3/update_credit.py` | `UpdateCreditLedger.mint_for_unique_R_commit`, `consume_one`, `snapshot`, `restore` | Only new unique R commits mint credits; zero credits block the learner. |
| `src/forcesmolvla/rft/stage3/protocol.py` | `TransportEnvelope`, `PolicyEpochGate.classify` | CPU protocol envelope and normal stale-revision drop. |
| `src/forcesmolvla/rft/stage3/publication.py` | `InMemoryRevisionStateMachine.stage`, `activate`, `rollback` | In-memory-only revision state machine with quiescent-boundary activation. |
| `src/forcesmolvla/rft/stage3/checkpoint.py` | `validate_online_checkpoint_metadata`, `cpu_round_trip_online_checkpoint` | JSON schema validation and CPU round-trip only; no exact-resume implementation. |
| `src/forcesmolvla/rft/stage3/__init__.py` | explicit exports | Stage-3 CPU API surface; no ROS, robot, server or publisher import. |

G2 tests:

- `tests/test_stage3_temporal_bridge.py::test_same_ack_can_causally_zoh_to_three_30hz_slots`
- `tests/test_stage3_temporal_bridge.py::test_future_missing_rejected_and_300ms_interpretation_fails`
- `tests/test_stage3_temporal_bridge.py::test_out_of_order_or_rejected_ack_fails_closed`
- `tests/test_stage3_replay_and_credit.py::test_R_D_membership_payload_dedupe_uid_and_credit_rules`
- `tests/test_stage3_replay_and_credit.py::test_credits_block_at_zero_and_round_trip_exactly`
- `tests/test_stage3_replay_and_credit.py::test_mixed_sampler_origin_and_expert_mask_prevent_R_self_imitation`
- `tests/test_stage3_losses.py::test_pure_online_td_has_no_calql_random_or_mc_and_uses_target_min`
- `tests/test_stage3_losses.py::test_all_terminal_rows_never_call_next_actor_or_target_critics`
- `tests/test_stage3_losses.py::test_expert_only_fm_zero_batch_is_graph_connected_exact_zero`
- `tests/test_stage3_losses.py::test_actor_objective_uses_min_q_and_actioncontract_v2_stops_gripper_q_gradient`
- `tests/test_stage3_protocol_and_publication.py::test_policy_epoch_stale_result_is_normal_drop`
- `tests/test_stage3_protocol_and_publication.py::test_revision_lifecycle_enforces_episode_boundary_and_rollback`
- `tests/test_stage3_protocol_and_publication.py::test_revision_identity_is_immutable_and_invalid_candidate_rejects`

Conclusion: G2 is a pure CPU primitive layer, not an Actor/Learner runtime. It
does not implement persistence/WAL, real publication, full learner scheduling,
GPU execution, exact resume, ROS, or robot control.

### G1A — real Phase-2 API compatibility audit

The compatibility closure uses production symbols rather than Stage-3 mirrors:

| Phase-2 dependency | Stage-3 use/evidence | Test |
|---|---|---|
| `src/forcesmolvla/rft/frozen_vlm_trainability.py::apply_frozen_vlm_trainability` | delegated by `contracts.py::apply_stage3_trainability` | `tests/test_stage3_contracts.py::test_stage3_trainability_reuses_frozen_vlm_contract` |
| Frozen prefix `no_grad`/detach path and Force K/V once | real model path exercised by a CPU-small fixture | `tests/test_stage3_recorded_ack_parity.py::test_real_phase2_frozen_prefix_path_is_no_grad_detached_and_force_kv_once` |
| `src/forcesmolvla/rft/critic_action_adapter_v2.py::critic_action_for_q_guidance_v2` | used for TCP6 differentiability and gripper stop-gradient | `tests/test_stage3_losses.py::test_actor_objective_uses_min_q_and_actioncontract_v2_stops_gripper_q_gradient` |
| `src/forcesmolvla/rft/critic.py::ForceAwareMacroCritic.forward` | called with real image/state/wrench/task/action/mask interface | `tests/test_stage3_recorded_ack_parity.py::test_online_td_calls_real_force_aware_macro_critic_interface` |
| `src/forcesmolvla/rft/losses.py::compute_min_twin_q_actor_loss` | compatibility checked against Stage-3 min-Q wrapper | Actor-objective and real-interface tests above |
| `src/forcesmolvla/action_delta.py::ActionDeltaProcessor.to_delta` | called by ACK action normalization and parity | transition/parity tests |
| `src/forcesmolvla/raw_to_lerobot_v3.py::prepare_episode` | canonical full per-episode Phase-2 converter called exactly once | `tests/test_stage3_recorded_ack_parity.py::test_stage2_parity_path_calls_production_prepare_episode_once` |
| Frozen normalizer exactly once | both Stage-2 and Stage-3 parity branches instrument the same frozen normalizer | synthetic parity tests |
| Actor/Critic image dtype/range and canonical task feature | real interface compatibility checked | `tests/test_stage3_recorded_ack_parity.py::test_real_phase2_actor_critic_image_range_and_task_feature_contract` |

`prepare_episode()` is the canonical per-episode converter: it consumes the raw
session episode plus runtime/calibration contracts and returns the complete
`PreparedEpisode` numeric result. `convert_dataset()` is the multi-episode
orchestration/writer and is intentionally not invoked by a one-episode parity
tool. Private helpers may execute inside `prepare_episode()`, but Stage-3 does
not compose those helpers to impersonate the converter.

Conclusion:

```text
G1A_PHASE2_API_COMPATIBILITY=PASS
G1A_FULL_CONVERTER_CALL_PATH=PASS
G1A_STAGE3_END_TO_END_RUNTIME_WIRING=UNVERIFIED
```

The PASS values cover callable production APIs and the synthetic converter
fixture only. There is no complete Stage-3 learner runtime to verify end to end.

### G1B — recorded-live ACK temporal parity tool

Implemented files:

- `schemas/stage3_recorded_ack_fixture.v1.schema.json`
- `schemas/stage3_temporal_parity_report.v1.schema.json`
- `src/forcesmolvla/rft/stage3/temporal_parity.py`
- `tools/preflight_stage3_recorded_ack_parity.py`
- `tests/test_stage3_recorded_ack_parity.py`

Key symbols are `validate_recorded_ack_fixture`,
`blocked_temporal_parity_report`, `_stage2_project`, `_stage3_project`,
`_comparison`, and `run_recorded_ack_parity`. `_stage2_project` calls production
`raw_to_lerobot_v3.prepare_episode()` exactly once. Converter numeric output is
compared separately from raw-stream ACK/gripper identity provenance.

G1B tests:

- `tests/test_stage3_recorded_ack_parity.py::test_synthetic_fixture_exercises_both_paths_but_cannot_open_formal_gate`
- `tests/test_stage3_recorded_ack_parity.py::test_stage2_parity_path_calls_production_prepare_episode_once`
- `tests/test_stage3_recorded_ack_parity.py::test_missing_raw_episode_fails_closed`
- `tests/test_stage3_recorded_ack_parity.py::test_missing_ack_id_fails_closed`
- `tests/test_stage3_recorded_ack_parity.py::test_missing_gripper_identity_fails_closed`
- `tests/test_stage3_recorded_ack_parity.py::test_missing_recorded_fixture_is_schema_valid_blocked_report`
- `tests/test_stage3_recorded_ack_parity.py::test_recorded_gripper_identity_mismatch_fails_closed`
- `tests/test_stage3_recorded_ack_parity.py::test_online_td_calls_real_force_aware_macro_critic_interface`
- `tests/test_stage3_recorded_ack_parity.py::test_real_phase2_actor_critic_image_range_and_task_feature_contract`
- `tests/test_stage3_recorded_ack_parity.py::test_real_phase2_frozen_prefix_path_is_no_grad_detached_and_force_kv_once`

Conclusion: the tool is ready, but there is no qualified recorded-live fixture.
Synthetic input proves tooling behavior only and is forbidden from opening the
formal gate.

## 3. Current gate state

```text
G1A_PHASE2_API_COMPATIBILITY=PASS
G1A_FULL_CONVERTER_CALL_PATH=PASS
G1A_STAGE3_END_TO_END_RUNTIME_WIRING=UNVERIFIED
G1B_RECORDED_ACK_PARITY_TOOL=READY
G1_TEMPORAL_PARITY_GATE=BLOCKED
RECORDED_FIXTURE_CAPTURE_REQUIRED=true
G1_GATE_PASSED=false
G2_FORMAL_GATE=BLOCKED_ON_G1
G3_AND_LATER=NOT_RUN
ROBOT_EXECUTION_AUTHORIZED=false
```

The missing recorded-live fixture blocks formal temporal parity and robot
execution. It does not permanently prohibit a later, separately authorized G3
provisional offline fake Actor/Learner loopback. Such work must remain fake
robot/synthetic ACK only until another authorization changes the gate.

## 4. Parent/checkpoint and runtime inventory

All recomputed SHAs below were obtained by read-only byte hashing. Directory
tree digests use sorted records of `relative_path`, NUL, file SHA, newline. No
large PyTorch/safetensors payload was deserialized and no GPU was started.

### Parent and checkpoint candidates

1. **Historical complete cycle210 learner**

   ```text
   logical role = exact Phase-2 continuation candidate
   selected = false
   absolute path = /home/rlc123/ForceSmolVLA/artifacts/development/stage2/stage2b_throughput_v2_half_pass_run.v1/checkpoint_cycle_000210
   repo-relative path = artifacts/development/stage2/stage2b_throughput_v2_half_pass_run.v1/checkpoint_cycle_000210
   format = Stage-2 cycle-boundary learner directory
   size = unavailable
   SHA-256 = b514b50d118cb3edaa6e5e135e1a2cf7340062d11c16cb58bed437581c082e08
   SHA evidence = manifest_only historical value
   required = Actor, Q1/Q2, targets, optimizer/scheduler, RNG/sampler, cursor, manifest
   existence = absent
   load compatibility = UNVERIFIED_NOT_AVAILABLE
   notes = retained manifest SHA 1d1644ed6b51a1a61cb66e0b6a4ae6831bcced025530dbb5583b5d90ea4083fa is evidence only, not payload
   ```

2. **Cycle210 evaluation Actor artifact**

   ```text
   logical role = evaluation-smoke-only Actor runtime artifact
   selected = false
   absolute path = /home/rlc123/ForceSmolVLA/artifacts/development/stage2/stage2b_cycle210_evaluation_smoke_checkpoint.v1
   repo-relative path = artifacts/development/stage2/stage2b_cycle210_evaluation_smoke_checkpoint.v1
   format = directory; safetensors + config + runtime manifests
   size = 1,425,391,406 file bytes
   SHA-256 = 0945ab6d984663b82c2546f2b70eb190e59907bbc87a44fa806fda0f729682b7
   SHA evidence = recomputed
   required = model.safetensors, config.json, artifact_manifest.json, normalizer, ActionContract-v2
   existence = present
   load compatibility = prior strict Actor coverage 574/574, missing=0, unexpected=0; not rerun here
   notes = no Critics/targets/optimizer/scheduler/RNG/sampler; not a learner parent
   ```

3. **Latest available Phase-2 Actor candidate**

   ```text
   logical role = cycle210 full Actor weights for inference/evaluation
   selected = false
   absolute path = /home/rlc123/ForceSmolVLA/artifacts/development/stage2/stage2b_cycle210_evaluation_smoke_checkpoint.v1/model.safetensors
   repo-relative path = artifacts/development/stage2/stage2b_cycle210_evaluation_smoke_checkpoint.v1/model.safetensors
   format = safetensors
   size = 1,420,094,908 bytes
   SHA-256 = e24c1d6bb0a778921659514ac47c692b952178aa39af2601ccf0fc32bf94774d
   SHA evidence = recomputed
   required = 574 Actor tensors; source Actor-state digest 73b35435e943823bb88c54decf68ce4bf08f39100999c5770b071aa76c3cf4c3
   existence = present
   load compatibility = prior strict export load pass; not rerun here
   notes = cannot be silently combined with old/fresh Critics as exact cycle210 continuation
   ```

4. **G7A-r2 checkpoint container**

   ```text
   logical role = Critic-warmup reconstruction candidate
   selected = false
   absolute path = /home/rlc123/ForceSmolVLA/artifacts/development/stage2/g7a_r2_critic_warmup_checkpoint
   repo-relative path = artifacts/development/stage2/g7a_r2_critic_warmup_checkpoint
   format = 18-file Stage-2 checkpoint directory
   size = 296,019,846 file bytes
   SHA-256 = f8c08b9058d173211a7306d370a97a848bfc1f7569ac52e6cc88baacff0c0d40
   SHA evidence = recomputed
   required = Q1/Q2, targets, Critic optimizer/scheduler, RNG, sampler, counters, source/config manifests
   existence = present
   load compatibility = prior fresh-process strict-load pass; not rerun here
   notes = historical marker NOT_AN_APPROVED_LONG_TRAIN_PARENT remains in force
   ```

5. **G7A-r2 online Q1 / online Q2**

   ```text
   logical roles = online Critic Q1; online Critic Q2
   selected = false; false
   absolute paths = /home/rlc123/ForceSmolVLA/artifacts/development/stage2/g7a_r2_critic_warmup_checkpoint/models/q1_state.pt ; .../models/q2_state.pt
   repo-relative paths = artifacts/development/stage2/g7a_r2_critic_warmup_checkpoint/models/q1_state.pt ; .../models/q2_state.pt
   format = PyTorch state payloads
   sizes = 56,463,747; 56,463,747 bytes
   SHA-256 = 13f464ea2c07184dd6a564af6743872af2e2c06cbf21e23dfe29f20363bb4a66 ; 86022ddc6b78ea06919c0d02d7a7b7ee718d958df35f509d2a53ce9584ed53ec
   SHA evidence = recomputed; recomputed
   required = complete G2 ForceAwareMacroCritic state_dict including canonical_task_feature
   existence = present; present
   load compatibility = prior strict model load pass; not deserialized here
   notes = independent online Critics; Stage-3 ownership/binding pending
   ```

6. **G7A-r2 target Q1 / target Q2**

   ```text
   logical roles = target Critic Q1; target Critic Q2
   selected = false; false
   absolute paths = /home/rlc123/ForceSmolVLA/artifacts/development/stage2/g7a_r2_critic_warmup_checkpoint/models/q1_target_state.pt ; .../models/q2_target_state.pt
   repo-relative paths = artifacts/development/stage2/g7a_r2_critic_warmup_checkpoint/models/q1_target_state.pt ; .../models/q2_target_state.pt
   format = PyTorch state payloads
   sizes = 56,464,699; 56,464,699 bytes
   SHA-256 = da3e80a552c65be49e5f9d7f911cd1c3b2e466e4c1402a71181a5de4d24e62ea ; a113fe238acbbf6921eb736a988c2dfd6b8074d01369786c860961ff6a1e6810
   SHA evidence = recomputed; recomputed
   required = complete G2 target Critic state_dict
   existence = present; present
   load compatibility = prior strict model load pass; not deserialized here
   notes = target state has no optimizer ownership
   ```

7. **G7A-r2 preserved r5 Actor parent**

   ```text
   logical role = frozen Stage-1 r5 Actor bound by G7A-r2
   selected = false
   absolute path = /home/rlc123/ForceSmolVLA/outputs/development/task2_lerobotv3_full_sft_10k_r5/checkpoints/step_010000
   repo-relative path = outputs/development/task2_lerobotv3_full_sft_10k_r5/checkpoints/step_010000
   format = LeRobot/ForceSmolVLA runtime checkpoint directory
   size = 3,887,526,794 file bytes
   SHA-256 = 01665f899ad34e4ba048a46bcffe3a0f819fc0f7e42c6d0c9a2db662a3665379
   SHA evidence = manifest_only
   required = model.safetensors, config.json, artifact_manifest.json, trainability_manifest.json
   existence = present; all required top-level files checked
   load compatibility = prior G7A binding says bitwise unchanged; not reloaded/rehashed as a tree here
   notes = G7A-r2 stores no Actor and binds this r5 checkpoint
   ```

### Architecture and runtime contracts

8. **Actor architecture config** — logical role Actor runtime architecture;
   selected false; absolute path
   `/home/rlc123/ForceSmolVLA/artifacts/development/stage2/stage2b_cycle210_evaluation_smoke_checkpoint.v1/config.json`;
   repo-relative equivalent; JSON; 3,038 bytes; SHA-256
   `27147413fd1062e77c312e3a9f5221318df138878de20099faa1a0012292d24b`;
   evidence recomputed; required keys include `force_smolvla`, two cameras,
   state7/wrench6/action7, `H=50`, `N=10`; present; prior runtime export
   validation passed; configuration only.

9. **Critic architecture config** — logical role G2 mask-aware Twin-Q;
   selected false; absolute path
   `/home/rlc123/ForceSmolVLA/configs/stage2_g2_force_aware_twin_q.development.yaml`;
   repo-relative equivalent; YAML; 3,565 bytes; SHA-256
   `ab23300edc478cff53a82215702fd659cc838933426f611042c9c6a3a88b7dcc`;
   evidence recomputed; required sections `critic_interface`, `observation`,
   `topology`, `resnet_asset`, `targets`, `actor_q_contract`; present; prior G2
   topology validation passed; action `[K=3,7]`, mask `[3]`.

10. **Frozen normalizer** — logical role Stage-1-frozen state/wrench/delta
    normalizer; selected false; absolute path
    `/home/rlc123/ForceSmolVLA/artifacts/development/stage2/stage2b_cycle210_evaluation_smoke_checkpoint.v1/manifests/normalizer_manifest.json`;
    repo-relative equivalent; JSON; 137,827 bytes; SHA-256
    `c053d6aadd9db1dd7e365afdb08ef020d10b990b2eec1a9103ffca5b1a1f6e7e`;
    evidence recomputed; required state/wrench/delta action statistics and
    feature order; present; binding matches G7A/cycle210 manifests, not loaded;
    must be applied exactly once.

11. **ActionContract-v2** — logical role mixed continuous/discrete Kx7 Critic
    action; selected false; absolute path
    `/home/rlc123/ForceSmolVLA/artifacts/development/stage2/stage2b_cycle210_evaluation_smoke_checkpoint.v1/manifests/stage2_action_contract.v2.development.json`;
    repo-relative equivalent; JSON; 2,020 bytes; SHA-256
    `163af8b6a1eb23d39a1f1afd76098348501d6212e6b7ce4b5ec44ea5f31721d6`;
    evidence recomputed; required TCP6 Q gradient, binary gripper projection,
    gripper Q stop-gradient and invalid-slot mask; present; Stage-3 API tests
    pass; public execution authorization remains separate.

12. **Canonical 256D task feature** — logical role frozen task feature inside
    each Critic state; selected false; evidence-container absolute path
    `/home/rlc123/ForceSmolVLA/artifacts/development/stage2/s2_g2_twin_q_topology.json`;
    repo-relative equivalent; JSON tensor-digest evidence; container size 45,781
    bytes; logical tensor SHA-256
    `b40aa90369c4c355405ed7234dfb1a5935ee3dd05df73c7847132e2b34d44103`;
    SHA evidence manifest_only; required dimension 256 and state key
    `canonical_task_feature`; evidence container present; prior G2/G7A strict
    load passed, tensor not deserialized here. The container file itself was
    recomputed as `9a0b006bd13a2fca844e3f4b118b1347dce4d1022f03671ca3293728588a90a2`.

13. **Calibration contract** — logical role wrench calibration/frame binding;
    selected false; absolute path
    `/home/rlc123/ForceSmolVLA/artifacts/development/stage2/stage2b_cycle210_evaluation_smoke_checkpoint.v1/manifests/calibration_bundle.development.json`;
    repo-relative equivalent; JSON; 1,267 bytes; SHA-256
    `f041784b78e08f1359e27b8f841e6a61f687b59bce1b3db841ca4986937a53f6`;
    evidence recomputed; required bias/sign/mass/COM/static transform/frames;
    present; runtime export binding present but not loaded; `formal_ready=false`.

14. **Converter runtime contract** — logical role Phase-2 temporal/runtime
    conversion contract; selected false; absolute path
    `/home/rlc123/ForceSmolVLA/artifacts/development/stage2/stage2b_cycle210_evaluation_smoke_checkpoint.v1/manifests/converter_runtime_spec.task2.development.json`;
    repo-relative equivalent; JSON; 2,624 bytes; SHA-256
    `f6312bc580311077a5eec56c53c3013ab6cebd9744463cf4354651cdc0d616ea`;
    evidence recomputed; required controller grid, ACK association, cameras,
    clock map, wrench filter and split; present; `prepare_episode` fixture tests
    pass; development approval fields remain pending.

## 5. Parent binding

No parent has been selected. The current state is deliberately neither an
`exact_phase2_continuation` nor a `new_hybrid_stage3_bootstrap`:

```text
G0_FINAL_PARENT_BINDING=PENDING
CROSS_STAGE_OPTIMIZER_REBUILT=NOT_RUN
```

Candidate Actor choices are the cycle210 evaluation Actor or the r5 Actor bound
by G7A-r2, subject to explicit approval. Candidate online Critics are G7A-r2
Q1/Q2; candidate target Critics are G7A-r2 target Q1/Q2. Combining cycle210
Actor with G7A-r2 or fresh Critics would be a new hybrid bootstrap, not an exact
cycle210 continuation, and must be labeled and approved as such.

Regardless of the eventual parent choice, Stage-3 Actor and Critic optimizers
must be newly created after binding approval. Stage-2 optimizer state must not
cross the stage boundary. No optimizer has been created in G1/G2/G1A/G1B.

## 6. ConRFT alignment boundary

The high-level route retained from ConRFT is:

```text
async Actor/Learner
R/D 50:50
intervention dual membership
training_starts=100
online pure TD
critic:actor updates=2:1
periodic policy publication
```

ForceRFT intentionally retains the following differences:

```text
SmolVLA flow matching
Frozen-VLM + trainable Force/Fusion/MoE/Action Expert
ACK-authoritative Kx7 macro action
expert-only FM
min Twin-Q Actor guidance
TCP6 Q-gradient
gripper Q stop-gradient
sample credits/revision/checkpoint safety mechanisms
```

A future implementation must not silently replace this with ConRFT's
whole-mixed-batch consistency loss, mean-Q Actor guidance, or full 7D Q-gradient.
The frozen ConRFT reference is commit
`a779fde7fa5db5a469960a8490c100f35b41b49e`.

## 7. Tests and reproduction

Collection command:

```bash
CUDA_VISIBLE_DEVICES='' \
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
PYTHONPATH=src:vendor/lerobot/src \
/home/rlc123/anaconda3/envs/forcesmolvla/bin/python \
-m pytest --collect-only -q tests/test_stage3_*.py
```

Execution command:

```bash
CUDA_VISIBLE_DEVICES='' \
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
PYTHONPATH=src:vendor/lerobot/src \
/home/rlc123/anaconda3/envs/forcesmolvla/bin/python \
-m pytest -vv tests/test_stage3_*.py
```

Current result:

```text
32 tests collected
32 passed
STAGE3_CPU_TESTS_FAILED=0
STAGE3_UNEXPECTED_SKIPS=0
ROS_IMPORT_COUNT=0
ROBOT_CONNECTION_COUNT=0
ROBOT_COMMAND_COUNT=0
```

The 32 exact node IDs are the G1/G2/G1A/G1B test IDs listed in Sections 2 and
the following no-robot tests:

- `tests/test_stage3_no_robot_imports.py::test_stage3_cpu_modules_have_no_ros_robot_or_network_imports`
- `tests/test_stage3_no_robot_imports.py::test_importing_stage3_stays_cpu_only_and_does_not_connect_or_command`

## 8. G3 next scope — description only, not executed

A new session may request separate authorization for:

```text
ConRFT-style provisional offline fake Actor/Learner loopback
fake robot/synthetic ACK only
episode seal -> R/D -> 50/50 sampler
training_starts=100
2 Critic : 1 Actor
pure online TD
expert-only FM + min-Q guidance
periodic staged policy revision
CPU/small fake modules first
no ROS
no robot
no real publisher
no checkpoint mutation
```

This handoff does not grant that authorization and does not bind a parent.

## 9. Protected scope

Do not modify, move, delete, re-sign, or reinterpret:

- all Phase-1 and Phase-2 configs, source, checkpoints, artifacts and reports;
- Stage-1 r5, G7A-r2, cycle210 evaluation Actor, Reward Classifier,
  detector-G1, normalizer, split and ActionContract-v2;
- `tools/serve_policy.py`;
- `/home/rlc123/fr3_client_ws/scripts/deploy_forcesmolvla.py`;
- raw converter, temporal, action-delta, normalizer, Critic, Stage-2 losses,
  Flow sampling, Critic action adapter and Frozen-VLM implementation;
- robot, controller, recorder, ROS/RTC and public inference paths;
- `artifacts/development/stage3/stage3_parent_inventory.v1.json`;
- `artifacts/development/stage3/stage3_parent_inventory_manifest.v1.json`;
- `docs/stage3_parent_inventory_report.v1.md`;
- any other user-owned pre-existing uncommitted file not explicitly listed in
  the handoff commit.

Final boundary:

```text
G0_FINAL_PARENT_BINDING=PENDING
G3_AND_LATER=NOT_RUN
ROBOT_CONNECTION_COUNT=0
ROBOT_COMMAND_COUNT=0
ROBOT_EXECUTION_AUTHORIZED=false
```
