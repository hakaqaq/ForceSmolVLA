# Stage-3 runtime temporal/safety report v1

## Outcome and scope

G7C1A CPU-only pre-freeze audit: **PASS**. The Stage-3 regression collected 202 tests and passed all 202. This report freezes CPU primitive behavior only; it does not approve production integration, production cadence, recorded-live temporal parity, the G7 formal gate, or G8.

Evidence classification used throughout:

- **code-audited**: derived from the named implementation and covered by CPU tests where a test is listed.
- **recorded-live-unverified**: no live production recording closes the claim.
- **benchmark-only**: G7 synthetic latency measurements provide preliminary coverage only.
- **unbound**: the production interface does not currently provide the required identity or contract.

Baseline parent: `stage3-online-hil@d9d6ecb468fb9301bf9b2196ecd32654fe7fd1c4`.

## Five independent temporal quantities

| Temporal category | Frozen value | Evidence class | Status |
|---|---|---|---|
| Model horizon timebase | `H50_MODEL_TIMEBASE_HZ=30` | code-audited configuration | frozen for the CPU contract |
| Pose reference dispatch | `POSE_REFERENCE_DISPATCH_HZ=10` | code-audited configuration | frozen for the CPU contract |
| Controller internal servo | `CONTROLLER_INTERNAL_SERVO_HZ=UNVERIFIED` | recorded-live-unverified | not a production contract |
| Transition/projection | `STAGE3_PROJECTION_GRID_HZ=30`; `CONTRACT_TRANSITION_MACRO_HZ=10`; `PRODUCTION_TRANSITION_COMMIT_HZ=UNVERIFIED` | CPU projection is code-audited; production commit is recorded-live-unverified | production binding incomplete |
| Policy request/refresh | `POLICY_REQUEST_TRIGGER=event_driven_low_watermark`; `POLICY_REQUEST_HZ_MEASURED=UNVERIFIED`; `POLICY_INFERENCE_10HZ_REQUIRED=false`; `PRODUCTION_SAFE_INFERENCE_REFRESH_RATE=UNVERIFIED` | trigger is the CPU contract; rates are recorded-live-unverified | no frequency is approved as a production contract |

These quantities are deliberately independent. In particular, a 100 ms dispatch/macro interval does not imply 10 Hz policy inference.

The current four-dispatch headroom represents approximately 400 ms. It is less than the G7 concurrent maximum service latency of approximately 443 ms and excludes additional overhead. Therefore `CURRENT_LOW_WATERMARK_APPROVED=false`; that benchmark is preliminary latency coverage, not a production low-watermark, maximum-action-age, or open-loop-safety approval.

## Runtime contract

- Thread ownership: `single_owner_event_loop_only`. The ledger captures the creating thread, and every stateful public entry point/property fails closed with a STOP latch on cross-thread access.
- Clock provenance: request, reference, result, selection, dispatch, Pose ACK, gripper ACK, and projection grid carry an explicit clock-domain identity. Cross-domain comparisons or transition creation are rejected.
- Chunk semantics: `ACTION_SLOT_FIFO_PRESENT=false`, `H50_ACTIONS_CACHED=50`, `MAX_SELECTIONS_PER_ADOPTED_CHUNK=8`, and `SELECTED_INDEX_POLICY=rational_time_based_sparse_selection`.
- Selection phase: index selection is anchored to `t_ref_ns` on the rational 30 Hz grid. CPU fault injection covers dispatch jitter that produces index increments of 2, 3, and 4; a fixed increment of 3 is not assumed.
- Authority boundary: only the safety-adapter output `post_adapter_absolute7` whose Pose and gripper acknowledgements are both accepted can become an `AcceptedAck` and enter `causal_zoh_ack_macro`.
- Fail-closed behavior: chunk age, selected index, dispatch count, refresh deadline/headroom, stale generation/epoch/revision, takeover/reset/revision flush, ACK lifecycle faults, and exhausted safe action all yield quarantine, HOLD, or latched STOP according to the primitive contract.

## Requirement → implementation → test matrix

| Requirement | Implementation (file:symbol) | CPU test/evidence | Result |
|---|---|---|---|
| Single event-loop owner; cross-thread fail-closed | `src/forcesmolvla/rft/stage3/runtime_contract.py:RationalH50SelectionLedger.__init__/_assert_owner` | `test_single_owner_event_loop_rejects_cross_thread_call_and_latches_stop` | PASS |
| Clock identity on all comparable runtime timestamps | `runtime_contract.py:ChunkRequestIdentity.validate`, `ChunkResultIdentity.validate`, `RationalH50SelectionLedger.adopt_result/begin_dispatch/record_pose_ack/record_gripper_ack` | `test_runtime_clock_domain_is_required_for_every_comparable_timestamp` | PASS |
| Cross-clock transition rejection | `runtime_contract.py:SelectionLedgerEntry.to_accepted_ack`, `project_acknowledged_runtime_macro` | `test_runtime_clock_domain_is_required_for_every_comparable_timestamp` | PASS |
| Rational H50 selection with a fixed anchor/phase | `runtime_contract.py:rational_h50_index`, `RationalH50SelectionLedger.begin_dispatch` | `test_rational_selection_fault_injection_accepts_index_steps_2_3_4`; `test_fixed_anchor_rejects_future_phase_and_repeated_index` | PASS |
| Maximum chunk age/index/dispatch count | `runtime_contract.py:RuntimeSafetyLimits`, `begin_dispatch` | `test_chunk_age_selected_index_and_dispatch_count_fail_closed` | PASS |
| Time-based low-watermark; current 400 ms not approved | `runtime_contract.py:refresh_assessment`; `CURRENT_LOW_WATERMARK_APPROVED` | `test_low_watermark_is_time_based_and_current_400ms_is_not_approved` | PASS |
| Refresh deadline and additional headroom | `runtime_contract.py:RuntimeSafetyLimits.refresh_required_coverage_ns`, `refresh_assessment` | `test_refresh_must_be_pinned_before_entering_additional_headroom` | PASS |
| Stale result generation/epoch/revision rejection | `runtime_contract.py:adopt_result/_identity_is_current` | `test_stale_result_and_takeover_reset_revision_generations_are_rejected` | PASS |
| Takeover/reset/revision flush invalidates pending commands and late ACKs | `runtime_contract.py:human_takeover_flush/reset_home_flush/policy_revision_flush/_flush` | `test_ack_after_generation_flush_is_rejected[takeover/reset/revision]` | PASS |
| Exact duplicate ACK is idempotent; conflicting duplicate STOPs | `runtime_contract.py:_entry_for_ack/record_pose_ack/record_gripper_ack` | `test_duplicate_ack_is_idempotent_only_for_the_exact_same_payload` | PASS |
| ACK before command, after deadline, or after STOP is rejected | `runtime_contract.py:_entry_for_ack/record_pose_ack/record_gripper_ack/expire_missing_acks` | `test_ack_before_command_after_deadline_and_after_stop_latch_is_rejected` | PASS |
| Missing/rejected/mismatched Pose or gripper ACK fails closed | `runtime_contract.py:record_pose_ack/record_gripper_ack/expire_missing_acks` | `test_pose_and_gripper_ack_fail_closed_on_missing_rejected_or_mismatch` | PASS |
| Partial Pose/gripper ACK cannot produce `AcceptedAck` | `runtime_contract.py:commit_dispatch/SelectionLedgerEntry.to_accepted_ack` | `test_partial_ack_cannot_create_accepted_action_or_transition` | PASS |
| Only accepted post-adapter absolute7 is authoritative | `runtime_contract.py:begin_dispatch/SelectionLedgerEntry.to_accepted_ack` | `test_only_dual_ack_post_adapter_absolute7_becomes_transition_authority` | PASS |
| Committed ledger → `AcceptedAck` → causal ZOH, including steps 2/3/4 | `runtime_contract.py:project_acknowledged_runtime_macro`; `transition.py:causal_zoh_ack_macro` | `test_rational_selection_fault_injection_accepts_index_steps_2_3_4` | PASS |
| No safe action gives explicit HOLD; uncertain command outcome STOPs | `runtime_contract.py:refresh_assessment/expire_missing_acks` | `test_no_safe_action_is_explicit_hold_and_unknown_command_outcome_is_stop` | PASS |
| Import does not start ROS/network/robot/CUDA work | Stage-3 package imports | `test_stage3_import_has_no_ros_network_robot_or_cuda_side_effects`; `test_stage3_no_robot_imports.py` | PASS |
| Production gripper unchanged/no-op provenance | `/home/rlc123/fr3_client_ws/scripts/deploy_forcesmolvla.py:run_async_policy_loop`; inherited `record_franka_spacemouse_publisher.py:_send_gripper_goal` | read-only production call-path audit | UNBOUND |
| Runtime ledger persistence and resume | `src/forcesmolvla/rft/stage3/checkpoint.py` G5P payload boundary | read-only checkpoint field audit | UNVERIFIED |

## ACK lifecycle and gripper closure

The CPU ledger implements:

- duplicate ACK with the same identity and payload: idempotent;
- duplicate ACK with a conflicting payload: rejected and STOP latched;
- ACK before its command: rejected;
- ACK after deadline, generation flush, revision flush, takeover, reset/Home, or STOP: rejected and cannot commit or clear STOP;
- Pose-only or gripper-only acknowledgement: cannot create `AcceptedAck`.

The production read-only audit found that `deploy_forcesmolvla.py:run_async_policy_loop` sends a gripper goal only when the desired state changes and no goal is active, after Pose acknowledgement. It does not bind a terminal gripper acknowledgement to each selected chunk/transition. When the gripper is unchanged, the path supplies neither a new command/ACK nor explicit held/no-op provenance referencing the last accepted gripper command. The inherited `_send_gripper_goal` has local command sequencing and asynchronous result handling, but the current deployment loop does not close that identity into the authoritative action7 transition.

Consequently:

```text
GRIPPER_NOOP_ACK_POLICY=UNBOUND
FULL_ACTION7_ACK_CLOSURE=false
PRODUCTION_INTEGRATION_BLOCKED_ON_GRIPPER_ACK=true
```

No synthetic acknowledgement is created, and copied gripper values are not treated as command provenance.

## Normalization boundary

```text
OBSERVATION_STATE_NORMALIZATION_ONCE=code-audited
OBSERVATION_WRENCH_NORMALIZATION_ONCE=code-audited
ACTION_DELTA_DENORMALIZATION_ONCE=code-audited
RECORDED_LIVE_ACCEPTED_MACRO_NORMALIZATION_ONCE=UNVERIFIED
```

The first three statements are source-level audits. The fourth remains unverified until recorded-live evidence shows the exact accepted macro path. CPU synthetic traces do not open that gate.

## Persistence boundary

The G5P checkpoint payload does not include the runtime selection ledger, open requests, pending commands, ACK identities, STOP latch, clock identity, or dispatch lifecycle. Therefore:

```text
RUNTIME_LEDGER_PERSISTED=false
PRODUCTION_RUNTIME_LEDGER_RESUME=UNVERIFIED
G5_PRODUCTION_DURABLE_RESUME=UNVERIFIED
```

## Complete Stage-3 collection

Command:

```bash
pytest --collect-only -q tests/test_stage3_*.py
```

Result: **202 tests collected**.

```text
tests/test_stage3_ack_transition.py::test_finalize_uid_digest_and_schema_are_stable
tests/test_stage3_ack_transition.py::test_human_expert_requires_ack_source_and_intervention
tests/test_stage3_ack_transition.py::test_reward_terminal_matrix_and_quarantine
tests/test_stage3_checkpoint.py::test_checkpoint_schema_and_cpu_json_round_trip
tests/test_stage3_checkpoint.py::test_checkpoint_fails_nonboundary_and_counter_or_credit_drift
tests/test_stage3_contracts.py::test_g1_contracts_are_cross_consistent_and_locked
tests/test_stage3_contracts.py::test_stage3_trainability_reuses_frozen_vlm_contract
tests/test_stage3_exact_resume.py::test_atomic_checkpoint_completion_and_fresh_object_restore
tests/test_stage3_exact_resume.py::test_partial_checkpoint_and_missing_completion_marker_fail_closed
tests/test_stage3_exact_resume.py::test_tampered_payload_sha_fails_closed
tests/test_stage3_exact_resume.py::test_wrong_parent_config_or_source_binding_fails_before_restore
tests/test_stage3_exact_resume.py::test_missing_optimizer_state_and_group_reorder_fail_closed
tests/test_stage3_exact_resume.py::test_rng_omission_and_corruption_fail_closed
tests/test_stage3_exact_resume.py::test_control_state_faults_fail_closed[credit-drift-<lambda>-CREDIT_COUNTER_DRIFT]
tests/test_stage3_exact_resume.py::test_control_state_faults_fail_closed[counter-drift-<lambda>-COUNTER_DRIFT]
tests/test_stage3_exact_resume.py::test_control_state_faults_fail_closed[unsealed-<lambda>-BOUNDARY_NOT_QUIESCENT]
tests/test_stage3_exact_resume.py::test_control_state_faults_fail_closed[pending-revision-<lambda>-PENDING_REVISION]
tests/test_stage3_exact_resume.py::test_mid_cycle_save_is_rejected_before_any_checkpoint_is_written
tests/test_stage3_exact_resume.py::test_cold_or_warm_decoded_image_cache_does_not_change_model_state
tests/test_stage3_gpu_coexistence_contract.py::test_config_freezes_full_workload_and_safety_scope
tests/test_stage3_gpu_coexistence_contract.py::test_config_rejects_degraded_or_expanded_scope[path0-16]
tests/test_stage3_gpu_coexistence_contract.py::test_config_rejects_degraded_or_expanded_scope[path1-8]
tests/test_stage3_gpu_coexistence_contract.py::test_config_rejects_degraded_or_expanded_scope[path2-2]
tests/test_stage3_gpu_coexistence_contract.py::test_config_rejects_degraded_or_expanded_scope[path3-100ms]
tests/test_stage3_gpu_coexistence_contract.py::test_config_rejects_degraded_or_expanded_scope[path4-True]
tests/test_stage3_gpu_coexistence_contract.py::test_config_rejects_degraded_or_expanded_scope[path5-True]
tests/test_stage3_gpu_coexistence_contract.py::test_config_rejects_degraded_or_expanded_scope[path6-True]
tests/test_stage3_gpu_coexistence_contract.py::test_config_rejects_degraded_or_expanded_scope[path7-resident]
tests/test_stage3_gpu_coexistence_contract.py::test_source_audit_binds_real_symbols_and_leaves_approval_fields_unbound
tests/test_stage3_gpu_coexistence_contract.py::test_benchmark_only_decoded_cache_is_byte_bounded_lru
tests/test_stage3_gpu_coexistence_contract.py::test_request_summary_never_calls_macro_period_an_approved_deadline
tests/test_stage3_gpu_coexistence_contract.py::test_fixed_action_comparison_reports_exact_parity
tests/test_stage3_gpu_coexistence_contract.py::test_fixed_action_comparison_reports_first_difference_without_tolerance_growth
tests/test_stage3_gpu_coexistence_contract.py::test_current_g7_closure_is_separate_from_historical_g6_closure
tests/test_stage3_gpu_coexistence_contract.py::test_schema_accepts_all_mode_statuses
tests/test_stage3_gpu_coexistence_contract.py::test_existing_raw_semantic_audit_recomputes_counts_and_is_deterministic
tests/test_stage3_gpu_coexistence_contract.py::test_existing_raw_queue_and_time_sliced_semantics_are_fail_closed
tests/test_stage3_gpu_coexistence_contract.py::test_tool_has_worker_only_cuda_imports_and_no_network_server_entrypoint
tests/test_stage3_gpu_coexistence_contract.py::test_separate_device_is_explicitly_not_run_in_schema_contract
tests/test_stage3_gpu_coexistence_contract.py::test_g7b_missing_each_required_timestamp_fails[episode_last_release_ns]
tests/test_stage3_gpu_coexistence_contract.py::test_g7b_missing_each_required_timestamp_fails[episode_queue_drained_ns]
tests/test_stage3_gpu_coexistence_contract.py::test_g7b_missing_each_required_timestamp_fails[episode_worker_exit_ns]
tests/test_stage3_gpu_coexistence_contract.py::test_g7b_missing_each_required_timestamp_fails[pre_learner_gap_start_ns]
tests/test_stage3_gpu_coexistence_contract.py::test_g7b_missing_each_required_timestamp_fails[pre_learner_gap_end_ns]
tests/test_stage3_gpu_coexistence_contract.py::test_g7b_missing_each_required_timestamp_fails[learner_spawn_requested_ns]
tests/test_stage3_gpu_coexistence_contract.py::test_g7b_missing_each_required_timestamp_fails[learner_process_spawn_ns]
tests/test_stage3_gpu_coexistence_contract.py::test_g7b_missing_each_required_timestamp_fails[learner_model_ready_ns]
tests/test_stage3_gpu_coexistence_contract.py::test_g7b_missing_each_required_timestamp_fails[learner_warmup_cycle_start_ns]
tests/test_stage3_gpu_coexistence_contract.py::test_g7b_missing_each_required_timestamp_fails[learner_warmup_cycle_end_ns]
tests/test_stage3_gpu_coexistence_contract.py::test_g7b_missing_each_required_timestamp_fails[learner_measured_cycle_1_start_ns]
tests/test_stage3_gpu_coexistence_contract.py::test_g7b_missing_each_required_timestamp_fails[learner_measured_cycle_1_end_ns]
tests/test_stage3_gpu_coexistence_contract.py::test_g7b_missing_each_required_timestamp_fails[learner_measured_cycle_2_start_ns]
tests/test_stage3_gpu_coexistence_contract.py::test_g7b_missing_each_required_timestamp_fails[learner_measured_cycle_2_end_ns]
tests/test_stage3_gpu_coexistence_contract.py::test_g7b_missing_each_required_timestamp_fails[learner_measured_cycle_3_start_ns]
tests/test_stage3_gpu_coexistence_contract.py::test_g7b_missing_each_required_timestamp_fails[learner_measured_cycle_3_end_ns]
tests/test_stage3_gpu_coexistence_contract.py::test_g7b_missing_each_required_timestamp_fails[learner_worker_exit_ns]
tests/test_stage3_gpu_coexistence_contract.py::test_g7b_missing_each_required_timestamp_fails[pre_resume_gap_start_ns]
tests/test_stage3_gpu_coexistence_contract.py::test_g7b_missing_each_required_timestamp_fails[pre_resume_gap_end_ns]
tests/test_stage3_gpu_coexistence_contract.py::test_g7b_missing_each_required_timestamp_fails[resume_requested_ns]
tests/test_stage3_gpu_coexistence_contract.py::test_g7b_missing_each_required_timestamp_fails[resume_process_spawn_ns]
tests/test_stage3_gpu_coexistence_contract.py::test_g7b_missing_each_required_timestamp_fails[resume_model_ready_ns]
tests/test_stage3_gpu_coexistence_contract.py::test_g7b_missing_each_required_timestamp_fails[resume_first_request_release_ns]
tests/test_stage3_gpu_coexistence_contract.py::test_g7b_missing_each_required_timestamp_fails[resume_first_result_ready_ns]
tests/test_stage3_gpu_coexistence_contract.py::test_g7b_missing_each_required_timestamp_fails[resume_queue_drained_ns]
tests/test_stage3_gpu_coexistence_contract.py::test_g7b_missing_each_required_timestamp_fails[resume_worker_exit_ns]
tests/test_stage3_gpu_coexistence_contract.py::test_g7b_out_of_order_timestamp_fails
tests/test_stage3_gpu_coexistence_contract.py::test_g7b_negative_duration_fails
tests/test_stage3_gpu_coexistence_contract.py::test_g7b_cycle_overlap_fails
tests/test_stage3_gpu_coexistence_contract.py::test_g7b_gap_mislabeled_as_budget_fails
tests/test_stage3_gpu_coexistence_contract.py::test_g7b_cold_swap_cannot_be_mislabeled_resident[TIME_SLICED_TOPOLOGY-resident-G7B_TOPOLOGY]
tests/test_stage3_gpu_coexistence_contract.py::test_g7b_cold_swap_cannot_be_mislabeled_resident[RESIDENT_TIME_SLICING-PASS-G7B_RESIDENT]
tests/test_stage3_gpu_coexistence_contract.py::test_g7b_valid_trace_and_derived_durations_are_exactly_recomputable
tests/test_stage3_gpu_coexistence_contract.py::test_g7b_base_component_digest_drift_fails
tests/test_stage3_gpu_coexistence_contract.py::test_g7b_targeted_entrypoint_calls_only_time_sliced_mode
tests/test_stage3_gpu_preflight_contract.py::test_config_freezes_exact_g4p_scope
tests/test_stage3_gpu_preflight_contract.py::test_config_rejects_batch_or_loss_semantic_reduction
tests/test_stage3_gpu_preflight_contract.py::test_fixed_real_row_selection_is_nonoverlapping_and_pool_exact
tests/test_stage3_gpu_preflight_contract.py::test_report_schema_requires_safe_pass_evidence
tests/test_stage3_gpu_preflight_contract.py::test_evidence_freeze_recomputes_each_cycle_gradient_ratio
tests/test_stage3_gpu_preflight_contract.py::test_gpu_tool_uses_real_production_primitives_and_fail_closed_loading
tests/test_stage3_gpu_preflight_contract.py::test_gpu_tool_has_no_robot_ros_network_or_process_imports_and_import_is_cpu_only
tests/test_stage3_gpu_preflight_contract.py::test_schema_is_draft_2020_12_and_config_sha_is_recordable
tests/test_stage3_loopback.py::test_end_to_end_synthetic_H50_ack_K7_seal_replay_learner_staged_revision
tests/test_stage3_loopback.py::test_rational_30hz_grid_has_fixed_10hz_anchor_phase
tests/test_stage3_loopback.py::test_99_unique_R_blocks_and_100_unique_R_unlocks
tests/test_stage3_loopback.py::test_exact_R_D_50_50_and_intervention_dual_membership
tests/test_stage3_loopback.py::test_two_critic_one_actor_and_two_polyak_updates
tests/test_stage3_loopback.py::test_calql_monkeypatch_to_raise_remains_uncalled
tests/test_stage3_loopback.py::test_actioncontract_v2_q_gradient_and_test_optimizer_ownership
tests/test_stage3_loopback.py::test_zero_credit_backpressure
tests/test_stage3_loopback.py::test_duplicate_commit_idempotence_and_conflicting_digest_rejection
tests/test_stage3_loopback.py::test_human_takeover_invalidates_stale_policy_chunk
tests/test_stage3_loopback.py::test_partial_missing_rejected_and_stale_ack_quarantine_without_replay
tests/test_stage3_loopback.py::test_revision_staged_but_never_activated
tests/test_stage3_loopback.py::test_frozen_normalizer_applied_exactly_once_per_accepted_macro
tests/test_stage3_loopback.py::test_two_identical_seeded_runs_have_same_canonical_report_digest
tests/test_stage3_loopback.py::test_recorded_live_missing_fixture_returns_schema_valid_BLOCKED
tests/test_stage3_loopback.py::test_cli_writes_schema_valid_recorded_live_BLOCKED_report
tests/test_stage3_losses.py::test_pure_online_td_has_no_calql_random_or_mc_and_uses_target_min
tests/test_stage3_losses.py::test_all_terminal_rows_never_call_next_actor_or_target_critics
tests/test_stage3_losses.py::test_expert_only_fm_zero_batch_is_graph_connected_exact_zero
tests/test_stage3_losses.py::test_actor_objective_uses_min_q_and_actioncontract_v2_stops_gripper_q_gradient
tests/test_stage3_no_robot_imports.py::test_stage3_cpu_modules_have_no_ros_robot_or_network_imports
tests/test_stage3_no_robot_imports.py::test_importing_stage3_stays_cpu_only_and_does_not_connect_or_command
tests/test_stage3_parent_contract.py::test_valid_approved_hybrid_binding_schema
tests/test_stage3_parent_contract.py::test_cycle210_evaluation_actor_is_selected_and_not_a_learner_resume
tests/test_stage3_parent_contract.py::test_g7a_r2_online_and_target_twin_q_are_selected
tests/test_stage3_parent_contract.py::test_g7a_r5_is_retained_but_explicitly_unselected
tests/test_stage3_parent_contract.py::test_binding_is_not_exact_cycle210_continuation
tests/test_stage3_parent_contract.py::test_missing_cycle210_full_learner_payload_is_explicit_and_not_masked
tests/test_stage3_parent_contract.py::test_missing_parent_artifact_fails_closed[Actor]
tests/test_stage3_parent_contract.py::test_missing_parent_artifact_fails_closed[online_q1]
tests/test_stage3_parent_contract.py::test_missing_parent_artifact_fails_closed[target_q1]
tests/test_stage3_parent_contract.py::test_mismatched_parent_sha_fails_closed[Actor]
tests/test_stage3_parent_contract.py::test_mismatched_parent_sha_fails_closed[online_q2]
tests/test_stage3_parent_contract.py::test_mismatched_parent_sha_fails_closed[target_q2]
tests/test_stage3_parent_contract.py::test_critic_architecture_key_shape_dtype_mismatch_fails_closed[missing-key]
tests/test_stage3_parent_contract.py::test_critic_architecture_key_shape_dtype_mismatch_fails_closed[unexpected-key]
tests/test_stage3_parent_contract.py::test_critic_architecture_key_shape_dtype_mismatch_fails_closed[shape-mismatch]
tests/test_stage3_parent_contract.py::test_critic_architecture_key_shape_dtype_mismatch_fails_closed[dtype-mismatch]
tests/test_stage3_parent_contract.py::test_cross_component_mismatch_fails_closed[normalizer]
tests/test_stage3_parent_contract.py::test_cross_component_mismatch_fails_closed[action-contract]
tests/test_stage3_parent_contract.py::test_cross_component_mismatch_fails_closed[task-feature]
tests/test_stage3_parent_contract.py::test_cross_component_mismatch_fails_closed[calibration]
tests/test_stage3_parent_contract.py::test_cross_component_mismatch_fails_closed[runtime]
tests/test_stage3_parent_contract.py::test_inherited_optimizer_rng_sampler_or_instantiation_is_rejected[inherit_actor_optimizer]
tests/test_stage3_parent_contract.py::test_inherited_optimizer_rng_sampler_or_instantiation_is_rejected[inherit_critic_optimizer]
tests/test_stage3_parent_contract.py::test_inherited_optimizer_rng_sampler_or_instantiation_is_rejected[inherit_scheduler]
tests/test_stage3_parent_contract.py::test_inherited_optimizer_rng_sampler_or_instantiation_is_rejected[inherit_rng]
tests/test_stage3_parent_contract.py::test_inherited_optimizer_rng_sampler_or_instantiation_is_rejected[inherit_sampler]
tests/test_stage3_parent_contract.py::test_inherited_optimizer_rng_sampler_or_instantiation_is_rejected[instantiated_in_this_round]
tests/test_stage3_parent_contract.py::test_initial_actor_freeze_and_q_guidance_lock
tests/test_stage3_parent_contract.py::test_real_cpu_preflight_is_complete_for_hybrid_and_does_not_initialize_cuda
tests/test_stage3_parent_contract.py::test_parent_module_and_cli_have_no_ros_robot_serve_deploy_or_network_imports
tests/test_stage3_parent_contract.py::test_binding_config_file_sha_is_stable_and_schema_is_draft_2020_12
tests/test_stage3_policy_revision_loopback.py::test_freeze_and_descendant_with_unchanged_bound_closure_pass
tests/test_stage3_policy_revision_loopback.py::test_git_native_recursive_pathspec_includes_zero_to_many_components
tests/test_stage3_policy_revision_loopback.py::test_nested_bound_path_removed_fails_closed
tests/test_stage3_policy_revision_loopback.py::test_nested_bound_path_added_fails_closed
tests/test_stage3_policy_revision_loopback.py::test_dirty_nested_bound_content_and_path_changes_fail_closed
tests/test_stage3_policy_revision_loopback.py::test_invalid_recursive_pathspecs_fail_closed[/src/**/*.py-G6C_ABSOLUTE_BOUND_PATH]
tests/test_stage3_policy_revision_loopback.py::test_invalid_recursive_pathspecs_fail_closed[src/../outside.py-G6C_BOUND_PATH_TRAVERSAL]
tests/test_stage3_policy_revision_loopback.py::test_invalid_recursive_pathspecs_fail_closed[:(glob)src/**/*.py-G6C_INVALID_PATHSPEC_MAGIC]
tests/test_stage3_policy_revision_loopback.py::test_missing_freeze_and_non_descendant_fail_closed
tests/test_stage3_policy_revision_loopback.py::test_historical_artifact_tamper_is_rejected
tests/test_stage3_policy_revision_loopback.py::test_explicit_frozen_verification_rejects_bound_file_tamper
tests/test_stage3_policy_revision_loopback.py::test_recursive_source_binding_covers_required_real_sources_and_blockers
tests/test_stage3_policy_revision_loopback.py::test_immutable_content_addressed_export_is_atomic_idempotent_and_read_only
tests/test_stage3_policy_revision_loopback.py::test_crash_before_atomic_rename_and_registry_replace_never_publish_partial_state
tests/test_stage3_policy_revision_loopback.py::test_lifecycle_serialization_pending_rollback_and_fresh_recovery_fail_closed
tests/test_stage3_policy_revision_loopback.py::test_each_quiescent_activation_condition_is_fail_closed[override0]
tests/test_stage3_policy_revision_loopback.py::test_each_quiescent_activation_condition_is_fail_closed[override1]
tests/test_stage3_policy_revision_loopback.py::test_each_quiescent_activation_condition_is_fail_closed[override2]
tests/test_stage3_policy_revision_loopback.py::test_each_quiescent_activation_condition_is_fail_closed[override3]
tests/test_stage3_policy_revision_loopback.py::test_each_quiescent_activation_condition_is_fail_closed[override4]
tests/test_stage3_policy_revision_loopback.py::test_each_quiescent_activation_condition_is_fail_closed[override5]
tests/test_stage3_policy_revision_loopback.py::test_each_quiescent_activation_condition_is_fail_closed[override6]
tests/test_stage3_policy_revision_loopback.py::test_one_episode_pins_revision_model_and_epoch_while_new_candidate_is_pending
tests/test_stage3_policy_revision_loopback.py::test_policy_epoch_gate_stale_drops_old_model_request_chunk_revision_and_epoch
tests/test_stage3_policy_revision_loopback.py::test_cross_revision_observation_ack_or_transition_is_quarantinable
tests/test_stage3_policy_revision_loopback.py::test_invalid_candidate_cannot_enter_pending
tests/test_stage3_policy_revision_loopback.py::test_cli_runs_twice_with_identical_canonical_report_and_full_fault_evidence
tests/test_stage3_policy_revision_loopback.py::test_cli_has_no_ros_robot_or_network_server_imports
tests/test_stage3_policy_revision_loopback.py::test_checked_in_report_is_schema_valid_and_canonically_self_signed
tests/test_stage3_protocol_and_publication.py::test_policy_epoch_stale_result_is_normal_drop
tests/test_stage3_protocol_and_publication.py::test_revision_lifecycle_enforces_episode_boundary_and_rollback
tests/test_stage3_protocol_and_publication.py::test_revision_identity_is_immutable_and_invalid_candidate_rejects
tests/test_stage3_recorded_ack_parity.py::test_synthetic_fixture_exercises_both_paths_but_cannot_open_formal_gate
tests/test_stage3_recorded_ack_parity.py::test_stage2_parity_path_calls_production_prepare_episode_once
tests/test_stage3_recorded_ack_parity.py::test_missing_raw_episode_fails_closed
tests/test_stage3_recorded_ack_parity.py::test_missing_ack_id_fails_closed
tests/test_stage3_recorded_ack_parity.py::test_missing_gripper_identity_fails_closed
tests/test_stage3_recorded_ack_parity.py::test_missing_recorded_fixture_is_schema_valid_blocked_report
tests/test_stage3_recorded_ack_parity.py::test_recorded_gripper_identity_mismatch_fails_closed
tests/test_stage3_recorded_ack_parity.py::test_online_td_calls_real_force_aware_macro_critic_interface
tests/test_stage3_recorded_ack_parity.py::test_real_phase2_actor_critic_image_range_and_task_feature_contract
tests/test_stage3_recorded_ack_parity.py::test_real_phase2_frozen_prefix_path_is_no_grad_detached_and_force_kv_once
tests/test_stage3_replay_and_credit.py::test_R_D_membership_payload_dedupe_uid_and_credit_rules
tests/test_stage3_replay_and_credit.py::test_credits_block_at_zero_and_round_trip_exactly
tests/test_stage3_replay_and_credit.py::test_mixed_sampler_origin_and_expert_mask_prevent_R_self_imitation
tests/test_stage3_runtime_contract.py::test_frozen_temporal_and_normalization_facts_match_development_config
tests/test_stage3_runtime_contract.py::test_single_owner_event_loop_rejects_cross_thread_call_and_latches_stop
tests/test_stage3_runtime_contract.py::test_runtime_clock_domain_is_required_for_every_comparable_timestamp
tests/test_stage3_runtime_contract.py::test_rational_selection_fault_injection_accepts_index_steps_2_3_4
tests/test_stage3_runtime_contract.py::test_fixed_anchor_rejects_future_phase_and_repeated_index
tests/test_stage3_runtime_contract.py::test_chunk_age_selected_index_and_dispatch_count_fail_closed
tests/test_stage3_runtime_contract.py::test_low_watermark_is_time_based_and_current_400ms_is_not_approved
tests/test_stage3_runtime_contract.py::test_refresh_must_be_pinned_before_entering_additional_headroom
tests/test_stage3_runtime_contract.py::test_stale_result_and_takeover_reset_revision_generations_are_rejected
tests/test_stage3_runtime_contract.py::test_pose_and_gripper_ack_fail_closed_on_missing_rejected_or_mismatch
tests/test_stage3_runtime_contract.py::test_duplicate_ack_is_idempotent_only_for_the_exact_same_payload
tests/test_stage3_runtime_contract.py::test_ack_before_command_after_deadline_and_after_stop_latch_is_rejected
tests/test_stage3_runtime_contract.py::test_ack_after_generation_flush_is_rejected[takeover]
tests/test_stage3_runtime_contract.py::test_ack_after_generation_flush_is_rejected[reset]
tests/test_stage3_runtime_contract.py::test_ack_after_generation_flush_is_rejected[revision]
tests/test_stage3_runtime_contract.py::test_partial_ack_cannot_create_accepted_action_or_transition
tests/test_stage3_runtime_contract.py::test_only_dual_ack_post_adapter_absolute7_becomes_transition_authority
tests/test_stage3_runtime_contract.py::test_no_safe_action_is_explicit_hold_and_unknown_command_outcome_is_stop
tests/test_stage3_runtime_contract.py::test_stage3_import_has_no_ros_network_robot_or_cuda_side_effects
tests/test_stage3_temporal_bridge.py::test_same_ack_can_causally_zoh_to_three_30hz_slots
tests/test_stage3_temporal_bridge.py::test_future_missing_rejected_and_300ms_interpretation_fails
tests/test_stage3_temporal_bridge.py::test_out_of_order_or_rejected_ack_fails_closed
```

Full regression command:

```bash
CUDA_VISIBLE_DEVICES='' \
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
PYTHONPATH=src:vendor/lerobot/src \
/home/rlc123/anaconda3/envs/forcesmolvla/bin/python \
-m pytest -vv tests/test_stage3_*.py
```

Result: **202 passed, 0 failed in 26.33 s**.

Additional checks:

| Check | Result |
|---|---|
| Python compile with bytecode redirected to `/tmp` | PASS |
| Development JSON parse | PASS |
| Evidence JSON schema and canonical SHA validation | PASS |
| `git diff --check` / staged `git diff --cached --check` | PASS |
| Stage-3 ROS/network/robot/CUDA import side-effect tests | PASS |
| CUDA initialized | false |
| Network server / ROS started | false |
| Robot connections / commands | 0 / 0 |

## Production blockers

1. Production gripper unchanged/no-op command and ACK provenance is unbound.
2. Full post-adapter absolute7 Pose plus gripper ACK closure is absent.
3. The CPU runtime ledger is not integrated into production serve/deploy/recorder/robot paths.
4. Production safe refresh rate, maximum chunk/action age, and maximum open-loop exposure remain unverified.
5. Controller internal servo and production transition commit frequencies remain unverified.
6. Recorded-live temporal and authoritative accepted-action parity evidence is absent.
7. Runtime ledger persistence and production resume semantics are absent from G5P checkpoints.

## Frozen evidence identity

Canonical JSON artifact: `artifacts/development/stage3/stage3_runtime_temporal_safety.v1.json`

Canonical payload SHA-256 (computed after removing `canonical_report_sha256` and canonical JSON serialization):

```text
280ccccaa076d4635c8c07f684ca138f86a28c3d58e474d5057ae602cf3a5857
```

Final gates:

```text
G7C1_CPU_PRIMITIVES=PASS
G7C1_FULL_REGRESSION=PASS
G7C1_EVIDENCE_FROZEN=true
PRODUCTION_CADENCE_BINDING_COMPLETE=false
RECORDED_LIVE_TEMPORAL_PARITY=false
G7_FORMAL_GATE_PASSED=false
G8_STARTED=false
CUDA_INITIALIZED=false
ROBOT_COMMAND_COUNT=0
ROBOT_EXECUTION_AUTHORIZED=false
PUSHED=false
```
