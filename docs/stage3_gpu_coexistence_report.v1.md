# Stage-3 G7B targeted time-sliced timestamp report

Result: `PASS`; this means the targeted cold-process-swap workload completed and its timestamp evidence passed consistency checks. It does not approve a production topology or cadence.

- Base G7A canonical SHA-256: `1fd51e03eaa57c10412f4b38e2c4671edcd3abb474f7cec9ac45a800e4dacadb`.
- `TIME_SLICED_TOPOLOGY=cold_process_swap`; `RESIDENT_TIME_SLICING=NOT_RUN`; `REAL_RESET_HOME_WINDOW_USED=false`.
- `G7P_RESULT_SEMANTICS=PASS_MEANS_MEASUREMENT_COMPLETED_ONLY`.
- `inter_phase_gap_ms=1000`; `INTER_PHASE_GAP_IS_EXECUTION_BUDGET=false`. The gap is a coordinator delay, not learner execution budget.
- Clock: `CLOCK_MONOTONIC` via `time.monotonic_ns()`; Linux processes on the same boot share a comparable monotonic clock domain.
- Production reset/Home window, request cadence, action-queue low-watermark, refresh cadence, staleness safety limit, deadline, and topology remain unverified/unbound.
- H=50 is the model output horizon, not authorization to execute the full chunk open-loop for 1.67 seconds.

## Preserved G7A components

| Component | Before digest | After digest | Unchanged |
| --- | --- | --- | --- |
| `inference_only` | `5af3dc87a30cc7816e6d7d634639f0412f74b04b0cd98843111a950724b27ccc` | `5af3dc87a30cc7816e6d7d634639f0412f74b04b0cd98843111a950724b27ccc` | `true` |
| `learner_only` | `3cebeaf4ffc4350fa0939bd4c3e946344f52e5bf10b3696f0ddb3582b13f96b6` | `3cebeaf4ffc4350fa0939bd4c3e946344f52e5bf10b3696f0ddb3582b13f96b6` | `true` |
| `concurrent` | `7f83588bd51d9d05c8e17b0b0e72e4e43994d8d9aaeaf50b73fef90f825eb618` | `7f83588bd51d9d05c8e17b0b0e72e4e43994d8d9aaeaf50b73fef90f825eb618` | `true` |
| `environment` | `cc0685cffd1298f3fc956ecb700aaf4d31059ee9fc940ac0c2e00d35851ba379` | `cc0685cffd1298f3fc956ecb700aaf4d31059ee9fc940ac0c2e00d35851ba379` | `true` |
| `checkpoint_bindings` | `85e52bcafe4829abb8bc9ab3b58b5b85e827404a16e94937707e92f97e37beb3` | `85e52bcafe4829abb8bc9ab3b58b5b85e827404a16e94937707e92f97e37beb3` | `true` |
| `action_semantics` | `9520e50873d117ba2de2b0f88e6a8db12a861d3d5cdc2279f3c0384879f61837` | `9520e50873d117ba2de2b0f88e6a8db12a861d3d5cdc2279f3c0384879f61837` | `true` |

## Absolute monotonic timestamps

| Event | monotonic ns |
| --- | ---: |
| `episode_last_release_ns` | 428507404187680 |
| `episode_queue_drained_ns` | 428508825104145 |
| `episode_worker_exit_ns` | 428511785669184 |
| `pre_learner_gap_start_ns` | 428511800723032 |
| `pre_learner_gap_end_ns` | 428512800874307 |
| `learner_spawn_requested_ns` | 428512801038011 |
| `learner_process_spawn_ns` | 428512801565747 |
| `learner_model_ready_ns` | 428528552091575 |
| `learner_warmup_cycle_start_ns` | 428543235767984 |
| `learner_warmup_cycle_end_ns` | 428559173094402 |
| `learner_measured_cycle_1_start_ns` | 428559173724306 |
| `learner_measured_cycle_1_end_ns` | 428574410943249 |
| `learner_measured_cycle_2_start_ns` | 428574410981694 |
| `learner_measured_cycle_2_end_ns` | 428589698374209 |
| `learner_measured_cycle_3_start_ns` | 428589698425331 |
| `learner_measured_cycle_3_end_ns` | 428605082096513 |
| `learner_worker_exit_ns` | 428607394031341 |
| `pre_resume_gap_start_ns` | 428607440331262 |
| `pre_resume_gap_end_ns` | 428608440492178 |
| `resume_requested_ns` | 428608440645703 |
| `resume_process_spawn_ns` | 428608441338557 |
| `resume_model_ready_ns` | 428623635720843 |
| `resume_first_request_release_ns` | 428625406527244 |
| `resume_first_result_ready_ns` | 428625835248587 |
| `resume_queue_drained_ns` | 428631365034142 |
| `resume_worker_exit_ns` | 428634269028614 |
| `resume_first_service_start_ns` | 428625438417346 |
| `resume_first_service_end_ns` | 428625835135400 |

## Recomputed durations

| Metric | Value |
| --- | ---: |
| `episode_drain_duration_ms` | 1420.916465 ms |
| `pre_learner_gap_actual_ms` | 1000.151275 ms |
| `learner_process_load_ms` | 15750.525828 ms |
| `warmup_cycle_ms` | 15937.326418 ms |
| `measured_cycle_ms` | [15237.218943, 15287.392515, 15383.671182] |
| `measured_cycles_total_ms` | 45908.282640 ms |
| `learner_phase_total_ms` | 94592.993330 ms |
| `pre_resume_gap_actual_ms` | 1000.160916 ms |
| `resume_model_load_ms` | 15194.382286 ms |
| `resume_first_inference_service_ms` | 396.718054 ms |
| `resume_spawn_to_first_ready_ms` | 17393.910030 ms |
| `episode_drain_to_resume_spawn_ms` | 99616.234412 ms |
| `episode_drain_to_first_resumed_action_ms` | 117010.144442 ms |
| `full_policy_unavailability_ms` | 117010.144442 ms |
| `MINIMUM_MEASURED_SINGLE_JOINT_CYCLE_MS` | 15237.218943 ms |
| `FULL_MEASURED_LEARNER_PHASE_MS` | 94592.993330 ms |
| `COLD_RESUME_SPAWN_TO_FIRST_READY_MS` | 17393.910030 ms |
| `FULL_COLD_SWAP_INTERRUPTION_MS` | 117010.144442 ms |
| `PRODUCTION_REQUIRED_RESET_HOME_WINDOW_MS` | UNVERIFIED |

`COLD_RESUME_SPAWN_TO_FIRST_READY_MS` is cold restart latency and is not mixed into steady-state inference latency. `FULL_MEASURED_LEARNER_PHASE_MS` spans learner spawn request through worker exit (including load, setup, warm-up, three measured cycles, and teardown); a single cycle is not presented as the full required reset/Home window.

## Targeted request accounting

| Stream | Scheduled | Completed | Dropped | Drop rate | p99 status |
| --- | ---: | ---: | ---: | ---: | --- |
| `time_sliced_episode` | 90 | 80 | 10 | 11.11% | `PROVISIONAL_INSUFFICIENT_SAMPLES` |
| `time_sliced_resume` | 30 | 30 | 0 | 0.00% | `PROVISIONAL_INSUFFICIENT_SAMPLES` |

## Conclusion boundary

- `G7P_TIME_SLICED_SEMANTICS=VERIFIED_COLD_PROCESS_SWAP_ONLY` and `G7P_EVIDENCE_FREEZE_ALLOWED=true`.
- `G7P_100MS_SYNTHETIC_GRID_FEASIBLE=false`; cold swapping can remove learner contention but cannot fix inference-only service time above 100 ms.
- `G7P_PROVISIONAL_TOPOLOGY_CANDIDATE=NONE`, `PRODUCTION_COLD_PROCESS_SWAP_APPROVED=false`, `PRODUCTION_REQUEST_CADENCE_VALIDATED=false`, `PRODUCTION_DEADLINE_VALIDATED=false`, `PRODUCTION_GPU_TOPOLOGY_APPROVED=false`, `G7_FORMAL_GATE_PASSED=false`.
- `PRODUCTION_CHECKPOINT_WRITES=0`; no server, ROS, robot, replay, publication, activation, or G8 path ran.

Canonical report SHA-256: `52bec6086eb43e9f44ddb090538c7b02ce32ef78f255350c127a0d50b64367a5`
