---
type: "code"
date: "2026-09-08T16:49:24.338311+00:00"
question: "Convert raw task4 recordings to LeRobot v3 and run ForceSmolVLA offline SFT with task2/task3 parameters, saving only step 010000"
contributor: "graphify"
outcome: "useful"
source_nodes: ["convert_franka_raw_to_lerobot_v3", "train_forcesmolvla_sft"]
---

# Q: Convert raw task4 recordings to LeRobot v3 and run ForceSmolVLA offline SFT with task2/task3 parameters, saving only step 010000

## Answer

Converted 97 of 100 task4 episodes into datasets/task4_lerobotv3 (134811 frames); excluded 3 episodes by existing quality gates. Validation and action-target parity passed. Restored the exact missing P8 readiness checkpoint by deterministic replay with model and artifact hashes matching the preserved report. Completed 10000 SFT updates over 40000 samples and saved only outputs/task4/sft/checkpoints/forcesmolvla_sft_step_010000. Final validation L_flow 0.2073232532; best 0.2033908963 at step 9500. Relevant CPU tests: 57 passed.

## Outcome

- Signal: useful

## Source Nodes

- convert_franka_raw_to_lerobot_v3
- train_forcesmolvla_sft