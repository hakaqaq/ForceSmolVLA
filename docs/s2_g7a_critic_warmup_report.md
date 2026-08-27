# Stage-2 G7-A Critic-only warm-up report

Status: `G7A_CRITIC_WARMUP_MECHANICS = fail`.

G7-A stopped before update 1. The frozen r5 Actor produced at least one gripper candidate outside the existing `[-0.01, 0.095] m` fail-closed range while computing the prescribed update-0 diagnostics. The final attempt used CUDA-generated deterministic Flow noise, matching the G5 execution device; it passed the fixed train probe and then failed on the complete validation diagnostic before any optimizer or Polyak operation.

The enforced exception was:

```text
ValueError: model gripper candidate is outside the frozen [-0.01,0.095] m tolerance
```

No clipping, endpoint substitution, rejection/resampling, seed search, row exclusion, or tolerance expansion was used. Any of those would change the frozen G4/G5 action contract. The earlier implementation-only failure caused by an unsupported Flow-purpose label was corrected before this final attempt; it also occurred before any optimizer update.

## Counters and state

```text
CRITIC_WARMUP_UPDATES = 0
CRITIC_SCHEDULER_STEPS = 0
POLYAK_UPDATES_PER_TARGET = 0
ACTOR_UPDATES = 0
ACTOR_SCHEDULER_STEPS = 0
TARGET_ACTOR = none
G7B_STARTED = no
```

No G7-A training checkpoint or success output directory was created. The final attempt is preserved at `artifacts/development/stage2/g7a_failed_2963435/`; its log SHA256 is `609cdd064c4203785f3ceaab0a0b55bad6c75052afb1338bb0ba89f3ed94f911`.

The related regression suite passed `21/21` tests. The G2–G6 source closure still matches `stage2_source_manifest.v8_g6.json`; the frozen r5, Stage-1 dataset, G5 checkpoint and G6 output tree bindings recorded immediately before the final attempt are unchanged by the worker's write scope.

Test transitions/images, manual G1, manual labels, and Reward Classifier inference/updates were not accessed. Train and validation were read only for the prescribed update-0 diagnostics. No learning metric, update-256 diagnostic, or 32-batch Q/FM gradient-scale measurement exists because proceeding would have required bypassing the frozen gripper contract.

## Required status

```text
G7A_CRITIC_WARMUP_MECHANICS = fail
CRITIC_WARMUP_UPDATES = 0
ACTOR_UPDATES = 0
CRITIC_NUMERICALLY_STABLE = no
Q_GUIDANCE_SCALE_MEASURED = no
ETA_G7B_APPROVED = no
G7B_STARTED = no
LONG_RUN_AUTHORIZED = no
ROBOT_EXECUTION_AUTHORIZED = false
NEXT_ALLOWED_ACTION = request_G7A_gripper_candidate_contract_resolution
```

`CRITIC_NUMERICALLY_STABLE = no` means the required 256-update stability test was not completed; it does not assert that a NaN or Inf occurred. G7-B is not eligible while G7-A is failed.

The research limits remain `ALL_SUCCESS_DEMOS = true`, `REWARD_MODEL_TRAINING_OVERLAP = true`, and `UNBIASED_POLICY_EVALUATION = false`. No policy-improvement, failure-recovery, OOD-conservatism, or reward-generalization claim is made.
