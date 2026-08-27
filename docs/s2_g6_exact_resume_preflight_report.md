# Stage-2 G6 fresh-process exact-resume preflight

Status: `G6_FRESH_PROCESS_EXACT_RESUME = pass`.

Branch A PID `2913833` rebuilt S0, replayed cycle 1 without loading G5 training state, matched the G5 S1 canonical training state exactly, saved a side-effect-free cycle-1 reference, and continued from the same in-memory objects through cycle 2. Branch B PID `2916409` used a new Python interpreter and CUDA context, strictly restored G5 S1 with RNG restoration last, and executed only cycle 2.

Cycle-2 canonical trace digest: `ed3f2cc2ae321a10bdf097aa15110cf8889199f425fdbf8baaa4e7afd62c85bf`. Final canonical training-state digest: `f529fc44fec16af9fcb857c51d59e05484ca9ee545ed8c9e55e825f23ef16812`. All tensor comparisons used original dtype contiguous bytes with `rtol=0`, `atol=0`, and `equal_nan=false`.

All five isolated negative-loader tests rejected before model updates, sampler draws, Polyak, or training-RNG consumption. Validation/test reads, manual G1/label opens, and Reward Classifier inference/updates were zero.

The branch checkpoints remain `DEVELOPMENT_EXACT_RESUME_TEST_ONLY`, `NOT_FOR_DEPLOYMENT`, `NOT_FOR_POLICY_EVALUATION`, and `NOT_AN_APPROVED_LONG_TRAIN_PARENT`. Cycle 3 and G7 did not run.

G6 proves only that cycle-boundary training state restores exactly under this frozen software/hardware configuration. It does not establish hyperparameter quality, Critic convergence, policy improvement, failure recovery, or reproducibility across GPUs/software versions.
