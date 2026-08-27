# Stage-2 G4 loss implementation report

Status: `PASS_DEVELOPMENT_ZERO_UPDATE_PREFLIGHT` on `NVIDIA GeForce RTX 4090 D`.

## Implemented formulas

- Twin-Q TD uses every detector-G1 train transition and exactly `r + stored_discount * min(Q1_target,Q2_target)`. Terminal rows are filtered before next-Actor/target-Q evaluation; their measured calls were `{'actor': 0, 'q1': 0, 'q2': 0}`.
- Conservative loss is a **Cal-QL-style finite-candidate conservative objective**, not importance-corrected exact CQL. Its normalized LSE has `3M+1=7` test-only terms, dataset Q exactly once, and the MC lower bound only on the `3M=6` candidate values.
- Actor-Q is `-mean((Q1+Q2)/2)` using online critics. TCP6 remains differentiable and the decoded gripper gradient is exactly zero; full H=50 Flow Matching retains a nonzero gripper gradient.
- Router balance and z terms are each computed once per Actor objective. No target Actor exists.

## Numerical and gradient evidence

NumPy/PyTorch fp32 maximum absolute error: `0` (tolerance `2e-06`). The bf16 Actor to fp32 Critic interface and all loss values were finite. Three fixed observation/noise/candidate calculations were exact-repeatable.

Critic backward produced nonzero gradients in both online critics, none in the Actor, targets, or frozen critic backbones. Actor-Q reverse-mode gradients reached the Vision/VLM, Flow projections, Action Expert, ForceMLP, Fusion, an actually routed expert, Force Action Adapter, and router; critic parameter gradients stayed absent.

## Data access and immutability

Only `10075` automatic detector-G1 train rows were returned. Validation/test transition reads were `0/0`; manual G1/manual label opens were `0/0`. Actor, critics, r5, classifier, Stage-1 P4-P9 artifacts, dataset tree, G1, and prior artifacts were byte/state identical before and after.

## Still unapproved for G5

`beta`, `eta`, Cal-QL `alpha`, candidate count `M`, temperature `T`, clip min/max, Polyak `tau`, and the random proposal distribution remain `null/unapproved`. G5 must separately approve training-cycle scheduling, loss coefficients/proposals, optimizer ownership, gradient accumulation, target-update timing, checkpoint/resume semantics, and train-only sampling.

Development limitation remains: reward-model training overlap is true, unbiased reward-model evaluation is false, and these all-success demonstrations do not constitute formal offline-RL validation with failures.
