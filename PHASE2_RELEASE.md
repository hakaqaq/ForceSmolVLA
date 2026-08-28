# ForceSmolVLA v2.1.0 — Phase-2 development release

`v2.1.0` packages the tested Stage-2 development pipeline on top of the
`v2.0.0` ForceRFT implementation. The released training boundary is:

```text
Stage-1: full-model force-conditioned behavior adaptation
Stage-2A: Twin-Q warm-up with the Actor frozen
Stage-2B: frozen-backbone value-guided force-action refinement
```

## Included

- Frozen-VLM TrainabilityContract and detached prefix path.
- ActionContract-v2 mixed continuous TCP6 / discrete gripper Critic adapter.
- Mask-aware K=3 Twin-Q, target ownership, TD and finite-candidate Cal-QL-style
  losses, differentiable N=10 Flow Actor-Q guidance, and exact resume.
- Throughput-v2 bounded decoded-image cache and frozen-prefix reuse without
  changing H=50, N=10, Cal-QL M=2, deterministic sampling, or loss semantics.
- The C64 / Actor-B24 development long-run recipe and the durable cycle-210
  boundary: 210 joint cycles, 5,040 Actor transitions, approximately 0.5 Actor
  pass. The attempted continuation did not create a durable 1.0-pass result.
- A cycle-210 evaluation-smoke export contract and exact direct/public/HTTP
  offline parity evidence. This is not a deployment release.

The append-only artifacts preserve failed and superseded development evidence;
passing later stages does not rewrite earlier outcomes.

## Research and safety limits

```text
claim_scope = development_only
durable_actor_coverage = 0.5_pass
one_pass_completed = no
deployment_release_authorized = no
robot_execution_authorized = false
formal_detector_approved = no
reward_model_training_overlap = true
unbiased_policy_evaluation = false
```

All current demonstrations are successful demonstrations. Reward-classifier
training overlaps the automatic detector-G1 RL train episodes. The release
therefore does not establish unbiased reward-model generalization, failure
recovery, OOD conservatism, policy improvement, or production readiness.

## GitHub exclusions

Datasets, decoded-image caches, temporary benchmark traces, model weights,
Critic/target states, optimizer/scheduler state, RNG/sampler state, and local
training/evaluation checkpoints remain outside Git. Lightweight reports and
manifests retain their SHA-256 bindings to those local append-only artifacts.

In particular, the cycle-210 evaluation checkpoint is an
`evaluation_smoke_only` local artifact. Its original training checkpoint keeps
`NOT_FOR_DEPLOYMENT` and `NOT_FOR_POLICY_EVALUATION`; neither is a public model
release or an approved long-run training parent.
