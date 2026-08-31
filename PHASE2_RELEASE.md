# ForceSmolVLA v2.1.1 — Phase-2 development release

`v2.1.1` is an artifact-retention cleanup of the tested `v2.1.0` Stage-2
development pipeline. It does not change model architecture, training math,
ActionContract-v2, public inference, or safety behavior. The released training
boundary remains:

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
`evaluation_smoke_only` local artifact. The retained provenance of its original
training checkpoint records `NOT_FOR_DEPLOYMENT` and
`NOT_FOR_POLICY_EVALUATION`; neither artifact is a public model release or an
approved long-run training parent.

## Local retention after Phase-2 completion

The local cleanup removes heavyweight G5/G6/G7 smoke and exact-resume payloads,
failed/interrupted run directories, throughput benchmark work checkpoints,
Python test caches, and the superseded cycle-105 boundary. Git-tracked reports,
tests, and manifests remain in the repository.

The self-contained cycle-210 evaluation-smoke checkpoint and its dedicated live
binding remain the deployment-smoke inputs. The original cycle-210 training
checkpoint's retained provenance records its original `NOT_FOR_DEPLOYMENT` and
`NOT_FOR_POLICY_EVALUATION` status. Its recovery payload has been pruned, so
exact optimizer-state resume from that boundary is no longer available locally.
See
The current retention and recovery rules are maintained in
[`docs/forcerft_end_to_end_user_guide.md`](docs/forcerft_end_to_end_user_guide.md#12-数据和-checkpoint-保留策略).
The historical machine-readable cleanup record remains
[`phase2_pipeline_retention.v1.json`](artifacts/development/stage2/phase2_pipeline_retention.v1.json).
