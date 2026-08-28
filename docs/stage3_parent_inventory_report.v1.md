# Stage-3 Parent Inventory Audit v1

Audit time: `2026-08-28T12:27:05+08:00`

This was a read-only parent-state and source audit against `origin/main` commit `d00384330eed01ddd0dfe3bf435e4545fb679a71` and release `v2.1.1`. No Phase-2 reconstruction, Stage-3 training, online collection, or robot execution was started.

## Outcome

The applicable classification is **Option B**:

- The complete cycle210 learner checkpoint is no longer present on this machine or in the searched local backup locations.
- The retained cycle210 evaluation checkpoint is a valid Actor inference artifact, not a complete learner parent.
- G7A-r2 is present and strictly loadable under its original contract, including online/target Twin-Q, Critic optimizer and scheduler, RNG, sampler, and its frozen r5 Actor binding.
- G7A-r2 remains historically marked `NOT_AN_APPROVED_LONG_TRAIN_PARENT`. It is sufficient to propose a deterministic Phase-2 reconstruction, but this audit does not authorize or execute it.
- Stage-3 remains unimplemented. There is no candidate Stage-3 parent binding until cycle210 is recovered or separately approved for deterministic reconstruction.

Therefore:

```text
STAGE3_PARENT_OPTION = B
STAGE3_PARENT_BINDING = blocked
STAGE3_IMPLEMENTATION_STATUS = not_implemented
```

## Repository identity

The local checkout, local `origin/main`, remote `origin/main`, and peeled `v2.1.1` tag all resolve to:

```text
d00384330eed01ddd0dfe3bf435e4545fb679a71
```

The annotated tag object is `0498e8fd84b65e267f27ec8c52514d15cac3b315`. The GitHub release exists at `https://github.com/hakaqaq/ForceSmolVLA/releases/tag/v2.1.1` and is marked as a prerelease.

## Cycle210 full learner checkpoint

The audit searched the repository, `/home/rlc123`, `/mnt`, `/media`, `/tmp`, and the local trash directory for `checkpoint_cycle_000210`, its model states, and known manifest names. No complete checkpoint directory or backup payload was found. No external backup volume was mounted.

The missing state includes:

```text
models/actor_state.pt
models/q1_state.pt
models/q2_state.pt
models/q1_target_state.pt
models/q2_target_state.pt
optimizer states
scheduler state
RNG state
sampler state
cycle cursor
training manifest payload
```

Historical expected values remain:

```text
cycle210 checkpoint tree SHA256 =
b514b50d118cb3edaa6e5e135e1a2cf7340062d11c16cb58bed437581c082e08

cycle210 manifest SHA256 =
1d1644ed6b51a1a61cb66e0b6a4ae6831bcced025530dbb5583b5d90ea4083fa
```

A byte-identical copy of the historical manifest survives under the evaluation export's `provenance/` directory. It describes the removed learner checkpoint; it is not the learner checkpoint itself. Since the payload is absent, the full tree SHA cannot be recomputed and its match status is `not_available`.

## Cycle210 evaluation checkpoint boundary

The retained artifact is:

```text
artifacts/development/stage2/
stage2b_cycle210_evaluation_smoke_checkpoint.v1
```

Its artifact manifest and evaluation-scope validation pass. The relevant bindings are:

```text
source Actor state SHA256 =
73b35435e943823bb88c54decf68ce4bf08f39100999c5770b071aa76c3cf4c3

runtime model SHA256 =
e24c1d6bb0a778921659514ac47c692b952178aa39af2601ccf0fc32bf94774d
```

The export records 574/574 Actor tensors, zero missing or unexpected keys, and exact parity for 347 frozen-VLM tensors. It intentionally exports no Critic, optimizer, scheduler, RNG, or sampler state and explicitly sets `training_parent_allowed=false`. Its capability is therefore limited to Actor inference. It must not be promoted to a learner checkpoint, and old or freshly initialized Critics must not be silently attached to it.

## G7A-r2 payload and strict load

The retained checkpoint is:

```text
artifacts/development/stage2/g7a_r2_critic_warmup_checkpoint
```

Its complete tree contains 18 files and hashes to:

```text
f8c08b9058d173211a7306d370a97a848bfc1f7569ac52e6cc88baacff0c0d40
```

The checkpoint manifest SHA is:

```text
2e0902076cb12a1391613230679730d035155528c9be01bd17dce960d5e707f7
```

Manifest closure and the 25-file source closure both validate. The required payloads exist and match their manifest hashes:

| Payload | SHA-256 |
|---|---|
| `models/q1_state.pt` | `13f464ea2c07184dd6a564af6743872af2e2c06cbf21e23dfe29f20363bb4a66` |
| `models/q2_state.pt` | `86022ddc6b78ea06919c0d02d7a7b7ee718d958df35f509d2a53ce9584ed53ec` |
| `models/q1_target_state.pt` | `da3e80a552c65be49e5f9d7f911cd1c3b2e466e4c1402a71181a5de4d24e62ea` |
| `models/q2_target_state.pt` | `a113fe238acbbf6921eb736a988c2dfd6b8074d01369786c860961ff6a1e6810` |
| `optimizers/critic_optimizer_state.pt` | `019ea251b7e36240840aba2e95c5e9c5951da95661d43589d1a0a35e8b2e99ce` |
| `schedulers/critic_scheduler_state.pt` | `d9054e582ab55c941c2ae5f0c386e0df4efe560d1b35bb1f358800b0cae75f45` |
| `state/rng_states.pt` | `026c6f8b9ee72a1c5f6b3674b743d8e815552d15bd533fefedd8fe8338c2e56b` |
| `state/sampler_states.pt` | `a77956ed073cdfcb367ea3f3e42c91435d1d7d8dd1ac9619b44e55257d0558e8` |

G7A-r2 does not duplicate the Actor state. Its manifest binds the Actor to the frozen Stage-1 r5 tree, whose recomputed tree SHA is:

```text
01665f899ad34e4ba048a46bcffe3a0f819fc0f7e42c6d0c9a2db662a3665379
```

The Actor binding records 574 parameter tensors with digest `ba425f642ac5bd4a2fd199262d43d71c5a337763af76e785b3e08ad41da326d5`. The checkpoint also binds the normalizer, ActionContract-v2, and automatic detector-G1 transition view.

A fresh-process, read-only strict restoration was performed using the existing G7A verification worker. It restored model, Critic optimizer, scheduler, sampler, and RNG state; RNG was restored last; no random draws, sampler draws, optimizer updates, parameter updates, or dataset reads occurred afterward. ActionContract-v2 verification passed. This establishes:

```text
G7A_R2_PAYLOAD_AVAILABLE = yes
G7A_R2_STRICT_LOAD = pass
```

It does not change G7A-r2's historical authorization markers:

```text
DEVELOPMENT_G7A_CRITIC_WARMUP_ONLY
NOT_AN_APPROVED_LONG_TRAIN_PARENT
NOT_FOR_POLICY_EVALUATION
NOT_FOR_DEPLOYMENT
```

## Proposed deterministic Phase-2 reconstruction — not executed

If separately authorized, Option B would reconstruct cycle210 from G7A-r2 with the frozen historical recipe:

```text
Actor batch = 24
Critic batch = 64
Flow inference subbatch = 4
Critic : Actor = 2 : 1
H = 50
Flow N = 10
Cal-QL M = 2
eta = 3.0
beta = 1.0
Cal-QL alpha = 0.1
Polyak tau = 0.005
joint cycles = 210
```

Reference command, provided only as a proposal and **not executed**:

```bash
PYTHONHASHSEED=42 CUBLAS_WORKSPACE_CONFIG=:4096:8 \
PYTHONPATH=src:vendor/lerobot/src:tools:. \
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
/home/rlc123/anaconda3/envs/forcesmolvla/bin/python \
tools/run_stage2b_throughput_v2_authorized_half_pass.py
```

Expected compute is about 1.8 hours. Historical process RSS peaked near 36.55 GiB with an 8 GiB decoded-image cache. At audit time, host `MemAvailable` was about 44.57 GiB, leaving little margin above the historical startup gate.

A reconstruction must not be accepted solely because it finishes. It must reproduce the expected Actor, checkpoint-tree, and manifest SHA values; strictly load Actor/Q/target/optimizer/scheduler state; retain exact deterministic sample identities, Flow noise, sampler and RNG trajectories; keep all numerical states finite; and preserve ActionContract-v2.

One evidence limitation must remain explicit: the raw historical per-cycle trace was removed during retention cleanup. A future reconstruction can deterministically regenerate its draw plan from the frozen parent/config/seeds, but cannot compare that trace to a retained original trace file. To avoid confusing reconstructed evidence with the deleted original checkpoint, a separately approved append-only output identity should be used.

## Stage-3 implementation inventory

No Stage-3 learner implementation exists in this release. The configuration explicitly records:

```text
online_hil_vlm_frozen.implementation_status = not_implemented
```

The source contains low-level support for freezing Vision/SmolVLM parameters and keeping the VLM in eval mode. That is scaffolding, not a Stage-3 TrainabilityContract or training system.

| Capability | Actual status | Nearby code that must not be misclassified |
|---|---|---|
| Stage-3 TrainabilityContract | `not_implemented` | Low-level frozen-VLM mode scaffolding only |
| ACK-aligned transition contract | `not_implemented` | Deployment-side controller ACK validation only |
| Intervention/expert labeling | `not_implemented` | Current policy action metadata uses `intervention=false` |
| Online/demo mixed replay | `not_implemented` | No ForceSmolVLA replay mixer is wired |
| Pure online TD-only Critic loss | `not_implemented` | Existing Twin-Q loss still constructs Cal-QL candidates even when alpha is zero |
| Expert-masked FM loss | `not_implemented` | No expert mask path exists |
| Versioned policy publication | `not_implemented` | Metadata fields are not a learner publication protocol |
| Sample-credit backpressure | `not_implemented` | No credit accounting/backpressure path exists |
| Stage-3 exact resume | `not_implemented` | Stage-2 exact resume does not close online replay/publication state |
| Fake Actor/Learner loopback | `not_implemented` | Generic vendored RL utilities are not integrated |
| Single-GPU coexistence preflight | `not_implemented` | Documented only as a future stress-test candidate |

In particular, `compute_twin_q_critic_loss()` cannot be described as a pure online TD-only implementation merely by setting `calql_alpha=0`: candidate tensors are still required and Cal-QL candidate Q values are still constructed before their weighted contribution is zeroed.

The repository architecture query confirmed that the connected live path is inference/deployment and controller-ACK validation; it did not reveal an integrated Stage-3 learner graph. Direct source inspection was used to distinguish those adjacent mechanisms from actual Stage-3 implementation.

## Final status

```text
REPOSITORY_HEAD = d00384330eed01ddd0dfe3bf435e4545fb679a71

CYCLE210_EVALUATION_ACTOR_AVAILABLE = yes
CYCLE210_FULL_TRAINING_STATE_AVAILABLE = no
CYCLE210_FULL_TREE_SHA_MATCH = not_available

G7A_R2_PAYLOAD_AVAILABLE = yes
G7A_R2_STRICT_LOAD = pass

STAGE3_PARENT_OPTION = B
STAGE3_PARENT_BINDING = blocked
STAGE3_IMPLEMENTATION_STATUS = not_implemented

PHASE2_RECONSTRUCTION_STARTED = no
STAGE3_TRAINING_STARTED = no
ONLINE_DATA_COLLECTION_STARTED = no
ROBOT_EXECUTION_AUTHORIZED = false
```
