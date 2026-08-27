# Stage-1 / Stage-2 Trainability and Batch Scaling

## Outcome

Frozen-VLM TrainabilityContract v2 and its GPU preflight **passed**. The recommended offline throughput configuration is Actor B24 / Critic B128. Actor B32 and Critic B256 each OOMed in all three independent processes and are rejected. No benchmark state was checkpointed or retained.

The same-GPU online-coexistence candidate is B24/B64, pending a separate concurrent stress test. It leaves 6.45 GiB versus 4.24 GiB for B24/B128. Neither authorizes online or robot execution.

## TrainabilityContract v2

- Frozen: Vision Encoder, SmolVLM/token embeddings, state-to-prefix projection; always eval and excluded from the Actor optimizer.
- Trainable: ForceMLP, Fusion/MoE/router, Force Action Adapter, Action Expert and Action I/O.
- Frozen/trainable Actor parameters: 350,196,864 / 155,423,477.
- Exact frozen/full forward parity before updates: pass; frozen parameter/buffer hashes after temporary updates: unchanged.
- Prefix representation/cache detached; Force K/V prepared once per chunk.
- ActionContract-v2 and public execution behavior are unchanged.

Existing full-Actor G7-B remains `historical_valid_development_mechanics`; its checkpoint is not a long-run parent.

## Stage-1 scaling

| Batch | Median samples/s | Peak reserved GiB | Decision |
|---:|---:|---:|---|
| 4 | 5.047 | 8.62 | measured |
| 8 | 6.979 | 12.41 | recommended |
| 16 | 7.042 | 20.15 | measured |

B16 improved median throughput only 0.91% over B8, below the frozen 5% rule, while reaching 20.15 GiB. Therefore B24/B32 were not run. At B8, 40,000 exposures are projected to take 1.59 h. Equal sample exposure does not make the optimization trajectory equivalent; LR, warmup, scheduler and batch-local MoE losses require separate approval.

## Stage-2 scaling

### Actor-only

| Batch | Median transitions/s | Peak reserved GiB | Result |
|---:|---:|---:|---|
| 4 | 2.133 | 5.61 | PASS |
| 8 | 2.514 | 8.15 | PASS |
| 16 | 2.991 | 12.82 | PASS |
| 24 | 3.200 | 17.62 | PASS |
| 32 | — | — | OOM 3/3 |

### Critic-only

| Batch | Median transitions/s | Peak reserved GiB | Result |
|---:|---:|---:|---|
| 32 | 1.373 | 6.01 | PASS |
| 64 | 1.694 | 9.93 | PASS |
| 128 | 2.176 | 17.77 | PASS |
| 256 | — | — | OOM 3/3 |

### Joint combinations

| Actor/Critic | Cycle s | Actor tr/s | Critic tr/s | Reserved GiB |
|---|---:|---:|---:|---:|
| 24/128 | 125.70 | 0.190 | 2.030 | 19.74 |
| 16/128 | 125.30 | 0.128 | 2.050 | 19.17 |
| 24/64 | 84.26 | 0.289 | 1.541 | 17.54 |

At the same historical B4/C16 layout, Frozen-VLM uses 31.16 s/cycle versus the supplied 41.58 s/cycle full-Actor baseline: 1.33x speedup and 18.4% lower reserved memory. Steady-state timing excludes public audit, checkpoint, process load and report generation. Prefix timing is embedded in the Flow components and is separately identified in the JSON artifact to prevent double counting.

## Frozen-VLM gradient scale

At eta=10, beta=1 on the selected B24 physical Actor batch:

- raw `||g_Q|| / ||g_FM||`: 0.019085
- weighted ratio: 0.190849
- cosine similarity: -0.097523
- TCP6 Q gradient: 0.00029399127
- gripper Q gradient: exact 0.0
- gripper FM gradient: 0.0011514535
- one discarded step normalized TCP drift: 0.015724
- binary gripper change rate: 0.000
- raw-gripper out-of-public-tolerance rate before/after: 0.000/0.000

Because eta=10 yields a 0.19085 weighted ratio and mildly opposing global cosine, the proposed next approval value is eta=3 (analytical expected ratio 0.05725); eta=3 was not run or approved.

## Actor-transition budgets (B24/C128)

Budget projections use the reported average steady-state throughput of 28.5457 cycles/hour (about 126.11 seconds/cycle), while all original mean, median and range measurements remain unchanged.

| Target Actor passes | Cycles | Actor exposure | Critic exposure | Critic passes | Projected time |
|---:|---:|---:|---:|---:|---:|
| 0.5 | 210 | 5040 | 53760 | 5.34 | 7.36 h |
| 1.0 | 420 | 10080 | 107520 | 10.67 | 14.71 h |
| 2.0 | 840 | 20160 | 215040 | 21.34 | 29.43 h |

The proposed starting budget is 0.5 Actor pass, followed by review; 1 and 2 passes are projections, not convergence claims. G7-A's 256 B16 Critic updates equal 4,096 samples (0.407 transition pass); whether to accept that parent as-is or top it up toward 0.5 pass needs explicit approval.

## ConRFT boundary

ConRFT batch sizes, 20k pretraining steps, and online `steps_per_update=50` are not directly transferable. ConRFT updates a lighter consistency policy/critic while freezing larger VLA representations. ForceRFT performs full-model force-conditioned behavior adaptation in Stage-1, then frozen-backbone value-guided force-action refinement with native N=10 Flow in Stage-2. `steps_per_update=50` is an asynchronous learner publication cadence, not a requirement to train ForceRFT 50 times per new batch. Stage-2 Twin-Q does not update VLM parameters.

## Stop state

`LONG_RUN_RECIPE_PROPOSED=yes`; `LONG_RUN_AUTHORIZED=no`; `LONG_RUN_STARTED=no`; `ROBOT_EXECUTION_AUTHORIZED=false`. Validation/test/manual G1/manual labels/Reward Classifier reads were all zero.
