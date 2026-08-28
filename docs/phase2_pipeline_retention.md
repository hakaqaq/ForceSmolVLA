# Phase-2 pipeline retention

This document records the append-only local cleanup performed after the
Phase-2 development pipeline was completed. It is a storage-retention change,
not an algorithm, checkpoint-weight, inference, ActionContract, or safety
change.

## Retained final pipeline inputs

- Stage-1 r5 parent checkpoint and manifests.
- Reward Classifier checkpoint and frozen Detector/G1 provenance.
- G7-A-r2 critic-warmup parent as compact training provenance.
- Self-contained cycle-210 evaluation-smoke checkpoint.
- Cycle-210 evaluation deployment binding.
- Source code, tests, reports, schemas, and lightweight SHA-bound manifests.

The public serving checkpoint is:

```text
artifacts/development/stage2/
stage2b_cycle210_evaluation_smoke_checkpoint.v1
```

Its model SHA-256 is:

```text
e24c1d6bb0a778921659514ac47c692b952178aa39af2601ccf0fc32bf94774d
```

The corresponding binding is:

```text
artifacts/development/live/
task2_cycle210_evaluation_smoke_binding.v1.json
```

with SHA-256:

```text
6f15f33aedbf4327388012dc7a0418de09f05ba070833ac95c092b95104471d5
```

## Removed local payloads

- G5 single-cycle model/optimizer/scheduler/RNG payloads for v1 and v2.
- G6 exact-resume branch model/optimizer/scheduler/RNG payloads for v1 and v2.
- G7-B joint-smoke model/optimizer/scheduler/RNG payloads.
- Throughput-v2 temporary benchmark and exact-resume work directories.
- Interrupted and superseded Stage-2B long-run work directories.
- Superseded cycle-105 checkpoint.
- Python bytecode, pytest, and Ruff caches.

Git-tracked lightweight evidence under those directories was retained. The
cleanup released approximately 46 GiB locally.

The ignored `stage2b_throughput_v2_half_pass_run.v1` run directory was removed
as a unit, including its cycle-210 optimizer/recovery state. Therefore:

```text
cycle210_evaluation_and_serving = available
cycle210_exact_training_resume = unavailable_locally
```

The evaluation-smoke checkpoint is still development-only. This cleanup does
not authorize deployment release, online updates, or robot execution.
