# Stage-2 throughput-v2 timing audit

The audit passes. The slow cycle is caused by the synchronous data path and many
small policy Flow batches, not by a training-semantics or GPU fault.

The selected B24/C128 benchmark used synchronized wall-clock timing. Cycle,
Flow, Actor, Critic, optimizer, and Polyak GPU regions are bounded by
`torch.cuda.synchronize()`; CUDA Events are not used. The data timers begin only
after the preceding GPU region has synchronized, and all current H2D copies use
blocking `.to(device)` without pinned memory or `non_blocking=True`. Therefore
the measured 60.8407 seconds is real synchronous CPU/I/O/preprocessing/H2D wait,
not queued CUDA work charged to data loading.

The old instrumentation wraps the whole `build_batch()` call. It cannot
retroactively split dataset indexing, Parquet reading, PIL decoding, transforms,
normalization, collate, and H2D into trustworthy individual seconds. Those
fields remain explicitly `not_individually_instrumented`; throughput-v2 must add
timers at their actual boundaries.

## Per-cycle static work

- Five independent `build_batch()` calls: TD and Cal-QL for each of two Critic
  updates, then one Actor batch.
- 1,072 current/next observation instances and 2,144 two-camera PIL decodes.
- The Actor H=50 action fetch requests future rows through the same raw-row path,
  which currently reads unused camera columns for those future action-only rows.
- 1,536 Cal-QL candidates: 512 random, 512 current-policy, 512 next-policy.
- Typically 1,302–1,304 policy action chunks, 321 Flow subbatches, and 3,210
  Euler velocity evaluations at `subbatch=4`.
- Prefix is computed once per Flow subbatch and reused for all ten Euler steps;
  it is not recomputed ten times. Typical total is 322 prefix prefills including
  FM. M=2 candidate expansion creates 512 avoidable duplicate prefix instances,
  and the same 24 Actor observations are prefetched independently for FM and
  Actor-Q.

`num_workers=0` is descriptive but not currently a performance knob: the
runtime path calls `build_batch()` directly and does not use a PyTorch
`DataLoader`. Candidate A therefore requires a real persistent prefetch/staging
path rather than changing one YAML value.

All counts preserve ActionContract-v2, H=50, N=10, B24/C128, 2:1 updates, M=2,
losses, seeds, sample ordering, and the Frozen-VLM TrainabilityContract.
