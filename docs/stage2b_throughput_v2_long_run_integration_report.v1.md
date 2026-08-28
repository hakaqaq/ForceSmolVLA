# Stage-2B throughput-v2 long-run integration report

## Outcome

Candidate B has been integrated into an append-only bounded-cache worker and passed the 210-cycle data-only cache stress test, fresh-process exact-resume test, and three-repeat C64/C96/C128 benchmark. No 210-cycle long-run was started.

`candidate_B_prefix_cache` means the fastest previously screened implementation that satisfied the frozen numerical/training contracts. It does **not** mean formal long-run had already been integrated, nor that C128 was globally optimal. The formal retest selected Actor B24 / Critic C64 for minimum fixed-Actor-pass wall time.

## Frozen implementation boundary

- Parquet files are materialized at most once per process.
- Dual-camera images use an 8 GiB bounded decoded LRU with eight CPU decode workers.
- Cal-QL M=2 candidates share only frozen VLM PrefixContext/KV; each keeps independent fixed Flow noise and the full N=10 integration.
- No trainable Force/Action representation is cached.
- Hot-loop full-model SHA, development Polyak tensor audit, `gc.collect()` and `torch.cuda.empty_cache()` are removed; full state is audited only at preflight/boundary scope.
- Flow inference subbatch remains 4. Rejected B8/B16/grouped-Flow candidates remain rejected; no tolerance was relaxed.

## 210-cycle bounded-cache stress

- Unique row references: 31,351
- Unique images: 20,226
- Decoded working set: 17.36 GiB
- Cache limit / peak: 8.00 / 8.00 GiB
- Peak process RSS: 36.54 GiB
- Hit rate: 44.84%; evictions: 237,511
- Cold start: 31.29 s; steady data-only latency: 1.047 s/cycle

The decoded cache is strictly bounded. Peak RSS remains a deployment-planning consideration because materialized compressed Parquet payloads are resident; it did not grow beyond the measured stable bound during the full 210-cycle draw plan.

## Exact resume

Branch A ran two continuous cycles. Branch B ran one cycle, saved an audit-only recovery checkpoint, strict-loaded it in a new process, and ran cycle 2. All canonical comparisons passed at `rtol=0`, `atol=0`: rows, Flow noise/actions, loss and Q traces, gradients, optimizer/target deltas, sampler/RNG state, and final model state. Cycle-2 digest: `16ef0eb5b8148efcbea4fbb8c7e9fc74214b8befe3b52813898055d38cf47b81`.

## Critic batch retest

| Critic batch | Mean cycle (s) | Median | P95 | cycles/hour | Actor transitions/s | TD transitions/s | Cal-QL transitions/s | Peak reserved GiB | Peak RSS GiB |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 64 | 30.184 | 30.271 | 30.813 | 119.269 | 0.7952 | 4.2409 | 4.2409 | 17.65 | 32.62 |
| 96 | 44.723 | 44.494 | 45.432 | 80.496 | 0.5367 | 4.2936 | 4.2936 | 17.48 | 34.95 |
| 128 | 58.654 | 58.389 | 59.813 | 61.377 | 0.4093 | 4.3654 | 4.3654 | 17.89 | 36.00 |


C64 is selected because it minimizes fixed Actor-pass wall time while all numerical, ownership, mask, frozen-hash, VRAM and ActionContract checks pass. C128 exposes twice as many TD rows and twice as many independent Cal-QL rows per cycle; that exposure tradeoff must be considered before authorizing a long run. The report does not combine TD and Cal-QL memberships into an ambiguous `2 × critic_batch` count.

## Projected budgets (mean steady-state throughput)

- 0.5 Actor pass, 210 cycles: 1.761 h
- 1.0 Actor pass, 420 cycles: 3.521 h
- 2.0 Actor passes, 840 cycles: 7.043 h

For C64, 0.5 pass is 5,040 Actor rows, 26,880 TD memberships, and 26,880 independent Cal-QL memberships. These are projections, not authorization.

## Final state

```text
THROUGHPUT_V2_BENCHMARK = pass
CANDIDATE_B_LONG_RUN_INTEGRATION = pass
BOUNDED_CACHE_210_CYCLE_PREFLIGHT = pass
EXACT_RESUME = pass
FINAL_ACTOR_BATCH = 24
FINAL_CRITIC_BATCH = 64
FINAL_FLOW_INFERENCE_SUBBATCH = 4
FINAL_CYCLES_PER_HOUR = 119.268677
PROJECTED_0_5_ACTOR_PASS_RUNTIME = 1.761_hours
PROJECTED_1_0_ACTOR_PASS_RUNTIME = 3.521_hours
PROJECTED_2_0_ACTOR_PASS_RUNTIME = 7.043_hours
OLD_CYCLE105_CHECKPOINT_ALLOWED_AS_PARENT = no
LONG_RUN_RECIPE_PROPOSED = yes
LONG_RUN_AUTHORIZED = no
LONG_RUN_STARTED = no
ROBOT_EXECUTION_AUTHORIZED = false
```

Source manifest SHA-256: `81b0e74c817ed751c12b1482321ece67f7702ff374730282c284971053350275`  
Final artifact SHA-256: `6446ff21ee275e77f5072129fd1022dd37c0a0f776f8371571f0ba2ef542718d`
