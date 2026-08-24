# P4 v4.2 scoped-bf16 acceptance report

Date: 2026-08-20 (Asia/Shanghai)  
Status: `PASS_DEVELOPMENT_ONLY`  
Formal/production eligibility: `false`

## Approved scope

The hash-bound development configuration separates the P4 bf16 comparisons:

- `prefix_hidden_atol=0.3` applies only to full-vs-prefill hidden states at valid physical prefix tokens.
- `velocity_cache_atol=0.1` applies only to the five declared velocity/cache/10-step floating-point comparisons.
- `rtol=0.0` and the P4 fp32 threshold `atol=1e-5` are unchanged.
- Prefix layout, masks, physical length, invalid suffix velocity, cache append-crop restoration, and cache snapshots remain exact contracts.
- P8 was not changed by this approval. Formal P4/P8 bf16 thresholds remain `null` and `unapproved`.

Acceptance config SHA256: `fefe35a92bcbf22de67c7c7b43e9f97d2658afef41745182b2b8207f750592f4`.

## Implementation and regression checks

- `src/forcesmolvla/acceptance.py` fail-closes on comparison-scope drift, structural-scope drift, or any non-null formal bf16 threshold.
- `tools/preflight_p4_bare_parity.py` evaluates prefix and velocity groups independently and reports exact structural contracts separately.
- Focused acceptance tests: `10 passed`.
- Full CPU regression suite: `134 passed`.

## Complete P4 rerun

Both runs used LeRobot commit `30da8e687a6dfc617fcd94afc367ac7071c376ce`, strict offline reload, seed 4107, batch size 2, the same source binding SHA256 `2afbb09b2702e0bb5b405606fed95e6dfbf7b977e224491a4022788a50d6de68`, and no network access.

| Precision | Prefix hidden max abs | Max declared velocity/cache error | 10-step error | Invalid velocity | Peak CUDA allocated/reserved | Result |
|---|---:|---:|---:|---:|---:|---|
| fp32 | 0 | 3.0994415e-6 | 0 | 0 exact | 1,971,325,440 / 2,097,152,000 B | pass |
| bf16 | 0.25 | 0.04691458 | 0.00980625 | 0 exact | 1,309,543,936 / 1,409,286,144 B | pass |

All exact structural fields passed in both reports: prefix layout, prefix mask, physical prefix length 177, invalid suffix velocity zero, cache append-crop restoration, and unchanged cache snapshots. Debug whole-cache comparison remains outside the formal hot path.

Artifacts:

- `artifacts/development/p4_v4_2_r4_bare_parity_gpu_fp32.json`, SHA256 `9b3a053b5129608cfcebe0198b959522ca59a0878608add7220d9db384598398`.
- `artifacts/development/p4_v4_2_r4_bare_parity_gpu_bf16.json`, SHA256 `ceadf8df7bec1a7c0ea38924321149c70f90bf6a5ccac339efa29747ab678d95`.

The r3 P4 reports are historical measurements and are invalid for the current gate because the acceptance config and P4 runner source binding changed.

## Gate boundary

P4 is complete only for development use. P5 has not been entered. P8 and formal acceptance require independent evidence and approval; neither inherits these P4 bf16 thresholds.
