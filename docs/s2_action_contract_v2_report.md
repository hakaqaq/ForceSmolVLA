# Stage-2 ActionContract v2 reclosure report

Status: `ACTION_CONTRACT_V2 = pass`.

## Contract

The Stage-2 Critic action remains `[B, K=3, 7]`. TCP dimensions `0:6` remain
normalized continuous xyz/rpy. The gripper dimension is canonicalized only on
the internal Critic path as:

```text
normalized Flow gripper
-> frozen normalizer inverse
-> finite check
-> frozen binary decoder
-> exactly 0.0 or 0.085 m
-> frozen normalizer
-> stop-gradient
```

This is deliberately distinct from public execution authorization. The public
path retains its existing tolerance, validation, exception, absolute-action,
RuleSpec, and controller behavior.

The approved research description is:

> value-guided 6-DoF Cartesian refinement with imitation-regularized discrete
> gripper control

The implementation does not claim differentiable Twin-Q optimization of the
full 7D action.

## ActionContract-v2 evidence

- The original offender `g_flow_normalized=1.71746826171875` inverse-normalizes
  to `0.09740415960550308 m`, maps internally to the frozen `0.085 m` endpoint,
  and remains rejected by the unchanged public tolerance path.
- Normalized Critic gripper endpoints are
  `[-0.7100614905357361, 1.4083287715911865]`.
- Every finite internal input deterministically maps to one of the two frozen
  endpoints; NaN and Inf fail closed.
- Decoder threshold direction and tie rule have golden parity with the existing
  public decoder.
- Internal public-validator, absolute-inverse, RuleSpec, and public decoder call
  counts are zero.
- Clipping, resampling, binary STE, and public threshold changes are absent.
- The synthetic projector diagnostic reports
  `raw_gripper_out_of_public_tolerance_rate=0.9984999895095825`; this is a raw
  Flow-distribution diagnostic, not a public-validity rate.

Primary artifact:
`artifacts/development/stage2/s2_action_contract_v2_preflight.v2.json`.

## Sequential reclosure

The required fail-closed sequence completed without modifying the historical
v1 evidence:

| Gate | Result | Scope |
| --- | --- | --- |
| ActionContract-v2 | pass | total-binary internal projection |
| G3-v2 | pass | differentiable Flow and gradient contract |
| G4-v2 | pass | zero-update losses and ownership |
| G5-v2 | pass | fresh single-cycle mechanics only |
| G6-v2 | pass | fresh-process exact resume |
| G7-A-r2 | pass | 256-step Critic-only warm-up |

G7-A-r2 executed 256 Critic optimizer/scheduler updates and 256 Polyak updates
per target. Actor optimizer/scheduler updates were zero, Actor parameters and
floating buffers remained bitwise unchanged, and a fresh process strictly
loaded the final checkpoint. The actual training-path
`raw_gripper_out_of_public_tolerance_rate` was
`1.9958884697523103e-05`; internal binary projection allowed deterministic
duplicates (`binary_gripper_pattern_duplicate_rate=0.9223399796419376`) without
resampling.

The 32 frozen train-only probes measured median raw
`||g_Q||/||g_FM||=0.0033034774856795924`. Among the zero-update candidates,
only `eta=10.0` had a median weighted ratio inside `[0.01, 0.10]`. This is a
measurement, not an approval; `ETA_G7B_APPROVED = no`.

An earlier foreground attempt was infrastructure-interrupted at update 176 by
the one-hour tool-session limit. It produced no final checkpoint, was not
resumed, and was excluded from acceptance. The accepted run restarted from the
frozen r5 Actor and fixed-seed fresh G2 Critics at update 0.

## Safety and research boundary

```text
ACTION_CONTRACT_V1 = historical_superseded
INTERNAL_GRIPPER_PROJECTION = total_binary
PUBLIC_INFERENCE_BEHAVIOR_CHANGED = no
PUBLIC_TOLERANCE_CHANGED = no
CLIPPING_ADDED = no
RESAMPLING_ADDED = no
BINARY_STE_ADDED = no
G7A_R1_FAIL = preserved
ACTOR_UPDATES_IN_G7A = 0
ETA_G7B_APPROVED = no
G7B_STARTED = no
LONG_RUN_AUTHORIZED = no
ROBOT_EXECUTION_AUTHORIZED = false
```

All data remain success demonstrations, and the reward model overlaps the RL
train episodes. These results establish development mechanics, numerical
behavior, and gradient scale only; they do not establish policy improvement,
failure recovery, OOD conservatism, unbiased policy evaluation, formal reward
generalization, deployment readiness, or robot-execution authorization.
