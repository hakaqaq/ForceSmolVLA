# Stage-2B throughput-v2 long-run status

The durable 0.5 Actor-pass boundary completed successfully. The subsequently
authorized continuation toward 1.0 pass stopped fail-closed when host available
memory crossed the frozen 8 GiB runtime floor. No automatic retry or parameter
change was performed.

## Interpretation of throughput results

```text
C128_EQUIVALENT_INFRA_SPEEDUP = 2.20x

C64_TOTAL_WALL_CLOCK_SPEEDUP = 4.28x
C64_SEMANTICALLY_IDENTICAL_TO_C128 = no
C64_STATUS = approved_long_run_hyperparameter
```

C64 keeps two Critic optimizer updates for each Actor optimizer update, but
uses half the Critic row exposure per cycle relative to C128. Its additional
wall-clock improvement is therefore not described as an equivalent
infrastructure-only speedup.

## Durable 0.5-pass result

The cycle-210 checkpoint is the latest complete and strictly loadable training
boundary:

```text
completed cycles       = 210
Actor exposure         = 5,040 transitions
TD row membership      = 26,880
Cal-QL row membership  = 26,880
cycle mean/median/P95  = 30.594 / 30.321 / 30.570 s
steady cycle time      = 6,424.697 s (1.785 h)
end-to-end child time  = 6,475.343 s (1.799 h)
cycles/hour            = 117.671
Actor transitions/s    = 0.78447
Critic TD transitions/s = 4.18386
Critic Cal-QL transitions/s = 4.18386
GPU utilization mean  = 50.838%
peak allocated VRAM   = 18,292,726,784 bytes
peak reserved VRAM    = 19,048,431,616 bytes
peak process RSS      = 40,343,429,120 bytes
```

Checkpoint bindings:

```text
cycle105 tree SHA256 =
d5fbf277929160e8808bc9f27086ad5ff7cd63c39fe2dd95d3abdcbe536de8eb

cycle210 tree SHA256 =
b514b50d118cb3edaa6e5e135e1a2cf7340062d11c16cb58bed437581c082e08
```

Across the 210 durable cycles, all recorded losses and gradients were finite.
TCP6 Q-gradient remained nonzero, gripper Q-gradient remained exactly zero,
and gripper Flow-Matching gradient remained nonzero. Frozen-VLM and
ActionContract-v2 checks passed. Validation, test, manual G1, manual labels,
and Reward Classifier inference access counts were all zero.

## 1.0-pass continuation safe stop

The continuation strictly resumed at cycle 211. Its last fully printed cycle
was cycle 293, but the worker was terminated before its final cycle-boundary
checkpoint/result payload. Those cycle-211--293 updates existed only in the
volatile process and are not a recoverable or authorized model artifact.

```text
failure                        = MEM_AVAILABLE_HARD_LIMIT
configured floor               = 8,589,934,592 bytes
observed minimum               = 8,584,093,696 bytes
threshold deficit              = 5,840,896 bytes
peak continuation process RSS  = 40,681,844,736 bytes
worker return code             = -15 (ordinary termination)
cycle420 checkpoint created    = no
latest durable cycle           = 210
```

This is not an OOM, numerical instability, ActionContract failure, or model
failure. It is the explicitly approved host-RAM safety gate acting as designed.
The process was not restarted with a smaller batch, larger cache allowance, or
different training semantics.

## Final state

```text
LONG_RUN_AUTHORIZED = yes_for_420_total_cycles
DURABLE_COMPLETED_CYCLES = 210
DURABLE_ACTOR_COVERAGE = 0.5_pass
ONE_PASS_COMPLETED = no
LONG_RUN_COMPLETED = stopped
AUTO_RESTART = no
AUTO_CONTINUE_BEYOND_420 = no
DEPLOYMENT_CHECKPOINT_AUTHORIZED = no
ROBOT_EXECUTION_AUTHORIZED = false
```

Any future attempt to reach cycle 420 must receive a new authorization and
must resume from the retained cycle-210 checkpoint. The discarded volatile
cycle-293 state cannot be used as a parent.
