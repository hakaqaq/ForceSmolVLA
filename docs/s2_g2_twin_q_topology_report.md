# Stage-2 G2 Force-aware Twin-Q topology report

Status: `PASS_DEVELOPMENT_TOPOLOGY_ZERO_UPDATE_PREFLIGHT`.

The implemented interface is a mask-aware 3-step macro-action critic at 10 Hz, not a single-step critic. Each online critic consumes two cameras, the hash-bound canonical task feature, normalized state7/wrench6, normalized executed action `[B,3,7]`, and prefix mask `[B,3]`.

## Topology and ownership

The concatenated feature is 1283D (256+256+128+128+128+384+3) and the fusion path is `1283→1024→512→256→1`. New trainable layers use LayerNorm+SiLU. The frozen ConRFT backbone retains its native four-group GroupNorm and ReLU so safe-NPZ tensor semantics are not silently changed.

- `q1`: trainable=4,292,097, frozen=9,811,584, total=14,103,681
- `q2`: trainable=4,292,097, frozen=9,811,584, total=14,103,681
- `q1_target`: trainable=0, frozen=14,103,681, total=14,103,681
- `q2_target`: trainable=0, frozen=14,103,681, total=14,103,681

Q1/Q2 share neither module objects nor parameter/buffer storage. Targets are exact deep copies at initialization, permanently eval, require no gradient, and are absent from optimizers. No optimizer was created.

## Real GPU zero-update preflight

Device: `NVIDIA GeForce RTX 4090 D`. Real detector-G1 train rows returned: 10,075; validation/test rows returned: 0/0.

| Batch | Forward median (ms) | Backward median (ms) | Peak allocated (MiB) | Peak incremental (MiB) |
|---:|---:|---:|---:|---:|
| 1 | 3.301 | 1.527 | 293.1 | 0.3 |
| 4 | 3.591 | 1.477 | 302.9 | 4.8 |
| 16 | 5.284 | 1.601 | 442.0 | 123.8 |

All required camera/state/wrench/action/gripper sensitivities, valid-action gradients, exact padding invariance, mask rejection, target initialization, storage independence, and synthetic Polyak checks passed.

## Scope limits

- `STRICT_ZERO_FRAME_ALIGNMENT_ACCEPTANCE = FAIL_preserved`
- `DEVELOPMENT_DETECTOR_OPERATIONAL = yes`
- `FORMAL_DETECTOR_APPROVED = no`
- `REWARD_MODEL_TRAINING_OVERLAP = true`
- `UNBIASED_REWARD_MODEL_EVALUATION = false`
- TD, Cal-QL, Actor-Q loss, optimizer creation, and all parameter updates remain unimplemented/unapproved.

Acceptance checks: 17/17 passed.
