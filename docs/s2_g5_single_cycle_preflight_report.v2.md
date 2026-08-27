# Stage-2 G5 development single-cycle preflight

Status: `PASS_DEVELOPMENT_SINGLE_CYCLE_ONLY` on `NVIDIA GeForce RTX 4090 D`.

Exactly one disposable cycle ran: 2 Critic optimizer updates, 2 Polyak updates per target, and 1 Actor optimizer update. Critic/Actor scheduler steps were 2/1; target-Actor updates were 0. A second cycle, G6/G7, evaluation, export, and robot execution did not run.

## Loss and update evidence

| Critic step | TD Q1/Q2 | Cal-QL Q1/Q2 | Twin-Q total | pre/post clip norm |
|---:|---|---|---:|---|
| 1 | 1.66371e-06 / 4.82602e-05 | 0.359296 / 0.355613 | 0.0357704 | 0.464006 / 0.464006 |
| 2 | 0.00418828 / 0.00389979 | 0.262319 / 0.26432 | 0.030376 | 0.266166 / 0.266166 |

Actor losses: FM `0.151972`, Actor-Q `-0.041687`, balance `0.999778`, z `8.81092e-05`, weighted total `0.161553`. TCP6 Actor-Q gradient was nonzero, gripper Actor-Q gradient was exactly `0.0`, and gripper Flow-Matching gradient was nonzero.

The diagnostic `||eta*grad_Q|| / ||beta*grad_FM||` was `0.000146801` on one fixed train microbatch. It is measurement-only and did not alter eta, beta, or either learning rate.

## Ownership, data, and checkpoint

Only the 10,075 automatic detector-G1 train transitions were available. TD, Cal-QL, Actor, empirical proposal, and every Flow/noise stream had independent serialized state; batch identities were unique. Validation/test reads, manual G1/label opens, Reward Classifier inference/updates, and robot actions were all zero.

The atomic checkpoint is marked `DEVELOPMENT_SINGLE_CYCLE_ONLY`, `NOT_FOR_DEPLOYMENT`, `NOT_FOR_POLICY_EVALUATION`, and `NOT_AN_APPROVED_LONG_TRAIN_PARENT`; exact resume remains untested and reserved for G6.

## Limits

`2 Critic : 1 Actor` is a ConRFT-inspired development recipe, not a proven-optimal update ratio. `M=2`, `eta=0.01`, `alpha=0.1`, and empirical whole-macro proposals are single-cycle mechanics values only. All demonstrations are successes, and Reward Classifier training overlaps automatic-G1 RL train episodes, so this smoke cannot establish failure recovery or policy improvement.


## ActionContract v2

This append-only rerun uses total binary internal gripper projection. Internal critic
canonicalization is not public execution authorization. Public tolerance, exception,
RuleSpec, and controller behavior are unchanged. No clipping, resampling, or binary
STE is used. Random empirical candidates are already frozen normalized endpoints;
their v2 projection is therefore an exact identity after endpoint validation.
