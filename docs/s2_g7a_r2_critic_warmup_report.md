# Stage-2 G7-A Critic-only warm-up report

Status: `G7A_CRITIC_WARMUP_MECHANICS = pass`.

The worker started from the frozen Stage-1 r5 Actor and fresh G2 seed-0 Twin-Q; no G5/G6 checkpoint training state was loaded. Exactly 256 Critic optimizer/scheduler steps and 256 Polyak updates per target ran. Actor optimizer/scheduler/update counts remained zero, and Actor parameters plus floating buffers matched r5 bitwise before and after.

| Update | Dataset | Rows | TD MSE | Cal-QL term | Total critic loss | Q/MC MAE | Spearman |
|---:|---|---:|---:|---:|---:|---:|---:|
| 0 | train_probe | 128 | 0.295925 | 0.304777 | 0.326402 | 0.538665 | -0.514676451958457 |
| 0 | validation | 1205 | 0.00415309 | 0.328508 | 0.0370039 | 0.373592 | -0.3269900482212364 |
| 256 | train_probe | 128 | 0.228489 | 0.213903 | 0.24988 | 0.431864 | 0.9172856959473215 |
| 256 | validation | 1205 | 0.00417257 | 0.237874 | 0.0279599 | 0.272239 | 0.967260213363741 |

The fixed 32-batch train-only scale probe measured median raw `||g_Q||/||g_FM|| = 0.00330348` with p10/p90 `0.0014203/0.00677479` and maximum `0.01674`. Median cosine similarity was `0.0235471`. Measurement-only eta candidates whose median weighted ratio fell in `[0.01, 0.10]`: `[10.0]`. No eta was selected or approved, and no Actor update occurred.

All fixed train/validation row IDs, Flow noises, timesteps, and proposals were frozen before update 0. Validation was evaluated only at updates 0 and 256 and was not used for search, early stopping, or checkpoint selection. Test transition/image reads, manual G1/label reads, and Reward Classifier inference/updates were zero.

The checkpoint is `DEVELOPMENT_G7A_CRITIC_WARMUP_ONLY`, `NOT_FOR_DEPLOYMENT`, `NOT_FOR_POLICY_EVALUATION`, `NOT_AN_APPROVED_LONG_TRAIN_PARENT`, and `APPROVED_ONLY_FOR_G7B_IF_EXPLICITLY_AUTHORIZED`. A second fresh process strictly loaded it without an update or sampler draw.

All demonstrations are successes; Reward Classifier training overlaps the RL train episodes; unbiased policy evaluation is false. G7-A establishes only Critic warm-up numerical behavior and Q-gradient scale. It does not demonstrate policy improvement, failure recovery, OOD conservatism, or reward-model generalization.


## ActionContract v2 and r1 preservation

`G7A_R1_FAIL` remains preserved with zero optimizer/Polyak/Actor updates and no
checkpoint. Its numerical-stability status is `not_measured`. This r2 run uses
total-binary internal gripper projection; public execution behavior and tolerance
are unchanged. No clipping, resampling, or binary STE was added.
