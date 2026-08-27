# Stage-2 G7-B development joint-smoke report

Status: **PASS (development mechanics only)**. The frozen G7-A-r2 Critic warm-up checkpoint was loaded at update 256; no G5/G6 smoke checkpoint was used. Exactly eight joint cycles ran, each with two Critic updates followed by one Actor update.

| cycle | Critic loss #1/#2 | TD Q1/Q2 #1→#2 | Cal-QL Q1/Q2 #1→#2 | FM | Actor-Q | Actor total | seconds |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.025062 / 0.021319 | 0.0010522/0.0010416 → 0.0010652/0.0011578 | 0.24019/0.24012 → 0.20218/0.20197 | 0.216155 | -0.109497 | -0.868818 | 42.14 |
| 2 | 0.0175834 / 0.064972 | 0.00097613/0.00101 → 0.048738/0.047679 | 0.16632/0.16548 → 0.16802/0.16724 | 0.222888 | -0.124451 | -1.01152 | 41.19 |
| 3 | 0.0215099 / 0.0174195 | 0.0021598/0.0024709 → 0.0029917/0.0032678 | 0.19302/0.19087 → 0.1437/0.14209 | 0.232552 | -0.136169 | -1.11911 | 41.96 |
| 4 | 0.0670204 / 0.0677964 | 0.04769/0.046854 → 0.048385/0.047024 | 0.19823/0.19674 → 0.20135/0.20048 | 0.306864 | -0.160095 | -1.28409 | 41.36 |
| 5 | 0.0265283 / 0.0301418 | 0.007177/0.0068045 → 0.0082456/0.0078071 | 0.19517/0.19558 → 0.2208/0.22151 | 0.180583 | -0.121216 | -1.02151 | 40.77 |
| 6 | 0.0252129 / 0.0237763 | 0.0066779/0.005767 → 0.0023609/0.0015338 | 0.18593/0.19388 → 0.21199/0.22459 | 0.240232 | -0.0864258 | -0.613977 | 41.79 |
| 7 | 0.0369342 / 0.0718214 | 0.0010362/0.00066441 → 0.046306/0.042326 | 0.35689/0.36479 → 0.27486/0.27525 | 0.142787 | -0.104736 | -0.894527 | 40.71 |
| 8 | 0.0291698 / 0.0160314 | 0.00093901/0.0012534 → 0.0015472/0.0033393 | 0.28459/0.27688 → 0.14234/0.12942 | 0.155665 | -0.131714 | -1.15137 | 42.21 |

## Gradient and action contract

Unweighted `||g_Q||/||g_FM||` median/P95/max: `0.00706433` / `0.0112785` / `0.0125057`. With the smoke-only eta=10, the weighted values are `0.0706433` / `0.112785` / `0.125057`. Gradient-cosine median/P95/max: `-0.0219567` / `0.0945467` / `0.104402`.

Every cycle had nonzero TCP6 Q-gradient, exactly zero gripper Q-gradient, and nonzero Flow-Matching gripper gradient. The v2 total-binary internal projection remained separate from public execution authorization. The raw out-of-public-tolerance rate is a distribution diagnostic only; it neither clipped nor resampled Critic candidates. Fixed train observations and fixed noise were used for normalized TCP drift/binary-gripper diagnostics. `predict_action_chunk()` completed before training and after every cycle under the unchanged public RuleSpec.

The per-cycle TD-target means (Critic substeps 1/2), Cal-QL Q1/Q2 values, and fixed-observation action diagnostics were:

| cycle | TD target mean #1/#2 | Cal-QL Q1 #1/#2 | Cal-QL Q2 #1/#2 | normalized TCP drift L2 | binary gripper change rate |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.0755282 / 0.0785431 | 0.240187 / 0.202180 | 0.240116 / 0.201971 | 0.0215187 | 0 |
| 2 | 0.0728911 / 0.133830 | 0.166325 / 0.168022 | 0.165483 / 0.167241 | 0.0343362 | 0 |
| 3 | 0.0764890 / 0.0743992 | 0.193020 / 0.143704 | 0.190872 / 0.142090 | 0.0350354 | 0 |
| 4 | 0.134317 / 0.134352 | 0.198230 / 0.201351 | 0.196738 / 0.200483 | 0.0503847 | 0 |
| 5 | 0.0765254 / 0.0796899 | 0.195170 / 0.220797 | 0.195581 / 0.221511 | 0.0450246 | 0 |
| 6 | 0.0821546 / 0.0787344 | 0.185930 / 0.211987 | 0.193880 / 0.224593 | 0.0709413 | 0 |
| 7 | 0.0806233 / 0.134152 | 0.356890 / 0.274863 | 0.364787 / 0.275249 | 0.0697729 | 0 |
| 8 | 0.0780858 / 0.0783725 | 0.284592 / 0.142340 | 0.276880 / 0.129422 | 0.0745627 | 0 |

All nine public diagnostic calls (baseline plus one after every cycle) succeeded. Internal `raw_gripper_out_of_public_tolerance_rate` was `0.0`; this is not a public-validity metric.

## Runtime

Peak CUDA allocation/reservation was 7,719,208,448 / 7,929,331,712 bytes. Median/mean cycle latency was 41.58 / 41.52 seconds (range 40.71–42.21 seconds); this implies about one Actor update every 41.5 seconds for this offline smoke workload on the RTX 4090D. It is a throughput estimate only, not an authorized online control/update period. The eight-cycle training body took 340.95 seconds and sampled 352 Flow chunks with 3,520 Euler velocity evaluations.

## Ownership, access, and limits

Actor, Q1/Q2, and both Polyak targets changed in their authorized substeps; frozen ResNet backbones did not. The final atomic checkpoint passed a fresh-process strict model/optimizer/scheduler/sampler/RNG load. Validation/test transitions, manual G1, manual labels, and Reward Classifier inference/updates were all zero.

This run does not authorize a long run, policy evaluation, export, online HIL, ROS/RTC, or robot execution. Eta=10.0 is approved only for these eight development cycles. The all-success dataset and reward-model training overlap remain unchanged, so no policy-improvement, recovery, unbiased-evaluation, or deployment claim is made.
