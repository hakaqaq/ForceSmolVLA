# Stage-2 throughput-v2 benchmark

本轮仅执行临时 benchmark updates。所有候选从同一 G7-A-r2 父状态、相同样本顺序和 RNG 启动；候选参数均丢弃，没有训练 checkpoint，也没有恢复或启动 long-run。

## 结果

| Candidate | mean cycle (s) | Actor trans/s | Critic trans/s | cycles/h | speedup | peak reserved (GiB) | equivalence |
|---|---:|---:|---:|---:|---:|---:|---|
| baseline_current_implementation | 129.256 | 0.1857 | 1.9806 | 27.852 | 1.000× | 18.81 | bitwise_exact |
| candidate_A_async_data_pipeline | 64.567 | 0.3717 | 3.9649 | 55.756 | 2.002× | 17.84 | bitwise_exact |
| candidate_B_prefix_cache | 58.050 | 0.4134 | 4.4100 | 62.015 | 2.227× | 17.91 | numerically_equivalent_with_declared_tolerance |
| candidate_C_flow_subbatch_8 | 32.685 | 0.7343 | 7.8323 | 110.142 | 3.955× | 18.08 | failed |
| candidate_D_flow_subbatch_16 | 21.274 | 1.1281 | 12.0334 | 169.220 | 6.076× | 18.66 | failed |
| candidate_E_grouped_td_calql_flow | 17.661 | 1.3589 | 14.4953 | 203.840 | 7.319× | 22.00 | failed |

推荐：`candidate_B_prefix_cache`。选择依据是通过数值等价、finite、ActionContract-v2、冻结 hash 和公共接口检查后的最高 mean steady-state throughput，不是最大显存占用。

## Actor transition-pass 预算（mean steady-state）

| Budget | cycles | projected hours | Actor exposure | Critic exposure |
|---|---:|---:|---:|---:|
| 0.5_actor_pass | 210 | 3.39 | 5040 | 53760 |
| 1.0_actor_pass | 420 | 6.77 | 10080 | 107520 |
| 2.0_actor_passes | 840 | 13.55 | 20160 | 215040 |

启动/冷 cache 成本单独报告，未计入 steady-state cycle throughput。按用户要求，本轮每候选仅运行 1 warm-up + 1 measured cycle，因此用于快速筛选，不对 P95 或跨重复波动作统计性主张。

Action 等价性在真实 ActionContract-v2 Critic K×7 域比较：TCP6 使用预先声明的 bf16 容差，binary gripper endpoint 必须 exact。未投影 H×7 raw Flow 只作为诊断，不作为 Critic 输入等价性的错误门槛。B8/B16/E 因 TCP 超差或 gripper endpoint 翻转被拒绝，即使吞吐更高也不推荐。

```text
CURRENT_RUN_STATUS = valid_interrupted_long_run_pilot
CURRENT_CHECKPOINT_STATUS = audit_only_cycle105_latest; no_cycle136_checkpoint
AUTO_RESUME = no
THROUGHPUT_V2_AUTHORIZED = yes
THROUGHPUT_V2_LONG_RUN = no
THROUGHPUT_V2_TRAINING_CHECKPOINT = no
RESTART_0_5_PASS_AUTHORIZED = no
AUTO_EXTEND_TO_1_0_PASS = no
LONG_RUN_EXTENSION_AUTHORIZED = no
ROBOT_EXECUTION_AUTHORIZED = false
```
