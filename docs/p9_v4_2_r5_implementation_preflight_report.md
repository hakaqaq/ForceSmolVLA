# P9 v4.2 r5 task2 scope/provenance preflight report

日期：2026-08-21（Asia/Shanghai）  
状态：`DEVELOPMENT_GATE_PASS / PRODUCTION_SHADOW_FAIL_CLOSED`  
acceptance：`development_only`，`formal_eligible=false`

## 结论

P9 r5 task2-scoped algorithmic development replay 已通过；P4–P9 development
implementation functionally complete。r5 通过独立 hash-bound amendment 解决了 v4.2
可见文本中 P9 `task1_v4_1` 与实际 `task2_lerobotv3` 的 scope 冲突，不修改或重跑
P4–P8。

task2 原始根目录只有一个 session 级 `session.json`，但该文件没有物理
`session_id`。因此 r5 明确记录 `physical_session_id=null`，并仅使用确定性的
collection-scope label `task2_collection_scope_f87ae11e9831`；它不是物理 session
身份。r4/P7 fixture 中的 `task1_within_session` 被分类为 task2 的无效 legacy metadata，
r5 records 中已替换且保留替换 provenance。

长程 development SFT 预算已统一为 80,000 samples；B4×1 下派生为 20,000
optimizer updates。训练 wrapper 会拒绝 10,000-update/40,000-sample 漂移，并在未来
validation fixture 中使用相同 collection-scope provenance。本次没有启动训练。

## 调度语义

本次冻结并验收：

```text
t_candidate = t_ready_controller + transport
j = ceil((t_candidate_controller_ns - tau0_controller_ns) / Delta_a)
  = ceil((t_candidate_controller_ns - tau0_controller_ns)
         * action_period_denominator / action_period_numerator_ns)
```

其中 `Delta_a=1/30 s`，本次 `t_candidate-tau0=205 ms`，未取整索引为 `6.15`，
所以 `j=7`。该公式不使用 `t_apply`；早期基于 `t_apply` 的简化表述不适用于 r5。

## 测试与绑定

- P9-and-upstream：`149 passed`，耗时 `114.81 s`。
- JUnit SHA256：`e340987439f6d68c331e2dccb3e7e47eeb9569d999e094952aa281a092fc791e`。
- P9 config SHA256：`efa467732c9d80f462927cce2a360677cf76cd0b2ac484658ca084c2ab6cff3d`。
- P9-only amendment SHA256：`5d5ae90782b8d4582f3b9285de5523b9a437a4f5b6a6457aee5aa84b42ae3263`。
- task2 data/session/training scope SHA256：`28d66cf68f3d2befb6f8e41b8f4b0252f08cae93ebc40463d3e7199371fd7bd2`。
- 原始 task2 `session.json` SHA256：`f87ae11e98312688a1673bba4581898f27ec2e7045bcc87d8afc8aea816791f2`。
- source binding SHA256：`5f96275f02adafe7b1b52bb1981d79e3027afabf4d22f9cb66eaaec70e701474`。
- P8 gate/source SHA256 仍为
  `27fd7846c380875a5969d8e54e919508a202e4e75a3be6a9e24af9cafd46ca24` /
  `30f8eaad5b8894cf2c88053dd9e4db9a324593ce2f45e7e7acf9cca2b27a8147`。

## 真实离线 inference/replay

- GPU：NVIDIA GeForce RTX 4090 D。
- 输入：task2 val episode 7 frame 0；B=2、双相机、H=50、execution horizon=3。
- CUDA/wall latency：`585.577 ms / 0.585690 s`。
- peak allocated/reserved：`1,552,972,288 / 1,673,527,296 bytes`。
- 11 项 acceptance assertions 全部为 true。
- candidate valid=false；唯一 reason=`SHADOW_END_TO_APPLY_EXCEEDED`。
- `j=7`，actual dispatched indices=`[]`，replay exact=true。
- pose/wrench source timestamp 加 conversion manifest 的整数 `clock_offset_ns` 后，
  精确等于 records 中的 host-monotonic timestamp。
- `ros_connected=false`、`rtc_configured=false`、`native_queue_used=false`、
  `robot_actions_sent=0`。

## 最终可独立核验工件

- JUnit：[p9_v4_2_r5_pytest.xml](../artifacts/development/p9_v4_2_r5_pytest.xml)
- source binding：[p9_v4_2_r5_source_binding.json](../artifacts/development/p9_v4_2_r5_source_binding.json)
- resolved config：[p9_v4_2_r5_resolved_config.json](../artifacts/development/p9_v4_2_r5_resolved_config.json)
- records：[p9_v4_2_r5_records.json](../artifacts/development/p9_v4_2_r5_records.json)
- replay：[p9_v4_2_r5_replay.json](../artifacts/development/p9_v4_2_r5_replay.json)
- final gate：[p9_v4_2_r5_gpu_preflight.json](../artifacts/development/p9_v4_2_r5_gpu_preflight.json)

对应 SHA256：

```text
e340987439f6d68c331e2dccb3e7e47eeb9569d999e094952aa281a092fc791e  JUnit
5f96275f02adafe7b1b52bb1981d79e3027afabf4d22f9cb66eaaec70e701474  source binding
d20e96d48b572ef0273bacbcc26980a6ab76c96170c35470ad61d7b0985d0880  resolved config
c576b82e65c57112ae83984b074acec50004793d71c5b0f4dae08b0cfb03acf3  records
f48d035687ecff78296f9e21a56d70cdb616d3cf62c405b2cb0941ccf13feef6  replay
03c85ff9305fe4d64bdaea360585a4faaea6037c162f4fc9dec3ad5ffcd5d507  final gate
```

## Formal/production 边界

production sensor→controller/GPU→controller clock map、正式 Shadow 阈值、trusted
detached signature algorithm/key/approver/verifier 仍未冻结。production/formal resolver
继续 fail-closed。P9 r5 不授权 ROS、RTC、在线 HIL、native queue 或机器人动作发送，
也不支持跨 session 泛化结论。
