# P9 task2 input approval addendum

日期：2026-08-21（Asia/Shanghai）  
状态：`development_only`

用户在批准进入 P9 后明确确认当前离线数据目录为
`/home/rlc123/ForceSmolVLA/datasets/task2_lerobotv3`。因此本次 P9 只读
record/replay 使用该目录，以及 `golden_fixtures` 和 `tests/fixtures` 中明确标记的
test-only 工件。

该确认不批准 production/formal Shadow，不批准任何 ROS、RTC、Franky queue、实时
控制接口或机器人动作发送，也不把 test-only 阈值或 synthetic clock map 提升为正式
工件。P8 r4 checkpoint 与 task2 normalizer/calibration 的严格兼容性检查保持启用。

为避免使已通过的 P8 r4 source binding 失效，本 addendum 不修改 P8 已绑定的
`ForceSmolVLA_Implementation_Spec_v4_2.md`；本次输入修订由
`configs/p9_task2_scope_amendment.development.json` 与
`configs/p9_shadow_replay.development.json` 独立 hash-bound，不要求重跑 P4–P8。

task2 原始根目录的单个 `session.json` 没有显式物理 `session_id`。因此不得沿用
`task1_within_session`，也不得自行发明物理 session 身份。P9 与后续 task2 development
SFT 使用 `task2_collection_scope_f87ae11e9831` 作为 collection-scope label，并明确
记录 `physical_session_id=null`；结论仍仅限 episode-disjoint、within-collection
development，不声称跨 session 泛化。

训练预算继续服从 v4.2：80,000 samples，B4×1 下派生为 20,000 optimizer updates。
10,000-update 旧状态文本无效，本 addendum 不构成新的缩减实验设计。
task2 下游训练入口采用 `final_update_only`：validation 仍每 2,000 samples 运行，
但 step 1、周期节点和 new-best 都不保存；仅 update 20,000 原子保存最终 checkpoint。
该 override 不修改 P7/P8-bound 的离线 recipe，P8 r4 证据保持有效。

P9 调度索引使用
`j=ceil((t_candidate_controller_ns-tau0_controller_ns)/Delta_a)`，其中
`t_candidate=t_ready_controller+transport`。r5 的差值为 205 ms，在 30 Hz 下为 6.15，
故 `j=7`；不使用基于 `t_apply` 的早期简化公式。
