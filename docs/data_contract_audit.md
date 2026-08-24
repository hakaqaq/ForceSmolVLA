# ForceSmolVLA v4.1 数据契约检查报告

检查日期：2026-08-19  
目标契约：Force-SmolVLA v4.1 available-sensor profile  
只读输入：`/home/rlc123/fr3_client_ws/datasets/task1`  
目标输出：`/home/rlc123/ForceSmolVLA/datasets/task1_forcesmolvla_v4_1`  
结论：`FORMAL_WITHIN_SESSION_OFFLINE_SFT_ELIGIBLE_AFTER_EPISODE_GATES`

## 1. SmolVLA 与本项目真正需要的数据

上游 SmolVLA 需要 observation、至少一幅 RGB 图像、task/language、state 和 future action chunk；冻结候选版本把短 state/action pad 到 32D，并读取 LeRobot v3.0 数据。

ForceSmolVLA v4.1 在此之上使用现有可采集模态：

- `camera1` = D435 third-person RGB，`camera2` = D405 wrist RGB，顺序固定；
- 7D TCP state（第7维保持 measured gripper width m）与独立 wrench6；
- 原始 7D absolute TCP action 和原始 prompt；
- 约 100 Hz measured TCP pose，不要求 measured joint-q 或 FK；
- 每个 raw wrench 只选择最新 `t_pose <= t_wrench` 的 TCP pose（causal ZOH）；
- 静态 `T_TCP<-sensor`、sign、bias、payload gravity、rotation 和 TCP moment shift；
- 每个 wrench/tuple 保留 pose/raw/filter timestamps、pose age、calibration id 和 validity；
- episode-disjoint train/val/test，normalizer 只拟合 eligible train episodes。

当前数据只有一个 recording session，因此可用于正式的 within-session offline fine-tuning，但不得据此声称跨 session 泛化。

## 2. 原始 task1 实测清单

| 项目 | 实测 |
|---|---|
| raw format | `fr3-hilserl-impedance-native-raw-v5` |
| raw size/file count | 约 8.7 GiB / 84,766 files |
| recording session / episode | 1 / 50 |
| prompt | `Pick up the purple disk and slide its center hole onto the wooden peg until fully seated.` |
| session manifest SHA256 | `7c5817d5139ce5cee34ef86f5dfb50e5ed1996cd9939e1b3b7c7f96d4261bbf8` |
| legacy converter SHA256 | `f8dc57b0a69b29be36b8fb62820d4284c37b4687def4cc27bad4075040f96b1a` |
| D435 external RGB | 约 29.98 Hz，480×640 RGB，serial `250343062953` |
| D405 wrist RGB | 约 29.99 Hz，480×640 RGB，serial `242223072007` |
| measured TCP pose | 约 100 Hz，`fr3_link0`，含 source/receive timestamp |
| raw notch wrench | 约 499.5 Hz，sensor frame，含 source/receive timestamp |
| accepted reference | 约 100 Hz，absolute TCP pose + gripper target |
| reference acknowledgement | 约 10 Hz，含 accepted flag、pose、sequence 与 ack timing |
| gripper state | 约 500 Hz，同时含 `controller_position` 与 `width_m` |
| calibration | bias、downstream mass/CoM、gravity、sign 和 TCP/sensor 静态关系存在 |

原始目录保持完全只读。已有 `/home/rlc123/ForceVLA/datasets/task1` 仅作为 ForceVLA/v2.1 解析参考，也保持只读；v4.1 converter 直接读取 raw root。

## 3. causal-ZOH pose-age 实测

审计算法：对每个 `wrench_notch_sensor.source_stamp_ns = t_w`，选择最大的 `measured_tcp_pose.source_stamp_ns = t_pose`，且严格满足 `t_pose <= t_w`。两个字段属于同一 ROS source timestamp domain，因此本统计不使用 receive time，也不做插值。

| 指标 | 结果 |
|---|---:|
| episodes | 50 |
| pose samples | 139,978 |
| raw wrench samples | 699,418 |
| matched causal samples | 699,134 |
| episode 起始处无历史 pose | 284 |
| pose timestamp monotonic violations | 0 |
| wrench timestamp monotonic violations | 0 |
| pose-age P50 | **5.026 ms** |
| pose-age P95 | **9.536 ms** |
| pose-age P99 | **10.161 ms** |
| pose-age maximum | **11.776 ms** |

审计工具：`/home/rlc123/ForceSmolVLA/tools/audit_pose_age.py`。无历史 pose 的起始 wrench 必须标记 invalid，不能借用未来 pose；后续 tuple 只有使用的 wrench/filter 样本有效时才可进入数据集。

### max_pose_age 候选（未批准）

`max_pose_age_ms` 是必填配置且没有默认值。基于当前数据最大实测值 11.775707 ms，候选值为 **12.0 ms**（把当前最大值向上取整到 1 ms）；这会保留所有已有 causal matches，但不改变 284 个无历史 pose 样本的 invalid 状态。

该候选仅为 `development-only / approval_pending`。在实验负责人明确批准前，不把 12.0 ms 写成正式 RuleSpec 阈值，正式训练入口必须因阈值未批准而 fail-closed。

## 4. v4.1 字段判定

| v4.1 项目 | 结果 | 处置 |
|---|---|---|
| 两路 RGB 与顺序 | PASS | 固定 D435→camera1、D405→camera2 |
| raw/source timestamps | PASS | 原样保留并写 selection provenance |
| measured TCP pose | PASS | causal ZOH；不要求 q/FK |
| raw wrench finite | PASS（schema） | converter 逐样本复核 |
| device status | PARTIAL | raw stream未见独立 status 字段；manifest 必须显式记为 unavailable，不可伪造 |
| saturation | PENDING THRESHOLD | 保留 raw axes；full-scale/margin待 RuleSpec 批准 |
| timestamp monotonicity | PASS | 本次全量检查违规为 0 |
| drift | PENDING THRESHOLD | 数据可计算；窗口/阈值待批准 |
| filter reset/warm-up | IMPLEMENTATION REQUIRED | 每 episode reset，warm-up invalid |
| static `T_TCP<-sensor` | PASS/PARTIAL | 当前 calibration/tool profile可解析；方向、id、hash须由 fixture 冻结 |
| sign/bias/payload/rotation/moment shift | IMPLEMENTATION REQUIRED | 必须与现有已验证转换路径做 golden parity |
| original 7D absolute action | PASS | 第7维保留 `target_gripper_width_m`，不改写为 joint/controller action |
| prompt | PASS | 原字符串及 hash均保留 |
| episode-disjoint split | PASS BY DESIGN | 只在 quality gates 后划分 |
| train-only normalizer | PASS BY DESIGN | val/test 禁止 fit/update |
| LeRobot v3.0 | IMPLEMENTATION REQUIRED | 旧 v2.1 产物不复用 |

两相机 causal latest-only development audit使用每 episode首个 `reference_ack.receive_monotonic_ns` 作为待批准 grid anchor，测得41,933个双相机 tick：D435 age P50/P95/P99/max=`16.058/31.725/33.035/33.643 ms`，D405=`17.049/31.426/32.855/33.438 ms`，相机间 skew=`10.252/26.261/28.823/32.786 ms`。据此提出 camera max age 34 ms、skew 33 ms候选；它们依赖 grid/clock语义获批，目前仍为 development-only。

全量 ack/action association audit显示：13,842个 accepted `reference_ack` 全部能关联到它之前最新的 `accepted_reference`；position error最大为0，sign-invariant quaternion geodesic最大为`4.214685e-8 rad`，reference在ack之前`0.035271–0.417518 ms`，future association和rejected ack均为0。候选语义因此为：按 receive monotonic causal latest关联，验证 pose一致，再从同一 record取得 absolute pose与`target_gripper_width_m`，之后按ack-associated target做ZOH。该候选仍需批准。

## 5. 新 v4.1 转换契约

输出目录固定为 `/home/rlc123/ForceSmolVLA/datasets/task1_forcesmolvla_v4_1`；目录已存在即 fail-fast，不提供隐式 overwrite。

1. 直接读取只读 raw task1；不从旧 v2.1 dataset 二次转换。
2. 保留 source tree hash、converter/version hash、LeRobot commit、环境 lock、camera order、全部选择 timestamp/age、prompt/hash 和原始 7D absolute action。
3. `observation.state` 只含 7D TCP/gripper-width state；`observation.wrench` 是独立 6D feature。旧 13D state不得作为新 storage contract。
4. 每个 wrench 使用 measured TCP pose causal ZOH；禁止 future pose、linear、SLERP 或 SE(3) 插值。
5. 先 sign、bias、payload gravity，再旋转到 base，并把 moment shift 到 TCP；每一步由 calibration id/hash和 fixture追踪。
6. 保留 raw wrench finite、可用的 device status、saturation、timestamp monotonicity、drift、filter reset/warm-up结果；不可用字段必须声明 unavailable，不得补造。
7. 每路相机只选 `timestamp <= t_ref` 的最新帧；缺失/过期/skew策略由待批准 RuleSpec 控制。
8. quality gate 只排除失败 episode；通过的 episode进入冻结的 episode-disjoint split。
9. normalizer 只对 train episode fit一次；stats绑定 split、calibration、geometry和converter hashes。
10. 报告固定写“calibrated TCP wrench conditioned on measured TCP pose”和“within-session offline fine-tuning”。

## 6. 转换 manifest 最低字段

- raw root realpath、raw format、session.json hash；
- canonical source-tree manifest/hash（relative path、size、SHA256）；
- converter version/source hash、LeRobot commit、environment lock hash；
- output schema=`forcesmolvla-v4.1`、LeRobot codebase version、fps；
- camera keys/order/role/model/serial/color/resolution和 source timestamps；
- state/action/wrench names、frames、units、handedness和 gripper语义；
- prompt原文/hash；
- measured-TCP lookup=`causal_zoh`、`allow_future_pose=false`、必填 `max_pose_age_ms`；
- `calibration_id`、calibration/T_TCP<-sensor/geometry/filter hashes；
- 每 episode/tick 的 source ids、raw/filter/pose/image/action timestamps、ages、validity和 exclusion codes；
- deterministic episode split manifest；
- train-only normalizer fit manifest及输入 episode ids/hash。

## 7. 尚待批准、不得猜测的字段

- `max_pose_age_ms`：候选 12.0 ms，待实验负责人批准；
- camera max age 与 inter-camera skew；
- controller grid anchor、ROS source→upper-host clock map estimator/confidence及 id/hash；
- 10 Hz `reference_ack` 如何精确证明100 Hz `accepted_reference`中的 absolute pose与 gripper target；
- wrench full-scale/margin、drift窗口/统计量/阈值；
- gravity residual、TCP lever-arm 和 filter warm-up thresholds；
- train/val/test episode比例和 split seed；
- detached signature 的算法、key id/public-key来源、签名编码、批准人 identity/role、approval id/timestamp、canonicalization规则及信任根。

这些未冻结项不阻止 P0–P4 代码、转换 dry-run/smoke/parity；它们阻止相应正式 gate 被标记为 approved。正式 SFT入口在全部必需 gate 批准前 fail-closed，但不再以单 session、缺 joint-q/FK 或约 100 Hz measured TCP pose为阻塞理由。

## 8. development-only全量转换与v2.1对比结果

2026-08-19 已从只读raw目录直接生成
`/home/rlc123/ForceSmolVLA/datasets/task1_forcesmolvla_v4_1`：50 episodes、40,780
frames、0 exclusions、约32 GiB。manifest保持`artifact_status=development_only`、
`formal_ready=false`；重算84,766个raw文件后source tree SHA256仍为
`ae2e353733c2a3208b07ba84c409d89c2c0ab423c184253cf8532fa262caff2e`。

全量Parquet检查无错误：所有state7/wrench6/action7 finite；相机、pose和action选择均
无future sample；raw/filter wrench timestamp一致；30 Hz rational grid、frame/global
index、age/skew重算、calibration index和validity全部一致。tuple上的wrench geometry
pose-age P50/P95/P99/max为`4.914/9.566/10.196/11.677 ms`。当前action-ack age
P50/P95/P99/max为`49.678/95.230/99.107/109.831 ms`，报告测量值但未擅自新增批准阈值。

与legacy v2.1相比并非仅容器版本不同：v2.1为41,615 frames，v3少835 frames，50个
episode各少15–18帧，主要来自每episode 250个500 Hz causal-filter warm-up样本排除和
因果共同区间/grid边界。v2.1把float64 state7+wrench6合为state13，并使用nearest相机、
pose/gripper插值及regularized/interpolated wrench；v3拆为float32 state7与独立wrench6，
使用latest-causal选择、ack-associated action、episode filter reset和逐tuple provenance。
因此数值允许接近但不会逐帧相同；按最近时间线对齐后的绝对差异P99为state各元素聚合
`4.99e-4`、wrench各元素聚合`9.40e-2`、action各元素聚合`3.03e-3`。

唯一结构警告是冻结LeRobot生成的`meta/info.json`仍把存储范围写为`train=0:50`。
权威实验split是独立manifest的40/5/5；`forcesmolvla.dataset_v3.load_dataset_split`
已强制按该manifest传入episode indices，并在formal模式拒绝当前development-only产物。
完整机器可读报告：`artifacts/development/task1_v3_validation_comparison.json`。
