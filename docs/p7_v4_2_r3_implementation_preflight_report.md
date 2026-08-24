# P7 v4.2 r3 implementation/preflight report

日期：2026-08-20（Asia/Shanghai）  
结论：`PASS_DEVELOPMENT_ONLY`  
formal/production eligibility：`false`  
下一阶段：`P8_READY_AWAITING_APPROVAL`（本次未进入 P8）

## 阶段边界

P7 的真实训练回归固定为一次 B4×1 single-pass：一次联合
`forward -> backward -> clip -> AdamW.step -> scheduler.step`。B2×8 exact two-pass
只在全新模型实例上作为短程 acceptance oracle 运行；它未成为长程 SFT 路径，且
`long_running_sft_allowed=false`。未创建 checkpoint、未进入 P8/P9、未连接 ROS/RTC、
未发送机器人动作。

全部产物均为 `acceptance_status=development_only`、`formal_eligible=false`，trusted
detached signature 与 formal approvals 仍为空。

## 代码与配置改动

- `configs/p7_training_recipe.development.yaml`：冻结 active B4×1 single-pass、独立
  B2×8 exact oracle、`L=L_flow+0.01L_balance+0.001L_z`、global-valid-feature
  flow reduction、1000-update warmup、20000-update cosine decay 与最终 LR `2.5e-6`。
- `tools/preflight_p7_two_pass_gpu.py`：真实 single-pass、独立梯度来源审计、Additive
  对照、全新实例 exact oracle、固定 validation 与 fail-closed P6 prerequisite。
- `tools/build_p7_source_binding.py`：绑定四项 P6 哈希、实际 import roots、测试/JUnit、
  base assets、LeRobot commit/source、ForceSmolVLA source 和 51 文件数据树。
- `tests/test_p7_gate_contract.py`：验证 P6 prerequisite tamper rejection，以及
  single-pass/oracle 的角色隔离。

核心模型源码未改；P4/P5/P6 已通过工件的 SHA256 在 P7 前后保持不变。

## P6 prerequisite 与测试证据

- P6 static spec：`af36eddd...26526f`
- P6 source binding：`dbe7b6ba...48d59`
- P6 resolved config：`701ed2e3...214f`
- P6 gate result：`71ae6020...c706f`
- 完整 CPU suite：140 passed，0 failed，0 errors，0 skipped；28 个测试源码进入 binding。
- dataset storage tree：51 files，SHA256 `f9935b64...a58d`。
- 实际 `forcesmolvla.__file__` 与 `lerobot.__file__` 均位于被哈希的项目/vendor 目录。

## 真实 RTX 4090 D single-pass gate

- 数据：`local/task2_lerobotv3` train split，双相机，H=50，B4×1。
- 模型：ForceToken-MoE，505,620,341 parameters，全部 `requires_grad=True`。
- `vlm_with_expert.forward`：真实 optimizer update 中恰好 1 次。
- loss：`L_flow=10.57443237`、`L_balance=2.59700179`、`L_z=3.66760969`、
  weighted total=`10.60407066`。
- routing：572 valid tokens，counts=`[4,15,549,4]`；四专家均被路由，计数和恰好
  等于 valid tokens，无 token drop。
- base 梯度：vision、VLM text、Action Expert 与 action I/O 均为 100% nonzero；唯一
  missing tensor 是不参与动作 loss 的 `lm_head.weight`。
- CUDA latency：472.918 ms；wall time：0.4731 s。
- process peak allocated/reserved：6,312,640,000 / 6,924,795,904 bytes
  （约 5.88 / 6.45 GiB）。该数值只适用于本次 P7 B4×1 preflight，不能外推为长程训练峰值。

## 独立梯度来源审计

- 零初始化第一步：仅 `L_flow` backward 时，`W_out` weight/bias 均为非零梯度。
- 完成一次 optimizer step 后：仅 `L_flow` backward 时，ForceMLP 4/4、Fusion 36/36、
  Cross-Attention Q/K/V 6/6、conditioner 5/5，以及四个被路由 expert 各 4/4
  parameter tensors 均为有限非零梯度。
- 仅 `L_balance` backward：router 2/2 tensors 非零。
- 仅 `L_z` backward：router 2/2 tensors 非零。

因此 P7 gate 没有用 auxiliary loss 的梯度掩盖 Force residual 未影响动作路径的问题。

## Additive 对照与 exact oracle

- main/Additive：574 个 state tensors 完全相等，参数量同为 505,620,341；Force init、
  optimizer groups 完全相同。
- 相同 validation input/noise/time 下 step-0 native `L_flow` 均为
  `10.27765941619873`。两者唯一结构差异继续是 `Q_main=S+C`、`Q_add=C`。
- exact two-pass：B2×8、16 samples、一个独立 optimizer step；pass-A/B router
  probability max error=`0.0`，2288 valid router tokens，5306 valid flow features，
  route counts=`[19,49,2204,16]`。
- exact oracle CUDA/wall：5083.757 ms / 5.0839 s；process peak allocated/reserved：
  8,444,544,000 / 8,923,381,760 bytes（约 7.86 / 8.31 GiB）。

## 固定 validation

单次 development update 后，两次 single-pass fixed validation `L_flow` 均为
`10.263299942016602`，Python float exact replay。checkpoint selection 仍只允许该
single-pass global-valid-feature-weighted `L_flow`；router auxiliary loss 不参与选择。

## 工件 SHA256

- P7 recipe：`2d5ee53db97fc457a40c042aea07b4325f2a59f2f2b0b5e3d1d5826b874935bb`
- JUnit：`e87cf40a446f92e347568e8d6a9f858fd4f00ae930d403317cdc3e1ae72a0b03`
- source binding：`9b03847c5a627a0e598558accbc3591da1fd44093bbf3f03772c0d5504c04d52`
- resolved config：`05a23602d6142d5c1699fe6189c1e03b36e0c2b0659ab7922cdc4668ed5776c5`
- validation fixture：`547e49166b0df605b144300e9baef899edfa8830924d57f0335d1fc4510e6079`
- GPU result：`56f9db0d767bb8008372f45cbf965ab7886663f1a23c7795834632430aa035d7`

r2 仅因无效的 PyTorch module-hook 计数方法而中止；没有生成 gate/resolved/fixture。
r3 改为包裹真实 `vlm_with_expert.forward`，重新运行 140 tests、重新绑定全部源码后通过。
r2 不得用于 acceptance。

## 剩余 blocker

1. P8 strict save/reload、fresh-process cold start、exact resume dry-run 与 Force full parity
   尚未按当前 P7 binding 执行。
2. 长程 development SFT 在 P8 exact-resume 和 Force parity 通过前继续阻断。
3. P9 offline record/replay Shadow 尚未进入。
4. formal thresholds、trusted detached signature algorithm/key/approver/verifier 仍未批准。

用户 `chanzhang` 的 PID 80524 在 P7 前后均保持运行，占用约 3576 MiB；未终止、未改写。
