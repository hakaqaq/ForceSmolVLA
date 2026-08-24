# P5 v4.2 r2 implementation/preflight report

日期：2026-08-20（Asia/Shanghai）  
状态：`P5_DEVELOPMENT_GATE_PASS`  
acceptance：`development_only`；formal eligible：`false`

## 结论

P5 Dense-Compute 已在 NVIDIA GeForce RTX 4090 D 上通过当前 source-bound gate。输入是 `task2_lerobotv3` train split 的四个真实样本，B4×1、双相机、H=50、absolute action7 经冻结的 delta/normalizer 训练路径、wrench6；执行两次真实 full forward、backward 和 AdamW `optimizer.step()`。

精度语义为全参数训练加 bf16 autocast 混合精度；模型参数与 AdamW 更新状态不是全部以 bf16 存储或更新。

没有使用 CPU fallback、LoRA、冻结 VLM、batch 降级或简化架构；没有连接 ROS/RTC，也没有发送机器人动作。本报告只关闭 P5 development gate，尚未进入 P6。

## P4 入口绑定

P5 在加载模型前和写出结果前各校验一次 P4 r4：

- acceptance config SHA256：`fefe35a92bcbf22de67c7c7b43e9f97d2658afef41745182b2b8207f750592f4`。
- P4 shared source binding：`2afbb09b2702e0bb5b405606fed95e6dfbf7b977e224491a4022788a50d6de68`。
- fp32 artifact SHA256：`9b3a053b5129608cfcebe0198b959522ca59a0878608add7220d9db384598398`。
- bf16 artifact SHA256：`ceadf8df7bec1a7c0ea38924321149c70f90bf6a5ccac339efa29747ab678d95`。

两份 P4 artifact 在 P5 后哈希未改变。P5 runner 同时要求 `gate_status=pass`、`development_only`、`formal_eligible=false`、空 missing/unexpected keys，以及全部 structural exact contract 为 true。

## 代码和契约变化

- P5 静态配置显式绑定上述 P4 r4 证据，并冻结 no-weight-decay 类别：bias、normalization、embedding、alpha、learned action slot。
- P5 runner 的 optimizer 分组与长程 single-pass 训练共用同一参数分类函数；560 个 trainable parameter tensors 恰好各出现一次。
- 新增版本化 source-binding builder；拒绝覆盖已有 evidence，并绑定完整 ForceSmolVLA Python package、P4/P5 runner/config、base assets、LeRobot vendor 和 task2 manifests。
- 旧 P5 artifact 与报告保留为 historical，不用于当前 gate。

## 测试与真实 GPU preflight

| 项目 | 结果 |
|---|---|
| P5 focused tests | 23 passed |
| full CPU regression | 136 passed |
| GPU | NVIDIA GeForce RTX 4090 D, 24 GiB |
| real batch | task2 train indices `[0,1,2,3]`, B4×1, 2 cameras, H=50 |
| parameters | 483,483,697 total/trainable；0 frozen |
| Force parameters | 33,437,521 |
| optimizer partition | 326 decay + 234 no-decay = 560 exact-once tensors |
| step-2 vision | 197/197 nonzero-gradient tensors |
| step-2 VLM text | 146/146 |
| step-2 Action Expert | 145/145 |
| step-2 action I/O | 10/10 |
| step-2 Force | 60/60 |
| only allowed grad=None | `model.vlm_with_expert.vlm.lm_head.weight` |
| peak allocated | 8,238,857,728 bytes（7.67 GiB） |
| peak reserved | 8,944,353,280 bytes（8.33 GiB） |
| two-step measured wall | 1.4283 s（共享 GPU，仅作 development measurement） |

上述 7.67/8.33 GiB 只代表 P5 DenseCompute 两步 preflight 的进程内峰值，不得外推为长程训练或 P6 MoE 的显存需求。

Step 1 loss/gradient norm 为 `11.65648 / 165.27750`；Step 2 为 `3.98119 / 66.47807`。两步的唯一梯度来源均为 `L_flow`，P5 Dense 不含 router auxiliary loss。第一步确认零初始化 `W_out` 获得非零梯度；一次 optimizer step 后，第二步确认 Force upstream 与完整动作基础路径获得有限非零梯度。

## 工件

- static P5 config SHA256：`aed6eb93a8e2f6bd198df59cc69426abc4b9c061dccab39e32ce6ba5b0978156`。
- P5 source binding：`artifacts/development/p5_v4_2_r2_source_binding.json`，SHA256 `541365f0a9cb93c12329d12f508844d4e93b50726c5e0276cc4a0b7a8a202a96`。
- resolved config：`artifacts/development/p5_v4_2_r2_resolved_config.json`，SHA256 `ad31aaf7d71a574782020e7ed097a1810f284a1c7c098f7da1ec15d41621dd5f`。
- GPU preflight：`artifacts/development/p5_v4_2_r2_gpu_preflight.json`，SHA256 `046ece5dbb4e2c96c605d8b0a5c814add5549af123eda5b5e51097e58f7c1df8`。
- Force initialization tensor SHA256：`72d4ad42eb81dc3197169e7e7506b25b54e985b926eebe18ce87a693c36ad5ec`。
- optimizer group-name SHA256：`7c64477f1349803a14b580b9c3fa2fbca63c6f1cd9a52a39b8b10538b4d458a6`。

## Gate 边界与剩余 blocker

- P6 尚未进入，必须由实验负责人按顺序 gate 明确批准。
- P6–P9 的旧报告、binding 和 checkpoint 仍为 historical，不能继承为当前 pass。
- P4 的 development-only bf16 阈值不传播到 P8 或 formal；formal P4/P8 阈值仍为 null/unapproved。
- trusted detached signature 的算法、key、approver/verifier 与 formal quality/Shadow thresholds 仍未冻结，formal/production resolver 继续 fail-closed。
