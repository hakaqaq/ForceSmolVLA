# P5 v4.2 remediation implementation/preflight report

> Historical notice（2026-08-20）：本报告的 source binding 早于 P4 r4 scoped-bf16 acceptance，且已因后续 spec/Force 源码变化失效。当前有效 P5 证据见 `p5_v4_2_r2_implementation_preflight_report.md`；本文件不得用于进入 P6。

日期：2026-08-20（Asia/Shanghai）  
状态：`P5_DEVELOPMENT_GATE_PASS`  
acceptance：`development_only`；formal eligible：`false`

## 结论

修订后的 P5 已在 RTX 4090 D 上通过。输入为 `task2_lerobotv3` train split 的 4 个真实样本，B4×1、双相机、H=50、7D action、wrench6；执行两次真实 full forward、backward 和 AdamW optimizer step。未使用 CPU fallback、LoRA、冻结 VLM 或简化架构，机器人动作发送数为 0。

本次只通过 P5。按 v4.2 顺序 gate，P6/P7/P8/P9 旧 pass 均为 historical，尚未进入新 P6。

## 代码与契约修订

- P5 从 conversion manifest 读取并校验 repo_id，不再硬编码已不存在的 task1。
- P5 固定四个连续真实 train samples，拒绝不足 B4 的输入。
- source binding 同时验证 LeRobot commit/clean worktree、vendor 文件、完整 ForceSmolVLA Python package、P5/config/spec、base checkpoint、constructor tree 和 task2 conversion/split/normalizer manifests。
- 初始化 seed=42 覆盖 Python、NumPy、PyTorch CPU/CUDA；resolved config 保存初始化 tensor hash。
- 两步梯度 gate：第 1 步允许零初始化 W_out 阻断上游 Force 梯度；第 2 步要求 vision、VLM text、Action Expert、action I/O 和全部 Force tensors 的梯度均 finite/nonzero。

## 测试与测量

| 项目 | 结果 |
|---|---|
| CPU regression | 117 passed |
| GPU | NVIDIA GeForce RTX 4090 D, 24 GiB |
| real batch | task2 train indices `[0,1,2,3]`, B4×1, 2 cameras, H=50 |
| parameters | 483,483,697 total/trainable；0 frozen |
| step-2 vision | 197/197 nonzero-gradient tensors |
| step-2 VLM text | 146/146 |
| step-2 Action Expert | 145/145 |
| step-2 action I/O | 10/10 |
| step-2 Force | 60/60 |
| only allowed grad=None | `model.vlm_with_expert.vlm.lm_head.weight` |
| peak allocated | 8,237,054,976 bytes（7.67 GiB） |
| peak reserved | 8,946,450,432 bytes（8.33 GiB） |
| two-step measured wall | 1.480 s；共享 GPU，因此只作 development measurement |

## 工件

- P5 report SHA256：`0028c8aeb9bd9b62d401ec8ca61644d25dbd4f56cb0da821b45066a54e4034b0`
- resolved config SHA256：`3eb0f737830a86ec739cfc9b695f4ce0ef5422ea54f2fb6b96b46cf5544e2216`
- source binding SHA256：`53798a5c25529cec79fc28013346f59862d377dfc2d8cb7f66f5a1d634f967c2`
- Force initialization tensor SHA256：`72d4ad42eb81dc3197169e7e7506b25b54e985b926eebe18ce87a693c36ad5ec`

## 剩余 blocker

- 进入 P6 需要实验负责人按顺序 gate 明确批准。
- P6–P9 必须基于当前 source binding 重新生成，不得继承旧 pass/checkpoint。
- P4/prefix parity 的正式 atol/rtol、质量/Shadow 阈值、trusted detached signature 算法/key/approver 仍未批准；formal/production resolver 继续 fail-closed。
- P8 strict reload 未重新通过前禁止进入 P9；不连接 ROS/RTC/机器人接口。
