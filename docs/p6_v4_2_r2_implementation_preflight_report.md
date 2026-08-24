# P6 v4.2 r2 DenseParam–MoE implementation/preflight report

日期：2026-08-20（Asia/Shanghai）  
状态：`P6_DEVELOPMENT_GATE_PASS`  
acceptance：`development_only`；formal eligible：`false`；P7 started：`false`

## 结论

P6 DenseParam–MoE construction/equivalence gate 已在 NVIDIA GeForce RTX 4090 D 上通过。两个变体均使用 `task2_lerobotv3` train `[0,1,2,3]`、B4×1、双相机、H=50，执行两次真实全参数训练 + bf16 autocast 混合精度 forward/backward/clip/AdamW step。

模型参数及 optimizer state 实际同时包含 bf16 与 fp32；不得表述为全部参数以 bf16 存储或更新。没有 CPU fallback、LoRA、VLM 冻结、batch 降级或架构简化；没有连接 ROS/RTC，也没有发送机器人动作。本报告只关闭 P6 development gate，尚未进入 P7。

## P5 prerequisite 与执行来源

P6 静态配置和 source binding 精确冻结四项 P5 哈希：

- P5 result：`046ece5dbb4e2c96c605d8b0a5c814add5549af123eda5b5e51097e58f7c1df8`。
- P5 source binding：`541365f0a9cb93c12329d12f508844d4e93b50726c5e0276cc4a0b7a8a202a96`。
- P5 resolved config：`ad31aaf7d71a574782020e7ed097a1810f284a1c7c098f7da1ec15d41621dd5f`。
- P5 static spec：`aed6eb93a8e2f6bd198df59cc69426abc4b9c061dccab39e32ce6ba5b0978156`。

运行时 `__file__` 也已 fail-closed 核验：

- ForceSmolVLA：`/home/rlc123/ForceSmolVLA/src/forcesmolvla/__init__.py`。
- LeRobot：`/home/rlc123/ForceSmolVLA/vendor/lerobot/src/lerobot/__init__.py`。

因此不存在校验目录 A、执行已安装副本 B 的路径漂移。

## 测试与数据树证据

- P6 focused tests：`25 passed`。
- 完整测试：`138 passed`，failures/errors/skipped 均为 0。
- JUnit XML SHA256：`b150e0be94df78ac61183fb8a3e27b744cfeaa7192512d62ab30461826b8c770`。
- P6 binding 包含 27 个 `tests/test_*.py` 的逐文件 SHA256。
- 数据存储绑定包含 47 个 parquet shard 和 4 个 metadata 文件，共 51 个文件；当前数据集没有 video 文件。
- data/meta/videos storage tree SHA256：`f9935b6479dc851e49444669065d20b8aef8cb3ad382f77f53391f701a55a58d`。

## Construction 与 equivalence

| 项目 | DenseParam | MoE |
|---|---:|---:|
| refiner | `960→15364→960` | 4×`960→3840→960` + `Linear(960,4)` router |
| total/trainable/frozen | 505,621,301 / 505,621,301 / 0 | 505,620,341 / 505,620,341 / 0 |
| Force parameters | 55,575,125 | 55,574,165 |
| optimizer tensors | 560/560 exact once | 574/574 exact once |
| step-2 Force nonzero gradients | 60/60 | 74/74 |
| peak allocated | 8,619,155,968 B（8.03 GiB） | 8,608,746,496 B（8.02 GiB） |
| peak reserved | 9,393,143,808 B（8.75 GiB） | 9,344,909,312 B（8.70 GiB） |

共同 Fusion/Adapter 初始化 SHA256 为 `ad9c21f4e7156adcfe66cc5e047d82529d227d961c351cf0173baeca51666152`，两变体一致。由于唯一 `W_out` 零初始化，两变体首步 flow loss 完全相同，absolute difference=`0.0`。

两者第二步均获得 vision 197/197、VLM text 146/146、Action Expert 145/145、action I/O 10/10 非零梯度；唯一允许的 `grad=None` 是不参与动作损失的 `lm_head.weight`。

MoE 为 temperature=1 的 deterministic Top-1、capacity-free、no-drop。两步 route counts 分别为 `[4,16,548,4]` 与 `[4,58,502,8]`，均对应 572 个 valid tokens、dropped=0；第二步四个 expert 全部激活并获得梯度。DenseParam 与 MoE 参数差为 960；MoE active MACs/token=7,377,600，与 DenseCompute reference 7,372,800 的相对差为 0.0651%。

上述显存和时延只代表各变体两步 P6 preflight 的进程内测量，不得外推为长程训练峰值。

## 工件

- static P6 config：`configs/p6_dense_param_moe.development.json`，SHA256 `af36eddd0a1af40d3cdecabdf5f4b3a0bfb85d384208f6d6fc715f908226526f`。
- JUnit：`artifacts/development/p6_v4_2_r2_pytest.xml`，SHA256 `b150e0be94df78ac61183fb8a3e27b744cfeaa7192512d62ab30461826b8c770`。
- source binding：`artifacts/development/p6_v4_2_r2_source_binding.json`，SHA256 `dbe7b6ba2fd1ebe50e61ad1e4015803bf23b0b5c26ec6e9778717f0357f48d59`。
- resolved config：`artifacts/development/p6_v4_2_r2_resolved_config.json`，SHA256 `701ed2e37e0730d9d55f1f765ec88d74ef35f317855e266bca44235d7fbe214f`。
- GPU result：`artifacts/development/p6_v4_2_r2_gpu_preflight.json`，SHA256 `71ae6020770e758d054a8ce3c4ac5b74bec89eb039f5d9310c35130b4b0c706f`。
- DenseParam initialization SHA256：`2a8f3193a277410463752dc3bd8b1e244a36511b782b7c727678565b9d706c90`。
- MoE initialization SHA256：`c94558bf0e4fde02737e5f30969a2dc61f3568aead05630bbfe1a0816bb0f735`。

## Gate 边界

- P7 尚未进入；router auxiliary loss、Additive 对照与 exact two-pass oracle 仍需独立 P7 证据。
- P8 strict reload/resume/full parity 未重新通过，禁止进入 P9。
- formal P4/P8 thresholds、quality/Shadow thresholds 与 trusted detached signature 算法/key/approver/verifier 仍未批准，formal/production 继续 fail-closed。
