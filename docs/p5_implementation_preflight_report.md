# P5 ForceToken-Dense-Compute implementation/preflight report

> Historical notice（2026-08-20）：本报告属于旧 v4.1 source/config。当前有效 P5 证据见 `p5_v4_2_remediation_preflight_report.md`；旧 hash、B1 测量和 downstream acceptance 不再有效。

日期：2026-08-19（Asia/Shanghai）  
状态：`P5_DEVELOPMENT_GATE_PASS`  
acceptance：`development_only`  
formal eligible：`false`

## 结论

P5 已按 v4.1 冻结结构实现，并在 NVIDIA GeForce RTX 4090 D 上用当前 task1 v4.1 train split 的真实双相机样本完成 H=50、全参数 bf16 forward/backward/AdamW optimizer.step gate。未使用 CPU fallback、冻结 VLM、LoRA 或简化结构。P6 尚未开始，等待 P5 报告获确认后再进入。

## 冻结结构

- fusion physical spans：camera1=`[0,64)`、camera2=`[64,128)`、language=`[128,176)`；state token excluded；Force slot=`176`；`N_fused_physical=177`。
- ForceMLP：`6 -> 960 -> 960`，SiLU，无 raw-wrench LayerNorm。
- FusionBlock：2 blocks、8 heads、FFN `960 -> 3840 -> 960`、GELU、dropout=0。
- DenseCompute refiner：`960 -> 3840 -> 960`；active MACs/token=`7,372,800`，与 MoE reference `7,377,600` 的相对差为 `0.0650618%`。
- guidance projection：`960 -> 720`，fp32。
- ForceCrossAttention：严格 single-head，`num_heads=1`、`head_dim=720`、`scale=1/sqrt(720)`；显式 Q/K/V；没有内部 O/out_proj/`nn.MultiheadAttention`。
- adapter 唯一输出投影为 `W_out=Linear(720,720,bias=True)`，weight/bias 全零初始化；`alpha=atanh(1e-3)`。
- Q/K/V、logits、softmax、W_out、residual add 和原生 action output head 均在 autocast-disabled fp32 区域。
- 初始化 seed=42，同时固定 Python、NumPy、PyTorch CPU/CUDA RNG；初始化 tensor SHA256=`c8a4cfa58ecac504e24476ee47d921d82f4b516238f898fa691dcfac0f8f8050`。

## 代码改动

- `src/forcesmolvla/force_token.py`：ForceMLP、masked 8-head FusionBlock、DenseCompute refiner、ForceContext、严格单头 ForceCrossAttention 和零初始化 residual adapter。
- `src/forcesmolvla/modeling_forcesmolvla.py`：将同一 adapter hook 接入 training forward、reference full-prefix velocity 与 cached denoise；P5 强制 normalized wrench6；cached path 每个 observation 只构造一次 ForceContext。
- `src/forcesmolvla/configuration_forcesmolvla.py`：冻结 P5 variant、development-only、seed、2×8-head fusion 与 1-head adapter字段。
- `src/forcesmolvla/checkpoint.py`：base load 仅允许新增 P5 state keys 缺失；任何其他 missing/unexpected key fail-fast。
- `configs/p5_force_token_dense_compute.development.json`：完整静态 resolved 候选，明确 formal=false、签名/批准为 null。
- `tools/preflight_p5_dense_compute_gpu.py`：CUDA-only真实数据 gate；禁止 CPU fallback 和 OOM降级；验证 source hashes、参数集合、显存、延迟及两步梯度覆盖。

## 测试结果

- 全套 unit/static tests：`66 passed`。
- 验证内容包括：精确 camera/language spans、state exclusion、right-padding mask、Force slot always-valid、single-head公式、invalid-key `-inf`、invalid-query strict zero、无内部 out projection、初始 residual parity、W_out首步非零梯度、seed42 hash确定性。
- base checkpoint：518 source tensors；allowlist丢弃18个旧 SO100 normalizer tensors；500个 base model tensors加载；missing恰好等于60个新增 P5 parameter tensors；unexpected=0。

## RTX 4090 D real-batch preflight

| 项目 | 结果 |
|---|---:|
| GPU | NVIDIA GeForce RTX 4090 D，24 GiB |
| 数据 | `task1_forcesmolvla_v4_1` train，sample 0 |
| batch / cameras / H | 1 / 2 / 50 |
| total/trainable/frozen parameters | 483,483,697 / 483,483,697 / 0 |
| Force parameters | 33,437,521 |
| peak allocated | 5,973,689,856 bytes = 5.563 GiB |
| peak reserved | 6,673,137,664 bytes = 6.215 GiB |
| step 1 forward/backward/optimizer | 325.653 / 108.661 / 145.598 ms |
| step 2 forward/backward/optimizer | 44.971 / 50.702 / 98.889 ms |
| step 1 loss | 6.5666728 |
| step 2 loss | 5.5999827 |
| step 2 Force nonzero-gradient coverage | 60/60 tensors = 100% |
| CPU fallback / architecture downgrade | false / false |

第1步由于唯一 `W_out` 为零初始化，Force上游分支按设计得到零梯度，而 `W_out.weight/bias` 两个张量得到非零梯度。第一次 optimizer.step 后，第2步全部60个 Force parameter tensors均得到非零梯度。

base 参数中 `vlm.lm_head.weight` 保持 `requires_grad=True`，但 action训练路径不调用语言生成 head，所以两步均为 `grad=None`；这属于“可训练但本路径未使用”，不是冻结。其余499/500个 base parameter tensors在第2步均有非零梯度。

## 工件与哈希

- GPU preflight：`artifacts/development/p5_gpu_preflight.json`，SHA256=`ebf22cd13b3b446d69f86b3414782d2195918afd24838649e1f26db041a5b491`
- resolved config：`artifacts/development/p5_resolved_config.json`，SHA256=`2d2f17f2aab90a2105c8a98ffca2ef11b1abea4b6120ab683793297f2bd44588`
- source binding：`artifacts/development/source_binding.json`，SHA256=`efdca58a4fabf8ea1ebde719e3fb3e09de788af169b60b826b3a5c1b52de594b`
- static P5 spec：SHA256=`38d7a3d847c8a53cef36cb55a70c6fcb438d6600c37a241f5e10bfa54a1082b9`

## 剩余 blocker

- P6 尚未获本阶段入口确认，因此没有实现 MoE/router/two-pass loss。
- P8 strict checkpoint reload 尚未执行；在它通过前不能进入 P9。
- trusted detached signature 的算法、密钥、批准人仍未冻结。
- formal quality/shadow thresholds仍缺批准；当前所有 P5产物只能用于 development SFT/development checkpoint，不能用于 formal checkpoint acceptance、formal evaluation 或 Shadow acceptance。
