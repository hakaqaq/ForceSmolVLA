# P6 Dense-Param/MoE implementation and preflight report

> Historical notice（2026-08-20）：v4.2 源码、初始化和 source binding 已改变，本报告仅保留历史测量，不是当前 P6 acceptance；必须在新 P5 后重新 gated 执行。

日期：2026-08-19（Asia/Shanghai）  
状态：`P6_DEVELOPMENT_GATE_PASS`  
acceptance：`development_only`  
formal eligible：`false`  
P7 started：`false`

## 结论

P6 已完成 ForceToken-Dense-Param 与 ForceToken-MoE 的结构、参数/active-compute预算、deterministic top-1 routing和真实 RTX 4090 D preflight。两个变体均使用当前 task1 v4.1 train split真实双相机样本、H=50、offline full-parameter bf16 forward/backward/AdamW step；没有 CPU fallback、VLM冻结、LoRA或结构降级。

本阶段严格停在 P6：没有实现 P7 two-pass router auxiliary loss，也没有实现 Additive adapter。

## 结构与预算

| 项目 | Dense-Param | MoE |
|---|---:|---:|
| context refiner | `960→15364→960` | 4×`960→3840→960` |
| router | 无 | `Linear(960,4,bias=True)` |
| routing | 无 | temperature=1、top-1、capacity-free、zero-drop |
| refiner参数（含LN） | 29,517,124 | 29,516,164 |
| refiner参数差 | 960 | reference |
| 相对参数差 | 0.0032525% | reference |
| Force参数 | 55,575,125 | 55,574,165 |
| 模型总参数 | 505,621,301 | 505,620,341 |
| active MACs/valid token | parameter-match control | 7,377,600 |

Dense-Compute reference active MACs/token为7,372,800；相对 MoE 差异为0.0650618%，满足小于1%的compute-match约束。Dense-Param只标记为parameter-matched，不标记为compute-matched。

MoE实现固定为：独立具名`norm`、`router`和4个`experts.{0..3}`；router logits/softmax为fp32；`torch.argmax`平局取最小expert id；每个valid token只计算一个expert；无capacity、top-2、fallback或token drop。

## 代码改动

- `src/forcesmolvla/force_token.py`：加入DenseParamRefiner、4-expert Top1MoERefiner、RouterState以及共享ForceToken fusion外壳。
- `src/forcesmolvla/configuration_forcesmolvla.py`：注册`force_token_dense_param`与`force_token_moe`，冻结`h_param=15364`、E=4、temperature=1、top_k=1、capacity-free/zero-drop。
- `src/forcesmolvla/modeling_forcesmolvla.py`：按显式variant构造对应refiner；base checkpoint仍只允许当前variant新增参数缺失。
- `configs/p6_dense_param_moe.development.json`：冻结P6参数/计算预算和阶段边界。
- `tools/preflight_p6_variants_gpu.py`：对两个变体分别执行真实数据、全参数、CUDA-only两步gate。

共享fusion/adapter路径保持不变。P5 Dense-Compute seed42初始化回归值仍为：`c8a4cfa58ecac504e24476ee47d921d82f4b516238f898fa691dcfac0f8f8050`。

## 测试结果

- 全套tests：`75 passed`。
- P6专项覆盖：
  - `h_param=15364`唯一最近整数解与小于0.1%参数预算；
  - Dense-Param/MoE实际state_dict参数计数；
  - B=1、batch permutation和竞争co-batch下route id、p-vector与输出不变；
  - capacity-free no-drop和每token仅一个active expert；
  - router tie-break为最小expert id；
  - router logits/probabilities在autocast中仍为fp32；
  - invalid token route=-1、probability/output严格为零；Force slot始终有route。

首次GPU运行在MoE expert bf16输出写入fp32 dispatch buffer时fail-fast，且未生成任何pass artifact。修复后将`p_route * expert_output`显式在fp32计算再写入refiner buffer，并从Dense-Param开始完整重跑；最终结果如下。

## RTX 4090 D real-batch preflight

共同输入：task1 v4.1 train sample 0，batch=1，两相机，H=50，7D action，6D wrench，seed=42。

| 项目 | Dense-Param | MoE |
|---|---:|---:|
| total/trainable/frozen | 505,621,301 / 505,621,301 / 0 | 505,620,341 / 505,620,341 / 0 |
| peak allocated | 5.984 GiB | 5.985 GiB |
| peak reserved | 6.623 GiB | 6.547 GiB |
| step2 forward | 44.562 ms | 44.747 ms |
| step2 backward | 48.152 ms | 49.998 ms |
| step2 optimizer | 106.035 ms | 105.305 ms |
| step2 Force nonzero-grad coverage | 60/60 | 74/74 |
| initialization tensor SHA256 | `9026f598ba2362a0390af30c079401e138713cccea6d9e31402c198a5e4667e1` | `7cc70cb564f039faffcbd5a8bea8b6b2e99896b9f45eec959a5772b31b8a65d1` |

MoE route accounting：

- step1：`[28,4,116,0]`，valid=148，dropped=0；expert3未收到token不是drop。
- step2：`[33,6,108,1]`，valid=148，dropped=0；4个expert均实际激活。

与P5相同，`vlm.lm_head.weight`保持`requires_grad=True`但action路径不调用language generation head，因此为`grad=None`，不是冻结；其余base parameter tensors第二步均得到非零梯度。

## 工件与哈希

- GPU preflight：`artifacts/development/p6_gpu_preflight.json`，SHA256=`adfd4a6e791f1d615ce8191728328cd6b82f246475a29809cf284d9ba11ccde6`
- resolved config：`artifacts/development/p6_resolved_config.json`，SHA256=`2b207d239b7c5964d6dbdc124989585f4130782f9d4e20561986f20b86afcebf`
- source binding：`artifacts/development/p6_source_binding.json`，SHA256=`d04bc9e593e17bb6b04898ca99a65a6e6efc21b5b12cc6cef81b205aaee590ca`
- static P6 spec：`configs/p6_dense_param_moe.development.json`，SHA256=`8fec86d6c316ddc566f2528bce59ff3c6b602bb8166a0a14323ec77c6b996ddf`

## 剩余blocker

- P7未获入口确认：two-pass全局`L_balance/L_z`和parameter/init-matched Additive尚未实现。
- P8 strict checkpoint reload/cold-start尚未执行，因此不能进入P9。
- trusted detached signature算法、密钥、批准人以及formal thresholds仍未冻结。
- 全部P6工件仅允许development SFT/checkpoint，不得用于formal checkpoint/evaluation/Shadow acceptance。
