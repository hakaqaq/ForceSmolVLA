# ForceSmolVLA implementation specification v4.2

状态：development-only source-of-truth  
日期：2026-08-20（Asia/Shanghai）  
平台：单张 RTX 4090D 24 GiB；独立 Conda 环境  
继承：v4.1 available-sensor 数据/几何契约；本文件覆盖其训练、推理、验收和 checkpoint 冲突项。

## 1. 方法定位与理论动机：支持离线全模型适配的紧凑型力觉 Flow-VLA

我们以约 450M 参数的 SmolVLA 为骨干，构建完整规模约 505.6M 参数的力觉条件化 Flow Actor，使动作计算路径中的视觉—语言主干、力觉融合模块与 Action Expert 能够在离线阶段进行端到端联合适配。相较 ForceVLA 所依赖的约 3.3B 参数级 `π0`，该设计显著降低了完整力觉策略适配的计算门槛，并为后续消费级 GPU 上冻结 VLM 的在线更新提供紧凑的 Actor 初始化。

这一改动并非将 ForceVLA 中的 `π0` 直接替换为较小的 VLA。ForceVLA 建立在 `π0` 的双专家联合 self-attention 拓扑之上，其 post-VLM force-fusion 分支能够构造与动作序列对齐的加性 guidance。相比之下，SmolVLA 将机器人状态纳入 VLM prefix，并通过交错的 prefix cross-attention 与 causal action self-attention 生成动作。其 post-VLM prefix 不包含与未来 `H` 个动作位置天然对齐的表示，尽管 Action Expert 内部仍具有 `H` 个 action suffix hidden。因此，ForceVLA 的 action-aligned additive interface 无法在保持动作对齐与缓存语义的条件下直接迁移到 SmolVLA。

为解决这一结构不兼容问题，我们保留 ForceVLA-inspired post-VLM force fusion，在原生 prefix K/V cache 之外构建 observation-conditioned Force Context，并提出 Action-Query Force Residual Adapter。该适配器以当前 Action Expert hidden 为主要查询，同时显式结合 noisy action、动作位置和 flow timestep，从固定的 Force Context 中动态检索与当前动作生成状态相关的力觉信息。由此，ForceVLA 中 guidance 分支不显式以当前 noisy action、flow timestep 和 Action Expert hidden 为查询的加性 guidance，被重新构造为 denoising-state-conditioned force residual。

Force 分支使用 post-VLM contextual prefix hidden 构建 Force Context，但不接收、不拼接或修改 SmolVLA 原生 `past_key_values`；原生 denoiser 的 suffix K/V append–crop 过程保持不变。因此，该方法在离线阶段支持完整力觉 Actor 的端到端学习，同时保留 SmolVLA 的 cached flow-generation 路径，并为后续冻结 VLM、仅更新力觉—动作模块与 Actor–Critic 的在线阶段提供模型和计算基础。

继承与创新边界固定如下：

- 继承自 ForceVLA 的思想：post-VLM force fusion 与稀疏 MoE force refinement。
- 本项目的新结构：Action-Query Force Residual Adapter，以及它对 SmolVLA cached Flow Action Expert、action suffix、noisy action 与 flow timestep 的适配。
- v4.2 当前证明范围：离线全参数 Force-conditioned Actor；冻结 VLM 的在线 Actor–Critic 仅是后续能力，不得表述为已经实现或验收。

## 2. 优先级与边界

1. 本文件是当前实现的最高优先规格；v4.1 仅提供未被本文件覆盖的数据/几何细节。
2. P5→P6→P7→P8→P9 必须顺序 gated。任一阶段源码、配置或绑定工件改变后，该阶段及所有下游旧 pass 自动失效。
3. 当前 P5–P9 仅允许 development SFT、synthetic/smoke、development checkpoint 和纯离线 replay。
4. 不连接 ROS、Franky、RTC 或机器人控制接口，不读取实时控制队列，不发送机器人动作。
5. production/formal resolver 对 null、未批准 provenance、缺可信 detached signature、缺 clock map 或不匹配工件一律 fail-closed。签名算法、key、批准人和 verifier 未冻结前，任何自称 `verified` 的字段都不能解锁 formal 模式。

## 3. 数据契约

- raw task1/task2 永远只读；转换数据写入独立 LeRobot v3 目录，不覆盖 raw。
- 数据模态固定为两路 RGB、7D measured TCP state、calibrated TCP wrench6、7D absolute target 和 prompt。
- wrench 几何使用约 100 Hz measured TCP pose；对 wrench timestamp 仅 latest-causal ZOH，禁止 future interpolation。
- 不要求 joint-q、joint FK、1 kHz pose、2 ms pose age、每 session calibration bundle 或 session-disjoint split。
- split 固定 episode-disjoint；当前结论只能称为 within-session offline fine-tuning，不声称跨 session 泛化。
- episode/tuple eligibility、causal timestamps、calibration/geometry hashes、repo_id、split 与 conversion manifest 必须逐项 fail-closed。
- normalizer 只拟合 train episodes。state7/wrench6 每个 eligible tuple 各贡献一行；delta-action7 必须与真实 H=50 训练标签完全同分布：每个 `t_ref` 使用该时刻 raw measured state7，将同 episode 内未来最多 50 个 valid absolute action7 整个 chunk 经 `ActionDeltaProcessor.to_delta` 转换后逐 valid target 拟合，right-padded tail 不得参与；val/test 不得参与。该定义与已验证 ForceVLA 的“前 6 维 chunk-relative、夹爪绝对值”处理一致。
- 上述混合目标的规范名称为 `action_target7=[delta_xyz, delta_rpy, absolute_gripper_width_m]`；`delta_action7` 仅作为现有 Python/checkpoint 键名保留，不表示第 7 维为 delta。
- `ActionTargetPopulationParityGate` 必须由独立 oracle 重建 train split 的完整有效 `(episode_id, anchor_t, horizon_k)` 总体，并 exact 比较 valid pair count、有序 pair hash、展平 float64 target tensor hash、逐维 mean/std/min/max/分位数及 k=0..49 分 horizon 统计。改变 masked padding 值或加入 val/test 不得改变结果；gripper 在 to/from delta 前后必须保持绝对宽度；必须包含 `action[t+k]-state[t] != action[t+k]-state[t+k] (k>0)` sentinel。ForceVLA 数值只能作为辅助证据，不能作为 acceptance oracle。
- `observation.wrench` 必须显式出现在 checkpoint input schema，类型 ENV、shape=(6,)；继承的 STATE/ACTION/ENV normalizer 均为 Identity/disconnected，自定义 normalizer 是唯一 owner。

## 4. 冻结 SmolVLA 拓扑

基于 pinned LeRobot v0.6.0 commit `30da8e687a6dfc617fcd94afc367ac7071c376ce` 和本地 smolvla_base revision `d5ef92b547b2bf36bdd50f18ea6ed6463cb5c5af`：

```text
resize_imgs_with_padding = [512,512]
use_cache = true
add_image_special_tokens = false
attention_mode = cross_attn
num_vlm_layers = 16
num_expert_layers = 0       # pinned checkpoint 的实际值；<=0 表示与 VLM 同层数
self_attn_every_n_layers = 2
expert_width_multiplier = 0.75
tokenizer_max_length = 48, right-pad/right-truncate
prefix physical layout = camera1[0,64), camera2[64,128), language[128,176), state[176,177)
```

真实 `embed_prefix()` 的 valid mask、span、physical length 和 full/prefill hidden parity 必须验收，不能只测硬编码常量。

## 5. Force 架构

- 主模型是 ForceToken-MoE；Dense-Compute、Dense-Param、Additive 和 Cartesian7D 是独立对照。
- Force fusion 选择 `[0,176)`，排除 state token；force slot=176，物理长度=177。
- FusionBlock 是 2 层、8-head；ForceCrossAttention 独立固定 single-head、D=720、scale=`1/sqrt(720)`。
- Q/K/V 和唯一 `W_out` 均为 `Linear(720,720,bias=True)`；禁止 `nn.MultiheadAttention`、head split 和额外内部 O/out projection。
- `W_out` weight/bias 零初始化；Q/K/V、logits、softmax、W_out、residual add 和 action output head 在 autocast-disabled fp32 区域。
- invalid key 在 softmax 前为 `-inf`；invalid query 输出严格为零；force slot 永远 valid。
- Force K/V 每个 action chunk 只投影一次，10 个 Euler step 复用 `PreparedForceContext`；动态 step 只重算 Q。
- Dense/MoE 的 ForceMLP、segment/position embedding、FusionBlocks、guidance projection 和 adapter 必须使用分模块 seed，从相同公共初始化开始；只允许 variant-specific refiner 不同。
- MoE router 固定为 `Linear(960,4,bias=true)`，weight 使用 seed=42 派生的确定性 `Normal(0,0.02)`、bias 为 0；该尺度继承 ForceVLA/FlaxFormer `RouterWeights` 默认初始化。为保持 capacity-free deterministic Top-1 契约，不使用 ForceVLA 的 routing jitter。

### 5.1 Action-Query Adapter 唯一公式

对 flow step `t` 和 action position `k`：

```text
C[t,k] = learned_action_slot[k]
         + W_a(noisy_action7[t,k])
         + W_t(t)

q[t,k] = suffix_out[t,k] + C[t,k]
delta_h[t,k] = tanh(alpha) * W_out(
    Attn(W_Q(q[t,k]), prepared_K_force, prepared_V_force)
)
refined_suffix[t,k] = suffix_out[t,k] + delta_h[t,k]
velocity[t,k] = shared_fp32_action_out_proj(refined_suffix[t,k])
```

- `learned_action_slot[k]` 是唯一 action-position embedding；`W_a=Linear(7,720,bias=true)`，`W_t=Linear(1,720,bias=true)`。
- `alpha_init=atanh(1e-3)`；`W_out` weight/bias 全零初始化。residual 只注入 Action Expert 的最后 H 个 suffix hidden，不能写回 VLM prefix/cache。
- Additive 对照的唯一差异为 `q_additive[t,k]=C[t,k]`，即只移除 `suffix_out[t,k]`；所有参数名、shape、tensor 初始化、数据、mask、precision、noise、timestep 和 optimizer 必须相同。
- MoE 固定 4 experts、deterministic Top-1、capacity-free、no token drop、no fallback expert；logit tie 由 `argmax` 选择最小 expert id。只执行被选 expert。

### 5.2 PreparedForceContext 生命周期

`PreparedForceContext` 除 fp32 K/V 与 fused-valid mask 外，必须绑定：

```text
chunk_id[B]
sample_id[B]
context_generation
model_generation
device
dtype=torch.float32
```

新 chunk/sample、policy reset、任一参数被 optimizer step 原地更新、device 或 dtype 改变后，旧 PreparedForceContext 必须 fail-fast 为 stale。它只在一次 action chunk 的 10 个 Euler steps 内有效，不能保存进 checkpoint、跨调用复用或成为 prefix cache 的一部分。

## 6. 离线训练

当前 `offline_full_finetune`：

- 所有现存 VLM、vision encoder、Action Expert、action I/O 和 Force 参数 `requires_grad=true`；不得 LoRA、不得冻结、不得静默减结构。
- AdamW：lr=1e-4、betas=(0.9,0.95)、eps=1e-8、weight decay=1e-10、grad clip=10。
- B4×gradient-accumulation-1；H=50；双相机；bf16 outer autocast + 已声明 fp32 islands。
- 当前 task2 development SFT 主预算=40,000 samples；10,000 optimizer updates 仅是派生值。
- 长程 SFT 每 batch 只允许一次共享 full forward、一次 backward、一次 optimizer.step。
- 唯一训练目标为 `L = L_flow + 0.01*L_balance + 0.001*L_z`。`L_flow` 用整个物理 batch 的 global valid-feature count 归一化，不能按 sample/chunk 平均后再平均。
- scheduler 继承 1,000/20,000 preset，并按 LeRobot 短训练规则缩放为 warmup 500 updates，随后 cosine decay，到第 10,000 update 为 `2.5e-6`；peak LR 为 `1e-4`。
- bias、任意 normalization 参数、embedding、`alpha` 和 `learned_action_slot` 的 weight decay 为 0；其他参数为 `1e-10`。每个 trainable parameter 必须恰好出现在一个 optimizer group。
- P7 exact two-pass 仅是短程 routing/gradient/numerical oracle；不得进入长程 SFT。
- training 与 best-checkpoint validation 均调用 `forward_single_pass_training_terms()`；selection metric 仅使用全局 valid-feature 加权 `L_flow`。
- 当前固定两样本 validation 只允许 development checkpoint selection；formal checkpoint selection 必须遍历完整 held-out val split，并另行冻结计算预算。
- 真实梯度审计至少两步：第一步因 W_out=0 允许 Force 上游零梯度；第二步 vision、VLM text、Action Expert、action I/O、ForceMLP/Fusion/被路由 experts/QKV/conditioner 必须有 finite nonzero gradient。唯一允许的 base `grad=None` 是未参与 action loss 的 `vlm.lm_head.weight`。
- 梯度来源必须分开审计，不能只对 total loss 做一次 backward：
  1. 初始化时仅对 `L_flow` backward，`W_out` 必须 finite/nonzero；
  2. 完成一次 optimizer step 后，仅对 `L_flow` backward，ForceMLP、Fusion、adapter Q/K/V、conditioner 和被路由 expert 必须 finite/nonzero；
  3. 对 `0.01*L_balance+0.001*L_z` 单独 backward，router weight/bias 必须 finite/nonzero。

未来 `online_hil_vlm_frozen` 是独立阶段：届时才冻结 VLM/vision。跨阶段不得恢复 optimizer/scheduler/scaler/accumulation state。

## 7. 推理 API 与 action 语义

- 私有 `_predict_normalized_delta_chunk()` 返回 normalized delta7，不消费 ChunkContext。
- 公共 `predict_action_chunk()` 是唯一 handoff API，使用 `@torch.inference_mode()` 并在方法内部执行 `self.eval()`；不得要求调用方代做。内部按唯一顺序执行：

```text
normalized delta7
-> custom action unnormalize exactly once
-> fail-closed binary gripper decode (candidate [-0.01,0.095] m; threshold 0.0425 m; output {0,0.085} m)
-> whole-chunk ActionDeltaProcessor.from_delta(raw_state_snapshot)
-> ZYX principal-chart canonicalization
-> intrinsic RuleSpec safety checks
-> raw absolute [B,H,7]
```

- 公开输出第 7 维永远是 `target_gripper_width_m` absolute width，范围 [0,0.1] m；不得转换为 controller position。task2 的训练 target population 在该维只有 `{0,0.085}` m，因此公共 API 使用与已验证 ForceVLA 控制路径一致的显式二值解码：连续模型候选必须先落入 `[-0.01,0.095]` m，否则 fail-closed；候选 `<0.0425` m 解码为 `0` m，候选 `>=0.0425` m 解码为 `0.085` m。该步骤是二值控制语义解码，不是 clipping；前 6 维不得改变。
- mixed delta 定义固定为：

```text
delta7 = [target_xyz - raw_state_snapshot_xyz,
          wrap_to_pi(target_rpy - raw_state_snapshot_rpy),
          target_gripper_width_m]
```

  第 7 维在 forward/inverse 中都不得与 state 相减或相加。
- roll/yaw∈[-pi,pi)，pitch∈[-pi/2,pi/2]；距 gimbal lock 小于 2° 必须拒绝。
- workspace、orientation、adjacent translation/SO(3)、gripper range/rate 和首目标 continuity 必须来自显式 hash-bound RuleSpec；禁止 Python 隐式生产阈值。
- 任一 batch 元素 inverse、artifact 或 safety gate 失败时，整个 batch 的 chunk ID 均不消费；仅在全 batch 成功后原子消费全部 ID。
- private/public 输出都固定 `[B,H,7]`、`torch.float32`、与模型输入相同 device。invalid right-padded horizon tail 在 public 输出中必须精确为 0，并由同一 ChunkContext mask 保证永不进入 dispatch；本项目禁用 inherited `select_action`/RTC queue。
- `raw_state_snapshot` 必须是与该 sample 的 `t_ref` 相同、未归一化的 measured TCP state7；不允许调用方用 processor 输出或其他时刻 state 替代。
- public failure 必须抛出带稳定 `.code` 的 `ActionInferenceError`。完成 custom inverse 后禁止任何第二个 LeRobot action postprocessor/unnormalizer。
- checkpoint 必须保存 ActionDeltaSpec、normalizer manifest 和 processor graph；调用前必须重算并匹配 normalizer/calibration/geometry hashes。RuleSpec 在阈值未批准时只能作为显式 runtime test-only binding，不能伪装成 checkpoint 内的 production RuleSpec。
- Shadow 的 clock/age/transport/dispatch continuity 仍由 P9 resolver/arbiter 处理，不能由 policy 假装完成。

## 8. Checkpoint/resume

- checkpoint 必须保存模型、constructor assets、trainability manifest、source binding、resolved config、dataset/conversion/split/normalizer/calibration/geometry manifests、optimizer/scheduler/scaler/RNG/sampler 和 accumulation state。
- active SFT checkpoint batching 固定 B4×1，不得保留旧 B2×8 元数据。
- resume 在恢复 optimizer state 前逐项比较 training stage、source-binding hash、resolved-config hash、dataset manifest hashes、optimizer groups、B4×1 accumulation phase 与 sampler binding。
- 每个新 best validation 必须对应一个已落盘且完整校验的 checkpoint；不得只更新 best scalar。
- strict reload 必须 local-only、断网、fresh process；exact fileset、payload SHA256 和 trainability 必须匹配。

## 9. Gate 顺序、parity 与 P9

顺序固定为：

```text
P4 bare SmolVLA topology/layout/full-prefill/cache baseline
 -> P5 Dense
 -> P6 MoE
 -> P7 single-pass training + exact-two-pass oracle
 -> P8 Force-MoE checkpoint/reload/resume/full parity
 -> P9 pure-offline Shadow replay
```

- P4 只验裸 Cartesian7D/SmolVLA baseline，不依赖尚未接受的 Force 模块。
- P8 使用真实 ForceToken-MoE，必测：zero-init Force/Cartesian7D 共同 fp32 output parity、真实 prefix layout、per-sample valid length、valid full/prefill hidden、B>=2 heterogeneous language、batch/single、one-step cached/full、完整 10-step cached/uncached、raw-state reachability/state-leakage、state/action/noise `[7:32]` 独立扰动、invalid-tail 扰动、cache unchanged 和 Force K/V 一次投影。
- debug cache 的全 tensor clone/compare 只能在 gate 中启用，不能进入训练或正式推理热路径。
- parity `atol/rtol` 必须来自 versioned、hash-bound acceptance config；禁止操作者用 CLI 临时放宽。development 数值必须明确 non-production，未批准不得成为 formal threshold。
- 80k-sample development SFT 只有在 P8 exact-resume dry-run 通过后才允许启动。P9 只 gate Shadow，不反向阻止已通过 P8 的纯离线 SFT。
- P9 仅做 task1_v4_1 + golden/test fixtures 的纯离线 record/replay；production 缺 trusted clock map 时 `candidate_valid=false`。
- P9 写 `gate_status=pass` 前必须断言 replay exact、absolute finite、candidate validity 与 reasons 自洽、outcome 与 test-only expected fixture 相同、dispatch indices 精确且 invalid candidate 无 dispatch。
- synthetic/test-only clock、threshold、key 和数值不得进入 formal command/checkpoint/report，也不得回填为候选生产阈值。

## 10. 当前验收状态

本次新增生命周期/API/梯度来源契约会再次改变 initialization/source/config hashes，因此此前 P4、P5、P6、P7、P8、P9 development pass 全部降级为 historical evidence。必须从 P4 baseline 开始顺序重跑；P4 未通过不得进入 P5，P5 未通过不得进入 P6/P7，P8 strict reload 未通过不得启动 80k-sample development SFT 或进入 P9。

当前允许：CPU unit/static regression、数据只读验证、development smoke、显式 test-only RuleSpec。  
当前禁止：沿用旧 checkpoint acceptance、正式训练/评测、production Shadow、在线 HIL、机器人动作发送。
