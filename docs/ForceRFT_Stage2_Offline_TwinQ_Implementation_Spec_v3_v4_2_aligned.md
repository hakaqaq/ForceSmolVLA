# ForceRFT 第二阶段：基于 v4.2 的离线 Force-aware Twin-Q 强化微调实施规范

> 文档类型：可直接交给 Codex 的实现规格  
> 版本：v3（ForceSmolVLA v4.2 对齐版）  
> 状态：development-only；不授权在线 HIL 或真机执行  
> 第一阶段唯一基线：`ForceSmolVLA_Implementation_Spec_v4_2`  
> 论文方法名：**ForceRFT**  
> 现有代码包名：`forcesmolvla`（第二阶段不重命名）  
> 初始化：通过 v4.2 验收并完成离线 SFT 的 ForceSmolVLA Actor checkpoint  
> 第二阶段算法标识：`offline_force_rft`（只属于 RFT trainer，不写入 Actor 的 `training_stage`）

---

## 0. 本版相对第二阶段 v2 的关键修订

本规范不是重新实现第一阶段，而是在 v4.2 Actor 外部增加离线
Actor–Critic 训练层。以下修改用于消除第二阶段 v2 与 v4.2 的冲突，并降低
对已验证第一阶段代码的侵入。

1. **第一阶段 Actor 配置保持不变。**  
   Actor 的 `training_stage` 继续为 v4.2 已支持的
   `offline_full_finetune`。第二阶段阶段名只记录在外部
   `ResolvedOfflineRFTConfig` 和 RFT checkpoint 中。不得为了表示 RFT
   而修改 `ForceSmolVLAConfig.__post_init__()`、
   `apply_training_stage()` 或公共 checkpoint config。

2. **采用 sidecar 实现，不修改 v4.2 核心推理路径。**  
   第二阶段优先只新增 `src/forcesmolvla/rft/` 下的模块。直接调用现有
   `policy.model.sample_actions_masked()` 完成带梯度的十步 Flow sampling；
   不修改 `modeling_forcesmolvla.py`、`context.py`、
   `configuration_forcesmolvla.py`、`serve_policy.py` 和真机部署代码。

3. **不再把 controller ACK、formal RuleSpec 或签名验证作为离线 RFT
   前置条件。**  
   第二阶段使用 LeRobot v3 记录数据的离线 transition 语义。controller
   apply/ACK、真实调度延迟、formal safety authorization 属于第三阶段在线
   HIL/真机部署，不应反向否定第一阶段 checkpoint。

4. **Critic 的动作语义固定为单个 7D recorded target，而不是完整
   `H=50` 未执行计划。**  
   Actor 仍生成完整 `H=50` chunk，但离线 Q 默认评价 chunk 的第 0 个动作：
   `critic_action = normalized_action_chunk[:, 0, :]`。这避免用一个
   30 Hz next observation 为未执行的未来 49 个动作做错误归因。

5. **删除 `execution-equivalent ST projection`。**  
   v4.2 公共 API 对非法 whole-chunk action 执行 fail-closed，而不是将其
   clamp/project 到可执行域。因此，第二阶段不得声称训练时 hard projection
   与部署完全等价。Q 路径工作在冻结的 normalized `action_target7` 空间；
   v4.2 public inverse/RuleSpec 仅用于 detached validation。

6. **父 normalizer 只读复用，绝不在 RFT transition 上重新拟合。**  
   `action_target7` 的语义继续为
   `[delta_xyz, wrapped_delta_rpy, absolute_gripper_width_m]`；Python 中保留的
   `delta_action7` 键名不改变该语义。

7. **第一阶段 checkpoint 与证据不重写。**  
   Stage-2 使用 zero-update compatibility bridge 验证父 Actor；P9 只 gate
   Shadow，不是纯离线 RFT 的前置条件。第二阶段所有 artifact 写入独立目录。

8. **第二阶段 checkpoint 采用外层容器。**  
   `actor/` 子目录保持现有 v4.2 Actor checkpoint 格式；Critic、target、
   optimizer、RNG 和 transition 工件存放在外层 RFT checkpoint。现有
   `serve_policy.py` 只加载导出的 `actor/`，不需要理解 Critic。

9. **理论名称收紧。**  
   本阶段称为 `ConRFT-compatible Cal-QL-style estimator`。单轨迹 Monte
   Carlo return 是 empirical calibration reference，不宣称构成一般条件下
   可证明的行为价值下界。

10. **以父 checkpoint 的实际 resolved config 为准。**  
    v4.2 正文的有效第一阶段预算是 `40,000 samples / 10,000 updates`；其
    后部残留的 `80k-sample` 表述不得用于选择父 checkpoint。启动时必须从
    父 checkpoint 的 `resolved_training_config` 读取并绑定真实计数，禁止
    通过实验目录名猜测。

---

## 1. 研究目标、实现范围与非目标

### 1.1 第二阶段目标

从已完成第一阶段 SFT 的 ForceSmolVLA Actor 出发，新增两个相互独立的
force-aware critics (Q_{phi_1},Q_{phi_2}) 及对应 target critics，利用
离线 transition 数据实现：

1. force-aware Twin-Q 的离线 TD 学习；
2. ConRFT-compatible Cal-QL-style conservative calibration；
3. 价值梯度穿过 SmolVLA 原生 (N=10) Euler Flow integration；
4. Flow Matching、MoE auxiliary objective 与 Twin-Q guidance 的联合 Actor
   更新；
5. 离线阶段继续全参数训练 v4.2 Actor，包括 Vision、VLM、Force、Action
   Expert 与 Action I/O；
6. 在 RTX 4090D 上完成可恢复、可复现的 development 训练闭环。

### 1.2 本阶段明确不实现

- 在线 HIL 数据采集；
- 在线 replay buffer；
- 人工干预数据混合；
- 冻结 VLM 的在线更新；
- controller apply/ACK 对齐；
- ROS、RTC、Franky、SpaceMouse 或机器人动作发送；
- formal RuleSpec 签名与生产授权；
- 修改已有 v4.2 public action API；
- 将 SmolVLA Flow Action Expert 替换为 consistency-policy head；
- 将十步 Flow 简化为单步动作头。

第三阶段才实现 `online_hil_vlm_frozen`：继承第二阶段 Actor、Q1/Q2 和
targets，冻结 VLM/vision，继续更新 Force–Action 子空间与 Twin-Q。

---

## 2. 从 v4.2 无条件继承的 Actor 契约

下表中的第一阶段行为在第二阶段均不得改变。

| 类别 | v4.2 冻结内容 | 第二阶段要求 |
|---|---|---|
| 上游版本 | LeRobot commit `30da8e687a6dfc617fcd94afc367ac7071c376ce`；SmolVLA base revision `d5ef92b547b2bf36bdd50f18ea6ed6463cb5c5af` | 写入 parent manifest，不跟随 `main` |
| 输入 | 双相机、prompt、state7、wrench6 | current/next observation 使用相同顺序和预处理 |
| Prefix | camera1 `[0,64)`、camera2 `[64,128)`、language `[128,176)`、state `[176,177)` | 不向 prefix/cache 插入 Critic token |
| Force | post-VLM Force Context、4-expert capacity-free deterministic Top-1 MoE | 完整继承；不重初始化 router/expert/adapter |
| Adapter | Action Expert hidden + noisy action7 + action position + flow timestep | Q 梯度通过现有 adapter，不新增另一套 force 注入 |
| Flow | `H=50`、`N=10`、cached prefix、suffix append→crop | Q guidance 必须穿过完整十步，不 detach |
| 维度 | 模型内部 32D，真实 state/action 仅前 7D | 后 25 维 noise/action/gradient exact zero |
| 动作 | `action_target7=[delta_xyz, wrapped_delta_rpy, absolute_gripper_width_m]` | Critic 只读 normalized 7D，不把夹爪变成 delta |
| Normalizer | v4.2 custom normalizer exactly once；继承 normalizer disconnected | Stage-2 禁止重新拟合或二次 normalize |
| 训练 | `offline_full_finetune` 时 Actor 全参数可训练 | RFT Actor 仍保持该 stage |
| 推理 | public API 内部 eval/inference、unnormalize、二值夹爪、inverse、RuleSpec | 完整回归；不进入 Actor Q-gradient 路径 |
| 缓存 | Force branch 不读写 prefix K/V；Force K/V 每 sampling call 投影一次 | Stage-2 每个 sampling call 重新构造 ephemeral context |

父 Actor 的精确参数数量不得在第二阶段硬编码为 `505.6M` 或其他常数。
启动时从 checkpoint 统计参数名、shape、总量和 trainable 数，并与父 manifest
匹配。

### 2.1 父 checkpoint 最低资格

父 Actor 必须满足：

```text
v4.2 immutable source snapshot
+ P4→P8 development acceptance
+ 完成第一阶段 offline_full_finetune SFT
+ fresh-process strict reload
+ ActionTargetPopulationParityGate PASS
```

P9 只验证离线 Shadow，不是第二阶段纯离线训练的父资格条件。

`parent_actor_manifest.json` 至少绑定：

```text
parent_spec_sha256
parent_source_binding_sha256
parent_resolved_training_config_sha256
parent_checkpoint_payload_sha256
parent_topology_digest
parent_normalizer_manifest_sha256
parent_action_semantics_sha256
parent_processor_graph_sha256
parent_conversion_manifest_sha256
parent_split_manifest_sha256
parent_p4_to_p8_artifact_sha256s
```

---

## 3. 第二阶段离线 MDP 与动作语义

### 3.1 默认 transition profile

为避免重新引入 v4.2 已明确不要求的 controller clock map/ACK，第二阶段
默认使用数据帧级离线 MDP：

```yaml
profile_id: lerobot_frame_30hz_recorded_target_v1
dataset_fps: 30
transition_stride_frames: 1
critic_action_dim: 7
actor_slot_index: 0
behavior_action_source: lerobot_recorded_absolute_action7
apply_verified: false
claim_scope: recorded-command offline RFT
```

对 episode 内帧 (t)：

\[
o_t^F=
\left(I_t^{\mathrm{cam1}},I_t^{\mathrm{cam2}},\ell_t,s_t,w_t\right),
\]

\[
a_t^{\mathcal D}
=
\mathcal N_A\!\left(
\mathcal D_{s_t}(a_{t,\mathrm{abs}}^{\mathcal D})
\right)
\in\mathbb R^7,
\]

其中 \(\mathcal D_{s_t}\) 是 v4.2 的 `ActionDeltaProcessor.to_delta`，
\(\mathcal N_A\) 是父 checkpoint 冻结的 action normalizer。

一条 transition 定义为：

\[
\left(o_t^F,a_t^{\mathcal D},r_t,o_{t+1}^F,d_t\right).
\]

约束：

- `next_frame_index = frame_index + 1`；
- current/next 必须位于同一 episode 和同一 split；
- current action 必须是该帧 recorded target，不能使用整个未来 H-step label；
- true terminal 的 (d_t=1)，不得 bootstrap；
- 无合法 next observation 的异常 truncation 不进入 TD population；
- `episode_end` 不自动等价于 success，必须有明确 outcome source；
- 该 profile 不声称 target 已被 controller ACK 或实际施加。

如果后续希望把 Critic timebase 改为约 10 Hz，必须新建独立 profile 和
transition sidecar，例如 `stride_frames=3`；同时重新定义 reward aggregation
和 discount。不得在同一次训练中混用 30 Hz 与 10 Hz transition，也不得只
改 \(\gamma\) 而复用旧 sidecar。

### 3.2 Actor chunk 与 Critic action

Actor 保持生成完整动作序列：

\[
\hat{\mathbf A}_{\theta,t}^{H}
=
\Phi_\theta^{(N)}(\epsilon;o_t^F)
\in\mathbb R^{H\times7},
\qquad H=50,\;N=10.
\]

Critic 默认只评价第 0 个动作：

\[
\hat{\mathbf a}_{\theta,t}
=
S_0\left(\hat{\mathbf A}_{\theta,t}^{H}\right)
=
\hat{\mathbf A}_{\theta,t}^{H}[0]
\in\mathbb R^7.
\]

这是算法定义，不是 tensor slicing 的实现细节。原因是
\(o_{t+1}\) 只由当前 transition 内的 recorded action 直接关联；把完整
H=50 计划送入 Q，却只使用一帧 next observation 做 TD，会把未执行的未来
动作错误归因给当前状态转移。

后 49 个动作仍通过 Flow Matching 学习；由于 Actor 参数共享，第一动作上的
Q 梯度仍会更新 VLM、Force 模块和共享 Action Expert 向量场，但论文不得
宣称每个未来 action slot 都获得独立 Q supervision。

### 3.3 三类 mask 必须分离

```text
demo_action_valid_mask[B,H]     # 第一阶段 FM label 的 episode-tail padding
policy_suffix_valid_mask[B,H]   # RFT Flow sampling；默认全 true
action_feature_mask[B,H,32]     # 仅前7维 true
```

Critic action 固定为 `[B,7]`，不需要 variable-length Critic mask。任何
`[B,32]`、`[B,H,32]` 或带 false slot 的 Critic action 输入都应拒绝。

---

## 4. 冻结的数学目标

### 4.1 十步可微 Flow sampling

SmolVLA 的积分方向保持 v4.2/官方实现：

\[
\mathbf x^{(0)}=\epsilon,
\qquad
\Delta\tau=-\frac1N,
\qquad
\tau_n=1-\frac nN,
\]

\[
\mathbf x^{(n+1)}
=
\mathbf x^{(n)}
+
\Delta\tau\,
v_\theta(\mathbf x^{(n)},\tau_n;o_t^F),
\qquad n=0,\ldots,N-1,
\]

\[
\hat{\mathbf A}_{\theta,t}^{H}=\mathbf x^{(N)}.
\]

最后一次 velocity evaluation 位于 \(\tau=1/N\)，完成更新后的
\(\mathbf x^{(N)}\) 对应 \(\tau=0\)。不得反转积分方向。

`noise7` 只在 `[B,H,7]` 上采样；嵌入 32D 后 `[7:32]` 从创建起即为
exact zero。Development 初版固定 TD 与 Actor 每状态各采一个 noise sample：

```yaml
td_noise_samples_per_state: 1
actor_noise_samples_per_state: 1
```

不得在未修改公式和 reduction 的情况下静默增加采样数。

“梯度穿过全部十步”的理论递推为：

\[
J_{n+1}
=
\left(
I+\Delta\tau\frac{\partial v_n}{\partial \mathbf x^{(n)}}
\right)J_n
+
\Delta\tau\frac{\partial v_n}{\partial\theta}.
\]

验收不能只检查最终 loss 非零；必须用 directional derivative 或 cached/
uncached JVP/parameter-gradient parity 验证完整链路。

### 4.2 Twin-Q TD target

在 Critic update 中，next action 由当前 Actor 在 `eval()`、`no_grad()` 下
采样：

\[
\hat{\mathbf a}'_{t+1}
=
S_0\!\left[
\Phi_\theta^{(N)}(\epsilon';o_{t+1}^F)
\right].
\]

TD target 为：

\[
y_t
=
r_t
+
\gamma(1-d_t)
\min_{i\in\{1,2\}}
Q_{\bar\phi_i}
\left(o_{t+1}^F,\hat{\mathbf a}'_{t+1}\right).
\]

实现约束：

- \(Q_{\bar\phi_1},Q_{\bar\phi_2}\) 是 target critics；
- true terminal 直接使用 \(y_t=r_t\)，不得调用 next Actor 或 target Q；
- target 全部 stop-gradient；
- Q、target、TD reduction 和 log-sum-exp 使用 fp32；
- \(\gamma\) 是每 30 Hz transition 的 discount；
- `terminated` 与 `truncated` 分开保存，训练时才构造 bootstrap mask；
- TD next noise 与 Actor/CQL noise 使用独立、可恢复 RNG stream。

### 4.3 Empirical return 与 Cal-QL-style loss

对到达已标注 terminal 的完整 episode，经验 return 为：

\[
G_t^{\mathcal D}
=
\sum_{k=t}^{T-1}\gamma^{k-t}r_k.
\]

它是 empirical behavior-return calibration reference，不应写成一般条件下
可证明的 lower bound。没有完整 return 的 truncation row 可以进入 TD，但
不得进入 Cal-QL calibration batch。

候选集沿用 ConRFT 的三源结构：

\[
\mathcal C_t
=
\left\{
a_{t,m}^{\mathrm{rand}},
a_{t,m}^{\pi},
a_{t,m}^{\pi,\mathrm{next}}
\right\}_{m=1}^{M}.
\]

- `random`：先由 Stage-2 builder 从 train split 的 slot-0 behavior actions
  构建只读 `CriticActionPopulation`，再使用父 normalizer 转到 normalized
  7D proposal support 内采样；不得使用包含远期 horizon delta 的完整 H=50
  population 直接定义 slot-0 Critic proposal；
- `CriticActionPopulation` 只定义 proposal support，不拟合或覆盖父
  normalizer；夹爪只从父数据的离散 open/close support 采样；
- `current-policy`：从 \(o_t^F\) 采样完整 Flow chunk 后取 slot 0；
- `next-policy`：从 \(o_{t+1}^F\) 采样后取 slot 0，再与其他候选一样
  在 current \(Q(o_t^F,\cdot)\) 上评价；
- 因 Critic action 是 state-relative mixed delta7，next-policy candidate
  不执行 absolute→current-state rebase；
- 不在 32D padded action space 中采样。

对第 \(c\) 个 Critic，dataset Q 为：

\[
q_{c,t}^{\mathcal D}
=
Q_{\phi_c}(o_t^F,a_t^{\mathcal D}).
\]

对 OOD candidates 使用经验 return calibration：

\[
\widetilde q_{c,t}(a)
=
\max\left(Q_{\phi_c}(o_t^F,a),G_t^{\mathcal D}\right),
\qquad a\in\mathcal C_t.
\]

令 \(J=3M+1\)，则：

\[
\Delta_{\mathrm{Cal},c}(t)
=
T_{\mathrm{cql}}
\log\left[
\frac{
\exp(q_{c,t}^{\mathcal D}/T_{\mathrm{cql}})
+
\sum_{a\in\mathcal C_t}
\exp(\widetilde q_{c,t}(a)/T_{\mathrm{cql}})
}{3M+1}
\right]
-q_{c,t}^{\mathcal D}.
\]

每个 Critic 的损失为：

\[
\mathcal L_{Q_c}
=
\mathbb E_{t\sim\mathcal D_{\mathrm{TD}}}
\left[
\left(q_{c,t}^{\mathcal D}-y_t\right)^2
\right]
+
\alpha\,
\mathbb E_{t\sim\mathcal D_{\mathrm{Cal}}}
\left[
\operatorname{clip}
\left(
\Delta_{\mathrm{Cal},c}(t),c_{\min},c_{\max}
\right)
\right],
\]

\[
\mathcal L_Q^{\mathrm{off}}
=
\frac12
\left(
\mathcal L_{Q_1}+\mathcal L_{Q_2}
\right).
\]

`D_Cal` 必须由 `calibration_valid=true` 的独立 sampler 产生，并满足：

```text
calibration_valid
=> mc_return_valid
=> mc_return is finite
=> next observation exists for next-policy candidates
```

不能在混合 minibatch 中遇到零 valid rows 时返回零并仍声称估计
conditional mean。

实现名称固定为：

> ConRFT-compatible Cal-QL-style finite-sample estimator.

如果 random proposal 没有 density correction，不得声称严格复现
importance-sampled CQL。

### 4.4 Actor objective

完整 Actor 目标为：

\[
\mathcal L_{\mathrm{actor}}^{\mathrm{off}}
=
\beta\mathcal L_{\mathrm{FM}}
+0.01\mathcal L_{\mathrm{balance}}
+0.001\mathcal L_z
-\eta\,
\mathbb E
\left[
\frac{
Q_{\phi_1}(o_t^F,\hat{\mathbf a}_{\theta,t})
+Q_{\phi_2}(o_t^F,\hat{\mathbf a}_{\theta,t})
}{2}
\right].
\]

冻结以下语义：

- Actor guidance 使用 online Q1/Q2 的算术均值；
- TD target 使用 target Q1/Q2 的最小值；
- Critic 参数在 Actor update 中冻结，但 action input 不 detach；
- target critics 不参与 Actor objective；
- Q 梯度穿过完整十步 Flow；
- Top-1 expert index 不可微，router 继续由 balance/z loss 更新；
- Flow Matching 继续监督完整 H=50 demonstration chunk；
- Q guidance 只评价 slot 0 的 7D normalized action。

task2 的 gripper 是二值 absolute width，而 v4.2 public decode 不可微。初版
采用最保守且理论清晰的规则：Critic 输入仍包含第 7 维，但 Actor 的
Q-guidance 不向 gripper 维反传：

\[
\hat a_{\theta,t}^{Q}
=
\left[
\hat a_{\theta,t,1:6},
\operatorname{sg}(\hat a_{\theta,t,7})
\right].
\]

gripper 继续只由 Flow Matching 学习。不得在未写入方法与测试的情况下
静默加入 binary STE。

### 4.5 不在 Q-gradient 路径中执行 public safety transform

Actor Q 路径禁止调用：

```text
predict_action_chunk()
NumPy unnormalize
binary gripper decode
ActionDeltaProcessor.from_delta()
RuleSpec validation
controller target conversion
```

这些操作要么不可微，要么具有 v4.2 的 fail-closed 语义。训练时不做
hard clamp 或伪“execution-equivalent projection”。

validation 可在 detached action 上调用原 v4.2 public API，记录：

```text
whole_chunk_valid_rate
failure_code_histogram
workspace/orientation/continuity rejection rate
gripper candidate-range rejection rate
```

如果 invalid rate 上升，应降低 \(\eta\)、增加 FM 权重或停止训练；不能
把非法动作投影成合法动作后伪装成模型真实输出。

### 4.6 Target Critic 更新

初始化：

\[
\bar\phi_i\leftarrow\phi_i.
\]

每次 Critic optimizer step 后执行一次：

\[
\bar\phi_i
\leftarrow
(1-\tau)\bar\phi_i+\tau\phi_i,
\qquad i\in\{1,2\}.
\]

Polyak update 必须在 fp32、`torch.no_grad()` 下执行。配置中的 `tau` 遵循
上述约定，不得与反向约定混用。

---

## 5. Force-aware Twin-Q 架构

新增两个完全独立的 Critic；第一版不提供共享 encoder 选项，以避免 target
copy、optimizer 和 Polyak 所有权歧义。

建议 development topology：

```text
camera1 -> critic-specific ResNet-10 -> 256D
camera2 -> critic-specific ResNet-10 -> 256D
language token ids/mask -> embedding + masked pooling -> 128D
normalized state7 -> MLP -> 128D
normalized wrench6 -> MLP -> 128D
normalized action_target7 -> MLP -> 128D
concat -> MLP(1024, 512, 256, 1) -> scalar Q
```

硬约束：

- Q1/Q2 的所有 parameter objects 独立；
- target Q1/Q2 分别是 online Q 的 exact deep copy；
- Critic 只接收 `[B,7]` action；
- state7、wrench6、action7 使用父 custom normalizer exactly once；
- current/next 相机顺序与 v4.2 一致；
- 使用 GroupNorm/LayerNorm，不使用 BatchNorm；
- 初版 `dropout=0`；
- Q scalar、loss、log-sum-exp 和 Polyak 处于 fp32；
- target critics 永久 `eval()`、`requires_grad=False`；
- Critic observation encoder 与 Actor 不共享参数；
- 固定其他输入时改变有效 wrench，Q 应具有非零局部敏感性；
- 修改 action/state padding `[7:32]` 不得影响 Q，因为 Critic 根本不接收
  32D tensor。

具体通道数和 hidden size允许在 GPU preflight 后调整，但必须记录在
`force_aware_twin_q.development.yaml`；论文结果使用的 topology 在长训前
冻结。

---

## 6. 低侵入 sidecar 实现

### 6.1 第一阶段文件默认不修改

以下文件应视为 v4.2 core，第二阶段默认只读：

```text
src/forcesmolvla/configuration_forcesmolvla.py
src/forcesmolvla/modeling_forcesmolvla.py
src/forcesmolvla/context.py
src/forcesmolvla/prefix.py
src/forcesmolvla/force_token.py
src/forcesmolvla/action_delta.py
src/forcesmolvla/normalizer.py
src/forcesmolvla/checkpoint.py
tools/train_task2_full_gpu.py
tools/serve_policy.py
```

如果真实源码证明某一新功能无法 sidecar 化，Codex 必须先报告具体调用点、
为何无法复用和最小 patch；不得直接重写 v4.2 core。

### 6.2 新增模块

```text
src/forcesmolvla/rft/__init__.py
src/forcesmolvla/rft/config.py
src/forcesmolvla/rft/transition.py
src/forcesmolvla/rft/batching.py
src/forcesmolvla/rft/critic.py
src/forcesmolvla/rft/flow_sampling.py
src/forcesmolvla/rft/candidate_sampler.py
src/forcesmolvla/rft/losses.py
src/forcesmolvla/rft/training.py
src/forcesmolvla/rft/checkpoint.py

tools/build_task2_offline_rft_transitions.py
tools/preflight_s2_parent_bridge.py
tools/preflight_s2_transition_population.py
tools/preflight_s2_differentiable_flow_gpu.py
tools/preflight_s2_twinq_losses_gpu.py
tools/preflight_s2_single_cycle_gpu.py
tools/preflight_s2_resume.py
tools/train_task2_offline_rft_gpu.py
tools/export_stage2_actor.py
```

### 6.3 Differentiable Flow wrapper

`flow_sampling.py` 新增：

```python
def sample_normalized_action_chunk_with_grad(
    policy: ForceSmolVLAPolicy,
    batch: dict[str, torch.Tensor],
    noise7: torch.Tensor,
    *,
    call_id: str,
    purpose: Literal["actor_guidance", "td_next", "cql_current", "cql_next"],
) -> torch.Tensor:
    """Return float32 normalized action_target7 with shape [B,50,7]."""
```

实现必须：

1. 使用现有 `policy.prepare_images()`、`prepare_state()` 和
   `_prepare_wrench()`；
2. 构造全 true `policy_suffix_valid_mask[B,50]`；
3. 构造仅前 7 维 true 的 `action_feature_mask[B,50,32]`；
4. 将 `noise7[B,50,7]` exact zero-pad 为 32D；
5. 为该调用创建唯一 ephemeral `PreparedForceContextBinding`；
6. 直接调用现有 `policy.model.sample_actions_masked()`；
7. 返回 `actions32[..., :7].float()`；
8. 不调用 `_predict_normalized_delta_chunk()` 或 public inference；
9. 不创建/消费部署 `ChunkContext.chunk_id`；
10. 不缓存跨 call 的 PrefixContext/PreparedForceContext。

`sample_actions_masked()` 现有实现没有 `no_grad/inference_mode` decorator，
其 prefix、Force Context 和 Euler state 不应 detach，因此无需复制一套
denoiser。

每个 sampling call 内：

```text
prefix prefill: 1次
Force Context: 1次
Force K/V projection: 1次
Euler velocity evaluation: 10次
suffix cache append→crop: 每步1次
```

不同 `purpose`、不同 candidate microbatch 或不同 optimizer update 不得复用
prepared context。Ephemeral cache 不进入 checkpoint。

binding 字段直接复用 v4.2 现有类型，建议唯一构造为：

```python
PreparedForceContextBinding(
    chunk_id=tuple(
        f"rft:{purpose}:{call_id}:{row}" for row in range(batch_size)
    ),
    sample_id=tuple(batch["sample_identity"]),
    context_generation=policy._context_generation,
    model_generation=policy.model.parameter_generation(),
    device=state.device,
    dtype=torch.float32,
)
```

同一 batch 内 ID 必须唯一。任一 Actor `optimizer.step()` 后，参数
`_version` 变化必须使旧 binding stale；Critic-only step 不改变 Actor
generation。

### 6.4 Mode 与 observation-view 约束

- FM 路径：Actor `train()`；
- Actor Q-guidance：Actor `eval()` 但 autograd 开启；
- Critic update 的 next/candidate Actor：`eval()+no_grad()`；
- Actor update 的 Q1/Q2：`eval()`、参数冻结、action Jacobian 保留；
- 每个 context manager 退出时恢复所有子模块原 mode。

同一数学项中 Actor 与 Critic 必须看到同一个 observation view：

```text
Q(o_aug, pi(o_aug))       # 允许
Q(o_aug2, pi(o_aug1))     # 禁止
```

development 初版建议 Actor guidance 和 TD 全部使用 deterministic eval
preprocessing，不启用随机 crop；图像增强仅在 Critic supervised update 中
后续配置化。

---

## 7. Transition sidecar 与 RewardSpec

### 7.1 原始数据只读

不得修改原 LeRobot v3 数据。新增：

```text
datasets/task2_offline_rft_v3/
  transitions.parquet
  transition_manifest.json
  reward_spec.yaml
  outcome_labels.json
```

sidecar 只保存索引、reward、terminal 语义和 hash；图像/state/wrench/action
仍从原数据集读取。

### 7.2 最小 transition schema

```text
transition_id
episode_index
frame_index
next_frame_index              # terminal可为null
current_sample_identity
next_sample_identity          # terminal可为null
recorded_absolute_action7
normalized_action_target7
reward
terminated
truncated
bootstrap_mask
mc_return                     # 可为null
mc_return_valid
calibration_valid
outcome_label
outcome_source
dataset_tree_sha256
conversion_manifest_sha256
split_manifest_sha256
normalizer_manifest_sha256
action_semantics_sha256
reward_spec_sha256
transition_profile_sha256
critic_action_population_sha256
```

全量 gate 必须验证：

- `frame_index+1 == next_frame_index`（非 terminal）；
- current/next 同 episode、同 split；
- `next_identity[t] == current_identity[t+1]`；
- action 与原数据 row exact；
- `normalized_action_target7` 等于现有 H=50 training sample 的 slot 0；
- normalizer exactly once；
- gripper 在 transform 前后保持 absolute width；
- terminal 只出现在 episode 最后一条 outgoing transition；
- padding 不构成 transition；
- train/val/test transition 不交叉；
- transition count 与 ordered identity/tensor SHA 可重建。
- `calibration_valid` 必须蕴含有限 MC return 和合法 next observation。
- slot-0 `CriticActionPopulation` 只能由 train split 构建，val/test 不参与。

### 7.3 RewardSpec

Twin-Q 训练必须有真实定义的 reward、terminal 和 next observation。Codex
不得仅根据“episode 结束”猜测 success。

最小配置：

```yaml
profile: task2_offline_reward_v1
timebase_hz: 30
success_reward: null
failure_reward: null
step_reward: null
force_penalty:
  enabled: false
  coefficient: null
discount_gamma_per_transition: null
success_label_source: null
failure_label_source: null
truncation_rule: null
```

数值可由实验负责人批准后写入 development config，不需要引入 formal
detached signature。若 50 条轨迹全部为成功示范且只有 terminal success
reward，则论文结论必须限定为示范分布内的回报/进度估计，不能声称 Critic
学会失败恢复或危险接触辨别。

---

## 8. 训练更新与梯度所有权

### 8.1 初始化

```text
strict load v4.2 parent Actor weights and runtime artifacts
discard Stage-1 optimizer/scheduler/sampler state
keep Actor config.training_stage = offline_full_finetune
initialize Q1 and Q2 independently
deep-copy Q1 -> target Q1
deep-copy Q2 -> target Q2
create new Actor/Critic optimizers and schedulers
```

切换前后 Actor `state_dict` 必须 exact；不得重新初始化已经训练的
W_out、router、experts、adapter 或 VLM。

### 8.2 Critic update

```python
critic_optimizer.zero_grad(set_to_none=True)

with actor_eval_no_grad(policy):
    next_action = sample_slot0_for_nonterminal_rows(..., purpose="td_next")
    candidates = sample_calql_candidates(..., independent_rng=True)

with torch.no_grad():
    y = reward.clone()
    y[nonterminal] += gamma * torch.minimum(
        target_q1(next_obs, next_action),
        target_q2(next_obs, next_action),
    )

loss_q1 = td_loss_q1 + alpha * calql_loss_q1
loss_q2 = td_loss_q2 + alpha * calql_loss_q2
critic_loss = 0.5 * (loss_q1 + loss_q2)
critic_loss.backward()
clip_and_check_critic_gradients()
critic_optimizer.step()
polyak_update_(target_q1, q1, tau)
polyak_update_(target_q2, q2, tau)
```

必须保证：

- Critic step 前后 Actor 参数和 buffer exact unchanged；
- target critics 从不产生 `.grad`；
- 每个 Critic step 后恰好一次 Polyak；
- terminal rows不进入 next Actor/target Q；
- TD sampler 与 Cal sampler 分开，Cal batch 全部 `mc_return_valid=true`。

### 8.3 Actor update

```python
actor_optimizer.zero_grad(set_to_none=True)

with actor_train_mode(policy):
    flow_terms, feature_mask, router_state = (
        policy.forward_single_pass_training_terms(...)
    )
    fm_aux_loss = (
        beta * flow_loss
        + 0.01 * balance_loss
        + 0.001 * z_loss
    )
    fm_aux_loss.backward()

with actor_eval_with_grad(policy), critics_frozen_keep_input_grad(q1, q2):
    action_chunk = sample_normalized_action_chunk_with_grad(
        ..., purpose="actor_guidance"
    )
    action_q = action_chunk[:, 0, :]
    action_q = stop_gripper_q_gradient(action_q)
    q_mean = 0.5 * (q1(obs, action_q) + q2(obs, action_q))
    q_loss = -eta * q_mean.mean()
    q_loss.backward()

clip_and_check_actor_gradients()
actor_optimizer.step()  # 恰好一次
```

这是两个可加 objective 的分阶段 backward：

```text
1次 FM full forward/backward
+ 1次 N=10 differentiable Flow forward/backward
+ 1次 Actor optimizer.step
```

它不是第一阶段错误使用的 exact two-pass router oracle。不得为了减少时间
detach Euler state，也不得执行两次 Actor optimizer step。

### 8.4 初始 development recipe

以下值只作为 GPU preflight 起点，不直接构成论文最终超参数：

```yaml
critic_warmup_updates: 1000
critic_updates_per_cycle: 2
actor_updates_per_cycle: 1
actor_lr: 1.0e-5
critic_lr: 3.0e-4
beta_flow: 1.0
eta_q: 0.01
eta_warmup_actor_updates: 500
alpha_calql: 0.1
tau_polyak: 0.005
cql_temperature: 1.0
cql_candidates_per_source: 2
td_noise_samples_per_state: 1
actor_noise_samples_per_state: 1
critic_batch_size: 16
actor_microbatch_size: 1
actor_gradient_accumulation: 4
router_aux_scope: microbatch_local
precision: bf16_outer_with_v4_2_fp32_islands
```

在真实 4090D single-cycle preflight 后可降低 batch 或 candidate 数，但不得
改变 H=50、N=10、双相机、全参数 Actor 或 Flow 架构。最终正式实验配置
必须记录改变理由和吞吐/显存。

当 Actor 采用 microbatch accumulation 时：

- `L_flow` 必须按整个 accumulation window 的有效 feature 总数归一化；
- Q term 必须按整个 window 的 transition 数归一化；
- balance/z 初版定义为 microbatch-local auxiliary objective，再对 microbatch
  等权平均；
- 该 router auxiliary objective 不与第一阶段物理 B4 global objective
  声称数值等价；
- 整个 accumulation window 结束后仍只允许一次 Actor optimizer step。

### 8.5 训练计数

不得用一个模糊 `step` 表示所有更新。至少记录：

```text
cycle_index
transition_samples_seen
critic_optimizer_updates
actor_optimizer_updates
polyak_updates
flow_chunks_sampled
euler_velocity_evaluations
candidate_actions_sampled
```

正式预算不能直接复用第一阶段 `10,000 updates`。应依据 transition 数量、
Critic calibration、policy drift 和 4090D 吞吐单独确定。

---

## 9. Checkpoint、resume 与 Actor export

### 9.1 RFT training checkpoint

```text
rft_checkpoint/
  actor/                         # 完整v4.2-compatible Actor checkpoint
  critics/q1.safetensors
  critics/q2.safetensors
  critics/q1_target.safetensors
  critics/q2_target.safetensors
  actor_optimizer.pt
  critic_optimizer.pt
  actor_scheduler.pt
  critic_scheduler.pt
  counters.json
  sampler_state.json
  rng_state.pt
  transition_manifest.json
  reward_spec.yaml
  resolved_rft_config.yaml
  parent_actor_manifest.json
  source_binding.json
  checkpoint_manifest.json
```

如果 bf16 不使用 GradScaler，也必须写入：

```yaml
amp_scaler_enabled: false
```

### 9.2 保存边界

初版只允许在完整 training cycle 边界保存：

```text
所有计划的 Critic updates完成
+ Actor update完成
+ 两个 optimizer均已step/zero_grad
+ 不存在pending graph或partial accumulation
```

因此不需要保存 pending gradients。若未来要在 cycle 中途保存，则必须新增
`phase/critic_substep/actor_substep/accumulation_substep` 和 pending gradient
状态，不能只保存一个 cycle number。

使用：

```text
rolling recovery/latest（原子替换）
→ final checkpoint
→ fresh-process strict validation
→ 可选删除 recovery
```

### 9.3 Exact resume

中断前后必须比较下一次：

```text
transition IDs
TD/Actor/CQL RNG states
Flow noise和normalized action
Q1/Q2/target outputs
TD target与loss
optimizer/scheduler state
Polyak counter与参数
Actor fixed-noise output
```

初版 `num_workers=0`，避免预取队列破坏精确恢复。多 worker 只有在实现可
checkpoint 的主进程 index dispatcher 后才能启用。

### 9.4 Actor export

Stage-2 Actor 导出直接复制/保存为现有 v4.2 Actor artifact 结构；Critic 不
进入推理路径。`serve_policy.py` 无需理解新的 RFT checkpoint 类型。

新进程 local-only reload 后必须验证：

- fixed-noise normalized chunk；
- public absolute chunk；
- prefix/cache parity；
- Force K/V 投影次数；
- action/normalizer/processor artifacts；
- `robot_execution_authorized=false`。

---

## 10. Gate 顺序与验收标准

### S2-G0：Parent zero-update bridge

- 按 v4.2 strict loader 加载父 Actor；
- 父 topology、normalizer、processor 和 P4–P8 hashes匹配；
- Stage-2 wrapper 前后 Actor state_dict exact；
- fixed input/noise normalized action exact；
- public absolute output和错误码一致；
- 运行一次第一阶段 `forward_single_pass_training_terms()` regression；
- 不恢复第一阶段 optimizer/scheduler；
- 不修改父 checkpoint 和 P4–P9 artifacts。

### S2-G1：Transition、reward 与 normalizer

- 全量 transition population 可重建；
- 无 cross-episode/cross-split；
- current/next/action slot0 exact；
- reward/terminal/truncation truth table通过；
- terminal 不要求伪造 next observation；
- normalizer exactly once；
- inherited LeRobot normalizers仍 Identity/disconnected；
- raw、mixed-delta、normalized action字段不可混用；
- empirical MC return数值可独立重算。

### S2-G2：Twin-Q topology

- Q1/Q2 参数对象完全独立；
- target初始化与online max-abs error为0；
- target无梯度；
- Polyak公式exact；
- action shape不是 `[B,7]` 必须拒绝；
- force/state/image/action sensitivity报告；
- dropout=0、fp32 Q output/loss。

### S2-G3：Differentiable Flow

- eval fixed-noise wrapper 与现有 normalized sampler output parity；
- cached与`velocity_full()` uncached reference output parity；
- cached/uncached parameter-gradient或JVP parity；
- N=10 exact；
- prefix cache每步append→crop恢复；
- Force K/V每call一次；
- state/action/noise `[7:32]` 三类独立扰动不影响7D输出/梯度；
- Q directional derivative穿过全部Euler steps；
- full-tensor cache audit仅在gate启用，长训只记录O(1) counters。

### S2-G4：损失与梯度所有权

- Actor使用online Q均值；
- TD使用target Q最小值；
- terminal branch不调用next Actor/target Q；
- Cal-QL的3M+1、OOD-only clamp、LME、clip和Twin-Q mean正确；
- Cal batch全部mc-return valid；
- Critic step不改变Actor；
- Actor step不改变Q1/Q2/targets；
- Critic参数冻结时仍保留 `dQ/da`；
- Q梯度到达Vision、VLM、ForceMLP、Fusion、active experts、Adapter、Action
  Expert和Action I/O的共享参数；
- gripper只接受FM梯度。

### S2-G5：RTX 4090D single-cycle

真实执行：

```text
2 critic updates
+ 1 full-Actor update
```

报告：

```text
peak allocated/reserved memory
Critic update latency
FM forward/backward latency
10-step Flow forward/backward latency
Euler evaluation count
candidate sampling latency
Actor/Critic batch与accumulation
NaN/Inf count
```

禁止 CPU fallback、LoRA、冻结 VLM、减少 N/H、删除相机或使用 exact
two-pass oracle 作为长训循环。

### S2-G6：Cycle-boundary exact resume

- rolling recovery可原子恢复；
- 下一 transition IDs、RNG、noise、action、Q、target、loss和参数一致；
- Polyak次数不多不少；
- 无 pending accumulation；
- fresh-process local-only reload。

### S2-G7：短程 offline RFT smoke

- Critic TD loss在小型可学习fixture下降；
- Q1/Q2不被错误地绑定为同一网络；
- target Q finite；
- empirical-return calibration error下降或保持稳定；
- Flow validation loss无异常突增；
- fixed-noise policy drift受控；
- detached public inference whole-chunk valid rate不恶化到批准阈值之外。

只有 G0–G7 全部通过后，才能确定 development 长训预算。

---

## 11. 文件级实施顺序

1. 冻结 v4.2 source snapshot、父 checkpoint 和实际 resolved config；
2. 实现 `rft/config.py` 与 G0 parent bridge；
3. 实现 transition/reward sidecar builder 和全量 G1；
4. 实现两个独立 Critic、targets、Polyak 和 G2；
5. 实现 `flow_sampling.py` sidecar wrapper，不修改 Actor core；
6. 完成 cached/uncached output与gradient/JVP G3；
7. 实现纯函数 TD、Cal-QL和Actor losses，并与固定 ConRFT estimator tensor
   fixture对照；
8. 实现独立TD/Actor/CQL RNG streams和candidate sampler；
9. 实现 Critic update与Actor双目标单step update；
10. 完成 G4 gradient ownership；
11. 实现4090D single-cycle并确定可行batch/M；
12. 实现cycle-boundary checkpoint/exact resume；
13. 导出v4.2-compatible Actor并做fresh-process parity；
14. 运行短程G7；
15. 根据transition数量、Q calibration、policy drift和吞吐确定长训预算；
16. 第二阶段完成后，再单独制定第三阶段 frozen-VLM online HIL 规范。

---

## 12. Codex 执行边界

Codex 可以立即完成：

- sidecar源码；
- synthetic transition fixture；
- Twin-Q模块；
- differentiable Flow wrapper；
- loss数值测试；
- GPU single-cycle preflight；
- checkpoint/resume框架。

在以下条件未闭合前，不得启动真实第二阶段长训：

1. task2 RewardSpec；
2. success/failure/terminated/truncated标注来源；
3. G0父checkpoint bridge；
4. G1真实transition population；
5. G3十步cached gradient parity；
6. G5 4090D内存/耗时；
7. development长训超参数与预算。

这不是因为 v4.2 Actor 不合格，而是 Twin-Q 所需的 RL transition/reward 是
第一阶段 SFT 数据契约之外的新信息。

---

## 13. 论文贡献对应与表述边界

实现完成后，本阶段支持以下论文主张：

> ForceRFT 在不替换 SmolVLA 原生 Flow Action Expert 的前提下，引入力觉
> 感知 Twin-Q，并使价值梯度穿过完整十步 Euler flow sampling，从而直接
> 调整力觉条件化动作向量场。离线阶段保持完整 Actor 可训练，Flow Matching
> 约束策略先验，target Twin-Q 的 clipped backup 与经验校准的 conservative
> objective共同构成离线强化微调信号。

必须同时限定：

- Critic 是 observation-conditioned，而非已证明具备完整 Markov state；
- 当前离线 action source 是 recorded command/target，不是 verified actual
  apply；
- Q guidance直接作用于chunk第0个动作，共享参数使其影响整个Actor，但不
  声称未来49个slot均有独立Q标签；
- 只有成功示范时不能声称失败恢复；
- 单帧wrench仍主要是static force conditioning；
- 离线Q指标不能替代固定协议的真机成功率评估；
- frozen-VLM在线HIL属于第三阶段，不得在第二阶段完成前写成已验证结果。

---

## 14. 固定参考

- ForceSmolVLA 第一阶段：`ForceSmolVLA_Implementation_Spec_v4_2`
- ConRFT repository：<https://github.com/cccedric/conrft>
- ConRFT estimator reference commit：
  `a779fde7fa5db5a469960a8490c100f35b41b49e`
- ConRFT agent：
  <https://github.com/cccedric/conrft/blob/a779fde7fa5db5a469960a8490c100f35b41b49e/serl_launcher/serl_launcher/agents/continuous/conrft_single_octo_cp.py>
- SmolVLA/LeRobot pinned commit：
  `30da8e687a6dfc617fcd94afc367ac7071c376ce`
