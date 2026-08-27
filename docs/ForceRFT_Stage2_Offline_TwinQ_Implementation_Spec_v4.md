# ForceRFT 第二阶段：基于 v4.2 Actor 的离线 Force-aware Twin-Q 强化微调实施规范

> 文档类型：可直接交给 Codex 的实现规格  
> 版本：v4（10 Hz macro-action + Reward Detector 对齐版）  
> 状态：development-only；不授权在线 HIL 或真机执行  
> 第一阶段唯一基线：`ForceSmolVLA_Implementation_Spec_v4_2`  
> 论文方法名：**ForceRFT**  
> 现有代码包名：`forcesmolvla`（第二阶段不重命名）  
> 初始化：通过 v4.2 验收并完成离线 SFT 的 ForceSmolVLA r5 Actor checkpoint  
> 第二阶段算法标识：`offline_force_rft`（只属于 RFT trainer，不写入 Actor 的 `training_stage`）

---

## 0. 本版相对第二阶段 v3 的关键修订

本规范不是重新实现第一阶段，而是在 v4.2 Actor 外部增加离线
Actor–Critic 训练层。以下修改同时对齐已经完成的 Stage-2 G0/G3
development preflight、冻结的 10 Hz macro-action 语义，以及 ConRFT 风格的
视觉奖励分类器流程。

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

4. **Critic 的动作语义固定为 10 Hz nominal macro-action。**  
   数据记录频率为 30 Hz，策略决策频率约为 10 Hz，因此每个 Critic transition
   评价在一个决策周期内纳入的 `K=3` 个 recorded-command slot，而不是只评价
   slot 0，也不是评价完整 `H=50` 计划。Reward Detector 的 30 Hz 因果确认帧
   向后对齐到首个合法 10 Hz 决策边界，使当前 profile 的 Critic 输入始终为
   完整 `[B,K,7]`，不把未来 terminal 距离编码进 action mask。

5. **冻结 mixed continuous–discrete action contract。**  
   Twin-Q 前向观察完整 `K×7` 动作；Actor 的 Q-gradient 仅通过前 6 维
   Cartesian TCP 动作。第 7 维 gripper 按第一阶段公共契约解码为离散
   `0/0.085 m` endpoint，再经冻结 normalizer 映射回 Critic 输入空间并
   stop-gradient。完整 `H×7` 动作仍由 Flow Matching 监督。

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

9. **奖励判定与奖励数值分离。**  
   Reward Classifier 只输出逐帧成功置信度；冻结的 Reward Detector 将概率转为
   `causal_success_confirmation_frame`；10 Hz aligner 再得到 RL terminal
   boundary；RewardSpec 最后将普通/成功 transition 映射为
   reward、discount 与 terminal。三者不得混为一个隐式规则。

10. **奖励分类器以独立工具链接入。**  
    允许复用固定 ConRFT commit 的 ResNet-10 多相机 classifier、BCE 训练逻辑
    和 Flax checkpoint，但不得把其 Octo/SERL 环境直接并入 ForceRFT。
    LeRobot v3 adapter、独立验证、阈值冻结、逐 episode 人工复核和 sidecar
    是本项目新增的必要闭包。

11. **理论名称收紧。**  
   本阶段称为 `ConRFT-compatible Cal-QL-style estimator`。单轨迹 Monte
   Carlo return 是 empirical calibration reference，不宣称构成一般条件下
   可证明的行为价值下界。

12. **以父 checkpoint 的实际 resolved config 为准。**  
    v4.2 正文的有效第一阶段预算是 `40,000 samples / 10,000 updates`；其
    后部残留的 `80k-sample` 表述不得用于选择父 checkpoint。启动时必须从
    父 checkpoint 的 `resolved_training_config` 读取并绑定真实计数，禁止
    通过实验目录名猜测。

13. **Stage-2 源码闭包独立绑定。**  
    新增或尚未提交的 Stage-2 源码、配置、测试和工具不能只依赖 Git HEAD 或
    changed-file allowlist。`stage2_source_manifest.json` 必须逐文件记录路径、
    SHA256、大小、角色和运行时导入状态，并反向写入所有 G0/G3/R0/G1 artifact。

14. **Gate 顺序按真实进度修订。**  
    已完成的 G0 parent bridge 和独立 G3 differentiable-flow 仅作为
    development preflight 接受；先关闭其 source-closure、mixed-action 与
    fp32/bf16 gradient-parity 遗留项，再执行 R0 Reward Detector 和 G1
    transition。G2 Twin-Q 在拓扑、维度和参数预算批准前不得创建。

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
| 动作 | `action_target7=[delta_xyz, wrapped_delta_rpy, absolute_gripper_width_m]` | Critic 读取 normalized `K×7` macro-action；夹爪仍为 absolute width，不变成 delta |
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

## 3. 第二阶段离线 semi-MDP 与动作语义

### 3.1 冻结的 30 Hz 数据、10 Hz 决策 profile

第二阶段不重新引入 controller clock map/ACK，但必须使 Critic 的时间尺度与
计划中的 10 Hz 闭环一致。默认 profile 固定为：

```yaml
profile_id: lerobot_30hz_macro3_decision_aligned_v3
f_data_hz: 30
f_policy_hz: 10
anchor_stride_frames: 3
action_horizon: 50
critic_action_slots: 3
critic_action_features: 7
behavior_action_source: lerobot_recorded_absolute_action7
apply_verified: false
claim_scope: recorded-command macro-transition offline RFT
```

令 \(C_e\) 表示第 \(e\) 个 episode 由批准的 Reward Detector 产生并通过
人工 acceptance audit 的 `causal_success_confirmation_frame`。它是 30 Hz
逐帧 detector 的因果确认帧，不是 streak onset，也不是 episode 文件末帧。

为避免把“距离成功还剩 1/2/3 帧”作为 mask 泄漏给 Critic，RL terminal
必须对齐到以 episode start 为相位的 10 Hz 决策网格：

\[
t_{e,n}=t_{e,0}+nK,
\qquad K=3,
\]

\[
T_e^{\mathrm{RL}}
=
\min\left\{
t_{e,n}:t_{e,n}\ge C_e
\right\}.
\]

只有当该边界存在于有效 episode 内、成功状态在 \(C_e\) 到
\(T_e^{\mathrm{RL}}\) 之间未被人工判定为失效，且完整的 incoming K-slot
recorded-command macro-action 可重建时，该 episode 才能进入当前 G1 scope。
因此 \(0\le T_e^{\mathrm{RL}}-C_e\le K-1\)。对 episode 内决策帧
\(t=t_{e,n}<T_e^{\mathrm{RL}}\)：

\[
o_t^F=
\left(I_t^{\mathrm{cam1}},I_t^{\mathrm{cam2}},\ell_t,s_t,w_t\right).
\]

\[
t^+=t+K,
\qquad K=3.
\]

anchor 从 `episode_start` 按 3 帧递增，直到且仅到达
\(T_e^{\mathrm{RL}}\)。每条 transition 都包含完整 K 个 recorded-command
slot；当前 profile 不允许 partial-duration action。一条 transition 定义为：

\[
\xi_t=
\left(o_t^F,\mathbf A_t^{\mathcal D},r_t,
o_{t^+}^F,d_t,\delta_t\right),
\]

其中 \(d_t\) 是 `terminated`，\(\delta_t\) 是已经合并 bootstrap 语义后的
scalar discount。

### 3.2 Demonstration macro-action

Stage-2 必须复用第一阶段同一个 anchor-relative action 处理链。对
\(j=0,\ldots,K-1\)：

\[
\mathbf A_{t,j}^{\mathcal D}
=
\mathcal N_A\!\left[
\mathcal D_{s_t}
\left(a_{t+j,\mathrm{abs}}^{\mathcal D}\right)
\right]
\in\mathbb R^7.
\]

其中 \(\mathcal D_{s_t}\) 是 v4.2 的 `ActionDeltaProcessor.to_delta`，
\(\mathcal N_A\) 是父 checkpoint 冻结的 action normalizer。所有 slot 都以
同一个 raw anchor state \(s_t\) 为参考；不得改成
`action[t+j]-state[t+j]`。

固定长度 Critic tensor 定义为：

\[
\mathbf A_t^{\mathcal D}\in\mathbb R^{K\times7}.
\]

当前 transition 只包含记录动作 `action[t:t_plus]`，绝不包含
`action[t_plus]`。

约束：

- current/next 必须位于同一 episode 和同一 split；
- 所有 transition 都必须满足 `next_frame_index=frame_index+3`；
- terminal transition 的 `next_frame_index=T_e_RL`，且具有真实、决策对齐的
  terminal observation；
- 每个成功 episode 恰好一个 terminal transition；
- terminal observation 不生成 outgoing self-loop；
- `episode_end`、`saved=true` 或 `last_valid_frame` 均不能自动定义成功；
- 若 detector confirmation 后不存在合法的下一决策边界，该 episode 使 R0
  失败，不得退回 `last_valid_frame` 或构造 partial action；
- 该 profile 只表示 nominal 10 Hz 下聚合的 recorded command，不声称 target
  已被 controller ACK 或实际施加。

### 3.3 Actor chunk、Critic view 与 mixed action contract

Actor 保持生成完整动作序列：

\[
\hat{\mathbf A}_{\theta,t}^{H}
=
\Phi_\theta^{(N)}(\epsilon;o_t^F)
\in\mathbb R^{H\times7},
\qquad H=50,\;N=10.
\]

Critic 只读取计划中在一个 nominal 10 Hz policy decision 内纳入的前
\(K=3\) 个 recorded-command slot：

\[
S_K\!\left(\hat{\mathbf A}_{\theta,t}^{H}\right)
=
\hat{\mathbf A}_{\theta,t}^{H}[0:K]
\in\mathbb R^{K\times7}.
\]

对 Actor-Q 路径，前 6 维保持 continuous normalized Flow action。令
\(\mathcal N_{A,g}^{-1}\) 与 \(\mathcal N_{A,g}\) 分别表示冻结 gripper
normalizer 的逆变换与正变换，\(\mathcal B_g\) 表示第一阶段二值解码
`{0,0.085 m}`，则：

\[
\widetilde g_{\theta,t,j}
=
\operatorname{sg}\!\left(
\mathcal N_{A,g}\!\left[
\mathcal B_g\!\left(
\mathcal N_{A,g}^{-1}(\hat A_{\theta,t,j,7})
\right)
\right]
\right),
\]

\[
\hat{\mathbf U}_{\theta,t,j}^{Q}
=
\left[
\hat{\mathbf A}_{\theta,t,j,1:6},
\widetilde g_{\theta,t,j}
\right],
\qquad j=0,1,2.
\]

因此 Twin-Q 前向仍能区分 open/closed gripper，而 mixed-action projection
只允许 Actor-Q gradient 通过 6-DoF Cartesian motion：

\[
\frac{\partial \hat U_{\theta,t,j,d}^{Q}}
{\partial \hat A_{\theta,t,j,d}}
=
\begin{cases}
1,&d\in\{1,\ldots,6\},\\
0,&d=7.
\end{cases}
\]

gripper 继续由完整 \(H\times7\) Flow Matching objective 学习。论文统一称为：

> value-guided 6-DoF Cartesian refinement with imitation-regularized
> discrete gripper control.

对应配置固定为：

```text
critic_action_input_shape = [K,7]
critic_duration_mode = fixed_k
critic_receives_terminal_derived_mask = false
actor_q_guided_action_dims = [0,1,2,3,4,5]
gripper_objective = flow_matching_only
gripper_q_gradient = false
gripper_endpoint_width_m = [0.0,0.085]
```

下文的 \(\Pi^Q\) 表示逐 slot 应用上述 TCP identity 与
gripper decode→renormalize→stop-gradient 的 mixed-action projection；它不接收
terminal frame、remaining horizon 或 duration mask。

### 3.4 四类 mask 必须分离

```text
demo_action_valid_mask[B,H]       # 第一阶段FM label的episode-tail padding
policy_suffix_valid_mask[B,H]     # RFT Flow sampling；默认全true
action_feature_mask[B,H,32]       # 仅前7维true
actor_q_gradient_mask[B,K,7]      # 当前profile为[1,1,1,1,1,1,0]
```

Critic action 固定为 `[B,K,7]`。当前 profile 的 K 个 slot 全部有效，不向
Critic 提供由 terminal 派生的 duration/length/mask feature。任何 `[B,7]`、
`[B,32]`、`[B,H,32]` 或 variable-length Critic 输入都应拒绝。若未来确需
partial-duration option，必须把 duration 明确定义为策略可选择的 action，并
另行批准；不得复用本 profile。

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

target critic 不自动意味着存在 target Actor。本版明确冻结 bootstrap policy
\(\pi_{\mathrm{boot}}\) 为**当前 online Actor 的 `eval()+no_grad()` 视图**。
每个 training cycle 开始时建立只读参数版本；该 cycle 内所有 Critic substep
共用这一版本，Actor step 完成后，下一个 cycle 自然读取更新后的 online
Actor。不创建、不 Polyak 更新、也不 checkpoint 额外 EMA Actor。

不得使用数据集未来 action 代替 bootstrap-policy action。对非 terminal
macro-transition：

\[
\hat{\mathbf U}_{t^+}^{\mathrm{boot}}
=
\Pi^{Q}\!\left(
S_K\!\left[
\Phi_{\theta_{\mathrm{boot}}}^{(N)}
(\epsilon';o_{t^+}^F)
\right]
\right).
\]

TD target 为：

\[
y_t
=
r_t
+
\delta_t
\min_{i\in\{1,2\}}
Q_{\bar\phi_i}
\left(o_{t^+}^F,\hat{\mathbf U}_{t^+}^{\mathrm{boot}}\right).
\]

实现约束：

- \(Q_{\bar\phi_1},Q_{\bar\phi_2}\) 是 target critics；
- \(\delta_t=\gamma_{\mathrm{dec}}(1-d_t)\)，且当前
  \(\gamma_{\mathrm{dec}}=0.99\) 按 10 Hz policy decision 定义；
- true terminal 的 \(\delta_t=0\)，直接使用 \(y_t=r_t\)，不得调用
  bootstrap Actor 或 target Q；
- target 全部 stop-gradient；
- Q、target、TD reduction 和 log-sum-exp 使用 fp32；
- `terminated`、`truncated`、`bootstrap_mask` 和 `discount` 分开保存；当前
  task2 遇到任何 truncated episode 必须拒绝；
- TD next noise 与 Actor/CQL noise 使用独立、可恢复 RNG stream。
- \(\mathcal D_{\mathrm{TD}}\)、\(\mathcal D_{\mathrm{Cal}}\)、Actor FM、
  Actor-Q 与所有 candidate proposal 只能从 train split 采样；val/test 只用于
  detached evaluation，禁止 backward、optimizer、normalizer fitting 或
  proposal population 构建。

### 4.3 Empirical return 与 Cal-QL-style loss

对到达已标注 terminal 的完整 episode，按 macro-decision index \(n\) 反向
\[
G_{e,n}^{\mathcal D}
=
r_{e,n}
+
\delta_{e,n}G_{e,n+1}^{\mathcal D}.
\]

等价展开为：

\[
G_{e,n}^{\mathcal D}
=
\sum_{j=n}^{N_e-1}
\left(
\prod_{u=n}^{j-1}\delta_{e,u}
\right)r_{e,j}.
\]

下文将当前 macro-transition 对应的 \(G_{e,n}^{\mathcal D}\) 简写为
\(G_t^{\mathcal D}\)。

它是 empirical behavior-return calibration reference，不应写成一般条件下
可证明的 lower bound。当前 task2 的 RewardSpec 拒绝 truncation，因此没有
完整 return 的 episode 不进入 TD 或 Cal-QL population。

候选集沿用 ConRFT 的三源结构：

\[
\mathcal C_t
=
\left\{
\mathbf B_{t,m}^{\mathrm{rand}},
\mathbf B_{t,m}^{\pi,t},
\mathbf B_{t,m}^{\pi,t^+}
\right\}_{m=1}^{M}.
\]

- `random`：由 Stage-2 builder 从 train split 的真实完整 K-slot macro-actions
  构建只读 `CriticMacroActionPopulation`，保留 slot 间相关性和离散 gripper
  support；
- `CriticMacroActionPopulation` 已处于父 normalizer 的 normalized 语义，只
  定义 proposal support，不重新 normalization，也不拟合或覆盖父 normalizer；
- `current-policy`：从 \(o_t^F\) 采样完整 Flow chunk，取前 \(K\) 个 slot，
  经 \(\Pi^{Q}\) 构造 macro-action；
- `next-policy`：从 \(o_{t^+}^F\) 采样后取前 \(K\) 个 slot，并在 current
  \(Q(o_t^F,\cdot)\) 上评价；
- 因 Critic action 是 state-relative mixed delta7，next-policy candidate
  不执行 absolute→current-state rebase；
- 不在 32D padded action space 中采样。

对第 \(c\) 个 Critic，dataset Q 为：

\[
q_{c,t}^{\mathcal D}
=
Q_{\phi_c}\!\left(o_t^F,\mathbf A_t^{\mathcal D}\right).
\]

对 sampled candidates 使用经验 return calibration：

\[
\widetilde q_{c,t}(\mathbf B)
=
\max\left(
Q_{\phi_c}(o_t^F,\mathbf B),
\operatorname{sg}(G_t^{\mathcal D})
\right),
\qquad \mathbf B\in\mathcal C_t.
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
\sum_{\mathbf B\in\mathcal C_t}
\exp(\widetilde q_{c,t}(\mathbf B)/T_{\mathrm{cql}})
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

terminal row 具有真实 terminal next observation 和有限 MC return，因此可以
参与 candidate calibration；它只是不进入 TD bootstrap。不得因
`terminated=true` 自动排除唯一的正奖励 row。若 terminal row 使用
`next-policy` candidate，该 action 仅是 conservative calibration proposal，
不表示 terminal observation 存在 outgoing environment transition。

不能在混合 minibatch 中遇到零 valid rows 时返回零并仍声称估计
conditional mean。

实现名称固定为：

> ConRFT-compatible Cal-QL-style finite-sample estimator.

`3M+1`、三源候选和 log-mean-exp 是本文冻结的 finite-sample 实现变体；它们
不是 ConRFT 论文公式的逐式复现。如果 random proposal 没有 density
correction，不得声称严格复现 importance-sampled CQL。

### 4.4 Actor objective

完整 Actor 目标为：

\[
\mathcal L_{\mathrm{actor}}^{\mathrm{off}}
=
\beta\mathcal L_{\mathrm{FM}}
+0.01\mathcal L_{\mathrm{balance}}
+0.001\mathcal L_z
-\eta\,
\mathbb E_{t\sim\mathcal D,\,\epsilon}
\left[
\frac{
Q_{\phi_1}\!\left(o_t^F,\hat{\mathbf U}_{\theta,t}^{Q}\right)
+Q_{\phi_2}\!\left(o_t^F,\hat{\mathbf U}_{\theta,t}^{Q}\right)
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
- Q guidance 评价前 \(K=3\) 个完整、决策对齐的 macro-action slot；
- Actor-Q gradient 仅作用于有效 slot 的前 6 维 TCP action；Critic 前向仍
  观察离散 gripper endpoint。

进入 Twin-Q 的动作由第 3.3 节定义的 mixed-action projection 构造：

\[
\hat{\mathbf U}_{\theta,t}^{Q}
=
\Pi^{Q}\!\left(
S_K\!\left[\hat{\mathbf A}_{\theta,t}^{H}\right]
\right)
\in\mathbb R^{K\times7}.
\]

不得引入 binary STE；gripper 继续只由 Flow Matching 学习。

### 4.5 不在 Q-gradient 路径中执行 public safety transform

Actor Q 路径禁止调用：

```text
predict_action_chunk()
NumPy unnormalize
ActionDeltaProcessor.from_delta()
RuleSpec validation
controller target conversion
```

这些完整公共操作要么不可微，要么具有 v4.2 的 fail-closed 语义。唯一允许
的离散操作是 Stage-2 sidecar 中与第一阶段逐元素等价的纯 gripper
decode→renormalize→stop-gradient；其输入输出必须通过 public decode parity
test。训练时不对 TCP 动作做 hard clamp 或伪“execution-equivalent
projection”。

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
normalized macro-action[K,7] -> shared SlotMLP -> [K,128D]
+ action-position embedding -> flatten
concat -> MLP(1024, 512, 256, 1) -> scalar Q
```

硬约束：

- Q1/Q2 的所有 parameter objects 独立；
- target Q1/Q2 分别是 online Q 的 exact deep copy；
- Critic 接收固定 `[B,K,7]` action，其他 shape 必须拒绝；
- state7、wrench6、macro-action7 使用父 custom normalizer exactly once；
- action slot encoder 权重共享，但使用固定 action-position embedding 保留顺序；
- 当前 profile 不存在 invalid K-slot，也不把 terminal-derived duration/length
  作为 Critic 特征；
- current/next 相机顺序与 v4.2 一致；
- 使用 GroupNorm/LayerNorm，不使用 BatchNorm；
- 初版 `dropout=0`；
- Q scalar、loss、log-sum-exp 和 Polyak 处于 fp32；
- target critics 永久 `eval()`、`requires_grad=False`；
- Critic observation encoder 与 Actor 不共享参数；
- 固定其他输入时改变有效 wrench，Q 应具有非零局部敏感性；
- 修改 action/state padding `[7:32]` 不得影响 Q，因为 Critic 根本不接收
  32D tensor；
- synthetic probe 中，所有 `K×6` TCP 输入对 Q 的梯度必须非零；
- 切换有效 gripper open/closed endpoint 必须改变 Q 前向，Actor-Q 路径对
  gripper 的梯度必须精确为零。

上述结构是 G2 的候选 topology，而不是已批准实现。具体通道数、hidden size
和参数预算必须在进入 G2 前批准并写入
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
src/forcesmolvla/rft/reward_contract.py
src/forcesmolvla/rft/offline_transitions.py
src/forcesmolvla/rft/batching.py
src/forcesmolvla/rft/critic.py
src/forcesmolvla/rft/flow_sampling.py
src/forcesmolvla/rft/candidate_sampler.py
src/forcesmolvla/rft/losses.py
src/forcesmolvla/rft/training.py
src/forcesmolvla/rft/checkpoint.py

tools/reward_classifier/export_lerobot_v3_classifier_dataset.py
tools/reward_classifier/infer_reward_classifier_lerobot_v3.py
tools/reward_classifier/review_task2_success_frames.py
tools/build_task2_offline_rl_transitions.py
tools/preflight_s2_parent_bridge.py
tools/preflight_s2_transition_population.py
tools/preflight_s2_differentiable_flow_gpu.py
tools/preflight_s2_twinq_losses_gpu.py
tools/preflight_s2_single_cycle_gpu.py
tools/preflight_s2_resume.py
tools/train_task2_offline_rft_gpu.py
tools/export_stage2_actor.py
```

### 6.3 Stage-2 源码闭包

必须生成：

```text
artifacts/development/stage2/stage2_source_manifest.json
```

逐文件记录：

```text
relative_path
sha256
file_size
artifact_role
runtime_imported
```

manifest 必须覆盖所有实际执行或 import 的 Stage-2 源码、配置、测试、工具，
以及固定 ConRFT classifier commit 下被复用的文件。G0、G3、R0、G1 和后续
artifact 都必须写入该 manifest 的 SHA256。未纳入 manifest 的 Stage-2 文件
不得进入 acceptance 或训练。

### 6.4 Differentiable Flow wrapper

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

### 6.5 Mode 与 observation-view 约束

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

## 7. Reward Detector、Transition sidecar 与 RewardSpec

### 7.1 原始数据只读

`task2_lerobotv3` 是唯一且不可变的 observation/action 事实源。不得修改 raw
采集代码，不得修改或重新运行 v2.1/v3 converter，也不得向 v3 数据根目录
写入任何文件。运行 R0/G1 前后必须重算并证明 v3 data-tree SHA 不变。

所有派生工件写到根目录之外：

```text
labels/
  task2_episode_outcomes.v1.json
  task2_reward_predictions.v1.parquet
  task2_terminal_review.v1.json
datasets/task2_forcerft_rl_v1/
  transition_index.parquet
  rl_manifest.json
configs/
  stage2_action_contract.development.json
  stage2_reward_classifier_input.development.yaml
  stage2_reward_detector.development.yaml
  stage2_reward_spec.development.yaml
```

sidecar 只保存 v3 row reference、分类器输出、人工审计、reward/terminal
语义和 hash；图像/state/wrench/action 不复制，仍由冻结 v3 loader 读取。

### 7.2 Episode outcome 与成功时刻分离

当前 task2 的 47 条有效 episode 由操作者追溯确认为成功，但该事实只定义
episode outcome，不自动定义成功发生在哪一帧。必须逐 episode 枚举：

```text
raw_episode_id
output_episode_index
split
task_outcome=success
outcome_source=retrospective_operator_attestation
reviewer_id
review_timestamp
```

禁止从 `saved=true`、episode 文件结束、最后一个有效 tuple 或文件名自动
推断 success。`last_valid_frame` 只允许用于 synthetic/test-only fixture，
不得进入正式 G1 artifact。

### 7.3 ConRFT Reward Classifier 复用边界

固定 ConRFT commit 下可复用：

- ResNet-10 多相机 binary classifier；
- ImageNet 初始化与其 RGB normalization；
- BCE objective、正负各半 batch、random-crop augmentation；
- Flax checkpoint save/restore 逻辑。

若选择直接复用，classifier topology 保持：

```text
each camera
→ frozen ImageNet ResNet-10 pre-pooling features
→ learned spatial embeddings
→ multi-camera EncodingWrapper
→ Dense(256) → Dropout(0.1) → LayerNorm → ReLU → Dense(1)
→ success logit
```

`conrft_reward_commit` 必须从本地 clone 的实际 HEAD 读取并冻结，不能写
`main` 或使用无法解析的分支名。

不得直接复用：

- ConRFT 的 task-specific classifier checkpoint；
- `CONFIG_MAPPING`、Octo、SERL ReplayBuffer 或 Franka Gym 环境；
- 示例任务的 camera key/crop、阈值、夹爪/TCP 规则或 reward scale；
- 将 Flax checkpoint 直接交给 `torch.load()`。

推荐在独立 `conrft_reward` JAX 环境中训练分类器，并通过 LeRobot v3 adapter
读取 D435/D405。必须冻结并绑定：

```text
conrft_commit_sha
classifier_source_sha256s
classifier_checkpoint_sha256
pretrained_resnet10_sha256
camera_keys_and_order
RGB/BGR
resize/crop/pad
dtype_and_normalization
frame_stack
train/val/test_episode_split
```

frame stack 必须严格因果。若 stack 长度为 \(L\)，frame \(t\) 的 classifier
view 只能包含：

\[
\operatorname{view}(e,t)=
\left[e,\max(t_{e,0},t-L+1),\ldots,t\right].
\]

episode start 必须 reset；缺失历史只能按冻结规则复制最早可用帧或使用固定
padding，不得跨 episode，也不得读取 `frame>t`。evaluation crop 必须确定性。
R0 必须加入“任意扰动所有未来帧不改变当前 logit”的测试，并逐帧绑定输入 row
identity。若固定 ConRFT commit 的 `EncodingWrapper` 在 `stack>1` 时实际只消费
最后一帧，也必须在 adapter manifest 中记录并用 fixture 证明，不能仅凭配置名
推断时序行为。

分类器数据必须包含 task-specific positives、ordinary negatives 和接近成功的
hard negatives；建议负样本总量至少为正样本的 2–3 倍。训练与验证 episode
必须不相交。至少报告 AUROC、PR-AUC、FPR、FNR、precision、recall，以及用于
冻结阈值的 held-out confusion matrix。训练 accuracy 不能作为唯一验收。

positive 只表示该 observation 已经满足任务完成条件。若任务完成状态在后续
帧持续保持，这些帧可以作为经规则/人工确认的 classifier positive samples；
但它们不会产生多个 RL success reward 或多个 terminal transition。不得把
“接近插入完成”样本误作正例。

不得把 ConRFT 示例脚本中的 `num_epochs=150` 直接解释为150次完整数据遍历；
其原循环每次只抽取一个平衡 batch。新的配置必须分别记录 optimizer updates、
examples seen 和等效 dataset passes。

现有 47 条示范可以辅助构造候选帧，但不得同时作为 classifier 训练集和未经
人工复核的最终自动标签集，否则形成循环验证。

### 7.4 Reward Detector 与人工复核

分类器输出逐帧 logit/probability，不直接输出 terminal。冻结的 detector
将逐帧概率转为候选成功时刻：

```yaml
detector_id: task2_success_detector_v1
inference_source: immutable_task2_lerobotv3
inference_rate_hz: 30
probability_threshold: null        # R0 held-out validation后批准
consecutive_positive_frames: null  # R0测量后批准
max_detection_delay_frames: null   # 相对人工physical completion，R0后批准
causal_confirmation: true
episode_boundary_reset: true
post_success_transition_policy: stop_at_first_10hz_boundary
manual_episode_review_required: true
last_valid_frame_fallback: disabled
```

令 \(p_{e,t}\) 为逐帧成功概率、\(\tau_R\) 为冻结阈值、\(M_R\) 为连续帧数，
则因果 detector confirmation 定义为：

\[
C_e^{\mathrm{det}}
=
\min\left\{
t:\;p_{e,t-M_R+1},\ldots,p_{e,t}\ge\tau_R
\right\}.
\]

该时刻是第 \(M_R\) 个连续 positive 到达后的确认帧，不得利用未来帧把
terminal 回填到 streak onset
\(S_e=C_e^{\mathrm{det}}-M_R+1\)。detector streak 必须在每个 episode start
清零且不得跨 episode。随后必须由人工逐 episode 审计：

```text
causal_success_confirmation_frame
streak_onset_frame                 # 仅审计，不作为terminal
audited_physical_completion_frame
detection_delay_frames
review_decision=accept/retrain_detector/reject_episode
reviewer_id
review_notes
```

令 \(P_e\) 为人工标注的 physical completion frame，冻结
\(D_{\max}\) 为 held-out collection runs 上批准的最大 detection delay。只有
满足下列条件才能 `accept`：

\[
0\le C_e^{\mathrm{det}}-P_e\le D_{\max},
\]

并且不存在 pre-completion false positive、集合非空、
\(C_e^{\mathrm{det}}>t_{e,0}\)，且第 3.1 节的 10 Hz terminal boundary 可构造。
无 crossing、负 delay、超出最大 delay、需要人工改写 detector frame 或无法
构造 incoming macro-transition 时，R0 必须整体失败并重新训练/校准 detector；
不得回退到 `last_valid_frame`。

正式 RL `terminal_frame` 是第 3.1 节由已接受
\(C_e^{\mathrm{det}}\) 得到的 \(T_e^{\mathrm{RL}}\)，不是物理完成帧或 streak
onset。人工审计只作 accept/retrain/reject，不静默改写 frame。其来源固定为：

```text
approved_reward_detector
+ per-episode_acceptance_audit
+ first_10hz_decision_boundary_at_or_after_confirmation
```

当前 scope 要求 47/47 episode 全部 `accept`；任何 `no_candidate`、
`retrain_detector` 或 `reject_episode` 都阻断 G1。若未来缩小数据范围，必须
版本化新的 dataset scope 并重新批准，不能仍声称覆盖 47 条。

除帧级 AUROC/PR-AUC 外，R0 必须报告 episode-level premature-terminal rate、
missed-success rate、no-trigger count、每 episode 首触发混淆，以及相对
\(P_e\) 的 confirmation-delay 分布。\(\tau_R\)、\(M_R\) 与 \(D_{\max}\)
必须只在独立 held-out collection runs 上冻结。

分类器评价的是 observation frame。成功 reward 必须赋给到达
\(T_e^{\mathrm{RL}}\) 的 incoming K-slot macro-transition，不能赋给 terminal
frame 上并不存在的 `action[T_e_RL]`。

### 7.5 最小 transition schema

```text
transition_id
raw_episode_id
output_episode_index
split
anchor_frame_index
next_frame_index
terminal_frame_index
detector_confirmation_frame_index
streak_onset_frame_index
audited_physical_completion_frame_index
detection_delay_frames
terminal_alignment_delay_frames
current_sample_identity
next_sample_identity
macro_action_slots            # 固定K=3
recorded_action_row_indices   # [K]，全部有效
normalized_macro_action_sha256
reward
terminated
truncated
bootstrap_mask
discount
mc_return
mc_return_valid
calibration_valid
outcome_label
outcome_source
classifier_probability_at_confirmation
classifier_probability_at_rl_terminal
classifier_checkpoint_sha256
reward_classifier_dataset_manifest_sha256
reward_classifier_manifest_sha256
reward_predictions_sidecar_sha256
reward_detector_spec_sha256
terminal_review_sha256
dataset_tree_sha256
conversion_manifest_sha256
split_manifest_sha256
normalizer_manifest_sha256
action_semantics_sha256
reward_spec_sha256
transition_profile_sha256
critic_macro_action_population_sha256
stage2_source_manifest_sha256
```

全量 gate 必须验证：

- 所有 row：`next_frame_index=anchor_frame_index+3`；
- terminal：`next_frame_index=terminal_frame_index=T_e_RL`；
- `0 <= terminal_frame_index-detector_confirmation_frame_index <= 2`；
- current/next 同 episode、同 split；
- 相邻 macro-transition 满足上一条 `next_identity` 等于下一条
  `current_identity`；
- 三个 action row 与原 v3 数据 exact，当前 transition 不包含
  `action[next_frame_index]`；
- 重建的前 K 个 delta/normalized slot 与同 anchor 的第一阶段 H=50 loader
  逐元素一致；
- normalizer exactly once；
- gripper 在 transform 前后保持 absolute width；
- 每个 episode 恰好一条到达真实 terminal observation 的 terminal
  transition；不生成 terminal self-loop；
- 当前 profile 不存在 terminal-tail padding 或 duration mask；
- train/val/test transition 不交叉；
- 47 条有效 episode 全覆盖、无重复、无遗漏；terminal reward 总数恰好为47；
- transition count 必须等于
  \(\sum_e(T_e^{\mathrm{RL}}-t_{e,0})/K\)，且每项均为整数，ordered
  identity/tensor SHA 可重建；
- `calibration_valid` 必须蕴含有限 MC return 和合法 next observation。
- `CriticMacroActionPopulation` 只能由 train split 构建，val/test 不参与。
- `D_TD`、`D_Cal`、Actor FM/Actor-Q 和所有 proposal sampler 只能读取 train；
  val/test 只允许 detached evaluation。

### 7.6 Development RewardSpec、terminal 与 MC return

Twin-Q 训练必须有真实定义的 reward、terminal 和 next observation。Codex
不得仅根据“episode 结束”猜测 success。

Reward Classifier/Detector 只决定成功时刻，不决定 reward scale。当前
development-only RewardSpec 冻结为：

```yaml
profile: task2_offline_sparse_success_macro3_v3
timebase_hz: 10
success_terminal_reward: 1.0
nonterminal_reward: 0.0
gamma_per_policy_decision: 0.99
force_penalty:
  enabled: false
failure_semantics: reserved_unapproved
truncated_semantics: reject_if_encountered_in_current_task2
success_outcome_source: retrospective_operator_attestation
terminal_frame_source: approved_detector_plus_audit_plus_10hz_boundary_alignment
formal_status: unapproved
```

对 episode \(e\) 的 macro-transition \(n\)：

\[
d_{e,n}=\mathbb 1[t_{e,n+1}=T_e^{\mathrm{RL}}],
\qquad
r_{e,n}=d_{e,n},
\]

\[
b_{e,n}=1-d_{e,n},
\qquad
\delta_{e,n}=0.99\,b_{e,n}.
\]

因此只有到达首个不早于因果成功确认的 10 Hz 决策边界之
macro-transition 获得 reward 1；成功 observation 不产生新的 outgoing action
或额外 reward。MC return 按第 4.3 节从 terminal 反向计算。

formal RewardSpec 继续为 unapproved。当前47条轨迹全部为成功示范且只有
terminal success reward，因此论文结论必须限定为示范分布内的回报/进度估计，
不能声称 Critic 已学会失败恢复、危险接触辨别或一般性的在线最优价值函数。

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

with frozen_cycle_bootstrap_actor_eval_no_grad(policy):
    next_chunk = sample_action_chunk_for_nonterminal_rows(..., purpose="td_next")
    next_macro_action = build_critic_macro_action(next_chunk[:, :K])
    candidates = sample_calql_candidates(..., independent_rng=True)

with torch.no_grad():
    y = reward.clone()
    y[nonterminal] += discount[nonterminal] * torch.minimum(
        target_q1(next_obs, next_macro_action),
        target_q2(next_obs, next_macro_action),
    )

loss_q1 = td_loss_q1 + alpha * calql_loss_q1
loss_q2 = td_loss_q2 + alpha * calql_loss_q2
critic_loss = 0.5 * (loss_q1 + loss_q2)
critic_loss.backward()
clip_and_check_critic_gradients()
critic_optimizer.step()
critic_optimizer.zero_grad(set_to_none=True)
polyak_update_(target_q1, q1, tau)
polyak_update_(target_q2, q2, tau)
```

必须保证：

- Critic step 前后 Actor 参数和 buffer exact unchanged；
- target critics 从不产生 `.grad`；
- 每个 Critic step 后恰好一次 Polyak；
- terminal rows不进入 next Actor/target Q；
- bootstrap action 来自已冻结所有权的 `pi_bootstrap`，不能替换为数据集未来
  action；
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
    action_q = build_mixed_critic_macro_action_with_grad(action_chunk[:, :K, :])
    q_mean = 0.5 * (
        q1(obs, action_q)
        + q2(obs, action_q)
    )
    q_loss = -eta * q_mean.mean()
    q_loss.backward()

clip_and_check_actor_gradients()
actor_optimizer.step()  # 恰好一次
actor_optimizer.zero_grad(set_to_none=True)
```

这是两个可加 objective 的分阶段 backward：

```text
1次 FM full forward/backward
+ 1次 N=10 differentiable Flow forward/backward
+ 1次 Actor optimizer.step
```

它不是第一阶段错误使用的 exact two-pass router oracle。不得为了减少时间
detach Euler state，也不得执行两次 Actor optimizer step。

`build_mixed_critic_macro_action_with_grad()` 必须满足：全部 `K×6` TCP
gradient 保留、gripper gradient 精确为零，同时
open/closed gripper 的前向 Q 值可区分。

### 8.4 初始 development recipe

以下值只作为 GPU preflight 起点，不直接构成论文最终超参数：

```yaml
critic_warmup_updates: 1000
critic_updates_per_cycle: 2
actor_updates_per_cycle: 1
critic_action_slots: 3
gamma_per_policy_decision: 0.99
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

warmup 语义固定为：

1. 前 `critic_warmup_updates=1000` 个 Critic optimizer update 为
   critic-only；不执行 Actor FM/Q backward 或 Actor optimizer step；
2. warmup 完成后开始 Actor update，Flow Matching 从第一个 Actor update 起
   始终使用完整权重 \(\beta\)；
3. 令 \(k\) 为 warmup 后从 1 开始计数的 Actor update，则

\[
\eta(k)=\eta_q\min\left(1,\frac{k}{500}\right).
\]

不得把 critic warmup 误记为 training cycle 数，也不得在 warmup 中静默更新
Actor buffer。

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
  task2_episode_outcomes.v1.json
  task2_terminal_review.v1.json
  reward_classifier_dataset_manifest.json
  reward_classifier_manifest.json
  reward_detector_spec.yaml
  reward_spec.yaml
  action_contract.json
  resolved_rft_config.yaml
  parent_actor_manifest.json
  stage2_source_manifest.json
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
+ Actor phase完成（critic-only warmup时显式记录actor_update_performed=false）
+ 所有本cycle计划执行的optimizer已step，两个optimizer均无pending grad
+ 不存在pending graph或partial accumulation
```

在 critic-only warmup cycle 中，Actor optimizer 没有 step，但仍必须保持
zero-grad、参数/buffer exact unchanged；“两个 optimizer 均已 step”只适用于
包含 Actor update 的 cycle。

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

允许的实际顺序为：

```text
G0/G3 development preflight
→ G0/G3遗留闭包
→ R0 Reward Detector
→ G1 Transition
→ G2 Twin-Q
→ G4–G7
```

G3 的独立可微 Flow preflight 可以早于 G1/G2，但不能据此创建 Critic 或启动
训练。

### S2-G0：Parent zero-update bridge

- 按 v4.2 strict loader 加载父 Actor；
- 父 topology、normalizer、processor 和 P4–P8 hashes匹配；
- Stage-2 wrapper 前后 Actor state_dict exact；
- fixed input/noise normalized action exact；
- public absolute output和错误码一致；
- 运行一次第一阶段 `forward_single_pass_training_terms()` regression；
- 不恢复第一阶段 optimizer/scheduler；
- 不修改父 checkpoint 和 P4–P9 artifacts；
- `stage2_source_manifest.json` 覆盖所有实际执行/import的Stage-2文件，并将
  manifest SHA写入G0/G3 artifacts。

### S2-R0：Reward Classifier 与 Reward Detector

- ConRFT classifier 源码固定到精确 commit，并使用独立 JAX 环境；
- D435/D405 key、顺序、RGB、resize/crop、normalization 和 causal frame
  stack冻结；episode边界reset且未来帧扰动不改变当前logit；
- classifier train/val/test episode-disjoint；
- positives、ordinary negatives 和 hard negatives 数据来源可审计；
- held-out AUROC、PR-AUC、FPR/FNR、precision/recall、confusion matrix，以及
  episode-level premature/miss/no-trigger/confirmation-delay指标报告；
- checkpoint、预训练ResNet、预处理和源码SHA完整绑定；
- 对不可变LeRobot v3以30 Hz逐帧推理；
- threshold、连续帧规则与最大 detection delay 冻结后才生成 confirmation；
- 47条episode逐条审计且必须47/47 `accept`；任一no-trigger/retrain/reject使
  R0整体失败；
- detector confirmation向后对齐到首个10 Hz决策边界，alignment delay必须
  位于`[0,2]`帧且完整incoming K-slot action存在；
- v3 data-tree SHA运行前后不变；
- formal classifier threshold与RewardSpec仍保持unapproved。

### S2-G1：Transition、reward 与 normalizer

- 全量 transition population 可重建；
- 无 cross-episode/cross-split；
- 47条episode全覆盖且每条恰好一个terminal macro-transition；
- current/next与`action[t:next_t]`无off-by-one；transition不包含
  `action[next_t]`；
- Critic action为完整`[K,7]`，且不存在terminal-derived duration/mask输入；
- terminal transition到达真实terminal observation，不生成self-loop；
- reward/terminal/truncation truth table通过；
- terminal reward数量恰好为47；
- normalizer exactly once；
- inherited LeRobot normalizers仍 Identity/disconnected；
- raw、mixed-delta、normalized action字段不可混用；
- G1三个absolute/delta/normalized action slot与同anchor第一阶段loader逐元素
  一致，且对应Stage-1 `demo_action_valid_mask`均为true；
- empirical MC return按10 Hz macro index可独立重算；
- 输出不复制图像，只保存v3 row reference和RL字段。

### S2-G2：Twin-Q topology

- Q1/Q2 参数对象完全独立；
- target初始化与online max-abs error为0；
- target无梯度；
- Polyak公式exact；
- action shape不是 `[B,K,7]` 必须拒绝；
- 不接收terminal-derived mask/length feature；
- 所有`K×6` TCP维在synthetic probe中具有非零action gradient；
- open/closed gripper改变Q前向，但Actor-Q gripper gradient精确为零；
- force/state/image/action sensitivity报告；
- dropout=0、fp32 Q output/loss。

### S2-G3：Differentiable Flow

- eval fixed-noise wrapper 与现有 normalized sampler output parity；
- cached与`velocity_full()` uncached reference output parity；
- cached/uncached parameter-gradient或JVP parity分别在fp32和bf16下运行；
- 至少3次固定输入重复，报告per-module relative L2、gradient cosine、最大
  absolute error和repeat-to-repeat variance；
- 当前约`relative L2≈0.0142`只作为既有development measurement；不得据此
  自行批准或放宽formal threshold；
- N=10 exact；
- prefix cache每步append→crop恢复；
- Force K/V每call一次；
- state/action/noise `[7:32]` 三类独立扰动不影响7D输出/梯度；
- Q directional derivative穿过全部Euler steps；
- synthetic Q probe显式依赖所有`K×6` TCP动作；gripper Q-gradient为零，
  FM对gripper仍有非零梯度；
- full-tensor cache audit仅在gate启用，长训只记录O(1) counters。

### S2-G4：损失与梯度所有权

- Actor使用online Q均值；
- TD使用target Q最小值；
- `pi_bootstrap`固定为cycle内只读的current online Actor eval视图，且不得用
  数据集未来action替代；
- terminal branch因discount=0而不调用bootstrap Actor/target Q；
- Cal-QL的3M+1、candidate-only clamp（dataset q不参与clamp）、LME、clip和
  Twin-Q mean正确；
- Cal batch全部mc-return valid；
- Critic step不改变Actor；
- Actor step不改变Q1/Q2/targets；
- Critic参数冻结时仍保留 `dQ/da`；
- Q梯度到达Vision、VLM、ForceMLP、Fusion、active experts、Adapter、Action
  Expert和Action I/O的共享参数；
- Q梯度只覆盖K个slot的6D TCP；gripper只接受FM梯度。

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

只有 R0 与 G0–G7 全部通过后，才能确定 development 长训预算。

---

## 11. 文件级实施顺序

1. 冻结 v4.2 source snapshot、父 checkpoint 和实际 resolved config；
2. 接受现有G0/G3 development preflight，但补齐Stage-2 source manifest；
3. 关闭mixed-action contract与fp32/bf16 cached/uncached gradient parity；
4. 固定ConRFT classifier commit与LeRobot v3 adapter；
5. 采集/整理奖励分类器正例、负例与hard negatives，训练并完成独立验证；
6. 冻结Reward Detector阈值、连续帧和最大延迟规则，对v3做因果逐帧推理，
   人工复核47条episode并对齐10 Hz RL terminal boundary；
7. 实现transition/reward sidecar builder和全量G1；
8. 批准G2 Twin-Q topology、维度和参数预算；
9. 实现两个独立Critic、targets、Polyak和G2；
10. 实现纯函数 TD、Cal-QL和Actor losses，并与固定 ConRFT estimator tensor
   fixture对照；
11. 实现独立TD/Actor/CQL RNG streams和macro-candidate sampler；
12. 实现Critic update与Actor双目标单step update；
13. 完成G4 gradient ownership；
14. 实现4090D single-cycle并确定可行batch/M；
15. 实现cycle-boundary checkpoint/exact resume；
16. 导出v4.2-compatible Actor并做fresh-process parity；
17. 运行短程G7；
18. 根据transition数量、Q calibration、policy drift和吞吐确定长训预算；
19. 第二阶段完成后，再单独制定第三阶段 frozen-VLM online HIL 规范。

---

## 12. Codex 执行边界

当前 handoff 中，Codex 只可以立即完成：

- G0/G3 source closure；
- mixed-action synthetic probes；
- fp32/bf16 differentiable-Flow gradient parity；
- Reward Classifier/LeRobot v3 adapter的接口和synthetic测试；
- G1 transition builder框架和synthetic fixture。

在奖励分类器 checkpoint、detector threshold 和逐episode review未冻结前，
不得生成正式 G1 artifact。G2 topology未批准前不得创建Twin-Q、target、
optimizer、Cal-QL loss或训练循环。

在以下条件未闭合前，不得启动真实第二阶段长训：

1. Stage-2 source closure；
2. task2 Reward Classifier与Reward Detector；
3. task2 RewardSpec；
4. success/failure/terminated/truncated标注来源；
5. G0父checkpoint bridge；
6. G1真实macro-transition population；
7. G2 topology批准；
8. G3十步cached gradient parity；
9. G5 4090D内存/耗时；
10. development长训超参数与预算。

这不是因为 v4.2 Actor 不合格，而是 Twin-Q 所需的 RL transition/reward 是
第一阶段 SFT 数据契约之外的新信息。

---

## 13. 论文贡献对应与表述边界

实现完成后，本阶段支持以下论文主张：

> ForceRFT 在不替换 SmolVLA 原生 Flow Action Expert 的前提下，引入力觉
> 感知 Twin-Q，并使价值梯度穿过完整十步 Euler flow sampling，从而直接
> 调整力觉条件化动作向量场。离线阶段保持完整 Actor 可训练，Flow Matching
> 约束策略先验，target Twin-Q 的 clipped backup 与经验校准的 conservative
> objective共同构成离线强化微调信号。Twin-Q 对一个10 Hz决策周期内实际
> 执行macro-action的有效6-DoF TCP分量提供可微价值引导；离散夹爪参与价值
> 条件化，但由示范目标监督。

必须同时限定：

- Critic 是 observation-conditioned，而非已证明具备完整 Markov state；
- 当前离线 action source 是 recorded command/target，不是 verified actual
  apply；
- Q guidance直接作用于chunk前K=3个nominal recorded-command slot的6D TCP
  分量，共享参数使
  其影响整个Actor，但不声称其余H-K个slot具有独立Q标签；
- Reward Classifier只提供任务完成检测，不构成本文Actor或Critic创新；其
  checkpoint、阈值和人工复核必须作为实验协议披露；
- 只有成功示范时不能声称失败恢复；
- 单帧wrench仍主要是static force conditioning；
- 离线Q指标不能替代固定协议的真机成功率评估；
- frozen-VLM在线HIL属于第三阶段，不得在第二阶段完成前写成已验证结果。

---

## 14. 固定参考

- ForceSmolVLA 第一阶段：`ForceSmolVLA_Implementation_Spec_v4_2`
- ConRFT repository：<https://github.com/cccedric/conrft>
- ConRFT license：Apache-2.0；复用源码时保留许可证与论文引用
- ConRFT estimator reference commit：
  `a779fde7fa5db5a469960a8490c100f35b41b49e`
- ConRFT reward classifier：
  <https://github.com/cccedric/conrft/blob/a779fde7fa5db5a469960a8490c100f35b41b49e/serl_launcher/serl_launcher/networks/reward_classifier.py>
- ConRFT reward-classifier training：
  <https://github.com/cccedric/conrft/blob/a779fde7fa5db5a469960a8490c100f35b41b49e/examples/train_reward_classifier.py>
- ConRFT classifier-data collection：
  <https://github.com/cccedric/conrft/blob/a779fde7fa5db5a469960a8490c100f35b41b49e/examples/record_success_fail.py>
- ConRFT agent：
  <https://github.com/cccedric/conrft/blob/a779fde7fa5db5a469960a8490c100f35b41b49e/serl_launcher/serl_launcher/agents/continuous/conrft_single_octo_cp.py>
- SmolVLA/LeRobot pinned commit：
  `30da8e687a6dfc617fcd94afc367ac7071c376ce`
