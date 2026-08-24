# Force-SmolVLA 离线实现规格（修订版 v4.1 available-sensor data profile，供 Codex 执行）

> **已由 v4.2 取代。** 自 2026-08-20 起，训练、推理 API、checkpoint/resume、validation 与 P5–P9 验收以 `ForceSmolVLA_Implementation_Spec_v4_2.md` 为唯一优先 source-of-truth；本文件仅保留 available-sensor 数据与几何契约的历史全文。两者冲突时必须执行 v4.2。

> 状态：Implementation Plan — Revised v4.1  
> 平台：Ubuntu 22.04；单张 RTX 4090D 24 GB  
> 训练：纯离线 Flow-Matching SFT；部署：只读 Shadow  
> 本文件完整取代 v4 的数据采集与几何要求。当前 task1 转换为 LeRobot v3 后可用于正式、within-session offline fine-tuning。任何已批准的数据质量 gate、P0 断言、工件兼容性检查或必测项目失败，均阻止对应 episode 进入正式训练、评测或 Shadow。
> 2026-08-19训练阶段修订：当前offline SFT为全参数微调；未来human-in-the-loop online fine-tuning才切换为VLM/vision冻结。两阶段不得共享optimizer state。
> 2026-08-20训练recipe修订：active offline SFT采用ForceVLA×SmolVLA单遍联合更新；P7 exact two-pass仅作为路由、梯度和数值一致性的acceptance oracle，不进入长程SFT主循环。训练预算以samples/epochs为主口径，optimizer updates只允许作为由有效batch size推导的实现指标。

v4.1 采用 available-sensor profile：约 100 Hz measured TCP pose + raw wrench + 两路 RGB + 7D absolute action。measured joint-q、joint FK、1 kHz pose、固定 2 ms pose age、每 session 独立 calibration bundle和 session-disjoint split均不是本实现的数据前置条件。所有论文与日志统一使用“calibrated TCP wrench conditioned on measured TCP pose”，不得声称 joint-FK 或 1 kHz geometry fidelity。

## 1. 目标、主路径与边界

实现独立的 ForceSmolVLAPolicy：以 SmolVLA 视觉/语言前缀为条件，以经标定的 6D TCP wrench 形成一个 Force Token；该 token 在 post-VLM 固定物理布局中与图像/语言 token 融合，再通过 Action-Query Force Residual Adapter 修正原生 Action Expert 的 velocity。当前offline阶段该视觉/语言前缀参与全参数微调；未来online HIL阶段才冻结。

    D435 RGB + D405 RGB + prompt + 7D state
      -> SmolVLM prefix prefill（offline trainable；online HIL frozen）
      -> PrefixContext(prefix_out, PrefixLayout, prefix cache)

    raw HEX-E wrench + causal geometry/quality pipeline
      -> canonical compensated TCP wrench6
      -> ForceMLP -> Force Token
      -> ForceToken-Dense 或 ForceToken-MoE context refiner
      -> immutable ForceContext

    noisy packed 32D action + flow time + native Action Expert
      -> suffix_out
      -> unique fp32 Force-Action residual hook
      -> native fp32 action_out_proj
      -> 32D velocity；只有前 7D 可用

主实验为 ForceToken-MoE；ForceToken-Dense、ForceConcat、NoForce、Additive Guidance 是预注册的独立对照，不得在同一 checkpoint 中临时切换结构。

当前offline阶段明确排除 Twin-Q、critic、online RL、reward/replay、HIL、真机 action 下发、Franky/RTC 队列改造、ForceVLA JAX/Flax 权重迁移、逐层 force KV 注入和未记录 LoRA。未来HIL必须作为`online_hil_vlm_frozen`独立阶段另行实现和批准。Shadow 只读传感器和记录，不发送机器人动作。

## 2. 冻结 revision、配置和 RTC 隔离

### 2.1 源码绑定

训练前必须写入并签名 source_binding.json：

    lerobot repository URL, frozen commit, wheel/git SHA256
    configuration_smolvla.py SHA256
    modeling_smolvla.py SHA256
    ForceSmolVLA code SHA256
    base checkpoint repo/revision/local SHA256
    config-class and policy-factory registration symbols
    num_steps -> ForceVLAFlowMatching.sample_actions/euler_integrate binding
    rtc_config -> policy.init_rtc_processor and model._rtc_enabled binding
    every custom override class/method source SHA256

不得依赖 main 的未冻结行为。字段、方法或掩码语义与本规格不一致时，preflight 必须失败，而不是尝试兼容猜测。

### 2.2 唯一允许的运行时值

从 base checkpoint 解析并写入 resolved_force_config.json 的具体值；该文件签名后才可构造模型。最少包含：

    config.type = force_smolvla
    chunk_size = H = 50
    max_state_dim = 32
    max_action_dim = 32
    n_obs_steps = 1
    observation_delta_indices = [0]
    use_cache = True
    attention_mode = cross_attn
    config.num_steps = 10
    n_action_steps = base checkpoint 的原始值，原样保留
    shadow.execution_horizon = K = 3
    prefix_length = N_prefix_physical
    pad_language_to = max_length
    tokenizer_max_length = L_lang
    adapt_to_pi_aloha = False
    use_delta_joint_actions_aloha = False
    empty_cameras = 0
    training_stage = offline_full_finetune | online_hil_vlm_frozen

唯一的采样步数字段是 ForceSmolVLAConfig.num_steps。禁止定义或读取 FlowMatching 实例上的第二个 num_steps 成员。K=3 只由 ShadowExecutor 消费；不得作为 sample_actions、euler_integrate、RTC 或原生 queue 参数传入。n_action_steps 不等于 K，也不决定 Shadow 实际消费长度。

训练 full-forward 显式 use_cache=False；cached prefill 与 denoise 显式 use_cache=True。二者均必须满足第 8、11 节的数值 parity。

### 2.3 RTC 必须 fail-closed

仅设 rtc_config=None 不足以关闭当前父类能力。ForceSmolVLAPolicy 和 ForceVLAFlowMatching 必须实际使用下列等价覆盖：

    ForceSmolVLAPolicy.supports_rtc() -> False
    ForceSmolVLAPolicy.init_rtc_processor():
        assert config.rtc_config is None
        rtc_processor = None
    ForceSmolVLAPolicy._rtc_enabled() -> False
    ForceVLAFlowMatching._rtc_enabled():
        assert config.rtc_config is None
        return False

任何非 None 的 RTCConfig，即使其 enabled=false，也使 preflight、reload 和 Shadow 失败。predict_action_chunk 拒绝 inference_delay、prev_chunk_left_over、execution_horizon 及所有 RTC kwargs。构造器必须实例化 ForceVLAFlowMatching，不能遗留普通 VLAFlowMatching。

## 3. 二相机、语言和固定 physical prefix

唯一允许的视觉配置为二相机重定版：

    tuple(config.image_features.keys()) =
      (observation.images.camera1, observation.images.camera2)
    camera1 = D435 third-person RGB
    camera2 = D405 wrist RGB
    两个 feature 均为 FeatureType.VISUAL
    empty_cameras = 0

若 base checkpoint 有第三 slot，必须在 ForceSmolVLAConfig 创建时完成二相机重定和权重 allowlist 验证；禁止 placeholder、empty camera 或运行时选择。batch 中必须存在且只存在两个上述 visual key，否则 fail-fast。

visual_language_manifest.json 逐项冻结：

    camera slot key/order/FeatureType
    RGB（禁止 BGR）源色彩顺序、resolution、dtype、range
    resize/pad/crop/interpolation/normalization
    tokenizer revision、prompt template、chat template
    language_tensor_shape = [B,L_lang]
    padding=max_length, padding_side=right
    truncation=true, truncation_side=right
    config.pad_language_to=max_length
    output keys = observation.language.tokens,
                  observation.language.attention_mask
    N_prefix_physical and config.prefix_length equality
    max_camera_age_ms=66.7
    max_intercamera_skew_ms=33.3
    missing-image policy=fail_fast

在 t_ref 对每路选择最新且仅满足 t_image<=t_ref、t_ref-t_image<=66.7 ms 的帧；两相机时间差不得大于 33.3 ms。不得用未来帧、补零、复制另一相机或静默丢失 slot。raw task 字符串不得直接传入 policy forward。

## 4. 7D state/action、坐标和标签语义

    state7 =
      [tcp_x,tcp_y,tcp_z,tcp_roll,tcp_pitch,tcp_yaw,gripper_position]
    absolute_action7[k] =
      [target_x,target_y,target_z,target_roll,target_pitch,target_yaw,
       target_gripper_position]

state、action 与 wrench 使用同一 task TCP、同一 fr3_link0、同一单位和右手系。各 source timestamp 与 clock map 必须保留。位置单位 m。姿态为右手 ZYX：

    R = Rz(yaw) @ Ry(pitch) @ Rx(roll)
    roll,yaw in [-pi,pi)
    pitch in [-pi/2,pi/2]

abs(abs(pitch)-pi/2)<2 deg 的 state 或 target 为 orientation-invalid，不能进入训练、正式指标或 Shadow candidate。source rotation 必须先转矩阵、再以此 principal chart canonicalize。任何无法解释的约 2pi branch jump 为 schema failure。

第 7 维固定保留当前 raw dataset 的连续 Robotiq width 语义：

    state[6] = gripper_state.width_m
    action[6] = accepted_reference.target_gripper_width_m
    unit = m

原始 width 字段、方向、观测范围与任何 clipping必须入 manifest；不得把 width-m静默换成 motor/controller position。`controller_position`仅作为 provenance/audit保留。本阶段只做离线 SFT，不构造 flange控制命令、不要求 joint FK，且不发送机器人动作。action标签保持原始 7D absolute TCP target；未来部署所需的 gripper/TCP-to-controller命令变换另行冻结，不得倒推改变本数据集标签。

原始 absolute target 到训练 target 的唯一规则：

    delta_xyz[k] = target_xyz(tau_k) - state_xyz(t_ref)
    delta_rpy[k] = wrap_to_pi(target_rpy(tau_k) - state_rpy(t_ref))
    delta_action7[k] = [delta_xyz,delta_rpy,target_gripper(tau_k)]

ActionDeltaProcessor.from_delta 对前三维加 state_xyz，对 RPY 使用 canonicalize_ZYX(state_rpy+delta_rpy)，第 7 维原样保留。inverse 后再执行 orientation、workspace 与 gripper-range 检查。它不是 stateless action postprocessor；必须使用该 chunk 自己的 raw_state_snapshot。

## 5. WrenchGeometrySpec、因果处理和质量闸门

### 5.1 WrenchGeometrySpec 是必需工件

wrench_geometry_spec.json 必须包含 schema_version、SHA256、全部 frame/单位/手性、时钟域、transform source、静态外参、算法与 failure code。正式 approval/signature 机制冻结前，该工件标记为 development-only；正式命令对缺失的批准信息 fail-closed。其唯一主路径如下；pose 与 wrench 的 source timestamp 必须处于同一已声明时钟域，或先由冻结 clock map 映射到同一时间轴：

    raw wrench sample at t_w in sensor frame S
    measured TCP pose stream T_B_C(t_pose), approximately 100 Hz
    B = fr3_link0, C = task TCP, S = wrench sensor

    dynamic_pose_source = measured_tcp_pose
    lookup_mode = causal_zoh; interpolation=disabled
    allow_future_pose = false
    max_pose_age_ms = required configuration; no implicit default
    i* = max{i | t_pose[i] <= t_w}
    t_pose = t_pose[i*]
    require 0 <= pose_age_ms=(t_w-t_pose)/1e6 <= max_pose_age_ms

    static calibration supplies T_C_S = T_TCP<-sensor
    T_B_S(t_w) = T_B_C(t_pose) @ T_C_S

T_C_S、payload mass/CoM、sensor sign matrix、source topics/fields、calibration_id 与必要 clock map 均由 calibration artifact 冻结。不得用 action label、desired pose、command target 或 t_w 之后的 pose；也不得线性、SLERP、SE(3) 或任何非因果插值。不得以 joint-q/FK 或 1 kHz geometry fidelity 描述该路径。

从传感器到 TCP 的唯一力矩变换为：

    [F_S,M_S] = signed_raw - bias
    [F_Sc,M_Sc] = payload_gravity_and_gravity_moment_compensate(
                       [F_S,M_S], T_B_S(t_w), calibration_artifact)
    F_B = R_BS(t_w) @ F_Sc
    M_B_at_S = R_BS(t_w) @ M_Sc
    p_BS = translation(T_B_S(t_w))
    p_BC = translation(T_B_C(t_w))
    M_B_at_C = M_B_at_S + (p_BS-p_BC) x F_B
    wrench6_BC = [F_B,M_B_at_C]

这里 compensation 的 sign、payload reference frame 与 moment convention 必须由一个 reference fixture 验证；不得在 policy 内重复 compensation。统一术语为“calibrated TCP wrench conditioned on measured TCP pose”；不得写作 external wrench、joint-FK wrench 或 1 kHz geometry。

每一 raw wrench/tuple 必须记录 raw timestamp、filter input/output timestamp、t_pose、pose_age_ms、calibration_id、T_B_C/T_B_S source id、T_C_S hash、clock-map id/hash（若使用）以及 finite/device-status/saturation/timestamp-monotonicity/drift/filter-reset/warm-up/geometry validity。geometry_valid=false、pose 缺失/过期、future-pose 尝试或 frame/calibration 不匹配均使该 raw sample 无效；这种 raw sample 不得更新 filter state。在 Shadow 中它使 candidate_valid=false，绝不以旧 geometry 继续推理。

### 5.2 过滤和因果重采样

wrench_filter_resample_spec.json 固定：

    raw sample rate and source clock
    causal filter family/order and exact SOS or b/a coefficients
    cutoff=8 Hz, library/version
    initialization=first-valid steady-state
    reset=every episode
    warm-up disposition=invalid
    output rate=30 Hz
    resampler=causal ZOH of latest filtered sample
    source sample time must be <= t_ref

禁止双向滤波、跨 episode filter state、线性插值和 future sample。filter warm-up 期间的 tick 不可产生 tuple。

### 5.3 force quality gate 与可审计 YAML

先于 split 冻结 force_quality_thresholds.yaml：

    force_quality_valid(episode) =
      bias_drift_pass
      and gravity_residual_pass
      and tcp_lever_arm_validation_pass
      and saturation_pass
      and clock_offset_pass
      and wrench_geometry_pass

force_quality_thresholds.yaml 和 shadow_safety_thresholds.yaml 必须服从同一 RuleSpec schema。每一条 rule 必含：

    rule_id, schema_version, description
    input fields and input artifact hashes
    frame, unit, clock domain
    exact formula/formula_revision
    window length, sampling rule, aggregation statistic
    numeric threshold, comparator, threshold provenance
    calibration/protocol source and approval id
    failure_code, severity, remediation
    golden-fixture id/hash and expected result

force_quality_thresholds.yaml 在冻结前必须填满（不得留 symbolic/null/default）下列 rule 的 expression、window、unit、numeric threshold、threshold provenance 与 failure code：

    WQ_BIAS_DRIFT:
      max_axis(abs(TheilSenSlope(bias_estimate_axis(t),t)))
      over the declared zero-contact window, unit=N/s or Nm/s
    WQ_GRAVITY_RESIDUAL:
      median over declared free-space/static-pose window of
      norm(F_compensated) and norm(M_compensated), units=N and Nm
    WQ_TCP_LEVER_ARM:
      percentile_95 over declared lever-arm fixture of
      norm(M_tcp_measured-M_tcp_reference), unit=Nm
    WQ_SATURATION:
      max_axis(count(abs(raw_axis)>=full_scale_axis-margin_axis)/N), unit=fraction
    WQ_CLOCK_OFFSET:
      abs(offset)+confidence_bound, unit=ms
    WQ_GEOMETRY:
      all(wrench_geometry_valid) and max(pose_age_ms)<=configured max_pose_age_ms,
      unit=bool/ms; max_pose_age_ms has no implicit default

shadow_safety_thresholds.yaml 必须以同一方式填满 SS_WORKSPACE、SS_ORIENTATION、SS_DELTA_XYZ、SS_DELTA_ROT_GEODESIC、SS_GRIPPER_RANGE_RATE、SS_CONTINUITY、SS_OBSERVATION_AGE、SS_TRANSPORT、SS_END_TO_APPLY、SS_SLOT_LATENESS、SS_EXPIRED_RATE、SS_MISSED_TICK_RATE 和 SS_HOLD_OVERRUN。每项 scope 必须为 session、tuple、candidate 或 run_aggregate；candidate fail 只能拒绝该 candidate，aggregate fail 必须标记整个 Shadow run 不通过。任何 Python 常数若未出现在这两个签名 YAML 中即为启动失败。wrench_quality_manifest.json 记录 serial、Compute Box、bias、payload/TF、bundle hash、全部统计量和判定。

    eligible_episode = temporal_valid and force_quality_valid

任意 false 的 episode 不得进入 normalizer fitting、任何 split、sampler、训练、验证、正式指标或 Shadow acceptance。

## 6. 7D-to-32D packing、feature mask 与 processor 图

必须显式存在：

    state_feature_mask [B,32]
    action_feature_mask [B,H,32]
    action_valid_mask [B,H]
    flow_valid_mask [B,H,32] = action_feature_mask
    suffix_valid_mask [B,H] = action_valid_mask

主模型：

    state_active_dim=7
    state_feature_mask[b,d]=1[d<7]
    action_feature_mask[b,k,d]=action_valid_mask[b,k] * 1[d<7]

ForceConcat baseline 例外：

    state_active_dim=13
    state32[:,:13] = [normalize_state7, normalize_wrench6]
    state padding=19D；action padding仍为25D

ForceConcat-13D 是独立的 pre-VLM state-injection baseline，不是主模型的“7D state path”。它的 ForceConcatStatePackerProjection 固定为无参数 identity pack，added parameters=0；它使用独立、仅 train-split fit 的 state7/wrench6 normalizer、13D mask 和 resolved config，绝不把 raw 13D 先拼接后共享一个 std。它没有 Force Token、fusion/refiner 或 adapter；其 action target、flow/noise、split、updates 和 validation scalar 与 SmolVLA-Cartesian7D 相同。

唯一的 NoForce 为 PostVLM-NoForceAdapter：它保留所选 ForceToken 主模型完全相同的 ForceMLP、selection、fusion/refiner、adapter、masks、params 和 compute；只把已经过正常 wrench normalizer 的 wrench6 替换为零，且 Force Token 仍有效。禁止将“全零 Force Token”和“mask/drop Force Token”混为同一个 NoForce 对照。

state/action padding 不承载信号。所有 25 个 action pad 维不得参与 noise、interpolation、self-attention 内容、loss、normalizer statistics、inverse transform 或有效 7D 输出。所有无效 feature 在 state projection 前后、action projection 前后、flow interpolation 前后和 Euler update 后置零。改变任何 padding 值不得改变前 7D velocity 或 absolute output。

合法顺序唯一：

    raw state7/wrench6/absolute action7
    -> ActionDeltaProcessor.to_delta on raw fields
    -> custom state7/wrench6/delta_action7 normalizers exactly once
    -> 32D pack and masks
    -> image/language processor -> policy

Shadow：

    raw observation + raw ChunkContext
    -> runtime compatibility gate
    -> custom normalizers exactly once
    -> predict_action_chunk
    -> extract active 7D, unnormalize once
    -> whole-chunk ActionDeltaProcessor.from_delta(raw_state_snapshot)
    -> absolute [B,H,7] handoff

继承 checkpoint 的 STATE/ACTION normalizer 必须 removed、Identity 或可证明 disconnected；否则启动失败。所有 inherited Aloha/delta-joint transforms 必须 disabled/Identity。processor_graph_manifest.json 冻结 call graph、callable hash、keys、shapes、dtypes、normalizer owner 与 call count。

ChunkContext 的每个字段均 batch-bound：

    raw_state_snapshot [B,7]
    t_ref, tau_0, clock_domain_id [B]
    episode_id, session_id, sample_id, chunk_id [B]
    action_valid_mask/suffix_valid_mask [B,H]
    calibration_bundle_hash, wrench_geometry_spec_hash, normalizer_hash [B]
    calibration_mapping_hash_or_none, wrench_geometry_valid,
      runtime_artifact_compatible [B]
    selected state/wrench/image IDs/timestamps/ages [B,...]

禁止 batch scalar、global last_state 或 processor 丢失 ChunkContext。

### 6.1 calibration/normalizer 兼容性闸门

checkpoint 必须包含：

    canonical_calibration_bundle_sha256
    wrench_geometry_spec_sha256
    normalizer_stats_sha256
    normalizer_fitted_calibration_bundle_sha256
    accepted_calibration_bridge_allowlist[]

每次训练、evaluation、reload 和 Shadow 在 normalizer 调用前检查：

    calibration_compatible =
      runtime_normalizer_sha256 == normalizer_stats_sha256
      and runtime_wrench_geometry_spec_sha256
          == wrench_geometry_spec_sha256
      and normalizer_fitted_calibration_bundle_sha256
          == canonical_calibration_bundle_sha256
      and (
        runtime_bundle_sha256 == canonical_calibration_bundle_sha256
        OR
        an allowlisted calibration bridge exactly maps
        runtime_bundle_sha256 -> canonical_calibration_bundle_sha256
        and its SHA256, source/target serials, formula, units, frame,
        golden residual test and approval id all verify
      )

calibration bridge 只可在 wrench normalizer 之前运行；不得在 normalizer 空间拟合临时补偿。任何 hash/version/serial/Compute Box 更换不匹配、geometry spec 不匹配、bridge 缺失、bridge 测试失败或 normalizer hash 不匹配，均为 CALIBRATION_NORMALIZER_INCOMPATIBLE：训练/评测 fail-fast；Shadow candidate_valid=false，且不运行 model。失败样本的 normalizer invocation count 必须为零。不得仅记录 version 后继续使用旧 wrench normalizer。normalizer 只能用 eligible train raw tuples fit；stats 随后冻结，validation/test/Shadow 只可 apply。

## 7. 时间锚点、tuple 构造与 split

转换器只读原始数据。先定义 controller-clock reference grid：

    session_epoch = first 30-Hz grid tick >= session_start_ack_timestamp
    t_ref[q] = session_epoch + q/30 s, q=0,1,...

每个 grid tick 都记录 q、clock-map version 与选择结果。无效 tick 只 reject，不移动 epoch、不填补、不重新编号。

    Delta_a = 1/30 s
    tau_0 = t_ref + Delta_0
    tau_k = tau_0 + k*Delta_a, k=0,...,49

label[k] 仅为 acknowledgement 证明在 tau_k 生效的 controller-applied ZOH target。输入 image/state/wrench 只能取时间 <=t_ref 的数据；future action 仅可作为 label。state/wrench 分别选择各自最新有效样本，最大 age 均为 33.3 ms；缺失、过期、无 clock map 或 geometry invalid 均 reject tuple。

action_valid_mask[k]=true 当且仅当精确 acknowledgement/ZOH label 存在、在 episode 内、并通过 target语义检查。尾块仅在 sum(action_valid_mask)>=K=3 时保留；m=1、2 一律丢弃。Flow loss 在 batch、gradient accumulation 和 DDP 上按全局 valid feature token 分子/分母加权，不允许按 chunk 平均。

先运行逐 episode 数据质量 gate，再按冻结的 episode_id 做 episode-disjoint train/val/test split。同一 episode 的任何 tuple 不得跨 split；当前数据只有一个 recording session，这不阻止正式离线 SFT，但所有结果必须明确标注为 within-session offline fine-tuning，不得作为跨 session 泛化结论。只有 split 后 eligible train episodes 的 raw tuples 能 fit normalizers；validation/test 只 apply 冻结 stats。主实验使用 uniform eligible-chunk sampler；contact-aware sampler 默认为 disabled。

## 8. PrefixContext、cache 和 B>=2 parity

encode_prefix 必须返回：

    prefix_out [B,N_prefix_physical,D_vlm]
    prefix_valid_mask [B,N_prefix_physical]
    prefix_segment_ids [B,N_prefix_physical]
    PrefixLayout(camera1 span,camera2 span,language span,state span,pad span,
                 prefix_position_ids,N_prefix_physical,N_prefix_valid)
    past_key_values

N_prefix_physical 是 cache crop 的长度，包含右 padding；N_prefix_valid 仅用于 token/position validity，绝不可用于 crop。native denoise 每步可 append suffix K/V 后 crop(N_prefix_physical)；本规格中的“不可变 cache”仅指 force branch 不改动 prefix cache content。每个 Euler step 后必须断言 cache length/content 回到 prefill 值。

fused selection 固定物理索引，不做 ragged compaction：

    camera1 physical span, camera2 physical span,
    L_lang right-padded language span, one final force slot

state span 和 native prefix padding span 不进入 fusion selection。fused_valid_mask 包含 camera/language valid mask 和 Force slot=true。fusion_include_state_token=false 只称 state-token-excluded；必须测量 contextual state leakage，不能称严格 FVL-only。

在 eval、dropout=0、B>=2、同一 physical prefix length而不同语言 valid length的异构 batch下，比较：

    contextual valid prefix_full vs prefix_prefill
    cached vs reference full-prefix velocity
    prefix cache K/V content/physical length after every Euler step

固定 raw observation、wrench、noise、time；容差由 resolved config 写死。任何 physical/valid length 混用、camera 顺序变更或 cache content 变化均失败。

## 9. Force Token、Dense/MoE resolved configs 和精确结构

### 9.1 所有 ForceToken variants 的共同结构

preflight 从冻结 base 读取 D_vlm、D_expert、D_action；写为具体正整数并签名。若 D_vlm!=D_expert，唯一允许的跨宽度路径为 guidance_projection=Linear(D_vlm,D_expert,bias=true)。构造器必须断言 shapes、head divisibility、state_dict 与 resolved config 一致。

共同 force encoder：

    h = SiLU(Linear(6,D_vlm)(normalized_wrench6))
    force_token = Linear(D_vlm,D_vlm)(h)          # [B,D_vlm]

两层 Xavier-uniform、bias=0；不得对 raw wrench6 作 LayerNorm。所有新模块参数存储为 fp32，标准 AdamW state 因此也为 fp32；matmul/attention activations 可在 bf16 autocast。router logits、softmax、aux losses、guidance projection、adapter 和 action head 使用第 11 节规定的 fp32 path。

固定 Force Token 拼接公式，禁止隐式加法：

    X_prefix = gather(prefix_out, fusion_selection_indices)
             # [B,N_selected_prefix,D_vlm]
    F = ForceMLP(normalized_wrench6)              # [B,D_vlm]
    X_tokens = cat([X_prefix, F[:,None,:]], dim=1)
             # [B,N_fused_physical,D_vlm]
    segment_ids =
      [CAMERA1]*N_cam1 + [CAMERA2]*N_cam2
      + [LANGUAGE]*L_lang + [FORCE]
    X0 = (X_tokens
          + segment_embedding[segment_ids]
          + fusion_position_embedding[0:N_fused_physical])
         * fused_valid_mask[:,:,None]

Force slot index 固定为 N_fused_physical-1，segment id 固定为 FORCE；camera1、camera2、language、force 四类 id 不得在 checkpoint 间重排。fusion_position_embedding 是固定 physical table，不以每样本 valid length 决定位置。

共同两层 FusionBlock，l=1,2：

    A_l = MHA(LN_attn_l(X_{l-1}), valid=fused_valid_mask)
    Y_l = (X_{l-1}+A_l) * fused_valid_mask[...,None]
    B_l = Linear(D,4D)->GELU->Linear(4D,D)(LN_ffn_l(Y_l))
    X_l = (Y_l+B_l) * fused_valid_mask[...,None]

LN 为 LayerNorm(eps=1e-5)，MHA heads=8、Q/K/V/O 均 Linear(D,D,bias=true)，dropout=0，所有 init 为 config 指定 Xavier-uniform/bias=0。无效 key score 在 softmax 前置 -inf，无效 query output 精确为 0。

### 9.2 三个明确 resolved context-refiner 配置

不得再使用没有 budget 的“Dense”。每个实验各有独立 resolved config、checkpoint type 和 manifest。

| ID | context refiner（对 X2） | budget 定义 | router |
| --- | --- | --- | --- |
| ForceToken-Dense-Compute | Z=X2+DenseMLP(LN_dense(X2)); DenseMLP: D->4D->D, GELU | 与 top-1 MoE 单 token active MACs 匹配，偏差<=1% | 无 |
| ForceToken-Dense-Param | Z=X2+DenseMLP(LN_dense(X2)); D->h_param->D, GELU | preflight 解 h_param，使总 trainable 参数相对 MoE <=0.1%；记录 exact count | 无 |
| ForceToken-MoE | Z=X2+p_route*Expert_route(LN_moe(X2)) | 主实验；4 个专家，active 为 top-1 | capacity-free deterministic top-1 |

ForceToken-Dense-Compute 是 P5/P6 成本验收的唯一 Dense 结构；ForceToken-Dense-Param 是参数量对照。它们分别报告 compute-matched 与 parameter-matched，禁止声称一个 Dense 同时匹配二者。

令 D=D_vlm，忽略共同 LayerNorm 时：

    P_dense_compute_refiner = 8D^2 + 5D
    P_moe_refiner = 4*(8D^2+5D) + (4D+4)
                   = 32D^2 + 24D + 4
    active_MAC_dense/token = 8D^2
    active_MAC_moe/token = 8D^2 + 4D + D

ForceToken-Dense-Param 的 h_param 是 preflight 解出的非负整数：

    h_param = argmin_h |h*(2D+1)+D - (32D^2+24D+4)|

并硬断言参数差相对 MoE <=0.1%。所有 count/MAC 均写入 resolved config 与最终实验矩阵；不能把 Dense-Compute 称为 parameter-matched。

对任一 valid token i，Dense 的唯一实现为：

    u_i = Linear(D,4D)->GELU->Linear(4D,D)(LN_dense(X2_i))
    Z_i = (X2_i+u_i) if fused_valid[i] else 0

Dense-Param 仅将中间宽度 4D 替换为已签名 h_param；共同两层 FusionBlock、ForceMLP、selection、projection、adapter 和 action head 不得改变。

ForceToken-MoE 的精确模块：

    LN_moe = LayerNorm(D,eps=1e-5)
    router = Linear(D, E=4, bias=true)
    logits_i = router(LN_moe(X2_i)).float()
    p_i = softmax(logits_i)
    e_i = argmax(p_i)                    # stable deterministic tie-break=min id
    Expert_e = Linear(D,4D)->GELU->Linear(4D,D), bias=true
    Z_i = (X2_i + p_i[e_i]*Expert_ei(LN_moe(X2_i)))
          if fused_valid[i] else 0

router、每个 expert、每个 norm 均为独立具名 state_dict 模块。router 与 expert Xavier-uniform/bias=0；无 warm-up、top2、capacity、fallback 或 token drop。测试应验证同一固定 token 在 B=1、batch permutation、竞争 co-batch 中 route id、p-vector 与输出不变，并验证 zero drop；不要求不同样本去同一 expert。

每个 config 还必须固定并写入：

    D_vlm,D_expert,D_action
    fusion_num_blocks=2, fusion_num_heads=8, fusion_ffn_hidden_dim=4*D_vlm
    force_token_count=1, position/segment scheme
    force_encoder type=ForceMLP only
    Dense hidden dim 或 E=4/expert_hidden=4*D_vlm
    router temperature=1, top_k=1, capacity_free=true
    all dropout=0, all init seeds/hashes
    guidance projection/Q/K/V/W_out shapes and initialization

## 10. 分阶段 trainable set 与 Action-Query adapter

唯一允许的训练阶段为：

1. `offline_full_finetune`（当前）：所有现存模型参数都必须`requires_grad=True`，包括vision encoder、language/VLM backbone、Action Expert、state/action/time projections、action output head以及全部ForceMLP/fusion/refiner/router/guidance/adapter参数。顶层`model.train()`后VLM与vision必须处于train mode。若任何具名参数被冻结，启动失败；同时报告实际获得gradient的参数覆盖率，区分“可训练但本路径未使用”和“被冻结”。
2. `online_hil_vlm_frozen`（未来）：完整VLM（含vision encoder、language backbone和connector）必须`requires_grad=False`并保持eval mode；Action Expert、state/action/time projections、action output head以及全部Force模块继续可训练。该阶段未实现HIL数据/安全闭环前不得启动。

checkpoint、resolved config、日志和training recipe必须记录`training_stage`。从offline切换到online HIL时必须重新构造optimizer/scheduler，禁止加载offline optimizer state；反向切换同样禁止复用optimizer state。base weights、normalizer和数据split兼容性仍独立验证。

prefill 后仅一次构造：

    ForceContext(
      z_action_fp32 = guidance_projection_fp32(Z_fused.float()),
      fused_valid_mask = fused_valid_mask
    )

它与 cache 一样绑定本 batch，force branch 不修改 native prefix cache。唯一 velocity hook 位于 training forward、reference full-prefix denoise、cached denoise 的共同位置：

    suffix_out[:, -H:] -> velocity_from_suffix -> native action_out_proj

    learned_action_slot = Parameter[H,D_expert], Normal(0,0.02)
    time_projection = Linear(1,D_expert,bias=true), Xavier/zero-bias
    noisy_action_projection = Linear(7,D_expert,bias=true), Xavier/zero-bias
    sanitized_noisy_action7 = x_t[:,:,:7] * suffix_valid_mask[...,None]
    C = learned_action_slot[None] + time_projection(t[:,None,None])
        + noisy_action_projection(sanitized_noisy_action7)

    S = suffix_out[:, -H:].float()
    Q_main = S + C
    G = ForceCrossAttention(Q_main,
                            ForceContext.z_action_fp32,
                            query_valid_mask=suffix_valid_mask,
                            key_valid_mask=ForceContext.fused_valid_mask)
    S_hat = (S + tanh(alpha)*W_out(G)) * suffix_valid_mask[...,None]
    velocity32 = action_out_proj(S_hat) * action_feature_mask

ForceCrossAttention 固定为 `single_head_scaled_dot_product`：`num_heads=1`、`D_expert=head_dim=720`、`scale=1/sqrt(720)`，不拆 head，也不使用 `nn.MultiheadAttention`。Q=Linear(D_expert,D_expert,bias=true)(Q_main)，K/V 对 ForceContext.z_action_fp32 使用各自同形 Linear；该 primitive 直接返回 `softmax(QK^T/sqrt(720))V`，唯一输出投影是外部公式中的 W_out=Linear(D_expert,D_expert,bias=true)，不存在内部 out_proj/O 或任何额外 O。不得因 FusionBlock 使用 8-head 而改变这里的单头定义。Q/K/V 和 guidance_projection 均 Xavier-uniform/zero-bias。它为显式 fp32 attention：invalid-key scores=-inf before softmax，invalid-query output=0，Force slot 永远 valid，因此不得出现全 invalid key softmax。guidance_projection、Q/K/V、W_out、alpha、action_out_proj 参数与运算均处于 autocast-disabled fp32 path；普通 fusion 在 bf16 autocast 后，仅在 guidance projection 前 cast fp32。

P5 physical layout 固定为 `camera1=[0,64)`、`camera2=[64,128)`、`language=[128,176)`、`fusion_selection_indices=[0,176)`、`force_slot_index=176`、`N_fused_physical=177`；state token 不进入 fusion。语言保留 48 个 physical right-padded slots并使用对应 valid mask。P5 初始化 seed 固定为 42，同时设置 Python、NumPy、PyTorch CPU 与全部 CUDA RNG；resolved config 必须记录初始化 tensor SHA256。

W_out.weight/bias=0；alpha=atanh(1e-3)。这样初始 residual 精确为零，W_out 首步仍有梯度。初始 parity 对比必须用相同 autocast-disabled fp32 native action_out_proj。

## 11. 精确 Flow Matching、suffix temporal mask 与 Additive 对照

仅在 active 7D 生成随机性：

    epsilon7 ~ iid N(0,I), fp32 [B,H,7]
    epsilon32 = pack(epsilon7)*action_feature_mask
    a32 = pack(normalized_delta_action7)*action_feature_mask
    t ~ Beta(1.5,1.0)*0.999+0.001, fp32 [B]
    t3=t[:,None,None]
    x_t=(t3*epsilon32+(1-t3)*a32)*flow_valid_mask
    u_t=(epsilon32-a32)*flow_valid_mask
    L_flow=sum(flow_valid_mask*(velocity32-u_t)^2) / sum(flow_valid_mask)

采样：

    x_1=epsilon32
    N=config.num_steps=10; dt=-1/N
    t_m=1+m*dt, m=0,...,9
    x_{m+1}=(x_m+dt*velocity(x_m,t_m))*action_feature_mask

native suffix embedding 的 action_in_proj、sinusoidal time embedding、action-time MLP 必须完整保留；自定义 time conditioner 是额外模块，不是替代。ForceSmolVLAPolicy 必须 override parent loss reduction，取得 unreduced error 后按上式全局 mask/reduction；不得调用父类另一次 crop/mean。

native 默认将 suffix time mask 全设为 true，故本实现必须显式扩展 API：

    embed_suffix(x_t,t,suffix_valid_mask)
    make_att_2d_masks(..., suffix_valid_mask)

训练 full path 和 cached denoise 都传入同一 suffix_valid_mask。cached path 特别要求：

    prefix_to_suffix_valid =
      suffix_valid_mask[:,:,None] & prefix_valid_mask[:,None,:]

再与 suffix causal self-mask 拼接；不得只 broadcast prefix key mask 而遗漏 invalid suffix query。invalid tail 的 x/noise/embedding/query/output/residual/velocity 均为零。mask 检查与扰动非干扰测试均为必测；后者不能取代前者。

Additive Guidance 仅与 ForceToken-MoE 主模型比较：

    Q_add = C
    G_add = ForceCrossAttention(Q_add,ForceContext.z_action_fp32,
                                 query_valid_mask=suffix_valid_mask,
                                 key_valid_mask=ForceContext.fused_valid_mask)
    S_hat_add = (S + tanh(alpha)*W_out(G_add))*suffix_valid_mask[...,None]

唯一结构差异是 Q 是否加 S。二者完整 base state_dict、所有新模块名称/shape/count、初始化 tensor、optimizer group、updates、data order、precision、noise/time、valid-token budget 必须在 step 0 精确相同。Additive 的 parameter count、trainable set 和 initialization 是硬相同条件，不是仅报告条件。

## 12. 训练 recipe、单遍联合 SFT 与 exact two-pass gate

training_recipe.yaml 是 checkpoint 工件，主实验固定：

    training_stage=offline_full_finetune
    all existing model parameters requires_grad=true
    AdamW; lr=1e-4; betas=(0.9,0.95); eps=1e-8
    weight_decay=1e-10
    no_decay=bias,norm scale,embeddings,alpha
    grad_clip=10
    training_update_algorithm=single_pass_batch_local
    batch_per_gpu=4; gradient_accumulation=1; effective_batch=4 samples
    primary training budget=80000 samples
    equivalent epochs=80000/resolved_train_split_samples
    derived optimizer updates=80000/4=20000
    warmup=4000 samples（derived 1000 updates）
    cosine to 2.5e-6 at 80000 samples（derived update 20000）
    new-module dropout=0
    seeds=[42,43,44]; deterministic algorithms/cuBLAS settings
    uniform eligible-chunk sampler with serializable RNG/cursor
    checkpoint every 2000 samples（derived 500 updates）
    checkpoint selection=global-valid-feature-token-weighted fixed validation L_flow
    early stopping=disabled

active offline SFT 综合两条已验证路径：沿用官方 SmolVLA 的 Flow-Matching、AdamW、bf16 autocast和H=50，以80000个训练样本为主预算；沿用 ForceVLA 的 batch_per_gpu=4、force/action 分支在同一次 full forward 中联合求导、每 batch 一次 backward 和一次 optimizer.step。20000 optimizer updates仅为80000/4的派生实现指标。一次 full prefix+suffix VLM forward 产生的 prefix_out 同时构造 ForceContext 和 router state；禁止为 active SFT 额外执行 prefix-only VLM forward。offline阶段全部现有参数保持 requires_grad=true，不得静默切换 LoRA 或冻结 VLM/vision。

ForceToken-MoE 与 ForceToken-MoE-Additive 使用 L=L_flow+0.01*L_balance+0.001*L_z。active SFT 中 I 为当前单卡 physical batch 的全部 valid fusion token；N_I=|I|：

    pbar[e] = sum_I p_i,e / N_I
    rbar[e] = sum_I 1[argmax p_i=e] / N_I
    L_balance = E * sum_e(pbar[e] * rbar[e])
    L_z = sum_I(logsumexp(logits_i)^2) / N_I

rbar 的 argmax 不求导，pbar、L_z 与 L_flow 来自同一次带图 forward。当前范围固定单进程单张 RTX 4090D；若 torch.distributed 已初始化，active single-pass resolver 必须 fail-fast，不能把 batch-local auxiliary 静默冒充跨 rank 全局值。

P7 exact two-pass仍由`p7_training_recipe.development.yaml`冻结，但角色仅为训练前或短程回归中的routing/gradient/numerical acceptance oracle，不是 active SFT 更新算法。oracle固定预算为16 samples（8×B2）和一次派生optimizer update；调用必须显式声明`oracle_mode=true`，否则fail-fast。不得循环调用形成长程SFT。oracle对固定窗口执行：

    Pass A (no_grad, no optimizer update, deterministic):
      对整个 accumulation window 仅计算 prefix/fusion/router；
      all-reduce S_p[e]=sum_I p_i,e, S_r[e]=sum_I 1[argmax p_i=e], N_I
      pbar=S_p/N_I; rbar=S_r/N_I

    Pass B (with graph, same window/input/order/RNG):
      对每 microbatch q 使用
      Lbal_q = world_size * E * sum_e(
                 rbar_e * sum_{i in I_q} p_i,e / N_I)
      Lz_q = world_size * sum_{i in I_q}(logsumexp(logits_i)^2) / N_I
      Lflow_q = world_size * flow_numerator_q / N_flow_global
      backward(Lflow_q + .01*Lbal_q + .001*Lz_q)

其中 E=4；rbar 的 argmax 部分不求导，故 Pass B 的梯度与该固定全局定义在 tie 以外相同。Pass A/B 之间不得更新参数；dropout=0，必须保持相同数据与 deterministic path。N_I=0 时两项 auxiliary loss=0。logging value 为 L_balance=E*sum_e(pbar_e*rbar_e)。gate失败时不得启动 active SFT；gate通过后，active SFT不得调用Pass A。

exact gate 的flow N_flow_global 在 Pass B 前由mask预计算并all-reduce，每次backward以world_size缩放以抵消DDP梯度平均。active single-pass SFT的flow denominator是当前physical batch的valid feature token数。resume必须恢复optimizer/scheduler/scaler、所有RNG和sampler cursor；active recipe的accumulation phase固定为0。

SmolVLA-Cartesian7D、ForceConcat-13D 与两个 ForceToken-Dense variant 没有 router，硬设 L_balance=L_z=0，不运行 Pass A；PostVLM-NoForceAdapter 若其 resolved config 是 MoE，则按同一 MoE 两遍算法训练。任何变体都不得因缺 router 而改用不同的 flow denominator、更新数或 validation scalar。

fixed validation fixture 包含固定 tuple list、masks、epsilon7 tensor/hash、t[B] tensor/hash、ChunkContext/hash；eval 两次必须完全复现 scalar。held-out 数据不得影响 normalizer、checkpoint 选择或训练 sampler。

## 13. 30 Hz / 10 Hz / 1 kHz Shadow、兼容性和仲裁

固定：

    data/action rate=30 Hz; Delta_a=33.333... ms
    policy period=100 ms=10 Hz
    controller tick=1 kHz; Delta_c=1 ms
    H=50; K=3

select_action 和一切等价 native queue entrypoint 在任何 deque 改变前抛出 RuntimeError。predict_action_chunk(batch,ChunkContext,noise7_or_seed) 是唯一 API；它拒绝 caller 32D noise，除非逐元素等于受 mask 的 packed epsilon7。它绕过 inherited non-ACTION observation queues；reset 清空所有 action/non-action queue、pending ChunkContext、chunk registry 和 cache reference。

所有 scheduling 时间映射到 controller monotonic clock。ShadowClockMap 必须含 sensor->controller 和 GPU->controller mapping；shadow_clock_sync.yaml 固定 residual/drift 阈值。时钟 map 不存在/过期/超限即 candidate_valid=false。

    t_ready = synchronized GPU ready time
    t_candidate = t_ready + measured transport latency
    j=max(0,ceil((t_candidate-tau_0)/Delta_a))
    t_controller_apply=first 1-kHz tick >= max(t_candidate,tau_j)
    end_to_apply_age=t_controller_apply-t_ref
    slot_lateness=t_controller_apply-tau_j

j>=H 或 j+K>H 为 expired/tail-short，执行零项并重规划。候选 targets 为 absolute_chunk[j:j+K]。t_controller_apply 是实际 first controller slot，不是估计值。

Shadow 在模型前执行 runtime_compatibility_gate、WrenchGeometrySpec 和 force-quality 状态检查。任一 calibration/normalizer incompatibility、wrench geometry invalid、clock invalid 或 safety rule invalid 均使 candidate_valid=false；invalid newer generation 不得取消有效旧计划。

latest-generation-wins arbiter：

    candidate_valid =
      finite(targets) and j+K<=H
      and clock/age/compatibility/geometry valid
      and end_to_apply_age/slot_lateness within thresholds
      and workspace/orientation/per-Delta_a translation-rotation/gripper checks

    dispatch_valid(at actual arrival) =
      candidate_valid and no_hold_overrun
      and continuity(candidate_target[0],
                     target_held_immediately_before_actual_arrival)

只有 dispatch_valid generation 才能原子取消旧计划尚未开始的项；old ready after newer generation 为 stale-reject。每个实际 dispatch 的 hold 为 [arrival_q,arrival_{q+1})；无下一 target 的 ZOH 最大延伸 max_hold_extension_ms=100，超过为 hold_overrun/invalid。actual arrival/hold 必须严格单调、无重叠、无负 interval。指标只用 arbiter 的 actual_dispatched_indices，不得默认 action[0]。

shadow_safety_thresholds.yaml 以 RuleSpec schema 冻结 workspace、orientation、per-33.333ms translation/rotation、gripper range/rate、continuity、max observation age、latency/expired/missed tick 和 hold overrun。主阈值：

    P95(t_ready-t_ref)<=80 ms; P99<=100 ms
    max transport<=10 ms
    P95(end_to_apply_age)<=125 ms; P99<=145 ms; max<=150 ms
    max slot_lateness=1 ms
    expired_chunk_rate=0; missed_policy_tick_rate=0

每条 record 保存 clock map、t_ref/t_ready/t_candidate/t_controller_apply/tau_0/j、calibration/normalizer/geometry hashes、full normalized/absolute chunk、candidate/actual/cancelled indices/intervals、validity/reasons、noise seed/tensor hash、camera IDs/timestamps/hashes、prompt/token hash、full ChunkContext、model/config/all artifact hashes、RTC/queue-disabled status。record 必须可离线重放到相同 actual dispatch。

## 14. 严格 checkpoint/reload 与离线可复现

ForceSmolVLAConfig 以冻结 PreTrainedConfig 机制注册 type=force_smolvla；policy factory 必须导入注册。裸 SmolVLA config 被拒绝。

仅允许两条加载路径：

    from_base_for_initialization(local_base_dir,revision,sha256):
      strict=False 只允许一次；missing keys恰为新 force modules，
      unexpected=[]，所有 base keys 加载，ignore_mismatched_sizes=false。

    ForceSmolVLAPolicy.from_pretrained(local_force_ckpt_dir):
      仅本地完整 Force checkpoint；strict=True；
      local_files_only=true；拒绝 remote repo id、force_download 和裸 base config。

完整 checkpoint 内嵌 base_assets/（architecture config、tokenizer、processor、chat template 等 constructor assets）和完整 model.safetensors。constructor strict load 前只能读已签名的本地 base_assets，必须 load_vlm_weights=false，任何 Hub call 失败。

artifact 至少含 config.json、model.safetensors、base_assets/、custom normalizer stats、ActionDelta spec、feature/processor/visual/wrench geometry/quality/temporal/calibration manifests、resolved force architecture/dtype/flow/training configs、artifact_manifest.json 和由 CLI/registry 提供的 detached trusted manifest SHA/signature。每个 payload hash、calibration bridge allowlist、requirements.lock/environment manifest/wheelhouse hash 都须验证。resume 另存 optimizer/scheduler/scaler/RNG/sampler state。

checkpoint还必须记录`training_stage`和精确trainable/frozen parameter name/hash。resume时stage必须完全相同；stage不同只能加载model tensors并新建optimizer/scheduler，不得恢复optimizer/scaler/accumulation state。

在空 HF cache、无网络、HF_HUB_OFFLINE/TRANSFORMERS_OFFLINE/HF_DATASETS_OFFLINE=true、所有 hub APIs 被 monkeypatch deny 的干净进程中，strict reload 必须对固定 B=2 input/ChunkContext/noise/time 重现 PrefixContext/cache、velocity、absolute 7D chunk 和 normalizer call count。

## 15. 最终比较矩阵与报告纪律

| 实验 ID | state path | wrench 注入 | context refiner | 核心比较目的 | budget 规则 |
| --- | --- | --- | --- | --- | --- |
| SmolVLA-Cartesian7D | 7D | 无 | 无 | 裸 Cartesian 基线 | base |
| ForceConcat-13D | 独立 13D pre-VLM | state concat | 无 | naive state-level fusion | packer=0 参数；单独 normalizer/mask |
| PostVLM-NoForceAdapter | 7D | normalized wrench 置零，Force slot 保留 | 与指定主模型相同 | force-information 消融 | 形状/参数/计算与同名主模型相同 |
| ForceToken-Dense-Compute | 7D | post-VLM Force Token | Dense D->4D->D | active-compute 对照及 P5/P6 latency | MoE active MACs <=1% |
| ForceToken-Dense-Param | 7D | post-VLM Force Token | Dense D->h_param->D | total-parameter 对照 | MoE total params <=0.1% |
| ForceToken-MoE | 7D | post-VLM Force Token | 4-expert top-1 no-drop MoE | 主模型 | 记录 total/active FLOPs |
| ForceToken-MoE-Additive | 7D | 同 MoE | 同 MoE；仅 adapter Q 不含 S | ForceVLA-style injection control | 与 MoE 完全 parameter/init/update matched |

每行必须锁定 base checkpoint hash、resolved config hash、trainable parameter count、total parameters、active FLOPs、prefill/fusion/10-step denoise/postprocess latency、peak memory、相同 episode-disjoint train/val/test split、seed/noise/time/sampler/update budget。不得把 total-parameter-matched 与 active-compute-matched 混为同一个控制。

正式 held-out 评测固定在原始 episode-disjoint 分布；使用 paired fixed flow-noise seed，分别报告 translation、SO(3) geodesic rotation、gripper position/rate、contact/free-space/wrench/horizon 分层误差和实际 Shadow-consumed action。报告必须写明 within-session offline fine-tuning。单帧 wrench 结果只表述为 calibrated TCP wrench conditioned on measured TCP pose，不能推出跨 session 泛化、动态力反馈控制或物理因果性。

## 16. 必测项目

    test_two_camera_config_contract.py
    test_visual_fail_fast.py
    test_prefix_length_and_heterogeneous_batch.py
    test_prefix_layout_reload_contract.py
    test_prefix_cache_restore.py
    test_num_steps_source_binding.py
    test_rtc_fail_closed.py
    test_shadow_horizon_not_native_queue.py
    test_feature_dimension_masks.py
    test_action_padding_native_key.py
    test_suffix_temporal_mask_contract.py
    test_invalid_tail_noninterference.py
    test_fixed_noise_7d_contract.py
    test_flow_matching_contract_and_hook_trace.py
    test_force_cross_attention_valid_mask.py
    test_wrench_geometry_spec.py
    test_wrench_geometry_no_future_tf.py
    test_shadow_wrench_geometry_fail_close.py
    test_wrench_quality_gate.py
    test_quality_safety_rule_schema.py
    test_runtime_calibration_normalizer_compatibility_gate.py
    test_normalizer_train_only_fit.py
    test_processor_graph_sentinel.py
    test_single_normalization_ownership.py
    test_chunk_context_batch_binding.py
    test_tuple_time_anchor_and_no_future_read.py
    test_reference_grid_generation.py
    test_episode_disjoint_split.py
    test_tail_chunk_global_token_weighting.py
    test_measured_tcp_pose_causal_zoh.py
    test_pose_age_required_and_no_future.py
    test_rpy_branch_and_gripper_semantics.py
    test_force_token_concatenation_and_segments.py
    test_resolved_architecture_contract.py
    test_dense_moe_resolved_config_budget.py
    test_moe_no_drop_and_batch_invariance.py
    test_global_router_aux_two_pass_reduction.py
    test_dtype_fp32_adapter_contract.py
    test_fp32_parameter_and_optimizer_contract.py
    test_additive_parameter_match.py
    test_additive_initial_state_equality.py
    test_additive_mask_and_noise_contract.py
    test_offline_full_parameter_train_mode.py
    test_online_hil_vlm_freeze_lock.py
    test_training_stage_optimizer_boundary.py
    test_select_action_rejected.py
    test_delta_inverse_before_shadow_handoff.py
    test_reset_invalidates_chunk_context.py
    test_shadow_tick_schedule.py
    test_shadow_supersession_and_monotonic_intervals.py
    test_shadow_invalid_new_does_not_supersede.py
    test_shadow_clock_domain_and_control_age.py
    test_shadow_prediction_validity.py
    test_shadow_record_replay.py
    test_base_allowlist_load.py
    test_force_checkpoint_strict_reload.py
    test_artifact_tamper_fail_fast.py
    test_force_reload_missing_base_assets_fails.py
    test_offline_cold_start.py
    test_validation_fixture_determinism.py
    test_final_experiment_matrix_contract.py

## 17. Definition of Done 和执行顺序

在开始正式 SFT 前，以下必须为真：

- [ ] source binding、二相机 config、N_prefix_physical、num_steps=10 与 RTC fail-closed 全部验证；
- [ ] 7D feature mask、suffix temporal mask、Flow Matching 三路径 hook 和 cache parity 均通过；
- [ ] WrenchGeometrySpec 的 measured-TCP causal-ZOH/pose-age/no-future 规则、quality gate 和 runtime calibration/normalizer gate 均 fail-close；
- [ ] Dense-Compute、Dense-Param、MoE 的结构、公式、参数/计算预算与最终实验矩阵均已签名；
- [ ] Force Token concat、四类 segment id、ForceContext 投影与 adapter fp32 cast point 已测试；
- [ ] active single-pass SFT与P7 exact two-pass gate已隔离；optimizer、scheduler、seed、sampler、validation scalar已冻结；
- [ ] offline全参数train mode、online HIL VLM freeze lock和跨stage optimizer拒绝均已验证；
- [ ] ActionDelta whole-chunk inverse、原生 queue 拒绝、K 与 n_action_steps 隔离已测试；
- [ ] Shadow 的 controller-clock、actual consumed index、latest-wins、safety gate、replay 和 hold-overrun 全部通过；
- [ ] strict Force reload 与断网冷启动通过；
- [ ] 没有任何 online RL、HIL、reward、replay 或真机 action 发送路径。

推荐执行顺序：

    P0  freeze source binding, two-camera/base config, no-RTC overrides
    P1  WrenchGeometrySpec, calibration bundle, quality/safety RuleSpec
    P2  temporal conversion, split, normalizer/compatibility gates
    P3  7D masks, processor graph, ActionDelta and native suffix-mask path
    P4  PrefixLayout/cache heterogeneous parity
    P5  sign ForceToken-Dense-Compute and benchmark latency/memory
    P6  sign ForceToken-Dense-Param and ForceToken-MoE; verify budget contracts
    P7  exact two-pass gate；active ForceVLA×SmolVLA single-pass SFT recipe；then parameter-matched Additive
    P8  strict offline checkpoint/reload/cold-start
    P9  read-only Shadow timing, arbiter, safety and replay audit

任一 P0 测试未通过，不得跨入下一相关阶段；P9 未通过不得形成真机执行规格。
