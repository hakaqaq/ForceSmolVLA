# ForceSmolVLA P0–P4 实现计划

状态：`V4_1_IMPLEMENTATION_IN_PROGRESS`  
范围：P0–P4 + Cartesian7D baseline + formal v4.1 converter + smoke/parity tests  
本阶段排除：Dense/MoE、two-pass router loss、strict final checkpoint、Shadow运行、机器人动作。当前 task1的正式离线 SFT资格不再暂停；训练实际启动安排在 P0–P4和批准 gates通过后。

## 总体目录边界

计划中的最小工程布局：

```text
forcesmolvla/
  pyproject.toml
  environment.yml
  environment-manifest/        # conda explicit + pip freeze
  src/forcesmolvla/
  tests/
  artifacts/development/
  assets/                       # gitignored, only pinned upstream assets
  docs/
datasets/
  task1_forcesmolvla_v4_1/      # generated formal within-session dataset; never overwrite
```

代码只依赖固定 LeRobot public/core symbols和明确 source-bound internal hooks。不修改 `src/openpi/models/pi0_force.py`，不把 JAX/Flax 与新 PyTorch 环境混装。

## 全局 fail-closed 规则

所有 CLI 先运行统一 preflight gate。以下任一项为 false，正式 train/eval/Shadow 在 dataset stats、normalizer 或 model 调用前退出：

- `source_binding_verified`
- `environment_lock_verified`
- `base_assets_verified`
- `rulespec_approved_and_signature_verified`
- `dataset_profile == forcesmolvla-v4.1-available-sensor`
- `all_selected_episodes_quality_eligible`
- `episode_disjoint_split_verified`
- `calibration_geometry_normalizer_compatible`

P0–P4允许 development audit、正式数据转换 dry-run、smoke、single-batch overfit和接口测试。正式 converter/normalizer/train入口必须验证已批准 RuleSpec；没有隐式 fallback。单 session、缺 joint-q/FK或约 100 Hz measured TCP pose本身不触发拒绝。

## P0 — source binding、base config、two-camera、no-RTC

### 实现

1. 建立独立 `pyproject.toml` 和 Conda env `forcesmolvla`（Python 3.12），使用固定 LeRobot commit；禁止向 base或现有 forcevla env安装项目包。
2. GPU/driver/CUDA/bf16 preflight；当前 `nvidia-smi` 异常先作为硬阻塞调查。
3. 生成依赖 lock、environment manifest、wheel/git archive hash。
4. 下载固定 SmolVLA checkpoint和 SmolVLM constructor assets到 `assets/`；逐文件 SHA256验证。
5. 断网加载 base config/tokenizer/processor；任何 Hub 调用直接失败。
6. 实现并注册 `ForceSmolVLAConfig(type=force_smolvla)` 与 policy factory。
7. 两相机 config 重定：只允许 `camera1,camera2` 且顺序固定；state/action active dims=7。
8. 实现 `ForceSmolVLAPolicy`/`ForceVLAFlowMatching` 最小骨架，构造器必须实际使用后者。
9. 四重 RTC 禁用：policy `supports_rtc=false`、policy/model `_rtc_enabled=false`、`init_rtc_processor` 断言 None、inference kwargs reject。
10. 从实例解析 D_vlm/D_expert/D_action、image tokens和 physical prefix；写 development-only resolved config。
11. 对 checkpoint state dict 做 missing/unexpected allowlist；未列差异失败。

### 工件

- `source_binding.json`
- `environment_manifest.json`
- `base_asset_manifest.json`
- `resolved_force_config.json`
- `visual_language_manifest.json`
- development detached-signature placeholders（无算法/密钥假设）

### 关键测试

- `test_two_camera_config_contract.py`
- `test_visual_fail_fast.py`
- `test_num_steps_source_binding.py`
- `test_rtc_fail_closed.py`
- `test_base_allowlist_load.py`
- `test_force_reload_missing_base_assets_fails.py`
- `test_offline_cold_start.py`

### 退出条件

本地 SHA、strict offline constructor、factory registration、base allowlist、两相机 exact-set/order和 RTC tests 全绿。否则不进入依赖 P0 model semantics 的 P3/P4。

## P1 — WrenchGeometrySpec、calibration、RuleSpec

### 实现

1. 定义 JSON Schema 2020-12 的 `RuleSpec`，覆盖 v4.1 每条 rule 的全部字段。
2. 生成 `force_quality_thresholds.development.yaml` 与 `shadow_safety_thresholds.development.yaml` 候选。
3. 所有候选 rule 有显式 numeric value和 provenance，但顶层 `status=development_only`；不能冒充批准值。
4. 生成 deterministic golden fixtures：pass、threshold equality、just-fail、NaN/missing input、unit/frame/clock/hash mismatch。
5. 生成 `approval_checklist.yaml` 和签名字段待确认清单。
6. 定义 complete `WrenchGeometrySpec`、calibration artifact和 filter/resample spec schemas。
7. 写 reference wrench fixture验证 sign、gravity moment和 TCP lever arm。
8. geometry只接受约 100 Hz measured TCP pose + causal ZOH，严格要求 `t_pose<=t_wrench`；`max_pose_age_ms`必填且无默认值。每个 wrench/tuple记录 pose/raw/filter timestamp、pose age、calibration id和 validity。
9. 保留 finite、可用 device status、saturation、timestamp monotonicity、drift、filter reset/warm-up检查；缺失 device status明确标为 unavailable。

### 工件

- `rulespec.schema.json`
- `force_quality_thresholds.development.yaml`
- `shadow_safety_thresholds.development.yaml`
- `approval_checklist.yaml`
- `golden_fixtures/`
- `wrench_geometry_spec.schema.json`
- `calibration_bundle.schema.json`
- `wrench_filter_resample_spec.json`

### 关键测试

- `test_wrench_geometry_spec.py`
- `test_wrench_geometry_no_future_tf.py`
- `test_wrench_quality_gate.py`
- `test_quality_safety_rule_schema.py`
- reference fixture pass/fail tests

### 退出条件

开发 schema/fixtures 可以在未批准状态完成；正式 gate启用必须等待实验负责人批准和 detached signature机制确认。未批准不阻止 P2–P4 converter dry-run、synthetic/smoke和 parity开发，但正式 train/eval/Shadow入口必须 fail-closed。

## P2 — direct raw conversion、temporal tuples、split/normalizer gates

### 实现

1. 新 converter `tools/convert_franka_raw_to_lerobot_v3.py` 直接读取 `/home/rlc123/fr3_client_ws/datasets/task1`，不读取旧 v2.1 dataset；它只使用本工程 `src/forcesmolvla/` 内的转换实现。
2. 先实现只读 `audit` 子命令，再实现原子写入新目录；输出已存在即失败。
3. 生成固定 30 Hz session grid；每 tick记录完整 selection audit。
4. causal latest-only选择 state、wrench和两相机；禁止 future/nearest、linear、SLERP和跨 episode filter state。
5. camera order固定，RGB/serial/role逐 record 验证。
6. gripper state/action保留 raw `width_m` / `target_gripper_width_m`，controller position只作 audit；做单位/范围/无隐式转换 fixture。
7. absolute ack target形成 H=50 labels；尾部 mask至少有 K=3 才保留 chunk。
8. 每个 raw wrench以 measured TCP pose causal ZOH构造 sensor姿态，再执行 sign、bias、payload gravity、rotation和 TCP moment shift；geometry-invalid sample不更新 filter state。30 Hz tuple只取最新 causal filtered sample。
9. LeRobot v3.0输出使用两相机 + state7 + wrench6 + absolute action7；保留 provenance sidecars。
10. 逐 episode执行已批准数据质量 gate；只排除失败 episodes，并保留原因。
11. 对 eligible episodes做 deterministic episode-disjoint train/val/test split；同一 episode不得跨 split。
12. normalizer API只对 eligible train episodes的 raw tuples开放，并绑定 split/calibration/geometry hashes；val/test调用 fit/update必失败。

### 工件

- `/home/rlc123/ForceSmolVLA/datasets/task1_forcesmolvla_v4_1`
- `conversion_manifest.json`
- `source_files.sha256.jsonl`
- `tuple_audit.parquet`
- `exclusions.jsonl`
- `split_manifest.json`（episode-disjoint）
- `normalizer_fit_manifest.json`（train episodes only）

### 关键测试

- `test_tuple_time_anchor_and_no_future_read.py`
- `test_reference_grid_generation.py`
- `test_episode_disjoint_split.py`
- `test_tail_chunk_global_token_weighting.py`
- `test_normalizer_train_only_fit.py`
- `test_runtime_calibration_normalizer_compatibility_gate.py`
- `test_measured_tcp_pose_causal_zoh.py`
- `test_pose_age_required_and_no_future.py`
- `test_rpy_branch_and_gripper_semantics.py`

### smoke 范围

- 全 raw dataset只读 audit。
- 少量 episode转换与 deterministic replay。
- eligible train 单 batch加载、固定 batch hash、single-batch overfit。
- val/test episode不得影响 normalizer stats。

## P3 — Cartesian7D baseline、masks、processor graph、ActionDelta

### 实现

1. 先实现 `SmolVLA-Cartesian7D`，不加 Force Token/Dense/MoE。
2. state7/action7在 custom processor中各自 normalize一次，再 pad 到 32D。
3. inherited STATE/ACTION normalizer和 Aloha transforms必须 Identity/removed/disconnected。
4. 显式构造 state/action/action-valid/flow/suffix masks。
5. noise、flow interpolation、attention content、Euler update和 loss只允许前 7D active features。
6. `ActionDeltaProcessor` 对 raw absolute labels做唯一 delta变换；whole-chunk inverse绑定 raw state snapshot。
7. `ChunkContext` 全字段 batch-bound，reset后失效。
8. native suffix temporal mask必须显式支持 invalid tail，且 loss按全局 valid feature tokens加权。
9. 写 processor call graph/hash/key/shape/dtype/normalizer owner/call count manifest。

### 工件

- `processor_graph_manifest.json`
- `action_delta_spec.json`
- `feature_mask_spec.json`
- `flow_matching_spec.json`
- Cartesian7D smoke checkpoint（不用于正式 checkpoint selection）

### 关键测试

- `test_feature_dimension_masks.py`
- `test_action_padding_native_key.py`
- `test_suffix_temporal_mask_contract.py`
- `test_invalid_tail_noninterference.py`
- `test_fixed_noise_7d_contract.py`
- `test_single_normalization_ownership.py`
- `test_processor_graph_sentinel.py`
- `test_chunk_context_batch_binding.py`
- `test_delta_inverse_before_shadow_handoff.py`
- `test_reset_invalidates_chunk_context.py`

### 退出条件

padding mutation不得改变前 7D velocity；无效 horizon不得改变有效 horizon loss/gradient；normalizer调用次数严格为一次；Cartesian7D能完成 CPU synthetic 和 GPU单 batch forward/backward。

## P4 — PrefixLayout、cache、heterogeneous batch parity

### 实现

1. `encode_prefix` 返回 immutable `PrefixContext`：contextual prefix output、physical/valid masks、segment ids、PrefixLayout和prefix cache。
2. 固定 camera1/camera2/language/state physical spans；不做 ragged compaction。
3. full forward显式 `use_cache=false`；cached inference显式 `use_cache=true`。
4. suffix position、attention mask和cache crop分别使用其正确的 physical/valid语义。
5. 每个 Euler step后验证 cache physical length和 K/V内容恢复到 prefill值。
6. B>=2 heterogeneous language-valid-length batch与逐样本 reference做固定 noise/time parity。
7. 两相机 missing/extra/reordered、错误 prefix length和 cache mutation均 fail-fast。

### 工件

- `prefix_layout_spec.json`
- `prefix_cache_contract.json`
- parity report（CPU/GPU、dtype、tolerance、max error）

### 关键测试

- `test_prefix_length_and_heterogeneous_batch.py`
- `test_prefix_layout_reload_contract.py`
- `test_prefix_cache_restore.py`
- `test_two_camera_config_contract.py`
- `test_visual_fail_fast.py`

### 退出条件

同一 physical prefix、不同语言 valid length 的 B>=2 batch中，full-prefix、prefill和cached velocity在冻结容差内一致；每一步 cache长度/内容不变；reload后 PrefixLayout完全一致。

## P0–P4 完成后的停点

完成 P4 后只交付：

- 可断网构造的 Cartesian7D baseline。
- direct raw-to-v3 v4.1 converter及其审计产物。
- 两相机、mask、suffix、cache和single-batch smoke结果。
- P0/P1 工件的批准/签名待办。

此时仍不实现 Dense/MoE、two-pass router loss、strict final checkpoint或 Shadow，也不启动正式训练。进入 P5 前先提交 P0–P4 测试报告、GPU memory preflight和所有未批准字段清单。
