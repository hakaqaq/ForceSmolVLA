# ForceSmolVLA / ForceRFT 端到端用户手册

本文只描述最终固定 pipeline：

```text
ForceSmolVLA SFT
→ 人工奖励标注
→ 奖励分类器训练
→ 构建 frozen base policy + zero wrist-wrench residual actor + random Twin-Q bootstrap
→ 收集真实 sealed ACK replay
→ 达到 100 条合法 ACK 后执行 256-step ACK Critic warm-up
→ 每 cycle 进行 2 Twin-Q + 1 wrist-wrench residual Actor 在线训练
```

## 1. 目录与环境

```bash
export FORCESMOLVLA_ROOT=/home/rlc123/ForceSmolVLA
export FR3_WS=/home/rlc123/fr3_client_ws
export TASK_ID=task2
export TASK_OUTPUT_ROOT="$FORCESMOLVLA_ROOT/outputs/$TASK_ID"
export ONLINE_CAPTURE_ROOT="$FORCESMOLVLA_ROOT/datasets/${TASK_ID}_forcerft_online"
export RAW_ROOT="$FR3_WS/datasets/$TASK_ID"
export LEROBOT_DATASET="$FORCESMOLVLA_ROOT/datasets/${TASK_ID}_lerobotv3"
export MODEL_PYTHON=/home/rlc123/anaconda3/envs/forcesmolvla/bin/python
export ROBOT_PYTHON="$FR3_WS/.venv/bin/python"
```

训练机使用 Conda/PyTorch/CUDA；机器人侧使用 ROS 2 Humble。硬件链为 FR3、HEX-E、Robotiq、D435、D405 和 SpaceMouse。机器人运行前加载：

```bash
source /opt/ros/humble/setup.bash
source "$FR3_WS/install/setup.bash"
source "$FR3_WS/.venv/bin/activate"
export ROS_DOMAIN_ID=30 ROS_LOCALHOST_ONLY=0
```

目录规则固定为：原始转换数据使用 `datasets/{task_id}_lerobotv3`，训练产物使用 `outputs/{task_id}`，任务配置使用 `configs/forcerft/tasks/{task_id}.yaml`。CLI 均接受 `--task-id` 和显式路径覆盖参数。

## 2. 原生数据采集

原生采集入口来自机器人工作区，不属于本仓库：

```bash
"$ROBOT_PYTHON" "$FR3_WS/scripts/record_franka_hilserl_impedance.py" \
  --root "$RAW_ROOT" \
  --task "Pick up the purple ring and place it onto the red peg." \
  --episodes 10 \
  --episode-time 120 \
  --tool-profile onrobot_robotiq
```

录制界面中 `Enter` 保存、`d` 后回车丢弃、`q` 后回车停止。只有完整性检查通过的 episode 进入 `episodes/`；失败原始数据进入 `rejected_episodes/`，不能伪造 ACK、gripper goal 或图片来绕过检查。

每个 accepted episode 保留双相机 JPEG、state7、action7、calibrated TCP wrench6、accepted reference、Pose ACK、gripper target/status/state、controller state 和 `episode_result.json`。正式 online replay 通过绝对路径读取这些文件，因此被 replay 引用的原始 episode 不得移动或删除。

## 3. 转换为 LeRobot v3

```bash
"$MODEL_PYTHON" tools/convert_franka_raw_to_lerobot_v3.py \
  --input-root "$RAW_ROOT" \
  --output-root "$LEROBOT_DATASET" \
  --repo-id "${TASK_ID}_lerobotv3"
```

转换结果必须包含双相机、state7、wrench6、action7、episode/frame 索引、split、conversion manifest 和 normalizer manifest。normalization 在训练输入处只应用一次；已归一化数据不得再次归一化。

## 4. ForceSmolVLA 全量 SFT

```bash
export PYTHONHASHSEED=42 CUBLAS_WORKSPACE_CONFIG=:4096:8
"$MODEL_PYTHON" tools/train_forcesmolvla_sft.py \
  --dataset "$LEROBOT_DATASET" \
  --config "configs/train/${TASK_ID}.json" \
  --task-id "$TASK_ID" \
  --output-root "$TASK_OUTPUT_ROOT"
```

最终 SFT checkpoint：

```text
outputs/{task_id}/sft/checkpoints/forcesmolvla_sft_step_010000
```

它包含 Actor、optimizer/scheduler、RNG、sampler 和运行 manifests。需要续训时使用该 CLI 的 `--resume`；不要只加载 `model.safetensors` 后重建 optimizer。

## 5. 奖励人工标注与分类器

先选取人工标注 episode 子集，再生成 review bundle 和 frame labels。无需标注采集到的全部 episode；首版可从约 20 条开始，但训练与验证 episode 必须互斥，并覆盖 positive、ordinary negative 和 hard negative。未列入 reviewed labels 的 episode 不进入奖励分类器训练/验证，奖励器冻结后仍会对完整 LeRobot 数据集自动打分。

```bash
"$MODEL_PYTHON" tools/reward_classifier/label_reward_frames.py \
  --task-id "$TASK_ID" \
  --dataset-root "$LEROBOT_DATASET" \
  --train-episodes 16 \
  --val-episodes 4
```

构造分类器 cache 并训练：

```bash
"$MODEL_PYTHON" tools/reward_classifier/train_reward_classifier.py \
  --task-id "$TASK_ID" --output-root "$TASK_OUTPUT_ROOT" \
  --dataset-root "$LEROBOT_DATASET" \
  --reviewed-labels "$FORCESMOLVLA_ROOT/labels/${TASK_ID}_reward_frame_labels.json" \
  prepare-cache --cache-dir /absolute/path/to/reward_cache

"$MODEL_PYTHON" tools/reward_classifier/train_reward_classifier.py \
  --task-id "$TASK_ID" --output-root "$TASK_OUTPUT_ROOT" \
  --dataset-root "$LEROBOT_DATASET" \
  --reviewed-labels "$FORCESMOLVLA_ROOT/labels/${TASK_ID}_reward_frame_labels.json" \
  train --cache-dir /absolute/path/to/reward_cache
```

production detector checkpoint 固定在：

```text
outputs/{task_id}/reward_classifier/checkpoints/best/best_checkpoint.msgpack
```

## 6. 历史离线 reward/terminal 物化（不属于当前生产训练链）

```bash
"$MODEL_PYTHON" tools/materialize_reward_transitions.py \
  build --task-id "$TASK_ID" \
  --config "configs/tasks/$TASK_ID/forcerft_offline_reward_transitions.json" \
  --dataset-root "$LEROBOT_DATASET" \
  --reward-transition-root \
    "$FORCESMOLVLA_ROOT/datasets/${TASK_ID}_forcerft_offline_reward_transitions"
```

该产物只保留给旧方法实验对照，不用于 ACK-aligned residual Twin-Q、Residual Actor、Actor-Q 更新或 online replay 混合。当前生产训练链不执行本节命令。

## 7. 构建 online ACK-residual bootstrap checkpoint

```bash
"$MODEL_PYTHON" tools/build_forcerft_online_residual_bootstrap.py \
  --task-id "$TASK_ID" --output-root "$TASK_OUTPUT_ROOT" \
  --dataset-root "$LEROBOT_DATASET" \
  --frozen-base-policy-checkpoint \
    "outputs/$TASK_ID/sft/checkpoints/forcesmolvla_sft_step_010000"
```

输出：

```text
outputs/{task_id}/online_ack_residual/bootstrap_checkpoints/base_policy_zero_residual_random_twin_q
```

bootstrap checkpoint 保存 frozen base policy 的路径、严格零输出 wrist-wrench residual Actor、随机 ACK-aligned residual Twin-Q、targets、两个 optimizer 与运行计数；不读取 demonstration 或旧 offline Critic checkpoint。

## 8. 真实 ACK Critic warm-up 与 Residual Actor–Critic 训练

Learner 状态依次为 `ack_replay_collection → ack_critic_warmup → residual_actor_critic_training`。前 100 条正式 sealed ACK transition 期间 Actor/Twin-Q 均不更新；达到阈值后在同一进程执行一次 256-step Twin-Q warm-up，然后每 cycle 固定执行 `2 Twin-Q + 1 wrist-wrench residual Actor`。训练只读取低维 state/wrench/base/residual/ACK 数据，不运行第二份 base policy、Flow sampler 或图像 Critic。

## 9. HIL 与 online replay

`tools/serve_forcerft_residual_actor_critic.py` 是唯一 GPU owner；`tools/run_forcerft_integrated_capture.py` 是唯一机器人控制链。控制仍保持 H50、10 Hz、low-watermark inference、takeover generation、stale result rejection、Pose ACK 和 gripper authority。
若 inference 期间 wrench causal filter 因源间隙重置并切换 generation，旧 request/result 和未执行 chunk 会被作废；等待现有 250-sample warmup 完成后，同一 episode 使用 fresh observation 重新 inference，恢复等待期间不生成 transition。

episode seal 后，操作者输入 success/failure。技术记录完整的 success 与 failure episode 都通过 production bridge 的同一次 admission 调用物化 TD transition，并 append 到：

```text
outputs/{task_id}/online/replay
```

不再先 dry-run 后重复 admission。success 的 detector terminal 保持 `reward=1.0, terminated=true, bootstrap_mask=false, discount=0.0`；failure 使用 sealed episode 最后一个有效 Critic transition 作为零奖励 terminal，且同样不 bootstrap。operator failure 与 frozen detector success trigger 冲突时整条 episode 不进入训练。

task2 封口物化在 30 Hz 因果网格上允许双相机样本年龄不超过 `100 ms`，以覆盖正常调度抖动和偶发丢帧；双相机 skew 仍不得超过 `33 ms`，样本年龄超过 `100 ms` 仍拒绝进入 replay。历史 checkpoint 内保存的 provenance 不重写。

所有 `critic_td_valid=true` 的正式 sealed online ACK transition 进入同一个低维 Critic replay，包括 autonomous policy ACK、accepted human correction、success、failure 和 intervention-truncated boundary。旧 transition 缺少 base/residual 字段时按 `base=accepted behavior, residual=0` 兼容，不迁移 replay 目录。

只有带可靠 pre-takeover base action 的 human row 才提供 residual Actor 监督目标；缺少该基线时仅令 `human_residual_valid=false`，不拒绝 episode。人工姿态差继续使用 RPY delta。接管开始时清空旧 chunk/pending request/旧 observation、接管期间暂停 policy dispatch、释放后 fresh observation + fresh inference 的控制语义保持不变。

## 10. 持续在线 Actor/Learner

```bash
"$MODEL_PYTHON" tools/run_forcerft_online_loop.py \
  --task-id "$TASK_ID" \
  --output-root "$TASK_OUTPUT_ROOT" \
  --dataset-root "$LEROBOT_DATASET" \
  --max-episodes 100 \
  --capture-output-root "$ONLINE_CAPTURE_ROOT" \
  --ack-replay-root "$TASK_OUTPUT_ROOT/online" \
  --online-residual-bootstrap-checkpoint \
    "$TASK_OUTPUT_ROOT/online_ack_residual/bootstrap_checkpoints/base_policy_zero_residual_random_twin_q" \
  --task "Pick up the purple ring and place it onto the red peg." \
  --episode-time 120 \
  --tool-profile onrobot_robotiq \
  --policy-replan-steps 8 \
  --policy-queue-low-watermark 7 \
  --max-force-n 25 \
  --max-torque-nm 2 \
  --allow-development-policy-execution-smoke
```

在线 native episode 固定保存在 ForceSmolVLA 数据目录下，例如第一个 session 为
`/home/rlc123/ForceSmolVLA/datasets/{task_id}_forcerft_online_001`；不写入
`/home/rlc123/fr3_client_ws/datasets`。省略 `--capture-output-root` 时也使用这一仓库内默认目录。

Unified server 每次启动恢复一个 residual Actor/Twin-Q checkpoint，并跨 episode 常驻。`minimum_ack_transitions=100` 统计 success 与 failure episode 中全部正式 `critic_td_valid` ACK（policy 与 human）；达到阈值后一次性完成 256 个 Twin-Q optimizer step，再自动进入 `residual_actor_critic_training`。每 cycle 固定为 2 Twin-Q + 2 target Polyak + 1 residual Actor，不读取 demonstration、图像或 Flow/SFT reference。
因此 async capture manifest 中 `learner_started=false`（未达 100 条）和 `learner_started=true`（已达 100 条）都是合法状态；两种情况下 `current_episode_sampled_by_learner` 都必须为 `false`。

canonical online loop 在每个 episode 后只打印两行 capture/learner 摘要和一行 admission 摘要；完整 contract、stream quality 与 episode seal 继续保存在 session 文件中，不在终端重复展开。

启动时先选择 `outputs/{task_id}/online_ack_residual/training_checkpoints/` 中 cycle 最大且结构完整的 exact-resume checkpoint；没有可恢复 checkpoint 时只接受显式 `--online-residual-bootstrap-checkpoint`。模型结构、optimizer、loss、residual cap、batch 与调度全部以 checkpoint 的 `state/config.yaml` 为唯一权威；仓库当前公共 YAML 只用于算法字段一致性校验，不一致时以 `FORCERFT_EXACT_RESUME_CONFIG_MISMATCH` 停止。需要改变算法配置时必须建立新的 adaptation lineage，不能称为 exact-resume。frozen base policy 始终从 checkpoint 的 `frozen_base_policy_checkpoint` 加载，整个 online session 不变。

服务开放首个新 episode 前会扫描全部 sealed admissions，重建应得 cycle 总预算并与 checkpoint 已完成 cycle 比较。若存在历史欠账，`recovery_budget_drain_required=true`，canonical online loop 会先调用 `/runtime/drain-outstanding-budget`；欠账排空或显式失败前，`prepare-episode` 一律拒绝。这样即使进程在 formal admission 与正常 drain 之间退出，也不会跳过旧 episode 的训练预算。

`--allow-development-policy-execution-smoke` 是已有的显式机器人执行开关；它不选择模型，也不触发 publication、activation、candidate、profile 或 binding 流程。力限、takeover generation、stale-result rejection、ACK 和 recorder 单控制链保持不变。

在线推理只对反归一化后的 gripper candidate 做有限值饱和：低于 `-0.01 m` 按闭合端处理，高于 `0.095 m` 按打开端处理，二值判定阈值保持 `0.0425 m`，随后只输出精确的 `0.0 m` 或 `0.085 m`。`NaN/Inf` 继续拒绝；TCP6、力限和 action normalizer 不做裁剪或改写。

每累计 10 个真实 residual Actor optimizer step 生成只含 `residual_actor.pt` 的 lineage-isolated candidate，并只在下一 episode boundary 生效。每 20 residual Actor–Critic cycles 保存 exact-resume checkpoint；Critic warm-up 完成、candidate 实际激活以及 graceful exit 时也立即保存：

```text
outputs/{task_id}/online_ack_residual/training_checkpoints/residual_actor_critic_cycle_000020
outputs/{task_id}/online_ack_residual/training_checkpoints/residual_actor_critic_cycle_000040
```

只保留最新十个 checkpoint。每个达到启动阈值后的新 admission，其 residual Actor–Critic cycle 预算为 `min(10, max(1, ceil(new_critic_td_valid_rows / 64)))`；256-step Critic warm-up 不消耗该预算。采用固定的 warmup-only 边界语义：阈值前 admissions 只为 Critic warm-up 提供数据，不追溯产生 joint-cycle debt；首次跨过阈值的 admission 只按自身合法行数获得预算。

正常停止使用 recorder 的 `q`。系统停止新 learner cycle，等待正在进行的 optimizer step 完成，并保存最后完成 cycle；若该 cycle 恰好是 20 的倍数，只保存一次。Learner 异常失败时不保存可能只完成部分 optimizer step 的 checkpoint，也不修改原始 episode或把未封口 episode 加入 replay。
采集途中若因控制器、通信或进程错误退出，canonical online-loop 会自动删除本次未封口 session root 及 `.inprogress` 内容，不保留半条 episode。已存在 technical seal 的 session 不自动删除，即使后续 admission 失败，也保留供修复后重试。

例如 cycle45 退出保存 cycle45；cycle55 会保留最近的周期/事件 checkpoint，最多十个。candidate 不复制完整 ForceSmolVLA。

## 11. 保留与故障处理

必须保留：原始 D 数据和 LeRobot v3、SFT、reward classifier、materialized demo replay、历史 offline Critic 实验记录、最新十个 online exact-resume，以及 formal replay/WAL/outbox/admission 引用的所有 raw episode。

常见 fail-closed 原因：exact-resume checkpoint 不完整、推理 Actor 与 Learner checkpoint 不同源、原始 JPEG 缺失、takeover 后旧 result、gripper origin 不完整、ACK 缺失、checkpoint/replay UID 或 credit 不一致。不得用其他 episode 图片、虚假 command ID/ACK、重绑旧 generation 或修改原始 episode绕过。
