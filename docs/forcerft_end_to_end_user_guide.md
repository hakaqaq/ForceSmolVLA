# ForceSmolVLA / ForceRFT 端到端用户手册

本文只描述最终固定 pipeline：

```text
ForceSmolVLA SFT
→ 人工奖励标注
→ 奖励分类器训练
→ 离线 reward/terminal 物化
→ 离线 Twin-Q Critic warmup
→ Frozen-VLM Actor/Critic 离线联合训练
→ 完整 offline exact-resume checkpoint
→ HIL 在线采集
→ Online-R 达到 100 条
→ Actor/Learner 异步持续联合训练
```

## 1. 目录与环境

```bash
export FORCESMOLVLA_ROOT=/home/rlc123/ForceSmolVLA
export FR3_WS=/home/rlc123/fr3_client_ws
export TASK_ID=task2
export TASK_OUTPUT_ROOT="$FORCESMOLVLA_ROOT/outputs/$TASK_ID"
export RAW_ROOT="$FR3_WS/datasets/task2"
export LEROBOT_DATASET="$FORCESMOLVLA_ROOT/datasets/task2_lerobotv3"
export OFFLINE_REPLAY="$FORCESMOLVLA_ROOT/artifacts/development/stage2/g1_frozen_detector_transition_view.v1"
export MODEL_PYTHON=/home/rlc123/anaconda3/envs/forcesmolvla/bin/python
export ROBOT_PYTHON="$FR3_WS/.venv/bin/python"
# 只在用户明确部署某个完整 checkpoint 后设置；本 pipeline 不提供默认部署 profile。
export DEPLOYMENT_PROFILE=/absolute/path/to/explicitly_deployed_profile.json
```

训练机使用 Conda/PyTorch/CUDA；机器人侧使用 ROS 2 Humble。硬件链为 FR3、HEX-E、Robotiq、D435、D405 和 SpaceMouse。机器人运行前加载：

```bash
source /opt/ros/humble/setup.bash
source "$FR3_WS/install/setup.bash"
source "$FR3_WS/.venv/bin/activate"
export ROS_DOMAIN_ID=30 ROS_LOCALHOST_ONLY=0
```

目录规则固定为：`datasets/{task_id}` 只放训练数据，`outputs/{task_id}` 放该任务全部训练产物，`configs/` 只放配置。训练 CLI 均接受 `--task-id` 与 `--output-root`；省略 `--output-root` 时使用 `outputs/{task_id}`。

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
  --repo-id task2_lerobotv3
```

转换结果必须包含双相机、state7、wrench6、action7、episode/frame 索引、split、conversion manifest 和 normalizer manifest。normalization 在训练输入处只应用一次；已归一化数据不得再次归一化。

## 4. ForceSmolVLA 全量 SFT

```bash
export PYTHONHASHSEED=42 CUBLAS_WORKSPACE_CONFIG=:4096:8
"$MODEL_PYTHON" tools/train_forcesmolvla_sft.py \
  --dataset "$LEROBOT_DATASET" \
  --config configs/train/task2.json \
  --task-id "$TASK_ID" \
  --output-root "$TASK_OUTPUT_ROOT"
```

最终 SFT checkpoint：

```text
outputs/task2/sft/checkpoints/forcesmolvla_sft_step_010000
```

它包含 Actor、optimizer/scheduler、RNG、sampler 和运行 manifests。需要续训时使用该 CLI 的 `--resume`；不要只加载 `model.safetensors` 后重建 optimizer。

## 5. 奖励人工标注与分类器

先生成 review bundle，再写入人工 frame labels：

```bash
"$MODEL_PYTHON" tools/reward_classifier/build_task2_review_bundle.py --help
"$MODEL_PYTHON" tools/reward_classifier/annotate_reward_frames.py --help
```

构造分类器 cache 并训练：

```bash
"$MODEL_PYTHON" tools/reward_classifier/train_reward_classifier.py \
  --task-id "$TASK_ID" --output-root "$TASK_OUTPUT_ROOT" \
  prepare-cache --cache-dir /absolute/path/to/reward_cache

"$MODEL_PYTHON" tools/reward_classifier/train_reward_classifier.py \
  --task-id "$TASK_ID" --output-root "$TASK_OUTPUT_ROOT" \
  train --cache-dir /absolute/path/to/reward_cache
```

production detector checkpoint 固定在：

```text
outputs/task2/reward_classifier/checkpoints/best/best_checkpoint.msgpack
```

## 6. 离线 reward/terminal 物化

```bash
"$MODEL_PYTHON" tools/materialize_reward_transitions.py \
  --config configs/reward_transition_materialization.development.json \
  --output-root "$OFFLINE_REPLAY"
```

物化必须保持 episode/frame/transition 对齐，输出 reward、terminal、observation/next-observation 引用及 detector provenance。它不是在线 admission，不写 online WAL/outbox。

## 7. 离线 Twin-Q Critic warmup

```bash
"$MODEL_PYTHON" tools/train_twin_q_critic.py \
  --run --task-id "$TASK_ID" --output-root "$TASK_OUTPUT_ROOT"
```

输出：

```text
outputs/task2/offline/checkpoints/offline_twin_q_critic_warmup_step_000256
```

该 checkpoint 保存 Q1/Q2、target Q1/Q2、Critic optimizer/scheduler、RNG、sampler 与 counter。此阶段不更新 Actor。

## 8. Frozen-VLM Actor/Critic 离线联合训练

```bash
"$MODEL_PYTHON" tools/train_forcerft_actor_critic.py \
  --task-id "$TASK_ID" \
  --output-root "$TASK_OUTPUT_ROOT" \
  --offline-joint-cycles 210
```

输入是 SFT Actor、离线 Twin-Q Critic 和现有 demonstration replay。固定每 cycle 2 次 Critic、2 次 Polyak、1 次 Actor；只用 demo；VLM 冻结；FM 只来自 expert；Actor 使用 min Twin-Q guidance；TCP6 接收 Q-gradient，gripper Q-gradient 截止。不调用 Cal-QL/CQL/random candidate/online MC return。

输出完整 exact-resume checkpoint：

```text
outputs/task2/offline/checkpoints/offline_actor_critic_cycle_000210
```

它同时包含 Actor、Q1/Q2、targets、两个 optimizer、scheduler、RNG、sampler、cycle counter、normalizer、action contract 和 offline replay reference。Unified inference Actor 与 Learner 必须从这里读取同一 Actor 参数；Actor-only export 不能作为 Learner resume parent。

## 9. HIL 与 online replay

`tools/serve_forcerft_actor_learner.py` 是唯一 GPU owner；`tools/run_forcerft_integrated_capture.py` 是唯一机器人控制链。控制仍保持 H50、10 Hz、low-watermark inference、takeover generation、stale result rejection、Pose ACK 和 gripper authority。

episode seal 后，操作者输入 success/failure。success episode 通过 production bridge 的同一次 admission 调用物化 reward/terminal 并 append 到：

```text
outputs/task2/online/replay
```

不再先 dry-run 后重复 admission。human override、旧 generation proposal、缺失 action7 authority 或 warmup transition 不进入 replay；same UID 幂等。当前仍在采集的 episode 永远不能被 Learner 采样。

## 10. 持续在线 Actor/Learner

```bash
"$MODEL_PYTHON" tools/run_forcerft_online_loop.py \
  --task-id "$TASK_ID" \
  --output-root "$TASK_OUTPUT_ROOT" \
  --max-episodes 100 \
  --root-prefix "$FR3_WS/datasets/task2_forcerft_online" \
  --task "Pick up the purple ring and place it onto the red peg." \
  --episode-time 120 \
  --tool-profile onrobot_robotiq \
  --policy-replan-steps 8 \
  --policy-queue-low-watermark 7 \
  --max-force-n 25 \
  --max-torque-nm 2 \
  --deployment-profile "$DEPLOYMENT_PROFILE"
```

Unified server只加载一次 offline exact-resume checkpoint，并跨 episode 常驻。Online-R 少于 100 条时只采集；达到 100 后 Learner 连续运行。batch 为 50% demonstration + 50% sealed online replay；每 learner cycle 为 2 Critic + 2 Polyak + 1 Actor。

server 本身不加载 registry 的 active/previous rollback Actor，也不需要 deployment binding；它始终从 `outputs/task2/offline/checkpoints/offline_actor_critic_cycle_000210/actor` 读取推理 Actor，并由同一完整 checkpoint 恢复 Learner。机器人 policy-execute 只在用户明确部署某个完整 checkpoint 后才传入对应 deployment profile；本轮不会创建 actor export、candidate、binding 或 approval 文件。

每 5 cycles 只在内存广播 Actor，新参数从下一次 inference request 的新 H50 chunk 生效；同步点不写 checkpoint、不导出 package、不做 candidate validation。每 50 cycles 保存完整 checkpoint：

```text
outputs/task2/online/checkpoints/online_actor_critic_cycle_000050
outputs/task2/online/checkpoints/online_actor_critic_cycle_000100
```

只保留最新两个 checkpoint。online cycle 从在线 optimizer 首次启动时的 0 开始，与 episode 编号无关。

正常停止使用 recorder 的 `q`。系统停止新 learner cycle，等待正在进行的 optimizer step 完成，并保存最后完成 cycle；若该 cycle 恰好是 50 的倍数，只保存一次。异常失败时不修改原始 episode，不把未封口 episode 加入 replay。

例如 cycle45 退出保存 cycle45；cycle55 保留 cycle50 与 cycle55；cycle107 保留 cycle100 与 cycle107。每 5-cycle 广播不写磁盘，也不导出 Actor package。

Actor export 只在用户以后明确要求部署某个完整 checkpoint 时，才写入 `outputs/task2/exports/actor`；它不是本 pipeline 的自动阶段。

## 11. 保留与故障处理

必须保留：原始 D 数据和 LeRobot v3、SFT、reward classifier、materialized demo replay、offline Critic、offline exact-resume、最新两个 online exact-resume、formal replay/WAL/outbox/admission 引用的所有 raw episode、registry 和 active/previous rollback export。

常见 fail-closed 原因：deployment binding 不匹配、原始 JPEG 缺失、takeover 后旧 result、gripper origin 不完整、ACK 缺失、checkpoint/replay UID 或 credit 不一致。不得用其他 episode 图片、虚假 command ID/ACK、重绑旧 generation 或修改原始 episode绕过。
