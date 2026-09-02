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
export ONLINE_CAPTURE_PREFIX="$FORCESMOLVLA_ROOT/datasets/${TASK_ID}_forcerft_online"
export RAW_ROOT="$FR3_WS/datasets/task2"
export LEROBOT_DATASET="$FORCESMOLVLA_ROOT/datasets/task2_lerobotv3"
export OFFLINE_REPLAY="$FORCESMOLVLA_ROOT/artifacts/development/stage2/g1_frozen_detector_transition_view.v1"
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
若 inference 期间 wrench causal filter 因源间隙重置并切换 generation，旧 request/result 和未执行 chunk 会被作废；等待现有 250-sample warmup 完成后，同一 episode 使用 fresh observation 重新 inference，恢复等待期间不生成 transition。

episode seal 后，操作者输入 success/failure。技术记录完整的 success 与 failure episode 都通过 production bridge 的同一次 admission 调用物化 TD transition，并 append 到：

```text
outputs/task2/online/replay
```

不再先 dry-run 后重复 admission。success 的 detector terminal 保持 `reward=1.0, terminated=true, bootstrap_mask=false, discount=0.0`；failure 使用 sealed episode 最后一个有效 Critic transition 作为零奖励 terminal，且同样不 bootstrap。operator failure 与 frozen detector success trigger 冲突时整条 episode 不进入训练。封口时仍只形成两个 TD 逻辑池：

task2 封口物化在 30 Hz 因果网格上允许双相机样本年龄不超过 `100 ms`，以覆盖正常调度抖动和偶发丢帧；双相机 skew 仍不得超过 `33 ms`，样本年龄超过 `100 ms` 仍拒绝进入 replay。历史 checkpoint 内保存的 provenance 不重写。

```text
D_exp_TD = offline demonstrations + technically complete online human transitions from success/failure episodes
R_pol_TD = technically complete autonomous policy transitions from success/failure episodes
D_exp_FM = offline demonstrations + accepted human corrections from confirmed-success episodes
```

两类在线 transition 物理上仍只写入 `outputs/task2/online/replay`，通过独立的 `td_eligible`、`fm_eligible` 以及 `action_source=human|policy` 分区。自主动作从不作为 expert target；失败 episode 的人工纠正只进入 TD，不进入 FM。人工纠正不转换为 LeRobot v3，也不复制图像；observation 继续引用原始 native episode。

只有真实发布且由 Pose/gripper ACK 确认的人工命令才会物化为 human transition，其中只有 confirmed-success episode 的纠正是 expert/FM target。它保存实际执行的 post-adapter absolute7、takeover generation 和真实执行 provenance，不伪造 policy request/result/chunk/proposal/ACK。missing/rejected ACK、未执行输入、无法组成 observation/action/next-observation 的残缺记录，以及接管时作废的旧 policy chunk 后缀都不进入任何训练池；same UID 幂等。当前仍在采集、尚未封口的 episode 永远不能被 Learner 采样。

人工纠正按同一 30 Hz action grid 因果投影，并使用与 offline demonstration 相同的 TCP6 delta + absolute gripper adapter 和冻结 normalizer。H50 中以 `human_action_target[H,7]` 和 `human_action_valid_mask[H,7]` 只监督 confirmed-success episode 实际执行的人工槽，mask 为 false 的槽不参与 Flow Matching。自主 policy transition 只参与 Critic TD、Actor Q-guidance 和弱 TCP6 command-space behavior anchor，不产生 FM 梯度；anchor 直接复用 replay 中 ACK-authoritative behavior action，不约束 gripper。接管开始时清空旧 chunk/pending request/旧 observation、接管期间暂停 policy dispatch、释放后 fresh observation + fresh inference 的控制语义保持不变。

## 10. 持续在线 Actor/Learner

```bash
"$MODEL_PYTHON" tools/run_forcerft_online_loop.py \
  --task-id "$TASK_ID" \
  --output-root "$TASK_OUTPUT_ROOT" \
  --max-episodes 100 \
  --root-prefix "$ONLINE_CAPTURE_PREFIX" \
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
`/home/rlc123/ForceSmolVLA/datasets/task2_forcerft_online_001`；不写入
`/home/rlc123/fr3_client_ws/datasets`。省略 `--root-prefix` 时也使用这一仓库内默认前缀。

Unified server 每次启动只选择并加载一次完整 exact-resume checkpoint，并跨 episode 常驻。`training_starts=100` 统计 success 与 failure episode 中所有已接收的 autonomous policy TD transitions；人工纠正不会提前启动 Learner，也不铸造 policy update credit。达到 100 后 Learner 连续运行。每个 batch 固定为 50% D_exp_TD + 50% R_pol_TD，不增加第三个 replay pool；两半都进入 Critic TD 和 Actor Q-guidance，只有 `D_exp_FM` 进入 FM，ordinary autonomous rows 额外使用权重 `0.1` 的弱 TCP6 behavior anchor。每 learner cycle 为 2 Critic + 2 Polyak + 1 Actor。
因此 async capture manifest 中 `learner_started=false`（未达 100 条）和 `learner_started=true`（已达 100 条）都是合法状态；两种情况下 `current_episode_sampled_by_learner` 都必须为 `false`。

canonical online loop 在每个 episode 后只打印两行 capture/learner 摘要和一行 admission 摘要；完整 contract、stream quality 与 episode seal 继续保存在 session 文件中，不在终端重复展开。

server 本身不加载 registry 的 active/previous rollback Actor，也不需要 deployment profile、deployment binding 或 Actor-only export。启动时先选择 `outputs/task2/online/checkpoints/` 中 cycle 最大且结构、optimizer counter 与 scheduler 计数一致的 online exact-resume checkpoint；不一致的 online checkpoint 会被跳过，没有可恢复的 online checkpoint 时回退到 `outputs/task2/offline/checkpoints/offline_actor_critic_cycle_000210`。Inference Actor 固定为所选 checkpoint 的 `actor/`，Learner 从同一目录恢复 Critics、targets、两个 optimizer、scheduler、RNG、sampler、cycle counter 和 replay cursor，不允许混合启动。

`--allow-development-policy-execution-smoke` 是已有的显式机器人执行开关；它不选择模型，也不触发 publication、activation、candidate、profile 或 binding 流程。力限、takeover generation、stale-result rejection、ACK 和 recorder 单控制链保持不变。

在线推理只对反归一化后的 gripper candidate 做有限值饱和：低于 `-0.01 m` 按闭合端处理，高于 `0.095 m` 按打开端处理，二值判定阈值保持 `0.0425 m`，随后只输出精确的 `0.0 m` 或 `0.085 m`。`NaN/Inf` 继续拒绝；TCP6、力限和 action normalizer 不做裁剪或改写。

每 5 cycles 只在内存广播 Actor，新参数从下一次 inference request 的新 H50 chunk 生效；同步点不写 checkpoint、不导出 package、不做 candidate validation。每 50 cycles 保存完整 checkpoint：

```text
outputs/task2/online/checkpoints/online_actor_critic_cycle_000050
outputs/task2/online/checkpoints/online_actor_critic_cycle_000100
```

只保留最新两个 checkpoint。online cycle 从在线 optimizer 首次启动时的 0 开始，与 episode 编号无关。

正常停止使用 recorder 的 `q`。系统停止新 learner cycle，等待正在进行的 optimizer step 完成，并保存最后完成 cycle；若该 cycle 恰好是 50 的倍数，只保存一次。Learner 异常失败时不保存可能只完成部分 optimizer step 的 checkpoint，也不修改原始 episode或把未封口 episode 加入 replay。
采集途中若因控制器、通信或进程错误退出，canonical online-loop 会自动删除本次未封口 session root 及 `.inprogress` 内容，不保留半条 episode。已存在 technical seal 的 session 不自动删除，即使后续 admission 失败，也保留供修复后重试。

例如 cycle45 退出保存 cycle45；cycle55 保留 cycle50 与 cycle55；cycle107 保留 cycle100 与 cycle107。每 5-cycle 广播不写磁盘，也不导出 Actor package。

Actor export 只在用户以后明确要求部署某个完整 checkpoint 时，才写入 `outputs/task2/exports/actor`；它不是本 pipeline 的自动阶段。

## 11. 保留与故障处理

必须保留：原始 D 数据和 LeRobot v3、SFT、reward classifier、materialized demo replay、offline Critic、offline exact-resume、最新两个 online exact-resume、formal replay/WAL/outbox/admission 引用的所有 raw episode、registry 和 active/previous rollback export。

常见 fail-closed 原因：exact-resume checkpoint 不完整、推理 Actor 与 Learner checkpoint 不同源、原始 JPEG 缺失、takeover 后旧 result、gripper origin 不完整、ACK 缺失、checkpoint/replay UID 或 credit 不一致。不得用其他 episode 图片、虚假 command ID/ACK、重绑旧 generation 或修改原始 episode绕过。
