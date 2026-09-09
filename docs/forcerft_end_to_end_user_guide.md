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
outputs/{task_id}/online_ack_residual_filter_leash/bootstrap_checkpoints/base_policy_zero_residual_filter_leash_random_twin_q
```

bootstrap checkpoint 保存 frozen base policy 的路径、严格零输出 wrist-wrench residual Actor、随机 ACK-aligned residual Twin-Q、targets、两个 optimizer 与运行计数；不读取 demonstration 或旧 offline Critic checkpoint。

## 8. 真实 ACK Critic warm-up 与 Residual Actor–Critic 训练

Learner 状态依次为 `ack_replay_collection → ack_critic_warmup → residual_actor_critic_training`。累计不足 1000 条正式 TD 或不足 3 条有效 episode 时 Actor/Twin-Q 均不更新；同时达到阈值后在同一进程执行一次 256-step Twin-Q warm-up，然后按累计 `floor(unique_td_rows/8)` 额度执行联合 cycle，每 cycle 固定为 `2 Twin-Q + 1 wrist-wrench residual Actor attempt`。训练只读取低维 state/wrench/base/residual/ACK 数据，不运行第二份 base policy、Flow sampler 或图像 Critic。Twin-Q 输入为原 58 维低维特征再追加 `control_source` 与归一化的 accepted gripper command，共 60 维；Residual Actor 仍为 25 维输入且只输出 TCP6。

Critic 的行为标签始终由同一 residual-decision pose 下的 frozen-base absolute target 与 Controller ACK accepted absolute target 重表达后相减。每次 ACK 同时记录上层 target guard/绝对目标转换、客户端 measured-pose adapter/workspace，以及机器人端 reference filter 与非对称 pose leash 所需的最小执行上下文。Actor value 与 target-Q 都通过同一份可微 PyTorch 镜像将新候选映射到 ACK accepted residual；current 使用本行上下文，target 使用真实后继上下文。映射缺失或候选被真实 guard 拒绝时，只跳过相应 value/TD 项，不改写 terminal、reward 或 bootstrap mask。

## 9. HIL 与 online replay

`tools/serve_forcerft_residual_actor_critic.py` 是唯一 GPU owner；`tools/run_forcerft_integrated_capture.py` 是唯一机器人控制链。当前 chunk/slot 映射为：第 `n` 次真实 dispatch 使用当前已采纳 chunk `i(n)`，再按该 chunk 的 `t_ref_ns` 与 dispatch selection time 在 30 Hz 有理数时基上取 `j(n)=ceil(30(selection_ns-t_ref_ns))`；超出缓存 H50 便丢弃并重规划。模型在 request pose 下生成的 delta 先还原为 absolute chunk，选中槽位再以真实 decision pose 重表达给 Residual Actor。异步新 chunk 只在完成 lineage 检查后生效；最多从一个 chunk 派发 8 次，当前 CLI 的 replan/low-watermark 为 8/7；接管使旧 chunk 与 pending request 失效，释放后必须 fresh observation + fresh inference。只有实际发送且获得 Controller ACK 的 decision 才形成后继，HOLD 或被拒绝命令不虚构 transition。
若 inference 期间 wrench causal filter 因源间隙重置并切换 generation，旧 request/result 和未执行 chunk 会被作废；等待现有 250-sample warmup 完成后，同一 episode 使用 fresh observation 重新 inference，恢复等待期间不生成 transition。

episode seal 后，操作者输入 success/failure。技术记录完整的 success 与 failure episode 都通过 production bridge 的同一次 admission 调用物化 TD transition，并 append 到：

```text
outputs/{task_id}/online/replay
```

不再先 dry-run 后重复 admission。success 的 detector terminal 保持 `reward=1.0, terminated=true, bootstrap_mask=false, discount=0.0`；failure 使用 sealed episode 最后一个有效 Critic transition 作为零奖励 terminal，且同样不 bootstrap。operator failure 与 frozen detector success trigger 冲突时整条 episode 不进入训练。

task2 封口物化在 30 Hz 因果网格上允许双相机样本年龄不超过 `100 ms`，以覆盖正常调度抖动和偶发丢帧；双相机 skew 仍不得超过 `33 ms`，样本年龄超过 `100 ms` 仍拒绝进入 replay。历史 checkpoint 内保存的 provenance 不重写。

所有满足当前 filter/leash accepted-Q 语义且 `critic_td_valid=true` 的正式 sealed online ACK transition 进入同一个低维 Critic replay，包括 autonomous policy ACK、同一 human 控制段内的 accepted correction、success、failure 和 intervention-truncated boundary。Replay 分别记录已物化 transition 数和本次可用于 TD 的行数；缺少接受映射的非终止行可继续提供合法 BC，但不产生 TD 预算。若候选相关的 TD 暂时为空，Learner 保持等待且不推进 optimizer、warm-up 或 cycle 计数。缺少可信 frozen absolute base、真实 dispatch decision context、accepted action、filter/leash 上下文或真实后继关联的旧 transition 不自动补零，保留原始记录但排除相应训练用途。

这两点是当前实现相对旧论文文字的已知差异：合格 human transition 按 `control_source` 条件化后参与 Critic，且 Q 评价 controller 接受映射后的 residual；Actor 的 Q 项仍只抽 policy 样本。本轮不删除 human TD、不改奖励，也不把 Q 改回 pre-controller proposal。

只有带可靠 pre-takeover base action 的 human row 才提供 residual Actor 监督目标；缺少该基线时仅令 `human_residual_valid=false`，不拒绝 episode。人工姿态差继续使用 RPY delta，BC target 单独投影到 Residual Actor 的输出范围，原始 ACK、行为残差与 human TD 不被改写。接管开始时清空旧 chunk/pending request/旧 observation、接管期间暂停 policy dispatch、释放后 fresh observation + fresh inference 的控制语义保持不变。

takeover、release 与 reset 继续作为信用截断边界。因此 Q 是“给定控制源和夹爪命令的当前控制段内折扣终端成功反馈”代理，而不是完整 episode 自主成功率的无偏估计：在 `自主 A → 人工 B → 自主 C → 成功` 中，成功只沿 C 的 TD 链传播；B 仍通过 human TD 与有界 BC 提供训练信号，但 BC 不等价于补回任务奖励信用。

## 10. 持续在线 Actor/Learner

```bash
"$MODEL_PYTHON" tools/run_forcerft_online_loop.py \
  --task-id "$TASK_ID" \
  --output-root "$TASK_OUTPUT_ROOT" \
  --dataset-root "$LEROBOT_DATASET" \
  --max-episodes 100 \
  --capture-output-root "$ONLINE_CAPTURE_ROOT" \
  --ack-replay-root "$TASK_OUTPUT_ROOT/online_ack_residual_filter_leash/formal_replay" \
  --online-residual-bootstrap-checkpoint \
    "$TASK_OUTPUT_ROOT/online_ack_residual_filter_leash/bootstrap_checkpoints/base_policy_zero_residual_filter_leash_random_twin_q" \
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

Unified server 每次启动恢复一个 residual Actor/Twin-Q checkpoint，并在完成 checkpoint/replay 完整性验证后立即启动独立 learner worker。只有累计至少 1000 条唯一、正式接纳并实际物化为 `critic_td_valid` 的 transition，且这些 TD 来自至少 3 条正式 episode，才一次性完成 256 个 Twin-Q warm-up optimizer step并进入联合训练。当前协议下合格 human TD 与 policy TD 都计入；仅可用于 BC 而不能用于 TD 的 human 行不计。每个完成的联合 learner cycle 固定为 2 次 Twin-Q optimizer update、2 次对应的 target Polyak update 和 1 次 residual Actor update 尝试；Actor 因无有效支持而跳过时仍完成并消费本 cycle。warm-up 不计入联合 cycle。训练不读取 demonstration 图像或运行第二份 base policy/Flow sampler。

learner 与 recorder 并行：episode 正在执行时，只要推理优先协调器留出合理空隙，Critic 更新和 Actor 尝试都可继续；训练 Actor 与被执行端 pin 的 Actor 是两个实例。联合训练使用累计数据额度 `allowed_cycles=floor(unique_td_rows/8)`；一个 in-flight cycle 与已完成 cycle 都占用额度。没有新 TD 时墙钟时间、等待人工确认和重复 sampling 均不增加额度。正式 admission 的 HTTP 确认只校验 committed manifest 并唤醒 learner，额度仅在 replay refresh 实际物化和筛选 TD 后登记。额度耗尽或 TD 暂不可映射时 learner 通过可取消事件等待，collector 仍可直接准备下一条，不执行 drain 或前台 join。

async capture manifest 可以覆盖启动数据等待、Critic warm-up、额度等待、联合训练或 partial Q-cycle 进度；`current_episode_sampled_by_learner=false` 必须由 replay membership 与实际 batch provenance 共同证明。当前执行、未提交、未确认、rejected 与 discarded episode 不得进入 replay。

canonical online loop 在每个 episode 后只打印两行 capture/learner 摘要和一行 admission 摘要；完整 contract、stream quality 与 episode seal 继续保存在 session 文件中，不在终端重复展开。

启动时先选择 `outputs/{task_id}/online_ack_residual_filter_leash/training_checkpoints/` 中 cycle 最大且结构完整的 exact-resume checkpoint；也可用 `--learner-resume-checkpoint` 明确选择 checkpoint。没有可恢复 checkpoint 时只接受显式 `--online-residual-bootstrap-checkpoint`。模型结构、optimizer、loss、residual cap、batch、调度与 filter/leash accepted-Q 语义全部以 checkpoint 的 `state/config.yaml` 为唯一权威；缺少 TD 额度 ledger、逐轴 cap 或新 loss 配置的旧 checkpoint 不恢复，也不走旧调度迁移。

新 schema 的普通 resume 仍要求配置完全一致。新的 TD 额度 ledger、逐轴 cap、loss 与 Actor 学习率都属于算法状态；旧 checkpoint 或旧 bootstrap 不能迁移或静默套用新 YAML，必须从冻结 SFT 先验重新构建 zero-residual bootstrap。

`--allow-development-policy-execution-smoke` 是已有的显式机器人执行开关；它不选择模型，也不触发 publication、activation、candidate、profile 或 binding 流程。力限、takeover generation、stale-result rejection、ACK 和 recorder 单控制链保持不变。

在线推理只对反归一化后的 gripper candidate 做有限值饱和：低于 `-0.01 m` 按闭合端处理，高于 `0.095 m` 按打开端处理，二值判定阈值保持 `0.0425 m`，随后只输出精确的 `0.0 m` 或 `0.085 m`。`NaN/Inf` 继续拒绝；TCP6、力限和 action normalizer 不做裁剪或改写。

Residual Actor 的归一化输出逐轴为 `tanh(logit)*cap6`，其中 `cap6=min(0.1, physical_limit6/delta_action7.std[:6])`。物理 proposal 相对 frozen-base target 的每轴上限为平移 1 mm、RPY 0.5 度；这不是 TCP 实测运动、总旋转角或整条轨迹的保证。online Actor、target Actor、CPU inference 与 human BC 投影共用同一 `cap6`。真实 ACK-minus-base 行为残差保留原值，即使超过 cap 也不回写裁剪。

唯一发布/保存时钟是已完成的联合 learner cycle：完成 cycle 100、200、300……时发布 residual Actor candidate，完成 cycle 1000、2000、3000……时保存完整 exact-resume checkpoint。发布不取决于该 cycle 的 Actor optimizer 是否 applied；参数未变化时仍保留该周期 publication event，但相同权重 blob 可复用。候选记录导出时的 admission/TD 覆盖、实际 sample draws、逐轴 cap/loss 合同，并对每条已加载 episode 的确定性状态做轻量 proposal probe；它不是自主成功率评估。候选自己的训练数据未达到 1000 TD/3 episode 时不得激活，不能借激活时后来加入的数据补资格。

发布与执行激活是两个事件。episode 内执行 Actor 始终 pin；同一 episode 中产生多个候选时只保留最新 pending candidate，并在下一个既有 episode boundary 原子激活。激活不再触发完整训练 checkpoint。状态分别展示当前 learner cycle、last published cycle，以及 active Actor 的 publication cycle/revision/policy epoch。

完整 checkpoint 的周期保存示例：

```text
outputs/{task_id}/online_ack_residual_filter_leash/training_checkpoints/residual_actor_critic_cycle_001000
outputs/{task_id}/online_ack_residual_filter_leash/training_checkpoints/residual_actor_critic_cycle_002000
```

只保留最新十个 retention-managed 完整 checkpoint。Critic warm-up 完成和 graceful/operator 退出是非整千保存的明确例外；candidate publication 或 activation 不是。cycle 1000 同时按“先完成 publication 状态、再保存统一训练快照”的顺序处理，checkpoint 内的模型、targets、optimizers、计数、水位和 active/pending Actor 绑定来自同一个安全边界。

正常停止使用 recorder 的 `q`。系统先结束 capture/Actor window，再取消新的 learner 调度，在安全边界停止并幂等保存当前一致状态；不会等待连续训练“跑完”或凑到下一个 100/1000 周期。协调器中的 replay/推理覆盖等待均可被 stop 唤醒。若在两个 Q update 之间退出，`partial_cycle_q_updates` 明确记录部分进度，不能把它伪装成完整 cycle。Learner 异常失败时不修改原始 episode，也不把未封口 episode 加入 replay。
采集途中若因控制器、通信或进程错误退出，canonical online-loop 会自动删除本次未封口 session root 及 `.inprogress` 内容，不保留半条 episode。已存在 technical seal 的 session 不自动删除，即使后续 admission 失败，也保留供修复后重试。

`--max-episodes 1` 仍会在单条 episode 流程结束后保存已实际取得的进度并退出，不会为了制造更新数而 drain 旧预算。要观察采集与 learner 的真实重叠，应连续采集多条，例如设置 `--max-episodes 3`；CPU 测试只验证调度与并发语义，不能代表 GPU 推理延迟或实机吞吐。

状态/日志字段示例（仅示例，不是一次实际训练结果）：

```text
completed_learner_cycles=1200 partial_cycle_q_updates=0
total_twin_q_optimizer_steps=2656 warmup_twin_q_optimizer_steps=256
residual_actor_update_attempts=1200 residual_actor_optimizer_steps=1194
residual_actor_updates_skipped_no_gradient=6
last_published_cycle=1200 last_periodic_checkpoint_cycle=1000
last_checkpoint_cycle=1000
active_actor_online_cycle=1100 pending_actor_revision=task3-residual-policy-cycle-001200-g...
capture_window.delta.completed_learner_cycles=37
capture_window.current_episode_sampled=false
learner_wait_reason=insufficient_action_coverage learner_wait_ms=18.4
```

## 11. 保留与故障处理

必须保留：原始 D 数据和 LeRobot v3、SFT、reward classifier、materialized demo replay、历史 offline Critic 实验记录、最新十个 online exact-resume，以及 formal replay/WAL/outbox/admission 引用的所有 raw episode。

常见 fail-closed 原因：exact-resume checkpoint 不完整、推理 Actor 与 Learner checkpoint 不同源、原始 JPEG 缺失、takeover 后旧 result、gripper origin 不完整、ACK 缺失、checkpoint/replay UID 或 credit 不一致。不得用其他 episode 图片、虚假 command ID/ACK、重绑旧 generation 或修改原始 episode绕过。
