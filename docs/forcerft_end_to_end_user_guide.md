# ForceSmolVLA / ForceRFT 端到端用户手册

> 本手册依据当前仓库 CLI、配置、revision registry、checkpoint metadata 与正式 replay 引用整理。当前数据集、模型与部署清单均带有 development-only 约束；本文不把它们描述为未经批准的正式生产发布。

## 1. 系统概览

ForceSmolVLA 是以双相机图像、机器人状态和标定后的 TCP wrench 为条件的 H=50、action7 策略。公开动作由 TCP6 与 gripper 组成；离线 SFT 使用 flow matching 学习动作序列。ForceRFT 在该 Actor 上增加 Force-aware Twin-Q Critic，并用离线 demonstration replay `D` 与正式在线 replay `R` 继续训练。

数据流如下：

1. 原生 recorder 以独立 native-rate streams 保存 D435、D405、state7、wrench、gripper、raw/safe action 与 Pose ACK。
2. converter 在离线阶段做因果对齐，生成 LeRobot v3 数据集、episode split 和只由 train split 拟合的 frozen normalizer。
3. ForceSmolVLA SFT 全量更新已有模型参数，得到 offline SFT Actor parent。
4. reward classifier 在人工审核的 frame labels 上训练；冻结 detector 再把 frame score 物化成 reward、terminal 和 demonstration transition view。
5. Twin-Q 使用离线 `D` bootstrap。在线 ForceRFT Learner 恢复 Actor、Q1/Q2、target Q1/Q2、两个 optimizer、RNG、sampler 和 sample-credit 状态。
6. 在线阶段由 active Actor 控制机器人；Learner 只采样 episode 开始前已经 seal、bridge PASS、admission 的历史 `R` 和已绑定 `D`。当前正在采集的 episode 不可采样。
7. episode 结束后依次执行 production bridge、append-only admission、candidate export/publication、真实 Home witness 和 episode-boundary activation。revision 只在 episode 边界切换。

当前实现的 online joint cycle 保持既有算法：每轮 2 个 Critic TD optimizer step、2 个 target Polyak update、1 个 Actor optimizer step；R/D 为 50:50；FM 只用于 expert/demo；Actor guidance 使用 `min(Q1,Q2)`；Q gradient 只进入 TCP6，gripper 的 Q gradient 为零，但 gripper 仍可从 expert FM 获得梯度；VLM 保持 frozen/eval/no-grad。

### 1.1 从零开始的完整顺序

| 环节 | 输入 | 执行入口 | 主要输出 | 下一步用途 |
|---|---|---|---|---|
| 原生真机示教采集 | FR3、双相机、wrench、gripper、SpaceMouse | `$FR3_WS/scripts/record_franka_hilserl_impedance.py` | accepted native episode | LeRobot v3 转换 |
| 数据转换 | accepted native episodes | `tools/convert_franka_raw_to_lerobot_v3.py` | LeRobot v3、split、normalizer manifests | SFT、reward、offline replay |
| Actor 全量微调 | LeRobot v3 train split | `tools/train_forcesmolvla_sft.py` | offline SFT Actor checkpoint | Twin-Q/在线训练的 Actor parent |
| Reward 人工标注 | review bundle、双相机帧 | `tools/reward_classifier/annotate_reward_frames.py` | reviewed frame labels、label inventory | classifier 训练 |
| Reward classifier | reviewed labels、train/val 图像 | `tools/reward_classifier/train_reward_classifier.py` | frozen detector checkpoint | reward/terminal 物化 |
| Reward 与离线 replay | LeRobot v3、frozen detector | `tools/materialize_reward_transitions.py`、`tools/materialize_offline_demo_replay.py` | reward/terminal transition view、offline `D` | Twin-Q 与 joint learner |
| Twin-Q bootstrap | offline `D`、offline SFT Actor | `tools/train_twin_q_critic.py` | Q1/Q2、target Q1/Q2 | Critic warmup/joint update |
| Frozen-VLM joint update | Actor、Critics、正式 `R`、offline `D` | `tools/train_forcerft_critic_warmup.py`、`tools/train_forcerft_actor_critic.py` | exact-resume checkpoint、candidate Actor | offline validation/publication |
| 持续在线训练 | active Actor、latest checkpoint、formal replay | `tools/run_forcerft_online_loop.py` | 新 episode、append-only `R`、pending checkpoint/candidate | Home-boundary publication/activation |

每一行成功完成并通过该行的完整性检查后再进入下一行。不要用修改 manifest、伪造 ACK/terminal 或复制其他 episode 图片的方法跨过失败步骤。

## 2. 环境与硬件

### 2.1 目录与 Python

先按本机安装位置设置变量。不要把某次 revision、日期或 cycle 写进通用脚本。

```bash
export FORCESMOLVLA_ROOT=/path/to/ForceSmolVLA
export FR3_WS=/path/to/fr3_client_ws
export MODEL_PYTHON=/path/to/forcesmolvla/bin/python
export REWARD_PYTHON=/path/to/conrft_reward/bin/python
export ROBOT_PYTHON="$FR3_WS/.venv/bin/python"
export RAW_ROOT="$FR3_WS/datasets/task2"
export LEROBOT_DATASET="$FORCESMOLVLA_ROOT/datasets/task2_lerobotv3"
export REWARD_TRANSITION_ROOT=/absolute/path/to/reward_transition_view
export FORMAL_R_ROOT="<current-formal-online-replay-root>"
export CHECKPOINT_ROOT="$FORMAL_R_ROOT/checkpoints"
export REVISION_REGISTRY="<current-policy-revision-registry>"
cd "$FORCESMOLVLA_ROOT"
```

Actor/Critic 训练、validation、publication 和 activation 使用包含 PyTorch 与 `jsonschema` 的 `$MODEL_PYTHON`。Reward cache 准备用 `$MODEL_PYTHON`，Reward classifier 的 `train` 子命令使用项目绑定的 JAX/Flax `conrft_reward` 环境 `$REWARD_PYTHON`。机器人 Home/recording 使用 FR3 ROS 工作区的 Python。不要用缺少 `jsonschema` 的 robot venv 执行 revision activation。

机器人终端需要：

```bash
source /opt/ros/humble/setup.bash
source "$FR3_WS/install/setup.bash"
source "$FR3_WS/.venv/bin/activate"
export ROS_DOMAIN_ID=<domain-id>
export ROS_LOCALHOST_ONLY=0
```

### 2.2 硬件与绑定

当前真实链路包含：

- Franka FR3；
- OnRobot HEX-E force/torque sensor；
- Robotiq gripper；
- D435 external camera；
- D405 wrist camera；
- SpaceMouse，用于人工 takeover、TCP 平移/roll 和 gripper toggle；
- 单一 recorder-owned Cartesian impedance control chain。

`--tool-profile onrobot_robotiq` 必须与机器人侧发布的 `/fr3/tool_profile` 一致。当前工具定义来自 `$FR3_WS/config/tool_profiles.yaml`，标定绑定来自 `configs/calibration_bundle.development.json` 指向的真实 calibration JSON。不要在命令行重新定义另一套 TCP、wrench frame 或 Home joint target。

## 3. 原生数据采集

### 3.1 入口与依赖

原生入口是：

```text
$FR3_WS/scripts/record_franka_hilserl_impedance.py
```

它实际复用：

- `record_franka_forcevla.py`：recorder、camera/gripper/wrench/action worker 与 integrity checks；
- `record_franka_forcevla_raw.py`：native stream storage；
- `record_franka_spacemouse_publisher.py`：真实 Home、SpaceMouse、gripper 和 takeover arbiter。

### 3.2 采集命令

在 robot/ROS/tool-profile 服务已经就绪后运行：

```bash
"$ROBOT_PYTHON" "$FR3_WS/scripts/record_franka_hilserl_impedance.py" \
  --root "$RAW_ROOT" \
  --task "Pick up the purple ring and place it onto the red peg." \
  --episodes 10 \
  --episode-time 120 \
  --tool-profile onrobot_robotiq
```

常用交互：

- `Enter`：请求保存；所有 integrity checks 通过后才进入 `episodes/episode_*`；
- `d` 后 `Enter`：discard 当前 episode；
- `q` 后 `Enter`：停止，不保存当前 episode。

不要使用 `--skip-home` 作为常规流程。正常流程复用 recorder 中的 recorded Home，并等待到位与稳定。accepted episode 与 rejected raw data 分离；rejected episode 不可通过修改 manifest 或伪造 ACK 变成 accepted。

### 3.3 原生 streams

accepted episode 至少保存：D435 external RGB、D405 wrist RGB、measured TCP pose/state7、gripper state/target/goal status、标定 wrench stream、raw action、safe action、requested/accepted reference、reference/Pose ACK、controller state 与 episode result。各 stream 保持 native rate，30 Hz 对齐在 converter 或 bridge 阶段完成。

## 4. 转换为 LeRobot v3

### 4.1 转换

当前入口为 `tools/convert_franka_raw_to_lerobot_v3.py`，实现位于 `src/forcesmolvla/raw_to_lerobot_v3.py`。

```bash
"$MODEL_PYTHON" tools/convert_franka_raw_to_lerobot_v3.py \
  --raw-root "$RAW_ROOT" \
  --output-root "$LEROBOT_DATASET" \
  --repo-id local/task2_lerobotv3 \
  --project-root "$FORCESMOLVLA_ROOT" \
  --runtime-spec configs/converter_runtime_spec.task2.development.json \
  --development-only
```

在正式写出前可只检查输入：

```bash
"$MODEL_PYTHON" tools/convert_franka_raw_to_lerobot_v3.py \
  --raw-root "$RAW_ROOT" \
  --output-root "$LEROBOT_DATASET" \
  --repo-id local/task2_lerobotv3 \
  --project-root "$FORCESMOLVLA_ROOT" \
  --runtime-spec configs/converter_runtime_spec.task2.development.json \
  --preflight-only
```

`--development-only` 明确把当前开发数据写成不可冒充 formal dataset 的输出；只有将来具备独立批准的正式输入契约时才可省略。正式转换要求新的空输出目录，不覆盖已有 LeRobot 数据集。

### 4.2 数据契约

当前 task2 v3 manifest 固定：

- 两路 RGB：external D435 与 wrist D405；
- state7：TCP position/quaternion；
- calibrated TCP wrench6；
- action7：TCP6 + absolute gripper width；
- 按 episode 划分 train/val/test，不按 frame 混分；
- causal alignment，不读取未来样本；
- normalizer 只在 train episode 上拟合。

转换完成后至少核对：

```bash
"$MODEL_PYTHON" -m pytest -q tests/test_raw_to_lerobot_v3.py tests/test_lerobot_v3_smoke.py tests/test_split_normalizer.py
```

`normalizer_manifest.json` 是冻结输入。offline SFT、Critic、Actor joint training 和 policy inference 都只能应用一次，不得在后续环节重拟合或重复 normalize。

## 5. ForceSmolVLA 全量微调

### 5.1 配置与训练

当前 SFT 入口是 `tools/train_forcesmolvla_sft.py`，task2 实验配置是 `configs/train/task2.json`。完整训练契约来自 `configs/offline_sft_training_recipe.development.yaml`，其中绑定当前 canonical recipe `configs/forcesmolvla_sft_recipe.development.yaml`。

```bash
export PYTHONHASHSEED=42
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
"$MODEL_PYTHON" tools/train_forcesmolvla_sft.py \
  --dataset "$LEROBOT_DATASET" \
  --config configs/train/task2.json
```

精确恢复：

```bash
"$MODEL_PYTHON" tools/train_forcesmolvla_sft.py \
  --dataset "$LEROBOT_DATASET" \
  --config configs/train/task2.json \
  --resume /absolute/path/to/sft/checkpoint
```

当前 recipe 的主要语义是 40,000 training samples、batch/GPU=4、AdamW、bf16 autocast、force adapter/action head fp32，以及 `L_flow + 0.01*L_balance + 0.001*L_z`。offline full-finetune 要求所有已有模型参数 trainable；VLM、force branch、router/action expert 一起更新。不要把在线 ForceRFT 的 frozen-VLM 规则错误套用到这一步。

checkpoint 包含模型、optimizer、scheduler/RNG/dataloader 恢复契约及 runtime manifests。最终 offline SFT Actor parent 由 `configs/online_replay_bootstrap_parent_binding.v1.development.json` 动态绑定；不要按历史 cycle 名称猜测。

## 6. Reward 标注

### 6.1 人工 frame labels

当前 reviewed labels 位于 `labels/task2_reward_frame_labels.v2.reviewed.json`。使用现有 review bundle 逐 episode 查看同一 parquet row 的 D435 third-person 与 D405 wrist 帧，填写 confident completion boundary、ordinary negative、hard negative、ambiguous、reviewer 与 confidence。这个过程必须由人工完成；CLI 只验证和 ingest，不替代人工判断。

任务的唯一语义是：抓起紫色圆环，使中心孔与红色 peg 对齐，将圆环套到 peg 上，松开夹爪，并让圆环稳定支撑在 peg/base assembly 上。只有以下四项同时成立时才标为 positive：

1. 红色 peg 明确穿过紫色圆环中心孔；
2. 圆环已脱离夹爪，不再由夹爪支撑；
3. 圆环稳定落在 peg/base assembly 上；
4. 后续可观察帧中没有滑落、弹出或重新抓取。

`first_confident_complete_frame` 是双相机首次同时支持以上四项的帧。圆环仍在桌面、尚未抓取或明显远离 peg 属于 ordinary negative；已经到 peg 上方、近似同轴、接触或部分套入，但仍被夹爪支撑或尚未稳定，属于 hard negative；遮挡、双相机结论不一致或无法确认稳定性的帧必须标为 ambiguous/ignore，不能进入 classifier 训练、阈值选择或指标统计。不得根据 `saved=true`、episode 末帧、文件名、episode success 或示例视频时间推断完成帧。

每条 episode 至少记录 `last_confident_incomplete_frame`、`first_confident_complete_frame`、hard/ordinary/ambiguous 闭区间、`completion_visible`、`completion_stable`、`positive_available`、reviewer、时间、confidence 与 notes。成功后立即结束、没有足够后续帧验证稳定性时，必须令 `positive_available=false`，不能强制把末帧标为 positive。episode split 必须沿用转换时的冻结 split；同一 episode 不得跨 split，test 不参与训练或阈值选择。

标签 ingest/validation：

```bash
"$MODEL_PYTHON" tools/reward_classifier/annotate_reward_frames.py \
  --reviewed-labels labels/task2_reward_frame_labels.v2.reviewed.json \
  --dataset-root "$LEROBOT_DATASET" \
  --validate-only
```

ambiguous frame 完全排除；ordinary negative 和 hard negative 都映射为 0，positive 映射为 1。验证会拒绝 episode 泄漏、重叠区间、未标帧、缺失 reviewer 和不连续 positive 区间。

### 6.2 reward/terminal 物化

冻结 detector transition view 的当前入口：

```bash
"$MODEL_PYTHON" tools/materialize_reward_transitions.py build \
  --config configs/reward_transition_materialization.development.json \
  --dataset-root "$LEROBOT_DATASET" \
  --output-root "$REWARD_TRANSITION_ROOT"
```

当前 reward contract 使用 30 Hz detector score、阈值 0.83、连续 5 帧为正、第五个确认帧为 terminal；reward 是 sparse binary terminal。没有 manual boundary、episode-end、saved=true 或 last-frame fallback。detector 未触发的 episode/transition 按现有规则排除，不伪造 terminal。

`transition_index.parquet` 负责 episode/frame/transition 对齐；`frame_scores.parquet` 保存冻结 detector 输出。当前配置明确是 development scope，不能把它描述为外部正式 detector approval。

## 7. Reward classifier

### 7.1 数据 cache 与训练

```bash
export REWARD_CACHE=/absolute/path/to/reward_classifier_cache
export REWARD_RUN=/absolute/path/to/reward_classifier_run

"$MODEL_PYTHON" tools/reward_classifier/train_reward_classifier.py \
  --config configs/stage2_r0_reward_classifier_training.development.json \
  prepare-cache --cache-dir "$REWARD_CACHE"

"$REWARD_PYTHON" tools/reward_classifier/train_reward_classifier.py \
  --config configs/stage2_r0_reward_classifier_training.development.json \
  train --cache-dir "$REWARD_CACHE" --output-dir "$REWARD_RUN"
```

当前 classifier 为 frozen pretrained ResNet10 pre-pooling backbone，加双相机 learned embeddings/bottleneck 与 binary head。训练使用 Adam、150 optimizer updates、batch 256，positive 占 128，ordinary/hard negative 各 64；validation checkpoint 以最低 BCE 为主、PR-AUC 为 tie-break。

当前 production bridge 使用的 development detector checkpoint 路径由 `configs/reward_transition_materialization.development.json` 的 `frozen_inputs.classifier_checkpoint.path` 读取。不要手工替换 checkpoint 或只改文件名。

当前 reward materialization contract 使用 probability threshold `0.83`，并要求连续 5 个 30 Hz positive frame 才触发 detector terminal。单帧超过阈值不等于 terminal；阈值、连续帧数、checkpoint 与 split 都必须以该配置为准。历史校准或一次性测试报告不再作为运行入口。

## 8. Twin-Q Critic

Twin-Q 架构在 `src/forcesmolvla/rft/critic.py`，ActionContract-v2 adapter 在 `src/forcesmolvla/rft/critic_action_adapter_v2.py`。Critic 输入包含双相机、state7、wrench6、确定性的 256D canonical task feature 和 K=3 的 action7。

Actor 输出 H=50 的 absolute action7；执行端以 10 Hz dispatch，Critic 使用连续 3 个已执行 action7 构造 macro-action。TCP6 与 gripper 都必须经过同一个 ActionContract-v2 adapter，state7、wrench6 和 action normalizer 各应用且只应用一次。padding/valid mask、terminal mask、bootstrap mask 和 loss mask 不得混用；terminal transition 的 TD target 不调用 next Actor 或 target Q。

离线 demonstration replay 由冻结 detector view 构造：

```bash
"$MODEL_PYTHON" tools/materialize_offline_demo_replay.py \
  --dataset-root "$LEROBOT_DATASET" \
  --output-root /absolute/path/to/offline_demo_replay \
  --reward-labels labels/task2_reward_frame_labels.v2.reviewed.json \
  --reward-spec configs/stage2_reward_spec.development.yaml \
  --action-contract configs/stage2_action_contract.v2.development.json \
  --source-manifest artifacts/development/stage2/stage2_source_manifest.v4.json
```

当前真实 bootstrap Critic 是 `configs/online_replay_bootstrap_parent_binding.v1.development.json` 选定的 Q1/Q2 与 target Q1/Q2。canonical 训练入口是：

```bash
"$MODEL_PYTHON" tools/train_twin_q_critic.py --run
```

该入口复用正式 Critic worker、训练周期 primitives 和固定配置；`--run` 才会执行真实 Critic 训练。

## 9. Frozen-VLM Actor/Critic 微调

### 9.1 Parent binding

在线 Actor/Critic 初始化 parent 必须从 `configs/online_replay_bootstrap_parent_binding.v1.development.json` 读取：

- Actor：已批准的 offline SFT Actor export；
- Q1/Q2 与 target Q1/Q2：已绑定 Twin-Q bootstrap；
- frozen state7/wrench6/action normalizer；
- ActionContract-v2；
- canonical task feature；
- calibration/runtime binding。

### 9.2 Critic warmup

```bash
"$MODEL_PYTHON" tools/train_forcerft_critic_warmup.py \
  --steps 100 \
  --checkpoint "$CHECKPOINT_ROOT/online_replay_critic_warmup_step_<step>"
```

warmup 创建 fresh Critic optimizer，不创建 Actor optimizer；Actor frozen/eval/no-grad，仅用于 non-terminal TD target action；target Critics no-grad；每个 Critic step 后一次 Polyak update；terminal transition 不调用 next Actor/target Q。

### 9.3 Joint update 与 exact resume

```bash
"$MODEL_PYTHON" tools/train_forcerft_actor_critic.py \
  --cycles 10 \
  --resume-checkpoint "$CHECKPOINT_ROOT/<exact_resume_parent>" \
  --checkpoint "$CHECKPOINT_ROOT/<new_joint_checkpoint>" \
  --candidate-id <new-candidate-id>
```

exact resume 必须恢复 Actor、Q1/Q2、targets、Actor/Critic optimizer、RNG、sampler 和 sample credits；Critic optimizer 不可重建，Actor/Critic optimizer 参数不可重叠，frozen VLM 与 state-prefix projection 不进入 optimizer。

联合训练保持固定语义：每个 cycle 为 2 个 pure Twin-Q TD step、每步后 1 次 target Polyak update，以及 1 个 Actor step；R/D 为 50:50；FM loss 只作用于 expert/demo；Actor Q guidance 使用 `min(Q1,Q2)`；Q gradient 只进入 TCP6，gripper Q gradient 必须精确为零，但 gripper 仍可从 expert FM 获得梯度。VLM 始终 eval/no-grad。不得在该路径引入 Cal-QL、CQL、random candidate、MC return 或 online self-imitation FM。

### 9.4 验证与导出

零 optimizer-step 离线验证：

```bash
"$MODEL_PYTHON" tools/validate_forcerft_candidate.py \
  --checkpoint /absolute/path/to/joint_checkpoint \
  --expected-revision <model-revision> \
  --fixed-episode-id <episode-id>
```

标准导出/开发发布：

```bash
"$MODEL_PYTHON" tools/export_forcerft_candidate.py \
  --joint-checkpoint /absolute/path/to/joint_checkpoint \
  --destination /absolute/path/to/candidate_package.v1 \
  --deployment-profile /absolute/path/to/deployment.development.json \
  --deployment-binding /absolute/path/to/deployment_binding.v1.json \
  --candidate-revision-id <candidate-id> \
  --deployment-id <deployment-id> \
  --approval-id <approval-id>
```

exporter 必须生成 strict loader 所需的 `model.safetensors`、`config.json`、`artifact_manifest.json` 和必要 runtime manifests，只替换 Actor 权重，不修改 parent 或 joint checkpoint。

## 10. 在线 HIL

### 10.1 单 episode unified Actor/Learner

从 registry、active deployment profile 和 exact-resume metadata 读取以下变量，不手填历史 revision：

```bash
export SESSION_ID=<new-unique-session>
export EPISODE_ID=episode_000000
export CAPTURE_ROOT=<new-absolute-capture-root>
export DEPLOYMENT_PROFILE=<active-development-profile>
export DEPLOYMENT_BINDING=<active-binding>
export TRUSTED_BINDING_SHA=<profile.deployment_binding_sha256>
export ACTIVE_MODEL_REVISION=<registry-active-record.model_sha256>
export POLICY_EPOCH=<registry-state.policy_epoch>
export LEARNER_RESUME=<latest-checkpoint-for-active-revision>
export PENDING_CHECKPOINT=<new-unique-pending-checkpoint>
export PENDING_CANDIDATE_ID=<new-unique-pending-id>
```

统一 GPU owner/server：

```bash
"$MODEL_PYTHON" tools/serve_forcerft_actor_learner.py \
  --deployment-profile "$DEPLOYMENT_PROFILE" \
  --deployment-binding "$DEPLOYMENT_BINDING" \
  --trusted-deployment-binding-sha256 "$TRUSTED_BINDING_SHA" \
  --session-id "$SESSION_ID" \
  --episode-id "$EPISODE_ID" \
  --learner-resume-checkpoint "$LEARNER_RESUME" \
  --pending-checkpoint "$PENDING_CHECKPOINT" \
  --pending-candidate-id "$PENDING_CANDIDATE_ID" \
  --host 127.0.0.1 --port 8000 \
  --allow-development-robot-execution
```

另一个已经加载 ROS/robot 环境的终端运行 integrated capture：

```bash
"$ROBOT_PYTHON" tools/run_forcerft_integrated_capture.py \
  --mode policy-execute \
  --allow-development-policy-execution-smoke \
  --async-learner \
  --root "$CAPTURE_ROOT" \
  --task "Pick up the purple ring and place it onto the red peg." \
  --episodes 1 --episode-time 120 \
  --tool-profile onrobot_robotiq \
  --session-id "$SESSION_ID" --episode-id "$EPISODE_ID" \
  --policy-revision "$ACTIVE_MODEL_REVISION" \
  --policy-epoch "$POLICY_EPOCH" --takeover-generation 0 \
  --deployment-profile "$DEPLOYMENT_PROFILE" \
  --deployment-binding "$DEPLOYMENT_BINDING" \
  --policy-host 127.0.0.1 --policy-port 8000 \
  --policy-replan-steps 8 --policy-queue-low-watermark 7 \
  --max-force-n 25 --max-torque-nm 2 --launch
```

server 与 capture 的 session/episode identity 必须相同。Actor 在整个 episode pin 当前 revision；H50 cache 在 10 Hz dispatch，inference 优先于 learner micro-step。takeover 发生时旧 queue/request/result/proposal 全部失效，恢复必须用最新 observation、新 generation、新 request 和新 `t_ref`。

### 10.2 ACK、gripper authority 与 seal

accepted policy transition 需要 policy lineage、action7、Pose ACK 和 gripper authority 同时闭合。human override 不借用 policy ACK。gripper authority 只能来自真实 accepted NEW_COMMAND，或满足显式 takeover synchronization、fresh feedback、无冲突 pending command等条件的 `HELD_FROM_ACCEPTED_COMMAND` transfer；不生成 synthetic goal ID、ACK 或 terminal。

episode 成功保存后产生 technical seal。它记录 active Actor、Learner resume、2 Critic/1 Actor step、pending candidate、current episode sampling isolation、camera reconciliation、terminal observation，以及 `formal_replay=false`、`real_online_r=false`。原始 capture 永远不因后续 admission 被改写。

### 10.3 Bridge 与 admission

只读验收：

```bash
"$MODEL_PYTHON" tools/run_forcerft_production_bridge.py \
  --episode "$CAPTURE_ROOT/episodes/$EPISODE_ID" \
  --operator-task-outcome success \
  --dry-run
```

只有 technical seal、lineage、ACK、generation、gripper provenance、双相机/state7/calibrated wrench6、terminal 与 detector 全部通过，才执行 append-only admission：

```bash
"$MODEL_PYTHON" tools/run_forcerft_production_bridge.py \
  --episode "$CAPTURE_ROOT/episodes/$EPISODE_ID" \
  --state-root "$FORMAL_R_ROOT" \
  --operator-task-outcome success \
  --admit-formal-online-r
```

same UID + same digest 幂等；不同 digest fail closed。admission 写 episode-sealed WAL/outbox/replay 并 mint 新 unique-R credit。被排除的 warmup、human override、stale proposal 或缺少 authority 的 transition 不进入 replay；不能静默修复 identity。

## 11. Continuous online loop

### 11.1 单命令循环

`tools/run_forcerft_online_loop.py` 是当前 thin orchestrator。它复用 server、capture、bridge/admission、export、Home witness 和 activation，不重新实现训练算法。

```bash
"$MODEL_PYTHON" tools/run_forcerft_online_loop.py \
  --max-episodes 100 \
  --root-prefix /absolute/new/path/forcerft_online_run \
  --task "Pick up the purple ring and place it onto the red peg." \
  --episode-time 120 \
  --tool-profile onrobot_robotiq \
  --policy-replan-steps 8 \
  --policy-queue-low-watermark 7 \
  --max-force-n 25 \
  --max-torque-nm 2 \
  --formal-r-root "$FORMAL_R_ROOT" \
  --registry "$REVISION_REGISTRY" \
  --model-python "$MODEL_PYTHON" \
  --robot-python "$ROBOT_PYTHON"
```

每轮动态读取 registry active identity、唯一匹配的 deployment、active revision 对应 checkpoint，并在 resume 时合并 checkpoint 保存后 append-only admission 的新 UID/credits。Learner 只看到本轮开始前的  `R`；当前 episode sample count 必须为 0。

episode 保存并且 learner 完成后，程序提示：

```text
operator_task_outcome [success/failure]:
```

只有输入 `success` 且 bridge PASS，才 admission、publish、真实 Home 和 activate。任何一步失败都会打印 `[continuous] STOP:...` 并停止，不跳过该轮，也不激活失败 candidate。

### 11.2 Bootstrap

只有存在“已 bridge PASS 但尚未 admission”的上一 episode，且有与其对应的完整 pending checkpoint 时，才同时提供：

```bash
--bootstrap-episode /absolute/path/to/episode \
--bootstrap-checkpoint /absolute/path/to/pending_checkpoint
```

不要对已 admission/published/activated 的 episode 重复 bootstrap。

### 11.3 Restart 与安全停止

当前 orchestrator 不在同一 `root-prefix` 上续编号；它从 `_001` 开始并拒绝已存在的 capture root。因此重新启动必须选择全新的 `--root-prefix`。checkpoint/revision 则从 registry 自动接续，不会从旧 Actor 重新开始。

安全停止优先级：

1. 最干净：把 `--max-episodes` 设为计划数量，让当前轮完成 bridge/admission/activation 后退出。
2. 录制时不保存当前 episode：在 recorder 输入 `q` 后 `Enter`；orchestrator 因缺少完整 seal 停止，不会 admission/activate。
3. 紧急停止：先使用机器人既有急停/人工 takeover，再 `Ctrl+C`。orchestrator 的 `finally` 会终止 unified server，但仍应确认 robot controller 和 server 进程都已结束。
4. 若已经保存 episode但尚未输入 outcome，可输入 `failure` 使循环 fail closed。该 episode保留，但不会 admission/publish/activate。

不要删除失败 pending 或 capture 来“续跑”；先确认它们没有 registry、resume、replay 引用，再在 cleanup 流程中处理。

## 12. 数据和 checkpoint 保留策略

### 12.1 必须保留

- 最终 LeRobot v3 offline `D` 数据集及 conversion/split/normalizer manifests；
- v3 manifest、reward classifier、Critic 或从头复现实验引用的原始 raw episodes；
- formal replay/WAL/outbox admission 引用的每个原始 online episode，包括全部 JPEG 和 streams；
- active、immediate previous rollback、pending candidate package；
- latest exact-resume learner checkpoint；
- 至少一个 previous known-good exact-resume checkpoint；
- online bootstrap 所需的 offline SFT Actor、Q1/Q2、target Q1/Q2；
- frozen normalizer、ActionContract-v2、task feature、calibration；
- revision registry、active/previous deployment profile/binding；
- current continuous-loop capture root；
- current reward classifier/detector checkpoint。

正式 `R` 的 observation image 不是自包含复制：sampler 会根据 admission source episode 和 blob 路径重新打开原始 external/wrist JPEG。因此 raw episode 名字即使包含 smoke/validation/旧 cycle，只要被 replay 引用就绝对不能删除。移动目录也会破坏绝对路径引用。

每个 exact-resume checkpoint 自身保存模型、Critics/targets、optimizers、RNG、sampler、replay cursor 和 credits；`source_checkpoint` 还承担 provenance。旧 checkpoint 即使不是 loader 的运行时依赖，也可能仍被 previous package、deployment profile 或后继 metadata 以绝对路径引用。删除前必须同时检查 registry records、candidate package、deployment profile/binding、checkpoint metadata 和后继 checkpoint 的 source reference。

### 12.2 可归档或删除候选

只有同时满足“未被任何正式 replay、训练、报告复现、registry、active/previous/pending package 或 resume parent 引用”时，才可考虑：

- rejected/failed capture；
- 失败且未发布/未激活/未作为 resume parent 的 pending checkpoint；
- 未被 registry 引用的旧 candidate package；
- disposable preflight/synthetic loopback 输出；
- Python/pytest cache 和已结束任务的精确 `/tmp` 路径；
- 完整 v3 已替代且没有入口引用的旧格式 conversion 输出。

清理外部数据时必须以精确绝对路径和当时的 replay/checkpoint/registry 依赖核验为准，不得根据名称或日期批量删除。Graphify 不是生产 pipeline 依赖根，日常运行无需生成。

## 13. 常见故障

### 13.1 `POLICY_EXECUTE_SERVER_AUTHORIZATION_MISMATCH`

常见原因是 source closure 改动后 deployment binding 仍旧。必须使用现有 deployment binding builder 刷新 binding，并同步 profile 中的 binding identity；server 与 capture 必须使用同一 profile/binding/revision/trusted binding。不要绕过 strict loader 或手改 hash 字段。

### 13.2 `ConnectionRefusedError`

integrated capture 在 `policy-host:policy-port` 找不到 server。先确认 unified Actor/Learner server 已监听，且 session/episode identity 与 capture 完全一致。continuous loop 模式会自动管理 server，不需要另开普通 `serve_policy.py`。

### 13.3 Missing raw image

正式 replay 会打开 admission source episode 中的原始 JPEG。图片缺失时停止；优先从同一 episode 的真实备份/Trash 恢复原路径。不得用其他 episode 图片替代、伪造 blob identity 或悄悄排除已 admission transition。

### 13.4 Takeover stale result / epoch discontinuity

takeover 必须 flush queue，invalidate pending/ready old-generation result，并增加 generation。resume 使用 fresh observation/request/result/chunk/proposal 与新 `t_ref`。native arbiter 的 initial policy epoch/takeover generation 必须由 integrated capture 传入。不要 re-anchor 旧 H50 chunk、扩大 horizon 或把旧 result 改绑新 generation。

### 13.5 Gripper origin 或重复 toggle

SpaceMouse takeover 时 toggle 状态先从当前 generation 的 accepted gripper authority 同步；有效 reached 或有真实位移的 stalled 可作为 authority。无 authority 时只允许 fresh measured feedback；stale/unknown 不发 goal。零位移、rejected/error command 不更新 toggle。

bridge 中 takeover 后优先真实 NEW_COMMAND；只有显式 synchronization event、fresh matching feedback、无冲突 pending command和有效真实 origin 全部存在时，才可转移 `HELD_FROM_ACCEPTED_COMMAND`。不得创建 synthetic command/goal/ACK/terminal。

### 13.6 Checkpoint/replay reconciliation

checkpoint 可能早于最新 append-only admission。resume 必须恢复 checkpoint 模型、optimizer、RNG、sampler/credits，再只合并 checkpoint 后的新 UID；same UID 不重复加入或 mint credit。若 live replay count/credits 与恢复后 effective state 不一致，停止且不训练。

### 13.7 `ONLINE_REPLAY_CONTINUOUS_CAPTURE_ROOT_EXISTS` / candidate output exists

continuous loop 要求新 root-prefix 和新 pending/candidate 输出路径。重新启动使用新的 root-prefix；Actor/revision/checkpoint 从 registry 自动接续。不要删除仍可能是 latest resume、active/previous package 或 current capture root 的路径来强行通过。

### 13.8 Activation 缺少依赖或 Home witness

Home witness 必须由 `robot/deployment/reset_home_witness.py` 调用 recorder 的真实 Home 实现生成，不能从 CLI 传 `robot_home=true`。activation/status 用包含 `jsonschema` 的 `$MODEL_PYTHON`：

```bash
"$ROBOT_PYTHON" robot/deployment/reset_home_witness.py \
  --output /absolute/path/to/reset_home_quiescent.json \
  --previous-episode-seal /absolute/path/to/policy_execute_episode_seal.json \
  --interface-timeout 10 --home-timeout 30

"$MODEL_PYTHON" tools/activate_forcerft_policy_revision.py activate \
  --registry "$REVISION_REGISTRY" \
  --home-witness /absolute/path/to/reset_home_quiescent.json \
  --candidate-package /absolute/path/to/published_candidate.v1 \
  --candidate-id <candidate-id> \
  --candidate-revision <model-revision> \
  --current-active-revision <registry-current-active-id>

"$MODEL_PYTHON" tools/activate_forcerft_policy_revision.py status \
  --registry "$REVISION_REGISTRY"
```

activation 要求 episode inactive、request inactive、robot Home 且 quiescent。candidate 必须已经 development-published；activation 原子更新 active/previous/pending 与 policy epoch，不加载模型、不启动 server、不控制机器人。
