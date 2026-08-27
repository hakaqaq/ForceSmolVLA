# task2 Reward Classifier 人工帧级审核协议（development）

状态：`MANUAL_REVIEW_REQUIRED`。本协议只用于生成 Reward Classifier 候选标签；不生成 ForceRFT reward、terminal、MC return 或 G1 transition。

## 数据与界面

- 唯一图像事实源：只读 `datasets/task2_lerobotv3`。
- 频率：30 Hz；界面同步显示 `D435 third-person`（`observation.images.camera1`）和 `D405 wrist`（`observation.images.camera2`）的同一 parquet row。
- 每帧显示 `episode_id`、`output_episode_index`、`split`、`frame_index`、`timestamp` 和 LeRobot-v3 row reference。
- Bundle 不复制图像。只读 localhost server 按需读取 parquet 内嵌 PNG；浏览器中的人工输入只能导出 JSON，不能回写数据集。
- 训练候选以后可按每 3 帧下采样为 10 Hz；Reward Detector 阈值与连续帧校准必须保留完整 30 Hz 序列。

启动只读审核界面：

```bash
python tools/reward_classifier/serve_task2_label_ui.py
```

然后访问 `http://127.0.0.1:8765`。空白模板位于 `labels/task2_reward_frame_labels.v1.template.json`。人工导出的 JSON 不得覆盖该模板。

## 标签定义

- `positive = 1`：插头已明确完成物理插装，不是仅对齐、接触或部分进入；完成状态在后续可观察帧中保持成立。
- `ordinary_negative = 0`：插头明显尚未对齐、接触或进入目标插槽。
- `hard_negative = 0`：插头已经接近、对齐、发生接触或部分插入，但尚未满足完整插装条件。
- `ignore`：人工无法可靠判断的模糊边界帧；不得进入训练、阈值选择或指标统计。

冻结 conversion manifest 中的任务文本为 “Pick up the purple ring and place it onto the red peg.”，与本轮“插头完整插装”的审核措辞存在对象命名差异。界面同时显示两者。审核者必须按本轮人工定义判断，并在无法确认二者等价时暂停该 episode、将相关区间标为 ambiguous，并在 `notes` 记录；程序不做语义替换。

严禁用 `saved=true`、episode 末帧、`last_valid_frame`、文件名或 episode 成功标签推断完成帧。47 条 episode 的 `task_outcome=success`、`outcome_source=retrospective_operator_attestation` 只是 episode 级上下文，不是逐帧标签。

## 每条 episode 必填项

- `last_confident_incomplete_frame`
- `first_confident_complete_frame`
- `hard_negative_intervals`
- `ordinary_negative_intervals`
- `ambiguous_intervals`
- `completion_visible`
- `completion_stable`
- `positive_available`
- `reviewer_id`
- `review_timestamp`
- `confidence`
- `notes`

区间采用闭区间 JSON 表示，如 `[[120, 184], [220, 231]]`。区间必须在该 episode 的 frame 范围内且互不重叠。边界不可信的帧放入 `ambiguous_intervals`，不要勉强归类。

如果成功后立即停止、双相机中没有可信的 completed observation，设置 `positive_available=false`；不得把末帧强制标成 positive。`completion_visible` 或 `completion_stable` 不成立时，也必须在 notes 解释可观察性限制。

## 分组与可用性门槛

沿用第一阶段冻结 split，episode 不得移动：train 38、val 5、test 4。同一 episode 的任何 frame/row 不得跨 split。test 在训练和阈值选择期间不可见。

人工审核完成后按 split 统计：

- positive episode count
- ordinary-negative episode count
- hard-negative episode count
- ignored frame count

若 train 或 validation 缺少可信 positive 或 hard negative，则 `EXISTING_TASK2_CLASSIFIER_DATA_READY=no`，不得开始训练。

`development_heldout = episode_disjoint_within_task2`；`formal_heldout = independent_collection_run`。独立 collection 不是 development-only 训练申请的绝对前置条件，但论文 formal evaluation 前必须具备。

## 数据不足时的 classifier-only 补采

补采写入独立目录，不混入 `task2_lerobotv3`，也不要求重新采集完整 VLA demonstration：

- 成功后保持 1–2 秒的完整插装；
- 对齐但未插入；
- 接触但未插入；
- 部分插入后主动终止；
- 不同初始位置与独立 collection run。

本协议不批准 Reward Classifier 训练、threshold、consecutive-positive frames、最大检测延迟、task2 reward/terminal、G1 或 Twin-Q。
