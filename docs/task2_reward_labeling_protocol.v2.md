# task2 Reward Classifier 人工帧级审核协议 v2（development）

状态：`ACTIVE_FOR_MANUAL_LABELING`。v1 协议中的“插头完整插装”描述错误，保留为 historical、`INVALID_FOR_LABELING`，不得再用于人工审核。本协议只定义 Reward Classifier 候选标签，不生成 ForceRFT reward、terminal、MC return 或 G1 transition。

## 冻结任务语义

Canonical task prompt：

> Pick up the purple ring and place it onto the red peg.

Physical task description：

> Pick up the purple ring, align its center hole with the red peg, lower the ring over the peg, release it, and leave it stably supported by the red peg/base assembly.

Conversion manifest 中的 “purple ring onto red peg” 与 canonical task prompt 语义等价。该确认不要求重训 r5，也不要求重新转换 Stage-1 数据。

用户提供的示例视频约 23.87 秒、30 Hz；约 22 秒发生落位和释放，23 秒后终态稳定。该时间只验证任务语义，严禁直接转换为任何 LeRobot-v3 episode 的完成帧。精确 `first_confident_complete_frame` 必须由原始双相机 30 Hz row 人工审核确定。

## 数据与审核界面

- 唯一图像事实源：只读 `datasets/task2_lerobotv3`。
- 界面同步显示同一 parquet row 的 `D435 third-person`（`observation.images.camera1`）和 `D405 wrist`（`observation.images.camera2`）。
- 每帧显示 `episode_id`、`output_episode_index`、`split`、`frame_index`、`timestamp` 和 LeRobot-v3 row reference。
- Bundle 不复制图像。localhost GET-only server 按需读取 parquet 内嵌图像；人工输入只在浏览器内存中编辑并导出 JSON，不能回写数据集。
- 分类器训练候选以后可每 3 帧下采样为 10 Hz；Reward Detector 校准必须保留完整 30 Hz 序列。

启动：

```bash
python tools/reward_classifier/serve_task2_label_ui.py
```

访问 `http://127.0.0.1:8765`。服务器必须加载 `labels/task2_reward_frame_labels.v2.template.json` 和本协议。v1 模板不得被人工填写或用于训练。

## Positive：四项条件必须同时满足

`positive = 1` 必须同时满足：

1. 红色 peg 明确穿过紫色 ring 的中心孔；
2. ring 已经脱离夹爪，不再由夹爪支撑；
3. ring 稳定落在 red peg/base assembly 上；
4. 后续可观察帧中 ring 未滑落、弹出或重新被抓取。

`first_confident_complete_frame` 是上述四项首次同时得到双相机支持的帧。不得仅因 ring 与 peg 接触、同轴对齐或部分套入就标记 positive。

## Negative 与 ignore

`ordinary_negative = 0`：

- ring 仍放置在桌面；
- 尚未成功抓取；
- ring 与 peg 明显分离；
- ring 明显偏离 peg；
- 机器人仍处于普通搬运阶段。

`hard_negative = 0`：

- ring 已移动到 peg 上方；
- ring 与 peg 近似同轴；
- ring 已经接触 peg；
- ring 部分套入 peg；
- ring 看似落位但仍由夹爪抓持或支撑；
- ring 刚被释放但仍在移动、倾斜或稳定性无法确认。

`ambiguous/ignore`：

- 相机遮挡，无法确认 peg 是否穿过中心孔；
- 无法确认夹爪是否仍在支撑 ring；
- 双相机结论不一致；
- 无法判断 ring 是否已经稳定。

Ambiguous 帧不得进入 classifier 训练、threshold 选择或指标统计。

## 每条 episode 的人工字段

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

区间采用闭区间 JSON，如 `[[120, 184], [220, 231]]`，必须处于 episode frame 范围内且互不重叠。边界不可信的帧必须放入 `ambiguous_intervals`。

如果 episode 在成功后立即停止，双相机中没有足够后续帧验证条件 4，则设置 `positive_available=false`；不得强制把末帧标成 positive。47 条 episode 的 retrospective success attestation 只是 episode 级上下文，不是逐帧标签。

严禁用 `saved=true`、episode 末帧、`last_valid_frame`、文件名、episode success 标签或示例视频时间推断完成帧。

## Split 与停止线

沿用 Stage-1 冻结 episode split：train 38、val 5、test 4。同一 episode 的 frame/row 不得跨 split；test 在训练和 threshold 选择期间不可见。

人工审核完成后，必须按 split 统计 positive、ordinary-negative、hard-negative episode count 与 ignored frame count。若 train 或 validation 缺少可信 positive 或 hard negative，则 `EXISTING_TASK2_CLASSIFIER_DATA_READY=no`。

`development_heldout = episode_disjoint_within_task2`；`formal_heldout = independent_collection_run`。本协议不批准 Reward Classifier 训练、optimizer update、checkpoint、threshold、consecutive-positive frames、task2 reward/terminal、G1、G2 或 Twin-Q。
