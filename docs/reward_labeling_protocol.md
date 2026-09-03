# Reward frame labeling protocol

人工标注文件只需要包含明确选中的 episode 子集，不要求覆盖采集数据全集。建议首轮约 20 条，并将训练 episode 与验证 episode 分开；每个子集都应覆盖 positive、ordinary negative 和 hard negative。

Task2 的任务语义是：`Pick up the purple ring and place it onto the red peg.`

正样本必须能从双相机确认：红色 peg 明确穿过紫色 ring 的中心孔，ring 已离开夹爪并稳定留在 peg/base 上。仅仅接触、对齐、仍由夹爪支撑或释放后状态不稳定，都不是正样本。看不清或两路相机证据冲突时标为 ambiguous。

分类器训练只读取 `labels/{task_id}_reward_frame_labels.json` 中列出的 episode。未列出的 episode 不参与人工监督；奖励器冻结后，完整的 `{task_id}_lerobotv3` 数据集仍会被自动打分。
