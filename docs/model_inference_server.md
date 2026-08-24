# ForceSmolVLA 模型推理服务

本工程只负责模型推理：严格加载 ForceSmolVLA checkpoint、校验随
checkpoint 保存的数据/normalizer/几何工件、构造模型输入并返回可执行语义的
absolute action7。这里不包含 ROS、相机、Franky 或 HIL-SERL 控制代码。

当前 checkpoint 和 RuleSpec 都是 `development_only`。默认服务只接受 P9 的
test-only RuleSpec，并发布 `robot_execution_allowed=false`；它可用于无动作预检，
不能用于真机执行。formal/production 仍然 fail-closed。

## 启动模型服务

```bash
source /home/rlc123/anaconda3/etc/profile.d/conda.sh
conda activate forcesmolvla
cd /home/rlc123/ForceSmolVLA

export CUDA_VISIBLE_DEVICES=0
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

python tools/serve_policy.py \
  --host 127.0.0.1 \
  --port 8000 \
  --checkpoint /home/rlc123/ForceSmolVLA/outputs/development/task2_lerobotv3_full_sft_10k_r3/checkpoints/step_010000
```

服务仅监听 loopback，避免在没有受信 clock map、认证和传输保护时跨机器暴露。
模型输入固定为 D435 camera1、D405 camera2、measured TCP state7、按 v4.1
causal ZOH 几何和固定 SOS 得到的 wrench6、prompt。输出固定为 `[50,7]`：

```text
[target_x, target_y, target_z,
 target_roll, target_pitch, target_yaw,
 target_gripper_width_m]
```

public `predict_action_chunk()` 已在模型进程内完成 delta unnormalize、absolute
inverse 和 action safety 检查。机器人客户端不得再运行第二个 LeRobot action
postprocessor。

## HTTP 边界

- `GET /healthz`：模型及绑定 metadata。
- `GET /metadata`：checkpoint、数据、相机、标定、滤波和时钟契约。
- `POST /infer`：一次 atomic observation，返回一个 H=50 absolute7 chunk。

HTTP 只是同机进程边界。异步性位于机器人客户端：一个正在推理的请求加一个
latest-only pending observation，并通过带时间戳的 action queue 消费结果。模型
自身不启用 LeRobot RTC，也不修改 SmolVLA 原生 prefix cache。

