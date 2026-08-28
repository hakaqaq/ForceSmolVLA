# Stage-2 throughput-v2 timing audit v2

结论：这是可解决的训练系统工程瓶颈，不是 ForceRFT 算法不可行，也不是 RTX 4090D 性能异常。旧代码是 development acceptance 路径，完整性检查正确，但不适合作为 steady-state long-run 内循环。

## 已测周期归属

| 部分 | 平均时间 | 占比 |
|---|---:|---:|
| 同步数据构造 | 60.84 s | 48.06% |
| Critic policy Flow | 57.64 s | 45.53% |
| Critic forward/backward | 0.91 s | 0.71% |
| Actor FM + Q backward | 0.78 s | 0.62% |
| Polyak | 0.67 s | 0.53% |
| SHA、GC、同步及其他审计 | 5.76 s | 4.55% |

计时使用 `time.perf_counter()`；cycle、Flow、Actor 和 Critic GPU 阶段具有 `torch.cuda.synchronize()` 边界。数据构造前序 GPU 工作已同步，H2D 又是 blocking `.to(device)`，因此 60.84 s 不是 CUDA 异步任务误记到 data loading。历史运行没有分别插桩 Parquet、Pillow、transform、collate 与 H2D，报告不虚构这些子项秒数。

## 源码闭环

- 每个 cycle 调用 `build_batch()` 5 次，约 2144 次双相机 Pillow RGB decode；每次重新 `pq.read_table()`，没有 DataLoader、row cache、decoded-image cache、pinned memory 或 non-blocking H2D。
- Actor H=50 target 的未来行只需要 action，但旧路径仍读取相机列。
- C128 在 Flow subbatch=4 下，每个 Critic update 约 160 个 Flow 调用；每 cycle 两次 Critic update 即约 320 个 prefix prefill、3200 个 Euler velocity evaluation、1280 个 policy chunks。
- N=10 内 prefix/KV 只构造一次是正确的；重复发生在下一个 B4 subbatch、candidate 和 critic update。
- `module_state_sha256()` 对每个 tensor 执行 `detach().cpu()`。Actor 在每个 cycle 被完整 SHA 约 8 次；Polyak verifier 还逐 tensor clone/hash。每个 cycle 有 3 次 `gc.collect()` 与 3 次 `torch.cuda.empty_cache()`。

现有 B24/C128 的全周期 GPU mean 约 27.25%，而非观察窗口中的瞬时 50%。B24/C64 的 Actor-pass wall time 更短，但 Critic exposure 减半；这属于预算目标选择，不是 C128 算法错误。

## 优化边界

throughput-v2 可以把 row/image cache、CPU prefetch、冻结 prefix 候选复用、Flow B8/B16 与 grouped TD/Cal-QL scheduling 放到 append-only 路径，并把完整 SHA/逐 tensor Polyak 审计移到进程及 checkpoint 边界。N=10、H=50、M=2、2:1、B24/C128、loss、随机序列、ActionContract-v2 与梯度契约保持不变。

仅隐藏 60.84 s 数据等待的理论上限约为 1.925×。1.5–2× 是待 GPU benchmark 验证的假设，不是已实现结果。

```text
ALGORITHM_NOT_FEASIBLE = no
GPU_HARDWARE_INSUFFICIENT = no
SYSTEM_IMPLEMENTATION_BOTTLENECK = yes
BOTTLENECK_SOLVABLE = yes
ONE_LINE_CONFIGURATION_FIX = no
TRAINING_PIPELINE_REFACTOR_REQUIRED = yes
EXPECTED_SPEEDUP = approximately_1.5_to_2x_pending_benchmark
```
