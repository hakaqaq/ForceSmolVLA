# P9 v4.2 r4 implementation/preflight report

> 历史说明：r4 的 algorithmic replay 与原始 SHA256 仍可核验，但它早于 P9-only
> task2 scope/session-provenance amendment。当前 acceptance 证据为 r5；见
> `docs/p9_v4_2_r5_implementation_preflight_report.md`。该更新不否定 r4 的数值结果，
> 只使其不再作为当前 scope-binding 入口。

日期：2026-08-21（Asia/Shanghai）  
状态：`DEVELOPMENT_GATE_PASS / PRODUCTION_SHADOW_FAIL_CLOSED`  
acceptance：`development_only`，`formal_eligible=false`

## 结论

P9 r4 纯离线 record/replay Shadow gate 已通过。输入限定为用户确认的
`datasets/task2_lerobotv3`、P8 r4 development checkpoint，以及
`tests/fixtures`/`golden_fixtures` 中的 test-only 工件。该 gate 没有连接 ROS、RTC、
Franky queue、实时控制接口，也没有发送机器人动作。

真实推理输出为 finite absolute action7。test-only candidate 因冻结 fixture 中的
`SHADOW_END_TO_APPLY_EXCEEDED` 被拒绝，实际 dispatch indices 为空；record replay、
candidate validity/reasons 与 dispatch 全部 exact。production/formal Shadow 继续
fail-closed。

## 实现修订

- P9 config 独立绑定 P8 r4 source/resolved/cold/gate/checkpoint manifest 五项哈希，
  并固定 task2 数据目录和 checkpoint 路径。
- 新增 `tools/build_p9_source_binding.py`，绑定全部 147 项测试、实际 import roots、
  P9 源码/fixture、task2 manifest 与 51 文件 storage tree。
- task2 相机 timestamp 已是 host-monotonic；pose/wrench 是 device source stamp。
  P9 使用 conversion manifest 内每 episode 已冻结的整数 `clock_offset_ns` 执行
  `host_monotonic_ns = source_stamp_ns + clock_offset_ns`，再进入 synthetic test clock
  map；不使用浮点 age 反推时间戳。
- records/resolved/replay 改为所有 acceptance assertions 通过后才原子性落盘；最终
  gate artifact 最后写入。
- 没有修改 `src/forcesmolvla/shadow.py` 或任何 P8 已绑定源码，因此 P8 r4 哈希保持不变。

## 测试与绑定

- P9-and-upstream：`147 passed`。
- JUnit SHA256：`fcb4bf80f89e46c0c44f0d4858b2ad85e5881569cdac624365629d14d6d1ac04`。
- source binding SHA256：`009b9812a273a8e51610b72e2d360b052af432efe874c383c0d22ffd99d583e7`。
- task2 storage tree：51 files，SHA256
  `f9935b6479dc851e49444669065d20b8aef8cb3ad382f77f53391f701a55a58d`。
- P8 r4 gate/checkpoint manifest SHA256：
  `27fd7846c380875a5969d8e54e919508a202e4e75a3be6a9e24af9cafd46ca24` / 
  `91a1e2cfeaa8edc510c66eecb3e8b9ec92b17461d917d910be89ea052d8ae10a`。

## 真实离线 inference/replay

- GPU：NVIDIA GeForce RTX 4090 D。
- 输入：task2 val episode 7 frame 0；B=2，双相机，H=50，execution horizon=3。
- CUDA/wall latency：`590.478 ms / 0.590589 s`。
- peak allocated/reserved：`1,552,972,288 / 1,673,527,296 bytes`，约
  `1.45 / 1.56 GiB`。
- candidate valid=false；reasons=`[SHADOW_END_TO_APPLY_EXCEEDED]`。
- actual dispatched indices=`[]`；replay exact=true。
- absolute finite、candidate/reasons consistency、expected outcome、dispatch 和
  invalid-no-dispatch 共七项 acceptance assertions 全部为 true。
- ROS connected=false；RTC configured=false；native queue used=false；robot actions
  sent=0。

## 最终工件

- P9 config：SHA256 `54966af7b4c836849cc09d6b7e39e22efd7f0efa2badd79c73dea51ef7678c18`。
- resolved config：SHA256 `67e2f5cddbd4a2df0b97d11d27f96ec31f269a340cf34454adb7f3276cf13f65`。
- records：SHA256 `b0bf284346fddccc8d1c0d438680f184fe9d08b24599df7a98eba5f5b8d40263`。
- replay：SHA256 `ddfbec7753164697c8da29f82b9ea058f866b06b8839ba692b3e1d8df803136e`。
- final gate：SHA256 `7b301e8408d0f656a5e0275d93ad7a593d958dc54a22af326c9efedddebc0131`。

## 失败尝试记录

- r2 在写任何 records/replay/gate 前因 import-root helper 调用签名错误 fail-fast；只有
  JUnit 和 source binding，不能用于 acceptance。
- r3 replay exact，但暴露 task2 pose/wrench device source stamps 被错误当作
  host-monotonic，导致额外 `SHADOW_OBSERVATION_FROM_FUTURE`。没有生成 PASS；留下的
  records/replay/resolved 仅为诊断工件，不得用于 acceptance。
- r4 使用 conversion manifest 的整数 clock map 修复根因，并重新运行全部测试和绑定。

## Formal/production blocker

1. production sensor→controller 与 GPU→controller clock map 不存在。
2. Shadow safety thresholds 仍为 null/unapproved；test-only 数值不得提升为候选正式阈值。
3. trusted detached signature algorithm/key/approver/verifier 未冻结。
4. task2 结果仅是 within-session algorithmic development replay，不支持 production Shadow
   或跨 session 泛化结论。

P9 PASS 不授权实时连接、在线 HIL 或机器人动作发送；也没有自动启动 task2 长程训练。
