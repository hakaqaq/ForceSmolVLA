# P8 v4.2 r4 implementation/preflight report

日期：2026-08-21（Asia/Shanghai）  
状态：`DEVELOPMENT_GATE_PASS / P9_NOT_STARTED`  
acceptance：`development_only`，`formal_eligible=false`

## 结论

P8 r4 已通过。该原子 gate 在真实 task2 数据上执行一次 B4×1 single-pass 全参数更新，先写完整 payload manifest，再释放主进程模型并从空 HF cache 启动 fresh process。strict local reload、checkpoint fileset/hash、fixed validation、optimizer/scheduler/scaler、Python/NumPy/PyTorch CPU/CUDA RNG、sampler continuation、accumulation phase 和 Force cached parity 均满足契约。

`long_development_sft_unlocked=true`，但这只解锁 development SFT。P9 未启动、训练未启动、机器人动作发送数为 0；formal resolver 因缺少 trusted detached signature/approval 按预期 fail-closed。

## P8 实现与约束

- 新增独立 `configs/p8_parity_acceptance.development.json`，避免修改 P4 acceptance config 而使 P4→P7 prerequisite 失效。
- development FP32 使用 `atol=1e-5, rtol=0`；BF16 valid-prefix hidden 使用 `atol=0.3`，velocity/cache/10-step 使用 `atol=0.1, rtol=0`。formal P8 字段保持 null/unapproved。
- P8 parity 覆盖真实 Force-MoE：full/prefill hidden、cached/uncached velocity、完整 10-step denoise、zero-init shared-fp32 projection、7D/25D padding 隔离、invalid horizon tail、prefix layout/mask/physical length、cache append→crop 与 K/V snapshot。
- Force K/V 投影调用数必须严格为每个 chunk `k=1, v=1`；结构契约及无效输出使用 exact 判断，不受数值容差放宽。
- P8 source binding 绑定 P7 r4 五项 prerequisite、实际 import roots、30 个测试源码、JUnit 报告、task2 的 51 文件 storage tree、P8 代码与配置。
- P8 checkpoint contract 要求 self-contained 82 payloads、exact fileset/size/SHA256、无 symlink、embedded constructor assets、严格本地加载和 formal fail-closed。

## 测试与 parity

- P8-and-upstream tests：`138 passed, 0 failed, 0 errors, 0 skipped`。
- JUnit：`artifacts/development/p8_v4_2_r3_pytest.xml`，SHA256 `1373f3a5ab2f9c945144724eb20d810104c5d83a16095a3f6ffdcb998d086aba`。
- source binding：`artifacts/development/p8_v4_2_r3_source_binding.json`，SHA256 `30f8eaad5b8894cf2c88053dd9e4db9a324593ce2f45e7e7acf9cca2b27a8147`。
- dataset storage tree：51 files，SHA256 `f9935b6479dc851e49444669065d20b8aef8cb3ad382f77f53391f701a55a58d`。
- FP32 artifact：SHA256 `8a473364734135221ea393508341cc7bab898fe57d755d2f3fc8227a2e911939`；valid-prefix error `0`；最大 velocity/cache error `2.86102294921875e-6 <= 1e-5`；10-step cached/uncached error `0`。
- BF16 artifact：SHA256 `0e24323eb55e17c6e3dea779e28076e07101e93c8d559bacd163bc2eb3ea9dec`；valid-prefix error `0.25 <= 0.3`；最大 velocity/cache error `0.06651902198791504 <= 0.1`；10-step cached/uncached error `0.00947539508342743 <= 0.1`。
- invalid velocity 误差、state/action/noise padding 扰动、invalid tail 扰动均为 `0`；所有结构检查通过。

## 真实 B4×1 update

- GPU：NVIDIA GeForce RTX 4090 D，24 GiB。
- 数据：`datasets/task2_lerobotv3`，repo id `local/task2_lerobotv3`，train split；双相机，`H=50`。
- batch：4 samples，1 microbatch，1 optimizer step；sample indices=`[20952, 3648, 819, 24299]`。
- algorithm：`single_pass_batch_local`；CPU fallback=false；architecture downgrade=false；全部参数可训练。
- valid flow features=`1400`；valid router tokens=`572`；route counts=`[4,15,549,4]`。
- loss：flow=`10.574432373046875`，balance=`2.597001791000366`，z=`3.667609691619873`，weighted total=`10.604070663452148`。
- gradient norm before clip=`103.3202133178711`。
- CUDA/wall latency=`1266.375 ms / 1.266444 s`。
- peak allocated/reserved=`6,304,516,608 / 6,916,407,296 bytes`，约 `5.87 / 6.44 GiB`。该值仅为 P8 单步 preflight，不代表长程训练峰值。

## Checkpoint 与 cold reload

- checkpoint：`outputs/development/p8_v4_2_r4_checkpoint_seed42_step000001`。
- payload count=`82`；artifact manifest SHA256 `91a1e2cfeaa8edc510c66eecb3e8b9ec92b17461d917d910be89ea052d8ae10a`；model SHA256 `9208415ae66aa9b6afad5b2202caa6c9cffb9aed9b8e5d2342524696e6ec8a99`。
- fresh-process gate：pass；`strict=true`，`local_files_only=true`，`force_download=false`，empty HF cache before=true。
- network attempts=`0`；Hub API attempts=`0`；Hub/model cache files after=`0`。
- exact resume dry-run=true；RNG continuation exact=true；sampler continuation exact=true；parity exact=true。
- source/cold fixed validation `L_flow=10.263299942016602`，exact。
- optimizer state SHA256 `d13ac19b300ff73b7f0b5eb93839a0249481ce049c0754d6fac232c40585a27e`。
- cold peak allocated/reserved=`4,340,162,048 / 4,429,185,024 bytes`。
- formal rejection=`FORMAL_FORCE_CHECKPOINT_SIGNATURE_OR_APPROVAL_MISSING`。

## 最终工件

- P8 acceptance config：SHA256 `e86f936c493cf80ad2c053b2e801e422634080978cdc1168b9b41c54362a6c2d`。
- P8 checkpoint contract：SHA256 `58f5945dbdad871dbd27e199afde42e33174221d659ebbc6ba0f1ded5f26631d`。
- resolved config：`artifacts/development/p8_v4_2_r4_resolved_config.json`，SHA256 `4415bfecba31b62aed6c585d33b713dbdc0e77c7b45c2c1dc2bf17cb604e0497`。
- cold result：`artifacts/development/p8_v4_2_r4_cold_start_result.json`，SHA256 `57d34df8bdb5880817d7d8799d6328693a1961ef19b48b44ccd612ba241e18c8`。
- final gate：`artifacts/development/p8_v4_2_r4_gpu_preflight.json`，SHA256 `27fd7846c380875a5969d8e54e919508a202e4e75a3be6a9e24af9cafd46ca24`。

## 剩余 blocker 与边界

1. formal P8 阈值仍未批准；development 容差不得传播到 formal。
2. trusted detached signature algorithm/key/approver/verifier 尚未冻结；formal checkpoint acceptance 继续 fail-closed。
3. P9 pure-offline record/replay Shadow 尚未启动，必须作为下一独立 gate 执行。
4. 本次按批准要求停在 P9 入口；未启动 P9，未启动 task2 长程训练，未发送机器人动作。

历史文件 `docs/p8_implementation_preflight_report.md` 描述早期 B2×8/task1 实现，已失效且仅供追溯；不得用于当前 acceptance、resume、checkpoint selection 或 SFT 解锁。
