# ForceRFT Stage-2 R0 Preparation Report

Status: `PREPARED_NOT_TRAINED`

## Authority and isolation

- Active Stage-2 source manifest: `artifacts/development/stage2/stage2_source_manifest.v4_r0prep.json`
- Manifest SHA256: `7ad7155290292b13b847c76de244d861a15b8c8c525b1fd014816f60928b5d49`
- Parent v4 manifest SHA256: `defa5b1d1a975c465154ac62e009863163947065127c557b5600025ce77b29eb`
- ConRFT repository: `/home/rlc123/conrft`
- ConRFT HEAD: `a779fde7fa5db5a469960a8490c100f35b41b49e`
- ConRFT worktree: clean and unmodified
- Isolated environment: `conrft_reward`
- Environment lock SHA256: `b9680391e91d1839258c79df6d1bdf17f12ea6a33e7c380bc6494821b27c3808`

Resolved runtime: Python 3.10.20, JAX/JAXlib 0.4.20, Flax 0.8.0,
Optax 0.1.5, NumPy 1.24.3, CUDA 11.8, cuDNN 8.9.6.50, and an RTX
4090 D. `pip check` reports no broken requirements when inherited ROS,
`PYTHONPATH`, and system CUDA library paths are excluded.

The fixed ConRFT commit's repository copy of `resnet10_params.pkl` is a
truncated pickle and was preserved unchanged. The runtime asset was fetched
from the public SERL release URL declared by that same fixed source in
`train_utils.py`; it is stored outside ConRFT and bound at SHA256
`175745d43d30233eb01b5369465d1c24c11b8ee71ccb734cc1c1bca13e07f57b`.

## Input contract

- `frame_stack = 1`; no past or future frame is accepted.
- Source camera order is `observation.images.camera1` (D435 third-person),
  followed by `observation.images.camera2` (D405 wrist).
- Source tensors are RGB uint8 CHW `[3,480,640]`.
- ConRFT tensors are RGB uint8 BTHWC `[1,1,480,640,3]` for each camera.
- The adapter does no resize, crop, padding, or normalization.
- ConRFT's ResNet owns bilinear resize to 128×128 and the only ImageNet
  normalization.
- Native ConRFT training augmentation remains random crop with padding 4 and
  two batch dimensions; deterministic evaluation applies no crop.
- Episode changes reset adapter identity; retained frame count is zero.

The adapter read one real embedded-PNG v3 row without writing or copying an
image into an output dataset. Repeated adapter output was exact.

## Synthetic classifier evidence

Unmodified ConRFT `create_classifier()` created one synthetic Flax TrainState.
The two camera keys were both observed by the forward pass. Adapter and direct
ConRFT logits were exactly equal, repeated logits were exactly equal, and a
future-frame mutation did not affect the current result.

- Synthetic logit: `-0.431196391582489`
- Synthetic probability: `0.3938406705856323`
- BCE loss: `0.5006123781204224`
- D435 forward sensitivity max-abs: `0.3071921169757843`
- D405 forward sensitivity max-abs: `0.3645095229148865`
- Frozen pretrained ResNet gradient: exact zero
- Trainable classifier gradient: nonzero
- TrainState step: 0 before and after
- Parameter SHA256: exact before/after
- Optimizer updates: 0
- Checkpoint saved: no
- Octo/SERL ReplayBuffer/Franka Gym instantiated: no

## Reward Detector preparation

Development configuration leaves `probability_threshold`,
`consecutive_positive_frames`, and `max_detection_delay_frames` null.
`last_valid_frame_fallback` is disabled. Synthetic-only tests cover causal
streak confirmation, episode reset, no future access, no backfill to streak
onset, and forward alignment to the first 10 Hz boundary with 0–2 frame delay.

## Classifier data inventory and blocker

The inventory is complete but contains zero eligible frame labels in all three
classes (`positive`, `ordinary_negative`, `hard_negative`), zero classifier
train/val/test records, and no independent held-out collection. The 47 task2
episode outcome attestations are not frame labels and were not converted into
positives or terminals.

Before requesting R0 training approval, collection and human review must
provide:

1. episode- and collection-disjoint train, validation, and test frame labels;
2. a separate held-out collection used only for final acceptance;
3. reviewed positives that visibly satisfy completion;
4. ordinary negatives and hard negatives covering aligned/contact/partial-
   insertion states that are not complete;
5. complete row/camera identity, reviewer, timestamp, notes, and split
   provenance under the frozen schema.

The exact collection/sample-count target remains unapproved. No label may be
derived from `saved=true`, episode end, last valid frame, filename, gripper
state, or an image/action heuristic.

`BLOCKER = reward_classifier_labeled_data_missing`

## Frozen assets and regression disposition

- v3 data tree before/after SHA256:
  `daa3d3b876cddc25caa4effa1e7ac8c55e875738367304c4d51a18653118aa01`
- r5 checkpoint tree before/after SHA256:
  `9c8748bd62ed8ba76d7d25a22a02cfacb7d9e5889ef5c4ed0c8770f007a4dd42`
- Conversion, split, normalizer, and all P4–P9 artifact hashes were exact
  before and after.
- The old v4 manifest, G0 artifact, G3 artifact, and G3 precision matrix kept
  their approved SHA256 values.
- Historical P5/P7 source-binding failures are classified
  `HISTORICAL_EXPECTED_SOURCE_MISMATCH`; their historical snapshots were not
  rewritten.
- Two frozen-parent G0 tests reject the new append-only R0 files by design;
  they are reported as source-registry/allowlist mismatches, not behavior
  regressions. The old v4 closure independently validates, the new R0 closure
  passes, and 34 selected G0/G3/Stage-1 behavior regression tests pass.

No real classifier training, classifier checkpoint, task2 probability,
reward, terminal, G1 transition, Critic, target network, Cal-QL loss, or RFT
optimizer was created.
