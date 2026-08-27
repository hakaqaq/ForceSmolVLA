# R0 one-shot development test evaluation

- Decision: **FAIL**
- One-shot test artifact SHA256: `ec02e805c84a80778be9e16525d751da058a28bd1ebde85f394b33875b6a3988`
- DetectorSpec/disposition SHA256: `d8e586575b5cd64fb19277f6b1770c42e1e85ee530ae953ce1c200c6caf4f257`
- Frozen checkpoint SHA256: `6b4e366baa55993d150cb3dd86e67a1d708e58d836b123a0c433190835021510`
- Frozen candidate SHA256: `d493c9f398a2f14ae5e11d1d1cf44ef769c66759c61220eed53e00eedb2d3362`
- Frozen validation calibration SHA256: `5d52475ce518eef2315bbf6908140d318d89d025311efdb3f7e8c8204d6bdb47`
- Scope: four frozen development test episodes; exactly one frozen candidate; no reselection.

## Frozen detector

`tau=0.83`, `M=5`, `30 Hz`, causal current-frame trigger, latch enabled.

## Episode results

| episode | completion | trigger | delay frames | delay ms | early | missed | pre max run | post min run | post max run |
|---|---:|---:|---:|---:|:---:|:---:|---:|---:|---:|
| episode_000005 | 747 | 744 | -3 | -100.000 | yes | no | 7 | 64 | 64 |
| episode_000021 | 694 | 698 | 4 | 133.333 | no | no | 0 | 23 | 23 |
| episode_000025 | 742 | 741 | -1 | -33.333 | yes | no | 5 | 35 | 35 |
| episode_000033 | 713 | 717 | 4 | 133.333 | no | no | 0 | 22 | 22 |

## Acceptance

- early triggers: 2 (required 0)
- missed successes: 0 (required 0)
- maximum delay: 4 frames / 133.333 ms (required <=6 / <=200.0)

## Frame metrics at frozen tau

- BCE: 0.014604929
- ROC-AUC: 0.999769797
- PR-AUC: 0.995316872
- balanced accuracy: 0.997928177
- positive recall: 1.000000000
- ordinary-negative FPR: 0.000000000
- hard-negative FPR: 0.016460905
- confusion matrix: {'true_negative': 2884, 'false_positive': 12, 'false_negative': 0, 'true_positive': 144}
- longest pre-completion positive run: 7
- shortest post-completion positive run: 22
- shortest sustained post-completion run across episodes: 22

The JSON artifact contains all 3,040 original-order frame probabilities and exact metric definitions.

## Audit and status

- Test GPU inference invocation count: 1; frozen candidate parameter sets evaluated: 1.
- One sandboxed CPU-backend preflight stopped before model creation with 0 test frames inferred; its evidence is preserved.
- Eval mode only; no dropout, crop, random augmentation, or optimizer update.
- No train/validation image was opened by this run; no reward/terminal, G1/G2, critic, Cal-QL, Actor, online, or robot artifact was created.

```text
DETECTOR_CANDIDATE_APPROVED_FOR_ONE_SHOT_TEST = yes
ONE_SHOT_TEST_EVALUATION = complete
ONE_SHOT_DEVELOPMENT_TEST_ACCEPTANCE = FAIL
DEVELOPMENT_DETECTOR_SPEC_APPROVED = no
FORMAL_DETECTOR_SPEC_APPROVED = no
PRODUCTION_DETECTOR_SPEC_APPROVED = no
CLASSIFIER_CHECKPOINT_FROZEN = yes
CLASSIFIER_RETRAINED = no
OPTIMIZER_UPDATES = 0
TEST_EVALUATED = yes_once
TASK2_REWARD_TERMINAL_CREATED = no
G1_CREATED = no
G2_CREATED = no
NEXT_ALLOWED_ACTION = return_to_validation_redesign_or_collect_independent_calibration_episodes
```
