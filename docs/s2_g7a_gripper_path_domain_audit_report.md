# G7-A gripper path domain audit

`G7A_GRIPPER_PATH_DOMAIN_AUDIT = pass`; the result is `FAILURE_SCOPE = true_action_contract_error`, so G7-A r2 was not started.

The zero-update replay reproduced the first offending sample as:

```text
episode_id = episode_000017
row_id = 4480|episode_000017|222|225
sampling_purpose = calql_next_policy_candidate
candidate_index = 0
action_slot = 2
executed_action_mask = [true, true, true]
g_flow_normalized = 1.71746826171875
g_unnormalized_continuous_width_m = 0.09740415960550308
g_public_decoded_endpoint_m = null (public decoder rejected it)
g_critic_normalized = null (internal adapter aborted)
```

The `[-0.01, 0.095] m` comparison occurs after inverse normalization, on `g_unnormalized_continuous_width_m`. A normalized value was not compared directly with meter bounds.

The rejection occurs in the real Critic internal path:

```text
calql_next_policy_candidate
→ critic_action_for_q_guidance
→ copied public candidate-range check
→ ValueError
```

It is not a detached public-validity diagnostic. Public `predict_action_chunk`, absolute inverse, RuleSpec validation, and controller conversion were not called. The detached audit separately called `decode_binary_gripper_width`, recorded the rejection without replacement, and left Python/NumPy/torch CPU/CUDA RNG unchanged.

The current source matches the old hash-bound contract: `stage2_action_contract.development.json` names `forcesmolvla.action_delta.decode_binary_gripper_width` as the decode owner, and the frozen G3/G4 tests explicitly require out-of-range candidates to be rejected by `critic_action_for_q_guidance`. Therefore the newly frozen boundary principle conflicts with the existing G3/G4 contract. There is no existing no-public-validation internal adapter that G7-A can merely rewire to. Creating one in G7-A would reimplement the contract; changing the existing one would modify frozen G3/G4. Neither is authorized.

The permanent regression suite passed 25/25 tests, covering action shape/mask/padding, discrete critic endpoints, TCP/gripper gradient ownership, detached diagnostic reproducibility and the exact offending fixture.

R1 evidence remains byte-identical:

```text
r1 artifact SHA256 = 1a9b6aa20c88df7cacddfaf29bbfb6e7a18270461ccd18cb074101a3ee371001
r1 report SHA256   = bba3075124a0773bec31f2161ed2e513c52c3834e043a71489c76a46a1277cef
CRITIC_NUMERICAL_STABILITY = not_measured
```

No optimizer, Polyak, or Actor update ran. No r2 checkpoint or r2 source manifest was created. Public inference behavior and its safety threshold remain unchanged; no clipping or resampling was added.

```text
G7A_R1_FAIL = preserved
GRIPPER_PATH_DOMAIN_AUDIT = pass
FAILURE_SCOPE = true_action_contract_error
NORMALIZED_VALUE_COMPARED_AS_METERS = no
PUBLIC_INFERENCE_BEHAVIOR_CHANGED = no
PUBLIC_SAFETY_THRESHOLD_CHANGED = no
CLIPPING_OR_RESAMPLING_ADDED = no
G7A_R2_CRITIC_WARMUP = not_started
CRITIC_WARMUP_UPDATES = 0
ACTOR_UPDATES = 0
ETA_G7B_APPROVED = no
G7B_STARTED = no
LONG_RUN_AUTHORIZED = no
ROBOT_EXECUTION_AUTHORIZED = false
NEXT_ALLOWED_ACTION = request_action_contract_revision_approval
```
