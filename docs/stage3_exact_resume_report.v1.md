# Stage-3 G5P isolated learner exact-resume preflight

Status: `G5P isolated learner exact-resume preflight passed.`

This is a ForceRFT engineering safety extension. ConRFT does not provide the exact-resume implementation required here. It is not production online durable resume, policy publication, Critic warmup, GPU coexistence, online collection, or robot evidence.

## Audited interfaces

- `stage3/checkpoint.py`: the prior symbols `validate_online_checkpoint_metadata` and `cpu_round_trip_online_checkpoint` covered metadata only. G5P adds safetensors/restricted-state atomic save, validation, and strict load.
- `stage3/learner.py`: `ProvisionalStage3Learner.run_joint_cycle` is a CPU synthetic loopback learner; GPU G5P instead reuses the real G4P `_critic_step` and `_actor_step` ForceRFT primitives.
- `stage3/loopback.py`: `run_synthetic_loopback` remains G3P synthetic evidence and is not used as production replay.
- `canonical_state.py`: reuses `canonical_digest`, `module_record`, `optimizer_parameter_name_groups`, and `optimizer_record`.
- `exact_resume.py`: Phase-2 primitives were audited but their optimizer/RNG/sampler state was not inherited.

## Fresh subprocess evidence

- Branch A PID `2413687`: cycles 1 and 2 continuously.
- Branch B1 PID `2415071`: cycle 1, atomic checkpoint, full exit.
- Branch B2 PID `2416262`: strict checkpoint load and cycle 2.
- Disposable checkpoint: `/tmp/forcesmolvla_g5p_e5s7bryk/b1_disposable_exact_resume_checkpoint` (2573154713 bytes).
- Canonical content digest: `b0d24880e02f0eff3f18f22930b3fe8bbc1ebd8f9cfa9da825d27a08533d1058`.

All model, optimizer, RNG, loss/Q, captured gradient, parameter delta, Polyak, sampler, replay/credit/counter, and revision comparisons are exact. `allclose` is not an acceptance criterion.

## Limits

- `REAL_ONLINE_R_USED=false`; R is `synthetic_preflight_R_only` over the frozen real observation pipeline.
- `PRODUCTION_WAL_OUTBOX_RESUME_VALIDATED=false` and `G5_PRODUCTION_DURABLE_RESUME=UNVERIFIED`.
- No policy revision was activated; no ROS/network/robot path was entered.
- `G6_AND_LATER=NOT_RUN`.
