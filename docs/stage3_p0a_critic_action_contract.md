# Stage-3 P0-A Critic Action Contract

Status: P0-A closed on `stage3-online-hil`.  Formal parity is computed at runtime
from hash-bound recorded-live evidence; a stale binding or synthetic fixture
returns `BLOCKED`.

## Canonical contract

The sole production contract is `CriticActionContract` in
`forcesmolvla.rft.critic_action_adapter_v2`.  Its version is persisted on every
Critic transition.  Constants copied into a replay loader, bridge, or trainer are
not an alternative contract.

- Reference grid: strict rational 30 Hz monotonic grid.
- Execution grid: 10 Hz.
- Critic action: `K=3`, seven features per slot.
- Nominal transition: ticks `t`, `t+1/30 s`, `t+2/30 s`; next observation at
  `t+3/30 s`, exactly 100,000,000 ns after `t`.
- Behavior authority: identifier-matched, controller-accepted Pose and gripper
  ACKs.  Proposals, requests, and rejected/missing/stale ACKs are never behavior.
- Temporal projection: latest causal ACK at each valid tick, then zero-order
  hold.  A repeated ACK ID denotes a real held command; it is not a newly minted
  ACK.
- TCP6: accepted absolute target converted to current-anchor-relative delta,
  with canonical orientation handling from `ActionDeltaProcessor`.
- Gripper: accepted canonical absolute endpoint.  Actor Q projection is binary
  and stop-gradient for this feature.
- Normalization: the frozen parent `delta_action7` normalizer, exactly once.  Its
  manifest hash is transition provenance.

## Command-effective anchor

Autonomous-policy Critic transitions start at the first rational 30 Hz
observation tick at which the controller-accepted command is already effective.
The accepted 10 Hz command is held through the three Critic slots.  Therefore the
Actor and bootstrap-Actor candidate mapping is `[candidate, candidate,
candidate]`, derived by the shared rational execution selector rather than by a
literal index tuple.  A policy replay transition whose ACK phase or subsequent
authority change violates that command-effective interval is quarantined; it is
not treated as approximately aligned.

The bridge persists both the number and ratio of policy transitions quarantined
for a mid-macro effective-command change.  The recorded-live positive fixture
has zero such quarantines; this is an exact contract, not an approximation.

The current observation is the observation at that effective tick, the reward is
for the same physical interval, and the nominal next observation is exactly the
fourth rational tick.  Actor Q-guidance and bootstrap TD use the identical
candidate projection.

## Behavior mappings

Policy and human behavior both call `build_ack_behavior_macro`.  It performs the
same grid construction, ACK identity/age checks, causal selection, ZOH,
absolute-to-anchor-relative conversion, endpoint canonicalization, normalization,
mask construction, and lineage validation.  Policy macros use command-effective
10 Hz anchors.  Human commands may refresh inside the 100 ms macro; their real
accepted ACK identities are retained per slot.

Offline demonstrations use the same 30 Hz three-slot coordinates, normalizer,
prefix behavior mask, and 100 ms outcome semantics.  Their accepted converted
actions have `offline_demonstration` authority and the dataset/normalizer
manifests provide provenance.

Human supervision has two deliberately separate values:

- `human_action_target_h50` and `human_action_valid_mask_h50` are sparse masked
  Flow-Matching supervision.  Missing features remain masked and receive no FM
  gradient.
- `human_behavior_action_k3` and `human_behavior_mask_k3` are ACK-authoritative
  Critic TD inputs.  They are never sliced out of the H50 target.

## Partial terminal and boundary macros

The global policy is a masked shortened macro.  The tensor shape remains `[3,7]`
and the behavior mask is a non-empty prefix.  If a terminal boundary follows two
executed slots, the mask is `[true,true,false]`; invalid slots use deterministic
zero padding after normalization and cannot affect Q.  The next observation is
the real boundary, `terminated=true`, `truncated=false`, `bootstrap_mask=false`,
and `discount=0`.

An intervention boundary remains a truncation, not an environment terminal:
`terminated=false`, `truncated=true`, `bootstrap_mask=false`, and `discount=0`.
Neither Actor nor target Critics are evaluated for it.  No reward after takeover
can bootstrap through the boundary.

Ordinary full transitions use `terminated=false`, `truncated=false`,
`bootstrap_mask=true`, and `discount=0.99`, representing one nominal 100 ms
decision.

## Fail-closed lineage

A macro cannot cross episode, action source, takeover generation, reset
generation, policy revision, clock domain, controller authority, or an
incompatible chunk boundary.  Missing/rejected/stale ACKs and invalidated policy
suffixes are excluded.  Compatible accepted chunk refreshes are permitted only
when controller authority and every other lineage field match; actual per-slot
ACK/chunk provenance is retained.

Every persisted transition records the contract version, transition and episode
identity, source, revision and generations, chunk, anchor/grid/next timestamps,
source command/ACK/dispatch/model identities, behavior mask, duration, discount,
and normalizer manifest hash.

## Recorded-live gate

Synthetic fixtures remain unit-test inputs and are categorically ineligible to
unlock the formal gate.  The formal parity evidence must bind immutable hashes
and real monotonic/ACK provenance for policy behavior, human behavior, offline
demo behavior, Actor candidate, and bootstrap candidate.  The gate remains
blocked if any path, identity check, numeric comparison, source-tree binding, or
normalizer binding is missing or stale.  The parity tool never connects to or
commands a robot.

The accepted positive evidence is
`golden_fixtures/stage3_p0a_recorded_live_evidence.v1.json`, derived from
`datasets/online/000/episodes/episode_000000`.  It reconstructs 517 policy and
83 human transitions, checks one real partial terminal transition and one real
offline-demo partial transition, and verifies recorded Actor/bootstrap
candidates against the shared projection.  Its formal gate result is `PASS` and
its robot-command count is zero.
