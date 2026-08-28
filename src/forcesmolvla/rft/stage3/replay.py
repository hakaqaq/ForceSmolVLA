"""Canonical payload store with logical R_online / D_expert memberships."""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Literal, Mapping

from .transition import canonical_json_bytes, validate_ack_transition
from .update_credit import UpdateCreditLedger


R_ONLINE = "R_online"
D_EXPERT = "D_expert"
Origin = Literal["online", "offline_demonstration"]


class ReplayDigestCollisionError(RuntimeError):
    pass


@dataclass(frozen=True)
class ReplayCommitResult:
    transition_uid: str
    new_payload: bool
    added_memberships: tuple[str, ...]
    credit_minted: bool
    idempotent_noop: bool
    evicted_online_uid: str | None


def memberships_for_transition(payload: Mapping, *, origin: Origin) -> tuple[str, ...]:
    if origin == "offline_demonstration":
        return (D_EXPERT,)
    if origin != "online":
        raise ValueError("STAGE3_REPLAY_ORIGIN_INVALID")
    owners = payload["behavior_ack"]["slot_owner"]
    return (R_ONLINE, D_EXPERT) if "human_intervention" in owners else (R_ONLINE,)


class Stage3Replay:
    """In-memory G2 replay; persistence and WAL transactions are deferred to G3+."""

    def __init__(
        self,
        *,
        max_online_transitions: int,
        credit_ledger: UpdateCreditLedger | None = None,
    ) -> None:
        if max_online_transitions <= 0:
            raise ValueError("STAGE3_REPLAY_ONLINE_CAPACITY_INVALID")
        self.max_online_transitions = int(max_online_transitions)
        self.credit_ledger = credit_ledger
        self._payload_bytes: dict[str, bytes] = {}
        self._digests: dict[str, str] = {}
        self._memberships: dict[str, dict[str, None]] = {R_ONLINE: {}, D_EXPERT: {}}

    def commit(self, payload: Mapping, *, origin: Origin) -> ReplayCommitResult:
        value = validate_ack_transition(payload)
        uid = value["identity"]["transition_uid"]
        digest = value["integrity"]["canonical_payload_sha256"]
        memberships = memberships_for_transition(value, origin=origin)
        existing = self._digests.get(uid)
        if existing is not None and existing != digest:
            raise ReplayDigestCollisionError(f"STAGE3_REPLAY_UID_DIGEST_COLLISION:{uid}")
        if existing == digest:
            return ReplayCommitResult(
                transition_uid=uid,
                new_payload=False,
                added_memberships=(),
                credit_minted=False,
                idempotent_noop=True,
                evicted_online_uid=None,
            )
        new_payload = existing is None
        if new_payload:
            self._payload_bytes[uid] = canonical_json_bytes(value)
            self._digests[uid] = digest
        added = []
        for pool in memberships:
            if uid not in self._memberships[pool]:
                self._memberships[pool][uid] = None
                added.append(pool)
        credit_minted = False
        if R_ONLINE in added and self.credit_ledger is not None:
            credit_minted = self.credit_ledger.mint_for_unique_online_transition(uid)
        evicted = self._evict_online_if_needed()
        return ReplayCommitResult(
            transition_uid=uid,
            new_payload=new_payload,
            added_memberships=tuple(added),
            credit_minted=credit_minted,
            idempotent_noop=not new_payload and not added,
            evicted_online_uid=evicted,
        )

    def _evict_online_if_needed(self) -> str | None:
        if len(self._memberships[R_ONLINE]) <= self.max_online_transitions:
            return None
        uid = next(iter(self._memberships[R_ONLINE]))
        del self._memberships[R_ONLINE][uid]
        if uid not in self._memberships[D_EXPERT]:
            del self._payload_bytes[uid]
            del self._digests[uid]
        return uid

    def membership_uids(self, pool: str) -> tuple[str, ...]:
        if pool not in self._memberships:
            raise KeyError(f"STAGE3_REPLAY_POOL_UNKNOWN:{pool}")
        return tuple(self._memberships[pool])

    def get_payload(self, transition_uid: str) -> dict:
        return json.loads(self._payload_bytes[transition_uid])

    @property
    def canonical_payload_count(self) -> int:
        return len(self._payload_bytes)

    def audit(self) -> dict:
        overlap = set(self._memberships[R_ONLINE]) & set(self._memberships[D_EXPERT])
        return {
            "canonical_payload_count": self.canonical_payload_count,
            "R_online_membership_count": len(self._memberships[R_ONLINE]),
            "D_expert_membership_count": len(self._memberships[D_EXPERT]),
            "dual_membership_count": len(overlap),
            "canonical_payload_copies_per_uid": 1,
        }
