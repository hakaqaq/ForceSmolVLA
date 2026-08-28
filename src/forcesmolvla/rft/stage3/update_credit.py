"""Sample-credit ledger: unique committed online data bounds learner work."""

from __future__ import annotations

from dataclasses import dataclass
import threading
import time
from typing import Mapping


class CreditsUnavailable(RuntimeError):
    pass


@dataclass(frozen=True)
class CreditSnapshot:
    minted: int
    consumed: int
    available: int
    credited_transition_count: int


class UpdateCreditLedger:
    def __init__(self, *, credits_per_transition: int, credits_per_joint_cycle: int) -> None:
        if credits_per_transition <= 0 or credits_per_joint_cycle <= 0:
            raise ValueError("STAGE3_CREDIT_RATE_MUST_BE_POSITIVE")
        self.credits_per_transition = int(credits_per_transition)
        self.credits_per_joint_cycle = int(credits_per_joint_cycle)
        self._minted = 0
        self._consumed = 0
        self._credited_uids: set[str] = set()
        self._condition = threading.Condition()

    @property
    def available(self) -> int:
        return self._minted - self._consumed

    def mint_for_unique_online_transition(self, transition_uid: str) -> bool:
        if not transition_uid:
            raise ValueError("STAGE3_CREDIT_UID_EMPTY")
        with self._condition:
            if transition_uid in self._credited_uids:
                return False
            self._credited_uids.add(transition_uid)
            self._minted += self.credits_per_transition
            self._condition.notify_all()
            return True

    def can_consume_joint_cycle(self) -> bool:
        with self._condition:
            return self.available >= self.credits_per_joint_cycle

    def consume_joint_cycle(self, *, block: bool = False, timeout: float | None = None) -> None:
        with self._condition:
            if block:
                deadline = None if timeout is None else time.monotonic() + timeout
                while self.available < self.credits_per_joint_cycle:
                    remaining = None if deadline is None else deadline - time.monotonic()
                    if remaining is not None and remaining <= 0:
                        raise CreditsUnavailable("STAGE3_LEARNER_BLOCKED_NO_CREDITS")
                    self._condition.wait(remaining)
            elif self.available < self.credits_per_joint_cycle:
                raise CreditsUnavailable("STAGE3_LEARNER_BLOCKED_NO_CREDITS")
            self._consumed += self.credits_per_joint_cycle

    def snapshot(self) -> CreditSnapshot:
        with self._condition:
            return CreditSnapshot(
                minted=self._minted,
                consumed=self._consumed,
                available=self.available,
                credited_transition_count=len(self._credited_uids),
            )

    def state_dict(self) -> dict:
        with self._condition:
            return {
                "credits_per_transition": self.credits_per_transition,
                "credits_per_joint_cycle": self.credits_per_joint_cycle,
                "minted": self._minted,
                "consumed": self._consumed,
                "credited_uids": sorted(self._credited_uids),
            }

    @classmethod
    def from_state_dict(cls, state: Mapping) -> "UpdateCreditLedger":
        ledger = cls(
            credits_per_transition=int(state["credits_per_transition"]),
            credits_per_joint_cycle=int(state["credits_per_joint_cycle"]),
        )
        ledger._minted = int(state["minted"])
        ledger._consumed = int(state["consumed"])
        ledger._credited_uids = {str(value) for value in state["credited_uids"]}
        if (
            ledger._minted != len(ledger._credited_uids) * ledger.credits_per_transition
            or ledger._consumed < 0
            or ledger.available < 0
            or ledger._consumed % ledger.credits_per_joint_cycle != 0
        ):
            raise ValueError("STAGE3_CREDIT_STATE_INCONSISTENT")
        return ledger
