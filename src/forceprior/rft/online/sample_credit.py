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
            raise ValueError("ONLINE_REPLAY_CREDIT_RATE_MUST_BE_POSITIVE")
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
            raise ValueError("ONLINE_REPLAY_CREDIT_UID_EMPTY")
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
                        raise CreditsUnavailable("ONLINE_REPLAY_LEARNER_BLOCKED_NO_CREDITS")
                    self._condition.wait(remaining)
            elif self.available < self.credits_per_joint_cycle:
                raise CreditsUnavailable("ONLINE_REPLAY_LEARNER_BLOCKED_NO_CREDITS")
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
            raise ValueError("ONLINE_REPLAY_CREDIT_STATE_INCONSISTENT")
        return ledger


@dataclass(frozen=True)
class TdCycleCreditSnapshot:
    unique_td_rows: int
    distinct_td_episodes: int
    allowed_cycles: int
    completed_cycles: int
    in_flight_cycle: int | None
    available_cycles: int


class TdCycleCreditLedger:
    """Cumulative unique-TD budget for asynchronous joint training cycles."""

    SCHEMA = "forceprior-td-cycle-credit-ledger-v1"

    def __init__(self, *, new_td_rows_per_cycle: int) -> None:
        if new_td_rows_per_cycle < 1:
            raise ValueError("FORCERFT_TD_CREDIT_RATE_INVALID")
        self.new_td_rows_per_cycle = int(new_td_rows_per_cycle)
        self._admissions: dict[str, dict[str, object]] = {}
        self._credited_td_uids: set[str] = set()
        self._in_flight_cycle: int | None = None
        self._lock = threading.RLock()

    def register_admission(
        self,
        *,
        admission_id: str,
        episode_id: str,
        td_uids: set[str],
    ) -> bool:
        if not admission_id or not episode_id or "" in td_uids:
            raise ValueError("FORCERFT_TD_CREDIT_ADMISSION_INVALID")
        record = {
            "episode_id": episode_id,
            "critic_td_valid_rows": len(td_uids),
            "td_uids": sorted(td_uids),
        }
        with self._lock:
            existing = self._admissions.get(admission_id)
            if existing is not None:
                if existing != record:
                    raise ValueError("FORCERFT_TD_CREDIT_ADMISSION_CHANGED")
                return False
            overlap = self._credited_td_uids.intersection(td_uids)
            if overlap:
                raise ValueError("FORCERFT_TD_CREDIT_UID_DUPLICATE")
            self._admissions[admission_id] = record
            self._credited_td_uids.update(td_uids)
            return True

    def startup_ready(
        self, *, minimum_td_rows: int, minimum_episodes: int
    ) -> bool:
        snapshot = self.snapshot(completed_cycles=0)
        return (
            snapshot.unique_td_rows >= minimum_td_rows
            and snapshot.distinct_td_episodes >= minimum_episodes
        )

    def snapshot(self, *, completed_cycles: int) -> TdCycleCreditSnapshot:
        with self._lock:
            unique_td_rows = len(self._credited_td_uids)
            episodes = {
                str(record["episode_id"])
                for record in self._admissions.values()
                if int(record["critic_td_valid_rows"]) > 0
            }
            allowed = unique_td_rows // self.new_td_rows_per_cycle
            reserved = int(completed_cycles) + int(
                self._in_flight_cycle is not None
            )
            return TdCycleCreditSnapshot(
                unique_td_rows=unique_td_rows,
                distinct_td_episodes=len(episodes),
                allowed_cycles=allowed,
                completed_cycles=int(completed_cycles),
                in_flight_cycle=self._in_flight_cycle,
                available_cycles=max(0, allowed - reserved),
            )

    def reserve_cycle(self, *, completed_cycles: int) -> bool:
        with self._lock:
            expected = int(completed_cycles) + 1
            if self._in_flight_cycle is not None:
                if self._in_flight_cycle != expected:
                    raise ValueError("FORCERFT_TD_CREDIT_IN_FLIGHT_INVALID")
                return True
            if self.snapshot(completed_cycles=completed_cycles).available_cycles < 1:
                return False
            self._in_flight_cycle = expected
            return True

    def complete_cycle(self, *, completed_cycle: int) -> None:
        with self._lock:
            if self._in_flight_cycle != int(completed_cycle):
                raise ValueError("FORCERFT_TD_CREDIT_COMPLETION_INVALID")
            self._in_flight_cycle = None

    def state_dict(self) -> dict[str, object]:
        with self._lock:
            return {
                "schema": self.SCHEMA,
                "new_td_rows_per_cycle": self.new_td_rows_per_cycle,
                "admissions": {
                    key: dict(value) for key, value in self._admissions.items()
                },
                "in_flight_cycle": self._in_flight_cycle,
            }

    @classmethod
    def from_state_dict(cls, state: Mapping) -> "TdCycleCreditLedger":
        if state.get("schema") != cls.SCHEMA:
            raise ValueError("FORCERFT_TD_CREDIT_SCHEMA_INVALID")
        ledger = cls(new_td_rows_per_cycle=int(state["new_td_rows_per_cycle"]))
        admissions = state.get("admissions")
        if not isinstance(admissions, Mapping):
            raise ValueError("FORCERFT_TD_CREDIT_STATE_INVALID")
        for admission_id, value in admissions.items():
            if not isinstance(value, Mapping):
                raise ValueError("FORCERFT_TD_CREDIT_STATE_INVALID")
            td_uids = [str(uid) for uid in value["td_uids"]]
            if len(td_uids) != len(set(td_uids)):
                raise ValueError("FORCERFT_TD_CREDIT_STATE_INVALID")
            ledger.register_admission(
                admission_id=str(admission_id),
                episode_id=str(value["episode_id"]),
                td_uids=set(td_uids),
            )
            if int(value["critic_td_valid_rows"]) != len(td_uids):
                raise ValueError("FORCERFT_TD_CREDIT_STATE_INVALID")
        in_flight = state.get("in_flight_cycle")
        ledger._in_flight_cycle = None if in_flight is None else int(in_flight)
        return ledger
