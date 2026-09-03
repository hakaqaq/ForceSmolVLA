#!/usr/bin/env python3
"""Summarize Twin-Q rankings from same-observation intervention probes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


ACTION_NAMES = (
    "pre_takeover_policy",
    "accepted_human_correction",
    "frozen_sft_reference",
    "current_actor",
)


def summarize_same_state_rankings(
    comparisons: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for comparison in comparisons:
        observation_id = str(comparison["observation_id"])
        actions = comparison["actions"]
        values: dict[str, dict[str, Any]] = {}
        for name in ACTION_NAMES:
            record = actions[name]
            q1 = float(record["q1"])
            q2 = float(record["q2"])
            action = np.asarray(record["action_tcp6"], dtype=np.float64)
            if action.shape != (3, 6) or not np.isfinite(action).all():
                raise ValueError("FORCERFT_SAME_STATE_AUDIT_ACTION_INVALID")
            values[name] = {
                "q1": q1,
                "q2": q2,
                "min_twin_q": min(q1, q2),
                "twin_disagreement": abs(q1 - q2),
                "action": action,
            }
        rows.append(
            {
                "observation_id": observation_id,
                "actions": {
                    name: {
                        key: value
                        for key, value in record.items()
                        if key != "action"
                    }
                    for name, record in values.items()
                },
                "actor_sft_action_distance": float(
                    np.linalg.norm(
                        values["current_actor"]["action"]
                        - values["frozen_sft_reference"]["action"]
                    )
                ),
                "actor_human_action_distance": float(
                    np.linalg.norm(
                        values["current_actor"]["action"]
                        - values["accepted_human_correction"]["action"]
                    )
                ),
            }
        )
    count = len(rows)
    fraction = lambda predicate: (  # noqa: E731
        None if count == 0 else sum(map(predicate, rows)) / count
    )
    return {
        "comparison_count": count,
        "human_gt_policy_fraction": fraction(
            lambda row: row["actions"]["accepted_human_correction"]["min_twin_q"]
            > row["actions"]["pre_takeover_policy"]["min_twin_q"]
        ),
        "sft_gt_policy_fraction": fraction(
            lambda row: row["actions"]["frozen_sft_reference"]["min_twin_q"]
            > row["actions"]["pre_takeover_policy"]["min_twin_q"]
        ),
        "actor_gt_sft_fraction": fraction(
            lambda row: row["actions"]["current_actor"]["min_twin_q"]
            > row["actions"]["frozen_sft_reference"]["min_twin_q"]
        ),
        "comparisons": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--same-state-probe", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = json.loads(args.same_state_probe.read_text(encoding="utf-8"))
    result = {
        "readiness_mode": "automatic_readiness",
        "critic_checkpoint": payload["critic_checkpoint"],
        "action_contract_version": payload["action_contract_version"],
        "same_observation_required": True,
        **summarize_same_state_rankings(payload["comparisons"]),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
