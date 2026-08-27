from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch

from forcesmolvla.rft.g7a import (
    G7A_CHECKPOINT_MARKERS,
    G7A_COUNTERS,
    aggregate_gradient_probes,
    grouped_regression,
    regression_metrics,
    save_g7a_checkpoint,
    select_fixed_critic_probe,
    spearman_correlation,
    validate_g7a_checkpoint,
    verify_source_manifest,
)


def test_spearman_and_regression_are_deterministic_with_ties() -> None:
    assert spearman_correlation([1, 2, 2, 4], [10, 20, 20, 40]) == pytest.approx(1.0)
    metrics = regression_metrics([1.0, 3.0], [2.0, 2.0])
    assert metrics["mae"] == 1.0
    assert metrics["rmse"] == 1.0
    assert metrics["bias"] == 0.0
    assert metrics["spearman"] is None


def test_fixed_probe_includes_all_terminal_and_partial_rows() -> None:
    rows = [
        {"terminated": index == 9, "executed_steps": 2 if index == 7 else 3}
        for index in range(10)
    ]
    selected = select_fixed_critic_probe(rows, 5)
    assert len(selected) == len(set(selected)) == 5
    assert {7, 9}.issubset(selected)


def test_grouped_regression_covers_terminal_steps_and_distance() -> None:
    rows = [
        {"terminated": False, "executed_steps": 3, "policy_decision_distance": 8,
         "q_mean": 0.2, "mc_return": 0.3},
        {"terminated": True, "executed_steps": 1, "policy_decision_distance": 0,
         "q_mean": 1.0, "mc_return": 1.0},
    ]
    groups = grouped_regression(rows)
    assert groups["terminal=true"]["count"] == 1
    assert groups["executed_steps=1"]["mae"] == 0.0
    assert groups["distance=6_20"]["count"] == 1


def test_eta_candidates_are_measurement_only() -> None:
    probes = [
        {
            "global": {"raw_q_over_fm": value, "cosine_similarity": 0.0,
                       "fm_norm": 2.0, "q_norm": 2.0 * value},
            "modules": {
                "router": {"raw_q_over_fm": value, "cosine_similarity": 0.0,
                           "fm_norm": 1.0, "q_norm": value}
            },
        }
        for value in (0.5, 1.0, 1.5)
    ]
    result = aggregate_gradient_probes(probes, [0.01, 0.1], [0.01, 0.10])
    assert result["eta_selected_or_approved"] is False
    assert result["candidates_with_median_in_reference_band"] == [0.01, 0.1]
    assert math.isfinite(result["modules"]["router"]["raw_q_over_fm"]["median"])


def test_g7a_checkpoint_is_atomic_integrity_bound_and_critic_only(tmp_path: Path) -> None:
    critics = {
        name: torch.nn.Linear(2, 1)
        for name in ("q1", "q2", "q1_target", "q2_target")
    }
    optimizer = torch.optim.Adam(
        [*critics["q1"].parameters(), *critics["q2"].parameters()], lr=3e-4
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    checkpoint = tmp_path / "checkpoint"
    manifest = save_g7a_checkpoint(
        checkpoint,
        critics=critics,
        critic_optimizer=optimizer,
        critic_scheduler=scheduler,
        counters=G7A_COUNTERS,
        sampler_states={"td": {"draws": 256}},
        rng_states={"fixture": True},
        actor_binding={"optimizer_created": False},
        ownership_manifest={"actor_optimizer_created": 0},
        fixed_diagnostics_manifest={"frozen": True},
        protected_snapshot={"unchanged": True},
        startup_snapshot_bytes={"config.yaml": b"frozen\n"},
    )
    assert all(manifest[key] == value for key, value in G7A_CHECKPOINT_MARKERS.items())
    assert manifest["actor_state_stored"] is False
    assert validate_g7a_checkpoint(checkpoint)["counters"] == G7A_COUNTERS

    target = checkpoint / "manifests/actor_binding.json"
    target.write_bytes(target.read_bytes() + b" ")
    with pytest.raises(RuntimeError, match="G7A_CHECKPOINT_INTERNAL_FILE_SHA_MISMATCH"):
        validate_g7a_checkpoint(checkpoint)


def test_source_manifest_fails_closed_on_drift(tmp_path: Path) -> None:
    source = tmp_path / "source.py"
    source.write_text("frozen = True\n")
    import hashlib
    manifest = tmp_path / "manifest.json"
    manifest.write_text(__import__("json").dumps({
        "schema_version": "forcesmolvla_stage2_source_manifest.v9_g7a",
        "manual_g1_or_manual_label_in_runtime_closure": False,
        "files": {"source": {
            "path": "source.py", "file_size": source.stat().st_size,
            "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        }},
    }))
    verify_source_manifest(tmp_path, manifest)
    source.write_text("frozen = False\n")
    with pytest.raises(RuntimeError, match="G7A_SOURCE_FILE_(SIZE|SHA)_MISMATCH"):
        verify_source_manifest(tmp_path, manifest)
