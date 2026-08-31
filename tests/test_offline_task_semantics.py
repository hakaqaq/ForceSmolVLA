from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
V1_PROTOCOL = ROOT / "docs/task2_reward_labeling_protocol.md"
V1_TEMPLATE = ROOT / "labels/task2_reward_frame_labels.v1.template.json"
V2_PROTOCOL = ROOT / "docs/task2_reward_labeling_protocol.v2.md"
V2_TEMPLATE = ROOT / "labels/task2_reward_frame_labels.v2.template.json"
AUDIT = ROOT / "artifacts/development/stage2/task_semantics_audit.v4.json"
SERVER = ROOT / "tools/reward_classifier/serve_task2_label_ui.py"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_v1_is_byte_exact_historical() -> None:
    assert sha256(V1_PROTOCOL) == "31213136e922fd827a364a232fcddd631db2d141c26cdb02231376f4aa004d79"
    assert sha256(V1_TEMPLATE) == "bb4df9b118e138894b8c2bd0d9d6d128fa1c35f6240ffc74b03be8fceaf765fa"


def test_v2_template_is_blank_and_has_frozen_ring_on_peg_contract() -> None:
    value = json.loads(V2_TEMPLATE.read_text())
    assert value["schema_version"] == "force_rft_task2_reward_frame_labels.v2"
    assert value["canonical_task_prompt"] == "Pick up the purple ring and place it onto the red peg."
    assert len(value["positive_all_of"]) == 4
    assert "gripper" in value["positive_all_of"][1]
    assert "subsequent observable frames" in value["positive_all_of"][3]
    assert value["conversion_task_text_semantically_equivalent"] is True
    assert value["example_video_semantics_only"]["may_map_to_lerobot_frame"] is False
    assert value["episode_count"] == len(value["episodes"]) == 47
    for episode in value["episodes"]:
        assert episode["manual_review_status"] == "unreviewed"
        assert episode["first_confident_complete_frame"] is None
        assert episode["last_confident_incomplete_frame"] is None
        assert episode["positive_available"] is None
        assert episode["hard_negative_intervals"] == []
        assert episode["ordinary_negative_intervals"] == []
        assert episode["ambiguous_intervals"] == []


def test_server_rejects_v1_and_loads_v2() -> None:
    spec = importlib.util.spec_from_file_location("task2_label_server_v2", SERVER)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    contract = module.load_label_contract(V2_TEMPLATE, V2_PROTOCOL)
    assert contract["schema_version"].endswith(".v2")
    try:
        module.load_label_contract(V1_TEMPLATE, V1_PROTOCOL)
    except RuntimeError as exc:
        assert "v2 labeling contract" in str(exc)
    else:
        raise AssertionError("v1 contract must be rejected")
    source = SERVER.read_text()
    assert "task2_reward_frame_labels.v2.template.json" in source
    assert "task2_reward_labeling_protocol.v2.md" in source
    ui = (ROOT / "tools/reward_classifier/task2_label_ui.html").read_text()
    assert "task2_reward_frame_labels.v2.reviewed.json" in ui
    assert "peg 穿过 ring 中心孔" in ui
    assert "完整物理插装" not in ui


def test_task_semantics_audit_disposition() -> None:
    audit = json.loads(AUDIT.read_text())
    assert audit["task_semantics_audit"] == "pass"
    assert audit["semantic_equivalence"] is True
    assert audit["stage1_task_text_invalid"] is False
    assert audit["r5_retraining_required"] is False
    assert audit["stage1_data_reconversion_required"] is False
    assert audit["historical_v1"]["disposition"] == "HISTORICAL_INVALID_FOR_LABELING"
    assert audit["active_v2"]["ui_uses_v2"] is True
    assert audit["reward_classifier_training_authorized"] is False
    assert audit["reward_or_terminal_created"] is False
