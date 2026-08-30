import json
from pathlib import Path
import pytest


ROOT = Path(__file__).parents[1]
from forcesmolvla.dataset_binding import (
    validate_dense_compute_prerequisite as _validate_dense_compute_prerequisite,
    validate_runtime_import_roots as _validate_runtime_import_roots,
    validate_variant_spec as _validate_spec,
)


def _spec():
    return json.loads(
        (ROOT / "configs/p6_dense_param_moe.development.json").read_text()
    )


def test_p6_freezes_all_four_p5_prerequisite_hashes_and_import_roots():
    spec = _spec()
    _validate_spec(spec)
    observed = _validate_dense_compute_prerequisite(
        ROOT,
        spec,
        dataset_root=ROOT / "datasets/task2_lerobotv3",
        repo_id="local/task2_lerobotv3",
    )
    assert observed == {
        "static_spec": "edc33ab18b015007039a2355c16768e736ceb1b3a4620b74e8d9fd540614ea19",
        "source_binding": "4a2d2ab0c515cc5b45687f46f736ac9cad05b88b6df5a15256cea3f7207c46f9",
        "resolved_config": "19834558943477d0d24662a8251f3e19f9609b29a55c9bb8265c32cc32d9c3ea",
        "gate_result": "cadfb8e399182f6c981fe57e943fed6d755648144cb37139568b3e7bc4a1b3ba",
    }
    imports = _validate_runtime_import_roots(ROOT)
    assert imports["forcesmolvla"].startswith(str(ROOT / "src/forcesmolvla"))
    assert imports["lerobot"].startswith(str(ROOT / "vendor/lerobot/src/lerobot"))


def test_p6_rejects_any_parent_p5_hash_drift():
    spec = _spec()
    spec["p5_prerequisite"]["gate_result"]["sha256"] = "0" * 64
    with pytest.raises(RuntimeError, match="P6_P5_GATE_RESULT_HASH_MISMATCH"):
        _validate_dense_compute_prerequisite(
            ROOT,
            spec,
            dataset_root=ROOT / "datasets/task2_lerobotv3",
            repo_id="local/task2_lerobotv3",
        )
