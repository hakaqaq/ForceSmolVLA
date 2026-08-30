import json
from pathlib import Path
import pytest


ROOT = Path(__file__).parents[1]
from forcesmolvla.dataset_binding import (
    validate_dataset_variant_prerequisite as _validate_dataset_variant_prerequisite,
)
from forcesmolvla.training_runtime import validate_training_recipe as _validate_recipe


def _recipe():
    return json.loads(
        (ROOT / "configs/p7_training_recipe.development.yaml").read_text()
    )


def test_p7_freezes_p6_and_separates_single_pass_from_exact_oracle():
    recipe = _recipe()
    _validate_recipe(recipe)
    assert recipe["single_pass_batching"] == {
        "batch_per_gpu": 4,
        "gradient_accumulation_microbatches": 1,
        "effective_samples_per_gpu_update": 4,
    }
    assert recipe["exact_two_pass_oracle_batching"] == {
        "batch_per_gpu": 2,
        "gradient_accumulation_microbatches": 8,
        "effective_samples_per_gpu_update": 16,
    }
    assert recipe["long_running_sft_allowed"] is False
    assert recipe["loss"]["active_training_router_algorithm"] == "single_pass_batch_local"
    assert (
        recipe["loss"]["acceptance_oracle_router_algorithm"]
        == "exact_two_pass_all_microbatches_all_ranks"
    )
    observed = _validate_dataset_variant_prerequisite(
        ROOT,
        recipe,
        dataset_root=ROOT / "datasets/task2_lerobotv3",
        repo_id="local/task2_lerobotv3",
    )
    assert observed == {
        "static_spec": "b3988b12665d1c9668da27dde3256ae47042996f55cd81384d4f32a5a7e27681",
        "source_binding": "291983591fce875b04b3632b1c503243b1881c3f92021c18765c494a27a24395",
        "resolved_config": "c57ce93f3cf96458427fc96ecc61bf2bb62d6e45bec013c57a9e9fe536185800",
        "gate_result": "e7adfa8968c9bbf11aee3377e7d54f8645f5166daa13e3ead73ce905de92be7a",
    }


def test_p7_rejects_any_parent_p6_hash_drift():
    recipe = _recipe()
    recipe["p6_prerequisite"]["gate_result"]["sha256"] = "0" * 64
    with pytest.raises(RuntimeError, match="P7_P6_GATE_RESULT_HASH_MISMATCH"):
        _validate_dataset_variant_prerequisite(
            ROOT,
            recipe,
            dataset_root=ROOT / "datasets/task2_lerobotv3",
            repo_id="local/task2_lerobotv3",
        )
