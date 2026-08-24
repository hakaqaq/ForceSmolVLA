import pytest
import torch

from forcesmolvla.prefix import PrefixLayout, assert_cache_unchanged, clone_cache


def test_prefix_layout_is_fixed_and_contiguous():
    layout = PrefixLayout()
    layout.validate()
    assert layout.physical_length == 177
    assert layout.camera1.length == 64
    assert layout.camera2.length == 64
    assert layout.language.length == 48
    assert layout.state.length == 1
    ids = layout.segment_ids(device="cpu")
    assert ids.tolist()[:64] == [0] * 64
    assert ids.tolist()[176] == 3


def test_prefix_cache_detects_length_and_content_mutation():
    cache = {0: {"key_states": torch.zeros(2, 177, 1), "value_states": torch.ones(2, 177, 1)}}
    snapshot = clone_cache(cache)
    assert_cache_unchanged(cache, snapshot, physical_length=177)
    cache[0]["key_states"][0, 0, 0] = 1
    with pytest.raises(RuntimeError, match="CONTENT_MUTATED"):
        assert_cache_unchanged(cache, snapshot, physical_length=177)


def test_valid_length_is_not_physical_length():
    valid = torch.tensor([[1] * 177, [1] * 146 + [0] * 31], dtype=torch.bool)
    assert valid.shape[1] == PrefixLayout().physical_length
    assert valid.sum(dim=1).tolist() == [177, 146]
