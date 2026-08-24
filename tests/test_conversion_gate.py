from pathlib import Path

import pytest

from forcesmolvla.conversion_gate import formal_conversion_preflight


ROOT = Path(__file__).parents[1]


def test_formal_conversion_fails_before_creating_output(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "session.json").write_text("{}")
    output = tmp_path / "output"
    with pytest.raises(PermissionError, match="approved RuleSpec"):
        formal_conversion_preflight(raw_root=raw, output_root=output, project_root=ROOT)
    assert not output.exists()


def test_formal_conversion_refuses_existing_output_first(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "session.json").write_text("{}")
    output = tmp_path / "output"
    output.mkdir()
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        formal_conversion_preflight(raw_root=raw, output_root=output, project_root=ROOT)
