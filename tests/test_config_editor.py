from pathlib import Path

import pytest

from ai_trader.config import editor


def test_validate_and_write_config_file(tmp_path, monkeypatch):
    monkeypatch.setattr(editor, "_root", lambda: Path(tmp_path))
    (tmp_path / "config.yaml").write_text("environment: development\n", encoding="utf-8")

    result = editor.write_config_file("config.yaml", "environment: production\n")

    assert result["content"] == "environment: production\n"
    assert (tmp_path / "config.yaml").read_text(encoding="utf-8") == result["content"]


def test_rejects_invalid_yaml(tmp_path, monkeypatch):
    monkeypatch.setattr(editor, "_root", lambda: Path(tmp_path))
    (tmp_path / "config.yaml").write_text("{}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Invalid YAML"):
        editor.write_config_file("config.yaml", "environment: [\n")


def test_reset_clears_personal_values(tmp_path, monkeypatch):
    monkeypatch.setattr(editor, "_root", lambda: Path(tmp_path))
    (tmp_path / "config.yaml").write_text(
        "broker:\n  api_key: committed-example\n  name: paper\n", encoding="utf-8"
    )
    monkeypatch.setattr(editor, "_git_default", lambda name: None)

    result = editor.reset_config_file("config.yaml")

    assert "committed-example" not in result["content"]
    assert "name: paper" in result["content"]
