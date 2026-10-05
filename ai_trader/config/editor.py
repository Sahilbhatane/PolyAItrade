"""Safe editor for operator-managed configuration files."""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path
from typing import Any

import yaml

CONFIG_FILES = ("config.yaml", "ml_config.yaml", "strategy_config.yaml", ".env")
_SECRET_PARTS = (
    "key",
    "secret",
    "token",
    "password",
    "webhook",
    "traderkey",
    "client_id",
    "api",
)


def _root() -> Path:
    return Path.cwd().resolve()


def _allowed_names() -> set[str]:
    names = set(CONFIG_FILES)
    for pattern in ("*.yaml", "*.yml", ".env.*"):
        names.update(path.name for path in _root().glob(pattern) if path.is_file())
    return names


def _path(name: str) -> Path:
    if name not in _allowed_names() or Path(name).name != name:
        raise ValueError(f"Unsupported configuration file: {name}")
    path = (_root() / name).resolve()
    if path.parent != _root():
        raise ValueError(f"Configuration file must be in the project root: {name}")
    return path


def list_config_files() -> list[str]:
    return sorted(_allowed_names())


def read_config_file(name: str) -> dict[str, str]:
    path = _path(name)
    try:
        content = path.read_text(encoding="utf-8") if path.exists() else ""
    except OSError as exc:
        raise ValueError(f"Could not read {name}: {exc}") from exc
    return {"name": name, "content": content}


def validate_config_file(name: str, content: str) -> None:
    if name == ".env" or name.startswith(".env."):
        for line_number, line in enumerate(content.splitlines(), 1):
            stripped = line.strip()
            if stripped and not stripped.startswith("#") and "=" not in stripped:
                raise ValueError(f"Invalid .env line {line_number}: expected KEY=VALUE")
        return
    try:
        parsed = yaml.safe_load(content)
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid YAML: {exc}") from exc
    if parsed is not None and not isinstance(parsed, dict):
        raise ValueError("Configuration YAML must contain a mapping at the top level")


def write_config_file(name: str, content: str) -> dict[str, str]:
    _path(name)
    validate_config_file(name, content)
    path = _path(name)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        temporary.write_text(content, encoding="utf-8", newline="")
        os.replace(temporary, path)
    except OSError as exc:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass
        raise ValueError(f"Could not write {name}: {exc}") from exc
    return {"name": name, "content": content}


def _git_default(name: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", "show", f"HEAD:{name}"],
            cwd=_root(),
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return None
    return result.stdout if result.returncode == 0 else None


def _is_secret_key(key: Any) -> bool:
    normalized = str(key).lower().replace("-", "_")
    return any(part in normalized for part in _SECRET_PARTS)


def _clear_secrets(value: Any, key: Any = "") -> Any:
    if isinstance(value, dict):
        return {k: _clear_secrets(v, k) for k, v in value.items()}
    if isinstance(value, list):
        return [_clear_secrets(v, key) for v in value]
    return "" if _is_secret_key(key) else value


def _reset_content(name: str) -> str:
    if name == ".env":
        example = _root() / ".env.example"
        source = example.read_text(encoding="utf-8") if example.exists() else ""
        return re.sub(r"^(\s*[A-Za-z_][A-Za-z0-9_]*=).*$", r"\1", source, flags=re.MULTILINE)

    source = _git_default(name)
    if source is None:
        source = (_root() / name).read_text(encoding="utf-8") if (_root() / name).exists() else ""
    try:
        parsed = yaml.safe_load(source)
    except yaml.YAMLError:
        return source
    return yaml.safe_dump(_clear_secrets(parsed), sort_keys=False) if parsed is not None else ""


def reset_config_file(name: str) -> dict[str, str]:
    content = _reset_content(name)
    return write_config_file(name, content)
