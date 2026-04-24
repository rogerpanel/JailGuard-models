"""YAML configuration loader with dot-access."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import yaml


class Config(dict):
    """dict subclass exposing keys as attributes, recursively."""

    def __init__(self, mapping: Mapping[str, Any] | None = None):
        super().__init__()
        if mapping is None:
            return
        for k, v in mapping.items():
            self[k] = Config(v) if isinstance(v, Mapping) else v

    def __getattr__(self, item: str) -> Any:
        try:
            return self[item]
        except KeyError as exc:
            raise AttributeError(item) from exc

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value


def load_config(path: str | Path) -> Config:
    path = Path(path)
    with path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    return Config(raw)
