"""Input/output utilities for configuration and artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml


PathLike = str | Path


def ensure_dir(path: PathLike) -> Path:
    """Create directory if it does not exist and return it as Path."""
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    return target


def load_yaml(path: PathLike) -> dict[str, Any]:
    """Load a YAML file and return a dictionary."""
    file_path = Path(path)
    with file_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise TypeError(f"YAML content must be a mapping, got: {type(data).__name__}")
    return data


def save_yaml(data: dict[str, Any], path: PathLike) -> Path:
    """Save a dictionary as YAML with stable key order disabled."""
    file_path = Path(path)
    ensure_dir(file_path.parent)
    with file_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)
    return file_path


def load_json(path: PathLike) -> dict[str, Any]:
    """Load a JSON file and return a dictionary."""
    file_path = Path(path)
    with file_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise TypeError(f"JSON content must be an object, got: {type(data).__name__}")
    return data


def save_json(data: dict[str, Any], path: PathLike, indent: int = 2) -> Path:
    """Save a dictionary to a JSON file."""
    file_path = Path(path)
    ensure_dir(file_path.parent)
    with file_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent)
        f.write("\n")
    return file_path
