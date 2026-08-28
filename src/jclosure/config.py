"""Strict, reproducible YAML configuration loading."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml


def _merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in override.items():
        if (
            key in out
            and isinstance(out[key], dict)
            and isinstance(value, dict)
        ):
            out[key] = _merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def load_config(path: str | Path) -> dict[str, Any]:
    """Load YAML with optional relative ``extends`` inheritance."""

    config_path = Path(path).resolve()
    with config_path.open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, dict):
        raise TypeError(f"configuration root must be a mapping: {config_path}")
    parent = raw.pop("extends", None)
    if parent is None:
        config = raw
    else:
        parent_path = (config_path.parent / str(parent)).resolve()
        if parent_path == config_path:
            raise ValueError("configuration cannot extend itself")
        config = _merge(load_config(parent_path), raw)
    config["_config_path"] = str(config_path)
    validate_config(config)
    return config


def validate_config(config: dict[str, Any]) -> None:
    for section in ("model", "lens", "jstate", "reproducibility", "outputs"):
        if section not in config:
            raise ValueError(f"missing required configuration section: {section}")
    if int(config["jstate"].get("k", 0)) <= 0:
        raise ValueError("jstate.k must be positive")
    if int(config["jstate"].get("concept_vocab_size", 0)) < 16:
        raise ValueError("jstate.concept_vocab_size must be at least 16")
    strengths = config.get("closure", {}).get("strengths", [])
    if any(float(value) < 0 for value in strengths):
        raise ValueError("closure strengths must be non-negative")


def _json_key_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_key_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_key_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_key_safe(item) for item in value]
    return value


def public_config(config: dict[str, Any]) -> dict[str, Any]:
    return _json_key_safe(
        {key: value for key, value in config.items() if not key.startswith("_")}
    )


def config_digest(config: dict[str, Any]) -> str:
    payload = json.dumps(
        public_config(config), sort_keys=True, separators=(",", ":"), default=str
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def resolve_output_path(config: dict[str, Any], key: str) -> Path:
    root = Path(config["outputs"].get("root", "."))
    if not root.is_absolute():
        config_dir = Path(config["_config_path"]).parent.parent
        root = (config_dir / root).resolve()
    return root / str(config["outputs"][key])
