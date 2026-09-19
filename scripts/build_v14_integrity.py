"""Create and verify a byte manifest for V14 code, freezes, records, and reports."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from jclosure.provenance import sha256_file, write_json_atomic

OUTPUT = Path("results/v14/processed/v14_integrity.json")


def _digest(value: dict) -> str:
    return hashlib.sha256(
        json.dumps(
            {key: item for key, item in value.items() if key != "manifest_digest"},
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()


def paths(root: Path) -> list[Path]:
    candidates = [
        *root.glob("artifacts/finite_causal_control_v14*.freeze.json"),
        root / "artifacts/v13_immutable.sha256.json",
        root / "configs/finite_causal_control_v14.yaml",
        *root.glob("results/v14/**/*"),
        *root.glob("reports/*_V14.md"),
        root / "reports/FINAL_REPORT.md",
        *root.glob("src/jclosure/*v14*.py"),
        *root.glob("src/jclosure/experiments/*v14*.py"),
        *root.glob("scripts/*v14*.py"),
        root / "tests/test_v14.py",
    ]
    return sorted(
        {path for path in candidates if path.is_file() and path != root / OUTPUT}
    )


def build(root: Path) -> dict:
    if (root / OUTPUT).exists():
        raise RuntimeError(
            "V14 integrity manifest already exists; do not silently rewrite"
        )
    hashes = {
        path.relative_to(root).as_posix(): sha256_file(path) for path in paths(root)
    }
    value = {
        "schema_version": 23,
        "protocol_version": "finite_causal_control_v14_integrity",
        "excluded_mutable_paths": ["reports/V14_COMPLETE_REPORT.md"],
        "file_count": len(hashes),
        "hashes": hashes,
    }
    value["manifest_digest"] = _digest(value)
    write_json_atomic(root / OUTPUT, value)
    return value


def verify(root: Path) -> dict:
    value = json.loads((root / OUTPUT).read_text(encoding="utf-8"))
    if value.get("manifest_digest") != _digest(value):
        raise RuntimeError("V14 integrity digest mismatch")
    changed = [
        name
        for name, expected in value["hashes"].items()
        if not (root / name).is_file() or sha256_file(root / name) != expected
    ]
    if changed:
        raise RuntimeError(f"V14 protected files changed: {changed[:10]}")
    return value


if __name__ == "__main__":
    root = Path.cwd()
    value = verify(root) if (root / OUTPUT).exists() else build(root)
    print(value["file_count"], value["manifest_digest"])
