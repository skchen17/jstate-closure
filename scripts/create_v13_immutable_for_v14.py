"""Guard every tracked V1–V13 file before V14, except cumulative FINAL_REPORT."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

from jclosure.protocol_v14 import BASELINE_COMMIT
from jclosure.provenance import sha256_file, write_json_atomic

OUTPUT = Path("artifacts/v13_immutable.sha256.json")


def _digest(value: dict) -> str:
    return hashlib.sha256(
        json.dumps(
            {key: item for key, item in value.items() if key != "guard_digest"},
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()


def build(root: Path) -> dict:
    if (root / OUTPUT).exists():
        raise RuntimeError("V13 guard already exists; never silently rewrite it")
    names = subprocess.check_output(
        ["git", "ls-tree", "-r", "--name-only", BASELINE_COMMIT],
        cwd=root,
        text=True,
    ).splitlines()
    names = [name for name in names if name != "reports/FINAL_REPORT.md"]
    hashes = {name: sha256_file(root / name) for name in names}
    value = {
        "schema_version": 23,
        "purpose": "byte guard for every tracked V1-V13 file",
        "baseline_commit": BASELINE_COMMIT,
        "excluded_mutable_paths": ["reports/FINAL_REPORT.md"],
        "file_count": len(hashes),
        "hashes": hashes,
    }
    value["guard_digest"] = _digest(value)
    write_json_atomic(root / OUTPUT, value)
    return value


def verify(root: Path) -> dict:
    value = json.loads((root / OUTPUT).read_text(encoding="utf-8"))
    if value.get("guard_digest") != _digest(value):
        raise RuntimeError("V13 immutable guard digest mismatch")
    changed = [
        name
        for name, expected in value["hashes"].items()
        if not (root / name).is_file() or sha256_file(root / name) != expected
    ]
    if changed:
        raise RuntimeError(f"V1-V13 tracked files changed: {changed[:10]}")
    return value


if __name__ == "__main__":
    root = Path.cwd()
    result = verify(root) if (root / OUTPUT).exists() else build(root)
    print(result["file_count"], result["guard_digest"])
