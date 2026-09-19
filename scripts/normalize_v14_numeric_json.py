"""Publish a strict-JSON copy of the preserved NaN-bearing V14 audit summary."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from jclosure.protocol_v14 import freeze_stage, verify_base
from jclosure.provenance import sha256_file, write_json_atomic

SOURCE = Path("results/v14/processed/jvp_finite_writeback_audit_v14.json")
OUTPUT = Path("results/v14/processed/jvp_finite_writeback_audit_v14_strict.json")


def clean(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, list):
        return [clean(item) for item in value]
    if isinstance(value, dict):
        return {key: clean(item) for key, item in value.items()}
    return value


def main() -> None:
    root = Path.cwd()
    verify_base(root)
    frozen = freeze_stage(
        root,
        "strict_json_correction",
        ["scripts/normalize_v14_numeric_json.py", str(SOURCE)],
        {
            "source_records_preserved": True,
            "correction": "represent undefined zero-norm cosines as JSON null rather than NaN",
            "estimand_changed": False,
        },
    )
    payload = clean(json.loads((root / SOURCE).read_text(encoding="utf-8")))
    payload["strict_json_correction_freeze_digest"] = frozen["freeze_digest"]
    payload["source_sha256"] = sha256_file(root / SOURCE)
    write_json_atomic(root / OUTPUT, payload)
    json.loads(
        (root / OUTPUT).read_text(encoding="utf-8"),
        parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)),
    )
    print(sha256_file(root / OUTPUT))


if __name__ == "__main__":
    main()
