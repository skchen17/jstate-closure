"""Preserve raw V15 JSON and create strictly valid copies for undefined metrics."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from jclosure.protocol_v15 import stage_freeze, verify
from jclosure.provenance import sha256_file, write_json_atomic


def _clean(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, list):
        return [_clean(item) for item in value]
    if isinstance(value, dict):
        return {key: _clean(item) for key, item in value.items()}
    return value


def main() -> None:
    root = Path.cwd()
    verify(root)
    directory = root / "results/v15/processed"
    source_paths = sorted(
        path for path in directory.glob("*.json")
        if not path.name.endswith("_strict.json") and path.name != "v15_integrity.json"
    )
    affected = []
    for path in source_paths:
        text = path.read_text(encoding="utf-8")
        try:
            json.loads(text, parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)))
        except ValueError:
            affected.append(path)
    if not affected:
        print("all V15 JSON is already strict")
        return
    stage = stage_freeze(root, "strict_json_correction", ["scripts/normalize_v15_numeric_json.py", *[str(path.relative_to(root)) for path in affected]], {"reason": "undefined numerical metrics serialized as nonstandard NaN/Infinity in original records", "original_records_preserved": True, "transformation": "nonfinite float to JSON null only"})
    records = []
    for path in affected:
        output = path.with_name(path.stem + "_strict.json")
        clean = _clean(json.loads(path.read_text(encoding="utf-8")))
        output.write_text(json.dumps(clean, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
        records.append({"source": str(path.relative_to(root)), "source_sha256": sha256_file(path), "strict": str(output.relative_to(root)), "strict_sha256": sha256_file(output)})
    manifest = {"role": "machine_record_correction", "freeze_digest": stage["freeze_digest"], "affected": records}
    write_json_atomic(directory / "strict_json_correction_v15.json", manifest)
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
