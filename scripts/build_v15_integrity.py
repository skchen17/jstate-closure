"""Guard V1--V14 tracked bytes and index all V15 report/data artifacts."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

from jclosure.protocol_v15 import PARENT, verify
from jclosure.provenance import sha256_file, write_json_atomic


def _lines(command: list[str], root: Path) -> list[str]:
    return subprocess.check_output(command, cwd=root, text=True).splitlines()


def main() -> None:
    root = Path.cwd()
    verify(root)
    tracked = set(_lines(["git", "ls-tree", "-r", "--name-only", PARENT], root))
    modified = set(_lines(["git", "diff", "--name-only", PARENT, "--", "."], root))
    overlap = sorted((tracked & modified) - {"reports/FINAL_REPORT.md"})
    if overlap:
        raise RuntimeError(f"V1--V14 tracked parents changed: {overlap[:20]}")
    patterns = (
        "configs/*v15*", "src/jclosure/protocol_v15.py", "src/jclosure/experiments/*v15.py",
        "src/jclosure/reporting_v15.py", "scripts/*v15*.py", "tests/test_v15.py",
        "artifacts/*v15*.freeze.json", "results/v15/processed/*", "reports/*_V15.md",
        "reports/FINAL_REPORT.md",
    )
    files = sorted({file for pattern in patterns for file in root.glob(pattern) if file.is_file() and file.name != "v15_integrity.json"})
    rows = [{"path": str(path.relative_to(root)).replace("\\", "/"), "bytes": path.stat().st_size, "sha256": sha256_file(path)} for path in files]
    payload = {
        "schema_version": 24, "protocol_version": "quantization_aware_causal_actuation_v15_integrity",
        "parent_commit": PARENT, "historical_tracked_file_count": len(tracked),
        "historical_modified_except_cumulative_final": overlap,
        "file_count": len(rows), "files": rows,
    }
    payload["index_digest"] = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    path = root / "results/v15/processed/v15_integrity.json"
    write_json_atomic(path, payload)
    print(json.dumps({"historical_file_count": len(tracked), "v15_file_count": len(rows), "index_digest": payload["index_digest"]}, sort_keys=True))


if __name__ == "__main__":
    main()
