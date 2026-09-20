"""Guard V1--V16 tracked bytes and index V17 outputs without editing history."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

from jclosure.protocol_v17 import PARENT, verify
from jclosure.provenance import sha256_file, write_json_atomic


def _lines(args: list[str], root: Path) -> list[str]:
    return subprocess.check_output(args, cwd=root, text=True).splitlines()


def main() -> None:
    root = Path.cwd()
    verify(root)
    v13 = json.loads((root / "artifacts/causal_geometry_v13_jvp.freeze.json").read_text())
    for name in ("artifacts/causal/v13/causal_features_v13.npz",
                 "artifacts/causal/v13/probe_directions_v13.pt"):
        if sha256_file(root / name) != v13["hashes"][name]:
            raise RuntimeError(f"V13 frozen large artifact changed: {name}")
    tracked = set(_lines(["git", "ls-tree", "-r", "--name-only", PARENT], root))
    changed = set(_lines(["git", "diff", "--name-only", PARENT, "--", "."], root))
    overlap = sorted((tracked & changed) - {"reports/FINAL_REPORT.md"})
    if overlap:
        raise RuntimeError(f"V1--V16 tracked bytes changed: {overlap[:20]}")
    patterns = ("configs/*v17*", "src/jclosure/protocol_v17.py", "src/jclosure/reporting_v17.py",
                "src/jclosure/experiments/*v17.py", "scripts/*v17*.py", "tests/test_v17.py",
                "artifacts/*v17*.freeze.json", "results/v17/processed/*",
                "reports/*_V17.md", "reports/FINAL_REPORT.md")
    files = sorted({path for pattern in patterns for path in root.glob(pattern)
                    if path.is_file() and path.name not in ("v17_integrity.json", "V17_COMPLETE_REPORT.md")})
    rows = [{"path": str(path.relative_to(root)).replace("\\", "/"),
             "bytes": path.stat().st_size, "sha256": sha256_file(path)} for path in files]
    payload = {"schema_version": 26, "protocol_version": "interventional_state_sufficiency_v17_integrity",
               "parent_commit": PARENT, "historical_tracked_file_count": len(tracked),
               "historical_modified_except_cumulative_final": overlap,
               "v13_large_artifact_hashes_verified": True,
               "file_count": len(rows), "files": rows}
    payload["index_digest"] = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    write_json_atomic(root / "results/v17/processed/v17_integrity.json", payload)
    print(json.dumps({"historical_file_count": len(tracked), "v17_file_count": len(rows),
                      "index_digest": payload["index_digest"]}, sort_keys=True))


if __name__ == "__main__":
    main()
