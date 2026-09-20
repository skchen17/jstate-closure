#!/usr/bin/env python3
"""Audit V20 freeze inputs and index version files before bundling."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from jclosure.protocol_v20 import PARENT, verify, verify_stage
from jclosure.provenance import sha256_file, write_json_atomic


def main() -> None:
    root = Path.cwd()
    base = verify(root)
    names = [p.name.removeprefix("compact_causal_response_operator_v20_").removesuffix(".freeze.json")
             for p in sorted((root / "artifacts").glob("compact_causal_response_operator_v20_*.freeze.json"))]
    verified = {name: verify_stage(root, name)["freeze_digest"] for name in names}
    tracked = subprocess.check_output(["git", "diff", "--name-only", PARENT, "--"], cwd=root, text=True).splitlines()
    if any(name != "reports/FINAL_REPORT.md" for name in tracked):
        raise RuntimeError(f"V1–V19 tracked content changed: {tracked}")
    prefixes = ("configs/compact_causal_response_operator_v20.yaml",
                "src/jclosure/protocol_v20.py", "src/jclosure/experiments/",
                "tests/test_v20_operator.py", "scripts/adjudicate_v20.py",
                "scripts/build_v20_reports.py", "scripts/build_v20_integrity.py",
                "scripts/inspect_v20_inputs.py", "scripts/amend_v20_adjudication.py",
                "scripts/summarize_v20_operator_geometry.py")
    paths = [root / prefixes[0], root / prefixes[1], root / prefixes[3],
             *sorted((root / "src/jclosure/experiments").glob("*v20.py")),
             *[root / x for x in prefixes[4:]],
             *sorted((root / "artifacts").glob("compact_causal_response_operator_v20*.freeze.json")),
             *sorted((root / "results/v20/processed").glob("*")),
             *sorted((root / "reports").glob("*_V20.md")),
             root / "reports/FINAL_REPORT.md"]
    paths = [p for p in paths if p.is_file() and p.name != "v20_integrity_index.json"]
    unique = sorted(set(paths))
    record = {str(p.relative_to(root)): {"bytes": p.stat().st_size, "sha256": sha256_file(p)} for p in unique}
    result = {"protocol_digest": base["freeze_digest"], "parent_commit": PARENT,
              "parent_tracked_changes": tracked, "verified_stage_digests": verified,
              "files": record, "file_count": len(record),
              "old_frozen_records_unchanged": True,
              "complete_report_bundle_not_yet_generated": True}
    target = root / "results/v20/processed/v20_integrity_index.json"
    write_json_atomic(target, result)
    print(json.dumps({"file_count": len(record), "verified_freeze_stages": len(verified),
                      "index_sha256": sha256_file(target)}, indent=2))


if __name__ == "__main__":
    main()
