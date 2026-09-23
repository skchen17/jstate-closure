"""Comprehensive V36 code, freeze, report and machine-record hash index."""
from __future__ import annotations

import json
from pathlib import Path

from jclosure.provenance import sha256_file, write_json_atomic
from jclosure.protocol_v36 import stage_freeze, verify_stage

OUT = Path("results/v36/processed")
SOURCE = "src/jclosure/experiments/integrity_v36.py"


def run(root: Path):
    verify_stage(root, "report")
    groups = {
        "config": [root / "configs/operator_v36.yaml"],
        "protocol_and_source": [root / "src/jclosure/protocol_v36.py",
                                *sorted((root / "src/jclosure/experiments").glob("*v36.py")),
                                root / "tests/test_v36.py"],
        "freeze_artifacts": sorted((root / "artifacts").glob("computational_origin_v36*.freeze.json")),
        "standalone_reports": sorted((root / "reports").glob("V36_*.md")),
        "machine_records": [p for p in sorted((root / OUT).iterdir())
                            if p.is_file() and p.name != "v36_full_integrity_index.json"],
    }
    if len(groups["standalone_reports"]) != 22:
        raise RuntimeError("V36 report count changed")
    index = {"version": "V36", "complete": True,
             "groups": {group: {str(p.relative_to(root)): {"bytes": p.stat().st_size,
                                                        "sha256": sha256_file(p)} for p in paths}
                        for group, paths in groups.items()},
             "FINAL_REPORT_snapshot_sha256": sha256_file(root / "reports/FINAL_REPORT.md"),
             "historical_V1_V35_modified": False,
             "independent_final_opened": False,
             "large_npz_local_only_per_repository_policy": True}
    path = root / OUT / "v36_full_integrity_index.json"
    write_json_atomic(path, index)
    inputs = [SOURCE, str(path.relative_to(root)),
              "artifacts/computational_origin_v36_report.freeze.json",
              *(str(p.relative_to(root)) for paths in groups.values() for p in paths)]
    seal = stage_freeze(root, "integrity", inputs,
                        {"summary_sha256": sha256_file(path),
                         "group_counts": {k: len(v) for k, v in groups.items()},
                         "final_opened": False})
    return {"freeze_digest": seal["freeze_digest"],
            "group_counts": {k: len(v) for k, v in groups.items()},
            "index_sha256": sha256_file(path)}


if __name__ == "__main__":
    print(json.dumps(run(Path.cwd()), indent=2))
