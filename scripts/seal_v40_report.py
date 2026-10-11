"""Seal the reviewed, self-contained V40 report bundle after machine audit."""
from __future__ import annotations

import json
from pathlib import Path

from jclosure.protocol_v40_a4 import stage_freeze, verify, verify_stage
from jclosure.protocol_v40_a5 import verify_stage as verify_a5_stage
from jclosure.provenance import sha256_file

SOURCE = "scripts/seal_v40_report.py"


def run(root: Path) -> dict:
    verify(root)
    for role in ("development", "validation", "independent_final"):
        for key in ("Q", "F"):
            verify_stage(root, f"analysis_functional_{key}_{role}")
            verify_stage(root, f"split_sensitivity_{key}_{role}")
            verify_stage(root, f"candidate_vector_{key}_{role}")
        verify_stage(root, f"analysis_temporal_Q_{role}")
        verify_stage(root, f"temporal_family_Q_{role}")
        verify_a5_stage(root, f"trajectory_analysis_Q_{role}")
    verify_stage(root, "final_opening")
    index_path = root / "results/v40/processed/formal_report_index_v40.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    if index["status"] != "EXECUTION_COMPLETE" or not index["v40_formal_results_only"]:
        raise RuntimeError("V40 report not execution-complete")
    for group in ("input_sha256", "report_sha256"):
        for name, expected in index[group].items():
            if sha256_file(root / name) != expected:
                raise RuntimeError(f"V40 report {group} drift: {name}")
    all_text = (root / "reports/V40_ALL_REPORTS.md").read_text(encoding="utf-8")
    for topic in ("V40_CHANNEL_FUNCTION_PROFILE.md", "V40_TASK_GROUNDED_RESULTS.md",
                  "V40_TEMPORAL_PERSISTENCE.md", "V40_CHANNEL_COMPARISON.md",
                  "V40_TRAJECTORY_CAUSALITY.md", "V40_COMPLETE_REPORT.md"):
        if f"<!-- {topic} | SHA256 " not in all_text:
            raise RuntimeError(f"V40 all-reports topic missing: {topic}")
    inputs = [SOURCE, "scripts/report_v40_formal.py", "scripts/build_v40_all_reports.py",
              str(index_path.relative_to(root)), *index["input_sha256"],
              *index["report_sha256"]]
    seal = stage_freeze(root, "final_report", inputs,
                        {"status": "EXECUTION_COMPLETE",
                         "index_sha256": sha256_file(index_path),
                         "all_reports_sha256": index["report_sha256"]["reports/V40_ALL_REPORTS.md"]})
    return {"freeze_digest": seal["freeze_digest"],
            "report_index": str(index_path.relative_to(root))}


if __name__ == "__main__":
    print(json.dumps(run(Path.cwd()), indent=2))
