"""Freeze the V21 standalone/cumulative report and integrity generation scheme."""

from __future__ import annotations

import json
from pathlib import Path

from jclosure.protocol_v21 import stage_freeze,verify_stage

REPORTS=(
    "ACTION_REPRESENTATION_CEILING_V21.md",
    "STATE_INFORMATION_CEILING_V21.md",
    "BOTTLENECK_FACTORIAL_V21.md",
    "ACTION_COVERAGE_AUDIT_V21.md",
    "ACTION_DATA_SCALING_V21.md",
    "PAIRED_JVP_FINITE_OPERATOR_V21.md",
    "DIFFERENTIAL_FINITE_ALIGNMENT_V21.md",
    "OPERATOR_ROTATION_DECOMPOSITION_V21.md",
    "HIGHER_ORDER_ACTION_GEOMETRY_V21.md",
    "COMPACT_OPERATOR_REOPEN_DECISION_V21.md",
    "STRICT_INTERFACE_AUDIT_V21.md",
    "EXECUTION_MANIFEST_V21.md",
    "V21_SCIENTIFIC_ANSWERS_V21.md",
)


def main():
    root=Path.cwd()
    verify_stage(root,"adjudication_design")
    result=stage_freeze(root,"reporting_design",
                        ["scripts/freeze_v21_reporting.py","scripts/build_v21_reports.py",
                         "scripts/build_v21_integrity.py",
                         "artifacts/action_coordinate_geometry_v21_adjudication_design.freeze.json"],
                        {"standalone_reports":list(REPORTS),
                         "single_file_bundle":"reports/V21_COMPLETE_REPORT.md",
                         "cumulative_final_report":"reports/FINAL_REPORT.md append V21_START/V21_END once only",
                         "integrity_index":"results/v21/processed/v21_integrity_index.json",
                         "report_source":"machine_readable_frozen_V21_results_only",
                         "no_historical_report_rewrite":True,
                         "final_six_action_responses_opened":False})
    print(json.dumps({"freeze_digest":result["freeze_digest"],
                      "standalone_report_count":len(REPORTS)},indent=2))


if __name__=="__main__":
    main()
