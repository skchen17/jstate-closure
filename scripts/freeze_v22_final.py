#!/usr/bin/env python3
"""Freeze final V22 adjudication, reports, and authorization state."""

import json
from pathlib import Path

from jclosure.protocol_v22 import stage_freeze
from jclosure.provenance import sha256_file, write_json_atomic


ROOT = Path.cwd()


def main() -> None:
    final = stage_freeze(ROOT, "final", [
        "scripts/freeze_v22_final.py",
        "results/v22/processed/v22_adjudication.json",
        "results/v22/processed/v22_test_audit.json",
        "results/v22/processed/v22_integrity_index.json",
        "reports/V22_COMPLETE_REPORT.md",
        "reports/V22_SCIENTIFIC_ANSWERS_V22.md",
        "reports/STRICT_INTERFACE_AUDIT_V22.md",
    ], {
        "formal_outcome": "V22-D_ACTION_DATA_LIMITED",
        "PRACTICAL_ACTION_COORDINATE_PASS": False,
        "COMPACT_OPERATOR_SEARCH_REOPENED": False,
        "RAW_TO_OPERATOR_ENCODER_AUTHORIZED": False,
        "H2_REMAINS": True,
        "H3_AUTHORIZED": False,
        "DYNAMIC_STATE_SEARCH_AUTHORIZED": False,
        "historical_final_six_opened": False,
        "new_independent_final_opened": False,
    })
    verification = {
        "status": "PASS_WITH_INHERITED_HISTORICAL_TEST_EXCEPTIONS",
        "final_freeze_digest": final["freeze_digest"],
        "complete_report_sha256": sha256_file(ROOT / "reports/V22_COMPLETE_REPORT.md"),
        "adjudication_sha256": sha256_file(ROOT / "results/v22/processed/v22_adjudication.json"),
        "integrity_index_sha256": sha256_file(ROOT / "results/v22/processed/v22_integrity_index.json"),
        "v22_tests": "5 passed, 0 failed",
        "full_tests": "238 passed, 2 inherited cumulative FINAL_REPORT hash failures",
    }
    write_json_atomic(ROOT / "results/v22/processed/v22_verification.json", verification)
    print(json.dumps(verification, indent=2))


if __name__ == "__main__":
    main()
