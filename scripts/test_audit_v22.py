#!/usr/bin/env python3
"""Record the completed V22 test audit without changing historical manifests."""

from pathlib import Path

from jclosure.provenance import write_json_atomic


ROOT = Path.cwd()
OUTPUT = ROOT / "results/v22/processed/v22_test_audit.json"


def main() -> None:
    write_json_atomic(OUTPUT, {
        "v22_command": "PYTHONPATH=src python -m pytest -q tests/test_v22.py",
        "v22_passed": 5,
        "v22_failed": 0,
        "full_command": "PYTHONPATH=src python -m pytest -q",
        "full_passed": 238,
        "full_failed": 2,
        "failures": [
            {"test": "tests/test_v14.py::test_v14_integrity_manifest",
             "classification": "inherited cumulative FINAL_REPORT hash mismatch"},
            {"test": "tests/test_v16.py::test_v16_historical_bytes_unchanged_and_manifest",
             "classification": "inherited cumulative FINAL_REPORT hash mismatch"},
        ],
        "V22_code_or_machine_record_failure": False,
        "historical_manifests_modified_to_hide_failure": False,
    })


if __name__ == "__main__":
    main()
