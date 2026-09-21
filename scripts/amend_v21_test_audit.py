"""Append-only correction for a summary parser that missed leading failures."""

from __future__ import annotations

import json
import re
from pathlib import Path

from jclosure.protocol_v21 import stage_freeze
from jclosure.provenance import sha256_file,write_json_atomic

SOURCE="scripts/amend_v21_test_audit.py"
OLD="results/v21/processed/v21_test_audit.json"
NEW="results/v21/processed/v21_test_audit_amendment_1.json"


def main():
    root=Path.cwd()
    original=json.loads((root/OLD).read_text())
    corrected=json.loads((root/OLD).read_text())
    for name,row in corrected["runs"].items():
        text=row["raw_stdout_tail"]
        first=re.search(r"(?m)^(\d+) failed, (\d+) passed",text)
        second=re.search(r"(?m)^(\d+) passed",text)
        if first:
            row["failed"],row["passed"]=int(first.group(1)),int(first.group(2))
        elif second:
            row["failed"],row["passed"]=0,int(second.group(1))
        else:
            raise RuntimeError(f"V21 test summary missing for {name}")
        if row["failed"]!=len(row["failed_tests"]):
            raise RuntimeError(f"V21 failed test name/count mismatch: {name}")
    if original["runs"]["all"]["failed"]!=0 or corrected["runs"]["all"]["failed"]!=2:
        raise RuntimeError("V21 expected inherited parser error not found")
    corrected["append_only_correction"]={"reason":"Initial regex found the later '233 passed' fragment inside '2 failed, 233 passed' and incorrectly recorded failed=0 despite exit code 1 and two listed failures.",
                                         "original_record_sha256":sha256_file(root/OLD),
                                         "original_record_preserved":True}
    write_json_atomic(root/NEW,corrected)
    freeze=stage_freeze(root,"test_audit_parse_amendment_1",[SOURCE,OLD,NEW],
                        {"original_record_sha256":sha256_file(root/OLD),
                         "corrected_record_sha256":sha256_file(root/NEW),
                         "v21_tests_passed":5,"all_tests_passed":233,"inherited_old_hash_tests_failed":2,
                         "historical_frozen_manifests_modified":False})
    print(json.dumps({"freeze_digest":freeze["freeze_digest"],
                      "corrected_counts":{name:{"passed":x["passed"],"failed":x["failed"]}
                                          for name,x in corrected["runs"].items()}},indent=2))


if __name__=="__main__":
    main()
