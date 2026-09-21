"""Run and record V21/all-version CPU test status without changing old manifests."""

from __future__ import annotations

import json
import re
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

from jclosure.provenance import write_json_atomic


def main():
    root=Path.cwd()
    executable=sys.executable
    runs={}
    for name,selection in (("v21_only",["tests/test_v21.py"]),("all",[])):
        command=[executable,"-m","pytest","-q",*selection]
        result=subprocess.run(command,cwd=root,text=True,stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT,check=False)
        matched=re.search(r"(\d+) passed(?:, (\d+) failed)?",result.stdout)
        if not matched:
            matched=re.search(r"(\d+) failed, (\d+) passed",result.stdout)
            failed=int(matched.group(1)) if matched else None
            passed=int(matched.group(2)) if matched else None
        else:
            passed=int(matched.group(1))
            failed=int(matched.group(2) or 0)
        failed_names=re.findall(r"^FAILED ([^ ]+) -",result.stdout,re.MULTILINE)
        runs[name]={"command":command,"exit_code":result.returncode,"passed":passed,
                    "failed":failed,"failed_tests":failed_names,
                    "raw_stdout_tail":result.stdout[-12000:]}
        print(f"V21 test audit {name}: exit={result.returncode}, passed={passed}, failed={failed}",flush=True)
    target=root/"results/v21/processed/v21_test_audit.json"
    target.parent.mkdir(parents=True,exist_ok=True)
    write_json_atomic(target,{"created_utc":datetime.now(UTC).isoformat(),
                              "runs":runs,"legacy_failures_preexisting_in_V20_report":
                              ["tests/test_v14.py::test_v14_integrity_manifest",
                               "tests/test_v16.py::test_v16_historical_bytes_unchanged_and_manifest"],
                              "old_frozen_manifests_modified":False})
    print(json.dumps({name:{"passed":x["passed"],"failed":x["failed"]} for name,x in runs.items()},indent=2))


if __name__=="__main__":
    main()
