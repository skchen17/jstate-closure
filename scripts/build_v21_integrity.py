"""Verify append-only V21 freezes and index all version files before bundling."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from jclosure.protocol_v21 import PARENT,verify,verify_stage
from jclosure.provenance import sha256_file,write_json_atomic


def main():
    root=Path.cwd()
    base=verify(root)
    freeze_files=sorted((root/"artifacts").glob("action_coordinate_geometry_v21_*.freeze.json"))
    stages={path.name.removeprefix("action_coordinate_geometry_v21_").removesuffix(".freeze.json"):
            verify_stage(root,path.name.removeprefix("action_coordinate_geometry_v21_").removesuffix(".freeze.json"))["freeze_digest"]
            for path in freeze_files}
    tracked=subprocess.check_output(["git","diff","--name-only",PARENT,"--","."],cwd=root,text=True).splitlines()
    if any(name!="reports/FINAL_REPORT.md" for name in tracked):
        raise RuntimeError(f"V1–V20 tracked content changed: {tracked}")
    adjudication=json.loads((root/"results/v21/processed/v21_adjudication.json").read_text())
    if adjudication["flags"]["FINAL_SIX_ACTION_RESPONSES_OPENED"] or adjudication["flags"]["DYNAMIC_STATE_SEARCH_AUTHORIZED"]:
        raise RuntimeError("V21 final or dynamics invariant violated")
    paths=[root/"configs/action_coordinate_geometry_v21.yaml",
           root/"src/jclosure/protocol_v21.py",
           root/"tests/test_v21.py",
           root/"artifacts/action_coordinate_geometry_v21.freeze.json",
           *freeze_files,
           *sorted((root/"src/jclosure/experiments").glob("*v21.py")),
           *sorted((root/"scripts").glob("*v21*.py")),
           *sorted((root/"results/v21/processed").glob("*")),
           *sorted((root/"reports").glob("*_V21.md")),
           root/"reports/FINAL_REPORT.md"]
    target=root/"results/v21/processed/v21_integrity_index.json"
    unique=sorted({path for path in paths if path.is_file() and path!=target})
    files={str(path.relative_to(root)):{"bytes":path.stat().st_size,"sha256":sha256_file(path)}
           for path in unique}
    result={"protocol_digest":base["freeze_digest"],"parent_commit":PARENT,
            "parent_tracked_changes":tracked,"verified_stage_digests":stages,
            "files":files,"file_count":len(files),"old_frozen_records_unchanged":True,
            "sealed_final_action_responses_unopened":True,
            "dynamic_state_search_unauthorized":True,
            "complete_report_bundle_not_yet_generated":True}
    write_json_atomic(target,result)
    print(json.dumps({"file_count":len(files),"verified_freeze_stages":len(stages),
                      "index_sha256":sha256_file(target)},indent=2))


if __name__=="__main__":
    main()
