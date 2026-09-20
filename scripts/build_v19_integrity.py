#!/usr/bin/env python3
"""V19 machine integrity index and old-version tracked-byte guard."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from jclosure.protocol_v19 import verify, verify_stage
from jclosure.provenance import sha256_file, write_json_atomic

ROOT = Path.cwd()
PARENT = "3d670f434c0f86bba87e2bbb8d0f9c2c2d567df1"


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def main():
    base = verify(ROOT)
    stage_names = ["splits", "interventions", "q_amendment_1", "q_amendment_2",
                   "teacher_amendment_3", "analysis", "diagnostics", "runtime_gpu1_amendment_4",
                   "diagnostic_gpu1_amendment_5", "execution_binding_amendment_6",
                   "gpu1_reverse_train_amendment_7"]
    if (ROOT / "artifacts/counterfactual_workspace_v19_compact_response.freeze.json").exists():
        stage_names.append("compact_response")
    stages = {name: verify_stage(ROOT, name)["freeze_digest"] for name in stage_names}
    old_paths = git("ls-tree", "-r", "--name-only", PARENT).splitlines()
    historical = [name for name in old_paths if name.startswith(("artifacts/", "configs/", "reports/", "results/", "src/", "tests/"))
                  and name != "reports/FINAL_REPORT.md"]
    modified = set(git("diff", "--name-only", PARENT, "--").splitlines())
    changed = [{"path": name, "status": "missing" if not (ROOT / name).exists() else "content_changed"}
               for name in historical if name in modified or not (ROOT / name).exists()]
    if changed:
        raise RuntimeError(f"V1–V18 tracked record changed: {changed[:10]}")
    paths = sorted([*ROOT.glob("artifacts/*v19*.json"), *ROOT.glob("configs/*v19*.yaml"),
                    *ROOT.glob("src/jclosure/*v19.py"), *ROOT.glob("src/jclosure/experiments/*v19.py"),
                    *ROOT.glob("scripts/*v19*.py"), *ROOT.glob("tests/*v19.py"),
                    *ROOT.glob("reports/*_V19.md"), *ROOT.glob("results/v19/processed/*")])
    paths = [path for path in paths if path.is_file() and path.name != "integrity_v19.json"]
    indexed = [{"path": str(path.relative_to(ROOT)), "bytes": path.stat().st_size,
                "sha256": sha256_file(path)} for path in paths]
    result = {"protocol_digest": base["freeze_digest"], "stage_digests": stages,
              "historical_parent_commit": PARENT, "historical_tracked_paths_checked": len(historical),
              "historical_changed": changed, "final_report_append_only_exception": True,
              "indexed_files": indexed, "indexed_count": len(indexed)}
    target = ROOT / "results/v19/processed/integrity_v19.json"
    write_json_atomic(target, result)
    print(json.dumps({"indexed_count": len(indexed), "historical_paths_checked": len(historical),
                      "historical_changed": len(changed), "path": str(target.relative_to(ROOT))}, indent=2))


if __name__ == "__main__":
    main()
