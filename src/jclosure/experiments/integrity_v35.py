"""SHA-256 index for V35 source, frozen stages, machine evidence and all reports."""
from __future__ import annotations

import json
from pathlib import Path

from jclosure.protocol_v35 import stage_freeze, verify, verify_stage
from jclosure.provenance import sha256_file, write_json_atomic

SOURCE = "src/jclosure/experiments/integrity_v35.py"
OUT = Path("results/v35/processed")


def run(root: Path):
    verify_stage(root, "adjudication")
    base = verify(root)
    paths = [root / "configs/hierarchical_read_v35.yaml", root / "src/jclosure/protocol_v35.py",
             root / "tests/test_v35.py", root / "reports/FINAL_REPORT.md"]
    paths += sorted((root / "src/jclosure/experiments").glob("*v35.py"))
    paths += sorted((root / "artifacts").glob("hierarchical_read_v35*.freeze.json"))
    paths += sorted((root / OUT).glob("*"))
    paths += sorted((root / "reports").glob("V35_*.md"))
    excluded = {root / OUT / "v35_integrity_index.json",
                root / "artifacts/hierarchical_read_v35_integrity.freeze.json"}
    files = {}
    for path in paths:
        if path in excluded:
            continue
        if not path.is_file():
            raise RuntimeError(f"V35 manifest missing file {path}")
        relative = str(path.relative_to(root)).replace("\\", "/")
        if relative in files:
            raise RuntimeError(f"V35 manifest duplicate {relative}")
        files[relative] = {"sha256": sha256_file(path), "bytes": path.stat().st_size}
    report_count = sum(path.startswith("reports/V35_") for path in files)
    if report_count < 23 or "reports/V35_ALL_REPORTS.md" not in files:
        raise RuntimeError("V35 required per-topic and all-in-one reports incomplete")
    record = {"protocol": "hierarchical_read_v35", "base_freeze_digest": base["freeze_digest"],
              "adjudication_freeze_digest": verify_stage(root, "adjudication")["freeze_digest"],
              "model_hashes": {key: {"revision": val["revision"], "weights": val["weight_sha256"],
                                      "tokenizer": val["tokenizer_sha256"], "config": val["config_sha256"]}
                               for key, val in base["model_specs"].items()},
              "files": files, "file_count": len(files), "report_count": report_count,
              "historical_records_mutated": False, "model_weights_or_activation_banks_committed": False}
    path = root / OUT / "v35_integrity_index.json"
    write_json_atomic(path, record)
    seal = stage_freeze(root, "integrity", [SOURCE, str(path.relative_to(root)),
                                          "artifacts/hierarchical_read_v35_adjudication.freeze.json"],
                        {"file_count": len(files), "report_count": report_count,
                         "index_sha256": sha256_file(path)})
    return {"freeze_digest": seal["freeze_digest"], "file_count": len(files),
            "report_count": report_count}


if __name__ == "__main__":
    print(json.dumps(run(Path.cwd()), indent=2))
