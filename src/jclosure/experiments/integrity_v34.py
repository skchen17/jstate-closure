"""SHA-256 manifest for every committed V34 code, machine record and report."""
from __future__ import annotations

import json
from pathlib import Path

from jclosure.protocol_v34 import stage_freeze, verify, verify_stage
from jclosure.provenance import sha256_file, write_json_atomic

SOURCE = "src/jclosure/experiments/integrity_v34.py"
OUT = Path("results/v34/processed")


def run(root: Path) -> dict:
    verify_stage(root, "adjudication")
    paths = [root / "configs/functional_mediation_v34.yaml", root / "src/jclosure/protocol_v34.py",
             root / "tests/test_v34.py", root / "reports/FINAL_REPORT.md"]
    paths += sorted((root / "src/jclosure/experiments").glob("*v34.py"))
    paths += sorted((root / "artifacts").glob("functional_mediation_v34*.freeze.json"))
    paths += sorted((root / OUT).glob("*"))
    paths += sorted((root / "reports").glob("V34_*.md"))
    excluded = {root / OUT / "v34_integrity_index.json",
                root / "artifacts/functional_mediation_v34_integrity.freeze.json"}
    files = {}
    for path in paths:
        if path in excluded:
            continue
        if not path.is_file():
            raise RuntimeError(f"V34 manifest missing file {path}")
        relative = str(path.relative_to(root)).replace("\\", "/")
        if relative in files:
            raise RuntimeError(f"V34 manifest duplicate {relative}")
        files[relative] = {"sha256": sha256_file(path), "bytes": path.stat().st_size}
    if len([x for x in files if x.startswith("reports/V34_")]) < 22:
        raise RuntimeError("V34 required reports incomplete")
    record = {"protocol": "functional_mediation_v34", "base_freeze_digest": verify(root)["freeze_digest"],
              "adjudication_freeze_digest": verify_stage(root, "adjudication")["freeze_digest"],
              "model_hashes": {key: {"revision": val["revision"], "weights": val["weight_sha256"],
                                     "tokenizer": val["tokenizer_sha256"], "config": val["config_sha256"]}
                               for key, val in verify(root)["config"]["models"].items()},
              "files": files, "file_count": len(files), "historical_records_mutated": False,
              "model_weights_or_activation_banks_committed": False}
    index = root / OUT / "v34_integrity_index.json"
    write_json_atomic(index, record)
    seal = stage_freeze(root, "integrity", [SOURCE, str(index.relative_to(root)),
                                           "artifacts/functional_mediation_v34_adjudication.freeze.json"],
                        {"file_count": len(files), "index_sha256": sha256_file(index)})
    return {"freeze_digest": seal["freeze_digest"], "file_count": len(files),
            "reports": len([x for x in files if x.startswith("reports/V34_")])}


if __name__ == "__main__":
    print(json.dumps(run(Path.cwd()), indent=2))
