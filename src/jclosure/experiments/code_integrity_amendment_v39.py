"""Append-only correction for V39 metric-audit source omitted by *v39.py glob."""
from __future__ import annotations

import json
from pathlib import Path

from jclosure.protocol_v39 import stage_freeze, verify_stage
from jclosure.provenance import sha256_file, write_json_atomic


OUT = Path("results/v39/processed")
SOURCE = "src/jclosure/experiments/code_integrity_amendment_v39.py"
MISSING = "src/jclosure/experiments/v39_metric_audit.py"


def run(root: Path) -> dict:
    verify_stage(root, "code_integrity")
    previous = json.loads((root / OUT / "v39_code_integrity_index.json").read_text())
    if MISSING in previous["sha256"]:
        raise RuntimeError("V39 code index already contains metric audit")
    files = [SOURCE, MISSING]
    result = {"amends": "results/v39/processed/v39_code_integrity_index.json",
              "reason": "The original *v39.py pattern does not match v39_metric_audit.py.",
              "additional_sha256": {name: sha256_file(root / name) for name in files},
              "original_index_unchanged": True}
    path = root / OUT / "v39_code_integrity_amendment.json"
    write_json_atomic(path, result)
    seal = stage_freeze(root, "code_integrity_amendment",
                        [*files, str(path.relative_to(root)),
                         "artifacts/interaction_genesis_v39_code_integrity.freeze.json"],
                        {"summary_sha256": sha256_file(path),
                         "original_index_unchanged": True})
    return {"freeze_digest": seal["freeze_digest"], **result}


if __name__ == "__main__":
    print(json.dumps(run(Path.cwd()), indent=2))
