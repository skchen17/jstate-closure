"""Supplemental immutable index of all V39 implementation and test source files."""
from __future__ import annotations

import json
from pathlib import Path

from jclosure.protocol_v39 import stage_freeze, verify_stage
from jclosure.provenance import sha256_file, write_json_atomic


OUT = Path("results/v39/processed")
SOURCE = "src/jclosure/experiments/code_integrity_v39.py"


def run(root: Path) -> dict:
    verify_stage(root, "report")
    files = [root / "src/jclosure/protocol_v39.py",
             *sorted((root / "src/jclosure/experiments").glob("*v39.py")),
             root / "tests/test_v39.py", root / "configs/genesis_v39.yaml",
             root / OUT / "v39_integrity_index.json"]
    indexed = {str(path.relative_to(root)): sha256_file(path) for path in files}
    path = root / OUT / "v39_code_integrity_index.json"
    if path.exists():
        raise RuntimeError("V39 code integrity index already exists")
    result = {"protocol": "interaction_genesis_v39",
              "supplements": "results/v39/processed/v39_integrity_index.json",
              "source_and_test_file_count": len(files)-2,
              "sha256": indexed}
    write_json_atomic(path, result)
    seal = stage_freeze(root, "code_integrity",
                        [*indexed, str(path.relative_to(root))],
                        {"file_count": len(files), "summary_sha256": sha256_file(path)})
    return {"freeze_digest": seal["freeze_digest"], "file_count": len(files)}


if __name__ == "__main__":
    print(json.dumps(run(Path.cwd()), indent=2))
