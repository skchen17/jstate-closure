"""Assemble one append-only V29 file containing every individual V29 report verbatim."""
from __future__ import annotations

import json
from pathlib import Path

from jclosure.protocol_v29 import stage_freeze, verify_stage
from jclosure.provenance import sha256_file, write_json_atomic
from jclosure.reporting_v29 import REPORTS

SOURCE = "src/jclosure/all_reports_v29.py"
TARGET = Path("reports/V29_ALL_REPORTS.md")
MANIFEST = Path("results/v29/processed/v29_all_reports_manifest.json")


def run(root: Path):
    verify_stage(root, "final")
    path = root / TARGET
    if path.exists():
        raise RuntimeError("V29 all-reports bundle already exists")
    sources = [root / "reports" / name for name in REPORTS]
    source_hashes = {str(p.relative_to(root)): sha256_file(p) for p in sources}
    sections = ["# V29 — All Reports in One File", "", "This bundle contains all 22 frozen V29 reports verbatim, in the required report order. The individual files remain the authoritative sources.", ""]
    for index, source in enumerate(sources, 1):
        sections.extend(("---", "", f"<!-- {index:02d}: {source.name}; sha256={source_hashes[str(source.relative_to(root))]} -->", "", source.read_text(encoding="utf-8").rstrip(), ""))
    path.write_text("\n".join(sections).rstrip() + "\n", encoding="utf-8")
    manifest = {"bundle": str(TARGET), "bundle_sha256": sha256_file(path), "source_report_count": len(sources), "source_sha256": source_hashes, "historical_final_opened": False}
    write_json_atomic(root / MANIFEST, manifest)
    fr = stage_freeze(root, "all_reports", [SOURCE, str(TARGET), str(MANIFEST), "artifacts/natural_write_content_v29_final.freeze.json", *source_hashes], manifest)
    return {"freeze_digest": fr["freeze_digest"], "bundle": str(TARGET), "reports": len(sources), "bundle_sha256": manifest["bundle_sha256"]}


if __name__ == "__main__":
    print(json.dumps(run(Path.cwd()), indent=2))
