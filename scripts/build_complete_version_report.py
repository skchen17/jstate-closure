#!/usr/bin/env python3
"""Build one self-contained Markdown report for a project version."""

from __future__ import annotations

import argparse
import hashlib
import subprocess
from datetime import UTC, datetime
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def extract_final_section(text: str, version: str) -> str:
    start = f"<!-- {version}_START -->"
    end = f"<!-- {version}_END -->"
    if start not in text or end not in text:
        return "No version-specific section was found in `reports/FINAL_REPORT.md`."
    return text.split(start, 1)[1].split(end, 1)[0].strip()


def report_order(version: str, names: list[Path]) -> list[Path]:
    if version == "V11":
        preferred = [
            "CHANNELWISE_CAUSAL_DECODER_AUDIT_V11.md",
            "ORACLE_LOWRANK_CAUSAL_STATE_V11.md",
            "FACTORIZED_CAUSAL_STATE_V11.md",
            "CAUSAL_STATE_MANIFOLD_AUDIT_V11.md",
            "CAUSAL_ERROR_AMPLIFICATION_V11.md",
            "ARCH_RESOLVED_CEILING_V11.md",
        ]
        lookup = {path.name: path for path in names}
        ordered = [lookup.pop(name) for name in preferred if name in lookup]
        return [*ordered, *sorted(lookup.values())]
    return sorted(names)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("version", help="version label such as V11")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    root = args.root.resolve()
    version = args.version.upper()
    reports_dir = root / "reports"
    output = reports_dir / f"{version}_COMPLETE_REPORT.md"
    sources = [
        path
        for path in reports_dir.glob(f"*_{version}.md")
        if path.resolve() != output.resolve()
    ]
    sources = report_order(version, sources)
    if not sources:
        raise SystemExit(f"no reports found for {version}")
    final_report = reports_dir / "FINAL_REPORT.md"
    final_section = extract_final_section(
        final_report.read_text(encoding="utf-8"), version
    )
    result_dir = root / "results" / version.lower() / "processed"
    machine_records = sorted(path for path in result_dir.glob("*") if path.is_file())
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip()
    lines = [
        f"# {version} complete report",
        "",
        "> This is the canonical single-file bundle for this version. It combines the ",
        "> version-specific FINAL_REPORT section, every standalone version report, and ",
        "> an integrity index of the machine-readable records. Standalone reports remain ",
        "> preserved for direct navigation.",
        "",
        "## Bundle provenance",
        "",
        f"- Source commit: `{commit}`",
        f"- Generated at: `{datetime.now(UTC).isoformat()}`",
        f"- Included standalone reports: `{len(sources)}`",
        f"- Indexed machine-record files: `{len(machine_records)}`",
        "- Generator: `scripts/build_complete_version_report.py`",
        "",
        "## Version summary and adjudication",
        "",
        final_section,
        "",
        "## Standalone report integrity index",
        "",
        "| report | SHA256 |",
        "|---|---|",
    ]
    for path in sources:
        lines.append(f"| `{path.relative_to(root)}` | `{sha256_file(path)}` |")
    lines.extend(
        [
            "",
            "## Machine-record integrity index",
            "",
            "| record | bytes | SHA256 |",
            "|---|---:|---|",
        ]
    )
    for path in machine_records:
        lines.append(
            f"| `{path.relative_to(root)}` | {path.stat().st_size} | "
            f"`{sha256_file(path)}` |"
        )
    for index, path in enumerate(sources, start=1):
        lines.extend(
            [
                "",
                "---",
                "",
                f"## Bundled report {index}: `{path.name}`",
                "",
                path.read_text(encoding="utf-8").strip(),
            ]
        )
    output.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    print(output.relative_to(root))


if __name__ == "__main__":
    main()
