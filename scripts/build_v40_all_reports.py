"""Build the version-local, self-contained V40 report bundle."""
from __future__ import annotations

import hashlib
from pathlib import Path

ORDER = (
    "V40_FROZEN_STARTING_POINT.md",
    "V40_CHANNEL_FUNCTION_PROFILE.md",
    "V40_TASK_GROUNDED_RESULTS.md",
    "V40_TEMPORAL_PERSISTENCE.md",
    "V40_CHANNEL_COMPARISON.md",
    "V40_TRAJECTORY_CAUSALITY.md",
    "V40_COMPLETE_REPORT.md",
)


def build(root: Path) -> Path:
    reports = root / "reports"
    lines = ["# V40 — All Reports", "",
             "Startup checkpoint only; this bundle contains no intervention outcomes.", ""]
    for name in ORDER:
        path = reports / name
        content = path.read_text(encoding="utf-8").strip()
        sha = hashlib.sha256(path.read_bytes()).hexdigest()
        lines.extend([f"<!-- {name} | SHA256 {sha} -->", "", content, "", "---", ""])
    output = reports / "V40_ALL_REPORTS.md"
    output.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return output


if __name__ == "__main__":
    print(build(Path.cwd()))
