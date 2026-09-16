"""Append the explicit v11 manifold-metric amendment to generated reports."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from jclosure.protocol_v11_manifold_amendment import FREEZE_PATH
from jclosure.provenance import sha256_file


def _read(root: Path, path: str | Path) -> dict[str, Any]:
    return json.loads((root / path).read_text(encoding="utf-8"))


def _append_once(path: Path, marker: str, text: str) -> None:
    current = path.read_text(encoding="utf-8")
    if marker in current:
        current = current.split(marker, 1)[0].rstrip()
    path.write_text(current + "\n\n" + text.rstrip() + "\n", encoding="utf-8")


def write_amended_reports(root: Path) -> dict[str, Any]:
    freeze = _read(root, FREEZE_PATH)
    manifold = _read(
        root, "results/v11/processed/causal_state_manifold_v11_amendment_1.json"
    )
    interaction = _read(
        root, "results/v11/processed/channel_interaction_nonadditivity_v11.json"
    )
    marker = "<!-- V11_MANIFOLD_AMENDMENT_1 -->"
    clean_rows = []
    for channel, values in manifold["aggregates"]["clean_zero"].items():
        clean_rows.append(
            f"| {channel} | undefined | {values['cycle_absolute_error']:.3e} |"
        )
    manifold_text = """{marker}
## Manifold metric amendment 1

The original clean-zero *relative* cycle error divided by `||query||=0` and is undefined.
The original record remains preserved; corrected records report it as null and report the
absolute score-space cycle error separately. All nonzero teacher/decoded relative cycle
metrics are unchanged.

| channel | relative cycle error | absolute cycle error |
|---|---:|---:|
{rows}

Amendment freeze: `{digest}`. Corrected records: `{records}` (`{records_hash}`).
""".format(
        marker=marker,
        rows="\n".join(clean_rows),
        digest=freeze["freeze_digest"],
        records=manifold["records"],
        records_hash=manifold["records_sha256"],
    )
    _append_once(
        root / "reports/CAUSAL_STATE_MANIFOLD_AUDIT_V11.md", marker, manifold_text
    )
    interaction_rows = []
    for condition, horizons in interaction["aggregates"].items():
        for horizon, values in sorted(horizons.items(), key=lambda item: int(item[0])):
            interaction_rows.append(
                f"| {condition} | {horizon} | {values['mean_joint_error_norm']:.4f} | "
                f"{values['mean_single_error_rss']:.4f} | "
                f"{values['mean_joint_to_rss_ratio']:.3f} |"
            )
    interaction_text = """{marker}
## Channel nonadditivity amendment 1

The retained records do not contain full error vectors, so exact cross terms cannot be
recovered. The frozen descriptive proxy compares observed joint error norm with the
root-sum-square of single-channel error norms. Ratio >1 is compatible with constructive
interaction; ratio <1 with cancellation.

| decoded channels | h | joint error | single RSS | joint/RSS |
|---|---:|---:|---:|---:|
{rows}

Amendment freeze: `{digest}`. Machine records: `{records}` (`{records_hash}`).
""".format(
        marker=marker,
        rows="\n".join(interaction_rows),
        digest=freeze["freeze_digest"],
        records=interaction["records"],
        records_hash=interaction["records_sha256"],
    )
    _append_once(
        root / "reports/CHANNELWISE_CAUSAL_DECODER_AUDIT_V11.md",
        marker,
        interaction_text,
    )
    final_text = """{marker}
### V11 manifold metric amendment 1

The clean-zero relative cycle metric in the original V11 manifold record is undefined
because its denominator is zero. It is superseded only for that row by
`causal_state_manifold_v11_amendment_1`; nonzero teacher/decoded metrics are unchanged.
Channel joint-error nonadditivity is additionally reported as a frozen norm proxy.
Amendment freeze: `{digest}`. Commands:

- `bash scripts/run_causal_geometry_v11_manifold_amendment.sh freeze`
- `bash scripts/run_causal_geometry_v11_manifold_amendment.sh analyze`
- `bash scripts/run_causal_geometry_v11_manifold_amendment.sh report`
""".format(marker=marker, digest=freeze["freeze_digest"])
    _append_once(root / "reports/FINAL_REPORT.md", marker, final_text)
    paths = [
        "reports/CHANNELWISE_CAUSAL_DECODER_AUDIT_V11.md",
        "reports/CAUSAL_STATE_MANIFOLD_AUDIT_V11.md",
        "reports/FINAL_REPORT.md",
    ]
    return {name: sha256_file(root / name) for name in paths}
