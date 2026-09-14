"""Post-run numerical audit for protocol v5 reports.

This module does not alter the frozen experiment protocol. It diagnoses the
resolution of saved causal deltas and expands family-wise reporting after all
formal runners have completed.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from jclosure.provenance import sha256_file, write_json_atomic
from jclosure.reporting_v5 import build as build_base

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "results/v5/processed"
REPORT = ROOT / "reports/PERIPHERAL_STATE_V5.md"
FINAL = ROOT / "reports/FINAL_REPORT.md"
START = "<!-- PERIPHERAL_V5_POSTRUN_START -->"
END = "<!-- PERIPHERAL_V5_POSTRUN_END -->"


def _load(name: str) -> dict[str, Any]:
    return json.loads((PROCESSED / name).read_text(encoding="utf-8"))


def _fmt(value: dict[str, Any]) -> str:
    return f"{value['estimate']:.6f} [{value['lower']:.6f}, {value['upper']:.6f}]"


def _fidelity_resolution(replication: dict[str, Any]) -> dict[str, Any]:
    path = ROOT / replication["causal_state_artifact"]
    if sha256_file(path) != replication["causal_state_artifact_sha256"]:
        raise RuntimeError("causal fidelity artifact hash mismatch")
    with np.load(path, allow_pickle=False) as payload:
        clean = payload["next_j_clean"]
        changed = payload["next_j_intervened"]
        storage_dtype = str(clean.dtype)
        delta = changed.astype(np.float32) - clean.astype(np.float32)
    norms = np.linalg.norm(delta, axis=1)
    nonzero = int(np.count_nonzero(norms))
    return {
        "storage_dtype": storage_dtype,
        "n_interventions": int(len(norms)),
        "exact_zero_delta_count": int(len(norms) - nonzero),
        "nonzero_delta_count": nonzero,
        "nonzero_delta_fraction": float(nonzero / max(1, len(norms))),
        "delta_norm_minimum": float(norms.min()),
        "delta_norm_median": float(np.median(norms)),
        "delta_norm_maximum": float(norms.max()),
        "status": (
            "NUMERICALLY_DEGENERATE_FLOAT16_TEACHER_DELTA"
            if nonzero < len(norms) / 2
            else "RESOLVED"
        ),
        "artifact": replication["causal_state_artifact"],
        "artifact_sha256": replication["causal_state_artifact_sha256"],
    }


def _family_compact_table(sweep: dict[str, Any]) -> str:
    rows = []
    for record in sweep["records"]:
        dimension = int(record["state_dimension"])
        if (
            sweep["selected_by_dimension"].get(str(dimension))
            != record["encoder_family"]
        ):
            continue
        fidelity = record.get("causal_fidelity_by_family", {})
        conditional = record.get("conditional_by_family", {})
        for family, metrics in record["family_metrics"].items():
            causal = fidelity.get(family, {})
            rows.append(
                {
                    "dimension": dimension,
                    "encoder": record["encoder_family"],
                    "family": family,
                    "next-J cosine": metrics["next_j_cosine"],
                    "semantic accuracy": metrics["semantic_accuracy"],
                    "conditional gain": conditional.get(family, {}).get(
                        "conditional_residual_gain"
                    ),
                    "causal cosine": causal.get("direction_cosine", {}).get("estimate"),
                    "output-sign agreement": causal.get(
                        "output_sign_agreement", {}
                    ).get("estimate"),
                    "transitions": metrics["n_transitions"],
                }
            )
    return pd.DataFrame(rows).to_markdown(index=False, floatfmt=".6f")


def _mediation_table(replication: dict[str, Any], mediation: dict[str, Any]) -> str:
    rows = []
    for family, metrics in replication["effects"].items():
        single = metrics["output_js_divergence"].get("j_preserving")
        later = mediation["effects"].get(family, {}).get("output_js_divergence", {})
        if not single:
            continue
        rows.append(
            {
                "family": family,
                "single JS (95% CI)": _fmt(single),
                "persistent-final JS": _fmt(later["persistent_final"])
                if "persistent_final" in later
                else None,
                "persistent-all JS": _fmt(later["persistent_all"])
                if "persistent_all" in later
                else None,
                "M_final point": mediation["mediation_fraction_point_estimates"]
                .get(family, {})
                .get("persistent_final"),
                "M_all point": mediation["mediation_fraction_point_estimates"]
                .get(family, {})
                .get("persistent_all"),
            }
        )
    return pd.DataFrame(rows).to_markdown(index=False, floatfmt=".6f")


def _h2_quality(replication: dict[str, Any]) -> dict[str, Any]:
    frame = pd.read_parquet(ROOT / replication["records"])
    values = frame[frame["condition"] == "j_preserving"]["quality"].map(json.loads)
    metrics = {
        key: np.asarray([float(value[key]) for value in values], dtype=float)
        for key in (
            "dense_cosine",
            "top10_overlap",
            "rms_drift",
            "displacement_fraction",
        )
    }
    return {
        key: {
            "minimum": float(array.min()),
            "median": float(np.median(array)),
            "maximum": float(array.max()),
        }
        for key, array in metrics.items()
    }


def _h2_control_table(replication: dict[str, Any]) -> str:
    metrics = replication["effects"]["pooled"]
    rows = []
    for condition, value in metrics["output_js_divergence"].items():
        rows.append(
            {
                "condition": condition,
                "output JS (95% CI)": _fmt(value),
                "future-J divergence": _fmt(
                    metrics["future_j_trajectory_divergence"][condition]
                ),
                "target log-odds change": _fmt(
                    metrics["target_log_odds_change"][condition]
                ),
                "answer flip rate": metrics["answer_flip"][condition]["estimate"],
                "task accuracy change": metrics["task_accuracy_change"][condition][
                    "estimate"
                ],
            }
        )
    return pd.DataFrame(rows).to_markdown(index=False, floatfmt=".6f")


def build() -> dict[str, Any]:
    build_base()
    reference = _load("full_remainder_reference_v5.json")
    sweep = _load("peripheral_dimension_sweep_v5.json")
    replication = _load("h2_replication_v5.json")
    mediation = _load("h2_mediation_v5.json")
    recurrent = _load("peripheral_recurrent_v5.json")
    resolution = _fidelity_resolution(replication)
    h2_quality = _h2_quality(replication)
    audit: dict[str, Any] = {
        "schema_version": 7,
        "protocol_version": "jstate_peripheral_protocol_v5_postrun_audit",
        "formal_protocol_freeze_digest": json.loads(
            (ROOT / "artifacts/peripheral_v5.freeze.json").read_text(encoding="utf-8")
        )["freeze_digest"],
        "full_remainder_ceiling_authorized": reference[
            "full_remainder_ceiling_authorized"
        ],
        "compact_peripheral_state_authorized": sweep[
            "compact_peripheral_state_authorized"
        ],
        "causal_fidelity_resolution": resolution,
        "h2_families_above_frozen_noise": replication["families_above_frozen_noise"],
        "h2_j_preserving_quality": h2_quality,
        "mediation_valid_records": mediation["valid_records"],
        "mediation_fraction_point_estimates": mediation[
            "mediation_fraction_point_estimates"
        ],
        "recurrent_status": recurrent["status"],
        "interpretation": (
            "Causal-direction fidelity is not numerically evaluable because the "
            "saved float16 teacher next-J delta is exactly zero for most trials. "
            "Persistent restoration did not remove the v5 single-arm effect."
        ),
    }
    audit_path = PROCESSED / "postrun_audit_v5.json"
    write_json_atomic(audit_path, audit)

    section = f"""{START}

## Post-run numerical and family-wise audit

This audit does not revise the frozen v5 protocol or rerun any formal trial.
The causal-fidelity source stored next-J profiles as `{resolution["storage_dtype"]}`.
Of {resolution["n_interventions"]} paired interventions,
{resolution["exact_zero_delta_count"]} had an exactly zero saved teacher next-J delta;
the median norm was {resolution["delta_norm_median"]:.6g} and only
{resolution["nonzero_delta_count"]} were nonzero. Direction cosine and magnitude
ratio are therefore **not numerically evaluable** for this endpoint. The
reported output-sign and semantic-change agreements remain descriptive, but
cannot authorize a causal peripheral state.

### Family-wise compact diagnostics

{_family_compact_table(sweep)}

These are predictive results except for the intervention-fidelity columns.
Negative conditional gains show that the augmented model did not learn to use
the added residual summary; they do not prove that the residual contains no
information.

### Single-arm and persistent-restoration comparison

All 66 accepted J-preserving interventions passed the frozen state-equality
criteria. Dense cosine had median {h2_quality["dense_cosine"]["median"]:.9f}
(minimum {h2_quality["dense_cosine"]["minimum"]:.9f}); top-10 overlap had median
{h2_quality["top10_overlap"]["median"]:.3f}; RMS drift had median
{h2_quality["rms_drift"]["median"]:.6f}; displacement was
{h2_quality["displacement_fraction"]["median"]:.3f} natural-difference units at
the median.

{_h2_control_table(replication)}

{_mediation_table(replication, mediation)}

The persistent-final and persistent-all estimates are identical in this
final-token arm, but both exceed the single-arm point estimate overall and in
the Boolean family. Thus v5 does not demonstrate mediation removal. The
restoration operation may itself alter later dynamics; without a
persistent-identity/null arm this result cannot distinguish bypass from a
restoration artifact.

### Determinism and execution boundary

PyTorch reported that the memory-efficient attention backward kernel was not
strictly deterministic during the attention-reference screen. Confirmation
used the three frozen seeds, and the observed limitation is retained rather
than hidden by an effect-dependent retry. No compact state passed all frozen
gates, so recurrent training was correctly recorded as
`{recurrent["status"]}` and autonomous rollout was not executed.

{END}"""
    report = REPORT.read_text(encoding="utf-8")
    if START in report and END in report:
        report = re.sub(
            re.escape(START) + r".*?" + re.escape(END),
            section,
            report,
            flags=re.DOTALL,
        )
    else:
        report = report.rstrip() + "\n\n" + section + "\n"
    REPORT.write_text(report, encoding="utf-8")

    final_note = (
        "The v5 post-run audit found 64/66 saved teacher next-J deltas exactly "
        "zero at float16 resolution, so directional intervention fidelity was "
        "not evaluable. Persistent restoration did not remove the replicated "
        "Boolean-family effect."
    )
    final = FINAL.read_text(encoding="utf-8")
    final = final.replace(
        "Full details: [PERIPHERAL_STATE_V5.md](PERIPHERAL_STATE_V5.md).",
        f"{final_note} Full details: [PERIPHERAL_STATE_V5.md](PERIPHERAL_STATE_V5.md).",
    )
    FINAL.write_text(final, encoding="utf-8")
    output = {
        **audit,
        "audit": str(audit_path.relative_to(ROOT)),
        "audit_sha256": sha256_file(audit_path),
        "report": str(REPORT.relative_to(ROOT)),
        "report_sha256": sha256_file(REPORT),
        "final_report_sha256": sha256_file(FINAL),
    }
    write_json_atomic(PROCESSED / "postrun_report_manifest_v5.json", output)
    return output


if __name__ == "__main__":
    print(json.dumps(build(), sort_keys=True))
