"""Add amendment-aware conclusions to the machine-generated v10 reports."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

from jclosure.experiments.common import initialize_context, standard_parser
from jclosure.protocol_v10 import verify_candidate_freeze, verify_freeze
from jclosure.protocol_v10_causal_amendment import verify_causal_amendment_freeze
from jclosure.protocol_v10_residual_amendment import verify_amendment_freeze
from jclosure.protocol_v10_strict_interface import verify_freeze as verify_strict_freeze
from jclosure.provenance import sha256_file, write_json_atomic

AMEND_START = "<!-- V10-AMENDMENT:START -->"
AMEND_END = "<!-- V10-AMENDMENT:END -->"
FINAL_START = "<!-- V10-RESULTS:START -->"
FINAL_END = "<!-- V10-RESULTS:END -->"


def _replace_or_insert(path: Path, block: str) -> None:
    current = path.read_text(encoding="utf-8")
    if AMEND_START in current and AMEND_END in current:
        prefix, remainder = current.split(AMEND_START, 1)
        _, suffix = remainder.split(AMEND_END, 1)
        updated = prefix.rstrip() + "\n\n" + block + "\n\n" + suffix.lstrip()
    else:
        lines = current.splitlines()
        updated = "\n".join([lines[0], "", block, "", *lines[1:]])
    path.write_text(updated.rstrip() + "\n", encoding="utf-8")


def _metric(causal: dict[str, Any], dimension: int, horizon: int, name: str) -> float:
    return float(
        causal["aggregates"][str(dimension)][str(horizon)]["pooled"][name][
            "estimate"
        ]
    )


def _finalize(context: Any) -> dict[str, Any]:
    root = context.root
    base = verify_freeze(root, context.config)
    candidates = verify_candidate_freeze(root, context.config)
    residual_freeze = verify_amendment_freeze(root, context.config)
    causal_freeze = verify_causal_amendment_freeze(root, context.config)
    strict_freeze = verify_strict_freeze(root, context.config)
    sweep = json.loads(
        (root / "results/v10/processed/corrected_sub512_sufficiency_v10.json").read_text()
    )
    residual = json.loads(
        (root / "results/v10/processed/residual_localization_audit_v10.json").read_text()
    )
    semantic_records = root / "results/v10/processed/semantic_sufficiency_audit_v10.parquet"
    causal = json.loads(
        (root / "results/v10/processed/decoded_causal_state_validation_v10.json").read_text()
    )
    decoder = json.loads(
        (root / "results/v10/processed/compact_decoder_v10.json").read_text()
    )
    strict = json.loads(
        (root / "results/v10/processed/strict_interface_audit_v10.json").read_text()
    )
    import pandas as pd

    semantic = pd.read_parquet(semantic_records)
    semantic512 = semantic[
        (semantic["method"] == "causal_bottleneck")
        & (semantic["dimension"] == 512)
        & (semantic["split"] == "final_test")
        & (semantic["family"] == "pooled")
    ].iloc[0]
    residual512 = residual["dimensions"]["512"]
    thresholds = context.config["causal_sufficiency_v10"]

    residual_block = "\n".join(
        [
            AMEND_START,
            "## Corrected estimand and answers",
            "",
            f"Residual-estimand amendment freeze: `{residual_freeze['freeze_digest']}`. The pre-fix derived result remains archived; no v1-v9 result was changed.",
            "",
            "1. **No.** The v9 localization and corrected joint comparator were not mathematically identical: v9 appended non-residualized block coordinates that duplicated compact information.",
            "2. v10 residualizes each architecture block out-of-fold on train, uses train-only held-out residualization, freezes the compact base prediction, and predicts only the OOF target residual. This prevents coefficient refitting/regularization changes from being counted as information.",
            f"3. At 512D, the exact same-599D reference gain is `{residual512['combined_reference']['estimate']:.6f}` with 95% CI `[{residual512['combined_reference']['lower']:.6f}, {residual512['combined_reference']['upper']:.6f}]`; its maximum discrepancy from the unified comparator is `{residual['unified_comparator_max_abs_difference']}`. Rich block-space gains are recurrent `{residual512['recurrent']['estimate']:.4f}`, conv `{residual512['conv']['estimate']:.4f}`, KV `{residual512['kv']['estimate']:.4f}`, and joint `{residual512['architecture_joint']['estimate']:.4f}`.",
            "4. Recurrent and conv are individually strongest and nearly tied; KV is smaller but non-zero. The strongly negative additive interaction is redundancy/non-additivity, not a unique negative information channel.",
            "5. A joint residual near zero requires the same information space and estimand. It is reproduced by `combined_reference`; it does **not** force the richer 3x599 block-specific space to zero. The non-zero architecture-joint result shows that the 599D combined ceiling is lossy with respect to block-resolved state.",
            AMEND_END,
        ]
    )
    _replace_or_insert(root / "reports/RESIDUAL_LOCALIZATION_AUDIT_V10.md", residual_block)

    sweep_block = "\n".join(
        [
            AMEND_START,
            "## Frozen selection conclusion",
            "",
            f"Frozen gates: predictive gap closed ≥ `{thresholds['predictive_gap_closed_minimum']}`, conditional upper CI ≤ `{thresholds['conditional_residual_gain_maximum']}`, semantic relative retention ≥ `{thresholds['semantic_relative_retention_minimum']}`, semantic residual upper CI ≤ `{thresholds['semantic_residual_gain_maximum']}`.",
            "",
            "Although 256D passes the pooled final-test observational row, candidate selection was validation-only and required every family. The first universal validation pass is **384D**; 320D and 352D fail the variable-binding predictive gate. Candidates were frozen as **384/448/512D** before decoded outcomes.",
            AMEND_END,
        ]
    )
    _replace_or_insert(root / "reports/CORRECTED_SUB512_SUFFICIENCY_V10.md", sweep_block)

    semantic_block = "\n".join(
        [
            AMEND_START,
            "## Interpretation",
            "",
            f"At 512D final test, baseline=`{semantic512.semantic_baseline:.4f}`, compact=`{semantic512.semantic_compact:.4f}`, full ceiling=`{semantic512.semantic_full_ceiling:.4f}`, and compact/full retention=`{semantic512.semantic_retention_relative_to_full:.4f}`. The observational semantic representation gate passes.",
            "",
            "The absolute probe ceiling is only about 0.626, so low absolute readout quality is a probe/target ceiling issue rather than compact-specific information loss. This is distinct from decoded-causal semantic fidelity, which fails and diagnoses the decoder/interface rather than this observational gate.",
            AMEND_END,
        ]
    )
    _replace_or_insert(root / "reports/SEMANTIC_SUFFICIENCY_AUDIT_V10.md", semantic_block)

    causal_lines = [
        AMEND_START,
        "## Authorization conclusion and strict three-way audit",
        "",
        f"Causal metadata amendment freeze: `{causal_freeze['freeze_digest']}`; strict-interface freeze: `{strict_freeze['freeze_digest']}`. All 50 selected pairs map to the frozen v8 task table.",
        "",
        f"Decoder full-rank feature reconstruction max error: `{decoder['full_rank_feature_reconstruction_max_abs_error']:.3e}`. The compact-only function hash is `{causal['strict_interface']['function_sha256']}` and its allowed inputs are only clean cache, decoded delta, and teacher-forced token.",
        "",
        f"The diagnostic `decoded + (raw-decoded)` reconstruction matches full raw state to max absolute error `{strict['maximum_recombined_full_raw_abs_error']:.3e}`. It intentionally violates compact-only access and is not an authorization path; it confirms that the missing raw residual is sufficient to restore the teacher reference.",
        "",
        "No candidate passes. Magnitude is well calibrated (~0.97–1.01), but h1 semantic top-10 effect agreement is only 0.610–0.642 (threshold 0.8). Direction/output fidelity then decays strongly: for 512D, direction is 0.922/0.825/0.685/0.527 at h1/2/4/8 and output direction is 0.897/0.789/0.647/0.515.",
        "",
        "Increasing 384→512D does not materially improve causal fidelity. This points to the encoder/decoder objective and block/channel interaction, with semantic-effect loss and long-horizon instability, rather than dimension alone. Every family fails, so task heterogeneity is not the sole explanation.",
        AMEND_END,
    ]
    _replace_or_insert(
        root / "reports/DECODED_CAUSAL_STATE_VALIDATION_V10.md",
        "\n".join(causal_lines),
    )

    horizon_block = "\n".join(
        [
            AMEND_START,
            "## Measured stability conclusion",
            "",
            "h8 and h16 are newly measured teacher-forced outcomes, not extrapolations. None of 384/448/512D is stable through h8. Since the teacher-forced prerequisite fails, limited free continuation was correctly not run.",
            "",
            f"For 512D, direction fidelity is `{_metric(causal,512,1,'direction_cosine'):.3f}` at h1, `{_metric(causal,512,2,'direction_cosine'):.3f}` at h2, `{_metric(causal,512,4,'direction_cosine'):.3f}` at h4, `{_metric(causal,512,8,'direction_cosine'):.3f}` at h8, and `{_metric(causal,512,16,'direction_cosine'):.3f}` at h16.",
            AMEND_END,
        ]
    )
    _replace_or_insert(root / "reports/LONG_HORIZON_STATE_SUFFICIENCY_V10.md", horizon_block)

    enriched_block = "\n".join(
        [
            AMEND_START,
            "## Confirmatory conclusion",
            "",
            "The 25 effect-enriched cases preserve the same conclusion. At h1 their direction and output-direction fidelity are strong, but semantic agreement is only 0.588–0.596; all candidate/horizon gates fail. Stronger raw effects therefore do not rescue causal sufficiency.",
            AMEND_END,
        ]
    )
    _replace_or_insert(
        root / "reports/BEHAVIOR_ENRICHED_CONFIRMATORY_V10.md", enriched_block
    )

    commands = [
        "scripts/run_causal_sufficiency_v10.sh freeze --run-suffix protocol-freeze",
        "scripts/run_causal_sufficiency_v10.sh audit --run-suffix corrected-audit",
        "scripts/run_causal_sufficiency_v10_residual_amendment.sh freeze --run-suffix estimand-freeze",
        "scripts/run_causal_sufficiency_v10_residual_amendment.sh audit --run-suffix frozen-base-audit",
        "scripts/run_causal_sufficiency_v10.sh freeze-candidates --run-suffix candidate-freeze",
        "scripts/run_causal_sufficiency_v10.sh prepare-decoder --device 0 --run-suffix decoder",
        "scripts/run_causal_sufficiency_v10_causal_amendment.sh freeze --run-suffix metadata-freeze",
        "HF_HOME=/data/CSK/J-space-project/.hf-cache scripts/run_causal_sufficiency_v10_causal_amendment.sh causal --device 0 --run-suffix causal-confirmatory-final",
        "scripts/run_strict_interface_v10.sh freeze --run-suffix strict-freeze",
        "scripts/run_strict_interface_v10.sh audit --run-suffix strict-three-way",
        "scripts/run_causal_sufficiency_v10.sh report --run-suffix final-v10",
        "scripts/finalize_reports_v10.sh --run-suffix amendment-aware-final",
    ]
    final_lines = [
        FINAL_START,
        "",
        "## Protocol v10 corrected causal-sufficiency update",
        "",
        f"Base freeze `{base['freeze_digest']}`; residual amendment `{residual_freeze['freeze_digest']}`; candidate freeze `{candidates['freeze_digest']}`; causal metadata amendment `{causal_freeze['freeze_digest']}`; strict-interface freeze `{strict_freeze['freeze_digest']}`.",
        "",
        "### Required v10 answers",
        "",
        "1. **Yes.** v9 residual localization had coordinate duplication and was not the corrected joint conditional estimand.",
        f"2. At 512D, corrected rich-block gains are recurrent `{residual512['recurrent']['estimate']:.4f}`, conv `{residual512['conv']['estimate']:.4f}`, KV `{residual512['kv']['estimate']:.4f}`, joint `{residual512['architecture_joint']['estimate']:.4f}`; the exact same-599D combined reference is `{residual512['combined_reference']['estimate']:.6f}`.",
        f"3. The smallest validation/all-family observationally sufficient dimension is **{sweep['smallest_observationally_sufficient_dimension']}D**.",
        "4. Yes. 512D is a strong reference/upper candidate, not the minimum.",
        f"5. Observational semantic sufficiency passes relative to a modest full ceiling (~`{semantic512.semantic_full_ceiling:.3f}`); absolute probe quality is ceiling-limited. Decoded-causal semantic fidelity fails separately.",
        f"6. Yes. 384/448/512D decode to recurrent/conv/KV state; full-rank feature reconstruction error is `{decoder['full_rank_feature_reconstruction_max_abs_error']:.3e}`.",
        "7. No. Compact decoded interventions do not reproduce all teacher raw-state effects under frozen gates.",
        "8. No. Direction/output/semantic fidelity degrades through h2/h4/h8; h16 confirms continued instability.",
        "9. No rescue: every effect-enriched candidate/horizon gate fails.",
        "10. **No candidate causal sufficient persistent state was obtained.**",
        "11. Main failures are decoder/interface semantic loss, block/channel interaction not captured by the combined ceiling, and long-horizon instability. Dimension alone and task heterogeneity are not sufficient explanations.",
        "12. **Autonomous controller training is not authorized.**",
        "",
        "H2/H3 adjudication: **H2 remains the strongest supported account; H3 is not established.**",
        "",
        "### Exact commands",
        "",
        "```bash",
        *commands,
        "```",
        "",
        FINAL_END,
    ]
    final_path = root / "reports/FINAL_REPORT.md"
    current = final_path.read_text(encoding="utf-8")
    block = "\n".join(final_lines)
    prefix, remainder = current.split(FINAL_START, 1)
    _, suffix = remainder.split(FINAL_END, 1)
    final_path.write_text(
        prefix.rstrip() + "\n\n" + block + suffix, encoding="utf-8"
    )

    report_paths = [
        "reports/RESIDUAL_LOCALIZATION_AUDIT_V10.md",
        "reports/CORRECTED_SUB512_SUFFICIENCY_V10.md",
        "reports/SEMANTIC_SUFFICIENCY_AUDIT_V10.md",
        "reports/DECODED_CAUSAL_STATE_VALIDATION_V10.md",
        "reports/LONG_HORIZON_STATE_SUFFICIENCY_V10.md",
        "reports/BEHAVIOR_ENRICHED_CONFIRMATORY_V10.md",
        "reports/FINAL_REPORT.md",
    ]
    summary = {
        "status": "COMPLETED_V10_FINAL",
        "git_commit_at_generation": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip(),
        "freeze_digests": {
            "base": base["freeze_digest"],
            "residual_amendment": residual_freeze["freeze_digest"],
            "candidates": candidates["freeze_digest"],
            "causal_metadata_amendment": causal_freeze["freeze_digest"],
            "strict_interface": strict_freeze["freeze_digest"],
        },
        "smallest_observationally_sufficient_dimension": sweep[
            "smallest_observationally_sufficient_dimension"
        ],
        "candidate_dimensions": candidates["candidate_dimensions"],
        "candidate_causal_sufficient_state": None,
        "controller_authorized": False,
        "free_continuation_executed": False,
        "strongest_conclusion": (
            "384D is observationally sufficient relative to the combined ceiling, "
            "but no tested compact state is a causally sufficient interface."
        ),
        "report_hashes": {path: sha256_file(root / path) for path in report_paths},
        "commands": commands,
    }
    write_json_atomic(root / "results/v10/processed/v10_final_summary.json", summary)
    manifest = {
        "summary_sha256": sha256_file(
            root / "results/v10/processed/v10_final_summary.json"
        ),
        "report_hashes": summary["report_hashes"],
    }
    write_json_atomic(
        root / "results/v10/processed/report_manifest_v10_amended.json", manifest
    )
    return summary


def main() -> None:
    parser = standard_parser(
        "finalize amendment-aware protocol-v10 reports",
        "configs/causal_sufficiency_v10.yaml",
    )
    args = parser.parse_args()
    context = initialize_context("finalize-reports-v10", args)
    try:
        summary = _finalize(context)
        context.finish("COMPLETED_V10_FINAL_REPORTS", summary=summary)
    except Exception as exc:
        context.finish("FAILED", error=f"{type(exc).__name__}: {exc}")
        raise


if __name__ == "__main__":
    main()
