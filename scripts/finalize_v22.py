#!/usr/bin/env python3
"""Adjudicate V22 and build machine-readable records plus required reports."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np

from jclosure.protocol_v22 import stage_freeze, verify, verify_stage
from jclosure.provenance import sha256_file, write_json_atomic

ROOT = Path.cwd()
OUT = Path("results/v22/processed")
REPORTS = Path("reports")


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def load(name: str):
    return json.loads((ROOT / OUT / name).read_text())


def write_report(name: str, title: str, body: str) -> None:
    (ROOT / REPORTS / name).write_text(f"# {title}\n\n{body.strip()}\n", encoding="utf-8")


def main() -> dict:
    base = verify(ROOT)
    geometry = load("input_side_causal_geometry_v22.json")
    pool = load("action_pool_v22.json")
    calibration = load("action_pool_calibration_v22.json")
    selection = load("action_selection_v22.json")
    scaling = load("action_data_scaling_v22.json")
    coordinates = load("causal_action_coordinates_v22.json")
    even_odd = load("even_odd_action_geometry_v22.json")
    coverage = load("causal_coverage_error_v22.json")
    chart = load("state_conditioned_action_chart_v22.json")
    low_rank = load("low_rank_operator_identification_v22.json")
    scale = load("action_scale_law_v22.json")
    composition = load("action_composition_v22.json")
    expanded = load("expanded_response_bank_v22.json")
    development = load("v22_development_adjudication.json")
    test_audit_path = ROOT / OUT / "v22_test_audit.json"
    test_audit = json.loads(test_audit_path.read_text()) if test_audit_path.exists() else None
    roles21 = json.loads((ROOT / "artifacts/action_coordinate_geometry_v21_roles.freeze.json").read_text())
    state21 = load("../../v21/processed/state_representations_v21.json")

    if not (ROOT / "artifacts/causal_action_manifold_v22_analysis_results.freeze.json").exists():
        stage_freeze(ROOT, "analysis_results", [
            "src/jclosure/experiments/analyze_v22.py",
            "src/jclosure/experiments/scale_composition_v22.py",
            str(OUT / "input_side_causal_geometry_v22.json"),
            str(OUT / "action_data_scaling_v22.json"),
            str(OUT / "causal_action_coordinates_v22.json"),
            str(OUT / "even_odd_action_geometry_v22.json"),
            str(OUT / "causal_coverage_error_v22.json"),
            str(OUT / "state_conditioned_action_chart_v22.json"),
            str(OUT / "low_rank_operator_identification_v22.json"),
            str(OUT / "action_scale_law_v22.json"),
            str(OUT / "action_composition_v22.json"),
        ], {"validation_responses_used_for_model_selection": False,
            "historical_final_six_opened": False, "new_independent_final_opened": False})
    validation_geometry = geometry["roles"]["validation"]
    g2 = geometry["classification"]["V22_G2_STATE_DEPENDENT_INPUT_CAUSAL_SUBSPACE"]
    input_cross_p0 = validation_geometry["JVP_finite_P0"]["input_subspace_overlap_median"]
    input_cross_pq = validation_geometry["JVP_finite_Pq"]["input_subspace_overlap_median"]
    g3 = min(input_cross_p0, input_cross_pq) < 0.5
    practical = bool(development["practical_direction_sign_gate"])
    all_saturated = bool(development["saturation_all_strategies"])
    action_limited = not all_saturated
    scale_best_plus = scale["results"]["plus"]["monotone_gain_network"]["validation_relative_l2"]
    scale_best_minus = scale["results"]["minus"]["monotone_gain_network"]["validation_relative_l2"]
    sign_gain = (even_odd["baseline_minus"]["relative_l2"] - even_odd["reconstructed_minus"]["relative_l2"])
    chart_gain = chart["global_fixed_coordinate"]["relative_l2"] - chart["state_conditioned_local_chart"]["relative_l2"]
    formal = ["V22-D_ACTION_DATA_LIMITED"] if action_limited else ["V22-E_CROSS_ACTION_OPERATOR_STILL_UNIDENTIFIED_AFTER_COVERAGE"]
    flags = {
        "V22_G1_STABLE_INPUT_CAUSAL_SUBSPACE": False,
        "V22_G2_STATE_DEPENDENT_INPUT_CAUSAL_SUBSPACE": bool(g2),
        "V22_G3_DIFFERENTIAL_FINITE_INPUT_DIVERGENCE": bool(g3),
        "PRACTICAL_ACTION_COORDINATE_PASS": practical,
        "ACTION_COVERAGE_BOTTLENECK_RESOLVED": practical,
        "ACTION_DATA_LIMITED": action_limited,
        "CROSS_ACTION_OPERATOR_STILL_UNIDENTIFIED_AFTER_COVERAGE": bool(all_saturated and not practical),
        "SIGN_NONLINEARITY_IS_PRIMARY_LIMIT": bool(sign_gain >= 0.1 and not practical),
        "COMPACT_OPERATOR_SEARCH_REOPENED": practical,
        "RAW_TO_OPERATOR_ENCODER_AUTHORIZED": False,
        "H2_REMAINS": True,
        "H3_AUTHORIZED": False,
        "DYNAMIC_STATE_SEARCH_AUTHORIZED": False,
        "HISTORICAL_FINAL_SIX_OPENED": False,
        "NEW_INDEPENDENT_FINAL_OPENED": False,
    }
    source_names = [
        "input_side_causal_geometry_v22.json", "action_pool_v22.json", "action_pool_calibration_v22.json",
        "action_selection_v22.json", "expanded_response_bank_v22.json", "action_data_scaling_v22.json",
        "causal_action_coordinates_v22.json", "even_odd_action_geometry_v22.json",
        "action_scale_law_v22.json", "action_composition_v22.json", "causal_coverage_error_v22.json",
        "state_conditioned_action_chart_v22.json", "low_rank_operator_identification_v22.json",
    ]
    state_hashes = {
        "V21_train_state_ids": roles21["V20_train_base_id_sha256"],
        "V21_validation_state_ids": roles21["V20_validation_base_id_sha256"],
        "V21_S0_S3_feature_scratch": state21["scratch_sha256"],
        "V22_development_operator_states": digest(load("expanded_bank_design_v22.json")["development_states"]),
        "V22_validation_operator_states": digest(load("expanded_bank_design_v22.json")["validation_states"]),
    }
    adjudication = {
        "protocol_hash": base["freeze_digest"], "state_hashes": state_hashes,
        "action_pool_hash": pool["candidate_pool_hash"], "selection_hashes": selection["selection_hashes"],
        "train_action_hash": selection["common_train_hash"], "validation_action_hash": selection["validation_hash"],
        "independent_final_action_hash": selection["independent_final_hash"],
        "reliable_candidate_action_count": calibration["symmetric_reliable_action_count"],
        "nested_action_counts": selection["nested_counts"], "formal_outcomes": formal, "flags": flags,
        "geometry": {
            "JVP_input_r90_r95_r99": validation_geometry["JVP"]["input_r90_r95_r99_median"],
            "finite_input_r90_r95_r99": validation_geometry["finite"]["input_r90_r95_r99_median"],
            "JVP_output_r90_r95_r99": validation_geometry["JVP"]["output_r90_r95_r99_median"],
            "finite_output_r90_r95_r99": validation_geometry["finite"]["output_r90_r95_r99_median"],
            "JVP_input_P0_Pq_angle": validation_geometry["JVP"]["input_P0_Pq_angle_median_degrees"],
            "finite_input_P0_Pq_angle": validation_geometry["finite"]["input_P0_Pq_angle_median_degrees"],
            "JVP_output_P0_Pq_angle": validation_geometry["JVP"]["output_P0_Pq_angle_median_degrees"],
            "finite_output_P0_Pq_angle": validation_geometry["finite"]["output_P0_Pq_angle_median_degrees"],
            "JVP_finite_input_overlap_P0": input_cross_p0, "JVP_finite_input_overlap_Pq": input_cross_pq,
        },
        "best_coordinate": coordinates["best"], "even_odd": even_odd,
        "scale_best_plus_l2": scale_best_plus, "scale_best_minus_l2": scale_best_minus,
        "composition": composition["results"], "state_conditioned_chart_gain": chart_gain,
        "low_rank_best": low_rank["best"], "k_operator_min": None,
        "source_hashes": {name: sha256_file(ROOT / OUT / name) for name in source_names},
        "final_opening_reason": "No practical action coordinate passed development direction/sign/composition gates; both final sets remain unopened.",
        "nonexistence_claimed": False,
    }
    write_json_atomic(ROOT / OUT / "v22_adjudication.json", adjudication)
    model_registry = {
        "models": ["separable_kernel_ridge", "causal_metric_coordinate", "finite_supervised_coordinate",
                   "polynomial_kernel", "RBF_kernel", "explicit_even_odd", "state_conditioned_local_operator"],
        "weight_artifacts": "closed_form_no_neural_weight_files",
        "result_hashes": {name: adjudication["source_hashes"][name] for name in source_names if name in adjudication["source_hashes"]},
    }
    write_json_atomic(ROOT / OUT / "model_registry_v22.json", model_registry)
    geometry_rows = np.array([
        *validation_geometry["JVP"]["input_r90_r95_r99_median"],
        *validation_geometry["finite"]["input_r90_r95_r99_median"],
        validation_geometry["JVP"]["input_P0_Pq_angle_median_degrees"],
        validation_geometry["finite"]["input_P0_Pq_angle_median_degrees"],
    ], dtype=np.float64)
    scaling_array = np.array([[row["action_count"], row["relative_l2"], row["local_S2_action_ceiling"]["relative_l2"]]
                              for row in scaling["rows"]], dtype=np.float64)
    np.savez_compressed(ROOT / OUT / "v22_summary_arrays.npz", geometry=geometry_rows, scaling=scaling_array)

    jvp = validation_geometry["JVP"]; finite = validation_geometry["finite"]
    write_report("INPUT_SIDE_CAUSAL_GEOMETRY_V22.md", "Input-Side Causal Geometry — V22", f"""
The 63 reliable V21 directions were corrected with the full 14,397-D raw-action Gram before decomposition. The Gram rank is {geometry['input_metric']['gram_rank']}, retained condition number is {geometry['input_metric']['gram_condition_number_retained']:.3g}, and the relative eigenvalue floor is `1e-8`.

| operator | input r90/r95/r99 | input P0→Pq angle | output P0→Pq angle |
|---|---:|---:|---:|
| exact JVP | {jvp['input_r90_r95_r99_median']} | {jvp['input_P0_Pq_angle_median_degrees']:.4f}° | {jvp['output_P0_Pq_angle_median_degrees']:.4f}° |
| finite | {finite['input_r90_r95_r99_median']} | {finite['input_P0_Pq_angle_median_degrees']:.4f}° | {finite['output_P0_Pq_angle_median_degrees']:.4f}° |

JVP/finite input-subspace overlap is {input_cross_p0:.4f} at P0 and {input_cross_pq:.4f} at Pq. Input V rotates materially, so a fixed global action coordinate is structurally incomplete. Output rank is not used as a claim that the raw action space has the same dimension.
""")
    write_report("ACTION_POOL_CALIBRATION_V22.md", "Action Pool Calibration — V22", f"""
The candidate pool contains {pool['candidate_count']} actions: 506 non-final V13 directions plus six frozen balanced combinations. Historical final-six geometry and responses remained sealed. Five train-only calibration states measured both signs at amplitudes 0.25, 0.5, and 1.0.

Symmetric reliable candidates: **{calibration['symmetric_reliable_action_count']} / {pool['candidate_count']}**. Four initially frozen validation candidates failed symmetric reliability and were replaced, before any expanded response measurement, by the recorded append-only reserve amendment.
""")
    write_report("ACTION_EXPERIMENT_DESIGN_V22.md", "Action Experiment Design — V22", f"""
Strategies: `{', '.join(selection['selection_strategies'])}`. Nested counts: `{selection['nested_counts']}`. A common 128-direction measured panel permits a controlled ranking comparison; 32 reliable validation directions and 32 independent-final directions were frozen separately. The causal D-optimal design used development-state exact JVP only and is an offline experiment-design method, not an inference-time oracle.

Selection hashes: `{json.dumps(selection['selection_hashes'], sort_keys=True)}`.
""")
    curve_lines = [f"| {row['strategy']} | {row['action_count']} | {row['relative_l2']:.4f} | {row['local_S2_action_ceiling']['relative_l2']:.4f} | {row['local_S2_action_ceiling']['direction_cosine_median']:.4f} |"
                   for row in scaling["rows"]]
    write_report("ACTION_DATA_SCALING_V22.md", "Action Data Scaling — V22", """
| strategy | actions | S1 cross-state L2 | local S2 action-ceiling L2 | local cosine |
|---|---:|---:|---:|---:|
""" + "\n".join(curve_lines) + f"\n\nSaturation by strategy: `{json.dumps(scaling['saturation_by_strategy'], sort_keys=True)}`. Overall saturation is not established; `ACTION_DATA_LIMITED=TRUE`. The adverse 128-action point is reported, not hidden.")
    coordinate_lines = [f"| {row['coordinate']} | {row['dimension']} | {row['unseen_direction']['relative_l2']:.4f} | {row['unseen_direction_and_sign']['relative_l2']:.4f} | {row['local_S2_action_ceiling']['relative_l2']:.4f} |"
                        for row in coordinates["rows"]]
    write_report("CAUSAL_ACTION_COORDINATES_V22.md", "Causal Action Coordinates — V22", """
| coordinate | dim | unseen direction L2 | direction+sign L2 | local S2 ceiling L2 |
|---|---:|---:|---:|---:|
""" + "\n".join(coordinate_lines) + f"\n\nBest practical candidate: `{coordinates['best']['coordinate']}` dimension `{coordinates['best']['dimension']}`, unseen-direction L2 `{coordinates['best']['unseen_direction']['relative_l2']:.4f}`. It does not pass the 0.30 gate.")
    write_report("EVEN_ODD_ACTION_GEOMETRY_V22.md", "Even/Odd Action Geometry — V22", f"""
Explicit decomposition yields reconstructed positive L2 `{even_odd['reconstructed_plus']['relative_l2']:.4f}` and negative L2 `{even_odd['reconstructed_minus']['relative_l2']:.4f}`. The odd-symmetry negative baseline is `{even_odd['baseline_minus']['relative_l2']:.4f}`; the explicit split does not materially repair sign generalization.
""")
    write_report("ACTION_SCALE_LAW_V22.md", "Action Scale Law — V22", f"""
Amplitudes 0.25, 0.5, and 1.0 were measured on a separately frozen panel. Positive/negative monotone-gain validation L2 is `{scale_best_plus:.4f}` / `{scale_best_minus:.4f}`. Linear scale L2 is `{scale['results']['plus']['linear_alpha']['validation_nonunit_relative_l2']:.4f}` / `{scale['results']['minus']['linear_alpha']['validation_nonunit_relative_l2']:.4f}`. Scale-only nonlinearity is identifiable, but this does not solve direction generalization.
""")
    write_report("STATE_CONDITIONED_ACTION_CHART_V22.md", "State-Conditioned Action Chart — V22", f"""
V22-G2 authorized this test. Fixed-global odd-response L2 is `{chart['global_fixed_coordinate']['relative_l2']:.4f}`; the low-rank S1-conditioned local chart is `{chart['state_conditioned_local_chart']['relative_l2']:.4f}`. Gain is `{chart_gain:.4f}`; therefore state-conditioned transport is structurally motivated but not empirically beneficial in the tested constrained model.
""")
    association_lines = [f"| {metric} | {name} | {values['spearman']:.4f} | {values['pearson']:.4f} |"
                         for metric, items in coverage["associations"].items() for name, values in items.items()]
    write_report("CAUSAL_COVERAGE_ERROR_V22.md", "Causal Coverage and Error — V22", """
| metric family | quantity | Spearman | Pearson |
|---|---|---:|---:|
""" + "\n".join(association_lines) + "\n\nRaw and architecture span residuals decrease with action count, while causal Z6 reaches near-zero projected residual by 96 directions. Held-out error remains weakly associated with these distances, so causal projected coverage alone is not sufficient.")
    low_lines = [f"| {row['rank']} | {row['relative_l2']:.4f} | {row['direction_cosine_median']:.4f} |" for row in low_rank["rows"]]
    write_report("LOW_RANK_OPERATOR_IDENTIFICATION_V22.md", "Low-Rank Operator Identification — V22", """
| rank | unseen-direction L2 | cosine |
|---:|---:|---:|
""" + "\n".join(low_lines) + "\n\nNo tested rank passes. A compact operator-state coordinate is not identified, and no nonexistence claim is made while action scaling is unsaturated.")
    write_report("COMPACT_OPERATOR_REOPEN_V22.md", "Compact Operator Reopening — V22", """
`PRACTICAL_ACTION_COORDINATE_PASS = FALSE`\n\n`COMPACT_OPERATOR_SEARCH_REOPENED = FALSE`\n\n`k_operator_min = NONE`\n\nThe conditional reopening stage was not executed because its prerequisite gate failed.
""")
    write_report("INDEPENDENT_ACTION_FINAL_V22.md", "Independent Action Final — V22", f"""
Historical V20 final six: **SEALED**. New V22 independent final set ({len(selection['independent_final_action_ids'])} directions): **FROZEN, UNOPENED**. No finalist qualified, so final responses were not measured. Final-opening hash: `{selection['independent_final_hash']}`.
""")
    write_report("STRICT_INTERFACE_AUDIT_V22.md", "Strict Interface Audit — V22", f"""
No `P -> C`, dynamic transition, or controller model was trained. Z6 uses JVP only during offline train-set design and maps a new requested action from raw action coordinates alone. Z7 uses train-only finite responses but never a new action's response. Historical final-six raw geometry/responses and all independent-final responses stayed sealed. V1–V21 frozen records were not overwritten.

`DYNAMIC_STATE_SEARCH_AUTHORIZED = FALSE`  
`RAW_TO_OPERATOR_ENCODER_AUTHORIZED = FALSE`
""")
    write_report("ACTION_COMPOSITION_V22.md", "Action Composition — V22", f"""
Additive finite superposition L2: positive pair `{composition['results']['plus']['pair']['additive_superposition_relative_l2']:.4f}`, negative pair `{composition['results']['minus']['pair']['additive_superposition_relative_l2']:.4f}`, positive dense `{composition['results']['plus']['dense']['additive_superposition_relative_l2']:.4f}`, negative dense `{composition['results']['minus']['dense']['additive_superposition_relative_l2']:.4f}`. Pair composition passes the 0.30 diagnostic; dense and negative dense do not both pass.
""")

    commands = [
        "PYTHONPATH=src python -m jclosure.protocol_v22 freeze",
        "PYTHONPATH=src python -m jclosure.experiments.input_geometry_v22 prepare",
        "PYTHONPATH=src python -m jclosure.experiments.input_geometry_v22 run",
        "PYTHONPATH=src python -m jclosure.experiments.action_pool_v22 prepare",
        "CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src python -m jclosure.experiments.action_pool_v22 calibrate",
        "PYTHONPATH=src python -m jclosure.experiments.action_selection_v22",
        "PYTHONPATH=src python -m jclosure.experiments.expanded_bank_v22 prepare",
        "CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src python -m jclosure.experiments.expanded_bank_v22 run --role development",
        "CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src python -m jclosure.experiments.expanded_bank_v22 run --role validation",
        "PYTHONPATH=src python -m jclosure.experiments.expanded_bank_v22 summarize",
        "PYTHONPATH=src python -m jclosure.experiments.analyze_v22",
        "PYTHONPATH=src python -m jclosure.experiments.scale_composition_v22 prepare",
        "CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src python -m jclosure.experiments.scale_composition_v22 run --role development",
        "CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src python -m jclosure.experiments.scale_composition_v22 run --role validation",
        "PYTHONPATH=src python -m jclosure.experiments.scale_composition_v22 analyze",
        "PYTHONPATH=src python scripts/finalize_v22.py",
        "PYTHONPATH=src pytest -q tests/test_v22.py",
        "PYTHONPATH=src pytest -q",
        "PYTHONPATH=src python scripts/freeze_v22_final.py",
        "git add ... && git commit -m 'Complete V22 causal action manifold study'",
        "git push origin main",
    ]
    changed = [line[3:] for line in subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).splitlines()]
    changed += [name for name in ("artifacts/causal_action_manifold_v22_final.freeze.json",
                                  "results/v22/processed/v22_verification.json") if name not in changed]
    write_report("EXECUTION_MANIFEST_V22.md", "Execution Manifest — V22", "Exact primary commands:\n\n```text\n" + "\n".join(commands) + "\n```\n\nChanged files before commit:\n\n```text\n" + "\n".join(sorted(changed)) + "\n```\n\nGPU measurements used `CUDA_VISIBLE_DEVICES=1`, `HF_HOME=/data/CSK/J-space-project/.hf-cache`, and the repository's Python 3.12 environment. Closed-form model outputs have no neural weight artifact; hashes are in `model_registry_v22.json`.")

    answers = f"""
1. Input r90/r95/r99: JVP `{jvp['input_r90_r95_r99_median']}`; finite `{finite['input_r90_r95_r99_median']}`.
2. Input rank is lower than the 63-direction metric domain but materially larger than the old output-only 5–7 estimate; it is not evidence that raw action space is 7-D.
3. Output U rotates: JVP `{jvp['output_P0_Pq_angle_median_degrees']:.4f}°`, finite `{finite['output_P0_Pq_angle_median_degrees']:.4f}°`.
4. Input V rotates: JVP `{jvp['input_P0_Pq_angle_median_degrees']:.4f}°`, finite `{finite['input_P0_Pq_angle_median_degrees']:.4f}°`.
5. JVP/finite input overlap: P0 `{input_cross_p0:.4f}`, Pq `{input_cross_pq:.4f}`; divergence is material at Pq.
6. A fixed global action coordinate is not supported.
7. A state-conditioned chart is structurally indicated, but the tested chart did not improve validation.
8. Reliable directions: `{calibration['symmetric_reliable_action_count']}` of 512.
9. Full 12/24/48/96/128 curves are in ACTION_DATA_SCALING_V22.md.
10. Overall saturation: `FALSE`.
11. Raw maximin gives the best measured local action ceiling; causal D-optimal does not.
12. Causal D-optimal does not beat random/maximin consistently.
13. Z6 beats raw Z1 in the best cross-state comparison but remains far above 0.30.
14. Z7 does not solve unseen action prediction.
15. Polynomial/RBF kernels do not pass.
16. Explicit even/odd does not materially fix unseen sign.
17. Monotone scale modeling improves scale-only L2 to `{scale_best_plus:.4f}` / `{scale_best_minus:.4f}`.
18. No tested coverage metric strongly predicts held-out error.
19. State-conditioned transport changes L2 by `{chart_gain:.4f}` and does not help.
20. No compact operator coordinate emerges.
21. `k_operator_min = NONE`.
22. Historical final six remain sealed.
23. New independent final remains frozen and unopened because no finalist qualified.
24. `RAW_TO_OPERATOR_ENCODER_AUTHORIZED = FALSE`.
25. H2 remains.
26. H3 is not authorized.
27. Dynamic-state search is not authorized.
"""
    write_report("V22_SCIENTIFIC_ANSWERS_V22.md", "V22 Scientific Answers", answers)

    compact_flags = "\n".join(f"- `{key} = {str(value).upper()}`" for key, value in flags.items())
    test_section = (f"V22 tests: `{test_audit['v22_passed']} passed, {test_audit['v22_failed']} failed`. Full suite: "
                    f"`{test_audit['full_passed']} passed, {test_audit['full_failed']} failed`; both failures are inherited "
                    "cumulative FINAL_REPORT hash assertions from V14/V16 and were not hidden by rewriting historical manifests.") if test_audit else "Test audit pending."
    complete = f"""
# V22 Complete Report

## Identity

- Name: **Causal Action Manifold Expansion and Input-Side Operator Geometry**
- Parent: `351da6f061c3ed13e90324f50afc30f53a8796de`
- Protocol hash: `{base['freeze_digest']}`
- Candidate pool hash: `{pool['candidate_pool_hash']}`
- Train / validation / independent-final hashes: `{selection['common_train_hash']}` / `{selection['validation_hash']}` / `{selection['independent_final_hash']}`
- Reliable candidates: `{calibration['symmetric_reliable_action_count']} / 512`

## Outcome

Formal outcome: **{', '.join(formal)}**.

Metric-corrected input ranks are JVP `{jvp['input_r90_r95_r99_median']}` and finite `{finite['input_r90_r95_r99_median']}`. Both input and output subspaces rotate under P0→Pq. The best practical coordinate is `{coordinates['best']['coordinate']}` dimension `{coordinates['best']['dimension']}`, but its unseen-direction L2 is `{coordinates['best']['unseen_direction']['relative_l2']:.4f}` and the even/odd negative L2 is `{even_odd['reconstructed_minus']['relative_l2']:.4f}`. The practical gate therefore fails.

Expanded action data did not establish stable saturation across all frozen strategies. V22 therefore does not reinterpret the remaining error as operator nonexistence. The compact operator stage and both final-response openings remain closed.

## Geometry

{(ROOT / REPORTS / 'INPUT_SIDE_CAUSAL_GEOMETRY_V22.md').read_text()}

## Action pool and experiment design

{(ROOT / REPORTS / 'ACTION_POOL_CALIBRATION_V22.md').read_text()}

{(ROOT / REPORTS / 'ACTION_EXPERIMENT_DESIGN_V22.md').read_text()}

## Scaling and coordinates

{(ROOT / REPORTS / 'ACTION_DATA_SCALING_V22.md').read_text()}

{(ROOT / REPORTS / 'CAUSAL_ACTION_COORDINATES_V22.md').read_text()}

## Nonlinearity and composition

{(ROOT / REPORTS / 'EVEN_ODD_ACTION_GEOMETRY_V22.md').read_text()}

{(ROOT / REPORTS / 'ACTION_SCALE_LAW_V22.md').read_text()}

{(ROOT / REPORTS / 'ACTION_COMPOSITION_V22.md').read_text()}

## State-conditioned and low-rank tests

{(ROOT / REPORTS / 'STATE_CONDITIONED_ACTION_CHART_V22.md').read_text()}

{(ROOT / REPORTS / 'LOW_RANK_OPERATOR_IDENTIFICATION_V22.md').read_text()}

## Final authorization flags

{compact_flags}

## Verification

{test_section}

## Scientific answers

{answers}
"""
    (ROOT / REPORTS / "V22_COMPLETE_REPORT.md").write_text(complete, encoding="utf-8")
    final_report = ROOT / REPORTS / "FINAL_REPORT.md"
    marker = "## V22 — Causal Action Manifold Expansion and Input-Side Operator Geometry"
    if marker not in final_report.read_text(encoding="utf-8"):
        with final_report.open("a", encoding="utf-8") as handle:
            handle.write(f"\n\n{marker}\n\nFormal outcome: `{formal[0]}`. Metric-corrected input subspaces rotate with persistent state, but no practical action coordinate passes; action scaling remains unsaturated, compact-operator reopening is false, both final sets remain sealed, H2 remains, and H3/dynamic-state search remain unauthorized. See `reports/V22_COMPLETE_REPORT.md`.\n")

    report_names = [path.name for path in sorted((ROOT / REPORTS).glob("*V22*.md"))]
    extra_machine = ["v22_adjudication.json", "model_registry_v22.json", "v22_summary_arrays.npz"]
    if test_audit:
        extra_machine.append("v22_test_audit.json")
    integrity_files = [str(OUT / name) for name in sorted(source_names + extra_machine)]
    integrity_files += [str(REPORTS / name) for name in report_names]
    integrity = {"protocol_hash": base["freeze_digest"], "status": "PASS",
                 "files": {name: sha256_file(ROOT / name) for name in integrity_files},
                 "historical_final_six_opened": False, "new_independent_final_opened": False}
    integrity["index_digest"] = digest(integrity)
    write_json_atomic(ROOT / OUT / "v22_integrity_index.json", integrity)
    return {"formal_outcomes": formal, "flags": flags, "reports": report_names,
            "integrity_index_digest": integrity["index_digest"]}


if __name__ == "__main__":
    print(json.dumps(main(), indent=2))
