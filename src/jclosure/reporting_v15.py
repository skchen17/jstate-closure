"""Evidence-gated V15 reports; keep diagnostic, development and final separate."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from jclosure.provenance import sha256_file

ROOT = Path.cwd()
OUT = ROOT / "results/v15/processed"
REPORTS = ROOT / "reports"


def _json(name: str) -> dict[str, Any]:
    return json.loads((OUT / name).read_text(encoding="utf-8"))


def _write(name: str, body: str) -> None:
    (REPORTS / name).write_text(body.rstrip() + "\n", encoding="utf-8")


def _table(rows: list[dict[str, Any]], columns: list[str], digits: int = 3) -> str:
    frame = pd.DataFrame(rows)
    return frame[columns].to_markdown(index=False, floatfmt=f".{digits}f") if len(frame) else "No measured rows."


def _comparison() -> pd.DataFrame:
    old = pd.read_parquet(ROOT / "results/v13/processed/moving_tangent_oracle_development_v13.parquet")
    old = old[(old.method == "static_local_causal_oracle") | ((old.method == "moving_tangent_interpolated_oracle") & (old.alpha == 1.0))]
    v14 = pd.read_parquet(ROOT / "results/v14/processed/closed_loop_fidelity_development_v14.parquet")
    v15 = pd.read_parquet(OUT / "control_fidelity_v15.parquet")
    columns = ["method", "horizon", "direction_cosine", "magnitude_ratio", "output_direction_cosine", "semantic_delta_agreement", "semantic_vector_cosine", "task_decision_sign_agreement"]
    combined = pd.concat([old[columns], v14[columns], v15[columns]], ignore_index=True)
    teacher = pd.DataFrame([
        {"method": "raw_teacher_label_self_comparison", "horizon": horizon,
         "direction_cosine": 1.0, "magnitude_ratio": 1.0,
         "output_direction_cosine": 1.0, "semantic_delta_agreement": 1.0,
         "semantic_vector_cosine": 1.0, "task_decision_sign_agreement": 1.0}
        for horizon in (1, 2, 4, 8)
    ])
    combined = pd.concat([combined, teacher], ignore_index=True)
    return combined.groupby(["method", "horizon"]).agg(direction=("direction_cosine", "mean"), magnitude=("magnitude_ratio", "mean"), output=("output_direction_cosine", "mean"), semantic_legacy=("semantic_delta_agreement", "mean"), semantic_continuous=("semantic_vector_cosine", "mean"), sign=("task_decision_sign_agreement", "mean")).reset_index()


def main() -> None:
    transfer = _json("actuator_transfer_corrected_v15.json")
    shadow = _json("shadow_accumulation_v15.json")
    precision = _json("precision_modes_v15.json")
    pilot = _json("autograd_vs_actuator_v15.json")
    linearity = _json("finite_response_linearity_v15.json")
    rank = _json("writable_rank_analysis_v15.json")
    channels = _json("channel_actuator_geometry_v15.json")
    basis = _json("actuator_basis_v15.json")
    repeat = _json("finite_response_repeat_v15.json")
    control = _json("finite_causal_control_v15.json")
    terminal = _json("control_terminal_v15.json")
    finalist = json.loads((ROOT / "artifacts/quantization_aware_actuation_v15_finalist_decision.freeze.json").read_text())
    splits = json.loads((ROOT / "artifacts/quantization_aware_actuation_v15_splits.freeze.json").read_text())
    config = json.loads((ROOT / "artifacts/quantization_aware_actuation_v15.freeze.json").read_text())
    comparison = _comparison()
    native_curve = pd.DataFrame(transfer["curves"])
    selected_curves = native_curve[native_curve.scale.isin([0.001, 0.01, 0.1, 0.25, 0.5, 1.0, 2.0])]
    shadow_rows = [row for row in shadow["pooled"] if row["increment"] == 0.01 and row["steps"] in (1, 4, 16)]
    _write("ACTUATOR_TRANSFER_FUNCTION_V15.md", f"""# Raw persistent-state actuator transfer — V15

Five frozen V13 train anchors × five predeclared probe directions were audited at 13 absolute scales and six state-relative ULP scales under seven precision/storage modes. The actual V13 `_apply_direction` writeback was read back tensor by tensor for REC, Conv, attention K and V at every touched layer. This is a **diagnostic train** measurement, not a causal control result. `ACTUATOR_DEADZONE_MAX_TESTED_SCALE` is the largest tested absolute scale with median exactly-zero fraction ≥0.5; it is not a physical discontinuity at that precise scale. `MIN_STATE_CHANGE_SCALE` is the first tested scale with any surviving element; `MIN_RELIABLE_STATE_SCALE` additionally requires median gain in [0.8,1.2] and cosine ≥0.95.

| Channel | Dead-zone max tested | First state change | First reliable state scale | Gain at scale 1 | Surviving fraction at scale 1 |
|---|---:|---:|---:|---:|---:|
{chr(10).join(f'| {name} | {item["ACTUATOR_DEADZONE_MAX_TESTED_SCALE"]} | {item["MIN_STATE_CHANGE_SCALE"]} | {item["MIN_RELIABLE_STATE_SCALE"]} | {item["REALIZED_REQUESTED_GAIN_AT_SCALE_1"]:.4f} | {item["SURVIVING_FRACTION_AT_SCALE_1"]:.3f} |' for name, item in transfer["deadzones"].items())}

The per-layer requested/realized norm, cosine, exact-zero fraction, sign flips, relative norm loss, max error, effective BF16 ULP, below-one/half-ULP fraction and surviving fraction are in `{transfer['raw_records']}`. The local ULP sweeps and all seven modes are in the same Parquet; per-channel/layer curves are in `actuator_transfer_corrected_v15.json`.

Absolute-scale native curves (median across measured layer/directions):

{_table(selected_curves.to_dict('records'), ['channel', 'scale', 'median_gain', 'median_cosine', 'median_zero_fraction', 'median_surviving_fraction'])}

At small scales Conv has the largest median zero-writeback region by the declared 50% rule; at scale 1 V survives in the smallest fraction of nonzero requested elements. These are different definitions of “largest quantization loss.” The initial summary `actuator_transfer_v15.json` is preserved: its empty curves were caused by using the pandas `DataFrame.mode` method instead of the `mode` column. The separate frozen correction recomputed only aggregates, not measured rows (digest `{transfer['freeze_digest']}`).

The FP32-shadow experiment accumulates repeated requests in FP32 and performs one BF16 cast immediately before hypothetical model consumption. For a single final write this equals cast-after-add of the total delta. It is compared with repeated BF16 state writes without changing model weights or forward consumption semantics; only raw realized state, not model output, was measured:

{_table(shadow_rows, ['mode', 'channel', 'increment', 'steps', 'median_gain', 'median_cosine', 'median_surviving_fraction'])}
""")
    modes = pd.DataFrame(precision["pooled"])
    modes = modes[(modes.epsilon == 1.0) & (modes.target.isin(["j", "logits", "workspace"]))]
    _write("AUTOGRAD_VS_ACTUATOR_OPERATOR_V15.md", f"""# Ideal autograd JVP versus actual finite actuator — V15

`Jv` is the frozen V13 exact-autograd differential column. `Rεv=[Y(W(P,+εv))-Y(W(P,-εv))]/(2ε)` uses real persistent-state writeback and the frozen J, selected-logit, semantic-continuous and workspace readouts. Target-block normalization uses only V13 **train** JVP columns: `{pilot['target_scales']}`. The fifth target is their dimension-normalized stack. No V13/V14 result was rewritten.

The first epsilon meeting the frozen selection rule was found for only **{pilot['selected_direction_channel_count']}/{pilot['requested_direction_channel_count']}** train state×direction×channel combinations; unselected trials remain in the Parquet rather than being dropped from the denominator. Selection requires both signed J effects ≥ `{config['config']['diagnostic']['min_causal_effect_norm']:.8f}`, ≥10× measured repeat noise, nonzero state writeback, state gain 0.2–5, and signed-effect norm ratio ≤5. This last check is only a coarse saturation screen, not a proof of local linearity.

Selected comparisons:

{_table(pilot['medians_by_target_channel'], ['channel', 'target', 'count', 'cosine', 'relative_l2', 'norm_ratio', 'realized_state_cosine'])}

Precision decomposition: construction/addition/storage/consumption/readout are explicit fields in `precision_modes_v15.parquet`. `native_fp32_add_bf16_writeback` and `cast_after_add` are the same canonical interface and produce identical outputs; native BF16 add quantizes the increment before addition. FP32 shadow REC/Conv and channel-specific FP32 modes alter diagnostic forward semantics, so they cannot replace the canonical result. Full FP32 KV model consumption is `{precision['fp32_kv_consumption']}`; no pretrained weight or query dtype was changed.

At ε=1 (selected diagnostic probes), precision comparisons:

{_table(modes.to_dict('records'), ['mode', 'target', 'status', 'count', 'median_cosine', 'median_relative_l2', 'median_realized_state_cosine'])}

At ε=1, canonical joint requested/realized state cosine is near 1 while logits/workspace output mismatch persists. This supports `AUTOGRAD_DIFFERENTIAL_OPERATOR_DOES_NOT_MATCH_FINITE_ACTUATOR_OPERATOR_IN_TESTED_REGIME`; simple BF16 rounding alone does not explain every output mismatch. The direction/target-specific adaptive rows and full signed projections, norm ratios, state realization and ε are in `autograd_vs_actuator_v15.parquet`. Repeatability max relative L2: `{repeat['max_repeat_relative_l2']}`.
""")
    _write("FINITE_RESPONSE_LINEARITY_V15.md", f"""# Finite-response linearity — V15

The frozen gate requires cosine ≥0.9, relative L2 ≤0.3, and norm ratio 0.7–1.3 for every SNR-qualified tested pair. Tests use one-sided effects relative to the clean state at ε=1: odd symmetry, additivity, homogeneity (factor 2), REC+Conv, and REC+Conv+KV. This is not the trivial algebraic oddness of a central difference definition.

{_table(linearity['by_test'], ['test', 'qualified', 'pass_count', 'median_cosine', 'median_relative_l2', 'median_norm_ratio'])}

`FINITE_RESPONSE_LINEARITY_GATE = {linearity['FINITE_RESPONSE_LINEARITY_GATE']}`. The interface is repeatable in the limited retest, but not sufficiently linear under this operating rule. No V15 differential Hessian/secant geometry was promoted from V14. Machine rows: `finite_response_linearity_v15.parquet` and `finite_response_repeat_v15.parquet`.
""")
    full = pd.DataFrame(rank["full_probe_spectra"])
    full = full[["base_trial_id", "operator", "direction_count", "rank_90", "rank_95", "rank_99", "stable_rank", "effective_rank"]]
    _write("WRITABLE_CAUSAL_RANK_V15.md", f"""# Restricted writable finite-response rank — V15

Actual finite-writeback response columns were measured at ε=1 in the **same V13 probe-coordinate domain** as frozen ideal JVP columns. Probe counts 64/128/256/512 were measured at one train anchor; 64 columns were measured at each of four additional train anchors. Each block is train-normalized before the stacked spectrum. This is not a full raw model-state intrinsic dimension.

Stacked-normalized spectra (joint channels):

{full.to_markdown(index=False, floatfmt='.2f')}

The signal-qualified subset uses both signed J effects above the frozen `{config['config']['diagnostic']['min_causal_effect_norm']:.8f}` threshold. Counts and restricted spectra:

{_table([{'base_trial_id': row['base_trial_id'], 'qualified_direction_count': row['qualified_direction_count'], 'rank_identifiable': row['rank_identifiable'], 'r95_R': row.get('finite', {}).get('rank_95'), 'r95_J': row.get('autograd', {}).get('rank_95')} for row in rank['restricted_snr_qualified_rank']], ['base_trial_id', 'qualified_direction_count', 'rank_identifiable', 'r95_R', 'r95_J'])}

At the 512-probe anchor, stacked `r95_R=14` versus `r95_J=7`; the five 64-probe anchors give `r95_R=8–10` versus `r95_J=5–6`. This supports the **restricted finite-response matrix** pattern in V15-D (`FINITE_WRITABLE_CAUSAL_OPERATOR_HIGHER_RANK_THAN_IDEAL_DIFFERENTIAL_OPERATOR`) but not a linear writable operator theorem. Full numerical spectra can include weak columns and quantization/readout noise; qualified-subset spectra have selection bias. A numeric `r95_R` alone does **not** establish `LOW_DIMENSIONAL_WRITABLE_CAUSAL_ACTUATOR_SUPPORTED` while the finite-response linearity gate fails. Complete singular vectors are reproducible from the response-vector list columns in `finite_operator_columns_v15.parquet`; the spectrum JSON preserves singular values.
""")
    geometry_rows = channels["channel_spectra"]
    _write("CHANNEL_ACTUATOR_GEOMETRY_V15.md", f"""# Channel-resolved actuator geometry — V15

`R_REC`, `R_CONV`, `R_KV` and `R_joint` were measured at the same ε, state and probe directions; K/V also have separate raw writeback transfer curves. Target-block and stacked spectra are in `finite_operator_spectra_v15.json`.

Stacked-normalized rank at 64 probes across five train anchors:

{_table(geometry_rows, ['base_trial_id', 'channel', 'rank_90', 'rank_95', 'rank_99', 'stable_rank'])}

SNR-qualified column fractions: `{channels['snr_qualified_fraction']}`. Median `||R_joint-(R_REC+R_CONV+R_KV)||/||R_joint||` by target: `{channels['interaction_median_by_target']}`. Interaction is finite-response nonadditivity, not a literal independent-channel decomposition. V14's Conv-removal ablation is preserved; V15 distinguishes its raw actuator dead-zone (larger than REC at small scales) from its actual causal response. Per-direction channel norms and interaction residuals: `channel_interaction_v15.parquet`.
The data support measuring joint REC/Conv/KV effects, but do not prove all three channels are intrinsically necessary: V14's restricted ablation found material Conv removal and no material KV removal under its criterion.
""")
    basis_rows = pd.DataFrame(basis["rows"])
    first_anchor = basis_rows[basis_rows.direction_count == basis_rows.direction_count.max()]
    _write("FINITE_CAUSAL_CONTROL_V15.md", f"""# Actuator-calibrated development control — V15

The compact basis is a realized-cost-weighted SVD of **finite response columns**, not PCA or raw autograd singular vectors. The denominator is a diagonal approximation using each column's measured realized-state norm; raw-state cross-Gram and nonlinear combination writeback are not assumed away. The basis coefficients and held-out-probe target-coverage comparison with an autograd-JVP target basis are machine-readable. **The finite basis covers less held-out-probe target energy than the autograd basis in the tested 512-probe anchor**; no basis superiority is claimed.

{_table(first_anchor.to_dict('records'), ['rank', 'r95_weighted', 'finite_basis_heldout_target_coverage', 'autograd_basis_heldout_finite_target_coverage'])}

On the same ten V13 validation cases, the V15 controller used actual ε=1 finite responses, train/frozen output scales, h1 / h1+h2+h4 / h1+h2+h4+h8 objectives, at most four steps, rank `min(8, local r95)`, and realized-response trust decisions. Every step records requested control norm, realized state norm, channel gains, prediction error, actual residual, acceptance and trust ratio. A teacher raw cache was used **only** to generate the teacher response label, never as candidate input or basis. These coordinates are numerical actuator controls, not cognitive or biological state variables.

Median residual curves:

{_table(control['residual_curves'], ['method', 'step', 'raw_ratio', 'weighted_ratio', 'acceptance'])}

Those stepwise medians have different surviving case sets and need not be monotone even when a within-case accepted step improves the weighted objective. The ten-case terminal analysis includes stopped cases:

{_table(terminal['pooled'], ['method', 'case_count', 'terminal_raw_ratio', 'terminal_weighted_ratio', 'raw_monotone_fraction', 'weighted_monotone_fraction', 'terminal_improvement_fraction'])}

Weighted residual is monotone by acceptance construction, but raw teacher residual was not monotone for every case and terminal improvement was not universal.

Same development-panel controller comparison (raw teacher is the target label, so its self-fidelity is trivially 1, not a competing learned controller):

{comparison.to_markdown(index=False, floatfmt='.3f')}

Residual decrease is **not** causal fidelity pass. The frozen finite-response linearity gate failed, so these results are exploratory development only; no independent V15 finalist was eligible. Historical legacy top-k semantic is reported alongside continuous semantic and has not been replaced.
""")
    _write("FINITE_CAUSAL_CONTROLLABILITY_V15.md", f"""# Empirical finite causal reachability — V15 development

The one-step columns are local finite-response basis directions. Multi-step columns concatenate the remeasured response maps after attempted writebacks in a common frozen target coordinate system; no full state transition Jacobian is claimed. Projection floors refer only to the tested rank-limited actuator span and weighted teacher residual, not global reachability.

{_table(control['reachability'], ['objective', 'one_step_rank', 'multi_step_rank', 'one_step_floor', 'multi_step_floor', 'outside_fraction'])}

`TARGET_OUTSIDE_TESTED_ACTUATOR_REACHABLE_SPACE` is recorded per case only when the multi-step projection floor exceeds the predeclared 0.2 threshold. Since the finite linearity gate failed, this is a descriptive development diagnostic, not a formal proof of actuator impossibility. Machine rows: `finite_controllability_v15.parquet`.
""")
    _write("STRICT_INTERFACE_AUDIT_V15.md", f"""# V15 strict interface and authorization audit

- V1–V14 results and protocols are read-only parents. V15 base freeze: `{config['freeze_digest']}`. The original transfer summary was retained; its correction has separate freeze `{transfer['freeze_digest']}`. Cumulative `FINAL_REPORT.md` is the declared mutable exception.
- Diagnostic train / development validation are disjoint: ID SHA256 `{splits['diagnostic_train']['id_sha256']}` / `{splits['development_validation']['id_sha256']}`. V15 development deliberately reuses the V13-validation/V14-development panel for comparison; it is **not independent confirmation**. No V13 final bank was used, and no V15 finalist evaluation occurred.
- Numerical repeat max relative L2: `{repeat['max_repeat_relative_l2']}`. Frozen finite-response linearity pass: `{linearity['FINITE_RESPONSE_LINEARITY_GATE']}`. Development h1 gate: `{finalist['development_h1_gate_pass']}`. Finalist: `{finalist['finalist_status']}` (digest `{finalist['freeze_digest']}`). Independent V15 confirmatory bank: `{finalist['independent_confirmatory_bank']}`.
- h1/h2/h4/h8 development metrics are in the control report; independent h1/h2/h4/h8: **not tested**. Legacy semantic remains separately gated. A continuous-only pass, if any, cannot override the historical full gate.
- H2 remains; `H3_CANDIDATE_INTERFACE_SUPPORTED = FALSE` and H3 complete-state interpretation is unsupported. `ABSOLUTE_REPLACEMENT_NOT_AUTHORIZED`; `AUTONOMOUS_CONTROLLER_AUTHORIZED = FALSE`. V15 control is editing, not absolute persistent-state encoding.
- No new curvature/Hessian claim was made after the finite-response linearity gate failure. Diagnostic FP32 modes changed state storage/consumption semantics and are never counted as canonical model behavior.
- Historical test compatibility: V14's frozen integrity test hashes the entire cumulative `FINAL_REPORT.md`, so the required V15 append makes that one old test fail. Its stored hash and V14 test were **not** changed. V15's parent guard verifies all 1,957 previously tracked paths with only the declared cumulative-report exception; every other historical file matches.

Formal procedural status: **V15-STOP — FINITE_RESPONSE_LINEARITY_GATE_FAILED**. Evidence supports a tested-regime ideal-vs-actual mismatch (V15-B) and the restricted finite-matrix higher-rank pattern (V15-D). Small-scale BF16 quantization is real but is not sufficient to explain every output discrepancy. V15-C/E/F are not established as authorization-level results.
""")
    command_lines = [
        "python -m jclosure.experiments.actuation_v15 --stage freeze",
        "python -m jclosure.experiments.actuation_v15 --stage raw",
        "python scripts/analyze_actuator_transfer_v15.py",
        "python scripts/freeze_v15_splits.py",
        "python -m jclosure.experiments.operator_v15 --stage pilot",
        "python -m jclosure.experiments.precision_v15",
        "python -m jclosure.experiments.operator_v15 --stage linearity",
        "python -m jclosure.experiments.operator_v15 --stage rank",
        "python scripts/analyze_writable_rank_v15.py",
        "python -m jclosure.experiments.basis_v15",
        "python -m jclosure.experiments.repeat_v15",
        "python -m jclosure.experiments.control_v15",
        "python scripts/analyze_control_v15.py",
        "python -m jclosure.experiments.shadow_v15",
        "python scripts/freeze_v15_finalist_decision.py",
        "python scripts/normalize_v15_numeric_json.py",
        "python -m jclosure.reporting_v15",
        "python scripts/build_v15_integrity.py",
        "python scripts/build_complete_version_report.py V15",
        "python -m pytest -q",
        "python -m pytest -q -k 'not test_v14_integrity_manifest'",
    ]
    freeze_paths = sorted((ROOT / "artifacts").glob("quantization_aware_actuation_v15*.freeze.json"))
    freeze_table = "\n".join(f"| `{path.relative_to(ROOT)}` | `{json.loads(path.read_text())['freeze_digest']}` | `{sha256_file(path)}` |" for path in freeze_paths)
    _write("EXECUTION_MANIFEST_V15.md", f"""# V15 execution manifest

Server workspace: `/data/CSK/J-space-project/jstate-closure` on `222.20.126.223`; parent commit `{config['parent_commit']}`. GPU stages used `HF_HOME=/data/CSK/J-space-project/.hf-cache` and `/home/user/anaconda3/bin/python`. Exact commands from the repository root:

```bash
{chr(10).join(command_lines)}
```

The first transfer summary had an aggregation-only bug and is preserved; `scripts/analyze_actuator_transfer_v15.py` produced a separate frozen correction. The independent confirmation command is absent because `{finalist['finalist_status']}`.

Full `pytest -q` has one expected historical test failure: `test_v14_integrity_manifest` compares the old whole-file SHA256 of the cumulative `reports/FINAL_REPORT.md`, which V15 must append. No V14 hash/test/result was modified. The filtered run excludes only this incompatible test and checks the remaining suite plus V15's explicit parent-byte guard.

| Freeze | Digest | File SHA256 |
|---|---|---|
{freeze_table}

Machine record hashes are in each JSON summary and `results/v15/processed/v15_integrity.json`. Changed files can be enumerated exactly with `git diff --name-only {config['parent_commit']}..HEAD` after commit (or `git status --short` before commit). 
""")
    marker_start = "<!-- V15_START -->"
    marker_end = "<!-- V15_END -->"
    final_path = REPORTS / "FINAL_REPORT.md"
    current = final_path.read_text(encoding="utf-8")
    section = f"""\n\n{marker_start}
## V15 — Quantization-Aware Causal Actuation

Formal status: **V15-STOP — FINITE_RESPONSE_LINEARITY_GATE_FAILED**. Real BF16 writeback dead-zones were quantified per REC/Conv/K/V; only {pilot['selected_direction_channel_count']}/{pilot['requested_direction_channel_count']} adaptive state×direction×channel combinations met the predeclared finite-effect selection rule. At accurate realized state perturbations, some output targets still disagree with frozen exact JVP, supporting tested-regime V15-B. The restricted finite-response matrix has higher r95 than ideal JVP (V15-D pattern), but the frozen finite-response linearity gate failed, so spectra and actuator-aware closed-loop results remain diagnostic/development rather than validated finite control. No eligible finalist or new independent confirmatory bank was created. H2 remains; H3 and absolute state replacement are not supported; autonomous-controller training is not authorized. The required cumulative-report append invalidates one old V14 whole-file integrity test; V15's guard confirms all other historical bytes are unchanged. See `reports/V15_COMPLETE_REPORT.md` for all standalone reports, machine records, frozen gates and limitations.
{marker_end}
"""
    if marker_start in current and marker_end in current:
        prefix, remainder = current.split(marker_start, 1)
        _, suffix = remainder.split(marker_end, 1)
        if suffix.strip():
            raise RuntimeError("V15 is not the last FINAL_REPORT section; refusing to rewrite")
        final_path.write_text(prefix.rstrip() + section, encoding="utf-8")
    elif marker_start not in current and marker_end not in current:
        final_path.write_text(current.rstrip() + section, encoding="utf-8")
    else:
        raise RuntimeError("unbalanced V15 FINAL_REPORT markers")
    print("V15 reports and FINAL_REPORT.md written")


if __name__ == "__main__":
    main()
