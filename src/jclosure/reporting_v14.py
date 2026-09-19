"""Evidence-gated V14 reports; never promote development to confirmation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from jclosure.provenance import sha256_file, write_json_atomic

ROOT = Path.cwd()
PROCESSED = ROOT / "results/v14/processed"
REPORTS = ROOT / "reports"


def _json(name: str) -> dict[str, Any]:
    return json.loads((PROCESSED / name).read_text(encoding="utf-8"))


def _write(name: str, value: str) -> None:
    (REPORTS / name).write_text(value.rstrip() + "\n", encoding="utf-8")


def _table(rows: list[dict[str, Any]], columns: list[str] | None = None) -> str:
    frame = pd.DataFrame(rows)
    if columns:
        frame = frame[columns]
    return frame.to_markdown(index=False, floatfmt=".3f")


def _fidelity_comparison(controller: dict[str, Any]) -> pd.DataFrame:
    old = pd.read_parquet(
        ROOT / "results/v13/processed/moving_tangent_oracle_development_v13.parquet"
    )
    old = old[
        (old.method == "static_local_causal_oracle")
        | ((old.method == "moving_tangent_interpolated_oracle") & (old.alpha == 1.0))
    ]
    current = pd.read_parquet(
        ROOT / "results/v14/processed/closed_loop_fidelity_development_v14.parquet"
    )
    columns = [
        "method",
        "horizon",
        "direction_cosine",
        "magnitude_ratio",
        "output_direction_cosine",
        "semantic_delta_agreement",
        "semantic_vector_cosine",
        "task_decision_sign_agreement",
    ]
    frame = pd.concat([old[columns], current[columns]], ignore_index=True)
    return (
        frame.groupby(["method", "horizon"])
        .agg(
            direction=("direction_cosine", "mean"),
            magnitude=("magnitude_ratio", "mean"),
            output=("output_direction_cosine", "mean"),
            semantic_legacy=("semantic_delta_agreement", "mean"),
            semantic_continuous=("semantic_vector_cosine", "mean"),
            sign=("task_decision_sign_agreement", "mean"),
        )
        .reset_index()
    )


def _gate(group: pd.DataFrame) -> bool:
    return bool(
        (group.direction >= 0.8).all()
        and group.magnitude.between(0.8, 1.2).all()
        and (group.output >= 0.8).all()
        and (group.semantic_legacy >= 0.8).all()
        and (group.sign >= 0.8).all()
        and set(group.horizon) == {1, 2, 4, 8}
    )


def main() -> None:
    numerical = _json("numerical_snr_summary_v14.json")
    curvature = _json("local_causal_curvature_v14.json")
    transport = _json("transport_analysis_v14.json")
    control = _json("closed_loop_development_v14.json")
    channels = _json("joint_channel_ablation_v14.json")
    split = json.loads(
        (ROOT / "artifacts/finite_causal_control_v14_splits.freeze.json").read_text()
    )
    finalist = json.loads(
        (
            ROOT / "artifacts/finite_causal_control_v14_finalist_decision.freeze.json"
        ).read_text()
    )
    guard = json.loads((ROOT / "artifacts/v13_immutable.sha256.json").read_text())
    pair = pd.read_parquet(PROCESSED / "transport_pairs_v14.parquet")
    loops = pd.read_parquet(PROCESSED / "holonomy_loops_v14.parquet")
    paths = pd.read_parquet(PROCESSED / "transported_path_dimension_v14.parquet")
    steps = pd.read_parquet(PROCESSED / "closed_loop_development_v14.parquet")
    fidelity = _fidelity_comparison(control)
    jvp = pd.read_parquet(PROCESSED / "numerical_snr_summary_v14.parquet")
    jvp_focus = jvp[jvp.epsilon.isin([1e-5, 0.001, 0.1, 1.0, 2.0])]
    alpha = pd.read_parquet(PROCESSED / "v13_alpha_snr_reanalysis_v14.parquet")
    alpha_h1 = (
        alpha[alpha.horizon == 1]
        .groupby(["scale", "direction_snr_label"])
        .agg(
            n=("base_trial_id", "size"),
            j=("j_direction", "mean"),
            output=("output_direction", "mean"),
        )
        .reset_index()
    )
    fp32 = pd.read_parquet(
        PROCESSED
        / "numerical_partial_fp32_extension_2/jvp_finite_writeback_audit_v14.parquet"
    )
    fp32_supported = bool(fp32.error.isna().all())
    partial_fp32_one = fp32[(fp32.epsilon == 1.0) & (fp32.target == "j")]
    baseline_one = pd.read_parquet(
        PROCESSED / "numerical_scale_extension_1/jvp_finite_writeback_audit_v14.parquet"
    )
    baseline_one = baseline_one[
        (baseline_one.epsilon == 1.0)
        & (baseline_one.target == "j")
        & (baseline_one.precision_mode == "v13_fp32_accumulate_bf16_writeback")
    ]
    _write(
        "JVP_FINITE_WRITEBACK_AUDIT_V14.md",
        f"""# JVP versus finite writeback — V14

The exact V13 autograd JVP was compared with central finite differences on five frozen train anchors, five predeclared direction families, four target blocks, and ε from `1e-5` to `2` under separately frozen base and extension protocols. This tests the finite BF16 writeback interface, not a new model weight state.

- Three clean repetitions per anchor were byte/deterministically identical at the target readout (`noise=0`); dividing by that zero floor would falsely imply infinite SNR.
- Empirical 5th-percentile nonzero writeback output floors: `{numerical["quantization_floors"]}`.
- Validation-only `MIN_CAUSAL_EFFECT_NORM`: `{numerical["min_causal_effect_norm"]:.8f}` (5th percentile of V13 validation raw-teacher J-effect norms across h1/h2/h4/h8).
- First ε satisfying the predeclared *all-target* median JVP equivalence rule (cosine ≥0.95 and relative L2 ≤0.20): `{numerical["first_all_target_jvp_reliable_epsilon"]}`. **No reliable scale was identified through ε=2.** At that scale the perturbation is finite, not an infinitesimal check.
- All-FP32 cache failed because BF16 attention query requires matching key/value dtype. REC/conv-only FP32 with BF16 KV supported: `{fp32_supported}`. FP64 finite-model reference is not feasible without changing frozen BF16 model weights.
- At ε=1, REC/conv-FP32/BF16-KV J median cosine/relative L2 was `{partial_fp32_one.cosine.median():.3f}` / `{partial_fp32_one.relative_l2.median():.3f}` versus ordinary V13 writeback `{baseline_one.cosine.median():.3f}` / `{baseline_one.relative_l2.median():.3f}`; partial FP32 does not materially fix the gate.

| ε | target | median cosine | median relative L2 | median quantization-floor SNR | pass fraction |
|---:|---|---:|---:|---:|---:|
{chr(10).join(f"| {row.epsilon:g} | {row.target} | {row.median_cosine:.3f} | {row.median_relative_l2:.3f} | {row.median_effective_snr:.2f} | {row.jvp_agreement_rate:.2f} |" for row in jvp_focus.itertuples())}

V13 alpha rows were retained and relabeled; no V13 result changed. Qualification uses `alpha × full-teacher J-effect norm` as a development-data proxy, not a newly measured scaled effect. h1 all-row versus SNR-qualified reinterpretation:

{_table(alpha_h1.to_dict("records"))}

All `α=.05/.10` h1 rows fall below the direction threshold. At `α=.25`, the 3 qualified h1 rows improve J direction to about 0.887 but output direction remains about 0.547. Later-horizon qualified rows also fail. Thus near-zero metric instability explains part, not all, of V13 finite-control failure.

Machine records: `results/v14/processed/jvp_finite_writeback_audit_v14.parquet`, both numerical extension folders, `numerical_snr_summary_v14.parquet`, and `v13_alpha_snr_reanalysis_v14.parquet`. The original JSON audit preserves its `NaN` undefined-cosine tokens; the separately frozen strict-JSON correction maps only those tokens to `null` in `jvp_finite_writeback_audit_v14_strict.json`.
""",
    )
    curvature_table = pd.DataFrame(curvature["pooled"])
    _write(
        "LOCAL_CAUSAL_CURVATURE_V14.md",
        f"""# Local causal curvature — V14

The top `{9}` local causal directions were taken from frozen V13 rank-512 JVP matrices, one train anchor per family. Central second differences at radius `1.0`, two mixed-direction pairs, and held-out radii `0.5` and `2.0` were measured. Since the infinitesimal JVP gate failed, these are **finite-scale curvature diagnostics, not validated differential Hessian estimates**.

{_table(curvature_table.to_dict("records"), ["radius", "target", "surrogate", "direction", "relative_l2", "magnitude"])}

- First-order valid radius under the all-target direction/magnitude/relative-error rule at tested held-out radii `0.5/2.0`: `{curvature["first_order_valid_radius"]}`.
- Second-order valid radius under the same tested radii: `{curvature["second_order_valid_radius"]}`.
- The quadratic term improves direction prediction at radius `0.5`, but relative errors remain above `0.20`; it does not validate a finite local surrogate.
- Mixed second-difference terms were retained in `results/v14/processed/local_causal_mixed_curvature_v14.parquet`.
""",
    )
    _write(
        "CAUSAL_TANGENT_TRANSPORT_V14.md",
        f"""# Causal tangent transport — V14

V13's ten train-anchor local matrices were aligned at rank 16 with explicit sign/index-invariant mappings. Same-prompt successive-position mean fidelity:

{_table(transport["same_prompt_comparison"])}

Procrustes and minimal Grassmann-geodesic endpoint transport coincide mathematically for these full-rank principal-angle alignments; their identical numbers are not independent replications. The reported `raw_direction_cosine` is actually a **512-dimensional probe-coordinate** cosine, not a metric-calibrated full raw-state cosine because the frozen probe directions need not be orthonormal. J-response alignment improves J-effect direction but can sacrifice probe-coordinate/output/semantic alignment, demonstrating a target-specific gauge choice. Only common selected logit IDs were compared across states; the comparison does not assert global coordinate identity or finite steering success.

Machine records: `results/v14/processed/transport_pairs_v14.parquet` (`{len(pair)}` rows).
""",
    )
    meaningful_loops = loops[loops.method != "unaligned"]
    _write(
        "CAUSAL_HOLONOMY_V14.md",
        f"""# Path dependence and holonomy — V14

Ten train-state loops A→B→C→A were constructed, with same-prompt A/B and C chosen by nearest frozen JVP matrix within family when available. This does not guarantee that C is close in full raw state or J-state distance. Mean results by mapping:

{_table(transport["holonomy"])}

Procrustes/geodesic coordinate return error is `{meaningful_loops[meaningful_loops.method == "procrustes"].coordinate_return_error.mean():.3f}`, with J-effect return error `{meaningful_loops[meaningful_loops.method == "procrustes"].j_effect_return_error.mean():.3f}`. This is measurable path dependence in the tested local atlas, not proof that a low-dimensional manifold is absent. The zero error of `unaligned` is a tautology (identity map), not evidence of flat geometry. Degree-style holonomy is geometrically interpretable only for the orthogonal maps; projector and J-response maps are not isometries. Channel-wise return error in physical cache space was not measured, so that part of the requested audit remains unidentified.

Machine records: `results/v14/processed/holonomy_loops_v14.parquet`.
""",
    )
    _write(
        "TRANSPORTED_PATH_DIMENSION_V14.md",
        f"""# Transported path dimension — V14

For ten three-state train paths, naive **probe-coordinate-basis** union r95 has median `{paths.naive_raw_union_r95.median():.1f}` and range `{paths.naive_raw_union_r95.min()}–{paths.naive_raw_union_r95.max()}` at local rank 16. Transported *coordinate* dimension is by construction at most 16. These quantities live in different ambient spaces; Procrustes rotation within a basis cannot change the union rank in the original probe-coordinate space. Therefore V14 does **not** claim that transport lowered V13's cumulative rank (`11` under its different frozen rank/path panel), much less full raw-state dimension.

Machine records: `results/v14/processed/transported_path_dimension_v14.parquet`.
""",
    )
    terminal = (
        steps.sort_values("step").groupby(["base_trial_id", "method"]).tail(1).copy()
    )
    terminal["class"] = np.select(
        [
            terminal.raw_residual_ratio <= 0.2,
            terminal.raw_residual_ratio <= 0.8,
            terminal.raw_residual_ratio >= 0.9,
        ],
        [
            "reachable_in_development",
            "partial_reduction_or_stall",
            "no_meaningful_reduction",
        ],
        default="intermediate_reduction",
    )
    reach = (
        terminal.groupby(["method", "family", "class"]).size().reset_index(name="cases")
    )
    _write(
        "FINITE_CAUSAL_REACHABILITY_V14.md",
        f"""# Finite causal reachability — V14 development only

Teacher-target residuals were measured for the same ten V13 validation cases used by the prior development oracle. The controller never receives a raw target cache; raw teacher state is used only to evaluate the teacher response label. Residual ratio is `||Y*−Y(P_k)|| / ||Y*−Y(P_0)||` for the selected horizon objective.

{_table(control["residual_curve"])}

Terminal case categories (development diagnostics, not independent reachability claims):

{_table(reach.to_dict("records"))}

Backtracking enforces improvement in the weighted objective, not necessarily every unweighted target or horizon. A failure here cannot establish V14-E while the local finite-response model and independent confirmatory panel remain unvalidated.

Machine records: `results/v14/processed/closed_loop_development_v14.parquet`.
""",
    )
    _write(
        "CLOSED_LOOP_CAUSAL_CONTROL_V14.md",
        f"""# Closed-loop finite causal control — V14 development

The tested controller uses a shared rank-9 REC/conv/KV direction basis, finite central response at ε=1 (because exact-JVP equivalence failed), ridge `0.01`, trust radius `0.5`, four steps, and backtracking `[1,.5,.25]`. At every accepted step the causal response is remeasured and the nearest V13 tangent atlas basis is retrieved. Only `{steps.nearest_atlas_index.nunique()}` atlas index was actually visited, so this run **did not test an effective between-basis transport switch**. It optimizes weighted future J, logits, continuous semantics, and workspace effects, **not raw target-state distance**. Horizon objectives were h1 and h1+h2+h4+h8.

Static V13, V13 moving-interpolated, and V14 closed-loop on the same validation panel:

{_table(fidelity.to_dict("records"))}

All-horizon frozen gate pass by method: `{ {method: _gate(group) for method, group in fidelity.groupby("method")} }`. Continuous semantic fidelity is reported separately and does not replace the legacy gate. This is a development experiment, not V14-D independent confirmation. Differences versus V13 are exploratory because the controller's finite-response derivative was not validated as a local Jacobian.

The all-horizon objective did not consistently improve h4/h8 over h1-only (h4 J direction fell from about 0.498 to 0.483; h8 rose only from about 0.511 to 0.524). It did not resolve long-horizon rotation or semantic divergence. h16 was not authorized.

Per-step predicted/actual improvement, trust ratio, control norm, residual, and atlas index are in `results/v14/processed/closed_loop_development_v14.parquet`.
""",
    )
    _write(
        "JOINT_CHANNEL_CAUSAL_REALIZATION_V14.md",
        f"""# Shared-coordinate channel realization — V14

The same local rank-3 coordinates were realized jointly in REC, convolution, and KV heads. Each channel was ablated at finite scale 1.0, without fitting a large decoder. Low-effect rows are retained but not classified as channel-required under the frozen `MIN_CAUSAL_EFFECT_NORM`.

- SNR-qualified ablation rows: `{channels["qualified_rows"]}`.
- Fraction marked `JOINT_CHANNEL_REQUIRED` when one-channel removal made J cosine <0.8 or norm ratio <0.8: `{channels["joint_required_fraction"]}`.

{_table(channels["by_channel"]) if channels["by_channel"] else "No SNR-qualified channel rows; channel necessity not identified."}

This is a finite writeback ablation, not a learned shared latent decoder or complete-state replacement. Machine records: `results/v14/processed/joint_channel_ablation_v14.parquet`.
""",
    )
    all_gate = any(
        _gate(group)
        for method, group in fidelity.groupby("method")
        if method.startswith("closed_loop")
    )
    _write(
        "STRICT_INTERFACE_AUDIT_V14.md",
        f"""# Strict interface and authorization audit — V14

- V1–V13 tracked-byte guard: `{guard["file_count"]}` files, digest `{guard["guard_digest"]}`; cumulative `FINAL_REPORT.md` is the declared mutable exception.
- V14 diagnostic train / development validation are disjoint; split hashes `{split["diagnostic_train"]["id_sha256"]}` / `{split["development_validation"]["id_sha256"]}`.
- A new independent confirmatory bank was **not** created or used. The numerical all-target JVP gate failed and the frozen finalist decision is `{finalist["finalist_status"]}` (digest `{finalist["freeze_digest"]}`). This is a required stop, not a final-test success.
- Raw teacher cache never enters the candidate control update or basis search; only teacher response labels define residuals. There is no direct target-cache bypass in the developmental controller.
- Development all-horizon gate pass: `{all_gate}`. Independent h1/h2/h4/h8 gate pass: **not tested**.
- Absolute state replacement: `ABSOLUTE_STATE_REPLACEMENT_NOT_AUTHORIZED`. V13/V14 controls are delta/edit interfaces, not absolute raw-state representations.
- H2 remains. H3 not supported. `AUTONOMOUS_CONTROLLER_AUTHORIZED = FALSE`.

Formal procedural outcome: `V14-STOP — NUMERICAL_INTERFACE_GATE_FAILED`. No V14-A–E scientific branch is fully established: a writeback floor is demonstrated, but all-target JVP/finite equivalence and independently confirmed finite control are not.
""",
    )
    freezes = sorted(ROOT.glob("artifacts/finite_causal_control_v14*.freeze.json"))
    freeze_rows = [
        {
            "freeze": str(path.relative_to(ROOT)),
            "SHA256": sha256_file(path),
            "digest": json.loads(path.read_text())["freeze_digest"],
        }
        for path in freezes
    ]
    _write(
        "EXECUTION_MANIFEST_V14.md",
        f"""# V14 execution and provenance manifest

Server workspace: `/data/CSK/J-space-project/jstate-closure` on `222.20.126.223`. Parent commit: `37df399e4bba8afaca6d99721fe8ceee653e78ee`. V1–V13 tracked guard digest: `{guard["guard_digest"]}`. Diagnostic/development ID hashes: `{split["diagnostic_train"]["id_sha256"]}` / `{split["development_validation"]["id_sha256"]}`. Independent-final status: `{split["independent_final"]["status"]}`.

Canonical commands (all from repo root with `/home/user/anaconda3/bin/python`; GPU stages use `HF_HOME=/data/CSK/J-space-project/.hf-cache`):

```bash
python -m jclosure.experiments.numerics_v14 --stage freeze
python -m jclosure.experiments.numerics_v14 --stage audit
python -m jclosure.experiments.numerics_v14_extension_1
python -m jclosure.experiments.analyze_numerics_v14
python -m jclosure.experiments.transport_v14 --stage analyze
python -m jclosure.experiments.curvature_v14
python -m jclosure.experiments.closed_loop_v14
python -m jclosure.experiments.numerics_v14_extension_2
python -m jclosure.experiments.channels_v14
python scripts/create_v13_immutable_for_v14.py
python scripts/freeze_v14_splits.py
python scripts/freeze_v14_finalist_decision.py
python scripts/normalize_v14_numeric_json.py
python -m jclosure.reporting_v14
python scripts/build_v14_integrity.py
python scripts/build_complete_version_report.py V14
python -m pytest -q
```

Frozen protocol and amendment hashes:

{_table(freeze_rows)}

Machine-record hashes are indexed in `reports/V14_COMPLETE_REPORT.md` for top-level files and in the nested extension summaries for extension records. Exact changed-file list after commit is obtained with `git diff --name-only 37df399e4bba8afaca6d99721fe8ceee653e78ee..HEAD`.
""",
    )
    final = REPORTS / "FINAL_REPORT.md"
    text = final.read_text(encoding="utf-8")
    section = f"""<!-- V14_START -->
## V14 — Finite Causal Control and Tangent Transport

Formal procedural outcome: **V14-STOP — NUMERICAL_INTERFACE_GATE_FAILED**. The BF16 persistent-state writeback creates a measured finite-effect floor; the frozen all-target exact-JVP/finite-difference equivalence gate did not pass through ε=2. `MIN_CAUSAL_EFFECT_NORM = {numerical["min_causal_effect_norm"]:.8f}`. SNR relabeling explains part of V13's small-alpha anomaly, not the later-horizon failures. First- and second-order valid radii: `{curvature["first_order_valid_radius"]}` / `{curvature["second_order_valid_radius"]}`. Train-atlas transport and holonomy are measurable, but they do not establish a global compact coordinate system.

Development closed-loop causal steering was run on ten validation cases; independent confirmation was **not** run because no numerically validated finalist was available. V14-A through V14-E are not fully established. H2 remains; H3 is not supported. Absolute state replacement and autonomous controller training are **not authorized**. V1–V13 conclusions remain frozen. See `reports/V14_COMPLETE_REPORT.md` for the full report bundle.
<!-- V14_END -->
"""
    if "<!-- V14_START -->" in text:
        prefix, remainder = text.split("<!-- V14_START -->", 1)
        _, suffix = remainder.split("<!-- V14_END -->", 1)
        final.write_text(
            (prefix.rstrip() + "\n\n" + section + suffix).rstrip() + "\n",
            encoding="utf-8",
        )
    else:
        final.write_text(text.rstrip() + "\n\n" + section, encoding="utf-8")
    report_paths = sorted(REPORTS.glob("*_V14.md"))
    index = {
        "protocol_version": "finite_causal_control_v14_reporting",
        "report_count": len(report_paths),
        "reports": {
            str(path.relative_to(ROOT)): sha256_file(path) for path in report_paths
        },
        "final_report_sha256": sha256_file(final),
        "formal_outcome": "V14-STOP — NUMERICAL_INTERFACE_GATE_FAILED",
        "independent_confirmatory": False,
    }
    write_json_atomic(PROCESSED / "report_integrity_v14.json", index)
    print(json.dumps(index, sort_keys=True)[:2000])


if __name__ == "__main__":
    main()
