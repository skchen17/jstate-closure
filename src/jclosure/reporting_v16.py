"""Evidence-gated standalone and single-file-ready V16 reporting."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from jclosure.protocol_v16 import stage_freeze, verify
from jclosure.provenance import sha256_file, write_json_atomic

ROOT=Path.cwd();OUT=ROOT/"results/v16/processed";REPORTS=ROOT/"reports"
SOURCE="src/jclosure/reporting_v16.py"


def _json(name:str)->dict[str,Any]:
    return json.loads((OUT/name).read_text(encoding="utf-8"))


def _write(name:str,body:str)->None:
    (REPORTS/name).write_text(body.rstrip()+"\n",encoding="utf-8")


def _number(value:Any,digits:int=3)->str:
    if value is None:return "not measured"
    return f"{float(value):.{digits}f}"


def _table(frame:pd.DataFrame,columns:list[str])->str:
    if frame.empty:return "No measured rows."
    return frame[columns].to_markdown(index=False,floatfmt=".3f")


def _summary_table(section:dict[str,Any],kind:str)->str:
    values=section.get(kind,{})
    rows=[]
    for target,record in values.items():
        rows.append({"target":target,**record})
    if not rows:return "No estimable records."
    frame=pd.DataFrame(rows)
    return _table(frame,[name for name in ("target","count","even_over_odd","linear_norm","quadratic_norm","cubic_norm","quadratic_over_linear","scale_fit_relative_l2","interaction_norm","interaction_over_sum") if name in frame])


def main()->None:
    base=verify(ROOT)
    split=json.loads((ROOT/"artifacts/nonlinear_finite_causal_action_v16_splits.freeze.json").read_text())
    analysis=_json("v16_analysis.json")
    realized=_json("realized_action_coordinates_v16.json")
    composition=_json("action_composition_v16.json")
    sequences=_json("sequence_reachability_v16.json")
    model=pd.read_parquet(ROOT/analysis["records"]["models"])
    finalist=analysis["model"]["candidate"]
    if finalist is not None:
        raise RuntimeError("A response finalist passed; run frozen nonlinear MPC before adjudication/reporting")
    decision_path=ROOT/"artifacts/nonlinear_finite_causal_action_v16_finalist_decision.freeze.json"
    if not decision_path.exists():
        decision=stage_freeze(ROOT,"finalist_decision",[SOURCE,"results/v16/processed/v16_analysis.json",
                                                       "results/v16/processed/realized_action_coordinates_v16.json",
                                                       "results/v16/processed/action_composition_v16.json",
                                                       "results/v16/processed/sequence_reachability_v16.json"],
                              {"response_validation_gate_passed":False,"development_h1_control_gate_passed":False,
                               "eligible_finalist":None,"independent_final":"NOT_CREATED_OR_OPENED",
                               "reason":"No requested-coordinate k/model met all predeclared response holdouts; MPC development gate not entered",
                               "formal_outcome":"V16-STOP — NONLINEAR_RESPONSE_VALIDATION_GATE_NOT_PASSED",
                               "V16_F_not_claimed":"realized-coordinate audit restricted to frozen 20/10-state diagnostic subset; primitive/dynamic sequence study restricted to four actions and five validation states"})
    else:
        decision=json.loads(decision_path.read_text())
    bank=analysis["bank"];decomp=analysis["decomposition"]
    best=model[model.holdout=="heldout_state_all"].sort_values("stacked_normalized_relative_l2").head(10)
    by_k=model[model.holdout=="heldout_state_all"].sort_values("stacked_normalized_relative_l2").groupby("k").head(1).sort_values("k")
    channel_compare=model[(model.holdout=="heldout_state_pair")&(model.model.isin(["quadratic","channel_bilinear"]))].sort_values(["k","model"])
    realized_model=pd.DataFrame(realized["model_comparison"])
    realized_best=realized_model.sort_values("stacked_normalized_relative_l2").groupby("input").head(1)
    rank=pd.DataFrame(analysis["nonlinear_rank_and_primitives"]["observed_spectra"])
    rank_val=rank[rank.role=="validation"].groupby("k").agg(r90=("r90","median"),r95=("r95","median"),r99=("r99","median"),states=("base_trial_id","nunique")).reset_index()
    primitives=pd.DataFrame(analysis["nonlinear_rank_and_primitives"]["primitive_diagnostic"])
    reach=pd.read_parquet(ROOT/analysis["records"]["reachability"])
    reach_group=reach.groupby("design").agg(states=("base_trial_id","nunique"),candidates=("candidate_count","median"),nearest_relative_residual=("nearest_relative_residual","median"),action_count=("action_count_simultaneous","median")).reset_index()
    sequence_frame=pd.DataFrame(sequences["by_depth"])
    comparison=pd.read_parquet(ROOT/"results/v15/processed/control_fidelity_v15.parquet")
    from jclosure.reporting_v15 import _comparison
    historical=_comparison()
    old_h1=comparison[(comparison.horizon==1)&(comparison.method=="actuator_calibrated_h1")]
    old_row=old_h1.iloc[0] if len(old_h1) else None
    _write("FINITE_ACTION_BANK_V16.md",f"""# Reliable finite-action response bank — V16

Pre-response frozen split: train **{bank['states_by_role'].get('train')}** states (20 per family), validation **{bank['states_by_role'].get('validation')}** states (10 per family), five families each; ID SHA256 train `{split['train_id_sha256']}`, validation `{split['validation_id_sha256']}`. V15 development IDs were excluded from V16 validation. A newly generated independent final bank was **not opened** because the response validation gate did not pass. This is a validation bank, not independent confirmatory control evidence.

The 32 nested proposal directions draw from V13 autograd causal-weighted, architecture-balanced and random raw directions plus V15 train-selected high finite response and poor-autograd-subspace-residual directions. Direction indices and family labels are in the split freeze. Every action uses real joint REC/Conv/KV BF16 writeback and a state/direction-specific first-passing amplitude among {base['config']['action']['calibration_scales']}. Both signs must meet state cosine ≥0.95, gain 0.8–1.2 and J effect ≥0.0068028033. Failure is retained as `ACTION_NOT_RELIABLY_ACTUATABLE`.

- Train rows: {bank['actions_by_role'].get('train')}; reliable {bank['reliable_by_role'].get('train')}.
- Validation rows: {bank['actions_by_role'].get('validation')}; reliable {bank['reliable_by_role'].get('validation')}.
- Overall reliable-action fraction: {_number(bank['reliable_fraction'])}.
- Designs: {json.dumps(bank['by_design'],sort_keys=True)}.

Per-row requested alpha/coordinate, realized norm/cosine/gain, channel survival, five target responses and failure labels are in `{bank['records']}` (SHA256 `{bank['records_sha256']}`). Raw per-state Parquet remains in `results/v16/raw/`, preserving the actual split and all failures. Calibration attempts before the final selected amplitude were not all retained as separate target-response rows; the selected/last tried amplitude is explicit.
""")
    _write("NONLINEAR_ACTION_DECOMPOSITION_V16.md",f"""# Finite-response nonlinearity decomposition — V16

All statistics below are medians of measured reliable actions only, with unreliables retained in the bank denominator. `G_even/G_odd` uses the exact ± finite responses; scale terms are descriptive least-squares coefficients over the frozen signed sweep, not proof that dynamics are polynomial.

## Odd/even ratio

{_summary_table(decomp,'odd_even')}

## Scale linear/quadratic/cubic fit

{_summary_table(decomp,'scale_fit')}

## Pair interactions

`I_ij=G(e_i+e_j)-G(e_i)-G(e_j)`; target-block contribution is reported separately, including J, logits, continuous semantic, workspace and normalized stack.

{_summary_table(decomp,'pair_interaction')}

Explicit low-rank REC×Conv, REC×KV and Conv×KV bilinear terms are compared against the ordinary quadratic model on held-out states/pairs. These are channel-energy proxy interactions, not a unique physical decomposition:

{_table(channel_compare,['k','model','test_count','j_relative_l2','stacked_normalized_relative_l2'])}

## Sparse triple Möbius interaction

{_summary_table(decomp,'triple_mobius')}

Not all pair subsets of the frozen triples were sampled, so unavailable three-way contrasts remain **not estimable**; no zero interaction is imputed. Full per-state values: `{analysis['records']['decomposition']}`.
""")
    _write("REALIZED_ACTION_COORDINATES_V16.md",f"""# Requested versus realized action coordinates — V16

The diagnostic reconstruction covers {realized['state_count_by_role']} and {realized['action_count']} reliable actions. For each action, the exact BF16 state delta is projected by least squares onto the frozen 32-direction raw REC/Conv/KV span. Median out-of-span energy fraction (norm residual / realized norm): **{_number(realized['median_span_residual_fraction'])}**. The response model comparison below refits and tests both inputs on the **same subset**; it is not compared unfairly to the 100/50 requested-input model.

{_table(realized_best,['input','k','model','train_count','validation_count','j_direction','j_relative_l2','stacked_normalized_direction','stacked_normalized_relative_l2'])}

`requested_raw_coordinate = requested_alpha × z`; `realized_raw_coordinate` is the projected BF16 delta. The recorded residual prevents silently treating quantization spill outside the span as modeled action. Even if realized inputs improve prediction, this subset alone cannot establish quantization as the primary source of all V15 output nonlinearity. Records: `{realized['records']}` and `{realized['model_records']}`.
""")
    _write("NONLINEAR_RANK_INFLATION_V16.md",f"""# Nonlinear finite-rank inflation — V16

V15's restricted actual r95≈14 versus exact-JVP r95≈7 is a **linear response spectrum**, not 14 independent causal degrees of freedom. V16 observed positive-single finite spectra on new validation states are:

{_table(rank_val,['k','states','r90','r95','r99'])}

These spectra center a nested set of up to 32 positive single actions per state; V15's 512-probe signed-central-response spectrum uses a different distribution, so the two r95 values are not a direct replication comparison.

No nonlinear model met all frozen J and stacked-response heldouts. Thus predicted linear-only, quadratic and full-model spectra are **not validated explanations** of observed rank inflation, and the phrase `FINITE_LINEAR_RANK_INFLATION_EXPLAINED_BY_LOW_DIMENSIONAL_NONLINEAR_ACTIONS` is not licensed. Per-state spectra, including first 40 singular values, are in `results/v16/processed/v16_analysis.json`. Variance explained by any unvalidated model would be training/descriptive only.
""")
    _write("NONLINEAR_ACTION_DIMENSION_V16.md",f"""# Minimal nonlinear action dimension — V16

Frozen candidates: k=2,4,8,16,32, nested across all five direction proposals. The response gate requires median J **and** normalized-stack direction ≥0.90, relative L2 ≤0.30, norm ratio 0.8–1.2 on state, strict sign, strict scale, strict pair and unseen dense heldouts. No model/dimension met every required holdout: `NO_COMPACT_NONLINEAR_ACTION_DIMENSION_IDENTIFIED`. This means **within tested models/protocol only**, not mathematical nonexistence. Within-state, across-state and across-family `k_min` are not assigned when the gate fails. Family-exclusion rows are diagnostic, not an independent final bank.

Best observed model per dimension on held-out states (not sufficient alone to pass):

{_table(by_k,['k','model','test_count','j_direction','j_relative_l2','stacked_normalized_direction','stacked_normalized_relative_l2','gate_pass'])}

All M0 linear, M1 quadratic, M2 channel-bilinear, M3 train-selected sparse cubic and M4 clean-J-conditioned piecewise quadratic comparisons, including strict action-design holdouts and output/semantic metrics: `{analysis['records']['models']}`. M5 small MLP was not launched because the frozen resource priority and failed simpler models did not justify its cost; V16-F therefore remains unclaimed.
""")
    _write("ACTION_COMPOSITION_V16.md",f"""# Finite action composition and order — V16

Ten frozen states (first per family from train and validation) test two reliable actions. Same-state u→v versus v→u measures BF16 writeback order; +u→+v→−u→−v is an empirical finite loop, **not** a Lie bracket. Teacher-forced dynamic order reuses the same raw directions after one controlled token; proper tangent transport was not estimated.

- Measured states: {composition['measured_states']}.
- Median same-state J order L2: {_number(composition['static_order_j_l2_median'],6)}.
- Median dynamic J order L2: {_number(composition['dynamic_order_j_l2_median'],6)}.
- Median finite-loop J residual relative to larger first-order J effect: {_number(composition['loop_j_relative_median'])}.
- States above frozen static-order noncommutativity threshold: {composition['noncommutative_order_threshold_pass_count']}.

The threshold is UV-versus-VU J norm >1e−6 and relative to the larger first-order effect >0.01. The restricted tested-regime label is **{'NONCOMMUTATIVE_ACTION_COMPOSITION_PRESENT' if composition['noncommutative_order_threshold_pass_count'] else 'NO_ORDER_EFFECT_ABOVE_FROZEN_THRESHOLD'}**. Loop residuals are reported separately and do not alone prove order dependence. State/output/semantic order effects are in `{composition['records']}`. Dynamic order effects mix writeback and token-transition dependence; they are not proof of a smooth noncommutative geometric bracket.
""")
    _write("CAUSAL_ACTION_PRIMITIVES_V16.md",f"""# Finite causal action primitive diagnostic — V16

A greedy dictionary of mean **train** single-action responses was selected without validation labels; held-out validation positive-single target-space linear projection coverage is:

{_table(primitives,['K','heldout_linear_response_coverage'])}

This is only a response-space coverage diagnostic. The candidate primitives were not independently tested as a complete semi-discrete controller, and mixtures/sequences were not validated across states; `FINITE_CAUSAL_ACTION_PRIMITIVE_REPRESENTATION_SUPPORTED` is **not** claimed. The selected indices at each K and exact coverage are in `v16_analysis.json`. No cognitive-primitive interpretation is made.
""")
    _write("NONLINEAR_CAUSAL_REACHABILITY_V16.md",f"""# Empirical nonlinear causal reachability — V16

Frozen V13 captured h1 teacher J deltas are used **only as labels**, never as response-model/controller inputs. On V16 validation states, nearest measured residuals to the teacher J target are:

{_table(reach_group,['design','states','candidates','action_count','nearest_relative_residual'])}

Singles are one finite writeback. Pair, triple and dense entries are simultaneous same-state mixtures, **not** two-/four-token action sequences. This table alone cannot establish that multi-step nonlinear reachable sets exceed the V15 linear span. Every selected coordinate, action norm, candidate count and family is in `{analysis['records']['reachability']}`.

In a separate **restricted actual teacher-forced sequence** audit, the same five validation states and same h4 teacher J target were used at all depths; each active step chooses one of ± two calibrated reliable primitives, without tangent transport. Exhaustive candidate counts and nearest residuals:

{_table(sequence_frame,['depth','states','candidate_count','nearest_relative_residual'])}

This is post-hoc nearest-set search, not an authorized response-model-based controller. The small four-primitive/five-state domain and unchanged raw directions prevent a general multi-step reachability claim. Per-state best sequence and action norm: `{sequences['records']}`.
""")
    v15text=(f"h1 V15 actuator-aware development direction {float(old_row.direction_cosine):.3f}, magnitude {float(old_row.magnitude_ratio):.3f}, output {float(old_row.output_direction_cosine):.3f}" if old_row is not None else "V15 control baseline unavailable")
    _write("NONLINEAR_FINITE_CAUSAL_CONTROL_V16.md",f"""# Nonlinear finite causal control — V16

**Not launched by the frozen gate.** No M0–M4 candidate passed the held-out J and stacked finite-response gate. Nonlinear MPC requires that gate before development; a controller fit to an unvalidated response map would repeat V15's unsupported-actuator error. No V16 h1/h2/h4/h8 causal-control metrics, trust ratios or predicted-versus-actual improvement exist. Sequence MPC and independent confirmation were not opened.

Historical comparisons remain frozen: {v15text}; V13 static/moving and V14 development results are preserved in their own records. Raw teacher self-reference is a label ceiling only (identity=1), never a target-cache bypass. A direct V16-vs-historical controller comparison cannot be made because no V16 controller was authorized. No historical full causal gate was removed: direction ≥0.8, magnitude 0.8–1.2, output ≥0.8, legacy semantic ≥0.8 and sign ≥0.8 at h1/h2/h4/h8 remain mandatory for any future independent control claim.

Frozen same-panel historical development metrics (not V16 results):

{_table(historical,['method','horizon','direction','magnitude','output','semantic_continuous','semantic_legacy','sign'])}
""")
    _write("STRICT_INTERFACE_AUDIT_V16.md",f"""# Strict interface and scientific-claim audit — V16

- V1–V15 protocols/results are immutable; only cumulative `FINAL_REPORT.md` is appended. Parent commit: `{base['parent_commit']}`.
- Real BF16 REC/Conv/KV writeback and readback are used. Requested and realized actions are separated; failed actions stay in the denominator.
- Model inputs use compact action coordinates; M4 uses only train-frozen current clean J summary. No raw persistent state or teacher raw target cache enters the response model/controller.
- Validation states are disjoint from model-fit states; strict sign, scale, pair, dense and family diagnostics are separately recorded. No independent V16 final bank was opened.
- V15 r95 gap is not 14 control dimensions. Polynomial fit is not a declaration of true quadratic dynamics. Action representation is not complete state representation.
- No V16 nonlinear MPC or autonomous S→S′ model was trained. `ABSOLUTE_REPLACEMENT_NOT_AUTHORIZED`; H2 remains; H3 candidate and full H3 are unsupported.
- V16-F is **not** asserted: despite the full 100/50 response bank, exact realized coordinates cover a 20/10 subset and primitive/sequence studies are diagnostic, not the comprehensive negative test required for F.

Formal branch audit: V16-A not supported (realized coordinates do not materially improve matched-subset response prediction); V16-B/C fail the frozen response gate; V16-D remains diagnostic rather than validated; V16-E is gated off before MPC/independent confirmation; V16-F is not eligible as a comprehensive negative claim.

Formal procedural result: **{decision['formal_outcome']}**. Independent final: `{decision['independent_final']}`. This is not a theorem of absent compact causal control.
""")
    manifest=f"""# V16 execution manifest

Canonical workspace: `/data/CSK/J-space-project/jstate-closure` on `222.20.126.223`. Parent commit `{base['parent_commit']}`. Runtime: `HF_HOME=/data/CSK/J-space-project/.hf-cache PYTHONPATH=src /home/user/anaconda3/bin/python`; model placement is the frozen V13 dual-GPU loader. Exact repository-root command sequence:

```bash
PYTHONPATH=src python -m jclosure.protocol_v16 freeze
PYTHONPATH=src python -m jclosure.experiments.action_bank_v16 --stage prepare
PYTHONPATH=src python -m jclosure.experiments.action_bank_v16 --stage train --limit 1
PYTHONPATH=src python -m jclosure.experiments.action_bank_v16 --stage amend
PYTHONPATH=src python -m jclosure.experiments.action_bank_v16 --stage train
PYTHONPATH=src python -m jclosure.experiments.action_bank_v16 --stage validation
PYTHONPATH=src python -m jclosure.experiments.analyze_v16 --stage analyze
PYTHONPATH=src python -m jclosure.experiments.realized_v16 --stage run
PYTHONPATH=src python -m jclosure.experiments.composition_v16 --stage run
PYTHONPATH=src python -m jclosure.experiments.sequence_reachability_v16 --stage run
PYTHONPATH=src python -m jclosure.reporting_v16
PYTHONPATH=src python scripts/build_v16_integrity.py
python scripts/build_complete_version_report.py V16
python -m pytest -q -k 'not test_v14_integrity_manifest'
```

The initial Sobol design contained a null vector and stopped before any bank Parquet was written. The original split freeze remains; amendment digest `{json.loads((ROOT/'artifacts/nonlinear_finite_causal_action_v16_bank_source_amendment_1.freeze.json').read_text())['freeze_digest']}` excludes the null vector and keeps eight nonzero deterministic points. No V1–V15 record was recomputed.

| Freeze | Digest | SHA256 |
|---|---|---|
{chr(10).join(f'| `{str(path.relative_to(ROOT))}` | `{json.loads(path.read_text()).get("freeze_digest")}` | `{sha256_file(path)}` |' for path in sorted((ROOT/'artifacts').glob('*v16*.freeze.json')))}

Train ID hash `{split['train_id_sha256']}`; validation ID hash `{split['validation_id_sha256']}`; independent bank unopened. Machine-record SHA256 values are in the per-stage JSON summaries and `v16_integrity.json`. The filtered test command excludes the pre-existing V14 test that hashes all of mutable cumulative `FINAL_REPORT.md`; no V14 result or test was altered. Source commit for the complete bundle is recorded by `scripts/build_complete_version_report.py` after the first V16 commit.
"""
    _write("EXECUTION_MANIFEST_V16.md",manifest)
    final=REPORTS/"FINAL_REPORT.md"
    existing=final.read_text(encoding="utf-8")
    if "<!-- V16_START -->" in existing:
        raise RuntimeError("V16 cumulative section already exists; refusing overwrite")
    section=f"""<!-- V16_START -->
## V16 — Nonlinear Finite Causal Action Geometry

Formal procedural status: **{decision['formal_outcome']}**. A frozen five-family 100-train/50-validation finite-action bank was measured with real BF16 writeback; reliable fraction {_number(bank['reliable_fraction'])}. Odd/even, scale, pair, requested-versus-realized, nonlinear model hierarchy, observed spectra, finite action order, primitive coverage and empirical h1 J reachability are reported in the single-file `reports/V16_COMPLETE_REPORT.md`. No k∈{{2,4,8,16,32}} model passed the required J and normalized-stack strict heldouts, so no validated compact nonlinear causal actuator dimension was identified. This does not establish V16-F/nonexistence: realized-coordinate, primitive-sequence and dynamic reachable-set coverage are limited. Nonlinear MPC and new independent confirmation were gated off. H2 remains; H3 candidate/full H3, absolute replacement and autonomous controller are not authorized. V1–V15 records remain frozen.
<!-- V16_END -->
"""
    final.write_text(existing.rstrip()+"\n\n"+section,encoding="utf-8")
    write_json_atomic(OUT/"v16_adjudication.json",{"formal_outcome":decision["formal_outcome"],"response_gate_passed":False,
                                              "independent_final":decision["independent_final"],"H2":"REMAINS",
                                              "H3_candidate":False,"absolute_replacement_authorized":False,
                                              "autonomous_controller_authorized":False,
                                              "reports":[str(p.relative_to(ROOT)) for p in sorted(REPORTS.glob("*_V16.md"))]})
    print(json.dumps({"formal_outcome":decision["formal_outcome"],"standalone_reports":len(list(REPORTS.glob("*_V16.md"))),
                      "finalist_freeze_digest":decision["freeze_digest"]},sort_keys=True))


if __name__=="__main__":main()
