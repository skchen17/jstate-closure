"""Build all V21 standalone scientific reports and append its cumulative section."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from jclosure.protocol_v21 import verify, verify_stage
from jclosure.provenance import sha256_file

ROOT=Path.cwd()
OUT=ROOT/"reports"
DATA=ROOT/"results/v21/processed"


def read(name:str)->dict:
    return json.loads((DATA/name).read_text(encoding="utf-8"))


def f(value,digits:int=4)->str:
    return "not measured" if value is None else f"{float(value):.{digits}f}"


def write(name:str,body:str)->None:
    path=OUT/name
    if path.exists():
        raise RuntimeError(f"V21 report already exists (append-only): {path}")
    path.write_text(body.strip()+"\n",encoding="utf-8")
    print(path.relative_to(ROOT),flush=True)


def _all_models()->list[dict]:
    rows=[]
    rows.extend(read("bottleneck_factorial_v21.json")["results"])
    rows.extend(read("bottleneck_factorial_extension_v21.json")["results"])
    rows.extend(read("z2_channel_normalization_correction_v21.json")["results"])
    rows.extend(read("realized_action_factorial_v21.json")["models"])
    for condition in read("neural_operator_models_v21.json")["conditions"]:
        rows.extend(condition["models"])
    return rows


def model(s:str,z:str,g:str)->dict:
    matches=[x for x in _all_models() if x["S"]==s and x["Z"]==z and x["model"]==g]
    if len(matches)!=1:
        raise RuntimeError(f"V21 model comparison not unique: {s}/{z}/{g}")
    return matches[0]


def metric(row:dict,name:str)->str:
    return f(row["metrics"][name]["stack_relative_l2"])


def _table(rows:list[dict])->str:
    lines=["| S | Z | G | seen/new state L2 | unseen direction L2 | unseen sign L2 | unseen direction+sign L2 | parameters |",
           "|---|---|---|---:|---:|---:|---:|---:|"]
    for row in rows:
        lines.append("| {S} | {Z} | {model} | {seen} | {direction} | {sign} | {both} | {parameters} |".format(
            S=row["S"],Z=row["Z"],model=row["model"],
            seen=metric(row,"seen_direction_new_state"),direction=metric(row,"unseen_direction"),
            sign=metric(row,"unseen_sign"),both=metric(row,"unseen_direction_and_sign"),
            parameters=row.get("parameter_count","kernel/closed-form")))
    return "\n".join(lines)


def build(root:Path)->dict:
    verify(root)
    verify_stage(root,"adjudication_design")
    verify_stage(root,"reporting_design")
    a=read("v21_adjudication.json")
    roles=json.loads((root/"artifacts/action_coordinate_geometry_v21_roles.freeze.json").read_text())
    states=read("state_representations_v21.json")
    raw=read("raw_state_reference_v21.json")
    actions=read("action_representation_v21.json")
    realized=read("realized_action_v21.json")
    coverage=read("action_coverage_audit_v21.json")
    scaling=read("action_data_scaling_v21.json")
    paired=read("paired_geometry_analysis_v21.json")
    bootstrap=read("paired_geometry_bootstrap_v21.json")
    z4=read("differential_oracle_ceiling_v21.json")
    z5=read("finite_second_order_oracle_v21.json")
    composition=read("scale_pair_dense_v21.json")
    test=read("v21_test_audit_amendment_1.json")
    v=paired["by_role"]["validation"]
    d=paired["by_role"]["development"]
    comp=a["comparisons"]
    flag=a["flags"]
    g2="G2_quadratic_action"
    s2z0=model("S2","Z0",g2)
    s2z1=model("S2","Z1",g2)
    s2z2=model("S2","Z2_train_floor_corrected",g2)
    s1z1=model("S1","Z1",g2)
    s4z1=model("S4","Z1",g2)

    write("ACTION_REPRESENTATION_CEILING_V21.md",f"""
# V21 practical action-coordinate ceiling

The six validation directions are distinct from the 12 training directions. All practical models use only positive train-action responses for fitting. No final-six response or raw final-action geometry was opened.

| fixed state/model | Z0 V20 descriptor | Z1 full requested | Z2 REC/Conv/KV corrected | Z3 realized BF16 Nyström |
|---|---:|---:|---:|---:|
| S2, G2 unseen direction relative L2 | {metric(s2z0,'unseen_direction')} | {metric(s2z1,'unseen_direction')} | {metric(s2z2,'unseen_direction')} | not separable for state-dependent Z3 |
| S2, G2 unseen sign relative L2 | {metric(s2z0,'unseen_sign')} | {metric(s2z1,'unseen_sign')} | {metric(s2z2,'unseen_sign')} | not separable |

Z3 was evaluated with the same frozen G4/G5 decoder class on S2 and S4. Best Z3 unseen-direction L2: **{f(comp['best_Z3_unseen_L2'])}**; Z3 is a 39-dimensional train-anchor approximation to the actual BF16 REC/Conv/KV delta, not the full raw action. Best G4/G5 S2×rich-requested-Z unseen-direction L2: **{f(comp['best_G4_G5_S2_rich_Z_unseen_L2'])}**. G2 and G5 are not a one-axis comparison; fixed-G comparisons are in the factorial record.

Z1 retains the complete 14,397-dimensional frozen V13 score coordinate through an exact linear Gram. Z2 preserves three channel blocks. The first Z2 run blew up because the train KV median was effectively zero; its original result is retained, and a train-only robust normalization was frozen before the corrected run. Z0 has 16 dimensions. Full Z1 materially improves over Z0 under fixed S2/G2 by **{f(comp['action_coordinate_gain_absolute_L2'])}** absolute L2, but **ACTION_COORDINATE_CEILING_PASS = {flag['ACTION_COORDINATE_CEILING_PASS']}** because every practical candidate fails strict direction/sign gates; no action coordinate is declared sufficient.

Machine records: `action_representation_v21.json`, `bottleneck_factorial_v21.json`, `z2_channel_normalization_correction_v21.json`, `realized_action_factorial_v21.json`.
""")

    state_rows=[]
    for s,definition in (("S0","full frozen current boundary J, 4,096-D; S0 dimension amendment"),
                         ("S1","V20 positive train-action fingerprint SVD k128"),
                         ("S2","uncompressed 12×288 positive train-action response fingerprint"),
                         ("S3","uncompressed 24×288 signed train-action response fingerprint"),
                         ("S4","93-D train-only, architecture-resolved raw-P Nyström reference; approximate raw-state kernel, not compact sufficiency")):
        row=model(s,"Z1",g2)
        dimension=93 if s=="S4" else states["dimensions"][s]
        state_rows.append(f"| {s} | {dimension} | {metric(row,'seen_direction_new_state')} | {metric(row,'unseen_direction')} | {metric(row,'unseen_sign')} | {definition} |")
    write("STATE_INFORMATION_CEILING_V21.md",f"""
# V21 state-information ceiling

The same Z1 requested-action representation and G2 decoder are held fixed throughout this table. S2/S3 are response-derived oracle contexts; S4 is a raw-state kernel approximation. None is a deployable learned compact state.

| state | feature dimension | seen/new state L2 | unseen direction L2 | unseen sign L2 | definition |
|---|---:|---:|---:|---:|---|
{chr(10).join(state_rows)}

The full S2 fingerprint improves on compact S1 by only **{f(comp['state_compression_gain_absolute_L2'])}** absolute unseen-direction L2, below the frozen 0.10 materiality rule. S4 does not outperform S2 under the same Z1/G2. This does not prove a compact state exists or that raw P is useless: S4 is a 30-anchor Nyström approximation, while the 12 training directions leave large gaps in the requested action space. S0's base config label said 128, but the frozen V20 boundary-J record actually supplies 4,096 coordinates; the pre-fit append-only dimension amendment uses all 4,096.

State feature hashes: S0–S3 `{states['scratch_sha256']}`; S4 `{raw['S4_feature_sha256']}`. Validation unseen-action responses were never inputs to S0–S4.
""")

    rows=_all_models()
    compact="\n".join(f"| {name} | {f(value['all_L2'])} | {value['qualified_states']} |" for name,value in
                       ((key,{"all_L2":x["all_rows"]["stack_relative_l2"],"qualified_states":x["reliability_qualified_states"]})
                        for key,x in composition["metrics"].items()))
    write("BOTTLENECK_FACTORIAL_V21.md",f"""
# V21 state × action × decoder factorial

The table keeps S, Z and G comparisons separated. Ridge values were chosen only on held-out development states/actions, then refit on 600 training operator states. G4/G5 architecture (width 64, 25 fixed epochs, seed 2021) was frozen in a separate design stage; the base config's exploratory 128/60 placeholders were not used. Parameters are reported for neural models. Z3 depends on P and therefore does not have an eligible separable G0–G3 kernel under this implementation.

{_table(rows)}

Additional reliability-qualified finite tests for the representative near-best S2×Z1×G2 candidate on 50 P0/Pq validation operator states:

| heldout action type | relative L2 | qualified states |
|---|---:|---:|
{compact}

The isolated action comparison S2×Z0×G2 → S2×Z1×G2 improves unseen direction by {f(comp['action_coordinate_gain_absolute_L2'])}, while S1→S2 at fixed Z1/G2 improves only {f(comp['state_compression_gain_absolute_L2'])}. Stronger G4/G5 models do not pass the strict gate. The original unnormalized Z2 numerical failure remains in `bottleneck_factorial_v21.json`; only `z2_channel_normalization_correction_v21.json` is used for scientific comparison.
""")

    covlines=[]
    for z,row in coverage["representations"].items():
        covlines.append(f"| {z} | {f(row['median_nearest_train_abs_cosine'])} | {f(row['median_train_span_relative_residual'])} |")
    write("ACTION_COVERAGE_AUDIT_V21.md",f"""
# V21 action-manifold coverage audit

| practical coordinate | median nearest train absolute cosine | median relative residual outside train span |
|---|---:|---:|
{chr(10).join(covlines)}

Z1/Z2 validation actions are far from the 12-direction train span, exceeding the frozen 0.25 residual coverage flag. Z0's small residual reflects its anchor-cosine descriptor, not true raw-action coverage. Z3's 39-dimensional projection similarly compresses raw BF16 geometry and cannot certify full raw support. Per-action values and REC/Conv/KV energy mismatch are in `results/v21/processed/action_coverage_audit_v21.json`.

Final six action responses and raw final-action geometry remain sealed; the requested final-coverage audit is withheld by the stronger independent-final rule, not marked passed.
""")

    scale_rows=[]
    for row in scaling["rows"]:
        value=row["validation_unseen_direction"]
        scale_rows.append(f"| {row['train_action_count']} | {f(value['stack_relative_l2'])} | {f(value['stack_cosine_median'])} | {f(row['selected_ridge'],5)} |")
    write("ACTION_DATA_SCALING_V21.md",f"""
# V21 nested action-data scaling

S2 full 12-action state context, Z1 and G2 were held fixed. Only decoder-supervision action count changed, so this is not a joint test of how many actions suffice to *measure* S2.

| train decoder actions | unseen-direction L2 | median cosine | train-selected ridge |
|---:|---:|---:|---:|
{chr(10).join(scale_rows)}

The 4→12 action L2 reduction is **{f(comp['action_count_4_to_12_L2_drop'])}**, with a Z1 validation train-span residual of **{f(comp['Z1_validation_span_residual_median'])}**. The frozen action-data-limited diagnostic is **{flag['ACTION_DATA_LIMITED']}**. Counts 16/18 are not available within the sealed V20 12-train/6-validation split without reassigning validation labels, so they were not used as training actions.
""")

    write("PAIRED_JVP_FINITE_OPERATOR_V21.md",f"""
# V21 paired exact differential and finite operator panel

Exactly matched P0/Pq persistent states were measured on **50 development** and **25 disjoint validation** bases, balanced across five families. All matrices use 64 frozen probes; one train-calibrated BF16-unreliable probe is retained in raw audit but excluded from reliability-qualified 63-column spectra. First 18 probes are the V20 12 train plus 6 validation actions; none of the final six responses was opened. The same frozen h1 288-D target and teacher token are used for exact JVP and signed central finite response.

| role | bases | JVP median r90/r95/r99 | finite median r90/r95/r99 | JVP P0/Pq median angle | finite P0/Pq median angle |
|---|---:|---|---|---:|---:|
| development | {d['base_states']} | {' / '.join(f(x,1) for x in d['JVP_median_r90_r95_r99'])} | {' / '.join(f(x,1) for x in d['finite_median_r90_r95_r99'])} | {f(d['JVP_P0_Pq_median_principal_angle'])}° | {f(d['finite_P0_Pq_median_principal_angle'])}° |
| validation | {v['base_states']} | {' / '.join(f(x,1) for x in v['JVP_median_r90_r95_r99'])} | {' / '.join(f(x,1) for x in v['finite_median_r90_r95_r99'])} | {f(v['JVP_P0_Pq_median_principal_angle'])}° | {f(v['finite_P0_Pq_median_principal_angle'])}° |

Exact JVP uses forward-mode `torch.func.jvp` for all 75 final-panel bases. The original reverse-over-reverse 17-base development run is preserved separately, never mixed. Five-probe same-state comparison found minimum column cosine {f(read('forward_ad_comparison_v21.json')['rows'][0]['cosine'],6)} and maximum relative difference below 0.01; see `forward_ad_comparison_v21.json` and the append-only method amendment. Raw matrices remain outside Git; `paired_jvp_finite_operator_v21.parquet` indexes their hashes.
""")

    val_ci=bootstrap["by_role"]["validation"]["CI95"]
    fam="\n".join(f"| {family} | {f(rho)} |" for family,rho in v["family_rotation_Spearman_rho"].items())
    write("DIFFERENTIAL_FINITE_ALIGNMENT_V21.md",f"""
# V21 differential–finite alignment on identical P0/Pq states

Validation Spearman correlation between per-pair JVP rotation and finite rotation is **{f(v['rotation_Spearman_rho'])}** (family-stratified base bootstrap 95% CI **[{f(val_ci['rotation_Spearman_rho'][0])}, {f(val_ci['rotation_Spearman_rho'][1])}]**); Pearson r **{f(v['rotation_Pearson_r'])}**. Same-state P0/Pq JVP–finite median dominant-subspace overlaps are **{f(v['JVP_finite_P0_median_subspace_overlap'])}/{f(v['JVP_finite_Pq_median_subspace_overlap'])}**, and P0 response-column median cosine is **{f(v['JVP_finite_P0_median_column_cosine'])}**.

| family | validation within-family Spearman rotation correlation |
|---|---:|
{fam}

Strong shared-geometry support under the corrected frozen **all five families positive** gate is **{paired['shared_geometry_strong_support']}**. Similar ranks alone are not treated as operator equality. The half/full finite-action scale and BF16 quantization remain potential sources of differential–finite divergence. Detailed per-base angles and overlaps: `paired_geometry_analysis_v21.parquet`; bootstrap: `paired_geometry_bootstrap_v21.json`.
""")

    write("OPERATOR_ROTATION_DECOMPOSITION_V21.md",f"""
# V21 rotation versus gain/spectrum/rank decomposition

For each identical P0/Pq pair the operator change was decomposed via output-space orthogonal Procrustes, gain and singular spectra. The original draft Procrustes formula mistakenly used `nuclear_norm(AᵀB)`; a pre-analysis CPU invariant caught this. The append-only correction uses the equivalent compact SVD of `B Aᵀ`, with no raw matrix change.

| validation finite P0→Pq statistic | median |
|---|---:|
| pre-alignment relative error | {f(v['finite_median_pre_alignment_error'])} |
| post-orthogonal-Procrustes relative error | {f(v['finite_median_post_Procrustes_error'])} |
| Pq/P0 Frobenius gain | {f(v['finite_median_gain_ratio'])} |
| singular-spectrum JS divergence | {f(v['finite_median_spectral_JS'],6)} |
| r95 difference | {f(v['finite_median_r95_difference'],1)} |
| JVP P0/Pq median principal angle | {f(v['JVP_P0_Pq_median_principal_angle'])}° |
| finite P0/Pq median principal angle | {f(v['finite_P0_Pq_median_principal_angle'])}° |

An error reduction after Procrustes is compatible with rotation, but not by itself causal proof that one operator is exactly a rotated copy of the other. Family-split association and JVP–finite overlap determine the stronger shared-mechanism claim.
""")

    direct=z5["direct_oracle"]
    z4_best=min(x["metrics"]["unseen_direction"]["stack_relative_l2"] for x in z4["state_conditioned_models"])
    z5_best=min(x["positive_unseen_direction"]["stack_relative_l2"] for x in z5["S2_state_conditioned_models"])
    write("HIGHER_ORDER_ACTION_GEOMETRY_V21.md",f"""
# V21 exact-JVP and finite second-order oracle ceilings

Z4 is the exact same-state, same-action JVP response on the 288-D normalized target; it is a target-side diagnostic oracle, not a deployable action coordinate. Z5 adds the finite central second difference measured at **half** the target action amplitude, so the full-amplitude response is never used as a Z5 feature. Z5 is not an exact Hessian.

| oracle diagnostic | validation unseen-direction positive L2 | negative L2 |
|---|---:|---:|
| Z4 direct JVP | {f(direct['JVP_only']['positive_unseen_direction']['stack_relative_l2'])} | {f(direct['JVP_only']['negative_unseen_direction']['stack_relative_l2'])} |
| Z5 direct Taylor D+Q/2 | {f(direct['direct_Taylor']['positive_unseen_direction']['stack_relative_l2'])} | {f(direct['direct_Taylor']['negative_unseen_direction']['stack_relative_l2'])} |
| Z5 train-fitted two-scalar map | {f(direct['train_scalar_pair']['positive_unseen_direction']['stack_relative_l2'])} | {f(direct['train_scalar_pair']['negative_unseen_direction']['stack_relative_l2'])} |
| Z4 best state-conditioned G4/G5 | {f(z4_best)} | see machine record |
| Z5 S2 best state-conditioned G4/G5 | {f(z5_best)} | see machine record |

Half-step actuator reliability: development **{f(z5['half_step_reliable_train_fraction'])}**, validation **{f(z5['half_step_reliable_validation_fraction'])}**. Any apparent oracle improvement does not establish a practical action representation or justify opening the final six. The same-action derivative/second-difference feature must never be mixed with Z0–Z3 deployment claims.
""")

    write("COMPACT_OPERATOR_REOPEN_DECISION_V21.md",f"""
# V21 compact-operator reopening and final-opening decision

`ACTION_COORDINATE_CEILING_PASS = {flag['ACTION_COORDINATE_CEILING_PASS']}`. Practical direction/sign gate pass count: **{a['practical_direction_sign_gate_count']}** of **{a['practical_candidate_count']}** evaluated rich-action candidates. The strict scale/pair/dense representative candidate also fails. Therefore `COMPACT_OPERATOR_SEARCH_REOPENED = {flag['COMPACT_OPERATOR_SEARCH_REOPENED']}`; no k=2…128 rerun is authorized.

`RAW_TO_OPERATOR_ENCODER_AUTHORIZED = {flag['RAW_TO_OPERATOR_ENCODER_AUTHORIZED']}` and `DYNAMIC_STATE_SEARCH_AUTHORIZED = {flag['DYNAMIC_STATE_SEARCH_AUTHORIZED']}`. No raw P→C encoder or `(J,C,a)→(J′,C′)` transition model was trained. The final six action responses are **SEALED / UNOPENED**, not failed empirical final tests. No independent finalist exists. H2 remains; H3 and autonomous control remain unauthorized.

Formal result: **{' + '.join(a['formal_outcomes'])}**. “Not identified under the tested action representation/model family” is deliberately narrower than “compact state does not exist.”
""")

    write("STRICT_INTERFACE_AUDIT_V21.md",f"""
# V21 strict scientific interface audit

| interface | status | evidence/caveat |
|---|---|
| V1–V20 frozen inputs | unchanged except cumulative FINAL_REPORT append | V21 parent commit and input hashes; integrity index |
| final six response labels | sealed | all V21 result flags false; no finalist gate |
| final six raw action geometry | sealed | coverage audit limited to validation six |
| state S versus action Z versus decoder G | separately varied | factorial and extension records; Z3's state dependence disclosed |
| practical Z0–Z3 versus oracle Z4/Z5 | strictly separated | oracle reports and adjudication ignore Z4/Z5 for practical ceiling |
| raw P→C | not trained | S4 is an input-side raw-state reference only |
| transition/controller | not trained | V20 `V21_DYNAMIC_STATE_SEARCH_AUTHORIZED=false` |
| S0 source dimension | append-only correction before model fitting | actual V20 full J is 4,096-D, not config label 128 |
| Z2 numerical scale | append-only correction | initial blow-up preserved, train-only robust floor before corrected run |
| Procrustes/family gate | append-only pre-analysis corrections | CPU invariant and base all-five-family rule restored |
| paired exact-JVP method | one homogeneous final panel | forward-mode full rerun; older 17-base matrices audit-only |
| G4/G5 capacity | constrained | frozen derived 64-wide/25-epoch design, not exhaustive model-class search |
| S4/Z3 raw coverage | approximate | 30-anchor state and 12-anchor realized-action Nyström references |

No H3, physical cache replacement, mathematical nonexistence, or autonomous control claim is made.
""")

    base=verify(root)
    freezes=sorted((root/"artifacts").glob("action_coordinate_geometry_v21*.freeze.json"))
    freeze_rows=[]
    for path in freezes:
        payload=json.loads(path.read_text())
        freeze_rows.append(f"| `{path.relative_to(root)}` | `{payload['freeze_digest']}` | `{sha256_file(path)}` |")
    commands=[
        "PYTHONPATH=src python -m jclosure.protocol_v21 freeze",
        "PYTHONPATH=src python -m jclosure.experiments.bank_v21",
        "PYTHONPATH=src python -m jclosure.experiments.action_representation_v21",
        "PYTHONPATH=src python -m jclosure.experiments.state_representations_v21",
        "PYTHONPATH=src python -m jclosure.experiments.probe_calibration_v21 calibrate",
        "PYTHONPATH=src python -m jclosure.experiments.factorial_v21 run",
        "PYTHONPATH=src python -m jclosure.experiments.z2_correction_v21 run",
        "PYTHONPATH=src python -m jclosure.experiments.action_scaling_v21 run",
        "PYTHONPATH=src python -m jclosure.experiments.neural_models_v21 run --state S2 --action Z1",
        "PYTHONPATH=src python -m jclosure.experiments.raw_state_reference_v21 run",
        "PYTHONPATH=src python -m jclosure.experiments.realized_action_v21 run",
        "PYTHONPATH=src python -m jclosure.experiments.factorial_extension_v21 run",
        "PYTHONPATH=src python -m jclosure.experiments.realized_factorial_v21 run",
        "PYTHONPATH=src python -m jclosure.experiments.coverage_audit_v21 run",
        "PYTHONPATH=src python -m jclosure.experiments.composition_v21 measure",
        "PYTHONPATH=src python -m jclosure.experiments.composition_v21 evaluate",
        "CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src python -m jclosure.experiments.second_order_v21 measure --role development",
        "CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src python -m jclosure.experiments.second_order_v21 measure --role validation",
        "CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src python -m jclosure.experiments.paired_forward_v21 run --role development",
        "CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src python -m jclosure.experiments.paired_forward_v21 run --role validation",
        "PYTHONPATH=src python -m jclosure.experiments.paired_forward_v21 summarize",
        "PYTHONPATH=src python -m jclosure.experiments.paired_forward_v21 analyze",
        "PYTHONPATH=src python -m jclosure.experiments.paired_forward_v21 oracle",
        "PYTHONPATH=src python -m jclosure.experiments.paired_forward_v21 second",
        "PYTHONPATH=src python -m jclosure.experiments.geometry_bootstrap_v21 run",
        "PYTHONPATH=src python scripts/adjudicate_v21.py run",
        "PYTHONPATH=src python scripts/test_audit_v21.py",
        "PYTHONPATH=src python scripts/build_v21_reports.py",
        "PYTHONPATH=src python scripts/build_v21_integrity.py",
        "python scripts/build_complete_version_report.py V21",
    ]
    write("EXECUTION_MANIFEST_V21.md",f"""
# V21 execution manifest

Parent Git commit: `{base['parent_commit']}`. V21 protocol digest: `{base['freeze_digest']}`; config SHA: `{base['config_sha256']}`. Train/validation state hashes: `{roles['V20_train_base_id_sha256']}` / `{roles['V20_validation_base_id_sha256']}`; V21 JVP development/validation base hashes: `{roles['jvp_development_id_sha256']}` / `{roles['jvp_validation_id_sha256']}`. V20 train/validation/final action hashes: `{json.dumps(json.loads((root/'artifacts/compact_causal_response_operator_v20_actions.freeze.json').read_text())['action_hashes'],sort_keys=True)}`. JVP probe hash: `{roles['jvp_probe_sha256']}`. S0–S3, S4 and Z3 feature hashes are in their standalone reports.

| freeze manifest | digest | file SHA256 |
|---|---|---|
{chr(10).join(freeze_rows)}

Primary commands below use the same module/stage arguments actually executed from the repository root; GPU measurements used the frozen `CUDA_VISIBLE_DEVICES=1`, `HF_HOME=/data/CSK/J-space-project/.hf-cache` and `PYTHONPATH=src` environment. Ancillary `prepare`/freeze commands and initial interrupted GPU-0/slow-JVP pilots are preserved in freeze records and method amendments.

```bash
{chr(10).join(commands)}
```

Raw paired matrices, full raw-state features, realized-action features, neural weights and half-step response banks remain under `/data/CSK/J-space-project/v21-action-geometry-work`, outside Git. They are indexed by hashes in processed machine records. Test audit: **{test['runs']['all']['passed']} passed, {test['runs']['all']['failed']} inherited old cumulative-report hash failures**; V21-only **{test['runs']['v21_only']['passed']} passed**. An append-only audit amendment corrects an initial summary-regex error without altering test output. No V1–V20 frozen manifest was rewritten.
""")

    questions=[
        f"1. **Primary bottleneck?** Action representation/coverage is the strongest observed practical limitation: fixed S2/G2 Z0→Z1 gains {f(comp['action_coordinate_gain_absolute_L2'])} L2, but no practical coordinate passes; model/finite geometry remain possible contributors.",
        f"2. **Does full Φ_train beat k128?** No material gain: {f(comp['state_compression_gain_absolute_L2'])} absolute L2 (<0.10 gate).",
        f"3. **Full requested coordinate vs V20 descriptor?** Yes, unseen L2 {f(z0)}→{f(z1)} under fixed S2/G2, still fails 0.30.",
        f"4. **Architecture resolved Z2?** Corrected Z2 {f(z2)} versus Z1 {f(z1)}: small further gain, not a pass.",
        f"5. **Realized BF16 Z3?** Best state-dependent Z3 {f(comp['best_Z3_unseen_L2'])}; not a pass; 39D Nyström caveat.",
        f"6. **Coverage sufficient?** No; Z1 train-span residual median {f(comp['Z1_validation_span_residual_median'])}.",
        f"7. **More actions help?** 4/8/12 decoder-action L2 {', '.join(f(x['validation_unseen_direction']['stack_relative_l2']) for x in scaling['rows'])}.",
        f"8. **Raw-state ceiling?** S4×Z1/G2 {f(s4)} versus S2 {f(s2)}; no gain for this 30-anchor raw kernel.",
        f"9. **Exact JVP predict finite unseen actions?** Direct Z4 L2 {f(z4['direct_oracle']['direct_identity']['metrics']['unseen_direction']['stack_relative_l2'])}; best state-conditioned {f(comp['Z4_best_state_conditioned_unseen_L2'])}, oracle only.",
        f"10. **Second order help?** Z5 direct Taylor {f(comp['Z5_direct_Taylor_unseen_L2'])} vs JVP-only {f(comp['Z4_direct_JVP_unseen_L2'])}; half-step oracle only, reliability {f(z5['half_step_reliable_validation_fraction'])}.",
        f"11–12. **Paired r90/r95/r99?** JVP {'/'.join(f(x,1) for x in v['JVP_median_r90_r95_r99'])}; finite {'/'.join(f(x,1) for x in v['finite_median_r90_r95_r99'])} (validation P0).",
        f"13–14. **Pq rotation?** JVP {f(v['JVP_P0_Pq_median_principal_angle'])}°, finite {f(v['finite_P0_Pq_median_principal_angle'])}°.",
        f"15. **Rotations correlated?** Validation Spearman {f(v['rotation_Spearman_rho'])}, CI [{f(val_ci['rotation_Spearman_rho'][0])}, {f(val_ci['rotation_Spearman_rho'][1])}].",
        f"16. **Rotation/gain/spectrum/rank?** Finite pre/post Procrustes {f(v['finite_median_pre_alignment_error'])}/{f(v['finite_median_post_Procrustes_error'])}, gain {f(v['finite_median_gain_ratio'])}, JS {f(v['finite_median_spectral_JS'],6)}, Δr95 {f(v['finite_median_r95_difference'],1)}.",
        f"17. **Same V13/V20 mechanism?** Strong paired support = {paired['shared_geometry_strong_support']}; do not infer equality from rank alone.",
        f"18. **Corrected practical action coordinate identified?** {flag['ACTION_COORDINATE_CEILING_PASS']}.",
        f"19. **Compact search reopened?** {flag['COMPACT_OPERATOR_SEARCH_REOPENED']}.",
        f"20. **Raw P→C authorized?** {flag['RAW_TO_OPERATOR_ENCODER_AUTHORIZED']}.",
        f"21–23. **H2/H3/dynamic search?** H2 remains {flag['H2_REMAINS']}; H3 {flag['H3_AUTHORIZED']}; dynamic search {flag['DYNAMIC_STATE_SEARCH_AUTHORIZED']}.",
    ]
    write("V21_SCIENTIFIC_ANSWERS_V21.md",f"""
# V21 scientific answers

{chr(10).join(questions)}

Formal result: **{' + '.join(a['formal_outcomes'])}**. This is a tested-regime result, not proof of mathematical nonexistence or a license for H3/autonomous control. Final six remain unopened.
""")

    final=root/"reports/FINAL_REPORT.md"
    before=final.read_bytes()
    if b"<!-- V21_START -->" in before:
        raise RuntimeError("V21 cumulative report section already exists")
    paragraph=f"""
<!-- V21_START -->
## V21 — Action Coordinate Ceiling and Paired Causal Operator Geometry

Formal outcome: **{' + '.join(a['formal_outcomes'])}**. Holding S2/G2 fixed, V20 Z0→full requested Z1 reduces unseen-direction relative L2 from **{f(z0)}** to **{f(z1)}**, but unseen sign is **{metric(s2z1,'unseen_sign')}** and no practical candidate passes the frozen cross-action gate. S1 k128→full S2 gains only **{f(comp['state_compression_gain_absolute_L2'])}** L2 under fixed Z1/G2. The Z1 validation train-span residual is **{f(comp['Z1_validation_span_residual_median'])}**, and 4→12 training actions improve L2 **{f(comp['action_count_4_to_12_L2_drop'])}**. Thus current action coverage remains a material limit; this does not establish compact-state nonexistence.

On 50 development + 25 disjoint validation P0/Pq bases, matched 64-probe exact-JVP and central finite operators have validation median r95 **{f(v['JVP_median_r90_r95_r99'][1],1)}/{f(v['finite_median_r90_r95_r99'][1],1)}**, P0/Pq median rotations **{f(v['JVP_P0_Pq_median_principal_angle'])}°/{f(v['finite_P0_Pq_median_principal_angle'])}°**, rotation Spearman **{f(v['rotation_Spearman_rho'])}**, and P0 JVP–finite subspace overlap **{f(v['JVP_finite_P0_median_subspace_overlap'])}**. Strong shared-geometry gate: **{paired['shared_geometry_strong_support']}**. Z4/Z5 are same-action diagnostic oracles, not deployable coordinates.

`COMPACT_OPERATOR_SEARCH_REOPENED={flag['COMPACT_OPERATOR_SEARCH_REOPENED']}`; `RAW_TO_OPERATOR_ENCODER_AUTHORIZED={flag['RAW_TO_OPERATOR_ENCODER_AUTHORIZED']}`; `DYNAMIC_STATE_SEARCH_AUTHORIZED={flag['DYNAMIC_STATE_SEARCH_AUTHORIZED']}`. Final six action responses remain sealed. H2 remains; H3, complete replacement and autonomous control are not authorized. Historical V1–V20 frozen inputs are unchanged. See `reports/V21_COMPLETE_REPORT.md` for every standalone report, amendment, machine-record hash, and limitation.
<!-- V21_END -->
""".strip()+"\n"
    with final.open("ab") as handle:
        if not before.endswith(b"\n"):
            handle.write(b"\n")
        handle.write(b"\n"+paragraph.encode("utf-8"))
    return {"standalone_reports":13,"final_report_sha256":sha256_file(final),
            "formal_outcomes":a["formal_outcomes"]}


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--root",type=Path,default=Path.cwd())
    args=parser.parse_args()
    ROOT=args.root.resolve()
    OUT=ROOT/"reports"
    DATA=ROOT/"results/v21/processed"
    print(json.dumps(build(ROOT),indent=2))
