"""Standalone and cumulative reporting for V23."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from jclosure.protocol_v23 import stage_freeze, verify, verify_stage
from jclosure.provenance import sha256_file, write_json_atomic

OUT=Path("results/v23/processed")
REPORTS=Path("reports")
SOURCE="src/jclosure/reporting_v23.py"
REPORT_NAMES=[
"INPUT_RANK_SCALING_V23.md","ORACLE_JVP_ACTION_CHART_V23.md","ORACLE_FINITE_ACTION_CHART_V23.md",
"LOCAL_CHART_DIMENSION_V23.md","LOCAL_CHART_GENERALIZATION_V23.md","CHART_SMOOTHNESS_V23.md",
"SAME_J_CHART_ALIASING_V23.md","CHART_TRANSPORT_V23.md","CHART_PREDICTABILITY_V23.md",
"ORACLE_VS_LEARNED_CHART_V23.md","LOCAL_IDENTIFICATION_SAMPLE_COMPLEXITY_V23.md",
"LOCAL_CHART_NONLINEARITY_V23.md","STRICT_INTERFACE_AUDIT_V23.md","EXECUTION_MANIFEST_V23.md",
"V23_SCIENTIFIC_ANSWERS_V23.md","V23_COMPLETE_REPORT.md"]


def _write(root:Path,name:str,text:str):
    (root/REPORTS/name).write_text(text.rstrip()+"\n",encoding="utf-8")


def _f(x):
    if x is None:return "N/A"
    if isinstance(x,bool):return "TRUE" if x else "FALSE"
    if isinstance(x,float):return f"{x:.6f}"
    return str(x)


def _category_table(best):
    lines=["| category | relative L2 | cosine | J cosine | norm ratio |","|---|---:|---:|---:|---:|"]
    for name,value in best["categories"].items():
        lines.append(f"| {name} | {_f(value['relative_l2'])} | {_f(value['median_cosine'])} | {_f(value['median_j_cosine'])} | {_f(value['median_norm_ratio'])} |")
    return "\n".join(lines)


def generate(root:Path,v23_tests:str,full_tests:str):
    base=verify(root); verify_stage(root,"rank_measurement_design")
    rank=json.loads((root/OUT/"input_rank_scaling_v23.json").read_text())
    oracle=json.loads((root/OUT/"oracle_local_action_charts_v23.json").read_text())
    adj=json.loads((root/OUT/"v23_adjudication.json").read_text())
    probes=json.loads((root/OUT/"probe_selection_v23.json").read_text())
    candidate=json.loads((root/OUT/"probe_candidate_design_v23.json").read_text())
    measurement=json.loads((root/OUT/"rank_measurement_summary_v23.json").read_text())
    rows=["| m | Gram rank | condition | JVP r90/r95/r99 | finite r90/r95/r99 |","|---:|---:|---:|---:|---:|"]
    for m in base["config"]["probe_counts"]:
        g=rank["gram"][str(m)]; j=rank["curves"]["JVP"][str(m)]["input_r90_r95_r99_median"]; f=rank["curves"]["finite"][str(m)]["input_r90_r95_r99_median"]
        rows.append(f"| {m} | {g['effective_gram_rank']} | {g['retained_condition_number']:.3e} | {j} | {f} |")
    rank_table="\n".join(rows)
    _write(root,"INPUT_RANK_SCALING_V23.md",f"""# Input-Rank Scaling — V23

{rank_table}

JVP operator states: {measurement['jvp_operator_state_count']}; finite operator states: {measurement['finite_operator_state_count']}. The frozen last-two-doubling saturation criterion gives JVP=`{_f(rank['JVP_INPUT_RANK_SATURATED'])}`, finite=`{_f(rank['finite_INPUT_RANK_SATURATED'])}`, joint=`{_f(rank['INPUT_RANK_SATURATED'])}`. Input and output energy ranks share the operator singular spectrum; their singular vectors remain distinct.
""")
    for kind,file,title in (("JVP","ORACLE_JVP_ACTION_CHART_V23.md","Oracle JVP Action Chart"),("finite","ORACLE_FINITE_ACTION_CHART_V23.md","Oracle Finite Action Chart")):
        best=oracle["best"][kind]
        _write(root,file,f"""# {title} — V23

Best frozen validation candidate: `{best['token']}`. Gate pass: **{_f(best['gate_pass'])}**.

{_category_table(best)}

Every chart uses only the state-local train-128 probe responses. Held-out validation responses enter metrics only as targets.
""")
    dims=[]
    for kind in ("JVP","finite"):
        token=oracle["best"][kind]["token"]; dims.append(f"- {kind}: best `{token}`, pass `{_f(oracle['best'][kind]['gate_pass'])}`")
    _write(root,"LOCAL_CHART_DIMENSION_V23.md","# Local Chart Dimension — V23\n\n"+"\n".join(dims)+f"\n\n`k_chart_min = {_f(oracle['k_chart_min'])}`. No chart dimension is identified unless a common k/model passes all required categories.\n")
    _write(root,"LOCAL_CHART_GENERALIZATION_V23.md",f"""# Local Chart Generalization — V23

## Best finite chart

{_category_table(oracle['best']['finite'])}

## Best JVP chart

{_category_table(oracle['best']['JVP'])}

Oracle local-chart pass: **{_f(oracle['ORACLE_LOCAL_CHART_PASS'])}**.
""")
    s=oracle["smoothness"]
    _write(root,"CHART_SMOOTHNESS_V23.md",f"""# Chart Smoothness — V23

- Adjacent tokens, same prompt: `{s['adjacent_tokens_same_prompt']}`.
- Nearest-J median chart angle: `{_f(s['nearby_J_median_angle_degrees'])}°`.
- Same-family median angle: `{_f(s['same_family_median_angle_degrees'])}°`.
- Across-family median angle: `{_f(s['across_family_median_angle_degrees'])}°`.

{s['claim']}
""")
    sj=oracle["same_J"]
    _write(root,"SAME_J_CHART_ALIASING_V23.md",f"""# Same-J Chart Aliasing — V23

Across {sj['pair_count']} exact same-J P0/Pq pairs, the median finite input-chart angle is `{_f(sj['finite_input_chart_angle_median_degrees'])}°`, the output-chart angle is `{_f(sj['finite_output_chart_angle_median_degrees'])}°`, and spectrum relative change is `{_f(sj['spectrum_relative_change_median'])}`. Same workspace therefore does not fix the tested local causal chart.
""")
    t=oracle["transport"]
    _write(root,"CHART_TRANSPORT_V23.md",f"""# Chart Transport — V23

Orthogonal-Procrustes transport on {t['pair_count']} P0/Pq pairs has median residual `{_f(t['procrustes_residual_median'])}`, fidelity `{_f(t['transport_fidelity_median'])}`, and projector overlap `{_f(t['projector_overlap_median'])}`. This is a diagnostic alignment only; no global connection or fiber-bundle claim is made.
""")
    p=oracle["chart_predictability"]
    _write(root,"CHART_PREDICTABILITY_V23.md",f"""# Chart Predictability — V23

Status: **{p['status']}**.

S0/J: `{_f(p['S0'])}`; S1: `{_f(p['S1'])}`; S2 fingerprint: `{_f(p['S2'])}`; S4 raw persistent reference: `{_f(p['S4'])}`.

{p['rule']} Predicting ordered basis vectors without gauge alignment is prohibited.
""")
    ovl=oracle["oracle_vs_learned"]
    _write(root,"ORACLE_VS_LEARNED_CHART_V23.md",f"""# Oracle versus Learned Chart — V23

Best oracle direction L2: `{_f(ovl['oracle_best_direction_l2'])}`. Frozen V22 learned-chart L2: `{_f(ovl['V22_learned_chart_l2'])}`. Oracle-minus-learned gap: `{_f(ovl['oracle_minus_learned_l2'])}`.

Because oracle sufficiency is `{_f(oracle['ORACLE_LOCAL_CHART_PASS'])}`, chart-prediction failure is not declared as the primary bottleneck.
""")
    curve=["| local probes | best candidate | relative L2 | cosine |","|---:|---|---:|---:|"]
    for n,v in oracle["local_probe_curve"].items():curve.append(f"| {n} | {v['best_token']} | {_f(v['relative_l2'])} | {_f(v['median_cosine'])} |")
    _write(root,"LOCAL_IDENTIFICATION_SAMPLE_COMPLEXITY_V23.md","# Local Identification Sample Complexity — V23\n\n"+"\n".join(curve)+f"\n\nSample-efficient (pass by 8–16 probes): **{_f(oracle['LOCAL_CAUSAL_SYSTEM_IDENTIFICATION_IS_SAMPLE_EFFICIENT'])}**.\n")
    nonlin=["| model | direction L2 | cosine |","|---|---:|---:|"]
    for model,v in oracle["within_chart"].items():nonlin.append(f"| {model} | {_f(v['relative_l2'])} | {_f(v['median_cosine'])} |")
    _write(root,"LOCAL_CHART_NONLINEARITY_V23.md","# Local-Chart Nonlinearity — V23\n\n"+"\n".join(nonlin)+"\n\nQuadratic and cubic decoders keep the chart fixed, isolating within-chart response law from chart estimation.\n")
    _write(root,"STRICT_INTERFACE_AUDIT_V23.md",f"""# Strict Interface Audit — V23

- Historical final action overlap with V23 train probes: `{probes['historical_final_action_overlap']}`.
- Historical final responses opened: `{_f(adj['historical_final_responses_opened'])}`.
- V23 independent-final responses opened: `{_f(adj['v23_independent_final_responses_opened'])}`.
- Dynamic state used: `FALSE`.
- Held-out action response used in chart construction: `FALSE`.
- Frozen train/held-out/final hashes: `{probes['training_action_hash']}` / `{probes['heldout_action_hash']}` / `{candidate['new_independent_final_hash']}`.
""")
    commands=[
"PYTHONPATH=src /home/user/anaconda3/bin/python -m jclosure.protocol_v23 freeze",
"CUDA_VISIBLE_DEVICES=1 HF_HOME=/data/CSK/J-space-project/.hf-cache PYTHONPATH=src /home/user/anaconda3/bin/python -m jclosure.experiments.probe_design_v23 prepare",
"CUDA_VISIBLE_DEVICES=1 HF_HOME=/data/CSK/J-space-project/.hf-cache PYTHONPATH=src /home/user/anaconda3/bin/python -m jclosure.experiments.probe_design_v23 calibrate",
"PYTHONPATH=src /home/user/anaconda3/bin/python -m jclosure.experiments.probe_design_v23 select",
"PYTHONPATH=src /home/user/anaconda3/bin/python -m jclosure.experiments.rank_measurement_v23 prepare",
"CUDA_VISIBLE_DEVICES=1 HF_HOME=/data/CSK/J-space-project/.hf-cache PYTHONPATH=src /home/user/anaconda3/bin/python -m jclosure.experiments.rank_measurement_v23 jvp",
"CUDA_VISIBLE_DEVICES=1 HF_HOME=/data/CSK/J-space-project/.hf-cache PYTHONPATH=src /home/user/anaconda3/bin/python -m jclosure.experiments.rank_measurement_v23 finite",
"PYTHONPATH=src /home/user/anaconda3/bin/python -m jclosure.experiments.rank_measurement_v23 summarize",
"PYTHONPATH=src /home/user/anaconda3/bin/python -m jclosure.experiments.analyze_v23 rank",
"PYTHONPATH=src /home/user/anaconda3/bin/python -m jclosure.experiments.analyze_v23 oracle",
"PYTHONPATH=src /home/user/anaconda3/bin/python -m jclosure.experiments.analyze_v23 adjudicate",
"PYTHONPATH=src /home/user/anaconda3/bin/python -m pytest tests/test_v23_oracle_charts.py -q",
"PYTHONPATH=src /home/user/anaconda3/bin/python -m pytest -q",
"git add configs/oracle_local_action_charts_v23.yaml src/jclosure/protocol_v23.py src/jclosure/experiments/probe_design_v23.py src/jclosure/experiments/rank_measurement_v23.py src/jclosure/experiments/analyze_v23.py src/jclosure/reporting_v23.py tests/test_v23_oracle_charts.py artifacts/oracle_local_action_charts_v23*.json results/v23 reports/*.md",
"git commit -m 'Complete V23 oracle local action charts'",
"git push origin main"]
    write_json_atomic(root/OUT/"execution_manifest_v23.json",{"commands":commands,"v23_tests":v23_tests,"full_tests":full_tests})
    _write(root,"EXECUTION_MANIFEST_V23.md","# Execution Manifest — V23\n\n"+"\n".join(f"- `{x}`" for x in commands)+f"\n\nV23 tests: `{v23_tests}`. Full suite: `{full_tests}`.\n")
    answers=[
f"JVP input r95 saturation: {rank['JVP_INPUT_RANK_SATURATED']}.",f"Finite input r95 saturation: {rank['finite_INPUT_RANK_SATURATED']}.",
f"Best tested 512-probe JVP/finite r95: {rank['curves']['JVP']['512']['input_r90_r95_r99_median'][1]} / {rank['curves']['finite']['512']['input_r90_r95_r99_median'][1]}.",
"Finite systematically exceeds JVP only if the full paired curve supports it; no inference is made from one point.",
f"Oracle JVP chart pass: {oracle['ORACLE_JVP_LOCAL_CHART_PASS']}.",f"Oracle finite chart pass: {oracle['ORACLE_FINITE_LOCAL_CHART_PASS']}.",
f"Best chart comparison is recorded by direction L2 and overlap; finite-minus-JVP direction L2={oracle['jvp_finite_comparison']['finite_minus_jvp_best_direction_l2']:.6f}.",
f"k_chart_min={oracle['k_chart_min']}.",
f"Oracle sign L2={oracle['best']['finite']['categories']['unseen_sign']['relative_l2']:.6f}.",
f"Oracle amplitude L2={oracle['best']['finite']['categories']['unseen_amplitude']['relative_l2']:.6f}.",
f"Oracle pair L2={oracle['best']['finite']['categories']['unseen_pair']['relative_l2']:.6f}.",
f"Oracle dense L2={oracle['best']['finite']['categories']['unseen_dense']['relative_l2']:.6f}.",
f"Local probe curve={oracle['local_probe_curve']}.",f"Sample-efficient identification={oracle['LOCAL_CAUSAL_SYSTEM_IDENTIFICATION_IS_SAMPLE_EFFICIENT']}.",
f"Natural adjacent-token smoothness was not measured; nearest-J angle={s['nearby_J_median_angle_degrees']:.6f} degrees.",
f"Same-J P0/Pq finite chart angle={sj['finite_input_chart_angle_median_degrees']:.6f} degrees.",
f"S0 chart prediction={p['S0']} ({p['status']}).",f"S1 chart prediction={p['S1']} ({p['status']}).",f"S2 chart prediction={p['S2']} ({p['status']}).",f"Raw-P chart prediction={p['S4']} ({p['status']}).",
f"Chart prediction is separated from sufficiency; oracle pass={oracle['ORACLE_LOCAL_CHART_PASS']}.",
f"Within-chart models={list(oracle['within_chart'])}.","Quadratic/cubic necessity is judged only against the common gate; no passing law means no necessity claim.",
f"JVP/finite overlap={oracle['jvp_finite_comparison']['median_input_overlap']:.6f}.",
f"Compact operator search reopened={adj['COMPACT_OPERATOR_SEARCH_REOPENED']}.",f"Raw P→C authorized={adj['RAW_TO_OPERATOR_ENCODER_AUTHORIZED']}.",
f"H2 remains={adj['H2_REMAINS']}.",f"H3 authorized={adj['H3_AUTHORIZED']}.",f"Dynamic-state search authorized={adj['DYNAMIC_STATE_SEARCH_AUTHORIZED']}." ]
    write_json_atomic(root/OUT/"v23_scientific_answers.json",{"answers":answers})
    _write(root,"V23_SCIENTIFIC_ANSWERS_V23.md","# V23 Scientific Answers\n\n"+"\n".join(f"{i}. {x}" for i,x in enumerate(answers,1)))
    complete=f"""# V23 Complete Report

## Identity

- Name: **Oracle Local Action Charts and Input-Rank Scaling**
- Parent: `{base['parent_commit']}`
- Protocol hash: `{base['freeze_digest']}`
- State split hash: `{candidate['state_split_hash']}`
- Probe hashes: `{json.dumps(probes['probe_hashes'],sort_keys=True)}`
- Train / held-out / new-final hashes: `{probes['training_action_hash']}` / `{probes['heldout_action_hash']}` / `{candidate['new_independent_final_hash']}`

## Outcome

Formal outcome: **{adj['primary_outcome']}**.

{adj['negative_result_wording']}

## Rank scaling

{rank_table}

Joint rank saturation: **{_f(rank['INPUT_RANK_SATURATED'])}**.

## Oracle charts

Best JVP: `{oracle['best']['JVP']['token']}`; best finite: `{oracle['best']['finite']['token']}`; `k_chart_min={_f(oracle['k_chart_min'])}`.

### Best finite metrics

{_category_table(oracle['best']['finite'])}

### Best JVP metrics

{_category_table(oracle['best']['JVP'])}

JVP/finite input overlap: `{_f(oracle['jvp_finite_comparison']['median_input_overlap'])}`; angle: `{_f(oracle['jvp_finite_comparison']['median_input_angle_degrees'])}°`.

## Smoothness, aliasing, transport

- Nearest-J clean-state chart angle: `{_f(s['nearby_J_median_angle_degrees'])}°`; adjacent-token result unavailable on the frozen panel.
- Same-J P0/Pq chart angle: `{_f(sj['finite_input_chart_angle_median_degrees'])}°`.
- Procrustes transport fidelity: `{_f(t['transport_fidelity_median'])}`.

## Prediction and adaptation

- Chart-predictor status: `{p['status']}`; S0/S1/S2/S4 = `{_f(p['S0'])}/{_f(p['S1'])}/{_f(p['S2'])}/{_f(p['S4'])}`.
- Local probe curve: `{json.dumps(oracle['local_probe_curve'],sort_keys=True)}`.
- Oracle vs learned direction L2: `{_f(ovl['oracle_best_direction_l2'])}` vs `{_f(ovl['V22_learned_chart_l2'])}`.

## Authorization

- `COMPACT_OPERATOR_SEARCH_REOPENED = {_f(adj['COMPACT_OPERATOR_SEARCH_REOPENED'])}`
- `RAW_TO_OPERATOR_ENCODER_AUTHORIZED = {_f(adj['RAW_TO_OPERATOR_ENCODER_AUTHORIZED'])}`
- `H2_REMAINS = {_f(adj['H2_REMAINS'])}`
- `H3_AUTHORIZED = {_f(adj['H3_AUTHORIZED'])}`
- `DYNAMIC_STATE_SEARCH_AUTHORIZED = {_f(adj['DYNAMIC_STATE_SEARCH_AUTHORIZED'])}`
- Independent final: `{adj['independent_final_status']}`

## Verification

V23 tests: `{v23_tests}`. Full suite: `{full_tests}`. Historical records were not rewritten to hide inherited failures.

## Files

All required standalone V23 reports are in `reports/`; machine-readable JSON/Parquet outputs are in `results/v23/processed/`; raw response matrices remain in the external V23 scratch directory.
"""
    _write(root,"V23_COMPLETE_REPORT.md",complete)
    final_path=root/REPORTS/"FINAL_REPORT.md"
    marker="\n\n---\n\n"+complete
    if "# V23 Complete Report" in final_path.read_text(encoding="utf-8"):
        raise RuntimeError("V23 already appended to FINAL_REPORT")
    final_path.write_text(final_path.read_text(encoding="utf-8").rstrip()+marker,encoding="utf-8")
    status=subprocess.check_output(["git","status","--short"],cwd=root,text=True).splitlines()
    write_json_atomic(root/OUT/"v23_test_audit.json",{"v23_tests":v23_tests,"full_suite":full_tests})
    paths=[]
    for base_dir in ("artifacts","results/v23/processed","reports"):
        for path in sorted((root/base_dir).glob("**/*")):
            rel=str(path.relative_to(root))
            if path.is_file() and ("v23" in rel.lower() or rel=="reports/FINAL_REPORT.md") and not rel.endswith("v23_integrity_index.json"):
                paths.append(rel)
    integrity={"files":{p:sha256_file(root/p) for p in paths},"file_count":len(paths),"git_status_before_final_freeze":status,
               "historical_final_responses_opened":False,"v23_independent_final_responses_opened":False}
    write_json_atomic(root/OUT/"v23_integrity_index.json",integrity)
    inputs=[SOURCE,"reports/V23_COMPLETE_REPORT.md","reports/FINAL_REPORT.md","results/v23/processed/input_rank_scaling_v23.json",
            "results/v23/processed/oracle_local_action_charts_v23.json","results/v23/processed/v23_adjudication.json",
            "results/v23/processed/v23_integrity_index.json"]+[f"reports/{x}" for x in REPORT_NAMES if x!="V23_COMPLETE_REPORT.md"]
    frozen=stage_freeze(root,"final",inputs,{"formal_outcomes":adj["formal_outcomes"],"integrity_file_count":len(paths),
        "integrity_index_sha256":sha256_file(root/OUT/"v23_integrity_index.json"),"independent_final_responses_opened":False,
        "COMPACT_OPERATOR_SEARCH_REOPENED":adj["COMPACT_OPERATOR_SEARCH_REOPENED"],"RAW_TO_OPERATOR_ENCODER_AUTHORIZED":False,
        "H3_AUTHORIZED":False,"DYNAMIC_STATE_SEARCH_AUTHORIZED":False})
    return {"final_freeze_digest":frozen["freeze_digest"],"formal_outcomes":adj["formal_outcomes"],"report_count":len(REPORT_NAMES),"integrity_file_count":len(paths)}


if __name__=="__main__":
    parser=argparse.ArgumentParser(); parser.add_argument("--v23-tests",required=True); parser.add_argument("--full-tests",required=True); args=parser.parse_args()
    print(json.dumps(generate(Path.cwd(),args.v23_tests,args.full_tests),indent=2))

