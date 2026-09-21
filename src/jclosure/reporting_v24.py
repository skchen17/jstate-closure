"""Standalone and cumulative V24 reports."""
from __future__ import annotations
import argparse,json,subprocess
from pathlib import Path
from jclosure.protocol_v24 import verify,verify_stage,stage_freeze
from jclosure.provenance import sha256_file,write_json_atomic
OUT=Path("results/v24/processed"); R=Path("reports"); SOURCE="src/jclosure/reporting_v24.py"
NAMES=["BROAD_CAUSAL_TARGET_AUDIT_V24.md","LAYERWISE_CAUSAL_RANK_V24.md","CAUSAL_RANK_COMPRESSION_PROFILE_V24.md","OUTPUT_BOTTLENECK_GENERALIZATION_V24.md","GLOBAL_VS_LOCAL_BOTTLENECK_V24.md","BOTTLENECK_ORIENTATION_DYNAMICS_V24.md","BOTTLENECK_CAUSAL_SUFFICIENCY_V24.md","BOTTLENECK_CAUSAL_NECESSITY_V24.md","BOTTLENECK_RESTORATION_V24.md","BOTTLENECK_TRANSPLANT_V24.md","BOTTLENECK_HORIZON_PROPAGATION_V24.md","BOTTLENECK_REGENERATION_V24.md","ARCHITECTURE_BOTTLENECK_LOCALIZATION_V24.md","WORKSPACE_BOTTLENECK_RELATION_V24.md","STRICT_INTERFACE_AUDIT_V24.md","EXECUTION_MANIFEST_V24.md","V24_SCIENTIFIC_ANSWERS_V24.md","V24_COMPLETE_REPORT.md"]
def w(root,n,s):(root/R/n).write_text(s.rstrip()+"\n",encoding="utf-8")
def f(x):
 if x is None:return "N/A"
 if isinstance(x,bool):return "TRUE" if x else "FALSE"
 if isinstance(x,float):return f"{x:.6f}"
 return str(x)
def generate(root:Path,vtests:str,fulltests:str):
 base=verify(root); final_protocol=verify_stage(root,"mediation_protocol"); mediation_results=verify_stage(root,"mediation_results_amendment"); d=json.loads((root/OUT/"bottleneck_design_v24.json").read_text()); c=json.loads((root/OUT/"bottleneck_candidate_selection_v24.json").read_text()); v=json.loads((root/OUT/"output_bottleneck_validation_v24.json").read_text()); m=json.loads((root/OUT/"causal_bottleneck_mediation_v24.json").read_text()); a=json.loads((root/OUT/"v24_adjudication.json").read_text()); natural=json.loads((root/OUT/"natural_transition_amendment_v24.json").read_text())
 layer=v["candidate_layer"]; k=v["candidate_k"]; p=v["compression_profile"]
 w(root,NAMES[0],f"# Broad Causal Target Audit — V24\n\nFrozen response-independent targets: T0 historical 288-D; T1 512 random orthogonal vocabulary coordinates; T2 256 random orthogonal residual coordinates at every layer; T3 128 attention/recurrent plus 128 MLP coordinates at ten transition layers; T4 balanced T0+T1+T2.\n\nAt 256 probes: early broad r95 `{p['early_hidden_broad_r95']}`, candidate `{p['candidate_hidden_broad_r95']}`, late `{p['late_hidden_broad_r95']}`, late T0 `{p['late_T0_r95']}`, random-vocabulary `{p['late_vocab_r95']}`. Representational pass: **{f(v['LOW_RANK_OUTPUT_BOTTLENECK_REPRESENTATIONAL_PASS'])}**.\n")
 rows=["| layer | residual r95 | broad r95 |","|---:|---:|---:|"]
 for l in base["config"]["architecture_layers"]: rows.append(f"| {l} | {v['rank_curves']['residual']['256'][l]} | {v['rank_curves']['broad_T4']['256'][l]} |")
 w(root,NAMES[1],"# Layerwise Causal Rank — V24\n\n"+"\n".join(rows)+f"\n\nFormal collapse layer: `{f(v['formal_rank_collapse_layer'])}`; diagnostic layer: `{layer}`; largest adjacent drop at layer `{v['rank_collapse_shape']['largest_adjacent_drop_layer']}` with Δr95 `{f(v['rank_collapse_shape']['largest_adjacent_r95_drop'])}`.\n")
 w(root,NAMES[2],f"# Causal Rank Compression Profile — V24\n\n`input JVP/finite r95@256 = {p['V23_input_JVP_r95_at_256']}/{p['V23_input_finite_r95_at_256']} -> early broad hidden {p['early_hidden_broad_r95']} -> candidate {p['candidate_hidden_broad_r95']} -> late broad {p['late_hidden_broad_r95']} -> T0 {p['late_T0_r95']}`. Output T0 saturation: **{f(v['output_T0_rank_saturated'])}**.\n")
 cov=["| basis | explained norm | residual | cosine |","|---|---:|---:|---:|"]
 for name,x in v["coverage"].items():cov.append(f"| {name} | {f(x['explained_norm_fraction_median'])} | {f(x['relative_residual_median'])} | {f(x['median_cosine'])} |")
 w(root,NAMES[3],"# Output Bottleneck Generalization — V24\n\n"+"\n".join(cov)+f"\n\nGlobal held-out pass `{f(v['heldout_global_pass'])}`; same-state oracle pass `{f(v['heldout_state_oracle_pass'])}`.\n")
 w(root,NAMES[4],f"# Global versus Local Bottleneck — V24\n\n"+"\n".join(cov)+f"\n\nLocal-oracle residual improvement over global: `{f(v['global_vs_local_oracle_residual_gain'])}`. Global outcome `{f(a['V24_D'])}`; moving outcome `{f(a['V24_C'])}`.\n")
 o=v["orientation_dynamics"]; sj=v["same_J_orientation"]
 w(root,NAMES[5],f"# Bottleneck Orientation Dynamics — V24\n\nSame-J P0/Pq angle `{f(sj['median_angle_degrees'])}°`, overlap `{f(sj['median_overlap'])}`. Nearest-J angle `{f(o['nearest_J_angle_median'])}°`; same-family `{f(o['same_family_angle_median'])}°`; across-family `{f(o['across_family_angle_median'])}°`; transport fidelity `{f(o['transport_fidelity_median'])}`. Stable dimension is not equated with stable orientation.\n")
 w(root,NAMES[6],f"# Bottleneck Causal Sufficiency — V24\n\nCandidate layer/k `{layer}/{k}`. B_ONLY relative L2 `{f(m['B_ONLY']['relative_l2'])}`, cosine `{f(m['B_ONLY']['cosine'])}`, norm ratio `{f(m['B_ONLY']['norm_ratio'])}`. Causal gate: **{f(m['CAUSAL_MEDIATION_GATE_PASS'])}**.\n")
 w(root,NAMES[7],f"# Bottleneck Causal Necessity — V24\n\nPERP_ONLY retained ratio `{f(m['PERP_ONLY']['retained_ratio'])}`, bootstrap 97.5% upper `{f(m['PERP_ONLY']['bootstrap_97_5_upper'])}`. FULL-minus-B retained ratio `{f(m['FULL_MINUS_B']['retained_ratio'])}`. Necessity is not inferred from PCA variance.\n")
 w(root,NAMES[8],f"# Bottleneck Restoration — V24\n\nPq→P0 restoration relative residual `{f(m['restoration']['relative_to_P0_median'])}`; descriptive intervention-mediated fraction `{f(m['restoration']['intervention_mediated_fraction_median'])}`. This is not a classical natural indirect effect.\n")
 w(root,NAMES[9],f"# Bottleneck Transplant — V24\n\nP0→Pq transplant relative L2 `{f(m['transplant']['relative_l2_median'])}`, cosine `{f(m['transplant']['cosine_median'])}`, norm ratio `{f(m['transplant']['norm_ratio_median'])}`.\n")
 w(root,NAMES[10],f"# Bottleneck Horizon Propagation — V24\n\nStatus: **{m['horizon_status']}**. h2/h4/h8 are not opened when the frozen h1 mediation gate fails.\n")
 w(root,NAMES[11],f"# Bottleneck Regeneration — V24\n\nStatus: **{m['regeneration']['status']}**. Median later-layer projection of PERP_ONLY effect into train response bases: `{f(m['regeneration']['later_projected_ratio_median'])}`. By layer: `{json.dumps(m['regeneration']['by_layer'],sort_keys=True)}`. Candidate layer 31 is the final layer, so no downstream regeneration estimand exists. V24-G: **{f(a['V24_G'])}**.\n")
 arch=a["architecture_localization"]
 w(root,NAMES[12],f"# Architecture Bottleneck Localization — V24\n\nNearest architecture transition layer `{arch['nearest_architecture_layer']}`. Attention/recurrent r95 `{f(arch['component_r95_median']['arch_attn'])}`; MLP r95 `{f(arch['component_r95_median']['arch_mlp'])}`. Channel-conditioned norms: `{json.dumps(arch['channel_effect_norm_median'],sort_keys=True)}`.\n\n{arch['wording']}\n")
 ws=a["workspace_relationship"]
 w(root,NAMES[13],f"# Workspace–Bottleneck Relation — V24\n\nJ-from-B train relative L2 `{f(ws['J_from_B_train_relative_l2'])}`. Same-J q-effect bottleneck coverage `{json.dumps(ws['same_J_q_effect_bottleneck_coverage'],sort_keys=True)}`. Workspace aliasing mediated: **{f(ws['WORKSPACE_ALIASING_IS_MEDIATED_BY_DOWNSTREAM_CAUSAL_BOTTLENECK'])}**.\n")
 w(root,NAMES[14],f"# Strict Interface Audit — V24\n\n- Historical final opened: `{f(a['historical_final_opened'])}`.\n- V24 independent final opened: `{f(a['v24_independent_final_opened'])}`.\n- Final state hash: `{d['state_hashes']['independent_final']}`.\n- Train/held-out action hashes: `{d['action_hashes']['train']}` / `{d['action_hashes']['heldout']}`.\n- Target projection hash: `{d['target_projection_sha256']}`.\n- Raw bottleneck basis hash: `{json.loads((root/OUT/'mediation_protocol_v24.json').read_text())['basis_sha256']}`.\n- Writeback min cosine / median gain: `{f(m['writeback_audit']['min_cosine'])}` / `{f(m['writeback_audit']['median_gain'])}`.\n- Only the frozen layer bottleneck component was written; raw persistent cache was not restored.\n")
 commands=["PYTHONPATH=src /home/user/anaconda3/bin/python -m jclosure.protocol_v24 freeze","PYTHONPATH=src /home/user/anaconda3/bin/python -m jclosure.experiments.bottleneck_design_v24","PYTHONPATH=src /home/user/anaconda3/bin/python -m jclosure.experiments.layer_response_v24 prepare","CUDA_VISIBLE_DEVICES=1 HF_HOME=/data/CSK/.cache/huggingface PYTHONPATH=src /home/user/anaconda3/bin/python -m jclosure.experiments.layer_response_v24 finite --role development","CUDA_VISIBLE_DEVICES=0 HF_HOME=/data/CSK/.cache/huggingface PYTHONPATH=src /home/user/anaconda3/bin/python -m jclosure.experiments.layer_response_v24 finite --role validation","CUDA_VISIBLE_DEVICES=0 HF_HOME=/data/CSK/.cache/huggingface PYTHONPATH=src /home/user/anaconda3/bin/python -m jclosure.experiments.layer_response_v24 jvp","CUDA_VISIBLE_DEVICES=0 HF_HOME=/data/CSK/.cache/huggingface PYTHONPATH=src /home/user/anaconda3/bin/python -m jclosure.experiments.layer_response_v24 natural  # audited failure: 8/10 rows expose h1 only","CUDA_VISIBLE_DEVICES=0 HF_HOME=/data/CSK/.cache/huggingface PYTHONPATH=src /home/user/anaconda3/bin/python -m jclosure.experiments.natural_response_v24_amendment","PYTHONPATH=src /home/user/anaconda3/bin/python -m jclosure.experiments.layer_response_summary_v24_amendment","CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src /home/user/anaconda3/bin/python -m jclosure.experiments.analyze_bottleneck_v24 select","CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src /home/user/anaconda3/bin/python -m jclosure.experiments.analyze_bottleneck_v24 validate","CUDA_VISIBLE_DEVICES=1 HF_HOME=/data/CSK/.cache/huggingface PYTHONPATH=src /home/user/anaconda3/bin/python -m jclosure.experiments.mediation_v24 basis","CUDA_VISIBLE_DEVICES=1 HF_HOME=/data/CSK/.cache/huggingface PYTHONPATH=src /home/user/anaconda3/bin/python -m jclosure.experiments.mediation_v24 run  # branches persisted; empty final-layer regeneration summary amended below","CUDA_VISIBLE_DEVICES=1 HF_HOME=/data/CSK/.cache/huggingface PYTHONPATH=src /home/user/anaconda3/bin/python -m jclosure.experiments.mediation_results_v24_amendment","CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src /home/user/anaconda3/bin/python -m jclosure.experiments.adjudicate_v24","PYTHONPATH=src /home/user/anaconda3/bin/python -m pytest tests/test_v24_output_bottleneck.py -q","PYTHONPATH=src /home/user/anaconda3/bin/python -m pytest -q","git commit -m 'Complete V24 causal output bottleneck validation'","git push origin main"]
 amendments={"natural_transition":{"design_count":natural["design_count"],"eligible":natural["eligible_transition_count"],"excluded_h1_only":natural["excluded_count"]},"mediation_regeneration":{"status":m["regeneration"]["status"],"branch_records_reused":m["branch_rows"]}}
 write_json_atomic(root/OUT/"execution_manifest_v24.json",{"commands":commands,"amendments":amendments,"v24_tests":vtests,"full_tests":fulltests}); w(root,NAMES[15],"# Execution Manifest — V24\n\n"+"\n".join(f"- `{x}`" for x in commands)+f"\n\nAppend-only amendments: `{json.dumps(amendments,sort_keys=True)}`. V24 tests `{vtests}`; full suite `{fulltests}`.\n")
 answers=[f"Broad-target low rank survives: {v['LOW_RANK_OUTPUT_BOTTLENECK_REPRESENTATIONAL_PASS']}.",f"T0 output rank saturates: {v['output_T0_rank_saturated']}.",f"Formal rank-collapse layer: {v['formal_rank_collapse_layer']}.",f"Largest adjacent rank change occurs at layer {v['rank_collapse_shape']['largest_adjacent_drop_layer']}.",f"Candidate is stable minimum: {v['rank_collapse_shape']['candidate_is_stable_minimum']}.","Family dimension stability is reported in layerwise parquet and frozen thresholds.","P0/Pq dimension uses identical probe sets and target projections.",f"Same-J orientation angle: {sj['median_angle_degrees']:.6f} degrees.",f"Global B pass: {v['heldout_global_pass']}.",f"Local oracle residual gain: {v['global_vs_local_oracle_residual_gain']:.6f}.",f"Held-out state-oracle pass: {v['heldout_state_oracle_pass']}.",f"B_ONLY: {m['B_ONLY']}.",f"PERP_ONLY retained ratio: {m['PERP_ONLY']['retained_ratio']:.6f}.",f"FULL-minus-B retained ratio: {m['FULL_MINUS_B']['retained_ratio']:.6f}.",f"Restoration residual: {m['restoration']['relative_to_P0_median']:.6f}.",f"Transplant metrics: {m['transplant']}.",f"Intervention-mediated fraction: {m['restoration']['intervention_mediated_fraction_median']:.6f}.",f"Horizon status: {m['horizon_status']}.",f"Regeneration ratio: {f(m['regeneration']['later_projected_ratio_median'])} ({m['regeneration']['status']}).",f"Architecture localization: {arch}.",f"Workspace relation: {ws}.",f"Same-J aliasing appears in B angle {sj['median_angle_degrees']:.6f} degrees.",f"Restoring B eliminates aliasing: {ws['WORKSPACE_ALIASING_IS_MEDIATED_BY_DOWNSTREAM_CAUSAL_BOTTLENECK']}.",f"Internal mediator vs readout outcome: {a['primary_outcome']}.",f"V24-A: {a['V24_A']}.",f"V24-B: {a['V24_B']}.",f"H2 remains: {a['H2_REMAINS']}.",f"H3 authorized: {a['H3_AUTHORIZED']}.",f"Dynamic search authorized: {a['DYNAMIC_STATE_SEARCH_AUTHORIZED']}." ]
 write_json_atomic(root/OUT/"v24_scientific_answers.json",{"answers":answers}); w(root,NAMES[16],"# V24 Scientific Answers\n\n"+"\n".join(f"{i}. {x}" for i,x in enumerate(answers,1)))
 complete=f"""# V24 Complete Report

## Identity

- Name: **Causal Output Bottleneck Validation and Layer Localization**
- Parent: `{base['parent_commit']}`
- Protocol hash: `{base['freeze_digest']}`
- State hashes: `{json.dumps(d['state_hashes'],sort_keys=True)}`
- Action hashes: `{json.dumps(d['action_hashes'],sort_keys=True)}`
- Target projection hash: `{d['target_projection_sha256']}`
- Candidate layer/k: `{layer}/{k}`
- Mediation protocol hash: `{final_protocol['freeze_digest']}`
- Mediation-results amendment hash: `{mediation_results['freeze_digest']}`
- Natural transitions: `{natural['eligible_transition_count']}` eligible h2−h1 rows; `{natural['excluded_count']}` audited h1-only exclusions from ten frozen design rows.

## Outcome

Formal outcome: **{a['primary_outcome']}**.

Representational bottleneck pass: `{f(v['LOW_RANK_OUTPUT_BOTTLENECK_REPRESENTATIONAL_PASS'])}`. Causal mediation gate: `{f(m['CAUSAL_MEDIATION_GATE_PASS'])}`. A causal mediator is not equated with a complete model state.

## Compression profile

- V23 input JVP/finite r95@256: `{p['V23_input_JVP_r95_at_256']}/{p['V23_input_finite_r95_at_256']}`.
- Early/candidate/late broad hidden r95: `{p['early_hidden_broad_r95']}/{p['candidate_hidden_broad_r95']}/{p['late_hidden_broad_r95']}`.
- Late T0/vocabulary r95: `{p['late_T0_r95']}/{p['late_vocab_r95']}`.
- Formal collapse layer: `{f(v['formal_rank_collapse_layer'])}`.

## Held-out coverage

{chr(10).join(cov)}

## Causal branches

- B_ONLY L2/cosine/norm: `{f(m['B_ONLY']['relative_l2'])}/{f(m['B_ONLY']['cosine'])}/{f(m['B_ONLY']['norm_ratio'])}`.
- PERP_ONLY retained, 97.5% upper: `{f(m['PERP_ONLY']['retained_ratio'])}/{f(m['PERP_ONLY']['bootstrap_97_5_upper'])}`.
- FULL-minus-B retained: `{f(m['FULL_MINUS_B']['retained_ratio'])}`.
- Restoration mediated fraction: `{f(m['restoration']['intervention_mediated_fraction_median'])}`.
- Transplant L2/cosine/norm: `{f(m['transplant']['relative_l2_median'])}/{f(m['transplant']['cosine_median'])}/{f(m['transplant']['norm_ratio_median'])}`.
- Regeneration ratio: `{f(m['regeneration']['later_projected_ratio_median'])}`.
- Horizon status: `{m['horizon_status']}`.

## Authorization

- `H2_REMAINS = {f(a['H2_REMAINS'])}`
- `H3_AUTHORIZED = {f(a['H3_AUTHORIZED'])}`
- `DYNAMIC_STATE_SEARCH_AUTHORIZED = {f(a['DYNAMIC_STATE_SEARCH_AUTHORIZED'])}`
- Independent final: `{a['independent_final_status']}`

## Verification

V24 tests: `{vtests}`. Full suite: `{fulltests}`. Historical cumulative-report hash failures remain visible.
"""
 w(root,NAMES[17],complete); final=root/R/"FINAL_REPORT.md"
 if "# V24 Complete Report" in final.read_text():raise RuntimeError("V24 already appended")
 final.write_text(final.read_text().rstrip()+"\n\n---\n\n"+complete,encoding="utf-8")
 write_json_atomic(root/OUT/"v24_test_audit.json",{"v24_tests":vtests,"full_suite":fulltests})
 paths=[]
 for base_dir in ("artifacts","results/v24/processed","reports"):
  for path in sorted((root/base_dir).glob("**/*")):
   rel=str(path.relative_to(root))
   if path.is_file() and ("v24" in rel.lower() or rel=="reports/FINAL_REPORT.md") and not rel.endswith("v24_integrity_index.json"):paths.append(rel)
 integrity={"files":{x:sha256_file(root/x) for x in paths},"file_count":len(paths),"historical_final_opened":False,"v24_independent_final_opened":False}; write_json_atomic(root/OUT/"v24_integrity_index.json",integrity)
 inputs=[SOURCE,"reports/V24_COMPLETE_REPORT.md","reports/FINAL_REPORT.md","results/v24/processed/output_bottleneck_validation_v24.json","results/v24/processed/causal_bottleneck_mediation_v24.json","results/v24/processed/v24_adjudication.json","results/v24/processed/v24_integrity_index.json"]+[f"reports/{x}" for x in NAMES[:-1]]
 frozen=stage_freeze(root,"final",inputs,{"formal_outcomes":a["formal_outcomes"],"integrity_index_sha256":sha256_file(root/OUT/"v24_integrity_index.json"),"historical_final_opened":False,"v24_independent_final_opened":False,"H3_AUTHORIZED":False,"DYNAMIC_STATE_SEARCH_AUTHORIZED":False})
 return {"final_freeze_digest":frozen["freeze_digest"],"formal_outcomes":a["formal_outcomes"],"reports":len(NAMES),"integrity_files":len(paths)}
if __name__=="__main__":
 p=argparse.ArgumentParser();p.add_argument("--v24-tests",required=True);p.add_argument("--full-tests",required=True);x=p.parse_args();print(json.dumps(generate(Path.cwd(),x.v24_tests,x.full_tests),indent=2))
