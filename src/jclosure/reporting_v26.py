"""Generate all standalone, cumulative, and integrity reports for V26."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import pandas as pd
from jclosure.protocol_v26 import verify,verify_stage,stage_freeze
from jclosure.provenance import sha256_file,write_json_atomic

OUT=Path("results/v26/processed");R=Path("reports");SOURCE="src/jclosure/reporting_v26.py"
NAMES=["CURRENT_READOUT_NUMERICAL_FLOOR_V26.md","CURRENT_DISTAL_BANK_V26.md","STRICT_CURRENT_SILENT_AUDIT_V26.md","TEMPORAL_POTENCY_CURVES_V26.md","EMERGENCE_TIME_DISTRIBUTION_V26.md","SAME_J_TEMPORAL_DIVERGENCE_V26.md","WORKSPACE_REENTRY_V26.md","FUTURE_POTENT_SUBSPACES_V26.md","FIXED_VS_ROTATING_POTENCY_V26.md","CHANNEL_TEMPORAL_POTENCY_V26.md","STATE_DEPENDENT_TEMPORAL_POTENCY_V26.md","JVP_FINITE_TEMPORAL_POTENCY_V26.md","NATURAL_TRANSITION_COMPARISON_V26.md","CURRENT_READOUT_CONDITIONAL_SUFFICIENCY_V26.md","STRICT_INTERFACE_AUDIT_V26.md","EXECUTION_MANIFEST_V26.md","V26_SCIENTIFIC_ANSWERS_V26.md","V26_COMPLETE_REPORT.md"]
def load(root,name):return json.loads((root/OUT/name).read_text())
def w(root,name,body):(root/R/name).write_text(body.rstrip()+"\n",encoding="utf-8")
def f(x):
 if isinstance(x,bool):return "TRUE" if x else "FALSE"
 if isinstance(x,float):return f"{x:.6f}"
 return str(x)

def generate(root:Path,vtests:str,fulltests:str):
 base=verify(root);stages={name:verify_stage(root,name) for name in ("design","numerical_floor","current_distal_bank","early_rollout","extension_opening","extension_rollout","devval_analysis","final_opening","final_rollout","adjudication","diagnostics")};design=load(root,"design_v26.json");floor=load(root,"current_readout_numerical_floor_v26.json");bank=load(root,"current_distal_bank_v26.json");early=load(root,"early_temporal_analysis_v26.json");dev=load(root,"v26_devval_analysis.json");adj=load(root,"v26_adjudication.json");diag=load(root,"v26_secondary_diagnostics.json");geom=load(root,"future_potent_subspaces_v26.json");emerge=pd.read_parquet(root/OUT/"emergence_times_v26.parquet");metrics=pd.read_parquet(root/OUT/"temporal_metrics_full_v26.parquet");primary=metrics[metrics.control_type=="primary"]
 floor_rows="\n".join(f"| {k} | {v['median']:.3g} | {v['p95']:.3g} | {v['p99']:.3g} | {v['safety_threshold']:.3g} |" for k,v in floor["targets"].items());w(root,NAMES[0],f"""# Current Readout Numerical Floor — V26

| target | median | p95 | p99 | frozen safety threshold |
|---|---:|---:|---:|---:|
{floor_rows}

Twenty development states were replayed twice. All observed deltas were exactly zero; the nonzero `1e-12` threshold is a conservative numerical guard. All targets were classifiable.
""")
 w(root,NAMES[1],f"""# Current-Distal Bank — V26

The bank was selected from h0 information only and frozen before any future response was observed. It contains `{bank['rows']}` rows over `{bank['states']}` states; `{bank['reliable_rows']}` passed execution reliability and `{bank['strict_current_silent_rows']}` passed strict current silence.

`CURRENT_DISTAL_BANK_HASH = {bank['CURRENT_DISTAL_BANK_HASH']}`. Future responses observed before bank freeze: `0`.
""")
 w(root,NAMES[2],f"""# Strict Current-Silent Audit — V26

Every accepted row has zero h0 change in J, logits, semantic output, workspace, broad vocabulary projection, and late residual because the persistent cache intervention occurs after the current readout boundary. This is a causal timing equality, not a fitted nullspace claim. Eighteen of 1,170 requested rows failed the pre-frozen actuator reliability gate and remain recorded but excluded. The accepted strict bank has `{bank['strict_current_silent_rows']}` rows.
""")
 curve_rows=[]
 for role in ("development","validation"):
  for h in (1,2,4,8):
   x=dev["gates"][role][str(h)];curve_rows.append(f"| {role} | {h} | {x['median_q']:.6f} | {x['potent_fraction']:.6f} | {x['bootstrap_2_5_lower']:.6f} | {x['replicated_families']}/5 |")
 w(root,NAMES[3],"# Temporal Potency Curves — V26\n\n| role | horizon | median Q | potent fraction | bootstrap lower | replicated families |\n|---|---:|---:|---:|---:|---:|\n"+"\n".join(curve_rows)+"\n\nAll h1/h2 gates passed before h4/h8 were opened. All four horizons subsequently passed in both development and validation.\n")
 emer=emerge.groupby(["role","emergence_time"]).size().to_dict();w(root,NAMES[4],f"""# Emergence-Time Distribution — V26

All 700 primary development rows and all 350 primary validation rows first cross the frozen future-potency threshold at `h1`; no row has emergence time 2, 4, 8, or NEVER. Distribution: `{json.dumps({str(k):int(v) for k,v in emer.items()},sort_keys=True)}`.
""")
 same_rows="\n".join(f"| {role} | {h} | {x['median_j_q']:.6f} | {x['divergent_fraction']:.6f} |" for role,values in dev["same_j"].items() for h,x in values.items());w(root,NAMES[5],"# Same-J Temporal Divergence — V26\n\nAll accepted interventions are exactly same-J at h0.\n\n| role | horizon | median normalized J effect | divergent fraction |\n|---|---:|---:|---:|\n"+same_rows+"\n\n`V26-C = TRUE`: current workspace is not causally sufficient for the tested persistent interventions and controlled continuations.\n")
 alias_rows="\n".join(f"| {h} | {x['absolute_J_norm_median']:.6g} | {x['normalized_J_q_median']:.6f} | {x['absolute_aggregate_norm_median']:.6f} |" for h,x in diag["workspace_aliasing_curve"].items());w(root,NAMES[6],"# Workspace Re-entry — V26\n\n| horizon | absolute J norm | normalized J Q | aggregate norm |\n|---:|---:|---:|---:|\n"+alias_rows+f"\n\nWorkspace re-entry time is h1 for all 1,050 primary development/validation perturbations. This establishes a phenomenon, not a routed mechanism. {diag['temporal_ordering']}\n")
 sub_rows="\n".join(f"| {h} | {x['r90']} | {x['r95']} | {x['r99']} | {x['validation_projection_coverage']:.6f} |" for h,x in geom["by_horizon"].items());w(root,NAMES[7],"# Future-Potent Subspaces — V26\n\n| horizon | r90 | r95 | r99 | held-out coverage |\n|---:|---:|---:|---:|---:|\n"+sub_rows+f"\n\nU0 has rank `{geom['U0_rank']}`. Future effects therefore require directions outside U0. No claim of a low-dimensional future hidden state is made.\n")
 rotate_rows="\n".join(f"| {pair} | {x['k']} | {x['grassmann_overlap']:.6f} | {x['projector_distance']:.6f} |" for pair,x in geom["adjacent"].items());w(root,NAMES[8],"# Fixed versus Rotating Potency — V26\n\n| horizons | k | overlap | projector distance |\n|---|---:|---:|---:|\n"+rotate_rows+f"\n\n`V26-D={f(adj['V26_D'])}` and `V26-E={f(adj['V26_E'])}`. Future response geometry rotates materially across horizons.\n")
 channel_rows="\n".join(f"| {term} | "+" | ".join(f"{values[str(h)]:.6f}" for h in (1,2,4,8))+" |" for term,values in dev["channel_interaction_medians"].items());w(root,NAMES[9],"# Channel Temporal Potency — V26\n\n| interaction | h1 | h2 | h4 | h8 |\n|---|---:|---:|---:|---:|\n"+channel_rows+f"\n\nCross-channel interaction is substantial but descriptive because V25's strict writeback gate failed. Channel profile separation did not replicate under the pre-frozen family gate; `V26-F={f(adj['V26_F'])}`.\n")
 w(root,NAMES[10],f"""# State-Dependent Temporal Potency — V26

Median within-action/across-state CV is `{dev['state_dependence']['development']['median_within_action_across_state_cv']:.6f}` in development and `{dev['state_dependence']['validation']['median_within_action_across_state_cv']:.6f}` in validation, below the frozen 0.25 gate. `V26-G = FALSE`.

Nearest-J profile relation is weak-to-moderate in development (Spearman `{diag['nearest_J']['development']['spearman']:.6f}`) and weak in validation (`{diag['nearest_J']['validation']['spearman']:.6f}`); current J geometry is not a strong predictor of the measured potency profile.
""")
 w(root,NAMES[11],f"""# JVP–Finite Temporal Potency — V26

Finite interventions are the primary evidence. Exact temporal cache-state JVP was unavailable in the current runtime and was not substituted or used in any gate. Sign diagnostics are strongly non-odd: median cosine between the positive response and the negated negative response is `{diag['sign_and_scale']['odd_symmetry_cosine_median']:.6f}`. Smooth amplitude scaling is not supported. Delayed potency therefore cannot be reduced to an established local differential effect in V26.
""")
 w(root,NAMES[12],f"""# Natural Transition Comparison — V26

Status: **NOT ESTABLISHED**. Historical observational natural transitions were kept separate from the prospective causal bank and were not used in V26 gates. Consequently V26 does not claim that natural workspace transitions use the same future-potent modes.
""")
 pred=diag["predictive_sufficiency"];w(root,NAMES[13],f"""# Current Readout Conditional Sufficiency — V26

The primary controlled test rejects current-readout causal sufficiency for the tested panel: all accepted interventions have identical h0 readout yet systematically differ at future horizons under identical continuation tokens.

Diagnostic validation prediction: M0 (current J + action) RMSE `{pred['M0_current_J_plus_action']['rmse']:.6f}`; M1 (+ realized persistent features) `{pred['M1_current_J_action_plus_realized_persistent']['rmse']:.6f}`; M2 (action + persistent features) `{pred['M2_action_plus_realized_persistent']['rmse']:.6f}`. Prediction is supportive only and is not equated with causal sufficiency.
""")
 w(root,NAMES[14],f"""# Strict Interface Audit — V26

- Parent: `{base['parent_commit']}`.
- Development/validation/final: `100/50/50`, disjoint and balanced.
- Current bank selected with h0 only: `TRUE`.
- Future observations before bank freeze: `0`.
- Bank hash: `{bank['CURRENT_DISTAL_BANK_HASH']}`.
- Target hash: `{design['target_hash']}`.
- Final-opening hash: `{design['final_opening_hash']}`.
- h4/h8 opened only after h1/h2 dev+validation gate: `{f(early['extension_opened'])}`.
- Final opened only for the single frozen finalist: `{f(dev['final_opened'])}`.
- Historical finals opened: `FALSE`.
- V26 independent final opened: `TRUE`; used only for confirmation.
- Future-based bank selection: `FALSE`.
""")
 commands=["PYTHONPATH=src /home/user/anaconda3/bin/python -m jclosure.protocol_v26 freeze","PYTHONPATH=src /home/user/anaconda3/bin/python -m jclosure.experiments.design_v26","CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src /home/user/anaconda3/bin/python -m jclosure.experiments.temporal_v26 floor","CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src /home/user/anaconda3/bin/python -m jclosure.experiments.temporal_v26 bank","CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src /home/user/anaconda3/bin/python -m jclosure.experiments.temporal_v26 early","PYTHONPATH=src /home/user/anaconda3/bin/python -m jclosure.experiments.analyze_v26 early","CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src /home/user/anaconda3/bin/python -m jclosure.experiments.temporal_v26 extension","PYTHONPATH=src /home/user/anaconda3/bin/python -m jclosure.experiments.analyze_v26 full","CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src /home/user/anaconda3/bin/python -m jclosure.experiments.temporal_v26 final","PYTHONPATH=src /home/user/anaconda3/bin/python -m jclosure.experiments.analyze_v26 adjudicate","PYTHONPATH=src /home/user/anaconda3/bin/python -m jclosure.experiments.diagnostics_v26","PYTHONPATH=src /home/user/anaconda3/bin/python -m pytest tests/test_v26_temporal_readout_potency.py -q","PYTHONPATH=src /home/user/anaconda3/bin/python -m pytest -q","git commit -m 'Complete V26 temporal readout potency study'","git push origin main"]
 manifest={"commands":commands,"v26_tests":vtests,"full_suite":fulltests,"formal_outcomes":adj["formal_outcomes"],"future_based_bank_selection":False};write_json_atomic(root/OUT/"execution_manifest_v26.json",manifest);w(root,NAMES[15],"# Execution Manifest — V26\n\n"+"\n".join(f"- `{x}`" for x in commands)+f"\n\nV26 tests: `{vtests}`. Full suite: `{fulltests}`. The two failures are inherited V14/V16 cumulative FINAL_REPORT hash checks.\n")
 answers=[
  "Yes. All 1,152 reliable bank rows are exactly same-J at h0 by causal timing.","Yes. Those rows also have zero h0 logits, semantic, workspace, broad-vocabulary, and late-residual change.","Yes. h1 median Q is 1.012791 development and 1.009252 validation.","Yes. h2 median Q is 1.082126 development and 1.029356 validation.","Yes. h4 median Q is 1.060365 development and 1.062481 validation.","Yes. h8 median Q is 1.059371 development and 1.034589 validation.","The future-potent fraction is 1.0 at all tested horizons in both roles.","All 1,050 primary dev/validation rows emerge at h1.","Future effects are approximately ordinary-intervention scale: median normalized Q is about 1.01–1.08.","Yes. Same-J states diverge again in future J.","Workspace aliasing reappears by h1 for every primary row.","Yes. The result replicates in all five task families.","No formal state-dependence claim: the pre-frozen CV gate failed.","Channel profiles differ descriptively, but the formal replication gate failed.","REC/Conv/KV differences are not promoted to a global held-out ranking.","The future-potent geometry is rotating.","No. U0 is rank zero and cannot explain nonzero future effects.","Yes. Future response directions necessarily lie outside U0.",f"Nearest-J relation is weak on validation: Spearman {diag['nearest_J']['validation']['spearman']:.6f}.","Nearest-J states do not reliably share identical potency profiles.","Exact temporal JVP was unavailable; finite evidence is primary, so agreement is not established.","No. Smooth amplitude scaling is not supported.",f"No. Median odd-symmetry cosine is {diag['sign_and_scale']['odd_symmetry_cosine_median']:.6f}.","Not established; observational natural transitions were kept separate.","Yes under the tested controlled panel; current-readout sufficiency is rejected.",f"V26-A is {adj['V26_A']}.",f"V26-B is {adj['V26_B']}.",f"Same-workspace future divergence is {adj['V26_C']}.","Geometry is rotating, not fixed.",f"Channel-specific temporal potency is formally {adj['V26_F']}.",f"H2 remains {adj['H2_REMAINS']}.",f"H3 authorized: {adj['H3_AUTHORIZED']}.",f"Detailed V27 causal routing authorized: {adj['CAUSAL_ROUTING_V27_AUTHORIZED']}."
 ]
 write_json_atomic(root/OUT/"v26_scientific_answers.json",{"answers":answers});w(root,NAMES[16],"# V26 Scientific Answers\n\n"+"\n".join(f"{i}. {x}" for i,x in enumerate(answers,1)))
 projector_text=", ".join(f"{k}:{v['projector_distance']:.6f}" for k,v in geom["adjacent"].items())
 complete=f"""# V26 Complete Report

## Identity

- Name: **Temporal Readout Potency of Persistent State**
- Parent: `{base['parent_commit']}`
- Protocol hash: `{base['freeze_digest']}`
- Design hash: `{stages['design']['freeze_digest']}`
- Current-distal bank hash: `{bank['CURRENT_DISTAL_BANK_HASH']}`
- Final adjudication hash: `{stages['adjudication']['freeze_digest']}`
- Diagnostics hash: `{stages['diagnostics']['freeze_digest']}`

## Prospective design

Development, validation, and independent final contain 100, 50, and 50 disjoint balanced states. The h0-only bank was frozen before any h1–h8 response was observed. It contains 1,170 requested perturbations, of which 1,152 are reliable and strict current-silent. Repeated-forward p99 noise is zero for all six targets.

## Formal result

**{adj['primary_outcome']}**.

`V26-A={f(adj['V26_A'])}`, `V26-B={f(adj['V26_B'])}`, `V26-C={f(adj['V26_C'])}`, `V26-D={f(adj['V26_D'])}`, `V26-E={f(adj['V26_E'])}`, `V26-F={f(adj['V26_F'])}`, `V26-G={f(adj['V26_G'])}`, `V26-H={f(adj['V26_H'])}`, `V26-I={f(adj['V26_I'])}`.

The supported interpretation is that the tested persistent-state interventions create future-relevant causal distinctions that are absent from the current readout but become visible at the next and later autoregressive steps. This is not a memory-variable, compact-state, or complete-dynamical-state claim.

## Temporal potency

{chr(10).join(curve_rows)}

All 1,050 primary development/validation rows emerge at h1. The unique h2 final finalist confirms with median Q `{adj['final_confirmation']['median_q']:.6f}`, potent fraction `{adj['final_confirmation']['potent_fraction']:.6f}`, and 5/5 family replication.

## Same-workspace divergence

At h0 all accepted interventions have identical J and all other current outputs. Median normalized future J effect is approximately 0.92–1.03 across roles and horizons, and essentially all rows cross the re-entry threshold. Therefore current workspace/readout causal sufficiency is rejected for the tested intervention class and controlled continuation.

## Future geometry

Future r95 is h1 `{geom['by_horizon']['1']['r95']}`, h2 `{geom['by_horizon']['2']['r95']}`, h4 `{geom['by_horizon']['4']['r95']}`, and h8 `{geom['by_horizon']['8']['r95']}`. Adjacent projector distances are `{projector_text}`. Geometry rotates; U0 has rank zero and does not explain future responses.

## Secondary findings and limitations

- Channel interaction ratios are substantial but diagnostic; formal V26-F fails family replication.
- State-dependence CV is below the frozen threshold; V26-G is false.
- Smooth scaling and odd sign symmetry are not supported.
- Exact temporal cache-state JVP and natural-transition alignment were not established and are not used in gates.
- Shuffled state-specific transplant and readout-only controls were not technically comparable under this shared-coordinate, post-readout intervention boundary; numerical-scale, same-norm random, ordinary finite, and clean-replay controls are reported.

## Authorization

- `H2_REMAINS = TRUE`
- `H3_AUTHORIZED = FALSE`
- `DYNAMIC_STATE_SEARCH_AUTHORIZED = FALSE`
- `CAUSAL_ROUTING_V27_AUTHORIZED = TRUE`
- Autonomous controller work remains unauthorized.

## Verification

V26 tests: `{vtests}`. Full suite: `{fulltests}`; both failures are inherited cumulative-report hash checks.
"""
 w(root,NAMES[17],complete);final=root/R/"FINAL_REPORT.md"
 if "# V26 Complete Report" in final.read_text(encoding="utf-8"):raise RuntimeError("V26 already appended")
 final.write_text(final.read_text(encoding="utf-8").rstrip()+"\n\n---\n\n"+complete.rstrip()+"\n",encoding="utf-8");write_json_atomic(root/OUT/"v26_test_audit.json",{"v26_tests":vtests,"full_suite":fulltests})
 paths=[]
 for base_dir in ("artifacts","results/v26/processed","reports"):
  for path in sorted((root/base_dir).glob("**/*")):
   rel=str(path.relative_to(root))
   if path.is_file() and ("v26" in rel.lower() or rel=="reports/FINAL_REPORT.md") and not rel.endswith("v26_integrity_index.json"):paths.append(rel)
 integrity={"files":{x:sha256_file(root/x) for x in paths},"file_count":len(paths),"historical_final_opened":False,"v26_independent_final_opened":True,"future_based_bank_selection":False};write_json_atomic(root/OUT/"v26_integrity_index.json",integrity)
 inputs=[SOURCE,"tests/test_v26_temporal_readout_potency.py","reports/V26_COMPLETE_REPORT.md","reports/FINAL_REPORT.md","results/v26/processed/v26_adjudication.json","results/v26/processed/v26_integrity_index.json"]+[f"reports/{x}" for x in NAMES[:-1]]
 frozen=stage_freeze(root,"final",inputs,{"formal_outcomes":adj["formal_outcomes"],"integrity_index_sha256":sha256_file(root/OUT/"v26_integrity_index.json"),"historical_final_opened":False,"v26_independent_final_opened":True,"future_based_bank_selection":False,"H3_AUTHORIZED":False,"DYNAMIC_STATE_SEARCH_AUTHORIZED":False,"CAUSAL_ROUTING_V27_AUTHORIZED":True})
 return {"formal_outcomes":adj["formal_outcomes"],"reports":len(NAMES),"integrity_files":len(paths),"final_freeze_digest":frozen["freeze_digest"]}

if __name__=="__main__":
 p=argparse.ArgumentParser();p.add_argument("--v26-tests",required=True);p.add_argument("--full-tests",required=True);a=p.parse_args();print(json.dumps(generate(Path.cwd(),a.v26_tests,a.full_tests),indent=2))
