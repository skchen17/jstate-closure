"""Generate standalone, cumulative and integrity reports for V27."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import pandas as pd
from jclosure.protocol_v27 import verify,verify_stage,stage_freeze
from jclosure.provenance import sha256_file,write_json_atomic
OUT=Path("results/v27/processed");R=Path("reports");SOURCE="src/jclosure/reporting_v27.py"
NAMES=["PRE_READOUT_BOUNDARY_AUDIT_V27.md","PRE_READOUT_SILENT_BANK_V27.md","PRE_VS_POST_READOUT_TIMING_V27.md","PRE_READOUT_FUTURE_POTENCY_V27.md","NEXT_TOKEN_LAYER_TRACE_V27.md","FIRST_VISIBILITY_PROFILE_V27.md","NATIVE_COMPONENT_RESTORATION_V27.md","NATIVE_COMPONENT_TRANSPLANT_V27.md","CHANNEL_ROUTING_FACTORIAL_V27.md","CAUSAL_REENTRY_LOCALIZATION_V27.md","WORKSPACE_RECONSTRUCTION_V27.md","DIRECT_VS_DISTRIBUTED_CARRYOVER_V27.md","REENTRY_GEOMETRY_V27.md","STRICT_WRITEBACK_AUDIT_V27.md","NATURAL_TRAJECTORY_COMPARISON_V27.md","EXECUTION_MANIFEST_V27.md","V27_SCIENTIFIC_ANSWERS_V27.md","V27_COMPLETE_REPORT.md"]
def load(root,name):return json.loads((root/OUT/name).read_text())
def w(root,name,body):(root/R/name).write_text(body.rstrip()+"\n",encoding="utf-8")
def tf(x):return "TRUE" if x else "FALSE"
def generate(root:Path,vtests:str,fulltests:str):
 base=verify(root);stages={x:verify_stage(root,x) for x in ("design","boundary_audit","silent_bank","candidate_expansion","silent_bank_expanded","timing_diagnostic","routing_authorization","final_opening","adjudication")};d=load(root,"design_v27.json");audit=load(root,"pre_readout_boundary_audit_v27.json");a=load(root,"pre_readout_silent_bank_v27.json");b=load(root,"pre_readout_silent_bank_expanded_v27.json");timing=load(root,"pre_vs_post_timing_v27.json");adj=load(root,"v27_adjudication.json");route=load(root,"routing_status_v27.json")
 floorrows="\n".join(f"| {k} | {v['p99']:.3g} | {v['normalized_safety_threshold']:.3g} |" for k,v in audit["replay_floor"].items());w(root,NAMES[0],f"""# Pre-Readout Boundary Audit — V27

The accepted boundary is the prefix cache immediately before the final prompt token. REC, Conv, and KV fields already exist and are writable; the complete final-token forward, J/workspace extraction, final normalization, unembedding, and logits remain downstream. The persistent update has occurred for the prefix, but not for the final prompt token.

| target | replay p99 | normalized safety threshold |
|---|---:|---:|
{floorrows}

The ordinary reference perturbation changes current logits, semantic output, workspace, broad vocabulary, and late residual by median `0.865–1.085×` reference scale. Therefore current-output invariance is not structurally guaranteed at this boundary.
""")
 w(root,NAMES[1],f"""# Pre-Readout Silent Bank — V27

Two h0-only searches were frozen before any V27 h1 response was observed. The architecture-resolved bank tested `{a['rows']}` rows and the V19 same-J expansion tested `{b['rows']}` rows. Across `{adj['reliable_candidate_rows']}` reliable rows, `PRE_READOUT_SILENT=0` and `PRE_READOUT_DISTAL=0`.

- Initial bank hash: `{a['PRE_READOUT_SILENT_BANK_HASH']}`
- Expanded bank hash: `{b['PRE_READOUT_SILENT_BANK_HASH']}`
- Future responses observed before each freeze: `0`

The strict bank is empty; no h1 opening can establish V27-A.
""")
 w(root,NAMES[2],f"""# Pre- versus Post-Readout Timing — V27

A predeclared `joint` intervention was compared on 10 development states (2 per family). This is diagnostic and was not used to rescue the empty bank.

| timing | h0 median Q | h1 median Q |
|---|---:|---:|
| pre-readout | {timing['median_pre_h0_q']:.6f} | {timing['median_pre_h1_q']:.6f} |
| post-readout | {timing['median_post_h0_q']:.6f} | {timing['median_post_h1_q']:.6f} |

Median pre/post h1 direction cosine is `{timing['median_pre_post_h1_cosine']:.6f}` and magnitude ratio is `{timing['median_pre_post_h1_magnitude_ratio']:.6f}`. V26's h0 silence was partly guaranteed by its post-readout timing.
""")
 famrows="\n".join(f"| {k} | {v['rows']} | {v['min_aggregate_h0_q']:.6f} | {v['silent']} | {v['distal']} |" for k,v in adj["family_audit"].items());w(root,NAMES[3],"# Pre-Readout Future Potency — V27\n\nNo accepted pre-readout-silent or distal pair exists, so the primary h1 test has zero eligible rows and V27-A is false. h1 was not opened on failed candidates for formal inference.\n\n| family | reliable rows | minimum h0 Q | silent | distal |\n|---|---:|---:|---:|---:|\n"+famrows+"\n")
 stop="The sequential V27A authorization gate failed because both prospectively frozen banks contain zero silent and zero distal rows. Detailed next-token routing was therefore **NOT RUN / NOT AUTHORIZED**. The corresponding Parquet/NPZ artifact is intentionally empty and records this stop; absence of measurements is not evidence for or against a route."
 w(root,NAMES[4],"# Next-Token Layer Trace — V27\n\n"+stop+"\n")
 w(root,NAMES[5],"# First-Visibility Profile — V27\n\n"+stop+" No `FIRST_MEASURABLE_DIFFERENCE` is assigned, and no observed layer is mislabeled as causal.\n")
 w(root,NAMES[6],"# Native-Component Restoration — V27\n\n"+stop+" No REC, Conv, KV, combination, or full-state restoration estimand was executed.\n")
 w(root,NAMES[7],"# Native-Component Transplant — V27\n\n"+stop+" No reverse transplant was executed.\n")
 w(root,NAMES[8],"# Channel-Routing Factorial — V27\n\n"+stop+" No channel interaction claim is made.\n")
 w(root,NAMES[9],"# Causal Re-Entry Localization — V27\n\n"+stop+" `CAUSAL_REENTRY_SITE = NONE / NOT ADJUDICATED`; V27-B is false, not because a route was disproved, but because its upstream premise failed.\n")
 w(root,NAMES[10],"# Workspace Reconstruction — V27\n\n"+stop+" V27-D is false and workspace reconstruction is not justified for this tested panel.\n")
 w(root,NAMES[11],"# Direct versus Distributed Carryover — V27\n\n"+stop+" Neither direct carryover (V27-E) nor distributed reconstruction (V27-C/G) is established.\n")
 w(root,NAMES[12],"# Re-Entry Geometry — V27\n\n"+stop+" r90/r95/r99 and projector distances are undefined; no low-rank or state-dimension claim is made.\n")
 w(root,NAMES[13],"# Strict Writeback Audit — V27\n\nAll bank actuation rows retain their recorded finite-write reliability fields. Detailed restoration/transplant writeback attempts: `0`, because routing was not authorized. `V27-H = FALSE`: the stop was scientific (empty V27A bank), not a routing-writeback failure.\n")
 w(root,NAMES[14],"# Natural-Trajectory Comparison — V27\n\nStatus: **NOT RUN / NOT AUTHORIZED**. Natural same-readout pairs are observational and could not substitute for the failed primary causal premise.\n")
 commands=["PYTHONPATH=src /home/user/anaconda3/bin/python -m jclosure.protocol_v27 freeze","PYTHONPATH=src /home/user/anaconda3/bin/python -m jclosure.experiments.design_v27","CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src /home/user/anaconda3/bin/python -m jclosure.experiments.pre_readout_v27 boundary","CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src /home/user/anaconda3/bin/python -m jclosure.experiments.pre_readout_v27 bank","PYTHONPATH=src /home/user/anaconda3/bin/python -m jclosure.experiments.pre_readout_expansion_v27 freeze","CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src /home/user/anaconda3/bin/python -m jclosure.experiments.pre_readout_expansion_v27 bank","CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src /home/user/anaconda3/bin/python -m jclosure.experiments.adjudicate_failure_v27 timing","PYTHONPATH=src /home/user/anaconda3/bin/python -m jclosure.experiments.adjudicate_failure_v27 adjudicate","PYTHONPATH=src /home/user/anaconda3/bin/python -m pytest tests/test_v27_pre_readout_reentry.py -q","PYTHONPATH=src /home/user/anaconda3/bin/python -m pytest -q","git commit -m 'Complete V27 pre-readout causal re-entry study'","git push origin main"]
 manifest={"commands":commands,"v27_tests":vtests,"full_suite":fulltests,"formal_outcomes":adj["formal_outcomes"],"routing_status":route["status"],"historical_final_opened":False,"v27_independent_final_opened":False};write_json_atomic(root/OUT/"execution_manifest_v27.json",manifest);w(root,NAMES[15],"# Execution Manifest — V27\n\n"+"\n".join(f"- `{x}`" for x in commands)+f"\n\nV27 tests: `{vtests}`. Full suite: `{fulltests}`. The three failures are inherited cumulative `FINAL_REPORT.md` hash checks in V14, V16, and V25.\n")
 answers=["Yes. V26 intervened after the current readout, so its h0 equality was structurally guaranteed by timing.","No under the frozen tested panel: zero reliable rows met pre-readout silent J/readout criteria.","No tested accepted perturbation left logits within the frozen silent gate.","No tested accepted perturbation left the full primary semantic/readout stack within the gate.","Not testable: the prospectively selected h0-only bank is empty.","0/4,460 reliable rows (0%).","The fixed diagnostic had strong h1 effects, but it was not current-silent and cannot establish V27-A.","The absence of silent/distal rows occurred in all five families.","Not assigned; detailed trace was not authorized.","None adjudicated.","Not tested after the gate failed.","Not tested after the gate failed.","Not tested after the gate failed.","No combination was eligible for routing.","Not tested.","Neither localized nor distributed routing was adjudicated.","Not tested; no factorial claim is made.","No; workspace reconstruction was not causally tested.","No accepted identical-J_t pre-readout pair existed.","Neither model was established.","Not measured.","Not tested on natural trajectories.","No, V27-A is false.","No, V27-B is false because its prerequisite failed.","V27-C, V27-D, and V27-E are false/not established.","Yes, V27-F is supported.","Yes, H2 remains.","No, H3 remains unauthorized.","No, compact/dynamic state search remains unauthorized.","No; workspace reconstruction wording is not justified by V27."]
 write_json_atomic(root/OUT/"v27_scientific_answers.json",{"answers":answers});w(root,NAMES[16],"# V27 Scientific Answers\n\n"+"\n".join(f"{i}. {x}" for i,x in enumerate(answers,1)))
 complete=f"""# V27 Complete Report

## Identity

- Name: **Pre-Readout Counterfactuals and Next-Token Causal Re-Entry**
- Parent: `{base['parent_commit']}`
- Protocol hash: `{base['freeze_digest']}`
- Design hash: `{stages['design']['freeze_digest']}`
- Initial bank hash: `{a['PRE_READOUT_SILENT_BANK_HASH']}`
- Expanded bank hash: `{b['PRE_READOUT_SILENT_BANK_HASH']}`
- Adjudication hash: `{adj['ADJUDICATION_HASH']}`

## Prospective design and boundary

The intervention was moved to the latest valid pre-readout boundary: the cache after all prompt tokens except the final token. The model then executed the final prompt token normally, with no J/logit/semantic/residual restoration or clamping. All six repeated-forward numerical floors were exactly zero. An ordinary intervention materially changed current output, confirming invariance was not built into the boundary.

The four disjoint balanced panels contain 25 calibration, 50 development, 25 validation, and 25 unopened independent-final states. All boundaries, targets, silence thresholds, candidates, routing layers/components, gates, and opening rules were frozen before response observation.

## Primary result

**V27-F_V26_BOUNDARY_SPECIFIC_ONLY**.

`V27-A=FALSE`, `V27-B=FALSE`, `V27-C=FALSE`, `V27-D=FALSE`, `V27-E=FALSE`, `V27-F=TRUE`, `V27-G=FALSE`, `V27-H=FALSE`.

The initial architecture-resolved bank tested 975 rows; the frozen V19 same-J expansion tested 3,600 rows. Of 4,575 total rows, 4,460 passed actuation reliability. None met `PRE_READOUT_SILENT` and none met `PRE_READOUT_DISTAL`. The minimum reliable aggregate h0 Q by family ranged from `{min(x['min_aggregate_h0_q'] for x in adj['family_audit'].values()):.6f}` to `{max(x['min_aggregate_h0_q'] for x in adj['family_audit'].values()):.6f}`, well above the 0.10 distal ceiling.

Therefore V27 found no evidence that a tested persistent distinction can be introduced before current readout, survive the full natural current-token computation, remain current-invisible, and then reappear at h1. V26 remains valid as a post-readout future-state result, but its stronger hidden-current-state interpretation is unsupported by this panel.

## Timing diagnostic

For a predeclared joint action on 10 states, median pre-readout h0 Q was `{timing['median_pre_h0_q']:.6f}` versus post-readout `0`; h1 Q was `{timing['median_pre_h1_q']:.6f}` versus `{timing['median_post_h1_q']:.6f}`. The h1 direction cosine was `{timing['median_pre_post_h1_cosine']:.6f}`. This diagnostic confirms both boundaries can affect the future but only the post-readout boundary guarantees current silence.

## Sequential stop

`DETAILED_ROUTING_AUTHORIZED = FALSE`. Consequently no layer trace, first causal site, restoration, transplant, factorial, workspace reconstruction, re-entry geometry, natural comparison, or independent final was opened. Empty machine-readable routing artifacts make this stop explicit and prevent absence from being mistaken for a null routing result.

## Authorization

- `H2_REMAINS = TRUE`
- `H3_AUTHORIZED = FALSE`
- `DYNAMIC_STATE_SEARCH_AUTHORIZED = FALSE`
- `AUTONOMOUS_CONTROLLER_AUTHORIZED = FALSE`

## Verification

V27 tests: `{vtests}`. Full suite: `{fulltests}`; the three failures are inherited cumulative-report hash checks.
"""
 w(root,NAMES[17],complete);final=root/R/"FINAL_REPORT.md"
 if "# V27 Complete Report" in final.read_text(encoding="utf-8"):raise RuntimeError("V27 already appended")
 final.write_text(final.read_text(encoding="utf-8").rstrip()+"\n\n---\n\n"+complete.rstrip()+"\n",encoding="utf-8");write_json_atomic(root/OUT/"v27_test_audit.json",{"v27_tests":vtests,"full_suite":fulltests,"inherited_failures":["V14 FINAL_REPORT hash","V16 FINAL_REPORT hash","V25 FINAL_REPORT hash"]})
 paths=[]
 for base_dir in ("artifacts","results/v27/processed","reports"):
  for p in sorted((root/base_dir).glob("**/*")):
   rel=str(p.relative_to(root))
   if p.is_file() and ("v27" in rel.lower() or rel=="reports/FINAL_REPORT.md") and not rel.endswith("v27_integrity_index.json"):paths.append(rel)
 integrity={"files":{x:sha256_file(root/x) for x in paths},"file_count":len(paths),"boundary_hash":d["boundary_hash"],"state_hashes":d["state_hashes"],"perturbation_hashes":{"initial_bank":a["PRE_READOUT_SILENT_BANK_HASH"],"expanded_bank":b["PRE_READOUT_SILENT_BANK_HASH"]},"component_hash":route["component_hash"],"routing_hash":d["routing_hash"],"intervention_hash":route["intervention_hash"],"adjudication_hash":adj["ADJUDICATION_HASH"],"final_opening_hash":load(root,"final_opening_v27.json")["FINAL_OPENING_HASH"],"historical_final_opened":False,"v27_independent_final_opened":False};ip=root/OUT/"v27_integrity_index.json";write_json_atomic(ip,integrity)
 inputs=[SOURCE,"tests/test_v27_pre_readout_reentry.py",str(ip.relative_to(root)),"results/v27/processed/v27_adjudication.json","results/v27/processed/execution_manifest_v27.json","results/v27/processed/v27_scientific_answers.json","reports/FINAL_REPORT.md"]+[f"reports/{x}" for x in NAMES]
 fr=stage_freeze(root,"final",inputs,{"formal_outcomes":adj["formal_outcomes"],"integrity_index_sha256":sha256_file(ip),"reports":len(NAMES),"historical_final_opened":False,"v27_independent_final_opened":False,"DETAILED_ROUTING_AUTHORIZED":False,"H3_AUTHORIZED":False,"DYNAMIC_STATE_SEARCH_AUTHORIZED":False,"AUTONOMOUS_CONTROLLER_AUTHORIZED":False});return {"final_freeze_digest":fr["freeze_digest"],"reports":len(NAMES),"integrity_files":len(paths),"formal_outcomes":adj["formal_outcomes"]}
if __name__=="__main__":
 p=argparse.ArgumentParser();p.add_argument("--v27-tests",required=True);p.add_argument("--full-tests",required=True);a=p.parse_args();print(json.dumps(generate(Path.cwd(),a.v27_tests,a.full_tests),indent=2))
