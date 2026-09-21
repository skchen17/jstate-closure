"""Architecture diagnostics and formal V24 adjudication."""
from __future__ import annotations
import json
from pathlib import Path
from collections import defaultdict
import numpy as np,pandas as pd
from jclosure.protocol_v24 import verify,verify_stage
from jclosure.provenance import sha256_file,write_json_atomic
from jclosure.experiments.analyze_bottleneck_v24 import _ranks,_coverage

OUT=Path("results/v24/processed"); SCRATCH=Path("/data/CSK/J-space-project/v24-output-bottleneck-work")
def run(root:Path)->dict:
 cfg=verify(root)["config"]; verify_stage(root,"mediation_protocol"); d=json.loads((root/OUT/"bottleneck_design_v24.json").read_text()); val=json.loads((root/OUT/"output_bottleneck_validation_v24.json").read_text()); med=json.loads((root/OUT/"causal_bottleneck_mediation_v24.json").read_text()); proto=json.loads((root/OUT/"mediation_protocol_v24.json").read_text())
 layer=int(val["candidate_layer"]); arch_layer=min(cfg["architecture_layers"],key=lambda x:abs(x-layer)); ai=cfg["architecture_layers"].index(arch_layer); rank_rows=[]; channel_norm=defaultdict(list); q_rows=[]
 ids=d["train_action_ids"]
 for item in d["development_states"]:
  z=np.load(root/SCRATCH/"finite_development"/f"layer_{item['base_trial_id']}.npz")
  for state in ("P0","Pq"):
   for target in ("arch_attn","arch_mlp"):
    y=np.asarray(z[f"{state}_{target}"][:256,ai],np.float64); r,s,e=_ranks(y.T); rank_rows.append({"base_trial_id":item["base_trial_id"],"family":item["family"],"state":state,"layer":arch_layer,"component":target,"r90":r[0],"r95":r[1],"r99":r[2],"stable_rank":s,"entropy_rank":e})
    for idx,action_id in enumerate(ids): channel_norm[(target,d["action_channel_tags"][action_id])].append(float(np.linalg.norm(y[idx])))
   q_rows.append(np.asarray(z["q_residual"][layer],np.float64))
  z.close()
 frame=pd.DataFrame(rank_rows); path=root/OUT/"architecture_localization_v24.parquet"; frame.to_parquet(path,index=False,compression="zstd")
 architecture={"nearest_architecture_layer":arch_layer,"component_r95_median":{name:float(frame[frame.component==name].r95.median()) for name in ("arch_attn","arch_mlp")},
  "channel_effect_norm_median":{f"{target}:{tag}":float(np.median(x)) for (target,tag),x in channel_norm.items()},"wording":"Architecture-resolved descriptive localization; no global channel ranking is inferred from one perturbation family.","rows_sha256":sha256_file(path)}
 with np.load(root/OUT/"projected_bottleneck_bases_v24.npz") as b:
  basis=np.asarray(b["global_basis"][:,:int(val["candidate_k"])],np.float64)
 qcoverage=_coverage(np.stack(q_rows),basis)
 A=bool(val["LOW_RANK_OUTPUT_BOTTLENECK_REPRESENTATIONAL_PASS"]); causal=bool(med["CAUSAL_MEDIATION_GATE_PASS"]); B=bool(A and causal and False)
 moving=bool(A and val["global_vs_local_oracle_residual_gain"]>=.10 and val["same_J_orientation"]["median_angle_degrees"]>=15)
 global_ok=bool(A and val["heldout_global_pass"])
 profile=val["compression_profile"]; readout_only=bool(not A and profile["late_T0_r95"]<=.6*profile["early_hidden_broad_r95"])
 distributed=not causal; regeneration_ratio=med["regeneration"]["later_projected_ratio_median"]
 regenerated=bool(regeneration_ratio is not None and med["PERP_ONLY"]["retained_ratio"]>cfg["causal_gate"]["perp_retained_ratio_max"] and regeneration_ratio>.5)
 outcomes=[]
 if A: outcomes.append("V24-A_LOW_RANK_OUTPUT_BOTTLENECK_CONFIRMED")
 if B: outcomes.append("V24-B_CAUSAL_BOTTLENECK_MEDIATES_PERSISTENT_EFFECTS")
 if moving: outcomes.append("V24-C_MOVING_CAUSAL_BOTTLENECK")
 if global_ok: outcomes.append("V24-D_GLOBAL_CAUSAL_BOTTLENECK")
 if readout_only: outcomes.append("V24-E_READOUT_LOW_RANK_ONLY")
 if distributed: outcomes.append("V24-F_DISTRIBUTED_CAUSAL_MEDIATION")
 if regenerated: outcomes.append("V24-G_BOTTLENECK_REGENERATED_FROM_ORTHOGONAL_STATE")
 if not outcomes: outcomes=["V24-F_DISTRIBUTED_CAUSAL_MEDIATION"]
 result={"formal_outcomes":outcomes,"primary_outcome":"+".join(outcomes),"V24_A":A,"V24_B":B,"V24_C":moving,"V24_D":global_ok,"V24_E":readout_only,"V24_F":distributed,"V24_G":regenerated,
  "architecture_localization":architecture,"workspace_relationship":{"J_from_B_train_relative_l2":proto["workspace_relation_train"]["J_from_B_train_relative_l2"],"same_J_q_effect_bottleneck_coverage":qcoverage,
  "WORKSPACE_ALIASING_IS_MEDIATED_BY_DOWNSTREAM_CAUSAL_BOTTLENECK":bool(B and med["restoration"]["intervention_mediated_fraction_median"]>=cfg["causal_gate"]["mediation_fraction_min"])},
  "independent_final_status":"FROZEN_AND_UNOPENED_NO_SINGLE_DEVELOPMENT_VALIDATION_MEDIATOR","historical_final_opened":False,"v24_independent_final_opened":False,
  "H2_REMAINS":True,"H3_AUTHORIZED":False,"DYNAMIC_STATE_SEARCH_AUTHORIZED":False,
  "wording":"Causal mediator is not equated with complete Markov state. Representational low rank and causal mediation are adjudicated separately."}
 write_json_atomic(root/OUT/"v24_adjudication.json",result); return result
if __name__=="__main__": print(json.dumps(run(Path.cwd()),indent=2))
