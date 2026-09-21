"""Append-only V25 correction for exact-zero response matrices."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np,pandas as pd
from jclosure.protocol_v25 import verify,verify_stage,stage_freeze
from jclosure.provenance import sha256_file,write_json_atomic

SOURCE="src/jclosure/experiments/zero_rank_amendment_v25.py";OUT=Path("results/v25/processed");SCRATCH=Path("/data/CSK/J-space-project/v24-output-bottleneck-work")
def _rank(y):
 s=np.linalg.svd(np.asarray(y,np.float64),compute_uv=False);e=s*s
 if float(e.sum())<=1e-20:return 0
 c=np.cumsum(e/e.sum());c[-1]=1;return min(int(np.searchsorted(c,.95)+1),min(y.shape))
def run(root:Path)->dict:
 verify_stage(root,"analysis");cfg=verify(root)["config"];d=json.loads((root/OUT/"design_v25.json").read_text());source=pd.read_parquet(root/OUT/"layerwise_causal_convergence_v25.parquet");corrected=source.copy();zero=corrected.participation_ratio<=1e-20
 for name in ("centered_r90","centered_r95","centered_r99"):corrected.loc[zero,name]=0;corrected[name]=corrected[name].clip(upper=256).astype(int)
 active=corrected[~zero];reference=int(active.iloc[0].layer);late=int(active.iloc[-1].layer);early_r95=int(active.iloc[0].centered_r95);late_r95=int(active.iloc[-1].centered_r95);early_cos=float(active.iloc[0].same_state_absolute_cosine);late_cos=float(active.iloc[-1].same_state_absolute_cosine)
 family={}
 for fam in sorted({x["family"] for x in d["rank_audit_states"]}):
  blocks={reference:[],late:[]}
  for item in [x for x in d["rank_audit_states"] if x["family"]==fam]:
   z=np.load(root/SCRATCH/"finite_validation"/f"layer_{item['base_trial_id']}.npz")
   for state in ("P0","Pq"):
    for l in blocks:
     y=np.asarray(z[f"{state}_residual"][-32:,l],np.float64);blocks[l].append(y-y.mean(0))
   z.close()
  family[fam]={"reference_layer":reference,"reference_r95":_rank(np.concatenate(blocks[reference])),"late_layer":late,"late_r95":_rank(np.concatenate(blocks[late]))}
 family_converged=all(x["late_r95"]<=x["reference_r95"]*(1-cfg["convergence_gate"]["centered_r95_reduction_fraction_min"]) for x in family.values())
 convergence=bool(late_r95<=early_r95*(1-cfg["convergence_gate"]["centered_r95_reduction_fraction_min"]) and late_cos-early_cos>=cfg["convergence_gate"]["absolute_cosine_increase_min"] and family_converged)
 path=root/OUT/"layerwise_causal_convergence_zero_corrected_v25.parquet";corrected.to_parquet(path,index=False,compression="zstd")
 result={"reason":"exact-zero matrices have rank zero; cumulative-energy search on a zero spectrum previously returned ambient dimension","zero_response_layers":corrected[zero].layer.astype(int).tolist(),"first_nonzero_layer":reference,"late_layer":late,"reference_r95":early_r95,"late_r95":late_r95,"r95_reduction_fraction":float(1-late_r95/max(early_r95,1)),"reference_absolute_cosine":early_cos,"late_absolute_cosine":late_cos,"absolute_cosine_change":float(late_cos-early_cos),"family_profile":family,"all_family_converged":family_converged,"MANY_TO_ONE_CAUSAL_CONVERGENCE":convergence,"INTERNAL_CAUSAL_CONVERGENCE_SUPPORTED":False,"many_to_one_mapping_status":"INCONCLUSIVE: frozen mapping source layers 0/7/15/23 have exact-zero observed responses","contraction_status":"layers 0-23 are zero-response; 23-to-24 is response emergence, not contraction; finite/JVP ratios reported only descriptively from layer 24 onward","original_records_overwritten":False,"corrected_profile_sha256":sha256_file(path),"historical_final_opened":False,"v25_independent_final_opened":False}
 target=root/OUT/"zero_rank_amendment_v25.json";write_json_atomic(target,result);frozen=stage_freeze(root,"zero_rank_amendment",[SOURCE,"artifacts/distributed_causal_interaction_v25_analysis.freeze.json",str(path.relative_to(root)),str(target.relative_to(root))],result);return {"freeze_digest":frozen["freeze_digest"],**result}
if __name__=="__main__":print(json.dumps(run(Path.cwd()),indent=2))
