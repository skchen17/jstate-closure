"""Integrate V25 interaction, convergence, readout, and realization evidence."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np,pandas as pd
from scipy.stats import spearmanr,pearsonr
from jclosure.protocol_v25 import verify,verify_stage,stage_freeze
from jclosure.provenance import sha256_file,write_json_atomic

SOURCE="src/jclosure/experiments/analyze_v25.py";OUT=Path("results/v25/processed");SCRATCH=Path("/data/CSK/J-space-project/v24-output-bottleneck-work")
def _rank(y):
 y=np.asarray(y,np.float64);s=np.linalg.svd(y,compute_uv=False);e=s*s;p=e/max(e.sum(),1e-30);c=np.cumsum(p);c[-1]=1
 return {"r90":min(int(np.searchsorted(c,.9)+1),min(y.shape)),"r95":min(int(np.searchsorted(c,.95)+1),min(y.shape)),"r99":min(int(np.searchsorted(c,.99)+1),min(y.shape)),"top1_fraction":float(p[0]),"first20":s[:20].tolist()}
def _views(y):
 y=np.asarray(y,np.float64);center=y-y.mean(0);return {"raw":_rank(y),"centered":_rank(center),"centered_normalized":_rank(center/np.maximum(np.linalg.norm(center,axis=1,keepdims=True),1e-12))}
def run(root:Path)->dict:
 cfg=verify(root)["config"];verify_stage(root,"interaction_estimand_amendment");verify_stage(root,"rank_convergence_amendment");d=json.loads((root/OUT/"design_v25.json").read_text());rs=json.loads((root/OUT/"rank_convergence_summary_v25.json").read_text());inter=json.loads((root/OUT/"distributed_interaction_summary_v25.json").read_text());corrected=json.loads((root/OUT/"interaction_estimand_amendment_v25.json").read_text());conv=pd.read_parquet(root/OUT/"layerwise_causal_convergence_corrected_v25.parquet");rank_audit=pd.read_parquet(root/OUT/"rank_construction_audit_v25.parquet")
 with np.load(root/OUT/"interaction_vectors_v25.npz") as z:raw=np.asarray(z["full_raw"],np.float64);broad=np.asarray(z["full_broad"],np.float64)
 with np.load(root/OUT/"corrected_interaction_vectors_v25.npz") as z:interaction=np.asarray(z["interaction_broad"],np.float64)
 with np.load(root/OUT/"late_hidden_projections_v25.npz") as z:idx=np.asarray(z["indices"],int);sign=np.asarray(z["sign"],np.float64)
 readout={"direct_full_hidden_2560":_views(raw),"direct_residual_256":_views(broad[:,:256]),"J_128":_views(broad[:,256:384]),"logits_32":_views(broad[:,384:416]),"semantic_32":_views(broad[:,416:448]),"workspace_96":_views(broad[:,448:544]),"random_vocab_512":_views(broad[:,544:])}
 for dim in cfg["late_projection_dimensions"]:readout[f"random_late_hidden_{dim}"]=_views(raw[:,idx[:dim]]*sign[:dim])
 interaction_rank={"full_effect":_views(broad),"interaction":_views(interaction)}
 write_json_atomic(root/OUT/"readout_compression_v25.json",readout);write_json_atomic(root/OUT/"interaction_rank_v25.json",interaction_rank)
 non=pd.read_parquet(root/OUT/"layerwise_nonlinear_interaction_v25.parquet");layer_inter=non.groupby("layer").interaction_ratio.median().reindex(cfg["layers"]);r95=conv.set_index("layer").centered_r95.reindex(cfg["layers"]);absolute=conv.set_index("layer").same_state_absolute_cosine.reindex(cfg["layers"])
 coupling={"interaction_vs_rank_reduction_spearman":float(spearmanr(layer_inter,r95.iloc[0]-r95).statistic),"interaction_vs_rank_reduction_pearson":float(pearsonr(layer_inter,r95.iloc[0]-r95).statistic),"interaction_vs_absolute_cosine_spearman":float(spearmanr(layer_inter,absolute).statistic),"interaction_vs_absolute_cosine_pearson":float(pearsonr(layer_inter,absolute).statistic),"wording":"association only; not interpreted as causal proof"}
 write_json_atomic(root/OUT/"interaction_convergence_coupling_v25.json",coupling)
 # Held-out all-family convergence audit on direct residual coordinates.
 family_profile={}
 for family in sorted({x["family"] for x in d["rank_audit_states"]}):
  blocks={0:[],31:[]}
  for item in [x for x in d["rank_audit_states"] if x["family"]==family]:
   z=np.load(root/SCRATCH/"finite_validation"/f"layer_{item['base_trial_id']}.npz")
   for state in ("P0","Pq"):
    for l in blocks:
     y=np.asarray(z[f"{state}_residual"][-32:,l],np.float64);blocks[l].append(y-y.mean(0))
   z.close()
  family_profile[family]={"early_r95":_rank(np.concatenate(blocks[0]))["r95"],"late_r95":_rank(np.concatenate(blocks[31]))["r95"]}
 early=int(conv.iloc[0].centered_r95);late=int(conv.iloc[-1].centered_r95);cos0=float(conv.iloc[0].same_state_absolute_cosine);cos1=float(conv.iloc[-1].same_state_absolute_cosine);family_replicated=all(x["late_r95"]<=x["early_r95"]*(1-cfg["convergence_gate"]["centered_r95_reduction_fraction_min"]) for x in family_profile.values())
 convergence_pass=bool(late<=early*(1-cfg["convergence_gate"]["centered_r95_reduction_fraction_min"]) and cos1-cos0>=cfg["convergence_gate"]["absolute_cosine_increase_min"] and family_replicated)
 direct_hidden=readout["direct_full_hidden_2560"]["centered"]["r95"];jrank=readout["J_128"]["centered"]["r95"];logrank=readout["logits_32"]["centered"]["r95"]
 readout_supported=bool(direct_hidden>=1.5*max(jrank,logrank));internal_supported=convergence_pass
 recon=pd.read_parquet(root/OUT/"same_j_reconstruction_v25.parquet");qual=[]
 for dim in cfg["realization_dimensions"]:
  sub=recon[(recon.method=="causal")&(recon.dimension==dim)];family_ok=all(((x.relative_l2.median()<=.3) and (x.cosine.median()>=.9) and (.8<=x.norm_ratio.median()<=1.2)) for _,x in sub.groupby("family"));
  if family_ok:qual.append(dim)
 realization=min(qual) if qual else None
 multi=pd.read_parquet(root/OUT/"three_component_interactions_v25.parquet");channel=pd.read_parquet(root/OUT/"channel_factorial_interaction_v25.parquet");order=pd.read_parquet(root/OUT/"intervention_order_v25.parquet")
 corrected_interaction={"broad_median_ratio":corrected["broad_interaction_ratio_median"],"bootstrap_2_5_lower":corrected["bootstrap_2_5_lower"],"family_medians":corrected["family_medians"],"additive_relative_l2_median":corrected["additive_relative_l2_median"],"interaction_cosine_median":corrected["interaction_cosine_median"],"STRONG_DISTRIBUTED_NONLINEAR_INTERACTION":corrected["STRONG_DISTRIBUTED_NONLINEAR_INTERACTION"],"Y11_vs_persistent_relative_l2_median":corrected["Y11_vs_persistent_relative_l2_median"],"writeback":corrected["writeback"]}
 corrected_cancellation={"index_median":corrected["cancellation_index_median"],"component_cosine_median":corrected["component_cosine_median"],"CAUSAL_CANCELLATION_SUPPORTED":corrected["CAUSAL_CANCELLATION_SUPPORTED"]}
 result={"rank_audit":rs["rank_audit"],"interaction_rank":interaction_rank,"interaction":corrected_interaction,"cancellation":corrected_cancellation,"background":inter["background"],"convergence":{"early_centered_r95":early,"late_centered_r95":late,"r95_reduction_fraction":float(1-late/max(early,1)),"early_absolute_cosine":cos0,"late_absolute_cosine":cos1,"family_profile":family_profile,"all_family_replicated":family_replicated,"MANY_TO_ONE_CAUSAL_CONVERGENCE":convergence_pass},"readout":{"metrics":readout,"READOUT_COMPRESSION_SUPPORTED":readout_supported,"INTERNAL_CAUSAL_CONVERGENCE_SUPPORTED":internal_supported},"coupling":coupling,"many_to_one_mapping":json.loads((root/OUT/"many_to_one_mapping_corrected_v25.json").read_text()),"JVP_finite_contraction":json.loads((root/OUT/"jvp_finite_contraction_v25.json").read_text()),"channel_factorial":{"median_interaction_by_term":channel.groupby("interaction").interaction_ratio.median().to_dict()},"three_component":{"triple_interaction_ratio_median":float(multi.triple_interaction_ratio.median())},"intervention_order":{"relative_difference_median":float(order.order_relative_difference.median()),"relative_difference_max":float(order.order_relative_difference.max())},"causal_effect_realization_dimension":realization,"CAUSAL_EFFECT_REALIZATION_DIMENSION_IDENTIFIED":realization is not None,"multihorizon_status":"NOT_OPENED_NO_FROZEN_ALL_FAMILY_TWO_TOKEN_CONFIRMATORY_PANEL","historical_final_opened":False,"v25_independent_final_opened":False}
 target=root/OUT/"v25_integrated_analysis.json";write_json_atomic(target,result);frozen=stage_freeze(root,"analysis",[SOURCE,"artifacts/distributed_causal_interaction_v25_interaction_estimand_amendment.freeze.json","artifacts/distributed_causal_interaction_v25_rank_convergence_amendment.freeze.json","results/v25/processed/readout_compression_v25.json","results/v25/processed/interaction_rank_v25.json","results/v25/processed/interaction_convergence_coupling_v25.json",str(target.relative_to(root))],{"analysis_sha256":sha256_file(target),"historical_final_opened":False,"v25_independent_final_opened":False});return {"freeze_digest":frozen["freeze_digest"],**result}
if __name__=="__main__":print(json.dumps(run(Path.cwd()),indent=2))
