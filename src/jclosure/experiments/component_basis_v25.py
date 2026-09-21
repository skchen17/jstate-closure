"""Train-only raw-hidden distributed component bases for V25."""
from __future__ import annotations
import json
from pathlib import Path
from collections import defaultdict
import numpy as np,torch
from jclosure.cache_v7 import clone_hybrid_cache
from jclosure.experiments import action_pool_v22 as actions
from jclosure.experiments import counterfactual_bank_v19 as v19
from jclosure.experiments.mediation_v24 import _context,_full_capture
from jclosure.experiments.layer_response_v24 import _projectors
from jclosure.protocol_v25 import verify,verify_stage,stage_freeze
from jclosure.provenance import sha256_file,write_json_atomic

SOURCE="src/jclosure/experiments/component_basis_v25.py";OUT=Path("results/v25/processed")
def _basis(x,k):
 _,_,vh=np.linalg.svd(np.asarray(x,np.float64),full_matrices=False);b=vh[:k].T
 for j in range(b.shape[1]):
  p=int(np.argmax(np.abs(b[:,j])));b[:,j]*=1 if b[p,j]>=0 else -1
 return b.astype(np.float32)
def _complete(seed,leading,k=256):
 rng=np.random.default_rng(seed);q,_=np.linalg.qr(np.concatenate([leading,rng.standard_normal((leading.shape[0],k))],1));return q[:,:k].astype(np.float32)

@torch.no_grad()
def run(root:Path)->dict:
 verify_stage(root,"design");cfg=verify(root)["config"];d=json.loads((root/OUT/"design_v25.json").read_text());v24=json.loads((root/"results/v24/processed/bottleneck_design_v24.json").read_text());candidate=json.loads((root/"results/v24/processed/bottleneck_candidate_selection_v24.json").read_text())
 design,split,prompts,teachers,qmap,bundle,dense,values,rec,att,measured,state_layer,ws_layers,ws_count=_context(root);layer=int(candidate["diagnostic_layer"]);device=next(bundle.hf_model.parameters()).device;proj=_projectors(root,device);jids=torch.as_tensor(split["selected_j"],device=device);lids=torch.as_tensor(split["selected_logits"],device=device);scales={x:float(v) for x,v in split["target_scales"].items()}
 rows=[];clean=[];family=defaultdict(list);action_ids=v24["train_action_ids"][:32]
 for number,item in enumerate(d["component_basis_states"],1):
  base=item["base_trial_id"];token=int(teachers[base][0]);pref=v19._prefill_history(bundle,str(prompts[base]["prompt"]),measured,dense,state_layer);p0=clone_hybrid_cache(pref["cache"]);q=qmap[item["q_name"]];pq=v19.apply(p0,1.0,v19._q_row(values["directions"],q,float(q["alpha"])),rec,att,"native_fp32_add_bf16_writeback")
  for state in (p0,pq):
   baseline=_full_capture(root,bundle,dense,state,token,pref["prompt_length"],jids,lids,scales,ws_layers,ws_count,max(measured),cfg["layers"],cfg["architecture_layers"],proj,layer);clean.append(baseline["raw_hidden"])
   for action_id in action_ids:
    edited=v19.apply(state,1.0,actions._row(values["directions"],v24["action_specs"][action_id],float(v24["action_alphas"][action_id]),1),rec,att,"native_fp32_add_bf16_writeback")
    full=_full_capture(root,bundle,dense,edited,token,pref["prompt_length"],jids,lids,scales,ws_layers,ws_count,max(measured),cfg["layers"],cfg["architecture_layers"],proj,layer);delta=full["raw_hidden"]-baseline["raw_hidden"];rows.append(delta);family[item["family"]].append(delta)
  print(f"V25 component basis {number}/10",flush=True)
 rows=np.stack(rows);clean=np.stack(clean);causal=_basis(rows,256);variance_leading=_basis(clean-clean.mean(0),min(19,len(clean)-1));variance=_complete(250028,variance_leading);random=_complete(250029,np.zeros((2560,0),np.float32));family_bases={f:_basis(np.stack(x),min(256,len(x))) for f,x in family.items()}
 path=root/OUT/"distributed_component_bases_v25.npz";np.savez_compressed(path,causal_basis=causal,variance_basis=variance,random_basis=random,**{f"family_{f}":b for f,b in family_bases.items()})
 result={"candidate_layer":layer,"basis_rows":len(rows),"basis_states":20,"basis_actions":32,"causal_basis_dimension":256,"variance_observed_dimension":int(variance_leading.shape[1]),"basis_sha256":sha256_file(path),"component_split":{"B1":"causal modes 1..24","B2":"causal modes 25..48","B3":"orthogonal remainder of each realized effect"},"historical_final_opened":False,"v25_independent_final_opened":False}
 target=root/OUT/"component_basis_v25.json";write_json_atomic(target,result);frozen=stage_freeze(root,"component_basis",[SOURCE,"artifacts/distributed_causal_interaction_v25_design.freeze.json",str(path.relative_to(root)),str(target.relative_to(root))],result);return {"freeze_digest":frozen["freeze_digest"],**result}

if __name__=="__main__":print(json.dumps(run(Path.cwd()),indent=2))
