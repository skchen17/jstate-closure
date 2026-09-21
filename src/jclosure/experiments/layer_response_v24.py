"""Broad-target layerwise finite and exact-JVP causal response capture for V24."""
from __future__ import annotations
import argparse,json,os
from pathlib import Path
import numpy as np,pandas as pd,torch
from jclosure.cache_v7 import clone_hybrid_cache
from jclosure.experiments import action_pool_v22 as actions
from jclosure.experiments import counterfactual_bank_v19 as v19
from jclosure.experiments import operator_bank_v20 as v20_bank
from jclosure.experiments import response_operator_v20 as v20_response
from jclosure.experiments.jvp_v12 import _apply_direction
from jclosure.experiments.paired_geometry_v21 import _torch_stack
from jclosure.protocol_v24 import verify,verify_stage,stage_freeze
from jclosure.provenance import sha256_file,write_json_atomic
from jclosure.recorder import ActivationRecorder

SOURCE="src/jclosure/experiments/layer_response_v24.py"; OUT=Path("results/v24/processed"); SCRATCH=Path("/data/CSK/J-space-project/v24-output-bottleneck-work")

def prepare(root:Path)->dict:
 design=verify_stage(root,"design"); cfg=verify(root)["config"]
 payload={"development_base_states":50,"validation_base_states":25,"persistent_states_per_base":["P0","Pq"],
 "finite_train_actions":256,"finite_heldout_validation_actions":32,"jvp_operator_states":20,"jvp_actions":32,
 "layers":cfg["layers"],"architecture_layers":cfg["architecture_layers"],
 "target_bundles":{"T0":288,"T1_random_orthogonal_vocabulary_coordinates":cfg["targets"]["vocabulary_random_coordinate_dimension"],
 "T2_random_orthogonal_residual_coordinates_per_layer":cfg["targets"]["residual_random_coordinate_dimension"],
 "T3_attention_or_recurrent_plus_MLP_coordinates_per_selected_layer":2*cfg["targets"]["architecture_random_coordinate_dimension"],
 "T4":"balanced normalized T0+T1+T2"},"finite_sign":"positive calibrated action; sign is frozen and identical across probe scaling",
 "natural_transitions":"separate h2-minus-h1 clean teacher-forced diagnostic on ten development bases",
 "historical_final_opened":False,"v24_independent_final_opened":False,"design_digest":design["freeze_digest"]}
 target=root/OUT/"layer_response_design_v24.json"; write_json_atomic(target,payload)
 frozen=stage_freeze(root,"layer_response_design",[SOURCE,"artifacts/causal_output_bottleneck_v24_design.freeze.json",str(target)],payload)
 return {"freeze_digest":frozen["freeze_digest"],**payload}

def _context(root:Path):
 d=json.loads((root/OUT/"bottleneck_design_v24.json").read_text()); split=json.loads((root/"artifacts/compact_causal_response_operator_v20_splits.freeze.json").read_text())
 operator=json.loads((root/"artifacts/compact_causal_response_operator_v20_operator_design.freeze.json").read_text())
 prompts=v20_bank.prompt_index(root); teachers=v20_response._teacher_map(root); q={x["name"]:x for x in operator["q"]}
 bundle,dense,_,values,rec,att,measured,state_layer,ws_layers,ws_count=v19._setup(root)
 return d,split,prompts,teachers,q,bundle,dense,values,rec,att,measured,state_layer,ws_layers,ws_count

def _projectors(root:Path,device):
 z=np.load(root/OUT/"target_projections_v24.npz")
 return {k:torch.as_tensor(z[k],device=device) for k in z.files if k.endswith("indices") or k.endswith("sign")}

def _arch_modules(bundle,layers):
 attn=[]; mlp=[]
 for i in layers:
  layer=bundle.layers[i]; attn.append(layer.linear_attn if hasattr(layer,"linear_attn") else layer.self_attn); mlp.append(layer.mlp)
 return attn,mlp

def _capture_torch(bundle,dense,cache,token,prompt_length,jids,lids,scales,ws_layers,ws_count,main,all_layers,arch_layers,proj,detach=True):
 device=next(bundle.hf_model.parameters()).device; attn_modules,mlp_modules=_arch_modules(bundle,arch_layers)
 with ActivationRecorder(bundle.layers,at=all_layers,clone=False,detach=detach) as rr, ActivationRecorder(attn_modules,at=range(len(attn_modules)),clone=False,detach=detach) as ar, ActivationRecorder(mlp_modules,at=range(len(mlp_modules)),clone=False,detach=detach) as mr:
  output=bundle.hf_model(input_ids=torch.tensor([[token]],device=device),attention_mask=torch.ones((1,prompt_length+1),device=device,dtype=torch.long),past_key_values=cache,use_cache=True)
 logits=output.logits[0,-1].float(); hidden=rr.activations[main][0,-1].float(); j=dense.dense_state(hidden,main)
 workspace=torch.cat([rr.activations[i][0,-1].float()[:ws_count] for i in ws_layers])
 parts={"j":j[jids],"logits":logits[lids],"semantic_continuous":torch.log_softmax(logits,dim=-1)[lids],"workspace":workspace}
 residual=torch.stack([rr.activations[i][0,-1].float()[proj["residual_indices"]]*proj["residual_sign"] for i in all_layers])
 arch_attn=torch.stack([ar.activations[i][0,-1].float()[proj["architecture_indices"]]*proj["architecture_sign"] for i in range(len(arch_layers))])
 arch_mlp=torch.stack([mr.activations[i][0,-1].float()[proj["architecture_indices"]]*proj["architecture_sign"] for i in range(len(arch_layers))])
 vocab=logits[proj["vocabulary_indices"]]*proj["vocabulary_sign"]
 return {"residual":residual,"arch_attn":arch_attn,"arch_mlp":arch_mlp,"vocab":vocab,"t0":_torch_stack(parts,scales)},output.past_key_values

def _np(x): return {k:v.detach().cpu().numpy().astype(np.float32) for k,v in x.items()}

@torch.no_grad()
def run_finite(root:Path,role:str,limit:int|None=None)->dict:
 verify_stage(root,"layer_response_design"); cfg=verify(root)["config"]
 d,split,prompts,teachers,qmap,bundle,dense,values,rec,att,measured,state_layer,ws_layers,ws_count=_context(root)
 device=next(bundle.hf_model.parameters()).device; proj=_projectors(root,device); jids=torch.as_tensor(split["selected_j"],device=device); lids=torch.as_tensor(split["selected_logits"],device=device)
 scales={k:float(v) for k,v in split["target_scales"].items()}; items=d[f"{role}_states"][:limit]; action_ids=d["train_action_ids"]+(d["heldout_action_ids"] if role=="validation" else [])
 specs=d["action_specs"]; all_layers=cfg["layers"]; arch_layers=cfg["architecture_layers"]; outputs=[]
 for number,item in enumerate(items,1):
  path=root/SCRATCH/f"finite_{role}"/f"layer_{item['base_trial_id']}.npz"; path.parent.mkdir(parents=True,exist_ok=True)
  if path.exists(): print(f"V24 finite {role} {number}/{len(items)} exists",flush=True); outputs.append(path); continue
  base=item["base_trial_id"]; token=int(teachers[base][0]); clean=v19._prefill_history(bundle,str(prompts[base]["prompt"]),measured,dense,state_layer); p0=clone_hybrid_cache(clean["cache"])
  q=qmap[item["q_name"]]; pq=v19.apply(p0,1.0,v19._q_row(values["directions"],q,float(q["alpha"])),rec,att,"native_fp32_add_bf16_writeback")
  arrays={}
  for state_name,state in (("P0",p0),("Pq",pq)):
   baseline,next_cache=_capture_torch(bundle,dense,state,token,clean["prompt_length"],jids,lids,scales,ws_layers,ws_count,max(measured),all_layers,arch_layers,proj); baseline=_np(baseline)
   banks={k:[] for k in baseline}
   for ai,action_id in enumerate(action_ids,1):
    row=actions._row(values["directions"],specs[action_id],float(d["action_alphas"][action_id]),1)
    edited=v19.apply(state,1.0,row,rec,att,"native_fp32_add_bf16_writeback")
    value,_=_capture_torch(bundle,dense,edited,token,clean["prompt_length"],jids,lids,scales,ws_layers,ws_count,max(measured),all_layers,arch_layers,proj); value=_np(value)
    for k in banks: banks[k].append((value[k]-baseline[k]).astype(np.float16))
    if ai%64==0: print(f"V24 finite {role} base={number}/{len(items)} state={state_name} action={ai}/{len(action_ids)}",flush=True)
   for k,v in banks.items(): arrays[f"{state_name}_{k}"]=np.stack(v)
   for k,v in baseline.items(): arrays[f"{state_name}_baseline_{k}"]=v.astype(np.float16)
  for k in ("residual","arch_attn","arch_mlp","vocab","t0"): arrays[f"q_{k}"]=(arrays[f"Pq_baseline_{k}"].astype(np.float32)-arrays[f"P0_baseline_{k}"].astype(np.float32)).astype(np.float16)
  tmp=path.with_suffix(".tmp.npz"); np.savez_compressed(tmp,**arrays,action_ids=np.asarray(action_ids),base_trial_id=np.asarray(base),family=np.asarray(item["family"]),q_name=np.asarray(item["q_name"])); os.replace(tmp,path); outputs.append(path)
 return {"role":role,"completed":len(outputs),"actions":len(action_ids),"operator_states":2*len(outputs)}

def run_jvp(root:Path,limit:int|None=None)->dict:
 verify_stage(root,"layer_response_design"); cfg=verify(root)["config"]
 d,split,prompts,teachers,qmap,bundle,dense,values,rec,att,measured,state_layer,ws_layers,ws_count=_context(root)
 for p in bundle.hf_model.parameters(): p.requires_grad_(False)
 device=next(bundle.hf_model.parameters()).device; proj=_projectors(root,device); jids=torch.as_tensor(split["selected_j"],device=device); lids=torch.as_tensor(split["selected_logits"],device=device)
 scales={k:float(v) for k,v in split["target_scales"].items()}; items=d["jvp_states"][:limit]; action_ids=d["train_action_ids"][:32]; specs=d["action_specs"]
 all_layers=cfg["layers"]; arch_layers=cfg["architecture_layers"]; outputs=[]
 for number,item in enumerate(items,1):
  path=root/SCRATCH/"jvp"/f"layer_jvp_{item['base_trial_id']}.npz"; path.parent.mkdir(parents=True,exist_ok=True)
  if path.exists(): outputs.append(path); continue
  base=item["base_trial_id"]; token=int(teachers[base][0]); clean=v19._prefill_history(bundle,str(prompts[base]["prompt"]),measured,dense,state_layer); p0=clone_hybrid_cache(clean["cache"]); q=qmap[item["q_name"]]; pq=v19.apply(p0,1.0,v19._q_row(values["directions"],q,float(q["alpha"])),rec,att,"native_fp32_add_bf16_writeback")
  arrays={}
  for state_name,state in (("P0",p0),("Pq",pq)):
   banks={k:[] for k in ("residual","arch_attn","arch_mlp","vocab","t0")}
   for ai,action_id in enumerate(action_ids,1):
    row=actions._row(values["directions"],specs[action_id],float(d["action_alphas"][action_id]),1)
    def target(eps):
     edited=_apply_direction(state,eps,row,rec,att); value,_=_capture_torch(bundle,dense,edited,token,clean["prompt_length"],jids,lids,scales,ws_layers,ws_count,max(measured),all_layers,arch_layers,proj,detach=False)
     return tuple(value[k] for k in ("residual","arch_attn","arch_mlp","vocab","t0"))
    eps=torch.zeros((),device=device,dtype=torch.float32); _,der=torch.func.jvp(target,(eps,),(torch.ones_like(eps),))
    for k,v in zip(banks,der,strict=True): banks[k].append(v.detach().cpu().numpy().astype(np.float16))
    if ai%8==0: print(f"V24 JVP base={number}/{len(items)} state={state_name} action={ai}/32",flush=True)
   for k,v in banks.items(): arrays[f"{state_name}_{k}"]=np.stack(v)
  tmp=path.with_suffix(".tmp.npz"); np.savez_compressed(tmp,**arrays,action_ids=np.asarray(action_ids),base_trial_id=np.asarray(base),family=np.asarray(item["family"]),q_name=np.asarray(item["q_name"])); os.replace(tmp,path); outputs.append(path)
 return {"completed":len(outputs),"operator_states":2*len(outputs),"actions":32}

@torch.no_grad()
def run_natural(root:Path)->dict:
 verify_stage(root,"layer_response_design"); cfg=verify(root)["config"]
 d,split,prompts,teachers,qmap,bundle,dense,values,rec,att,measured,state_layer,ws_layers,ws_count=_context(root)
 device=next(bundle.hf_model.parameters()).device; proj=_projectors(root,device); jids=torch.as_tensor(split["selected_j"],device=device); lids=torch.as_tensor(split["selected_logits"],device=device)
 scales={k:float(v) for k,v in split["target_scales"].items()}; outputs=[]
 for number,item in enumerate(d["natural_transition_states"],1):
  path=root/SCRATCH/"natural"/f"natural_{item['base_trial_id']}.npz"; path.parent.mkdir(parents=True,exist_ok=True)
  if path.exists(): outputs.append(path); continue
  old=v19._state_metadata(root,item); tokens=[int(x) for x in old["teacher_tokens_h8_or_h1"][:2]]
  if len(tokens)<2: raise RuntimeError("natural transition requires two frozen teacher tokens")
  clean=v19._prefill_history(bundle,str(old["prompt"]),measured,dense,state_layer)
  h1,next_cache=_capture_torch(bundle,dense,clean["cache"],tokens[0],clean["prompt_length"],jids,lids,scales,ws_layers,ws_count,max(measured),cfg["layers"],cfg["architecture_layers"],proj)
  h2,_=_capture_torch(bundle,dense,next_cache,tokens[1],clean["prompt_length"]+1,jids,lids,scales,ws_layers,ws_count,max(measured),cfg["layers"],cfg["architecture_layers"],proj)
  arrays={f"natural_{k}":(h2[k]-h1[k]).detach().cpu().numpy().astype(np.float16) for k in h1}
  np.savez_compressed(path,**arrays,base_trial_id=np.asarray(item["base_trial_id"]),family=np.asarray(item["family"])); outputs.append(path); print(f"V24 natural {number}/10",flush=True)
 return {"completed":len(outputs),"transition_count":len(outputs)}

def summarize(root:Path)->dict:
 d=json.loads((root/OUT/"bottleneck_design_v24.json").read_text()); rows=[]
 for role in ("development","validation"):
  for x in d[f"{role}_states"]:
   p=root/SCRATCH/f"finite_{role}"/f"layer_{x['base_trial_id']}.npz"
   if not p.exists(): raise RuntimeError(f"missing {p}")
   rows.append({"kind":"finite","role":role,**x,"path":str(p),"sha256":sha256_file(p)})
 for x in d["jvp_states"]:
  p=root/SCRATCH/"jvp"/f"layer_jvp_{x['base_trial_id']}.npz"
  if not p.exists(): raise RuntimeError(f"missing {p}")
  rows.append({"kind":"JVP","role":"development",**x,"path":str(p),"sha256":sha256_file(p)})
 for x in d["natural_transition_states"]:
  p=root/SCRATCH/"natural"/f"natural_{x['base_trial_id']}.npz"
  if not p.exists(): raise RuntimeError(f"missing {p}")
  rows.append({"kind":"natural","role":"development",**x,"path":str(p),"sha256":sha256_file(p)})
 frame=pd.DataFrame(rows); target=root/OUT/"layer_response_index_v24.parquet"; frame.to_parquet(target,index=False,compression="zstd")
 result={"finite_development_bases":50,"finite_validation_bases":25,"finite_operator_states":150,"jvp_base_states":10,"jvp_operator_states":20,
 "natural_transition_count":10,"index_sha256":sha256_file(target),"historical_final_opened":False,"v24_independent_final_opened":False}; write_json_atomic(root/OUT/"layer_response_summary_v24.json",result); return result

if __name__=="__main__":
 p=argparse.ArgumentParser(); p.add_argument("stage",choices=("prepare","finite","jvp","natural","summarize")); p.add_argument("--role",choices=("development","validation")); p.add_argument("--limit",type=int); a=p.parse_args(); root=Path.cwd()
 if a.stage=="prepare": ans=prepare(root)
 elif a.stage=="finite": ans=run_finite(root,a.role,a.limit)
 elif a.stage=="jvp": ans=run_jvp(root,a.limit)
 elif a.stage=="natural": ans=run_natural(root)
 else: ans=summarize(root)
 print(json.dumps(ans,indent=2))
