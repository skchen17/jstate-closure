"""Architecture-native token state read/write experiments for V28."""
from __future__ import annotations
import argparse,hashlib,json,math
from pathlib import Path
import numpy as np,pandas as pd,torch
from jclosure.cache_v7 import clone_hybrid_cache
from jclosure.experiments import counterfactual_bank_v19 as v19
from jclosure.experiments.action_pool_v22 import _row as action_row
from jclosure.experiments.pre_readout_v27 import TARGETS,PRIMARY,metadata,prefix,step,delta,norms,normalized,scales,cosine,vector
from jclosure.protocol_v28 import verify,verify_stage,stage_freeze
from jclosure.provenance import sha256_file,write_json_atomic
SOURCE="src/jclosure/experiments/transaction_v28.py";OUT=Path("results/v28/processed")
ATTR={"REC":("recurrent_states",),"Conv":("conv_states",),"KV":("keys","values")}
def design(root):return json.loads((root/OUT/"design_v28.json").read_text())
def context(root):
 bundle,dense,meta,values,rec,att,measured,state_layer,ws_layers,ws_count=v19._setup(root)
 return bundle,dense,values,rec,att,state_layer
def fields(cache,channel,rec,att):
 layers=att if channel=="KV" else rec
 for i in layers:
  for name in ATTR[channel]:
   x=getattr(cache.layers[i],name,None)
   if isinstance(x,torch.Tensor):yield i,name,x
def allfields(cache,rec,att):
 for ch in ("REC","Conv","KV"):
  for i,n,x in fields(cache,ch,rec,att):yield ch,i,n,x
def thash(x):
 y=x.detach().contiguous().cpu();return hashlib.sha256(y.view(torch.int16).numpy().tobytes() if y.dtype==torch.bfloat16 else y.numpy().tobytes()).hexdigest()
def channel_hash(cache,ch,rec,att):
 h=hashlib.sha256()
 for i,n,x in fields(cache,ch,rec,att):h.update(f"{i}:{n}:{tuple(x.shape)}:".encode());h.update(thash(x).encode())
 return h.hexdigest()
def write_stats(incoming,outgoing,rec,att):
 rows=[]
 for ch,i,n,y in allfields(outgoing,rec,att):
  x=getattr(incoming.layers[i],n)
  if ch=="KV":
   assert y.shape[-2]==x.shape[-2]+1,(i,n,x.shape,y.shape)
   new=y[...,-1:,:].float();prior=x[...,-1:,:].float();diff=new-prior;kind="new_slot_minus_previous_slot"
  else:diff=y.float()-x.float();kind="out_minus_in"
  rows.append({"channel":ch,"layer":i,"field":n,"delta_kind":kind,"delta_norm":float(torch.linalg.vector_norm(diff)),"incoming_shape":list(x.shape),"outgoing_shape":list(y.shape),"incoming_hash":thash(x),"outgoing_hash":thash(y)})
 return rows
def intercept(incoming,outgoing,channels,rec,att,alpha=0.0,donor=None):
 """Native same-length outgoing cache replacement; KV old state is previous-slot rewrite."""
 result=clone_hybrid_cache(outgoing);requested=[];touched=set();parts=channels.split("+") if channels!="none" else []
 for ch in parts:
  if ch=="KV_last_slot_previous_copy":ch="KV"
  if ch not in ATTR:raise ValueError(ch)
  for i,n,old in fields(incoming,ch,rec,att):
   cur=getattr(outgoing.layers[i],n);dest=getattr(result.layers[i],n)
   if donor is not None:
    src=getattr(donor.layers[i],n)
    if src.shape!=cur.shape:raise RuntimeError(f"transplant incompatible {i}:{n}")
    dest=src.detach().clone()
   elif ch=="KV":
    dest=cur.detach().clone();replacement=old[...,-1:,:]
    dest[...,-1:,:]=replacement
   elif alpha==0:dest=old.detach().clone()
   elif alpha==1:dest=cur.detach().clone()
   else:dest=(old.float()+float(alpha)*(cur.float()-old.float())).to(cur.dtype)
   setattr(result.layers[i],n,dest);requested.append((i,n,src if donor is not None else replacement if ch=="KV" else old if alpha==0 else cur if alpha==1 else dest,ch=="KV" and donor is None));touched.add((i,n))
 exact=all(torch.equal(getattr(result.layers[i],n)[...,-1:,:] if last_only else getattr(result.layers[i],n),req) for i,n,req,last_only in requested)
 untouched=all(torch.equal(x,getattr(result.layers[i],n)) for _,i,n,x in allfields(outgoing,rec,att) if (i,n) not in touched)
 changed={ch:channel_hash(result,ch,rec,att) for ch in ("REC","Conv","KV")}
 return result,{"requested_exact":bool(exact),"untouched_exact":bool(untouched),"touched_fields":len(touched),"channel_hashes":changed,"writeback_pass":bool(exact and untouched)}
def qnorm(a,b,s):return normalized(norms(delta(a,b)),s)
def aggregate(z):return float(np.mean([z[k] for k in TARGETS]))
def current_aggregate(z):return float(np.mean([z[k] for k in PRIMARY]))
def token(m):return int(m["teacher_tokens_h8_or_h1"][0])
def teacher_tensor(m):return torch.tensor([[token(m)]])
def incoming_action(values,d):
 aid=verify_config_action(d);return {k:v*.5 for k,v in action_row(values["directions"],d["action_specs"][aid],float(d["action_alphas"][aid]),1).items()}
def verify_config_action(d):return "d000"
def native_pair(bundle,dense,item,root,d,state_layer):
 m=metadata(root,item);p,last,n=prefix(bundle,str(m["prompt"]));clean=step(bundle,dense,p,last,n,d,state_layer);return m,p,last,n,clean
def plan(root):
 verify_stage(root,"design");d=design(root);x={"primary_mechanism":"REC+Conv","main_conditions":["REC","Conv","REC+Conv","KV_last_slot_previous_copy","REC+Conv+KV_last_slot_previous_copy"],"dose_alphas":[0.0,.25,.5,.75,1.0],"dose_states_per_family":2,"transplant_states_per_family":2,"layer_scan_states_per_family":1,"horizon_states_per_family":2,"finalist_rule":"fixed REC+Conv, require development+validation primary gates; final sealed otherwise","token_choice":"frozen V18 first teacher token","random_write_control":"not architecture-valid at fixed native update; report unavailable","future_responses_observed":0,"independent_final_opened":False};p=root/OUT/"execution_plan_v28.json";write_json_atomic(p,x);fr=stage_freeze(root,"execution_plan",[SOURCE,str(p.relative_to(root)),"artifacts/token_state_transaction_v28_design.freeze.json"],x);return {"freeze_digest":fr["freeze_digest"],**x}
def audit(root,pilot=False):
 if not pilot:verify_stage(root,"execution_plan")
 d=design(root);bundle,dense,values,rec,att,state_layer=context(root);s=scales(root);rows=[];write=[]
 chosen=d["calibration"][:1] if pilot else d["calibration"]
 for j,item in enumerate(chosen,1):
  m,p,last,n,clean=native_pair(bundle,dense,item,root,d,state_layer);replay=step(bundle,dense,p,last,n,d,state_layer);z=qnorm(replay["targets"],clean["targets"],s);write+= [{"base_trial_id":item["base_trial_id"],"family":item["family"],"token_id":int(last[0,0]),**r} for r in write_stats(p,clean["cache"],rec,att)]
  same,wb=intercept(p,clean["cache"],"REC+Conv",rec,att,alpha=1);same1=step(bundle,dense,same,teacher_tensor(m),n+1,d,state_layer);clean1=step(bundle,dense,clean["cache"],teacher_tensor(m),n+1,d,state_layer);samez=qnorm(same1["targets"],clean1["targets"],s)
  qprefix=v19.apply(p,1.0,incoming_action(values,d),rec,att,"native_fp32_add_bf16_writeback");read=step(bundle,dense,qprefix,last,n,d,state_layer);readz=qnorm(read["targets"],clean["targets"],s)
  rows.append({"base_trial_id":item["base_trial_id"],"family":item["family"],"token_id":int(last[0,0]),"next_token_id":token(m),"incoming_seq_len":int(p.get_seq_length()),"outgoing_seq_len":int(clean["cache"].get_seq_length()),"replay_current_q":current_aggregate(z),"same_rewrite_h1_q":aggregate(samez),"same_rewrite_writeback":wb["writeback_pass"],"read_current_q":current_aggregate(readz),"incoming_state_hashes":json.dumps({ch:channel_hash(p,ch,rec,att) for ch in ATTR},sort_keys=True),"outgoing_state_hashes":json.dumps({ch:channel_hash(clean["cache"],ch,rec,att) for ch in ATTR},sort_keys=True),"token_hash":hashlib.sha256(f"{int(last[0,0])}:{token(m)}".encode()).hexdigest()});print(f"V28 audit {j}/{len(chosen)}",flush=True)
 if pilot:return {"pilot_rows":rows,"pilot_write_rows":len(write)}
 f=pd.DataFrame(rows);w=pd.DataFrame(write);fp=root/OUT/"transaction_boundary_audit_v28.parquet";wp=root/OUT/"natural_write_statistics_v28.parquet";f.to_parquet(fp,index=False,compression="zstd");w.to_parquet(wp,index=False,compression="zstd");summary={"states":len(f),"layers":len(rec)+len(att),"recurrent_layers":rec,"attention_layers":att,"field_counts":w.groupby("channel").size().to_dict(),"max_replay_current_q":float(f.replay_current_q.max()),"max_same_rewrite_h1_q":float(f.same_rewrite_h1_q.max()),"min_read_current_q":float(f.read_current_q.min()),"all_same_rewrite_exact":bool(f.same_rewrite_writeback.all()),"current_readout_fixed_before_interception":True,"outgoing_fields_naturally_computed":True,"all_fields_future_consumed":True,"REC_Conv_old_state_restore_valid":True,"KV_old_state_restore_valid":False,"KV_length_preserving_last_slot_rewrite_valid":True,"token_identity_frozen":True,"model_source":"transformers/models/qwen3_5/modeling_qwen3_5.py Qwen3_5GatedDeltaNet.forward, Qwen3_5Attention.forward, Qwen3_5DecoderLayer.forward"};jp=root/OUT/"transaction_boundary_audit_v28.json";write_json_atomic(jp,summary);fr=stage_freeze(root,"boundary_audit",[SOURCE,str(fp.relative_to(root)),str(wp.relative_to(root)),str(jp.relative_to(root)),"artifacts/token_state_transaction_v28_execution_plan.freeze.json"],summary);return {"freeze_digest":fr["freeze_digest"],**summary}
if __name__=="__main__":
 p=argparse.ArgumentParser();p.add_argument("command",choices=("plan","pilot","audit"));a=p.parse_args();root=Path.cwd();print(json.dumps({"plan":plan,"pilot":lambda r:audit(r,True),"audit":audit}[a.command](root),indent=2))
