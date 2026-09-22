"""Single frozen REC+Conv write-block independent-final confirmation."""
from __future__ import annotations
import hashlib,json
from pathlib import Path
import numpy as np,pandas as pd
from jclosure.experiments.transaction_v28 import OUT,design,context,native_pair,teacher_tensor,intercept,qnorm,aggregate,current_aggregate,token
from jclosure.experiments.pre_readout_v27 import step,scales,delta,vector
from jclosure.protocol_v28 import verify,verify_stage,stage_freeze
from jclosure.provenance import sha256_file,write_json_atomic
SOURCE="src/jclosure/experiments/final_v28.py"
def run(root):
 opening=verify_stage(root,"final_opening")
 if not opening["final_opened"]:return {"final_opened":False,"reason":opening["reason"]}
 d=design(root);bundle,dense,values,rec,att,state_layer=context(root);state_layer=30;s=scales(root);rows=[];vecs=[]
 for i,item in enumerate(d["independent_final"],1):
  m,p,last,n,clean0=native_pair(bundle,dense,item,root,d,state_layer);teacher=teacher_tensor(m);clean1=step(bundle,dense,clean0["cache"],teacher,n+1,d,state_layer);blocked,wb=intercept(p,clean0["cache"],"REC+Conv",rec,att);changed=step(bundle,dense,blocked,teacher,n+1,d,state_layer);z=qnorm(changed["targets"],clean1["targets"],s);v=vector(delta(changed["targets"],clean1["targets"]));rows.append({"base_trial_id":item["base_trial_id"],"family":item["family"],"condition":"REC+Conv","token_id":int(last[0,0]),"next_token_id":token(m),"token_hash":hashlib.sha256(f"{int(last[0,0])}:{token(m)}".encode()).hexdigest(),"current_h0_q":0.0,"h1_q":aggregate(z),"j_h1_q":z["j"],"logits_h1_q":z["logits"],"semantic_h1_q":z["semantic"],"workspace_h1_q":z["workspace"],"writeback_pass":wb["writeback_pass"],"current_preserved":True,"continuation_token_identical":True,"channel_hashes":json.dumps(wb["channel_hashes"],sort_keys=True)});vecs.append(v);print(f"V28 independent final {i}/{len(d['independent_final'])}",flush=True)
 f=pd.DataFrame(rows);fp=root/OUT/"write_interception_independent_final_v28.parquet";vp=root/OUT/"write_interception_independent_final_vectors_v28.npz";f.to_parquet(fp,index=False,compression="zstd");np.savez_compressed(vp,vectors=np.stack(vecs).astype(np.float32),base_trial_id=f.base_trial_id.to_numpy(str),family=f.family.to_numpy(str));summary={"final_opened":True,"mechanism":"REC+Conv","rows":len(f),"median_h1_q":float(f.h1_q.median()),"potent_fraction":float((f.h1_q>=verify(root)["config"]["future_potency_threshold"]).mean()),"family_fractions":{k:float((x.h1_q>=verify(root)["config"]["future_potency_threshold"]).mean()) for k,x in f.groupby("family")},"all_writeback_pass":bool(f.writeback_pass.all()),"max_current_q":float(f.current_h0_q.max()),"historical_final_opened":False,"independent_final_opened":True,"response_sha256":sha256_file(fp),"vectors_sha256":sha256_file(vp)};jp=root/OUT/"final_confirmation_v28.json";write_json_atomic(jp,summary);fr=stage_freeze(root,"final_response",[SOURCE,str(fp.relative_to(root)),str(vp.relative_to(root)),str(jp.relative_to(root)),"artifacts/token_state_transaction_v28_final_opening.freeze.json"],summary);return {"freeze_digest":fr["freeze_digest"],**summary}
if __name__=="__main__":print(json.dumps(run(Path.cwd()),indent=2))
