"""Phase-B kernel-input/true-update instrumentation audit; no mediation claim."""
from __future__ import annotations

import gc
import json
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F

from jclosure.cache_v7 import clone_hybrid_cache
from jclosure.experiments import counterfactual_bank_v19 as v19
from jclosure.experiments.design_v33 import prompts
from jclosure.experiments.runtime_v33 import context, field_hashes, hd, prefix, step
from jclosure.experiments.transaction_v28 import thash
from jclosure.protocol_v33 import stage_freeze, verify, verify_stage
from jclosure.provenance import sha256_file, write_json_atomic
from jclosure.runtime_v3_1 import encode_direct_prompt

OUT=Path("results/v33/processed")
SOURCE="src/jclosure/experiments/phase_b_audit_v33.py"


def qwen_audit(root,old):
    row=old["development"][0]
    frame=pd.read_parquet(root/f"results/v18/processed/crossed_state_{row['source_role']}_{row['family']}_v18.parquet",filters=[[("base_trial_id","==",row["base_trial_id"])]] )
    prompt=str(frame.iloc[0]["prompt"])
    bundle,*_=v19._setup(root)
    model=bundle.hf_model
    ids=encode_direct_prompt(bundle,prompt)
    device=next(model.parameters()).device
    with torch.no_grad():
        incoming=clone_hybrid_cache(model(input_ids=ids[:,:-1].to(device),use_cache=True).past_key_values)
        donor=clone_hybrid_cache(model(input_ids=torch.tensor([[row["primary_token_id"]]],device=device),past_key_values=clone_hybrid_cache(incoming),attention_mask=torch.ones((1,ids.shape[1]),device=device,dtype=torch.long),use_cache=True).past_key_values)
    probe=int(row["future_probe_tokens"][0])
    capture={}
    originals=[]
    handles=[]
    try:
        for layer,block in enumerate(bundle.layers):
            if block.layer_type!="linear_attention":
                continue
            mixer=block.linear_attn
            local={}
            capture[layer]=local
            def hook_a(_m,_inp,out,loc=local):loc["raw_a"]=out.detach().clone()
            def hook_b(_m,_inp,out,loc=local):loc["raw_b"]=out.detach().clone()
            handles.extend([mixer.in_proj_a.register_forward_hook(hook_a),mixer.in_proj_b.register_forward_hook(hook_b)])
            original_conv=mixer.causal_conv1d_update
            def conv_wrap(*args,orig=original_conv,loc=local,**kwargs):
                out=orig(*args,**kwargs)
                loc["postconv"]=out.detach().clone()
                return out
            mixer.causal_conv1d_update=conv_wrap
            originals.append((mixer,"causal_conv1d_update",original_conv))
            original_kernel=mixer.recurrent_gated_delta_rule
            def kernel_wrap(query,key,value,*,g,beta,initial_state,output_final_state,use_qk_l2norm_in_kernel,orig=original_kernel,loc=local):
                loc.update({"q":query.detach().clone(),"k":key.detach().clone(),"v":value.detach().clone(),"g":g.detach().clone(),"beta":beta.detach().clone(),"initial":initial_state.detach().clone()})
                output=orig(query,key,value,g=g,beta=beta,initial_state=initial_state,output_final_state=output_final_state,use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel)
                loc["read"]=output[0].detach().clone()
                loc["state_arg"]=output[1].detach().clone()
                return output
            mixer.recurrent_gated_delta_rule=kernel_wrap
            originals.append((mixer,"recurrent_gated_delta_rule",original_kernel))
        with torch.no_grad():
            cache=clone_hybrid_cache(donor)
            output=model(input_ids=torch.tensor([[probe]],device=device),past_key_values=cache,attention_mask=torch.ones((1,ids.shape[1]+1),device=device,dtype=torch.long),use_cache=True)
        records=[]
        for layer,loc in capture.items():
            mixer=bundle.layers[layer].linear_attn
            gcheck=-mixer.A_log.float().exp()*F.softplus(loc["raw_a"].float()+mixer.dt_bias)
            bcheck=loc["raw_b"].sigmoid()
            gate_exact=torch.equal(gcheck,loc["g"])
            beta_exact=torch.equal(bcheck,loc["beta"])
            # Copy the actual one-token torch recurrence to expose the true delta term.
            from transformers.models.qwen3_5.modeling_qwen3_5 import l2norm
            q=l2norm(loc["q"],dim=-1,eps=1e-6).transpose(1,2).contiguous().float()
            k=l2norm(loc["k"],dim=-1,eps=1e-6).transpose(1,2).contiguous().float()
            v=loc["v"].transpose(1,2).contiguous().float()
            g=loc["g"].transpose(1,2).contiguous().float()
            beta=loc["beta"].transpose(1,2).contiguous().float()
            state=loc["initial"].to(v)
            decay=g[:,:,0].exp().unsqueeze(-1).unsqueeze(-1)
            decayed=state*decay
            kv_mem=(decayed*k[:,:,0].unsqueeze(-1)).sum(dim=-2)
            delta=(v[:,:,0]-kv_mem)*beta[:,:,0].unsqueeze(-1)
            write=k[:,:,0].unsqueeze(-1)*delta.unsqueeze(-2)
            predicted=decayed+write
            update_exact=torch.equal(predicted,loc["state_arg"])
            records.append({"layer":layer,"transformed_gate_exact":gate_exact,"beta_exact":beta_exact,"delta_term_exact_to_kernel_recurrence":update_exact,"postconv_hash":thash(loc["postconv"]),"q_postconv_hash":thash(loc["q"]),"k_postconv_hash":thash(loc["k"]),"v_postconv_hash":thash(loc["v"]),"g_consumed_hash":thash(loc["g"]),"beta_consumed_hash":thash(loc["beta"]),"true_delta_hash":thash(delta),"true_write_hash":thash(write),"read_hash":thash(loc["read"]),"outgoing_state_hash":thash(loc["state_arg"]),"cache_state_matches":torch.equal(output.past_key_values.layers[layer].recurrent_states,loc["state_arg"]),"kernel_callable":getattr(original_kernel,"__name__",str(original_kernel))})
        return {"model":"Qwen/Qwen3.5-4B","state_id":row["base_trial_id"],"probe_id":probe,"layers":records,"exact_gate_count":sum(r["transformed_gate_exact"] and r["beta_exact"] for r in records),"exact_delta_count":sum(r["delta_term_exact_to_kernel_recurrence"] for r in records),"all_cache_readback_exact":all(r["cache_state_matches"] for r in records),"future_mediation_performed":False}
    finally:
        for handle in handles:handle.remove()
        for obj,name,value in originals:setattr(obj,name,value)
        del model,bundle
        gc.collect()
        torch.cuda.empty_cache()


def falcon_audit(root,design):
    row=design["development"][0]
    lookup=prompts(root,design)
    model,tokenizer=context(root)
    incoming,_,length,ids=prefix(model,tokenizer,lookup[row["base_trial_id"]])
    donor=step(model,incoming,row["primary_token_id"],length)["cache"]
    cache=clone_hybrid_cache(donor)
    prior=[layer.recurrent_states.detach().clone() for layer in cache.layers]
    capture={}
    originals=[]
    handles=[]
    try:
        for layer,block in enumerate(model.model.layers):
            mixer=block.mamba
            local={}
            capture[layer]=local
            def hook_proj(_m,_inp,out,loc=local):loc["projected_raw"]=out.detach().clone()
            handles.append(mixer.in_proj.register_forward_hook(hook_proj))
            def hook_act(_m,_inp,out,loc=local):loc["postconv"]=out.detach().clone()
            handles.append(mixer.act.register_forward_hook(hook_act))
        original_update=cache.update_recurrent_state
        def update_wrap(state,layer_idx,**kwargs):
            capture[layer_idx]["state_arg"]=state.detach().clone()
            return original_update(state,layer_idx,**kwargs)
        cache.update_recurrent_state=update_wrap
        device=next(model.parameters()).device
        probe=int(row["future_probe_tokens"][0])
        with torch.no_grad():
            output=model(input_ids=torch.tensor([[probe]],device=device),past_key_values=cache,attention_mask=torch.ones((1,length+1),device=device,dtype=torch.long),use_cache=True)
        records=[]
        for layer,loc in capture.items():
            mixer=model.model.layers[layer].mamba
            projected=loc["projected_raw"]*mixer.mup_vector
            gate,_,dt=projected.split([mixer.intermediate_size,mixer.conv_dim,mixer.num_heads],dim=-1)
            hidden,B,C=torch.split(loc["postconv"],[mixer.intermediate_size,mixer.n_groups*mixer.ssm_state_size,mixer.n_groups*mixer.ssm_state_size],dim=-1)
            batch=hidden.shape[0]
            dt=dt[:,0,:][:,None,...]
            dt=dt.transpose(1,2).expand(batch,dt.shape[-1],mixer.head_dim)
            dt_bias=mixer.dt_bias[...,None].expand(mixer.dt_bias.shape[0],mixer.head_dim)
            dt=F.softplus(dt+dt_bias.to(dt.dtype))
            dt=torch.clamp(dt,mixer.time_step_limit[0],mixer.time_step_limit[1])
            A=-torch.exp(mixer.A_log.float())
            A=A[...,None,None].expand(mixer.num_heads,mixer.head_dim,mixer.ssm_state_size).float()
            dA=torch.exp(dt[...,None]*A).to(device=prior[layer].device)
            B=B.reshape(batch,mixer.n_groups,-1)[...,None,:]
            B=B.expand(batch,mixer.n_groups,mixer.num_heads//mixer.n_groups,B.shape[-1]).contiguous().reshape(batch,-1,B.shape[-1])
            dB=dt[...,None]*B[...,None,:]
            hidden=hidden.reshape(batch,-1,mixer.head_dim)
            dBx=(dB*hidden[...,None]).to(device=prior[layer].device)
            predicted=prior[layer]*dA+dBx
            update_exact=torch.equal(predicted,loc["state_arg"])
            records.append({"layer":layer,"postconv_x_hash":thash(hidden),"postconv_B_hash":thash(B),"postconv_C_hash":thash(C),"transformed_dt_hash":thash(dt),"decay_factor_hash":thash(dA),"actual_update_term_hash":thash(dBx),"update_exact_to_consumed_state_arg":update_exact,"outgoing_state_hash":thash(loc["state_arg"]),"cache_state_matches":torch.equal(output.past_key_values.layers[layer].recurrent_states,loc["state_arg"]),"qkv_homology":"x/B/C Mamba factorization; not identical Qwen delta q/k/v"})
        return {"model":"tiiuae/Falcon-H1-1.5B-Base","state_id":row["base_trial_id"],"probe_id":probe,"layers":records,"exact_update_count":sum(r["update_exact_to_consumed_state_arg"] for r in records),"all_cache_readback_exact":all(r["cache_state_matches"] for r in records),"future_mediation_performed":False}
    finally:
        for handle in handles:handle.remove()
        for obj,name,value in originals:setattr(obj,name,value)
        del model,tokenizer
        gc.collect()
        torch.cuda.empty_cache()


def run(root:Path):
    verify_stage(root,"primary_adjudication")
    primary=json.loads((root/OUT/"primary_adjudication_v33.json").read_text())
    if not primary["phase_B_opened"]:
        raise RuntimeError("V33 Phase B not authorized")
    old=json.loads((root/"results/v32/processed/design_v32.json").read_text())
    design=json.loads((root/OUT/"design_v33.json").read_text())
    model1=qwen_audit(root,old)
    model2=falcon_audit(root,design)
    result={"phase_B_authorized":True,"instrumentation_pilot_only":True,"model1":model1,"model2":model2,"bidirectional_mediation_executed":False,"shared_micro_mediator_identified":False,"H_or_I_claim_authorized":False,"reason":"Exact instrumentation and model homology audit do not by themselves satisfy remove/restore validation criterion"}
    path=root/OUT/"phase_b_instrumentation_v33.json"
    write_json_atomic(path,result)
    stage=stage_freeze(root,"phase_b_instrumentation",[SOURCE,str(path.relative_to(root)),"artifacts/cross_model_rec_conv_v33_primary_adjudication.freeze.json"],{"instrumentation_sha256":sha256_file(path),"bidirectional_mediation_executed":False,"shared_micro_mediator_identified":False})
    return {"freeze_digest":stage["freeze_digest"],"qwen_layers":len(model1["layers"]),"falcon_layers":len(model2["layers"]),"qwen_exact_delta":model1["exact_delta_count"],"falcon_exact_update":model2["exact_update_count"]}


if __name__=="__main__":
    print(json.dumps(run(Path.cwd()),indent=2))
