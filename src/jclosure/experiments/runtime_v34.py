"""Shared Qwen/Falcon native-cache transactions and five-block future endpoints."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer

from jclosure.cache_v7 import clone_hybrid_cache
from jclosure.experiments.transaction_v28 import thash
from jclosure.protocol_v34 import verify
from jclosure.recorder import ActivationRecorder


def hd(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":"),ensure_ascii=False).encode()).hexdigest()


def load(root:Path,key:str):
    spec=verify(root)["config"]["models"][key]
    if key=="Q":
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
    tokenizer=AutoTokenizer.from_pretrained(spec["local_path"],local_files_only=True,trust_remote_code=False)
    model=AutoModelForCausalLM.from_pretrained(spec["local_path"],local_files_only=True,dtype=torch.bfloat16,device_map=spec["device"],attn_implementation="eager",trust_remote_code=False).eval()
    if len(model.model.layers) <= spec["late_layer"]:raise RuntimeError("V34 model layer mismatch")
    if key=="Q":
        observed=[i for i,x in enumerate(model.model.layers) if x.layer_type=="linear_attention"]
        if observed!=spec["recurrent_layers"]:raise RuntimeError("V34 Qwen recurrent-layer mapping drift")
    else:
        if len(model.model.layers)!=len(spec["recurrent_layers"]):raise RuntimeError("V34 Falcon layer mapping drift")
    return model,tokenizer


def encode(model,tokenizer,key,prompt):
    if key=="Q":
        ids=tokenizer.apply_chat_template([{"role":"user","content":prompt}],tokenize=True,add_generation_prompt=True,enable_thinking=False,return_tensors="pt")
        if isinstance(ids,dict) or hasattr(ids,"get"):ids=ids["input_ids"]
    else:
        ids=tokenizer(prompt,return_tensors="pt",add_special_tokens=True).input_ids
    return ids.to(next(model.parameters()).device)


def prefix(model,tokenizer,key,prompt):
    ids=encode(model,tokenizer,key,prompt)
    if ids.shape[1]<2:raise RuntimeError("V34 prompt too short")
    with torch.no_grad():out=model(input_ids=ids[:,:-1],use_cache=True)
    return clone_hybrid_cache(out.past_key_values),int(ids.shape[1]),ids[:,:-1].tolist(),out.logits[0,-1].float()


def field_hashes(cache):
    result={}
    for channel,fields in (("REC",("recurrent_states",)),("Conv",("conv_states",)),("KV",("keys","values"))):
        h=hashlib.sha256()
        found=0
        for layer,item in enumerate(cache.layers):
            for field in fields:
                value=getattr(item,field,None)
                if isinstance(value,torch.Tensor):
                    h.update(f"{layer}:{field}:{tuple(value.shape)}:{value.dtype}:".encode());h.update(thash(value).encode());found+=1
        if not found:raise RuntimeError(f"V34 cache missing channel {channel}")
        result[channel]=h.hexdigest()
    return result


def native_swap(recipient,donor,channels,key):
    wanted=set(channels)
    if not wanted<={"REC","Conv","KV"}:raise ValueError(wanted)
    if recipient.get_seq_length()!=donor.get_seq_length() or len(recipient.layers)!=len(donor.layers):raise RuntimeError("V34 cache topology mismatch")
    out=clone_hybrid_cache(recipient)
    touched=[]
    for layer,(target,source,base) in enumerate(zip(out.layers,donor.layers,recipient.layers,strict=True)):
        for channel,fields in (("REC",("recurrent_states",)),("Conv",("conv_states",)),("KV",("keys","values"))):
            if channel not in wanted:continue
            for field in fields:
                bv,sv=getattr(base,field,None),getattr(source,field,None)
                if not isinstance(bv,torch.Tensor) and not isinstance(sv,torch.Tensor):continue
                if not isinstance(bv,torch.Tensor) or not isinstance(sv,torch.Tensor) or bv.shape!=sv.shape or bv.dtype!=sv.dtype:raise RuntimeError(f"V34 field mismatch {layer}:{field}")
                if channel=="KV":
                    if not torch.equal(bv[...,:-1,:],sv[...,:-1,:]):raise RuntimeError("V34 shared KV prefix changed")
                    value=bv.detach().clone();value[...,-1:,:]=sv[...,-1:,:]
                else:value=sv.detach().clone()
                setattr(target,field,value);touched.append((layer,field))
    if not touched:raise RuntimeError("V34 swap touched no fields")
    touched=set(touched)
    for layer,(target,source,base) in enumerate(zip(out.layers,donor.layers,recipient.layers,strict=True)):
        for field in ("recurrent_states","conv_states","keys","values"):
            value=getattr(target,field,None)
            if not isinstance(value,torch.Tensor):continue
            expected=getattr(source if (layer,field) in touched else base,field)
            if (layer,field) in touched and field in ("keys","values"):
                if not torch.equal(value[...,:-1,:],getattr(base,field)[...,:-1,:]) or not torch.equal(value[...,-1:,:],expected[...,-1:,:]):raise RuntimeError("V34 KV exact write fail")
            elif not torch.equal(value,expected):raise RuntimeError(f"V34 exact write fail {layer}:{field}")
    return out,{"channels":sorted(wanted),"touched_fields":len(touched),"requested_exact":True,"untouched_exact":True,"recipient_hashes":field_hashes(recipient),"donor_hashes":field_hashes(donor),"realized_hashes":field_hashes(out),"seq_length":int(out.get_seq_length())}


def targets(model,recorder,logits,bundle):
    device=logits.device
    selected=torch.as_tensor(bundle["selected_logits"],device=device)
    broad=torch.as_tensor(bundle["broad_logits"],device=device)
    signs=torch.as_tensor(bundle["broad_sign"],device=device,dtype=torch.float32)
    last=bundle["late_layer"]
    hidden=recorder.activations[last][0,-1].float()
    hi=torch.as_tensor(bundle["late_hidden_indices"],device=hidden.device)
    return {"logits":logits[selected].cpu().numpy().astype(np.float32),"semantic":torch.log_softmax(logits,-1)[selected].cpu().numpy().astype(np.float32),"broad_vocabulary":(logits[broad]*signs).cpu().numpy().astype(np.float32),"late_hidden":hidden[hi].cpu().numpy().astype(np.float32),"workspace":torch.cat([recorder.activations[i][0,-1].float()[:bundle["workspace_per_layer"]] for i in bundle["workspace_layers"]]).cpu().numpy().astype(np.float32)}


@torch.no_grad()
def step(model,cache,token,total_length,bundle=None):
    device=next(model.parameters()).device
    work=clone_hybrid_cache(cache)
    if work.get_seq_length()!=total_length-1:raise RuntimeError("V34 continuation position mismatch")
    kwargs={"input_ids":torch.tensor([[int(token)]],device=device),"past_key_values":work,"attention_mask":torch.ones((1,total_length),device=device,dtype=torch.long),"use_cache":True}
    if bundle is None:
        out=model(**kwargs)
        return {"cache":clone_hybrid_cache(out.past_key_values),"logits":out.logits[0,-1].float()}
    at=sorted(set(bundle["workspace_layers"]+[bundle["late_layer"]]))
    with ActivationRecorder(model.model.layers,at=at,clone=True,detach=True) as recorder:
        out=model(**kwargs)
    logits=out.logits[0,-1].float()
    return {"cache":clone_hybrid_cache(out.past_key_values),"logits":logits,"targets":targets(model,recorder,logits,bundle)}


def signature(outputs,scales):
    order=("logits","semantic","broad_vocabulary","late_hidden","workspace")
    return np.concatenate([np.concatenate([np.asarray(row["targets"][field],np.float64).ravel()/max(float(scales[field]),1e-12) for field in order]) for row in outputs])
