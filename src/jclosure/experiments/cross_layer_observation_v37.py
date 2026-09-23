"""Test frozen Q2→Q3/Q4 operator predictions with native later reads.

Only Q2 has an output patch. Q3/Q4 recurrent computations and all downstream
layers run natively with one target REC-state change; no later output is copied.
"""
from __future__ import annotations

import argparse
import json
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from jclosure.experiments.depth_common_v35 import capture, prepare
from jclosure.experiments.local_v36 import _local_cache, _module
from jclosure.experiments.read_hooks_v35 import ReadPatch
from jclosure.experiments.runtime_v34 import load, signature, step
from jclosure.experiments.transaction_v28 import thash
from jclosure.protocol_v37 import stage_freeze, verify_stage
from jclosure.provenance import sha256_file, write_json_atomic


OUT=Path("results/v37/processed")
SOURCE="src/jclosure/experiments/cross_layer_observation_v37.py"
SITES=("B23","P3","B34","P4")
CONTEXTS=("CC","JC")
STATES=("A","B")


def _cos(a,b):
    x,y=np.asarray(a,np.float64).ravel(),np.asarray(b,np.float64).ravel()
    d=float(np.linalg.norm(x)*np.linalg.norm(y))
    return float(np.dot(x,y)/d) if d>1e-8 else None


@torch.no_grad()
def _run_capture(model,key,cache,probe,length,bundle,layer,refs,mapping):
    module=_module(model,key,layer)
    seen={}
    def before(_m,args,kwargs):
        value=kwargs.get("hidden_states",args[0] if args else None)
        seen["hidden"]=value.detach().clone()
    def raw(_m,args):
        seen["raw"]=args[0].detach().clone()
    def output(_m,_args,value):
        seen["mixer"]=value.detach().clone()
    handles=[module.register_forward_pre_hook(before,with_kwargs=True),
             module.norm.register_forward_pre_hook(raw),
             module.register_forward_hook(output)]
    try:
        context=ReadPatch(model,key,refs,mapping) if mapping else nullcontext()
        with context:
            result=step(model,cache,probe,length+1,bundle)
        proof=context.proof() if mapping else None
    finally:
        for handle in handles:handle.remove()
    if set(seen)!={"hidden","raw","mixer"}:
        raise RuntimeError("V37 later native read capture incomplete")
    return result,seen,proof


@torch.no_grad()
def run(root:Path,key:str,role:str):
    if role not in ("development","validation"):raise ValueError(role)
    verify_stage(root,f"cross_layer_prediction_{key}_{role}")
    design=json.loads((root/OUT/f"design_{key}_v37.json").read_text())
    panel=json.loads((root/OUT/"panel_v37.json").read_text())
    prompts={row["base_trial_id"]:row["prompt"] for r in
             ("calibration","development","validation","independent_final") for row in panel[r]}
    pred=pd.read_parquet(root/OUT/f"cross_layer_prediction_{key}_{role}_v37.parquet")
    payload=np.load(root/OUT/f"cross_layer_prediction_vectors_{key}_{role}_v37.npz")
    raw_pred,mix_pred=payload["raw"],payload["mixer"]
    by_key={(row.state_id,row.position,int(row.probe_index)):row for row in pred.itertuples()}
    model,tokenizer=load(root,key)
    q2=design["relative_depth_layers"]["Q2"]
    rows,state_rows,audits=[],[],[]
    for n,item in enumerate(design[role],1):
        sid=item["base_trial_id"]
        caches=prepare(model,tokenizer,key,item,prompts[sid])
        length=caches["length"]
        outputs={position:{f"{context}_{state}":[] for context in CONTEXTS for state in STATES}
                 for position in SITES}
        per_probe={position:[] for position in SITES}
        for pi,probe in enumerate(item["future_probe_tokens"]):
            _,joint_ref=capture(model,key,caches["joint"],probe,length,design["target_bundle"])
            refs={"JOINT":joint_ref}
            q2_map={layer:"JOINT" for layer in q2}
            for position in SITES:
                layer=design["local_layers"][position]
                p=by_key[(sid,position,pi)]
                i=int(p.vector_index)
                if thash(torch.from_numpy(raw_pred[i]))!=p.raw_prediction_hash or \
                   thash(torch.from_numpy(mix_pred[i]))!=p.mixer_prediction_hash:
                    raise RuntimeError("V37 cross-layer frozen vector digest mismatch")
                observed={}
                for context,mapping in (("CC",{}),("JC",q2_map)):
                    for state in STATES:
                        if state=="A":cache=caches["conv"]
                        else:cache,_=_local_cache(caches["conv"],caches["joint"],caches["conv"],layer)
                        out,seen,proof=_run_capture(model,key,cache,probe,length,
                                   design["target_bundle"],layer,refs,mapping)
                        if thash(seen["hidden"])!=getattr(p,f"hidden_{context}_hash"):
                            raise RuntimeError(f"V37 hidden-context drift {key}:{sid}:{position}:{pi}:{context}")
                        outputs[position][f"{context}_{state}"].append(out)
                        observed[(context,state)]=seen
                        audits.append({"state_id":sid,"model":key,"role":role,
                                       "position":position,"probe_index":pi,"context":context,
                                       "state":state,"q2_patch_exact":bool(proof is None or proof["exact_writeback"]),
                                       "later_layers_native":True,
                                       "target_REC_requested_hash":thash(cache.layers[layer].recurrent_states)})
                raw=np.stack([(observed[(c,"B")]["raw"].float()-observed[(c,"A")]["raw"].float()).cpu().numpy().ravel()
                              for c in CONTEXTS])
                mixer=np.stack([(observed[(c,"B")]["mixer"].float()-observed[(c,"A")]["mixer"].float()).cpu().numpy().ravel()
                                for c in CONTEXTS])
                rp,mp=raw_pred[i].astype(np.float64),mix_pred[i].astype(np.float64)
                if raw.shape!=rp.shape or mixer.shape!=mp.shape:
                    raise RuntimeError("V37 later REC predicted/observed shape mismatch")
                raw_cos=_cos(rp[1]-rp[0],raw[1]-raw[0])
                mixer_cos=_cos(mp[1]-mp[0],mixer[1]-mixer[0])
                if np.max(np.abs(raw-rp))>.05 or np.max(np.abs(mixer-mp))>.05:
                    raise RuntimeError(f"V37 cross-layer native formula mismatch {key}:{sid}:{position}:{pi}")
                per_probe[position].append({"raw_change_cosine":raw_cos,
                   "mixer_change_cosine":mixer_cos,
                   "predicted_mixer_JC_greater":bool(np.linalg.norm(mp[1])>np.linalg.norm(mp[0])),
                   "observed_mixer_JC_greater":bool(np.linalg.norm(mixer[1])>np.linalg.norm(mixer[0])),
                   "mixer_change_norm":float(np.linalg.norm(mixer[1]-mixer[0]))})
                rows.append({"state_id":sid,"model":key,"role":role,"family":item["family"],
                             "position":position,"layer":layer,"probe_index":pi,
                             **per_probe[position][-1],
                             "factor_changed":bool(p.changed_factor_count>0),
                             "hidden_change_norm":p.hidden_change_norm,
                             "prediction_seal_verified":True,"natural_later_read":True})
        for position in SITES:
            ys={name:signature(values,design["calibration_clean_scales"])
                for name,values in outputs[position].items()}
            cc=ys["CC_B"]-ys["CC_A"]
            jc=ys["JC_B"]-ys["JC_A"]
            local_orders=[x["predicted_mixer_JC_greater"] for x in per_probe[position]]
            # State is the independent unit; six probe observations are repeated.
            vote=bool(np.mean(local_orders)>.5)
            future_order=bool(np.linalg.norm(jc)>np.linalg.norm(cc))
            state_rows.append({"state_id":sid,"model":key,"role":role,
                               "family":item["family"],"position":position,
                               "predicted_JC_future_order_vote":vote,
                               "observed_JC_future_order":future_order,
                               "future_order_correct":vote==future_order,
                               "future_effect_CC_norm":float(np.linalg.norm(cc)),
                               "future_effect_JC_norm":float(np.linalg.norm(jc)),
                               "future_effect_change_norm":float(np.linalg.norm(jc-cc)),
                               "future_effect_change_cosine_to_JC":_cos(jc-cc,jc),
                               "later_downstream_natural":True,
                               "Q2_patch_only":True})
        if n%5==0 or n==len(design[role]):
            print(f"V37 cross-layer observation {key} {role} {n}/{len(design[role])}",flush=True)
    frames={"local":pd.DataFrame(rows),"state":pd.DataFrame(state_rows),"audit":pd.DataFrame(audits)}
    files={name:root/OUT/f"cross_layer_{name}_{key}_{role}_v37.parquet" for name in frames}
    for name,frame in frames.items():frame.to_parquet(files[name],index=False,compression="zstd")
    summary={"model":key,"role":role,"states":len(design[role]),
             "local_rows":len(frames["local"]),"state_site_rows":len(frames["state"]),
             "median_mixer_change_cosine":float(frames["local"].mixer_change_cosine.median()),
             "future_order_accuracy":float(frames["state"].future_order_correct.mean()),
             "files_sha256":{name:sha256_file(path) for name,path in files.items()},
             "later_outputs_copied":False}
    path=root/OUT/f"cross_layer_observation_{key}_{role}_v37.json"
    write_json_atomic(path,summary)
    seal=stage_freeze(root,f"cross_layer_observation_{key}_{role}",
                      [SOURCE,str(path.relative_to(root)),
                       *[str(p.relative_to(root)) for p in files.values()],
                       f"artifacts/computational_origin_v37_cross_layer_prediction_{key}_{role}.freeze.json"],
                      {"model":key,"role":role,"summary_sha256":sha256_file(path),
                       "later_outputs_copied":False})
    return {"freeze_digest":seal["freeze_digest"],**summary}


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("model",choices=("Q","F"))
    p.add_argument("role",choices=("development","validation"))
    a=p.parse_args();print(json.dumps(run(Path.cwd(),a.model,a.role),indent=2))
