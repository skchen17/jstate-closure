"""Frozen-subset native REC controls and read-interface ceiling."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from jclosure.cache_v7 import clone_hybrid_cache
from jclosure.experiments.depth_common_v35 import capture, patch, prepare
from jclosure.experiments.natural_reconstruction_v37 import _selective_rec
from jclosure.experiments.runtime_v34 import field_hashes, hd, load, native_swap, prefix, signature, step
from jclosure.protocol_v38 import stage_freeze, verify_stage
from jclosure.provenance import sha256_file, write_json_atomic


OUT=Path("results/v38/processed")
SOURCE="src/jclosure/experiments/controls_v38.py"
CONDITIONS=("RECIPIENT","REC_ONLY","CONV_ONLY","JOINT","DONOR_CEILING",
            "Q234","Q1234","WRONG_TOKEN_Q234","SHUFFLED_Q234",
            "SIGNFLIP_Q234","READ_INTERFACE_Q234")


def _shuffled(base,donor,groups):
    work=clone_hybrid_cache(base)
    for name in ("Q2","Q3","Q4"):
        layers=groups[name]
        for target,source in zip(layers,reversed(layers)):
            x=donor.layers[source].recurrent_states
            y=work.layers[target].recurrent_states
            if x.shape!=y.shape or x.dtype!=y.dtype:
                raise RuntimeError("V38 layer shuffle topology mismatch")
            work.layers[target].recurrent_states=x.detach().clone()
    return work


def _signflip(base,donor,layers):
    work=clone_hybrid_cache(base)
    for layer in layers:
        x=base.layers[layer].recurrent_states
        y=donor.layers[layer].recurrent_states
        z=(2*x.float()-y.float()).to(x.dtype)
        if not torch.isfinite(z.float()).all():
            raise RuntimeError("V38 signflip nonfinite")
        work.layers[layer].recurrent_states=z
    return work


@torch.no_grad()
def run(root:Path,key:str,role:str):
    if role not in ("development","validation"):
        raise ValueError(role)
    verify_stage(root,f"trajectory_full_{key}_{role}")
    plan=json.loads((root/OUT/"execution_plan_v38.json").read_text())
    selected={sid for ids in plan["task_subsets"][role].values() for sid in ids}
    design=json.loads((root/OUT/f"design_{key}_v38.json").read_text())
    panel=json.loads((root/OUT/"panel_v38.json").read_text())
    prompts={row["base_trial_id"]:row["prompt"] for r in
             ("calibration","development","validation","independent_final") for row in panel[r]}
    groups=design["relative_depth_layers"]
    q234=groups["Q2"]+groups["Q3"]+groups["Q4"]
    q1234=groups["Q1"]+q234
    model,tokenizer=load(root,key)
    rows,vectors,audits=[],[],[]
    for n,item in enumerate([x for x in design[role] if x["base_trial_id"] in selected],1):
        sid=item["base_trial_id"]
        caches=prepare(model,tokenizer,key,item,prompts[sid])
        rec,_=native_swap(caches["recipient"],caches["donor"],["REC"],key)
        q234_cache,proof234=_selective_rec(caches["conv"],caches["joint"],q234)
        q1234_cache,proof1234=_selective_rec(caches["conv"],caches["joint"],q1234)
        incoming,length,ids,_=prefix(model,tokenizer,key,prompts[sid])
        if hd(ids)!=item["prefix_token_hash"] or field_hashes(incoming)!=item["incoming_state_hashes"]:
            raise RuntimeError("V38 wrong-token prefix drift")
        wrong=step(model,incoming,item["third_token_id"],length)["cache"]
        wrong_cache,wrong_proof=_selective_rec(caches["conv"],wrong,q234)
        shuffled=_shuffled(caches["conv"],caches["joint"],groups)
        signflip=_signflip(caches["conv"],caches["joint"],q234)
        branches={"RECIPIENT":caches["recipient"],"REC_ONLY":rec,
                  "CONV_ONLY":caches["conv"],"JOINT":caches["joint"],
                  "DONOR_CEILING":caches["donor"],"Q234":q234_cache,
                  "Q1234":q1234_cache,"WRONG_TOKEN_Q234":wrong_cache,
                  "SHUFFLED_Q234":shuffled,"SIGNFLIP_Q234":signflip}
        ys={}
        for name,cache in branches.items():
            ys[name]=signature([step(model,cache,probe,length+1,design["target_bundle"])
                                for probe in item["future_probe_tokens"]],
                               design["calibration_clean_scales"])
        interface=[]
        for probe in item["future_probe_tokens"]:
            _,joint_ref=capture(model,key,caches["joint"],probe,length,design["target_bundle"])
            out,proof=patch(model,key,caches["conv"],probe,length,design["target_bundle"],
                            {"JOINT":joint_ref},{layer:"JOINT" for layer in q234})
            if not proof["exact_writeback"]:
                raise RuntimeError("V38 interface writeback mismatch")
            interface.append(out)
        ys["READ_INTERFACE_Q234"]=signature(interface,design["calibration_clean_scales"])
        if field_hashes(q234_cache)["KV"]!=caches["recipient_KV_hash"] or \
           field_hashes(q234_cache)["Conv"]!=field_hashes(caches["conv"])["Conv"]:
            raise RuntimeError("V38 control channel drift")
        if not all(p["requested_hash"]==p["realized_hash"] for p in proof234+proof1234+wrong_proof):
            raise RuntimeError("V38 control REC writeback mismatch")
        donor=ys["DONOR_CEILING"]
        def error(name):return float(np.linalg.norm(donor-ys[name]))
        rows.append({"state_id":sid,"model":key,"role":role,"family":item["family"],
                     "vector_index":len(vectors),"recipient_error":error("RECIPIENT"),
                     "rec_only_error":error("REC_ONLY"),"conv_error":error("CONV_ONLY"),
                     "joint_error":error("JOINT"),"q234_error":error("Q234"),
                     "q1234_error":error("Q1234"),"wrong_error":error("WRONG_TOKEN_Q234"),
                     "shuffle_error":error("SHUFFLED_Q234"),
                     "signflip_error":error("SIGNFLIP_Q234"),
                     "interface_error":error("READ_INTERFACE_Q234"),
                     "q1_increment_norm":float(np.linalg.norm(ys["Q1234"]-ys["Q234"])),
                     "off_manifold_controls":"SHUFFLED_Q234,SIGNFLIP_Q234",
                     "native_q234_no_output_copy":True})
        vectors.append(np.stack([ys[x] for x in CONDITIONS]).astype(np.float32))
        audits.append({"state_id":sid,"model":key,"role":role,"q234_REC_hash":field_hashes(q234_cache)["REC"],
                       "donor_Conv_hash":field_hashes(caches["conv"])["Conv"],
                       "recipient_KV_hash":caches["recipient_KV_hash"],
                       "wrong_same_prefix":True,"interface_copy_only_ceiling":True})
        if n%5==0 or n==len(selected):
            print(f"V38 controls {key} {role} {n}/{len(selected)}",flush=True)
    frame=pd.DataFrame(rows)
    files={"rows":root/OUT/f"controls_{key}_{role}_v38.parquet",
           "vectors":root/OUT/f"controls_vectors_{key}_{role}_v38.npz",
           "audit":root/OUT/f"controls_audit_{key}_{role}_v38.parquet"}
    frame.to_parquet(files["rows"],index=False,compression="zstd")
    pd.DataFrame(audits).to_parquet(files["audit"],index=False,compression="zstd")
    np.savez_compressed(files["vectors"],vectors=np.stack(vectors),
                        conditions=np.asarray(CONDITIONS),state_ids=frame.state_id.to_numpy(str))
    summary={"model":key,"role":role,"states":len(frame),
             "medians":{x:float(frame[x].median()) for x in ("recipient_error","rec_only_error",
                 "conv_error","joint_error","q234_error","q1234_error","wrong_error",
                 "shuffle_error","signflip_error","interface_error","q1_increment_norm")},
             "files_sha256":{k:sha256_file(p) for k,p in files.items()},
             "off_manifold_not_natural":True,"interface_copy_not_mechanism":True}
    path=root/OUT/f"controls_{key}_{role}_v38.json"
    write_json_atomic(path,summary)
    seal=stage_freeze(root,f"controls_{key}_{role}",
                      [SOURCE,str(path.relative_to(root)),
                       *[str(p.relative_to(root)) for p in files.values()],
                       f"artifacts/trajectory_composition_v38_trajectory_full_{key}_{role}.freeze.json"],
                      {"model":key,"role":role,"summary_sha256":sha256_file(path)})
    return {"freeze_digest":seal["freeze_digest"],**summary}


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("model",choices=("Q","F"))
    p.add_argument("role",choices=("development","validation"))
    a=p.parse_args();print(json.dumps(run(Path.cwd(),a.model,a.role),indent=2))
