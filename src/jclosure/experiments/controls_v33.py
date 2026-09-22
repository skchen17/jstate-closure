"""Frozen matched/mismatched REC2 context and current-slot KV contrasts."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from jclosure.cache_v7 import clone_hybrid_cache
from jclosure.experiments.design_v33 import prompts
from jclosure.experiments.runtime_v33 import context, field_hashes, hd, native_swap, prefix, signature, step
from jclosure.protocol_v33 import stage_freeze, verify, verify_stage
from jclosure.provenance import sha256_file, write_json_atomic

OUT=Path("results/v33/processed")
SOURCE="src/jclosure/experiments/controls_v33.py"


def transplant_rec(recipient, donor):
    result=clone_hybrid_cache(recipient)
    for a,b in zip(result.layers,donor.layers,strict=True):
        if a.recurrent_states.shape!=b.recurrent_states.shape:
            raise RuntimeError("V33 cross-context REC shape mismatch")
        a.recurrent_states=b.recurrent_states.detach().clone()
    if field_hashes(result)["CONV2"]!=field_hashes(recipient)["CONV2"] or field_hashes(result)["KV"]!=field_hashes(recipient)["KV"]:
        raise RuntimeError("V33 non-REC context field changed")
    return result


def shuffle_rec(recipient, matched, seed):
    result=clone_hybrid_cache(recipient)
    rng=np.random.default_rng(seed)
    order=rng.permutation(len(result.layers))
    if np.array_equal(order,np.arange(len(order))):
        order=np.roll(order,1)
    for j,a in enumerate(result.layers):
        b=matched.layers[int(order[j])]
        if a.recurrent_states.shape!=b.recurrent_states.shape:
            raise RuntimeError("V33 shuffled layer mismatch")
        a.recurrent_states=b.recurrent_states.detach().clone()
    return result,order.tolist()


def random_rec(recipient, matched, seed):
    result=clone_hybrid_cache(recipient)
    generator=torch.Generator(device="cpu").manual_seed(seed)
    for a,b in zip(result.layers,matched.layers,strict=True):
        values=torch.randn(b.recurrent_states.shape,generator=generator,dtype=torch.float32).to(b.recurrent_states.device)
        norm=b.recurrent_states.float().norm()
        a.recurrent_states=(values*norm/values.norm().clamp_min(1e-12)).to(b.recurrent_states.dtype)
    return result


@torch.no_grad()
def run(root:Path):
    verify_stage(root,"context_plan")
    cfg=verify(root)["config"]
    design=json.loads((root/OUT/"design_v33.json").read_text())
    plan=json.loads((root/OUT/"context_plan_v33.json").read_text())
    lookup=prompts(root,design)
    model,tokenizer=context(root)
    by_id={r["base_trial_id"]:r for role in ("development","validation") for r in design[role]}
    saved={role:(pd.read_parquet(root/OUT/f"factorial_{role}_v33.parquet"),np.load(root/OUT/f"factorial_vectors_{role}_v33.npz")["vectors"].astype(np.float64)) for role in ("development","validation")}
    rows,audits,vectors=[],[],[]
    for n,control in enumerate(plan["rows"],1):
        sid=control["state_id"]
        item=by_id[sid]
        incoming,_,length,ids=prefix(model,tokenizer,lookup[sid])
        if hd(ids)!=item["prefix_token_hash"] or field_hashes(incoming)!=item["incoming_state_hashes"]:
            raise RuntimeError(f"V33 context incoming drift {sid}")
        rec=step(model,incoming,design["anchor_token_id"],length)["cache"]
        donor=step(model,incoming,item["primary_token_id"],length)["cache"]
        conv,nativeproof=native_swap(rec,donor,["CONV2"])
        joint,_=native_swap(rec,donor,["REC2","CONV2"])
        wrongtok=step(model,incoming,control["wrong_token_id"],length)["cache"]
        other={}
        for key,ref in (("SAME_FAMILY_WRONG_STATE_REC2","same_family_wrong_state_id"),("CROSS_FAMILY_WRONG_STATE_REC2","cross_family_wrong_state_id")):
            other_item=by_id[control[ref]]
            other_in,_,other_length,_=prefix(model,tokenizer,lookup[control[ref]])
            other[key]=step(model,other_in,other_item["primary_token_id"],other_length)["cache"]
        shuffled,order=shuffle_rec(conv,donor,control["random_seed"])
        caches={"MATCHED_REC2":joint,"RECIPIENT_REC2":conv,"WRONG_TOKEN_REC2":transplant_rec(conv,wrongtok),"SAME_FAMILY_WRONG_STATE_REC2":transplant_rec(conv,other["SAME_FAMILY_WRONG_STATE_REC2"]),"CROSS_FAMILY_WRONG_STATE_REC2":transplant_rec(conv,other["CROSS_FAMILY_WRONG_STATE_REC2"]),"SHUFFLED_REC2":shuffled,"RANDOM_SAME_NORM_REC2":random_rec(conv,donor,control["random_seed"])}
        kv,_=native_swap(rec,donor,["KV"])
        ck,_=native_swap(rec,donor,["CONV2","KV"])
        caches.update({"KV_ONLY":kv,"CONV2_PLUS_KV":ck,"REC2_CONV2_RECIPIENT_KV":joint,"FULL_DONOR":donor,"Y00":rec})
        responses={key:signature([step(model,cache,probe,length+1,design["target_bundle"]) for probe in item["future_probe_tokens"]],design["calibration_clean_scales"]) for key,cache in caches.items()}
        base=responses["Y00"]
        d=responses["FULL_DONOR"]-base
        dn=max(float(np.linalg.norm(d)),1e-12)
        fframe,fvectors=saved[control["role"]]
        row=fframe[fframe.state_id==sid]
        if len(row)!=1:
            raise RuntimeError("V33 context target missing factorial")
        old=fvectors[int(row.iloc[0].vector_index)]
        for name,index in (("Y00",0),("RECIPIENT_REC2",2),("MATCHED_REC2",3),("FULL_DONOR",4)):
            if not np.allclose(responses[name],old[index],rtol=1e-5,atol=1e-5):
                raise RuntimeError(f"V33 context factorial replay mismatch {sid}:{name}")
        for condition in plan["conditions"]+plan["KV_conditions"]:
            response=responses[condition]
            rows.append({"role":control["role"],"family":control["family"],"state_id":sid,"condition":condition,"relative_l2_to_donor":float(np.linalg.norm(d-(response-base)))/dn,"donor_direction_cosine":float(np.dot(response-base,d)/(max(float(np.linalg.norm(response-base)),1e-12)*dn)),"vector_index":len(vectors),"target_probe_hash":item["future_probe_hash"],"matched_conv_fixed":condition in plan["conditions"],"recipient_KV":condition in plan["conditions"] or condition=="REC2_CONV2_RECIPIENT_KV","cross_state_off_manifold":condition in ("SAME_FAMILY_WRONG_STATE_REC2","CROSS_FAMILY_WRONG_STATE_REC2","SHUFFLED_REC2","RANDOM_SAME_NORM_REC2")})
            vectors.append(response.astype(np.float32))
            audits.append({"role":control["role"],"state_id":sid,"condition":condition,"field_hashes":json.dumps(field_hashes(caches[condition]),sort_keys=True),"recipient_hashes":json.dumps(field_hashes(rec),sort_keys=True),"donor_hashes":json.dumps(field_hashes(donor),sort_keys=True),"wrong_token_id":control["wrong_token_id"],"same_state_id":control["same_family_wrong_state_id"],"cross_state_id":control["cross_family_wrong_state_id"],"shuffle_order":json.dumps(order),"random_seed":control["random_seed"],"exact_tensor_write":True})
        if n%5==0 or n==len(plan["rows"]):
            print(f"V33 context/KV {n}/{len(plan['rows'])}",flush=True)
    frame=pd.DataFrame(rows)
    paths={"rows":root/OUT/"controls_v33.parquet","vectors":root/OUT/"controls_vectors_v33.npz","audit":root/OUT/"controls_audit_v33.parquet"}
    frame.to_parquet(paths["rows"],index=False,compression="zstd")
    np.savez_compressed(paths["vectors"],vectors=np.stack(vectors))
    pd.DataFrame(audits).to_parquet(paths["audit"],index=False,compression="zstd")
    summary={"states":len(plan["rows"]),"rows":len(frame),"condition_median_error":frame.groupby("condition").relative_l2_to_donor.median().to_dict(),"data_sha256":{k:sha256_file(v) for k,v in paths.items()}}
    path=root/OUT/"controls_v33.json"
    write_json_atomic(path,summary)
    stage=stage_freeze(root,"controls",[SOURCE,*(str(v.relative_to(root)) for v in paths.values()),str(path.relative_to(root)),"artifacts/cross_model_rec_conv_v33_context_plan.freeze.json"],{"control_sha256":sha256_file(path),"rows":len(frame),"response_hash":hd(summary["data_sha256"])})
    return {"freeze_digest":stage["freeze_digest"],**{k:summary[k] for k in ("states","rows","condition_median_error")}}


if __name__=="__main__":
    print(json.dumps(run(Path.cwd()),indent=2))
