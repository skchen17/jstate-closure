"""Fresh-panel bidirectional functional-stage mediation in both hybrid LMs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from jclosure.experiments.design_v34 import prompt_lookup
from jclosure.experiments.functional_hooks_v34 import FunctionalIntervention
from jclosure.experiments.runtime_v34 import field_hashes,hd,load,native_swap,prefix,signature,step
from jclosure.protocol_v34 import stage_freeze,verify,verify_stage
from jclosure.provenance import sha256_file,write_json_atomic

OUT=Path("results/v34/processed")
SOURCE="src/jclosure/experiments/mediate_v34.py"


def cosine(a,b):
    x,y=float(np.linalg.norm(a)),float(np.linalg.norm(b))
    return float(np.dot(a,b)/(x*y)) if x*y>1e-12 else None


def audit_hashes(capture,stage):
    requested=[];realized=[];native=[];layers=[]
    for layer,record in sorted(capture.items()):
        if not record.get("exact_writeback"):raise RuntimeError(f"V34 stage writeback failed {stage}:{layer}")
        if "requested_hash" in record:
            r=[record["requested_hash"]];z=[record["realized_hash"]];n=[record["native_hash"]]
        else:
            r=[v for k,v in sorted(record.items()) if k.startswith(stage+"_") and k.endswith("_requested_hash")]
            z=[v for k,v in sorted(record.items()) if k.startswith(stage+"_") and k.endswith("_realized_hash")]
            n=[v for k,v in sorted(record.items()) if k.startswith(stage+"_") and k.endswith("_native_hash")]
        if not r or len(r)!=len(z) or r!=z:raise RuntimeError(f"V34 requested/realized hash mismatch {stage}:{layer}")
        requested.extend(r);realized.extend(z);native.extend(n)
        layers.append({"layer":layer,"requested_hashes":r,"realized_hashes":z,"native_hashes":n})
    return {"requested_stage_hash":hd(requested),"realized_stage_hash":hd(realized),"native_stage_hash":hd(native),"writeback_exact":requested==realized,"layer_records_json":json.dumps(layers,sort_keys=True)}


@torch.no_grad()
def run(root:Path,key:str,role:str):
    verify_stage(root,"mediation_plan")
    verify_stage(root,f"factorial_{key}_{role}")
    high_role="development" if role=="development" else "validation"
    gate=verify_stage(root,f"high_level_{high_role}")
    if not gate["formal_mediation_authorized"]:raise RuntimeError(f"V34 high-level correction failed: {high_role}")
    if role=="validation":
        for model in ("Q","F"):verify_stage(root,f"mediation_{model}_development")
    cfg=verify(root)["config"]
    plan=json.loads((root/OUT/"mediation_plan_v34.json").read_text())
    design=json.loads((root/OUT/f"design_{key}_v34.json").read_text())
    panel=json.loads((root/OUT/"panel_v34.json").read_text())
    lookup=prompt_lookup(root,panel)
    factorial=pd.read_parquet(root/OUT/f"factorial_{key}_{role}_v34.parquet")
    factual=np.load(root/OUT/f"factorial_vectors_{key}_{role}_v34.npz")["vectors"].astype(np.float64)
    by_state={row.state_id:row for row in factorial.itertuples()}
    model,tokenizer=load(root,key)
    stages=plan["stages_to_execute_formally"]
    rows,arrays,audits=[],[],[]
    for n,item in enumerate(design[role],1):
        sid=item["base_trial_id"]
        incoming,length,ids,_=prefix(model,tokenizer,key,lookup[sid])
        if hd(ids)!=item["prefix_token_hash"] or field_hashes(incoming)!=item["incoming_state_hashes"]:raise RuntimeError(f"V34 mediation incoming drift {key}:{sid}")
        recipient=step(model,incoming,item["recipient_token_id"],length)["cache"]
        donor=step(model,incoming,item["donor_token_id"],length)["cache"]
        conv,_=native_swap(recipient,donor,["Conv"],key)
        joint,_=native_swap(recipient,donor,["REC","Conv"],key)
        if field_hashes(conv)["KV"]!=field_hashes(recipient)["KV"] or field_hashes(joint)["KV"]!=field_hashes(recipient)["KV"]:raise RuntimeError("V34 mediation recipient KV drift")
        captures={"CONV_ONLY":[],"JOINT":[]}
        baselines={"CONV_ONLY":[],"JOINT":[]}
        patched={(stage,direction):[] for stage in stages for direction in ("REMOVE","RESTORE")}
        for probe_index,probe in enumerate(item["future_probe_tokens"]):
            for label,cache in (("CONV_ONLY",conv),("JOINT",joint)):
                with FunctionalIntervention(model,key) as instrument:
                    output=step(model,cache,probe,length+1,design["target_bundle"])
                captures[label].append(instrument.capture)
                baselines[label].append({"targets":output["targets"]})
            for stage in stages:
                for direction,base_label,ref_label,base_cache in (("REMOVE","JOINT","CONV_ONLY",joint),("RESTORE","CONV_ONLY","JOINT",conv)):
                    with FunctionalIntervention(model,key,stage,captures[ref_label][-1]) as instrument:
                        outcome=step(model,base_cache,probe,length+1,design["target_bundle"])
                    patched[(stage,direction)].append({"targets":outcome["targets"]})
                    proof=audit_hashes(instrument.capture,stage)
                    audits.append({"model_key":key,"role":role,"state_id":sid,"family":item["family"],"probe_index":probe_index,"probe_id":int(probe),"stage":stage,"direction":direction,"recipient_KV_hash":field_hashes(recipient)["KV"],**proof})
        yconv=signature(baselines["CONV_ONLY"],design["calibration_clean_scales"])
        yjoint=signature(baselines["JOINT"],design["calibration_clean_scales"])
        previous=factual[int(by_state[sid].vector_index)]
        if not np.allclose(yconv,previous[2],rtol=1e-5,atol=1e-5) or not np.allclose(yjoint,previous[3],rtol=1e-5,atol=1e-5):raise RuntimeError(f"V34 factual replay mismatch {key}:{sid}")
        yd=previous[4]
        e_conv=float(np.linalg.norm(yd-yconv));e_joint=float(np.linalg.norm(yd-yjoint));benefit=e_conv-e_joint
        true_correction=yjoint-yconv
        vecs=np.empty((len(stages),2,len(yconv)),dtype=np.float32)
        for si,stage in enumerate(stages):
            remove=signature(patched[(stage,"REMOVE")],design["calibration_clean_scales"])
            restore=signature(patched[(stage,"RESTORE")],design["calibration_clean_scales"])
            vecs[si,0],vecs[si,1]=remove,restore
            remove_error=float(np.linalg.norm(yd-remove))
            restore_error=float(np.linalg.norm(yd-restore))
            induced=restore-yconv
            valid=benefit>1e-12
            rows.append({"model_key":key,"role":role,"state_id":sid,"family":item["family"],"stage":stage,"stage_index":si,"vector_index":len(arrays),"status":"VALID" if valid else "NONPOSITIVE_BENEFIT","benefit_absolute":benefit,"conv_error":e_conv,"joint_error":e_joint,"remove_error":remove_error,"restore_error":restore_error,"removed_fraction":(remove_error-e_joint)/benefit if valid else None,"restored_fraction":(e_conv-restore_error)/benefit if valid else None,"reverse_correction_cosine":cosine(induced,true_correction),"reverse_magnitude_ratio":float(np.linalg.norm(induced))/max(float(np.linalg.norm(true_correction)),1e-12),"reverse_relative_l2_to_true_correction":float(np.linalg.norm(induced-true_correction))/max(float(np.linalg.norm(true_correction)),1e-12),"exact_writeback":True,"recipient_native_KV":True,"six_frozen_probes":True})
        arrays.append(vecs)
        if n%5==0 or n==len(design[role]):print(f"V34 mediation {key} {role} {n}/{len(design[role])}",flush=True)
    frame=pd.DataFrame(rows)
    files={"rows":root/OUT/f"mediation_{key}_{role}_v34.parquet","vectors":root/OUT/f"mediation_vectors_{key}_{role}_v34.npz","audit":root/OUT/f"mediation_audit_{key}_{role}_v34.parquet"}
    frame.to_parquet(files["rows"],index=False,compression="zstd")
    np.savez_compressed(files["vectors"],vectors=np.stack(arrays),state_ids=np.asarray([r["base_trial_id"] for r in design[role]],dtype=str),stages=np.asarray(stages,dtype=str),directions=np.asarray(["REMOVE","RESTORE"],dtype=str))
    pd.DataFrame(audits).to_parquet(files["audit"],index=False,compression="zstd")
    summary={"model_key":key,"role":role,"states":len(design[role]),"rows":len(frame),"probe_audit_rows":len(audits),"stages":stages,"median_removed":frame.groupby("stage").removed_fraction.median().to_dict(),"median_restored":frame.groupby("stage").restored_fraction.median().to_dict(),"median_reverse_cosine":frame.groupby("stage").reverse_correction_cosine.median().to_dict(),"files_sha256":{k:sha256_file(path) for k,path in files.items()},"stage5_status":"ALIASED_NOT_SEPARATELY_IDENTIFIABLE"}
    path=root/OUT/f"mediation_{key}_{role}_v34.json"
    write_json_atomic(path,summary)
    stage=stage_freeze(root,f"mediation_{key}_{role}",[SOURCE,*(str(x.relative_to(root)) for x in files.values()),str(path.relative_to(root)),"artifacts/functional_mediation_v34_mediation_plan.freeze.json"],{"model_key":key,"role":role,"summary_sha256":sha256_file(path),"rows":len(frame),"all_exact_writeback":True})
    return {"freeze_digest":stage["freeze_digest"],"model":key,"role":role,"states":len(design[role]),"median_removed":summary["median_removed"],"median_restored":summary["median_restored"]}


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("model",choices=("Q","F"))
    parser.add_argument("role",choices=("development","validation"))
    args=parser.parse_args()
    print(json.dumps(run(Path.cwd(),args.model,args.role),indent=2))
