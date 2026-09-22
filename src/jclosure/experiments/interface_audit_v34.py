"""Calibration-only bitwise native re-entry and five-stage writeback audit."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from jclosure.experiments.design_v34 import prompt_lookup
from jclosure.experiments.functional_hooks_v34 import FunctionalIntervention,STAGES
from jclosure.experiments.runtime_v34 import field_hashes,hd,load,native_swap,prefix,step
from jclosure.experiments.transaction_v28 import thash
from jclosure.protocol_v34 import stage_freeze,verify,verify_stage
from jclosure.provenance import write_json_atomic

OUT=Path("results/v34/processed")
SOURCE="src/jclosure/experiments/interface_audit_v34.py"


def value_hash(value):
    if isinstance(value,torch.Tensor):return thash(value)
    if isinstance(value,tuple):return [value_hash(x) for x in value]
    raise TypeError(type(value))


def run(root:Path,key:str):
    verify_stage(root,"design")
    config=verify(root)["config"]
    design=json.loads((root/OUT/f"design_{key}_v34.json").read_text())
    panel=json.loads((root/OUT/"panel_v34.json").read_text())
    lookup=prompt_lookup(root,panel)
    item=design["calibration"][0]
    model,tokenizer=load(root,key)
    incoming,length,ids,_=prefix(model,tokenizer,key,lookup[item["base_trial_id"]])
    if hd(ids)!=item["prefix_token_hash"] or field_hashes(incoming)!=item["incoming_state_hashes"]:raise RuntimeError("V34 audit incoming drift")
    recipient=step(model,incoming,item["recipient_token_id"],length)["cache"]
    donor=step(model,incoming,item["donor_token_id"],length)["cache"]
    conv,conv_proof=native_swap(recipient,donor,["Conv"],key)
    joint,joint_proof=native_swap(recipient,donor,["REC","Conv"],key)
    probe=int(item["future_probe_tokens"][0])
    target=design["target_bundle"]
    branches={"CONV_ONLY":conv,"JOINT":joint}
    baseline={}; captured={};replay={}
    for label,cache in branches.items():
        baseline[label]=step(model,cache,probe,length+1,target)
        with FunctionalIntervention(model,key) as instrument:
            replay[label]=step(model,cache,probe,length+1,target)
        captured[label]=instrument.capture
        if not torch.equal(baseline[label]["logits"],replay[label]["logits"]):raise RuntimeError(f"V34 {key} custom kernel logits not bitwise baseline: {label}")
        if field_hashes(baseline[label]["cache"])!=field_hashes(replay[label]["cache"]):raise RuntimeError(f"V34 {key} custom kernel cache not bitwise baseline: {label}")
        for endpoint in ("logits","semantic","broad_vocabulary","late_hidden","workspace"):
            if not np.array_equal(baseline[label]["targets"][endpoint],replay[label]["targets"][endpoint]):raise RuntimeError(f"V34 {key} endpoint drift {label}:{endpoint}")
    interventions=[]
    for stage in STAGES:
        for direction,base_label,reference_label in (("REMOVE","JOINT","CONV_ONLY"),("RESTORE","CONV_ONLY","JOINT")):
            with FunctionalIntervention(model,key,stage,captured[reference_label]) as instrument:
                out=step(model,branches[base_label],probe,length+1,target)
            if len(instrument.capture)!=len(config["models"][key]["recurrent_layers"]):raise RuntimeError("V34 incomplete layer instrumentation")
            for layer,record in instrument.capture.items():
                if not record.get("exact_writeback"):raise RuntimeError(f"V34 failed stage writeback {key}:{stage}:{layer}")
            interventions.append({"stage":stage,"direction":direction,"requested_realized_exact":True,"layers":len(instrument.capture),"logits_sha256":thash(out["logits"]),"per_layer":{str(layer):{"captured_stage_hashes":{s:value_hash(record[s]) for s in STAGES if s in record},"requested_hash":record.get("requested_hash"),"realized_hash":record.get("realized_hash"),"component_requested_hashes":{k:v for k,v in record.items() if k.endswith("_requested_hash")}} for layer,record in instrument.capture.items()}})
    result={"model_key":key,"state_id":item["base_trial_id"],"probe_id":probe,"branch_cache_hashes":{k:field_hashes(x) for k,x in branches.items()},"native_swap_proofs":{"CONV_ONLY":conv_proof,"JOINT":joint_proof},"bitwise_baseline_replay":True,"recurrent_layers_audited":len(config["models"][key]["recurrent_layers"]),"interventions":interventions,"stage_order":config["functional_stages"]["order"],"formal_development_or_validation_observed":False,"calibration_only":True}
    path=root/OUT/f"interface_audit_{key}_v34.json"
    write_json_atomic(path,result)
    stage=stage_freeze(root,f"interface_{key}",[SOURCE,"src/jclosure/experiments/functional_hooks_v34.py",str(path.relative_to(root)),"artifacts/functional_mediation_v34_design.freeze.json"],{"model_key":key,"audit_hash":hd(result),"bitwise_baseline_replay":True,"all_stage_writebacks_exact":True,"formal_development_or_validation_observed":False})
    return {"freeze_digest":stage["freeze_digest"],"model":key,"layers":result["recurrent_layers_audited"],"interventions":len(interventions)}


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("model",choices=("Q","F"))
    args=parser.parse_args()
    print(json.dumps(run(Path.cwd(),args.model),indent=2))
