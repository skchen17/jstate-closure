"""Explanatory eight-condition native trajectory traces on frozen subset."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch

from jclosure.experiments.depth_common_v35 import prepare
from jclosure.experiments.functional_hooks_v34 import FunctionalIntervention
from jclosure.experiments.natural_reconstruction_v37 import _selective_rec
from jclosure.experiments.runtime_v34 import load, step
from jclosure.experiments.transaction_v28 import thash
from jclosure.protocol_v38 import stage_freeze, verify, verify_stage
from jclosure.provenance import sha256_file, write_json_atomic
from jclosure.recorder import ActivationRecorder


OUT=Path("results/v38/processed")
SOURCE="src/jclosure/experiments/trace_v38.py"
STAGES=("POSTCONV_INPUT","TRANSFORMED_CONTROL","TRUE_UPDATE",
        "RECURRENT_READ","RESIDUAL_INTEGRATION")


def _tensor_digest(obj):
    if isinstance(obj,torch.Tensor):return thash(obj)
    if isinstance(obj,(tuple,list)):return [_tensor_digest(x) for x in obj]
    return None


@torch.no_grad()
def run(root:Path,key:str):
    verify_stage(root,f"trajectory_full_{key}_development")
    plan=json.loads((root/OUT/"execution_plan_v38.json").read_text())
    design=json.loads((root/OUT/f"design_{key}_v38.json").read_text())
    panel=json.loads((root/OUT/"panel_v38.json").read_text())
    prompts={x["base_trial_id"]:x["prompt"] for r in
             ("calibration","development","validation","independent_final") for x in panel[r]}
    ids={family:plan["task_subsets"]["development"][family][0]
         for family in verify(root)["config"]["families"]}
    items={x["base_trial_id"]:x for x in design["development"]}
    groups=design["relative_depth_layers"]
    conditions=verify(root)["config"]["conditions"]
    boundaries={name:layers[-1] for name,layers in groups.items() if name in ("Q2","Q3","Q4")}
    model,tokenizer=load(root,key)
    rows=[]
    for family,sid in ids.items():
        item=items[sid]
        caches=prepare(model,tokenizer,key,item,prompts[sid])
        for condition,chosen in conditions.items():
            if chosen:
                layers=[layer for group in chosen for layer in groups[group]]
                cache,_=_selective_rec(caches["conv"],caches["joint"],layers)
            else:cache=caches["conv"]
            before={name:{"REC":thash(cache.layers[layer].recurrent_states),
                          "Conv":thash(cache.layers[layer].conv_states)}
                    for name,layer in boundaries.items()}
            with ActivationRecorder(model.model.layers,at=sorted(boundaries.values()),
                                    clone=True,detach=True) as recorder:
                with FunctionalIntervention(model,key) as hook:
                    output=step(model,cache,item["future_probe_tokens"][0],
                                caches["length"]+1)
            stages={name:{stage:_tensor_digest(hook.capture[layer].get(stage)) for stage in STAGES}
                    for name,layer in boundaries.items()}
            hidden={name:thash(recorder.activations[layer]) for name,layer in boundaries.items()}
            rows.append({"state_id":sid,"family":family,"model":key,
                         "condition":condition,"probe_index":0,
                         "boundary_layers_json":json.dumps(boundaries,sort_keys=True),
                         "incoming_state_hashes_json":json.dumps(before,sort_keys=True),
                         "native_factors_and_read_hashes_json":json.dumps(stages,sort_keys=True),
                         "realized_hidden_hashes_json":json.dumps(hidden,sort_keys=True),
                         "output_logits_hash":thash(output["logits"]),
                         "future_output_copy":False})
        print(f"V38 trace {key} {family}",flush=True)
    frame=pd.DataFrame(rows)
    table=root/OUT/f"trajectory_trace_{key}_development_v38.parquet"
    frame.to_parquet(table,index=False,compression="zstd")
    summary={"model":key,"role":"development","states":len(ids),
             "conditions":list(conditions),"rows":len(frame),"boundaries":boundaries,
             "one_probe_explanatory_only":True,"future_output_copy":False,
             "table_sha256":sha256_file(table)}
    path=root/OUT/f"trajectory_trace_{key}_development_v38.json"
    write_json_atomic(path,summary)
    seal=stage_freeze(root,f"trajectory_trace_{key}_development",
                      [SOURCE,str(path.relative_to(root)),str(table.relative_to(root)),
                       f"artifacts/trajectory_composition_v38_trajectory_full_{key}_development.freeze.json"],
                      {"model":key,"summary_sha256":sha256_file(path),"future_output_copy":False})
    return {"freeze_digest":seal["freeze_digest"],**summary}


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("model",choices=("Q","F"))
    a=p.parse_args();print(json.dumps(run(Path.cwd(),a.model),indent=2))
