"""Freeze Q2-induced Q3/Q4 operator predictions before local REC probes.

The only replayed read outputs here are in Q2, as an upstream causal context
perturbation. Execution stops at the later local mixer input: no later REC
counterfactual or six-probe endpoint is observed in this stage.
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
from jclosure.experiments.local_v36 import _module
from jclosure.experiments.native_operator_v37 import finite_state_difference
from jclosure.experiments.operator_v36 import factors
from jclosure.experiments.read_hooks_v35 import ReadPatch
from jclosure.experiments.runtime_v34 import load, step
from jclosure.experiments.transaction_v28 import thash
from jclosure.protocol_v37 import stage_freeze, verify_stage
from jclosure.provenance import sha256_file, write_json_atomic


OUT=Path("results/v37/processed")
SOURCE="src/jclosure/experiments/cross_layer_prediction_v37.py"
SITES=("B23","P3","B34","P4")


class _AtEntry(Exception):
    pass


@torch.no_grad()
def _entry(model,key,cache,probe,length,bundle,layer,references,mapping):
    """Return only the incoming mixer hidden tensor; abort before later read."""
    module=_module(model,key,layer)
    captured={}
    def stop(_module,args,kwargs):
        hidden=kwargs.get("hidden_states",args[0] if args else None)
        captured["hidden"]=hidden.detach().clone()
        raise _AtEntry()
    handle=module.register_forward_pre_hook(stop,with_kwargs=True)
    try:
        context=ReadPatch(model,key,references,mapping) if mapping else nullcontext()
        with context:
            try:
                step(model,cache,probe,length+1,bundle)
            except _AtEntry:
                pass
    finally:
        handle.remove()
    if "hidden" not in captured:
        raise RuntimeError(f"V37 target entry not reached: {key}:{layer}")
    return captured["hidden"]


def _norm(x):
    return float(x.float().norm().item())


@torch.no_grad()
def run(root:Path,key:str,role:str):
    if role not in ("development","validation"):
        raise ValueError(role)
    high=verify_stage(root,f"high_level_{role}")
    if not high["both_pass"]:
        raise RuntimeError("V37 high-level cross-model gate failed")
    if role=="validation":
        verify_stage(root,"cross_layer_analysis_development")
    design=json.loads((root/OUT/f"design_{key}_v37.json").read_text())
    panel=json.loads((root/OUT/"panel_v37.json").read_text())
    prompts={row["base_trial_id"]:row["prompt"] for r in
             ("calibration","development","validation","independent_final") for row in panel[r]}
    q2=design["relative_depth_layers"]["Q2"]
    model,tokenizer=load(root,key)
    rows,raw_vectors,mixer_vectors=[],[],[]
    for n,item in enumerate(design[role],1):
        sid=item["base_trial_id"]
        caches=prepare(model,tokenizer,key,item,prompts[sid])
        length=caches["length"]
        for pi,probe in enumerate(item["future_probe_tokens"]):
            # A native joint replay supplies the frozen Q2 read reference only.
            # Its final endpoint is deliberately ignored in this stage.
            _,joint_ref=capture(model,key,caches["joint"],probe,length,design["target_bundle"])
            refs={"JOINT":joint_ref}
            mapping={layer:"JOINT" for layer in q2}
            for position in SITES:
                layer=design["local_layers"][position]
                module=_module(model,key,layer)
                hs={"CC":_entry(model,key,caches["conv"],probe,length,
                                design["target_bundle"],layer,refs,{}),
                    "JC":_entry(model,key,caches["conv"],probe,length,
                                design["target_bundle"],layer,refs,mapping)}
                effects={}
                factor_hashes={}
                for context,h in hs.items():
                    effect=finite_state_difference(key,module,h,caches["conv"],
                              caches["conv"].layers[layer].recurrent_states,
                              caches["joint"].layers[layer].recurrent_states)
                    effects[context]=effect
                    f=factors(key,module,h,caches["conv"])
                    factor_hashes[context]={name:thash(value) for name,value in f.items()
                                            if isinstance(value,torch.Tensor)}
                raw=np.stack([effects[c]["raw"].cpu().numpy().ravel() for c in ("CC","JC")]).astype(np.float32)
                mixer=np.stack([effects[c]["mixer"].cpu().numpy().ravel() for c in ("CC","JC")]).astype(np.float32)
                raw_vectors.append(raw)
                mixer_vectors.append(mixer)
                changed=sum(factor_hashes["CC"].get(name)!=factor_hashes["JC"].get(name)
                            for name in set(factor_hashes["CC"])|set(factor_hashes["JC"]))
                rows.append({"state_id":sid,"model":key,"role":role,"family":item["family"],
                             "position":position,"layer":layer,"probe_index":pi,"vector_index":len(rows),
                             "hidden_CC_hash":thash(hs["CC"]),"hidden_JC_hash":thash(hs["JC"]),
                             "hidden_change_norm":_norm(hs["JC"]-hs["CC"]),
                             "factor_hashes_json":json.dumps(factor_hashes,sort_keys=True),
                             "changed_factor_count":changed,
                             "predicted_raw_CC_norm":float(np.linalg.norm(raw[0])),
                             "predicted_raw_JC_norm":float(np.linalg.norm(raw[1])),
                             "predicted_mixer_CC_norm":float(np.linalg.norm(mixer[0])),
                             "predicted_mixer_JC_norm":float(np.linalg.norm(mixer[1])),
                             "predicted_raw_change_norm":float(np.linalg.norm(raw[1]-raw[0])),
                             "predicted_mixer_change_norm":float(np.linalg.norm(mixer[1]-mixer[0])),
                             "raw_prediction_hash":thash(torch.from_numpy(raw)),
                             "mixer_prediction_hash":thash(torch.from_numpy(mixer)),
                             "later_REC_outcome_observed":False,
                             "Q2_JOINT_output_injected_only_upstream":True})
        if n%5==0 or n==len(design[role]):
            print(f"V37 cross-layer prediction {key} {role} {n}/{len(design[role])}",flush=True)
    frame=pd.DataFrame(rows)
    files={"rows":root/OUT/f"cross_layer_prediction_{key}_{role}_v37.parquet",
           "vectors":root/OUT/f"cross_layer_prediction_vectors_{key}_{role}_v37.npz"}
    frame.to_parquet(files["rows"],index=False,compression="zstd")
    np.savez_compressed(files["vectors"],raw=np.stack(raw_vectors),mixer=np.stack(mixer_vectors),
                        state_ids=frame.state_id.to_numpy(str),positions=frame.position.to_numpy(str),
                        probe_indices=frame.probe_index.to_numpy(np.int8),contexts=np.asarray(("CC","JC")))
    summary={"model":key,"role":role,"states":len(design[role]),"prediction_rows":len(frame),
             "sites":SITES,"later_REC_outcomes_observed":False,
             "files_sha256":{name:sha256_file(path) for name,path in files.items()}}
    path=root/OUT/f"cross_layer_prediction_{key}_{role}_v37.json"
    write_json_atomic(path,summary)
    seal=stage_freeze(root,f"cross_layer_prediction_{key}_{role}",
                      [SOURCE,str(path.relative_to(root)),
                       *[str(p.relative_to(root)) for p in files.values()],
                       "src/jclosure/experiments/native_operator_v37.py"],
                      {"model":key,"role":role,"summary_sha256":sha256_file(path),
                       "later_REC_outcomes_observed":False})
    return {"freeze_digest":seal["freeze_digest"],**summary}


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("model",choices=("Q","F"))
    p.add_argument("role",choices=("development","validation"))
    a=p.parse_args();print(json.dumps(run(Path.cwd(),a.model,a.role),indent=2))
