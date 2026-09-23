"""Fresh V37 calibration audit of dependent old-state/update terms."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from jclosure.experiments.local_v36 import _capture, _module
from jclosure.experiments.operator_v36 import factors, parts
from jclosure.experiments.runtime_v34 import load, native_swap, prefix, step
from jclosure.experiments.transaction_v28 import thash
from jclosure.protocol_v37 import stage_freeze, verify_stage
from jclosure.provenance import sha256_file, write_json_atomic


OUT=Path("results/v37/processed")
SOURCE="src/jclosure/experiments/native_terms_v37.py"
LETTERS=("A","B","C")


def _norm(x):
    return float(x.float().norm().item())


def _read_term(key,f,term):
    if key=="Q":
        return (term.float()*f["q"].unsqueeze(-1)).sum(dim=-2).reshape(-1)
    c=f["C"]
    return (term.float()*c.unsqueeze(-2)).sum(dim=-1).reshape(-1)


@torch.no_grad()
def run(root:Path,key:str):
    verify_stage(root,f"calibration_{key}")
    design=json.loads((root/OUT/f"design_{key}_v37.json").read_text())
    panel=json.loads((root/OUT/"panel_v37.json").read_text())
    prompts={r["base_trial_id"]:r["prompt"] for r in panel["calibration"]}
    model,tokenizer=load(root,key)
    layers=list(design["local_layers"].values())
    rows=[]
    for n,item in enumerate(design["calibration"],1):
        sid=item["base_trial_id"]
        incoming,length,_,_=prefix(model,tokenizer,key,prompts[sid])
        caches={letter:step(model,incoming,token,length)["cache"]
                for letter,token in zip(LETTERS,(item["recipient_token_id"],
                                             item["donor_token_id"],item["third_token_id"]),strict=True)}
        conv,_=native_swap(caches["A"],caches["B"],["Conv"],key)
        for pi,probe in enumerate(item["future_probe_tokens"]):
            _,hidden,_,_=_capture(model,key,conv,probe,length,design["target_bundle"],layers)
            for position,layer in design["local_layers"].items():
                module=_module(model,key,layer)
                sa,sb=(caches[letter].layers[layer].recurrent_states for letter in ("A","B"))
                for operator in LETTERS:
                    f=factors(key,module,hidden[layer],caches[operator])
                    a,b=parts(key,f,sa),parts(key,f,sb)
                    old=b["decayed"].float()-a["decayed"].float()
                    update=b["update"].float()-a["update"].float()
                    old_read=_read_term(key,f,old)
                    update_read=_read_term(key,f,update)
                    raw=b["raw"].float().reshape(-1)-a["raw"].float().reshape(-1)
                    projected=old_read+update_read
                    if key=="Q":
                        input_write=f["k"].unsqueeze(-1)*(f["beta"].unsqueeze(-1).unsqueeze(-1)*f["v"].unsqueeze(-2))
                        removal_a=a["update"].float()-input_write
                        removal_b=b["update"].float()-input_write
                        input_hash=thash(input_write)
                        removal_difference_norm=_norm(removal_b-removal_a)
                    else:
                        input_hash=thash(f["dBx"])
                        removal_difference_norm=0.0
                    rows.append({"state_id":sid,"model":key,"family":item["family"],
                                 "position":position,"layer":layer,"probe_index":pi,
                                 "operator":operator,"old_state_hash_A":thash(sa),
                                 "old_state_hash_B":thash(sb),"factor_hashes_json":json.dumps(
                                     {name:thash(value) for name,value in f.items() if isinstance(value,torch.Tensor)},sort_keys=True),
                                 "input_write_hash":input_hash,
                                 "old_term_difference_norm":_norm(old),
                                 "update_term_difference_norm":_norm(update),
                                 "old_read_difference_norm":_norm(old_read),
                                 "update_read_difference_norm":_norm(update_read),
                                 "raw_difference_norm":_norm(raw),
                                 "read_term_sum_residual":_norm(raw-projected),
                                 "removal_difference_norm":removal_difference_norm,
                                 "update_independent_of_old_state":bool(torch.equal(a["update"],b["update"])),
                                 "calibration_only":True})
        if n%5==0 or n==len(design["calibration"]):
            print(f"V37 native terms {key} {n}/{len(design['calibration'])}",flush=True)
    frame=pd.DataFrame(rows)
    table=root/OUT/f"native_terms_{key}_v37.parquet"
    frame.to_parquet(table,index=False,compression="zstd")
    summary={"model":key,"calibration_states":len(design["calibration"]),
             "rows":len(frame),"median_old_read_difference_norm":float(frame.old_read_difference_norm.median()),
             "median_update_read_difference_norm":float(frame.update_read_difference_norm.median()),
             "median_raw_difference_norm":float(frame.raw_difference_norm.median()),
             "max_read_term_sum_residual":float(frame.read_term_sum_residual.max()),
             "fraction_update_independent_old_state":float(frame.update_independent_of_old_state.mean()),
             "calibration_only":True,"no_old_update_causal_dominance_claim":True,
             "table_sha256":sha256_file(table)}
    path=root/OUT/f"native_terms_{key}_v37.json"
    write_json_atomic(path,summary)
    seal=stage_freeze(root,f"native_terms_{key}",
                      [SOURCE,str(path.relative_to(root)),str(table.relative_to(root)),
                       f"artifacts/computational_origin_v37_calibration_{key}.freeze.json"],
                      {"model":key,"summary_sha256":sha256_file(path),"calibration_only":True})
    return {"freeze_digest":seal["freeze_digest"],**summary}


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("model",choices=("Q","F"))
    a=p.parse_args();print(json.dumps(run(Path.cwd(),a.model),indent=2))
