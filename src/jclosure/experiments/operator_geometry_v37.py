"""Calibration-only exact state-read operator spectra and Conv variation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from jclosure.experiments.local_v36 import _capture, _module
from jclosure.experiments.operator_v36 import factors
from jclosure.experiments.runtime_v34 import load, native_swap, prefix, step
from jclosure.experiments.transaction_v28 import thash
from jclosure.protocol_v37 import stage_freeze, verify_stage
from jclosure.provenance import sha256_file, write_json_atomic


OUT=Path("results/v37/processed")
SOURCE="src/jclosure/experiments/operator_geometry_v37.py"
LETTERS=("A","B","C")


def _rows(key,f):
    if key=="Q":
        q,k,beta,decay=f["q"].float(),f["k"].float(),f["beta"].float(),f["decay"].float()
        qk=(q*k).sum(dim=-1)
        return decay.unsqueeze(-1)*(q-beta.unsqueeze(-1)*qk.unsqueeze(-1)*k)
    return f["dA"].float()*f["C"].float().unsqueeze(-2)


def _spectrum(key,module,rows):
    singular=rows.square().sum(dim=-1).sqrt().reshape(-1)
    if key=="Q":singular=singular.repeat_interleave(module.head_v_dim)
    return torch.sort(singular,descending=True).values


def _cos(a,b):
    x,y=a.flatten(),b.flatten()
    denom=float(x.norm().item()*y.norm().item())
    return float(torch.dot(x,y).item()/denom) if denom>1e-12 else None


@torch.no_grad()
def run(root:Path,key:str):
    verify_stage(root,f"calibration_{key}")
    design=json.loads((root/OUT/f"design_{key}_v37.json").read_text())
    panel=json.loads((root/OUT/"panel_v37.json").read_text())
    prompts={row["base_trial_id"]:row["prompt"] for row in panel["calibration"]}
    model,tokenizer=load(root,key)
    layers=list(design["local_layers"].values())
    rows,spectra=[],[]
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
                operators={letter:_rows(key,factors(key,module,hidden[layer],caches[letter]))
                           for letter in LETTERS}
                for letter,operator in operators.items():
                    s=_spectrum(key,module,operator).cpu().numpy().astype(np.float32)
                    rows.append({"state_id":sid,"model":key,"family":item["family"],
                                 "position":position,"layer":layer,"probe_index":pi,
                                 "token_branch":letter,"spectrum_index":len(spectra),
                                 "operator_shape":str(tuple(operator.shape)),
                                 "operator_hash":thash(operator),
                                 "singular_values_hash":thash(torch.from_numpy(s)),
                                 "top_singular_value":float(s[0]),
                                 "median_singular_value":float(np.median(s)),
                                 "nonzero_singular_count":int(np.count_nonzero(s>1e-8)),
                                 "operator_norm":float(operator.norm().item()),
                                 "cosine_to_A":_cos(operator,operators["A"]),
                                 "norm_ratio_to_A":float(operator.norm().item()/max(operators["A"].norm().item(),1e-8)),
                                 "calibration_only":True,
                                 "state_operator_dependency":"independent of old REC under fixed h/Conv in real algebra"})
                    spectra.append(s)
        if n%5==0 or n==len(design["calibration"]):
            print(f"V37 operator geometry {key} {n}/{len(design['calibration'])}",flush=True)
    frame=pd.DataFrame(rows)
    files={"rows":root/OUT/f"operator_geometry_{key}_v37.parquet",
           "spectra":root/OUT/f"operator_spectra_{key}_v37.npz"}
    frame.to_parquet(files["rows"],index=False,compression="zstd")
    np.savez_compressed(files["spectra"],singular_values=np.stack(spectra),
                        state_ids=frame.state_id.to_numpy(str),
                        positions=frame.position.to_numpy(str),
                        token_branches=frame.token_branch.to_numpy(str))
    mismatch=frame[frame.token_branch!="A"]
    summary={"model":key,"calibration_states":len(design["calibration"]),
             "rows":len(frame),"singular_values_per_operator":len(spectra[0]),
             "median_operator_cosine_to_A":float(mismatch.cosine_to_A.median()),
             "median_operator_norm_ratio_to_A":float(mismatch.norm_ratio_to_A.median()),
             "calibration_only":True,"low_rank_causal_claim":False,
             "files_sha256":{name:sha256_file(path) for name,path in files.items()}}
    path=root/OUT/f"operator_geometry_{key}_v37.json"
    write_json_atomic(path,summary)
    seal=stage_freeze(root,f"operator_geometry_{key}",
                      [SOURCE,str(path.relative_to(root)),
                       *[str(p.relative_to(root)) for p in files.values()],
                       f"artifacts/computational_origin_v37_calibration_{key}.freeze.json"],
                      {"model":key,"summary_sha256":sha256_file(path),
                       "calibration_only":True})
    return {"freeze_digest":seal["freeze_digest"],**summary}


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("model",choices=("Q","F"))
    a=p.parse_args();print(json.dumps(run(Path.cwd(),a.model),indent=2))
