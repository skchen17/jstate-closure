"""Q2/Q3/Q4 REC-state reconstruction with native downstream computation.

The Conv-only recipient cache is the common baseline. Selected recurrent states
are swapped from the natural donor once, before the future token; all mixer
outputs and all later hidden/operator computations then evolve natively.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from jclosure.cache_v7 import clone_hybrid_cache
from jclosure.experiments.depth_common_v35 import prepare
from jclosure.experiments.runtime_v34 import load, signature, step
from jclosure.experiments.transaction_v28 import thash
from jclosure.protocol_v37 import stage_freeze, verify_stage
from jclosure.provenance import sha256_file, write_json_atomic


OUT=Path("results/v37/processed")
SOURCE="src/jclosure/experiments/natural_reconstruction_v37.py"
CONDITIONS=("CC","Q2_STATE","Q34_STATE","Q234_STATE","FULL_JOINT")


def _cos(a,b):
    x,y=np.asarray(a,np.float64).ravel(),np.asarray(b,np.float64).ravel()
    d=float(np.linalg.norm(x)*np.linalg.norm(y))
    return float(np.dot(x,y)/d) if d>1e-12 else None


def _selective_rec(base,donor,layers):
    work=clone_hybrid_cache(base)
    proofs=[]
    for layer in layers:
        source=donor.layers[layer].recurrent_states
        target=work.layers[layer].recurrent_states
        if source.shape!=target.shape or source.dtype!=target.dtype:
            raise RuntimeError(f"V37 selective REC topology mismatch: {layer}")
        work.layers[layer].recurrent_states=source.detach().clone()
        actual=work.layers[layer].recurrent_states
        if not torch.equal(actual,source):
            raise RuntimeError(f"V37 selective REC exact writeback failed: {layer}")
        proofs.append({"layer":layer,"requested_hash":thash(source),
                       "realized_hash":thash(actual),"baseline_hash":thash(target)})
    return work,proofs


@torch.no_grad()
def run(root:Path,key:str,role:str):
    if role not in ("development","validation"):
        raise ValueError(role)
    gate=verify_stage(root,f"read_gate_{role}")
    if not gate["both_pass"]:
        raise RuntimeError("V37 read interface gate failed")
    if role=="validation":
        verify_stage(root,"natural_reconstruction_Q_development")
        verify_stage(root,"natural_reconstruction_F_development")
    design=json.loads((root/OUT/f"design_{key}_v37.json").read_text())
    panel=json.loads((root/OUT/"panel_v37.json").read_text())
    prompts={row["base_trial_id"]:row["prompt"] for r in
             ("calibration","development","validation","independent_final") for row in panel[r]}
    high=pd.read_parquet(root/OUT/f"factorial_{key}_{role}_v37.parquet")
    hv=np.load(root/OUT/f"factorial_vectors_{key}_{role}_v37.npz")["vectors"].astype(np.float64)
    high_by_id={row.state_id:hv[int(row.vector_index)] for row in high.itertuples()}
    interface=pd.read_parquet(root/OUT/f"read_baselines_{key}_{role}_v37.parquet")
    iv=np.load(root/OUT/f"read_baselines_vectors_{key}_{role}_v37.npz")["vectors"].astype(np.float64)
    interface_by_id={row.state_id:iv[int(row.vector_index)] for row in interface.itertuples()}
    groups=design["relative_depth_layers"]
    selections={"Q2_STATE":groups["Q2"],
                "Q34_STATE":groups["Q3"]+groups["Q4"],
                "Q234_STATE":groups["Q2"]+groups["Q3"]+groups["Q4"]}
    model,tokenizer=load(root,key)
    rows,vectors,audits=[],[],[]
    for n,item in enumerate(design[role],1):
        sid=item["base_trial_id"]
        caches=prepare(model,tokenizer,key,item,prompts[sid])
        branches={"CC":caches["conv"],"FULL_JOINT":caches["joint"]}
        for name,layers in selections.items():
            branches[name],proof=_selective_rec(caches["conv"],caches["joint"],layers)
            audits.append({"state_id":sid,"model":key,"role":role,"condition":name,
                           "layer_count":len(layers),"proof_json":json.dumps(proof,sort_keys=True),
                           "all_requested_realized_exact":all(p["requested_hash"]==p["realized_hash"] for p in proof),
                           "later_output_copy":False})
        ys={}
        for name,cache in branches.items():
            output=[step(model,cache,probe,caches["length"]+1,design["target_bundle"])
                    for probe in item["future_probe_tokens"]]
            ys[name]=signature(output,design["calibration_clean_scales"])
        baseline=high_by_id[sid]
        if not np.allclose(ys["CC"],baseline[2],rtol=1e-5,atol=1e-5) or \
           not np.allclose(ys["FULL_JOINT"],baseline[3],rtol=1e-5,atol=1e-5):
            raise RuntimeError(f"V37 natural reconstruction factual replay drift {key}:{sid}")
        donor=baseline[4]
        econv=float(np.linalg.norm(donor-ys["CC"]))
        ejoint=float(np.linalg.norm(donor-ys["FULL_JOINT"]))
        benefit=econv-ejoint
        def gain(name):return econv-float(np.linalg.norm(donor-ys[name]))
        natural_q2_enablement=(float(np.linalg.norm(donor-ys["Q34_STATE"]))
                               -float(np.linalg.norm(donor-ys["Q234_STATE"])))
        interface_q2_enablement=interface_by_id[sid][2]-interface_by_id[sid][3]
        rows.append({"state_id":sid,"model":key,"role":role,"family":item["family"],
                     "vector_index":len(vectors),"natural_benefit_absolute":benefit,
                     "Q2_STATE_gain":gain("Q2_STATE"),
                     "Q34_STATE_gain":gain("Q34_STATE"),
                     "Q234_STATE_gain":gain("Q234_STATE"),
                     "Q234_fraction_of_full":gain("Q234_STATE")/benefit if benefit>1.0 else None,
                     "Q234_direction_cosine_to_full":_cos(ys["Q234_STATE"]-ys["CC"],ys["FULL_JOINT"]-ys["CC"]),
                     "natural_Q2_enablement_absolute":natural_q2_enablement,
                     "interface_Q2_enablement_vector_cosine":_cos(ys["Q234_STATE"]-ys["Q34_STATE"],
                                                                  interface_q2_enablement),
                     "six_probes_one_state":True,"native_downstream":True,
                     "prerecorded_Q3_Q4_outputs_copied":False})
        vectors.append(np.stack([ys[name] for name in CONDITIONS]).astype(np.float32))
        if n%5==0 or n==len(design[role]):
            print(f"V37 natural reconstruction {key} {role} {n}/{len(design[role])}",flush=True)
    frames={"rows":pd.DataFrame(rows),"audit":pd.DataFrame(audits)}
    files={"rows":root/OUT/f"natural_reconstruction_{key}_{role}_v37.parquet",
           "audit":root/OUT/f"natural_reconstruction_audit_{key}_{role}_v37.parquet",
           "vectors":root/OUT/f"natural_reconstruction_vectors_{key}_{role}_v37.npz"}
    frames["rows"].to_parquet(files["rows"],index=False,compression="zstd")
    frames["audit"].to_parquet(files["audit"],index=False,compression="zstd")
    np.savez_compressed(files["vectors"],vectors=np.stack(vectors),
                        conditions=np.asarray(CONDITIONS),
                        state_ids=frames["rows"].state_id.to_numpy(str))
    summary={"model":key,"role":role,"states":len(design[role]),
             "median_Q234_fraction_of_full":float(frames["rows"].Q234_fraction_of_full.median()),
             "median_Q234_direction_cosine":float(frames["rows"].Q234_direction_cosine_to_full.median()),
             "median_natural_Q2_enablement":float(frames["rows"].natural_Q2_enablement_absolute.median()),
             "later_outputs_copied":False,
             "files_sha256":{name:sha256_file(path) for name,path in files.items()}}
    path=root/OUT/f"natural_reconstruction_{key}_{role}_v37.json"
    write_json_atomic(path,summary)
    seal=stage_freeze(root,f"natural_reconstruction_{key}_{role}",
                      [SOURCE,str(path.relative_to(root)),
                       *[str(p.relative_to(root)) for p in files.values()],
                       f"artifacts/computational_origin_v37_read_gate_{role}.freeze.json"],
                      {"model":key,"role":role,"summary_sha256":sha256_file(path),
                       "later_outputs_copied":False})
    return {"freeze_digest":seal["freeze_digest"],**summary}


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("model",choices=("Q","F"))
    p.add_argument("role",choices=("development","validation"))
    a=p.parse_args();print(json.dumps(run(Path.cwd(),a.model,a.role),indent=2))
