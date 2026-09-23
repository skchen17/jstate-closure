"""Diagnostic finite-amplitude REC interpolation after F development class freeze."""
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
from jclosure.protocol_v38 import stage_freeze, verify, verify_stage
from jclosure.provenance import sha256_file, write_json_atomic


OUT=Path("results/v38/processed")
SOURCE="src/jclosure/experiments/dose_v38.py"
ORDER=("R000","R100","R010","R001","R110","R101","R011","R111")


def plan(root:Path)->dict:
    verify_stage(root,"composition_development")
    analysis=json.loads((root/OUT/"trajectory_analysis_F_development_v38.json").read_text())
    if not analysis["class_passes"]["higher_order"]:
        raise RuntimeError("V38 F development class not frozen")
    cfg=verify(root)["config"]
    prior=json.loads((root/OUT/"execution_plan_v38.json").read_text())
    ids=[sid for family in cfg["families"] for sid in prior["task_subsets"]["development"][family]]
    result={"model":"F","role":"development","state_ids":ids,
            "lambdas":cfg["dose_lambdas"],"conditions":ORDER,
            "interpolation":"recipient_REC + lambda*(donor_REC-recipient_REC) on donor Conv background",
            "lambda_1_is_exact_natural_donor_REC":True,
            "off_manifold_diagnostic_only":True,"no_cross_model_dose_class":True}
    path=root/OUT/"dose_plan_v38.json"
    write_json_atomic(path,result)
    seal=stage_freeze(root,"dose_plan",[SOURCE,str(path.relative_to(root)),
                      "artifacts/trajectory_composition_v38_composition_development.freeze.json"],
                      {"model":"F","states":len(ids),"off_manifold_diagnostic_only":True,
                       "summary_sha256":sha256_file(path)})
    return {"freeze_digest":seal["freeze_digest"],**result}


def _blend(base,donor,layers,lam):
    work=clone_hybrid_cache(base)
    for layer in layers:
        x=base.layers[layer].recurrent_states
        y=donor.layers[layer].recurrent_states
        z=y.detach().clone() if lam==1.0 else (x.float()+lam*(y.float()-x.float())).to(x.dtype)
        if not torch.isfinite(z.float()).all():raise RuntimeError("V38 dose nonfinite")
        work.layers[layer].recurrent_states=z
    return work


@torch.no_grad()
def run(root:Path)->dict:
    verify_stage(root,"dose_plan")
    p=json.loads((root/OUT/"dose_plan_v38.json").read_text())
    design=json.loads((root/OUT/"design_F_v38.json").read_text())
    panel=json.loads((root/OUT/"panel_v38.json").read_text())
    prompts={x["base_trial_id"]:x["prompt"] for r in
             ("calibration","development","validation","independent_final") for x in panel[r]}
    lookup={x["base_trial_id"]:x for x in design["development"]}
    cfg=verify(root)["config"]
    groups=design["relative_depth_layers"]
    model,tokenizer=load(root,"F")
    primary={}
    for stage in ("singles","pairs","full"):
        arch=np.load(root/OUT/f"trajectory_{stage}_F_development_v38.npz")
        for sid,vec in zip(arch["state_ids"].tolist(),arch["vectors"]):
            primary.setdefault(sid,{})
            for name,x in zip(arch["conditions"].tolist(),vec):primary[sid][name]=x
    rows,vectors=[],[]
    natural_replay_max=0.0
    for n,sid in enumerate(p["state_ids"],1):
        item=lookup[sid]
        caches=prepare(model,tokenizer,"F",item,prompts[sid])
        for lam in p["lambdas"]:
            ys=[]
            for name in ORDER:
                chosen=cfg["conditions"][name]
                layers=[layer for group in chosen for layer in groups[group]]
                cache=_blend(caches["conv"],caches["joint"],layers,float(lam))
                vector=signature([step(model,cache,probe,caches["length"]+1,design["target_bundle"])
                                  for probe in item["future_probe_tokens"]],
                                 design["calibration_clean_scales"])
                if lam==1.0:
                    natural_replay_max=max(natural_replay_max,
                                           float(np.max(np.abs(vector-primary[sid][name]))))
                ys.append(vector)
            y=np.stack(ys)
            e=y-y[0]
            e100,e010,e001,e110,e101,e011,e111=e[1:]
            additive=e100+e010+e001
            second=e110+e101+e011-e100-e010-e001
            denom=max(float(np.linalg.norm(e111)),cfg["effect_norm_floor"])
            rows.append({"state_id":sid,"family":item["family"],"lambda":lam,
                         "vector_index":len(vectors),
                         "additive_relative_error":float(np.linalg.norm(e111-additive))/denom,
                         "second_relative_error":float(np.linalg.norm(e111-second))/denom,
                         "threeway_fraction":float(np.linalg.norm(e111-second))/denom,
                         "natural_lambda":lam==1.0,"off_manifold":lam<1.0})
            vectors.append(y.astype(np.float32))
        if n%5==0 or n==len(p["state_ids"]):
            print(f"V38 F dose {n}/{len(p['state_ids'])}",flush=True)
    if natural_replay_max>1e-4:raise RuntimeError(f"V38 lambda=1 natural replay drift: {natural_replay_max}")
    frame=pd.DataFrame(rows)
    files={"rows":root/OUT/"dose_F_development_v38.parquet",
           "vectors":root/OUT/"dose_vectors_F_development_v38.npz"}
    frame.to_parquet(files["rows"],index=False,compression="zstd")
    np.savez_compressed(files["vectors"],vectors=np.stack(vectors),
                        conditions=np.asarray(ORDER),state_ids=frame.state_id.to_numpy(str),
                        lambdas=frame["lambda"].to_numpy(float))
    summary={"model":"F","states":len(p["state_ids"]),"rows":len(frame),
             "lambda_1_replay_max_abs":natural_replay_max,
             "by_lambda":{str(lam):{name:float(g[name].median()) for name in
                          ("additive_relative_error","second_relative_error","threeway_fraction")}
                          for lam,g in frame.groupby("lambda")},
             "off_manifold_diagnostic_only":True,
             "files_sha256":{k:sha256_file(v) for k,v in files.items()}}
    path=root/OUT/"dose_F_development_v38.json"
    write_json_atomic(path,summary)
    seal=stage_freeze(root,"dose_F_development",[SOURCE,str(path.relative_to(root)),
                      *[str(v.relative_to(root)) for v in files.values()],
                      "artifacts/trajectory_composition_v38_dose_plan.freeze.json"],
                      {"model":"F","summary_sha256":sha256_file(path),
                       "off_manifold_diagnostic_only":True})
    return {"freeze_digest":seal["freeze_digest"],**summary}


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("action",choices=("plan","run"))
    a=p.parse_args();print(json.dumps(plan(Path.cwd()) if a.action=="plan" else run(Path.cwd()),indent=2))
