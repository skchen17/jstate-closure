"""One-shot, preselected V37 local read-law independent-final test."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from jclosure.experiments.local_v36 import _capture, _local_cache, _module
from jclosure.experiments.native_operator_v37 import predict_with_factors
from jclosure.experiments.operator_v36 import factors, native_local
from jclosure.experiments.runtime_v34 import load, native_swap, prefix, step
from jclosure.experiments.transaction_v28 import thash
from jclosure.protocol_v37 import stage_freeze, verify, verify_stage
from jclosure.provenance import sha256_file, write_json_atomic


OUT = Path("results/v37/processed")
SOURCE = "src/jclosure/experiments/final_operator_v37.py"
LETTERS = ("A", "B", "C")


def _context(root: Path, key: str):
    verify_stage(root, "final_opening")
    opening = json.loads((root/OUT/"final_opening_v37.json").read_text())
    if not opening["opened"] or opening["selected_class"] != "READ_OPERATOR_MATCHING":
        raise RuntimeError("V37 independent final not opened for local read law")
    verify_stage(root, "pool_independent_final")
    design = json.loads((root/OUT/f"design_{key}_v37.json").read_text())
    panel = json.loads((root/OUT/"panel_v37.json").read_text())
    prompts = {r["base_trial_id"]: r["prompt"] for r in panel["independent_final"]}
    return design, prompts


@torch.no_grad()
def predict(root: Path, key: str):
    design, prompts = _context(root, key)
    model, tokenizer = load(root, key)
    layers = list(design["local_layers"].values())
    rows, raw_vectors, mixer_vectors = [], [], []
    for n, item in enumerate(design["independent_final"], 1):
        sid = item["base_trial_id"]
        incoming, length, _, _ = prefix(model, tokenizer, key, prompts[sid])
        caches = {letter: step(model, incoming, token, length)["cache"]
                  for letter, token in zip(LETTERS,
                                           (item["recipient_token_id"], item["donor_token_id"],
                                            item["third_token_id"]), strict=True)}
        conv, _ = native_swap(caches["A"], caches["B"], ["Conv"], key)
        for pi, probe in enumerate(item["future_probe_tokens"]):
            _, hidden, _, _ = _capture(model, key, conv, probe, length,
                                       design["target_bundle"], layers)
            for position, layer in design["local_layers"].items():
                module = _module(model, key, layer)
                raw, mixer = [], []
                for state in LETTERS:
                    raw_row, mixer_row = [], []
                    for operator in LETTERS:
                        f = factors(key, module, hidden[layer], caches[operator])
                        result = predict_with_factors(key, module, hidden[layer], f,
                                                      caches[state].layers[layer].recurrent_states)
                        raw_row.append(result["raw"].float().cpu().numpy().ravel())
                        mixer_row.append(result["mixer"].float().cpu().numpy().ravel())
                    raw.append(raw_row)
                    mixer.append(mixer_row)
                r = np.asarray(raw, np.float32)
                m = np.asarray(mixer, np.float32)
                raw_vectors.append(r)
                mixer_vectors.append(m)
                rows.append({"state_id": sid, "model": key, "family": item["family"],
                             "position": position, "layer": layer, "probe_index": pi,
                             "vector_index": len(rows), "hidden_hash": thash(hidden[layer]),
                             "raw_hash": thash(torch.from_numpy(r)),
                             "mixer_hash": thash(torch.from_numpy(m)),
                             "outcomes_observed": False})
        if n%5 == 0 or n == len(design["independent_final"]):
            print(f"V37 final prediction {key} {n}/{len(design['independent_final'])}",flush=True)
    frame = pd.DataFrame(rows)
    paths = {"rows": root/OUT/f"final_prediction_{key}_v37.parquet",
             "vectors": root/OUT/f"final_prediction_vectors_{key}_v37.npz"}
    frame.to_parquet(paths["rows"],index=False,compression="zstd")
    np.savez_compressed(paths["vectors"],raw=np.stack(raw_vectors),
                        mixer=np.stack(mixer_vectors),
                        state_ids=frame.state_id.to_numpy(str),
                        positions=frame.position.to_numpy(str),
                        probes=frame.probe_index.to_numpy(np.int8))
    summary = {"model":key,"states":len(design["independent_final"]),
               "prediction_rows":len(frame),"outcomes_observed":False,
               "files_sha256":{name:sha256_file(p) for name,p in paths.items()}}
    path = root/OUT/f"final_prediction_{key}_v37.json"
    write_json_atomic(path,summary)
    seal = stage_freeze(root,f"final_prediction_{key}",
                        [SOURCE,str(path.relative_to(root)),
                         *[str(p.relative_to(root)) for p in paths.values()],
                         "artifacts/computational_origin_v37_final_opening.freeze.json"],
                        {"model":key,"summary_sha256":sha256_file(path),
                         "outcomes_observed":False})
    return {"freeze_digest":seal["freeze_digest"],**summary}


def _cos(x,y):
    den=float(np.linalg.norm(x)*np.linalg.norm(y))
    return float(np.dot(x,y)/den) if den>1e-9 else None


def _ordering(x,y):
    good=[]
    for i in range(len(x)):
        for j in range(i+1,len(x)):
            dx,dy=float(x[i]-x[j]),float(y[i]-y[j])
            if abs(dx)>1e-6 and abs(dy)>1e-6:good.append((dx>0)==(dy>0))
    return float(np.mean(good)) if good else None


@torch.no_grad()
def observe(root: Path, key: str):
    design, prompts = _context(root, key)
    for model_key in ("Q", "F"):
        verify_stage(root,f"final_prediction_{model_key}")
    frame=pd.read_parquet(root/OUT/f"final_prediction_{key}_v37.parquet")
    frozen=np.load(root/OUT/f"final_prediction_vectors_{key}_v37.npz")
    expected_raw,expected_mixer=frozen["raw"],frozen["mixer"]
    by_key={(r.state_id,r.position,int(r.probe_index)):r for r in frame.itertuples()}
    model,tokenizer=load(root,key)
    layers=list(design["local_layers"].values())
    rows, proofs=[],[]
    for n,item in enumerate(design["independent_final"],1):
        sid=item["base_trial_id"]
        incoming,length,_,_=prefix(model,tokenizer,key,prompts[sid])
        caches={letter:step(model,incoming,token,length)["cache"]
                for letter,token in zip(LETTERS,(item["recipient_token_id"],
                                             item["donor_token_id"],item["third_token_id"]),strict=True)}
        conv,_=native_swap(caches["A"],caches["B"],["Conv"],key)
        for pi,probe in enumerate(item["future_probe_tokens"]):
            _,hidden,_,_=_capture(model,key,conv,probe,length,design["target_bundle"],layers)
            for position,layer in design["local_layers"].items():
                r=by_key[(sid,position,pi)]
                if thash(hidden[layer])!=r.hidden_hash:
                    raise RuntimeError("V37 final hidden input drift")
                idx=int(r.vector_index)
                pr,pm=expected_raw[idx],expected_mixer[idx]
                if thash(torch.from_numpy(pr))!=r.raw_hash or thash(torch.from_numpy(pm))!=r.mixer_hash:
                    raise RuntimeError("V37 final prediction hash drift")
                module=_module(model,key,layer)
                raw,mix=[],[]
                for state in LETTERS:
                    rr,mm=[],[]
                    for operator in LETTERS:
                        work,proof=_local_cache(caches["A"],caches[state],caches[operator],layer)
                        result=native_local(module,hidden[layer],work)
                        rr.append(result["raw"].float().cpu().numpy().ravel())
                        mm.append(result["mixer"].float().cpu().numpy().ravel())
                        proofs.append({"state_id":sid,"model":key,"position":position,
                                       "probe_index":pi,"cell":state+operator,
                                       "requested_REC":proof["REC_requested"],
                                       "realized_REC":proof["REC_realized"],
                                       "requested_Conv":proof["Conv_requested"],
                                       "realized_Conv":proof["Conv_realized"]})
                    raw.append(rr);mix.append(mm)
                raw,mix=np.asarray(raw,np.float32),np.asarray(mix,np.float32)
                error_raw=float(np.max(np.abs(raw-pr)))
                error_mixer=float(np.max(np.abs(mix-pm)))
                if error_raw>.05 or error_mixer>.05:
                    raise RuntimeError(f"V37 final local mismatch {key}:{sid}:{position}:{error_raw}/{error_mixer}")
                def metrics(p,o):
                    pdiff=p.reshape(9,-1)-p[0,0]
                    odiff=o.reshape(9,-1)-o[0,0]
                    cos=[_cos(a,b) for a,b in zip(pdiff[1:],odiff[1:],strict=True)]
                    pn=np.linalg.norm(pdiff,axis=1);on=np.linalg.norm(odiff,axis=1)
                    rank=float(pd.Series(pn).rank().corr(pd.Series(on).rank()))
                    return float(np.median([c for c in cos if c is not None])),rank,_ordering(pn,on)
                rc,rr,ro=metrics(pr,raw)
                mc,mr,mo=metrics(pm,mix)
                rows.append({"state_id":sid,"model":key,"family":item["family"],
                             "position":position,"probe_index":pi,
                             "raw_cosine":rc,"raw_rank":rr,"raw_ordering":ro,
                             "mixer_cosine":mc,"mixer_rank":mr,"mixer_ordering":mo,
                             "max_raw_error":error_raw,"max_mixer_error":error_mixer})
        if n%5==0 or n==len(design["independent_final"]):
            print(f"V37 final observation {key} {n}/{len(design['independent_final'])}",flush=True)
    local=pd.DataFrame(rows)
    proof=pd.DataFrame(proofs)
    paths={"rows":root/OUT/f"final_observation_{key}_v37.parquet",
           "proof":root/OUT/f"final_proof_{key}_v37.parquet"}
    local.to_parquet(paths["rows"],index=False,compression="zstd")
    proof.to_parquet(paths["proof"],index=False,compression="zstd")
    state=local.groupby(["state_id","family"],as_index=False).median(numeric_only=True)
    gate=verify(root)["config"]["operator_gate"]
    family={}
    for name,part in state.groupby("family"):
        scores={k:float(part[k].median()) for k in ("raw_cosine","raw_rank","raw_ordering",
                                                   "mixer_cosine","mixer_rank","mixer_ordering")}
        family[name]={"states":len(part),"scores":scores,
                      "pass":all((scores[k]>=v for k,v in (("raw_cosine",gate["local_effect_cosine_min"]),
                         ("raw_rank",gate["relative_magnitude_rank_min"]),
                         ("raw_ordering",gate["pairwise_ordering_min"]),
                         ("mixer_cosine",gate["local_effect_cosine_min"]),
                         ("mixer_rank",gate["relative_magnitude_rank_min"]),
                         ("mixer_ordering",gate["pairwise_ordering_min"]))))}
    whole={k:float(state[k].median()) for k in ("raw_cosine","raw_rank","raw_ordering",
                                                "mixer_cosine","mixer_rank","mixer_ordering")}
    passed=all((whole[k]>=v for k,v in (("raw_cosine",gate["local_effect_cosine_min"]),
                       ("raw_rank",gate["relative_magnitude_rank_min"]),
                       ("raw_ordering",gate["pairwise_ordering_min"]),
                       ("mixer_cosine",gate["local_effect_cosine_min"]),
                       ("mixer_rank",gate["relative_magnitude_rank_min"]),
                       ("mixer_ordering",gate["pairwise_ordering_min"]))))
    passed=bool(passed and sum(f["pass"] for f in family.values())>=gate["families_required"])
    summary={"model":key,"states":len(state),"local_rows":len(local),
             "whole":whole,"families":family,"local_read_law_pass":passed,
             "max_raw_error":float(local.max_raw_error.max()),
             "max_mixer_error":float(local.max_mixer_error.max()),
             "natural_downstream_task_effect_tested":False,
             "files_sha256":{name:sha256_file(p) for name,p in paths.items()}}
    path=root/OUT/f"final_observation_{key}_v37.json"
    write_json_atomic(path,summary)
    seal=stage_freeze(root,f"final_observation_{key}",
                      [SOURCE,str(path.relative_to(root)),
                       *[str(p.relative_to(root)) for p in paths.values()],
                       *[f"artifacts/computational_origin_v37_final_prediction_{k}.freeze.json" for k in ("Q","F")]],
                      {"model":key,"summary_sha256":sha256_file(path),"pass":passed})
    return {"freeze_digest":seal["freeze_digest"],**summary}


def analyze(root: Path):
    verify_stage(root,"final_observation_Q")
    verify_stage(root,"final_observation_F")
    models={k:json.loads((root/OUT/f"final_observation_{k}_v37.json").read_text()) for k in ("Q","F")}
    result={"version":"V37","selected_class":"READ_OPERATOR_MATCHING",
            "both_models_local_read_law_pass":all(x["local_read_law_pass"] for x in models.values()),
            "models":models,"level_two_local_computation_only":True,
            "cross_layer_mechanism_not_claimed":True,"task_function_not_tested":True,
            "v36_outcomes_excluded":True}
    path=root/OUT/"final_analysis_v37.json"
    write_json_atomic(path,result)
    seal=stage_freeze(root,"final_analysis",
                      [SOURCE,str(path.relative_to(root)),
                       *[f"artifacts/computational_origin_v37_final_observation_{k}.freeze.json" for k in ("Q","F")]],
                      {"both_pass":result["both_models_local_read_law_pass"],
                       "summary_sha256":sha256_file(path)})
    return {"freeze_digest":seal["freeze_digest"],**result}


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("command",choices=("predict","observe","analyze"))
    p.add_argument("model",nargs="?",choices=("Q","F"))
    a=p.parse_args()
    if a.command!="analyze" and a.model is None:p.error("model required")
    fn={"predict":predict,"observe":observe,"analyze":analyze}[a.command]
    print(json.dumps(fn(Path.cwd(),*([a.model] if a.model else [])),indent=2))
