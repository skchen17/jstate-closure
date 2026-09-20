"""Exact least-squares projection of BF16-realized V16 cache deltas to frozen actions."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from jclosure.experiments.actuation_v15 import apply, components, load_context
from jclosure.experiments.action_bank_v16 import _row_for_z, _verify_splits
from jclosure.experiments.analyze_v16 import _features, _metrics, _ridge, _summary, _channel_weights
from jclosure.experiments.persistent_channels_v7 import _prefill
from jclosure.protocol_v16 import stage_freeze, verify
from jclosure.provenance import sha256_file, write_json_atomic

OUT=Path("results/v16/processed")
RAW=Path("results/v16/raw")
SOURCE="src/jclosure/experiments/realized_v16.py"
PROJECTION_DEVICE="cuda:1"  # model placement leaves substantially more free memory here


def _basis(cache:Any,directions:dict[str,Any],indices:list[int],rec:list[int],att:list[int])->torch.Tensor:
    rows=[]
    for index in indices:
        row={name:directions[name][index] for name in ("recurrent","conv","kv")}
        pieces=[direction.reshape(-1).to(PROJECTION_DEVICE,dtype=torch.float32) for _,_,_,_,direction in components(cache,row,rec,att)]
        rows.append(torch.cat(pieces))
    return torch.stack(rows)


def _delta(cache:Any,edited:Any,row:dict[str,torch.Tensor],rec:list[int],att:list[int])->torch.Tensor:
    pieces=[]
    for channel,layer,name,old,_ in components(cache,row,rec,att):
        new=getattr(edited.layers[layer],name)
        if channel in ("keys","values"):
            new=new[..., :old.shape[-2],:]
        pieces.append((new.float()-old.float()).reshape(-1).to(PROJECTION_DEVICE))
    return torch.cat(pieces)


def _state_indices(split:dict[str,Any], role:str, count:int)->list[dict[str,str]]:
    by_family:dict[str,int]={}
    rows=[]
    for item in split[role]:
        family=item["family"]
        used=by_family.get(family,0)
        if used<count:
            rows.append(item);by_family[family]=used+1
    return rows


def run(root:Path)->dict[str,Any]:
    split=_verify_splits(root);base=verify(root)
    stage_path=root/"artifacts/nonlinear_finite_causal_action_v16_realized.freeze.json"
    if not stage_path.exists():
        stage=stage_freeze(root,"realized",[SOURCE,str(Path("artifacts/nonlinear_finite_causal_action_v16_splits.freeze.json"))],
                           {"role":"train_fit_validation_test","state_subset_per_family":{"train":4,"validation":2},
                            "projection":"least-squares of exact BF16 state delta onto 32 frozen raw directions; pseudo-inverse rtol 1e-5",
                            "model_comparison":"same train/validation rows for requested and realized inputs"})
    else:
        stage=json.loads(stage_path.read_text())
        if sha256_file(root/SOURCE)!=stage["input_hashes"][SOURCE]:raise RuntimeError("realized source changed")
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    bundle,dense_map,v13,metadata,values=load_context(root)
    v8=v13["persistent_state_v8"]
    measured=[int(x) for x in v8["intervention"]["measured_layers"]]
    rec=[int(x) for x in v8["persistent_components"]["recurrent_layers"]]
    att=[int(x) for x in v8["persistent_components"]["attention_layers"]]
    indices=[int(x) for x in split["direction_indices"]]
    for role,count in (("train",4),("validation",2)):
        for item in _state_indices(split,role,count):
            path=root/RAW/f"realized_{role}_{item['base_trial_id']}.parquet"
            if path.exists():continue
            base_id=item["base_trial_id"]
            bank=pd.read_parquet(root/RAW/f"bank_{role}_{base_id}.parquet")
            task=values["tasks"][str(metadata[base_id]["prompt_id"])]
            clean=_prefill(bundle,task.prompt,measured_layers=measured,dense_map=dense_map,
                           intervention_layer=int(v8["intervention"]["layer"]),candidate=None)
            cache=clean["cache"]
            basis=_basis(cache,values["directions"],indices,rec,att)
            gram=basis@basis.T
            inverse=torch.linalg.pinv(gram,rtol=1e-5)
            rows=[]
            for _,record in bank[bank.reliability_status=="RELIABLE"].iterrows():
                z=np.asarray(record.requested_z,dtype=np.float32)
                alpha=np.asarray(record.requested_alpha,dtype=np.float32)
                row=_row_for_z(values["directions"],indices,alpha,z)
                edited=apply(cache,1.0,row,rec,att,"native_fp32_add_bf16_writeback")
                delta=_delta(cache,edited,row,rec,att)
                product=basis@delta
                coefficients=inverse@product
                total=float(torch.dot(delta,delta).item())
                projected=float(torch.dot(coefficients,product).item())
                residual=math.sqrt(max(total-projected,0.0)/max(total,1e-20))
                rows.append({"base_trial_id":base_id,"family":item["family"],"role":role,
                             "design":record.design,"direction_index":record.direction_index,
                             "requested_raw_coordinate":(alpha*z).tolist(),
                             "realized_raw_coordinate":coefficients.cpu().numpy().astype(np.float32).tolist(),
                             "realized_delta_residual_fraction":residual,
                             "realized_delta_norm":math.sqrt(total),
                             "response_j":record.response_j,"response_logits":record.response_logits,
                             "response_semantic_continuous":record.response_semantic_continuous,
                             "response_workspace":record.response_workspace,
                             "freeze_digest":stage["freeze_digest"]})
            pd.DataFrame(rows).to_parquet(path,index=False,compression="zstd")
            print(f"realized {role} {base_id} {len(rows)}",flush=True)
    paths=[root/RAW/f"realized_{role}_{item['base_trial_id']}.parquet" for role,count in (("train",4),("validation",2)) for item in _state_indices(split,role,count)]
    if not all(x.exists() for x in paths):raise RuntimeError("realized reconstruction incomplete")
    frame=pd.concat([pd.read_parquet(path) for path in paths],ignore_index=True)
    scales=split["target_scales"]
    lengths={name:len(frame.iloc[0][f"response_{name}"]) for name in ("j","logits","semantic_continuous","workspace")}
    y=np.stack([np.concatenate([np.asarray(row[f"response_{name}"],dtype=np.float64)/(float(scales[name])*math.sqrt(lengths[name])) for name in lengths]) for _,row in frame.iterrows()])
    channel=_channel_weights(root,indices)
    results=[]
    for k in (2,4,8,16,32):
        for model in ("linear","quadratic","channel_bilinear"):
            for input_name in ("requested_raw_coordinate","realized_raw_coordinate"):
                z=np.stack(frame[input_name].map(lambda x:np.asarray(x[:k],dtype=np.float64)))
                features=_features(z,model,channel[:k])
                train=frame.role.to_numpy()=="train";valid=~train
                weight=_ridge(features[train],y[train],float(base["config"]["model"]["ridge"]))
                predicted=features[valid]@weight
                met=_metrics(y[valid],predicted,lengths)
                results.append({"k":k,"model":model,"input":input_name,"train_count":int(train.sum()),"validation_count":int(valid.sum()),
                                **{f"{name}_{metric}":value for name,part in met.items() for metric,value in part.items()}})
    model_frame=pd.DataFrame(results)
    OUT.mkdir(parents=True,exist_ok=True)
    path=root/OUT/"realized_action_coordinates_v16.parquet"
    frame.to_parquet(path,index=False,compression="zstd")
    model_path=root/OUT/"requested_vs_realized_models_v16.parquet"
    model_frame.to_parquet(model_path,index=False,compression="zstd")
    summary={"freeze_digest":stage["freeze_digest"],"records":str(path),"records_sha256":sha256_file(path),
             "model_records":str(model_path),"model_records_sha256":sha256_file(model_path),
             "state_count_by_role":frame.groupby("role").base_trial_id.nunique().to_dict(),
             "action_count":len(frame),"median_span_residual_fraction":_summary(frame.realized_delta_residual_fraction.tolist()),
             "model_comparison":model_frame.to_dict("records")}
    write_json_atomic(root/OUT/"realized_action_coordinates_v16.json",summary)
    return {k:summary[k] for k in ("state_count_by_role","action_count","median_span_residual_fraction")}


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--stage",choices=("run",),required=True);parser.parse_args()
    print(json.dumps(run(Path.cwd()),sort_keys=True))
