"""Predeclared V16 nonlinear-response, decomposition and reachability analysis."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from jclosure.experiments import geometry_v13 as geometry
from jclosure.experiments.action_bank_v16 import _verify_splits
from jclosure.protocol_v16 import stage_freeze, verify
from jclosure.provenance import sha256_file, write_json_atomic

OUT = Path("results/v16/processed")
RAW = Path("results/v16/raw")
SOURCE = "src/jclosure/experiments/analyze_v16.py"
SPLITS = Path("artifacts/nonlinear_finite_causal_action_v16_splits.freeze.json")
TARGETS = ("j", "logits", "semantic_continuous", "workspace", "stacked_normalized")


def _load(root: Path) -> pd.DataFrame:
    split = _verify_splits(root)
    paths = [root / RAW / f"bank_{role}_{item['base_trial_id']}.parquet" for role in ("train", "validation") for item in split[role]]
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise RuntimeError(f"V16 bank incomplete: {len(missing)} missing, first={missing[0]}")
    frame = pd.concat([pd.read_parquet(path) for path in paths], ignore_index=True)
    frame["action_key"] = frame.requested_z.map(lambda z: tuple(np.round(np.asarray(z, dtype=float), 5)))
    return frame


def _vector(row: pd.Series, target: str) -> np.ndarray:
    return np.asarray(row[f"response_{target}"], dtype=np.float64)


def _safe_cosine(a: np.ndarray, b: np.ndarray) -> float | None:
    norm = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / norm) if norm > 1e-15 else None


def _summary(x: list[float | None]) -> float | None:
    numbers = [float(v) for v in x if v is not None and np.isfinite(v)]
    return float(np.median(numbers)) if numbers else None


def _decomposition(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for (base_id, index), pair in frame[(frame.design == "single") & (frame.reliability_status == "RELIABLE")].groupby(["base_trial_id", "coordinate_index"]):
        if set(pair.sign.astype(int)) != {-1, 1}:
            continue
        plus = pair[pair.sign == 1].iloc[0]
        minus = pair[pair.sign == -1].iloc[0]
        for target in TARGETS:
            odd = (_vector(plus, target) - _vector(minus, target)) / 2
            even = (_vector(plus, target) + _vector(minus, target)) / 2
            rows.append({"kind": "odd_even", "base_trial_id": base_id, "family": plus.family,
                         "role": plus.role, "coordinate_index": str(int(index)), "target": target,
                         "even_over_odd": float(np.linalg.norm(even) / max(np.linalg.norm(odd), 1e-20)),
                         "odd_norm": float(np.linalg.norm(odd)), "even_norm": float(np.linalg.norm(even))})
    for base_id, group in frame[(frame.design.isin(["single", "scale", "pair", "triple"])) & (frame.reliability_status == "RELIABLE")].groupby("base_trial_id"):
        by_key = {row.action_key: row for _, row in group.iterrows()}
        width = len(group.iloc[0].requested_z)
        for index in range(4):
            for target in TARGETS:
                samples = []
                for sign in (-1.0, 1.0):
                    for scale in (0.5, 1.0, 1.5, 2.0):
                        z = np.zeros(width); z[index] = sign * scale
                        row = by_key.get(tuple(z))
                        if row is not None:
                            samples.append((sign * scale, _vector(row, target)))
                if len(samples) < 5:
                    continue
                a = np.asarray([x for x, _ in samples])
                y = np.stack([x for _, x in samples])
                coef = np.linalg.lstsq(np.column_stack([a, a * a, a**3]), y, rcond=None)[0]
                fitted = np.column_stack([a, a * a, a**3]) @ coef
                rows.append({"kind": "scale_fit", "base_trial_id": base_id, "family": group.iloc[0].family,
                             "role": group.iloc[0].role, "coordinate_index": str(index), "target": target,
                             "linear_norm": float(np.linalg.norm(coef[0])),
                             "quadratic_norm": float(np.linalg.norm(coef[1])),
                             "cubic_norm": float(np.linalg.norm(coef[2])),
                             "scale_fit_relative_l2": float(np.linalg.norm(y-fitted)/max(np.linalg.norm(y),1e-20)),
                             "quadratic_over_linear": float(np.linalg.norm(coef[1])/max(np.linalg.norm(coef[0]),1e-20))})
        for pair in ((0,1),(0,2),(1,3),(2,3)):
            a, b = pair
            za = np.zeros(width); za[a]=1
            zb = np.zeros(width); zb[b]=1
            zp = za+zb
            ra, rb, rp = (by_key.get(tuple(z)) for z in (za,zb,zp))
            if any(x is None for x in (ra,rb,rp)):
                continue
            for target in TARGETS:
                interaction = _vector(rp,target)-_vector(ra,target)-_vector(rb,target)
                rows.append({"kind":"pair_interaction","base_trial_id":base_id,"family":group.iloc[0].family,
                             "role":group.iloc[0].role,"coordinate_index":f"{a},{b}","target":target,
                             "interaction_norm":float(np.linalg.norm(interaction)),
                             "interaction_over_sum":float(np.linalg.norm(interaction)/max(np.linalg.norm(_vector(ra,target)+_vector(rb,target)),1e-20)),
                             "interaction_cosine_sum":_safe_cosine(interaction,_vector(ra,target)+_vector(rb,target))})
        for triple in ((0,1,2),(1,2,3)):
            z = np.zeros(width); z[list(triple)] = 1
            row = by_key.get(tuple(z))
            if row is None:
                continue
            for target in TARGETS:
                baseline = np.zeros_like(_vector(row,target))
                complete = True
                for size in (1,2):
                    from itertools import combinations
                    for subset in combinations(triple,size):
                        key = np.zeros(width); key[list(subset)] = 1
                        record = by_key.get(tuple(key))
                        if record is None:
                            complete = False; break
                        baseline += (-1 if size==1 else 1)*_vector(record,target)
                    if not complete: break
                if not complete: continue
                residual = _vector(row,target)-baseline
                rows.append({"kind":"triple_mobius","base_trial_id":base_id,"family":group.iloc[0].family,
                             "role":group.iloc[0].role,"coordinate_index":str(triple),"target":target,
                             "interaction_norm":float(np.linalg.norm(residual)),
                             "interaction_over_sum":float(np.linalg.norm(residual)/max(np.linalg.norm(_vector(row,target)),1e-20))})
    result = pd.DataFrame(rows)
    summary: dict[str, Any] = {}
    for kind, group in result.groupby("kind"):
        summary[kind] = {}
        for target, sub in group.groupby("target"):
            summary[kind][target] = {name: _summary(sub[name].tolist()) for name in ("even_over_odd","quadratic_over_linear","scale_fit_relative_l2","linear_norm","quadratic_norm","cubic_norm","interaction_norm","interaction_over_sum") if name in sub and sub[name].notna().any()}
            summary[kind][target]["count"] = int(len(sub))
    return result, summary


def _features(z: np.ndarray, model: str, channel: np.ndarray, cubic_indices: list[int] | None = None) -> np.ndarray:
    # All models have no intercept: a zero finite action predicts zero response.
    terms = [z]
    if model != "linear":
        outer = np.einsum("ni,nj->nij",z,z)
        iu = np.triu_indices(z.shape[1])
        terms.append(outer[:,iu[0],iu[1]])
    if model in ("channel_bilinear","sparse_cubic","piecewise_quadratic"):
        c = z @ channel
        terms.append(np.column_stack([c[:,0]*c[:,1],c[:,0]*c[:,2],c[:,1]*c[:,2]]))
    if model == "sparse_cubic":
        if cubic_indices is None:
            raise RuntimeError("sparse cubic terms must be selected using training rows")
        terms.append(z[:,cubic_indices]**3)
    return np.concatenate(terms,axis=1)


def _ridge(x: np.ndarray, y: np.ndarray, ridge: float) -> np.ndarray:
    gram = x.T @ x
    gram.flat[::len(gram)+1] += ridge
    return np.linalg.solve(gram, x.T @ y)


def _select_cubic(x:np.ndarray,y:np.ndarray,channel:np.ndarray,ridge:float)->list[int]:
    baseline=_features(x,"channel_bilinear",channel)
    residual=y-baseline@_ridge(baseline,y,ridge)
    cubic=x**3
    score=np.linalg.norm(cubic.T@residual,axis=1)/np.maximum(np.linalg.norm(cubic,axis=0),1e-20)
    return [int(i) for i in np.argsort(-score)[:min(16,x.shape[1])]]


def _metrics(y: np.ndarray, predicted: np.ndarray, lengths: dict[str,int]) -> dict[str,Any]:
    offsets = np.cumsum([0,*lengths.values()])
    out: dict[str,Any] = {}
    for i,target in enumerate(("j","logits","semantic_continuous","workspace")):
        actual = y[:,offsets[i]:offsets[i+1]]
        estimate = predicted[:,offsets[i]:offsets[i+1]]
        numer = np.einsum("ij,ij->i", actual,estimate)
        anorm = np.linalg.norm(actual,axis=1)
        enorm = np.linalg.norm(estimate,axis=1)
        good = (anorm>1e-12)&(enorm>1e-12)
        cosine = np.full(len(actual),np.nan)
        cosine[good]=numer[good]/(anorm[good]*enorm[good])
        out[target] = {"direction":_summary(cosine.tolist()),
                       "relative_l2":_summary((np.linalg.norm(actual-estimate,axis=1)/np.maximum(anorm,1e-20)).tolist()),
                       "magnitude_ratio":_summary((enorm/np.maximum(anorm,1e-20)).tolist())}
    anorm = np.linalg.norm(y,axis=1); enorm=np.linalg.norm(predicted,axis=1)
    good = (anorm>1e-12)&(enorm>1e-12)
    cos=np.full(len(y),np.nan); cos[good]=np.einsum("ij,ij->i",y[good],predicted[good])/(anorm[good]*enorm[good])
    out["stacked_normalized"]={"direction":_summary(cos.tolist()),
                                 "relative_l2":_summary((np.linalg.norm(y-predicted,axis=1)/np.maximum(anorm,1e-20)).tolist()),
                                 "magnitude_ratio":_summary((enorm/np.maximum(anorm,1e-20)).tolist())}
    return out


def _channel_weights(root: Path, indices: list[int]) -> np.ndarray:
    source=json.loads((root/"results/v13/processed/probe_directions_v13.json").read_text())["channel_score_energy"]
    energy=np.column_stack([np.asarray(source[name],dtype=float)[indices] for name in ("recurrent","conv","kv")])
    return np.sqrt(energy/np.maximum(energy.sum(axis=1,keepdims=True),1e-20))


def _model_compare(root: Path, frame: pd.DataFrame) -> tuple[pd.DataFrame,dict[str,Any]]:
    freeze=_verify_splits(root); cfg=verify(root)["config"]
    reliable=frame[frame.reliability_status=="RELIABLE"].copy()
    reliable=reliable[reliable.response_j.notna()].copy()
    lengths={name:len(np.asarray(reliable.iloc[0][f"response_{name}"])) for name in ("j","logits","semantic_continuous","workspace")}
    scales=freeze["target_scales"]
    for name in lengths:
        reliable[f"y_{name}"]=reliable[f"response_{name}"].map(lambda arr:np.asarray(arr,dtype=np.float64)/(float(scales[name])*math.sqrt(lengths[name])))
    context_data=geometry._load_features(root)
    lookup={str(item):i for i,item in enumerate(context_data["base_trial_id"].astype(str))}
    # Only current clean J is available at the control interface. V13's
    # layerwise causal *delta* embeds teacher intervention information and is
    # deliberately excluded from state-conditioned prediction.
    raw_context=np.asarray(context_data["endpoint__current_j_clean"],dtype=np.float32)
    train_ids=[x["base_trial_id"] for x in freeze["train"]]
    context_train=raw_context[[lookup[x] for x in train_ids]]
    mean=context_train.mean(axis=0)
    _,_,vt=np.linalg.svd(context_train-mean,full_matrices=False)
    context=((raw_context-mean)@vt[:4].T).astype(np.float32)
    train_context=context[[lookup[x] for x in train_ids],0]
    boundaries=np.quantile(train_context,[0.25,0.5,0.75])
    reliable["cluster"]=reliable.base_trial_id.map(lambda x:int(np.searchsorted(boundaries,context[lookup[x],0])))
    channel_all=_channel_weights(root,[int(x) for x in freeze["direction_indices"]])
    rows: list[dict[str,Any]]=[]
    gate=cfg["response_gate"]
    for k in cfg["action"]["candidate_dimensions"]:
        sub=reliable[reliable.requested_z.map(lambda z:np.count_nonzero(np.asarray(z)[int(k):])==0)].copy()
        sub["z"]=sub.requested_z.map(lambda z:np.asarray(z[:int(k)],dtype=np.float64))
        # Train deliberately excludes dense examples; validation tests new combinations.
        train=sub[(sub.role=="train")&(sub.design!="dense")]
        validation=sub[sub.role=="validation"]
        if len(train)<int(k)*5 or validation.empty:
            continue
        xtrain=np.stack(train.z); ytrain=np.stack([np.concatenate([row[f"y_{name}"] for name in lengths]) for _,row in train.iterrows()])
        channel=channel_all[:int(k)]
        for model in cfg["model"]["classes"]:
            cubic_indices=_select_cubic(xtrain,ytrain,channel,float(cfg["model"]["ridge"])) if model=="sparse_cubic" else None
            features=_features(xtrain,model,channel,cubic_indices)
            if model=="piecewise_quadratic":
                weights={int(c):_ridge(features[train.cluster.to_numpy()==c],ytrain[train.cluster.to_numpy()==c],float(cfg["model"]["ridge"])) for c in sorted(train.cluster.unique()) if sum(train.cluster.to_numpy()==c)>=features.shape[1]//2}
                if len(weights)<4: continue
            else:
                weights=_ridge(features,ytrain,float(cfg["model"]["ridge"]))
            for holdout, test in (("heldout_state_all",validation),
                                  ("heldout_state_single",validation[validation.design=="single"]),
                                  ("heldout_state_scale",validation[validation.design=="scale"]),
                                  ("heldout_state_pair",validation[validation.design=="pair"]),
                                  ("heldout_state_triple",validation[validation.design=="triple"]),
                                  ("heldout_state_dense",validation[validation.design=="dense"])):
                if test.empty: continue
                xtest=_features(np.stack(test.z),model,channel,cubic_indices)
                if model=="piecewise_quadratic":
                    pred=np.stack([xtest[i]@weights[int(c)] for i,c in enumerate(test.cluster.to_numpy())])
                else:
                    pred=xtest@weights
                ytest=np.stack([np.concatenate([row[f"y_{name}"] for name in lengths]) for _,row in test.iterrows()])
                met=_metrics(ytest,pred,lengths)
                passing=all(met[t]["direction"] is not None and met[t]["direction"]>=gate["direction_min"] and met[t]["relative_l2"] is not None and met[t]["relative_l2"]<=gate["relative_l2_max"] and met[t]["magnitude_ratio"] is not None and gate["magnitude_min"]<=met[t]["magnitude_ratio"]<=gate["magnitude_max"] for t in gate["required_targets"])
                rows.append({"k":int(k),"model":model,"input":"requested","holdout":holdout,
                             "train_count":len(train),"test_count":len(test),"gate_pass":passing,
                             **{f"{name}_{metric}":value for name,part in met.items() for metric,value in part.items()}})
            # These are genuine design holdouts: the named action pattern is
            # removed from model fitting, not merely assigned to another state.
            negative_single=validation[(validation.design=="single")&(validation.sign==-1)]
            positive_only=train[train.z.map(lambda z:bool(np.all(np.asarray(z)>=0)))]
            no_scale=train[train.design!="scale"]
            withheld_pairs={(0,1)} if int(k)==2 else {(1,3),(2,3)}
            def support(z:np.ndarray)->tuple[int,...]:
                return tuple(int(x) for x in np.flatnonzero(z))
            pair_test=validation[(validation.design=="pair")&validation.z.map(lambda z:support(z) in withheld_pairs)]
            pair_train=train[(train.design!="triple")&~train.z.map(lambda z:support(z) in withheld_pairs)]
            scenarios=(("strict_heldout_sign",positive_only,negative_single),
                       ("strict_heldout_scale",no_scale,validation[validation.design=="scale"]),
                       ("strict_heldout_pair",pair_train,pair_test))
            if model in ("linear","quadratic") and int(k) in (4,8,16,32):
                scenarios += tuple((f"diagnostic_heldout_family_{family}",
                                    train[train.family!=family],validation[validation.family==family])
                                   for family in sorted(train.family.unique()))
            for holdout,fit,test in scenarios:
                if fit.empty or test.empty or len(fit)<int(k)*5:
                    continue
                yf=np.stack([np.concatenate([row[f"y_{name}"] for name in lengths]) for _,row in fit.iterrows()])
                local_cubic=_select_cubic(np.stack(fit.z),yf,channel,float(cfg["model"]["ridge"])) if model=="sparse_cubic" else None
                xf=_features(np.stack(fit.z),model,channel,local_cubic)
                if model=="piecewise_quadratic":
                    local={int(c):_ridge(xf[fit.cluster.to_numpy()==c],yf[fit.cluster.to_numpy()==c],float(cfg["model"]["ridge"])) for c in sorted(fit.cluster.unique()) if sum(fit.cluster.to_numpy()==c)>=xf.shape[1]//2}
                    if not set(test.cluster.astype(int)).issubset(local):continue
                else:
                    local=_ridge(xf,yf,float(cfg["model"]["ridge"]))
                xt=_features(np.stack(test.z),model,channel,local_cubic)
                pred=np.stack([xt[i]@local[int(c)] for i,c in enumerate(test.cluster.to_numpy())]) if model=="piecewise_quadratic" else xt@local
                yt=np.stack([np.concatenate([row[f"y_{name}"] for name in lengths]) for _,row in test.iterrows()])
                met=_metrics(yt,pred,lengths)
                passing=all(met[t]["direction"] is not None and met[t]["direction"]>=gate["direction_min"] and met[t]["relative_l2"] is not None and met[t]["relative_l2"]<=gate["relative_l2_max"] and met[t]["magnitude_ratio"] is not None and gate["magnitude_min"]<=met[t]["magnitude_ratio"]<=gate["magnitude_max"] for t in gate["required_targets"])
                rows.append({"k":int(k),"model":model,"input":"requested","holdout":holdout,
                             "train_count":len(fit),"test_count":len(test),"gate_pass":passing,
                             **{f"{name}_{metric}":value for name,part in met.items() for metric,value in part.items()}})
    result=pd.DataFrame(rows)
    candidates=[]
    required=("heldout_state_all","strict_heldout_sign","strict_heldout_scale","strict_heldout_pair","heldout_state_dense")
    for (k,model), group in result.groupby(["k","model"]):
        if all(bool(group[group.holdout==holdout].gate_pass.any()) for holdout in required):
            candidates.append((int(k),str(model)))
    finalist=min(candidates,key=lambda x:(x[0],cfg["model"]["classes"].index(x[1]))) if candidates else None
    return result,{"candidate":finalist,"required_holdouts":list(required),"gate":gate,
                   "train_context_representation":"train-only 4D PCA of frozen current clean J; 4 bins from train quartiles of first component",
                   "context_input_is_not_raw_persistent_state":True,"target_block_lengths":lengths,
                   "model_rows":len(result)}


def _spectrum(matrix: np.ndarray) -> dict[str,Any]:
    if len(matrix)<2:return {"r90":None,"r95":None,"r99":None,"singular_values":[]}
    s=np.linalg.svd(matrix-matrix.mean(axis=0),compute_uv=False)
    energy=s*s; cumulative=np.cumsum(energy)/max(float(energy.sum()),1e-20)
    return {f"r{int(q*100)}":int(np.searchsorted(cumulative,q)+1) for q in (0.9,0.95,0.99)}|{"singular_values":s[:40].tolist()}


def _rank_and_primitives(frame:pd.DataFrame, models:dict[str,Any])->dict[str,Any]:
    singles=frame[(frame.design=="single")&(frame.reliability_status=="RELIABLE")&(frame.sign==1)]
    spectra=[]; primitive=[]
    for (role,state),group in singles.groupby(["role","base_trial_id"]):
        group=group.sort_values("coordinate_index")
        for k in (2,4,8,16,32):
            chosen=group[group.coordinate_index<k]
            if len(chosen)<2:continue
            matrix=np.stack([_vector(row,"stacked_normalized") for _,row in chosen.iterrows()])
            spectra.append({"role":role,"base_trial_id":state,"k":k,"sample_count":len(matrix),**_spectrum(matrix)})
    # A train-frozen greedy *response* dictionary, tested on unseen validation states.
    train=singles[singles.role=="train"]
    validation=singles[singles.role=="validation"]
    if not train.empty and not validation.empty:
        vectors=[]
        for index,group in train.groupby("coordinate_index"):
            vectors.append((int(index),np.mean(np.stack([_vector(row,"stacked_normalized") for _,row in group.iterrows()]),axis=0)))
        choices=[]; remaining=dict(vectors)
        val_train=np.stack([_vector(row,"stacked_normalized") for _,row in train.iterrows()])
        orthonormal=[]
        for count in range(1,min(32,len(remaining))+1):
            def gain(idx:int)->float:
                residual=remaining[idx].copy()
                for q in orthonormal:
                    residual-=q*np.dot(q,residual)
                norm=np.linalg.norm(residual)
                if norm<1e-12:return -math.inf
                q=residual/norm
                return float(np.sum((val_train@q)**2))
            candidate=max((x for x in remaining if x not in choices),key=gain)
            residual=remaining[candidate].copy()
            for q in orthonormal:
                residual-=q*np.dot(q,residual)
            orthonormal.append(residual/max(np.linalg.norm(residual),1e-20))
            choices.append(candidate)
            if count in (2,4,8,16,32):
                basis=np.stack([remaining[c] for c in choices],axis=1)
                val=np.stack([_vector(row,"stacked_normalized") for _,row in validation.iterrows()])
                coef=np.linalg.lstsq(basis,val.T,rcond=None)[0]
                coverage=float(np.linalg.norm(basis@coef)**2/max(np.linalg.norm(val)**2,1e-20))
                primitive.append({"K":count,"indices":choices.copy(),"heldout_linear_response_coverage":coverage,
                                  "limitation":"projection of one-step responses, not observed primitive sequence or finite nonlinear mixture"})
    return {"observed_spectra":spectra,"primitive_diagnostic":primitive,
            "rank_inflation_explained_by_validated_model":False if models["candidate"] is None else None}


def _reachability(root:Path,frame:pd.DataFrame,split:dict[str,Any])->tuple[pd.DataFrame,dict[str,Any]]:
    # Teacher raw states only define frozen labels here; no controller/model sees
    # these vectors as an input. Pair/triple writes are same-state combinations,
    # not multi-token sequential reachability.
    data=geometry._load_features(root)
    lookup={str(item):i for i,item in enumerate(data["base_trial_id"].astype(str))}
    selected=np.asarray(split["selected_j"],dtype=int)
    records=[]
    for state,group in frame[(frame.role=="validation")&(frame.reliability_status=="RELIABLE")].groupby("base_trial_id"):
        index=lookup[state]
        teacher=(data["endpoint__perturbed_j_h1"][index,selected].astype(np.float64)
                 -data["endpoint__clean_j_h1"][index,selected].astype(np.float64))
        teacher_norm=float(np.linalg.norm(teacher))
        for design in ("single","pair","triple","dense"):
            choices=group[group.design==design]
            if choices.empty:continue
            candidates=np.stack([_vector(row,"j") for _,row in choices.iterrows()])
            distances=np.linalg.norm(candidates-teacher[None],axis=1)
            best=int(np.argmin(distances));action=choices.iloc[best]
            records.append({"base_trial_id":state,"family":str(group.iloc[0].family),"design":design,
                            "action_count_simultaneous":int(np.count_nonzero(np.asarray(action.requested_z))),
                            "teacher_norm":teacher_norm,"nearest_relative_residual":float(distances[best]/max(teacher_norm,1e-20)),
                            "nearest_action_norm":float(np.linalg.norm(np.asarray(action.requested_z,dtype=float))),
                            "nearest_requested_z":action.requested_z,
                            "candidate_count":len(choices),"label_source":"frozen V13 captured h1 teacher delta; never model/controller input"})
    result=pd.DataFrame(records)
    medians={str(name):_summary(group.nearest_relative_residual.tolist()) for name,group in result.groupby("design")}
    return result,{"median_nearest_relative_residual_by_design":medians,
                   "one_step_is_empirical_singles":True,
                   "pair_triple_dense_are_same_state_combination_not_sequential_reachability":True,
                   "four_step_sequence_status":"NOT_MEASURED"}


def analyze(root:Path)->dict[str,Any]:
    base=verify(root); splits=_verify_splits(root)
    stage_path=root/"artifacts/nonlinear_finite_causal_action_v16_analysis.freeze.json"
    if not stage_path.exists():
        stage=stage_freeze(root,"analysis",[SOURCE,str(SPLITS),"artifacts/nonlinear_finite_causal_action_v16_bank_source_amendment_1.freeze.json"],
                           {"role":"train_fit_validation_test","response_gate":base["config"]["response_gate"],
                            "holdouts":["state","sign","scale","pair","dense","family"],
                            "analysis_must_not_open_independent_final":True})
    else:
        stage=json.loads(stage_path.read_text())
        if sha256_file(root/SOURCE)!=stage["input_hashes"][SOURCE]:raise RuntimeError("V16 analysis source changed")
    frame=_load(root)
    (root/OUT).mkdir(parents=True,exist_ok=True)
    bank_path=root/OUT/"finite_action_bank_v16.parquet"
    frame.drop(columns=["action_key"]).to_parquet(bank_path,index=False,compression="zstd")
    decomposed, decomposition=_decomposition(frame)
    decomposition_path=root/OUT/"nonlinear_decomposition_v16.parquet"
    decomposed.to_parquet(decomposition_path,index=False,compression="zstd")
    model_frame,model=_model_compare(root,frame)
    model_path=root/OUT/"nonlinear_model_comparison_v16.parquet"
    model_frame.to_parquet(model_path,index=False,compression="zstd")
    diagnostic=_rank_and_primitives(frame,model)
    reach,reach_summary=_reachability(root,frame,splits)
    reach_path=root/OUT/"nonlinear_reachability_v16.parquet"
    reach.to_parquet(reach_path,index=False,compression="zstd")
    summary={"protocol_version":base["protocol_version"],"base_freeze_digest":base["freeze_digest"],
             "split_freeze_digest":splits["freeze_digest"],"analysis_freeze_digest":stage["freeze_digest"],
             "bank":{"states_by_role":frame.groupby("role").base_trial_id.nunique().to_dict(),
                     "actions_by_role":frame.groupby("role").size().to_dict(),
                     "reliable_by_role":frame.groupby("role").reliability_status.apply(lambda x:int((x=="RELIABLE").sum())).to_dict(),
                     "reliable_fraction":float((frame.reliability_status=="RELIABLE").mean()),
                     "by_design":{str(name):{"count":int(len(group)),"reliable":int((group.reliability_status=="RELIABLE").sum())} for name,group in frame.groupby("design")},
                     "records":str(bank_path),"records_sha256":sha256_file(bank_path)},
             "decomposition":decomposition,"model":model,
             "nonlinear_rank_and_primitives":diagnostic,"reachability":reach_summary,
             "records":{"decomposition":str(decomposition_path),"decomposition_sha256":sha256_file(decomposition_path),
                        "models":str(model_path),"models_sha256":sha256_file(model_path),
                        "reachability":str(reach_path),"reachability_sha256":sha256_file(reach_path)}}
    write_json_atomic(root/OUT/"v16_analysis.json",summary)
    return summary


def main()->None:
    parser=argparse.ArgumentParser();parser.add_argument("--stage",choices=("analyze",),required=True)
    args=parser.parse_args()
    print(json.dumps({k:v for k,v in analyze(Path.cwd()).items() if k in ("bank","model")},default=str))


if __name__=="__main__":main()
