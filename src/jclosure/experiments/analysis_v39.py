"""State-wise trajectory algebra, bootstrap intervals and frozen-class adjudication."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from jclosure.experiments.trajectory_v39 import _saved
from jclosure.protocol_v39 import stage_freeze, verify, verify_stage
from jclosure.provenance import sha256_file, write_json_atomic


OUT = Path("results/v39/processed")
SOURCE = "src/jclosure/experiments/analysis_v39.py"
ROLES = ("development", "validation", "independent_final")


def norm(x):
    return np.linalg.norm(x, axis=-1)


def cosine(x, y):
    return np.sum(x*y, axis=-1) / np.maximum(norm(x)*norm(y), 1e-12)


def ci(values, seed, iterations):
    x = np.asarray(values, dtype=np.float64)
    x = x[np.isfinite(x)]
    if not len(x):
        return [None, None]
    rng = np.random.default_rng(seed)
    sampled = x[rng.integers(0, len(x), (iterations, len(x)))]
    return np.quantile(np.median(sampled, axis=1), [0.025, 0.975]).tolist()


def _aligned(*pairs):
    ids = [frame.state_id.tolist() for frame, _, _ in pairs]
    if not all(x == ids[0] for x in ids):
        raise RuntimeError("V39 state alignment drift")


def analyze(root: Path, key: str, role: str) -> dict:
    if role not in ROLES:
        raise ValueError(role)
    cfg = verify(root)["config"]
    single = _saved(root, key, role, "singles")
    pair = _saved(root, key, role, "pairs")
    predicted = _saved(root, key, role, "predict")
    full = _saved(root, key, role, "full")
    _aligned(single, pair, predicted, full)
    sf, sv, _ = single
    _, pv, _ = pair
    _, pred, _ = predicted
    _, fv, _ = full
    base = sv[:, 0]
    e100, e010, e001 = (sv[:, i]-base for i in (1, 2, 3))
    e110, e101, e011 = (pv[:, i]-base for i in range(3))
    actual = fv[:, 0]-base
    additive, second = pred[:, 0], pred[:, 1]
    direct_additive = e100+e010+e001
    direct_second = e110+e101+e011-e100-e010-e001
    if not np.allclose(additive, direct_additive, rtol=1e-5, atol=1e-4) or \
       not np.allclose(second, direct_second, rtol=1e-5, atol=1e-4):
        raise RuntimeError("V39 presealed prediction content mismatch")
    pair_terms = np.stack([e110-e100-e010, e101-e100-e001,
                           e011-e010-e001], axis=1)
    three = actual-direct_second
    exact_three = actual-e110-e101-e011+e100+e010+e001
    if not np.allclose(three, exact_three, rtol=1e-5, atol=1e-4):
        raise RuntimeError("V39 three-way identity mismatch")
    denom = np.maximum(norm(actual), cfg["effect_norm_floor"])
    add_err, second_err = norm(actual-additive)/denom, norm(actual-second)/denom
    add_cos, second_cos = cosine(actual, additive), cosine(actual, second)
    improvement = (add_err-second_err)/np.maximum(add_err, 1e-12)
    raw_norm = np.maximum(norm(actual), 1e-12)
    unit = actual/raw_norm[:, None]
    pair_proj = np.sum(pair_terms*unit[:, None, :], axis=-1) / raw_norm[:, None]
    three_proj = np.sum(three*unit, axis=-1) / raw_norm
    three_fraction = norm(three)/raw_norm
    three_cosine = cosine(three, actual)
    if not np.allclose(three_proj, three_fraction*three_cosine, rtol=1e-8, atol=1e-8):
        raise RuntimeError("V39 corrected p=f*cos identity failed")
    if np.any(np.abs(three_proj)>three_fraction+1e-8):
        raise RuntimeError("V39 |p| <= f identity failed")
    values = {"additive_relative_error": add_err,
              "additive_cosine": add_cos,
              "additive_magnitude_ratio": norm(additive)/denom,
              "second_relative_error": second_err,
              "second_cosine": second_cos,
              "second_magnitude_ratio": norm(second)/denom,
              "relative_error_improvement": improvement,
              "actual_effect_norm": norm(actual),
              "pair23_fraction": norm(pair_terms[:, 0])/denom,
              "pair24_fraction": norm(pair_terms[:, 1])/denom,
              "pair34_fraction": norm(pair_terms[:, 2])/denom,
              "pair23_projection": pair_proj[:, 0],
              "pair24_projection": pair_proj[:, 1],
              "pair34_projection": pair_proj[:, 2],
              "threeway_fraction": three_fraction,
              "threeway_projection": three_proj,
              "threeway_cosine": three_cosine,
              "q3_isolated_vs_given_q2_cosine": cosine(e010, e110-e100),
              "q4_isolated_vs_given_q2_cosine": cosine(e001, e101-e100),
              "q4_isolated_vs_given_q3_cosine": cosine(e001, e011-e010),
              "q4_isolated_vs_given_q23_cosine": cosine(e001, actual-e110),
              "q3_given_q2_magnitude_ratio": norm(e110-e100)/np.maximum(norm(e010),1.0),
              "q4_given_q23_magnitude_ratio": norm(actual-e110)/np.maximum(norm(e001),1.0),
              "isolated_q2_norm": norm(e100), "isolated_q3_norm": norm(e010),
              "isolated_q4_norm": norm(e001),
              "cumulative_q23_norm": norm(e110),
              "cumulative_q234_norm": norm(actual)}
    frame = sf[["state_id", "model", "role", "family", "vector_index"]].copy()
    for name, data in values.items():
        frame[name] = data
    frame["effect_below_floor"] = norm(actual) < cfg["effect_norm_floor"]
    # A probe is a repeated measure, never an independent bootstrap unit.
    width = actual.shape[1]//6
    if width*6 != actual.shape[1]:
        raise RuntimeError("V39 six-probe vector width mismatch")
    for probe in range(6):
        a = actual[:, probe*width:(probe+1)*width]
        frame[f"probe{probe+1}_additive_error"] = norm(a-additive[:,probe*width:(probe+1)*width])/np.maximum(norm(a),1.0)
        frame[f"probe{probe+1}_second_error"] = norm(a-second[:,probe*width:(probe+1)*width])/np.maximum(norm(a),1.0)
    high = pd.read_parquet(root/OUT/f"high_level_{key}_{role}_v39.parquet")
    hv = np.load(root/OUT/f"high_level_vectors_{key}_{role}_v39.npz")["vectors"].astype(np.float64)
    if high.state_id.tolist() != frame.state_id.tolist():
        raise RuntimeError("V39 high-level alignment drift")
    donor = hv[:, 2]
    joint = hv[:, 1]
    if not np.allclose(base, hv[:, 0], rtol=1e-5, atol=1e-4):
        raise RuntimeError("V39 R000 high-level baseline replay drift")
    econv = norm(donor-base)
    efull = norm(donor-joint)
    eq234 = norm(donor-fv[:,0])
    benefit = econv-efull
    frame["q234_fraction_of_full"] = np.where(benefit>1.0, (econv-eq234)/np.maximum(benefit,1e-12), np.nan)
    frame["q234_direction_cosine_to_full"] = cosine(actual, joint-base)
    frame["additive_donor_projection"] = np.sum(additive*(donor-base),axis=1)/np.maximum(norm(donor-base),1.0)
    frame["second_donor_projection"] = np.sum(second*(donor-base),axis=1)/np.maximum(norm(donor-base),1.0)
    frame["actual_donor_projection"] = np.sum(actual*(donor-base),axis=1)/np.maximum(norm(donor-base),1.0)
    table = root/OUT/f"trajectory_analysis_{key}_{role}_v39.parquet"
    frame.to_parquet(table,index=False,compression="zstd")
    iterations = cfg["bootstrap"]["iterations"]
    seed = cfg["bootstrap"]["seed"] + (0 if key=="Q" else 10000) + ROLES.index(role)
    fields = ["additive_relative_error","additive_cosine","second_relative_error",
              "second_cosine","relative_error_improvement","threeway_fraction",
              "threeway_projection","threeway_cosine",
              "q234_fraction_of_full","q234_direction_cosine_to_full"]
    metrics = {name: {"median": float(np.nanmedian(frame[name])),
                      "state_bootstrap_median_95ci": ci(frame[name],seed+i,iterations)}
               for i,name in enumerate(fields)}
    metrics["threeway_sign_stability"] = float(max(np.mean(three_proj>0),np.mean(three_proj<0)))
    metrics["threeway_dominant_sign"] = "positive" if np.mean(three_proj>0)>=np.mean(three_proj<0) else "negative"
    metrics["effect_below_floor_count"] = int(frame.effect_below_floor.sum())
    families = {}
    for family, g in frame.groupby("family"):
        families[family] = {name: float(np.nanmedian(g[name])) for name in fields}
        families[family]["threeway_sign_stability"] = float(max(np.mean(g.threeway_projection>0),
                                                                 np.mean(g.threeway_projection<0)))
    qgate = cfg["q234_gate"]
    hgate = cfg["falcon_higher_order_gate"]
    def q234_pass(data):
        return (data["q234_fraction_of_full"] >= qgate["median_fraction_min"] and
                data["q234_direction_cosine_to_full"] >= qgate["median_cosine_min"])
    def higher_pass(data):
        return (data["second_relative_error"] > hgate["second_order_relative_error_min_exclusive"] and
                data["threeway_fraction"] >= hgate["norm_fraction_min"] and
                data["threeway_sign_stability"] >= hgate["sign_stability_min"])
    med = {name: obj["median"] for name,obj in metrics.items() if isinstance(obj,dict) and "median" in obj}
    med["threeway_sign_stability"] = metrics["threeway_sign_stability"]
    family_counts = {"q234": sum(q234_pass(x) for x in families.values()),
                     "higher_order": sum(higher_pass(x) for x in families.values())}
    passed = {"q234": q234_pass(med) and
              family_counts["q234"] >= qgate["families_required"],
              "higher_order": higher_pass(med) and
              family_counts["higher_order"] >= hgate["families_required"]}
    summary = {"model":key,"role":role,"states":len(frame),"metrics":metrics,
               "family_metrics":families,"family_pass_counts":family_counts,
               "gate_passes":passed,"natural_Q234_no_output_copy":True,
               "corrected_projection_definition":"dot(I234,E111)/norm(E111)^2",
               "projection_identity_max_error":float(np.max(np.abs(three_proj-three_fraction*three_cosine))),
               "table_sha256":sha256_file(table),
               "prediction_sealed_before_R111":True}
    path=root/OUT/f"trajectory_analysis_{key}_{role}_v39.json"
    write_json_atomic(path,summary)
    seal=stage_freeze(root,f"trajectory_analysis_{key}_{role}",
                      [SOURCE,str(path.relative_to(root)),str(table.relative_to(root)),
                       f"artifacts/interaction_genesis_v39_trajectory_predict_{key}_{role}.freeze.json",
                       f"artifacts/interaction_genesis_v39_trajectory_full_{key}_{role}.freeze.json"],
                      {"model":key,"role":role,"summary_sha256":sha256_file(path),
                       "prediction_sealed_before_R111":True})
    return {"freeze_digest":seal["freeze_digest"],**summary}


def adjudicate(root:Path,role:str)->dict:
    if role not in ("development","validation"):
        raise ValueError(role)
    verify(root)
    summaries={}
    for key in ("Q","F"):
        path = root/OUT/f"trajectory_analysis_{key}_{role}_v39.json"
        if path.exists():
            verify_stage(root,f"trajectory_analysis_{key}_{role}")
            summaries[key]=json.loads(path.read_text())
    if "F" not in summaries:
        raise RuntimeError("Falcon primary factorial analysis missing")
    result={"role":role,"model_gates":{key:s["gate_passes"] for key,s in summaries.items()},
            "falcon_q234_pass":summaries["F"]["gate_passes"]["q234"],
            "falcon_higher_order_pass":summaries["F"]["gate_passes"]["higher_order"],
            "qwen_comparative_available":"Q" in summaries,
            "independent_final_opening_eligible":False,
            "mediator_not_yet_tested":True}
    path=root/OUT/f"factorial_gate_{role}_v39.json"
    write_json_atomic(path,result)
    seal=stage_freeze(root,f"factorial_gate_{role}",
                      [SOURCE,str(path.relative_to(root)),
                       *[f"artifacts/interaction_genesis_v39_trajectory_analysis_{key}_{role}.freeze.json"
                         for key in summaries]],
                      {"role":role,"falcon_q234_pass":result["falcon_q234_pass"],
                       "falcon_higher_order_pass":result["falcon_higher_order_pass"],
                       "independent_final_opening_eligible":False,
                       "summary_sha256":sha256_file(path)})
    return {"freeze_digest":seal["freeze_digest"],**result}


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("action",choices=("analyze","adjudicate"))
    p.add_argument("model",choices=("Q","F","both"));p.add_argument("role",choices=ROLES)
    a=p.parse_args()
    result=adjudicate(Path.cwd(),a.role) if a.action=="adjudicate" else analyze(Path.cwd(),a.model,a.role)
    print(json.dumps(result,indent=2))
