"""Frozen V35 quartile, curve, pair and serial causal hierarchy decisions."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from jclosure.protocol_v35 import stage_freeze, verify, verify_stage
from jclosure.provenance import sha256_file, write_json_atomic

OUT = Path("results/v35/processed")
SOURCE = "src/jclosure/experiments/depth_analysis_v35.py"
QUARTILES = ("Q1", "Q2", "Q3", "Q4")


def number(frame, condition, column, family=None):
    part = frame[(frame.condition == condition) & (frame.status == "VALID")]
    if family is not None:
        part = part[part.family == family]
    return float(part[column].median()) if len(part) else None


def stage_gate(frame, condition, threshold):
    cols = ("removed_fraction", "restored_fraction", "reverse_correction_cosine")
    limits = (("removed_fraction", threshold["median_removed_min"]),
              ("restored_fraction", threshold["median_restored_min"]),
              ("reverse_correction_cosine", threshold["median_cosine_min"]))
    med = {col: number(frame, condition, col) for col in cols}
    family = {}
    for name in sorted(frame.family.unique()):
        values = {col: number(frame, condition, col, name) for col in cols}
        family[name] = {**values, "pass": all(values[col] is not None and values[col] >= limit
                                                for col, limit in limits)}
    passed = (all(med[col] is not None and med[col] >= limit for col, limit in limits)
              and sum(v["pass"] for v in family.values()) >= threshold["families_required"])
    return {**med, "family": family, "families_passing": sum(v["pass"] for v in family.values()),
            "pass": bool(passed)}


def _late(frame, cfg, family=None):
    t = cfg["late_consolidation_gate"]
    vals = {name: {side: number(frame, name, col, family)
                   for side, col in (("REM", "removed_fraction"), ("REST", "restored_fraction"))}
            for name in ("Q1_Q2", "Q3_Q4")}
    if any(value is None for pair in vals.values() for value in pair.values()):
        return False, vals
    passed = all(vals["Q3_Q4"][side] >= t[f"late_half_{'removed' if side == 'REM' else 'restored'}_min"] and
                 vals["Q1_Q2"][side] <= t[f"early_half_{'removed' if side == 'REM' else 'restored'}_max"] and
                 vals["Q3_Q4"][side] - vals["Q1_Q2"][side] >= t["faster_difference_min"]
                 for side in ("REM", "REST"))
    return bool(passed), vals


def _progressive(frame, cfg, family=None):
    t = cfg["progressive_gate"]
    conditions = cfg["early_cumulative"]
    curves = {side: [number(frame, condition, col, family) for condition in conditions]
              for side, col in (("REM", "removed_fraction"), ("REST", "restored_fraction"))}
    singles = [number(frame, q, col, family) for q in QUARTILES
               for col in ("removed_fraction", "restored_fraction")]
    if any(value is None for curve in curves.values() for value in curve) or any(x is None for x in singles):
        return False, curves
    passed = all(all(curve[i+1] - curve[i] >= t["each_increment_min"] and
                     curve[i+1] >= curve[i] - t["monotonic_tolerance"] for i in range(3))
                 for curve in curves.values()) and max(singles) < t["single_quartile_max"]
    return bool(passed), curves


def _pair(frame, cfg, pair, kind, family=None):
    a, b = pair.split("_")[:2]
    threshold = cfg[f"{kind}_gate"]
    values = {name: {side: number(frame, name, col, family)
                     for side, col in (("REM", "removed_fraction"), ("REST", "restored_fraction"))}
              for name in (a, b, pair)}
    if any(x is None for value in values.values() for x in value.values()):
        return False, values
    if kind == "complementary":
        passed = all(values[pair][side] - max(values[a][side], values[b][side]) >=
                     threshold["pair_excess_over_best_single_min"] and
                     max(values[a][side], values[b][side]) < threshold["each_single_max"]
                     for side in ("REM", "REST"))
    else:
        passed = all(min(values[a][side], values[b][side]) >= threshold["each_single_min"] and
                     values[pair][side] - max(values[a][side], values[b][side]) <=
                     threshold["pair_increment_over_best_single_max"]
                     for side in ("REM", "REST"))
    return bool(passed), values


def _serial(frame, serial, cfg, early, late, family=None):
    t = cfg["serial_gate"]
    rows = serial[(serial.early == early) & (serial.late == late) & serial.benefit_positive]
    if family is not None:
        rows = rows[rows.family == family]
    pair = f"{early}_{late}"
    med = {"conditional_late_gain": float(rows.conditional_late_gain.median()) if len(rows) else None,
           "interaction_projection": float(rows.interaction_projection_to_correction.median()) if len(rows) else None}
    a, b, p = (number(frame, name, "removed_fraction", family) for name in (early, late, pair))
    med["removal_interaction"] = p - a - b if None not in (a, b, p) else None
    passed = (med["conditional_late_gain"] is not None and
              med["conditional_late_gain"] >= t["conditional_late_gain_min"] and
              med["removal_interaction"] >= t["removal_corrob_min"] and
              med["interaction_projection"] > 0)
    return bool(passed), med


def analyze_model(subset, leaveout, serial, cfg):
    gate = cfg["quartile_gate"]
    conditions = list(cfg["subset_conditions"])
    result = {"conditions": {condition: stage_gate(subset, condition, gate) for condition in conditions}}
    late_pass, late_values = _late(subset, cfg)
    progressive_pass, progressive_values = _progressive(subset, cfg)
    family = sorted(subset.family.unique())
    late_family = {name: _late(subset, cfg, name)[0] for name in family}
    progressive_family = {name: _progressive(subset, cfg, name)[0] for name in family}
    result["late_consolidation"] = {"pass": late_pass and sum(late_family.values()) >= cfg["late_consolidation_gate"]["families_required"],
                                   "values": late_values, "family_pass": late_family}
    result["progressive_accumulation"] = {"pass": progressive_pass and sum(progressive_family.values()) >= cfg["progressive_gate"]["families_required"],
                                          "early_curve": progressive_values, "family_pass": progressive_family}
    pairs = {}
    for pair in cfg["pair_conditions"]:
        pairs[pair] = {}
        for kind in ("complementary", "redundancy"):
            passed, values = _pair(subset, cfg, pair, kind)
            fam = {name: _pair(subset, cfg, pair, kind, name)[0] for name in family}
            pairs[pair][kind] = {"pass": bool(passed and sum(fam.values()) >= cfg[f"{kind}_gate"]["families_required"]),
                                 "values": values, "family_pass": fam}
    result["pairs"] = pairs
    serial_pairs = {}
    for early, late in cfg["serial_pairs"]:
        passed, med = _serial(subset, serial, cfg, early, late)
        fam = {name: _serial(subset, serial, cfg, early, late, name)[0] for name in family}
        serial_pairs[f"{early}_{late}"] = {"pass": bool(passed and sum(fam.values()) >= cfg["serial_gate"]["families_required"]),
                                           "values": med, "family_pass": fam}
    result["serial_pairs"] = serial_pairs
    result["leaveout"] = {name: {"median_loss_fraction": float(part.loss_fraction.median()),
                                  "family": {family_name: float(family_part.loss_fraction.median())
                                             for family_name, family_part in part.groupby("family")}}
                          for name, part in leaveout.groupby("condition")}
    result["single_quartile_passes"] = [q for q in QUARTILES if result["conditions"][q]["pass"]]
    result["strong_pair_passes"] = [p for p in cfg["pair_conditions"] if result["conditions"][p]["pass"]]
    result["complementary_pair_passes"] = [p for p in pairs if pairs[p]["complementary"]["pass"]]
    result["redundant_pair_passes"] = [p for p in pairs if pairs[p]["redundancy"]["pass"]]
    result["serial_pair_passes"] = [p for p in serial_pairs if serial_pairs[p]["pass"]]
    result["refinement_candidates"] = result["single_quartile_passes"] + [p for p in ("Q1_Q2", "Q2_Q3", "Q3_Q4") if p in result["strong_pair_passes"]]
    result["family_depth_profile"] = {name: {condition: {col: number(subset, condition, col, name)
                                                          for col in ("removed_fraction", "restored_fraction", "reverse_correction_cosine")}
                                            for condition in conditions} for name in family}
    result["token_category_depth_profile"] = {name: {condition: {col: float(part[part.condition == condition][col].median())
                                                                  for col in ("removed_fraction", "restored_fraction")}
                                                     for condition in QUARTILES}
                                              for name, part in subset.groupby("donor_token_category")}
    return result


def similarity(q, f, cfg):
    def vector(profile, col, names):
        return np.asarray([profile["conditions"][name][col] for name in names], dtype=float)
    def corr(a, b):
        return float(np.corrcoef(a, b)[0, 1]) if np.std(a) > 1e-12 and np.std(b) > 1e-12 else None
    out = {}
    for side, col in (("REM", "removed_fraction"), ("REST", "restored_fraction")):
        a, b = vector(q, col, QUARTILES), vector(f, col, QUARTILES)
        rank_a, rank_b = pd.Series(a).rank().to_numpy(), pd.Series(b).rank().to_numpy()
        out[side] = {"Q": a.tolist(), "F": b.tolist(), "spearman": corr(rank_a, rank_b),
                     "pearson": corr(a, b),
                     "cosine": float(np.dot(a,b) / max(float(np.linalg.norm(a)*np.linalg.norm(b)),1e-12)),
                     "early_curve_l2": float(np.linalg.norm(vector(q,col,cfg["early_cumulative"])-vector(f,col,cfg["early_cumulative"]))),
                     "late_curve_l2": float(np.linalg.norm(vector(q,col,cfg["late_cumulative"])-vector(f,col,cfg["late_cumulative"])))}
    gate = cfg["profile_similarity_gate"]
    out["descriptive_profile_similarity_pass"] = all(out[side][metric] is not None and out[side][metric] >= gate[f"{metric}_min"]
                                                      for side in ("REM", "REST") for metric in ("spearman", "pearson", "cosine"))
    return out


def pair_interactions(root, key, role, cfg, subset_frame):
    """Exact four-branch output interaction, not a sum of mediation fractions."""
    depth = np.load(root / OUT / f"depth_vectors_{key}_{role}_v35.npz")
    factual = np.load(root / OUT / f"factorial_vectors_{key}_{role}_v35.npz")["vectors"].astype(np.float64)
    factorial = pd.read_parquet(root / OUT / f"factorial_{key}_{role}_v35.parquet")
    idx = {row.state_id: int(row.vector_index) for row in factorial.itertuples()}
    names = list(depth["subset_conditions"])
    values = depth["subset"].astype(np.float64)
    state_ids = list(depth["state_ids"])
    family = {row.state_id: row.family for row in subset_frame[subset_frame.condition == "FULL"].itertuples()}
    out = {}
    for pair in cfg["pair_conditions"]:
        a, b = pair.split("_")
        rows = []
        for k, sid in enumerate(state_ids):
            facts = factual[idx[sid]]
            yconv, yjoint = facts[2], facts[3]
            correction = yjoint - yconv
            interaction = (values[k, names.index(pair), 1] - values[k, names.index(a), 1]
                           - values[k, names.index(b), 1] + yconv)
            norm = max(float(np.linalg.norm(correction)), 1e-12)
            rows.append({"family": family[sid],
                         "projection": float(np.dot(interaction, correction)) / (norm * norm),
                         "relative_norm": float(np.linalg.norm(interaction)) / norm})
        frame = pd.DataFrame(rows)
        out[pair] = {"interaction_definition": "Y_pair - Y_i - Y_j + Y_conv",
                     "median_projection_to_full_correction": float(frame.projection.median()),
                     "median_relative_norm": float(frame.relative_norm.median()),
                     "family_projection": {name: float(part.projection.median())
                                           for name, part in frame.groupby("family")}}
    return out


def run(root: Path, role: str):
    if role not in ("development", "validation"):
        raise ValueError(role)
    verify_stage(root, f"full_gate_{role}")
    cfg = verify(root)["config"]
    result = {"role": role, "models": {}, "thresholds_retuned": False,
              "historical_V34_exploratory_quartiles_used_as_formal": False}
    inputs = [SOURCE, f"artifacts/hierarchical_read_v35_full_gate_{role}.freeze.json"]
    for key in ("Q", "F"):
        verify_stage(root, f"depth_{key}_{role}")
        inputs.append(f"artifacts/hierarchical_read_v35_depth_{key}_{role}.freeze.json")
        subset = pd.read_parquet(root / OUT / f"depth_subset_{key}_{role}_v35.parquet")
        leaveout = pd.read_parquet(root / OUT / f"depth_leaveout_{key}_{role}_v35.parquet")
        serial = pd.read_parquet(root / OUT / f"depth_serial_{key}_{role}_v35.parquet")
        if subset.state_id.nunique() != (80 if role == "development" else 40):
            raise RuntimeError(f"V35 incomplete depth frame {key}:{role}")
        result["models"][key] = analyze_model(subset, leaveout, serial, cfg)
        result["models"][key]["pair_interactions"] = pair_interactions(root, key, role, cfg, subset)
    q, f = result["models"]["Q"], result["models"]["F"]
    result["cross_model_profile"] = similarity(q, f, cfg)
    result["shared_qualitative_this_role"] = {
        "late": q["late_consolidation"]["pass"] and f["late_consolidation"]["pass"],
        "progressive": q["progressive_accumulation"]["pass"] and f["progressive_accumulation"]["pass"],
        "complementary_pairs": sorted(set(q["complementary_pair_passes"]) & set(f["complementary_pair_passes"])),
        "redundant_pairs": sorted(set(q["redundant_pair_passes"]) & set(f["redundant_pair_passes"])),
        "serial_pairs": sorted(set(q["serial_pair_passes"]) & set(f["serial_pair_passes"])),
        "strong_quartiles": sorted(set(q["single_quartile_passes"]) & set(f["single_quartile_passes"])),
        "strong_pairs": sorted(set(q["strong_pair_passes"]) & set(f["strong_pair_passes"])),
    }
    if role == "validation":
        dev = json.loads((root / OUT / "depth_analysis_development_v35.json").read_text())
        d, v = dev["shared_qualitative_this_role"], result["shared_qualitative_this_role"]
        result["shared_qualitative_both_roles"] = {
            "late": d["late"] and v["late"], "progressive": d["progressive"] and v["progressive"],
            **{key: sorted(set(d[key]) & set(v[key])) for key in
               ("complementary_pairs", "redundant_pairs", "serial_pairs", "strong_quartiles", "strong_pairs")}}
        inputs.append("artifacts/hierarchical_read_v35_depth_analysis_development.freeze.json")
    path = root / OUT / f"depth_analysis_{role}_v35.json"
    write_json_atomic(path, result)
    inputs.append(str(path.relative_to(root)))
    seal = stage_freeze(root, f"depth_analysis_{role}", inputs,
                        {"role": role, "summary_sha256": sha256_file(path),
                         "shared_qualitative": result["shared_qualitative_this_role"]})
    return {"freeze_digest": seal["freeze_digest"], "role": role,
            "shared": result["shared_qualitative_this_role"],
            "profile": result["cross_model_profile"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("role", choices=("development", "validation"))
    args = parser.parse_args()
    print(json.dumps(run(Path.cwd(), args.role), indent=2))
