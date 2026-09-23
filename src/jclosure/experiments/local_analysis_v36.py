"""State-unit prospective development/validation gates for V36 local law."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from jclosure.provenance import sha256_file, write_json_atomic
from jclosure.protocol_v36 import stage_freeze, verify, verify_stage

OUT = Path("results/v36/processed")
SOURCE = "src/jclosure/experiments/local_analysis_v36.py"


def _bootstrap_median(values, seed, iterations=2000):
    x = np.asarray(values, np.float64)
    x = x[np.isfinite(x)]
    if not len(x): return [None, None]
    rng = np.random.default_rng(seed)
    med = np.median(x[rng.integers(0, len(x), size=(iterations, len(x)))], axis=1)
    return [float(np.quantile(med, .025)), float(np.quantile(med, .975))]


def _metric(part, column):
    x = pd.to_numeric(part[column], errors="coerce").dropna()
    return float(x.median()) if len(x) else None


def _gate(part, cfg):
    t = cfg["operator_gate"]
    c = _metric(part, "pred_cosine")
    r = _metric(part, "magnitude_rank")
    m = _metric(part, "matched_fraction")
    l = _metric(part, "local_fraction")
    pred = bool(c is not None and c >= t["local_predicted_observed_cosine_min"] and
                r is not None and r >= t["state_magnitude_rank_correlation_min"] and
                m is not None and m >= t["matched_superiority_fraction_min"])
    causal = bool(l is not None and l >= t["local_downstream_fraction_min"])
    return {"n_states": len(part), "median_predicted_observed_cosine": c,
            "median_magnitude_rank_spearman": r, "median_matched_superiority_fraction": m,
            "median_local_downstream_fraction": l,
            "prediction_pass": pred, "local_causal_pass": causal}


def _model(root, key, role, cfg):
    local = pd.read_parquet(root / OUT / f"local_{key}_{role}_v36.parquet")
    downstream = pd.read_parquet(root / OUT / f"local_state_{key}_{role}_v36.parquet")
    state = local.groupby(["state_id", "family"], as_index=False).agg(
        pred_cosine=("predicted_observed_cosine_B", "median"),
        magnitude_rank=("predicted_magnitude_rank_spearman", "median"),
        matched_fraction=("observed_matched_superior", "mean"),
        fixed_interaction=("mixer_interaction_norm", "median"),
        natural_interaction=("natural_mixer_interaction_norm", "median"),
        old_term=("old_state_term_difference_norm", "median"),
        dependent_update=("state_dependent_update_difference_norm", "median"),
        normalization_gain=("normalization_gain", "median"),
    )
    # Four layer fractions are a repeated measurement on one state, not four units.
    fractions = downstream.groupby("state_id", as_index=False).agg(
        local_fraction=("local_fraction", "median"),
        interaction_fraction=("interaction_fraction", "median"),
        natural_benefit_absolute=("natural_benefit_absolute", "median"),
        denominator_eligible=("denominator_eligible", "all"))
    state = state.merge(fractions, on="state_id", validate="one_to_one")
    state["fixed_to_natural_interaction_ratio"] = state.fixed_interaction / state.natural_interaction.clip(lower=1e-6)
    path = root / OUT / f"local_state_analysis_{key}_{role}_v36.parquet"
    state.to_parquet(path, index=False, compression="zstd")
    whole = _gate(state, cfg)
    families = {name: _gate(part, cfg) for name, part in state.groupby("family")}
    required = cfg["operator_gate"]["families_required"]
    prediction = whole["prediction_pass"] and sum(x["prediction_pass"] for x in families.values()) >= required
    causal = whole["local_causal_pass"] and sum(x["local_causal_pass"] for x in families.values()) >= required
    seed = cfg["seed"] + (0 if key == "Q" else 1) + (0 if role == "development" else 100)
    return {"state_count": len(state), "probe_layer_rows": len(local), "whole": whole,
            "family": families, "prediction_gate_pass": bool(prediction),
            "local_downstream_gate_pass": bool(causal),
            "same_class_operator_and_causal_pass": bool(prediction and causal),
            "fixed_to_natural_interaction_median": _metric(state, "fixed_to_natural_interaction_ratio"),
            "old_term_norm_median": _metric(state, "old_term"),
            "dependent_update_norm_median": _metric(state, "dependent_update"),
            "normalization_gain_median": _metric(state, "normalization_gain"),
            "bootstrap_ci_state_median_local_fraction": _bootstrap_median(state.local_fraction, seed),
            "bootstrap_ci_state_median_matched_fraction": _bootstrap_median(state.matched_fraction, seed + 1),
            "state_table_sha256": sha256_file(path)}


def run(root: Path, role: str):
    if role not in ("development", "validation"):
        raise ValueError(role)
    if role == "validation":
        verify_stage(root, "local_analysis_development")
    cfg = verify(root)["config"]
    models = {}
    for key in cfg["models"]:
        verify_stage(root, f"local_{key}_{role}")
        models[key] = _model(root, key, role, cfg)
    result = {"role": role, "models": models,
              "both_models_same_class_operator_and_causal_pass": all(x["same_class_operator_and_causal_pass"] for x in models.values()),
              "local_candidate_class": "READ_OPERATOR_MATCHING",
              "thresholds_retuned": False, "state_independence_unit": True,
              "six_probes_not_independent": True, "final_responses_seen": False}
    path = root / OUT / f"local_analysis_{role}_v36.json"
    write_json_atomic(path, result)
    seal = stage_freeze(root, f"local_analysis_{role}", [SOURCE, str(path.relative_to(root)),
                        *(f"artifacts/computational_origin_v36_local_{key}_{role}.freeze.json" for key in cfg["models"]),
                        *(f"results/v36/processed/local_state_analysis_{key}_{role}_v36.parquet" for key in cfg["models"])],
                        {"role": role, "summary_sha256": sha256_file(path),
                         "both_pass": result["both_models_same_class_operator_and_causal_pass"],
                         "final_responses_seen": False})
    return {"freeze_digest": seal["freeze_digest"], **result}


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("role", choices=("development", "validation")); a = p.parse_args()
    print(json.dumps(run(Path.cwd(), a.role), indent=2))
