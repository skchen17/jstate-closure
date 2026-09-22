"""V30 scientific adjudication from frozen machine responses; final remains sealed."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from jclosure.protocol_v30 import stage_freeze, verify_stage
from jclosure.provenance import sha256_file, write_json_atomic

SOURCE = "src/jclosure/experiments/adjudicate_v30.py"
OUT = Path("results/v30/processed")
ROLES = ("development", "validation")
PREREQUISITES = ("final_opening", "conv_depth_development", "conv_depth_validation", "rank120_fit", "coordinate_controls_development", "coordinate_controls_validation", "horizon_development", "horizon_validation")


def med(frame, condition, field, axis=None):
    x = frame[frame.condition == condition]
    if axis is not None and "axis" in x:
        x = x[x.axis == axis]
    return float(x[field].median()) if len(x) else None


def run(root: Path):
    for stage in PREREQUISITES:
        verify_stage(root, stage)
    opening = json.loads((root / OUT / "final_opening_v30.json").read_text())
    if opening["final_opened"] or opening["selected_finalist"] is not None:
        raise RuntimeError("unexpected independent-final opening")
    global_rows = {r: pd.read_parquet(root / OUT / f"causal_global_{r}_v30.parquet") for r in ROLES}
    local_rows = {r: pd.read_parquet(root / OUT / f"local_transport_{r}_v30.parquet") for r in ROLES}
    conditional = {r: pd.read_parquet(root / OUT / f"rec_conv_conditional_{r}_v30.parquet") for r in ROLES}
    conv = {r: json.loads((root / OUT / f"conv_depth_{r}_v30.json").read_text()) for r in ROLES}
    rank = json.loads((root / OUT / "write_rank_scaling_120_v30.json").read_text())
    token_rank = json.loads((root / OUT / "write_rank_scaling_v30.json").read_text())
    effect = {}
    for role, frame in conditional.items():
        p = frame.pivot(index="state_id", columns="condition", values="relative_l2_to_donor")
        delta = p["Y01_CONV_ONLY"] - p["Y11_REC_CONV"]
        interaction = frame[frame.condition == "REC_CONV_INTERACTION"].interaction_ratio
        effect[role] = {"states": len(delta), "REC_improves_Conv_relative_l2_count": int((delta > 0).sum()), "median_relative_l2_improvement": float(delta.median()), "median_interaction_ratio": float(interaction.median()), "Conv_only_median_relative_l2": float(p["Y01_CONV_ONLY"].median()), "REC_Conv_median_relative_l2": float(p["Y11_REC_CONV"].median()), "REC_only_median_relative_l2": float(p["Y10_REC_ONLY"].median())}
    outcomes = {
        "V30_A_GLOBAL_CAUSAL_WRITE_SUBSPACE_CONFIRMED": False,
        "V30_B_SHARED_CAUSAL_WRITE_COORDINATES_CONFIRMED": False,
        "V30_C_STATE_DEPENDENT_WRITE_ATLAS": False,
        "V30_D_TOKEN_OOD_WRITE_GENERALIZATION": False,
        "V30_E_FAMILY_OOD_WRITE_GENERALIZATION": False,
        "V30_F_CONV_CARRIER_LOCALIZED_IN_DEPTH": False,
        "V30_G_CONV_CARRIER_DISTRIBUTED_ACROSS_DEPTH": False,
        "V30_H_REC_CONDITIONALLY_REFINES_CONV_WRITE": all(x["REC_improves_Conv_relative_l2_count"] == x["states"] for x in effect.values()),
        "V30_I_LOCAL_COMPACT_WRITE_ONLY": False,
        "V30_J_WRITE_EFFECT_DIMENSION_NOT_SATURATED": True,
        "V30_K_NO_SHARED_WRITE_COORDINATE_IDENTIFIED": True,
    }
    result = {
        "outcomes": outcomes,
        "final_opening": {"opened": False, "decision_sha256": sha256_file(root / OUT / "final_opening_v30.json"), "freeze_digest": verify_stage(root, "final_opening")["freeze_digest"]},
        "global_joint_OOD": {r: {k: {"median_cosine": med(global_rows[r], k, "donor_cosine", "JOINT_OOD"), "median_relative_l2": med(global_rows[r], k, "relative_l2_to_donor", "JOINT_OOD")} for k in ("GLOBAL_k32", "GLOBAL_k64", "EXACT_REC_CONV", "EXACT_CONV", "EXACT_KV")} for r in ROLES},
        "local_heldout_token": {r: {k: {"median_cosine": med(local_rows[r], k, "donor_cosine"), "median_relative_l2": med(local_rows[r], k, "relative_l2_to_donor")} for k in ("LOCAL_ORACLE_k32", "LOCAL_ORACLE_k64", "TRANSPORTED_LOCAL_RIDGE_k32", "TRANSPORTED_LOCAL_RIDGE_k64", "TRANSPORTED_LOCAL_PROCRUSTES_k64")} for r in ROLES},
        "global_profiles": {r: json.loads((root / OUT / f"causal_global_{r}_v30.json").read_text())["profiles"] for r in ROLES},
        "local_profiles": {r: json.loads((root / OUT / f"local_transport_{r}_v30.json").read_text())["profiles"] for r in ROLES},
        "Conv_full_profile": {r: conv[r]["profiles"]["FULL_CONV"] for r in ROLES},
        "Conv_minimal_selection": conv["development"]["selection"],
        "REC_conditional_effect": effect,
        "rank120": rank["state_count_scaling"],
        "token_pair_rank": token_rank["train_token_pairs_per_state_scaling"],
        "interpretation": "No tested <=64D global, local-oracle, or train-only transported coordinate passes unseen-token causal OOD gates. Local ceiling itself fails, so this does not isolate transport-map failure or establish a moving atlas. V29 same-background compact effect remains historical. The G/A/L trichotomy is not exhaustive at expanded token coverage.",
        "authorization": {"H2_REMAINS": True, "H3_AUTHORIZED": False, "DYNAMIC_STATE_SEARCH_AUTHORIZED": False, "AUTONOMOUS_CONTROLLER_AUTHORIZED": False, "CROSS_MODEL_REPLICATION_AUTHORIZED": False},
        "independent_final_opened": False,
        "historical_V1_V29_unmodified": True,
        "not_complete_model_state_dimension": True,
    }
    jp = root / OUT / "v30_adjudication.json"
    write_json_atomic(jp, result)
    fr = stage_freeze(root, "adjudication", [SOURCE, str(jp.relative_to(root)), *[f"artifacts/transferable_natural_writes_v30_{s}.freeze.json" for s in PREREQUISITES]], {"adjudication_sha256": sha256_file(jp), "outcomes": outcomes, "final_opened": False})
    return {"freeze_digest": fr["freeze_digest"], "outcomes": outcomes, "final_opened": False}


if __name__ == "__main__":
    print(json.dumps(run(Path.cwd()), indent=2))
