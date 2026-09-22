"""Frozen-gate V32 statistics, restricted rotation test and adjudication."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from jclosure.protocol_v32 import stage_freeze, verify, verify_stage
from jclosure.provenance import sha256_file, write_json_atomic

OUT = Path("results/v32/processed")
SOURCE = "src/jclosure/experiments/analyze_v32.py"


def bootstrap_median_lower(values, iterations=2000, seed=320132, tail=0.025):
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 1 or not len(arr) or not np.all(np.isfinite(arr)):
        raise ValueError("bootstrap requires finite nonempty 1D values")
    rng = np.random.default_rng(seed)
    samples = arr[rng.integers(0, len(arr), size=(iterations, len(arr)))]
    return float(np.quantile(np.median(samples, axis=1), tail))


def rank_thresholds(matrix):
    x = np.asarray(matrix, dtype=np.float64)
    x = x - x.mean(axis=0, keepdims=True)
    gram = x @ x.T
    eigen = np.maximum(np.linalg.eigvalsh(gram), 0)[::-1]
    if float(eigen.sum()) <= 1e-12:
        return {"r90": 0, "r95": 0, "r99": 0, "effective_rank": 0.0}
    cumulative = np.cumsum(eigen) / eigen.sum()
    p = eigen / eigen.sum()
    return {"r90": int(np.searchsorted(cumulative, .90)+1), "r95": int(np.searchsorted(cumulative, .95)+1), "r99": int(np.searchsorted(cumulative, .99)+1), "effective_rank": float(np.exp(-np.sum(p[p>0]*np.log(p[p>0]))))}


def restricted_rotation_fit(C, RC, rank=32):
    C, RC = np.asarray(C, np.float64), np.asarray(RC, np.float64)
    stacked = np.concatenate([C, RC], axis=0)
    gram = stacked @ stacked.T
    eigen, u = np.linalg.eigh(gram)
    indices = np.argsort(eigen)[::-1][:rank]
    positive = eigen[indices] > 1e-12
    values, eigvec = eigen[indices][positive], u[:, indices][:, positive]
    basis = (eigvec.T @ stacked) / np.sqrt(values)[:, None]
    lhs, rhs = C @ basis.T, RC @ basis.T
    U, _, Vt = np.linalg.svd(lhs.T @ rhs, full_matrices=False)
    Q = U @ Vt
    return basis, Q


def restricted_rotation_predict(C, basis, Q):
    C = np.asarray(C, np.float64)
    projected = C @ basis.T
    return C + (projected @ Q - projected) @ basis


def gate_summary(frame, cfg):
    primary = frame[frame.primary].copy()
    boot = cfg["bootstrap"]
    repl = cfg["replication_gate"]
    align = cfg["alignment_gate"]
    family = {}
    for fam, group in primary.groupby("family"):
        family[fam] = {"rows": len(group), "positive_fraction": float(group.improvement_positive.mean()), "median_l2_reduction": float(group.relative_conv_error_reduction.median()), "median_alignment": float(group.residual_alignment_cosine.median())}
    positive = float(primary.improvement_positive.mean())
    reduction = float(primary.relative_conv_error_reduction.median())
    reduction_lb = bootstrap_median_lower(primary.relative_conv_error_reduction, boot["iterations"], boot["seed"], boot["lower_tail"])
    alignment = float(primary.residual_alignment_cosine.median())
    alignment_lb = bootstrap_median_lower(primary.residual_alignment_cosine, boot["iterations"], boot["seed"], boot["lower_tail"])
    repl_families = sum(x["positive_fraction"] >= repl["family_row_improvement_min"] and x["median_l2_reduction"] >= repl["median_l2_reduction_min"] for x in family.values())
    align_families = sum(x["median_alignment"] >= align["family_median_cosine_min"] for x in family.values())
    return {"rows": len(primary), "positive_fraction": positive, "median_l2_reduction": reduction, "reduction_bootstrap_lower": reduction_lb, "median_alignment": alignment, "alignment_bootstrap_lower": alignment_lb, "family": family, "replication_families_passing": repl_families, "alignment_families_passing": align_families, "replication_pass": bool(positive >= repl["row_improvement_min"] and reduction >= repl["median_l2_reduction_min"] and reduction_lb > repl["bootstrap_lower_bound_gt"] and repl_families >= repl["families_required"]), "alignment_pass": bool(alignment >= align["median_cosine_min"] and alignment_lb >= align["bootstrap_lower_bound_min"] and align_families >= align["families_required"])}


def context_summary(frame, cfg):
    primary = frame.pivot(index="state_id", columns="condition", values="relative_l2_to_target_donor")
    comp = ["WRONG_TOKEN_REC", "SAME_FAMILY_WRONG_STATE_REC", "CROSS_FAMILY_WRONG_STATE_REC", "SHUFFLED_REC"]
    all_better = np.logical_and.reduce([primary.MATCHED_REC.to_numpy() < primary[x].to_numpy() for x in comp])
    by_id = frame.drop_duplicates("state_id").set_index("state_id").family
    family = {fam: float(np.mean(all_better[np.where(primary.index.map(by_id).to_numpy() == fam)[0]])) for fam in sorted(frame.family.unique())}
    gate = cfg["context_gate"]
    return {"states": len(primary), "matched_better_than_all_four_fraction": float(np.mean(all_better)), "family_success_fraction": family, "families_passing": sum(x >= gate["matched_better_fraction_min"] for x in family.values()), "pass": bool(np.mean(all_better) >= gate["matched_better_fraction_min"] and sum(x >= gate["matched_better_fraction_min"] for x in family.values()) >= gate["families_required"]), "condition_median_error": frame.groupby("condition").relative_l2_to_target_donor.median().to_dict()}


def localization_summary(frame, cfg):
    gate = cfg["localization_gate"]
    success = frame.benefit_positive & (frame.removed_benefit_fraction >= gate["benefit_removed_min"]) & (frame.restored_benefit_fraction >= gate["benefit_restored_min"]) & (frame.remove_direction_cosine >= gate["donor_direction_cosine_min"]) & (frame.restore_direction_cosine >= gate["donor_direction_cosine_min"])
    family = {fam: float(success.loc[g.index].mean()) for fam, g in frame.groupby("family")}
    return {"rows": len(frame), "success_rows": int(success.sum()), "median_removed_fraction": float(frame.removed_benefit_fraction.median()), "median_restored_fraction": float(frame.restored_benefit_fraction.median()), "median_remove_cosine": float(frame.remove_direction_cosine.median()), "median_restore_cosine": float(frame.restore_direction_cosine.median()), "family_success_fraction": family, "families_passing": sum(x >= gate["family_success_fraction_min"] for x in family.values()), "pass": bool(success.mean() >= .5 and sum(x >= gate["family_success_fraction_min"] for x in family.values()) >= gate["families_required"] and frame.writeback_exact.all())}


def analyze(root: Path):
    for role in ("development", "validation"):
        for stage in (f"factorial_{role}", f"trace_{role}", f"intervention_{role}", f"context_{role}", f"layer_map_{role}", f"subsite_{role}"):
            verify_stage(root, stage)
    cfg = verify(root)["config"]
    # Preserve the immutable factorial rows and correct a non-outcome metadata
    # miscount in a separate append-only amendment.
    audit_amendment = {"scope": "factorial_audit_*_v32.parquet:field_count", "recorded_value": 120, "correct_touched_fields_by_condition": {"Y10_REC": 24, "Y01_Conv": 24, "Y11_REC_Conv": 48}, "correct_total_across_three_conditions": 96, "outcome_vectors_or_writeback_hashes_changed": False, "reason": "The original metadata expression 24*(2+2+1) counted 120 rather than 24+24+48=96; per-condition exact tensor equality checks were executed separately and are unaffected."}
    for role in ("development", "validation"):
        audit = pd.read_parquet(root / OUT / f"factorial_audit_{role}_v32.parquet")
        if not audit.field_count.eq(120).all():
            raise RuntimeError(f"V32 audit amendment assumption changed: {role}")
    amendment_path = root / OUT / "audit_metadata_amendment_v32.json"
    write_json_atomic(amendment_path, audit_amendment)
    stage_freeze(root, "audit_amendment", [SOURCE, str(amendment_path.relative_to(root)), "results/v32/processed/factorial_audit_development_v32.parquet", "results/v32/processed/factorial_audit_validation_v32.parquet"], {"amendment_sha256": sha256_file(amendment_path), "outcome_values_changed": False})
    frames = {r: pd.read_parquet(root / OUT / f"factorial_{r}_v32.parquet") for r in ("development", "validation")}
    gate = {r: gate_summary(frames[r], cfg) for r in frames}
    contexts = {r: context_summary(pd.read_parquet(root / OUT / f"context_{r}_v32.parquet"), cfg) for r in frames}
    locality = {r: localization_summary(pd.read_parquet(root / OUT / f"site_intervention_{r}_v32.parquet"), cfg) for r in frames}
    dev = frames["development"][frames["development"].primary]
    val = frames["validation"][frames["validation"].primary]
    vdev = np.load(root / OUT / "factorial_vectors_development_v32.npz")["vectors"][dev.vector_index.to_numpy()].astype(np.float64)
    vval = np.load(root / OUT / "factorial_vectors_validation_v32.npz")["vectors"][val.vector_index.to_numpy()].astype(np.float64)
    Cdev, RCdev, Gdev = vdev[:,2]-vdev[:,0], vdev[:,3]-vdev[:,0], vdev[:,3]-vdev[:,2]
    Cval, RCval = vval[:,2]-vval[:,0], vval[:,3]-vval[:,0]
    basis,Q = restricted_rotation_fit(Cdev, RCdev)
    pred = restricted_rotation_predict(Cval, basis, Q)
    Dval = vval[:,4]-vval[:,0]
    rot_err = np.linalg.norm(Dval-pred,axis=1)/np.maximum(np.linalg.norm(Dval,axis=1),1e-12)
    rotation = {"fit_role": "development_only", "test_role": "validation_only", "subspace_rank": int(len(basis)), "global_transform_only": True, "identity_outside_train_subspace": True, "median_rotation_relative_error": float(np.median(rot_err)), "median_true_joint_relative_error": float(val.joint_relative_l2.median()), "fraction_true_joint_beats_rotation": float(np.mean(val.joint_relative_l2.to_numpy() < rot_err))}
    rank = rank_thresholds(Gdev)
    site = json.loads((root / OUT / "site_selection_v32.json").read_text())
    formal = {"V32-A_REC_CORRECTION_REPLICATED": gate["development"]["replication_pass"] and gate["validation"]["replication_pass"], "V32-B_REC_RESIDUAL_CORRECTION_CONFIRMED": False, "V32-C_REC_CONTEXTUALIZES_CONV": False, "V32-D_LOCALIZED_CORRECTION_SITE": locality["development"]["pass"] and locality["validation"]["pass"], "V32-E_REC_GATE_MEDIATES_CONV_CORRECTION": False, "V32-F_REC_QKV_OR_UPDATE_MEDIATION": False, "V32-G_RESIDUAL_INTEGRATION_MEDIATION": False, "V32-H_LAYER_PAIRED_REC_CONV_ROUTING": False, "V32-I_DISTRIBUTED_REC_CONV_CORRECTION": False, "V32-J_REC_CORRECTION_NOT_GENERAL": False}
    formal["V32-B_REC_RESIDUAL_CORRECTION_CONFIRMED"] = bool(formal["V32-A_REC_CORRECTION_REPLICATED"] and gate["development"]["alignment_pass"] and gate["validation"]["alignment_pass"])
    formal["V32-J_REC_CORRECTION_NOT_GENERAL"] = not formal["V32-A_REC_CORRECTION_REPLICATED"]
    final_opened = bool(formal["V32-D_LOCALIZED_CORRECTION_SITE"])
    cross_model_authorized = bool(formal["V32-B_REC_RESIDUAL_CORRECTION_CONFIRMED"] or formal["V32-C_REC_CONTEXTUALIZES_CONV"] or formal["V32-D_LOCALIZED_CORRECTION_SITE"])
    payload = {"schema_version": 41, "protocol_version": "rec_conv_mechanism_v32", "factorial": gate, "context": contexts, "localization": locality, "rotation": rotation, "correction_dimensionality": rank, "selected_site": site, "formal_outcomes": formal, "independent_final_opened": final_opened, "cross_model_replication_authorized": cross_model_authorized, "H2_REMAINS": True, "H3_AUTHORIZED": False, "DYNAMIC_STATE_SEARCH_AUTHORIZED": False, "AUTONOMOUS_CONTROLLER_AUTHORIZED": False, "audit_metadata_amendment_sha256": sha256_file(amendment_path), "limits": ["Only one development-nominated recurrent-output site was formally tested; failure does not prove no other local site exists.", "Gate/qkv/normalized-read and residual-MLP probes are diagnostic because component candidates were specified after the frozen primary plan.", "Context matching list was frozen after factorial data but before context-specific responses; contextual claim has a narrower preregistration status.", "Global rotation falsification is restricted to a rank-32 train-only subspace with identity outside it.", "Factorial audit field_count metadata was corrected by append-only amendment; exact equality checks and outcome vectors were not changed."]}
    if final_opened:
        raise RuntimeError("V32 final opening requires separate frozen finalist implementation; do not silently open")
    path = root / OUT / "v32_adjudication.json"
    write_json_atomic(path, payload)
    final = {"independent_final_opened": False, "reason": "No development-selected recurrent-output site passed both development and validation localization gates", "candidate_layer": site["candidate_layer"], "final_state_ids_hash": hashlib.sha256(json.dumps([x["base_trial_id"] for x in json.loads((root / OUT / "design_v32.json").read_text())["independent_final"]], sort_keys=True).encode()).hexdigest(), "final_responses_observed": 0}
    fpath = root / OUT / "final_opening_v32.json"
    write_json_atomic(fpath, final)
    fst = stage_freeze(root, "final_opening", [SOURCE, str(fpath.relative_to(root)), str(path.relative_to(root)), "artifacts/rec_conv_mechanism_v32_site_selection.freeze.json"], {"final_opened": False, "final_opening_sha256": sha256_file(fpath), "adjudication_sha256": sha256_file(path)})
    return {"freeze_digest": fst["freeze_digest"], "formal_outcomes": formal, "independent_final_opened": False, "cross_model_replication_authorized": cross_model_authorized}


if __name__ == "__main__":
    print(json.dumps(analyze(Path.cwd()), indent=2))
