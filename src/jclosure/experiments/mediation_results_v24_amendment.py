"""Append-only V24 mediation summarization and BF16 writeback audit.

The frozen mediation runner completed every branch and persisted its branch
tables, then reached an empty regeneration table because the train-frozen
diagnostic candidate is the final model layer.  This module leaves those
branch records untouched, explicitly marks regeneration as not applicable,
and replays the exact BF16 write/readback arithmetic for every frozen write.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from jclosure.cache_v7 import clone_hybrid_cache
from jclosure.experiments import action_pool_v22 as actions
from jclosure.experiments import counterfactual_bank_v19 as v19
from jclosure.experiments.mediation_v24 import OUT, _context, _full_capture
from jclosure.experiments.layer_response_v24 import _projectors
from jclosure.protocol_v24 import stage_freeze, verify, verify_stage
from jclosure.provenance import sha256_file, write_json_atomic


SOURCE = "src/jclosure/experiments/mediation_results_v24_amendment.py"


def _audit(before: np.ndarray, requested: np.ndarray, label: str, metadata: dict) -> dict:
    before_bf16 = torch.as_tensor(before, dtype=torch.bfloat16)
    request = torch.as_tensor(requested, dtype=torch.float32)
    updated = (before_bf16.float() + request).to(torch.bfloat16)
    realized = updated.float() - before_bf16.float()
    rn = float(request.norm())
    zn = float(realized.norm())
    return {
        **metadata,
        "branch": label,
        "requested_norm": rn,
        "realized_norm": zn,
        "cosine": float(torch.dot(request, realized) / max(rn * zn, 1e-12)),
        "gain": zn / max(rn, 1e-12),
        "exact_bf16_roundtrip_simulation": True,
    }


@torch.no_grad()
def run(root: Path) -> dict:
    protocol = verify_stage(root, "mediation_protocol")
    cfg = verify(root)["config"]
    candidate = json.loads((root / OUT / "bottleneck_candidate_selection_v24.json").read_text())
    (
        design, split, prompts, teachers, qmap, bundle, dense, values, rec, att,
        measured, state_layer, ws_layers, ws_count,
    ) = _context(root)
    layer = int(protocol["candidate_layer"])
    device = next(bundle.hf_model.parameters()).device
    proj = _projectors(root, device)
    jids = torch.as_tensor(split["selected_j"], device=device)
    lids = torch.as_tensor(split["selected_logits"], device=device)
    target_scales = {key: float(value) for key, value in split["target_scales"].items()}
    with np.load(root / OUT / "raw_hidden_bottleneck_basis_v24.npz") as data:
        bases = {name: np.asarray(data[name], np.float32) for name in data.files}
    families = sorted({item["family"] for item in design["mediation_validation_states"]})
    audit_rows = []
    for number, item in enumerate(design["mediation_validation_states"], 1):
        base_id = item["base_trial_id"]
        token = int(teachers[base_id][0])
        pref = v19._prefill_history(bundle, str(prompts[base_id]["prompt"]), measured, dense, state_layer)
        p0 = clone_hybrid_cache(pref["cache"])
        q = qmap[item["q_name"]]
        pq = v19.apply(p0, 1.0, v19._q_row(values["directions"], q, float(q["alpha"])), rec, att,
                        "native_fp32_add_bf16_writeback")
        clean = {}
        for state_name, state in (("P0", p0), ("Pq", pq)):
            clean[state_name] = _full_capture(
                root, bundle, dense, state, token, pref["prompt_length"], jids, lids,
                target_scales, ws_layers, ws_count, max(measured), cfg["layers"],
                cfg["architecture_layers"], proj, layer,
            )
            for action_id in design["mediation_action_ids"]:
                edited = v19.apply(
                    state, 1.0,
                    actions._row(values["directions"], design["action_specs"][action_id],
                                 float(design["action_alphas"][action_id]), 1),
                    rec, att, "native_fp32_add_bf16_writeback",
                )
                full = _full_capture(
                    root, bundle, dense, edited, token, pref["prompt_length"], jids, lids,
                    target_scales, ws_layers, ws_count, max(measured), cfg["layers"],
                    cfg["architecture_layers"], proj, layer,
                )
                delta = full["raw_hidden"] - clean[state_name]["raw_hidden"]
                global_basis = bases["global_basis"]
                db = global_basis @ (global_basis.T @ delta)
                perp = delta - db
                meta = {"base_trial_id": base_id, "family": item["family"],
                        "state": state_name, "action_id": action_id}
                audit_rows.append(_audit(clean[state_name]["raw_hidden"], db, "B_ONLY", meta))
                audit_rows.append(_audit(clean[state_name]["raw_hidden"], perp, "PERP_ONLY", meta))
                audit_rows.append(_audit(full["raw_hidden"], -db, "FULL_MINUS_B", meta))
                for control in ("random_basis", "variance_basis", "random_causal_basis"):
                    control_basis = bases[control]
                    control_delta = control_basis @ (control_basis.T @ delta)
                    audit_rows.append(_audit(clean[state_name]["raw_hidden"], control_delta,
                                             f"CONTROL_{control}", meta))
                wrong = families[(families.index(item["family"]) + 1) % len(families)]
                wrong_basis = bases[f"family_{wrong}"]
                wrong_delta = wrong_basis @ (wrong_basis.T @ delta)
                audit_rows.append(_audit(clean[state_name]["raw_hidden"], wrong_delta,
                                         "CONTROL_shuffled", meta))
        q_delta = clean["Pq"]["raw_hidden"] - clean["P0"]["raw_hidden"]
        global_basis = bases["global_basis"]
        q_db = global_basis @ (global_basis.T @ q_delta)
        meta = {"base_trial_id": base_id, "family": item["family"], "state": "same_J", "action_id": "Pq-P0"}
        audit_rows.append(_audit(clean["Pq"]["raw_hidden"], -q_db, "RESTORE_Q_TO_0", meta))
        audit_rows.append(_audit(clean["P0"]["raw_hidden"], q_db, "TRANSPLANT_0_TO_Q", meta))
        print(f"V24 mediation writeback amendment {number}/10", flush=True)

    branch_path = root / OUT / "mediation_branches_v24.parquet"
    restore_path = root / OUT / "restoration_transplant_v24.parquet"
    regeneration_path = root / OUT / "bottleneck_regeneration_v24.parquet"
    frame = pd.read_parquet(branch_path)
    restore = pd.read_parquet(restore_path)
    audit = pd.DataFrame(audit_rows)
    audit_path = root / OUT / "strict_writeback_audit_v24.parquet"
    audit.to_parquet(audit_path, index=False, compression="zstd")

    def med(column: str) -> float:
        return float(frame[column].median())

    rng = np.random.default_rng(240028)
    perp_values = frame["PERP_ONLY_retained_ratio"].to_numpy()
    boots = [float(np.median(rng.choice(perp_values, len(perp_values), replace=True))) for _ in range(1000)]
    upper = float(np.quantile(boots, 0.975))
    summary = {
        "candidate_layer": layer,
        "candidate_k": int(protocol["candidate_k"]),
        "branch_rows": len(frame),
        "B_ONLY": {"relative_l2": med("B_ONLY_relative_l2"), "cosine": med("B_ONLY_cosine"),
                   "norm_ratio": med("B_ONLY_norm_ratio")},
        "PERP_ONLY": {"retained_ratio": med("PERP_ONLY_retained_ratio"), "bootstrap_97_5_upper": upper},
        "FULL_MINUS_B": {"retained_ratio": med("FULL_MINUS_B_retained_ratio")},
        "controls": {
            name: {"relative_l2": med(f"CONTROL_{name}_relative_l2"),
                   "cosine": med(f"CONTROL_{name}_cosine")}
            for name in ("random_basis", "variance_basis", "random_causal_basis", "shuffled")
        },
        "restoration": {
            "relative_to_P0_median": float(restore.restoration_relative_to_P0.median()),
            "intervention_mediated_fraction_median": float(restore.intervention_mediated_fraction.median()),
        },
        "transplant": {
            "relative_l2_median": float(restore.transplant_relative_l2_to_TE.median()),
            "cosine_median": float(restore.transplant_cosine_to_TE.median()),
            "norm_ratio_median": float(restore.transplant_norm_ratio.median()),
        },
        "regeneration": {
            "status": "NOT_APPLICABLE_CANDIDATE_IS_FINAL_LAYER",
            "later_projected_ratio_median": None,
            "by_layer": {},
        },
        "writeback_audit": {
            "count": len(audit),
            "min_cosine": float(audit.cosine.min()),
            "median_gain": float(audit.gain.median()),
            "parquet_sha256": sha256_file(audit_path),
        },
        "branch_parquet_sha256": sha256_file(branch_path),
        "restoration_parquet_sha256": sha256_file(restore_path),
        "regeneration_parquet_sha256": sha256_file(regeneration_path),
        "append_only_amendment": "empty regeneration table at final-layer diagnostic candidate",
        "historical_final_opened": False,
        "v24_independent_final_opened": False,
    }
    gate = cfg["causal_gate"]
    summary["CAUSAL_MEDIATION_GATE_PASS"] = bool(
        summary["B_ONLY"]["relative_l2"] <= gate["relative_l2_max"]
        and summary["B_ONLY"]["cosine"] >= gate["cosine_min"]
        and gate["magnitude_ratio_min"] <= summary["B_ONLY"]["norm_ratio"] <= gate["magnitude_ratio_max"]
        and summary["PERP_ONLY"]["retained_ratio"] <= gate["perp_retained_ratio_max"]
        and upper <= gate["bootstrap_upper_margin"]
        and summary["FULL_MINUS_B"]["retained_ratio"] <= gate["perp_retained_ratio_max"]
        and summary["restoration"]["intervention_mediated_fraction_median"] >= gate["mediation_fraction_min"]
    )
    summary["horizon_status"] = (
        "H1_FAILED_NO_H2_H4_H8_OPENING" if not summary["CAUSAL_MEDIATION_GATE_PASS"]
        else "H1_PASSED_H2_H4_H8_REQUIRED"
    )
    summary_path = root / OUT / "causal_bottleneck_mediation_v24.json"
    write_json_atomic(summary_path, summary)
    frozen = stage_freeze(
        root,
        "mediation_results_amendment",
        [SOURCE, str(branch_path.relative_to(root)), str(restore_path.relative_to(root)),
         str(regeneration_path.relative_to(root)), str(audit_path.relative_to(root)),
         str(summary_path.relative_to(root))],
        summary,
    )
    return {"freeze_digest": frozen["freeze_digest"], **summary}


if __name__ == "__main__":
    print(json.dumps(run(Path.cwd()), indent=2))
