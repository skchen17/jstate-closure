"""Freeze V31 methods and prospective causal evaluation fixtures."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from jclosure.protocol_v31 import stage_freeze, verify, verify_stage
from jclosure.provenance import write_json_atomic

OUT = Path("results/v31/processed")
SOURCE = "src/jclosure/experiments/plan_v31.py"
CONV = [i for i in range(32) if i % 4 != 3]


def hd(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def prepare(root: Path):
    verify_stage(root, "design")
    cfg = verify(root)["config"]
    design = json.loads((root / OUT / "design_v31.json").read_text())
    split = design["surface_composition_splits"]
    evaluation = {}
    for role, ids in (("development", design["development_holdout_ids"]), ("validation", [x["base_trial_id"] for x in design["validation"]])):
        by_id = {x["base_trial_id"]: x for x in design[role]}
        evaluation[role] = {}
        for sid in ids:
            item = by_id[sid]
            allowed = set(item["eligible_compositions"]["COMPOSITION_VALIDATION"])
            choices = [x for x in split["COMPOSITION_VALIDATION"] if x["composition_id"] in allowed]
            choices.sort(key=lambda x: hd([cfg["seed"], "HELDOUT_COMPOSITION", sid, x["composition_id"]]))
            if len(choices) < 2:
                raise RuntimeError(f"V31 composition evaluation shortfall: {sid}")
            evaluation[role][sid] = [x["composition_id"] for x in choices[:2]]
    factorial = {role: [sid for fam in cfg["families"] for sid in [x["base_trial_id"] for x in design[role] if x["family"] == fam and x["base_trial_id"] in evaluation[role]][:2]] for role in ("development", "validation")}
    quartiles = [CONV[i:i + 6] for i in (0, 6, 12, 18)]
    groups = {"single": [[i] for i in CONV], "quartile": quartiles, "half": [CONV[:12], CONV[12:]], "prefix": [CONV[:k] for k in (6, 12, 18, 24)], "suffix": [CONV[-k:] for k in (6, 12, 18, 24)], "leave_quartile_out": [[x for x in CONV if x not in g] for g in quartiles]}
    plan = {
        "fit_state_ids": design["fit_state_ids"],
        "fit_pair_ids_per_state": design["fit_pair_ids_per_state"],
        "fit_composition_ids_per_state": {x["base_trial_id"]: x["eligible_compositions"]["COMPOSITION_TRAIN"] for x in design["development"] if x["base_trial_id"] in design["fit_state_ids"]},
        "development_holdout_ids": design["development_holdout_ids"],
        "validation_ids": [x["base_trial_id"] for x in design["validation"]],
        "independent_final_ids": [x["base_trial_id"] for x in design["independent_final"]],
        "heldout_composition_eval": evaluation,
        "rec_conv_factorial_state_ids": factorial,
        "depth_group_layers": groups,
        "depth_route_rule": "On development only, select smallest Conv group with causal fidelity meeting gate; validate exact same frozen group on validation. No activation-norm-only route claims.",
        "primitive_model_rules": {
            "PCA": "Centered TRAIN write Gram eigendecomposition; project held-out write onto first M orthonormal directions. Geometry baseline, not a primitive claim.",
            "SPARSE_DICTIONARY": "TRAIN-only normalized K-SVD/orthogonal-matching-pursuit dictionary in empirical write span; M grid and active s grid frozen.",
            "CLUSTERED_PROTOTYPES": "TRAIN-only deterministic spherical k-means on normalized write Gram, then centroid primitives; sparse OMP reconstruction.",
            "FUNCTION_CONDITIONED": "Separate TRAIN-only dictionaries for frozen token-surface categories; no labels learned from responses.",
            "RESPONSE_FACTOR": "TRAIN-only cross-covariance/PLS between write coefficients and six-probe future response; validation excluded.",
            "CONV_DEPTH": "TRAIN-only depth-resolved Conv groups define primitives; all24 exact layer index and channel semantics retained.",
            "REC_CONDITIONAL_CONV": "TRAIN-only Conv primitive plus low-capacity REC correction predicted from frozen prewrite state and token metadata.",
        },
        "composition_model_rules": {
            "UNIT_ADDITIVE": "w(A)+w(B), where each w(token)=W(token)-W(anchor) in exactly the same incoming state; never use the tautological a->b + b->c identity.",
            "GLOBAL_SCALAR_GATED": "alpha*w(A)+beta*w(B); alpha,beta fit by TRAIN-only least squares on COMPOSITION_TRAIN AB writes.",
            "LOW_ORDER_INTERACTION": "alpha*w(A)+beta*w(B)+gamma*sqrt(D)*(w(A)*w(B))/(max(norm(w(A)),eps)*max(norm(w(B)),eps)); three global scalars fit TRAIN-only.",
            "STATE_CONDITIONED_SCALAR_GATED": "alpha(P)*w(A)+beta(P)*w(B), each scalar linear in standardized prewrite prompt length and frozen mean token rank; coefficients fit TRAIN-only with ridge=1.",
        },
        "composition_controls": ["A_ONLY", "B_ONLY", "RANDOM_B_WITHIN_STATE", "SIGN_FLIPPED_B", "EXACT_NATURAL_CEILING"],
        "fit_selection_rule": "All response-blind TRAIN contrasts from 100 development fit states; no held-out state/token/family future response used for dictionary fitting except RESPONSE_FACTOR uses TRAIN futures only.",
        "model_selection_rule": "Evaluate frozen grids on development; only development-passing mechanism may be tested on validation; priority from base config; no validation fitting or final-based selection.",
        "causal_gate": cfg["causal_gate"],
        "primitive_count_grid": cfg["primitive_count_grid"],
        "active_count_grid": cfg["active_count_grid"],
        "future_probe_count": cfg["future_probe_count"],
        "horizons": [1, 2, 4],
        "all_current_token_writes_unobserved_before_plan": True,
        "all_future_responses_unobserved_before_plan": True,
        "independent_final_opened": False,
    }
    path = root / OUT / "execution_plan_v31.json"
    write_json_atomic(path, plan)
    freeze = stage_freeze(root, "execution_plan", [SOURCE, str(path.relative_to(root)), "artifacts/compositional_natural_writes_v31_design.freeze.json"], {"plan_hash": hd(plan), "evaluation_hash": hd(evaluation), "depth_groups_hash": hd(groups), "future_responses_observed_before_plan": 0})
    return {"freeze_digest": freeze["freeze_digest"], "fit_states": len(plan["fit_state_ids"]), "development_holdout_states": len(plan["development_holdout_ids"]), "validation_states": len(plan["validation_ids"]), "factorial_states": {k: len(v) for k, v in factorial.items()}}


if __name__ == "__main__":
    print(json.dumps(prepare(Path.cwd()), indent=2))
