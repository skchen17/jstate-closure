"""Freeze V30 basis fitting, OOD tests, transport pairings, layer groups and gates."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from jclosure.protocol_v30 import verify, verify_stage, stage_freeze
from jclosure.provenance import write_json_atomic

SOURCE = "src/jclosure/experiments/plan_v30.py"
OUT = Path("results/v30/processed")
REC = [i for i in range(32) if i % 4 != 3]
ATT = [i for i in range(32) if i % 4 == 3]


def hd(x):
    return hashlib.sha256(json.dumps(x, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def pick(item, pairs, n, salt):
    eligible = [p for p in pairs if p["pair_id"] in item["eligible_pair_ids"]]
    eligible.sort(key=lambda p: hd([salt, item["base_trial_id"], p["pair_id"]]))
    if len(eligible) < n:
        raise RuntimeError(f"insufficient {salt} token pairs: {item['base_trial_id']}")
    return [x["pair_id"] for x in eligible[:n]]


def prepare(root: Path):
    verify_stage(root, "design")
    d = json.loads((root / OUT / "design_v30.json").read_text())
    cfg = verify(root)["config"]
    by_role = {r: [x for x in d["token_pair_library"] if x["role"] == r] for r in ("TOKEN_TRAIN", "TOKEN_VALIDATION", "TOKEN_FINAL")}
    fit, hold = [], []
    fitting = {}
    for fam in d["families"]:
        rows = [x for x in d["development"] if x["family"] == fam]
        fit += rows[:18]
        hold += rows[18:]
        for j, item in enumerate(rows[:18]):
            ordered = by_role["TOKEN_TRAIN"]
            selected = []
            for z in range(72):
                p = ordered[(j * 8 + z) % len(ordered)]
                if p["pair_id"] in item["eligible_pair_ids"]:
                    selected.append(p["pair_id"])
                if len(selected) == 8:
                    break
            if len(selected) != 8:
                raise RuntimeError("basis fit token coverage insufficient")
            fitting[item["base_trial_id"]] = selected
    targets = {"development": hold, "validation": d["validation"], "independent_final": d["independent_final"]}
    evaluations = {}
    transports = {}
    for role, items in targets.items():
        evaluations[role] = {}
        for item in items:
            sid = item["base_trial_id"]
            train = pick(item, by_role["TOKEN_TRAIN"], 2, "STATE_OOD")
            token_role = "TOKEN_FINAL" if role == "independent_final" else "TOKEN_VALIDATION"
            held = pick(item, by_role[token_role], 2, token_role)
            evaluations[role][sid] = {"state_OOD_pair_ids": train, "heldout_token_pair_ids": held}
            for pair_id in held:
                eligible_sources = [x for x in fit if x["family"] == item["family"] and pair_id in x["eligible_pair_ids"]]
                eligible_sources.sort(key=lambda x: hd([cfg["seed"], "TRANSPORT_SOURCE", sid, pair_id, x["base_trial_id"]]))
                if not eligible_sources:
                    raise RuntimeError("no eligible cross-state source")
                transports[f"{sid}:{pair_id}"] = eligible_sources[0]["base_trial_id"]
    seen_token_eval = {}
    for fam in d["families"]:
        for item in [x for x in fit if x["family"] == fam][:2]:
            seen_token_eval[item["base_trial_id"]] = pick(item, by_role["TOKEN_VALIDATION"], 2, "TOKEN_OOD_SEEN_STATE")
    conv_profile = {}
    for role, items in (("development", hold), ("validation", d["validation"])):
        chosen = [x for fam in d["families"] for x in [y for y in items if y["family"] == fam][:2]]
        conv_profile[role] = [{"base_trial_id": x["base_trial_id"], "pair_id": evaluations[role][x["base_trial_id"]]["heldout_token_pair_ids"][0]} for x in chosen]
    quartiles = [REC[i:i + 6] for i in (0, 6, 12, 18)]
    conv_groups = {"single": [[i] for i in REC], "quartile": quartiles, "half": [REC[:12], REC[12:]], "prefix": [REC[:i] for i in (6, 12, 18, 24)], "suffix": [REC[-i:] for i in (6, 12, 18, 24)], "leave_quartile_out": [[x for x in REC if x not in g] for g in quartiles]}
    plan = {"fit_state_ids": [x["base_trial_id"] for x in fit], "development_holdout_ids": [x["base_trial_id"] for x in hold], "validation_ids": [x["base_trial_id"] for x in d["validation"]], "independent_final_ids": [x["base_trial_id"] for x in d["independent_final"]], "basis_fit_pair_ids_per_state": fitting, "basis_fit_contrasts_per_state": 8, "eval_pairs": evaluations, "seen_state_token_OOD_pairs": seen_token_eval, "transport_source_state": transports, "local_basis_rule": "all eligible TOKEN_TRAIN pair contrasts in target state; never held-out-token write", "global_basis_rule": "centered response-blind PCA from 90 fit states x 8 frozen TOKEN_TRAIN pairs", "family_basis_rule": "same global fitting rule within family", "LOFO_basis_rule": "global fitting rows excluding held-out family", "coordinate_extract_rule": "orthogonal projection of source held-out natural write into TRAIN-only source local basis", "transport_fit_rule": "TRAIN-token-only paired source/target local coordinates; ridge alpha=1 and orthogonal Procrustes fixed", "transport_ridge_alpha": 1.0, "transport_methods": ["GLOBAL_FIXED", "FAMILY_FIXED", "LOCAL_ORACLE", "TRANSPORTED_LOCAL_RIDGE", "TRANSPORTED_LOCAL_PROCRUSTES", "RAW_SOURCE_DELTA_CONTROL"], "k_grid": cfg["k_grid"], "gates": cfg["causal_gate"], "future_probe_count": cfg["future_probe_count"], "conv_layers": REC, "attention_layers": ATT, "conv_groups": conv_groups, "conv_profile_states": conv_profile, "conv_minimal_selection_rule": "development-only smallest nested prefix/suffix/half/quartile passing reciprocal gate; fallback full24", "finalist_priority": cfg["finalist_priority"], "final_rule": cfg["final_rule"], "amplitudes": cfg["amplitudes"], "horizons": [1, 2, 4], "future_responses_observed_before_plan": 0, "independent_final_opened": False, "historical_final_opened": False}
    path = root / OUT / "execution_plan_v30.json"
    write_json_atomic(path, plan)
    fr = stage_freeze(root, "execution_plan", [SOURCE, str(path.relative_to(root)), "artifacts/transferable_natural_writes_v30_design.freeze.json"], {"plan_hash": hd(plan), "fit_state_hash": hd(plan["fit_state_ids"]), "OOD_pair_hash": hd(evaluations), "transport_pair_hash": hd(transports), "conv_group_hash": hd(conv_groups), "future_responses_observed_before_plan": 0})
    return {"freeze_digest": fr["freeze_digest"], "fit_states": len(fit), "development_holdout": len(hold), "validation": len(d["validation"]), "final": len(d["independent_final"]), "transport_pairs": len(transports)}


if __name__ == "__main__":
    print(json.dumps(prepare(Path.cwd()), indent=2))
