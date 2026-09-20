"""State-clustered V19 factorial statistics and matched-budget context comparison."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from jclosure.experiments.counterfactual_bank_v19 import OUT
from jclosure.protocol_v19 import stage_freeze, verify, verify_stage
from jclosure.provenance import sha256_file, write_json_atomic

SOURCE = "src/jclosure/experiments/analyze_v19.py"
BLOCKS = {"j": slice(0, 128), "logits": slice(128, 160),
          "semantic_continuous": slice(160, 192), "workspace": slice(192, 288),
          "stacked_normalized": slice(0, 288)}


def prepare(root: Path) -> dict:
    split = verify_stage(root, "splits")
    q = verify_stage(root, "q_amendment_2")
    teacher = verify_stage(root, "teacher_amendment_3")
    cfg = verify(root)["config"]
    return stage_freeze(root, "analysis",
                        [SOURCE, "artifacts/counterfactual_workspace_v19_q_amendment_2.freeze.json",
                         "artifacts/counterfactual_workspace_v19_teacher_amendment_3.freeze.json",
                         "results/v18/processed/clean_state_scores_v18.npz"],
                        {"split_freeze_digest": split["freeze_digest"],
                         "q_freeze_digest": q["freeze_digest"], "teacher_freeze_digest": teacher["freeze_digest"],
                         "primary_target": "stacked_normalized", "primary_horizon": 1,
                         "target_slices": {key: [value.start, value.stop] for key, value in BLOCKS.items()},
                         "bootstrap_unit": "base_state", "bootstrap_replicates": cfg["targets"]["bootstrap_replicates"],
                         "bootstrap_seed": cfg["targets"]["bootstrap_seed"],
                         "modulation_margin": cfg["targets"]["modulation_equivalence_margin"],
                         "dependence_lower_floor": cfg["targets"]["dependence_lower_floor"],
                         "natural_active_ratio_min": cfg["q"]["natural_active_ratio_min"],
                         "matched_action_only_for_primary": True,
                         "prediction_model": "fixed_ridge_linear_intercept_128_state_8_action",
                         "ridge_lambda": 1.0, "prediction_action_sampling": "one_hash_selected_development_signed_action_per_base_state",
                         "prediction_target_normalization": "frozen_V16_stack_and_train_mean_centered_denominator",
                         "prediction_raw_context": "frozen_V18_j_rec_conv_kv_128D",
                         "prediction_j_context": "frozen_V18_j_128D",
                         "prediction_budget": "same_states_one_row_per_state_same_136D_design_fixed_lambda",
                         "response_rows_observed_at_freeze": 0,
                         "validation_rows_observed_at_freeze": 0})


def _bank(root: Path, role: str) -> pd.DataFrame:
    paths = sorted((root / OUT).glob(f"factorial_{role}_*_v19.parquet"))
    if len(paths) != 5:
        raise RuntimeError(f"V19 {role} factorial family count {len(paths)} != 5")
    return pd.concat([pd.read_parquet(path) for path in paths], ignore_index=True)


def _ci_state(frame: pd.DataFrame, column: str, seed: int, reps: int) -> dict:
    state = frame.groupby("base_trial_id", sort=True)[column].median().to_numpy(dtype=float)
    if not len(state):
        return {"median": None, "ci95": [None, None], "states": 0}
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(state), size=(reps, len(state)))
    bs = np.median(state[draws], axis=1)
    return {"median": float(np.median(state)), "ci95": np.quantile(bs, [0.025, 0.975]).tolist(),
            "states": len(state)}


def _summary(frame: pd.DataFrame, cfg: dict, seed: int) -> dict:
    eligible = frame[frame.q_reliable & (frame.action_status == "MATCHED_REALIZED_ACTION")]
    result = {"all_rows": len(frame), "matched_rows": int((frame.action_status == "MATCHED_REALIZED_ACTION").sum()),
              "matched_rate": float((frame.action_status == "MATCHED_REALIZED_ACTION").mean()),
              "q_reliable_rate": float(frame.q_reliable.mean()),
              "eligible_rows": len(eligible), "eligible_states": int(eligible.base_trial_id.nunique()),
              "by_target_horizon": {}, "by_family": {}, "by_action": {}, "by_q_channel": {}, "by_q_name": {}}
    reps = int(cfg["targets"]["bootstrap_replicates"])
    for (target, h), group in eligible.groupby(["target", "horizon"], sort=True):
        key = f"{target}:h{h}"
        result["by_target_horizon"][key] = {
            "natural_norm": _ci_state(group, "natural_norm", seed + int(h), reps),
            "clean_action_norm": _ci_state(group, "clean_action_norm", seed + int(h) + 10, reps),
            "counterfactual_action_norm": _ci_state(group, "counterfactual_action_norm", seed + int(h) + 20, reps),
            "modulation_norm": _ci_state(group, "modulation_norm", seed + int(h) + 30, reps),
            "modulation_ratio": _ci_state(group, "modulation_ratio", seed + int(h) + 40, reps),
            "natural_to_action_ratio": _ci_state(group, "natural_to_action_ratio", seed + int(h) + 50, reps),
            "response_cosine_median": float(group.response_cosine.median()),
            "response_magnitude_ratio_median": float(group.response_magnitude_ratio.median()),
            "rows": len(group), "states": int(group.base_trial_id.nunique())}
    primary = eligible[(eligible.target == "stacked_normalized") & (eligible.horizon == 1)]
    for label, column in (("by_family", "family"), ("by_action", "coordinate_index"),
                          ("by_q_channel", "q_channel"), ("by_q_name", "q_name")):
        for name, group in primary.groupby(column, sort=True):
            result[label][str(name)] = {"modulation_ratio": _ci_state(group, "modulation_ratio", seed + 101, reps),
                                       "natural_to_action_ratio": _ci_state(group, "natural_to_action_ratio", seed + 102, reps),
                                       "matched_rate_all": float((frame[(frame.target == "stacked_normalized") & (frame.horizon == 1) & (frame[column] == name)].action_status == "MATCHED_REALIZED_ACTION").mean()),
                                       "rows": len(group), "states": int(group.base_trial_id.nunique())}
    return result


def analyze(root: Path, role: str) -> dict:
    design = verify_stage(root, "analysis")
    cfg = verify(root)["config"]
    bank = _bank(root, role)
    if not (bank.boundary_j_difference == 0).all() or not (bank.boundary_j_hash_p0 == bank.boundary_j_hash_pq).all():
        raise RuntimeError("V19 exact boundary-J invariant failed")
    if not (bank.p0_snapshot_hash != bank.pq_snapshot_hash).all():
        raise RuntimeError("V19 q failed to alter persistent snapshot")
    values = {key: np.stack(bank[key]).astype(np.float64) for key in ("y00_stack", "y01_stack", "y10_stack", "y11_stack")}
    y00, y01, y10, y11 = [values[key] for key in ("y00_stack", "y01_stack", "y10_stack", "y11_stack")]
    natural, r0, rq = y10 - y00, y01 - y00, y11 - y10
    mod = rq - r0
    parts = []
    base_columns = ["base_trial_id", "family", "role", "horizon", "q_name", "q_channel", "q_reliable",
                    "q_alpha", "coordinate_index", "action_sign", "action_role", "action_status",
                    "realized_action_pair_cosine", "realized_norm_ratio_pq_over_p0", "boundary_j_difference"]
    for target, sl in BLOCKS.items():
        n, a, b, m = [np.linalg.norm(x[:, sl], axis=1) for x in (natural, r0, rq, mod)]
        eps = float(cfg["targets"]["epsilon"])
        factor = (float(cfg["targets"].get("unused_scale", 1.0)) if target == "stacked_normalized"
                  else float(verify_stage(root, "splits")["target_scales"][target]) * math.sqrt(sl.stop - sl.start))
        cos = np.sum(r0[:, sl] * rq[:, sl], axis=1) / np.maximum(a * b, eps)
        frame = bank[base_columns].copy()
        frame["target"] = target
        frame["natural_norm"] = n
        frame["clean_action_norm"] = a
        frame["counterfactual_action_norm"] = b
        frame["modulation_norm"] = m
        frame["natural_raw_block_norm"] = n * factor
        frame["clean_action_raw_block_norm"] = a * factor
        frame["counterfactual_action_raw_block_norm"] = b * factor
        frame["modulation_raw_block_norm"] = m * factor
        frame["modulation_ratio"] = m / np.maximum(np.maximum(a, b), eps)
        frame["natural_to_action_ratio"] = n / np.maximum(a, eps)
        frame["response_cosine"] = cos
        frame["response_magnitude_ratio"] = b / np.maximum(a, eps)
        parts.append(frame)
    metrics = pd.concat(parts, ignore_index=True)
    path = root / OUT / f"factorial_metrics_{role}_v19.parquet"
    metrics.to_parquet(path, index=False, compression="zstd", compression_level=9)
    summary = _summary(metrics, cfg, int(cfg["targets"]["bootstrap_seed"]))
    summary.update({"role": role, "factorial_states": int(bank.base_trial_id.nunique()),
                    "factorial_rows": len(bank), "metrics_path": str(path.relative_to(root)),
                    "metrics_sha256": sha256_file(path), "analysis_freeze_digest": design["freeze_digest"],
                    "boundary_j_exact_all": True, "persistent_snapshot_distinct_all": True})
    write_json_atomic(root / OUT / f"factorial_summary_{role}_v19.json", summary)
    return {"role": role, "states": summary["factorial_states"], "rows": len(bank),
            "matched_rate": summary["matched_rate"]}


def _ridge_predict(xtr: np.ndarray, ytr: np.ndarray, xva: np.ndarray, lam: float) -> np.ndarray:
    mean_x = xtr.mean(0); mean_y = ytr.mean(0)
    x = xtr - mean_x
    gram = (x.T @ x) / len(x)
    gram.flat[::len(gram) + 1] += lam
    weights = np.linalg.solve(gram, (x.T @ (ytr - mean_y)) / len(x))
    return (xva - mean_x) @ weights + mean_y


def _errors(y: np.ndarray, pred_j: np.ndarray, pred_raw: np.ndarray, train_mean: np.ndarray,
            state_ids: list[str], seed: int, reps: int) -> dict:
    denominator = np.sum((y - train_mean)**2, axis=1)
    ej = np.sum((y - pred_j)**2, axis=1)
    er = np.sum((y - pred_raw)**2, axis=1)
    total = float(np.sum(denominator))
    point = float((math.sqrt(float(ej.sum())) - math.sqrt(float(er.sum()))) / max(math.sqrt(total), 1e-12))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(y), size=(reps, len(y)))
    den = denominator[idx].sum(1); aj = ej[idx].sum(1); ar = er[idx].sum(1)
    bs = (np.sqrt(aj) - np.sqrt(ar)) / np.maximum(np.sqrt(den), 1e-12)
    return {"raw_gain": point, "bootstrap_ci95": np.quantile(bs, [0.025, 0.975]).tolist(),
            "j_relative_l2": float(math.sqrt(float(ej.sum())) / max(math.sqrt(total), 1e-12)),
            "raw_relative_l2": float(math.sqrt(float(er.sum())) / max(math.sqrt(total), 1e-12)),
            "states": len(state_ids)}


def context(root: Path) -> dict:
    design = verify_stage(root, "analysis")
    split = verify_stage(root, "splits")
    cfg = verify(root)["config"]
    with np.load(root / "results/v18/processed/clean_state_scores_v18.npz", allow_pickle=False) as payload:
        scores = {name: payload[name] for name in ("base_trial_id", "j", "j_rec_conv_kv")}
    index = {str(key): i for i, key in enumerate(scores["base_trial_id"].astype(str))}
    banks = {role: _bank(root, role) for role in ("train", "validation")}
    outputs = {}
    for h in cfg["targets"]["horizons"]:
        data = {}
        for role, bank in banks.items():
            subset = bank[(bank.horizon == h) & (bank.action_role == "development")]
            rows = []
            for base_id, group in subset.groupby("base_trial_id", sort=True):
                unique = group.drop_duplicates(["coordinate_index", "action_sign"]).sort_values(["coordinate_index", "action_sign"])
                chosen = int(hashlib.sha256(f"v19-context:{base_id}".encode()).hexdigest(), 16) % len(unique)
                rows.append(unique.iloc[chosen])
            frame = pd.DataFrame(rows).sort_values("base_trial_id")
            ids = [str(x) for x in frame.base_trial_id]
            if len(ids) != (400 if role == "train" else 100) and h == 1:
                raise RuntimeError(f"V19 h1 {role} context panel incomplete")
            state_idx = np.asarray([index[x] for x in ids])
            action = np.zeros((len(ids), 8), dtype=np.float64)
            action_lookup = {(coord, sign): i for i, (coord, sign) in enumerate(
                (coord, sign) for coord in cfg["actions"]["development_coordinates"] for sign in cfg["actions"]["signs"])}
            for i, row in enumerate(frame.itertuples()):
                action[i, action_lookup[(int(row.coordinate_index), int(row.action_sign))]] = float(row.action_alpha)
            y00 = np.stack(frame.y00_stack).astype(np.float64)
            y01 = np.stack(frame.y01_stack).astype(np.float64)
            data[role] = {"ids": ids, "j": scores["j"][state_idx, :128].astype(np.float64),
                          "raw": scores["j_rec_conv_kv"][state_idx, :128].astype(np.float64),
                          "action": action, "natural": y00, "response": y01 - y00}
        outcomes = {}
        for endpoint in ("natural", "response"):
            tr, va = data["train"], data["validation"]
            atr = np.zeros_like(tr["action"]) if endpoint == "natural" else tr["action"]
            ava = np.zeros_like(va["action"]) if endpoint == "natural" else va["action"]
            ytr, yva = tr[endpoint], va[endpoint]
            xjtr = np.concatenate((tr["j"], atr), axis=1)
            xjva = np.concatenate((va["j"], ava), axis=1)
            xrtr = np.concatenate((tr["raw"], atr), axis=1)
            xrva = np.concatenate((va["raw"], ava), axis=1)
            pred_j = _ridge_predict(xjtr, ytr, xjva, float(design["ridge_lambda"]))
            pred_r = _ridge_predict(xrtr, ytr, xrva, float(design["ridge_lambda"]))
            outcomes[endpoint] = {name: _errors(yva[:, sl], pred_j[:, sl], pred_r[:, sl],
                                               ytr[:, sl].mean(0), va["ids"],
                                               int(cfg["targets"]["bootstrap_seed"]) + h, int(cfg["targets"]["bootstrap_replicates"]))
                                  for name, sl in BLOCKS.items()}
        outputs[str(h)] = {"train_states": len(data["train"]["ids"]), "validation_states": len(data["validation"]["ids"]),
                           "natural": outcomes["natural"], "action_response": outcomes["response"],
                           "delta_gain_stack": outcomes["natural"]["stacked_normalized"]["raw_gain"] - outcomes["response"]["stacked_normalized"]["raw_gain"]}
    result = {"analysis_freeze_digest": design["freeze_digest"], "horizons": outputs,
              "validation_is_development_not_independent": True,
              "model_limitation": "Fixed linear ridge is a matched-capacity comparison, not an exhaustive nonlinear raw-state ceiling."}
    write_json_atomic(root / OUT / "natural_vs_action_context_v19.json", result)
    return {"horizons": {h: {"natural_gain": x["natural"]["stacked_normalized"]["raw_gain"],
                             "action_gain": x["action_response"]["stacked_normalized"]["raw_gain"]}
                          for h, x in outputs.items()}}


def decide(root: Path) -> dict:
    design = verify_stage(root, "analysis")
    cfg = verify(root)["config"]
    summary = json.loads((root / OUT / "factorial_summary_validation_v19.json").read_text())
    context_result = json.loads((root / OUT / "natural_vs_action_context_v19.json").read_text())
    q = verify_stage(root, "q_amendment_2")
    active = {x["name"] for x in q["q"] if x["calibration_status"] == "RELIABLE" and x["natural_active_train_pilot"]}
    metrics = pd.read_parquet(root / OUT / "factorial_metrics_validation_v19.parquet")
    primary = metrics[(metrics.target == "stacked_normalized") & (metrics.horizon == 1) &
                      metrics.q_name.isin(active) & metrics.q_reliable &
                      (metrics.action_status == "MATCHED_REALIZED_ACTION")]
    ratio = _ci_state(primary, "modulation_ratio", 1981, int(cfg["targets"]["bootstrap_replicates"]))
    natural = _ci_state(primary, "natural_to_action_ratio", 1982, int(cfg["targets"]["bootstrap_replicates"]))
    matched_rate = summary["matched_rate"]
    material_n = natural["median"] is not None and natural["median"] >= cfg["q"]["natural_active_ratio_min"] and natural["ci95"][0] > 0
    family = primary.groupby("family").modulation_ratio.median()
    action = primary.groupby("coordinate_index").modulation_ratio.median()
    channels = primary.groupby("q_channel").modulation_ratio.median()
    dependence = (ratio["median"] is not None and ratio["median"] >= cfg["targets"]["modulation_equivalence_margin"]
                  and ratio["ci95"][0] > cfg["targets"]["dependence_lower_floor"]
                  and int((action >= cfg["targets"]["modulation_equivalence_margin"]).sum()) >= cfg["targets"]["require_multiple_actions"]
                  and int((channels >= cfg["targets"]["modulation_equivalence_margin"]).sum()) >= cfg["targets"]["require_multiple_q_families"]
                  and int((family >= cfg["targets"]["modulation_equivalence_margin"]).sum()) >= 2
                  and matched_rate >= 0.8)
    equivalence = (ratio["median"] is not None and ratio["ci95"][1] < cfg["targets"]["modulation_equivalence_margin"]
                   and all(family < cfg["targets"]["modulation_equivalence_margin"])
                   and all(action < cfg["targets"]["modulation_equivalence_margin"])
                   and all(channels < cfg["targets"]["modulation_equivalence_margin"])
                   and material_n and matched_rate >= 0.8)
    nat_gain = context_result["horizons"]["1"]["natural"]["stacked_normalized"]["raw_gain"]
    action_gain = context_result["horizons"]["1"]["action_response"]["stacked_normalized"]["raw_gain"]
    c_response = bool(dependence)
    c_dynamics = bool(material_n and nat_gain >= cfg["targets"]["raw_gain_material_min"] and not dependence)
    if matched_rate < 0.8:
        outcome = "V19-E — ACTION_RESPONSE_MODULATION_NOT_IDENTIFIABLE_UNDER_CURRENT_ACTUATOR"
    elif dependence:
        outcome = "V19-B — CURRENT_J_IS_NOT_SUFFICIENT_FOR_TESTED_FINITE_ACTION_RESPONSE_CONTEXT (development)"
    elif equivalence:
        outcome = "V19-A — PERSISTENT_NATURAL_DYNAMICS_J_RESPONSE_CONTEXT_DISSOCIATION (pending independent confirmation)"
    elif not material_n:
        outcome = "V19-D — TEST_INCONCLUSIVE_FOR_WORKSPACE_SUFFICIENCY"
    else:
        outcome = "V19-STOP — NO_EQUIVALENCE_OR_MATERIAL_DEPENDENCE_GATE"
    result = {"analysis_freeze_digest": design["freeze_digest"], "formal_outcome": outcome,
              "active_q_names_train_frozen": sorted(active), "primary_matched_rows": len(primary),
              "primary_states": int(primary.base_trial_id.nunique()), "matched_rate": matched_rate,
              "modulation_ratio_state_bootstrap": ratio, "natural_to_action_ratio_state_bootstrap": natural,
              "material_natural_effect": bool(material_n), "response_dependence_gate": bool(dependence),
              "response_equivalence_gate": bool(equivalence),
              "raw_gain_natural_h1_stack": nat_gain, "raw_gain_action_response_h1_stack": action_gain,
              "compact_response_context_search_authorized": c_response,
              "compact_natural_dynamics_search_authorized": c_dynamics,
              "compact_search_status": "PENDING_AUTHORIZED_SEARCH" if (c_response or c_dynamics) else "NOT_AUTHORIZED",
              "independent_final_status": "UNOPENED_PENDING_FROZEN_ELIGIBLE_FINALIST",
              "h2_remains": True, "h3_candidate_supported": False,
              "autonomous_state_model_authorized": False, "absolute_replacement_claim": False}
    write_json_atomic(root / OUT / "v19_adjudication.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("prepare", "analyze_train", "analyze_validation", "context", "decide"))
    args = parser.parse_args()
    root = Path.cwd()
    if args.stage == "prepare":
        result = prepare(root)
    elif args.stage.startswith("analyze_"):
        result = analyze(root, args.stage.split("_", 1)[1])
    else:
        result = {"context": context, "decide": decide}[args.stage](root)
    print(json.dumps(result, indent=2))
