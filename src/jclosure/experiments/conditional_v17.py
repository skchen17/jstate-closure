"""V17 cross-fitted conditional raw-residual and exact-action matched pairs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from jclosure.experiments.state_sufficiency_v17 import KERNELS, OUT, _kernel_set, _predict_key, _score, _split
from jclosure.protocol_v17 import stage_freeze, verify
from jclosure.provenance import sha256_file, write_json_atomic

SOURCE = "src/jclosure/experiments/conditional_v17.py"
DESIGN = {
    "status": "diagnostic_after_material_ceiling_gate_failed",
    "conditional_context": "C_empty_J_only",
    "train_only_oof_folds": 5,
    "raw_kernel_pca_rank": 32,
    "raw_residualization": "KRR_from_J_to_train_PCA_raw_features_train_only_crossfit_by_state",
    "target_residualization": "KRR_from_J_to_response_train_only_crossfit_by_state_per_action_key",
    "residual_predictor": "ridge_linear_no_intercept_per_action_key",
    "ridge": 0.1,
    "heldout_validation": "V16_validation_states_reused_exploratory_not_independent",
    "matched_pair_action": "same_coordinate_index_sign_and_exact_calibration_alpha",
    "matched_pair_type_a": "largest_raw_minus_J_distance_percentile_gap_top_decile",
    "matched_pair_type_c": "largest_J_minus_raw_distance_percentile_gap_top_decile",
    "matched_pair_type_b": "not_defined_without_compact_C",
    "bootstrap_unit": "validation_state",
    "bootstrap_replicates": 1000,
    "seed": 1717,
}


def freeze_design(root: Path) -> dict:
    split = _split(root)
    return stage_freeze(root, "conditional_design", [SOURCE, str(KERNELS), "results/v17/processed/state_context_ceiling_v17.json"],
                        {"parent_split_freeze_digest": split["freeze_digest"], "design": DESIGN})


def _verify_design(root: Path) -> dict:
    from jclosure.protocol_v17 import digest
    base = verify(root)
    value = json.loads((root / "artifacts/interventional_state_sufficiency_v17_conditional_design.freeze.json").read_text())
    if value["freeze_digest"] != digest(value) or value["base_freeze_digest"] != base["freeze_digest"]:
        raise RuntimeError("V17 conditional design freeze mismatch")
    amendment = None
    parent_digest = value["freeze_digest"]
    for number in (1, 2):
        amendment_path = root / f"artifacts/interventional_state_sufficiency_v17_conditional_source_amendment_{number}.freeze.json"
        if not amendment_path.exists():
            break
        amendment = json.loads(amendment_path.read_text())
        if amendment["freeze_digest"] != digest(amendment) or amendment["parent_conditional_freeze_digest"] != parent_digest:
            raise RuntimeError("V17 conditional source amendment mismatch")
        parent_digest = amendment["freeze_digest"]
    for path, expected in value["input_hashes"].items():
        if path == SOURCE and amendment is not None:
            expected = amendment["input_hashes"][SOURCE]
        if sha256_file(root / path) != expected:
            raise RuntimeError(f"V17 conditional input changed: {path}")
    return value


def _feature(k: np.ndarray, n_train: int, rank: int) -> tuple[np.ndarray, int]:
    values, vectors = np.linalg.eigh(k[:n_train, :n_train].astype(np.float64))
    positive = np.flatnonzero(values > max(float(values.max()) * 1e-8, 1e-10))
    chosen = positive[-rank:][::-1]
    if len(chosen) == 0:
        raise RuntimeError("No positive raw context kernel eigenvalues")
    scores = k[:, :n_train] @ (vectors[:, chosen] / np.sqrt(values[chosen])[None, :])
    return scores.astype(np.float32), len(chosen)


def _raw_residual(x: np.ndarray, j: np.ndarray, ids: list[str], n_train: int, ridge: float) -> tuple[np.ndarray, np.ndarray]:
    folds = np.asarray([int(hashlib.sha256(f"20260920:{key}".encode()).hexdigest(), 16) % 5 for key in ids[:n_train]])
    oof = np.zeros_like(x[:n_train])
    for fold in range(5):
        fit = np.flatnonzero(folds != fold)
        hold = np.flatnonzero(folds == fold)
        oof[hold] = x[hold] - _predict_key(j, fit, hold, x[fit], ridge)
    validation = x[n_train:] - _predict_key(j, np.arange(n_train), np.arange(n_train, len(ids)), x[:n_train], ridge)
    return oof, validation


def _bootstrap_gain(frame: pd.DataFrame, target: str, rng: np.random.Generator) -> list[float]:
    ids = sorted(frame.base_trial_id.unique())
    lookup = {key: np.flatnonzero(frame.base_trial_id.to_numpy() == key) for key in ids}
    actual = np.stack(frame[f"{target}_actual"])
    base = np.stack(frame[f"{target}_base"])
    corrected = np.stack(frame[f"{target}_corrected"])
    sums = np.asarray([[float((actual[lookup[key]] ** 2).sum()),
                        float(((actual[lookup[key]] - base[lookup[key]]) ** 2).sum()),
                        float(((actual[lookup[key]] - corrected[lookup[key]]) ** 2).sum())]
                       for key in ids])
    sample = rng.integers(0, len(ids), size=(1000, len(ids)))
    totals = sums[sample].sum(axis=1)
    gains = (np.sqrt(totals[:, 1]) - np.sqrt(totals[:, 2])) / np.sqrt(np.maximum(totals[:, 0], 1e-24))
    return np.quantile(gains, [0.025, 0.5, 0.975]).tolist()


def conditional(root: Path) -> dict:
    freeze = _verify_design(root)
    with np.load(root / KERNELS, allow_pickle=False) as payload:
        data = {name: payload[name] for name in payload.files}
    ids = data["base_trial_id"].astype(str).tolist()
    n_train = int((data["role"].astype(str) == "train").sum())
    id_to_index = {key: i for i, key in enumerate(ids)}
    kernels = _kernel_set(data)
    j = kernels["j"]
    bank = pd.read_parquet(root / "results/v16/processed/finite_action_bank_v16.parquet")
    bank = bank[(bank.design == "single") & (bank.reliability_status == "RELIABLE") & bank.base_trial_id.isin(ids)].copy()
    bank["state_index"] = bank.base_trial_id.map(id_to_index).astype(int)
    bank["action_key"] = bank.coordinate_index.astype(int).astype(str) + ":" + bank.sign.astype(int).astype(str)
    ceiling = json.loads((root / "results/v17/processed/state_context_ceiling_v17.json").read_text())
    chosen_ridge = float(ceiling["selected_ridge"])
    base_predictions = pd.read_parquet(root / "results/v17/processed/state_context_ceiling_predictions_v17.parquet")
    base_predictions = base_predictions[base_predictions.context == "j"].copy()
    base_predictions.set_index(["base_trial_id", "action_key"], inplace=True, verify_integrity=True)
    eligible = set(base_predictions.index.get_level_values("action_key"))
    rng = np.random.default_rng(DESIGN["seed"])
    results = []
    all_rows = []
    for context in ("rec", "conv", "kv", "rec+conv", "rec+kv", "conv+kv", "rec+conv+kv"):
        raw_k = np.mean([data[name] for name in context.split("+")], axis=0)
        features, rank = _feature(raw_k, n_train, DESIGN["raw_kernel_pca_rank"])
        oof_x, val_x = _raw_residual(features, j, ids, n_train, DESIGN["ridge"])
        rows = []
        for key in sorted(eligible):
            tr = bank[(bank.role == "train") & (bank.action_key == key)]
            va = bank[(bank.role == "validation") & (bank.action_key == key)]
            if tr.empty or va.empty:
                continue
            tr_idx = tr.state_index.to_numpy()
            va_idx = va.state_index.to_numpy()
            alpha_train = tr.calibration_alpha.to_numpy(dtype=np.float32)
            alpha_val = va.calibration_alpha.to_numpy(dtype=np.float32)
            folds = np.asarray([int(hashlib.sha256(f"20260920:{ids[i]}".encode()).hexdigest(), 16) % 5 for i in tr_idx])
            key_records = [{"base_trial_id": row.base_trial_id, "family": row.family, "action_key": key,
                            "calibration_alpha": float(row.calibration_alpha)} for row in va.itertuples()]
            for target, source in (("j", "response_j"), ("stack", "response_stacked_normalized")):
                y_train = np.stack(tr[source]).astype(np.float32) / alpha_train[:, None]
                residual = np.zeros_like(y_train)
                for fold in range(5):
                    fit = folds != fold
                    hold = ~fit
                    if not hold.any():
                        continue
                    residual[hold] = y_train[hold] - _predict_key(j, tr_idx[fit], tr_idx[hold], y_train[fit], chosen_ridge)
                x_train = oof_x[tr_idx].astype(np.float64)
                weights = np.linalg.solve(x_train.T @ x_train + DESIGN["ridge"] * np.eye(rank),
                                          x_train.T @ residual.astype(np.float64))
                correction = (val_x[va_idx - n_train] @ weights).astype(np.float32) * alpha_val[:, None]
                for i, row in enumerate(va.itertuples()):
                    baseline = base_predictions.loc[(row.base_trial_id, key)]
                    key_records[i][f"{target}_actual"] = baseline["j_actual" if target == "j" else "stack_actual"]
                    prediction = np.asarray(baseline["j_predicted" if target == "j" else "stack_predicted"], dtype=np.float32)
                    key_records[i][f"{target}_base"] = prediction.tolist()
                    key_records[i][f"{target}_corrected"] = (prediction + correction[i]).tolist()
            rows.extend(key_records)
        frame = pd.DataFrame(rows)
        frame.insert(0, "raw_context", context)
        metrics = {"raw_context": context, "raw_feature_rank": rank, "validation_rows": len(frame), "targets": {}, "by_family": {}}
        for target in ("j", "stack"):
            actual = np.stack(frame[f"{target}_actual"])
            baseline = np.stack(frame[f"{target}_base"])
            corrected = np.stack(frame[f"{target}_corrected"])
            a = _score(actual, baseline)
            b = _score(actual, corrected)
            metrics["targets"][target] = {"base": a, "corrected": b,
                                           "conditional_raw_gain_rel_l2": a["relative_l2"] - b["relative_l2"],
                                           "gain_bootstrap_ci95_and_median": _bootstrap_gain(frame, target, rng)}
        for family, part in frame.groupby("family"):
            actual = np.stack(part.stack_actual)
            base = np.stack(part.stack_base)
            corrected = np.stack(part.stack_corrected)
            metrics["by_family"][family] = _score(actual, base)["relative_l2"] - _score(actual, corrected)["relative_l2"]
        results.append(metrics)
        all_rows.append(frame)
    pd.concat(all_rows, ignore_index=True).to_parquet(root / OUT / "conditional_raw_residual_predictions_v17.parquet", index=False)
    result = {"design_freeze_digest": freeze["freeze_digest"], "conditional_context": "J_only_C_empty",
              "train_target_and_raw_residuals": "state_grouped_five_fold_OOF",
              "validation_role": "reused_V16_development_validation_not_independent",
              "results": results, "compact_C_sufficiency": "NOT_TESTED_CEILING_GATE_FAILED"}
    write_json_atomic(root / OUT / "conditional_raw_residual_v17.json", result)
    return result


def matched(root: Path) -> dict:
    freeze = _verify_design(root)
    with np.load(root / KERNELS, allow_pickle=False) as payload:
        data = {name: payload[name] for name in payload.files}
    ids = data["base_trial_id"].astype(str).tolist()
    roles = data["role"].astype(str)
    val_idx = np.flatnonzero(roles == "validation")
    id_to_index = {key: i for i, key in enumerate(ids)}
    raw = np.mean([data[name] for name in ("rec", "conv", "kv")], axis=0)
    j = data["j"]
    def distances(k):
        diag = np.diag(k)
        return np.sqrt(np.maximum(diag[:, None] + diag[None, :] - 2 * k, 0))
    dj, dr = distances(j), distances(raw)
    bank = pd.read_parquet(root / "results/v16/processed/finite_action_bank_v16.parquet")
    bank = bank[(bank.role == "validation") & (bank.design == "single") & (bank.reliability_status == "RELIABLE")].copy()
    bank["key"] = bank.coordinate_index.astype(int).astype(str) + ":" + bank.sign.astype(int).astype(str) + ":" + bank.calibration_alpha.astype(str)
    pair_rows = []
    for key, part in bank.groupby("key"):
        unique = part.drop_duplicates("base_trial_id")
        records = list(unique.itertuples())
        for a in range(len(records)):
            for b in range(a + 1, len(records)):
                left, right = records[a], records[b]
                i, l = id_to_index[left.base_trial_id], id_to_index[right.base_trial_id]
                if i not in val_idx or l not in val_idx:
                    continue
                ya = np.asarray(left.response_stacked_normalized, dtype=np.float32)
                yb = np.asarray(right.response_stacked_normalized, dtype=np.float32)
                pair_rows.append({"action_key_alpha": key, "state_a": left.base_trial_id, "state_b": right.base_trial_id,
                                  "family_a": left.family, "family_b": right.family,
                                  "j_distance": float(dj[i, l]), "raw_distance": float(dr[i, l]),
                                  "response_distance": float(np.linalg.norm(ya - yb)),
                                  "response_rel_divergence": float(np.linalg.norm(ya - yb) / max(0.5 * (np.linalg.norm(ya) + np.linalg.norm(yb)), 1e-12))})
    pairs = pd.DataFrame(pair_rows)
    if pairs.empty:
        raise RuntimeError("No exact-alpha matched action pairs")
    pairs["j_percentile"] = pairs.j_distance.rank(pct=True)
    pairs["raw_percentile"] = pairs.raw_distance.rank(pct=True)
    pairs["a_score"] = pairs.raw_percentile - pairs.j_percentile
    pairs["c_score"] = pairs.j_percentile - pairs.raw_percentile
    threshold_a = pairs.a_score.quantile(0.9)
    threshold_c = pairs.c_score.quantile(0.9)
    pairs["pair_type"] = "other"
    pairs.loc[pairs.a_score >= threshold_a, "pair_type"] = "A_similar_J_different_raw_rank_gap"
    pairs.loc[(pairs.c_score >= threshold_c) & (pairs.pair_type == "other"), "pair_type"] = "C_similar_raw_different_J_rank_gap"
    pairs.to_parquet(root / OUT / "matched_state_action_pairs_v17.parquet", index=False)
    summary = {"design_freeze_digest": freeze["freeze_digest"], "exact_same_action_key_and_alpha": True,
               "total_pairs": len(pairs), "pair_types": {}, "type_b": "NOT_DEFINED_NO_COMPACT_CONTEXT",
               "selection_caveat": "rank-gap matching, not absolute close-neighbor matching; descriptive, no causal replacement inference"}
    for label, part in pairs.groupby("pair_type"):
        summary["pair_types"][label] = {"count": len(part), "median_j_distance": float(part.j_distance.median()),
                                        "median_raw_distance": float(part.raw_distance.median()),
                                        "median_response_rel_divergence": float(part.response_rel_divergence.median())}
    write_json_atomic(root / OUT / "matched_state_action_v17.json", summary)
    return summary


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("freeze", "conditional", "matched"))
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    root = args.root.resolve()
    value = {"freeze": freeze_design, "conditional": conditional, "matched": matched}[args.stage](root)
    print(json.dumps(value, indent=2, default=str))
