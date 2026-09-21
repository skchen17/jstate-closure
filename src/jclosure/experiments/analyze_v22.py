"""V22 scaling, coordinates, coverage, even/odd, and local-operator analyses."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr, spearmanr

from jclosure.experiments import action_pool_v22 as action_bank
from jclosure.protocol_v22 import verify, verify_stage
from jclosure.provenance import sha256_file, write_json_atomic

SOURCE = "src/jclosure/experiments/analyze_v22.py"
OUT = Path("results/v22/processed")
RAW = Path("/data/CSK/J-space-project/v22-action-manifold-work/expanded_response_bank")
STATE = Path("/data/CSK/J-space-project/v21-action-geometry-work/state_representations_v21.npz")


def _load(root: Path) -> dict:
    verify_stage(root, "expanded_bank_design")
    design = json.loads((root / OUT / "expanded_bank_design_v22.json").read_text())
    selection = json.loads((root / OUT / "action_selection_v22.json").read_text())
    action_ids = selection["common_measured_train_action_ids"] + selection["validation_action_ids"]
    answer = {"action_ids": action_ids, "selection": selection}
    for role in ("development", "validation"):
        plus, minus, ids, families = [], [], [], []
        for item in design[f"{role}_states"]:
            with np.load(root / RAW / role / f"expanded_{item['base_trial_id']}.npz") as source:
                for state in ("P0", "Pq"):
                    plus.append(np.asarray(source[f"{state}_plus"], dtype=np.float64))
                    minus.append(np.asarray(source[f"{state}_minus"], dtype=np.float64))
                    ids.append(f"{item['base_trial_id']}::{state if state == 'P0' else item['q_name']}")
                    families.append(item["family"])
        answer[f"{role}_plus"] = np.stack(plus)
        answer[f"{role}_minus"] = np.stack(minus)
        answer[f"{role}_ids"] = np.asarray(ids)
        answer[f"{role}_families"] = np.asarray(families)
    with np.load(root / STATE) as source:
        for role, source_role in (("development", "train"), ("validation", "validation")):
            lookup = {str(value): index for index, value in enumerate(source[f"{source_role}_ids"])}
            positions = [lookup[str(value)] for value in answer[f"{role}_ids"]]
            answer[f"{role}_S1"] = np.asarray(source[f"S1_{source_role}"][positions], dtype=np.float64)
            answer[f"{role}_S2"] = np.asarray(source[f"S2_{source_role}"][positions], dtype=np.float64)
    return answer


def _action_features(root: Path, data: dict) -> dict:
    pool = json.loads((root / OUT / "action_pool_v22.json").read_text())
    specs = {item["action_id"]: item for item in pool["candidate_pool"]}
    directions = torch.load(root / "artifacts/causal/v13/probe_directions_v13.pt", map_location="cpu", weights_only=False)
    score = np.asarray(directions["score_directions"], dtype=np.float64)
    del directions
    x = np.stack([action_bank._score(specs[value], score) * data["selection"]["action_alphas"][value]
                  for value in data["action_ids"]])
    train = x[:128]
    median = max(float(np.median(np.linalg.norm(train, axis=1))), 1e-12)
    z1 = x / median
    blocks = x.reshape(len(x), 3, 4799)
    scales = np.median(np.linalg.norm(blocks[:128], axis=2), axis=0)
    z2 = (blocks / np.maximum(scales[None, :, None], 1e-12)).reshape(len(x), -1) / np.sqrt(3.0)
    with np.load(root / action_bank.SCRATCH / "action_coordinates_v22.npz") as source:
        probe_ids = np.asarray(source["probe_direction_indices"], dtype=int)
        response_covariance = np.asarray(source["response_covariance"], dtype=np.float64)
    probe = np.stack([score[value] for value in probe_ids])
    block_scale = np.asarray(data["selection"]["raw_block_scales"], dtype=np.float64)
    probe = (probe.reshape(len(probe), 3, 4799) / np.maximum(block_scale[None, :, None], 1e-12)).reshape(len(probe), -1)
    xwhite = (x.reshape(len(x), 3, 4799) / np.maximum(block_scale[None, :, None], 1e-12)).reshape(len(x), -1)
    kpp, kxp = probe @ probe.T, xwhite @ probe.T
    ridge = max(float(np.trace(kpp) / len(kpp)) * 1e-6, 1e-8)
    eigen, vectors = np.linalg.eigh(response_covariance)
    keep = eigen > max(float(eigen[-1]) * 1e-8, 1e-12)
    root_metric = vectors[:, keep] * np.sqrt(np.maximum(eigen[keep], 0))[None, :]
    z6 = kxp @ np.linalg.solve(kpp + ridge * np.eye(len(kpp)), root_metric)
    z6 /= max(float(np.median(np.linalg.norm(z6[:128], axis=1))), 1e-12)
    # Z7 is train-only finite-supervised: raw coordinate -> mean finite odd response.
    mean_odd = ((data["development_plus"] - data["development_minus"]) / 2.0).mean(axis=0)[:128]
    kernel = z2[:128] @ z2[:128].T
    cross = z2 @ z2[:128].T
    prediction = cross @ np.linalg.solve(kernel + 1e-3 * np.eye(128), mean_odd)
    _, _, vh = np.linalg.svd(prediction[:128], full_matrices=False)
    z7 = prediction @ vh[:64].T
    z7 /= max(float(np.median(np.linalg.norm(z7[:128], axis=1))), 1e-12)
    return {"Z1": z1, "Z2": z2, "Z6": z6, "Z7": z7}


def _kernel(features: np.ndarray, kind: str = "quadratic", dimension: int | None = None) -> np.ndarray:
    x = features
    if dimension is not None and x.shape[1] > dimension:
        _, _, vh = np.linalg.svd(x[:128], full_matrices=False)
        x = x @ vh[:dimension].T
    gram = x @ x.T
    diagonal = max(float(np.median(np.diag(gram[:128, :128]))), 1e-12)
    gram /= diagonal
    if kind == "linear": return 1.0 + gram
    if kind == "quadratic": return (1.0 + gram) ** 2
    if kind == "cubic": return (1.0 + gram) ** 3
    if kind == "rbf":
        diag = np.diag(gram)
        distance = np.maximum(diag[:, None] + diag[None, :] - 2 * gram, 0.0)
        positive = distance[:128, :128][distance[:128, :128] > 0]
        bandwidth = max(float(np.median(positive)), 1e-8)
        return np.exp(-distance / bandwidth)
    raise RuntimeError(kind)


def _state_kernel(train: np.ndarray, evaluation: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = train.mean(axis=0)
    left, right = train - mean, evaluation - mean
    scale = max(float(np.mean(np.sum(left * left, axis=1))) ** 0.5, 1e-12)
    left, right = left / scale, right / scale
    return 1.0 + left @ left.T, 1.0 + right @ left.T


def _predict(ks: np.ndarray, kn: np.ndarray, ka: np.ndarray, cross: np.ndarray,
             targets: np.ndarray, ridge: float) -> np.ndarray:
    es, us = np.linalg.eigh(ks)
    ea, ua = np.linalg.eigh(ka)
    transformed = np.einsum("ni,nad,am->imd", us, targets, ua, optimize=True)
    transformed /= (np.maximum(es, 0)[:, None, None] * np.maximum(ea, 0)[None, :, None] + ridge)
    return np.einsum("in,nmd,mj->ijd", kn @ us, transformed, ua.T @ cross, optimize=True)


def _metric(truth: np.ndarray, predicted: np.ndarray, families: np.ndarray) -> dict:
    flat_t, flat_p = truth.reshape(-1, truth.shape[-1]), predicted.reshape(-1, predicted.shape[-1])
    tn, pn = np.linalg.norm(flat_t, axis=1), np.linalg.norm(flat_p, axis=1)
    cosine = np.sum(flat_t * flat_p, axis=1) / np.maximum(tn * pn, 1e-12)
    relative = float(np.linalg.norm(flat_t - flat_p) / max(np.linalg.norm(flat_t), 1e-12))
    by_family = {}
    repeated = np.repeat(families, truth.shape[1])
    for family in sorted(set(families)):
        mask = repeated == family
        by_family[str(family)] = float(np.linalg.norm(flat_t[mask] - flat_p[mask]) / max(np.linalg.norm(flat_t[mask]), 1e-12))
    return {"relative_l2": relative, "direction_cosine_median": float(np.median(cosine)),
            "norm_ratio_median": float(np.median(pn / np.maximum(tn, 1e-12))),
            "family_relative_l2": by_family}


def _fit_eval(data: dict, kernel: np.ndarray, train_indices: list[int], sign: int = 1,
              target_kind: str = "plus", targets_override: np.ndarray | None = None,
              ridge: float = 0.01) -> tuple[dict, np.ndarray]:
    ks, kn = _state_kernel(data["development_S1"], data["validation_S1"])
    fit = np.asarray(train_indices, dtype=int)
    evaluation = np.arange(128, 160)
    ka = kernel[np.ix_(fit, fit)]
    cross = kernel[np.ix_(fit, evaluation)]
    targets = data["development_plus"][:, fit] if targets_override is None else targets_override[:, fit]
    truth = data[f"validation_{target_kind}"][:, evaluation]
    prediction = _predict(ks, kn, ka, cross, targets, ridge)
    if sign < 0:
        # Frozen odd-symmetry baseline. Explicit even/odd is evaluated separately.
        prediction = -prediction
    return _metric(truth, prediction, data["validation_families"]), prediction


def _local_action_ceiling(data: dict, kernel: np.ndarray, train_indices: list[int],
                          ridge: float = 0.01) -> tuple[dict, np.ndarray]:
    """Same-state action interpolation using measured train responses as S2-like context."""
    fit = np.asarray(train_indices, dtype=int)
    evaluation = np.arange(128, 160)
    ka = kernel[np.ix_(fit, fit)]
    cross = kernel[np.ix_(fit, evaluation)]
    weights = np.linalg.solve(ka + ridge * np.eye(len(fit)), data["validation_plus"][:, fit, :])
    prediction = np.einsum("ia,sir->sar", cross, weights)
    truth = data["validation_plus"][:, evaluation]
    return _metric(truth, prediction, data["validation_families"]), prediction


def _select_action_ridge(data: dict, kernel: np.ndarray, train_indices: list[int]) -> tuple[float, list[dict]]:
    indices = np.asarray(train_indices, dtype=int)
    split = max(8, int(0.75 * len(indices)))
    if split >= len(indices):
        split = len(indices) - 1
    fit, heldout = indices[:split], indices[split:]
    ka = kernel[np.ix_(fit, fit)]
    cross = kernel[np.ix_(fit, heldout)]
    truth = data["development_plus"][:, heldout]
    rows = []
    for ridge in (0.001, 0.01, 0.1, 1.0, 10.0, 100.0):
        weights = np.linalg.solve(ka + ridge * np.eye(len(fit)), data["development_plus"][:, fit, :])
        prediction = np.einsum("ia,sir->sar", cross, weights)
        relative = float(np.linalg.norm(truth - prediction) / max(np.linalg.norm(truth), 1e-12))
        rows.append({"ridge": ridge, "internal_action_holdout_relative_l2": relative})
    best = min(rows, key=lambda row: row["internal_action_holdout_relative_l2"])
    return float(best["ridge"]), rows


def _coverage(train: np.ndarray, heldout: np.ndarray) -> dict:
    norms = np.maximum(np.linalg.norm(train, axis=1), 1e-12)
    _, singular, vh = np.linalg.svd(train, full_matrices=False)
    rank = int((singular > max(float(singular[0]) * 1e-8, 1e-12)).sum())
    basis = vh[:rank]
    rows = []
    gram = train @ train.T
    inverse = np.linalg.pinv(gram + 1e-6 * max(float(np.trace(gram) / len(gram)), 1e-12) * np.eye(len(gram)))
    for value in heldout:
        norm = max(float(np.linalg.norm(value)), 1e-12)
        cosine = train @ value / (norms * norm)
        projection = value @ basis.T @ basis
        residual = float(np.linalg.norm(value - projection) / norm)
        cross = train @ value
        leverage = float(cross @ inverse @ cross / max(norm * norm, 1e-12))
        rows.append((float(np.max(np.abs(cosine))), residual, leverage))
    array = np.asarray(rows)
    return {"nearest_abs_cosine_median": float(np.median(array[:, 0])),
            "span_residual_median": float(np.median(array[:, 1])),
            "leverage_median": float(np.median(array[:, 2])), "per_action": array}


def run(root: Path) -> dict:
    data = _load(root)
    features = _action_features(root, data)
    selection = data["selection"]
    counts = selection["nested_counts"]
    scaling_rows, coverage_rows = [], []
    z1_kernel = _kernel(features["Z1"], "quadratic")
    for strategy, order in selection["strategy_rankings"].items():
        positions = {value: index for index, value in enumerate(data["action_ids"][:128])}
        for count in counts:
            fit = [positions[value] for value in order[:int(count)]]
            ridge, ridge_curve = _select_action_ridge(data, z1_kernel, fit)
            metric, prediction = _fit_eval(data, z1_kernel, fit, ridge=ridge)
            local_metric, local_prediction = _local_action_ceiling(data, z1_kernel, fit, ridge=ridge)
            entry = {"strategy": strategy, "action_count": int(count), **metric,
                     "local_S2_action_ceiling": local_metric, "selected_ridge": ridge,
                     "internal_ridge_curve": ridge_curve}
            scaling_rows.append(entry)
            for name in ("Z1", "Z2", "Z6"):
                cov = _coverage(features[name][fit], features[name][128:160])
                coverage_rows.append({"strategy": strategy, "action_count": int(count), "metric": name,
                                      **{key: value for key, value in cov.items() if key != "per_action"}})
    best_scaling = min(scaling_rows, key=lambda row: row["relative_l2"])
    improvements = {}
    for strategy in selection["selection_strategies"]:
        rows = [row for row in scaling_rows if row["strategy"] == strategy]
        improvements[strategy] = [rows[index - 1]["local_S2_action_ceiling"]["relative_l2"] -
                                  rows[index]["local_S2_action_ceiling"]["relative_l2"]
                                  for index in range(1, len(rows))]
    saturation = {strategy: bool(len(values) >= 2 and abs(values[-1]) < 0.02 and abs(values[-2]) < 0.02)
                  for strategy, values in improvements.items()}

    coordinate_rows = []
    coordinate_specs = [("Z1", "quadratic", None), ("Z2", "quadratic", None)]
    for dimension in (4, 8, 16, 32, 64, 128):
        coordinate_specs += [("Z6", "quadratic", dimension), ("Z7", "quadratic", dimension)]
    coordinate_specs += [("Z8_poly2", "quadratic", 64), ("Z8_poly3", "cubic", 64), ("Z8_rff", "rbf", 64)]
    for name, kind, dimension in coordinate_specs:
        source = "Z2" if name.startswith("Z8") else name
        kernel = _kernel(features[source], kind, dimension)
        ridge, ridge_curve = _select_action_ridge(data, kernel, list(range(128)))
        plus, plus_prediction = _fit_eval(data, kernel, list(range(128)), ridge=ridge)
        minus, minus_prediction = _fit_eval(data, kernel, list(range(128)), sign=-1, target_kind="minus", ridge=ridge)
        local, _ = _local_action_ceiling(data, kernel, list(range(128)), ridge=ridge)
        coordinate_rows.append({"coordinate": name, "dimension": dimension,
                                "unseen_direction": plus, "unseen_direction_and_sign": minus,
                                "local_S2_action_ceiling": local, "selected_ridge": ridge,
                                "internal_ridge_curve": ridge_curve})
    best_coordinate = min(coordinate_rows, key=lambda row: row["unseen_direction"]["relative_l2"])
    best_source = "Z2" if best_coordinate["coordinate"].startswith("Z8") else best_coordinate["coordinate"]
    best_kernel = _kernel(features[best_source],
                          "cubic" if best_coordinate["coordinate"] == "Z8_poly3" else
                          "rbf" if best_coordinate["coordinate"] == "Z8_rff" else "quadratic",
                          best_coordinate["dimension"])
    best_ridge = float(best_coordinate["selected_ridge"])

    odd_train = (data["development_plus"] - data["development_minus"]) / 2.0
    even_train = (data["development_plus"] + data["development_minus"]) / 2.0
    odd_val = (data["validation_plus"] - data["validation_minus"]) / 2.0
    even_val = (data["validation_plus"] + data["validation_minus"]) / 2.0
    ks, kn = _state_kernel(data["development_S1"], data["validation_S1"])
    ka = best_kernel[:128, :128]
    cross = best_kernel[:128, 128:160]
    odd_prediction = _predict(ks, kn, ka, cross, odd_train[:, :128], best_ridge)
    even_prediction = _predict(ks, kn, ka, cross, even_train[:, :128], best_ridge)
    plus_reconstruction = even_prediction + odd_prediction
    minus_reconstruction = even_prediction - odd_prediction
    even_odd = {
        "odd_component": _metric(odd_val[:, 128:160], odd_prediction, data["validation_families"]),
        "even_component": _metric(even_val[:, 128:160], even_prediction, data["validation_families"]),
        "reconstructed_plus": _metric(data["validation_plus"][:, 128:160], plus_reconstruction, data["validation_families"]),
        "reconstructed_minus": _metric(data["validation_minus"][:, 128:160], minus_reconstruction, data["validation_families"]),
        "baseline_minus": best_coordinate["unseen_direction_and_sign"],
    }

    # Coverage-error association for the frozen best practical development candidate.
    per_error = np.linalg.norm(data["validation_plus"][:, 128:160] - plus_reconstruction, axis=(0, 2)) / np.maximum(
        np.linalg.norm(data["validation_plus"][:, 128:160], axis=(0, 2)), 1e-12)
    associations = {}
    for name in ("Z1", "Z2", "Z6"):
        cov = _coverage(features[name][:128], features[name][128:160])["per_action"]
        associations[name] = {}
        for column, label in ((0, "nearest_abs_cosine"), (1, "span_residual"), (2, "leverage")):
            associations[name][label] = {"spearman": float(spearmanr(cov[:, column], per_error).statistic),
                                         "pearson": float(pearsonr(cov[:, column], per_error).statistic)}

    # State-conditioned local operator chart: S1 predicts a low-rank action->odd-response operator.
    z = features["Z6"]
    _, _, vh = np.linalg.svd(z[:128], full_matrices=False)
    z16 = z @ vh[:16].T
    zfit = z16[:128]
    inverse = np.linalg.inv(zfit.T @ zfit + 0.01 * np.eye(16)) @ zfit.T
    operators = np.stack([inverse @ odd_train[state, :128] for state in range(len(odd_train))])
    context = data["development_S1"]
    mean = context.mean(axis=0)
    _, _, context_vh = np.linalg.svd(context - mean, full_matrices=False)
    ctrain = (context - mean) @ context_vh[:8].T
    cval = (data["validation_S1"] - mean) @ context_vh[:8].T
    design_train = np.column_stack([np.ones(len(ctrain)), ctrain])
    design_val = np.column_stack([np.ones(len(cval)), cval])
    weights = np.linalg.solve(design_train.T @ design_train + 0.1 * np.eye(9),
                              design_train.T @ operators.reshape(len(operators), -1))
    predicted_operators = (design_val @ weights).reshape(len(cval), 16, 288)
    chart_prediction = np.einsum("ad,ndr->nar", z16[128:160], predicted_operators)
    global_prediction = np.einsum("ad,dr->ar", z16[128:160], operators.mean(axis=0))[None, :, :]
    global_prediction = np.repeat(global_prediction, len(cval), axis=0)
    chart = {"global_fixed_coordinate": _metric(odd_val[:, 128:160], global_prediction, data["validation_families"]),
             "state_conditioned_local_chart": _metric(odd_val[:, 128:160], chart_prediction, data["validation_families"]),
             "action_dimension": 16, "state_condition_dimension": 8,
             "eligible_from_V22_G2": True}

    low_rank = []
    for rank in (4, 8, 16, 32):
        # Truncate each predicted action-output operator; rank cannot exceed action coordinate width.
        predictions = []
        for operator in predicted_operators:
            u, singular, vh_operator = np.linalg.svd(operator, full_matrices=False)
            use = min(rank, len(singular))
            truncated = (u[:, :use] * singular[:use]) @ vh_operator[:use]
            predictions.append(z16[128:160] @ truncated)
        metric = _metric(odd_val[:, 128:160], np.stack(predictions), data["validation_families"])
        low_rank.append({"rank": rank, **metric})

    thresholds = verify(root)["config"]["gates"]
    finalist = even_odd["reconstructed_plus"]
    finalist_minus = even_odd["reconstructed_minus"]
    practical = (finalist["relative_l2"] <= thresholds["relative_l2_max"] and
                 finalist_minus["relative_l2"] <= thresholds["relative_l2_max"] and
                 finalist["direction_cosine_median"] >= thresholds["cosine_min"] and
                 finalist_minus["direction_cosine_median"] >= thresholds["cosine_min"] and
                 thresholds["norm_ratio_min"] <= finalist["norm_ratio_median"] <= thresholds["norm_ratio_max"] and
                 thresholds["norm_ratio_min"] <= finalist_minus["norm_ratio_median"] <= thresholds["norm_ratio_max"] and
                 max(finalist["family_relative_l2"].values()) <= thresholds["family_relative_l2_max"] and
                 max(finalist_minus["family_relative_l2"].values()) <= thresholds["family_relative_l2_max"])

    outputs = {
        "action_data_scaling_v22.json": {"rows": scaling_rows, "step_improvements": improvements,
                                         "saturation_by_strategy": saturation, "best": best_scaling},
        "causal_action_coordinates_v22.json": {"rows": coordinate_rows, "best": best_coordinate},
        "even_odd_action_geometry_v22.json": even_odd,
        "causal_coverage_error_v22.json": {"coverage_rows": coverage_rows, "associations": associations,
                                           "per_validation_action_error": per_error.tolist()},
        "state_conditioned_action_chart_v22.json": chart,
        "low_rank_operator_identification_v22.json": {"rows": low_rank, "best": min(low_rank, key=lambda row: row["relative_l2"])},
    }
    for name, payload in outputs.items():
        write_json_atomic(root / OUT / name, payload)
    pd.DataFrame(scaling_rows).to_parquet(root / OUT / "action_data_scaling_v22.parquet", index=False)
    adjudication = {
        "best_scaling": best_scaling, "best_coordinate": best_coordinate,
        "explicit_even_odd": even_odd, "state_conditioned_chart": chart,
        "practical_direction_sign_gate": bool(practical),
        "saturation_all_strategies": bool(all(saturation.values())),
        "action_data_limited": bool(not all(saturation.values())),
        "compact_operator_search_reopened": bool(practical),
        "historical_final_six_opened": False, "new_independent_final_opened": False,
    }
    write_json_atomic(root / OUT / "v22_development_adjudication.json", adjudication)
    return adjudication


if __name__ == "__main__":
    print(json.dumps(run(Path.cwd()), indent=2))
