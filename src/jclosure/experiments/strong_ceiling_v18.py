"""Unified, capacity-matched state × action response hierarchy for V18."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn

from jclosure.experiments.crossed_bank_v18 import OUT, _split
from jclosure.experiments.state_features_v18 import FEATURES, _design
from jclosure.protocol_v18 import digest, stage_freeze, verify
from jclosure.provenance import sha256_file, write_json_atomic

SOURCE = "src/jclosure/experiments/strong_ceiling_v18.py"
MODEL_FREEZE = Path("artifacts/strong_state_context_ceiling_v18_models.freeze.json")
MODEL_BINDING_AMENDMENT = Path("artifacts/strong_state_context_ceiling_v18_model_binding_amendment_6.freeze.json")
TARGETS = ("j", "logits", "semantic_continuous", "workspace", "stacked_normalized")
SLICES = {"j": slice(0, 128), "logits": slice(128, 160),
          "semantic_continuous": slice(160, 192), "workspace": slice(192, 288),
          "stacked_normalized": slice(0, 288)}
MODEL_CLASSES = ("M0_additive_linear", "M1_bilinear", "M2_quadratic_tensor",
                 "M3_low_rank_interaction", "M4_small_state_conditioned_mlp")
CONTEXTS = ("action_only", "j", "j_rec", "j_conv", "j_kv", "j_rec_conv", "j_rec_kv",
            "j_conv_kv", "j_rec_conv_kv", "full_raw")


def prepare(root: Path) -> dict:
    split = _split(root)
    features = _design(root)
    config = verify(root)["config"]
    detail = {
        "split_freeze_digest": split["freeze_digest"],
        "feature_freeze_digest": features["freeze_digest"],
        "primary_h1_cohort": "all_2000_train_400_validation_primary_alpha_0.5_reliable_rows_only",
        "horizon_cohort": "same_frozen_400_train_80_validation_states_actions_alpha_0.5",
        "horizons": [1, 2, 4, 8], "contexts": list(CONTEXTS), "model_classes": list(MODEL_CLASSES),
        "feature_dimension": 128, "diagnostic_dimension": 256,
        "target": "V16_normalized_stack_288_components",
        "action_vector": "eight_shared_signed_alpha_coordinates",
        "M0": "intercept_plus_state_plus_signed_action",
        "M1": "M0_plus_state_tensor_signed_action",
        "M2": "M1_plus_action_squared_and_state_tensor_action_squared",
        "M3": "rank_32_factorized_state_action_plus_linear_mains",
        "M4": "two_hidden_layer_128_GELU_MLP_of_state_and_action",
        "ridge_lambda_grid": config["models"]["ridge_grid"],
        "ridge_selection": "train_only_state_grouped_20_percent_internal_holdout_on_J_M1_h1_shared_across_conditions",
        "mlp_weight_decay_grid": config["models"]["mlp_weight_decay_grid"],
        "mlp_selection": "train_only_state_grouped_internal_holdout_on_J_per_neural_model_shared_across_conditions",
        "mlp_epochs": config["models"]["mlp_epochs"], "mlp_batch_size": 2048,
        "mlp_learning_rate": 0.001, "mlp_seed": config["models"]["mlp_seed"],
        "primary_ceiling_model": "M4_small_state_conditioned_mlp_equal_128_feature_budget",
        "horizon_ceiling_model": "M4_same_architecture_and_training_budget",
        "family_gate": "nonnegative_gain_every_family",
        "bootstrap_unit": "state", "bootstrap_replicates": 1000, "bootstrap_seed": 1819,
        "validation_not_used_for_selection": True,
    }
    return stage_freeze(root, "models", [SOURCE, str(FEATURES),
                                         "artifacts/strong_state_context_ceiling_v18_splits.freeze.json",
                                         "artifacts/strong_state_context_ceiling_v18_features.freeze.json"], detail)


def _model_design(root: Path) -> dict:
    base = verify(root)
    value = json.loads((root / MODEL_FREEZE).read_text())
    if value["freeze_digest"] != digest(value) or value["base_freeze_digest"] != base["freeze_digest"]:
        raise RuntimeError("V18 model freeze invalid")
    for path, expected in value["input_hashes"].items():
        observed = sha256_file(root / path)
        if observed != expected:
            if path != SOURCE or not (root / MODEL_BINDING_AMENDMENT).exists():
                raise RuntimeError(f"V18 model frozen input changed: {path}")
            amendment = json.loads((root / MODEL_BINDING_AMENDMENT).read_text())
            if (amendment["freeze_digest"] != digest(amendment)
                    or amendment["base_freeze_digest"] != base["freeze_digest"]
                    or amendment["prior_source_sha256"] != expected
                    or amendment["input_hashes"].get(SOURCE) != observed
                    or amendment["input_hashes"].get(str(MODEL_FREEZE)) != sha256_file(root / MODEL_FREEZE)):
                raise RuntimeError("V18 model-binding amendment invalid")
    return value


def amend_model_binding(root: Path) -> dict:
    frozen = json.loads((root / MODEL_FREEZE).read_text())
    return stage_freeze(root, "model_binding_amendment_6",
                        [SOURCE, str(MODEL_FREEZE),
                         "results/v18/processed/strong_ceiling_state_errors_pre_model_binding_amendment_v18.parquet",
                         "results/v18/processed/strong_model_train_selection_pre_model_binding_amendment_v18.json"],
                        {"reason": "Frozen primary/horizon model strings are descriptive M4 labels, not MODEL_CLASSES keys; first h1 run completed all 50 fits but failed at summary lookup. Bind both summaries to the already-frozen M4_small_state_conditioned_mlp class without changing data, hyperparameters, fits or gates.",
                         "prior_source_sha256": frozen["input_hashes"][SOURCE],
                         "frozen_primary_description": frozen["primary_ceiling_model"],
                         "frozen_horizon_description": frozen["horizon_ceiling_model"],
                         "bound_model_class": MODEL_CLASSES[-1],
                         "prior_predictions_archived": True,
                         "validation_not_used_to_select_binding": True})


def _load_bank(root: Path, role: str) -> pd.DataFrame:
    parts = sorted((root / OUT).glob(f"crossed_response_{role}_*_v18.parquet"))
    if len(parts) != 5:
        raise RuntimeError(f"V18 {role} crossed bank not aggregated across five families")
    return pd.concat([pd.read_parquet(path) for path in parts], ignore_index=True)


def _data(root: Path, horizon: int, panel: bool, alpha: float) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, np.ndarray]]:
    with np.load(root / FEATURES, allow_pickle=False) as payload:
        scores = {name: payload[name] for name in payload.files}
    index = {key: i for i, key in enumerate(scores["base_trial_id"].astype(str))}
    role_frames = []
    for role in ("train", "validation"):
        frame = _load_bank(root, role)
        frame = frame[(frame.horizon == horizon) & (frame.horizon_panel == panel if panel else frame.alpha == alpha)]
        frame = frame[(frame.alpha == alpha) & (frame.reliability_status == "RELIABLE")].copy()
        frame["state_index"] = frame.base_trial_id.map(index).astype(int)
        frame = frame.sort_values(["base_trial_id", "coordinate_index", "sign"]).reset_index(drop=True)
        role_frames.append(frame)
    return role_frames[0], role_frames[1], scores


def _action(frame: pd.DataFrame, coordinates: list[int]) -> np.ndarray:
    index = {coordinate: i for i, coordinate in enumerate(coordinates)}
    a = np.zeros((len(frame), len(coordinates)), dtype=np.float32)
    for i, row in enumerate(frame.itertuples()):
        a[i, index[int(row.coordinate_index)]] = float(row.sign * row.alpha)
    return a


def _matrix(state: np.ndarray, action: np.ndarray, model: str) -> np.ndarray:
    ones = np.ones((len(state), 1), dtype=np.float32)
    pieces = [ones, state, action]
    if model in ("M1_bilinear", "M2_quadratic_tensor"):
        pieces.append((state[:, :, None] * action[:, None, :]).reshape(len(state), -1))
    if model == "M2_quadratic_tensor":
        squared = np.square(action)
        pieces.extend((squared, (state[:, :, None] * squared[:, None, :]).reshape(len(state), -1)))
    return np.concatenate(pieces, axis=1)


def _ridge(x: np.ndarray, y: np.ndarray, lam: float) -> np.ndarray:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    xt = torch.from_numpy(x).to(device)
    yt = torch.from_numpy(y).to(device)
    gram = (xt.T @ xt) / len(x)
    rhs = (xt.T @ yt) / len(x)
    gram.diagonal().add_(lam)
    weights = torch.linalg.solve(gram, rhs)
    return weights.cpu().numpy().astype(np.float32)


class _LowRank(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, output_dim: int) -> None:
        super().__init__()
        self.state = nn.Linear(state_dim, 32, bias=False)
        self.action = nn.Linear(action_dim, 32, bias=False)
        self.output = nn.Linear(state_dim + action_dim + 32, output_dim)

    def forward(self, x: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        product = self.state(x) * self.action(a)
        return self.output(torch.cat((x, a, product), dim=-1))


class _SmallMLP(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, output_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(state_dim + action_dim, 128), nn.GELU(),
                                 nn.Linear(128, 128), nn.GELU(), nn.Linear(128, output_dim))

    def forward(self, x: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat((x, a), dim=-1))


def _neural(model: str, x: np.ndarray, a: np.ndarray, y: np.ndarray,
            weight_decay: float, seed: int, epochs: int) -> nn.Module:
    torch.manual_seed(seed)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    network = (_LowRank(x.shape[1], a.shape[1], y.shape[1]) if model == "M3_low_rank_interaction"
               else _SmallMLP(x.shape[1], a.shape[1], y.shape[1])).to(device)
    optimizer = torch.optim.AdamW(network.parameters(), lr=0.001, weight_decay=weight_decay)
    inputs = torch.from_numpy(x).to(device)
    actions = torch.from_numpy(a).to(device)
    targets = torch.from_numpy(y).to(device)
    rng = np.random.default_rng(seed)
    network.train()
    for _ in range(epochs):
        order = rng.permutation(len(x))
        for start in range(0, len(x), 2048):
            idx = torch.as_tensor(order[start:start + 2048], device=device)
            output = network(inputs[idx], actions[idx])
            loss = (output - targets[idx]).square().mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    return network.eval()


@torch.no_grad()
def _neural_predict(network: nn.Module, x: np.ndarray, a: np.ndarray) -> np.ndarray:
    device = next(network.parameters()).device
    return network(torch.from_numpy(x).to(device), torch.from_numpy(a).to(device)).cpu().numpy().astype(np.float32)


def _summary(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    a, p = actual.astype(np.float64), predicted.astype(np.float64)
    an = np.linalg.norm(a, axis=1)
    pn = np.linalg.norm(p, axis=1)
    direction = (a * p).sum(axis=1) / np.maximum(an * pn, 1e-12)
    sign_mask = np.abs(a) > 1e-6
    return {"relative_l2": float(np.linalg.norm(a - p) / max(np.linalg.norm(a), 1e-12)),
            "direction_median": float(np.median(direction)),
            "magnitude_ratio_median": float(np.median(pn / np.maximum(an, 1e-12))),
            "component_sign_agreement": float((np.sign(a[sign_mask]) == np.sign(p[sign_mask])).mean()) if sign_mask.any() else math.nan,
            "n": len(a)}


def _metrics(frame: pd.DataFrame, actual: np.ndarray, predicted: np.ndarray) -> dict:
    result = {name: _summary(actual[:, region], predicted[:, region]) for name, region in SLICES.items()}
    result["by_family"] = {family: {name: _summary(actual[frame.family.to_numpy() == family, region],
                                                    predicted[frame.family.to_numpy() == family, region])
                                    for name, region in SLICES.items()}
                           for family in sorted(frame.family.unique())}
    return result


def _internal_mask(frame: pd.DataFrame) -> np.ndarray:
    return np.asarray([int(hashlib.sha256(f"v18-internal:{key}".encode()).hexdigest(), 16) % 5 == 0
                       for key in frame.base_trial_id], dtype=bool)


def _select_hyperparameters(root: Path, train: pd.DataFrame, scores: dict, coordinates: list[int], design: dict) -> dict:
    state = scores["j"][train.state_index.to_numpy(), :128].astype(np.float32)
    action = _action(train, coordinates)
    y = np.stack(train.response_stacked_normalized).astype(np.float32)
    hold = _internal_mask(train)
    fit = ~hold
    if not hold.any() or not fit.any():
        raise RuntimeError("V18 internal training state holdout empty")
    ridge_scores = []
    m1 = _matrix(state, action, "M1_bilinear")
    for lam in design["ridge_lambda_grid"]:
        weights = _ridge(m1[fit], y[fit], float(lam))
        score = _summary(y[hold], m1[hold] @ weights)["relative_l2"]
        ridge_scores.append({"lambda": float(lam), "train_internal_stack_rel_l2": score})
    selected = min(ridge_scores, key=lambda row: row["train_internal_stack_rel_l2"])["lambda"]
    neural = {}
    for model in MODEL_CLASSES[3:]:
        trials = []
        for wd in design["mlp_weight_decay_grid"]:
            network = _neural(model, state[fit], action[fit], y[fit], float(wd),
                              int(design["mlp_seed"]), int(design["mlp_epochs"]))
            score = _summary(y[hold], _neural_predict(network, state[hold], action[hold]))["relative_l2"]
            trials.append({"weight_decay": float(wd), "train_internal_stack_rel_l2": score})
            del network
        neural[model] = {"trials": trials, "selected_weight_decay": min(trials, key=lambda x: x["train_internal_stack_rel_l2"])["weight_decay"]}
    return {"ridge_trials": ridge_scores, "selected_ridge_lambda": selected, "neural": neural,
            "train_internal_holdout_state_count": int(train.loc[hold, "base_trial_id"].nunique())}


def _fit_predict(model: str, state_train: np.ndarray, action_train: np.ndarray,
                 y_train: np.ndarray, state_val: np.ndarray, action_val: np.ndarray,
                 hyper: dict, design: dict) -> tuple[np.ndarray, int]:
    if model in MODEL_CLASSES[:3]:
        xtr = _matrix(state_train, action_train, model)
        xva = _matrix(state_val, action_val, model)
        weight = _ridge(xtr, y_train, hyper["selected_ridge_lambda"])
        return xva @ weight, int(weight.size)
    network = _neural(model, state_train, action_train, y_train,
                      hyper["neural"][model]["selected_weight_decay"],
                      int(design["mlp_seed"]), int(design["mlp_epochs"]))
    prediction = _neural_predict(network, state_val, action_val)
    parameters = sum(item.numel() for item in network.parameters())
    del network
    return prediction, int(parameters)


def _state_squares(frame: pd.DataFrame, actual: np.ndarray, predicted: np.ndarray,
                   model: str, context: str, horizon: int, cohort: str) -> pd.DataFrame:
    rows = []
    for state_id, idx in frame.groupby("base_trial_id", sort=True).indices.items():
        part = frame.iloc[idx]
        for target, region in SLICES.items():
            a, p = actual[idx, region], predicted[idx, region]
            rows.append({"base_trial_id": state_id, "family": str(part.family.iloc[0]),
                         "cohort": cohort, "horizon": horizon, "model": model, "context": context,
                         "target": target, "actual_squared_norm": float((a * a).sum()),
                         "error_squared_norm": float(((a - p) ** 2).sum()),
                         "action_rows": len(part)})
    return pd.DataFrame(rows)


def _evaluate_cohort(root: Path, horizon: int, panel: bool, model_names: tuple[str, ...],
                     contexts: tuple[str, ...], hyper: dict, design: dict) -> tuple[list[dict], pd.DataFrame]:
    train, val, scores = _data(root, horizon, panel, 0.5)
    split = _split(root)
    coordinates = [x["coordinate_index"] for x in split["actions"]]
    atr, ava = _action(train, coordinates), _action(val, coordinates)
    ytr = np.stack(train.response_stacked_normalized).astype(np.float32)
    yva = np.stack(val.response_stacked_normalized).astype(np.float32)
    cohort = "horizon_panel" if panel else "full_h1"
    metrics = []
    square_rows = []
    for context in contexts:
        state_train = (np.zeros((len(train), 128), dtype=np.float32) if context == "action_only"
                       else scores[context][train.state_index.to_numpy(), :128].astype(np.float32))
        state_val = (np.zeros((len(val), 128), dtype=np.float32) if context == "action_only"
                     else scores[context][val.state_index.to_numpy(), :128].astype(np.float32))
        for model in model_names:
            prediction, parameter_count = _fit_predict(model, state_train, atr, ytr, state_val, ava, hyper, design)
            record = {"cohort": cohort, "horizon": horizon, "context": context,
                      "model": model, "parameter_count": parameter_count,
                      "train_action_rows": len(train), "validation_action_rows": len(val),
                      "train_state_count": int(train.base_trial_id.nunique()),
                      "validation_state_count": int(val.base_trial_id.nunique()),
                      "metrics": _metrics(val, yva, prediction)}
            metrics.append(record)
            square_rows.append(_state_squares(val, yva, prediction, model, context, horizon, cohort))
            print(f"V18 {cohort} h{horizon} {context} {model} stack_rel_l2={record['metrics']['stacked_normalized']['relative_l2']:.4f}", flush=True)
    return metrics, pd.concat(square_rows, ignore_index=True)


def _bootstrap_gain(squares: pd.DataFrame, cohort: str, horizon: int, model: str,
                    left: str, right: str, target: str = "stacked_normalized") -> dict:
    subset = squares[(squares.cohort == cohort) & (squares.horizon == horizon) &
                     (squares.model == model) & (squares.target == target)]
    a = subset[subset.context == left].sort_values("base_trial_id").reset_index(drop=True)
    b = subset[subset.context == right].sort_values("base_trial_id").reset_index(drop=True)
    if not a.base_trial_id.equals(b.base_trial_id):
        raise RuntimeError("V18 paired bootstrap state alignment mismatch")
    val = np.stack((a.actual_squared_norm.to_numpy(), a.error_squared_norm.to_numpy(), b.error_squared_norm.to_numpy()), axis=1)
    rng = np.random.default_rng(1819 + horizon)
    chosen = rng.integers(0, len(a), size=(1000, len(a)))
    totals = val[chosen].sum(axis=1)
    gain = (np.sqrt(totals[:, 1]) - np.sqrt(totals[:, 2])) / np.sqrt(np.maximum(totals[:, 0], 1e-24))
    return {"gain_ci95": np.quantile(gain, [0.025, 0.975]).tolist(),
            "gain_median": float(np.median(gain)), "state_count": len(a)}


def run_h1(root: Path) -> dict:
    design = _model_design(root)
    train, _, scores = _data(root, 1, False, 0.5)
    split = _split(root)
    coordinates = [x["coordinate_index"] for x in split["actions"]]
    hyper = _select_hyperparameters(root, train, scores, coordinates, design)
    write_json_atomic(root / OUT / "strong_model_train_selection_v18.json", hyper)
    metrics, squares = _evaluate_cohort(root, 1, False, MODEL_CLASSES, CONTEXTS, hyper, design)
    squares.to_parquet(root / OUT / "strong_ceiling_state_errors_v18.parquet", index=False, compression="zstd")
    lookup = {(x["model"], x["context"]): x for x in metrics}
    if not str(design["primary_ceiling_model"]).startswith("M4"):
        raise RuntimeError("V18 frozen primary model is not M4")
    model = MODEL_CLASSES[-1]
    j = lookup[(model, "j")]["metrics"]
    raw = lookup[(model, "j_rec_conv_kv")]["metrics"]
    gain = {target: j[target]["relative_l2"] - raw[target]["relative_l2"] for target in TARGETS}
    family_gain = {family: j["by_family"][family]["stacked_normalized"]["relative_l2"] -
                   raw["by_family"][family]["stacked_normalized"]["relative_l2"]
                   for family in j["by_family"]}
    bootstrap = {target: _bootstrap_gain(squares, "full_h1", 1, model, "j", "j_rec_conv_kv", target)
                 for target in ("j", "stacked_normalized")}
    threshold = verify(root)["config"]["gates"]["material_raw_context_absolute_rel_l2_gain_min"]
    passed = (gain["j"] >= threshold and gain["stacked_normalized"] >= threshold and
              all(family_gain[name] >= 0 for name in family_gain) and
              all(bootstrap[target]["gain_ci95"][0] > 0 for target in bootstrap))
    result = {"model_freeze_digest": design["freeze_digest"], "hyperparameters_train_only": hyper,
              "model_results": metrics, "primary_model": model,
              "primary_joint_raw_gain_over_j": gain, "primary_family_stack_gain": family_gain,
              "primary_bootstrap": bootstrap, "material_h1_raw_context_gate_passed": passed,
              "validation_not_model_selection": True}
    write_json_atomic(root / OUT / "strong_h1_ceiling_v18.json", result)
    pd.DataFrame([{**{k: x[k] for k in ("cohort", "horizon", "context", "model", "parameter_count",
                                         "train_action_rows", "validation_action_rows")},
                   "j_relative_l2": x["metrics"]["j"]["relative_l2"],
                   "stack_relative_l2": x["metrics"]["stacked_normalized"]["relative_l2"]}
                  for x in metrics]).to_parquet(root / OUT / "strong_h1_model_matrix_v18.parquet", index=False)
    return {"material_gate": passed, "primary_joint_gain": gain,
            "model_freeze_digest": design["freeze_digest"]}


def run_horizons(root: Path) -> dict:
    design = _model_design(root)
    hyper = json.loads((root / OUT / "strong_model_train_selection_v18.json").read_text())
    if not str(design["horizon_ceiling_model"]).startswith("M4"):
        raise RuntimeError("V18 frozen horizon model is not M4")
    model = MODEL_CLASSES[-1]
    contexts = CONTEXTS[1:]
    all_metrics = []
    all_squares = []
    for horizon in (1, 2, 4, 8):
        metrics, squares = _evaluate_cohort(root, horizon, True, (model,), contexts, hyper, design)
        all_metrics.extend(metrics)
        all_squares.append(squares)
    square_frame = pd.concat(all_squares, ignore_index=True)
    square_frame.to_parquet(root / OUT / "horizon_context_state_errors_v18.parquet", index=False, compression="zstd")
    lookup = {(x["horizon"], x["context"]): x for x in all_metrics}
    curves = []
    threshold = verify(root)["config"]["gates"]["material_raw_context_absolute_rel_l2_gain_min"]
    for horizon in (1, 2, 4, 8):
        j = lookup[(horizon, "j")]["metrics"]
        gains = {}
        for context in contexts[1:]:
            row = lookup[(horizon, context)]["metrics"]
            gains[context] = {target: j[target]["relative_l2"] - row[target]["relative_l2"] for target in TARGETS}
        raw_gain = gains["j_rec_conv_kv"]
        bootstrap = {target: _bootstrap_gain(square_frame, "horizon_panel", horizon, model, "j", "j_rec_conv_kv", target)
                     for target in ("j", "stacked_normalized")}
        family = {f: j["by_family"][f]["stacked_normalized"]["relative_l2"] -
                  lookup[(horizon, "j_rec_conv_kv")]["metrics"]["by_family"][f]["stacked_normalized"]["relative_l2"]
                  for f in j["by_family"]}
        passed = (raw_gain["j"] >= threshold and raw_gain["stacked_normalized"] >= threshold and
                  all(value >= 0 for value in family.values()) and
                  all(bootstrap[t]["gain_ci95"][0] > 0 for t in bootstrap))
        curves.append({"horizon": horizon, "gains_over_j": gains, "raw_joint_gain": raw_gain,
                       "family_stack_gain": family, "bootstrap": bootstrap, "material_gate_passed": passed})
    result = {"model_freeze_digest": design["freeze_digest"], "model": model,
              "horizons": curves, "model_results": all_metrics,
              "same_state_action_panel_across_horizons": True,
              "earliest_material_horizon": next((row["horizon"] for row in curves if row["material_gate_passed"]), None)}
    write_json_atomic(root / OUT / "horizon_context_localization_v18.json", result)
    pd.DataFrame([{"horizon": row["horizon"], "context": context,
                   "j_gain": gain["j"], "stack_gain": gain["stacked_normalized"]}
                  for row in curves for context, gain in row["gains_over_j"].items()]).to_parquet(
                      root / OUT / "channel_horizon_gain_v18.parquet", index=False)
    return {"earliest_material_horizon": result["earliest_material_horizon"],
            "raw_gain_curve": [(x["horizon"], x["raw_joint_gain"]) for x in curves]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("prepare", "amend_model_binding", "h1", "horizons"))
    args = parser.parse_args()
    root = Path.cwd()
    result = {"prepare": prepare, "amend_model_binding": amend_model_binding,
              "h1": run_h1, "horizons": run_horizons}[args.stage](root)
    print(json.dumps(result, indent=2))
