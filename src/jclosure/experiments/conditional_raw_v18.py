"""Cross-fitted V18 fixed-base raw residual correction, only after raw ceiling gate."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from jclosure.experiments.crossed_bank_v18 import OUT, _split
from jclosure.experiments.state_features_v18 import FEATURES
from jclosure.experiments.strong_ceiling_v18 import (
    _action, _data, _model_design, _neural, _neural_predict, _summary,
)
from jclosure.protocol_v18 import digest, stage_freeze, verify
from jclosure.provenance import sha256_file, write_json_atomic

SOURCE = "src/jclosure/experiments/conditional_raw_v18.py"
FREEZE = Path("artifacts/strong_state_context_ceiling_v18_conditional_raw.freeze.json")


def prepare(root: Path) -> dict:
    model = _model_design(root)
    split = _split(root)
    return stage_freeze(root, "conditional_raw", [SOURCE, str(FEATURES),
                                                   "artifacts/strong_state_context_ceiling_v18_models.freeze.json"],
                        {"model_freeze_digest": model["freeze_digest"], "split_freeze_digest": split["freeze_digest"],
                         "base_model": "M4_J_action_frozen_training_hyperparameters",
                         "crossfit": "five_state_grouped_training_folds_no_validation_fit",
                         "raw_residualization": "ridge_raw_128D_on_J_128D_train_OOF",
                         "target_residualization": "five_state_grouped_M4_OOF",
                         "correction": "linear_raw_residual_and_raw_residual_tensor_signed_action_ridge",
                         "ridge": "frozen_train_selected_strong_ceiling_lambda",
                         "base_refit_after_raw_correction": False,
                         "horizons": [1, 2, 4, 8], "run_only_if_material_raw_ceiling": True,
                         "bootstrap_unit": "state", "bootstrap_replicates": 1000, "seed": 1821})


def _freeze(root: Path) -> dict:
    base = verify(root)
    value = json.loads((root / FREEZE).read_text())
    if value["freeze_digest"] != digest(value) or value["base_freeze_digest"] != base["freeze_digest"]:
        raise RuntimeError("V18 conditional raw freeze invalid")
    for path, expected in value["input_hashes"].items():
        if sha256_file(root / path) != expected:
            raise RuntimeError(f"V18 conditional frozen input changed: {path}")
    return value


def _state_folds(ids: np.ndarray) -> np.ndarray:
    return np.asarray([int(hashlib.sha256(f"v18-cond:{key}".encode()).hexdigest(), 16) % 5 for key in ids])


def _residualized_raw(raw_train: np.ndarray, j_train: np.ndarray, raw_val: np.ndarray,
                      j_val: np.ndarray, folds: np.ndarray, ridge: float) -> tuple[np.ndarray, np.ndarray]:
    oof = np.zeros_like(raw_train)
    for fold in range(5):
        fit = folds != fold
        hold = ~fit
        w = np.linalg.solve(j_train[fit].T @ j_train[fit] + ridge * np.eye(j_train.shape[1]),
                            j_train[fit].T @ raw_train[fit])
        oof[hold] = raw_train[hold] - j_train[hold] @ w
    w = np.linalg.solve(j_train.T @ j_train + ridge * np.eye(j_train.shape[1]), j_train.T @ raw_train)
    validation = raw_val - j_val @ w
    return oof.astype(np.float32), validation.astype(np.float32)


def _correction_features(raw_residual: np.ndarray, action: np.ndarray) -> np.ndarray:
    return np.concatenate((raw_residual,
                           (raw_residual[:, :, None] * action[:, None, :]).reshape(len(action), -1)), axis=1)


def _bootstrap(frame: pd.DataFrame) -> list[float]:
    group = frame.groupby("base_trial_id")[["actual_sq", "base_error_sq", "corrected_error_sq"]].sum().to_numpy()
    rng = np.random.default_rng(1821)
    sample = rng.integers(0, len(group), size=(1000, len(group)))
    total = group[sample].sum(axis=1)
    gain = (np.sqrt(total[:, 1]) - np.sqrt(total[:, 2])) / np.sqrt(np.maximum(total[:, 0], 1e-24))
    return np.quantile(gain, [0.025, 0.5, 0.975]).tolist()


def run(root: Path) -> dict:
    freeze = _freeze(root)
    h1 = json.loads((root / OUT / "strong_h1_ceiling_v18.json").read_text())
    horizons = json.loads((root / OUT / "horizon_context_localization_v18.json").read_text())
    if not h1["material_h1_raw_context_gate_passed"] and horizons["earliest_material_horizon"] is None:
        raise RuntimeError("V18 conditional raw correction not eligible: no material raw ceiling")
    hyper = json.loads((root / OUT / "strong_model_train_selection_v18.json").read_text())
    model = _model_design(root)
    split = _split(root)
    coordinates = [x["coordinate_index"] for x in split["actions"]]
    with np.load(root / FEATURES, allow_pickle=False) as payload:
        scores = {name: payload[name] for name in payload.files}
    results = []
    all_rows = []
    for horizon in (1, 2, 4, 8):
        panel = horizon != 1
        train, val, _ = _data(root, horizon, panel, 0.5)
        if horizon == 1:
            # Primary h1 conditional test uses the full 2000/400 crossed cohort.
            panel = False
        jtr = scores["j"][train.state_index.to_numpy(), :128].astype(np.float32)
        jva = scores["j"][val.state_index.to_numpy(), :128].astype(np.float32)
        rawtr = scores["full_raw"][train.state_index.to_numpy(), :128].astype(np.float32)
        rawva = scores["full_raw"][val.state_index.to_numpy(), :128].astype(np.float32)
        atr, ava = _action(train, coordinates), _action(val, coordinates)
        ytr = np.stack(train.response_stacked_normalized).astype(np.float32)
        yva = np.stack(val.response_stacked_normalized).astype(np.float32)
        folds = _state_folds(train.base_trial_id.to_numpy())
        oof_target = np.zeros_like(ytr)
        for fold in range(5):
            fit, hold = folds != fold, folds == fold
            network = _neural("M4_small_state_conditioned_mlp", jtr[fit], atr[fit], ytr[fit],
                              hyper["neural"]["M4_small_state_conditioned_mlp"]["selected_weight_decay"],
                              int(model["mlp_seed"]), int(model["mlp_epochs"]))
            oof_target[hold] = ytr[hold] - _neural_predict(network, jtr[hold], atr[hold])
            del network
        base_network = _neural("M4_small_state_conditioned_mlp", jtr, atr, ytr,
                               hyper["neural"]["M4_small_state_conditioned_mlp"]["selected_weight_decay"],
                               int(model["mlp_seed"]), int(model["mlp_epochs"]))
        base_pred = _neural_predict(base_network, jva, ava)
        del base_network
        raw_res_tr, raw_res_va = _residualized_raw(rawtr, jtr, rawva, jva, folds,
                                                   float(hyper["selected_ridge_lambda"]))
        xtr = _correction_features(raw_res_tr, atr).astype(np.float64)
        xva = _correction_features(raw_res_va, ava).astype(np.float64)
        ridge = float(hyper["selected_ridge_lambda"])
        w = np.linalg.solve(xtr.T @ xtr / len(xtr) + ridge * np.eye(xtr.shape[1]),
                            xtr.T @ oof_target.astype(np.float64) / len(xtr))
        corrected = base_pred + (xva @ w).astype(np.float32)
        base_score = _summary(yva, base_pred)
        corrected_score = _summary(yva, corrected)
        parts = []
        for target, region in (("j", slice(0, 128)), ("stacked_normalized", slice(None))):
            part = pd.DataFrame({"base_trial_id": val.base_trial_id,
                                 "family": val.family, "horizon": horizon, "target": target,
                                 "actual_sq": (yva[:, region] ** 2).sum(axis=1),
                                 "base_error_sq": ((yva[:, region] - base_pred[:, region]) ** 2).sum(axis=1),
                                 "corrected_error_sq": ((yva[:, region] - corrected[:, region]) ** 2).sum(axis=1)})
            parts.append(part)
        frame = pd.concat(parts, ignore_index=True)
        all_rows.append(frame)
        gains = {}
        for target in ("j", "stacked_normalized"):
            part = frame[frame.target == target]
            base = float(np.sqrt(part.base_error_sq.sum() / part.actual_sq.sum()))
            corr = float(np.sqrt(part.corrected_error_sq.sum() / part.actual_sq.sum()))
            gains[target] = {"base_rel_l2": base, "corrected_rel_l2": corr,
                             "conditional_raw_gain": base - corr, "bootstrap_ci95_and_median": _bootstrap(part)}
        results.append({"horizon": horizon, "cohort": "full_h1" if horizon == 1 else "horizon_panel",
                        "train_states": int(train.base_trial_id.nunique()), "validation_states": int(val.base_trial_id.nunique()),
                        "train_rows": len(train), "validation_rows": len(val),
                        "gains": gains, "base_stack_score": base_score, "corrected_stack_score": corrected_score})
        print(f"V18 conditional h{horizon} stack gain={gains['stacked_normalized']['conditional_raw_gain']:.4f}", flush=True)
    pd.concat(all_rows, ignore_index=True).to_parquet(root / OUT / "conditional_raw_state_errors_v18.parquet", index=False)
    result = {"freeze_digest": freeze["freeze_digest"], "results": results,
              "train_only_OOF_target_and_raw_residualization": True,
              "base_fixed_for_correction": True, "validation_not_fit": True,
              "not_a_compact_C_conditional_independence_test": True}
    write_json_atomic(root / OUT / "strong_conditional_raw_context_v18.json", result)
    return {"freeze_digest": freeze["freeze_digest"], "gain_curve": [(r["horizon"], r["gains"]) for r in results]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("prepare", "run"))
    args = parser.parse_args()
    root = Path.cwd()
    result = {"prepare": prepare, "run": run}[args.stage](root)
    print(json.dumps(result, indent=2))
