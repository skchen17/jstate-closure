"""V18 current clean snapshot versus pre-action J-history comparison."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from jclosure.experiments.crossed_bank_v18 import OUT, _split
from jclosure.experiments.state_features_v18 import FEATURES
from jclosure.experiments.strong_ceiling_v18 import (
    _action, _data, _fit_predict, _metrics, _model_design, _state_squares,
)
from jclosure.protocol_v18 import digest, stage_freeze, verify
from jclosure.provenance import sha256_file, write_json_atomic

SOURCE = "src/jclosure/experiments/history_v18.py"
FREEZE = Path("artifacts/strong_state_context_ceiling_v18_history.freeze.json")
ARRAY_AMENDMENT = Path("artifacts/strong_state_context_ceiling_v18_history_array_amendment_7.freeze.json")
SCORES = OUT / "history_context_scores_v18.npz"


def prepare(root: Path) -> dict:
    split = _split(root)
    model = _model_design(root)
    return stage_freeze(root, "history", [SOURCE, "artifacts/strong_state_context_ceiling_v18_splits.freeze.json",
                                           "artifacts/strong_state_context_ceiling_v18_models.freeze.json"],
                        {"split_freeze_digest": split["freeze_digest"],
                         "model_freeze_digest": model["freeze_digest"],
                         "history_source": "last_4_prompt_prefill_J_positions_pre_action_only",
                         "histories": ["last_2", "last_4"],
                         "context_comparators": ["current_j", "last_2_j", "last_4_j", "current_j_raw",
                                                 "last_2_j_plus_raw", "last_4_j_plus_raw"],
                         "history_feature_PCA": "train_only_centered_randomized_rank_128",
                         "joint_history_raw_projection": "train_only_centered_PCA_rank_128_of_two_128D_sources",
                         "model": "frozen_M4_small_state_conditioned_mlp_same_budget",
                         "horizons": [1, 2, 4, 8], "prior_intervention_actions": "unavailable_first_action_at_control_point",
                         "validation_not_used_for_feature_or_model_selection": True})


def _freeze(root: Path) -> dict:
    base = verify(root)
    value = json.loads((root / FREEZE).read_text())
    if value["freeze_digest"] != digest(value) or value["base_freeze_digest"] != base["freeze_digest"]:
        raise RuntimeError("V18 history freeze invalid")
    for path, expected in value["input_hashes"].items():
        observed = sha256_file(root / path)
        if observed != expected:
            if path != SOURCE or not (root / ARRAY_AMENDMENT).exists():
                raise RuntimeError(f"V18 history input changed: {path}")
            amendment = json.loads((root / ARRAY_AMENDMENT).read_text())
            if (amendment["freeze_digest"] != digest(amendment)
                    or amendment["base_freeze_digest"] != base["freeze_digest"]
                    or amendment["prior_source_sha256"] != expected
                    or amendment["input_hashes"].get(SOURCE) != observed
                    or amendment["input_hashes"].get(str(FREEZE)) != sha256_file(root / FREEZE)):
                raise RuntimeError("V18 history-array amendment invalid")
    return value


def amend_history_array(root: Path) -> dict:
    frozen = json.loads((root / FREEZE).read_text())
    return stage_freeze(root, "history_array_amendment_7", [SOURCE, str(FREEZE)],
                        {"reason": "Pandas reads nested Parquet history_j_last4 as an object array of four float arrays; np.asarray(..., dtype=float32) raises before feature fitting. Explicitly stack the four saved arrays without changing positions, values, split, comparator or model.",
                         "prior_source_sha256": frozen["input_hashes"][SOURCE],
                         "prior_history_features_or_fits_created": False,
                         "history_values_changed": False,
                         "input_shape": [4, 4096]})


def _load_states(root: Path, role: str) -> pd.DataFrame:
    paths = sorted((root / OUT).glob(f"crossed_state_{role}_*_v18.parquet"))
    if len(paths) != 5:
        raise RuntimeError(f"V18 {role} state metadata incomplete")
    return pd.concat([pd.read_parquet(path) for path in paths], ignore_index=True)


def build_scores(root: Path) -> dict:
    freeze = _freeze(root)
    with np.load(root / FEATURES, allow_pickle=False) as payload:
        feature = {name: payload[name] for name in payload.files}
    ids = feature["base_trial_id"].astype(str).tolist()
    index = {key: i for i, key in enumerate(ids)}
    n_train = len(_split(root)["train"])
    states = pd.concat((_load_states(root, "train"), _load_states(root, "validation")), ignore_index=True)
    if set(states.base_trial_id) != set(ids):
        raise RuntimeError("V18 history state IDs do not match clean raw feature IDs")
    histories = np.zeros((len(ids), 4, 4096), dtype=np.float32)
    for row in states.itertuples():
        histories[index[row.base_trial_id]] = np.stack(
            [np.asarray(position, dtype=np.float32) for position in row.history_j_last4]
        )
    output = {"current_j": feature["j"][:, :128].astype(np.float32),
              "current_j_raw": feature["j_rec_conv_kv"][:, :128].astype(np.float32)}
    metadata = {}
    for length in (2, 4):
        raw = histories[:, -length:, :].reshape(len(ids), -1)
        mean = raw[:n_train].mean(axis=0)
        pca = PCA(n_components=128, svd_solver="randomized", random_state=1818)
        scores = pca.fit_transform(raw[:n_train] - mean)
        projected = pca.transform(raw - mean)
        scale = np.maximum(scores.std(axis=0), 1e-5)
        history_score = (projected / scale).astype(np.float32)
        output[f"last_{length}_j"] = history_score
        combined = np.concatenate((history_score, output["current_j_raw"]), axis=1)
        mean_combined = combined[:n_train].mean(axis=0)
        pca_joint = PCA(n_components=128, svd_solver="full")
        train_joint = pca_joint.fit_transform(combined[:n_train] - mean_combined)
        both = pca_joint.transform(combined - mean_combined)
        output[f"last_{length}_j_plus_raw"] = (both / np.maximum(train_joint.std(axis=0), 1e-5)).astype(np.float32)
        metadata[f"last_{length}_j"] = {"PCA_explained_variance_fraction": float(pca.explained_variance_ratio_.sum())}
    np.savez_compressed(root / SCORES, **output, base_trial_id=np.asarray(ids))
    result = {"freeze_digest": freeze["freeze_digest"], "scores_path": str(SCORES),
              "scores_sha256": sha256_file(root / SCORES), "metadata": metadata,
              "no_future_or_action_response_input": True}
    write_json_atomic(root / OUT / "history_context_scores_v18.json", result)
    return result


def evaluate(root: Path) -> dict:
    freeze = _freeze(root)
    hyper = json.loads((root / OUT / "strong_model_train_selection_v18.json").read_text())
    model_design = _model_design(root)
    split = _split(root)
    coordinates = [x["coordinate_index"] for x in split["actions"]]
    with np.load(root / SCORES, allow_pickle=False) as payload:
        scores = {name: payload[name] for name in payload.files}
    ids = scores["base_trial_id"].astype(str).tolist()
    index = {key: i for i, key in enumerate(ids)}
    rows = []
    squares = []
    for horizon in (1, 2, 4, 8):
        train, val, _ = _data(root, horizon, True, 0.5)
        train_idx = np.asarray([index[key] for key in train.base_trial_id])
        val_idx = np.asarray([index[key] for key in val.base_trial_id])
        atr, ava = _action(train, coordinates), _action(val, coordinates)
        ytr = np.stack(train.response_stacked_normalized).astype(np.float32)
        yva = np.stack(val.response_stacked_normalized).astype(np.float32)
        for context in freeze["context_comparators"]:
            pred, params = _fit_predict("M4_small_state_conditioned_mlp", scores[context][train_idx], atr, ytr,
                                        scores[context][val_idx], ava, hyper, model_design)
            metrics = _metrics(val, yva, pred)
            rows.append({"horizon": horizon, "context": context, "model": "M4_small_state_conditioned_mlp",
                         "parameter_count": params, "train_rows": len(train), "validation_rows": len(val), "metrics": metrics})
            square = _state_squares(val, yva, pred, "M4_small_state_conditioned_mlp", context,
                                    horizon, "history_panel")
            squares.append(square)
            print(f"V18 history h{horizon} {context} stack_rel_l2={metrics['stacked_normalized']['relative_l2']:.4f}", flush=True)
    frame = pd.concat(squares, ignore_index=True)
    frame.to_parquet(root / OUT / "history_context_state_errors_v18.parquet", index=False, compression="zstd")
    lookup = {(row["horizon"], row["context"]): row for row in rows}
    gain_rows = []
    for horizon in (1, 2, 4, 8):
        for length in (2, 4):
            base = lookup[(horizon, "current_j")]["metrics"]["stacked_normalized"]["relative_l2"]
            history = lookup[(horizon, f"last_{length}_j")]["metrics"]["stacked_normalized"]["relative_l2"]
            raw = lookup[(horizon, "current_j_raw")]["metrics"]["stacked_normalized"]["relative_l2"]
            joint = lookup[(horizon, f"last_{length}_j_plus_raw")]["metrics"]["stacked_normalized"]["relative_l2"]
            gain_rows.append({"horizon": horizon, "history_length": length,
                              "history_gain_over_current_j": base - history,
                              "raw_gain_over_current_j": base - raw,
                              "history_incremental_gain_after_raw": raw - joint})
    pd.DataFrame(gain_rows).to_parquet(root / OUT / "history_incremental_gain_v18.parquet", index=False)
    result = {"freeze_digest": freeze["freeze_digest"], "results": rows, "gain_curves": gain_rows,
              "prior_action_history": "NOT_AVAILABLE_FIRST_INTERVENTION_AT_CONTROL_POINT",
              "current_snapshot_markov_sufficiency": "NOT_INFERRED_FROM_PREDICTIVE_GAIN_ALONE"}
    write_json_atomic(root / OUT / "history_vs_snapshot_v18.json", result)
    return {"gain_curves": gain_rows, "freeze_digest": freeze["freeze_digest"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("prepare", "amend_history_array", "scores", "evaluate"))
    args = parser.parse_args()
    root = Path.cwd()
    result = {"prepare": prepare, "amend_history_array": amend_history_array,
              "scores": build_scores, "evaluate": evaluate}[args.stage](root)
    print(json.dumps(result, indent=2))
