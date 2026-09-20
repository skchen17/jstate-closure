"""V18 unified-model held-out sign and nested amplitude diagnostics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from jclosure.experiments.crossed_bank_v18 import OUT, _split
from jclosure.experiments.state_features_v18 import FEATURES
from jclosure.experiments.strong_ceiling_v18 import _action, _data, _fit_predict, _metrics, _model_design
from jclosure.protocol_v18 import digest, stage_freeze, verify
from jclosure.provenance import sha256_file, write_json_atomic

SOURCE = "src/jclosure/experiments/action_generalization_v18.py"
FREEZE = Path("artifacts/strong_state_context_ceiling_v18_action_generalization.freeze.json")


def prepare(root: Path) -> dict:
    split = _split(root)
    model = _model_design(root)
    return stage_freeze(root, "action_generalization", [SOURCE, str(FEATURES),
                                                         "artifacts/strong_state_context_ceiling_v18_models.freeze.json"],
                        {"split_freeze_digest": split["freeze_digest"], "model_freeze_digest": model["freeze_digest"],
                         "contexts": ["j", "j_rec_conv_kv"], "model": "frozen_M4",
                         "holdouts": ["alpha_1_from_alpha_0_5", "negative_sign_from_positive_sign"],
                         "validation_role": "V18_development_validation_not_independent",
                         "unseen_direction_pair_dense": "not_in_shared_action_bank_not_tested",
                         "train_hyperparameters": "frozen_train_only_h1_selection",
                         "no_validation_refitting": True})


def _freeze(root: Path) -> dict:
    base = verify(root)
    value = json.loads((root / FREEZE).read_text())
    if value["freeze_digest"] != digest(value) or value["base_freeze_digest"] != base["freeze_digest"]:
        raise RuntimeError("V18 action-generalization freeze invalid")
    for path, expected in value["input_hashes"].items():
        if sha256_file(root / path) != expected:
            raise RuntimeError(f"V18 action-generalization source changed: {path}")
    return value


def _all_rows(root: Path, role: str) -> pd.DataFrame:
    parts = sorted((root / OUT).glob(f"crossed_response_{role}_*_v18.parquet"))
    if len(parts) != 5:
        raise RuntimeError("V18 crossed bank incomplete")
    return pd.concat([pd.read_parquet(path) for path in parts], ignore_index=True)


def run(root: Path) -> dict:
    freeze = _freeze(root)
    split = _split(root)
    coordinates = [x["coordinate_index"] for x in split["actions"]]
    hyper = json.loads((root / OUT / "strong_model_train_selection_v18.json").read_text())
    model = _model_design(root)
    with np.load(root / FEATURES, allow_pickle=False) as payload:
        features = {name: payload[name] for name in payload.files}
    index = {key: i for i, key in enumerate(features["base_trial_id"].astype(str))}
    train = _all_rows(root, "train")
    validation = _all_rows(root, "validation")
    train = train[(train.horizon == 1) & (train.alpha == 0.5) &
                  (train.reliability_status == "RELIABLE")].copy()
    validation = validation[(validation.horizon == 1) &
                            (validation.reliability_status == "RELIABLE")].copy()
    results = []
    for holdout in freeze["holdouts"]:
        if holdout == "alpha_1_from_alpha_0_5":
            tr, va = train, validation[(validation.alpha == 1.0) & validation.horizon_panel]
        else:
            tr, va = train[train.sign == 1], validation[(validation.alpha == 0.5) & (validation.sign == -1)]
        tr = tr.sort_values(["base_trial_id", "coordinate_index", "sign"]).reset_index(drop=True)
        va = va.sort_values(["base_trial_id", "coordinate_index", "sign"]).reset_index(drop=True)
        atr, ava = _action(tr, coordinates), _action(va, coordinates)
        ytr = np.stack(tr.response_stacked_normalized).astype(np.float32)
        yva = np.stack(va.response_stacked_normalized).astype(np.float32)
        for context in freeze["contexts"]:
            xtr = features[context][np.asarray([index[key] for key in tr.base_trial_id]), :128].astype(np.float32)
            xva = features[context][np.asarray([index[key] for key in va.base_trial_id]), :128].astype(np.float32)
            pred, params = _fit_predict("M4_small_state_conditioned_mlp", xtr, atr, ytr, xva, ava, hyper, model)
            results.append({"holdout": holdout, "context": context,
                            "train_rows": len(tr), "validation_rows": len(va),
                            "train_states": int(tr.base_trial_id.nunique()),
                            "validation_states": int(va.base_trial_id.nunique()),
                            "parameter_count": params, "metrics": _metrics(va, yva, pred)})
            print(f"V18 {holdout} {context} stack={results[-1]['metrics']['stacked_normalized']['relative_l2']:.4f}", flush=True)
    lookup = {(row["holdout"], row["context"]): row for row in results}
    gains = {holdout: {target: lookup[(holdout, "j")]["metrics"][target]["relative_l2"] -
                      lookup[(holdout, "j_rec_conv_kv")]["metrics"][target]["relative_l2"]
                      for target in ("j", "stacked_normalized")}
             for holdout in freeze["holdouts"]}
    pd.DataFrame([{"holdout": row["holdout"], "context": row["context"],
                   "j_relative_l2": row["metrics"]["j"]["relative_l2"],
                   "stack_relative_l2": row["metrics"]["stacked_normalized"]["relative_l2"]}
                  for row in results]).to_parquet(root / OUT / "action_generalization_v18.parquet", index=False)
    result = {"freeze_digest": freeze["freeze_digest"], "results": results,
              "raw_context_gain_on_unseen_action_designs": gains,
              "unseen_pair_dense_direction": "NOT_TESTED_BY_THIS_BANK"}
    write_json_atomic(root / OUT / "action_generalization_v18.json", result)
    return {"gains": gains, "freeze_digest": freeze["freeze_digest"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("prepare", "run"))
    args = parser.parse_args()
    root = Path.cwd()
    result = {"prepare": prepare, "run": run}[args.stage](root)
    print(json.dumps(result, indent=2))
