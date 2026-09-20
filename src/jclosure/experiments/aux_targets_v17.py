"""Frozen supplemental h1 target audit for the V17 state-context ceiling."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from jclosure.experiments.state_sufficiency_v17 import KERNELS, OUT, _kernel_set, _predict_key, _score, _split
from jclosure.protocol_v17 import stage_freeze, verify
from jclosure.provenance import sha256_file, write_json_atomic

SOURCE = "src/jclosure/experiments/aux_targets_v17.py"
DESIGN = {"targets": ["j", "logits", "semantic_continuous", "workspace", "stacked_normalized"],
          "contexts": ["action_only", "j", "j+rec", "j+conv", "j+kv", "j+rec+conv+kv", "full_raw_reference"],
          "model": "same_frozen_per_action_kernel_ridge_as_ceiling",
          "ridge": "train_only_selected_ceiling_ridge",
          "validation": "V16_development_validation_reused",
          "legacy_semantic": "not_in_V16_finite_action_bank; continuous_semantic_proxy_only"}


def freeze_design(root: Path) -> dict:
    split = _split(root)
    return stage_freeze(root, "aux_targets", [SOURCE, str(KERNELS), "results/v17/processed/state_context_ceiling_v17.json"],
                        {"parent_split_freeze_digest": split["freeze_digest"], "design": DESIGN})


def evaluate(root: Path) -> dict:
    from jclosure.protocol_v17 import digest
    verify(root)
    freeze = json.loads((root / "artifacts/interventional_state_sufficiency_v17_aux_targets.freeze.json").read_text())
    if freeze["freeze_digest"] != digest(freeze):
        raise RuntimeError("V17 auxiliary target freeze mismatch")
    for path, expected in freeze["input_hashes"].items():
        if sha256_file(root / path) != expected:
            raise RuntimeError(f"V17 auxiliary input changed: {path}")
    with np.load(root / KERNELS, allow_pickle=False) as payload:
        data = {name: payload[name] for name in payload.files}
    ids = data["base_trial_id"].astype(str).tolist()
    index = {key: i for i, key in enumerate(ids)}
    kernels = _kernel_set(data)
    bank = pd.read_parquet(root / "results/v16/processed/finite_action_bank_v16.parquet")
    bank = bank[(bank.design == "single") & (bank.reliability_status == "RELIABLE") & bank.base_trial_id.isin(index)].copy()
    bank["state_index"] = bank.base_trial_id.map(index).astype(int)
    bank["action_key"] = bank.coordinate_index.astype(int).astype(str) + ":" + bank.sign.astype(int).astype(str)
    ceiling = json.loads((root / "results/v17/processed/state_context_ceiling_v17.json").read_text())
    ridge = float(ceiling["selected_ridge"])
    rows = []
    for name in DESIGN["contexts"]:
        kernel = kernels[name]
        for target in DESIGN["targets"]:
            actual, predicted, families, keys = [], [], [], []
            for key in sorted(bank.action_key.unique()):
                tr = bank[(bank.role == "train") & (bank.action_key == key)]
                va = bank[(bank.role == "validation") & (bank.action_key == key)]
                if len(tr) < 15 or va.empty:
                    continue
                ytr = np.stack(tr[f"response_{target}"]).astype(np.float32)
                yva = np.stack(va[f"response_{target}"]).astype(np.float32)
                atr = tr.calibration_alpha.to_numpy(dtype=np.float32)
                ava = va.calibration_alpha.to_numpy(dtype=np.float32)
                prediction = _predict_key(kernel, tr.state_index.to_numpy(), va.state_index.to_numpy(),
                                          ytr / atr[:, None], ridge) * ava[:, None]
                actual.extend(yva)
                predicted.extend(prediction)
                families.extend(va.family.astype(str).tolist())
                keys.extend([key] * len(va))
            a, p = np.stack(actual), np.stack(predicted)
            result = _score(a, p)
            component_sign = (np.sign(a) == np.sign(p))[np.abs(a) > 1e-6]
            result["component_sign_agreement"] = float(component_sign.mean()) if component_sign.size else None
            result["context"] = name
            result["target"] = target
            result["by_family_relative_l2"] = {family: _score(a[np.asarray(families) == family], p[np.asarray(families) == family])["relative_l2"]
                                               for family in sorted(set(families))}
            rows.append(result)
    pd.DataFrame(rows).to_parquet(root / OUT / "h1_all_target_metrics_v17.parquet", index=False)
    summary = {"freeze_digest": freeze["freeze_digest"], "selected_ridge": ridge,
               "metrics": rows, "legacy_semantic_metric": "UNAVAILABLE_FROM_V16_BANK",
               "continuous_semantic_metric": "MEASURED", "independent": False}
    write_json_atomic(root / OUT / "h1_all_target_metrics_v17.json", summary)
    return {"freeze_digest": freeze["freeze_digest"], "record_count": len(rows)}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("freeze", "evaluate"))
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    print(json.dumps({"freeze": freeze_design, "evaluate": evaluate}[args.stage](args.root.resolve()), indent=2))
