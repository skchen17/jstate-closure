"""Frozen V20 state split and response-bank support."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from jclosure.protocol_v20 import stage_freeze, verify, verify_stage

SOURCE = "src/jclosure/experiments/operator_bank_v20.py"
OUT = Path("results/v20/processed")
SCRATCH = Path("/data/CSK/J-space-project/v20-operator-work")


def id_digest(items: list[dict[str, Any]]) -> str:
    return hashlib.sha256("\n".join(sorted(x["base_trial_id"] for x in items)).encode()).hexdigest()


def prompt_index(root: Path) -> dict[str, dict[str, Any]]:
    payload = json.loads((root / "data/v13/causal_bank_selected.json").read_text())
    result = {str(x["example_id"]): x for x in payload["items"]}
    if len(result) != len(payload["items"]):
        raise RuntimeError("V13 prompt-bank duplicate ID")
    return result


def prepare(root: Path) -> dict[str, Any]:
    cfg = verify(root)["config"]
    index = prompt_index(root)
    v13 = json.loads((root / "artifacts/causal_bank_v13_splits.freeze.json").read_text())
    v18 = json.loads((root / "artifacts/strong_state_context_ceiling_v18_splits.freeze.json").read_text())
    v19 = json.loads((root / "artifacts/counterfactual_workspace_v19_splits.freeze.json").read_text())
    v16 = json.loads((root / "artifacts/nonlinear_finite_causal_action_v16_splits.freeze.json").read_text())
    old = {x["base_trial_id"] for role in ("train", "validation") for x in v18[role]}
    old |= {x["base_trial_id"] for role in ("train", "validation", "calibration") for x in v19[role]}
    old |= {x["base_trial_id"] for role in ("train", "validation") for x in v16[role]}
    seed = int(cfg["roles"]["seed"])
    families = cfg["roles"]["families"]
    roles: dict[str, list[dict[str, Any]]] = {name: [] for name in
        ("independent_v19_confirmation", "q_calibration", "operator_train", "operator_validation", "reserved_independent_v20_final")}
    for family in families:
        final_pool = [key for key in v13["final_test"]["base_trial_ids"]
                      if index[key]["family"] == family and key not in old]
        final_pool.sort(key=lambda key: hashlib.sha256(f"{seed}:independent:{key}".encode()).hexdigest())
        n_final = int(cfg["roles"]["independent_V19_confirmation_per_family"])
        if len(final_pool) < n_final:
            raise RuntimeError(f"V20 independent/{family} only {len(final_pool)}")
        roles["independent_v19_confirmation"].extend(
            {"base_trial_id": key, "family": family, "role": "independent_v19_confirmation"}
            for key in final_pool[:n_final])
        roles["reserved_independent_v20_final"].extend(
            {"base_trial_id": key, "family": family, "role": "reserved_independent_v20_final"}
            for key in final_pool[n_final:])
        train_pool = [key for key in v13["train"]["base_trial_ids"]
                      if index[key]["family"] == family and key not in old]
        train_pool.sort(key=lambda key: hashlib.sha256(f"{seed}:train:{key}".encode()).hexdigest())
        nc = int(cfg["roles"]["q_calibration_per_family"])
        nt = int(cfg["roles"]["operator_train_per_family"])
        if len(train_pool) < nc + nt:
            raise RuntimeError(f"V20 train/{family} only {len(train_pool)}")
        roles["q_calibration"].extend(
            {"base_trial_id": key, "family": family, "role": "q_calibration"}
            for key in train_pool[:nc])
        roles["operator_train"].extend(
            {"base_trial_id": key, "family": family, "role": "operator_train"}
            for key in train_pool[nc:nc + nt])
        val_pool = [key for key in v13["validation"]["base_trial_ids"]
                    if index[key]["family"] == family and key not in old]
        val_pool.sort(key=lambda key: hashlib.sha256(f"{seed}:validation:{key}".encode()).hexdigest())
        nv = int(cfg["roles"]["operator_validation_per_family"])
        if len(val_pool) < nv:
            raise RuntimeError(f"V20 validation/{family} only {len(val_pool)}")
        roles["operator_validation"].extend(
            {"base_trial_id": key, "family": family, "role": "operator_validation"}
            for key in val_pool[:nv])
    ids = [x["base_trial_id"] for rows in roles.values() for x in rows]
    if len(ids) != len(set(ids)) or set(ids) & old:
        raise RuntimeError("V20 split leak or duplicate")
    expected_source = {"independent_v19_confirmation": "final_test", "reserved_independent_v20_final": "final_test",
                       "q_calibration": "train", "operator_train": "train", "operator_validation": "validation"}
    if any(index[item["base_trial_id"]]["split"] != expected_source[name]
           for name, rows in roles.items() for item in rows):
        raise RuntimeError("V20 prompt source split mismatch")
    detail = {
        **roles,
        "role_id_sha256": {name: id_digest(rows) for name, rows in roles.items()},
        "prior_V18_V19_V16_state_overlap": 0,
        "independent_confirmation_V13_source_split": "final_test",
        "operator_train_V13_source_split": "train",
        "operator_validation_V13_source_split": "validation",
        "selected_j": v19["selected_j"],
        "selected_logits": v19["selected_logits"],
        "target_scales": v19["target_scales"],
        "independent_V20_final_opened": False,
        "responses_already_observed": 0,
    }
    return stage_freeze(root, "splits", [SOURCE, "data/v13/causal_bank_selected.json",
                                         "artifacts/causal_bank_v13_splits.freeze.json",
                                         "artifacts/nonlinear_finite_causal_action_v16_splits.freeze.json",
                                         "artifacts/strong_state_context_ceiling_v18_splits.freeze.json",
                                         "artifacts/counterfactual_workspace_v19_splits.freeze.json"], detail)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("prepare",))
    args = parser.parse_args()
    print(json.dumps(prepare(Path.cwd()), indent=2))
