"""S0–S3 V21 oracle state contexts from only V20 train-action measurements."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from jclosure.experiments import bank_v21 as bank
from jclosure.experiments import operator_model_v20 as v20_model
from jclosure.protocol_v21 import stage_freeze, verify_stage
from jclosure.provenance import sha256_file, write_json_atomic

SOURCE = "src/jclosure/experiments/state_representations_v21.py"
SCRATCH = bank.SCRATCH / "state_representations_v21.npz"
SUMMARY = bank.OUT / "state_representations_v21.json"


def _j(root: Path, role: str, ids: np.ndarray) -> np.ndarray:
    frame = pd.read_parquet(root / f"results/v20/processed/operator_state_metadata_{role}_v20.parquet")
    by_id = frame.set_index("base_trial_id")
    return np.stack([np.asarray(by_id.loc[value.split("::")[0]].boundary_j_vector, dtype=np.float32)
                     for value in ids])


def prepare(root: Path) -> dict:
    verify_stage(root, "roles")
    operator = json.loads((root / "artifacts/compact_causal_response_operator_v20_operator_design.freeze.json").read_text())
    train = v20_model._load(root, "operator_train", operator)
    validation = v20_model._load(root, "operator_validation", operator)
    if len(train["ids"]) != 600 or len(validation["ids"]) != 200:
        raise RuntimeError("V21 S0-S3 operator-state count mismatch")
    pca = v20_model._pca(v20_model._fingerprints(train), v20_model._fingerprints(validation))
    ctrain, cval, _ = v20_model._coordinate(pca, 128)
    features = {
        "S0_train": _j(root, "operator_train", train["ids"]),
        "S0_validation": _j(root, "operator_validation", validation["ids"]),
        "S1_train": ctrain,
        "S1_validation": cval,
        "S2_train": train["Y"][("train", 1)].reshape(len(train["ids"]), -1),
        "S2_validation": validation["Y"][("train", 1)].reshape(len(validation["ids"]), -1),
        "S3_train": np.concatenate((train["Y"][("train", 1)], train["Y"][("train", -1)]), axis=1).reshape(len(train["ids"]), -1),
        "S3_validation": np.concatenate((validation["Y"][("train", 1)], validation["Y"][("train", -1)]), axis=1).reshape(len(validation["ids"]), -1),
        "train_ids": train["ids"], "validation_ids": validation["ids"],
        "train_family": train["family"], "validation_family": validation["family"],
        "train_base_ids": train["base_ids"], "validation_base_ids": validation["base_ids"],
    }
    SCRATCH.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(SCRATCH, **features)
    summary = {"train_operator_states": len(train["ids"]), "validation_operator_states": len(validation["ids"]),
               "definitions": {
                   "S0": "frozen 128 selected current boundary J values; same for P0/Pq within a base",
                   "S1": "V20 centered positive 12-train-action fingerprint SVD coordinates k128; train-only basis and scaling",
                   "S2": "complete 12x288 positive train-action response fingerprint, no compression",
                   "S3": "complete positive+negative 24x288 train-action response fingerprint, no compression"},
               "dimensions": {key[:2]: int(features[key].shape[1]) for key in ("S0_train", "S1_train", "S2_train", "S3_train")},
               "train_positive_fingerprint_energy_k128": float(pca["energy"][:128].sum()),
               "training_states_used_to_fit_S1_basis": len(train["ids"]),
               "validation_unseen_action_responses_used_for_S0_S3": False,
               "scratch_not_for_git": str(SCRATCH), "scratch_sha256": sha256_file(SCRATCH),
               "S4_status": "PENDING_RAW_P_ARCHITECTURE_KERNEL_EXTRACTION",
               "V20_final_six_action_responses_opened": False}
    target = root / SUMMARY
    target.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(target, summary)
    freeze = stage_freeze(root, "state_representations",
                          [SOURCE, "artifacts/action_coordinate_geometry_v21_roles.freeze.json",
                           "results/v20/processed/response_operator_operator_train_v20.json",
                           "results/v20/processed/response_operator_operator_validation_v20.json",
                           str(SUMMARY)],
                          {"state_feature_scratch_sha256": summary["scratch_sha256"],
                           "state_feature_summary_sha256": sha256_file(target),
                           "S1_S2_S3_validation_direction_responses_used": False,
                           "S4_pending_separate_freeze": True,
                           "V20_final_six_action_responses_opened": False})
    return {"freeze_digest": freeze["freeze_digest"], "dimensions": summary["dimensions"]}


if __name__ == "__main__":
    print(json.dumps(prepare(Path.cwd()), indent=2))
