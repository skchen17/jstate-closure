"""V18 truly near-J matched finite interventions with no threshold relaxation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from jclosure.experiments.crossed_bank_v18 import OUT, _split
from jclosure.experiments.state_features_v18 import KERNELS
from jclosure.protocol_v18 import digest, stage_freeze, verify
from jclosure.provenance import sha256_file, write_json_atomic

SOURCE = "src/jclosure/experiments/strict_match_v18.py"
FREEZE = Path("artifacts/strong_state_context_ceiling_v18_strict_match.freeze.json")


def prepare(root: Path) -> dict:
    split = _split(root)
    cfg = verify(root)["config"]["gates"]
    return stage_freeze(root, "strict_match", [SOURCE, str(KERNELS),
                                                "artifacts/strong_state_context_ceiling_v18_splits.freeze.json"],
                        {"split_freeze_digest": split["freeze_digest"],
                         "J_max_distance": float(cfg["strict_j_match_max_normalized_distance"]),
                         "raw_different_min_distance": float(cfg["strict_raw_difference_min_normalized_distance"]),
                         "raw_near_max_distance": 0.25, "different_J_min_distance": 0.5,
                         "response_rel_divergence_min": float(cfg["strict_response_divergence_min"]),
                         "same_family": True, "same_template": True,
                         "maximum_prompt_token_length_difference": 4,
                         "exact_shared_action_key": "coordinate_sign_alpha_0.5_reliable_both",
                         "one_to_one_pair_assignment": "greedy_ascending_J_distance_no_state_reuse_per_type",
                         "threshold_relaxation": "prohibited",
                         "no_match_outcome": "MATCH_NOT_IDENTIFIED"})


def _freeze(root: Path) -> dict:
    base = verify(root)
    value = json.loads((root / FREEZE).read_text())
    if value["freeze_digest"] != digest(value) or value["base_freeze_digest"] != base["freeze_digest"]:
        raise RuntimeError("V18 strict-match freeze invalid")
    for path, expected in value["input_hashes"].items():
        if sha256_file(root / path) != expected:
            raise RuntimeError(f"V18 strict-match input changed: {path}")
    return value


def _distance(kernel: np.ndarray) -> np.ndarray:
    diag = np.diag(kernel)
    return np.sqrt(np.maximum(diag[:, None] + diag[None, :] - 2 * kernel, 0.0))


def _greedy(rows: list[dict]) -> list[dict]:
    used = set()
    selected = []
    for row in sorted(rows, key=lambda x: (x["j_distance"], x["state_a"], x["state_b"])):
        if row["state_a"] in used or row["state_b"] in used:
            continue
        used.update((row["state_a"], row["state_b"]))
        selected.append(row)
    return selected


def run(root: Path) -> dict:
    freeze = _freeze(root)
    with np.load(root / KERNELS, allow_pickle=False) as payload:
        data = {name: payload[name] for name in payload.files}
    ids = data["base_trial_id"].astype(str).tolist()
    index = {key: i for i, key in enumerate(ids)}
    dj = _distance(data["j"])
    raw = np.mean([data[name] for name in ("rec", "conv", "kv")], axis=0)
    dr = _distance(raw)
    state_paths = sorted((root / OUT).glob("crossed_state_validation_*_v18.parquet"))
    response_paths = sorted((root / OUT).glob("crossed_response_validation_*_v18.parquet"))
    if len(state_paths) != 5 or len(response_paths) != 5:
        raise RuntimeError("V18 validation bank incomplete for strict matching")
    states = pd.concat([pd.read_parquet(path) for path in state_paths], ignore_index=True)
    responses = pd.concat([pd.read_parquet(path) for path in response_paths], ignore_index=True)
    responses = responses[(responses.horizon == 1) & (responses.alpha == 0.5) &
                          (responses.reliability_status == "RELIABLE")].copy()
    responses["action_key"] = (responses.coordinate_index.astype(int).astype(str) + ":" +
                               responses.sign.astype(int).astype(str) + ":0.5")
    by_state = {key: frame.set_index("action_key") for key, frame in responses.groupby("base_trial_id")}
    candidates = {name: [] for name in ("near_J_different_raw", "near_J_near_raw", "different_J_near_raw")}
    for _, group in states.groupby(["family", "template_id"], sort=True):
        rows = list(group.itertuples())
        for a in range(len(rows)):
            for b in range(a + 1, len(rows)):
                left, right = rows[a], rows[b]
                if abs(int(left.prompt_length) - int(right.prompt_length)) > freeze["maximum_prompt_token_length_difference"]:
                    continue
                i, j = index[left.base_trial_id], index[right.base_trial_id]
                jd, rd = float(dj[i, j]), float(dr[i, j])
                detail = {"state_a": left.base_trial_id, "state_b": right.base_trial_id,
                          "family": left.family, "template_id": left.template_id,
                          "j_distance": jd, "raw_distance": rd,
                          "prompt_length_a": int(left.prompt_length), "prompt_length_b": int(right.prompt_length)}
                if jd <= freeze["J_max_distance"] and rd >= freeze["raw_different_min_distance"]:
                    candidates["near_J_different_raw"].append(detail)
                if jd <= freeze["J_max_distance"] and rd <= freeze["raw_near_max_distance"]:
                    candidates["near_J_near_raw"].append(detail)
                if jd >= freeze["different_J_min_distance"] and rd <= freeze["raw_near_max_distance"]:
                    candidates["different_J_near_raw"].append(detail)
    rows = []
    pair_counts = {}
    for pair_type, possibilities in candidates.items():
        selected = _greedy(possibilities)
        pair_counts[pair_type] = {"candidate_state_pairs": len(possibilities),
                                  "selected_one_to_one_state_pairs": len(selected)}
        for pair in selected:
            a, b = by_state.get(pair["state_a"]), by_state.get(pair["state_b"])
            if a is None or b is None:
                continue
            for action_key in sorted(set(a.index) & set(b.index)):
                ya = np.asarray(a.loc[action_key, "response_stacked_normalized"], dtype=np.float32)
                yb = np.asarray(b.loc[action_key, "response_stacked_normalized"], dtype=np.float32)
                divergence = float(np.linalg.norm(ya - yb) / max(0.5 * (np.linalg.norm(ya) + np.linalg.norm(yb)), 1e-12))
                rows.append({**pair, "pair_type": pair_type, "action_key": action_key,
                             "response_relative_divergence": divergence,
                             "divergence_exceeds_frozen_threshold": divergence >= freeze["response_rel_divergence_min"]})
    frame = pd.DataFrame(rows)
    if frame.empty:
        frame = pd.DataFrame(columns=["state_a", "state_b", "family", "template_id", "j_distance", "raw_distance",
                                      "prompt_length_a", "prompt_length_b", "pair_type", "action_key",
                                      "response_relative_divergence", "divergence_exceeds_frozen_threshold"])
    frame.to_parquet(root / OUT / "strict_matched_interventions_v18.parquet", index=False)
    summaries = {}
    for pair_type in candidates:
        part = frame[frame.pair_type == pair_type]
        summaries[pair_type] = {**pair_counts[pair_type], "measured_action_pairs": len(part),
                                "median_response_relative_divergence": float(part.response_relative_divergence.median()) if len(part) else None,
                                "fraction_above_response_threshold": float(part.divergence_exceeds_frozen_threshold.mean()) if len(part) else None}
    status = ("MATCH_NOT_IDENTIFIED" if summaries["near_J_different_raw"]["measured_action_pairs"] == 0
              else "STRICT_MATCH_MEASURED_DESCRIPTIVE")
    result = {"freeze_digest": freeze["freeze_digest"], "status": status,
              "pair_types": summaries, "thresholds_never_relaxed": True,
              "strict_match_is_not_physical_context_swap": True}
    write_json_atomic(root / OUT / "strict_matched_intervention_v18.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("prepare", "run"))
    args = parser.parse_args()
    root = Path.cwd()
    result = {"prepare": prepare, "run": run}[args.stage](root)
    print(json.dumps(result, indent=2))
