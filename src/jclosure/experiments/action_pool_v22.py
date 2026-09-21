"""Frozen 512-action pool, BF16 calibration, and train-only action designs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from jclosure.experiments import counterfactual_bank_v19 as v19
from jclosure.experiments.action_bank_v16 import _readback
from jclosure.experiments.response_operator_v20 import _action_ok
from jclosure.protocol_v22 import stage_freeze, verify, verify_stage
from jclosure.provenance import sha256_file, write_json_atomic

SOURCE = "src/jclosure/experiments/action_pool_v22.py"
OUT = Path("results/v22/processed")
SCRATCH = Path("/data/CSK/J-space-project/v22-action-manifold-work")
V21_RAW = Path("/data/CSK/J-space-project/v21-action-geometry-work/paired_geometry_forward_ad/development")
CHANNELS = ("recurrent", "conv", "kv")


def _digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _hashed(items: list[dict], seed: str) -> list[dict]:
    return sorted(items, key=lambda item: hashlib.sha256(f"{seed}:{item['action_id']}".encode()).hexdigest())


def _family_map(summary: dict) -> dict[int, str]:
    answer = {}
    priority = ["causal_weighted", "architecture_balanced", "random_raw", "high_variance_pca", "low_variance"]
    for name in priority:
        for index in summary["probe_families"].get(name, []):
            answer.setdefault(int(index), name)
    return answer


def prepare(root: Path) -> dict:
    config = verify(root)["config"]
    v20 = json.loads((root / "artifacts/compact_causal_response_operator_v20_actions.freeze.json").read_text())
    summary = json.loads((root / "results/v13/processed/probe_directions_v13.json").read_text())
    historical_final = {int(item["direction_index"]) for item in v20["partitions"]["final_heldout"]}
    families = _family_map(summary)
    originals = [{"action_id": f"d{index:03d}", "kind": "direction", "direction_index": index,
                  "components": [[index, 1.0]], "source_family": families.get(index, "unclassified")}
                 for index in range(512) if index not in historical_final]
    ordered = _hashed(originals, "V22-DERIVED-COMBINATION-SOURCE")
    combinations = []
    for number in range(6):
        left, right = ordered[2 * number], ordered[2 * number + 1]
        sign = -1.0 if number % 2 else 1.0
        combinations.append({"action_id": f"c{number:03d}", "kind": "balanced_combination",
                             "direction_index": None,
                             "components": [[left["direction_index"], 2 ** -0.5],
                                            [right["direction_index"], sign * 2 ** -0.5]],
                             "source_family": "orthogonal_random_combination"})
    pool = originals + combinations
    if len(pool) != int(config["candidate_pool"]["count"]):
        raise RuntimeError("V22 action-pool count mismatch")
    split_order = _hashed(pool, "V22-INDEPENDENT-PARTITIONS")
    final = split_order[:32]
    validation = split_order[32:64]
    development = split_order[64:]
    payload = {
        "candidate_pool": pool,
        "candidate_count": len(pool),
        "source_family_counts": {name: sum(item["source_family"] == name for item in pool)
                                 for name in sorted({item["source_family"] for item in pool})},
        "development_action_ids": [item["action_id"] for item in development],
        "validation_action_ids": [item["action_id"] for item in validation],
        "independent_final_action_ids": [item["action_id"] for item in final],
        "historical_final_direction_indices": sorted(historical_final),
        "historical_final_raw_geometry_inspected": False,
        "historical_final_responses_opened": False,
        "new_independent_final_responses_opened": False,
        "candidate_pool_hash": _digest(pool),
        "development_hash": _digest([item["action_id"] for item in development]),
        "validation_hash": _digest([item["action_id"] for item in validation]),
        "independent_final_hash": _digest([item["action_id"] for item in final]),
        "response_labels_observed_before_split": 0,
    }
    target = root / OUT / "action_pool_v22.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(target, payload)
    frozen = stage_freeze(root, "action_pool_and_partitions", [SOURCE,
        "artifacts/compact_causal_response_operator_v20_actions.freeze.json",
        "results/v13/processed/probe_directions_v13.json", str(OUT / "action_pool_v22.json")],
        {key: payload[key] for key in ("candidate_count", "candidate_pool_hash", "development_hash",
                                       "validation_hash", "independent_final_hash",
                                       "historical_final_responses_opened", "new_independent_final_responses_opened")})
    return {"freeze_digest": frozen["freeze_digest"], **{key: payload[key] for key in (
        "candidate_count", "candidate_pool_hash", "development_hash", "validation_hash", "independent_final_hash")}}


def _row(directions: dict, spec: dict, alpha: float, sign: int) -> dict[str, torch.Tensor]:
    return {channel: sum(float(weight) * directions[channel][int(index)].float()
                         for index, weight in spec["components"]) * (float(alpha) * int(sign))
            for channel in CHANNELS}


def calibrate(root: Path) -> dict:
    verify_stage(root, "action_pool_and_partitions")
    config = verify(root)["config"]
    pool = json.loads((root / OUT / "action_pool_v22.json").read_text())
    roles = json.loads((root / "artifacts/action_coordinate_geometry_v21_roles.freeze.json").read_text())
    operator = json.loads((root / "artifacts/compact_causal_response_operator_v20_operator_design.freeze.json").read_text())
    by_family = {}
    for item in roles["jvp_development"]:
        by_family.setdefault(item["family"], item["base_trial_id"])
    base_ids = [by_family[name] for name in sorted(by_family)]
    prompts = __import__("jclosure.experiments.operator_bank_v20", fromlist=["prompt_index"]).prompt_index(root)
    bundle, dense, _, values, rec, att, measured, state_layer, _, _ = v19._setup(root)
    directions = values["directions"]
    rows = []
    specs = pool["candidate_pool"]
    alphas = [float(value) for value in config["candidate_pool"]["alpha_candidates"]]
    for base_number, base_id in enumerate(base_ids, 1):
        clean = v19._prefill_history(bundle, str(prompts[base_id]["prompt"]), measured, dense, state_layer)
        p0 = clean["cache"]
        for action_number, spec in enumerate(specs, 1):
            for alpha in alphas:
                for sign in (-1, 1):
                    requested = _row(directions, spec, alpha, sign)
                    edited = v19.apply(p0, 1.0, requested, rec, att, "native_fp32_add_bf16_writeback")
                    actual = _readback(p0, edited, requested, rec, att)
                    rows.append({"base_trial_id": base_id, "action_id": spec["action_id"],
                                 "source_family": spec["source_family"], "alpha": alpha, "sign": sign,
                                 "requested_raw_norm": actual["requested_state_norm"],
                                 "realized_bf16_norm": actual["realized_state_norm"],
                                 "requested_realized_cosine": actual["realized_state_cosine"],
                                 "gain": actual["realized_state_gain"],
                                 "REC_survival": actual["channel_survival"]["recurrent"],
                                 "Conv_survival": actual["channel_survival"]["conv"],
                                 "K_survival": actual["channel_survival"]["keys"],
                                 "V_survival": actual["channel_survival"]["values"],
                                 "reliable": bool(_action_ok(actual, operator["action_reliability"]))})
            if action_number % 64 == 0:
                print(f"V22 calibration base={base_number}/{len(base_ids)} action={action_number}/{len(specs)}", flush=True)
    frame = pd.DataFrame(rows)
    target = root / OUT / "action_pool_calibration_v22.parquet"
    frame.to_parquet(target, index=False, compression="zstd")
    selected = []
    required = float(config["candidate_pool"]["symmetric_reliability_fraction_min"])
    for spec in specs:
        group = frame[frame.action_id == spec["action_id"]]
        candidates = []
        detail = {}
        for alpha in alphas:
            fractions = {str(sign): float(group[(group.alpha == alpha) & (group.sign == sign)].reliable.mean()) for sign in (-1, 1)}
            detail[str(alpha)] = fractions
            if min(fractions.values()) >= required:
                candidates.append(alpha)
        alpha = min(candidates) if candidates else None
        chosen = group[group.alpha == alpha] if alpha is not None else group[group.alpha == max(alphas)]
        selected.append({"action_id": spec["action_id"], "selected_alpha": alpha,
                         "positive_reliability": detail[str(alpha)]["1"] if alpha is not None else 0.0,
                         "negative_reliability": detail[str(alpha)]["-1"] if alpha is not None else 0.0,
                         "candidate_reliability": detail,
                         "median_requested_norm": float(chosen.requested_raw_norm.median()),
                         "median_realized_norm": float(chosen.realized_bf16_norm.median()),
                         "median_cosine": float(chosen.requested_realized_cosine.median()),
                         "median_gain": float(chosen.gain.median()),
                         "median_REC_survival": float(chosen.REC_survival.median()),
                         "median_Conv_survival": float(chosen.Conv_survival.median()),
                         "median_K_survival": float(chosen.K_survival.median()),
                         "median_V_survival": float(chosen.V_survival.median())})
    summary = {"candidate_count": len(specs), "calibration_states": base_ids,
               "calibration_rows": len(frame), "by_action": selected,
               "symmetric_reliable_action_count": sum(item["selected_alpha"] is not None for item in selected),
               "parquet_sha256": sha256_file(target), "historical_final_responses_opened": False,
               "new_independent_final_responses_opened": False}
    summary_target = root / OUT / "action_pool_calibration_v22.json"
    write_json_atomic(summary_target, summary)
    frozen = stage_freeze(root, "action_reliability", [SOURCE, str(OUT / "action_pool_v22.json"), str(target), str(OUT / "action_pool_calibration_v22.json")],
                          {"symmetric_reliable_action_count": summary["symmetric_reliable_action_count"],
                           "calibration_sha256": summary["parquet_sha256"],
                           "historical_final_responses_opened": False,
                           "new_independent_final_responses_opened": False})
    return {"freeze_digest": frozen["freeze_digest"],
            "symmetric_reliable_action_count": summary["symmetric_reliable_action_count"]}


def _score(spec: dict, score: np.ndarray) -> np.ndarray:
    vector = sum(float(weight) * score[int(index)] for index, weight in spec["components"])
    return np.asarray(vector, dtype=np.float64)


def _maximin(features: np.ndarray, ids: list[str], count: int, seed: str) -> list[str]:
    norms = np.maximum(np.linalg.norm(features, axis=1, keepdims=True), 1e-12)
    unit = features / norms
    first = min(range(len(ids)), key=lambda i: hashlib.sha256(f"{seed}:{ids[i]}".encode()).hexdigest())
    chosen = [first]
    best = np.abs(unit @ unit[first])
    while len(chosen) < count:
        candidates = [i for i in range(len(ids)) if i not in chosen]
        value = min(candidates, key=lambda i: (best[i], hashlib.sha256(f"{seed}:{ids[i]}".encode()).hexdigest()))
        chosen.append(value)
        best = np.maximum(best, np.abs(unit @ unit[value]))
    return [ids[index] for index in chosen]


def select(root: Path) -> dict:
    verify_stage(root, "action_reliability")
    config = verify(root)["config"]
    pool = json.loads((root / OUT / "action_pool_v22.json").read_text())
    calibration = json.loads((root / OUT / "action_pool_calibration_v22.json").read_text())
    reliable = {item["action_id"]: item for item in calibration["by_action"] if item["selected_alpha"] is not None}
    development_ids = [value for value in pool["development_action_ids"] if value in reliable]
    if len(development_ids) < 128:
        raise RuntimeError(f"V22 reliable development pool too small: {len(development_ids)}")
    specs = {item["action_id"]: item for item in pool["candidate_pool"]}
    directions = torch.load(root / "artifacts/causal/v13/probe_directions_v13.pt", map_location="cpu", weights_only=False)
    score = np.asarray(directions["score_directions"], dtype=np.float64)
    del directions
    features = np.stack([_score(specs[value], score) * float(reliable[value]["selected_alpha"]) for value in development_ids])
    block = features.reshape(len(features), 3, 4799)
    block_scale = np.median(np.linalg.norm(block, axis=2), axis=0)
    whitened = (block / np.maximum(block_scale[None, :, None], 1e-12)).reshape(len(features), -1)
    random_order = [item["action_id"] for item in _hashed([specs[value] for value in development_ids], "V22-RANDOM-ARCH-BALANCED")]
    # Interleave source families to make the deterministic random baseline architecture balanced.
    family_bins = {}
    for value in random_order:
        family_bins.setdefault(specs[value]["source_family"], []).append(value)
    random_balanced = []
    while len(random_balanced) < len(random_order):
        for family in sorted(family_bins):
            if family_bins[family]:
                random_balanced.append(family_bins[family].pop(0))
    maximin = _maximin(whitened, development_ids, len(development_ids), "V22-MAXIMIN")
    # Train-only causal embedding: Nyström continuation from V21 development exact-JVP covariance.
    roles = json.loads((root / "artifacts/action_coordinate_geometry_v21_roles.freeze.json").read_text())
    scales = json.loads((root / "artifacts/action_coordinate_geometry_v21_probe_scales.freeze.json").read_text())
    probe_ids = [int(value) for value in roles["jvp_probe_direction_indices"] if scales["selected_alpha_by_direction"][str(value)] is not None]
    probe_columns = [roles["jvp_probe_direction_indices"].index(value) for value in probe_ids]
    response_cov = np.zeros((len(probe_ids), len(probe_ids)), dtype=np.float64)
    count = 0
    for item in roles["jvp_development"]:
        with np.load(root / V21_RAW / f"paired_{item['base_trial_id']}.npz") as source:
            for state in ("P0", "Pq"):
                y = np.asarray(source[f"{state}_JVP"][:, probe_columns], dtype=np.float64)
                response_cov += y.T @ y
                count += 1
    response_cov /= count
    probe_features = np.stack([score[value] * float(scales["selected_alpha_by_direction"][str(value)]) for value in probe_ids])
    probe_block = probe_features.reshape(len(probe_features), 3, 4799)
    probe_white = (probe_block / np.maximum(block_scale[None, :, None], 1e-12)).reshape(len(probe_features), -1)
    kpp = probe_white @ probe_white.T
    kcp = whitened @ probe_white.T
    ridge = max(float(np.trace(kpp) / len(kpp)) * 1e-6, 1e-8)
    eigen, vectors = np.linalg.eigh(response_cov)
    keep = eigen > max(float(eigen[-1]) * 1e-8, 1e-12)
    causal_root = vectors[:, keep] * np.sqrt(np.maximum(eigen[keep], 0))[None, :]
    causal_embedding = kcp @ np.linalg.solve(kpp + ridge * np.eye(len(kpp)), causal_root)
    # Greedy D-optimal design in the estimated causal embedding.
    causal_embedding /= max(float(np.median(np.linalg.norm(causal_embedding, axis=1))), 1e-12)
    inverse = np.eye(causal_embedding.shape[1]) / 1e-3
    remaining = set(range(len(development_ids)))
    causal_order = []
    while remaining:
        chosen = max(remaining, key=lambda i: (float(causal_embedding[i] @ inverse @ causal_embedding[i]),
                                               hashlib.sha256(f"V22-DOPT:{development_ids[i]}".encode()).hexdigest()))
        vector = causal_embedding[chosen]
        iv = inverse @ vector
        inverse -= np.outer(iv, iv) / (1.0 + float(vector @ iv))
        causal_order.append(development_ids[chosen])
        remaining.remove(chosen)
    common = []
    for order, quota in ((random_balanced, 43), (maximin, 43), (causal_order, 42)):
        for value in order:
            if value not in common:
                common.append(value)
            if sum(item in common for item in order[:order.index(value)+1]) >= quota and len(common) >= sum(q for _, q in ((random_balanced,43),(maximin,43),(causal_order,42)) if _ is order):
                pass
        # quota is enforced by a simpler fill below; union can overlap.
    # Deterministically cap/fill to one common 128-direction measured panel.
    union = []
    for order, quota in ((random_balanced, 43), (maximin, 43), (causal_order, 42)):
        taken = 0
        for value in order:
            if value not in union:
                union.append(value); taken += 1
            if taken == quota:
                break
    for value in causal_order + maximin + random_balanced:
        if len(union) >= 128:
            break
        if value not in union:
            union.append(value)
    common = union[:128]
    rankings = {}
    for name, order in (("random_architecture_balanced", random_balanced),
                        ("raw_geometry_maximin", maximin), ("causal_d_optimal", causal_order)):
        rankings[name] = [value for value in order if value in set(common)]
        rankings[name] += [value for value in common if value not in rankings[name]]
    validation = [value for value in pool["validation_action_ids"] if value in reliable]
    if len(validation) < 32:
        raise RuntimeError(f"V22 reliable validation set too small: {len(validation)}")
    validation = validation[:32]
    action_alphas = {value: float(reliable[value]["selected_alpha"]) for value in common + validation}
    result = {
        "selection_strategies": list(rankings), "nested_counts": config["action_partitions"]["nested_counts"],
        "strategy_rankings": rankings, "common_measured_train_action_ids": common,
        "validation_action_ids": validation,
        "independent_final_action_ids": pool["independent_final_action_ids"],
        "action_alphas": action_alphas,
        "common_train_hash": _digest(common), "validation_hash": _digest(validation),
        "independent_final_hash": pool["independent_final_hash"],
        "selection_hashes": {name: _digest(order) for name, order in rankings.items()},
        "raw_block_scales": block_scale.tolist(), "causal_embedding_rank": int(causal_embedding.shape[1]),
        "causal_metric_source": "V21 development exact-JVP P0/Pq only; validation and final response labels excluded",
        "historical_final_responses_opened": False, "new_independent_final_responses_opened": False,
    }
    target = root / OUT / "action_selection_v22.json"
    write_json_atomic(target, result)
    scratch = root / SCRATCH / "action_coordinates_v22.npz"
    np.savez_compressed(scratch, development_ids=np.asarray(development_ids), raw_whitened=whitened.astype(np.float32),
                        causal_embedding=causal_embedding.astype(np.float32), probe_direction_indices=np.asarray(probe_ids),
                        response_covariance=response_cov.astype(np.float32))
    result["coordinate_scratch_sha256"] = sha256_file(scratch)
    write_json_atomic(target, result)
    frozen = stage_freeze(root, "action_selection", [SOURCE,
        str(OUT / "action_pool_v22.json"), str(OUT / "action_pool_calibration_v22.json"),
        "results/v21/processed/paired_jvp_finite_operator_v21.parquet", str(target)],
        {key: result[key] for key in ("selection_strategies", "nested_counts", "common_train_hash",
                                      "validation_hash", "independent_final_hash", "selection_hashes",
                                      "coordinate_scratch_sha256", "historical_final_responses_opened",
                                      "new_independent_final_responses_opened")})
    return {"freeze_digest": frozen["freeze_digest"], "train_count": len(common),
            "validation_count": len(validation), "selection_hashes": result["selection_hashes"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("prepare", "calibrate", "select"))
    args = parser.parse_args()
    answer = {"prepare": prepare, "calibrate": calibrate, "select": select}[args.stage](Path.cwd())
    print(json.dumps(answer, indent=2))
