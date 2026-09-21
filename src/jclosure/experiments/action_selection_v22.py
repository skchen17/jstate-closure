"""Append-only reliability amendment and train-only V22 action selection."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from jclosure.experiments import action_pool_v22 as bank
from jclosure.protocol_v22 import stage_freeze, verify, verify_stage
from jclosure.provenance import sha256_file, write_json_atomic

SOURCE = "src/jclosure/experiments/action_selection_v22.py"


def run(root: Path) -> dict:
    verify_stage(root, "action_reliability")
    config = verify(root)["config"]
    pool = json.loads((root / bank.OUT / "action_pool_v22.json").read_text())
    calibration = json.loads((root / bank.OUT / "action_pool_calibration_v22.json").read_text())
    specs = {item["action_id"]: item for item in pool["candidate_pool"]}
    reliable = {item["action_id"]: item for item in calibration["by_action"] if item["selected_alpha"] is not None}
    development_ids = [value for value in pool["development_action_ids"] if value in reliable]
    original_validation = [value for value in pool["validation_action_ids"] if value in reliable]
    needed = 32 - len(original_validation)
    reserve = [item["action_id"] for item in bank._hashed(
        [specs[value] for value in development_ids], "V22-VALIDATION-RELIABILITY-RESERVE")[:max(needed, 0)]]
    validation = (original_validation + reserve)[:32]
    development_ids = [value for value in development_ids if value not in set(reserve)]
    if len(validation) != 32 or len(development_ids) < 128:
        raise RuntimeError("V22 reliability amendment cannot populate required partitions")

    directions = torch.load(root / "artifacts/causal/v13/probe_directions_v13.pt", map_location="cpu", weights_only=False)
    score = np.asarray(directions["score_directions"], dtype=np.float64)
    del directions
    features = np.stack([bank._score(specs[value], score) * float(reliable[value]["selected_alpha"])
                         for value in development_ids])
    block = features.reshape(len(features), 3, 4799)
    block_scale = np.median(np.linalg.norm(block, axis=2), axis=0)
    whitened = (block / np.maximum(block_scale[None, :, None], 1e-12)).reshape(len(features), -1)

    random_order = [item["action_id"] for item in bank._hashed(
        [specs[value] for value in development_ids], "V22-RANDOM-ARCH-BALANCED")]
    bins = {}
    for value in random_order:
        bins.setdefault(specs[value]["source_family"], []).append(value)
    random_balanced = []
    while len(random_balanced) < len(random_order):
        for family in sorted(bins):
            if bins[family]:
                random_balanced.append(bins[family].pop(0))
    maximin = bank._maximin(whitened, development_ids, len(development_ids), "V22-MAXIMIN")

    roles = json.loads((root / "artifacts/action_coordinate_geometry_v21_roles.freeze.json").read_text())
    scales = json.loads((root / "artifacts/action_coordinate_geometry_v21_probe_scales.freeze.json").read_text())
    probe_ids = [int(value) for value in roles["jvp_probe_direction_indices"]
                 if scales["selected_alpha_by_direction"][str(value)] is not None]
    columns = [roles["jvp_probe_direction_indices"].index(value) for value in probe_ids]
    response_covariance = np.zeros((len(probe_ids), len(probe_ids)), dtype=np.float64)
    count = 0
    for item in roles["jvp_development"]:
        with np.load(root / bank.V21_RAW / f"paired_{item['base_trial_id']}.npz") as source:
            for state in ("P0", "Pq"):
                response = np.asarray(source[f"{state}_JVP"][:, columns], dtype=np.float64)
                response_covariance += response.T @ response
                count += 1
    response_covariance /= count
    probe = np.stack([score[value] * float(scales["selected_alpha_by_direction"][str(value)]) for value in probe_ids])
    probe = (probe.reshape(len(probe), 3, 4799) / np.maximum(block_scale[None, :, None], 1e-12)).reshape(len(probe), -1)
    kpp, kcp = probe @ probe.T, whitened @ probe.T
    ridge = max(float(np.trace(kpp) / len(kpp)) * 1e-6, 1e-8)
    eigen, vectors = np.linalg.eigh(response_covariance)
    keep = eigen > max(float(eigen[-1]) * 1e-8, 1e-12)
    causal_root = vectors[:, keep] * np.sqrt(np.maximum(eigen[keep], 0))[None, :]
    causal_embedding = kcp @ np.linalg.solve(kpp + ridge * np.eye(len(kpp)), causal_root)
    causal_embedding /= max(float(np.median(np.linalg.norm(causal_embedding, axis=1))), 1e-12)
    inverse = np.eye(causal_embedding.shape[1]) / 1e-3
    remaining = set(range(len(development_ids)))
    causal_order = []
    while remaining:
        chosen = max(remaining, key=lambda index: (
            float(causal_embedding[index] @ inverse @ causal_embedding[index]),
            hashlib.sha256(f"V22-DOPT:{development_ids[index]}".encode()).hexdigest()))
        vector = causal_embedding[chosen]
        transformed = inverse @ vector
        inverse -= np.outer(transformed, transformed) / (1.0 + float(vector @ transformed))
        causal_order.append(development_ids[chosen])
        remaining.remove(chosen)

    common = []
    for order, quota in ((random_balanced, 43), (maximin, 43), (causal_order, 42)):
        taken = 0
        for value in order:
            if value not in common:
                common.append(value)
                taken += 1
            if taken == quota:
                break
    for value in causal_order + maximin + random_balanced:
        if len(common) >= 128:
            break
        if value not in common:
            common.append(value)
    common = common[:128]
    rankings = {}
    for name, order in (("random_architecture_balanced", random_balanced),
                        ("raw_geometry_maximin", maximin), ("causal_d_optimal", causal_order)):
        rankings[name] = [value for value in order if value in set(common)]
        rankings[name] += [value for value in common if value not in rankings[name]]
    action_alphas = {value: float(reliable[value]["selected_alpha"]) for value in common + validation}
    result = {
        "selection_strategies": list(rankings),
        "nested_counts": config["action_partitions"]["nested_counts"],
        "strategy_rankings": rankings,
        "common_measured_train_action_ids": common,
        "validation_action_ids": validation,
        "independent_final_action_ids": pool["independent_final_action_ids"],
        "action_alphas": action_alphas,
        "common_train_hash": bank._digest(common),
        "validation_hash": bank._digest(validation),
        "independent_final_hash": pool["independent_final_hash"],
        "selection_hashes": {name: bank._digest(order) for name, order in rankings.items()},
        "raw_block_scales": block_scale.tolist(),
        "causal_embedding_rank": int(causal_embedding.shape[1]),
        "causal_metric_source": "V21 development exact-JVP P0/Pq only",
        "validation_reliability_amendment": {
            "frozen_validation_reliable_count": len(original_validation),
            "reserve_action_ids": reserve,
            "reason": "four pre-response validation candidates failed frozen symmetric BF16 reliability",
            "timing": "before expanded development or validation responses",
        },
        "historical_final_responses_opened": False,
        "new_independent_final_responses_opened": False,
    }
    scratch = root / bank.SCRATCH / "action_coordinates_v22.npz"
    np.savez_compressed(scratch, development_ids=np.asarray(development_ids),
                        raw_whitened=whitened.astype(np.float32), causal_embedding=causal_embedding.astype(np.float32),
                        probe_direction_indices=np.asarray(probe_ids), response_covariance=response_covariance.astype(np.float32))
    result["coordinate_scratch_sha256"] = sha256_file(scratch)
    target = root / bank.OUT / "action_selection_v22.json"
    write_json_atomic(target, result)
    frozen = stage_freeze(root, "action_selection", [SOURCE,
        str(bank.OUT / "action_pool_v22.json"), str(bank.OUT / "action_pool_calibration_v22.json"),
        "results/v21/processed/paired_jvp_finite_operator_v21.parquet", str(target)],
        {key: result[key] for key in ("selection_strategies", "nested_counts", "common_train_hash",
                                      "validation_hash", "independent_final_hash", "selection_hashes",
                                      "coordinate_scratch_sha256", "validation_reliability_amendment",
                                      "historical_final_responses_opened", "new_independent_final_responses_opened")})
    return {"freeze_digest": frozen["freeze_digest"], "train_count": len(common),
            "validation_count": len(validation), "selection_hashes": result["selection_hashes"]}


if __name__ == "__main__":
    print(json.dumps(run(Path.cwd()), indent=2))
