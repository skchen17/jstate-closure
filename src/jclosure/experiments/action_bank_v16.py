"""V16 finite-action bank: disjoint states, frozen designs, real BF16 writeback."""

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
from scipy.stats import qmc

from jclosure.config import load_config
from jclosure.experiments import geometry_v13 as geometry
from jclosure.experiments.actuation_v15 import apply, components, load_context, transfer_metrics
from jclosure.experiments.numerics_v14 import _evaluate
from jclosure.experiments.operator_v15 import TARGETS, cosine, stack, target_scales
from jclosure.experiments.persistent_channels_v7 import _prefill, _teacher_tokens
from jclosure.protocol_v16 import stage_freeze, verify
from jclosure.provenance import sha256_file, write_json_atomic

OUT = Path("results/v16/processed")
RAW = Path("results/v16/raw")
SOURCE = "src/jclosure/experiments/action_bank_v16.py"
SPLITS = Path("artifacts/nonlinear_finite_causal_action_v16_splits.freeze.json")


def _hash_ids(ids: list[str]) -> str:
    return hashlib.sha256("\n".join(sorted(ids)).encode()).hexdigest()


def _selected_directions(root: Path, count: dict[str, int]) -> tuple[list[int], list[str]]:
    prior = json.loads((root / "results/v13/processed/probe_directions_v13.json").read_text())
    pools = prior["probe_families"]
    selected: list[int] = []
    labels: list[str] = []

    def add(index: int, label: str) -> bool:
        if index in selected:
            return False
        selected.append(index)
        labels.append(label)
        return True

    for label in ("causal_weighted", "architecture_balanced", "random_raw"):
        for index in pools[label][: count[label]]:
            add(int(index), label)
    finite = pd.read_parquet(root / "results/v15/processed/finite_operator_columns_v15.parquet")
    joint = finite[finite.channel == "joint"]
    score = joint.groupby("direction_index").j_response_norm.median().sort_values(ascending=False)
    for index in score.index:
        if sum(x == "high_finite_response" for x in labels) >= count["high_finite_response"]:
            break
        add(int(index), "high_finite_response")
    # Residual to the *autograd response subspace* on a frozen V15 train anchor.
    # This is proposal selection, never a new V16 validation observation.
    anchor = str(joint.base_trial_id.iloc[0])
    matrix_row = pd.read_parquet(root / "results/v13/processed/causal_probe_scaling_v13.parquet")
    matrix_row = matrix_row[(matrix_row.base_trial_id == anchor) & (matrix_row.probe_family == "mixed") & (matrix_row.probe_direction_count == 512) & (matrix_row.token_position == 0)]
    with np.load(root / str(matrix_row.iloc[0].matrix_path), allow_pickle=False) as payload:
        from jclosure.experiments.numerics_v14 import _slices

        sl = _slices(payload)["j"]
        ideal = payload["matrix"][sl, :].astype(np.float64)
    u, _, _ = np.linalg.svd(ideal, full_matrices=False)
    basis = u[:, :8]
    residuals: list[tuple[float, int]] = []
    for _, record in joint[joint.base_trial_id == anchor].iterrows():
        response = np.asarray(record.j_response, dtype=np.float64)
        residual = response - basis @ (basis.T @ response)
        residuals.append((float(np.linalg.norm(residual)), int(record.direction_index)))
    for _, index in sorted(residuals, reverse=True):
        if sum(x == "residual_finite_response" for x in labels) >= count["residual_finite_response"]:
            break
        add(index, "residual_finite_response")
    if len(selected) != sum(count.values()):
        raise RuntimeError(f"direction selection insufficient: {len(selected)}")
    # Nested candidate k must include all five proposal families early.
    order = sorted(range(len(selected)), key=lambda i: (sum(labels[j] == labels[i] for j in range(i)), labels[i]))
    return [selected[i] for i in order], [labels[i] for i in order]


def prepare(root: Path) -> dict[str, Any]:
    base = verify(root)
    cfg = base["config"]
    data = geometry._load_features(root)
    ids = data["base_trial_id"].astype(str)
    families = data["family"].astype(str)
    splits = data["split"].astype(str)
    prior = json.loads((root / "artifacts/quantization_aware_actuation_v15_splits.freeze.json").read_text())
    excluded_validation = set(prior["development_validation"]["base_trial_ids"])
    selected: dict[str, list[dict[str, str]]] = {}
    for role, source, number in (
        ("train", "train", int(cfg["roles"]["train_per_family"])),
        ("validation", "validation", int(cfg["roles"]["validation_per_family"])),
    ):
        rows = []
        for family in sorted(set(families)):
            candidates = [str(ids[i]) for i in np.flatnonzero((splits == source) & (families == family))]
            if role == "validation" and cfg["roles"]["exclude_prior_v15_development_from_validation"]:
                candidates = [item for item in candidates if item not in excluded_validation]
            candidates.sort(key=lambda item: hashlib.sha256(f"{cfg['roles']['seed']}:{item}".encode()).hexdigest())
            if len(candidates) < number:
                raise RuntimeError(f"not enough {role} states for {family}: {len(candidates)}")
            rows.extend({"base_trial_id": item, "family": family, "role": role} for item in candidates[:number])
        selected[role] = rows
    train_ids = [row["base_trial_id"] for row in selected["train"]]
    validation_ids = [row["base_trial_id"] for row in selected["validation"]]
    if set(train_ids) & set(validation_ids):
        raise RuntimeError("V16 train/validation overlap")
    direction_indices, direction_families = _selected_directions(root, cfg["action"]["proposal_family_counts"])
    rows = pd.read_parquet(root / "results/v13/processed/causal_probe_scaling_v13.parquet")
    reference = rows[(rows.probe_family == "mixed") & (rows.probe_direction_count == 512) & (rows.token_position == 0)].sort_values("base_trial_id").iloc[0]
    with np.load(root / str(reference.matrix_path), allow_pickle=False) as payload:
        j_ids = payload["selected_j"].astype(int).tolist()
        logit_ids = payload["selected_logits"].astype(int).tolist()
    scales = target_scales(root, rows[(rows.probe_family == "mixed") & (rows.probe_direction_count == 512) & (rows.token_position == 0)].sort_values("base_trial_id").groupby("family", sort=True).head(1).to_dict("records"))
    detail = {
        "role": "train_and_validation_pre_response",
        "train": selected["train"], "validation": selected["validation"],
        "train_id_sha256": _hash_ids(train_ids), "validation_id_sha256": _hash_ids(validation_ids),
        "independent_final": "NOT_CREATED_OR_OPENED_PENDING_RESPONSE_AND_CONTROL_GATES",
        "direction_indices": direction_indices, "direction_families": direction_families,
        "selected_j": j_ids, "selected_logits": logit_ids, "target_scales": scales,
        "frozen_design": cfg["action"], "frozen_response_gate": cfg["response_gate"],
    }
    result = stage_freeze(root, "splits", [SOURCE, "results/v13/processed/probe_directions_v13.json", "results/v15/processed/finite_operator_columns_v15.parquet", "artifacts/quantization_aware_actuation_v15_splits.freeze.json"], detail)
    return result


def _verify_splits(root: Path) -> dict[str, Any]:
    base = verify(root)
    value = json.loads((root / SPLITS).read_text())
    from jclosure.protocol_v16 import digest

    if value["freeze_digest"] != digest(value) or value["base_freeze_digest"] != base["freeze_digest"]:
        raise RuntimeError("V16 split freeze invalid")
    amendment_path = root / "artifacts/nonlinear_finite_causal_action_v16_bank_source_amendment_1.freeze.json"
    amendment = None
    if amendment_path.exists():
        amendment = json.loads(amendment_path.read_text())
        if amendment["freeze_digest"] != digest(amendment) or amendment["parent_split_freeze_digest"] != value["freeze_digest"]:
            raise RuntimeError("V16 bank-source amendment invalid")
    for name, expected in value["input_hashes"].items():
        if name == SOURCE and amendment is not None:
            expected = amendment["input_hashes"][SOURCE]
        if sha256_file(root / name) != expected:
            raise RuntimeError(f"V16 frozen input changed: {name}")
    return value


def _readback(cache: Any, edited: Any, row: dict[str, torch.Tensor], rec: list[int], att: list[int]) -> dict[str, Any]:
    requested_sq = realized_sq = dot = 0.0
    channel: dict[str, list[float]] = {name: [] for name in ("recurrent", "conv", "keys", "values")}
    for name, layer, attr, old, direction in components(cache, row, rec, att):
        new = getattr(edited.layers[layer], attr)
        if name in ("keys", "values"):
            new = new[..., : old.shape[-2], :]
        metrics = transfer_metrics(old, direction, new, 1.0)
        a, b = float(metrics["requested_norm"]), float(metrics["realized_norm"])
        requested_sq += a * a
        realized_sq += b * b
        dot += float(metrics["cosine"] or 0.0) * a * b
        channel[name].append(float(metrics["surviving_fraction"]))
    a, b = math.sqrt(requested_sq), math.sqrt(realized_sq)
    return {"requested_state_norm": a, "realized_state_norm": b,
            "realized_state_gain": b / a if a else None,
            "realized_state_cosine": dot / (a * b) if a * b else None,
            "channel_survival": {name: float(np.mean(values)) if values else None for name, values in channel.items()}}


def _row_for_z(directions: dict[str, Any], indices: list[int], alpha: np.ndarray, z: np.ndarray) -> dict[str, torch.Tensor]:
    active = np.flatnonzero(z)
    if not len(active):
        raise ValueError("zero action is not measured as a finite response")
    if len(active) == 1:
        i = int(active[0])
        return {name: directions[name][indices[i]].float() * float(alpha[i] * z[i]) for name in ("recurrent", "conv", "kv")}
    weights = torch.as_tensor((alpha * z)[active], dtype=torch.float32)
    chosen = [indices[int(i)] for i in active]
    return {name: torch.tensordot(weights, directions[name][chosen].float(), dims=([0], [0])) for name in ("recurrent", "conv", "kv")}


def _measure(cache: Any, kwargs: dict[str, Any], baseline: dict[str, np.ndarray], directions: dict[str, Any], indices: list[int], alpha: np.ndarray, z: np.ndarray, rec: list[int], att: list[int], scales: dict[str, float]) -> dict[str, Any]:
    row = _row_for_z(directions, indices, alpha, z)
    edited = apply(cache, 1.0, row, rec, att, "native_fp32_add_bf16_writeback")
    realized = _readback(cache, edited, row, rec, att)
    response = _evaluate(cache=edited, **kwargs)
    delta = {name: (response[name] - baseline[name]).astype(np.float32) for name in TARGETS}
    delta["stacked_normalized"] = stack(delta, scales).astype(np.float32)
    return {**realized, **{f"response_{name}": delta[name].tolist() for name in (*TARGETS, "stacked_normalized")},
            "j_effect_norm": float(np.linalg.norm(delta["j"])), "requested_z": z.astype(np.float32).tolist(),
            "requested_alpha": alpha.astype(np.float32).tolist()}


def _design(cfg: dict[str, Any], split: str, state_index: int, family_index: int) -> list[tuple[str, np.ndarray]]:
    width = int(cfg["proposal_count"])
    output: list[tuple[str, np.ndarray]] = []
    if state_index >= int(cfg[f"design_{split}_extra_states_per_family"]):
        return output
    for index in cfg["scale_indices"]:
        for multiplier in cfg["scale_multipliers"]:
            z = np.zeros(width, dtype=np.float32); z[int(index)] = float(multiplier)
            if abs(multiplier) != 1.0:  # the ±1 singles already appear in the base bank
                output.append(("scale", z))
    for pair in cfg["pair_indices"]:
        for signs in cfg["pair_signs"]:
            z = np.zeros(width, dtype=np.float32)
            z[int(pair[0])], z[int(pair[1])] = signs
            output.append(("pair", z))
    for triple in cfg["triples"]:
        for last_sign in (-1.0, 1.0):
            z = np.zeros(width, dtype=np.float32)
            z[[int(x) for x in triple]] = [1.0, 1.0, last_sign]
            output.append(("triple", z))
    sobol = qmc.Sobol(d=4, scramble=False)
    sequence = sobol.random_base2(m=4)
    dense_added = 0
    for point in sequence:
        z = np.zeros(width, dtype=np.float32)
        z[:4] = np.asarray(2.0 * point - 1.0, dtype=np.float32)
        if not np.any(z):
            continue  # Sobol centre is a null intervention, not a finite action.
        output.append(("dense", z))
        dense_added += 1
        if dense_added == int(cfg["dense_sobol_count"]):
            break
    return output


def run(root: Path, split: str, limit: int | None = None) -> dict[str, Any]:
    declaration = _verify_splits(root)
    cfg = verify(root)["config"]
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    bundle, dense_map, v13, metadata, values = load_context(root)
    directions = values["directions"]
    v8 = v13["persistent_state_v8"]
    measured = [int(x) for x in v8["intervention"]["measured_layers"]]
    rec = [int(x) for x in v8["persistent_components"]["recurrent_layers"]]
    att = [int(x) for x in v8["persistent_components"]["attention_layers"]]
    jvp = v13["causal_geometry_v13"]["jvp"]
    indices = [int(x) for x in declaration["direction_indices"]]
    scales = {name: float(x) for name, x in declaration["target_scales"].items()}
    ids = declaration[split]
    if limit is not None:
        ids = ids[:limit]
    RAW.mkdir(parents=True, exist_ok=True)
    for state_index, item in enumerate(ids):
        base_id = item["base_trial_id"]
        path = root / RAW / f"bank_{split}_{base_id}.parquet"
        if path.exists():
            continue
        task = values["tasks"][str(metadata[base_id]["prompt_id"])]
        clean = _prefill(bundle, task.prompt, measured_layers=measured, dense_map=dense_map,
                         intervention_layer=int(v8["intervention"]["layer"]), candidate=None)
        cache = clean["cache"]
        token = _teacher_tokens(bundle, clean, count=1, measured_layers=measured, dense_map=dense_map)[0]
        kwargs = dict(bundle=bundle, dense_map=dense_map, token=token,
                      prompt_length=int(clean["prompt_length"]),
                      selected_j=np.asarray(declaration["selected_j"], dtype=int),
                      selected_logits=np.asarray(declaration["selected_logits"], dtype=int),
                      workspace_layers=[int(x) for x in jvp["workspace_layers"]],
                      workspace_count=int(jvp["selected_workspace_count"]), main_layer=max(measured))
        baseline = _evaluate(cache=cache, **kwargs)
        alpha = np.zeros(len(indices), dtype=np.float32)
        records: list[dict[str, Any]] = []
        for i in range(len(indices)):
            candidate = np.zeros(len(indices), dtype=np.float32); candidate[i] = 1.0
            last = None
            for scale in cfg["action"]["calibration_scales"]:
                trial_alpha = alpha.copy(); trial_alpha[i] = float(scale)
                plus = _measure(cache, kwargs, baseline, directions, indices, trial_alpha, candidate, rec, att, scales)
                minus = _measure(cache, kwargs, baseline, directions, indices, trial_alpha, -candidate, rec, att, scales)
                last = (plus, minus, float(scale))
                state_ok = all(x["realized_state_cosine"] is not None and x["realized_state_cosine"] >= cfg["action"]["minimum_state_cosine"] and x["realized_state_gain"] is not None and cfg["action"]["minimum_state_gain"] <= x["realized_state_gain"] <= cfg["action"]["maximum_state_gain"] for x in (plus, minus))
                effect_ok = min(plus["j_effect_norm"], minus["j_effect_norm"]) >= cfg["action"]["minimum_signed_effect_norm"]
                if state_ok and effect_ok:
                    alpha[i] = float(scale)
                    break
            if last is None:
                raise RuntimeError("empty calibration grid")
            for sign, observation in ((1, last[0]), (-1, last[1])):
                records.append({"base_trial_id": base_id, "family": item["family"], "role": split,
                                "design": "single", "direction_index": indices[i], "coordinate_index": i,
                                "proposal_family": declaration["direction_families"][i], "sign": sign,
                                "reliability_status": "RELIABLE" if alpha[i] else "ACTION_NOT_RELIABLY_ACTUATABLE",
                                "calibration_alpha": float(last[2]), "freeze_digest": declaration["freeze_digest"], **observation})
        local_index = state_index % 20 if split == "train" else state_index % 10
        for design, z in _design(cfg["action"], split, local_index, state_index // (20 if split == "train" else 10)):
            if any(alpha[i] <= 0 for i in np.flatnonzero(z)):
                records.append({"base_trial_id": base_id, "family": item["family"], "role": split,
                                "design": design, "direction_index": None, "coordinate_index": None,
                                "proposal_family": "mixed", "sign": None,
                                "reliability_status": "ACTION_NOT_RELIABLY_ACTUATABLE",
                                "freeze_digest": declaration["freeze_digest"], "requested_z": z.tolist(),
                                "requested_alpha": alpha.tolist()})
                continue
            observation = _measure(cache, kwargs, baseline, directions, indices, alpha, z, rec, att, scales)
            reliable = observation["realized_state_cosine"] is not None and observation["realized_state_cosine"] >= cfg["action"]["minimum_state_cosine"] and observation["realized_state_gain"] is not None and cfg["action"]["minimum_state_gain"] <= observation["realized_state_gain"] <= cfg["action"]["maximum_state_gain"] and observation["j_effect_norm"] >= cfg["action"]["minimum_signed_effect_norm"]
            records.append({"base_trial_id": base_id, "family": item["family"], "role": split,
                            "design": design, "direction_index": None, "coordinate_index": None,
                            "proposal_family": "mixed", "sign": None,
                            "reliability_status": "RELIABLE" if reliable else "ACTION_NOT_RELIABLY_ACTUATABLE",
                            "freeze_digest": declaration["freeze_digest"], **observation})
        pd.DataFrame(records).to_parquet(path, index=False, compression="zstd")
        write_json_atomic(root / OUT / f"progress_{split}_v16.json", {
            "split": split, "completed_states": state_index + 1, "total_states": len(declaration[split]),
            "last_state": base_id, "last_records_sha256": sha256_file(path),
        })
        print(f"{split} {state_index + 1}/{len(declaration[split])} {base_id} records={len(records)}", flush=True)
    paths = [root / RAW / f"bank_{split}_{x['base_trial_id']}.parquet" for x in declaration[split]]
    return {"complete": all(path.exists() for path in paths), "finished": sum(path.exists() for path in paths), "total": len(paths)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("prepare", "amend", "train", "validation", "verify"), required=True)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    root = Path.cwd()
    if args.stage == "prepare":
        print(prepare(root)["freeze_digest"])
    elif args.stage == "amend":
        prior = json.loads((root / SPLITS).read_text())
        existing = list((root / RAW).glob("bank_*.parquet")) if (root / RAW).exists() else []
        if existing:
            raise RuntimeError("source amendment requires no completed V16 bank records")
        result = stage_freeze(root, "bank_source_amendment_1", [SOURCE, str(SPLITS)], {
            "parent_split_freeze_digest": prior["freeze_digest"],
            "old_source_sha256": prior["input_hashes"][SOURCE],
            "reason": "frozen deterministic Sobol design included an all-zero vector; first state stopped before any bank record was written",
            "change": "skip null Sobol vector and take the first eight nonzero vectors from the same deterministic sequence",
            "prior_v16_bank_records": 0,
        })
        print(result["freeze_digest"])
    elif args.stage == "verify":
        print(_verify_splits(root)["freeze_digest"])
    else:
        print(json.dumps(run(root, args.stage, args.limit), sort_keys=True))


if __name__ == "__main__":
    main()
