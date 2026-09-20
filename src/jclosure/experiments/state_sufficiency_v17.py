"""V17 clean-state context ceiling using the frozen V16 finite-action bank.

Only *clean current* V13 cache tensors enter predictors. V13 perturbed tensors,
future endpoints, and intervention labels never enter the context kernels.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from jclosure.protocol_v17 import stage_freeze, verify
from jclosure.provenance import sha256_file, write_json_atomic

OUT = Path("results/v17/processed")
SPLITS = Path("artifacts/interventional_state_sufficiency_v17_splits.freeze.json")
SOURCE = "src/jclosure/experiments/state_sufficiency_v17.py"
KERNELS = OUT / "clean_state_kernels_v17.npz"
SCORES = OUT / "state_context_ceiling_v17.json"
PREDICTIONS = OUT / "state_context_ceiling_predictions_v17.parquet"


def _hash_ids(ids: list[str]) -> str:
    return hashlib.sha256("\n".join(sorted(ids)).encode()).hexdigest()


def prepare(root: Path) -> dict:
    base = verify(root)
    v16 = json.loads((root / "artifacts/nonlinear_finite_causal_action_v16_splits.freeze.json").read_text())
    bank = pd.read_parquet(root / base["config"]["source"]["action_bank"])
    selected = {}
    for role in ("train", "validation"):
        rows = v16[role]
        ids = [str(row["base_trial_id"]) for row in rows]
        bank_ids = set(bank.loc[bank.role == role, "base_trial_id"].astype(str))
        if set(ids) != bank_ids:
            raise RuntimeError(f"V16 {role} bank IDs do not match split freeze")
        selected[role] = rows
    if {x["base_trial_id"] for x in selected["train"]} & {x["base_trial_id"] for x in selected["validation"]}:
        raise RuntimeError("train/validation overlap")
    detail = {
        "train": selected["train"], "validation": selected["validation"],
        "train_id_sha256": _hash_ids([x["base_trial_id"] for x in selected["train"]]),
        "validation_id_sha256": _hash_ids([x["base_trial_id"] for x in selected["validation"]]),
        "independent_final": "UNOPENED_PENDING_ALL_DEVELOPMENT_GATES",
        "source_v16_split_freeze_digest": v16["freeze_digest"],
        "reused_v16_responses": True,
        "no_v13_final_or_v14_v15_development_as_final": True,
    }
    return stage_freeze(root, "splits", [SOURCE, "artifacts/nonlinear_finite_causal_action_v16_splits.freeze.json", "results/v16/processed/finite_action_bank_v16.parquet"], detail)


def _split(root: Path) -> dict:
    from jclosure.protocol_v17 import digest
    base = verify(root)
    value = json.loads((root / SPLITS).read_text())
    if value["freeze_digest"] != digest(value) or value["base_freeze_digest"] != base["freeze_digest"]:
        raise RuntimeError("V17 split freeze invalid")
    amendment = None
    parent_digest = value["freeze_digest"]
    for number in (1, 2):
        amendment_path = root / f"artifacts/interventional_state_sufficiency_v17_source_amendment_{number}.freeze.json"
        if not amendment_path.exists():
            break
        amendment = json.loads(amendment_path.read_text())
        if amendment["freeze_digest"] != digest(amendment) or amendment["parent_split_freeze_digest"] != parent_digest:
            raise RuntimeError("V17 source amendment invalid")
        parent_digest = amendment["freeze_digest"]
    for path, expected in value["input_hashes"].items():
        if path == SOURCE and amendment is not None:
            expected = amendment["input_hashes"][SOURCE]
        if sha256_file(root / path) != expected:
            raise RuntimeError(f"V17 split input changed: {path}")
    return value


def _load_clean_states(root: Path, split: dict) -> tuple[dict[str, np.ndarray], list[str], list[str], list[int], list[str]]:
    rows = split["train"] + split["validation"]
    ids = [x["base_trial_id"] for x in rows]
    families = [x["family"] for x in rows]
    roles = [x["role"] for x in rows]
    wanted = set(ids)
    state = {}
    prompt_len = {}
    for role in ("train", "validation"):
        manifest = json.loads((root / f"results/v13/processed/causal_capture_{role}_v13.json").read_text())
        for shard in manifest["state_shards"]:
            if not wanted.intersection(shard["base_trial_ids"]):
                continue
            path = root / shard["path"]
            if sha256_file(path) != shard["sha256"]:
                raise RuntimeError(f"V13 source state hash mismatch: {path}")
            payload = torch.load(path, map_location="cpu", weights_only=False)
            for row in payload["rows"]:
                key = str(row["base_trial_id"])
                if key not in wanted:
                    continue
                clean = row["clean"]
                kv_blocks = []
                for layer in ("27", "31"):
                    for name in ("keys", "values"):
                        tensor = clean["kv"][layer][name]
                        if tensor.shape[-2] > 256:
                            raise RuntimeError("V17 KV token width exceeds frozen 256")
                        padded = torch.zeros((tensor.shape[0], 256, tensor.shape[-1]), dtype=torch.float32)
                        padded[:, :tensor.shape[-2], :] = tensor.float()
                        kv_blocks.append(padded.flatten())
                state[key] = {
                    "rec": clean["recurrent"].flatten().float().numpy().astype(np.float16),
                    "conv": clean["conv"].flatten().float().numpy().astype(np.float16),
                    "kv": torch.cat(kv_blocks).numpy().astype(np.float16),
                }
                prompt_len[key] = int(row["prompt_length"])
    if set(state) != wanted:
        raise RuntimeError(f"Missing {len(wanted - set(state))} selected clean states")
    endpoints = {}
    for role in ("train", "validation"):
        manifest = json.loads((root / f"results/v13/processed/causal_capture_{role}_v13.json").read_text())
        with np.load(root / manifest["endpoint_artifact"], allow_pickle=False) as payload:
            source_ids = payload["base_trial_id"].astype(str)
            current = payload["current_j_clean"]
            for i, key in enumerate(source_ids):
                if key in wanted:
                    endpoints[key] = current[i].astype(np.float16)
    if set(endpoints) != wanted:
        raise RuntimeError("Missing current clean J endpoint")
    arrays = {name: np.stack([state[key][name] for key in ids]) for name in ("rec", "conv", "kv")}
    arrays["j"] = np.stack([endpoints[key] for key in ids])
    return arrays, ids, families, [prompt_len[key] for key in ids], roles


def _gram(x: np.ndarray, train_count: int) -> np.ndarray:
    device = "cuda:1" if torch.cuda.device_count() > 1 else ("cuda:0" if torch.cuda.is_available() else "cpu")
    width = x.shape[1]
    result = np.zeros((len(x), len(x)), dtype=np.float64)
    # Feature chunks keep the large REC tensor below GPU/CPU memory limits.
    for start in range(0, width, 65536):
        chunk = torch.from_numpy(x[:, start : start + 65536].copy()).to(device=device, dtype=torch.float32)
        result += (chunk @ chunk.T).cpu().numpy().astype(np.float64)
        del chunk
    # Train-only centering of the full raw linear kernel; validation never fits.
    tr_mean_col = result[:train_count].mean(axis=0, keepdims=True)
    tr_mean_all = float(result[:train_count, :train_count].mean())
    result = result - tr_mean_col - tr_mean_col.T + tr_mean_all
    scale = float(np.diag(result[:train_count, :train_count]).mean())
    if scale <= 0:
        raise RuntimeError("Degenerate clean-state kernel")
    return (result / scale).astype(np.float32)


def build_kernels(root: Path) -> dict:
    split = _split(root)
    arrays, ids, families, prompt_lengths, roles = _load_clean_states(root, split)
    n_train = len(split["train"])
    kernels = {name: _gram(value, n_train) for name, value in arrays.items()}
    target = root / KERNELS
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(target, **kernels, base_trial_id=np.asarray(ids), family=np.asarray(families),
                        prompt_length=np.asarray(prompt_lengths), role=np.asarray(roles))
    summary = {
        "clean_source_only": True, "raw_channels": ["rec", "conv", "kv"],
        "v13_perturbed_states_used": False, "teacher_future_used_as_context": False,
        "state_count": len(ids), "train_state_count": n_train,
        "validation_state_count": len(ids) - n_train,
        "source_tensor_shapes": {name: list(value.shape) for name, value in arrays.items()},
        "kernel_artifact": str(KERNELS), "kernel_sha256": sha256_file(target),
        "split_freeze_digest": split["freeze_digest"],
    }
    write_json_atomic(root / OUT / "clean_state_kernels_v17.json", summary)
    return summary


def _kernel_set(data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    j = data["j"]
    raw = {name: data[name] for name in ("rec", "conv", "kv")}
    out = {"action_only": np.zeros_like(j), "j": j}
    for names in (("rec",), ("conv",), ("kv",), ("rec", "conv"), ("rec", "kv"), ("conv", "kv"), ("rec", "conv", "kv")):
        label = "+".join(names)
        combined = np.mean([raw[name] for name in names], axis=0)
        out[f"j+{label}"] = 0.5 * j + 0.5 * combined
    out["full_raw_reference"] = np.mean([raw[name] for name in raw], axis=0)
    return out


def _predict_key(k: np.ndarray, train_state: np.ndarray, test_state: np.ndarray,
                 y_train: np.ndarray, ridge: float) -> np.ndarray:
    if len(train_state) == 0:
        raise RuntimeError("Action key has no train rows")
    mean = y_train.mean(axis=0, keepdims=True)
    if not np.any(k):
        return np.repeat(mean, len(test_state), axis=0)
    kt = k[np.ix_(train_state, train_state)].astype(np.float64)
    pred = k[np.ix_(test_state, train_state)].astype(np.float64)
    weights = np.linalg.solve(kt + ridge * np.eye(len(train_state)), y_train - mean)
    return (mean + pred @ weights).astype(np.float32)


def _score(actual: np.ndarray, predicted: np.ndarray) -> dict:
    a = actual.astype(np.float64)
    p = predicted.astype(np.float64)
    error = np.linalg.norm(a - p, axis=1)
    norm = np.linalg.norm(a, axis=1)
    cos = (a * p).sum(axis=1) / np.maximum(norm * np.linalg.norm(p, axis=1), 1e-12)
    return {"relative_l2": float(np.linalg.norm(a - p) / max(np.linalg.norm(a), 1e-12)),
            "direction_median": float(np.median(cos)),
            "magnitude_ratio_median": float(np.median(np.linalg.norm(p, axis=1) / np.maximum(norm, 1e-12))),
            "n": len(actual)}


def ceiling(root: Path) -> dict:
    split = _split(root)
    base = verify(root)
    cfg = base["config"]
    with np.load(root / KERNELS, allow_pickle=False) as payload:
        data = {name: payload[name] for name in payload.files}
    ids = data["base_trial_id"].astype(str).tolist()
    roles = data["role"].astype(str)
    id_to_index = {key: i for i, key in enumerate(ids)}
    kernels = _kernel_set(data)
    bank = pd.read_parquet(root / cfg["source"]["action_bank"])
    bank = bank[(bank.design == "single") & (bank.reliability_status.str.lower() == cfg["source"]["action_reliability"])].copy()
    bank = bank[bank.base_trial_id.isin(id_to_index)]
    bank["state_index"] = bank.base_trial_id.map(id_to_index).astype(int)
    bank["action_key"] = bank.coordinate_index.astype(int).astype(str) + ":" + bank.sign.astype(int).astype(str)
    bank = bank[bank.calibration_alpha.astype(float) > 0]
    train = bank[bank.role == "train"]
    val = bank[bank.role == "validation"]
    keys = sorted(set(train.action_key) & set(val.action_key))
    min_states = int(cfg["design"]["response_min_train_states_per_action"])
    keys = [key for key in keys if len(train[train.action_key == key]) >= min_states]
    if not keys:
        raise RuntimeError("No eligible reliable single-action keys")
    train_folds = {i: int(hashlib.sha256(f"{cfg['design']['cv_seed']}:{ids[i]}".encode()).hexdigest(), 16) % 5
                   for i in np.flatnonzero(roles == "train")}
    # Lambda is selected once on training states for the J baseline and then
    # held fixed for every architecture comparison on untouched validation.
    cv = []
    for ridge in cfg["design"]["ridge_lambda_grid"]:
        truth, preds = [], []
        for key in keys:
            part = train[train.action_key == key]
            state_idx = part.state_index.to_numpy()
            targets = np.stack(part.response_stacked_normalized.to_numpy()).astype(np.float32)
            alpha = part.calibration_alpha.to_numpy(dtype=np.float32)
            for fold in range(5):
                fit = np.asarray([train_folds[int(i)] != fold for i in state_idx])
                hold = ~fit
                if not hold.any() or fit.sum() < min_states:
                    continue
                prediction = _predict_key(kernels["j"], state_idx[fit], state_idx[hold],
                                          targets[fit] / alpha[fit, None], float(ridge)) * alpha[hold, None]
                truth.append(targets[hold])
                preds.append(prediction)
        metric = _score(np.concatenate(truth), np.concatenate(preds))["relative_l2"]
        cv.append({"ridge": float(ridge), "train_oof_stacked_relative_l2": metric})
    chosen = min(cv, key=lambda x: x["train_oof_stacked_relative_l2"])["ridge"]
    records = []
    prediction_rows = []
    for name, kernel in kernels.items():
        actual_j, pred_j, actual_stack, pred_stack, state_list, key_list, family_list = [], [], [], [], [], [], []
        for key in keys:
            tr = train[train.action_key == key]
            va = val[val.action_key == key]
            if va.empty:
                continue
            tr_idx = tr.state_index.to_numpy()
            va_idx = va.state_index.to_numpy()
            tr_alpha = tr.calibration_alpha.to_numpy(dtype=np.float32)
            va_alpha = va.calibration_alpha.to_numpy(dtype=np.float32)
            for target, actual, predicted in (("j", actual_j, pred_j), ("stacked_normalized", actual_stack, pred_stack)):
                y_train = np.stack(tr[f"response_{target}"].to_numpy()).astype(np.float32)
                y_val = np.stack(va[f"response_{target}"].to_numpy()).astype(np.float32)
                y_pred = _predict_key(kernel, tr_idx, va_idx, y_train / tr_alpha[:, None], chosen) * va_alpha[:, None]
                actual.extend(y_val)
                predicted.extend(y_pred)
            state_list.extend(va.base_trial_id.astype(str).tolist())
            key_list.extend([key] * len(va))
            family_list.extend(va.family.astype(str).tolist())
        aj, pj = np.stack(actual_j), np.stack(pred_j)
        ast, pst = np.stack(actual_stack), np.stack(pred_stack)
        metrics = {"context": name, "ridge": chosen, "j": _score(aj, pj),
                   "stacked_normalized": _score(ast, pst), "by_family": {}}
        state_array = np.asarray(state_list)
        family_array = np.asarray(family_list)
        for family in sorted(set(family_list)):
            mask = family_array == family
            metrics["by_family"][family] = {"j": _score(aj[mask], pj[mask]), "stacked_normalized": _score(ast[mask], pst[mask])}
        records.append(metrics)
        for i, state_id in enumerate(state_list):
            prediction_rows.append({"context": name, "base_trial_id": state_id, "family": family_list[i],
                                    "action_key": key_list[i], "j_actual": aj[i].tolist(), "j_predicted": pj[i].tolist(),
                                    "stack_actual": ast[i].tolist(), "stack_predicted": pst[i].tolist()})
    frame = pd.DataFrame(prediction_rows)
    frame.to_parquet(root / PREDICTIONS, index=False)
    by_name = {row["context"]: row for row in records}
    raw_gain = {target: by_name["j"][target]["relative_l2"] - by_name["j+rec+conv+kv"][target]["relative_l2"]
                for target in ("j", "stacked_normalized")}
    # Paired bootstrap by validation state, preserving all action rows per state.
    rng = np.random.default_rng(cfg["conditional_test"]["bootstrap_seed"])
    val_ids = sorted(set(frame.base_trial_id))
    bootstrap = {}
    for target, actual_col, predicted_col in (("j", "j_actual", "j_predicted"), ("stacked_normalized", "stack_actual", "stack_predicted")):
        a = frame[frame.context == "j"].reset_index(drop=True)
        b = frame[frame.context == "j+rec+conv+kv"].reset_index(drop=True)
        if not (a.base_trial_id.equals(b.base_trial_id) and a.action_key.equals(b.action_key)):
            raise RuntimeError("paired ceiling alignment failed")
        actual = np.stack(a[actual_col].to_numpy())
        pj = np.stack(a[predicted_col].to_numpy())
        pr = np.stack(b[predicted_col].to_numpy())
        index_by_id = {key: np.flatnonzero(a.base_trial_id.to_numpy() == key) for key in val_ids}
        draws = []
        for _ in range(int(cfg["conditional_test"]["bootstrap_replicates"])):
            sample = rng.choice(val_ids, len(val_ids), replace=True)
            idx = np.concatenate([index_by_id[key] for key in sample])
            draws.append(_score(actual[idx], pj[idx])["relative_l2"] - _score(actual[idx], pr[idx])["relative_l2"])
        bootstrap[target] = {"gain_ci_95": np.quantile(draws, [0.025, 0.975]).tolist(),
                             "gain_median": float(np.median(draws))}
    gate = float(cfg["gates"]["material_raw_ceiling_relative_l2_gain_min"])
    material = all(raw_gain[t] >= gate and bootstrap[t]["gain_ci_95"][0] > 0 for t in raw_gain)
    result = {"split_freeze_digest": split["freeze_digest"], "kernel_sha256": sha256_file(root / KERNELS),
              "action_design": "reliable_v16_single_only", "action_key_count": len(keys),
              "validation_action_rows": len(frame[frame.context == "j"]),
              "train_only_ridge_cv": cv, "selected_ridge": chosen, "results": records,
              "raw_joint_gain_over_j": raw_gain, "bootstrap": bootstrap,
              "material_raw_context_gate_passed": material,
              "gate_requires_both_j_and_stack_absolute_rel_l2_gain": gate,
              "other_action_designs_and_horizons": "NOT_EVALUATED_BY_CEILING_STAGE",
              "independent_final": "UNOPENED"}
    write_json_atomic(root / SCORES, result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("prepare", "kernels", "ceiling"))
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    root = args.root.resolve()
    value = {"prepare": prepare, "kernels": build_kernels, "ceiling": ceiling}[args.stage](root)
    print(json.dumps(value, indent=2, default=str))
