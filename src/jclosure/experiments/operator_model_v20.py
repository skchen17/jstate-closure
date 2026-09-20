"""Train-action-only oracle operator coordinates and cross-action decoder sweep."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from jclosure.experiments import operator_bank_v20 as bank
from jclosure.protocol_v20 import stage_freeze, verify, verify_stage
from jclosure.provenance import sha256_file, write_json_atomic

SOURCE = "src/jclosure/experiments/operator_model_v20.py"
DESCRIPTORS = bank.OUT / "action_descriptors_v20.json"
SUMMARY = bank.OUT / "oracle_operator_search_v20.json"
BLOCKS = {"j": slice(0, 128), "logits": slice(128, 160),
          "semantic_continuous": slice(160, 192), "workspace": slice(192, 288),
          "stacked_normalized": slice(0, 288)}


def _descriptors(root: Path, actions: dict) -> dict:
    directions = torch.load(root / "artifacts/causal/v13/probe_directions_v13.pt", map_location="cpu", weights_only=False)
    rows = actions["actions"]
    anchors = actions["partitions"]["train"]
    all_coords = sorted({x["coordinate_index"] for x in rows})
    parts = {}
    norms = {}
    for row in rows:
        coord = int(row["coordinate_index"])
        idx = int(row["direction_index"])
        vectors = [directions[name][idx].float().reshape(-1) for name in ("recurrent", "conv", "kv")]
        parts[coord] = vectors
        by_channel = [float(torch.linalg.vector_norm(x)) for x in vectors]
        norms[coord] = by_channel + [math.sqrt(sum(x*x for x in by_channel))]
    train_coords = [int(x["coordinate_index"]) for x in anchors]
    raw = {}
    for coord in all_coords:
        cosines = []
        for anchor in train_coords:
            dot = sum(float(torch.dot(a, b)) for a, b in zip(parts[coord], parts[anchor], strict=True))
            cosines.append(dot / max(norms[coord][-1] * norms[anchor][-1], 1e-12))
        raw[coord] = cosines + norms[coord]
    base = {int(x["coordinate_index"]): float(x["base_alpha"]) for x in rows}
    train_matrix = np.stack([np.asarray(raw[coord]) * base[coord] for coord in train_coords])
    scale = np.maximum(np.max(np.abs(train_matrix), axis=0), 1e-6)
    encoded = {str(coord): np.clip(np.asarray(raw[coord]) * base[coord] / scale, -5, 5).tolist()
               for coord in all_coords}
    return {"train_anchor_coordinates": train_coords, "action_coordinates": all_coords,
            "raw_descriptor": "full_raw_REC_Conv_KV_cosines_to_12_train_anchors_plus_4_channel_norms",
            "raw_direction_file_sha256": sha256_file(root / "artifacts/causal/v13/probe_directions_v13.pt"),
            "train_only_normalization": scale.tolist(), "clip_abs": 5.0,
            "positive_base_scale_z": encoded, "dimension": len(next(iter(encoded.values()))),
            "negative_sign_rule": "z(-a)=-z(a)", "amplitude_rule": "z(beta*a)=beta*z(a)",
            "composition_rule": "z(a+b)=z(a)+z(b)",
            "heldout_response_labels_used": False}


def prepare(root: Path) -> dict:
    operator = verify_stage(root, "operator_design")
    actions = verify_stage(root, "actions")
    cfg = verify(root)["config"]["operator_analysis"]
    descriptor = _descriptors(root, actions)
    path = root / DESCRIPTORS
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(path, descriptor)
    return stage_freeze(root, "operator_analysis",
                        [SOURCE, str(DESCRIPTORS),
                         "artifacts/compact_causal_response_operator_v20_operator_design.freeze.json",
                         "artifacts/compact_causal_response_operator_v20_actions.freeze.json"],
                        {"models": cfg["models"], "dimensions": cfg["dimensions"],
                         "ridge_lambda": cfg["ridge_lambda"],
                         "oracle_coordinate": "train_base_state_positive_train_action_response_fingerprint_centered_SVD_only",
                         "decoder_training": "operator_train_states_positive_train_actions_only",
                         "evaluation": "operator_validation_states_unseen_direction_positive_and_negative_plus_train_direction_unseen_negative",
                         "model_selection": "minimum_operator_validation_unseen_direction_positive_stack_relative_L2",
                         "small_nonlinear_model": {"hidden": 128, "layers": 2, "epochs": 30,
                                                   "batch_size": 512, "learning_rate": 0.001,
                                                   "weight_decay": 0.0001, "seed": 2021},
                         "heldout_gates": {"cosine_min": cfg["heldout_cosine_min"],
                                           "relative_l2_max": cfg["heldout_relative_l2_max"],
                                           "norm_ratio_min": cfg["heldout_norm_ratio_min"],
                                           "norm_ratio_max": cfg["heldout_norm_ratio_max"],
                                           "family_relative_l2_max": cfg["family_relative_l2_max"]},
                         "bootstrap_unit": cfg["bootstrap_unit"],
                         "bootstrap_replicates": cfg["bootstrap_replicates"],
                         "bootstrap_seed": cfg["bootstrap_seed"],
                         "descriptor_sha256": sha256_file(path),
                         "validation_response_rows_already_observed": 0,
                         "sealed_final_action_responses_observed": 0,
                         "independent_V20_final_opened": False})


def _load(root: Path, role: str, design: dict) -> dict:
    paths = sorted((root / bank.OUT).glob(f"response_operator_{role}_*_v20.parquet"))
    if len(paths) != 5:
        raise RuntimeError(f"V20 {role} operator bank incomplete")
    frame = pd.concat([pd.read_parquet(path) for path in paths], ignore_index=True)
    train_actions = [int(x["coordinate_index"]) for x in design["shared_measured_actions"][:design["train_action_count"]]]
    val_actions = [int(x["coordinate_index"]) for x in design["shared_measured_actions"][design["train_action_count"]:]]
    groups = list(frame.groupby("operator_state_id", sort=True))
    result = {"ids": [], "base_ids": [], "family": [], "q_name": [], "eligible": [], "rows": len(frame)}
    grids = {(partition, sign): [] for partition in ("train", "validation") for sign in (-1, 1)}
    for state_id, group in groups:
        lookup = group.set_index(["action_coordinate", "action_sign"])
        if len(lookup) != 2 * (len(train_actions) + len(val_actions)):
            raise RuntimeError(f"V20 operator incomplete shared grid: {state_id}")
        result["ids"].append(state_id)
        first = group.iloc[0]
        result["base_ids"].append(str(first.base_trial_id))
        result["family"].append(str(first.family))
        result["q_name"].append(str(first.q_name))
        inference_ok = True
        for partition, actions in (("train", train_actions), ("validation", val_actions)):
            for sign in (-1, 1):
                rows = [lookup.loc[(coord, sign)] for coord in actions]
                grids[(partition, sign)].append(np.stack([np.asarray(row.response_stack, dtype=np.float32) for row in rows]))
                if partition == "train" and sign == 1:
                    inference_ok = all(bool(row.q_reliable and row.action_reliable and row.matched_vs_P0) for row in rows)
        result["eligible"].append(bool(inference_ok))
    result.update({"ids": np.asarray(result["ids"]), "base_ids": np.asarray(result["base_ids"]),
                   "family": np.asarray(result["family"]), "q_name": np.asarray(result["q_name"]),
                   "eligible": np.asarray(result["eligible"], dtype=bool)})
    result["Y"] = {key: np.stack(values) for key, values in grids.items()}
    return result


def _pca(train: np.ndarray, validation: np.ndarray) -> dict:
    mean = train.mean(axis=0, keepdims=True)
    centered = train - mean
    gram = centered.astype(np.float64) @ centered.astype(np.float64).T
    eigen, u = np.linalg.eigh(gram)
    order = np.argsort(eigen)[::-1]
    eigen = np.maximum(eigen[order], 0.0)
    u = u[:, order]
    singular = np.sqrt(eigen)
    keep = singular > max(float(singular[0]) * 1e-8, 1e-9)
    v = (centered.T @ u[:, keep]) / singular[keep][None, :]
    energy = eigen / max(float(eigen.sum()), 1e-20)
    return {"mean": mean.reshape(-1), "basis": v, "singular": singular,
            "energy": energy, "train_coordinate": centered @ v,
            "validation_coordinate": (validation - mean) @ v,
            "effective_rank": int(keep.sum())}


def _action_matrix(descriptor: dict, coordinates: list[int], sign: int = 1,
                   multiplier: float = 1.0) -> np.ndarray:
    return np.stack([np.asarray(descriptor["positive_base_scale_z"][str(coord)], dtype=np.float32)
                     * sign * multiplier for coord in coordinates])


def _design(c: np.ndarray, z: np.ndarray) -> np.ndarray:
    states, actions = len(c), len(z)
    expanded_c = np.repeat(c, actions, axis=0)
    expanded_z = np.tile(z, (states, 1))
    interaction = (expanded_c[:, :, None] * expanded_z[:, None, :]).reshape(states * actions, -1)
    return np.concatenate((expanded_z, interaction), axis=1).astype(np.float32)


def _ridge(x: np.ndarray, y: np.ndarray, lam: float) -> np.ndarray:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    xt = torch.from_numpy(x.astype(np.float32)).to(device)
    yt = torch.from_numpy(y.astype(np.float32)).to(device)
    gram = xt.T @ xt / len(xt)
    gram.diagonal().add_(lam)
    rhs = xt.T @ yt / len(xt)
    return torch.linalg.solve(gram, rhs).cpu().numpy()


def _metric(y: np.ndarray, pred: np.ndarray, family: np.ndarray, base_ids: np.ndarray) -> dict:
    flat = y.reshape(-1, y.shape[-1]).astype(np.float64)
    estimate = pred.reshape(-1, pred.shape[-1]).astype(np.float64)
    error = flat - estimate
    family_rows = np.repeat(family, y.shape[1])
    norm_y = np.linalg.norm(flat, axis=1)
    norm_p = np.linalg.norm(estimate, axis=1)
    cosines = np.sum(flat * estimate, axis=1) / np.maximum(norm_y * norm_p, 1e-8)
    ratios = norm_p / np.maximum(norm_y, 1e-8)
    values = {"rows": len(flat), "base_states": int(len(set(base_ids))),
              "stack_relative_l2": float(np.linalg.norm(error) / max(np.linalg.norm(flat), 1e-8)),
              "j_relative_l2": float(np.linalg.norm(error[:, :128]) / max(np.linalg.norm(flat[:, :128]), 1e-8)),
              "stack_cosine_median": float(np.median(cosines)),
              "j_cosine_median": float(np.median(np.sum(flat[:, :128] * estimate[:, :128], axis=1) /
                                                 np.maximum(np.linalg.norm(flat[:, :128], axis=1) *
                                                            np.linalg.norm(estimate[:, :128], axis=1), 1e-8))),
              "stack_norm_ratio_median": float(np.median(ratios)),
              "j_norm_ratio_median": float(np.median(np.linalg.norm(estimate[:, :128], axis=1) /
                                                  np.maximum(np.linalg.norm(flat[:, :128], axis=1), 1e-8)))}
    values["family_stack_relative_l2"] = {
        name: float(np.linalg.norm(error[family_rows == name]) / max(np.linalg.norm(flat[family_rows == name]), 1e-8))
        for name in sorted(set(family))}
    return values


def _gate(metrics: dict, design: dict) -> bool:
    threshold = design["heldout_gates"]
    return (metrics["stack_relative_l2"] <= threshold["relative_l2_max"]
            and metrics["j_relative_l2"] <= threshold["relative_l2_max"]
            and metrics["stack_cosine_median"] >= threshold["cosine_min"]
            and metrics["j_cosine_median"] >= threshold["cosine_min"]
            and threshold["norm_ratio_min"] <= metrics["stack_norm_ratio_median"] <= threshold["norm_ratio_max"]
            and threshold["norm_ratio_min"] <= metrics["j_norm_ratio_median"] <= threshold["norm_ratio_max"]
            and all(x <= threshold["family_relative_l2_max"] for x in metrics["family_stack_relative_l2"].values()))


class Fitted:
    def __init__(self, model: str, k: int, mean: np.ndarray, basis: np.ndarray,
                 coordinate_scale: np.ndarray, ztrain: np.ndarray, ytrain: np.ndarray,
                 ctrain: np.ndarray, design: dict):
        self.model, self.k = model, k
        self.mean, self.basis = mean, basis
        self.coordinate_scale = coordinate_scale
        self.ztrain = ztrain
        self.design = design
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.nn = None
        self.weight = None
        self.out_basis = None
        self.profile_weight = None
        if model == "response_svd":
            profiles = np.concatenate((mean.reshape(1, len(ztrain), -1),
                                       basis.T.reshape(k, len(ztrain), -1)), axis=0)
            z = ztrain.astype(np.float32)
            self.profile_weight = _ridge(z, profiles.reshape(len(profiles), len(z), -1).transpose(1, 0, 2).reshape(len(z), -1),
                                         float(design["ridge_lambda"]))
        else:
            x = _design(ctrain, ztrain)
            y = ytrain.reshape(-1, ytrain.shape[-1]).astype(np.float32)
            if model == "tucker_factorization":
                _, _, vh = np.linalg.svd(y, full_matrices=False)
                rank = min(k, 64, y.shape[-1])
                self.out_basis = vh[:rank].T.astype(np.float32)
                self.weight = _ridge(x, y @ self.out_basis, float(design["ridge_lambda"]))
            elif model in ("bilinear_latent_operator", "reduced_rank_regression"):
                self.weight = _ridge(x, y, float(design["ridge_lambda"]))
                if model == "reduced_rank_regression":
                    u, s, vh = np.linalg.svd(self.weight, full_matrices=False)
                    rank = min(k, len(s))
                    self.weight = ((u[:, :rank] * s[:rank]) @ vh[:rank]).astype(np.float32)
            elif model == "small_nonlinear_latent_operator":
                spec = design["small_nonlinear_model"]
                torch.manual_seed(int(spec["seed"]) + k)
                self.nn = torch.nn.Sequential(torch.nn.Linear(x.shape[1], int(spec["hidden"])),
                                              torch.nn.GELU(),
                                              torch.nn.Linear(int(spec["hidden"]), int(spec["hidden"])),
                                              torch.nn.GELU(),
                                              torch.nn.Linear(int(spec["hidden"]), y.shape[1])).to(self.device)
                optim = torch.optim.AdamW(self.nn.parameters(), lr=float(spec["learning_rate"]),
                                          weight_decay=float(spec["weight_decay"]))
                xx = torch.from_numpy(x).to(self.device)
                yy = torch.from_numpy(y).to(self.device)
                rng = np.random.default_rng(int(spec["seed"]) + k)
                for _ in range(int(spec["epochs"])):
                    for indices in np.array_split(rng.permutation(len(x)), math.ceil(len(x) / int(spec["batch_size"]))):
                        optim.zero_grad()
                        loss = torch.mean((self.nn(xx[indices]) - yy[indices]) ** 2)
                        loss.backward()
                        optim.step()
                self.nn.eval()
            else:
                raise RuntimeError(f"unknown V20 model {model}")

    def predict(self, c: np.ndarray, z: np.ndarray) -> np.ndarray:
        if self.model == "response_svd":
            profiles = z.astype(np.float32) @ self.profile_weight
            profiles = profiles.reshape(len(z), self.k + 1, -1)
            result = profiles[:, 0][None, :, :] + np.einsum("nk,akd->nad", c * self.coordinate_scale, profiles[:, 1:])
            return result.astype(np.float32)
        x = _design(c, z)
        if self.nn is not None:
            with torch.no_grad():
                output = self.nn(torch.from_numpy(x).to(self.device)).cpu().numpy()
        else:
            output = x @ self.weight
            if self.out_basis is not None:
                output = output @ self.out_basis.T
        return output.reshape(len(c), len(z), -1).astype(np.float32)


def _fingerprints(data: dict) -> np.ndarray:
    return data["Y"][("train", 1)].reshape(len(data["ids"]), -1).astype(np.float32)


def _coordinate(pca: dict, k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    scale = np.maximum(pca["train_coordinate"][:, :k].std(axis=0), 1e-6)
    return (pca["train_coordinate"][:, :k].astype(np.float32) / scale,
            pca["validation_coordinate"][:, :k].astype(np.float32) / scale,
            scale.astype(np.float32))


def search(root: Path) -> dict:
    design = verify_stage(root, "operator_analysis")
    operator = verify_stage(root, "operator_design")
    descriptor = json.loads((root / DESCRIPTORS).read_text())
    if sha256_file(root / DESCRIPTORS) != design["descriptor_sha256"]:
        raise RuntimeError("V20 action descriptor drift")
    train = _load(root, "operator_train", operator)
    validation = _load(root, "operator_validation", operator)
    pca = _pca(_fingerprints(train), _fingerprints(validation))
    train_coords = [int(x["coordinate_index"]) for x in operator["shared_measured_actions"][:operator["train_action_count"]]]
    val_coords = [int(x["coordinate_index"]) for x in operator["shared_measured_actions"][operator["train_action_count"]:]]
    ztrain = _action_matrix(descriptor, train_coords, 1)
    zval = _action_matrix(descriptor, val_coords, 1)
    spectra = []
    fit_results = []
    for k in design["dimensions"]:
        rank_supported = k <= pca["effective_rank"]
        fraction = float(np.sum(pca["energy"][:k]))
        if not rank_supported:
            spectra.append({"k": k, "rank_supported": False, "train_positive_fingerprint_explained_variance": fraction})
            continue
        ctrain, cval, cscale = _coordinate(pca, k)
        reconstruction_train = pca["mean"] + pca["train_coordinate"][:, :k] @ pca["basis"][:, :k].T
        reconstruction_val = pca["mean"] + pca["validation_coordinate"][:, :k] @ pca["basis"][:, :k].T
        spectra.append({"k": k, "rank_supported": True, "train_positive_fingerprint_explained_variance": fraction,
                        "train_fingerprint_relative_l2": float(np.linalg.norm(_fingerprints(train) - reconstruction_train) /
                                                               max(np.linalg.norm(_fingerprints(train)), 1e-8)),
                        "validation_train_action_fingerprint_relative_l2": float(np.linalg.norm(_fingerprints(validation) - reconstruction_val) /
                                                                                 max(np.linalg.norm(_fingerprints(validation)), 1e-8))})
        for model in design["models"]:
            fitted = Fitted(model, k, pca["mean"], pca["basis"][:, :k], cscale, ztrain,
                            train["Y"][("train", 1)], ctrain, design)
            held = fitted.predict(cval, zval)
            unseen_sign = fitted.predict(cval, -ztrain)
            both = fitted.predict(cval, -zval)
            seen = fitted.predict(cval, ztrain)
            metrics = {
                "unseen_direction_positive": _metric(validation["Y"][("validation", 1)], held,
                                                      validation["family"], validation["base_ids"]),
                "unseen_sign_train_direction": _metric(validation["Y"][("train", -1)], unseen_sign,
                                                       validation["family"], validation["base_ids"]),
                "unseen_direction_and_sign": _metric(validation["Y"][("validation", -1)], both,
                                                      validation["family"], validation["base_ids"]),
                "seen_direction_new_state": _metric(validation["Y"][("train", 1)], seen,
                                                    validation["family"], validation["base_ids"]),
            }
            preliminary = (train["eligible"].all() and validation["eligible"].all()
                           and _gate(metrics["unseen_direction_positive"], design)
                           and _gate(metrics["unseen_sign_train_direction"], design)
                           and _gate(metrics["unseen_direction_and_sign"], design))
            fit_results.append({"model": model, "k": k, "preliminary_cross_action_gate": bool(preliminary),
                                "metrics": metrics})
            print(f"V20 oracle {model} k={k} heldout={metrics['unseen_direction_positive']['stack_relative_l2']:.4f} sign={metrics['unseen_sign_train_direction']['stack_relative_l2']:.4f}", flush=True)
    if not fit_results:
        raise RuntimeError("V20 no rank-supported operator candidate")
    chosen = min(fit_results, key=lambda x: x["metrics"]["unseen_direction_positive"]["stack_relative_l2"])
    eligible = [x for x in fit_results if x["preliminary_cross_action_gate"]]
    k_min = min((x["k"] for x in eligible), default=None)
    result = {"operator_analysis_freeze_digest": design["freeze_digest"],
              "train_base_states": int(len(set(train["base_ids"]))),
              "validation_base_states": int(len(set(validation["base_ids"]))),
              "train_operator_states": len(train["ids"]), "validation_operator_states": len(validation["ids"]),
              "train_coordinate_inference_reliable_states": int(train["eligible"].sum()),
              "validation_coordinate_inference_reliable_states": int(validation["eligible"].sum()),
              "all_states_retained_in_metrics": True,
              "train_action_count": len(train_coords), "unseen_validation_action_count": len(val_coords),
              "effective_train_fingerprint_rank": pca["effective_rank"],
              "rank_curve": spectra, "model_sweep": fit_results,
              "selected_model_for_diagnostics": chosen["model"], "selected_k_for_diagnostics": chosen["k"],
              "selected_validation_unseen_direction_relative_l2": chosen["metrics"]["unseen_direction_positive"]["stack_relative_l2"],
              "preliminary_k_operator_min": k_min,
              "preliminary_operator_compactness_gate": bool(eligible),
              "final_k_operator_min": None,
              "final_operator_gate_pending_sign_scale_composition": True,
              "sealed_final_action_directions_opened": False,
              "independent_V20_final_opened": False,
              "anti_leak": "Coordinates inferred solely from positive train-action responses; decoder fit solely to positive train actions on operator_train states; validation-action responses absent from coordinate and decoder fitting."}
    write_json_atomic(root / SUMMARY, result)
    return {"selected_model": chosen["model"], "selected_k": chosen["k"],
            "heldout_relative_l2": chosen["metrics"]["unseen_direction_positive"]["stack_relative_l2"],
            "preliminary_k_operator_min": k_min}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("prepare", "search"))
    args = parser.parse_args()
    result = {"prepare": prepare, "search": search}[args.stage](Path.cwd())
    if args.stage == "prepare":
        result = {"freeze_digest": result["freeze_digest"], "descriptor_sha256": result["descriptor_sha256"]}
    print(json.dumps(result, indent=2))
