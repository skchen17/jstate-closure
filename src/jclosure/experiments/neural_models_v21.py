"""Frozen small MLP and low-rank hypernetwork shared finite-response decoders."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn

from jclosure.experiments import bank_v21 as bank
from jclosure.experiments import factorial_v21 as fact
from jclosure.experiments import operator_model_v20 as v20
from jclosure.experiments.z2_correction_v21 import _corrected_gram
from jclosure.protocol_v21 import stage_freeze, verify, verify_stage
from jclosure.provenance import sha256_file, write_json_atomic

SOURCE = "src/jclosure/experiments/neural_models_v21.py"
CONDITIONS = [("S1", "Z0"), ("S2", "Z0"), ("S2", "Z1"), ("S2", "Z2"),
              ("S3", "Z1"), ("S0", "Z1")]
SUMMARY = bank.OUT / "neural_operator_models_v21.json"
WEIGHTS = bank.SCRATCH / "neural_weights"


class JointMLP(nn.Module):
    def __init__(self, sdim: int, zdim: int, output: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(sdim + zdim, 64), nn.ReLU(),
                                 nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, output))

    def forward(self, s, z):
        return self.net(torch.cat((s, z), dim=-1))


class LowRankHyper(nn.Module):
    def __init__(self, sdim: int, zdim: int, output: int):
        super().__init__()
        self.rank = 8
        self.output = output
        self.state = nn.Sequential(nn.Linear(sdim, 64), nn.ReLU(),
                                   nn.Linear(64, self.rank * output))
        self.action = nn.Linear(zdim, self.rank, bias=False)

    def forward(self, s, z):
        basis = self.state(s).reshape(-1, self.rank, self.output)
        coefficients = self.action(z)
        return torch.einsum("br,brd->bd", coefficients, basis)


def _embed(gram: np.ndarray) -> np.ndarray:
    train = (gram[:12, :12] + gram[:12, :12].T) / 2
    eigen, vectors = np.linalg.eigh(train)
    keep = eigen > max(float(eigen[-1]) * 1e-9, 1e-10)
    values = np.sqrt(eigen[keep])
    train_coord = vectors[:, keep] * values[None]
    val_coord = gram[12:18, :12] @ (vectors[:, keep] / values[None])
    return np.vstack((train_coord, val_coord)).astype(np.float32)


def _action_features(descriptor: dict, z: str) -> np.ndarray:
    if z == "Z0":
        matrix = np.asarray(descriptor["Z0_exact_V20_descriptor"], dtype=np.float32)
    elif z == "Z2":
        gram, _ = _corrected_gram(descriptor)
        matrix = _embed(gram)
    else:
        matrix = _embed(fact._action_gram(descriptor, z))
    median = max(float(np.median(np.linalg.norm(matrix[:12], axis=1))), 1e-6)
    return matrix / median


def prepare(root: Path) -> dict:
    verify_stage(root, "factorial_design")
    verify_stage(root, "z2_channel_normalization_amendment_1")
    return stage_freeze(root, "neural_model_design",
                        [SOURCE, "artifacts/action_coordinate_geometry_v21_factorial_design.freeze.json",
                         "artifacts/action_coordinate_geometry_v21_z2_channel_normalization_amendment_1.freeze.json"],
                        {"conditions": [{"S": s, "Z": z} for s,z in CONDITIONS],
                         "models": ["G4_joint_MLP", "G5_rank8_hypernetwork"],
                         "input_state": "complete_frozen_state_feature_without_dimension_reduction",
                         "action_embedding": "exact_train_action_Gram_eigenspace_and_validation_Nystrom_projection",
                         "hidden_width": 64, "hidden_layers": 2, "hypernetwork_rank": 8,
                         "epochs": 25, "batch_size": 256, "learning_rate": 0.001,
                         "weight_decay": 0.0001, "seed": 2021,
                         "selection": "hyperparameters_and_epochs_fixed_before_V21_validation_response_access; no_validation_response_in_training_or_selection",
                         "training": "600_V20_operator_train_states_x_12_positive_train_actions_only",
                         "evaluation": "200_V20_operator_validation_states_x_seen_unseen_direction_and_sign",
                         "weight_storage": "scratch_outside_git_repository",
                         "final_six_action_responses_opened": False})


def _train_one(s_train: np.ndarray, z: np.ndarray, y: np.ndarray, model_name: str,
               design: dict, weight_path: Path) -> nn.Module:
    torch.manual_seed(int(design["seed"]))
    np.random.seed(int(design["seed"]))
    random.seed(int(design["seed"]))
    torch.set_num_threads(4)
    n, a, d = y.shape
    s = torch.from_numpy(np.repeat(s_train, a, axis=0).astype(np.float32))
    zz = torch.from_numpy(np.tile(z[None, :a, :], (n, 1, 1)).reshape(n*a, -1).astype(np.float32))
    target = torch.from_numpy(y.reshape(n*a, d).astype(np.float32))
    model = JointMLP(s.shape[1], zz.shape[1], d) if model_name == "G4_joint_MLP" else LowRankHyper(s.shape[1], zz.shape[1], d)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(design["learning_rate"]),
                                  weight_decay=float(design["weight_decay"]))
    rng = np.random.default_rng(int(design["seed"]))
    model.train()
    for epoch in range(int(design["epochs"])):
        order = rng.permutation(n*a)
        total = 0.0
        for start in range(0, len(order), int(design["batch_size"])):
            index = order[start:start + int(design["batch_size"])]
            optimizer.zero_grad(set_to_none=True)
            prediction = model(s[index], zz[index])
            loss = ((prediction - target[index]) ** 2).mean()
            loss.backward()
            optimizer.step()
            total += float(loss.item())
        if (epoch + 1) % 5 == 0:
            print(f"V21 neural {weight_path.stem} epoch={epoch+1} train_mse_sum={total:.5g}", flush=True)
    weight_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), weight_path)
    return model.eval()


def _predict(model: nn.Module, s: np.ndarray, z: np.ndarray, indices: list[int], sign: int) -> np.ndarray:
    n = len(s)
    state = torch.from_numpy(np.repeat(s, len(indices), axis=0).astype(np.float32))
    action = torch.from_numpy(np.tile((z[indices] * sign)[None], (n, 1, 1)).reshape(n*len(indices), -1).astype(np.float32))
    chunks = []
    with torch.no_grad():
        for start in range(0, len(state), 256):
            chunks.append(model(state[start:start+256], action[start:start+256]).numpy())
    return np.vstack(chunks).reshape(n, len(indices), -1)


def run(root: Path, s: str, z: str) -> dict:
    design = verify_stage(root, "neural_model_design")
    if (s,z) not in CONDITIONS:
        raise RuntimeError("V21 neural condition not frozen")
    output = root / bank.OUT / f"neural_{s}_{z}_v21.json"
    if output.exists():
        raise RuntimeError("V21 neural result already exists; append-only policy")
    state = fact._load_state()
    descriptor = json.loads((root / "results/v21/processed/action_representation_v21.json").read_text())
    actions = _action_features(descriptor, z)
    raw_train = state[f"{s}_train"].astype(np.float32)
    raw_val = state[f"{s}_validation"].astype(np.float32)
    mean = raw_train.mean(axis=0)
    norm = max(float(np.mean(np.sum((raw_train-mean)**2, axis=1)))**0.5, 1e-8)
    s_train = (raw_train-mean)/norm
    s_val = (raw_val-mean)/norm
    op = json.loads((root / "artifacts/compact_causal_response_operator_v20_operator_design.freeze.json").read_text())
    train = v20._load(root, "operator_train", op)
    val = v20._load(root, "operator_validation", op)
    config = verify(root)["config"]["gates"]
    gate = {"heldout_gates": {"relative_l2_max": config["relative_l2_max"],
                              "cosine_min": config["stack_median_cosine_min"],
                              "norm_ratio_min": config["norm_ratio_min"],
                              "norm_ratio_max": config["norm_ratio_max"],
                              "family_relative_l2_max": config["family_relative_l2_max"]}}
    rows = []
    for name in design["models"]:
        path = WEIGHTS / f"{s}_{z}_{name}.pt"
        model = _train_one(s_train, actions, train["Y"][("train", 1)], name, design, path)
        metrics = {}
        for test, indices, sign, truth in (
            ("seen_direction_new_state", list(range(12)), 1, val["Y"][("train", 1)]),
            ("unseen_direction", list(range(12,18)), 1, val["Y"][("validation", 1)]),
            ("unseen_sign", list(range(12)), -1, val["Y"][("train", -1)]),
            ("unseen_direction_and_sign", list(range(12,18)), -1, val["Y"][("validation", -1)])):
            prediction = _predict(model, s_val, actions, indices, sign)
            metrics[test] = fact._metric(truth, prediction, val["family"], val["base_ids"])
        rows.append({"S": s, "Z": z, "model": name,
                     "parameter_count": sum(p.numel() for p in model.parameters()),
                     "weights_sha256": sha256_file(path), "weights_scratch_path": str(path),
                     "metrics": metrics,
                     "direction_and_sign_preliminary_gate": bool(v20._gate(metrics["unseen_direction"], gate)
                                                                 and v20._gate(metrics["unseen_sign"], gate))})
        print(f"V21 neural {s} {z} {name} unseen={metrics['unseen_direction']['stack_relative_l2']:.5f}", flush=True)
    result = {"design_digest": design["freeze_digest"], "state_dimension": int(raw_train.shape[1]),
              "action_embedding_dimension": int(actions.shape[1]), "models": rows,
              "final_six_action_responses_opened": False}
    write_json_atomic(output, result)
    return {"condition": f"{s}x{z}", "best_unseen_direction": min(x["metrics"]["unseen_direction"]["stack_relative_l2"] for x in rows)}


def summarize(root: Path) -> dict:
    design = verify_stage(root, "neural_model_design")
    paths = [root / bank.OUT / f"neural_{s}_{z}_v21.json" for s,z in CONDITIONS]
    if not all(path.exists() for path in paths):
        raise RuntimeError("V21 neural condition results incomplete")
    records = [json.loads(path.read_text()) for path in paths]
    result = {"design_digest": design["freeze_digest"], "conditions": records,
              "result_sha256": {path.name: sha256_file(path) for path in paths},
              "final_six_action_responses_opened": False}
    write_json_atomic(root / SUMMARY, result)
    return {"conditions": len(records), "models": sum(len(x["models"]) for x in records)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("prepare", "run", "summarize"))
    parser.add_argument("--state")
    parser.add_argument("--action")
    args = parser.parse_args()
    if args.stage == "run" and (not args.state or not args.action):
        parser.error("--state and --action required for run")
    result = {"prepare": lambda: prepare(Path.cwd()),
              "run": lambda: run(Path.cwd(), args.state, args.action),
              "summarize": lambda: summarize(Path.cwd())}[args.stage]()
    print(json.dumps(result, indent=2))
