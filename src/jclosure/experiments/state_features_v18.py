"""Train-frozen strong current-state kernels from V13 clean persistent tensors."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from jclosure.experiments.crossed_bank_v18 import OUT, SCRATCH, _split
from jclosure.protocol_v18 import digest, stage_freeze, verify
from jclosure.provenance import sha256_file, write_json_atomic

SOURCE = "src/jclosure/experiments/state_features_v18.py"
DESIGN_FREEZE = Path("artifacts/strong_state_context_ceiling_v18_features.freeze.json")
KERNELS = OUT / "clean_state_kernels_v18.npz"
FEATURES = OUT / "clean_state_scores_v18.npz"


def prepare(root: Path) -> dict:
    split = _split(root)
    ids = [x["base_trial_id"] for x in split["train"]]
    landmarks = sorted(ids, key=lambda key: hashlib.sha256(f"20260921:landmark:{key}".encode()).hexdigest())[:256]
    return stage_freeze(root, "features", [SOURCE, str(Path("artifacts/strong_state_context_ceiling_v18_splits.freeze.json"))],
                        {"split_freeze_digest": split["freeze_digest"],
                         "landmark_base_trial_ids": landmarks, "landmark_id_sha256": hashlib.sha256("\n".join(landmarks).encode()).hexdigest(),
                         "raw_source": "V13_clean_only", "kv_padding_tokens": 256,
                         "raw_kernel": "equal_linear_and_train_median_RBF_each_architecture_channel",
                         "kernel_center_and_scale": "train_only", "nystrom": "train_landmarks_only",
                         "feature_dimensions": [128, 256], "rank_policy": "no_silent_clamp"})


def _design(root: Path) -> dict:
    base = verify(root)
    value = json.loads((root / DESIGN_FREEZE).read_text())
    if value["freeze_digest"] != digest(value) or value["base_freeze_digest"] != base["freeze_digest"]:
        raise RuntimeError("V18 feature freeze invalid")
    for path, expected in value["input_hashes"].items():
        if sha256_file(root / path) != expected:
            raise RuntimeError(f"V18 feature source changed: {path}")
    return value


def _alloc(name: str, count: int, width: int) -> np.memmap:
    SCRATCH.mkdir(parents=True, exist_ok=True)
    path = SCRATCH / f"clean_{name}_v18.f16"
    return np.memmap(path, dtype=np.float16, mode="w+", shape=(count, width))


def extract(root: Path) -> dict:
    design = _design(root)
    split = _split(root)
    rows = split["train"] + split["validation"]
    ids = [x["base_trial_id"] for x in rows]
    count = len(ids)
    index = {key: i for i, key in enumerate(ids)}
    arrays = {"rec": _alloc("rec", count, 3145728),
              "conv": _alloc("conv", count, 196608),
              "kv": _alloc("kv", count, 1048576),
              "j": _alloc("j", count, 4096)}
    seen = set()
    verified_shards = []
    for role in ("train", "validation"):
        manifest = json.loads((root / f"results/v13/processed/causal_capture_{role}_v13.json").read_text())
        for shard in manifest["state_shards"]:
            selected = set(shard["base_trial_ids"]) & set(index)
            if not selected:
                continue
            path = root / shard["path"]
            if sha256_file(path) != shard["sha256"]:
                raise RuntimeError(f"V13 frozen clean-state shard hash mismatch: {path}")
            verified_shards.append({"path": shard["path"], "sha256": shard["sha256"]})
            payload = torch.load(path, map_location="cpu", weights_only=False)
            for row in payload["rows"]:
                key = str(row["base_trial_id"])
                if key not in selected:
                    continue
                clean = row["clean"]
                position = index[key]
                arrays["rec"][position] = clean["recurrent"].flatten().float().numpy().astype(np.float16)
                arrays["conv"][position] = clean["conv"].flatten().float().numpy().astype(np.float16)
                blocks = []
                for layer in ("27", "31"):
                    for name in ("keys", "values"):
                        tensor = clean["kv"][layer][name]
                        if tensor.shape[-2] > 256:
                            raise RuntimeError("V18 KV prompt exceeds frozen token width 256")
                        padded = torch.zeros((tensor.shape[0], 256, tensor.shape[-1]), dtype=torch.float32)
                        padded[:, :tensor.shape[-2], :] = tensor.float()
                        blocks.append(padded.flatten())
                arrays["kv"][position] = torch.cat(blocks).numpy().astype(np.float16)
                seen.add(key)
        with np.load(root / manifest["endpoint_artifact"], allow_pickle=False) as payload:
            for i, key in enumerate(payload["base_trial_id"].astype(str)):
                if key in index:
                    arrays["j"][index[key]] = payload["current_j_clean"][i].astype(np.float16)
    if seen != set(ids):
        raise RuntimeError(f"Missing {len(set(ids)-seen)} V18 clean states")
    for value in arrays.values():
        value.flush()
    result = {"feature_freeze_digest": design["freeze_digest"], "states": count,
              "train": len(split["train"]), "validation": len(split["validation"]),
              "verified_source_shards": verified_shards,
              "raw_duplicate_scratch_only": str(SCRATCH),
              "source_data": "V13_clean_current_REC_Conv_KV_and_J_no_perturbed_or_future"}
    write_json_atomic(root / OUT / "clean_state_extract_v18.json", result)
    return {"states": count, "verified_shards": len(verified_shards), "feature_freeze_digest": design["freeze_digest"]}


def _gram(name: str, n: int, width: int) -> np.ndarray:
    mm = np.memmap(SCRATCH / f"clean_{name}_v18.f16", dtype=np.float16, mode="r", shape=(n, width))
    gram = np.zeros((n, n), dtype=np.float64)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch.backends.cuda.matmul.allow_tf32 = True
    for start in range(0, width, 16384):
        chunk = torch.from_numpy(np.asarray(mm[:, start:start + 16384], dtype=np.float32).copy()).to(device)
        gram += (chunk @ chunk.T).cpu().numpy().astype(np.float64)
        del chunk
    return gram


def _center_scale(gram: np.ndarray, n_train: int) -> tuple[np.ndarray, float]:
    mean = gram[:n_train].mean(axis=0)
    centered = gram - mean[None, :] - mean[:, None] + float(gram[:n_train, :n_train].mean())
    scale = float(np.diag(centered[:n_train, :n_train]).mean())
    if scale <= 0:
        raise RuntimeError("V18 degenerate train state kernel")
    return centered / scale, scale


def kernels(root: Path) -> dict:
    design = _design(root)
    split = _split(root)
    ids = [x["base_trial_id"] for x in split["train"] + split["validation"]]
    n = len(ids)
    n_train = len(split["train"])
    dimensions = {"rec": 3145728, "conv": 196608, "kv": 1048576, "j": 4096}
    output = {}
    metadata = {}
    for name, width in dimensions.items():
        raw = _gram(name, n, width)
        linear, scale = _center_scale(raw, n_train)
        dist = np.maximum(np.diag(raw)[:, None] + np.diag(raw)[None, :] - 2 * raw, 0.0)
        train_dist = dist[:n_train, :n_train]
        bandwidth = float(np.median(train_dist[np.triu_indices(n_train, k=1)]))
        if bandwidth <= 0:
            raise RuntimeError(f"V18 degenerate RBF bandwidth: {name}")
        rbf, _ = _center_scale(np.exp(-dist / bandwidth), n_train)
        kernel, _ = _center_scale(0.5 * linear + 0.5 * rbf, n_train)
        output[name] = kernel.astype(np.float32)
        metadata[name] = {"raw_train_centered_trace": scale, "train_median_squared_distance": bandwidth,
                          "train_kernel_rank_1e_minus_6": int((np.linalg.eigvalsh(kernel[:n_train, :n_train]) > 1e-6).sum())}
        print(f"V18 clean kernel {name} complete", flush=True)
    np.savez_compressed(root / KERNELS, **output, base_trial_id=np.asarray(ids),
                        family=np.asarray([x["family"] for x in split["train"] + split["validation"]]),
                        role=np.asarray([x["role"] for x in split["train"] + split["validation"]]))
    result = {"feature_freeze_digest": design["freeze_digest"], "kernel_path": str(KERNELS),
              "kernel_sha256": sha256_file(root / KERNELS), "metadata": metadata,
              "train_only_bandwidth_center_scale": True}
    write_json_atomic(root / OUT / "clean_state_kernels_v18.json", result)
    return result


def _contexts(payload: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    j = payload["j"]
    raw = {name: payload[name] for name in ("rec", "conv", "kv")}
    contexts = {"j": j, "full_raw": np.mean(list(raw.values()), axis=0)}
    for names in (("rec",), ("conv",), ("kv",), ("rec", "conv"), ("rec", "kv"),
                  ("conv", "kv"), ("rec", "conv", "kv")):
        contexts["j_" + "_".join(names)] = 0.5 * j + 0.5 * np.mean([raw[name] for name in names], axis=0)
    return contexts


def scores(root: Path) -> dict:
    design = _design(root)
    split = _split(root)
    with np.load(root / KERNELS, allow_pickle=False) as payload:
        data = {name: payload[name] for name in payload.files}
    ids = data["base_trial_id"].astype(str).tolist()
    index = {key: i for i, key in enumerate(ids)}
    landmark_indices = np.asarray([index[key] for key in design["landmark_base_trial_ids"]])
    contexts = _contexts(data)
    output = {}
    ranks = {}
    for name, kernel in contexts.items():
        w = kernel[np.ix_(landmark_indices, landmark_indices)].astype(np.float64)
        eigenvalue, eigenvector = np.linalg.eigh(w)
        keep = eigenvalue > max(float(eigenvalue.max()) * 1e-8, 1e-10)
        if keep.sum() < 256:
            raise RuntimeError(f"V18 {name} Nyström rank {keep.sum()} below frozen 256; no silent clamp")
        order = np.argsort(eigenvalue)[-256:][::-1]
        x = kernel[:, landmark_indices] @ (eigenvector[:, order] / np.sqrt(eigenvalue[order])[None, :])
        mean = x[:len(split["train"])].mean(axis=0)
        std = x[:len(split["train"])].std(axis=0)
        std = np.maximum(std, 1e-5)
        output[name] = ((x - mean) / std).astype(np.float32)
        ranks[name] = int(keep.sum())
    np.savez_compressed(root / FEATURES, **output, base_trial_id=np.asarray(ids),
                        family=data["family"], role=data["role"])
    result = {"feature_freeze_digest": design["freeze_digest"], "scores_path": str(FEATURES),
              "scores_sha256": sha256_file(root / FEATURES), "nystrom_ranks": ranks,
              "feature_dimension": 256, "train_only_landmarks_and_standardization": True}
    write_json_atomic(root / OUT / "clean_state_scores_v18.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("prepare", "extract", "kernels", "scores"))
    args = parser.parse_args()
    root = Path.cwd()
    result = {"prepare": prepare, "extract": extract, "kernels": kernels, "scores": scores}[args.stage](root)
    print(json.dumps(result, indent=2))
