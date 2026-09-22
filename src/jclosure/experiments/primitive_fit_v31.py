"""TRAIN-only PCA, sparse dictionary and clustered-prototype write models."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.decomposition import MiniBatchDictionaryLearning

from jclosure.experiments.primitive_matrix_v31 import OUT, SCRATCH
from jclosure.protocol_v31 import stage_freeze, verify, verify_stage
from jclosure.provenance import sha256_file, write_json_atomic

SOURCE = "src/jclosure/experiments/primitive_fit_v31.py"
LATENT = 256
M_MAX = 128


def sha_array(x):
    return hashlib.sha256(memoryview(np.ascontiguousarray(x))).hexdigest()


def stream_basis(matrix, coefficients, destination, rows_per_block=8, block_columns=524288):
    n, dim = matrix.shape
    k = coefficients.shape[0]
    if not torch.cuda.is_available():
        raise RuntimeError("V31 full write-basis materialization requires CUDA")
    basis = np.memmap(destination, dtype=np.float32, mode="w+", shape=(k, dim))
    for col in range(0, dim, block_columns):
        end = min(dim, col + block_columns)
        basis_gpu = torch.zeros((k, end - col), dtype=torch.float32, device="cuda")
        for start in range(0, n, rows_per_block):
            stop = min(n, start + rows_per_block)
            block = torch.from_numpy(np.ascontiguousarray(matrix[start:stop, col:end])).to("cuda")
            weights = torch.from_numpy(np.ascontiguousarray(coefficients[:, start:stop])).to("cuda")
            basis_gpu.addmm_(weights, block)
            del block, weights
        basis[:, col:end] = basis_gpu.cpu().numpy()
        del basis_gpu
        print(f"V31 PCA basis stream columns {end}/{dim}", flush=True)
    basis.flush()
    torch.cuda.empty_cache()
    return basis


def rotate_basis(parent, atoms, destination, block_columns=524288):
    k, dim = atoms.shape[0], parent.shape[1]
    output = np.memmap(destination, dtype=np.float32, mode="w+", shape=(k, dim))
    left = torch.from_numpy(np.ascontiguousarray(atoms.astype(np.float32))).to("cuda")
    for start in range(0, dim, block_columns):
        stop = min(dim, start + block_columns)
        block = torch.from_numpy(np.ascontiguousarray(parent[:, start:stop])).to("cuda")
        rotated = left @ block
        output[:, start:stop] = rotated.cpu().numpy()
        del block, rotated
        if start % (block_columns * 5) == 0:
            print(f"V31 dictionary basis rotation {start}/{dim}", flush=True)
    output.flush()
    return output


def run(root: Path):
    verify_stage(root, "primitive_geometry")
    cfg = verify(root)["config"]
    manifest = json.loads((root / OUT / "primitive_training_rows_v31.json").read_text())
    n, dim = manifest["nrows"], manifest["dimension"]
    matrix = np.memmap(SCRATCH / "train_write_matrix_v31.f32", dtype=np.float32, mode="r", shape=(n, dim))
    eig = np.load(SCRATCH / "train_write_eigenvalues_v31.npy")
    vec = np.load(SCRATCH / "train_write_eigenvectors_v31.npy")
    if len(eig) < LATENT or eig[LATENT - 1] <= 1e-8:
        raise RuntimeError("V31 256-dimensional empirical TRAIN span unsupported")
    mean = np.zeros(dim, np.float64)
    for start in range(0, n, 8):
        mean += np.asarray(matrix[start:start + 8].sum(0, dtype=np.float64))
    mean = (mean / n).astype(np.float32)
    mean_path = SCRATCH / "primitive_mean_v31.f32"
    mean_mem = np.memmap(mean_path, dtype=np.float32, mode="w+", shape=(dim,))
    mean_mem[:] = mean
    mean_mem.flush()
    coeff = (vec[:, :LATENT] / np.sqrt(eig[:LATENT])[None, :]).T.astype(np.float32)
    pca_path = SCRATCH / "basis_PCA256_v31.f32"
    pca_basis = stream_basis(matrix, coeff, pca_path)
    score = (vec[:, :LATENT] * np.sqrt(eig[:LATENT])[None, :]).astype(np.float32)
    scale = max(float(np.median(np.linalg.norm(score, axis=1))), 1e-12)
    z = score / scale
    sparse = MiniBatchDictionaryLearning(n_components=M_MAX, alpha=0.1, max_iter=200, fit_algorithm="cd", batch_size=128, random_state=cfg["seed"], transform_algorithm="omp", transform_n_nonzero_coefs=4)
    sparse.fit(z)
    codes = sparse.transform(z)
    usage = np.count_nonzero(abs(codes) > 1e-9, axis=0)
    order_sparse = np.lexsort((np.arange(M_MAX), -usage))
    sparse_atoms = sparse.components_[order_sparse].astype(np.float32)
    sparse_atoms /= np.maximum(np.linalg.norm(sparse_atoms, axis=1, keepdims=True), 1e-12)
    print("V31 sparse dictionary latent fit complete", flush=True)
    cluster = KMeans(n_clusters=M_MAX, n_init=5, random_state=cfg["seed"])
    label = cluster.fit_predict(z)
    counts = np.bincount(label, minlength=M_MAX)
    order_cluster = np.lexsort((np.arange(M_MAX), -counts))
    cluster_atoms = cluster.cluster_centers_[order_cluster].astype(np.float32)
    cluster_atoms /= np.maximum(np.linalg.norm(cluster_atoms, axis=1, keepdims=True), 1e-12)
    print("V31 clustered prototypes latent fit complete", flush=True)
    sparse_path = SCRATCH / "basis_SPARSE128_v31.f32"
    cluster_path = SCRATCH / "basis_PROTOTYPE128_v31.f32"
    rotate_basis(pca_basis, sparse_atoms, sparse_path)
    rotate_basis(pca_basis, cluster_atoms, cluster_path)
    coordinate_path = root / OUT / "primitive_dictionary_coordinates_v31.npz"
    np.savez_compressed(coordinate_path, pca_eigenvalues=eig[:LATENT], pca_training_scores=score, sparse_atoms=sparse_atoms, sparse_training_usage=usage[order_sparse], prototype_atoms=cluster_atoms, prototype_training_counts=counts[order_cluster])
    result = {"TRAIN_rows": n, "write_dimension": dim, "latent_fit_span": LATENT, "basis_count": M_MAX, "tested_M_grid": cfg["primitive_count_grid"], "tested_active_count_grid": cfg["active_count_grid"], "PCA_is_geometry_baseline_not_confirmed_primitive": True, "SPARSE_DICTIONARY": {"method": "MiniBatchDictionaryLearning", "alpha": 0.1, "max_iter": 200, "random_seed": cfg["seed"], "rank_order": "decreasing TRAIN OMP four-active usage", "basis_path": str(sparse_path), "basis_sha256": sha256_file(sparse_path)}, "CLUSTERED_PROTOTYPES": {"method": "KMeans on 256-dimensional TRAIN PCA scores", "n_init": 5, "random_seed": cfg["seed"], "rank_order": "decreasing TRAIN cluster count", "basis_path": str(cluster_path), "basis_sha256": sha256_file(cluster_path)}, "PCA": {"basis_path": str(pca_path), "basis_sha256": sha256_file(pca_path)}, "mean_path": str(mean_path), "mean_sha256": sha256_file(mean_path), "coordinate_npz_sha256": sha256_file(coordinate_path), "heldout_write_used_in_fit": False, "future_response_used_in_fit": False, "scratch_not_committed": True}
    result_path = root / OUT / "primitive_dictionary_fit_v31.json"
    write_json_atomic(result_path, result)
    freeze = stage_freeze(root, "primitive_dictionary_fit", [SOURCE, str(result_path.relative_to(root)), str(coordinate_path.relative_to(root)), "artifacts/compositional_natural_writes_v31_primitive_geometry.freeze.json"], {"result_sha256": sha256_file(result_path), "basis_sha256": {k: result[k]["basis_sha256"] for k in ("PCA", "SPARSE_DICTIONARY", "CLUSTERED_PROTOTYPES")}, "heldout_response_seen": False})
    return {"freeze_digest": freeze["freeze_digest"], "models": list(("PCA", "SPARSE_DICTIONARY", "CLUSTERED_PROTOTYPES")), "basis_count": M_MAX}


if __name__ == "__main__":
    print(json.dumps(run(Path.cwd()), indent=2))
