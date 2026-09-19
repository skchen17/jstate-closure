"""Actuator-weighted response SVD and held-out-probe target coverage."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from jclosure.experiments import operator_v15 as operator
from jclosure.experiments.numerics_v14 import _panels, _slices
from jclosure.protocol_v15 import stage_freeze, verify
from jclosure.provenance import sha256_file, write_json_atomic

OUT = Path("results/v15/processed")


def _r95(matrix: np.ndarray) -> int:
    singular = np.linalg.svd(matrix, compute_uv=False)
    energy = singular**2
    cumulative = np.cumsum(energy) / max(float(energy.sum()), 1e-20)
    return int(np.searchsorted(cumulative, 0.95) + 1)


def _coverage(basis: np.ndarray, targets: np.ndarray) -> float:
    denominator = float(np.linalg.norm(targets) ** 2)
    if denominator <= 1e-20:
        return 0.0
    return float(np.linalg.norm(basis.T @ targets) ** 2 / denominator)


def main() -> None:
    root = Path.cwd()
    base = verify(root)
    stage_path = root / "artifacts/quantization_aware_actuation_v15_basis.freeze.json"
    if not stage_path.exists():
        stage = stage_freeze(root, "basis", ["src/jclosure/experiments/basis_v15.py", "results/v15/processed/finite_operator_columns_v15.parquet", "results/v15/processed/writable_causal_rank_v15.json"], {"role": "diagnostic_train", "method": "SVD of target-normalized finite response columns divided by realized-state norm; diagonal physical cost approximation", "heldout_probe_rule": "first half basis / second half target coverage", "ranks": base["config"]["development"]["ranks"], "not_pca": True, "not_full_physical_state_metric": True})
    else:
        stage = json.loads(stage_path.read_text())
    frame = pd.read_parquet(root / OUT / "finite_operator_columns_v15.parquet")
    panel = _panels(root, 1)
    scales = json.loads((root / OUT / "writable_causal_rank_v15.json").read_text())["target_scales"]
    rows = []
    basis_payload = {}
    for declaration in panel:
        base_id = str(declaration["base_trial_id"])
        group = frame[(frame.base_trial_id == base_id) & (frame.channel == "joint")].sort_values("direction_index")
        if len(group) < 64:
            raise RuntimeError("finite response matrix has fewer than 64 columns")
        count = len(group)
        finite = np.column_stack([
            operator.stack({name: np.asarray(item[f"{name}_response"], dtype=np.float64) for name in operator.TARGETS}, scales)
            for item in group.to_dict("records")
        ])
        with np.load(root / declaration["matrix_path"], allow_pickle=False) as payload:
            if sha256_file(root / declaration["matrix_path"]) != declaration["matrix_sha256"]:
                raise RuntimeError("V13 ideal matrix hash mismatch")
            matrix = payload["matrix"].astype(np.float64)
            slices = _slices(payload)
        ideal = np.column_stack([
            operator.stack({name: matrix[slices[name], i] for name in operator.TARGETS}, scales)
            for i in range(count)
        ])
        train_count = count // 2
        cost = group.realized_state_norm.to_numpy(dtype=np.float64)[:train_count]
        floor = max(float(np.median(cost[cost > 0])) * 0.01, 1e-12) if np.any(cost > 0) else 1e-12
        inv_cost = 1.0 / np.maximum(cost, floor)
        weighted = finite[:, :train_count] * inv_cost[None, :]
        u_finite, singular, vh = np.linalg.svd(weighted, full_matrices=False)
        u_ideal, _, _ = np.linalg.svd(ideal[:, :train_count], full_matrices=False)
        r95 = _r95(weighted)
        ranks = sorted(set([*base["config"]["development"]["ranks"], r95]))
        basis_payload[base_id] = {"train_direction_count": train_count, "r95_weighted": r95, "coefficient_rows": {}}
        for rank in ranks:
            rank = min(int(rank), train_count, u_finite.shape[1])
            # Each row is a combination of V13 raw probe directions. The
            # diagonal realized-cost approximation is explicit, not a full
            # generalized SVD over the physical cache-state Gram matrix.
            coefficients = vh[:rank] * inv_cost[None, :]
            basis_payload[base_id]["coefficient_rows"][str(rank)] = coefficients.astype(np.float32).tolist()
            rows.append({
                "role": "diagnostic_train", "freeze_digest": stage["freeze_digest"],
                "base_trial_id": base_id, "direction_count": count,
                "train_direction_count": train_count, "heldout_direction_count": count - train_count,
                "rank": rank, "r95_weighted": r95,
                "finite_basis_train_response_energy": _coverage(u_finite[:, :rank], finite[:, :train_count]),
                "finite_basis_heldout_target_coverage": _coverage(u_finite[:, :rank], finite[:, train_count:]),
                "autograd_basis_heldout_finite_target_coverage": _coverage(u_ideal[:, :rank], finite[:, train_count:]),
                "cost_floor": floor,
            })
    path = root / OUT / "actuator_basis_coverage_v15.parquet"
    pd.DataFrame(rows).to_parquet(path, index=False, compression="zstd")
    coefficient_path = root / OUT / "actuator_basis_coefficients_v15.json"
    write_json_atomic(coefficient_path, {"role": "diagnostic_train", "freeze_digest": stage["freeze_digest"], "by_anchor": basis_payload})
    summary = {"role": "diagnostic_train", "freeze_digest": stage["freeze_digest"], "coverage_records": str(path), "coverage_sha256": sha256_file(path), "coefficients": str(coefficient_path), "coefficients_sha256": sha256_file(coefficient_path), "metric_warning": "diagonal realized-state cost only; linear combination writeback may be nonlinear and must be remeasured", "rows": rows}
    write_json_atomic(root / OUT / "actuator_basis_v15.json", summary)
    print(json.dumps(summary, sort_keys=True)[:3000])


if __name__ == "__main__":
    main()
