#!/usr/bin/env python3
"""Compare a V13 batched exact-JVP diagnostic against its scalar reference."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _spectrum(matrix: np.ndarray) -> dict[str, Any]:
    singular = np.linalg.svd(matrix.astype(np.float64), compute_uv=False)
    energy = singular**2
    probability = energy / max(float(energy.sum()), 1e-20)
    cumulative = np.cumsum(probability)
    return {
        "r90": int(np.searchsorted(cumulative, 0.90) + 1),
        "r95": int(np.searchsorted(cumulative, 0.95) + 1),
        "r99": int(np.searchsorted(cumulative, 0.99) + 1),
        "stable_rank": float(energy.sum() / max(float(energy[0]), 1e-20)),
        "effective_rank": float(
            np.exp(
                -np.sum(
                    probability * np.log(np.maximum(probability, np.finfo(float).tiny))
                )
            )
        ),
        "leading_singular_values": singular[:20].tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scalar", type=Path, required=True)
    parser.add_argument("--batched", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    args = parser.parse_args()
    with np.load(args.scalar, allow_pickle=False) as payload:
        scalar = payload["matrix"].astype(np.float64)
    with np.load(args.batched, allow_pickle=False) as payload:
        batched = payload["matrix"].astype(np.float64)
    if scalar.shape != batched.shape:
        raise ValueError(f"matrix shape mismatch: {scalar.shape} != {batched.shape}")
    absolute = np.abs(batched - scalar)
    scalar_norm = np.linalg.norm(scalar, axis=0)
    batched_norm = np.linalg.norm(batched, axis=0)
    cosine = np.sum(scalar * batched, axis=0) / np.maximum(
        scalar_norm * batched_norm, 1e-20
    )
    curves = {
        str(size): {
            "scalar": _spectrum(scalar[:, :size]),
            "batched": _spectrum(batched[:, :size]),
        }
        for size in (64, 128, 256, 512)
    }
    checks = {
        "prefix_r90_r95_r99_exact_match": all(
            curves[str(size)]["scalar"][name] == curves[str(size)]["batched"][name]
            for size in (64, 128, 256, 512)
            for name in ("r90", "r95", "r99")
        ),
        "maximum_mean_absolute_difference": float(absolute.mean()) <= 0.001,
        "minimum_mean_column_cosine": float(cosine.mean()) >= 0.999,
        "minimum_worst_column_cosine": float(cosine.min()) >= 0.998,
        "maximum_relative_frobenius_error": float(
            np.linalg.norm(batched - scalar) / max(float(np.linalg.norm(scalar)), 1e-20)
        )
        <= 0.02,
    }
    record = {
        "schema_version": 22,
        "protocol_version": "causal_path_geometry_v13_jvp_batch_equivalence",
        "batch_size": args.batch_size,
        "scalar_matrix": str(args.scalar),
        "scalar_sha256": _sha256(args.scalar),
        "batched_matrix": str(args.batched),
        "batched_sha256": _sha256(args.batched),
        "shape": list(scalar.shape),
        "maximum_absolute_difference": float(absolute.max()),
        "mean_absolute_difference": float(absolute.mean()),
        "relative_frobenius_error": float(
            np.linalg.norm(batched - scalar) / max(float(np.linalg.norm(scalar)), 1e-20)
        ),
        "minimum_column_cosine": float(cosine.min()),
        "mean_column_cosine": float(cosine.mean()),
        "probe_scaling_comparison": curves,
        "acceptance_checks": checks,
        "passed": all(checks.values()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
