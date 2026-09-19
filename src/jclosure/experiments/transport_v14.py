"""Explicit alignment, path transport, and holonomy of V13 local JVP bases."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from jclosure.protocol_v14 import freeze_stage, verify_base
from jclosure.provenance import sha256_file, write_json_atomic

ROOT = Path("results/v14/processed")
RANK = 16
RIDGE = 1e-4


def _cosine(left: np.ndarray, right: np.ndarray) -> float | None:
    norm = float(np.linalg.norm(left) * np.linalg.norm(right))
    return None if norm < 1e-20 else float(np.dot(left, right) / norm)


def _states(root: Path) -> list[dict[str, Any]]:
    frame = pd.read_parquet(
        root / "results/v13/processed/causal_probe_scaling_v13.parquet"
    )
    frame = frame[
        (frame.probe_family == "mixed") & (frame.probe_direction_count == 512)
    ].sort_values(["family", "base_trial_id", "token_position"])
    output = []
    for row in frame.to_dict("records"):
        path = root / row["matrix_path"]
        if sha256_file(path) != row["matrix_sha256"]:
            raise RuntimeError(f"frozen V13 JVP matrix changed: {path}")
        with np.load(path, allow_pickle=False) as data:
            matrix = data["matrix"].astype(np.float64)
            slices = json.loads(str(data["slices_json"].item()))
            logits = data["selected_logits"].astype(int)
        _, singular, vh = np.linalg.svd(matrix, full_matrices=False)
        output.append(
            {
                "id": f"{row['base_trial_id']}:p{row['token_position']}",
                "base_trial_id": str(row["base_trial_id"]),
                "family": str(row["family"]),
                "position": int(row["token_position"]),
                "matrix": matrix,
                "basis": vh[:RANK],
                "singular": singular,
                "slices": slices,
                "logits": logits,
            }
        )
    return output


def _alignment(a: dict[str, Any], b: dict[str, Any], method: str) -> np.ndarray:
    ba, bb = a["basis"], b["basis"]
    if method == "unaligned":
        return np.eye(RANK)
    if method == "projector":
        return ba @ bb.T
    overlap = ba @ bb.T
    u, _, vt = np.linalg.svd(overlap, full_matrices=False)
    if method in ("procrustes", "grassmann_geodesic"):
        # The endpoint of minimal-geodesic parallel transport equals the
        # Procrustes isometry when all principal angles are below 90 degrees.
        return u @ vt
    if method == "j_response":
        ja = a["matrix"][a["slices"]["j_h1"]] @ ba.T
        jb = b["matrix"][b["slices"]["j_h1"]] @ bb.T
        return ja.T @ jb @ np.linalg.inv(jb.T @ jb + RIDGE * np.eye(RANK))
    raise ValueError(method)


def _common_indices(
    a: dict[str, Any], b: dict[str, Any], target: str
) -> tuple[np.ndarray, np.ndarray]:
    if target in ("j", "workspace"):
        if target == "j":
            names = ["j_h1"]
        else:
            names = ["workspace_l23_h1", "workspace_l26_h1", "workspace_l30_h1"]
        return (
            np.asarray([i for name in names for i in a["slices"][name]], dtype=int),
            np.asarray([i for name in names for i in b["slices"][name]], dtype=int),
        )
    common = sorted(set(a["logits"].tolist()) & set(b["logits"].tolist()))
    prefix = "logits_h1" if target == "logits" else "semantic_h1"
    index_a = {
        int(token): int(a["slices"][prefix][i]) for i, token in enumerate(a["logits"])
    }
    index_b = {
        int(token): int(b["slices"][prefix][i]) for i, token in enumerate(b["logits"])
    }
    return np.asarray([index_a[token] for token in common]), np.asarray(
        [index_b[token] for token in common]
    )


def _effect_cosine(
    a: dict[str, Any], b: dict[str, Any], transform: np.ndarray, target: str
) -> tuple[float | None, int]:
    ia, ib = _common_indices(a, b, target)
    if not len(ia):
        return None, 0
    response_a = a["matrix"][ia] @ a["basis"].T
    response_b = b["matrix"][ib] @ b["basis"].T @ transform.T
    scores = [_cosine(response_a[:, i], response_b[:, i]) for i in range(RANK)]
    qualified = [value for value in scores if value is not None]
    return (float(np.mean(qualified)) if qualified else None), len(ia)


def _rank95(matrix: np.ndarray) -> int:
    singular = np.linalg.svd(matrix, compute_uv=False)
    energy = singular**2
    return int(np.searchsorted(np.cumsum(energy) / energy.sum(), 0.95) + 1)


def analyze(root: Path) -> dict[str, Any]:
    freeze = verify_base(root)
    stage = root / "artifacts/finite_causal_control_v14_transport.freeze.json"
    if not stage.exists():
        freeze_stage(
            root,
            "transport",
            [
                "src/jclosure/experiments/transport_v14.py",
                "results/v13/processed/causal_probe_scaling_v13.parquet",
            ],
            {
                "rank": RANK,
                "j_response_ridge": RIDGE,
                "methods": [
                    "unaligned",
                    "procrustes",
                    "projector",
                    "grassmann_geodesic",
                    "j_response",
                ],
                "selection_uses_final_test": False,
                "source_split": "V13 train exact-JVP anchors",
            },
        )
    states = _states(root)
    methods = [
        "unaligned",
        "procrustes",
        "projector",
        "grassmann_geodesic",
        "j_response",
    ]
    pair_rows = []
    for a in states:
        for b in states:
            if a["id"] == b["id"]:
                continue
            relation = (
                "successive_position"
                if a["base_trial_id"] == b["base_trial_id"]
                else "same_family"
                if a["family"] == b["family"]
                else "across_family"
            )
            for method in methods:
                transform = _alignment(a, b, method)
                raw_a = a["basis"]
                raw_b = transform @ b["basis"]
                row: dict[str, Any] = {
                    "source": a["id"],
                    "destination": b["id"],
                    "relation": relation,
                    "method": method,
                    "raw_direction_cosine": float(
                        np.mean([_cosine(raw_a[i], raw_b[i]) for i in range(RANK)])
                    ),
                    "coordinate_norm_ratio": float(
                        np.linalg.norm(transform) / np.sqrt(RANK)
                    ),
                }
                for target in ("j", "logits", "semantic_continuous", "workspace"):
                    (
                        row[f"{target}_direction_cosine"],
                        row[f"{target}_common_dimensions"],
                    ) = _effect_cosine(a, b, transform, target)
                pair_rows.append(row)
    by_case: dict[str, list[dict[str, Any]]] = {}
    for state in states:
        by_case.setdefault(state["base_trial_id"], []).append(state)
    loops = []
    path_rows = []
    for case, positions in by_case.items():
        if len(positions) != 2:
            continue
        a, b = sorted(positions, key=lambda value: value["position"])
        candidates = [
            other
            for other in states
            if other["base_trial_id"] != case and other["family"] == a["family"]
        ]
        if not candidates:
            candidates = [other for other in states if other["base_trial_id"] != case]
        c = min(
            candidates, key=lambda value: np.linalg.norm(a["matrix"] - value["matrix"])
        )
        raw_union = np.vstack([a["basis"], b["basis"], c["basis"]])
        raw_rank = _rank95(raw_union)
        path_rows.append(
            {
                "case": case,
                "states": [a["id"], b["id"], c["id"]],
                "naive_raw_union_r95": raw_rank,
                "transported_coordinate_dimension": RANK,
                "rank_comparison_interpretation": "different ambient spaces; coordinate dimension is not raw-state span rank",
            }
        )
        for method in methods:
            ab = _alignment(a, b, method)
            bc = _alignment(b, c, method)
            ca = _alignment(c, a, method)
            ac = _alignment(a, c, method)
            around = ab @ bc @ ca
            indirect = ab @ bc
            direct = ac
            ja = a["matrix"][a["slices"]["j_h1"]] @ a["basis"].T
            loops.append(
                {
                    "case": case,
                    "method": method,
                    "loop": [a["id"], b["id"], c["id"], a["id"]],
                    "coordinate_return_error": float(
                        np.linalg.norm(around - np.eye(RANK)) / np.sqrt(RANK)
                    ),
                    "subspace_holonomy_degrees": float(
                        np.degrees(np.arccos(np.clip((np.trace(around) / RANK), -1, 1)))
                    ),
                    "j_effect_return_error": float(
                        np.linalg.norm(ja @ (around - np.eye(RANK)).T)
                        / max(np.linalg.norm(ja), 1e-20)
                    ),
                    "direct_indirect_coordinate_error": float(
                        np.linalg.norm(indirect - direct) / np.sqrt(RANK)
                    ),
                }
            )
    ROOT.mkdir(parents=True, exist_ok=True)
    outputs = {
        "transport_pairs": pd.DataFrame(pair_rows),
        "holonomy_loops": pd.DataFrame(loops),
        "transported_path_dimension": pd.DataFrame(path_rows),
    }
    paths = {}
    for name, frame in outputs.items():
        path = root / ROOT / f"{name}_v14.parquet"
        frame.to_parquet(path, index=False, compression="zstd")
        paths[name] = {
            "path": str(path.relative_to(root)),
            "sha256": sha256_file(path),
            "rows": len(frame),
        }
    summary = {
        "protocol_version": "finite_causal_control_v14_transport",
        "parent_freeze_digest": freeze["freeze_digest"],
        "stage_freeze_sha256": sha256_file(stage),
        "rank": RANK,
        "records": paths,
        "same_prompt_comparison": (
            outputs["transport_pairs"]
            .query('relation == "successive_position"')
            .groupby("method")[
                [
                    "raw_direction_cosine",
                    "j_direction_cosine",
                    "logits_direction_cosine",
                    "semantic_continuous_direction_cosine",
                    "workspace_direction_cosine",
                ]
            ]
            .mean()
            .reset_index()
            .to_dict("records")
        ),
        "holonomy": outputs["holonomy_loops"]
        .groupby("method")[
            [
                "coordinate_return_error",
                "subspace_holonomy_degrees",
                "j_effect_return_error",
                "direct_indirect_coordinate_error",
            ]
        ]
        .mean()
        .reset_index()
        .to_dict("records"),
    }
    write_json_atomic(root / ROOT / "transport_analysis_v14.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["analyze"], required=True)
    parser.parse_args()
    print(json.dumps(analyze(Path.cwd()), sort_keys=True)[:2000])


if __name__ == "__main__":
    main()
