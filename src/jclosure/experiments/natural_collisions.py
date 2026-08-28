"""Phase 4 observational search for natural J-state collisions."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import OneHotEncoder

from jclosure.datasets import (
    TaskExample,
    generate_equal_intermediate_pairs,
    generate_graph_traversal,
)
from jclosure.experiments.closure import (
    _flexible_examples,
    _record_clean,
    _task_pool,
)
from jclosure.experiments.common import (
    concept_vocabulary_path,
    initialize_context,
    require_phase0_gate,
    standard_parser,
)
from jclosure.jstate import ConceptVocabulary, JStateEncoder, jstate_distance
from jclosure.metrics import jensen_shannon_from_logits
from jclosure.model import load_model_bundle
from jclosure.provenance import append_jsonl, write_json_atomic


def _collision_examples(context: Any, target: int) -> list[TaskExample]:
    pairs = generate_equal_intermediate_pairs(max(100, target), seed=context.seed)
    arithmetic = [item for pair in pairs for item in pair]
    base = _task_pool(context, target)
    graphs = generate_graph_traversal(max(100, target), seed=context.seed + 7)
    data_root = context.root / context.config["data"]["upstream_root"]
    flexible = _flexible_examples(data_root)
    # Keep a deterministic order while dropping exact duplicate IDs.
    values = [*arithmetic, *base, *graphs, *flexible]
    return list({item.example_id: item for item in values}.values())


def _states_for_layer(
    runs: list[Any], layer: int, encoder: JStateEncoder
) -> tuple[np.ndarray, np.ndarray]:
    dense: list[np.ndarray] = []
    remainder: list[np.ndarray] = []
    for run in runs:
        activation = run.activations[layer]
        dense.append(encoder.encode(activation, layer).dense_scores.float().numpy())
        remainder.append(encoder.decompose(activation, layer).remainder.float().numpy())
    return np.stack(dense), np.stack(remainder)


def select_collision_pairs(
    dense: np.ndarray,
    remainder: np.ndarray,
    template_ids: list[str],
    *,
    seed: int,
    candidate_neighbors: int = 64,
    near_j_quantile: float = 0.25,
) -> list[tuple[int, int, float, float]]:
    """PCA retrieve, exact rerank, and maximize remainder distance in a J stratum."""

    if len(dense) < 3:
        return []
    dimensions = min(256, dense.shape[1], len(dense) - 1)
    projected = PCA(n_components=dimensions, random_state=seed).fit_transform(dense)
    neighbors = min(candidate_neighbors + 1, len(dense))
    indices = (
        NearestNeighbors(n_neighbors=neighbors, algorithm="auto")
        .fit(projected)
        .kneighbors(projected, return_distance=False)
    )
    selected: list[tuple[int, int, float, float]] = []
    for anchor, candidates in enumerate(indices):
        exact: list[tuple[int, float, float]] = []
        for candidate in candidates:
            candidate = int(candidate)
            if candidate == anchor or template_ids[candidate] == template_ids[anchor]:
                continue
            j_distance = float(1.0 - np.clip(dense[anchor] @ dense[candidate], -1, 1))
            r_distance = float(np.linalg.norm(remainder[anchor] - remainder[candidate]))
            exact.append((candidate, j_distance, r_distance))
        if not exact:
            continue
        cutoff = float(np.quantile([item[1] for item in exact], near_j_quantile))
        eligible = [item for item in exact if item[1] <= cutoff + 1e-12]
        donor, d_j, d_r = max(eligible, key=lambda item: (item[2], -item[1], -item[0]))
        selected.append((anchor, donor, d_j, d_r))
    return selected


def _regression(records: pd.DataFrame) -> dict[str, Any]:
    if len(records) < 10:
        return {"status": "INSUFFICIENT_DATA", "n": len(records)}
    categorical = OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore")
    controls = categorical.fit_transform(records[["layer", "task_family"]].astype(str))
    x = np.column_stack(
        [records["current_j_distance"], records["remainder_distance"], controls]
    )
    results: dict[str, Any] = {"status": "COMPLETED", "n": len(records)}
    for outcome in ("future_j_distance", "output_js_divergence"):
        model = LinearRegression().fit(x, records[outcome])
        results[outcome] = {
            "remainder_coefficient": float(model.coef_[1]),
            "current_j_coefficient": float(model.coef_[0]),
            "r_squared": float(model.score(x, records[outcome])),
        }
    return results


def main() -> None:
    parser = standard_parser(
        "Search for observational natural J-state collisions", "configs/confirm.yaml"
    )
    parser.add_argument("--candidate-neighbors", type=int, default=64)
    args = parser.parse_args()
    context = initialize_context("natural-collisions", args)
    try:
        if args.dry_run:
            context.finish("DRY_RUN")
            return
        gate = require_phase0_gate(context)
        bundle = load_model_bundle(context.config)
        vocabulary = ConceptVocabulary.from_json(
            concept_vocabulary_path(context)
        )
        encoder = JStateEncoder.from_lens(
            bundle.lens,
            bundle.unembedding_weight,
            vocabulary,
            k=int(context.config["jstate"]["k"]),
        )
        band = [int(value) for value in gate["workspace_band"]]
        target = int(context.config.get("run", {}).get("valid_per_cell", 500))
        examples = _collision_examples(context, target)
        runs = _record_clean(bundle, examples, band, args.limit or max(500, target * 4))
        raw_path = context.raw_dir / context.run_id / "collisions.jsonl"
        records: list[dict[str, Any]] = []
        for layer_index, layer in enumerate(band[:-1]):
            dense, remainder = _states_for_layer(runs, layer, encoder)
            pairs = select_collision_pairs(
                dense,
                remainder,
                [run.example.template_id for run in runs],
                seed=context.seed + layer,
                candidate_neighbors=args.candidate_neighbors,
            )
            for anchor_index, donor_index, current_j, remainder_distance in pairs:
                anchor, donor = runs[anchor_index], runs[donor_index]
                future = [
                    jstate_distance(
                        encoder.encode(anchor.activations[later], later),
                        encoder.encode(donor.activations[later], later),
                    )
                    for later in band[layer_index + 1 :]
                ]
                record = {
                    "schema_version": 1,
                    "run_id": context.run_id,
                    "anchor_id": anchor.example.example_id,
                    "donor_id": donor.example.example_id,
                    "template_id": anchor.example.template_id,
                    "donor_template_id": donor.example.template_id,
                    "task_family": anchor.example.family,
                    "donor_task_family": donor.example.family,
                    "layer": layer,
                    "position": -1,
                    "current_j_distance": current_j,
                    "remainder_distance": remainder_distance,
                    "future_j_distance": float(np.mean(future)) if future else 0.0,
                    "future_j_distances": future,
                    "output_js_divergence": jensen_shannon_from_logits(
                        anchor.logits, donor.logits
                    ),
                    "same_early_intermediate": bool(
                        anchor.example.intermediates
                        and donor.example.intermediates
                        and anchor.example.intermediates[0]
                        == donor.example.intermediates[0]
                    ),
                    "observational_only": True,
                    "seed": context.seed,
                }
                records.append(record)
        append_jsonl(raw_path, records)
        table = pd.DataFrame(records)
        table.to_parquet(
            context.processed_dir / f"natural_collisions_{context.run_id}.parquet",
            index=False,
        )
        regression = _regression(table)
        write_json_atomic(
            context.processed_dir / f"natural_collision_regression_{context.run_id}.json",
            {"schema_version": 1, "run_id": context.run_id, **regression},
        )
        context.finish(
            "COMPLETED",
            teacher_correct_runs=len(runs),
            collision_pairs=len(records),
            observational_only=True,
        )
    except KeyboardInterrupt:
        context.finish("FAILED", error="KeyboardInterrupt: run cancelled")
        raise
    except Exception as exc:
        context.finish("FAILED", error=f"{type(exc).__name__}: {exc}")
        raise


if __name__ == "__main__":
    main()
