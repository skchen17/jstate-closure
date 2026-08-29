"""Conditional low-dimensional operational-state search for protocol v3.

This stage estimates held-out next-state prediction and reconstruction for
fixed compression families.  It deliberately does not authorize a compact
state until separate Phase-0-regression and causal-intervention records exist.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge

from jclosure.experiments.common import initialize_context, standard_parser
from jclosure.experiments.geometry_v3 import _load_bank
from jclosure.geometry import DenseJMap
from jclosure.jstate import ConceptVocabulary, JStateEncoder
from jclosure.model import load_model_bundle
from jclosure.protocol_v3 import verify_v3_freeze
from jclosure.provenance import sha256_file, write_json_atomic


def _bank_path(root: Path, freeze: dict[str, Any]) -> Path:
    paths = [
        root / value
        for value in freeze["data_hashes"]
        if value.endswith("activation_bank_manifest.jsonl")
    ]
    if len(paths) != 1:
        raise RuntimeError("v3 freeze does not identify one activation bank")
    return paths[0]


def _latest_completed_geometry_dirs(root: Path, stage: str) -> list[Path]:
    latest: dict[int, tuple[str, Path]] = {}
    for path in sorted((root / "results/v3/raw").glob("geometry-v3-*/manifest.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("status") != "COMPLETED" or payload.get("stage") != stage:
            continue
        if stage == "pareto" and payload.get("limit") is not None:
            continue
        shard = int(payload.get("shard_index", 0))
        created = str(payload.get("created_at", ""))
        if shard not in latest or created > latest[shard][0]:
            latest[shard] = (created, path.parent)
    return [latest[index][1] for index in sorted(latest)]


def _triggered(
    root: Path, freeze: dict[str, Any] | None = None
) -> tuple[bool, dict[str, Any]]:
    if freeze is not None:
        frozen_paths = [root / path for path in freeze.get("data_hashes", {})]
        local_paths = [
            path
            for path in frozen_paths
            if path.name.startswith("local_spectra-")
            and not path.name.endswith("-smoke.parquet")
        ]
        pareto_paths = [
            path
            for path in frozen_paths
            if path.name.startswith("pareto_records-shard-")
        ]
    else:
        spectrum_dirs = _latest_completed_geometry_dirs(root, "spectrum")
        local_paths = [
            path
            for directory in spectrum_dirs
            for path in sorted(directory.glob("local_spectra-*.parquet"))
            if not path.name.endswith("-smoke.parquet")
        ]
        pareto_dirs = _latest_completed_geometry_dirs(root, "pareto")
        pareto_paths = [
            path
            for directory in pareto_dirs
            for path in sorted(directory.glob("pareto_records-shard-*.parquet"))
        ]
    if not local_paths:
        return False, {"reason": "local_geometry_missing"}
    local = pd.concat([pd.read_parquet(path) for path in local_paths], ignore_index=True)

    def value(mapping: Any, key: str) -> float:
        return float(mapping.get(key, np.nan)) if isinstance(mapping, dict) else np.nan

    ranks = local["tolerance_ranks"].map(
        lambda item: value(item, "relative_1e-04")
    )
    tangent = local["tangent_null_dimensions"].map(
        lambda item: value(item, "relative_1e-04")
    )
    near_injective = bool(
        ranks.median() >= 0.99 * (2560 - 1) or tangent.median() <= 25
    )
    max_displacement = None
    if pareto_paths:
        pareto = pd.concat(
            [pd.read_parquet(path) for path in pareto_paths], ignore_index=True
        )
        feasible = pareto[
            (pareto["dense_cosine"] >= 0.995)
            & (pareto["top10_overlap"] >= 0.8)
            & (pareto["rms_drift"] <= 0.02)
            & pareto["natural"]
        ]
        if not feasible.empty:
            max_displacement = float(feasible["displacement_fraction"].max())
    triggered = near_injective or max_displacement is None or max_displacement < 0.20
    return triggered, {
        "near_injective": near_injective,
        "median_rank_1e4": float(ranks.median()),
        "median_tangent_null_1e4": float(tangent.median()),
        "max_formal_displacement": max_displacement,
    }


def _cosine_rows(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    numerator = np.sum(left * right, axis=1)
    denominator = np.linalg.norm(left, axis=1) * np.linalg.norm(right, axis=1)
    return numerator / np.maximum(denominator, 1e-12)


def _clusters(values: np.ndarray, dimension: int) -> np.ndarray:
    output = np.zeros((len(values), dimension), dtype=np.float32)
    counts = np.zeros(dimension, dtype=np.float32)
    for index in range(values.shape[1]):
        cluster = index % dimension
        output[:, cluster] += values[:, index]
        counts[cluster] += 1
    return output / np.maximum(counts, 1)


def _constrained_predictive_components(
    predictive_rows: np.ndarray,
    *,
    feature_count: int,
    frozen_indices: list[int] | tuple[int, ...],
) -> tuple[np.ndarray, np.ndarray]:
    frozen = sorted({int(index) for index in frozen_indices})
    if any(index < 0 or index >= feature_count for index in frozen):
        raise ValueError("frozen concept index is outside the state vector")
    fixed = np.zeros((feature_count, len(frozen)), dtype=np.float64)
    if frozen:
        fixed[np.asarray(frozen), np.arange(len(frozen))] = 1.0
    learned_rows = np.asarray(predictive_rows, dtype=np.float64).copy()
    if frozen:
        learned_rows[:, frozen] = 0.0
    _, singular_values, right = np.linalg.svd(learned_rows, full_matrices=False)
    threshold = max(singular_values.max(initial=0.0), 1.0) * 1e-10
    rank = int(np.count_nonzero(singular_values > threshold))
    return fixed.astype(np.float32), right[:rank].T.astype(np.float32)


def _assemble_constrained_basis(
    fixed: np.ndarray, learned: np.ndarray, dimension: int
) -> np.ndarray | None:
    if fixed.shape[1] > dimension:
        return None
    needed = dimension - fixed.shape[1]
    if learned.shape[1] < needed:
        return None
    return np.concatenate([fixed, learned[:, :needed]], axis=1).astype(np.float32)


def _constrained_predictive_basis(
    predictive_rows: np.ndarray,
    *,
    dimension: int,
    feature_count: int,
    frozen_indices: list[int] | tuple[int, ...],
) -> np.ndarray | None:
    fixed, learned = _constrained_predictive_components(
        predictive_rows,
        feature_count=feature_count,
        frozen_indices=frozen_indices,
    )
    return _assemble_constrained_basis(fixed, learned, dimension)


def _ridge_predict(
    train_x: np.ndarray,
    test_x: np.ndarray,
    train_target: np.ndarray,
    target_pca: PCA,
) -> np.ndarray:
    model = Ridge(alpha=1.0, random_state=0)
    model.fit(train_x, train_target)
    return target_pca.inverse_transform(model.predict(test_x))


def _extract(
    root: Path,
    records: list[dict[str, Any]],
    layers: list[int],
    encoder: JStateEncoder,
    dense_map: DenseJMap,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for record in records:
        payload = torch.load(root / record["activation_path"], map_location="cpu")
        for current, following in zip(layers, layers[1:], strict=False):
            h = payload["activations"][current][-1].float()
            next_h = payload["activations"][following][-1].float()
            state = dense_map.dense_state(h, current).cpu().numpy()
            next_state = dense_map.dense_state(next_h, following).cpu().numpy()
            decomposition = encoder.decompose(h, current)
            remainder = decomposition.remainder.cpu().numpy()
            sparse_state = np.zeros(len(encoder.vocabulary.token_ids), dtype=np.float32)
            sparse_state[decomposition.atom_indices.cpu().numpy()] = (
                decomposition.coefficients.detach().cpu().float().numpy()
            )
            rows.append(
                {
                    "prompt_id": record["prompt_id"],
                    "task_family": record["task_family"],
                    "split": record["split"],
                    "layer": current,
                    "next_layer": following,
                    "state": state,
                    "next_state": next_state,
                    "remainder": remainder,
                    "sparse_state": sparse_state,
                }
            )
    return pd.DataFrame(rows)


def _single_token_ids(tokenizer: Any, surface: str) -> set[int]:
    output: set[int] = set()
    for candidate in (surface, " " + surface):
        encoded = tokenizer.encode(candidate, add_special_tokens=False)
        if len(encoded) == 1:
            output.add(int(encoded[0]))
    return output


def _frozen_concept_indices(
    root: Path, vocabulary: ConceptVocabulary, tokenizer: Any
) -> tuple[list[int], dict[str, Any]]:
    calibration_path = root / "results/processed/phase0_v2_calibration.json"
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    expected_readout_hash = str(calibration["readout_records_sha256"])
    readout_candidates = sorted(
        (root / "results/raw").glob("phase0-v2-*/readout_records_v2.parquet")
    )
    readout_path = next(
        (path for path in readout_candidates if sha256_file(path) == expected_readout_hash),
        None,
    )
    if readout_path is None:
        raise RuntimeError("frozen Phase 0 v2 readout records were not found by hash")
    readout = pd.read_parquet(readout_path)
    token_ids = {
        int(value)
        for value in readout.loc[
            (readout["method"] == "jacobian") & readout["tokenizable"],
            "winning_token_id",
        ].dropna()
    }
    positive_paths = sorted(
        (root / "results/raw").glob("phase0-v2-*/positive_control_records_v2.parquet")
    )
    positive_path = positive_paths[-1] if positive_paths else None
    if positive_path is not None:
        positive = pd.read_parquet(positive_path)
        for column in ("source", "target"):
            for surface in positive[column].dropna().astype(str).unique():
                token_ids.update(_single_token_ids(tokenizer, surface))
    by_token = {token_id: index for index, token_id in enumerate(vocabulary.token_ids)}
    indices = sorted(by_token[token_id] for token_id in token_ids if token_id in by_token)
    return indices, {
        "readout_records": str(readout_path.relative_to(root)),
        "readout_records_sha256": expected_readout_hash,
        "positive_control_records": (
            None if positive_path is None else str(positive_path.relative_to(root))
        ),
        "positive_control_records_sha256": (
            None if positive_path is None else sha256_file(positive_path)
        ),
        "candidate_token_ids": len(token_ids),
        "dictionary_indices": len(indices),
    }


def run_search(
    frame: pd.DataFrame,
    dimensions: list[int],
    *,
    frozen_concept_indices: list[int] | tuple[int, ...] = (),
) -> tuple[pd.DataFrame, dict[str, Any]]:
    train = frame[frame["split"] == "fit"]
    test = frame[frame["split"] == "audit"]
    if train.empty or test.empty:
        raise RuntimeError("low-dimensional search requires fit and audit states")
    train_x = np.stack(train["state"])
    test_x = np.stack(test["state"])
    train_y = np.stack(train["next_state"])
    test_y = np.stack(test["next_state"])
    train_r = np.stack(train["remainder"])
    test_r = np.stack(test["remainder"])
    target_components = min(512, len(train_y) - 1, train_y.shape[1])
    target_pca = PCA(target_components, random_state=0).fit(train_y)
    train_target = target_pca.transform(train_y)
    persistence = float(np.median(_cosine_rows(test_x, test_y)))
    oracle_pca = PCA(
        min(512, len(train_r) - 1, train_r.shape[1]), random_state=0
    ).fit(train_r)
    oracle_train = np.concatenate([train_x, oracle_pca.transform(train_r)], axis=1)
    oracle_test = np.concatenate([test_x, oracle_pca.transform(test_r)], axis=1)
    oracle_prediction = _ridge_predict(
        oracle_train, oracle_test, train_target, target_pca
    )
    oracle_cosine = float(np.median(_cosine_rows(oracle_prediction, test_y)))
    rows: list[dict[str, Any]] = []
    if "sparse_state" in frame.columns:
        train_sparse = np.stack(train["sparse_state"])
        test_sparse = np.stack(test["sparse_state"])
        sparse_prediction = _ridge_predict(
            train_sparse, test_sparse, train_target, target_pca
        )
        active_atoms = int(
            max(
                np.count_nonzero(train_sparse, axis=1).max(initial=0),
                np.count_nonzero(test_sparse, axis=1).max(initial=0),
            )
        )
        rows.append(
            {
                "candidate": "sparse_active_atoms",
                "dimension": 2 * active_atoms,
                "feature_width": train_sparse.shape[1],
                "next_state_cosine_median": float(
                    np.median(_cosine_rows(sparse_prediction, test_y))
                ),
                "state_reconstruction_cosine_median": None,
                "frozen_concept_count": 0,
            }
        )
    for dimension in dimensions:
        actual = min(int(dimension), len(train_x) - 1, train_x.shape[1])
        pca = PCA(actual, random_state=0).fit(train_x)
        train_pca = pca.transform(train_x)
        test_pca = pca.transform(test_x)
        prediction = _ridge_predict(train_pca, test_pca, train_target, target_pca)
        reconstruction = pca.inverse_transform(test_pca)
        rows.append(
            {
                "candidate": "dense_profile_pca",
                "dimension": actual,
                "next_state_cosine_median": float(
                    np.median(_cosine_rows(prediction, test_y))
                ),
                "state_reconstruction_cosine_median": float(
                    np.median(_cosine_rows(reconstruction, test_x))
                ),
            }
        )
        train_cluster = _clusters(train_x, actual)
        test_cluster = _clusters(test_x, actual)
        prediction = _ridge_predict(
            train_cluster, test_cluster, train_target, target_pca
        )
        rows.append(
            {
                "candidate": "deterministic_concept_clusters",
                "dimension": actual,
                "next_state_cosine_median": float(
                    np.median(_cosine_rows(prediction, test_y))
                ),
                "state_reconstruction_cosine_median": None,
            }
        )
    full_model = Ridge(alpha=1.0, random_state=0).fit(train_x, train_target)
    # Ridge coef is [target_components, concepts]; right singular vectors are
    # the supervised input bottleneck basis.
    _, _, right = np.linalg.svd(full_model.coef_, full_matrices=False)
    constrained_fixed, constrained_learned = _constrained_predictive_components(
        right,
        feature_count=train_x.shape[1],
        frozen_indices=frozen_concept_indices,
    )
    for dimension in dimensions:
        actual = min(int(dimension), right.shape[0])
        basis = right[:actual].T
        prediction = _ridge_predict(
            train_x @ basis, test_x @ basis, train_target, target_pca
        )
        rows.append(
            {
                "candidate": "predictive_linear_bottleneck",
                "dimension": actual,
                "next_state_cosine_median": float(
                    np.median(_cosine_rows(prediction, test_y))
                ),
                "state_reconstruction_cosine_median": float(
                    np.median(_cosine_rows((test_x @ basis) @ basis.T, test_x))
                ),
            }
        )
        constrained_basis = _assemble_constrained_basis(
            constrained_fixed, constrained_learned, actual
        )
        if constrained_basis is not None:
            constrained_prediction = _ridge_predict(
                train_x @ constrained_basis,
                test_x @ constrained_basis,
                train_target,
                target_pca,
            )
            rows.append(
                {
                    "candidate": "constrained_learned_encoder",
                    "dimension": actual,
                    "next_state_cosine_median": float(
                        np.median(_cosine_rows(constrained_prediction, test_y))
                    ),
                    "state_reconstruction_cosine_median": float(
                        np.median(
                            _cosine_rows(
                                (test_x @ constrained_basis) @ constrained_basis.T,
                                test_x,
                            )
                        )
                    ),
                    "frozen_concept_count": len(set(frozen_concept_indices)),
                    "frozen_concepts_exactly_embedded": True,
                }
            )
    result = pd.DataFrame(rows)
    denominator = oracle_cosine - persistence
    result["oracle_gap_closed"] = (
        (result["next_state_cosine_median"] - persistence) / denominator
        if denominator > 1e-12
        else np.nan
    )
    result["persistence_cosine_median"] = persistence
    result["remainder_oracle_cosine_median"] = oracle_cosine
    return result, {
        "train_samples": len(train),
        "test_samples": len(test),
        "persistence_cosine_median": persistence,
        "remainder_oracle_cosine_median": oracle_cosine,
        "oracle_improves_over_persistence": denominator > 1e-12,
        "frozen_concept_count": len(set(frozen_concept_indices)),
    }


def main() -> None:
    parser = standard_parser(
        "Search for compact operational states after dense feasibility failure",
        "configs/geometry_v3.yaml",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    context = initialize_context("lowdim-search-v3", args)
    try:
        freeze = verify_v3_freeze(
            context.root, require_behavioral_authorization=False
        )
        triggered, trigger = _triggered(context.root, freeze)
        if not triggered and not args.force:
            context.finish("NOT_TRIGGERED", trigger=trigger)
            return
        if args.dry_run:
            context.finish("DRY_RUN", trigger=trigger)
            return
        bundle = load_model_bundle(context.config)
        vocabulary = ConceptVocabulary.from_json(
            context.root / "results/processed/concept_vocabulary_v2_4096.json"
        )
        encoder = JStateEncoder.from_lens(
            bundle.lens,
            bundle.unembedding_weight,
            vocabulary,
            k=int(context.config["jstate"]["k"]),
            lazy=True,
            protocol_version="exploratory_protocol_v3",
            direction_chunk_size=int(
                context.config["jstate"].get("direction_chunk_size", 512)
            ),
        )
        dense_map = DenseJMap.from_encoder(encoder)
        records = _load_bank(context.root, _bank_path(context.root, freeze))
        if args.limit is not None:
            records = records[: args.limit]
        layers = [int(value) for value in context.config["geometry"]["candidate_layers"]]
        frame = _extract(context.root, records, layers, encoder, dense_map)
        frozen_indices, frozen_provenance = _frozen_concept_indices(
            context.root, vocabulary, bundle.tokenizer
        )
        results, summary = run_search(
            frame,
            [int(value) for value in context.config["lowdim"]["dimensions"]],
            frozen_concept_indices=frozen_indices,
        )
        result_path = context.processed_dir / "lowdim_search.parquet"
        results.to_parquet(result_path, index=False)
        prediction_pass = results[
            (results["dimension"] <= int(context.config["lowdim"]["max_dimension"]))
            & (
                results["oracle_gap_closed"]
                >= float(context.config["lowdim"]["oracle_gap_closed"])
            )
        ]
        authorization = {
            "schema_version": 3,
            "protocol_version": "exploratory_protocol_v3",
            "authorized": False,
            "prediction_candidates_passing": prediction_pass.to_dict("records"),
            "phase0_regression": "UNEXECUTED",
            "intervention_retention": "UNEXECUTED",
            "frozen_concept_provenance": frozen_provenance,
            "reason": (
                "Prediction screening alone cannot authorize a compact state. "
                "Frozen Phase 0 pass@10 retention and causal intervention retention "
                "records are required."
            ),
            "trigger": trigger,
            **summary,
        }
        authorization_path = (
            context.processed_dir / "compact_state_authorization.json"
        )
        write_json_atomic(authorization_path, authorization)
        context.finish(
            "COMPLETED_NOT_AUTHORIZED",
            trigger=trigger,
            results=str(result_path.relative_to(context.root)),
            authorization=str(authorization_path.relative_to(context.root)),
        )
    except KeyboardInterrupt:
        context.finish("FAILED", error="KeyboardInterrupt: run cancelled")
        raise
    except Exception as exc:
        context.finish("FAILED", error=f"{type(exc).__name__}: {exc}")
        raise


if __name__ == "__main__":
    main()
