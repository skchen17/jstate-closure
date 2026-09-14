"""Screen 64--512D J-centered predictive states and build a Pareto record."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge

from jclosure.compact_memory_v3_1 import LinearRepresentation, row_cosine
from jclosure.experiments.common import initialize_context, standard_parser
from jclosure.experiments.compact_memory_v3_1 import (
    _causal_regression_pairs,
    _phase0_regression_profiles,
    _profile_pass10,
)
from jclosure.experiments.traces_v4 import ACTION_SURFACES
from jclosure.geometry import DenseJMap
from jclosure.jstate import ConceptVocabulary, JStateEncoder
from jclosure.model import load_model_bundle
from jclosure.predictive_state_v4 import (
    PredictiveLossWeights,
    decode_numpy,
    encode_numpy,
    train_predictive_bottleneck,
)
from jclosure.protocol_v4 import verify_program_freeze
from jclosure.provenance import sha256_file, write_json_atomic
from jclosure.records_v4 import PredictiveStateRecord

PROTOCOL = "predictive_state_v4"


def _trace_summary(context: Any) -> dict[str, Any]:
    return json.loads(
        (context.processed_dir / "teacher_traces_v4.json").read_text(encoding="utf-8")
    )


def _domain_arrays(
    context: Any, summary: dict[str, Any], domain: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]:
    record = next(value for value in summary["domains"] if value["domain"] == domain)
    if sha256_file(context.root / record["tensor_path"]) != record["tensor_sha256"]:
        raise RuntimeError(f"trace hash mismatch for {domain}")
    with np.load(context.root / record["tensor_path"], allow_pickle=False) as payload:
        states = payload["j_profiles"].astype(np.float32)
        full = payload["full_states"].astype(np.float32)
        actions = payload["actions"].astype(np.int64)
    trajectories = pd.read_parquet(context.root / record["trajectories"])
    pairs_x = []
    pairs_y = []
    pair_actions = []
    metadata = []
    for row in trajectories.itertuples():
        start, stop = int(row.start), int(row.stop)
        if stop - start < 2:
            continue
        pairs_x.append(states[start : stop - 1])
        pairs_y.append(states[start + 1 : stop])
        pair_actions.append(actions[start + 1 : stop])
        metadata.append(
            {
                "example_id": str(row.example_id),
                "family": str(row.family),
                "horizon": int(row.horizon),
                "states": states[start:stop],
                "full_states": full[start:stop],
                "actions": actions[start:stop],
            }
        )
    return (
        np.concatenate(pairs_x),
        np.concatenate(pairs_y),
        np.concatenate(pair_actions),
        metadata,
    )


def _encoder(context: Any, bundle: Any) -> tuple[Any, Any, DenseJMap]:
    size = int(context.config["compact_state_v4"]["dictionary_size"])
    vocabulary = ConceptVocabulary.from_json(
        context.root / "results/processed" / f"concept_vocabulary_v2_{size}.json"
    )
    encoder = JStateEncoder.from_lens(
        bundle.lens,
        bundle.unembedding_weight,
        vocabulary,
        k=int(context.config["jstate"]["k"]),
        lazy=True,
        protocol_version=PROTOCOL,
        direction_chunk_size=int(
            context.config["jstate"].get("direction_chunk_size", 512)
        ),
    )
    return vocabulary, encoder, DenseJMap.from_encoder(encoder)


def _linear_representation(
    family: str, dimension: int, train_x: np.ndarray, train_y: np.ndarray
) -> LinearRepresentation:
    mean = train_x.mean(axis=0)
    centered = train_x - mean
    actual = min(dimension, len(train_x) - 1, train_x.shape[1])
    if family == "pca":
        model = PCA(actual, svd_solver="randomized", random_state=0).fit(train_x)
        encoder = model.components_.T
    elif family == "sparse_j":
        covariance = np.mean(centered * (train_y - train_y.mean(axis=0)), axis=0)
        scores = np.var(centered, axis=0) * (np.abs(covariance) + 1e-8)
        selected = np.argsort(-scores, kind="stable")[:actual]
        encoder = np.zeros((train_x.shape[1], actual), dtype=np.float32)
        encoder[selected, np.arange(actual)] = 1
    else:
        raise ValueError(f"unsupported linear representation: {family}")
    latent = centered @ encoder
    decoder = Ridge(alpha=1e-4, fit_intercept=False).fit(latent, centered).coef_.T
    return LinearRepresentation(
        family=family,
        dimension=actual,
        mean=mean.astype(np.float32),
        encoder=encoder.astype(np.float32),
        decoder=decoder.astype(np.float32),
    )


def _retention(
    encode: Callable[[np.ndarray], np.ndarray],
    decode: Callable[[np.ndarray], np.ndarray],
    phase0: list[dict[str, Any]],
    causal: list[tuple[np.ndarray, np.ndarray]],
    original_phase0: float,
) -> dict[str, Any]:
    reconstructed_phase0 = (
        float(
            np.mean(
                [
                    _profile_pass10(
                        decode(encode(value["profile"][None]))[0],
                        value["concept_indices"],
                    )
                    for value in phase0
                ]
            )
        )
        if phase0
        else 0.0
    )
    cosines = []
    magnitudes = []
    for clean, swapped in causal:
        original = swapped - clean
        reconstructed = (
            decode(encode(swapped[None]))[0] - decode(encode(clean[None]))[0]
        )
        cosines.append(float(row_cosine(reconstructed[None], original[None])[0]))
        original_norm = float(np.linalg.norm(original))
        ratio = float(np.linalg.norm(reconstructed)) / max(original_norm, 1e-12)
        magnitudes.append(min(ratio, 1 / max(ratio, 1e-12)))
    return {
        "semantic_pass10_original": original_phase0,
        "semantic_pass10_reconstructed": reconstructed_phase0,
        "semantic_retention": reconstructed_phase0 / max(original_phase0, 1e-12),
        "causal_trials": len(causal),
        "causal_delta_cosine_median": float(np.median(cosines)) if cosines else None,
        "causal_direction_retention": (
            float(np.mean(np.asarray(cosines) >= 0.8)) if cosines else 0.0
        ),
        "causal_magnitude_retention": (
            float(np.median(magnitudes)) if magnitudes else 0.0
        ),
    }


def _screen(context: Any, bundle: Any) -> dict[str, Any]:
    summary = _trace_summary(context)
    train_x, train_y, train_actions, train_trajectories = _domain_arrays(
        context, summary, "train"
    )
    validation_x, validation_y, _, validation_trajectories = _domain_arrays(
        context, summary, "validation"
    )
    vocabulary, encoder, dense_map = _encoder(context, bundle)
    context.config["compact_memory"] = {
        "workspace_layers": context.config["compact_state_v4"]["workspace_layers"]
    }
    phase0 = _phase0_regression_profiles(context, bundle, dense_map, vocabulary)
    causal = _causal_regression_pairs(context, bundle, encoder, dense_map)
    original_phase0 = (
        float(
            np.mean(
                [
                    _profile_pass10(value["profile"], value["concept_indices"])
                    for value in phase0
                ]
            )
        )
        if phase0
        else 0.0
    )
    section = context.config["compact_state_v4"]
    weights = PredictiveLossWeights(**section["objective_weights"])
    device = next(bundle.hf_model.parameters()).device
    artifact_root = context.root / "artifacts/predictive_state/v4" / context.run_id
    artifact_root.mkdir(parents=True, exist_ok=True)
    rows = []
    for dimension in (int(value) for value in section["dimensions"]):
        for family in section["families"]:
            if family in {"pca", "sparse_j"}:
                representation = _linear_representation(
                    str(family), dimension, train_x, train_y
                )
                transition = Ridge(alpha=1.0).fit(
                    representation.encode(train_x), train_y
                )
                encode = representation.encode
                decode = representation.decode
                prediction = transition.predict(encode(validation_x))
                path = artifact_root / f"{family}-{dimension}.npz"
                np.savez_compressed(
                    path,
                    **representation.state_dict(),
                    transition_coef=transition.coef_,
                    transition_intercept=transition.intercept_,
                )
                training_history: list[dict[str, Any]] = []
            else:
                nonlinear = family == "learned_predictive_encoder"
                model, training_history = train_predictive_bottleneck(
                    train_x,
                    train_y,
                    train_actions,
                    validation_x,
                    validation_y,
                    latent_dim=dimension,
                    action_count=len(ACTION_SURFACES),
                    nonlinear=nonlinear,
                    causal_pairs=causal,
                    weights=weights,
                    epochs=int(section["screen_epochs"]),
                    patience=int(section["screen_patience"]),
                    seed=int(context.seed),
                    device=device,
                )
                def encode(values: np.ndarray, model: Any = model) -> np.ndarray:
                    return encode_numpy(model, values, device)

                def decode(values: np.ndarray, model: Any = model) -> np.ndarray:
                    return decode_numpy(model, values, device)
                with torch.no_grad():
                    latent = model.encoder(
                        torch.from_numpy(validation_x).float().to(device)
                    )
                    prediction = (
                        torch.nn.functional.normalize(
                            model.decoder(model.transition(latent)), dim=-1
                        )
                        .cpu()
                        .numpy()
                    )
                path = artifact_root / f"{family}-{dimension}.pt"
                torch.save(
                    {
                        "family": str(family),
                        "dimension": dimension,
                        "state_dict": model.state_dict(),
                        "input_dim": train_x.shape[1],
                        "action_count": len(ACTION_SURFACES),
                        "nonlinear": nonlinear,
                    },
                    path,
                )
            reconstructed = decode(encode(validation_x))
            retention = _retention(encode, decode, phase0, causal, original_phase0)
            future_cosine = float(np.median(row_cosine(prediction, validation_y)))
            reconstruction_cosine = float(
                np.median(row_cosine(reconstructed, validation_x))
            )
            gate = bool(
                retention["semantic_retention"]
                >= float(section["minimum_semantic_retention"])
                and retention["causal_direction_retention"]
                >= float(section["minimum_causal_direction_retention"])
                and retention["causal_magnitude_retention"]
                >= float(section["minimum_causal_magnitude_retention"])
            )
            record = PredictiveStateRecord(
                representation=str(family),
                state_dimension=dimension,
                split="validation",
                semantic_retention=float(retention["semantic_retention"]),
                causal_direction_retention=float(
                    retention["causal_direction_retention"]
                ),
                causal_magnitude_retention=float(
                    retention["causal_magnitude_retention"]
                ),
                future_prediction_cosine=future_cosine,
                current_reconstruction_cosine=reconstruction_cosine,
            ).to_dict()
            record.update(
                {
                    **retention,
                    "gate_passed": gate,
                    "representation_path": str(path.relative_to(context.root)),
                    "representation_sha256": sha256_file(path),
                    "training_history": training_history,
                    "teacher_correct_train_trajectories": len(train_trajectories),
                    "teacher_correct_validation_trajectories": len(
                        validation_trajectories
                    ),
                }
            )
            rows.append(record)
    eligible = [value for value in rows if value["gate_passed"]]
    selected = (
        sorted(
            eligible,
            key=lambda value: (
                -float(value["future_prediction_cosine"]),
                int(value["state_dimension"]),
                str(value["representation"]),
            ),
        )[0]
        if eligible
        else None
    )
    records_path = context.processed_dir / "predictive_state_pareto_v4.parquet"
    pd.DataFrame(rows).drop(columns=["training_history"]).to_parquet(
        records_path, index=False, compression="zstd"
    )
    result = {
        "schema_version": 6,
        "protocol_version": PROTOCOL,
        "run_id": context.run_id,
        "records": str(records_path.relative_to(context.root)),
        "phase0_regression_items": len(phase0),
        "causal_regression_trials": len(causal),
        "selected": selected,
        "compact_state_authorized": selected is not None,
        "pareto": rows,
    }
    output = context.processed_dir / "predictive_state_screen_v4.json"
    write_json_atomic(output, result)
    return result


def main() -> None:
    parser = standard_parser(
        "Screen protocol-v4 compact predictive states", "configs/predictive_v4.yaml"
    )
    args = parser.parse_args()
    context = initialize_context("predictive-state-v4", args)
    try:
        verify_program_freeze(context.root, context.config)
        if args.dry_run:
            context.finish("DRY_RUN")
            return
        bundle = load_model_bundle(context.config)
        result = _screen(context, bundle)
        context.finish(
            "COMPLETED",
            compact_state_authorized=result["compact_state_authorized"],
            selected=result["selected"],
        )
    except KeyboardInterrupt:
        context.finish("FAILED", error="KeyboardInterrupt")
        raise
    except Exception as exc:
        context.finish("FAILED", error=f"{type(exc).__name__}: {exc}")
        raise


if __name__ == "__main__":
    main()
