"""Freeze v12, fit data/rank models, and serialize strict intervention states."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from jclosure.experiments.common import initialize_context, standard_parser
from jclosure.experiments.compress_persistent_v8 import (
    _capture_manifests,
    _load_raw_deltas,
)
from jclosure.experiments.decoded_causal_v10 import (
    _decode_block,
    _dual_reconstruction_alpha,
    _raw_order,
)
from jclosure.experiments.sufficiency_v9 import _load_data
from jclosure.protocol_v12 import (
    PREPARED_FREEZE_PATH,
    PROTOCOL_V12,
    SCHEMA_VERSION_V12,
    build_base_freeze,
    build_derived_freeze,
    verify_base_freeze,
)
from jclosure.provenance import sha256_file, write_json_atomic
from jclosure.state_models_v12 import (
    CHANNELS,
    V12StateBank,
    fit_causal_basis,
    fit_pca_basis,
    nested_indices,
    project_basis,
    scaling_records,
    target_bundle,
)

METHOD_SPECS = Path("results/v12/processed/method_specs_v12.json")
SCALING_RECORDS = Path("results/v12/processed/data_rank_scaling_v12.parquet")
DEVELOPMENT_STATES = Path("results/v12/processed/development_states_v12.json")
SCALING_STATES = Path("results/v12/processed/scaling_validation_states_v12.json")
CONFIRMATORY_STATES = Path("results/v12/processed/confirmatory_states_v12.json")


def _bank(
    context: Any, base: dict[str, Any]
) -> tuple[V12StateBank, dict[str, Any], np.ndarray]:
    data = _load_data(context.root)
    lookup = {str(value): index for index, value in enumerate(data["ids"].tolist())}
    ids = base["train_scaling"]["nested_base_trial_ids"]["600"]
    train = np.asarray([lookup[str(value)] for value in ids], dtype=int)
    return V12StateBank(data, train), data, train


def _selected(data: dict[str, Any], ids: list[str]) -> np.ndarray:
    lookup = {str(value): index for index, value in enumerate(data["ids"].tolist())}
    return np.asarray([lookup[str(value)] for value in ids], dtype=int)


def _write_specs(context: Any, base: dict[str, Any]) -> dict[str, Any]:
    bank, data, _ = _bank(context, base)
    section = context.config["causal_geometry_v12"]
    oracle = bank.specs(
        [int(value) for value in section["local_geometry"]["oracle_dimensions"]]
    )
    rows = scaling_records(data, base, context.config)
    scaling_frame = pd.DataFrame(rows)
    path = context.root / SCALING_RECORDS
    path.parent.mkdir(parents=True, exist_ok=True)
    scaling_frame.to_parquet(path, index=False, compression="zstd")
    causal_scaling = []
    for row in rows:
        if row["status"] != "IDENTIFIED" or int(row["dimension"]) > 512:
            continue
        causal_scaling.append(
            {
                "method": (
                    f"scaling_{row['method']}_n{int(row['train_size'])}"
                    f"_d{int(row['dimension'])}"
                ),
                "kind": str(row["method"]),
                "class": f"scaling_{row['method']}",
                "train_size": int(row["train_size"]),
                "dimension": int(row["dimension"]),
                "status": "IDENTIFIED",
            }
        )
    payload = {
        "schema_version": SCHEMA_VERSION_V12,
        "protocol_version": PROTOCOL_V12,
        "run_id": context.run_id,
        "source_freeze_digest": base["freeze_digest"],
        "selection_split": "validation",
        "confirmatory_used": False,
        "primary_neighbors": int(section["local_geometry"]["primary_neighbors"]),
        "oracle_methods": oracle,
        "identified_oracle_methods": [
            value for value in oracle if value["status"] == "IDENTIFIED"
        ],
        "causal_scaling_methods": causal_scaling,
        "data_rank_records": str(SCALING_RECORDS),
        "data_rank_records_sha256": sha256_file(path),
    }
    write_json_atomic(context.root / METHOD_SPECS, payload)
    return payload


def _save_states(
    context: Any,
    *,
    spec: dict[str, Any],
    ids: list[str],
    inputs: dict[str, tuple[np.ndarray, np.ndarray]],
    latent: np.ndarray,
    train: np.ndarray,
    raw: dict[str, torch.Tensor],
    split: str,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    decoder = context.config["causal_geometry_v12"]["decoder"]
    requested = int(context.config["model"].get("device", 0))
    device = torch.device(f"cuda:{requested}" if torch.cuda.is_available() else "cpu")
    raw_names = {"recurrent": "recurrent", "conv": "conv", "kv": "kv_full"}
    decoded: dict[str, torch.Tensor] = {}
    diagnostics = {}
    for name in CHANNELS:
        source_scores, decoded_scores = inputs[name]
        alpha, detail = _dual_reconstruction_alpha(source_scores[train], decoded_scores)
        decoded[name] = _decode_block(
            raw[raw_names[name]],
            train,
            alpha,
            device=device,
            chunk_size=int(decoder["reconstruction_chunk"]),
        )
        diagnostics[name] = detail
    artifact_root = (
        context.root
        / "artifacts/causal/v12"
        / context.run_id
        / split
        / str(spec["method"])
    )
    artifact_root.mkdir(parents=True, exist_ok=True)
    shards = []
    shard_size = int(decoder["shard_size"])
    for start in range(0, len(ids), shard_size):
        stop = min(start + shard_size, len(ids))
        rows = [
            {
                "base_trial_id": ids[index],
                "method": spec["method"],
                "recurrent": decoded["recurrent"][index].clone(),
                "conv": decoded["conv"][index].clone(),
                "kv": decoded["kv"][index].clone(),
            }
            for index in range(start, stop)
        ]
        path = artifact_root / f"state_{start:03d}_{stop:03d}.pt"
        torch.save(
            {"format": "v12_architecture_resolved_delta_bf16", "rows": rows}, path
        )
        shards.append(
            {
                "path": str(path.relative_to(context.root)),
                "sha256": sha256_file(path),
                "base_trial_ids": ids[start:stop],
            }
        )
    del decoded
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {
        "spec": spec,
        "shards": shards,
        "decoder_diagnostics": diagnostics,
        "selected_latent_mean_square": float(np.mean(latent.astype(np.float64) ** 2)),
        "model_metadata": metadata,
    }


def _raw_data(context: Any, data: dict[str, Any]) -> dict[str, torch.Tensor]:
    raw, raw_ids, _, _, _ = _load_raw_deltas(context, _capture_manifests(context.root))
    return _raw_order(raw, raw_ids, data["ids"])


def _prepare_oracles(
    context: Any, base: dict[str, Any], *, split: str, specs: list[dict[str, Any]]
) -> dict[str, Any]:
    bank, data, train = _bank(context, base)
    ids = [str(value) for value in base[split]["base_trial_ids"]]
    selected = _selected(data, ids)
    raw = _raw_data(context, data)
    neighbors = int(
        context.config["causal_geometry_v12"]["local_geometry"]["primary_neighbors"]
    )
    methods = []
    progress = context.raw_dir / context.run_id / f"prepare_{split}_progress.json"
    for index, spec in enumerate(specs):
        inputs, latent, detail = bank.reconstruction_inputs(
            spec, selected, neighbors=neighbors
        )
        methods.append(
            _save_states(
                context,
                spec=spec,
                ids=ids,
                inputs=inputs,
                latent=latent,
                train=train,
                raw=raw,
                split=split,
                metadata=detail,
            )
        )
        write_json_atomic(
            progress, {"status": "RUNNING", "completed": index + 1, "total": len(specs)}
        )
    payload = {
        "schema_version": SCHEMA_VERSION_V12,
        "protocol_version": PROTOCOL_V12,
        "run_id": context.run_id,
        "source_freeze_digest": base["freeze_digest"],
        "split": split,
        "base_trial_ids": ids,
        "method_count": len(methods),
        "methods": methods,
    }
    target = DEVELOPMENT_STATES if split == "development" else CONFIRMATORY_STATES
    write_json_atomic(context.root / target, payload)
    write_json_atomic(
        progress, {"status": "COMPLETED", "completed": len(specs), "total": len(specs)}
    )
    return payload


def _scaling_inputs(
    data: dict[str, Any], train: np.ndarray, selected: np.ndarray, spec: dict[str, Any]
) -> tuple[dict[str, tuple[np.ndarray, np.ndarray]], np.ndarray, dict[str, Any]]:
    dimension = int(spec["dimension"])
    rank = int(data["features"].shape[1])
    kind = str(spec["kind"])
    if kind == "combined_pca":
        model = fit_pca_basis(data["features"], train)
        reconstructed, latent = project_basis(
            model, data["features"][selected], dimension
        )
        return (
            {name: (data["features"], reconstructed) for name in CHANNELS},
            latent,
            {"empirical_rank": model.rank},
        )
    model = (
        fit_pca_basis(data["layerwise"], train)
        if kind == "architecture_joint_pca"
        else fit_causal_basis(data["layerwise"], target_bundle(data), train)
    )
    reconstructed, latent = project_basis(model, data["layerwise"][selected], dimension)
    return (
        {
            name: (
                data["layerwise"][:, index * rank : (index + 1) * rank],
                reconstructed[:, index * rank : (index + 1) * rank],
            )
            for index, name in enumerate(CHANNELS)
        },
        latent,
        {"empirical_rank": model.rank},
    )


def _prepare_scaling(
    context: Any, base: dict[str, Any], specs: list[dict[str, Any]]
) -> dict[str, Any]:
    data = _load_data(context.root)
    train_sets = nested_indices(data, base)
    ids = [str(value) for value in base["validation_panel"]["base_trial_ids"]]
    selected = _selected(data, ids)
    raw = _raw_data(context, data)
    methods = []
    for spec in specs:
        train = train_sets[int(spec["train_size"])]
        inputs, latent, detail = _scaling_inputs(data, train, selected, spec)
        methods.append(
            _save_states(
                context,
                spec=spec,
                ids=ids,
                inputs=inputs,
                latent=latent,
                train=train,
                raw=raw,
                split="validation_scaling",
                metadata=detail,
            )
        )
    payload = {
        "schema_version": SCHEMA_VERSION_V12,
        "protocol_version": PROTOCOL_V12,
        "run_id": context.run_id,
        "split": "validation_panel",
        "base_trial_ids": ids,
        "method_count": len(methods),
        "methods": methods,
    }
    write_json_atomic(context.root / SCALING_STATES, payload)
    return payload


def _freeze_prepared(context: Any) -> dict[str, Any]:
    specs = json.loads((context.root / METHOD_SPECS).read_text())
    return build_derived_freeze(
        context.root,
        context.config,
        path=PREPARED_FREEZE_PATH,
        purpose="freeze v12 data/rank models and development state shards",
        inputs=[METHOD_SPECS, SCALING_RECORDS, DEVELOPMENT_STATES, SCALING_STATES],
        payload={
            "oracle_method_count": len(specs["identified_oracle_methods"]),
            "causal_scaling_method_count": len(specs["causal_scaling_methods"]),
            "confirmatory_used": False,
        },
    )


def main() -> None:
    parser = standard_parser(
        "prepare causal geometry v12", "configs/causal_geometry_v12.yaml"
    )
    parser.add_argument(
        "--stage",
        required=True,
        choices=(
            "freeze-base",
            "fit-models",
            "prepare-development",
            "prepare-scaling",
            "freeze-prepared",
            "prepare-confirmatory",
        ),
    )
    args = parser.parse_args()
    context = initialize_context("prepare-v12", args)
    try:
        if args.stage == "freeze-base":
            result = build_base_freeze(context.root, context.config)
        else:
            base = verify_base_freeze(context.root, context.config)
            if args.stage == "fit-models":
                result = _write_specs(context, base)
            else:
                specs = json.loads((context.root / METHOD_SPECS).read_text())
                if args.stage == "prepare-development":
                    result = _prepare_oracles(
                        context,
                        base,
                        split="development",
                        specs=specs["identified_oracle_methods"],
                    )
                elif args.stage == "prepare-scaling":
                    result = _prepare_scaling(
                        context, base, specs["causal_scaling_methods"]
                    )
                elif args.stage == "freeze-prepared":
                    result = _freeze_prepared(context)
                else:
                    confirm = json.loads(
                        (
                            context.root
                            / "artifacts/causal_geometry_v12_confirmatory.freeze.json"
                        ).read_text()
                    )
                    result = _prepare_oracles(
                        context,
                        base,
                        split="confirmatory",
                        specs=confirm["method_specs"],
                    )
        context.finish("COMPLETED_V12_PREPARATION", summary=result)
    except Exception as exc:
        context.fail_progress(f"{type(exc).__name__}: {exc}")
        context.finish("FAILED", error=f"{type(exc).__name__}: {exc}")
        raise


if __name__ == "__main__":
    main()
