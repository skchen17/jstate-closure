"""Freeze v11 splits, select score-space models, and decode state artifacts."""

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
from jclosure.protocol_v11 import (
    CONFIRM_FREEZE_PATH,
    PREPARED_FREEZE_PATH,
    PROTOCOL_V11,
    SCHEMA_VERSION_V11,
    build_base_freeze,
    build_derived_freeze,
    verify_base_freeze,
    verify_derived_freeze,
)
from jclosure.provenance import sha256_file, write_json_atomic
from jclosure.state_models_v11 import CHANNELS, ScoreBank, build_method_specs

MODEL_SPECS = Path("results/v11/processed/method_specs_v11.json")
ALLOCATION_SEARCH = Path("results/v11/processed/allocation_search_v11.parquet")
DEVELOPMENT_STATES = Path("results/v11/processed/development_states_v11.json")
CONFIRMATORY_STATES = Path("results/v11/processed/confirmatory_states_v11.json")


def effect_weights(root: Path, ids: np.ndarray) -> np.ndarray:
    frame = pd.read_parquet(
        root / "results/v8/processed/structured_component_screen_v8.parquet"
    )
    frame = frame[frame["condition"] == "R7"].copy()
    columns = (
        "output_js_divergence",
        "future_j_trajectory_divergence",
        "target_log_odds_abs_change",
    )
    normalized = []
    for column in columns:
        raw = frame[column].astype(float)
        scale = float(raw.std())
        normalized.append((raw - float(raw.mean())) / (scale if scale > 1e-12 else 1.0))
    frame["_effect"] = sum(normalized) / len(normalized)
    lookup = dict(
        zip(frame["base_trial_id"].astype(str), frame["_effect"], strict=True)
    )
    score = np.asarray([float(lookup[str(value)]) for value in ids], dtype=np.float32)
    score = score - float(score.min()) + 0.25
    return score / max(float(score.mean()), 1e-12)


def build_bank(context: Any) -> tuple[ScoreBank, dict[str, Any], np.ndarray]:
    data = _load_data(context.root)
    train = np.flatnonzero(data["splits"] == "train")
    weights = effect_weights(context.root, data["ids"])
    bank = ScoreBank(
        data,
        train,
        weights,
        context.config["sufficiency_v9"]["semantic_objective_weights"],
    )
    return bank, data, train


def _select_models(context: Any, base: dict[str, Any]) -> None:
    bank, data, _ = build_bank(context)
    validation = np.flatnonzero(data["splits"] == "validation")
    specs, rows = build_method_specs(bank, context.config, validation)
    output = context.root / MODEL_SPECS
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": SCHEMA_VERSION_V11,
        "protocol_version": PROTOCOL_V11,
        "run_id": context.run_id,
        "source_freeze_digest": base["freeze_digest"],
        "selection_split": "validation",
        "final_confirmatory_not_used": True,
        "method_count": len(specs),
        "methods": specs,
        "allocation_search_records": str(ALLOCATION_SEARCH),
    }
    pd.DataFrame(rows).to_parquet(
        context.root / ALLOCATION_SEARCH, index=False, compression="zstd"
    )
    payload["allocation_search_sha256"] = sha256_file(context.root / ALLOCATION_SEARCH)
    write_json_atomic(output, payload)
    context.finish("COMPLETED_V11_MODEL_SELECTION", summary=payload)


def _state_ids(base: dict[str, Any], split: str) -> list[str]:
    if split == "development":
        return [str(value) for value in base["development"]["base_trial_ids"]]
    return [str(value) for value in base["confirmatory"]["base_trial_ids"]]


def _decode_method(
    *,
    context: Any,
    bank: ScoreBank,
    data: dict[str, Any],
    train: np.ndarray,
    raw: dict[str, torch.Tensor],
    spec: dict[str, Any],
    selected: np.ndarray,
    selected_ids: list[str],
    split: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    inputs, latent, model_metadata = bank.decode_inputs(spec)
    decoder_config = context.config["causal_geometry_v11"]["decoder"]
    requested_device = int(context.config["model"].get("device", 0))
    device = torch.device(
        f"cuda:{requested_device}" if torch.cuda.is_available() else "cpu"
    )
    raw_names = {"recurrent": "recurrent", "conv": "conv", "kv": "kv_full"}
    decoded: dict[str, torch.Tensor] = {}
    diagnostics: dict[str, Any] = {}
    for name in CHANNELS:
        train_scores, decoded_scores = inputs[name]
        alpha, detail = _dual_reconstruction_alpha(
            train_scores[train], decoded_scores[selected]
        )
        decoded[name] = _decode_block(
            raw[raw_names[name]],
            train,
            alpha,
            device=device,
            chunk_size=int(decoder_config["reconstruction_chunk"]),
        )
        diagnostics[name] = detail
    artifact_root = (
        context.root
        / "artifacts/causal/v11"
        / context.run_id
        / split
        / str(spec["method"])
    )
    artifact_root.mkdir(parents=True, exist_ok=True)
    declarations: list[dict[str, Any]] = []
    shard_size = int(decoder_config["shard_size"])
    for start in range(0, len(selected_ids), shard_size):
        stop = min(start + shard_size, len(selected_ids))
        rows = [
            {
                "base_trial_id": selected_ids[index],
                "method": spec["method"],
                # clone prevents each view from serializing the full backing tensor
                "recurrent": decoded["recurrent"][index].clone(),
                "conv": decoded["conv"][index].clone(),
                "kv": decoded["kv"][index].clone(),
            }
            for index in range(start, stop)
        ]
        path = artifact_root / f"state_{start:03d}_{stop:03d}.pt"
        torch.save(
            {
                "format": "architecture_resolved_state_v11_bf16",
                "rows": rows,
            },
            path,
        )
        declarations.append(
            {
                "path": str(path.relative_to(context.root)),
                "sha256": sha256_file(path),
                "base_trial_ids": selected_ids[start:stop],
            }
        )
    latent_energy = float(np.mean(latent[selected].astype(np.float64) ** 2))
    del decoded
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return declarations, {
        "model_metadata": model_metadata,
        "decoder_diagnostics": diagnostics,
        "selected_latent_mean_square": latent_energy,
    }


def _prepare_states(
    context: Any,
    base: dict[str, Any],
    *,
    split: str,
    specs: list[dict[str, Any]],
    summary_path: Path,
) -> None:
    bank, data, train = build_bank(context)
    ids = _state_ids(base, split)
    lookup = {str(value): index for index, value in enumerate(data["ids"].tolist())}
    selected = np.asarray([lookup[value] for value in ids], dtype=int)
    raw, raw_ids, _, _, _ = _load_raw_deltas(context, _capture_manifests(context.root))
    raw = _raw_order(raw, raw_ids, data["ids"])
    methods: list[dict[str, Any]] = []
    progress = context.raw_dir / context.run_id / "prepare_progress.json"
    for method_index, spec in enumerate(specs):
        declarations, detail = _decode_method(
            context=context,
            bank=bank,
            data=data,
            train=train,
            raw=raw,
            spec=spec,
            selected=selected,
            selected_ids=ids,
            split=split,
        )
        methods.append({"spec": spec, "shards": declarations, **detail})
        write_json_atomic(
            progress,
            {
                "status": "RUNNING",
                "completed_methods": method_index + 1,
                "total_methods": len(specs),
                "last_method": spec["method"],
            },
        )
    payload = {
        "schema_version": SCHEMA_VERSION_V11,
        "protocol_version": PROTOCOL_V11,
        "run_id": context.run_id,
        "source_freeze_digest": base["freeze_digest"],
        "split": split,
        "base_trial_ids": ids,
        "method_count": len(methods),
        "methods": methods,
    }
    write_json_atomic(context.root / summary_path, payload)
    write_json_atomic(
        progress,
        {
            "status": "COMPLETED",
            "completed_methods": len(specs),
            "total_methods": len(specs),
        },
    )
    context.finish("COMPLETED_V11_STATE_PREPARATION", summary=payload)


def _freeze_prepared(context: Any, base: dict[str, Any]) -> None:
    specs = json.loads((context.root / MODEL_SPECS).read_text())
    states = json.loads((context.root / DEVELOPMENT_STATES).read_text())
    value = build_derived_freeze(
        context.root,
        context.config,
        path=PREPARED_FREEZE_PATH,
        purpose="freeze validation-selected v11 method specs and development state shards",
        inputs=[MODEL_SPECS, ALLOCATION_SEARCH, DEVELOPMENT_STATES],
        payload={
            "method_count": specs["method_count"],
            "development_state_run_id": states["run_id"],
            "methods": [value["method"] for value in specs["methods"]],
        },
    )
    context.finish("COMPLETED_V11_PREPARED_FREEZE", freeze=value)


def main() -> None:
    parser = standard_parser(
        "prepare architecture-resolved causal geometry protocol v11",
        "configs/causal_geometry_v11.yaml",
    )
    parser.add_argument(
        "--stage",
        required=True,
        choices=(
            "freeze-base",
            "select-models",
            "prepare-development",
            "freeze-prepared",
            "prepare-confirmatory",
        ),
    )
    args = parser.parse_args()
    context = initialize_context("prepare-v11", args)
    try:
        if args.stage == "freeze-base":
            context.finish(
                "COMPLETED_V11_BASE_FREEZE",
                freeze=build_base_freeze(context.root, context.config),
            )
            return
        base = verify_base_freeze(context.root, context.config)
        if args.stage == "select-models":
            _select_models(context, base)
        elif args.stage == "prepare-development":
            specs = json.loads((context.root / MODEL_SPECS).read_text())["methods"]
            _prepare_states(
                context,
                base,
                split="development",
                specs=specs,
                summary_path=DEVELOPMENT_STATES,
            )
        elif args.stage == "freeze-prepared":
            _freeze_prepared(context, base)
        else:
            confirm = verify_derived_freeze(
                context.root, context.config, CONFIRM_FREEZE_PATH
            )
            _prepare_states(
                context,
                base,
                split="confirmatory",
                specs=confirm["method_specs"],
                summary_path=CONFIRMATORY_STATES,
            )
    except Exception as exc:
        context.fail_progress(f"{type(exc).__name__}: {exc}")
        context.finish("FAILED", error=f"{type(exc).__name__}: {exc}")
        raise


if __name__ == "__main__":
    main()
