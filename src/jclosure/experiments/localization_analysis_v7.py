"""Corrective, source-frozen analysis for completed v7 localization records."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd

from jclosure.config import config_digest
from jclosure.experiments.common import initialize_context, standard_parser
from jclosure.protocol_v7 import digest
from jclosure.protocol_v7_stage2 import verify_stage_freeze
from jclosure.provenance import sha256_file, write_json_atomic
from jclosure.records_v7 import PROTOCOL_V7
from jclosure.statistics import clustered_bootstrap_ci

FREEZE_PATH = Path("artifacts/channel_localization_analysis_v7.freeze.json")


def _source_manifest(root: Path) -> tuple[Path, dict[str, Any]]:
    manifests = []
    for path in (root / "results/v7/raw").glob(
        "channel-localization-v7-*/manifest.json"
    ):
        value = json.loads(path.read_text(encoding="utf-8"))
        if value.get("status") == "COMPLETED_LOCALIZATION":
            manifests.append((path, value))
    if not manifests:
        raise RuntimeError("no completed localization source")
    return sorted(manifests, key=lambda item: item[1]["run_id"])[-1]


def _build_freeze(root: Path, config: dict[str, Any]) -> dict[str, Any]:
    parent = verify_stage_freeze(root, config, "localization")
    source_path, source = _source_manifest(root)
    paths = [
        root / config["_config_path"],
        root / "src/jclosure/experiments/localization_analysis_v7.py",
        root / "scripts/run_localization_analysis_v7.sh",
        source_path,
        root / source["records"],
        root / "artifacts/channel_localization_v7.freeze.json",
    ]
    payload: dict[str, Any] = {
        "schema_version": 9,
        "protocol_version": PROTOCOL_V7,
        "stage": "corrective_localization_analysis",
        "config_digest": config_digest(config),
        "parent_freeze_digest": parent["freeze_digest"],
        "source_run_id": source["run_id"],
        "source_records": source["records"],
        "hashes": {str(path.relative_to(root)): sha256_file(path) for path in paths},
    }
    payload["freeze_digest"] = digest(payload)
    write_json_atomic(root / FREEZE_PATH, payload)
    return payload


def _verify_freeze(root: Path, config: dict[str, Any]) -> dict[str, Any]:
    verify_stage_freeze(root, config, "localization")
    payload = json.loads((root / FREEZE_PATH).read_text(encoding="utf-8"))
    if payload.get("config_digest") != config_digest(config):
        raise RuntimeError("localization analysis config mismatch")
    if payload.get("freeze_digest") != digest(payload):
        raise RuntimeError("localization analysis freeze digest mismatch")
    failures = []
    for relative, expected in payload["hashes"].items():
        path = root / relative
        observed = sha256_file(path) if path.is_file() else "MISSING"
        if observed != expected:
            failures.append(relative)
    if failures:
        raise RuntimeError(f"localization analysis frozen input mismatch: {failures}")
    return payload


def _ci(frame: pd.DataFrame, column: str, seed: int, resamples: int) -> dict[str, Any]:
    if frame.empty:
        return {"estimate": None, "lower": None, "upper": None, "n_clusters": 0}
    return asdict(
        clustered_bootstrap_ci(
            frame,
            cluster_col="prompt_id",
            value_col=column,
            seed=seed,
            n_resamples=resamples,
        )
    )


def _score(summary: dict[str, Any], condition: str) -> float:
    value = summary[condition]["direction_cosine_to_full"]["estimate"]
    return float(value) if value is not None else float("-inf")


def _analyze(context: Any, freeze: dict[str, Any]) -> dict[str, Any]:
    rows = [
        json.loads(line)
        for line in (context.root / freeze["source_records"])
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    frame = pd.DataFrame(rows)
    for column in (
        "next_j_l2",
        "output_js_divergence",
        "direction_cosine_to_full",
        "magnitude_ratio_to_full",
        "output_delta_cosine_to_full",
    ):
        frame[column] = frame["metrics"].map(lambda value, key=column: value.get(key))
    seed = int(context.config["reproducibility"]["bootstrap_seed"])
    resamples = int(context.config["persistent_channels_v7"]["bootstrap_resamples"])
    summaries: dict[str, Any] = {}
    metrics = (
        "next_j_l2",
        "output_js_divergence",
        "direction_cosine_to_full",
        "magnitude_ratio_to_full",
        "output_delta_cosine_to_full",
    )
    for split in ("localization_fit", "attribution_test"):
        subset = frame[(frame["split"] == split) & frame["valid"]]
        summaries[split] = {
            condition: {
                metric: _ci(values.dropna(subset=[metric]), metric, seed, resamples)
                for metric in metrics
            }
            for condition, values in subset.groupby("condition", sort=True)
        }
    fit = summaries["localization_fit"]
    section = context.config["persistent_channels_v7"]
    rec_layers = [int(value) for value in section["downstream_recurrent_layers"]]
    kv_layers = [int(value) for value in section["downstream_attention_layers"]]
    selected_rec_layers = sorted(
        rec_layers,
        key=lambda layer: _score(fit, f"rec_layer_{layer}_both"),
        reverse=True,
    )
    selected_kv_layer = max(
        kv_layers, key=lambda layer: _score(fit, f"kv_layer_{layer}")
    )
    selected_kv_heads = sorted(
        range(4), key=lambda head: _score(fit, f"kv_head_{head}"), reverse=True
    )
    output_frame = frame.copy()
    output_frame["metrics"] = output_frame["metrics"].map(
        lambda value: json.dumps(value, sort_keys=True)
    )
    output_frame["spec"] = output_frame["spec"].map(
        lambda value: json.dumps(value, sort_keys=True)
    )
    output_path = context.processed_dir / "channel_localization_v7.parquet"
    output_frame.to_parquet(output_path, index=False, compression="zstd")
    summary = {
        "schema_version": 9,
        "protocol_version": PROTOCOL_V7,
        "run_id": context.run_id,
        "source_run_id": freeze["source_run_id"],
        "source_freeze_digest": freeze["freeze_digest"],
        "summaries": summaries,
        "selected_recurrent_layers_ranked": selected_rec_layers,
        "selected_kv_layer": selected_kv_layer,
        "selected_kv_heads_ranked": selected_kv_heads,
        "records": str(output_path.relative_to(context.root)),
        "corrective_note": "None-valued structural-zero arms sort below finite effects; scientific records are unchanged.",
    }
    write_json_atomic(context.processed_dir / "channel_localization_v7.json", summary)
    context.finish("COMPLETED_LOCALIZATION_ANALYSIS", summary=summary)
    return summary


def main() -> None:
    parser = standard_parser(
        "Corrective v7 localization analysis", "configs/persistent_channels_v7.yaml"
    )
    parser.add_argument("--stage", choices=("freeze", "analyze"), required=True)
    args = parser.parse_args()
    context = initialize_context("localization-analysis-v7", args)
    try:
        if args.stage == "freeze":
            freeze = _build_freeze(context.root, context.config)
            context.finish("COMPLETED_FREEZE", freeze=freeze)
        else:
            _analyze(context, _verify_freeze(context.root, context.config))
    except Exception as exc:
        context.finish("FAILED", error=f"{type(exc).__name__}: {exc}")
        raise


if __name__ == "__main__":
    main()
