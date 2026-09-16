"""Numerically verify the compact-only/full/raw-residual interface comparison."""

from __future__ import annotations

import gc
import json
from pathlib import Path
from typing import Any

import pandas as pd
import torch

from jclosure.experiments.common import initialize_context, standard_parser
from jclosure.experiments.compress_persistent_v8 import (
    _capture_manifests,
    _load_raw_deltas,
)
from jclosure.protocol_v10 import verify_candidate_freeze
from jclosure.protocol_v10_strict_interface import (
    PROTOCOL,
    build_freeze,
    verify_freeze,
)
from jclosure.provenance import sha256_file, write_json_atomic

SUMMARY = Path("results/v10/processed/strict_interface_audit_v10.json")
RECORDS = Path("results/v10/processed/strict_interface_audit_v10.parquet")


def recombination_metrics(
    decoded: torch.Tensor, teacher: torch.Tensor
) -> dict[str, float]:
    decoded_f = decoded.float()
    teacher_f = teacher.float()
    residual = teacher_f - decoded_f
    recombined = decoded_f + residual
    teacher_norm = torch.linalg.vector_norm(teacher_f)
    decoded_norm = torch.linalg.vector_norm(decoded_f)
    residual_norm = torch.linalg.vector_norm(residual)
    cosine = torch.sum(decoded_f * teacher_f) / torch.clamp(
        decoded_norm * teacher_norm, min=1e-20
    )
    return {
        "decoded_teacher_cosine": float(cosine),
        "raw_residual_relative_l2": float(
            residual_norm / torch.clamp(teacher_norm, min=1e-20)
        ),
        "recombined_full_raw_max_abs_error": float(
            torch.max(torch.abs(recombined - teacher_f))
        ),
    }


def _audit(context: Any, freeze: dict[str, Any]) -> None:
    candidate = verify_candidate_freeze(context.root, context.config)
    decoder = json.loads(
        (context.root / "results/v10/processed/compact_decoder_v10.json").read_text()
    )
    causal = json.loads(
        (
            context.root
            / "results/v10/processed/decoded_causal_state_validation_v10.json"
        ).read_text()
    )
    raw, raw_ids, _, _, _ = _load_raw_deltas(
        context, _capture_manifests(context.root)
    )
    raw_lookup = {str(value): index for index, value in enumerate(raw_ids)}
    raw_names = {"recurrent": "recurrent", "conv": "conv", "kv": "kv_full"}
    rows: list[dict[str, Any]] = []
    for declaration in decoder["shards"]:
        path = context.root / declaration["path"]
        if sha256_file(path) != declaration["sha256"]:
            raise RuntimeError(f"decoded shard hash mismatch: {path}")
        payload = torch.load(path, map_location="cpu", weights_only=False)
        for row in payload["rows"]:
            base_id = str(row["base_trial_id"])
            source = raw_lookup[base_id]
            record: dict[str, Any] = {
                "protocol_version": PROTOCOL,
                "run_id": context.run_id,
                "base_trial_id": base_id,
                "dimension": int(row["dimension"]),
            }
            for name, raw_name in raw_names.items():
                metrics = recombination_metrics(row[name], raw[raw_name][source])
                record.update({f"{name}_{key}": value for key, value in metrics.items()})
            rows.append(record)
        del payload
        gc.collect()
    frame = pd.DataFrame(rows).sort_values(["dimension", "base_trial_id"])
    frame.to_parquet(context.root / RECORDS, index=False, compression="zstd")
    identity_columns = [
        column for column in frame if column.endswith("recombined_full_raw_max_abs_error")
    ]
    similarity: dict[str, Any] = {}
    for dimension, values in frame.groupby("dimension", sort=True):
        similarity[str(dimension)] = {
            name: {
                "decoded_teacher_cosine_mean": float(
                    values[f"{name}_decoded_teacher_cosine"].mean()
                ),
                "raw_residual_relative_l2_mean": float(
                    values[f"{name}_raw_residual_relative_l2"].mean()
                ),
            }
            for name in raw_names
        }
    payload = {
        "protocol_version": PROTOCOL,
        "run_id": context.run_id,
        "source_freeze_digest": freeze["freeze_digest"],
        "candidate_freeze_digest": candidate["freeze_digest"],
        "causal_run_id": causal["run_id"],
        "strict_compact_only_interface": causal["strict_interface"],
        "three_way_comparison": {
            "compact_only": (
                "Experimentally evaluated; failed frozen causal gates for every "
                "candidate."
            ),
            "full_raw": "Teacher intervention reference.",
            "compact_plus_raw_residual": (
                "Diagnostic only and intentionally violates compact-only access; "
                "numerically reconstructs the full raw delta."
            ),
        },
        "maximum_recombined_full_raw_abs_error": float(
            frame[identity_columns].to_numpy().max()
        ),
        "state_similarity": similarity,
        "records": str(RECORDS),
        "records_sha256": sha256_file(context.root / RECORDS),
    }
    write_json_atomic(context.root / SUMMARY, payload)
    context.finish("COMPLETED_V10_STRICT_INTERFACE_AUDIT", summary=payload)


def main() -> None:
    parser = standard_parser(
        "strict-interface three-way state audit v10",
        "configs/causal_sufficiency_v10.yaml",
    )
    parser.add_argument("--stage", required=True, choices=("freeze", "audit"))
    args = parser.parse_args()
    context = initialize_context("strict-interface-v10", args)
    try:
        if args.stage == "freeze":
            context.finish(
                "COMPLETED_V10_STRICT_INTERFACE_FREEZE",
                freeze=build_freeze(context.root, context.config),
            )
            return
        freeze = verify_freeze(context.root, context.config)
        if args.dry_run:
            context.finish("DRY_RUN", source_freeze_digest=freeze["freeze_digest"])
            return
        _audit(context, freeze)
    except Exception as exc:
        context.fail_progress(f"{type(exc).__name__}: {exc}")
        context.finish("FAILED", error=f"{type(exc).__name__}: {exc}")
        raise


if __name__ == "__main__":
    main()
