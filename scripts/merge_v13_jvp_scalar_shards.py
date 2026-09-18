#!/usr/bin/env python3
"""Merge frozen V13 scalar-JVP execution shards into canonical records."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from jclosure.experiments.geometry_v13 import (
    GEOMETRY_FREEZE,
    JVP_MATRIX_ROOT,
    JVP_RECORDS,
    JVP_SUMMARY,
)
from jclosure.experiments.runtime_v13_jvp_scalar_sharded import (
    AMENDMENT_PATH,
    ANCHORS_PER_SHARD,
    PROTOCOL,
    SHARD_COUNT,
    _verify,
)
from jclosure.provenance import sha256_file, write_json_atomic


def _digest(value: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def main() -> None:
    root = Path.cwd()
    amendment = _verify(root)
    jvp_freeze = json.loads((root / GEOMETRY_FREEZE).read_text())
    frames = []
    summaries = []
    canonical_root = root / JVP_MATRIX_ROOT
    canonical_root.mkdir(parents=True, exist_ok=True)
    for shard_index in range(SHARD_COUNT):
        records_path = (
            root
            / f"results/v13/processed/causal_probe_scaling_scalar_shard_{shard_index}_v13.parquet"
        )
        summary_path = (
            root
            / f"results/v13/processed/causal_probe_scaling_scalar_shard_{shard_index}_v13.json"
        )
        frame = pd.read_parquet(records_path)
        summaries.append(json.loads(summary_path.read_text()))
        old_root = f"results/v13/processed/jvp_matrices_scalar_shard_{shard_index}"
        for local_index in range(ANCHORS_PER_SHARD):
            global_index = shard_index * ANCHORS_PER_SHARD + local_index
            for position in (0, 1):
                source = (
                    root
                    / old_root
                    / f"anchor_{local_index:02d}_position_{position}.npz"
                )
                destination = (
                    canonical_root
                    / f"anchor_{global_index:02d}_position_{position}.npz"
                )
                if destination.exists():
                    if sha256_file(destination) != sha256_file(source):
                        raise RuntimeError(
                            f"refusing to replace non-identical matrix: {destination}"
                        )
                else:
                    shutil.copy2(source, destination)
                old_path = (
                    f"{old_root}/anchor_{local_index:02d}_position_{position}.npz"
                )
                new_path = str(destination.relative_to(root))
                mask = frame["matrix_path"] == old_path
                frame.loc[mask, "matrix_path"] = new_path
                frame.loc[mask, "matrix_sha256"] = sha256_file(destination)
        frames.append(frame)
    merged = pd.concat(frames, ignore_index=True)
    expected_records = SHARD_COUNT * ANCHORS_PER_SHARD * 2 * 9
    if len(merged) != expected_records:
        raise RuntimeError(f"expected {expected_records} records, found {len(merged)}")
    if merged[["base_trial_id", "token_position"]].drop_duplicates().shape[0] != 20:
        raise RuntimeError("merged scalar JVP states are not unique and complete")
    output_path = root / JVP_RECORDS
    merged.to_parquet(output_path, index=False, compression="zstd")
    mixed = merged[merged.probe_family == "mixed"]
    curves = {
        str(size): {
            "median_r90": float(group.rank_90.median()),
            "median_r95": float(group.rank_95.median()),
            "median_r99": float(group.rank_99.median()),
            "mean_stable_rank": float(group.stable_rank.mean()),
            "mean_effective_rank": float(group.effective_rank.mean()),
        }
        for size, group in mixed.groupby("probe_direction_count")
    }
    robustness = {
        str(name): {
            "median_r95": float(group.rank_95.median()),
            "mean_r95": float(group.rank_95.mean()),
        }
        for name, group in merged[merged.probe_family != "mixed"].groupby(
            "probe_family"
        )
    }
    sensitivity_names = sorted(summaries[0]["probe_family_mean_column_sensitivity"])
    sensitivity = {
        name: float(
            np.mean(
                [
                    summary["probe_family_mean_column_sensitivity"][name]
                    for summary in summaries
                ]
            )
        )
        for name in sensitivity_names
    }
    summary: dict[str, Any] = {
        "schema_version": 22,
        "protocol_version": PROTOCOL,
        "run_ids": [summary["run_id"] for summary in summaries],
        "source_freeze_digest": jvp_freeze["freeze_digest"],
        "runtime_amendment_digest": amendment["freeze_digest"],
        "exact_autograd_jvp": True,
        "direction_batch_size": 1,
        "execution_shards": SHARD_COUNT,
        "full_raw_jacobian_claimed": False,
        "local_state_count": 20,
        "probe_scaling": curves,
        "probe_family_robustness": robustness,
        "probe_family_mean_column_sensitivity": sensitivity,
        "records": str(JVP_RECORDS),
        "records_sha256": sha256_file(output_path),
    }
    write_json_atomic(root / JVP_SUMMARY, summary)
    merge_record = {
        "schema_version": 22,
        "protocol_version": PROTOCOL,
        "source_amendment": str(AMENDMENT_PATH),
        "source_amendment_digest": amendment["freeze_digest"],
        "shard_record_hashes": {
            str(index): sha256_file(
                root
                / f"results/v13/processed/causal_probe_scaling_scalar_shard_{index}_v13.parquet"
            )
            for index in range(SHARD_COUNT)
        },
        "canonical_records": str(JVP_RECORDS),
        "canonical_records_sha256": sha256_file(output_path),
        "canonical_summary": str(JVP_SUMMARY),
        "canonical_summary_payload_digest": _digest(summary),
        "local_state_count": 20,
        "record_count": len(merged),
    }
    write_json_atomic(
        root / "results/v13/processed/jvp_scalar_shard_merge_v13.json",
        merge_record,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
