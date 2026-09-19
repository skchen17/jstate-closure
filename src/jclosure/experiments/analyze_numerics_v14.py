"""Freeze a development-only effect threshold and reinterpret V13 alpha rows."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from jclosure.protocol_v14 import freeze_stage, verify_base
from jclosure.provenance import sha256_file, write_json_atomic

OUT = Path("results/v14/processed")


def _finite_records(root: Path) -> pd.DataFrame:
    paths = [
        root / OUT / "jvp_finite_writeback_audit_v14.parquet",
        root
        / OUT
        / "numerical_scale_extension_1/jvp_finite_writeback_audit_v14.parquet",
    ]
    return pd.concat([pd.read_parquet(path) for path in paths], ignore_index=True)


def analyze(root: Path) -> dict[str, Any]:
    base = verify_base(root)
    v13_oracle = pd.read_parquet(
        root / "results/v13/processed/moving_tangent_oracle_development_v13.parquet"
    )
    teacher = v13_oracle[v13_oracle.method == "raw_teacher_intervention"].copy()
    threshold_quantile = float(
        base["config"]["numerical_audit"]["minimum_effect_quantile"]
    )
    minimum_norm = float(teacher.teacher_j_effect_norm.quantile(threshold_quantile))
    finite = _finite_records(root)
    baseline = finite[
        (finite.precision_mode == "v13_fp32_accumulate_bf16_writeback")
        & finite.error.isna()
    ].copy()
    floors: dict[str, float] = {}
    for target, group in baseline.groupby("target"):
        observed = group[group.target_effect_norm > 0].target_effect_norm
        floors[target] = float(observed.quantile(0.05))
    stage = root / "artifacts/finite_causal_control_v14_snr_threshold.freeze.json"
    if not stage.exists():
        frozen = freeze_stage(
            root,
            "snr_threshold",
            [
                "src/jclosure/experiments/analyze_numerics_v14.py",
                "results/v13/processed/moving_tangent_oracle_development_v13.parquet",
                "results/v13/processed/local_linearity_radius_v13.parquet",
                "results/v14/processed/jvp_finite_writeback_audit_v14.parquet",
                "results/v14/processed/numerical_scale_extension_1/jvp_finite_writeback_audit_v14.parquet",
            ],
            {
                "development_only": True,
                "teacher_effect_quantile": threshold_quantile,
                "min_causal_effect_norm": minimum_norm,
                "target_quantization_floors": floors,
                "reliability_rule": "all targets median cosine>=0.95 and median relative L2<=0.20",
                "signal_floor_rule": "5th percentile of nonzero finite-writeback effect per target",
            },
        )
    else:
        frozen = json.loads(stage.read_text(encoding="utf-8"))
        if abs(float(frozen["min_causal_effect_norm"]) - minimum_norm) > 1e-12:
            raise RuntimeError("development threshold changed after freeze")
    baseline["effective_snr"] = baseline.apply(
        lambda row: float(row.target_effect_norm) / floors[str(row.target)], axis=1
    )
    baseline["jvp_agrees"] = (
        baseline.cosine >= float(base["config"]["numerical_audit"]["jvp_cosine_gate"])
    ) & (
        baseline.relative_l2
        <= float(base["config"]["numerical_audit"]["jvp_relative_error_gate"])
    )
    pooled = (
        baseline.groupby(["epsilon", "target"])
        .agg(
            median_cosine=("cosine", "median"),
            median_relative_l2=("relative_l2", "median"),
            median_effective_snr=("effective_snr", "median"),
            jvp_agreement_rate=("jvp_agrees", "mean"),
            count=("jvp_agrees", "size"),
        )
        .reset_index()
    )
    pooled_path = root / OUT / "numerical_snr_summary_v14.parquet"
    pooled.to_parquet(pooled_path, index=False, compression="zstd")
    reliable = pooled[
        (pooled.median_cosine >= 0.95) & (pooled.median_relative_l2 <= 0.2)
    ]
    shared = set(pooled.target.unique())
    epsilons = sorted(pooled.epsilon.unique())
    first_all_target_reliable = next(
        (
            float(eps)
            for eps in epsilons
            if set(reliable[reliable.epsilon == eps].target.unique()) == shared
        ),
        None,
    )
    linearity = pd.read_parquet(
        root / "results/v13/processed/local_linearity_radius_v13.parquet"
    )
    teacher_norms = teacher[
        ["base_trial_id", "horizon", "teacher_j_effect_norm"]
    ].drop_duplicates()
    linearity = linearity.merge(
        teacher_norms, on=["base_trial_id", "horizon"], validate="many_to_one"
    )
    linearity["expected_scaled_teacher_norm"] = (
        linearity.scale * linearity.teacher_j_effect_norm
    )
    linearity["direction_snr_label"] = np.where(
        linearity.expected_scaled_teacher_norm >= minimum_norm,
        "SNR_QUALIFIED",
        "BELOW_DIRECTION_SNR_THRESHOLD",
    )
    linearity_path = root / OUT / "v13_alpha_snr_reanalysis_v14.parquet"
    linearity.to_parquet(linearity_path, index=False, compression="zstd")
    alpha_summary = (
        linearity.groupby(["scale", "horizon", "direction_snr_label"])
        .agg(
            rows=("base_trial_id", "size"),
            j_direction=("j_direction", "mean"),
            output_direction=("output_direction", "mean"),
            j_relative_error=("j_relative_linearity_error", "mean"),
        )
        .reset_index()
    )
    oracle = v13_oracle.copy()
    oracle["direction_snr_label"] = np.where(
        oracle.teacher_j_effect_norm >= minimum_norm,
        "SNR_QUALIFIED",
        "BELOW_DIRECTION_SNR_THRESHOLD",
    )
    oracle_path = root / OUT / "v13_oracle_snr_reanalysis_v14.parquet"
    oracle.to_parquet(oracle_path, index=False, compression="zstd")
    summary = {
        "protocol_version": "finite_causal_control_v14_snr_threshold",
        "source_freeze_digest": frozen["freeze_digest"],
        "min_causal_effect_norm": minimum_norm,
        "quantization_floors": floors,
        "first_all_target_jvp_reliable_epsilon": first_all_target_reliable,
        "clean_repeat_noise_is_zero": True,
        "effective_snr_uses_quantization_floor_not_clean_repeat_zero": True,
        "alpha_summary": alpha_summary.to_dict("records"),
        "records": {
            name: {"path": str(path.relative_to(root)), "sha256": sha256_file(path)}
            for name, path in (
                ("numerical_snr", pooled_path),
                ("v13_alpha", linearity_path),
                ("v13_oracle", oracle_path),
            )
        },
        "v13_frozen_outcomes_changed": False,
    }
    write_json_atomic(root / OUT / "numerical_snr_summary_v14.json", summary)
    return summary


if __name__ == "__main__":
    print(json.dumps(analyze(Path.cwd()), sort_keys=True)[:2000])
