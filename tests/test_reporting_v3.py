from pathlib import Path

import pandas as pd
import pytest

from jclosure.provenance import write_json_atomic
from jclosure.reporting_v3 import (
    _format_calibration_layers,
    _format_lowdim_results,
    _geometry_sources,
    _latest_completed_closure_sources,
    summarize_closure_v3,
    summarize_pareto_v3,
)


def _write_record(path: Path, value: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"value": [value]}).to_parquet(path, index=False)


def _write_manifest(path: Path, *, stage: str, shard: int, created: str) -> None:
    write_json_atomic(
        path,
        {
            "status": "COMPLETED",
            "stage": stage,
            "shard_index": shard,
            "created_at": created,
            "limit": None,
        },
    )


def test_geometry_sources_do_not_promote_smoke_to_formal(tmp_path: Path) -> None:
    run = tmp_path / "results/v3/raw/geometry-v3-smoke"
    _write_record(run / "map_spectra-smoke.parquet", 1)
    _write_record(run / "local_spectra-smoke.parquet", 2)

    maps, local, paths, scope = _geometry_sources(tmp_path)

    assert scope == "smoke"
    assert maps["value"].tolist() == [1]
    assert local["value"].tolist() == [2]
    assert len(paths) == 2


def test_geometry_sources_prefer_formal_records(tmp_path: Path) -> None:
    smoke = tmp_path / "results/v3/raw/geometry-v3-smoke"
    formal = tmp_path / "results/v3/raw/geometry-v3-formal"
    _write_record(smoke / "map_spectra-smoke.parquet", 1)
    _write_record(smoke / "local_spectra-smoke.parquet", 2)
    _write_record(formal / "map_spectra-shard-0.parquet", 3)
    _write_record(formal / "local_spectra-shard-0.parquet", 4)
    _write_manifest(
        formal / "manifest.json", stage="spectrum", shard=0, created="2026-01-01"
    )

    maps, local, paths, scope = _geometry_sources(tmp_path)

    assert scope == "formal"
    assert maps["value"].tolist() == [3]
    assert local["value"].tolist() == [4]
    assert all("smoke" not in path.name for path in paths)


def test_geometry_sources_use_latest_completed_shard(tmp_path: Path) -> None:
    old = tmp_path / "results/v3/raw/geometry-v3-old"
    new = tmp_path / "results/v3/raw/geometry-v3-new"
    for run, value, created in ((old, 1, "2026-01-01"), (new, 2, "2026-01-02")):
        _write_record(run / "map_spectra-shard-0.parquet", value)
        _write_record(run / "local_spectra-shard-0.parquet", value)
        _write_manifest(
            run / "manifest.json", stage="spectrum", shard=0, created=created
        )

    maps, local, paths, scope = _geometry_sources(tmp_path)

    assert scope == "formal"
    assert maps["value"].tolist() == [2]
    assert local["value"].tolist() == [2]
    assert all(path.parent == new for path in paths)


def test_closure_summary_uses_positive_gate_and_identity_null():
    rows = []
    for prompt, positive, remainder in (("p1", 0.1, 0.01), ("p2", 0.2, 0.02)):
        for condition, value in (
            ("identity", 1e-6),
            ("j_positive", positive),
            ("state_preserving", remainder),
        ):
            rows.append(
                {
                    "prompt_id": prompt,
                    "base_trial_id": prompt,
                    "protocol_key": "dense-4096",
                    "state_definition": "V3-Dense",
                    "dictionary_size": 4096,
                    "task_family": "arithmetic",
                    "position_scope": "final",
                    "source": "activation_difference",
                    "strength": 0.25,
                    "condition": condition,
                    "clamp_mode": "single",
                    "valid": True,
                    "metrics": {"js_divergence": value},
                }
            )
    summary = summarize_closure_v3(pd.DataFrame(rows), n_resamples=200, seed=7)
    remainder = summary[summary["condition"] == "state_preserving"].iloc[0]
    assert remainder["positive_control_usable"]
    assert remainder["null_threshold"] == 1e-4
    assert remainder["normalized_remainder_eta"] == pytest.approx(0.1)
    assert remainder["normalized_remainder_eta_ci_upper"] == pytest.approx(0.1)


def test_closure_sources_require_a_complete_shard_group(tmp_path: Path):
    for shard in (0, 1):
        run = tmp_path / "results/v3/raw" / f"closure-v3-run-{shard}"
        write_json_atomic(
            run / "manifest.json",
            {
                "status": "COMPLETED",
                "created_at": f"2026-01-0{shard + 1}",
                "run_id": f"r{shard}",
                "shard_group_id": "paired-run",
                "shard_index": shard,
                "shard_count": 2,
            },
        )
        trial = run / "trials/arithmetic" / f"part-shard-{shard:03d}.jsonl"
        trial.parent.mkdir(parents=True, exist_ok=True)
        trial.write_text('{"prompt_id":"p' + str(shard) + '"}\n', encoding="utf-8")
    frame, paths, group = _latest_completed_closure_sources(tmp_path)
    assert group == "paired-run"
    assert sorted(frame["prompt_id"]) == ["p0", "p1"]
    assert len(paths) == 2


def test_pareto_summary_keeps_dense_and_sparse_gates_independent():
    common = {
        "layer": 24,
        "dictionary_size": 4096,
        "null_tolerance": float("nan"),
        "strength": 0.2,
        "rms_drift": 0.01,
        "displacement_fraction": 0.25,
        "natural": True,
        "valid": True,
        "exclusion_reason": None,
        "optimization_status": "NOT_APPLICABLE",
    }
    rows = [
        {
            **common,
            "prompt_id": "dense",
            "paired_trial_id": "dense",
            "state_definition": "V3-Dense",
            "method": "isotropic_random",
            "dense_cosine": 0.999,
            "top10_overlap": 0.9,
            "sparse_support_f1": 0.0,
            "sparse_weighted_jaccard": 0.0,
            "sparse_coefficient_cosine": 0.0,
            "sparse_reconstruction_cosine": 0.0,
        },
        {
            **common,
            "prompt_id": "sparse-pass",
            "paired_trial_id": "sparse-pass",
            "state_definition": "V3-Sparse",
            "method": "sparse_remainder",
            "dense_cosine": 0.5,
            "top10_overlap": 0.1,
            "sparse_support_f1": 0.9,
            "sparse_weighted_jaccard": 0.97,
            "sparse_coefficient_cosine": 0.999,
            "sparse_reconstruction_cosine": 0.999,
        },
        {
            **common,
            "prompt_id": "sparse-fail",
            "paired_trial_id": "sparse-fail",
            "state_definition": "V3-Sparse",
            "method": "sparse_remainder",
            "dense_cosine": 1.0,
            "top10_overlap": 1.0,
            "sparse_support_f1": 0.9,
            "sparse_weighted_jaccard": 0.5,
            "sparse_coefficient_cosine": 0.999,
            "sparse_reconstruction_cosine": 0.999,
        },
    ]
    summary = summarize_pareto_v3(pd.DataFrame(rows))
    dense = summary[summary["state_definition"] == "V3-Dense"].iloc[0]
    sparse = summary[summary["state_definition"] == "V3-Sparse"].iloc[0]
    assert dense["formal_valid_rows"] == 1
    assert sparse["formal_valid_rows"] == 1
    assert sparse["state_equal_rows"] == 1


def test_calibration_table_reports_strict_and_formal_counts():
    text = _format_calibration_layers(
        {
            "layers": [
                {
                    "dictionary_size": 4096,
                    "layer": 23,
                    "method": "dense_optimized",
                    "strict_valid": 175,
                    "formal_natural_valid": 175,
                    "attempted": 200,
                    "natural_fraction_among_valid": 1.0,
                    "eligible": True,
                    "reasons": [],
                }
            ]
        }
    )
    assert "175/200" in text
    assert "| yes | - |" in text


def test_lowdim_table_is_generated_from_saved_metrics():
    text = _format_lowdim_results(
        pd.DataFrame(
            [
                {
                    "candidate": "dense_profile_pca",
                    "dimension": 128,
                    "next_state_cosine_median": 0.97,
                    "oracle_gap_closed": 0.48,
                    "state_reconstruction_cosine_median": 0.99,
                }
            ]
        )
    )
    assert "dense_profile_pca" in text
    assert "0.480000" in text
