from pathlib import Path

import pandas as pd

from jclosure.provenance import write_json_atomic
from jclosure.reporting_v3 import _geometry_sources


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
