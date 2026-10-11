"""Guard against reading independent-final outcomes before the opening seal."""
from scripts.open_v40_final import premature_final_artifacts


def test_premature_final_scan_detects_raw_and_processed(tmp_path):
    raw = tmp_path / "results/v40/raw"
    processed = tmp_path / "results/v40/processed"
    raw.mkdir(parents=True)
    processed.mkdir(parents=True)
    assert premature_final_artifacts(tmp_path) == []
    (raw / "formal_functional_Q_independent_final_a4-run.jsonl").touch()
    (processed / "trajectory_analysis_Q_independent_final_v40.json").touch()
    assert premature_final_artifacts(tmp_path) == [
        "results/v40/processed/trajectory_analysis_Q_independent_final_v40.json",
        "results/v40/raw/formal_functional_Q_independent_final_a4-run.jsonl",
    ]
