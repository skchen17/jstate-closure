from pathlib import Path

import pytest

from jclosure.baseline_guard import verify_manifest
from jclosure.config import config_digest
from jclosure.protocol_v3 import verify_hashes, verify_v3_behavioral_config
from jclosure.provenance import sha256_file
from jclosure.records import ClampSchedule, TrialRecord

ROOT = Path(__file__).resolve().parents[1]


def test_v2_hash_regression_manifest_passes():
    result = verify_manifest(
        ROOT, ROOT / "artifacts/phase0_v2_immutable.sha256.json"
    )
    assert result["status"] == "PASSED"
    assert result["verified_files"] >= 55


def test_trial_record_loads_v1_v2_and_v3():
    required = {
        "run_id": "r",
        "prompt_id": "p",
        "task_family": "f",
        "layer": 2,
        "position": -1,
        "intervention": "clean",
        "valid": True,
        "metrics": {},
        "seed": 1,
    }
    assert TrialRecord.from_dict({**required, "schema_version": 1}).schema_version == 1
    assert TrialRecord.from_dict({**required, "schema_version": 2}).schema_version == 2
    assert TrialRecord.from_dict({**required, "schema_version": 3}).schema_version == 3


def test_clamp_modes_produce_distinct_hook_patterns():
    common = {
        "protocol_version": "exploratory_protocol_v3",
        "initial_layer": 24,
        "future_layers": [25, 26],
        "position_scope": "all_non_padding",
        "initial_positions": [1, 2, 3],
        "final_position": 3,
        "state_definition": "V3-Dense",
        "dictionary_size": 4096,
    }
    single = ClampSchedule.build(mode="single", **common)
    final = ClampSchedule.build(mode="persistent_final", **common)
    all_positions = ClampSchedule.build(mode="persistent_all", **common)
    assert single.modified_layer_positions == ((24, 1), (24, 2), (24, 3))
    assert final.modified_layer_positions[-2:] == ((25, 3), (26, 3))
    assert (25, 1) not in final.modified_layer_positions
    assert (25, 1) in all_positions.modified_layer_positions
    assert len({single.modified_layer_positions, final.modified_layer_positions, all_positions.modified_layer_positions}) == 3


def test_clamp_schedule_rejects_unknown_mode():
    with pytest.raises(ValueError, match="unknown clamp mode"):
        ClampSchedule.build(
            protocol_version="exploratory_protocol_v3",
            mode="persistent",
            initial_layer=1,
            future_layers=[],
            position_scope="final",
            initial_positions=[2],
            final_position=2,
            state_definition="V3-Sparse",
            dictionary_size=4096,
        )


def test_v3_freeze_hash_guard_rejects_modified_input(tmp_path):
    target = tmp_path / "input.txt"
    target.write_text("frozen", encoding="utf-8")
    hashes = {"input.txt": sha256_file(target)}
    verify_hashes(tmp_path, hashes)
    target.write_text("changed", encoding="utf-8")
    with pytest.raises(RuntimeError, match="freeze hash mismatch"):
        verify_hashes(tmp_path, hashes)


def test_v3_behavioral_config_must_match_frozen_digest():
    config = {"run": {"stage": "closure_v3_pilot"}, "seed": 7}
    freeze = {
        "behavioral_config_digests": {
            "configs/closure_v3_pilot.yaml": config_digest(config)
        }
    }
    assert verify_v3_behavioral_config(freeze, config) == config_digest(config)
    with pytest.raises(RuntimeError, match="not one of the frozen"):
        verify_v3_behavioral_config(
            freeze, {"run": {"stage": "closure_v3_confirm"}, "seed": 7}
        )
