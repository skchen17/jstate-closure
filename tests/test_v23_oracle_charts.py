import json
from pathlib import Path

import numpy as np

from jclosure.protocol_v23 import verify, verify_stage

ROOT=Path(__file__).resolve().parents[1]


def test_v23_protocol_and_frozen_stages():
    assert verify(ROOT)["freeze_digest"]
    for stage in ("probe_candidate_design","combo_calibration","probe_selection_and_metric","rank_measurement_design"):
        assert verify_stage(ROOT,stage)["freeze_digest"]


def test_v23_probe_anti_leakage_and_hashes():
    design=json.loads((ROOT/"results/v23/processed/probe_candidate_design_v23.json").read_text())
    selected=json.loads((ROOT/"results/v23/processed/probe_selection_v23.json").read_text())
    assert len(selected["train_action_ids"])==512 and len(set(selected["train_action_ids"]))==512
    assert len(selected["heldout_validation_action_ids"])==32
    assert not set(selected["train_action_ids"]) & set(selected["heldout_validation_action_ids"])
    assert not set(selected["train_action_ids"]) & set(design["historical_v22_final_action_ids"])
    assert selected["historical_final_action_overlap"]==[]


def test_v23_measurements_are_complete_and_shaped():
    design=json.loads((ROOT/"results/v23/processed/probe_candidate_design_v23.json").read_text())
    scratch=Path("/data/CSK/J-space-project/v23-oracle-chart-work")
    for kind,role in (("jvp","jvp_rank_development"),("finite","finite_rank_development")):
        for item in design["state_roles"][role]:
            with np.load(scratch/f"rank_{kind}"/f"{kind}_{item['base_trial_id']}.npz") as z:
                if kind=="jvp": assert z["P0_JVP"].shape==z["Pq_JVP"].shape==(288,512)
                else: assert z["P0_plus"].shape==z["Pq_minus"].shape==(512,288)


def test_v23_rank_scaling_and_saturation_are_complete():
    value=json.loads((ROOT/"results/v23/processed/input_rank_scaling_v23.json").read_text())
    assert set(value["gram"])=={"64","128","256","512"}
    for kind in ("JVP","finite"):
        assert set(value["curves"][kind])=={"64","128","256","512"}
        assert len(value[f"{kind}_saturation_checks"])==2


def test_v23_oracle_gate_and_authorization_consistency():
    oracle=json.loads((ROOT/"results/v23/processed/oracle_local_action_charts_v23.json").read_text())
    adj=json.loads((ROOT/"results/v23/processed/v23_adjudication.json").read_text())
    required={"unseen_direction","unseen_sign","unseen_amplitude","unseen_pair","unseen_dense"}
    for kind in ("JVP","finite"):
        assert required <= set(oracle["best"][kind]["categories"])
    assert adj["COMPACT_OPERATOR_SEARCH_REOPENED"] == (adj["ORACLE_LOCAL_CHART_PASS"] or adj["LEARNED_LOCAL_CHART_PASS"])
    assert not adj["H3_AUTHORIZED"] and not adj["DYNAMIC_STATE_SEARCH_AUTHORIZED"]
    assert not adj["historical_final_responses_opened"] and not adj["v23_independent_final_responses_opened"]


def test_v23_machine_outputs_exist():
    for name in ("input_rank_scaling_rows_v23.parquet","oracle_chart_metrics_v23.parquet","v23_adjudication.json"):
        assert (ROOT/"results/v23/processed"/name).is_file()
