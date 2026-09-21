"""Final V21 cross-file invariant verification."""

from __future__ import annotations

import json
from pathlib import Path

from jclosure.protocol_v21 import verify,verify_stage
from jclosure.provenance import sha256_file,write_json_atomic

REQUIRED=(
    "action_representation_v21.json","state_representations_v21.json","raw_state_reference_v21.json",
    "realized_action_v21.json","bottleneck_factorial_v21.json","bottleneck_factorial_extension_v21.json",
    "z2_channel_normalization_correction_v21.json","neural_operator_models_v21.json",
    "realized_action_factorial_v21.json","action_coverage_audit_v21.json","action_data_scaling_v21.json",
    "scale_pair_dense_v21.json","paired_jvp_finite_operator_v21.json","paired_geometry_analysis_v21.json",
    "paired_geometry_bootstrap_v21.json","differential_oracle_ceiling_v21.json",
    "finite_second_order_oracle_v21.json","v21_adjudication.json",
    "v21_test_audit.json","v21_test_audit_amendment_1.json")


def main():
    root=Path.cwd()
    base=verify(root)
    for path in sorted((root/"artifacts").glob("action_coordinate_geometry_v21_*.freeze.json")):
        verify_stage(root,path.name.removeprefix("action_coordinate_geometry_v21_").removesuffix(".freeze.json"))
    paths=[root/"results/v21/processed"/name for name in REQUIRED]
    missing=[str(path.relative_to(root)) for path in paths if not path.exists()]
    if missing:
        raise RuntimeError(f"V21 required records missing: {missing}")
    adjudication=json.loads((root/"results/v21/processed/v21_adjudication.json").read_text())
    flags=adjudication["flags"]
    if flags["FINAL_SIX_ACTION_RESPONSES_OPENED"] or flags["DYNAMIC_STATE_SEARCH_AUTHORIZED"] or flags["RAW_TO_OPERATOR_ENCODER_AUTHORIZED"]:
        raise RuntimeError("V21 sealed/auth invariant violated")
    paired=json.loads((root/"results/v21/processed/paired_jvp_finite_operator_v21.json").read_text())
    if paired["development_base_states"]!=50 or paired["validation_base_states"]!=25 or not paired["old_reverse_over_reverse_matrices_excluded"]:
        raise RuntimeError("V21 paired panel incomplete or mixed")
    geometry=json.loads((root/"results/v21/processed/paired_geometry_analysis_v21.json").read_text())
    if geometry["records"]!=75 or geometry["exact_autograd_method"]!="torch.func.jvp_forward_AD":
        raise RuntimeError("V21 geometry analysis method/count drift")
    test=json.loads((root/"results/v21/processed/v21_test_audit_amendment_1.json").read_text())
    if test["runs"]["v21_only"]["failed"]!=0 or test["runs"]["all"]["failed"]!=2:
        raise RuntimeError("V21 test audit counts drift")
    result={"protocol_digest":base["freeze_digest"],"required_record_count":len(paths),
            "required_record_hashes":{str(path.relative_to(root)):sha256_file(path) for path in paths},
            "verified_stage_count":len(list((root/"artifacts").glob("action_coordinate_geometry_v21_*.freeze.json"))),
            "paired_base_count":geometry["records"],"final_six_action_responses_opened":False,
            "dynamic_state_search_authorized":False,"raw_to_operator_encoder_authorized":False,
            "status":"PASS"}
    target=root/"results/v21/processed/v21_verification.json"
    write_json_atomic(target,result)
    print(json.dumps(result,indent=2))


if __name__=="__main__":
    main()
