"""Frozen V21 bottleneck and paired-geometry scientific adjudication."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from jclosure.protocol_v21 import stage_freeze, verify, verify_stage
from jclosure.provenance import sha256_file, write_json_atomic

SOURCE="scripts/adjudicate_v21.py"
OUTPUT=Path("results/v21/processed/v21_adjudication.json")


def prepare(root:Path)->dict:
    config=verify(root)["config"]
    verify_stage(root,"factorial_design")
    verify_stage(root,"factorial_extension_design")
    verify_stage(root,"realized_action_factorial_design")
    verify_stage(root,"differential_oracle_design")
    verify_stage(root,"second_order_design")
    verify_stage(root,"paired_geometry_family_gate_amendment_2")
    return stage_freeze(root,"adjudication_design",
                        [SOURCE,"configs/action_coordinate_geometry_v21.yaml",
                         "artifacts/action_coordinate_geometry_v21_paired_geometry_family_gate_amendment_2.freeze.json"],
                        {"practical_action_ceiling_rule":"At_least_one_S2_or_S4_practical_Z1_Z2_Z3_candidate_passes_frozen_V20_direction_sign_and_new_scale_pair_dense_gates; no_Z4_Z5_in_practical_claim",
                         "state_compression_material_gain_min":config["gates"]["state_compression_material_l2_gain_min"],
                         "action_coordinate_material_gain_min":config["gates"]["action_coordinate_material_l2_gain_min"],
                         "model_class_material_gain_min":config["gates"]["model_class_material_l2_gain_min"],
                         "action_data_limited_rule":"Z1_median_validation_span_residual_above_0.25_and_4_to_12_action_L2_drop_above_0.10",
                         "compact_search_reopen_only_if_practical_ceiling_passes":True,
                         "raw_P_to_operator_encoder_only_if_compact_cross_action_passes":True,
                         "H2_remains_default":True,"H3_authorized":False,
                         "DYNAMIC_STATE_SEARCH_AUTHORIZED":False,
                         "final_six_action_responses_opened":False})


def _read(root:Path,name:str)->dict:
    return json.loads((root/"results/v21/processed"/name).read_text())


def _practical(root:Path)->list[dict]:
    rows=[]
    for name,key in (("bottleneck_factorial_v21.json","results"),
                     ("bottleneck_factorial_extension_v21.json","results"),
                     ("z2_channel_normalization_correction_v21.json","results"),
                     ("realized_action_factorial_v21.json","models")):
        for row in _read(root,name)[key]:
            if row["S"] in ("S2","S4") and row["Z"].startswith(("Z1","Z2","Z3")):
                rows.append(row)
    for condition in _read(root,"neural_operator_models_v21.json")["conditions"]:
        for row in condition["models"]:
            if row["S"] in ("S2","S4") and row["Z"].startswith(("Z1","Z2","Z3")):
                rows.append(row)
    return rows


def run(root:Path)->dict:
    design=verify_stage(root,"adjudication_design")
    main=_read(root,"bottleneck_factorial_v21.json")
    extension=_read(root,"bottleneck_factorial_extension_v21.json")
    corrected=_read(root,"z2_channel_normalization_correction_v21.json")
    z3=_read(root,"realized_action_factorial_v21.json")
    scale=_read(root,"action_data_scaling_v21.json")
    coverage=_read(root,"action_coverage_audit_v21.json")
    composition=_read(root,"scale_pair_dense_v21.json")
    geometry=_read(root,"paired_geometry_analysis_v21.json")
    z4=_read(root,"differential_oracle_ceiling_v21.json")
    z5=_read(root,"finite_second_order_oracle_v21.json")
    practical=_practical(root)
    practical_direction_sign=[x for x in practical if x["direction_and_sign_preliminary_gate"]]
    additional=composition["metrics"]
    composition_pass=all(x["reliable_rows"] is not None
                         and x["reliable_rows"]["stack_relative_l2"]<=verify(root)["config"]["gates"]["relative_l2_max"]
                         for x in additional.values())
    action_pass=bool(practical_direction_sign and composition_pass)
    def pick(rows,s,z,model):
        current=[x for x in rows if x["S"]==s and x["Z"]==z and x["model"]==model]
        if len(current)!=1:
            raise RuntimeError(f"V21 bottleneck comparison missing {s}/{z}/{model}")
        return current[0]
    g2="G2_quadratic_action"
    s1=pick(extension["results"],"S1","Z1",g2)["metrics"]["unseen_direction"]["stack_relative_l2"]
    s2=pick(main["results"],"S2","Z1",g2)["metrics"]["unseen_direction"]["stack_relative_l2"]
    z0=pick(main["results"],"S2","Z0",g2)["metrics"]["unseen_direction"]["stack_relative_l2"]
    z1=s2
    z2=pick(corrected["results"],"S2","Z2_train_floor_corrected",g2)["metrics"]["unseen_direction"]["stack_relative_l2"]
    s4=pick(extension["results"],"S4","Z1",g2)["metrics"]["unseen_direction"]["stack_relative_l2"]
    neural_best=min(row["metrics"]["unseen_direction"]["stack_relative_l2"]
                    for condition in _read(root,"neural_operator_models_v21.json")["conditions"]
                    for row in condition["models"] if row["S"]=="S2" and row["Z"] in ("Z1","Z2"))
    z3_best=min(row["metrics"]["unseen_direction"]["stack_relative_l2"] for row in z3["models"])
    state_gain=s1-s2
    action_gain=z0-z1
    model_gain=z1-neural_best
    data_drop=scale["rows"][0]["validation_unseen_direction"]["stack_relative_l2"]-scale["rows"][-1]["validation_unseen_direction"]["stack_relative_l2"]
    action_data_limited=(coverage["representations"]["Z1"]["median_train_span_relative_residual"]>0.25 and data_drop>0.10)
    z4_best=min(row["metrics"]["unseen_direction"]["stack_relative_l2"] for row in z4["state_conditioned_models"])
    z5_direct=z5["direct_oracle"]["direct_Taylor"]["positive_unseen_direction"]["stack_relative_l2"]
    z4_direct=z5["direct_oracle"]["JVP_only"]["positive_unseen_direction"]["stack_relative_l2"]
    results={"ACTION_COORDINATE_CEILING_PASS":action_pass,
             "STATE_COMPRESSION_BOTTLENECK_IDENTIFIED":bool(state_gain>=design["state_compression_material_gain_min"]),
             "ACTION_REPRESENTATION_MATERIAL_IMPROVEMENT":bool(action_gain>=design["action_coordinate_material_gain_min"]),
             "SHARED_MODEL_BOTTLENECK_IDENTIFIED":bool(model_gain>=design["model_class_material_gain_min"] and action_pass),
             "ACTION_DATA_LIMITED":bool(action_data_limited),
             "SHARED_STATE_DEPENDENT_CAUSAL_GEOMETRY_STRONG_SUPPORT":bool(geometry["shared_geometry_strong_support"]),
             "HIGHER_ORDER_TAYLOR_IMPROVES_OVER_JVP":bool(z4_direct-z5_direct>=design["action_coordinate_material_gain_min"]),
             "COMPACT_OPERATOR_SEARCH_REOPENED":bool(action_pass),
             "RAW_TO_OPERATOR_ENCODER_AUTHORIZED":False,
             "DYNAMIC_STATE_SEARCH_AUTHORIZED":False,
             "H2_REMAINS":True,"H3_AUTHORIZED":False,
             "FINAL_SIX_ACTION_RESPONSES_OPENED":False}
    if action_pass:
        raise RuntimeError("V21 practical action ceiling unexpectedly passed: compact search/final gating required before adjudication")
    formal=["V21-F_CROSS_ACTION_OPERATOR_REMAINS_UNIDENTIFIED_UNDER_TESTED_PRACTICAL_FAMILIES"]
    if action_data_limited:
        formal.append("V21-G_ACTION_DATA_LIMITED")
    if geometry["shared_geometry_strong_support"]:
        formal.append("V21-E_SHARED_STATE_DEPENDENT_CAUSAL_GEOMETRY")
    result={"design_digest":design["freeze_digest"],"formal_outcomes":formal,
            "flags":results,"comparisons":{"S1_Z1_G2_unseen_L2":s1,"S2_Z1_G2_unseen_L2":s2,
                                    "S4_Z1_G2_unseen_L2":s4,"S2_Z0_G2_unseen_L2":z0,
                                    "S2_Z2_corrected_G2_unseen_L2":z2,"best_Z3_unseen_L2":z3_best,
                                    "best_G4_G5_S2_rich_Z_unseen_L2":neural_best,
                                    "state_compression_gain_absolute_L2":state_gain,
                                    "action_coordinate_gain_absolute_L2":action_gain,
                                    "stronger_model_gain_absolute_L2":model_gain,
                                    "action_count_4_to_12_L2_drop":data_drop,
                                    "Z1_validation_span_residual_median":coverage["representations"]["Z1"]["median_train_span_relative_residual"],
                                    "Z4_best_state_conditioned_unseen_L2":z4_best,
                                    "Z4_direct_JVP_unseen_L2":z4_direct,
                                    "Z5_direct_Taylor_unseen_L2":z5_direct},
            "practical_candidate_count":len(practical),
            "practical_direction_sign_gate_count":len(practical_direction_sign),
            "scale_pair_dense_representative_candidate_pass":composition_pass,
            "final_opening_reason":"No practical candidate passed development direction/sign gates; sealed final six remain unopened.",
            "oracle_Z4_Z5_not_deployable":True,
            "state_S4_Nystrom_and_action_Z3_Nystrom_are_approximations":True,
            "nonexistence_claimed":False,
            "H3_or_autonomous_controller_claimed":False,
            "source_hashes":{name:sha256_file(root/"results/v21/processed"/name) for name in
                             ("bottleneck_factorial_v21.json","bottleneck_factorial_extension_v21.json",
                              "z2_channel_normalization_correction_v21.json","realized_action_factorial_v21.json",
                              "neural_operator_models_v21.json","action_data_scaling_v21.json",
                              "action_coverage_audit_v21.json","scale_pair_dense_v21.json",
                              "paired_geometry_analysis_v21.json","differential_oracle_ceiling_v21.json",
                              "finite_second_order_oracle_v21.json")}}
    write_json_atomic(root/OUTPUT,result)
    return {"formal_outcomes":formal,"flags":results,"best_practical_unseen_L2":min(x["metrics"]["unseen_direction"]["stack_relative_l2"] for x in practical)}


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("stage",choices=("prepare","run"))
    args=parser.parse_args()
    print(json.dumps(prepare(Path.cwd()) if args.stage=="prepare" else run(Path.cwd()),indent=2))
