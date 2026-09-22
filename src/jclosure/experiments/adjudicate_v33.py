"""Final locked V33 adjudication, including context, KV, final and Phase B limits."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from jclosure.experiments.analyze_primary_v33 import summary
from jclosure.experiments.runtime_v33 import hd
from jclosure.protocol_v33 import stage_freeze, verify, verify_stage
from jclosure.provenance import sha256_file, write_json_atomic

OUT=Path("results/v33/processed")
SOURCE="src/jclosure/experiments/adjudicate_v33.py"


def run(root:Path):
    for stage in ("primary_adjudication","final_opening","factorial_independent_final","context_plan","controls","phase_b_instrumentation"):
        verify_stage(root,stage)
    cfg=verify(root)["config"]
    primary=json.loads((root/OUT/"primary_adjudication_v33.json").read_text())
    final_frame=pd.read_parquet(root/OUT/"factorial_independent_final_v33.parquet")
    final=summary(final_frame,cfg)
    controls=pd.read_parquet(root/OUT/"controls_v33.parquet")
    pivot=controls.pivot(index="state_id",columns="condition",values="relative_l2_to_donor")
    meta=controls.drop_duplicates("state_id").set_index("state_id")
    keys=("WRONG_TOKEN_REC2","SAME_FAMILY_WRONG_STATE_REC2","SHUFFLED_REC2")
    comparison={k:float((pivot.MATCHED_REC2<pivot[k]).mean()) for k in (*keys,"CROSS_FAMILY_WRONG_STATE_REC2","RANDOM_SAME_NORM_REC2")}
    jointly=(np.logical_and.reduce([(pivot.MATCHED_REC2.to_numpy()<pivot[k].to_numpy()) for k in keys]))
    rolefamily={}
    for role in ("development","validation"):
        subset=[sid for sid in pivot.index if meta.loc[sid,"role"]==role]
        rolefamily[role]={}
        for family in cfg["families"]:
            group=[sid for sid in subset if meta.loc[sid,"family"]==family]
            rolefamily[role][family]={"states":len(group),"matched_better_than_all_primary_mismatches_fraction":float(np.mean([all(pivot.loc[sid,"MATCHED_REC2"]<pivot.loc[sid,k] for k in keys) for sid in group]))}
    threshold=cfg["structural_gate"]["context_matched_better_fraction_min"]
    family_min=cfg["structural_gate"]["context_families_required"]
    context_pass=bool(all(sum(x["matched_better_than_all_primary_mismatches_fraction"]>=threshold for x in rolefamily[role].values())>=family_min for role in rolefamily) and float(np.mean(jointly))>=threshold)
    medians=controls.groupby("condition").relative_l2_to_donor.median().to_dict()
    kv={"subset_states":len(pivot),"median_relative_error":{k:medians[k] for k in ("KV_ONLY","CONV2_PLUS_KV","REC2_CONV2_RECIPIENT_KV")},"KV_only_not_global_importance_claim":True}
    phase=json.loads((root/OUT/"phase_b_instrumentation_v33.json").read_text())
    formal=primary["level_2_formal_replication"] and final["replication_pass"] and final["alignment_pass"]
    # A is based on independent dev+validation; the opened final separately confirms.
    A=bool(primary["level_2_formal_replication"])
    B=bool(primary["V33_B_preliminary"] and final["conv_dominance_pass"])
    C=bool(primary["V33_C_preliminary"] and final["median_interaction_ratio"]>=cfg["structural_gate"]["interaction_ratio_median_min"])
    D=bool(A and context_pass)
    formal_outcomes={"V33-A_CROSS_MODEL_REC_CORRECTION_REPLICATED":A,"V33-B_CONV_DOMINANT_HANDOFF_REPLICATED":B,"V33-C_CONDITIONAL_NOT_ADDITIVE_OR_GAIN_ONLY":C,"V33-D_MATCHED_CONTEXT_CORRECTION_REPLICATED":D,"V33-E_CROSS_MODEL_CHANNEL_ORGANIZATION_PARTIAL":bool(not A and (B or primary["level_1_qualitative_replication"])),"V33-F_ARCHITECTURE_SEMANTICS_NOT_COMPARABLE":False,"V33-G_MODEL_SPECIFIC_REC_CORRECTION":False,"V33-H_SHARED_MICRO_MEDIATOR_IDENTIFIED":False,"V33-I_SHARED_EFFECT_WITH_DIFFERENT_MICRO_MEDIATORS":False}
    result={"schema_version":42,"protocol_version":"cross_model_rec_conv_v33","model2_id":cfg["model2"]["id"],"architecture_comparable":True,"model1_historical_final_reopened":False,"model2_independent_final_opened":True,"phase_B_authorized":True,"phase_B_instrumentation_pilot_complete":True,"phase_B_bidirectional_mediation_completed":False,"model2_factorial":{**primary["factorial"],"independent_final":final},"model2_context":{"states":len(pivot),"matched_better_fraction":comparison,"matched_better_than_all_three_fraction":float(np.mean(jointly)),"role_family":rolefamily,"context_gate_pass":context_pass,"condition_median_error":medians,"cross_state_off_manifold_caveat":True},"model2_KV":kv,"gain_rotation_controls":primary["gain_rotation_controls"],"formal_outcomes":formal_outcomes,"level_1_qualitative_replication":primary["level_1_qualitative_replication"],"level_2_formal_replication":A,"level_3_structural_replication":bool(A and C and D),"final_confirmation_pass":bool(final["replication_pass"] and final["alignment_pass"]),"phase_B_exact_instrumentation":{"model1_exact_gate_layers":phase["model1"]["exact_gate_count"],"model1_exact_delta_layers":phase["model1"]["exact_delta_count"],"model2_exact_update_layers":phase["model2"]["exact_update_count"],"bidirectional_mediation_executed":False},"H_I_status":"NOT_ESTABLISHED: instrumentation pilot does not test bidirectional mediation","universal_mechanism_claim":False,"application_benefit_claim":False,"thresholds_retuned":False,"H2_REMAINS":True,"H3_AUTHORIZED":False,"DYNAMIC_STATE_SEARCH_AUTHORIZED":False,"AUTONOMOUS_CONTROLLER_AUTHORIZED":False}
    path=root/OUT/"v33_adjudication.json"
    write_json_atomic(path,result)
    stage=stage_freeze(root,"adjudication",[SOURCE,str(path.relative_to(root)),"artifacts/cross_model_rec_conv_v33_phase_b_instrumentation.freeze.json","artifacts/cross_model_rec_conv_v33_controls.freeze.json","artifacts/cross_model_rec_conv_v33_factorial_independent_final.freeze.json"],{"adjudication_sha256":sha256_file(path),"adjudication_hash":hd(result),"formal_outcomes":formal_outcomes,"phase_B_bidirectional_mediation_completed":False})
    return {"freeze_digest":stage["freeze_digest"],"formal_outcomes":formal_outcomes,"final_confirmation_pass":result["final_confirmation_pass"],"level_3_structural_replication":result["level_3_structural_replication"]}


if __name__=="__main__":
    print(json.dumps(run(Path.cwd()),indent=2))
