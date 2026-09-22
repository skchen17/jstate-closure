"""Frozen model-2 replication gates and development-only gain/rotation controls."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from jclosure.experiments.analyze_v32 import bootstrap_median_lower, restricted_rotation_fit, restricted_rotation_predict
from jclosure.experiments.runtime_v33 import hd
from jclosure.protocol_v33 import stage_freeze, verify, verify_stage
from jclosure.provenance import sha256_file, write_json_atomic

OUT = Path("results/v33/processed")
SOURCE = "src/jclosure/experiments/analyze_primary_v33.py"


def summary(frame, cfg):
    boot, repl, align, conv = (cfg[x] for x in ("bootstrap", "replication_gate", "alignment_gate", "conv_dominance_gate"))
    family = {}
    for fam, group in frame.groupby("family"):
        family[fam] = {"rows": len(group), "positive_fraction": float(group.improvement_positive.mean()), "median_l2_reduction": float(group.relative_conv_error_reduction.median()), "median_alignment": float(group.residual_alignment_cosine.median()), "median_conv_relative_l2": float(group.conv_relative_l2.median()), "median_rec_relative_l2": float(group.rec_only_relative_l2.median()), "median_conv_to_rec_error_ratio": float(group.conv_to_rec_error_ratio.median())}
    pos = float(frame.improvement_positive.mean())
    red = float(frame.relative_conv_error_reduction.median())
    red_lb = bootstrap_median_lower(frame.relative_conv_error_reduction, boot["iterations"], boot["seed"], boot["lower_tail"])
    alignment = float(frame.residual_alignment_cosine.median())
    align_lb = bootstrap_median_lower(frame.residual_alignment_cosine, boot["iterations"], boot["seed"], boot["lower_tail"])
    rf = sum(x["positive_fraction"] >= repl["family_row_improvement_min"] and x["median_l2_reduction"] >= repl["median_l2_reduction_min"] for x in family.values())
    af = sum(x["median_alignment"] >= align["family_median_cosine_min"] for x in family.values())
    cf = sum(x["median_conv_relative_l2"] <= conv["median_conv_relative_l2_max"] and x["median_conv_to_rec_error_ratio"] <= conv["conv_to_rec_error_ratio_max"] for x in family.values())
    cp = float(frame.conv_relative_l2.median()) <= conv["median_conv_relative_l2_max"] and float(frame.conv_to_rec_error_ratio.median()) <= conv["conv_to_rec_error_ratio_max"] and cf >= conv["families_required"]
    rp = pos >= repl["row_improvement_min"] and red >= repl["median_l2_reduction_min"] and red_lb > repl["bootstrap_lower_bound_gt"] and rf >= repl["families_required"]
    ap = alignment >= align["median_cosine_min"] and align_lb >= align["bootstrap_lower_bound_min"] and af >= align["families_required"]
    return {"rows": len(frame), "positive_fraction": pos, "median_l2_reduction": red, "reduction_bootstrap_lower": red_lb, "median_alignment": alignment, "alignment_bootstrap_lower": align_lb, "median_conv_relative_l2": float(frame.conv_relative_l2.median()), "median_rec_relative_l2": float(frame.rec_only_relative_l2.median()), "median_joint_relative_l2": float(frame.joint_relative_l2.median()), "median_conv_to_rec_error_ratio": float(frame.conv_to_rec_error_ratio.median()), "median_interaction_ratio": float(frame.interaction_ratio.median()), "median_rec_to_conv_effect_norm_ratio": float(frame.rec_to_conv_effect_norm_ratio.median()), "joint_beats_oracle_scalar_fraction": float(frame.joint_beats_scalar_gain.mean()), "family": family, "replication_families_passing": rf, "alignment_families_passing": af, "conv_families_passing": cf, "replication_pass": bool(rp), "alignment_pass": bool(ap), "conv_dominance_pass": bool(cp)}


def run(root: Path):
    for role in ("development", "validation"):
        verify_stage(root, f"factorial_{role}")
    cfg = verify(root)["config"]
    frames = {role: pd.read_parquet(root / OUT / f"factorial_{role}_v33.parquet") for role in ("development", "validation")}
    gate = {role: summary(frame, cfg) for role,frame in frames.items()}
    vs = {role: np.load(root / OUT / f"factorial_vectors_{role}_v33.npz")["vectors"].astype(np.float64) for role in frames}
    Cdev, RCdev = vs["development"][:,2]-vs["development"][:,0], vs["development"][:,3]-vs["development"][:,0]
    Cval, Dval = vs["validation"][:,2]-vs["validation"][:,0], vs["validation"][:,4]-vs["validation"][:,0]
    alpha = float(np.sum(Cdev * (vs["development"][:,4]-vs["development"][:,0])) / max(float(np.sum(Cdev*Cdev)),1e-12))
    global_err = np.linalg.norm(Dval-alpha*Cval, axis=1)/np.maximum(np.linalg.norm(Dval,axis=1),1e-12)
    basis,Q = restricted_rotation_fit(Cdev, RCdev,rank=32)
    pred = restricted_rotation_predict(Cval,basis,Q)
    rotation_err = np.linalg.norm(Dval-pred,axis=1)/np.maximum(np.linalg.norm(Dval,axis=1),1e-12)
    val = frames["validation"]
    controls = {"global_gain": {"fit_role":"development_only","test_role":"validation_only","alpha":alpha,"median_relative_error":float(np.median(global_err)),"true_joint_better_fraction":float(np.mean(val.joint_relative_l2.to_numpy()<global_err))},"oracle_scalar": {"descriptive_rowwise_oracle": True,"validation_joint_better_fraction":float(val.joint_beats_scalar_gain.mean()),"validation_median_scalar_relative_error":float(val.scalar_gain_error_relative.median()),"validation_median_true_joint_relative_error":float(val.joint_relative_l2.median())},"restricted_rotation": {"fit_role":"development_only","test_role":"validation_only","rank":int(len(basis)),"global_transform_only":True,"identity_outside_train_subspace":True,"median_relative_error":float(np.median(rotation_err)),"true_joint_better_fraction":float(np.mean(val.joint_relative_l2.to_numpy()<rotation_err)),"rules_out_all_nonlinear_rotations":False}}
    formal = bool(all(gate[role]["replication_pass"] and gate[role]["alignment_pass"] for role in gate))
    conv = bool(all(gate[role]["conv_dominance_pass"] for role in gate))
    level1 = bool(all(gate[role]["median_conv_relative_l2"] < gate[role]["median_rec_relative_l2"] and gate[role]["median_joint_relative_l2"] < gate[role]["median_conv_relative_l2"] for role in gate))
    structural_gain = controls["oracle_scalar"]["validation_joint_better_fraction"] >= cfg["structural_gate"]["joint_beats_scalar_fraction_min"]
    interaction = gate["validation"]["median_interaction_ratio"] >= cfg["structural_gate"]["interaction_ratio_median_min"]
    result={"schema_version":42,"protocol_version":"cross_model_rec_conv_v33","architecture_comparable":True,"factorial":gate,"gain_rotation_controls":controls,"level_1_qualitative_replication":level1,"level_2_formal_replication":formal,"V33_A_preliminary":formal,"V33_B_preliminary":conv,"V33_C_preliminary":bool(formal and structural_gain and interaction),"structural_gain_test":structural_gain,"interaction_material":interaction,"independent_final_opened":formal,"phase_B_opened":formal,"thresholds_retuned":False,"model_1_final_opened":False}
    path=root/OUT/"primary_adjudication_v33.json"
    write_json_atomic(path,result)
    stage=stage_freeze(root,"primary_adjudication",[SOURCE,str(path.relative_to(root)),"artifacts/cross_model_rec_conv_v33_factorial_development.freeze.json","artifacts/cross_model_rec_conv_v33_factorial_validation.freeze.json"],{"primary_adjudication_sha256":sha256_file(path),"formal_replication":formal,"final_open":formal,"phase_b_open":formal})
    opening={"opened":formal,"basis":"locked development and validation correction + alignment gates","development_pass":gate["development"]["replication_pass"] and gate["development"]["alignment_pass"],"validation_pass":gate["validation"]["replication_pass"] and gate["validation"]["alignment_pass"],"mechanism_definition_hash":hd(json.loads((root/OUT/"execution_plan_v33.json").read_text())),"thresholds_retuned":False,"independent_final_responses_observed_before_opening":0}
    opath=root/OUT/"final_opening_v33.json"
    write_json_atomic(opath,opening)
    ostage=stage_freeze(root,"final_opening",[SOURCE,str(opath.relative_to(root)),str(path.relative_to(root))],{"opened":formal,"final_opening_sha256":sha256_file(opath),"phase_B_opened":formal})
    return {"freeze_digest":stage["freeze_digest"],"opening_digest":ostage["freeze_digest"],"formal_replication":formal,"conv_dominance":conv,"development":gate["development"],"validation":gate["validation"]}


if __name__ == "__main__":
    print(json.dumps(run(Path.cwd()),indent=2))
