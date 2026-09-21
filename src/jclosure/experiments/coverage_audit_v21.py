"""Validation-only practical action geometry and train-span coverage audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from jclosure.experiments import bank_v21 as bank
from jclosure.experiments.z2_correction_v21 import _corrected_gram
from jclosure.protocol_v21 import stage_freeze, verify_stage
from jclosure.provenance import write_json_atomic

SOURCE = "src/jclosure/experiments/coverage_audit_v21.py"
SUMMARY = bank.OUT / "action_coverage_audit_v21.json"


def prepare(root: Path) -> dict:
    verify_stage(root,"action_representations")
    verify_stage(root,"z2_channel_normalization_amendment_1")
    verify_stage(root,"realized_action_representations")
    return stage_freeze(root,"coverage_audit_design",
                        [SOURCE,"artifacts/action_coordinate_geometry_v21_action_representations.freeze.json",
                         "artifacts/action_coordinate_geometry_v21_z2_channel_normalization_amendment_1.freeze.json",
                         "artifacts/action_coordinate_geometry_v21_realized_action_representations.freeze.json"],
                        {"representations":["Z0","Z1","Z2_corrected","Z3_actual_BF16"],
                         "heldout_actions":"six_V20_validation_only",
                         "final_six_action_raw_geometry_opened":False,
                         "final_six_action_responses_opened":False,
                         "metrics":["nearest_train_abs_cosine","train_span_relative_residual","norm_ratio","channel_energy_mismatch"]})


def _kernel_coverage(gram: np.ndarray) -> list[dict]:
    train=gram[:12,:12]
    eigen,u=np.linalg.eigh((train+train.T)/2)
    keep=eigen>max(float(eigen[-1])*1e-9,1e-10)
    pinv=(u[:,keep]/eigen[keep])@u[:,keep].T
    norm_train=np.sqrt(np.maximum(np.diag(train),0))
    rows=[]
    for index in range(12,18):
        cross=gram[:12,index]
        norm=max(float(np.sqrt(max(gram[index,index],0))),1e-12)
        cosine=cross/np.maximum(norm_train*norm,1e-12)
        projected=float(cross@pinv@cross)
        residual=float(np.sqrt(max(gram[index,index]-projected,0))/norm)
        rows.append({"validation_action_index_in_open_18":index,
                     "nearest_train_abs_cosine":float(np.max(np.abs(cosine))),
                     "train_span_relative_residual":residual,
                     "norm_over_train_median":norm/max(float(np.median(norm_train)),1e-12)})
    return rows


def run(root: Path) -> dict:
    design=verify_stage(root,"coverage_audit_design")
    descriptor=json.loads((root/"results/v21/processed/action_representation_v21.json").read_text())
    z3=json.loads((root/"results/v21/processed/realized_action_coverage_v21.json").read_text())
    z0=np.asarray(descriptor["Z0_exact_V20_descriptor"],dtype=np.float64)
    g0=z0@z0.T
    g1=np.asarray(descriptor["Z1_exact_linear_kernel"]["exact_requested_Gram"],dtype=np.float64)
    g2,_=_corrected_gram(descriptor)
    rows={"Z0":_kernel_coverage(g0),"Z1":_kernel_coverage(g1),"Z2_corrected":_kernel_coverage(g2)}
    channels=descriptor["Z2_channel_exact_linear_kernels"]
    for i in range(6):
        channel_energy={channel:float(channels[channel]["coverage"][i]["validation_norm_over_train_median"])
                        for channel in ("REC","Conv","KV")}
        rows["Z2_corrected"][i]["channel_energy_mismatch_ratio"] = channel_energy
    summary={}
    for name,items in rows.items():
        summary[name]={"by_validation_action":items,
                       "median_nearest_train_abs_cosine":float(np.median([x["nearest_train_abs_cosine"] for x in items])),
                       "median_train_span_relative_residual":float(np.median([x["train_span_relative_residual"] for x in items]))}
    summary["Z3_actual_BF16"]=z3
    result={"design_digest":design["freeze_digest"],"representations":summary,
            "final_six_action_raw_geometry_opened":False,
            "final_six_action_responses_opened":False,
            "interpretation_rule":"High_validation_to_train_span_residual_flags_action_coverage_limitation_not_state_nonexistence"}
    target=root/SUMMARY
    target.parent.mkdir(parents=True,exist_ok=True)
    write_json_atomic(target,result)
    return {name:{"nearest_abs_cosine":row["median_nearest_train_abs_cosine"],
                  "span_residual":row["median_train_span_relative_residual"]}
            for name,row in summary.items()}


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("stage",choices=("prepare","run"))
    args=parser.parse_args()
    print(json.dumps(prepare(Path.cwd()) if args.stage=="prepare" else run(Path.cwd()),indent=2))
