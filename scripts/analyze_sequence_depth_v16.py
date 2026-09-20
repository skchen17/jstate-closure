"""Frozen V16 correction: exact-depth versus at-most-depth reachable sets."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from jclosure.protocol_v16 import stage_freeze, verify
from jclosure.provenance import sha256_file, write_json_atomic

SOURCE="scripts/analyze_sequence_depth_v16.py"
INPUT="results/v16/processed/sequence_reachability_v16.parquet"
OUT="results/v16/processed/sequence_reachability_cumulative_v16.parquet"
SUMMARY="results/v16/processed/sequence_reachability_cumulative_v16.json"
REPORT="reports/SEQUENCE_REACHABILITY_AMENDMENT_V16.md"


def main()->None:
    root=Path.cwd();verify(root)
    freeze=stage_freeze(root,"sequence_depth_correction",[SOURCE,INPUT,"results/v16/processed/sequence_reachability_v16.json"],
                        {"role":"validation_diagnostic_post_measurement_correction",
                         "reason":"original depth rows denote exactly 1/2/4 active actions; union over shallower depths is required for at-most-depth reachability",
                         "no_new_model_forward_or_teacher_label":True})
    frame=pd.read_parquet(root/INPUT)
    frame=frame[frame.status=="MEASURED"].copy()
    rows=[]
    for (state,family),group in frame.groupby(["base_trial_id","family"]):
        seen=[]
        for depth in (1,2,4):
            own=group[group.depth==depth]
            if len(own)!=1:raise RuntimeError(f"missing/duplicate depth {depth} for {state}")
            seen.append(own.iloc[0])
            best=min(seen,key=lambda row:float(row.nearest_relative_residual))
            rows.append({"base_trial_id":state,"family":family,"max_actions":depth,
                         "candidate_count_at_most":int(sum(int(row.candidate_count) for row in seen)),
                         "nearest_relative_residual_at_most":float(best.nearest_relative_residual),
                         "best_exact_action_count":int(best.depth),"best_sequence":best.nearest_sequence,
                         "teacher_norm":float(best.teacher_norm),"freeze_digest":freeze["freeze_digest"]})
    result=pd.DataFrame(rows)
    result.to_parquet(root/OUT,index=False,compression="zstd")
    pooled=result.groupby("max_actions").agg(states=("base_trial_id","nunique"),
                                               candidate_count_at_most=("candidate_count_at_most","median"),
                                               nearest_relative_residual_at_most=("nearest_relative_residual_at_most","median"),
                                               median_best_exact_action_count=("best_exact_action_count","median")).reset_index()
    summary={"freeze_digest":freeze["freeze_digest"],"source_exact_depth_records":INPUT,
             "source_exact_depth_sha256":sha256_file(root/INPUT),"records":OUT,
             "records_sha256":sha256_file(root/OUT),"by_max_actions":pooled.to_dict("records"),
             "interpretation":"finite union over exact-depth candidate sets, same frozen h4 teacher J target",
             "additional_model_forwards":0}
    write_json_atomic(root/SUMMARY,summary)
    body=f"""# V16 sequence-depth interpretation amendment

The original `sequence_reachability_v16.parquet` records **exactly** 1, 2 or 4 active actions. Those sets are not nested. This frozen, post-measurement correction takes their union for an **at-most**-depth comparison; it changes no action response, teacher label, split or historical record.

{pooled.to_markdown(index=False,floatfmt='.3f')}

The five-state four-primitive search is still limited, and even the best at-most-four residual is far above the teacher target norm. It does not support general nonlinear controllability or authorize MPC. Per-state best exact count and sequence: `{OUT}` (SHA256 `{summary['records_sha256']}`). Correction freeze digest `{freeze['freeze_digest']}`.
"""
    (root/REPORT).write_text(body,encoding="utf-8")
    print(json.dumps({"freeze_digest":freeze["freeze_digest"],"by_max_actions":summary["by_max_actions"]},sort_keys=True))


if __name__=="__main__":main()
