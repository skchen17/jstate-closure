"""Freeze model-2 confirmatory context-control mapping before control responses."""
from __future__ import annotations

import json
from pathlib import Path

from jclosure.experiments.runtime_v33 import hd
from jclosure.protocol_v33 import stage_freeze, verify, verify_stage
from jclosure.provenance import sha256_file, write_json_atomic

OUT=Path("results/v33/processed")
SOURCE="src/jclosure/experiments/context_plan_v33.py"
SEED=330133


def run(root:Path):
    for role in ("development","validation"):
        verify_stage(root,f"factorial_{role}")
    cfg=verify(root)["config"]
    design=json.loads((root/OUT/"design_v33.json").read_text())
    execution=json.loads((root/OUT/"execution_plan_v33.json").read_text())
    rows=[]
    for role in ("development","validation"):
        role_rows=design[role]
        by_id={r["base_trial_id"]:r for r in role_rows}
        for family in cfg["families"]:
            for sid in execution["context_subset"][role][family]:
                target=by_id[sid]
                alternatives=[tid for tid in target["eligible_token_ids"] if tid!=target["primary_token_id"]]
                if not alternatives:
                    raise RuntimeError(f"V33 wrong-token candidate shortage {sid}")
                wrong_token=min(alternatives,key=lambda t:hd([SEED,sid,"wrong_token",t]))
                same=[r for r in role_rows if r["family"]==family and r["base_trial_id"]!=sid]
                cross=[r for r in role_rows if r["family"]!=family]
                shuffle=[r for r in role_rows if r["base_trial_id"]!=sid]
                picks={"same_family_wrong_state_id":min(same,key=lambda r:hd([SEED,sid,"same",r["base_trial_id"]]))["base_trial_id"],"cross_family_wrong_state_id":min(cross,key=lambda r:hd([SEED,sid,"cross",r["base_trial_id"]]))["base_trial_id"],"shuffle_state_id":min(shuffle,key=lambda r:hd([SEED,sid,"shuffle",r["base_trial_id"]]))["base_trial_id"]}
                rows.append({"role":role,"family":family,"state_id":sid,"wrong_token_id":wrong_token,**picks,"random_seed":int(hd([SEED,sid,"random"])[:8],16)})
    if len(rows)!=20:
        raise RuntimeError("V33 context plan must contain 20 states")
    payload={"rows":rows,"conditions":cfg["context_controls"],"KV_conditions":["KV_ONLY","CONV2_PLUS_KV","REC2_CONV2_RECIPIENT_KV"],"mapping_rule":"deterministic hash of frozen state/token IDs only; no response-ranked selection","control_responses_observed_before_freeze":False,"cross_state_off_manifold_caveat":True}
    path=root/OUT/"context_plan_v33.json"
    write_json_atomic(path,payload)
    stage=stage_freeze(root,"context_plan",[SOURCE,str(path.relative_to(root)),"artifacts/cross_model_rec_conv_v33_factorial_development.freeze.json","artifacts/cross_model_rec_conv_v33_factorial_validation.freeze.json"],{"plan_hash":hd(payload),"control_responses_observed_before_freeze":False})
    return {"freeze_digest":stage["freeze_digest"],"rows":len(rows)}


if __name__=="__main__":
    print(json.dumps(run(Path.cwd()),indent=2))
