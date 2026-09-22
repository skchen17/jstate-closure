"""Paired disjoint V34 semantic-state panel, before model write observations."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from jclosure.protocol_v34 import stage_freeze,verify
from jclosure.provenance import write_json_atomic

OUT=Path("results/v34/processed")
SOURCE="src/jclosure/experiments/panel_v34.py"
ROLES=("calibration","development","validation","independent_final")


def hd(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":"),ensure_ascii=False).encode()).hexdigest()


def run(root:Path):
    cfg=verify(root)["config"]
    used={r["base_trial_id"] for v in cfg["source_exclusions"] for role in ROLES for r in json.loads((root/f"results/v{v}/processed/design_v{v}.json").read_text()).get(role,[]) if "base_trial_id" in r}
    roles={role:[] for role in ROLES}
    for family in cfg["families"]:
        for source in ("train","validation"):
            frame=pd.read_parquet(root/f"results/v18/processed/crossed_state_{source}_{family}_v18.parquet",columns=["base_trial_id","prompt"])
            candidates=[x for x in frame.to_dict("records") if x["base_trial_id"] not in used]
            candidates.sort(key=lambda r:hd([cfg["seed"],family,source,r["base_trial_id"]]))
            if source=="train":
                assignments=(("calibration",4),("development",16),("independent_final",8))
            else:
                assignments=(("validation",8),)
            offset=0
            for role,count in assignments:
                if len(candidates)<offset+count:raise RuntimeError(f"V34 insufficient fresh states {family}/{source}")
                for row in candidates[offset:offset+count]:
                    roles[role].append({"base_trial_id":row["base_trial_id"],"family":family,"source_role":source,"role":role,"prompt_sha256":hashlib.sha256(str(row["prompt"]).encode()).hexdigest()})
                offset+=count
    ids=[r["base_trial_id"] for role in ROLES for r in roles[role]]
    if len(ids)!=180 or len(set(ids))!=180 or set(ids)&used:raise RuntimeError("V34 disjointness violation")
    payload={**roles,"families":cfg["families"],"exclusion_versions":cfg["source_exclusions"],"excluded_state_count":len(used),"paired_semantic_state_ids_for_both_models":True,"role_hashes":{r:hd([x["base_trial_id"] for x in roles[r]]) for r in ROLES},"write_or_future_response_observed_before_freeze":False,"historical_independent_finals_reopened":False}
    OUT.mkdir(parents=True,exist_ok=True)
    path=root/OUT/"panel_v34.json"
    write_json_atomic(path,payload)
    freeze=stage_freeze(root,"panel",[SOURCE,str(path.relative_to(root))],{"panel_hash":hd(payload),"role_hashes":payload["role_hashes"],"write_or_future_response_observed_before_freeze":False})
    return {"freeze_digest":freeze["freeze_digest"],"excluded":len(used),"roles":{r:len(roles[r]) for r in ROLES}}


if __name__=="__main__":
    print(json.dumps(run(Path.cwd()),indent=2))
