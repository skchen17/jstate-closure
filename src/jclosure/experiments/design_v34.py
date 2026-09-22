"""Model-local, response-blind V34 current-token forks and shared-form endpoint design."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from jclosure.experiments.design_v31 import usable
from jclosure.experiments.runtime_v34 import field_hashes,hd,load,prefix,step
from jclosure.protocol_v34 import stage_freeze,verify,verify_stage
from jclosure.provenance import write_json_atomic

OUT=Path("results/v34/processed")
ROLES=("calibration","development","validation","independent_final")
SOURCE="src/jclosure/experiments/design_v34.py"


def prompt_lookup(root,panel):
    lookup={}
    needed={(row["source_role"],row["family"]) for role in ROLES for row in panel[role]}
    for source,family in sorted(needed):
        table=pd.read_parquet(root/f"results/v18/processed/crossed_state_{source}_{family}_v18.parquet",columns=["base_trial_id","prompt"])
        lookup.update({str(x.base_trial_id):str(x.prompt) for x in table.itertuples(index=False)})
    return lookup


@torch.no_grad()
def run(root:Path,key:str):
    verify_stage(root,"panel")
    cfg=verify(root)["config"]
    spec=cfg["models"][key]
    panel=json.loads((root/OUT/"panel_v34.json").read_text())
    lookup=prompt_lookup(root,panel)
    previous=json.loads((root/spec["candidate_library_source"]).read_text())
    library=previous["token_library"]
    model,tokenizer=load(root,key)
    device=next(model.parameters()).device
    rng=np.random.default_rng(cfg["seed"]+(0 if key=="Q" else 1))
    vocab=int(model.get_output_embeddings().weight.shape[0])
    hidden_size=int(model.config.hidden_size)
    bundle={"selected_logits":rng.choice(vocab,64,replace=False).tolist(),"broad_logits":rng.choice(vocab,64,replace=False).tolist(),"broad_sign":rng.choice([-1,1],64).tolist(),"late_hidden_indices":rng.choice(hidden_size,64,replace=False).tolist(),"workspace_layers":spec["workspace_layers"],"workspace_per_layer":32,"late_layer":spec["late_layer"],"J_analogue":"SECONDARY_ONLY_Q" if key=="Q" else "J_NOT_COMPARABLE"}
    roles={role:[] for role in ROLES}
    clean_rows=[]
    for role in ROLES:
        for n,item in enumerate(panel[role],1):
            sid=item["base_trial_id"]
            prompt=lookup[sid]
            if hashlib.sha256(prompt.encode()).hexdigest()!=item["prompt_sha256"]:raise RuntimeError(f"V34 prompt drift {sid}")
            incoming,length,ids,logits=prefix(model,tokenizer,key,prompt)
            if length>cfg["context_limit"]:raise RuntimeError(f"V34 context too long {sid}")
            ranked=torch.topk(logits,k=cfg["token_selection"]["rank_max"]).indices.tolist()
            rank={int(t):i+1 for i,t in enumerate(ranked)}
            if spec["anchor_token_id"] not in rank:raise RuntimeError(f"V34 anchor ineligible {key}:{sid}")
            eligible=[x for x in library if int(x["token_id"]) in rank and int(x["token_id"])!=spec["anchor_token_id"]]
            if not eligible:raise RuntimeError(f"V34 no natural donor token {key}:{sid}")
            donor=min(eligible,key=lambda x:hd([cfg["seed"],key,sid,x["token_id"]]))
            probes=[int(t) for t in ranked if usable(tokenizer.decode([int(t)]))][:cfg["endpoint"]["future_probes"]]
            if len(probes)!=6:raise RuntimeError("V34 probe shortage")
            hashes=field_hashes(incoming)
            row={**item,"prefix_token_hash":hd(ids),"fork_total_length":length,"incoming_state_hashes":hashes,"incoming_state_hash":hd(hashes),"ranked_token_hash":hd(ranked),"recipient_token_id":spec["anchor_token_id"],"donor_token_id":int(donor["token_id"]),"donor_token_category":donor["category"],"recipient_rank":rank[spec["anchor_token_id"]],"donor_rank":rank[int(donor["token_id"])],"token_pair_hash":hd([spec["anchor_token_id"],int(donor["token_id"])]),"eligible_count":len(eligible),"future_probe_tokens":probes,"future_probe_hash":hd(probes)}
            roles[role].append(row)
            if role=="calibration":
                clean=step(model,incoming,spec["anchor_token_id"],length)["cache"]
                for probe in probes:
                    clean_rows.append(step(model,clean,probe,length+1,bundle)["targets"])
            if n%10==0 or n==len(panel[role]):print(f"V34 design {key} {role} {n}/{len(panel[role])}",flush=True)
    scales={}
    for block in ("logits","semantic","broad_vocabulary","late_hidden","workspace"):
        matrix=np.stack([r[block] for r in clean_rows]).astype(np.float64)
        scales[block]=max(float(np.sqrt(np.mean((matrix-matrix.mean(axis=0))**2))),1e-6)
    result={"model_key":key,"model_id":spec["id"],"model_revision":spec["revision"],**roles,"anchor_token_id":spec["anchor_token_id"],"token_library_source":spec["candidate_library_source"],"token_library_hash":hd(library),"target_bundle":bundle,"calibration_clean_scales":scales,"role_hashes":panel["role_hashes"],"same_semantic_state_ids_as_other_model":True,"current_token_write_geometry_used_for_selection":False,"future_causal_response_observed_before_freeze":False,"historical_final_reopened":False}
    path=root/OUT/f"design_{key}_v34.json"
    write_json_atomic(path,result)
    stage=stage_freeze(root,f"design_{key}",[SOURCE,str(path.relative_to(root)),"artifacts/functional_mediation_v34_panel.freeze.json"],{"model_key":key,"design_hash":hd(result),"role_hashes":panel["role_hashes"],"future_causal_response_observed_before_freeze":False})
    return {"freeze_digest":stage["freeze_digest"],"model":key,"scales":scales,"roles":{r:len(roles[r]) for r in ROLES}}


def finalize(root:Path):
    for key in ("Q","F"):verify_stage(root,f"design_{key}")
    cfg=verify(root)["config"]
    q=json.loads((root/OUT/"design_Q_v34.json").read_text())
    f=json.loads((root/OUT/"design_F_v34.json").read_text())
    for role in ROLES:
        if [x["base_trial_id"] for x in q[role]] != [x["base_trial_id"] for x in f[role]]:raise RuntimeError("V34 semantic pairing drift")
    plan={"models":["Q","F"],"roles":{r:[x["base_trial_id"] for x in q[r]] for r in ROLES},"factorial":cfg["factorial"],"primary_KV":cfg["primary_KV"],"estimands":{"D":"Ydonor-Y00","C":"Y01-Y00","R":"Y10-Y00","RC":"Y11-Y00","E_conv":"D-C","R_given_C":"Y11-Y01","B_REC":"||D-C||-||D-RC||","rho_REC":"B_REC/||D-C||"},"stage_order":cfg["functional_stages"]["order"],"stage_scope":cfg["functional_stages"]["layer_scope"],"gate":cfg["stage_gate"],"high_level_gate":cfg["high_level_gate"],"pipeline_sets":cfg["pipeline_sets"],"final_rule":cfg["final_rule"],"later_horizon_rule":cfg["later_horizon_rule"],"thresholds_retuned":False,"causal_response_observed_before_freeze":False}
    path=root/OUT/"execution_plan_v34.json"
    write_json_atomic(path,plan)
    stage=stage_freeze(root,"design",[SOURCE,str(path.relative_to(root)),"artifacts/functional_mediation_v34_design_Q.freeze.json","artifacts/functional_mediation_v34_design_F.freeze.json"],{"plan_hash":hd(plan),"causal_response_observed_before_freeze":False})
    return {"freeze_digest":stage["freeze_digest"],"paired_states":sum(len(plan["roles"][r]) for r in ROLES)}


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("model",choices=("Q","F","finalize"))
    args=parser.parse_args()
    print(json.dumps(finalize(Path.cwd()) if args.model=="finalize" else run(Path.cwd(),args.model),indent=2))
