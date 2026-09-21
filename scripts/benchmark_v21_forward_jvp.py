"""Read-only engineering benchmark for forward-mode exact V21 JVP."""

from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from jclosure.experiments import bank_v21 as bank
from jclosure.experiments import counterfactual_bank_v19 as v19
from jclosure.experiments import operator_bank_v20 as v20_bank
from jclosure.experiments import response_operator_v20 as v20_response
from jclosure.experiments.paired_geometry_v21 import _one_jvp, _target, _torch_stack
from jclosure.protocol_v21 import verify_stage


def main():
    root=Path.cwd()
    roles=verify_stage(root,"roles")
    design=verify_stage(root,"paired_geometry_design")
    split=json.loads((root/"artifacts/compact_causal_response_operator_v20_splits.freeze.json").read_text())
    prompts=v20_bank.prompt_index(root)
    teacher=v20_response._teacher_map(root)
    bundle,dense,_,values,rec,att,measured,state_layer,ws_layers,ws_count=v19._setup(root)
    for parameter in bundle.hf_model.parameters():
        parameter.requires_grad_(False)
    item=roles["jvp_development"][0]
    clean=v19._prefill_history(bundle,str(prompts[item["base_trial_id"]]["prompt"]),measured,dense,state_layer)
    state=clean["cache"]
    token=int(teacher[item["base_trial_id"]][0])
    device=next(bundle.hf_model.parameters()).device
    jids=torch.as_tensor(np.asarray(split["selected_j"],dtype=int),device=device,dtype=torch.long)
    lids=torch.as_tensor(np.asarray(split["selected_logits"],dtype=int),device=device,dtype=torch.long)
    scales={key:float(value) for key,value in split["target_scales"].items()}
    records=[]
    for index in design["probe_direction_indices"][:5]:
        row=v19._action_row(values["directions"],int(index),float(design["probe_alpha_by_direction"][str(index)]),1)
        def target(epsilon):
            return _torch_stack(_target(bundle,dense,state,clean["prompt_length"],token,jids,lids,
                                        ws_layers,ws_count,max(measured),row,rec,att,epsilon),scales)
        t0=perf_counter()
        backward=_one_jvp(bundle,dense,state,clean["prompt_length"],token,jids,lids,ws_layers,ws_count,
                          max(measured),row,rec,att,scales)
        t1=perf_counter()
        epsilon=torch.zeros((),device=device,dtype=torch.float32)
        t2=perf_counter()
        _,forward=torch.func.jvp(target,(epsilon,),(torch.ones_like(epsilon),))
        t3=perf_counter()
        forward=forward.detach().cpu().numpy().astype(np.float32)
        records.append({"direction_index":int(index),"backward_seconds":t1-t0,"forward_seconds":t3-t2,
                        "relative_l2_difference":float(np.linalg.norm(forward-backward)/max(np.linalg.norm(backward),1e-12)),
                        "cosine":float(np.dot(forward,backward)/max(np.linalg.norm(forward)*np.linalg.norm(backward),1e-12))})
    print(json.dumps(records,indent=2))


if __name__=="__main__":
    main()
