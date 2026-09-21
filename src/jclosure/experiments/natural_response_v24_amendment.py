"""Append-only V24 natural-transition eligibility amendment.

The frozen V24 design selected ten historical V19 horizon-panel rows.  Eight
of those rows expose only an h1 teacher token, so an h2-minus-h1 transition is
not defined.  This module measures only the two pre-existing rows with at
least two frozen teacher tokens and records every exclusion explicitly.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from jclosure.experiments import counterfactual_bank_v19 as v19
from jclosure.experiments.layer_response_v24 import (
    OUT,
    SCRATCH,
    _capture_torch,
    _context,
    _projectors,
)
from jclosure.protocol_v24 import stage_freeze, verify, verify_stage
from jclosure.provenance import sha256_file, write_json_atomic


SOURCE = "src/jclosure/experiments/natural_response_v24_amendment.py"


@torch.no_grad()
def run(root: Path) -> dict:
    verify_stage(root, "layer_response_design")
    cfg = verify(root)["config"]
    (
        design,
        split,
        _prompts,
        _teachers,
        _qmap,
        bundle,
        dense,
        _values,
        _rec,
        _att,
        measured,
        state_layer,
        ws_layers,
        ws_count,
    ) = _context(root)
    device = next(bundle.hf_model.parameters()).device
    proj = _projectors(root, device)
    jids = torch.as_tensor(split["selected_j"], device=device)
    lids = torch.as_tensor(split["selected_logits"], device=device)
    scales = {key: float(value) for key, value in split["target_scales"].items()}
    included, excluded = [], []
    for number, item in enumerate(design["natural_transition_states"], 1):
        old = v19._state_metadata(root, item)
        tokens = [int(x) for x in old["teacher_tokens_h8_or_h1"]]
        if len(tokens) < 2:
            excluded.append(
                {
                    "base_trial_id": item["base_trial_id"],
                    "family": item["family"],
                    "available_teacher_tokens": len(tokens),
                    "reason": "h2-minus-h1 undefined: historical row exposes only h1",
                }
            )
            continue
        path = root / SCRATCH / "natural" / f"natural_{item['base_trial_id']}.npz"
        path.parent.mkdir(parents=True, exist_ok=True)
        clean = v19._prefill_history(bundle, str(old["prompt"]), measured, dense, state_layer)
        h1, next_cache = _capture_torch(
            bundle, dense, clean["cache"], tokens[0], clean["prompt_length"],
            jids, lids, scales, ws_layers, ws_count, max(measured),
            cfg["layers"], cfg["architecture_layers"], proj,
        )
        h2, _ = _capture_torch(
            bundle, dense, next_cache, tokens[1], clean["prompt_length"] + 1,
            jids, lids, scales, ws_layers, ws_count, max(measured),
            cfg["layers"], cfg["architecture_layers"], proj,
        )
        arrays = {
            f"natural_{key}": (h2[key] - h1[key]).detach().cpu().numpy().astype(np.float16)
            for key in h1
        }
        np.savez_compressed(
            path,
            **arrays,
            base_trial_id=np.asarray(item["base_trial_id"]),
            family=np.asarray(item["family"]),
        )
        included.append(
            {
                "base_trial_id": item["base_trial_id"],
                "family": item["family"],
                "available_teacher_tokens": len(tokens),
                "path": str(path),
                "sha256": sha256_file(path),
            }
        )
        print(f"V24 natural amendment eligible={len(included)} design_row={number}/10", flush=True)
    payload = {
        "design_count": len(design["natural_transition_states"]),
        "eligible_transition_count": len(included),
        "excluded_count": len(excluded),
        "included": included,
        "excluded": excluded,
        "estimand": "clean teacher-forced h2-minus-h1 only where two frozen historical tokens exist",
        "frozen_layer_response_source_unchanged": True,
        "historical_final_opened": False,
        "v24_independent_final_opened": False,
    }
    target = root / OUT / "natural_transition_amendment_v24.json"
    write_json_atomic(target, payload)
    frozen = stage_freeze(
        root,
        "natural_transition_amendment",
        [
            SOURCE,
            "artifacts/causal_output_bottleneck_v24_layer_response_design.freeze.json",
            str(target.relative_to(root)),
        ],
        payload,
    )
    return {"freeze_digest": frozen["freeze_digest"], **payload}


if __name__ == "__main__":
    print(json.dumps(run(Path.cwd()), indent=2))
