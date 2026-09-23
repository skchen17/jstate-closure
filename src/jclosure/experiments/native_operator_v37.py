"""V37 native source-derived local prediction, with no fitted validation parameters."""
from __future__ import annotations

import torch
import torch.nn.functional as F

from jclosure.experiments.operator_v36 import factors, parts


def predict_with_factors(key, module, hidden, f, recurrent_state):
    """Predict raw read, normalized read, mixer and state before native replay.

    The exact installed one-token recurrence is evaluated from its actual
    factors. This is algebraic replay, not a learned regression model.
    """
    p = parts(key, f, recurrent_state)
    raw = p["raw"]
    if key == "Q":
        batch, time, _ = hidden.shape
        norm = module.norm(raw.reshape(-1, module.head_v_dim),
                           f["gate"].reshape(-1, module.head_v_dim))
        mixer = module.out_proj(norm.reshape(batch, time, -1))
    else:
        scan = module.norm(raw, f["gate"]) if module.mamba_rms_norm else raw * F.silu(f["gate"])
        norm = scan
        mixer = module.out_proj(scan.to(hidden.dtype))
    return {"raw": raw, "normalized": norm, "mixer": mixer,
            "new_state": p["new_state"], "old_term": p["decayed"],
            "update_term": p["update"], "factors": f}


def predict_cell(key, module, hidden, cache, recurrent_state):
    return predict_with_factors(key, module, hidden,
                                factors(key, module, hidden, cache), recurrent_state)


def finite_state_difference(key, module, hidden, cache, source_state, target_state):
    """Finite precision-aware ΔS effect under one frozen read condition."""
    source = predict_cell(key, module, hidden, cache, source_state)
    target = predict_cell(key, module, hidden, cache, target_state)
    return {name: target[name].float() - source[name].float()
            for name in ("raw", "normalized", "mixer", "new_state", "old_term", "update_term")}
