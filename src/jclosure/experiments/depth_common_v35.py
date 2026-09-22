"""Shared exact native read captures, depth patches and benefit metrics."""
from __future__ import annotations

import numpy as np

from jclosure.experiments.functional_hooks_v34 import FunctionalIntervention
from jclosure.experiments.read_hooks_v35 import ReadPatch
from jclosure.experiments.runtime_v34 import field_hashes, hd, native_swap, prefix, step


def cosine(a, b):
    x, y = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    return float(np.dot(a, b) / (x * y)) if x * y > 1e-12 else None


def prepare(model, tokenizer, key, item, prompt):
    incoming, length, ids, _ = prefix(model, tokenizer, key, prompt)
    if hd(ids) != item["prefix_token_hash"] or field_hashes(incoming) != item["incoming_state_hashes"]:
        raise RuntimeError(f"V35 incoming drift {key}:{item['base_trial_id']}")
    recipient = step(model, incoming, item["recipient_token_id"], length)["cache"]
    donor = step(model, incoming, item["donor_token_id"], length)["cache"]
    conv, _ = native_swap(recipient, donor, ["Conv"], key)
    joint, _ = native_swap(recipient, donor, ["REC", "Conv"], key)
    hashes = field_hashes(recipient)
    if field_hashes(conv)["KV"] != hashes["KV"] or field_hashes(joint)["KV"] != hashes["KV"]:
        raise RuntimeError("V35 recipient KV drift")
    return {"recipient": recipient, "donor": donor, "conv": conv, "joint": joint,
            "length": length, "recipient_KV_hash": hashes["KV"]}


def capture(model, key, cache, probe, length, bundle):
    with FunctionalIntervention(model, key) as hook:
        out = step(model, cache, probe, length + 1, bundle)
    if len(hook.capture) != 24:
        raise RuntimeError("V35 incomplete native read capture")
    return out, hook.capture


def patch(model, key, cache, probe, length, bundle, references, source_by_layer):
    with ReadPatch(model, key, references, source_by_layer) as hook:
        out = step(model, cache, probe, length + 1, bundle)
    return out, hook.proof()


def benefit_metrics(yd, yconv, yjoint, remove, restore, donor_effect_norm):
    e_conv = float(np.linalg.norm(yd - yconv))
    e_joint = float(np.linalg.norm(yd - yjoint))
    benefit = e_conv - e_joint
    remove_error = float(np.linalg.norm(yd - remove))
    restore_error = float(np.linalg.norm(yd - restore))
    correction = yjoint - yconv
    induced = restore - yconv
    valid = benefit > 1e-12
    donor_norm = max(float(donor_effect_norm), 1e-12)
    return {"status": "VALID" if valid else "NONPOSITIVE_BENEFIT",
            "benefit_absolute": benefit, "conv_error": e_conv, "joint_error": e_joint,
            "remove_error": remove_error, "restore_error": restore_error,
            "removed_fraction": (remove_error - e_joint) / benefit if valid else None,
            "restored_fraction": (e_conv - restore_error) / benefit if valid else None,
            "reverse_correction_cosine": cosine(induced, correction),
            "reverse_magnitude_ratio": float(np.linalg.norm(induced)) / max(float(np.linalg.norm(correction)), 1e-12),
            "reverse_relative_l2_to_true_correction": float(np.linalg.norm(induced - correction)) / max(float(np.linalg.norm(correction)), 1e-12),
            "remove_donor_relative_l2": remove_error / donor_norm,
            "restore_donor_relative_l2": restore_error / donor_norm}
