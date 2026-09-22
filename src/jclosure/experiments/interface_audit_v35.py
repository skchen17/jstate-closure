"""Calibration-only native-read equivalence and exact depth-patch audit."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from jclosure.experiments.design_v34 import prompt_lookup
from jclosure.experiments.functional_hooks_v34 import FunctionalIntervention
from jclosure.experiments.read_hooks_v35 import ReadPatch
from jclosure.experiments.runtime_v34 import field_hashes, hd, load, native_swap, prefix, step
from jclosure.experiments.transaction_v28 import thash
from jclosure.protocol_v35 import stage_freeze, verify, verify_stage
from jclosure.provenance import write_json_atomic

OUT = Path("results/v35/processed")
SOURCE = "src/jclosure/experiments/interface_audit_v35.py"


@torch.no_grad()
def run(root: Path, key: str):
    verify_stage(root, "design")
    cfg = verify(root)["config"]
    design = json.loads((root / OUT / f"design_{key}_v35.json").read_text())
    panel = json.loads((root / OUT / "panel_v35.json").read_text())
    lookup = prompt_lookup(root, panel)
    item = design["calibration"][0]
    model, tokenizer = load(root, key)
    incoming, length, ids, _ = prefix(model, tokenizer, key, lookup[item["base_trial_id"]])
    if hd(ids) != item["prefix_token_hash"] or field_hashes(incoming) != item["incoming_state_hashes"]:
        raise RuntimeError("V35 audit incoming drift")
    recipient = step(model, incoming, item["recipient_token_id"], length)["cache"]
    donor = step(model, incoming, item["donor_token_id"], length)["cache"]
    conv, conv_proof = native_swap(recipient, donor, ["Conv"], key)
    joint, joint_proof = native_swap(recipient, donor, ["REC", "Conv"], key)
    probe = int(item["future_probe_tokens"][0])
    bundle = design["target_bundle"]
    branches = {"CONV": conv, "JOINT": joint}
    baseline, capture = {}, {}
    for name, cache in branches.items():
        baseline[name] = step(model, cache, probe, length + 1, bundle)
        with FunctionalIntervention(model, key) as hook:
            replay = step(model, cache, probe, length + 1, bundle)
        capture[name] = hook.capture
        if not torch.equal(baseline[name]["logits"], replay["logits"]):
            raise RuntimeError(f"V35 V34-native read replay differs {key}:{name}")
        if field_hashes(baseline[name]["cache"]) != field_hashes(replay["cache"]):
            raise RuntimeError(f"V35 V34-native cache replay differs {key}:{name}")
        for field in ("logits", "semantic", "broad_vocabulary", "late_hidden", "workspace"):
            if not np.array_equal(baseline[name]["targets"][field], replay["targets"][field]):
                raise RuntimeError(f"V35 endpoint drift {key}:{name}:{field}")
    groups = design["relative_depth_layers"]
    conditions = design["condition_layers"]
    tests = []
    for condition in ("Q1", "Q2", "Q3", "Q4", "Q1_Q2", "Q3_Q4", "FULL"):
        for direction, base_name, ref_name in (("REMOVE", "JOINT", "CONV"), ("RESTORE", "CONV", "JOINT")):
            mapping = {layer: ref_name for layer in conditions[condition]}
            with ReadPatch(model, key, capture, mapping) as patch:
                out = step(model, branches[base_name], probe, length + 1, bundle)
            proof = patch.proof()
            tests.append({"condition": condition, "direction": direction, "proof": proof,
                          "logits_hash": thash(out["logits"]), "cache_hashes": field_hashes(out["cache"])})
    for missing, group in cfg["leave_out_conditions"].items():
        mapping = {layer: ("CONV" if layer in groups[group] else "JOINT")
                   for layer in conditions["FULL"]}
        with ReadPatch(model, key, capture, mapping) as patch:
            out = step(model, conv, probe, length + 1, bundle)
        tests.append({"condition": missing, "direction": "FULL_MINUS", "proof": patch.proof(),
                      "logits_hash": thash(out["logits"]), "cache_hashes": field_hashes(out["cache"])})
    for a, b in cfg["serial_pairs"]:
        for early, late in (("JOINT", "CONV"), ("CONV", "JOINT")):
            mapping = {**{layer: early for layer in groups[a]},
                       **{layer: late for layer in groups[b]}}
            with ReadPatch(model, key, capture, mapping) as patch:
                out = step(model, conv, probe, length + 1, bundle)
            tests.append({"condition": f"SERIAL_{a}_{b}_{early}_{late}", "direction": "HYBRID",
                          "proof": patch.proof(), "logits_hash": thash(out["logits"]),
                          "cache_hashes": field_hashes(out["cache"])})
    result = {"model_key": key, "state_id": item["base_trial_id"], "probe_id": probe,
              "V34_exact_recurrent_read_interface_reused": True,
              "bitwise_baseline_replay": True, "recurrent_layers": len(conditions["FULL"]),
              "native_swap_proofs": {"CONV": conv_proof, "JOINT": joint_proof},
              "branch_cache_hashes": {name: field_hashes(cache) for name, cache in branches.items()},
              "group_layers": groups, "group_hash": hd(groups),
              "condition_layers_hash": hd(conditions),
              "tests": tests, "all_writeback_exact": all(x["proof"]["exact_writeback"] for x in tests),
              "formal_development_or_validation_observed": False, "calibration_only": True}
    path = root / OUT / f"interface_audit_{key}_v35.json"
    write_json_atomic(path, result)
    seal = stage_freeze(root, f"interface_{key}",
                        [SOURCE, "src/jclosure/experiments/read_hooks_v35.py", str(path.relative_to(root)),
                         "artifacts/hierarchical_read_v35_design.freeze.json"],
                        {"model_key": key, "audit_hash": hd(result), "bitwise_baseline_replay": True,
                         "all_writebacks_exact": True, "formal_development_or_validation_observed": False})
    return {"freeze_digest": seal["freeze_digest"], "model": key,
            "recurrent_layers": result["recurrent_layers"], "interventions": len(tests)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=("Q", "F"))
    args = parser.parse_args()
    print(json.dumps(run(Path.cwd(), args.model), indent=2))
