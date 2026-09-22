"""Post-localization read-stage relative-depth and h2/h4 propagation analyses."""
from __future__ import annotations

import argparse
import json
from contextlib import AbstractContextManager
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from jclosure.experiments.design_v34 import prompt_lookup
from jclosure.experiments.functional_hooks_v34 import FunctionalIntervention
from jclosure.experiments.mediate_v34 import cosine
from jclosure.experiments.runtime_v34 import load, native_swap, prefix, signature, step
from jclosure.experiments.transaction_v28 import thash
from jclosure.protocol_v34 import stage_freeze, verify, verify_stage
from jclosure.provenance import sha256_file, write_json_atomic

OUT = Path("results/v34/processed")
SOURCE = "src/jclosure/experiments/read_followup_v34.py"


class SelectedReadPatch(AbstractContextManager):
    """Exact mixer-output patch on a predeclared relative-depth layer group."""

    def __init__(self, model, key, reference, selected_layers):
        self.model, self.key, self.reference = model, key, reference
        self.selected_layers = set(selected_layers)
        self.handles = []
        self.audit = {}

    def __enter__(self):
        for layer in sorted(self.selected_layers):
            block = self.model.model.layers[layer]
            mixer = block.linear_attn if self.key == "Q" else block.mamba
            requested = self.reference[layer]["RECURRENT_READ"]

            def hook(_module, _input, output, layer=layer, requested=requested):
                if requested.shape != output.shape or requested.dtype != output.dtype:
                    raise RuntimeError(f"V34 depth topology mismatch {layer}")
                realized = requested.detach().clone()
                if not torch.equal(requested, realized):
                    raise RuntimeError("V34 depth exact write failed")
                self.audit[layer] = {"requested_hash": thash(requested),
                                     "realized_hash": thash(realized),
                                     "native_hash": thash(output), "exact_writeback": True}
                return realized

            self.handles.append(mixer.register_forward_hook(hook))
        return self

    def __exit__(self, exc_type, exc, tb):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        return False


def _capture(model, key, cache, token, position, bundle):
    with FunctionalIntervention(model, key) as hook:
        out = step(model, cache, token, position, bundle)
    return out, hook.capture


def _trajectory(model, key, cache, tokens, start_length, bundle, reference=None):
    outputs = []
    current = cache
    for index, token in enumerate(tokens, 1):
        if index == 1 and reference is not None:
            with FunctionalIntervention(model, key, "RECURRENT_READ", reference) as hook:
                result = step(model, current, token, start_length + index, bundle)
            if len(hook.capture) != 24 or not all(x.get("exact_writeback") for x in hook.capture.values()):
                raise RuntimeError("V34 horizon exact write failed")
        else:
            result = step(model, current, token, start_length + index, bundle)
        current = result["cache"]
        outputs.append(result)
    return outputs


@torch.no_grad()
def run(root: Path, key: str, mode: str) -> dict:
    if mode not in ("depth", "horizons"):
        raise ValueError(mode)
    verify_stage(root, "stage_analysis_validation")
    analysis = json.loads((root / OUT / "stage_analysis_validation_v34.json").read_text())
    if "RECURRENT_READ" not in analysis["shared_stage_passes_development_and_validation"]:
        raise RuntimeError("V34 read follow-up not authorized")
    cfg = verify(root)["config"]
    plan = json.loads((root / OUT / "mediation_plan_v34.json").read_text())
    panel = json.loads((root / OUT / "panel_v34.json").read_text())
    design = json.loads((root / OUT / f"design_{key}_v34.json").read_text())
    subset = {sid for values in plan["KV_subset"]["validation"].values() for sid in values}
    if len(subset) != 5:
        raise RuntimeError("V34 frozen one-per-family follow-up subset drift")
    model, tokenizer = load(root, key)
    lookup = prompt_lookup(root, panel)
    recurrent_layers = cfg["models"][key]["recurrent_layers"]
    groups = {f"quarter_{i+1}": recurrent_layers[6*i:6*(i+1)] for i in range(4)}
    groups["full_depth"] = recurrent_layers
    rows = []
    for count, item in enumerate((x for x in design["validation"] if x["base_trial_id"] in subset), 1):
        sid = item["base_trial_id"]
        incoming, length, _, _ = prefix(model, tokenizer, key, lookup[sid])
        recipient = step(model, incoming, item["recipient_token_id"], length)["cache"]
        donor = step(model, incoming, item["donor_token_id"], length)["cache"]
        conv, _ = native_swap(recipient, donor, ["Conv"], key)
        joint, _ = native_swap(recipient, donor, ["REC", "Conv"], key)
        bundle, scales = design["target_bundle"], design["calibration_clean_scales"]
        if mode == "depth":
            captures = []
            natural = {"CONV": [], "JOINT": [], "DONOR": []}
            patched = {(group, direction): [] for group in groups for direction in ("REMOVE", "RESTORE")}
            for probe in item["future_probe_tokens"]:
                out, conv_ref = _capture(model, key, conv, probe, length + 1, bundle)
                natural["CONV"].append(out)
                out, joint_ref = _capture(model, key, joint, probe, length + 1, bundle)
                natural["JOINT"].append(out)
                natural["DONOR"].append(step(model, donor, probe, length + 1, bundle))
                captures.append((conv_ref, joint_ref))
                for group_name, layers in groups.items():
                    for direction, base, reference in (("REMOVE", joint, conv_ref), ("RESTORE", conv, joint_ref)):
                        with SelectedReadPatch(model, key, reference, layers) as hook:
                            outcome = step(model, base, probe, length + 1, bundle)
                        if len(hook.audit) != len(layers) or not all(x["requested_hash"] == x["realized_hash"] for x in hook.audit.values()):
                            raise RuntimeError("V34 partial-depth audit failed")
                        patched[(group_name, direction)].append(outcome)
            yconv, yjoint, yd = [signature(natural[x], scales) for x in ("CONV", "JOINT", "DONOR")]
            e_conv, e_joint = float(np.linalg.norm(yd-yconv)), float(np.linalg.norm(yd-yjoint))
            benefit = e_conv-e_joint
            for group_name, layers in groups.items():
                remove, restore = (signature(patched[(group_name, x)], scales) for x in ("REMOVE", "RESTORE"))
                rows.append({"model_key": key, "state_id": sid, "family": item["family"], "group": group_name,
                             "relative_depth_start": min(recurrent_layers.index(x) for x in layers)/24,
                             "relative_depth_end": (max(recurrent_layers.index(x) for x in layers)+1)/24,
                             "layers": json.dumps(layers), "benefit_positive": benefit > 0,
                             "removed_fraction": (float(np.linalg.norm(yd-remove))-e_joint)/benefit if benefit > 1e-12 else None,
                             "restored_fraction": (e_conv-float(np.linalg.norm(yd-restore)))/benefit if benefit > 1e-12 else None,
                             "reverse_cosine": cosine(restore-yconv, yjoint-yconv), "exact_writeback": True})
        else:
            tokens = item["future_probe_tokens"][:4]
            _, conv_ref = _capture(model, key, conv, tokens[0], length+1, bundle)
            _, joint_ref = _capture(model, key, joint, tokens[0], length+1, bundle)
            paths = {"CONV": _trajectory(model,key,conv,tokens,length,bundle),
                     "JOINT": _trajectory(model,key,joint,tokens,length,bundle),
                     "DONOR": _trajectory(model,key,donor,tokens,length,bundle),
                     "REMOVE": _trajectory(model,key,joint,tokens,length,bundle,conv_ref),
                     "RESTORE": _trajectory(model,key,conv,tokens,length,bundle,joint_ref)}
            for horizon in (2,4):
                vec = {name: signature([outputs[horizon-1]], scales) for name, outputs in paths.items()}
                ec = float(np.linalg.norm(vec["DONOR"]-vec["CONV"]))
                ej = float(np.linalg.norm(vec["DONOR"]-vec["JOINT"]))
                benefit = ec-ej
                rows.append({"model_key": key, "state_id": sid, "family": item["family"], "horizon": horizon,
                             "token_sequence_hash": __import__("hashlib").sha256(json.dumps(tokens).encode()).hexdigest(),
                             "benefit_positive": benefit > 0,
                             "removed_fraction": (float(np.linalg.norm(vec["DONOR"]-vec["REMOVE"]))-ej)/benefit if benefit > 1e-12 else None,
                             "restored_fraction": (ec-float(np.linalg.norm(vec["DONOR"]-vec["RESTORE"])))/benefit if benefit > 1e-12 else None,
                             "reverse_cosine": cosine(vec["RESTORE"]-vec["CONV"],vec["JOINT"]-vec["CONV"]),
                             "h1_only_intervention": True, "later_steps_unpatched": True})
        print(f"V34 {mode} {key} {count}/{len(subset)}", flush=True)
    frame = pd.DataFrame(rows)
    path = root / OUT / f"{mode}_{key}_validation_v34.parquet"
    frame.to_parquet(path, index=False, compression="zstd")
    group = "group" if mode == "depth" else "horizon"
    summary = {"model_key": key, "role": "validation", "mode": mode, "states": len(subset),
               "groups": {str(name): {"positive": int(part.benefit_positive.sum()),
                                      "median_removed": float(part.removed_fraction.median()),
                                      "median_restored": float(part.restored_fraction.median()),
                                      "median_cosine": float(part.reverse_cosine.median())}
                          for name, part in frame.groupby(group)},
               "exploratory_small_subset": True, "data_sha256": sha256_file(path)}
    summary_path = root / OUT / f"{mode}_{key}_validation_v34.json"
    write_json_atomic(summary_path, summary)
    seal = stage_freeze(root, f"{mode}_{key}_validation",
                        [SOURCE, str(path.relative_to(root)), str(summary_path.relative_to(root)),
                         "artifacts/functional_mediation_v34_stage_analysis_validation.freeze.json",
                         "artifacts/functional_mediation_v34_mediation_plan.freeze.json"],
                        {"model_key": key, "mode": mode, "summary_sha256": sha256_file(summary_path)})
    return {"freeze_digest": seal["freeze_digest"], **summary}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=("Q", "F"))
    parser.add_argument("mode", choices=("depth", "horizons"))
    args = parser.parse_args()
    print(json.dumps(run(Path.cwd(), args.model, args.mode), indent=2))
