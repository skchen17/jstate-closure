"""Exploratory exact individual-component mediation on a frozen validation subset."""
from __future__ import annotations

import argparse
import json
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import jclosure.experiments.functional_hooks_v34 as hooks
from jclosure.experiments.design_v34 import prompt_lookup
from jclosure.experiments.functional_hooks_v34 import FunctionalIntervention
from jclosure.experiments.mediate_v34 import cosine
from jclosure.experiments.runtime_v34 import load, native_swap, prefix, signature, step
from jclosure.experiments.transaction_v28 import thash
from jclosure.protocol_v34 import stage_freeze, verify_stage
from jclosure.provenance import sha256_file, write_json_atomic

OUT = Path("results/v34/processed")
SOURCE = "src/jclosure/experiments/component_followup_v34.py"
COMPONENTS = {"Q": [("POSTCONV_INPUT", "q", 0), ("POSTCONV_INPUT", "k", 1),
                    ("POSTCONV_INPUT", "v", 2), ("TRANSFORMED_CONTROL", "g", 0),
                    ("TRANSFORMED_CONTROL", "beta", 1)],
              "F": [("POSTCONV_INPUT", "x", 0), ("POSTCONV_INPUT", "B", 1),
                    ("POSTCONV_INPUT", "C", 2)]}


@contextmanager
def component_scope(key, stage, index, widths=None):
    """Only the named native component is substituted; other components evolve naturally."""
    if key == "Q":
        original = hooks._replacement_component

        def replace(reference, current, record, name):
            if name == f"{stage}_{index}":
                return original(reference, current, record, name)
            return current

        hooks._replacement_component = replace
        try:
            yield
        finally:
            hooks._replacement_component = original
    else:
        if stage != "POSTCONV_INPUT" or widths is None:
            raise RuntimeError("V34 Falcon component topology unavailable")
        original = hooks._replacement

        def replace(reference, layer, requested_stage, current, record):
            if requested_stage != stage:
                return original(reference, layer, requested_stage, current, record)
            ref = reference[layer][stage]
            if ref.shape != current.shape or ref.dtype != current.dtype:
                raise RuntimeError("V34 Falcon component topology mismatch")
            start = sum(widths[:index]); stop = start + widths[index]
            realized = current.detach().clone()
            realized[..., start:stop] = ref[..., start:stop]
            if not torch.equal(realized[..., start:stop], ref[..., start:stop]):
                raise RuntimeError("V34 Falcon component exact write failed")
            record["requested_hash"] = thash(ref[..., start:stop])
            record["realized_hash"] = thash(realized[..., start:stop])
            record["native_hash"] = thash(current[..., start:stop])
            record["exact_writeback"] = True
            return realized

        hooks._replacement = replace
        try:
            yield
        finally:
            hooks._replacement = original


@torch.no_grad()
def run(root: Path, key: str) -> dict:
    verify_stage(root, "stage_analysis_validation")
    plan = json.loads((root / OUT / "mediation_plan_v34.json").read_text())
    design = json.loads((root / OUT / f"design_{key}_v34.json").read_text())
    panel = json.loads((root / OUT / "panel_v34.json").read_text())
    lookup = prompt_lookup(root, panel)
    subset = {sid for values in plan["KV_subset"]["validation"].values() for sid in values}
    if len(subset) != 5:
        raise RuntimeError("V34 component subset drift")
    factorial = pd.read_parquet(root / OUT / f"factorial_{key}_validation_v34.parquet")
    vectors = np.load(root / OUT / f"factorial_vectors_{key}_validation_v34.npz")["vectors"].astype(np.float64)
    by_state = {x.state_id: x for x in factorial.itertuples()}
    model, tokenizer = load(root, key)
    widths = None
    if key == "F":
        mixer = model.model.layers[0].mamba
        widths = [mixer.intermediate_size, mixer.n_groups*mixer.ssm_state_size,
                  mixer.n_groups*mixer.ssm_state_size]
    rows = []
    for count, item in enumerate((x for x in design["validation"] if x["base_trial_id"] in subset), 1):
        sid = item["base_trial_id"]
        incoming, length, _, _ = prefix(model, tokenizer, key, lookup[sid])
        recipient = step(model, incoming, item["recipient_token_id"], length)["cache"]
        donor = step(model, incoming, item["donor_token_id"], length)["cache"]
        conv, _ = native_swap(recipient, donor, ["Conv"], key)
        joint, _ = native_swap(recipient, donor, ["REC", "Conv"], key)
        probes = item["future_probe_tokens"]
        bundle, scales = design["target_bundle"], design["calibration_clean_scales"]
        ref = {"CONV": [], "JOINT": []}
        for probe in probes:
            for label, cache in (("CONV", conv), ("JOINT", joint)):
                with FunctionalIntervention(model, key) as instrument:
                    step(model, cache, probe, length+1, bundle)
                ref[label].append(instrument.capture)
        yconv, yjoint, yd = vectors[int(by_state[sid].vector_index)][[2,3,4]]
        ec, ej = float(np.linalg.norm(yd-yconv)), float(np.linalg.norm(yd-yjoint))
        benefit = ec-ej
        if benefit <= 0:
            raise RuntimeError("V34 component control state has nonpositive benefit")
        for stage, name, index in COMPONENTS[key]:
            outcomes = {"REMOVE": [], "RESTORE": []}
            for i, probe in enumerate(probes):
                for direction, base, reference in (("REMOVE", joint, ref["CONV"][i]),
                                                   ("RESTORE", conv, ref["JOINT"][i])):
                    with component_scope(key, stage, index, widths):
                        with FunctionalIntervention(model, key, stage, reference) as instrument:
                            output = step(model, base, probe, length+1, bundle)
                    expected = f"{stage}_{index}_requested_hash" if key == "Q" else "requested_hash"
                    realized = f"{stage}_{index}_realized_hash" if key == "Q" else "realized_hash"
                    if len(instrument.capture) != 24 or not all(x.get(expected) == x.get(realized) and x.get(expected) for x in instrument.capture.values()):
                        raise RuntimeError("V34 component exact write audit failed")
                    outcomes[direction].append(output)
            remove = signature(outcomes["REMOVE"], scales)
            restore = signature(outcomes["RESTORE"], scales)
            rows.append({"model_key": key, "state_id": sid, "family": item["family"],
                         "stage": stage, "component": name, "exact_writeback": True,
                         "removed_fraction": (float(np.linalg.norm(yd-remove))-ej)/benefit,
                         "restored_fraction": (ec-float(np.linalg.norm(yd-restore)))/benefit,
                         "reverse_cosine": cosine(restore-yconv,yjoint-yconv)})
        print(f"V34 individual components {key} {count}/{len(subset)}", flush=True)
    frame = pd.DataFrame(rows)
    path = root / OUT / f"components_{key}_validation_v34.parquet"
    frame.to_parquet(path, index=False, compression="zstd")
    summary = {"model_key": key, "states": len(subset), "exploratory_components": True,
               "Falcon_dt_decay_coupled_not_separately_varied": key == "F",
               "components": {f"{stage}:{component}": {"median_removed": float(part.removed_fraction.median()),
                                                    "median_restored": float(part.restored_fraction.median()),
                                                    "median_cosine": float(part.reverse_cosine.median())}
                              for (stage,component),part in frame.groupby(["stage","component"])},
               "data_sha256": sha256_file(path)}
    summary_path = root / OUT / f"components_{key}_validation_v34.json"
    write_json_atomic(summary_path, summary)
    seal = stage_freeze(root, f"components_{key}_validation",
                        [SOURCE, str(path.relative_to(root)), str(summary_path.relative_to(root)),
                         "artifacts/functional_mediation_v34_mediation_plan.freeze.json",
                         "artifacts/functional_mediation_v34_stage_analysis_validation.freeze.json"],
                        {"model_key": key, "summary_sha256": sha256_file(summary_path)})
    return {"freeze_digest": seal["freeze_digest"], **summary}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=("Q","F"))
    args = parser.parse_args()
    print(json.dumps(run(Path.cwd(), args.model), indent=2))
