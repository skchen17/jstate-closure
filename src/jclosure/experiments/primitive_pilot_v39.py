"""Calibration-only primitive removal screen; no formal dev/validation outcomes."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from jclosure.experiments.depth_common_v35 import prepare
from jclosure.experiments.functional_hooks_v39 import FunctionalIntervention
from jclosure.experiments.natural_reconstruction_v37 import _selective_rec
from jclosure.experiments.primitive_hook_v39 import FalconPrimitiveHook
from jclosure.experiments.runtime_v34 import load, signature, step
from jclosure.protocol_v39 import stage_freeze, verify, verify_stage
from jclosure.provenance import sha256_file, write_json_atomic


OUT = Path("results/v39/processed")
SOURCE = "src/jclosure/experiments/primitive_pilot_v39.py"
ORDER = ("R000", "R100", "R010", "R001", "R110", "R101", "R011", "R111")
FACTOR_MAP = {
    "F1_POSTCONV_READ_FACTOR_C": {"C": "C"},
    "F2_POSTCONV_UPDATE_FACTORS_X_B": {"X": "X", "B": "B"},
    "F3_DECAY_CONTROL_DT_DA": {"DT": "TRANSFORMED_CONTROL"},
    "F4_RECURRENT_STATE_INPUT_S": {"S": "S"},
    "F5_RECURRENT_UPDATE_DBX": {"DBX": "TRUE_UPDATE"},
    "F6_OUTPUT_GATE": {"OUTPUT_GATE": "OUTPUT_GATE"},
    "F7_COMBINED_RECURRENCE_INPUTS": {"X": "X", "B": "B", "C": "C",
                                      "DT": "TRANSFORMED_CONTROL", "S": "S"},
    "F8_RESIDUAL_INPUT_CONTEXT": {"MIXER_INPUT": "MIXER_INPUT"},
}


def second(factors: dict, name: str) -> tuple[torch.Tensor, float]:
    terms = [factors[label][name].to(torch.float32) for label in ORDER]
    x0, x1, x2, x3, x12, x13, x23, x123 = terms
    raw = x12 + x13 + x23 - x1 - x2 - x3 + x0
    result = raw.to(factors["R111"][name].dtype)
    gap = float(torch.linalg.vector_norm(raw-result.to(torch.float32)).item())
    return result, gap


@torch.no_grad()
def run(root: Path) -> dict:
    verify_stage(root, "candidate_site_plan")
    verify_stage(root, "primitive_interface_audit_F")
    cfg = verify(root)["config"]
    plan = json.loads((root / OUT / "candidate_site_plan_v39.json").read_text())
    design = json.loads((root / OUT / "design_F_v39.json").read_text())
    groups = design["relative_depth_layers"]
    sites = {family: int(value["layer"]) for family, value in plan["sites"].items()
             if value["eligible"]}
    if not sites:
        raise RuntimeError("No trace-qualified candidate sites")
    unique_sites = sorted(set(sites.values()))
    model, tokenizer = load(root, "F")
    rows, vectors = [], []
    for n, item in enumerate(design["calibration"], 1):
        caches = prepare(model, tokenizer, "F", item, item["prompt"])
        probe = item["future_probe_tokens"][0]
        ys, factors = {}, {}
        for label in ORDER:
            chosen = [li for group in cfg["conditions"][label] for li in groups[group]]
            cache, _ = (_selective_rec(caches["conv"], caches["joint"], chosen)
                        if chosen else (caches["conv"], []))
            with FunctionalIntervention(model, "F") as hook:
                observed = step(model, cache, probe, caches["length"]+1,
                                design["target_bundle"])
            ys[label] = signature([observed], design["calibration_clean_scales"])
            factors[label] = {li: {name: value.detach().clone() for name, value in hook.capture[li].items()
                                   if isinstance(value, torch.Tensor)} for li in unique_sites}
        x0, x1, x2, x3, x12, x13, x23, x123 = (ys[label] for label in ORDER)
        original = x123-x12-x13-x23+x1+x2+x3-x0
        denom = max(float(np.dot(original, original)), 1e-12)
        cache111, _ = _selective_rec(caches["conv"], caches["joint"],
                                     groups["Q2"]+groups["Q3"]+groups["Q4"])
        for family in cfg["candidate_families_order"]:
            if family not in sites:
                continue
            li = sites[family]
            mapping = FACTOR_MAP[family]
            replacements, gaps = {}, {}
            for target, source in mapping.items():
                selected = {label: {source: factors[label][li][source]} for label in ORDER}
                replacements[target], gaps[target] = second(selected, source)
            with FalconPrimitiveHook(model, li, replacements) as intervention:
                modified = step(model, cache111, probe, caches["length"]+1,
                                design["target_bundle"])
            if not all(intervention.capture[k+"_requested_hash"] ==
                       intervention.capture[k+"_realized_hash"] for k in replacements):
                raise RuntimeError(f"Pilot primitive writeback mismatch: {family}:{item['base_trial_id']}")
            changed = signature([modified], design["calibration_clean_scales"])
            after = original + changed - x123
            removed_direction = float(np.dot(original-after, original)/denom)
            remaining_ratio = float(np.linalg.norm(after)/max(np.linalg.norm(original), 1e-12))
            rows.append({"state_id": item["base_trial_id"], "family": item["family"],
                         "candidate": family, "layer": li, "probe_index": 0,
                         "removed_direction": removed_direction,
                         "remaining_norm_ratio": remaining_ratio,
                         "original_norm": float(np.linalg.norm(original)),
                         "after_norm": float(np.linalg.norm(after)),
                         "quantization_gap_json": json.dumps(gaps, sort_keys=True),
                         "requested_realized_exact": True,
                         "native_descendants_recomputed": True,
                         "future_output_copy": False,
                         "vector_index": len(vectors)})
            vectors.append(np.stack([original, after]).astype(np.float32))
        print(f"V39 primitive pilot {n}/{len(design['calibration'])}", flush=True)
    frame = pd.DataFrame(rows)
    table = root / OUT / "primitive_pilot_F_v39.parquet"
    archive = root / OUT / "primitive_pilot_vectors_F_v39.npz"
    frame.to_parquet(table, index=False, compression="zstd")
    np.savez_compressed(archive, vectors=np.stack(vectors),
                        state_ids=frame.state_id.to_numpy(str),
                        candidates=frame.candidate.to_numpy(str))
    gate = cfg["necessity_gate"]
    screening = {}
    for family, data in frame.groupby("candidate"):
        by_family = {f: {"median_removed_direction": float(g.removed_direction.median()),
                         "median_remaining_norm_ratio": float(g.remaining_norm_ratio.median())}
                     for f, g in data.groupby("family")}
        family_count = sum(v["median_removed_direction"] >= gate["median_removed_direction_min"]
                           and v["median_remaining_norm_ratio"] <= gate["median_remaining_norm_ratio_max"]
                           for v in by_family.values())
        med_dir = float(data.removed_direction.median())
        med_ratio = float(data.remaining_norm_ratio.median())
        screening[family] = {"layer": sites[family], "pilot_states": len(data),
                             "median_removed_direction": med_dir,
                             "median_remaining_norm_ratio": med_ratio,
                             "family_metrics": by_family, "families_passing": family_count,
                             "screen_pass": med_dir >= gate["median_removed_direction_min"] and
                             med_ratio <= gate["median_remaining_norm_ratio_max"] and
                             family_count >= gate["families_required"]}
    summary = {"role": "calibration_pilot", "formal_intervention_outcomes_used": False,
               "screening": screening,
               "files_sha256": {"rows": sha256_file(table), "vectors": sha256_file(archive)}}
    path = root / OUT / "primitive_pilot_F_v39.json"
    write_json_atomic(path, summary)
    seal = stage_freeze(root, "primitive_pilot_F",
                        [SOURCE, str(path.relative_to(root)), str(table.relative_to(root)),
                         str(archive.relative_to(root)),
                         "artifacts/interaction_genesis_v39_candidate_site_plan.freeze.json",
                         "artifacts/interaction_genesis_v39_primitive_interface_audit_F.freeze.json"],
                        {"calibration_only": True, "summary_sha256": sha256_file(path)})
    return {"freeze_digest": seal["freeze_digest"], **summary}


if __name__ == "__main__":
    print(json.dumps(run(Path.cwd()), indent=2))
