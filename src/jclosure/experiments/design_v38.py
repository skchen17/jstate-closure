"""Response-blind V38 token forks, probes, trajectory sites, and execution plan."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from jclosure.experiments.design_v31 import usable
from jclosure.experiments.runtime_v34 import field_hashes, hd, load, prefix, step
from jclosure.protocol_v34 import verify as verify_v34
from jclosure.protocol_v38 import stage_freeze, verify, verify_stage
from jclosure.provenance import write_json_atomic


OUT = Path("results/v38/processed")
SOURCE = "src/jclosure/experiments/design_v38.py"
ROLES = ("calibration", "development", "validation", "independent_final")
BLOCKS = ("logits", "semantic", "broad_vocabulary", "late_hidden", "workspace")


@torch.no_grad()
def run(root: Path, key: str) -> dict:
    verify_stage(root, "panel")
    cfg = verify(root)["config"]
    spec = verify_v34(root)["config"]["models"][key]
    panel = json.loads((root / OUT / "panel_v38.json").read_text())
    library = json.loads((root / spec["candidate_library_source"]).read_text())["token_library"]
    model, tokenizer = load(root, key)
    rng = np.random.default_rng(cfg["seed"] + (0 if key == "Q" else 1))
    vocab = int(model.get_output_embeddings().weight.shape[0])
    hidden = int(model.config.hidden_size)
    bundle = {"selected_logits": rng.choice(vocab, 64, replace=False).tolist(),
              "broad_logits": rng.choice(vocab, 64, replace=False).tolist(),
              "broad_sign": rng.choice([-1, 1], 64).tolist(),
              "late_hidden_indices": rng.choice(hidden, 64, replace=False).tolist(),
              "workspace_layers": spec["workspace_layers"], "workspace_per_layer": 32,
              "late_layer": spec["late_layer"],
              "J_analogue": "SECONDARY_ONLY_Q" if key == "Q" else "J_NOT_COMPARABLE"}
    recurrent = list(spec["recurrent_layers"])
    if len(recurrent) != 24 or len(set(recurrent)) != 24:
        raise RuntimeError(f"Recurrent topology drift: {key}")
    groups = {name: [recurrent[int(i)] for i in positions]
              for name, positions in cfg["relative_depth_groups"].items()}
    roles = {role: [] for role in ROLES}
    clean_rows = []
    for role in ROLES:
        for n, item in enumerate(panel[role], 1):
            sid, prompt = item["base_trial_id"], item["prompt"]
            if hashlib.sha256(prompt.encode()).hexdigest() != item["prompt_sha256"]:
                raise RuntimeError(f"Prompt drift: {sid}")
            incoming, length, ids, logits = prefix(model, tokenizer, key, prompt)
            if length > 512:
                raise RuntimeError(f"Prompt too long: {sid}")
            ranked = torch.topk(logits, k=cfg["token_selection"]["rank_max"]).indices.tolist()
            rank = {int(token): i + 1 for i, token in enumerate(ranked)}
            anchor = int(spec["anchor_token_id"])
            if anchor not in rank:
                raise RuntimeError(f"Anchor ineligible: {key}:{sid}")
            eligible = [row for row in library if int(row["token_id"]) in rank
                        and int(row["token_id"]) != anchor]
            eligible.sort(key=lambda row: hd([cfg["seed"], key, sid, row["token_id"]]))
            if len(eligible) < 2:
                raise RuntimeError(f"Natural donor shortage: {key}:{sid}")
            donor, third = eligible[:2]
            probes = [int(token) for token in ranked
                      if usable(tokenizer.decode([int(token)]))][:cfg["token_selection"]["future_probes"]]
            if len(probes) != 6:
                raise RuntimeError(f"Probe shortage: {key}:{sid}")
            hashes = field_hashes(incoming)
            roles[role].append({**item, "prefix_token_hash": hd(ids), "fork_total_length": length,
                                "incoming_state_hashes": hashes, "incoming_state_hash": hd(hashes),
                                "ranked_token_hash": hd(ranked), "recipient_token_id": anchor,
                                "donor_token_id": int(donor["token_id"]),
                                "third_token_id": int(third["token_id"]),
                                "donor_token_category": donor["category"],
                                "third_token_category": third["category"],
                                "recipient_rank": rank[anchor],
                                "donor_rank": rank[int(donor["token_id"])],
                                "token_pair_hash": hd([anchor, int(donor["token_id"])]),
                                "token_triple_hash": hd([anchor, int(donor["token_id"]), int(third["token_id"])]),
                                "eligible_count": len(eligible),
                                "future_probe_tokens": probes, "future_probe_hash": hd(probes)})
            if role == "calibration":
                clean = step(model, incoming, anchor, length)["cache"]
                clean_rows.extend(step(model, clean, probe, length + 1, bundle)["targets"]
                                  for probe in probes)
            if n % 10 == 0 or n == len(panel[role]):
                print(f"V38 design {key} {role} {n}/{len(panel[role])}", flush=True)
    scales = {}
    for block in BLOCKS:
        matrix = np.stack([row[block] for row in clean_rows]).astype(np.float64)
        scales[block] = max(float(np.sqrt(np.mean((matrix - matrix.mean(axis=0)) ** 2))), 1e-6)
    result = {"model_key": key, "model_id": spec["id"], "model_revision": spec["revision"],
              **roles, "target_bundle": bundle, "calibration_clean_scales": scales,
              "anchor_token_id": anchor, "token_library_source": spec["candidate_library_source"],
              "token_library_hash": hd(library), "recurrent_layers": recurrent,
              "relative_depth_layers": groups, "role_hashes": panel["role_hashes"],
              "same_semantic_state_ids_as_other_model": True,
              "future_causal_response_observed_before_freeze": False}
    path = root / OUT / f"design_{key}_v38.json"
    write_json_atomic(path, result)
    seal = stage_freeze(root, f"design_{key}", [SOURCE, str(path.relative_to(root)),
                                              "artifacts/trajectory_composition_v38_panel.freeze.json"],
                        {"model_key": key, "design_hash": hd(result),
                         "role_hashes": result["role_hashes"],
                         "future_causal_response_observed_before_freeze": False})
    return {"freeze_digest": seal["freeze_digest"], "model": key,
            "roles": {role: len(roles[role]) for role in ROLES}}


def finalize(root: Path) -> dict:
    for key in ("Q", "F"):
        verify_stage(root, f"design_{key}")
    cfg = verify(root)["config"]
    designs = {key: json.loads((root / OUT / f"design_{key}_v38.json").read_text())
               for key in ("Q", "F")}
    for role in ROLES:
        if [x["base_trial_id"] for x in designs["Q"][role]] != [x["base_trial_id"] for x in designs["F"][role]]:
            raise RuntimeError(f"Cross-model pairing drift: {role}")
    subsets = {}
    for role in ("development", "validation"):
        subsets[role] = {family: [x["base_trial_id"] for x in designs["Q"][role]
                                  if x["family"] == family][:cfg["task_phase_b_per_family"][role]]
                         for family in cfg["families"]}
    plan = {"roles": {role: [x["base_trial_id"] for x in designs["Q"][role]] for role in ROLES},
            "groups": {key: designs[key]["relative_depth_layers"] for key in ("Q", "F")},
            "conditions": cfg["conditions"], "stage_order": cfg["stage_order"],
            "formulas": cfg["primary_interaction_formulas"],
            "gates": {name: val for name, val in cfg.items() if name.endswith("_gate")},
            "finalist_priority": cfg["finalist_priority"],
            "denominator_floor": cfg["benefit_denominator_absolute_floor"],
            "independent_final_rule": cfg["independent_final_rule"],
            "task_subsets": subsets,
            "response_observed_before_freeze": False}
    path = root / OUT / "execution_plan_v38.json"
    write_json_atomic(path, plan)
    seal = stage_freeze(root, "design", [SOURCE, str(path.relative_to(root)),
                                         *[f"artifacts/trajectory_composition_v38_design_{key}.freeze.json"
                                           for key in ("Q", "F")]],
                        {"plan_hash": hd(plan), "response_observed_before_freeze": False})
    return {"freeze_digest": seal["freeze_digest"], "paired_states": sum(len(v) for v in plan["roles"].values())}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=("Q", "F", "finalize"))
    args = parser.parse_args()
    print(json.dumps(finalize(Path.cwd()) if args.model == "finalize" else run(Path.cwd(), args.model), indent=2))
