"""Response-blind paired V36 forks, endpoints, third token and fixed local layers."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from jclosure.experiments.design_v31 import usable
from jclosure.experiments.design_v34 import prompt_lookup
from jclosure.experiments.runtime_v34 import field_hashes, hd, load, prefix, step
from jclosure.protocol_v36 import stage_freeze, verify, verify_stage
from jclosure.provenance import write_json_atomic

OUT = Path("results/v36/processed")
ROLES = ("calibration", "development", "validation", "independent_final")
SOURCE = "src/jclosure/experiments/design_v36.py"


@torch.no_grad()
def run(root: Path, key: str):
    verify_stage(root, "panel")
    base = verify(root)
    cfg, spec = base["config"], base["model_specs"][key]
    panel = json.loads((root / OUT / "panel_v36.json").read_text())
    lookup = prompt_lookup(root, panel)
    prior = json.loads((root / spec["candidate_library_source"]).read_text())
    library = prior["token_library"]
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
    recurrent = spec["recurrent_layers"]
    groups = {name: [recurrent[i] for i in positions] for name, positions in cfg["relative_depth_groups"].items()}
    local_layers = [recurrent[i] for i in cfg["local_representative_positions"]]
    if len(set(recurrent)) != 24 or len(set(local_layers)) != 4:
        raise RuntimeError(f"V36 recurrent mapping drift {key}")
    roles = {role: [] for role in ROLES}
    clean_rows = []
    for role in ROLES:
        for n, item in enumerate(panel[role], 1):
            sid = item["base_trial_id"]
            prompt = lookup[sid]
            if hashlib.sha256(prompt.encode()).hexdigest() != item["prompt_sha256"]:
                raise RuntimeError(f"V36 prompt drift {sid}")
            incoming, length, ids, logits = prefix(model, tokenizer, key, prompt)
            if length > 512:
                raise RuntimeError(f"V36 context too long {sid}")
            ranked = torch.topk(logits, k=cfg["token_selection"]["rank_max"]).indices.tolist()
            rank = {int(token): i + 1 for i, token in enumerate(ranked)}
            anchor = spec["anchor_token_id"]
            if anchor not in rank:
                raise RuntimeError(f"V36 anchor ineligible {key}:{sid}")
            eligible = [x for x in library if int(x["token_id"]) in rank and int(x["token_id"]) != anchor]
            if not eligible:
                raise RuntimeError(f"V36 no natural donor token {key}:{sid}")
            ordered = sorted(eligible, key=lambda x: hd([cfg["seed"], key, sid, x["token_id"]]))
            donor = ordered[0]
            third = next((x for x in ordered[1:] if int(x["token_id"]) != int(donor["token_id"])), None)
            if third is None:
                raise RuntimeError(f"V36 no distinct third natural token {key}:{sid}")
            probes = [int(token) for token in ranked if usable(tokenizer.decode([int(token)]))][:6]
            if len(probes) != 6:
                raise RuntimeError("V36 probe shortage")
            hashes = field_hashes(incoming)
            roles[role].append({**item, "prefix_token_hash": hd(ids), "fork_total_length": length,
                                "incoming_state_hashes": hashes, "incoming_state_hash": hd(hashes),
                                "ranked_token_hash": hd(ranked), "recipient_token_id": anchor,
                                "donor_token_id": int(donor["token_id"]),
                                "third_token_id": int(third["token_id"]),
                                "third_token_category": third["category"],
                                "donor_token_category": donor["category"],
                                "recipient_rank": rank[anchor], "donor_rank": rank[int(donor["token_id"])],
                                "token_pair_hash": hd([anchor, int(donor["token_id"])]),
                                "token_triple_hash": hd([anchor, int(donor["token_id"]), int(third["token_id"])]),
                                "eligible_count": len(eligible), "future_probe_tokens": probes,
                                "future_probe_hash": hd(probes)})
            if role == "calibration":
                clean = step(model, incoming, anchor, length)["cache"]
                for probe in probes:
                    clean_rows.append(step(model, clean, probe, length + 1, bundle)["targets"])
            if n % 10 == 0 or n == len(panel[role]):
                print(f"V36 design {key} {role} {n}/{len(panel[role])}", flush=True)
    scales = {}
    for block in ("logits", "semantic", "broad_vocabulary", "late_hidden", "workspace"):
        matrix = np.stack([row[block] for row in clean_rows]).astype(np.float64)
        scales[block] = max(float(np.sqrt(np.mean((matrix - matrix.mean(axis=0)) ** 2))), 1e-6)
    result = {"model_key": key, "model_id": spec["id"], "model_revision": spec["revision"],
              **roles, "anchor_token_id": spec["anchor_token_id"],
              "token_library_source": spec["candidate_library_source"],
              "token_library_hash": hd(library), "target_bundle": bundle,
              "calibration_clean_scales": scales, "role_hashes": panel["role_hashes"],
              "recurrent_layers": recurrent, "relative_depth_layers": groups,
              "local_layers": local_layers,
              "relative_depth_group_hash": hd(groups), "local_layer_hash": hd(local_layers),
              "same_semantic_state_ids_as_other_model": True,
              "current_token_write_geometry_used_for_selection": False,
              "future_causal_response_observed_before_freeze": False,
              "historical_final_reopened": False}
    path = root / OUT / f"design_{key}_v36.json"
    write_json_atomic(path, result)
    seal = stage_freeze(root, f"design_{key}", [SOURCE, str(path.relative_to(root)),
                                              "artifacts/computational_origin_v36_panel.freeze.json"],
                        {"model_key": key, "design_hash": hd(result),
                         "role_hashes": panel["role_hashes"],
                         "local_layer_hash": result["local_layer_hash"],
                         "future_causal_response_observed_before_freeze": False})
    return {"freeze_digest": seal["freeze_digest"], "model": key,
            "scales": scales, "roles": {role: len(roles[role]) for role in ROLES}}


def finalize(root: Path):
    for key in ("Q", "F"):
        verify_stage(root, f"design_{key}")
    cfg = verify(root)["config"]
    q = json.loads((root / OUT / "design_Q_v36.json").read_text())
    f = json.loads((root / OUT / "design_F_v36.json").read_text())
    for role in ROLES:
        if [x["base_trial_id"] for x in q[role]] != [x["base_trial_id"] for x in f[role]]:
            raise RuntimeError("V36 semantic pairing drift")
    plan = {"models": ["Q", "F"], "roles": {role: [x["base_trial_id"] for x in q[role]] for role in ROLES},
            "factorial": cfg["factorial"], "primary_KV": cfg["primary_KV"],
            "estimands": {"D": "Ydonor-Y00", "C": "Y01-Y00", "R": "Y10-Y00",
                           "RC": "Y11-Y00", "E_conv": "D-C", "R_given_C": "Y11-Y01",
                           "B_REC": "||D-C||-||D-RC||", "rho_REC": "B_REC/||D-C||"},
            "shared_endpoint_definition": cfg["shared_endpoint"],
            "groups_Q": q["relative_depth_layers"], "groups_F": f["relative_depth_layers"],
            "local_layers_Q": q["local_layers"], "local_layers_F": f["local_layers"],
            "operator_comparison": cfg["operator_comparison"],
            "local_factorial": cfg["local_factorial"],
            "gates": {name: value for name, value in cfg.items() if name.endswith("_gate")},
            "final_rule": cfg["final_rule"],
            "thresholds_retuned": False, "causal_response_observed_before_freeze": False}
    path = root / OUT / "execution_plan_v36.json"
    write_json_atomic(path, plan)
    seal = stage_freeze(root, "design", [SOURCE, str(path.relative_to(root)),
                                         "artifacts/computational_origin_v36_design_Q.freeze.json",
                                         "artifacts/computational_origin_v36_design_F.freeze.json"],
                        {"plan_hash": hd(plan), "causal_response_observed_before_freeze": False})
    return {"freeze_digest": seal["freeze_digest"], "paired_states": sum(len(plan["roles"][r]) for r in ROLES),
            "local_layers_per_model": 4}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=("Q", "F", "finalize"))
    args = parser.parse_args()
    print(json.dumps(finalize(Path.cwd()) if args.model == "finalize" else run(Path.cwd(), args.model), indent=2))
