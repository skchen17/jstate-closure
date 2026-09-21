"""Freeze V23 state/action roles, calibrate generated probes, and freeze the nested metric."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from jclosure.experiments import action_pool_v22 as v22_actions
from jclosure.experiments import counterfactual_bank_v19 as v19
from jclosure.experiments.action_bank_v16 import _readback
from jclosure.experiments.response_operator_v20 import _action_ok
from jclosure.protocol_v23 import stage_freeze, verify, verify_stage
from jclosure.provenance import sha256_file, write_json_atomic

SOURCE = "src/jclosure/experiments/probe_design_v23.py"
OUT = Path("results/v23/processed")
SCRATCH = Path("/data/CSK/J-space-project/v23-oracle-chart-work")


def _digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _hashed(values, seed: str):
    return sorted(values, key=lambda value: hashlib.sha256(f"{seed}:{value}".encode()).hexdigest())


def _balanced(ids: list[str], specs: dict[str, dict], seed: str) -> list[str]:
    bins: dict[str, list[str]] = {}
    for value in _hashed(ids, seed):
        bins.setdefault(specs[value]["source_family"], []).append(value)
    answer = []
    while any(bins.values()):
        for family in sorted(bins):
            if bins[family]:
                answer.append(bins[family].pop(0))
    return answer


def prepare(root: Path) -> dict:
    config = verify(root)["config"]
    roles = json.loads((root / "artifacts/action_coordinate_geometry_v21_roles.freeze.json").read_text())
    pool = json.loads((root / "results/v22/processed/action_pool_v22.json").read_text())
    calibration = json.loads((root / "results/v22/processed/action_pool_calibration_v22.json").read_text())
    selection = json.loads((root / "results/v22/processed/action_selection_v22.json").read_text())
    specs = {item["action_id"]: item for item in pool["candidate_pool"]}
    reliable = {item["action_id"]: item for item in calibration["by_action"] if item["selected_alpha"] is not None}
    final_ids = set(selection["independent_final_action_ids"])
    validation_ids = set(selection["validation_action_ids"])
    eligible = sorted(set(reliable) - final_ids - validation_ids)
    common = list(selection["common_measured_train_action_ids"])
    if len(common) != 128 or not set(common) <= set(eligible):
        raise RuntimeError("V23 expected the reliable V22 common-128 panel")
    originals = _balanced(common, specs, "V23-COMMON")
    originals += _balanced(sorted(set(eligible) - set(common)), specs, "V23-EXPANSION")
    if len(originals) != 419:
        raise RuntimeError(f"V23 reliable non-final/non-validation count changed: {len(originals)}")
    sources = _hashed(originals, "V23-COMBO-SOURCES")
    combos = []
    for number in range(int(config["panels"]["generated_combo_candidates"])):
        left = specs[sources[(2 * number) % len(sources)]]
        right = specs[sources[(2 * number + 1) % len(sources)]]
        sign = -1.0 if number % 2 else 1.0
        combos.append({
            "action_id": f"v23c{number:03d}", "kind": "balanced_combination", "direction_index": None,
            "components": [[int(i), float(w) * 2 ** -0.5] for i, w in left["components"]]
                          + [[int(i), sign * float(w) * 2 ** -0.5] for i, w in right["components"]],
            "source_family": ["REC_dominant", "Conv_dominant", "KV_dominant", "joint_REC_Conv_KV"][number % 4],
            "source_action_ids": [left["action_id"], right["action_id"]],
        })
    by_family = {}
    for item in roles["jvp_development"]:
        by_family.setdefault(item["family"], []).append(item)
    jvp_states = [item for family in sorted(by_family) for item in by_family[family][:2]]
    finite_states = [by_family[family][0] for family in sorted(by_family)]
    expanded = json.loads((root / "results/v22/processed/expanded_bank_design_v22.json").read_text())
    state_roles = {
        "jvp_rank_development": jvp_states,
        "finite_rank_development": finite_states,
        "oracle_development": expanded["development_states"],
        "oracle_validation": expanded["validation_states"],
        "persistent_states_per_base": ["P0", "Pq"],
    }
    final_specs = []
    for number in range(int(config["panels"]["independent_final_actions"])):
        left = specs[sources[(300 + 2 * number) % len(sources)]]
        right = specs[sources[(301 + 2 * number) % len(sources)]]
        final_specs.append({"action_id": f"v23final{number:02d}", "components": [left["components"], right["components"]],
                            "source_action_ids": [left["action_id"], right["action_id"]]})
    payload = {
        "state_roles": state_roles,
        "reliable_original_action_ids": originals,
        "v22_common_train_set": common,
        "heldout_validation_action_ids": selection["validation_action_ids"],
        "historical_v22_final_action_ids": selection["independent_final_action_ids"],
        "generated_combo_candidates": combos,
        "new_independent_final_specs": final_specs,
        "state_split_hash": _digest(state_roles),
        "candidate_combo_hash": _digest(combos),
        "heldout_action_hash": _digest(selection["validation_action_ids"]),
        "historical_final_hash": selection["independent_final_hash"],
        "new_independent_final_hash": _digest(final_specs),
        "response_labels_observed_before_freeze": 0,
        "historical_final_responses_opened": False,
        "v23_independent_final_responses_opened": False,
    }
    target = root / OUT / "probe_candidate_design_v23.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(target, payload)
    frozen = stage_freeze(root, "probe_candidate_design", [SOURCE,
        "results/v22/processed/action_pool_v22.json", "results/v22/processed/action_pool_calibration_v22.json",
        "results/v22/processed/action_selection_v22.json", "results/v22/processed/expanded_bank_design_v22.json",
        "artifacts/action_coordinate_geometry_v21_roles.freeze.json", str(target)],
        {key: payload[key] for key in ("state_split_hash", "candidate_combo_hash", "heldout_action_hash",
                                       "historical_final_hash", "new_independent_final_hash",
                                       "historical_final_responses_opened", "v23_independent_final_responses_opened")})
    return {"freeze_digest": frozen["freeze_digest"], "jvp_operator_states": 2 * len(jvp_states),
            "finite_operator_states": 2 * len(finite_states), **{key: payload[key] for key in (
                "state_split_hash", "candidate_combo_hash", "heldout_action_hash", "new_independent_final_hash")}}


def calibrate(root: Path) -> dict:
    verify_stage(root, "probe_candidate_design")
    design = json.loads((root / OUT / "probe_candidate_design_v23.json").read_text())
    operator = json.loads((root / "artifacts/compact_causal_response_operator_v20_operator_design.freeze.json").read_text())
    prompts = __import__("jclosure.experiments.operator_bank_v20", fromlist=["prompt_index"]).prompt_index(root)
    bundle, dense, _, values, rec, att, measured, state_layer, _, _ = v19._setup(root)
    directions = values["directions"]
    rows = []
    for base_number, item in enumerate(design["state_roles"]["finite_rank_development"], 1):
        base_id = item["base_trial_id"]
        clean = v19._prefill_history(bundle, str(prompts[base_id]["prompt"]), measured, dense, state_layer)
        p0 = clean["cache"]
        for action_number, spec in enumerate(design["generated_combo_candidates"], 1):
            for alpha in (0.25, 0.5, 1.0):
                for sign in (-1, 1):
                    requested = v22_actions._row(directions, spec, alpha, sign)
                    edited = v19.apply(p0, 1.0, requested, rec, att, "native_fp32_add_bf16_writeback")
                    actual = _readback(p0, edited, requested, rec, att)
                    rows.append({"base_trial_id": base_id, "family": item["family"], "action_id": spec["action_id"],
                                 "alpha": alpha, "sign": sign, "reliable": bool(_action_ok(actual, operator["action_reliability"])),
                                 "cosine": actual["realized_state_cosine"], "gain": actual["realized_state_gain"],
                                 "requested_norm": actual["requested_state_norm"], "realized_norm": actual["realized_state_norm"]})
            if action_number % 32 == 0:
                print(f"V23 combo calibration base={base_number}/5 action={action_number}/128", flush=True)
    frame = pd.DataFrame(rows)
    parquet = root / OUT / "generated_combo_calibration_v23.parquet"
    frame.to_parquet(parquet, index=False, compression="zstd")
    selected = []
    for spec in design["generated_combo_candidates"]:
        group = frame[frame.action_id == spec["action_id"]]
        detail = {}
        valid = []
        for alpha in (0.25, 0.5, 1.0):
            fractions = {str(sign): float(group[(group.alpha == alpha) & (group.sign == sign)].reliable.mean()) for sign in (-1, 1)}
            detail[str(alpha)] = fractions
            if min(fractions.values()) >= 0.8:
                valid.append(alpha)
        selected.append({"action_id": spec["action_id"], "selected_alpha": min(valid) if valid else None,
                         "candidate_reliability": detail})
    summary = {"candidate_count": 128, "reliable_count": sum(x["selected_alpha"] is not None for x in selected),
               "by_action": selected, "parquet_sha256": sha256_file(parquet),
               "historical_final_responses_opened": False, "v23_independent_final_responses_opened": False}
    target = root / OUT / "generated_combo_calibration_v23.json"
    write_json_atomic(target, summary)
    frozen = stage_freeze(root, "combo_calibration", [SOURCE,
        "artifacts/oracle_local_action_charts_v23_probe_candidate_design.freeze.json", str(parquet), str(target)], summary)
    return {"freeze_digest": frozen["freeze_digest"], "reliable_count": summary["reliable_count"]}


def select(root: Path) -> dict:
    config = verify(root)["config"]
    verify_stage(root, "combo_calibration")
    design = json.loads((root / OUT / "probe_candidate_design_v23.json").read_text())
    combo_cal = json.loads((root / OUT / "generated_combo_calibration_v23.json").read_text())
    pool = json.loads((root / "results/v22/processed/action_pool_v22.json").read_text())
    v22_cal = json.loads((root / "results/v22/processed/action_pool_calibration_v22.json").read_text())
    specs = {item["action_id"]: item for item in pool["candidate_pool"]}
    specs.update({item["action_id"]: item for item in design["generated_combo_candidates"]})
    alphas = {item["action_id"]: float(item["selected_alpha"]) for item in v22_cal["by_action"] if item["selected_alpha"] is not None}
    reliable_combos = [item for item in combo_cal["by_action"] if item["selected_alpha"] is not None]
    if len(reliable_combos) < 93:
        raise RuntimeError(f"V23 needs 93 reliable generated probes, found {len(reliable_combos)}")
    common = _balanced(design["v22_common_train_set"], specs, "V23-COMMON")
    extras = [value for value in design["reliable_original_action_ids"] if value not in set(common)]
    train_ids = common + extras + [item["action_id"] for item in reliable_combos[:93]]
    if len(train_ids) != 512 or len(set(train_ids)) != 512:
        raise RuntimeError("V23 nested training probe construction failed")
    for item in reliable_combos[:93]:
        alphas[item["action_id"]] = float(item["selected_alpha"])
    action_specs = [specs[value] for value in train_ids]
    directions = torch.load(root / "artifacts/causal/v13/probe_directions_v13.pt", map_location="cpu", weights_only=False)
    score = np.asarray(directions["score_directions"], dtype=np.float64)
    x = np.stack([v22_actions._score(specs[value], score) * alphas[value] for value in train_ids])
    x_hold = np.stack([v22_actions._score(specs[value], score) * float(json.loads((root / "results/v22/processed/action_selection_v22.json").read_text())["action_alphas"][value])
                       for value in design["heldout_validation_action_ids"]])
    del directions, score
    metric_path = root / SCRATCH / "probe_metric_v23.npz"
    metric_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(metric_path, action_matrix=x.astype(np.float32), heldout_action_matrix=x_hold.astype(np.float32),
                        train_action_ids=np.asarray(train_ids), heldout_action_ids=np.asarray(design["heldout_validation_action_ids"]),
                        action_alphas=np.asarray([alphas[x] for x in train_ids], dtype=np.float32))
    family_counts = {}
    for m in config["probe_counts"]:
        family_counts[str(m)] = {family: sum(specs[value]["source_family"] == family for value in train_ids[:m])
                                 for family in sorted({specs[value]["source_family"] for value in train_ids[:m]})}
    result = {
        "nested_probe_counts": config["probe_counts"], "train_action_ids": train_ids,
        "heldout_validation_action_ids": design["heldout_validation_action_ids"], "action_specs": action_specs,
        "action_alphas": {value: alphas[value] for value in train_ids}, "family_counts_by_m": family_counts,
        "probe_hashes": {str(m): _digest(train_ids[:m]) for m in config["probe_counts"]},
        "training_action_hash": _digest(train_ids), "heldout_action_hash": design["heldout_action_hash"],
        "metric_scratch_sha256": sha256_file(metric_path), "metric_rule": config["geometry"],
        "historical_final_action_overlap": sorted(set(train_ids) & set(design["historical_v22_final_action_ids"])),
        "historical_final_responses_opened": False, "v23_independent_final_responses_opened": False,
    }
    target = root / OUT / "probe_selection_v23.json"
    write_json_atomic(target, result)
    frozen = stage_freeze(root, "probe_selection_and_metric", [SOURCE,
        "artifacts/oracle_local_action_charts_v23_combo_calibration.freeze.json",
        "results/v22/processed/action_pool_v22.json", "results/v22/processed/action_pool_calibration_v22.json", str(target)],
        {key: result[key] for key in ("nested_probe_counts", "probe_hashes", "training_action_hash", "heldout_action_hash",
                                      "metric_scratch_sha256", "metric_rule", "historical_final_action_overlap",
                                      "historical_final_responses_opened", "v23_independent_final_responses_opened")})
    return {"freeze_digest": frozen["freeze_digest"], "train_count": 512, "heldout_count": 32,
            "probe_hashes": result["probe_hashes"], "training_action_hash": result["training_action_hash"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("prepare", "calibrate", "select"))
    args = parser.parse_args()
    answer = {"prepare": prepare, "calibrate": calibrate, "select": select}[args.stage](Path.cwd())
    print(json.dumps(answer, indent=2))
