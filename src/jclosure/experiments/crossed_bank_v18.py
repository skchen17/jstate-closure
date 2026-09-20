"""V18 crossed clean-state × shared finite-action bank with exact horizons."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from jclosure.cache_v7 import clone_hybrid_cache
from jclosure.experiments import geometry_v13 as geometry
from jclosure.experiments import actuation_v15 as actuation
from jclosure.experiments.action_bank_v16 import _readback
from jclosure.experiments.actuation_v15 import apply
from jclosure.experiments.operator_v15 import TARGETS, stack
from jclosure.model import load_model_bundle
from jclosure.protocol_v18 import digest, stage_freeze, verify
from jclosure.provenance import sha256_file, write_json_atomic
from jclosure.recorder import ActivationRecorder
from jclosure.runtime_v3_1 import encode_direct_prompt

SOURCE = "src/jclosure/experiments/crossed_bank_v18.py"
SPLITS = Path("artifacts/strong_state_context_ceiling_v18_splits.freeze.json")
ACTION_AMENDMENT = Path("artifacts/strong_state_context_ceiling_v18_action_selection_amendment_1.freeze.json")
STATE_AMENDMENT = Path("artifacts/strong_state_context_ceiling_v18_state_layer_amendment_2.freeze.json")
ALIGNMENT_AMENDMENT = Path("artifacts/strong_state_context_ceiling_v18_reference_alignment_amendment_3.freeze.json")
RUNTIME_AMENDMENT = Path("artifacts/strong_state_context_ceiling_v18_runtime_placement_amendment_4.freeze.json")
TEACHER_AMENDMENT = Path("artifacts/strong_state_context_ceiling_v18_teacher_drift_amendment_5.freeze.json")
OUT = Path("results/v18/processed")
SCRATCH = Path("/data/CSK/J-space-project/v18-crossed-bank-work")


def _hash_ids(ids: list[str]) -> str:
    return hashlib.sha256("\n".join(sorted(ids)).encode()).hexdigest()


def amend_actions(root: Path) -> dict:
    if (root / SPLITS).exists():
        raise RuntimeError("V18 action-selection amendment must precede split freeze")
    return stage_freeze(root, "action_selection_amendment_1",
                        ["configs/strong_state_context_ceiling_v18.yaml",
                         "results/v16/processed/finite_action_bank_v16.parquet"],
                        {"reason": "V16 train-only calibration shows indices 18 and 8 generally require alpha 1.0; replace with train-reliable alpha-0.5 indices 17 and 2 before any V18 response observation",
                         "old_coordinate_indices": [5, 1, 7, 20, 11, 19, 18, 8],
                         "corrected_coordinate_indices": [5, 1, 7, 20, 11, 19, 17, 2],
                         "new_v18_responses_at_amendment": 0})


def prepare(root: Path) -> dict[str, Any]:
    base = verify(root)
    cfg = base["config"]
    with np.load(root / geometry.FEATURES, allow_pickle=False) as payload:
        ids = payload["base_trial_id"].astype(str)
        families = payload["family"].astype(str)
        source_splits = payload["split"].astype(str)
    v16 = json.loads((root / "artifacts/nonlinear_finite_causal_action_v16_splits.freeze.json").read_text())
    excluded = {role: {x["base_trial_id"] for x in v16[role]} for role in ("train", "validation")}
    selected: dict[str, list[dict[str, Any]]] = {}
    for role in ("train", "validation"):
        n = int(cfg["roles"][f"{role}_per_family"])
        nh = int(cfg["roles"][f"horizon_{role}_per_family"])
        role_rows = []
        for family in sorted(set(families)):
            pool = [str(ids[i]) for i in np.flatnonzero((source_splits == role) & (families == family))
                    if str(ids[i]) not in excluded[role]]
            pool.sort(key=lambda key: hashlib.sha256(f"{cfg['roles']['seed']}:{role}:{key}".encode()).hexdigest())
            if len(pool) < n:
                raise RuntimeError(f"V18 {role}/{family}: only {len(pool)} source states, require {n}")
            role_rows.extend({"base_trial_id": key, "family": family, "role": role,
                              "horizon_panel": i < nh} for i, key in enumerate(pool[:n]))
        selected[role] = role_rows
    if {x["base_trial_id"] for x in selected["train"]} & {x["base_trial_id"] for x in selected["validation"]}:
        raise RuntimeError("V18 train/validation overlap")
    bank = pd.read_parquet(root / "results/v16/processed/finite_action_bank_v16.parquet")
    bank = bank[(bank.role == "train") & (bank.design == "single")]
    amendment = json.loads((root / ACTION_AMENDMENT).read_text())
    if amendment["freeze_digest"] != digest(amendment) or amendment["base_freeze_digest"] != base["freeze_digest"]:
        raise RuntimeError("V18 action-selection amendment invalid")
    for path, expected in amendment["input_hashes"].items():
        if sha256_file(root / path) != expected:
            raise RuntimeError(f"V18 action amendment input changed: {path}")
    actions = []
    for coordinate in amendment["corrected_coordinate_indices"]:
        rows = bank[bank.coordinate_index == coordinate]
        if len(rows) != 200 or (rows.reliability_status != "RELIABLE").any() or float(rows.j_effect_norm.median()) < cfg["actions"]["reliability_min_j_effect_norm"]:
            raise RuntimeError(f"V18 action {coordinate} not supported by V16 train-only calibration")
        if float(rows.calibration_alpha.median()) != cfg["actions"]["primary_alpha"]:
            raise RuntimeError(f"V18 action {coordinate} alpha calibration mismatch")
        actions.append({"coordinate_index": int(coordinate),
                        "direction_index": int(v16["direction_indices"][coordinate]),
                        "proposal_family": v16["direction_families"][coordinate],
                        "v16_train_reliable_fraction": 1.0,
                        "v16_train_median_j_effect_norm": float(rows.j_effect_norm.median())})
    detail = {
        "train": selected["train"], "validation": selected["validation"],
        "train_id_sha256": _hash_ids([x["base_trial_id"] for x in selected["train"]]),
        "validation_id_sha256": _hash_ids([x["base_trial_id"] for x in selected["validation"]]),
        "horizon_train_id_sha256": _hash_ids([x["base_trial_id"] for x in selected["train"] if x["horizon_panel"]]),
        "horizon_validation_id_sha256": _hash_ids([x["base_trial_id"] for x in selected["validation"] if x["horizon_panel"]]),
        "actions": actions, "selected_j": v16["selected_j"], "selected_logits": v16["selected_logits"],
        "target_scales": v16["target_scales"],
        "independent_final": "UNOPENED_NEW_PROMPT_BANK_ONLY_AFTER_FINALIST",
        "v16_development_ids_excluded_from_v18_roles": True,
        "frozen_v16_action_source_split_digest": v16["freeze_digest"],
        "action_selection_amendment_digest": amendment["freeze_digest"],
    }
    return stage_freeze(root, "splits", [SOURCE, "results/v16/processed/finite_action_bank_v16.parquet",
                                         "artifacts/nonlinear_finite_causal_action_v16_splits.freeze.json",
                                         "results/v13/processed/causal_capture_train_v13.json",
                                         "results/v13/processed/causal_capture_validation_v13.json"], detail)


def _split(root: Path) -> dict:
    base = verify(root)
    value = json.loads((root / SPLITS).read_text())
    if value["freeze_digest"] != digest(value) or value["base_freeze_digest"] != base["freeze_digest"]:
        raise RuntimeError("V18 split freeze invalid")
    for path, expected in value["input_hashes"].items():
        observed = sha256_file(root / path)
        if observed != expected:
            if path != SOURCE or not (root / STATE_AMENDMENT).exists():
                raise RuntimeError(f"V18 frozen split input changed: {path}")
            layer_amendment = json.loads((root / STATE_AMENDMENT).read_text())
            if (layer_amendment["freeze_digest"] != digest(layer_amendment)
                    or layer_amendment["base_freeze_digest"] != base["freeze_digest"]
                    or layer_amendment["prior_source_sha256"] != expected
                    or layer_amendment["input_hashes"].get(str(SPLITS)) != sha256_file(root / SPLITS)):
                raise RuntimeError("V18 state-layer amendment invalid")
            if (root / ALIGNMENT_AMENDMENT).exists():
                alignment = json.loads((root / ALIGNMENT_AMENDMENT).read_text())
                if (alignment["freeze_digest"] != digest(alignment)
                        or alignment["base_freeze_digest"] != base["freeze_digest"]
                        or alignment["prior_source_sha256"] != layer_amendment["input_hashes"][SOURCE]
                        or alignment["input_hashes"].get(str(STATE_AMENDMENT)) != sha256_file(root / STATE_AMENDMENT)):
                    raise RuntimeError("V18 reference-alignment amendment invalid")
                if (root / RUNTIME_AMENDMENT).exists():
                    runtime = json.loads((root / RUNTIME_AMENDMENT).read_text())
                    if (runtime["freeze_digest"] != digest(runtime)
                            or runtime["base_freeze_digest"] != base["freeze_digest"]
                            or runtime["prior_source_sha256"] != alignment["input_hashes"][SOURCE]
                            or runtime["input_hashes"].get(str(ALIGNMENT_AMENDMENT)) != sha256_file(root / ALIGNMENT_AMENDMENT)):
                        raise RuntimeError("V18 runtime-placement amendment invalid")
                    if (root / TEACHER_AMENDMENT).exists():
                        teacher = json.loads((root / TEACHER_AMENDMENT).read_text())
                        if (teacher["freeze_digest"] != digest(teacher)
                                or teacher["base_freeze_digest"] != base["freeze_digest"]
                                or teacher["prior_source_sha256"] != runtime["input_hashes"][SOURCE]
                                or teacher["input_hashes"].get(SOURCE) != observed
                                or teacher["input_hashes"].get(str(RUNTIME_AMENDMENT)) != sha256_file(root / RUNTIME_AMENDMENT)):
                            raise RuntimeError("V18 teacher-drift amendment invalid")
                    elif runtime["input_hashes"].get(SOURCE) != observed:
                        raise RuntimeError("V18 amended source changed without teacher-drift amendment")
                elif alignment["input_hashes"].get(SOURCE) != observed:
                    raise RuntimeError("V18 amended source changed without runtime amendment")
            elif layer_amendment["input_hashes"].get(SOURCE) != observed:
                raise RuntimeError("V18 amended source changed without alignment amendment")
    return value


def amend_state_layer(root: Path) -> dict:
    base = verify(root)
    split = json.loads((root / SPLITS).read_text())
    prior_hash = split["input_hashes"][SOURCE]
    if prior_hash == sha256_file(root / SOURCE):
        raise RuntimeError("No source change to amend")
    existing = sum(1 for role in ("train", "validation")
                   for item in split[role] if _state_paths(role, str(item["base_trial_id"]))[1].exists())
    return stage_freeze(root, "state_layer_amendment_2", [SOURCE, str(SPLITS)],
                        {"reason": "V18 pilot history/current J was captured from output layer 30, whereas frozen V13 clean current J and V17 state context use intervention layer 23. Correct pre-action J history to layer 23 with cloned activations; keep output response readout at layer 30.",
                         "prior_source_sha256": prior_hash,
                         "pilot_state_records_before_correction": existing,
                         "response_policy": "retain unchanged action-response records; archive and recompute state metadata only",
                         "prior_state_records_location": str(SCRATCH / "state_layer_amendment_2_prior"),
                         "v13_reference_check": "corrected_current_j_rel_l2_below_0.01_each_state"})


def amend_reference_alignment(root: Path) -> dict:
    layer = json.loads((root / STATE_AMENDMENT).read_text())
    return stage_freeze(root, "reference_alignment_amendment_3", [SOURCE, str(STATE_AMENDMENT)],
                        {"reason": "First corrected live layer-23 J had 0.016768 relative L2 versus frozen V13 current J, consistent with replay numerical drift. Use exact frozen V13 clean J for current state and final position of history; keep live pre-action positions -4,-3,-2 and record drift.",
                         "prior_source_sha256": layer["input_hashes"][SOURCE],
                         "observed_first_state_live_to_v13_rel_l2": 0.016767999157309532,
                         "current_state_source": "frozen_V13_clean_endpoint_layer_23",
                         "history_source": "live_layer_23_previous_three_plus_frozen_V13_current",
                         "response_policy": "original_action_responses_unchanged"})


def amend_runtime_placement(root: Path) -> dict:
    alignment = json.loads((root / ALIGNMENT_AMENDMENT).read_text())
    split = json.loads((root / SPLITS).read_text())
    existing = sum(1 for item in split["train"]
                   if all(path.exists() for path in _state_paths("train", str(item["base_trial_id"]))))
    return stage_freeze(root, "runtime_placement_amendment_4",
                        [SOURCE, str(ALIGNMENT_AMENDMENT), "scripts/benchmark_v18_single_gpu.py"],
                        {"reason": "GPU 1 concurrently saturated; V18 model placement on GPU 0 alone preserves weight/dtype/action/estimand and removes inter-GPU transfer bottleneck.",
                         "prior_source_sha256": alignment["input_hashes"][SOURCE],
                         "pilot_train_states_at_amendment": existing,
                         "model_device_map": {"": 0}, "model_weights_changed": False,
                         "read_only_equivalence_test": "same frozen normal state, all 16 h1 action normalized stacks exactly equal to dual-GPU pilot, median and max relative L2 0.0",
                         "benchmark_single_gpu_state_seconds": 2.2727047111839056,
                         "benchmark_gpu0_peak_allocated_GiB": 8.10276746749878,
                         "prior_response_policy": "retain dual-GPU pilot response records; no recomputation"})


def amend_teacher_drift(root: Path) -> dict:
    runtime = json.loads((root / RUNTIME_AMENDMENT).read_text())
    return stage_freeze(root, "teacher_drift_amendment_5",
                        [SOURCE, str(RUNTIME_AMENDMENT), "scripts/audit_v18_teacher_drift.py"],
                        {"reason": "Second validation clean replay differs from frozen V13 teacher token only at zero-based position 7; V18 teacher-forced estimand uses newly generated same-state clean greedy tokens, while V13 token agreement is descriptive, not an eligibility filter.",
                         "prior_source_sha256": runtime["input_hashes"][SOURCE],
                         "first_observed_validation_id": "v13-validation-572620683928d47d21c6",
                         "first_observed_mismatch_zero_based": 7,
                         "first_seven_tokens_identical": True,
                         "teacher_policy": "V18_clean_greedy_same_state_teacher_forcing",
                         "historical_V13_tokens_changed": False,
                         "prior_pilot_response_policy": "retain_unchanged",
                         "comparison_policy": "compare_only_shared_frozen_V13_prefix_and_audit_all_states"})


def load_context_v18(root: Path) -> tuple[Any, Any, dict[str, Any], dict[str, Any], dict[str, Any]]:
    """V13 context with identical weights and V18-only all-on-GPU0 placement."""
    original_loader = actuation._load_model

    def single_gpu_loader(config: dict[str, Any]) -> Any:
        amended = copy.deepcopy(config)
        amended["model"]["device_map"] = {"": 0}
        amended["model"].pop("max_memory", None)
        amended["model"].pop("offload_folder", None)
        return load_model_bundle(amended)

    actuation._load_model = single_gpu_loader
    try:
        return actuation.load_context(root)
    finally:
        actuation._load_model = original_loader


@torch.no_grad()
def _prefill_history(bundle: Any, prompt: str, measured: list[int], dense_map: Any, state_layer: int) -> dict:
    input_ids = encode_direct_prompt(bundle, prompt)
    with ActivationRecorder(bundle.layers, at=measured, clone=True, detach=True) as recorder:
        output = bundle.hf_model(input_ids=input_ids, use_cache=True)
    hidden = recorder.activations[state_layer][0]
    if hidden.shape[0] < 4:
        raise RuntimeError("V18 prompt shorter than four-token J history")
    history = torch.stack([dense_map.dense_state(hidden[-4 + i].float(), state_layer) for i in range(4)])
    return {"cache": clone_hybrid_cache(output.past_key_values),
            "logits": output.logits[0, -1].float(),
            "prompt_length": int(input_ids.shape[1]),
            "history_j": history.cpu().numpy().astype(np.float16)}


@torch.no_grad()
def _trajectory(bundle: Any, dense_map: Any, cache: Any, first_logits: torch.Tensor | None,
                tokens: list[int] | None, count: int, prompt_length: int,
                selected_j: np.ndarray, selected_logits: np.ndarray,
                workspace_layers: list[int], workspace_count: int, main: int) -> tuple[list[int], dict[int, dict[str, np.ndarray]]]:
    device = next(bundle.hf_model.parameters()).device
    working = clone_hybrid_cache(cache)
    logits = first_logits
    used: list[int] = []
    endpoints: dict[int, dict[str, np.ndarray]] = {}
    for step in range(count):
        token = int(tokens[step]) if tokens is not None else int(torch.argmax(logits).item())
        used.append(token)
        with ActivationRecorder(bundle.layers, at=workspace_layers, clone=False, detach=True) as recorder:
            output = bundle.hf_model(
                input_ids=torch.tensor([[token]], device=device),
                attention_mask=torch.ones((1, prompt_length + step + 1), device=device, dtype=torch.long),
                past_key_values=working,
                use_cache=True,
            )
        working = output.past_key_values
        logits = output.logits[0, -1].float()
        if step + 1 not in (1, 2, 4, 8):
            continue
        hidden = recorder.activations[main][0, -1].float()
        j = dense_map.dense_state(hidden, main)
        selection = torch.as_tensor(selected_j, device=j.device)
        logit_selection = torch.as_tensor(selected_logits, device=logits.device)
        workspace = torch.cat([recorder.activations[layer][0, -1].float()[:workspace_count]
                               for layer in workspace_layers])
        endpoints[step + 1] = {
            "j": j[selection].cpu().numpy().astype(np.float32),
            "logits": logits[logit_selection].cpu().numpy().astype(np.float32),
            "semantic_continuous": torch.log_softmax(logits, dim=-1)[logit_selection].cpu().numpy().astype(np.float32),
            "workspace": workspace.cpu().numpy().astype(np.float32),
        }
    return used, endpoints


def _state_paths(role: str, base_id: str) -> tuple[Path, Path]:
    directory = SCRATCH / role
    return directory / f"response_{base_id}.parquet", directory / f"state_{base_id}.parquet"


def run(root: Path, role: str, limit: int | None = None) -> dict:
    split = _split(root)
    config = verify(root)["config"]
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    bundle, dense_map, v13, metadata, values = load_context_v18(root)
    directions = values["directions"]
    v8 = v13["persistent_state_v8"]
    measured = [int(x) for x in v8["intervention"]["measured_layers"]]
    rec = [int(x) for x in v8["persistent_components"]["recurrent_layers"]]
    att = [int(x) for x in v8["persistent_components"]["attention_layers"]]
    jvp = v13["causal_geometry_v13"]["jvp"]
    workspace_layers = [int(x) for x in jvp["workspace_layers"]]
    workspace_count = int(jvp["selected_workspace_count"])
    main = max(measured)
    state_layer = int(v8["intervention"]["layer"])
    all_ids = [str(x["base_trial_id"]) for x in split["train"] + split["validation"]]
    j_reference = np.memmap(SCRATCH / "clean_j_v18.f16", dtype=np.float16,
                            mode="r", shape=(len(all_ids), 4096))
    reference_index = {key: i for i, key in enumerate(all_ids)}
    selected_j = np.asarray(split["selected_j"], dtype=int)
    selected_logits = np.asarray(split["selected_logits"], dtype=int)
    scales = {key: float(value) for key, value in split["target_scales"].items()}
    items = split[role][:limit] if limit is not None else split[role]
    (SCRATCH / role).mkdir(parents=True, exist_ok=True)
    completed = 0
    for number, item in enumerate(items):
        base_id = str(item["base_trial_id"])
        response_path, state_path = _state_paths(role, base_id)
        if response_path.exists() and state_path.exists():
            completed += 1
            continue
        pair = metadata[base_id]
        task = values["tasks"][str(pair["prompt_id"])]
        clean = _prefill_history(bundle, task.prompt, measured, dense_map, state_layer)
        observed_j = clean["history_j"][-1].astype(np.float32)
        frozen_j = np.asarray(j_reference[reference_index[base_id]], dtype=np.float32)
        current_j_rel_l2 = float(np.linalg.norm(observed_j - frozen_j) / max(np.linalg.norm(frozen_j), 1e-12))
        history_j = clean["history_j"].astype(np.float32)
        history_j[-1] = frozen_j
        count = 8 if item["horizon_panel"] else 1
        tokens, baseline = _trajectory(bundle, dense_map, clean["cache"], clean["logits"], None,
                                       count, clean["prompt_length"], selected_j, selected_logits,
                                       workspace_layers, workspace_count, main)
        prior_tokens = [int(x) for x in pair.get("teacher_tokens", [])]
        shared_token_count = min(len(tokens), len(prior_tokens))
        teacher_token_match = tokens[:shared_token_count] == prior_tokens[:shared_token_count]
        state_row = {"base_trial_id": base_id, "family": item["family"], "role": role,
                     "prompt_id": task.example_id, "template_id": task.template_id,
                     "prompt": task.prompt, "prompt_length": clean["prompt_length"],
                     "token_position": clean["prompt_length"] - 1,
                     "teacher_correct_v13": bool(pair["teacher_correct"]),
                     "teacher_token_match_v13": teacher_token_match,
                     "teacher_token_compared_count_v13": shared_token_count,
                     "teacher_tokens_h8_or_h1": tokens,
                     "horizon_panel": bool(item["horizon_panel"]),
                     "history_j_last4": history_j.tolist(),
                     "current_j": frozen_j.tolist(),
                     "state_j_layer": state_layer, "response_j_layer": main,
                     "live_current_j_v13_rel_l2": current_j_rel_l2,
                     "state_layer_amendment_digest": json.loads((root / STATE_AMENDMENT).read_text())["freeze_digest"],
                     "reference_alignment_amendment_digest": json.loads((root / ALIGNMENT_AMENDMENT).read_text())["freeze_digest"],
                     "source_split_freeze_digest": split["freeze_digest"]}
        action_rows = []
        for action in split["actions"]:
            direction = int(action["direction_index"])
            for alpha in ([config["actions"]["primary_alpha"], config["actions"]["nested_amplitude_alpha"]]
                          if item["horizon_panel"] else [config["actions"]["primary_alpha"]]):
                for sign in config["actions"]["signs"]:
                    row = {name: directions[name][direction].float() * float(alpha * sign)
                           for name in ("recurrent", "conv", "kv")}
                    edited = apply(clean["cache"], 1.0, row, rec, att, "native_fp32_add_bf16_writeback")
                    realized = _readback(clean["cache"], edited, row, rec, att)
                    _, endpoint = _trajectory(bundle, dense_map, edited, None, tokens, count,
                                              clean["prompt_length"], selected_j, selected_logits,
                                              workspace_layers, workspace_count, main)
                    for horizon, result in endpoint.items():
                        delta = {target: (result[target] - baseline[horizon][target]).astype(np.float32)
                                 for target in TARGETS}
                        delta["stacked_normalized"] = stack(delta, scales).astype(np.float32)
                        effect = float(np.linalg.norm(delta["j"]))
                        reliable = (realized["realized_state_cosine"] is not None
                                    and realized["realized_state_cosine"] >= config["actions"]["reliability_min_state_cosine"]
                                    and realized["realized_state_gain"] is not None
                                    and config["actions"]["reliability_state_gain_min"] <= realized["realized_state_gain"] <= config["actions"]["reliability_state_gain_max"]
                                    and float(np.linalg.norm(endpoint[1]["j"] - baseline[1]["j"])) >= config["actions"]["reliability_min_j_effect_norm"])
                        action_rows.append({"base_trial_id": base_id, "family": item["family"], "role": role,
                                            "prompt_id": task.example_id, "token_position": clean["prompt_length"] - 1,
                                            "horizon_panel": bool(item["horizon_panel"]),
                                            "horizon": horizon, "coordinate_index": action["coordinate_index"],
                                            "direction_index": direction, "proposal_family": action["proposal_family"],
                                            "sign": sign, "alpha": float(alpha),
                                            "reliability_status": "RELIABLE" if reliable else "ACTION_NOT_RELIABLY_ACTUATABLE",
                                            "j_effect_norm_h1": float(np.linalg.norm(endpoint[1]["j"] - baseline[1]["j"])),
                                            **realized,
                                            **{f"response_{target}": delta[target].tolist()
                                               for target in (*TARGETS, "stacked_normalized")},
                                            "split_freeze_digest": split["freeze_digest"]})
        pd.DataFrame(action_rows).to_parquet(response_path, index=False, compression="zstd")
        pd.DataFrame([state_row]).to_parquet(state_path, index=False, compression="zstd")
        completed += 1
        if completed % 10 == 0 or completed == len(items):
            print(f"V18 {role} {completed}/{len(items)} {base_id} actions={len(action_rows)}", flush=True)
    return {"role": role, "completed": completed, "selected": len(items), "scratch": str(SCRATCH / role)}


def repair_state_metadata(root: Path, role: str, limit: int | None = None) -> dict:
    split = _split(root)
    all_items = split[role]
    selected = [(i, item) for i, item in enumerate(all_items)
                if all(path.exists() for path in _state_paths(role, str(item["base_trial_id"])))
                and "state_j_layer" not in pd.read_parquet(_state_paths(role, str(item["base_trial_id"]))[1], columns=None).columns]
    selected = selected[:limit] if limit is not None else selected
    archive = SCRATCH / "state_layer_amendment_2_prior" / role
    archive.mkdir(parents=True, exist_ok=True)
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    bundle, dense_map, v13, _, values = load_context_v18(root)
    v8 = v13["persistent_state_v8"]
    measured = [int(x) for x in v8["intervention"]["measured_layers"]]
    state_layer = int(v8["intervention"]["layer"])
    all_ids = [str(x["base_trial_id"]) for x in split["train"] + split["validation"]]
    reference_index = {key: i for i, key in enumerate(all_ids)}
    j_reference = np.memmap(SCRATCH / "clean_j_v18.f16", dtype=np.float16,
                            mode="r", shape=(len(all_ids), 4096))
    records = []
    for number, (_, item) in enumerate(selected, 1):
        base_id = str(item["base_trial_id"])
        response_path, state_path = _state_paths(role, base_id)
        prior_path = archive / state_path.name
        old = pd.read_parquet(state_path)
        if prior_path.exists():
            raise RuntimeError(f"V18 prior state archive already exists: {prior_path}")
        clean = _prefill_history(bundle, str(old.iloc[0]["prompt"]), measured, dense_map, state_layer)
        observed_j = clean["history_j"][-1].astype(np.float32)
        frozen_j = np.asarray(j_reference[reference_index[base_id]], dtype=np.float32)
        relative = float(np.linalg.norm(observed_j - frozen_j) / max(np.linalg.norm(frozen_j), 1e-12))
        history_j = clean["history_j"].astype(np.float32)
        history_j[-1] = frozen_j
        response_hash = sha256_file(response_path)
        prior_hash = sha256_file(state_path)
        shutil.copy2(state_path, prior_path)
        updated = old.copy()
        updated.at[0, "history_j_last4"] = history_j.tolist()
        updated.at[0, "current_j"] = frozen_j.tolist()
        updated["state_j_layer"] = state_layer
        updated["response_j_layer"] = max(measured)
        updated["live_current_j_v13_rel_l2"] = relative
        updated["state_layer_amendment_digest"] = json.loads((root / STATE_AMENDMENT).read_text())["freeze_digest"]
        updated["reference_alignment_amendment_digest"] = json.loads((root / ALIGNMENT_AMENDMENT).read_text())["freeze_digest"]
        temporary = state_path.with_suffix(".corrected.parquet")
        updated.to_parquet(temporary, index=False, compression="zstd")
        temporary.replace(state_path)
        if sha256_file(response_path) != response_hash:
            raise RuntimeError(f"V18 response changed during metadata repair: {base_id}")
        records.append({"base_trial_id": base_id, "prior_state_sha256": prior_hash,
                        "corrected_state_sha256": sha256_file(state_path),
                        "response_sha256_unchanged": response_hash,
                        "live_current_j_v13_rel_l2": relative,
                        "corrected_current_j_v13_rel_l2": 0.0})
        if number % 25 == 0 or number == len(selected):
            print(f"V18 repaired {role} {number}/{len(selected)}", flush=True)
    result = {"role": role, "records": records, "archive": str(archive),
              "state_layer_amendment_digest": json.loads((root / STATE_AMENDMENT).read_text())["freeze_digest"],
              "reference_alignment_amendment_digest": json.loads((root / ALIGNMENT_AMENDMENT).read_text())["freeze_digest"]}
    write_json_atomic(root / OUT / f"state_layer_amendment_2_{role}_v18.json", result)
    return {"role": role, "repaired": len(records), "maximum_rel_l2": max((r["corrected_current_j_v13_rel_l2"] for r in records), default=None)}


def aggregate(root: Path, role: str) -> dict:
    split = _split(root)
    items = split[role]
    OUT.mkdir(parents=True, exist_ok=True)
    outputs = []
    for family in sorted({x["family"] for x in items}):
        response_parts, state_parts = [], []
        family_items = [x for x in items if x["family"] == family]
        for item in family_items:
            response_path, state_path = _state_paths(role, item["base_trial_id"])
            if not response_path.exists() or not state_path.exists():
                raise RuntimeError(f"V18 {role} bank incomplete: {item['base_trial_id']}")
            response_parts.append(pd.read_parquet(response_path))
            state_parts.append(pd.read_parquet(state_path))
        for label, parts in (("response", response_parts), ("state", state_parts)):
            frame = pd.concat(parts, ignore_index=True)
            path = root / OUT / f"crossed_{label}_{role}_{family}_v18.parquet"
            frame.to_parquet(path, index=False, compression="zstd", compression_level=9)
            outputs.append({"path": str(path.relative_to(root)), "rows": len(frame), "sha256": sha256_file(path)})
    result = {"role": role, "selected_states": len(items), "complete": True, "outputs": outputs,
              "split_freeze_digest": split["freeze_digest"]}
    write_json_atomic(root / OUT / f"crossed_bank_{role}_v18.json", result)
    return result


def audit_teacher_tokens(root: Path) -> dict:
    split = _split(root)
    metadata = geometry._pair_metadata(root)
    rows = []
    for role in ("train", "validation"):
        for item in split[role]:
            key = str(item["base_trial_id"])
            _, state_path = _state_paths(role, key)
            if not state_path.exists():
                raise RuntimeError(f"V18 teacher audit state missing: {key}")
            state = pd.read_parquet(state_path).iloc[0]
            observed = [int(x) for x in state.teacher_tokens_h8_or_h1]
            frozen = [int(x) for x in metadata[key].get("teacher_tokens", [])]
            compared = min(len(observed), len(frozen))
            first = next((i for i, (a, b) in enumerate(zip(observed, frozen)) if a != b), None)
            rows.append({"base_trial_id": key, "role": role, "family": item["family"],
                         "horizon_panel": bool(item["horizon_panel"]),
                         "v18_generated_count": len(observed), "v13_frozen_count": len(frozen),
                         "compared_prefix_count": compared,
                         "first_mismatch_zero_based": first,
                         "shared_prefix_exact": first is None,
                         "h1_token_exact": bool(observed and frozen and observed[0] == frozen[0])})
    frame = pd.DataFrame(rows)
    path = root / OUT / "teacher_token_drift_audit_v18.parquet"
    frame.to_parquet(path, index=False, compression="zstd")
    summary = {"state_count": len(frame), "h1_token_exact_count": int(frame.h1_token_exact.sum()),
               "shared_prefix_exact_count": int(frame.shared_prefix_exact.sum()),
               "first_mismatch_step_counts": {str(k): int(v) for k, v in frame.first_mismatch_zero_based.value_counts().items()},
               "v13_frozen_tokens_unchanged": True,
               "v18_teacher_policy": "new_same_state_clean_greedy_teacher_forcing",
               "audit_parquet_sha256": sha256_file(path),
               "teacher_drift_amendment_digest": json.loads((root / TEACHER_AMENDMENT).read_text())["freeze_digest"]}
    write_json_atomic(root / OUT / "teacher_token_drift_audit_v18.json", summary)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("amend_actions", "amend_state_layer", "amend_reference_alignment", "amend_runtime_placement", "amend_teacher_drift", "repair_train", "repair_validation", "prepare", "train", "validation", "aggregate_train", "aggregate_validation", "audit_teacher", "verify"))
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    root = Path.cwd()
    if args.stage == "amend_actions":
        result = amend_actions(root)
    elif args.stage == "amend_state_layer":
        result = amend_state_layer(root)
    elif args.stage == "amend_reference_alignment":
        result = amend_reference_alignment(root)
    elif args.stage == "amend_runtime_placement":
        result = amend_runtime_placement(root)
    elif args.stage == "amend_teacher_drift":
        result = amend_teacher_drift(root)
    elif args.stage.startswith("repair_"):
        result = repair_state_metadata(root, args.stage.split("_", 1)[1], args.limit)
    elif args.stage == "prepare":
        result = prepare(root)
    elif args.stage == "verify":
        result = _split(root)
    elif args.stage == "audit_teacher":
        result = audit_teacher_tokens(root)
    elif args.stage.startswith("aggregate_"):
        result = aggregate(root, args.stage.split("_", 1)[1])
    else:
        result = run(root, args.stage, args.limit)
    if args.stage in ("amend_actions", "amend_state_layer", "amend_reference_alignment", "amend_runtime_placement", "amend_teacher_drift", "prepare", "verify"):
        print(result["freeze_digest"])
    else:
        print(json.dumps(result, indent=2))
