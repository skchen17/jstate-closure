"""Finite writable operator, JVP comparison, linearity and restricted rank."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from jclosure.config import load_config
from jclosure.datasets_v8 import load_tasks
from jclosure.experiments import geometry_v13 as geometry
from jclosure.experiments.actuation_v15 import apply, components, load_context, transfer_metrics
from jclosure.experiments.jvp_v12 import _spectrum
from jclosure.experiments.numerics_v14 import _evaluate, _panels, _slices
from jclosure.experiments.persistent_channels_v7 import _prefill, _teacher_tokens
from jclosure.protocol_v15 import stage_freeze, verify
from jclosure.provenance import sha256_file, write_json_atomic

OUT = Path("results/v15/processed")
TARGETS = ("j", "logits", "semantic_continuous", "workspace")


def cosine(a: np.ndarray, b: np.ndarray) -> float | None:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denom) if denom > 1e-20 else None


def _select(row: dict[str, torch.Tensor], channel: str) -> dict[str, torch.Tensor]:
    return {
        name: value if channel == "joint" or channel == name or (channel == "kv" and name == "kv") else torch.zeros_like(value)
        for name, value in row.items()
    }


def _setup(root: Path) -> tuple[dict[str, Any], Any, Any, dict[str, Any], dict[str, Any], list[int], list[int], list[int]]:
    frozen = verify(root)
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    bundle, dense_map, config, metadata, values = load_context(root)
    v8 = config["persistent_state_v8"]
    measured = [int(x) for x in v8["intervention"]["measured_layers"]]
    recurrent = [int(x) for x in v8["persistent_components"]["recurrent_layers"]]
    attention = [int(x) for x in v8["persistent_components"]["attention_layers"]]
    return frozen, bundle, dense_map, metadata, values, measured, recurrent, attention


def _anchor(root: Path, declaration: dict[str, Any], bundle: Any, dense_map: Any, metadata: dict[str, Any], values: dict[str, Any], measured: list[int]) -> tuple[Any, dict[str, Any], dict[str, np.ndarray], np.ndarray, dict[str, np.ndarray]]:
    base_id = str(declaration["base_trial_id"])
    pair = metadata[base_id]
    task = values["tasks"][str(pair["prompt_id"])]
    config = load_config(root / "configs/causal_geometry_v13.yaml")
    v8 = config["persistent_state_v8"]
    clean = _prefill(
        bundle, task.prompt, measured_layers=measured, dense_map=dense_map,
        intervention_layer=int(v8["intervention"]["layer"]), candidate=None,
    )
    token = _teacher_tokens(bundle, clean, count=1, measured_layers=measured, dense_map=dense_map)[0]
    with np.load(root / declaration["matrix_path"], allow_pickle=False) as payload:
        if sha256_file(root / declaration["matrix_path"]) != declaration["matrix_sha256"]:
            raise RuntimeError("V13 frozen matrix hash mismatch")
        matrix = payload["matrix"].astype(np.float64)
        j_ids = payload["selected_j"].astype(int)
        logit_ids = payload["selected_logits"].astype(int)
        slices = _slices(payload)
    jvp = config["causal_geometry_v13"]["jvp"]
    kwargs = dict(
        bundle=bundle, dense_map=dense_map, token=token,
        prompt_length=int(clean["prompt_length"]), selected_j=j_ids,
        selected_logits=logit_ids,
        workspace_layers=[int(x) for x in jvp["workspace_layers"]],
        workspace_count=int(jvp["selected_workspace_count"]),
        main_layer=max(measured),
    )
    baseline = _evaluate(cache=clean["cache"], **kwargs)
    return clean["cache"], kwargs, baseline, matrix, slices


def target_scales(root: Path, panel: list[dict[str, Any]]) -> dict[str, float]:
    norms: dict[str, list[float]] = {name: [] for name in TARGETS}
    for anchor in panel:
        with np.load(root / anchor["matrix_path"], allow_pickle=False) as payload:
            matrix = payload["matrix"].astype(np.float64)
            slices = _slices(payload)
            for name in TARGETS:
                block = matrix[slices[name], :]
                norms[name].extend(np.linalg.norm(block, axis=0).tolist())
    return {name: max(float(np.median(data)), 1e-12) for name, data in norms.items()}


def stack(values: dict[str, np.ndarray], scales: dict[str, float]) -> np.ndarray:
    return np.concatenate([values[name] / (scales[name] * math.sqrt(len(values[name]))) for name in TARGETS])


def _effects(cache: Any, kwargs: dict[str, Any], baseline: dict[str, np.ndarray], row: dict[str, torch.Tensor], recurrent: list[int], attention: list[int], epsilon: float, mode: str = "native_fp32_add_bf16_writeback") -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray], Any, Any]:
    plus_cache = apply(cache, epsilon, row, recurrent, attention, mode)
    minus_cache = apply(cache, -epsilon, row, recurrent, attention, mode)
    plus = _evaluate(cache=plus_cache, **kwargs)
    minus = _evaluate(cache=minus_cache, **kwargs)
    return ({name: plus[name] - baseline[name] for name in TARGETS},
            {name: minus[name] - baseline[name] for name in TARGETS},
            {name: (plus[name] - minus[name]) / (2 * epsilon) for name in TARGETS},
            plus_cache, minus_cache)


def _realization(cache: Any, edited: Any, row: dict[str, torch.Tensor], recurrent: list[int], attention: list[int], epsilon: float) -> dict[str, float | None]:
    requested_sq = realized_sq = dot = 0.0
    zero = survival = below = count = 0.0
    for channel, layer, name, old, direction in components(cache, row, recurrent, attention):
        new = getattr(edited.layers[layer], name)
        if channel in ("keys", "values"):
            new = new[..., : old.shape[-2], :]
        metrics = transfer_metrics(old, direction, new, epsilon)
        a = float(metrics["requested_norm"])
        b = float(metrics["realized_norm"])
        requested_sq += a * a
        realized_sq += b * b
        dot += (metrics["cosine"] or 0) * a * b
        size = float(metrics["element_count"] if "element_count" in metrics else old.numel())
        zero += float(metrics["zero_fraction"]) * size
        survival += float(metrics["surviving_fraction"]) * size
        below += float(metrics["below_one_ulp_fraction"]) * size
        count += size
    norm_requested, norm_realized = math.sqrt(requested_sq), math.sqrt(realized_sq)
    return {
        "requested_norm": norm_requested, "realized_norm": norm_realized,
        "gain": norm_realized / norm_requested if norm_requested else None,
        "cosine": dot / (norm_requested * norm_realized) if norm_requested * norm_realized else None,
        "zero_fraction": zero / count if count else None,
        "surviving_fraction": survival / count if count else None,
        "below_one_ulp_fraction": below / count if count else None,
    }


def pilot(root: Path) -> dict[str, Any]:
    frozen = verify(root)
    config = frozen["config"]
    section = config["operator"]
    stage_path = root / "artifacts/quantization_aware_actuation_v15_pilot.freeze.json"
    if not stage_path.exists():
        stage = stage_freeze(root, "pilot", ["src/jclosure/experiments/operator_v15.py", "results/v15/processed/actuator_transfer_v15.parquet"], {"role": "diagnostic_train", "selection_rule": "first candidate with both signed J effects >= V14 threshold, nonzero writeback, state gain in frozen interval, and signed-effect norm ratio <=5; no post-hoc epsilon adjustment"})
    else:
        stage = json.loads(stage_path.read_text())
    frozen, bundle, dense_map, metadata, values, measured, recurrent, attention = _setup(root)
    panel = _panels(root, int(config["diagnostic"]["anchors_per_family"]))
    scales = target_scales(root, panel)
    rows: list[dict[str, Any]] = []
    for anchor_number, declaration in enumerate(panel):
        cache, kwargs, baseline, matrix, slices = _anchor(root, declaration, bundle, dense_map, metadata, values, measured)
        clean_repeat = _evaluate(cache=cache, **kwargs)
        clean_noise_j = float(np.linalg.norm(clean_repeat["j"] - baseline["j"]))
        for direction_index in section["pilot_direction_indices"]:
            raw = {name: values["directions"][name][int(direction_index)] for name in ("recurrent", "conv", "kv")}
            for channel in section["channel_modes"]:
                row = _select(raw, channel)
                selected = False
                for epsilon in section["adaptive_epsilon_candidates"]:
                    epsilon = float(epsilon)
                    plus, minus, response, plus_cache, minus_cache = _effects(cache, kwargs, baseline, row, recurrent, attention, epsilon)
                    realized = _realization(cache, plus_cache, row, recurrent, attention, epsilon)
                    j_plus = float(np.linalg.norm(plus["j"]))
                    j_minus = float(np.linalg.norm(minus["j"]))
                    effect_ratio = max(j_plus, j_minus) / max(min(j_plus, j_minus), 1e-20)
                    gain = realized["gain"] or 0.0
                    eligible = bool(
                        realized["realized_norm"] > config["diagnostic"]["realized_state_deadzone_norm"]
                        and min(j_plus, j_minus) >= config["diagnostic"]["min_causal_effect_norm"]
                        and min(j_plus, j_minus) / max(clean_noise_j, 1e-20) >= section["target_effect_snr_min"]
                        and config["diagnostic"]["saturation_gain_min"] <= gain <= config["diagnostic"]["saturation_gain_max"]
                        and effect_ratio <= 5.0
                    )
                    ideals = {name: matrix[slices[name], int(direction_index)] for name in TARGETS}
                    ideals["stacked_normalized"] = stack(ideals, scales)
                    finite_by_target = {**response, "stacked_normalized": stack(response, scales)}
                    plus_by_target = {**plus, "stacked_normalized": stack(plus, scales)}
                    minus_by_target = {**minus, "stacked_normalized": stack(minus, scales)}
                    for target in (*TARGETS, "stacked_normalized"):
                        ideal = ideals[target]
                        finite = finite_by_target[target]
                        rows.append({
                            "role": "diagnostic_train", "freeze_digest": stage["freeze_digest"],
                            "base_trial_id": str(declaration["base_trial_id"]), "family": str(declaration["family"]),
                            "direction_index": int(direction_index), "channel": channel,
                            "epsilon": epsilon, "selected": eligible and not selected,
                            "eligible": eligible, "target": target,
                            "j_plus_effect_norm": j_plus, "j_minus_effect_norm": j_minus,
                            "clean_repeat_noise_j": clean_noise_j,
                            "signed_effect_ratio": effect_ratio,
                            "ideal_norm": float(np.linalg.norm(ideal)), "finite_norm": float(np.linalg.norm(finite)),
                            "cosine": cosine(ideal, finite),
                            "relative_l2": float(np.linalg.norm(finite - ideal) / max(float(np.linalg.norm(ideal)), 1e-20)),
                            "norm_ratio": float(np.linalg.norm(finite) / max(float(np.linalg.norm(ideal)), 1e-20)),
                            "signed_projection": float(np.dot(ideal, finite) / max(float(np.linalg.norm(ideal)) ** 2, 1e-20)),
                            "plus_effect_norm": float(np.linalg.norm(plus_by_target[target])),
                            "minus_effect_norm": float(np.linalg.norm(minus_by_target[target])),
                            "realized_state_cosine": realized["cosine"],
                            "realized_state_gain": gain,
                            "realized_state_norm": realized["realized_norm"],
                            "requested_state_norm": realized["requested_norm"],
                            "realized_state_surviving_fraction": realized["surviving_fraction"],
                        })
                    del plus_cache, minus_cache
                    if eligible:
                        selected = True
                        break
        write_json_atomic(root / OUT / "pilot_progress_v15.json", {"completed_anchors": anchor_number + 1, "total_anchors": len(panel)})
    frame = pd.DataFrame(rows)
    path = root / OUT / "autograd_vs_actuator_v15.parquet"
    frame.to_parquet(path, index=False, compression="zstd")
    selected = frame[frame.selected]
    summary = {
        "role": "diagnostic_train", "freeze_digest": stage["freeze_digest"],
        "records": str(path), "records_sha256": sha256_file(path),
        "target_scales": scales,
        "selected_direction_channel_count": int(selected.drop_duplicates(["base_trial_id", "direction_index", "channel"]).shape[0]),
        "requested_direction_channel_count": int(len(panel) * len(section["pilot_direction_indices"]) * len(section["channel_modes"])),
        "medians_by_target_channel": selected.groupby(["channel", "target"]).agg(cosine=("cosine", "median"), relative_l2=("relative_l2", "median"), norm_ratio=("norm_ratio", "median"), realized_state_cosine=("realized_state_cosine", "median"), count=("cosine", "size")).reset_index().to_dict("records"),
    }
    write_json_atomic(root / OUT / "autograd_vs_actuator_v15.json", summary)
    return summary


def _rank_summary(matrix: np.ndarray) -> dict[str, Any]:
    data, _ = _spectrum(matrix, [0.9, 0.95, 0.99])
    return {key: data[key] for key in ("stable_rank", "effective_rank", "rank_90", "rank_95", "rank_99", "singular_values")}


def rank(root: Path) -> dict[str, Any]:
    frozen = verify(root)
    section = frozen["config"]["operator"]
    stage_path = root / "artifacts/quantization_aware_actuation_v15_rank.freeze.json"
    if not stage_path.exists():
        stage = stage_freeze(root, "rank", ["src/jclosure/experiments/operator_v15.py", "results/v15/processed/autograd_vs_actuator_v15.parquet"], {"role": "diagnostic_train", "fixed_finite_scale": section["rank_response_epsilon"], "rank_probe_counts": section["rank_probe_counts"], "rank_anchor_count": section["rank_anchor_count"], "interpretation": "restricted frozen 512-probe domain, not intrinsic full-state rank"})
    else:
        stage = json.loads(stage_path.read_text())
    frozen, bundle, dense_map, metadata, values, measured, recurrent, attention = _setup(root)
    panel = _panels(root, int(frozen["config"]["diagnostic"]["anchors_per_family"]))[: int(section["rank_anchor_count"])]
    scales = target_scales(root, panel)
    direction_count = max(int(x) for x in section["rank_probe_counts"])
    # Full 512-probe scaling on the first anchor; 64 probes on all five anchors
    # isolate state stability without pretending five 512-probe maps were measured.
    rows: list[dict[str, Any]] = []
    spectra: list[dict[str, Any]] = []
    for anchor_number, declaration in enumerate(panel):
        cache, kwargs, baseline, matrix, slices = _anchor(root, declaration, bundle, dense_map, metadata, values, measured)
        count = direction_count if anchor_number == 0 else min(section["rank_probe_counts"])
        by_channel: dict[str, list[dict[str, np.ndarray]]] = {x: [] for x in section["channel_modes"]}
        for direction_index in range(count):
            raw = {name: values["directions"][name][direction_index] for name in ("recurrent", "conv", "kv")}
            for channel in section["channel_modes"]:
                row = _select(raw, channel)
                plus, minus, response, plus_cache, minus_cache = _effects(cache, kwargs, baseline, row, recurrent, attention, float(section["rank_response_epsilon"]))
                realized = _realization(cache, plus_cache, row, recurrent, attention, float(section["rank_response_epsilon"]))
                by_channel[channel].append(response)
                rows.append({
                    "role": "diagnostic_train", "freeze_digest": stage["freeze_digest"],
                    "base_trial_id": str(declaration["base_trial_id"]), "family": str(declaration["family"]),
                    "direction_index": direction_index, "channel": channel,
                    "epsilon": float(section["rank_response_epsilon"]),
                    "realized_state_cosine": realized["cosine"], "realized_state_gain": realized["gain"],
                    "realized_state_norm": realized["realized_norm"],
                    **{f"{name}_response_norm": float(np.linalg.norm(response[name])) for name in TARGETS},
                    **{f"{name}_response": response[name].astype(np.float32).tolist() for name in TARGETS},
                    **{f"{name}_plus_norm": float(np.linalg.norm(plus[name])) for name in TARGETS},
                    **{f"{name}_minus_norm": float(np.linalg.norm(minus[name])) for name in TARGETS},
                })
                del plus_cache, minus_cache
            if (direction_index + 1) % 32 == 0:
                write_json_atomic(root / OUT / "rank_progress_v15.json", {"completed_anchor": anchor_number + 1, "completed_directions": direction_index + 1, "total_directions": count})
        for n in section["rank_probe_counts"]:
            n = int(n)
            if n > count:
                continue
            for channel, responses in by_channel.items():
                for target in (*TARGETS, "stacked_normalized"):
                    if target == "stacked_normalized":
                        finite = np.column_stack([stack(x, scales) for x in responses[:n]])
                        ideal = np.column_stack([stack({name: matrix[slices[name], i] for name in TARGETS}, scales) for i in range(n)]) if channel == "joint" else None
                    else:
                        finite = np.column_stack([x[target] / scales[target] for x in responses[:n]])
                        ideal = matrix[slices[target], :n] / scales[target] if channel == "joint" else None
                    spectra.append({"base_trial_id": str(declaration["base_trial_id"]), "family": str(declaration["family"]), "channel": channel, "target": target, "direction_count": n, "operator": "finite_writeback", **_rank_summary(finite)})
                    if ideal is not None:
                        spectra.append({"base_trial_id": str(declaration["base_trial_id"]), "family": str(declaration["family"]), "channel": channel, "target": target, "direction_count": n, "operator": "autograd_jvp", **_rank_summary(ideal)})
        # Write each completed anchor before the next expensive prefill.
        (root / OUT).mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_parquet(root / OUT / "finite_operator_columns_v15.parquet", index=False, compression="zstd")
        write_json_atomic(root / OUT / "finite_operator_spectra_v15.json", {"role": "diagnostic_train", "freeze_digest": stage["freeze_digest"], "rows": spectra})
    records = root / OUT / "finite_operator_columns_v15.parquet"
    spectrum_path = root / OUT / "finite_operator_spectra_v15.json"
    summary = {"role": "diagnostic_train", "freeze_digest": stage["freeze_digest"], "columns": str(records), "columns_sha256": sha256_file(records), "spectra": str(spectrum_path), "spectra_sha256": sha256_file(spectrum_path), "target_scales": scales, "rank_512_anchor_count": 1, "rank_64_anchor_count": len(panel), "rank_is_restricted_probe_rank": True}
    write_json_atomic(root / OUT / "writable_causal_rank_v15.json", summary)
    return summary


def linearity(root: Path) -> dict[str, Any]:
    frozen = verify(root)
    section = frozen["config"]["operator"]
    stage_path = root / "artifacts/quantization_aware_actuation_v15_linearity.freeze.json"
    if not stage_path.exists():
        stage = stage_freeze(root, "linearity", ["src/jclosure/experiments/operator_v15.py", "results/v15/processed/autograd_vs_actuator_v15.parquet"], {"role": "diagnostic_train", "pairs": [[0, 103], [206, 308]], "epsilon": 1.0, "tests": ["odd", "additive", "homogeneous", "cross_channel"], "gate": {key: section[key] for key in ("linearity_cosine_min", "linearity_relative_error_max", "linearity_norm_ratio_min", "linearity_norm_ratio_max")}})
    else:
        stage = json.loads(stage_path.read_text())
    frozen, bundle, dense_map, metadata, values, measured, recurrent, attention = _setup(root)
    panel = _panels(root, int(frozen["config"]["diagnostic"]["anchors_per_family"]))
    scales = target_scales(root, panel)
    rows = []
    for declaration in panel:
        cache, kwargs, baseline, _, _ = _anchor(root, declaration, bundle, dense_map, metadata, values, measured)
        for first, second in ((0, 103), (206, 308)):
            r1 = {name: values["directions"][name][first] for name in ("recurrent", "conv", "kv")}
            r2 = {name: values["directions"][name][second] for name in ("recurrent", "conv", "kv")}
            add = {name: r1[name] + r2[name] for name in r1}
            double = {name: 2 * r1[name] for name in r1}
            effects = {}
            for label, row in (("v1", r1), ("v2", r2), ("sum", add), ("double", double), ("rec", _select(r1, "recurrent")), ("conv", _select(r1, "conv")), ("kv", _select(r1, "kv")), ("rec_conv", {name: r1[name] if name != "kv" else torch.zeros_like(r1[name]) for name in r1})):
                plus, minus, _, plus_cache, minus_cache = _effects(cache, kwargs, baseline, row, recurrent, attention, 1.0)
                effects[label] = (stack(plus, scales), stack(minus, scales))
                del plus_cache, minus_cache
            comparisons = {
                "odd": (effects["v1"][0], -effects["v1"][1]),
                "additive": (effects["sum"][0], effects["v1"][0] + effects["v2"][0]),
                "homogeneous": (effects["double"][0], 2 * effects["v1"][0]),
                "rec_conv": (effects["rec_conv"][0], effects["rec"][0] + effects["conv"][0]),
                "rec_conv_kv": (effects["v1"][0], effects["rec"][0] + effects["conv"][0] + effects["kv"][0]),
            }
            for name, (observed, predicted) in comparisons.items():
                norm_observed = float(np.linalg.norm(observed))
                norm_predicted = float(np.linalg.norm(predicted))
                cos = cosine(observed, predicted)
                relative = float(np.linalg.norm(observed - predicted) / max(norm_predicted, 1e-20))
                ratio = norm_observed / max(norm_predicted, 1e-20)
                qualified = min(norm_observed, norm_predicted) >= frozen["config"]["diagnostic"]["min_causal_effect_norm"]
                passed = bool(qualified and cos is not None and cos >= section["linearity_cosine_min"] and relative <= section["linearity_relative_error_max"] and section["linearity_norm_ratio_min"] <= ratio <= section["linearity_norm_ratio_max"])
                rows.append({"role": "diagnostic_train", "freeze_digest": stage["freeze_digest"], "base_trial_id": str(declaration["base_trial_id"]), "pair": f"{first}:{second}", "test": name, "epsilon": 1.0, "cosine": cos, "relative_l2": relative, "norm_ratio": ratio, "qualified": qualified, "passed": passed})
    path = root / OUT / "finite_response_linearity_v15.parquet"
    pd.DataFrame(rows).to_parquet(path, index=False, compression="zstd")
    frame = pd.DataFrame(rows)
    summary = {"role": "diagnostic_train", "freeze_digest": stage["freeze_digest"], "records": str(path), "records_sha256": sha256_file(path), "by_test": frame.groupby("test").agg(qualified=("qualified", "sum"), pass_count=("passed", "sum"), median_cosine=("cosine", "median"), median_relative_l2=("relative_l2", "median"), median_norm_ratio=("norm_ratio", "median")).reset_index().to_dict("records"), "FINITE_RESPONSE_LINEARITY_GATE": bool(frame.qualified.all() and frame.passed.all())}
    write_json_atomic(root / OUT / "finite_response_linearity_v15.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("pilot", "linearity", "rank"), required=True)
    args = parser.parse_args()
    root = Path.cwd()
    result = {"pilot": pilot, "linearity": linearity, "rank": rank}[args.stage](root)
    print(json.dumps(result, sort_keys=True)[:4000])


if __name__ == "__main__":
    main()
