"""Calibrate individual intervention layers after the locked Phase 0 v2 gate."""

from __future__ import annotations

import json
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from jclosure.clamp import ClampThresholds, one_shot_clamp
from jclosure.datasets import (
    TaskExample,
    normalize_prompt,
    task_examples_from_json,
    upstream_multihop,
)
from jclosure.experiments.common import (
    initialize_context,
    require_phase0_v2_gate,
    standard_parser,
)
from jclosure.experiments.validate_lens import _positive_control_records
from jclosure.interventions import non_j_direction, replace_activation, steer_activation
from jclosure.jstate import ConceptVocabulary, JStateEncoder, jstate_similarity
from jclosure.model import load_model_bundle
from jclosure.phase0 import item_mrr_advantages, mean_item_hit
from jclosure.provenance import append_jsonl, write_json_atomic
from jclosure.recorder import ActivationRecorder, ResidualEditor
from jclosure.records import LayerCalibrationRecord
from jclosure.statistics import clustered_bootstrap_ci, numerical_null_threshold


def _edit(
    activation: torch.Tensor,
    layer: int,
    *,
    kind: str,
    value: torch.Tensor,
) -> torch.Tensor:
    del layer
    if kind == "zero":
        return steer_activation(activation, value, strength=0.0, positions=(-1,))
    return replace_activation(activation, value, positions=(-1,))


def _balanced_examples(root: Path, count_per_family: int) -> list[TaskExample]:
    fresh_dir = root / "data/phase0_v2"
    fresh_multi = task_examples_from_json(fresh_dir / "fresh_multihop.json")
    fresh_order = task_examples_from_json(fresh_dir / "fresh_order_ops.json")
    seen = {normalize_prompt(item.prompt) for item in fresh_multi}
    multihop = list(fresh_multi)
    for item in upstream_multihop(root / "data/upstream/anthropic"):
        normalized = normalize_prompt(item.prompt)
        if normalized not in seen:
            multihop.append(item)
            seen.add(normalized)
        if len(multihop) >= count_per_family:
            break
    if len(multihop) < count_per_family or len(fresh_order) < count_per_family:
        raise RuntimeError("insufficient unique states for balanced layer calibration")
    return [*multihop[:count_per_family], *fresh_order[:count_per_family]]


def _candidate_layers(records: pd.DataFrame, band: list[int], threshold: float) -> list[int]:
    families = ("factual_two_hop", "order_of_operations")
    return [
        layer
        for layer in band
        if all(mean_item_hit(records, layer=layer, family=family) >= threshold for family in families)
    ]


def _layer_rank_stats(
    records: pd.DataFrame, layer: int, config: dict[str, Any]
) -> tuple[dict[str, Any] | None, dict[str, float]]:
    item = item_mrr_advantages(records, layers=[layer])
    clusters = records[["example_id", "prompt_cluster"]].drop_duplicates("example_id")
    item = item.merge(clusters, on="example_id", how="left")
    points = item.groupby("family")["mrr_advantage"].mean().to_dict()
    if item["prompt_cluster"].nunique() < 2:
        return None, {str(key): float(value) for key, value in points.items()}
    ci = clustered_bootstrap_ci(
        item,
        cluster_col="prompt_cluster",
        value_col="mrr_advantage",
        n_resamples=int(config["validation"]["bootstrap_resamples"]),
        seed=int(config["reproducibility"]["bootstrap_seed"]) + layer,
    )
    return ci.__dict__, {str(key): float(value) for key, value in points.items()}


def _positive_stats(records: pd.DataFrame, config: dict[str, Any]) -> dict[str, Any]:
    floor = float(config["statistics"]["null_js_floor"])
    null = records[records["strength"] == 0]["target_log_odds_shift"].abs()
    actual = records[records["strength"] == 1]
    threshold = numerical_null_threshold(
        null.to_numpy(),
        floor=floor,
        quantile=float(config["statistics"]["null_quantile"]),
    )
    if actual["example_id"].nunique() < 2:
        return {"passed": False, "ci": None, "null_threshold": threshold}
    ci = clustered_bootstrap_ci(
        actual,
        cluster_col="example_id",
        value_col="target_log_odds_shift",
        n_resamples=int(config["validation"]["bootstrap_resamples"]),
        seed=int(config["reproducibility"]["bootstrap_seed"]),
    )
    return {"passed": ci.lower > threshold, "ci": ci.__dict__, "null_threshold": threshold}


def main() -> None:
    parser = standard_parser(
        "Calibrate closure-eligible measured-J layers",
        "configs/phase0_v2_confirmatory.yaml",
    )
    args = parser.parse_args()
    context = initialize_context("layer-calibration", args)
    try:
        gate = require_phase0_v2_gate(context)
        if args.dry_run:
            context.finish("DRY_RUN", phase0_v2_gate=gate["run_id"])
            return
        readout_path = context.raw_dir / gate["run_id"] / "readout_records_v2.parquet"
        readout = pd.read_parquet(readout_path)
        band = [int(layer) for layer in gate["workspace_band"]]
        threshold = float(context.config["validation"]["hit10_threshold"])
        candidates = _candidate_layers(readout, band, threshold)
        if not candidates:
            write_json_atomic(
                context.processed_dir / "layer_calibration.json",
                {
                    "schema_version": 2,
                    "protocol_version": "phase0_protocol_v2",
                    "phase0_v2_gate_run_id": gate["run_id"],
                    "eligible_layers": [],
                    "reason": "no layer met family-wise per-layer hit@10",
                },
            )
            context.finish("COMPLETED", eligible_layers=[])
            return

        bundle = load_model_bundle(context.config)
        vocabulary = ConceptVocabulary.from_json(
            context.processed_dir / "concept_vocabulary_v2_4096.json"
        )
        encoder = JStateEncoder.from_lens(
            bundle.lens,
            bundle.unembedding_weight,
            vocabulary,
            k=int(context.config["jstate"]["k"]),
            lazy=True,
            protocol_version="phase0_protocol_v2",
            direction_chunk_size=int(context.config["jstate"]["direction_chunk_size"]),
        )
        examples = _balanced_examples(context.root, 100)
        if args.limit is not None:
            examples = examples[: args.limit]
        activations: list[dict[int, torch.Tensor]] = []
        inputs: list[torch.Tensor] = []
        clean_logits: list[torch.Tensor] = []
        for example in examples:
            input_ids = bundle.lens_model.encode(example.prompt, max_length=512)
            with ActivationRecorder(bundle.layers, at=candidates) as recorder:
                with torch.no_grad():
                    logits = bundle.forward_logits(input_ids)[0, -1].float().cpu()
            inputs.append(input_ids)
            clean_logits.append(logits)
            activations.append(
                {
                    layer: recorder.activations[layer][0, -1].detach().float().cpu()
                    for layer in candidates
                }
            )

        # One composed zero-strength run and one composed identity run cover every layer.
        reference_input = inputs[0]
        reference_logits = clean_logits[0]
        zero = {
            layer: partial(
                _edit,
                kind="zero",
                value=encoder.dictionary(layer)[0],
            )
            for layer in candidates
        }
        identity = {
            layer: partial(_edit, kind="identity", value=activations[0][layer])
            for layer in candidates
        }
        with ResidualEditor(bundle.layers, zero):
            with torch.no_grad():
                zero_logits = bundle.forward_logits(reference_input)[0, -1].float().cpu()
        with ResidualEditor(bundle.layers, identity):
            with torch.no_grad():
                identity_logits = bundle.forward_logits(reference_input)[0, -1].float().cpu()
        numerical_pass = bool(
            torch.equal(reference_logits, zero_logits)
            and torch.equal(reference_logits, identity_logits)
        )

        # Exact deterministic reruns quantify numerical J-state stability.
        stability: dict[int, list[float]] = {layer: [] for layer in candidates}
        for index in range(min(8, len(inputs))):
            with ActivationRecorder(bundle.layers, at=candidates) as recorder:
                with torch.no_grad():
                    bundle.forward_logits(inputs[index])
            for layer in candidates:
                first = encoder.encode(activations[index][layer].to(bundle.hf_model.device), layer)
                second_vector = recorder.activations[layer][0, -1].detach().float()
                second = encoder.encode(second_vector, layer)
                stability[layer].append(jstate_similarity(first, second))

        flexible = json.loads(
            (
                context.root
                / context.config["data"]["upstream_root"]
                / "experiments/flexible-generalization.json"
            ).read_text(encoding="utf-8")
        )
        positive_frames: list[pd.DataFrame] = []
        clamp_records: list[dict[str, Any]] = []
        outputs: list[dict[str, Any]] = []
        device = bundle.hf_model.device
        thresholds = ClampThresholds(
            dense_cosine=float(context.config["jstate"]["dense_cosine_threshold"]),
            top10_overlap=float(context.config["jstate"]["top10_overlap_threshold"]),
            rms_drift=float(context.config["jstate"]["rms_drift_threshold"]),
            min_remainder_fraction=float(context.config["jstate"]["min_remainder_fraction"]),
        )
        for layer in candidates:
            rank_ci, family_points = _layer_rank_stats(readout, layer, context.config)
            positive = pd.DataFrame(
                _positive_control_records(
                    bundle,
                    encoder,
                    vocabulary,
                    flexible,
                    [layer],
                    int(context.config["validation"]["positive_control_trials"]),
                )
            )
            positive["layer"] = layer
            positive_frames.append(positive)
            positive_summary = _positive_stats(positive, context.config)
            passed_clamps = 0
            for index, example in enumerate(examples):
                donor_index = (
                    (index + len(examples) // 2) % len(examples)
                    if len(examples) > 1
                    else index
                )
                clean = activations[index][layer].to(device)
                donor = activations[donor_index][layer].to(device)
                difference = donor - clean
                natural_scale = float(torch.linalg.vector_norm(difference.float()).item())
                stripped, _ = non_j_direction(
                    difference,
                    encoder.dictionary(layer, device),
                    k=encoder.k,
                )
                stripped_norm = torch.linalg.vector_norm(stripped.float())
                if natural_scale <= 1e-12 or float(stripped_norm) <= 1e-12:
                    clamp_records.append(
                        {
                            "schema_version": 2,
                            "layer": layer,
                            "prompt_id": example.example_id,
                            "task_family": example.family,
                            "valid": False,
                            "exclusion_reason": "degenerate_natural_difference",
                        }
                    )
                    continue
                delta = stripped.float() * (0.25 * natural_scale / float(stripped_norm))
                result = one_shot_clamp(
                    clean,
                    clean + delta,
                    layer=layer,
                    encoder=encoder,
                    thresholds=thresholds,
                    natural_scale=natural_scale,
                )
                passed_clamps += int(result.passed)
                clamp_records.append(
                    {
                        "schema_version": 2,
                        "layer": layer,
                        "prompt_id": example.example_id,
                        "task_family": example.family,
                        "valid": result.passed,
                        "exclusion_reason": None if result.passed else ",".join(result.failure_reasons),
                        "dense_cosine": result.dense_cosine,
                        "top10_overlap": result.top10_overlap,
                        "rms_drift": result.activation_rms_drift,
                        "remainder_fraction": result.remainder_fraction,
                    }
                )
            clamp_rate = passed_clamps / max(len(examples), 1)
            rank_lower = None if rank_ci is None else float(rank_ci["lower"])
            stability_median = float(np.median(stability[layer])) if stability[layer] else 0.0
            reasons: list[str] = []
            multihop_hit = mean_item_hit(
                readout, layer=layer, family="factual_two_hop"
            )
            order_hit = mean_item_hit(
                readout, layer=layer, family="order_of_operations"
            )
            if multihop_hit < threshold or order_hit < threshold:
                reasons.append("family_hit10")
            if rank_lower is None or rank_lower <= 0 or any(value < 0 for value in family_points.values()):
                reasons.append("rank_advantage")
            if not positive_summary["passed"]:
                reasons.append("positive_control")
            if clamp_rate < 0.80:
                reasons.append("clamp_valid_rate")
            if not numerical_pass:
                reasons.append("numerical_hooks")
            if stability_median < 0.999:
                reasons.append("deterministic_stability")
            record = LayerCalibrationRecord(
                layer=layer,
                multihop_hit10=multihop_hit,
                order_ops_hit10=order_hit,
                rank_advantage_ci_lower=rank_lower,
                positive_control_ci_lower=(
                    None if positive_summary["ci"] is None else positive_summary["ci"]["lower"]
                ),
                clamp_valid_rate=clamp_rate,
                numerical_checks_passed=numerical_pass,
                eligible=not reasons,
                reasons=tuple(reasons),
            )
            outputs.append(
                {
                    **record.__dict__,
                    "rank_advantage_ci": rank_ci,
                    "rank_advantage_by_family": family_points,
                    "positive_control": positive_summary,
                    "stability_median_dense_cosine": stability_median,
                    "stability_values": stability[layer],
                    "clamp_attempts": len(examples),
                    "clamp_valid": passed_clamps,
                }
            )

        run_dir = context.raw_dir / context.run_id
        append_jsonl(run_dir / "clamp_calibration.jsonl", clamp_records)
        pd.concat(positive_frames, ignore_index=True).to_parquet(
            run_dir / "positive_control_by_layer.parquet", index=False
        )
        eligible_layers = [int(item["layer"]) for item in outputs if item["eligible"]]
        artifact = {
            "schema_version": 2,
            "protocol_version": "phase0_protocol_v2",
            "run_id": context.run_id,
            "phase0_v2_gate_run_id": gate["run_id"],
            "candidate_layers": candidates,
            "eligible_layers": eligible_layers,
            "criteria": {
                "family_hit10": threshold,
                "rank_advantage_ci_lower": 0.0,
                "positive_control_above_null": True,
                "clamp_valid_rate": 0.80,
                "stability_dense_cosine": 0.999,
                "numerical_hooks_exact": True,
            },
            "numerical": {
                "zero_max_logit_error": float(torch.max(torch.abs(reference_logits - zero_logits))),
                "identity_max_logit_error": float(
                    torch.max(torch.abs(reference_logits - identity_logits))
                ),
                "passed": numerical_pass,
            },
            "layers": outputs,
        }
        write_json_atomic(context.processed_dir / "layer_calibration.json", artifact)
        pd.json_normalize(outputs, sep=".").to_parquet(
            context.processed_dir / "layer_calibration.parquet", index=False
        )
        report = [
            "# Closure-eligible layer calibration",
            "",
            "## Material Passport",
            "",
            f"- Run ID: `{context.run_id}`",
            "- Verification Status: VERIFIED",
            f"- Phase 0 v2 gate: `{gate['run_id']}`",
            f"- Readout candidates: `{candidates}`",
            f"- Closure-eligible layers: `{eligible_layers}`",
            "",
            "## Per-layer records",
            "",
            f"`{json.dumps(outputs, sort_keys=True)}`",
            "",
        ]
        (context.reports_dir / "LAYER_CALIBRATION.md").write_text(
            "\n".join(report), encoding="utf-8"
        )
        context.finish("COMPLETED", candidate_layers=candidates, eligible_layers=eligible_layers)
    except KeyboardInterrupt:
        context.finish("FAILED", error="KeyboardInterrupt: run cancelled")
        raise
    except Exception as exc:
        context.finish("FAILED", error=f"{type(exc).__name__}: {exc}")
        raise


if __name__ == "__main__":
    main()
