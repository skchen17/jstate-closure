"""Phase 0: validate J-lens readout, identify a workspace band, and gate use."""

from __future__ import annotations

import hashlib
import json
import sys
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from jclosure.clamp import ClampThresholds, one_shot_clamp
from jclosure.config import config_digest
from jclosure.datasets import upstream_multihop, upstream_order_ops
from jclosure.experiments.common import (
    concept_vocabulary_path,
    initialize_context,
    phase0_gate_path,
    standard_parser,
)
from jclosure.interventions import (
    coordinate_swap_activation,
    matched_random_direction,
    non_j_direction,
    replace_activation,
    steer_activation,
)
from jclosure.jstate import JStateEncoder, build_concept_vocabulary
from jclosure.metrics import jensen_shannon_from_logits, token_log_odds
from jclosure.model import load_model_bundle
from jclosure.provenance import sha256_file, write_json_atomic
from jclosure.recorder import ActivationRecorder, ResidualEditor
from jclosure.statistics import clustered_bootstrap_ci, numerical_null_threshold


def _split(example_id: str, discovery_fraction: float) -> str:
    value = int(hashlib.sha256(example_id.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    return "discovery" if value < discovery_fraction else "validation"


def _single_token_ids(tokenizer: Any, text: str) -> list[int]:
    found: list[int] = []
    for variant in (text, " " + text.strip()):
        ids = tokenizer.encode(variant, add_special_tokens=False)
        if len(ids) == 1 and int(ids[0]) not in found:
            found.append(int(ids[0]))
    return found


def _rank(logits: torch.Tensor, token_ids: list[int]) -> int | None:
    if not token_ids:
        return None
    scores = logits.float().reshape(-1)
    ids = torch.tensor(token_ids, dtype=torch.long)
    best = scores[ids].max()
    return int(torch.sum(scores > best).item()) + 1


def _load_flexible(path: Path) -> dict[str, Any]:
    return json.loads((path / "experiments/flexible-generalization.json").read_text())


def _mandatory_concepts(
    multihop, order_ops, flexible: dict[str, Any]
) -> list[str]:
    values: set[str] = set()
    for example in [*multihop, *order_ops]:
        values.add(example.answer)
        values.update(example.intermediates)
    for category in flexible["categories"]:
        values.update(str(value) for value in category["args"])
        for function in category["funcs"]:
            values.update(str(value) for value in function["answers"].values())
            values.add(str(function["name"]))
    values.update(["addition", "subtraction", "multiplication", "division"])
    return sorted(values)


def _readout_records(bundle, examples, family: str, config: dict[str, Any], limit: int | None):
    records: list[dict[str, Any]] = []
    layers = bundle.lens.source_layers
    min_position = int(config["validation"]["min_primary_position"])
    discovery_fraction = float(config["validation"]["discovery_fraction"])
    selected = examples if limit is None else examples[:limit]
    for example in selected:
        input_ids = bundle.lens_model.encode(
            example.prompt, max_length=int(config["model"].get("max_seq_len", 512))
        )
        position = int(input_ids.shape[1] - 1)
        lens_logits, _, _ = bundle.lens.apply(
            bundle.lens_model,
            example.prompt,
            layers=layers,
            positions=[-1],
            max_seq_len=int(config["model"].get("max_seq_len", 512)),
            use_jacobian=True,
        )
        logit_logits, _, _ = bundle.lens.apply(
            bundle.lens_model,
            example.prompt,
            layers=layers,
            positions=[-1],
            max_seq_len=int(config["model"].get("max_seq_len", 512)),
            use_jacobian=False,
        )
        for concept in example.intermediates:
            token_ids = _single_token_ids(bundle.tokenizer, concept)
            copied = concept.casefold() in example.prompt.casefold()
            primary = position >= min_position and not copied and bool(token_ids)
            for layer in layers:
                for method, logits_by_layer in (
                    ("jacobian", lens_logits),
                    ("logit", logit_logits),
                ):
                    rank = _rank(logits_by_layer[layer][0], token_ids)
                    records.append(
                        {
                            "example_id": example.example_id,
                            "template_id": example.template_id,
                            "family": family,
                            "concept": concept,
                            "layer": int(layer),
                            "position": position,
                            "method": method,
                            "rank": rank,
                            "hit1": bool(rank is not None and rank <= 1),
                            "hit5": bool(rank is not None and rank <= 5),
                            "hit10": bool(rank is not None and rank <= 10),
                            "primary": primary,
                            "excluded_first16": position < min_position,
                            "excluded_literal_copy": copied,
                            "split": _split(example.example_id, discovery_fraction),
                        }
                    )
    return records


def select_workspace_band(
    records: pd.DataFrame,
    *,
    peak_fraction: float,
    min_layers: int,
) -> tuple[list[int], pd.DataFrame]:
    primary = records[
        records["primary"]
        & (records["split"] == "discovery")
        & records["rank"].notna()
    ].copy()
    if primary.empty:
        return [], pd.DataFrame()
    primary["reciprocal_rank"] = 1.0 / primary["rank"].astype(float)
    pivot = primary.pivot_table(
        index=["example_id", "concept", "family", "layer"],
        columns="method",
        values="reciprocal_rank",
        aggfunc="first",
    ).dropna(subset=["jacobian", "logit"])
    pivot["advantage"] = pivot["jacobian"] - pivot["logit"]
    layer_scores = (
        pivot.reset_index()
        .groupby("layer", as_index=False)["advantage"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .rename(
            columns={
                "mean": "advantage",
                "std": "prompt_stability_std",
                "count": "prompt_concept_count",
            }
        )
        .sort_values("layer")
    )
    layer_scores["smoothed_advantage"] = layer_scores["advantage"].rolling(
        3, center=True, min_periods=1
    ).mean()
    if layer_scores.empty or float(layer_scores["smoothed_advantage"].max()) <= 0:
        return [], layer_scores
    peak_row = layer_scores.loc[layer_scores["smoothed_advantage"].idxmax()]
    peak_layer = int(peak_row["layer"])
    threshold = peak_fraction * float(peak_row["smoothed_advantage"])
    eligible = {
        int(row.layer)
        for row in layer_scores.itertuples()
        if row.smoothed_advantage >= threshold and row.advantage >= 0
    }
    if peak_layer not in eligible:
        return [], layer_scores
    band = [peak_layer]
    while band[0] - 1 in eligible:
        band.insert(0, band[0] - 1)
    while band[-1] + 1 in eligible:
        band.append(band[-1] + 1)
    return (band if len(band) >= min_layers else []), layer_scores


def _positive_control_records(
    bundle,
    encoder: JStateEncoder,
    vocabulary,
    flexible: dict[str, Any],
    band: list[int],
    max_trials: int,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    token_to_index = {token: index for index, token in enumerate(vocabulary.token_ids)}
    trial_index = 0
    for category in flexible["categories"]:
        args = list(category["args"])
        for function in category["funcs"]:
            for source_index, source in enumerate(args):
                target = args[(source_index + 1) % len(args)]
                prompt = str(function["template"]).format(arg=source)
                source_ids = _single_token_ids(bundle.tokenizer, source)
                target_ids = _single_token_ids(bundle.tokenizer, target)
                answer_ids = _single_token_ids(bundle.tokenizer, function["answers"][source])
                swapped_answer_ids = _single_token_ids(bundle.tokenizer, function["answers"][target])
                if not source_ids or not target_ids or not answer_ids or not swapped_answer_ids:
                    continue
                source_id = next((value for value in source_ids if value in token_to_index), None)
                target_id = next((value for value in target_ids if value in token_to_index), None)
                if source_id is None or target_id is None:
                    continue
                input_ids = bundle.lens_model.encode(prompt, max_length=512)
                with torch.no_grad():
                    clean_logits = bundle.forward_logits(input_ids)[0, -1].float().cpu()
                if int(torch.argmax(clean_logits)) not in answer_ids:
                    continue
                baseline = token_log_odds(clean_logits, swapped_answer_ids[0])
                for strength in (0.0, 1.0):
                    transforms = {}
                    for layer in band:
                        dictionary = encoder.dictionary(layer)
                        source_direction = dictionary[token_to_index[source_id]]
                        target_direction = dictionary[token_to_index[target_id]]
                        transforms[layer] = partial(
                            _swap_transform,
                            source_direction=source_direction,
                            target_direction=target_direction,
                            strength=strength,
                        )
                    with ResidualEditor(bundle.layers, transforms):
                        with torch.no_grad():
                            logits = bundle.forward_logits(input_ids)[0, -1].float().cpu()
                    records.append(
                        {
                            "example_id": f"{category['name']}:{function['name']}:{source}->{target}",
                            "category": category["name"],
                            "function": function["name"],
                            "source": source,
                            "target": target,
                            "strength": strength,
                            "target_log_odds_shift": token_log_odds(
                                logits, swapped_answer_ids[0]
                            )
                            - baseline,
                            "target_rank": _rank(logits, swapped_answer_ids),
                        }
                    )
                trial_index += 1
                if trial_index >= max_trials:
                    return records
    return records


def _swap_transform(
    activation: torch.Tensor,
    layer: int,
    *,
    source_direction: torch.Tensor,
    target_direction: torch.Tensor,
    strength: float,
    positions: tuple[int, ...] | None = None,
) -> torch.Tensor:
    del layer
    return coordinate_swap_activation(
        activation,
        source_direction,
        target_direction,
        strength=strength,
        positions=positions,
    )


def _edit_final(
    activation: torch.Tensor,
    layer: int,
    *,
    kind: str,
    vector: torch.Tensor,
) -> torch.Tensor:
    del layer
    if kind == "steer":
        return steer_activation(activation, vector, strength=1.0, positions=(-1,))
    if kind == "replace":
        return replace_activation(activation, vector, positions=(-1,))
    raise ValueError(kind)


def _gpu_smoke(
    context,
    bundle,
    encoder: JStateEncoder,
    flexible: dict[str, Any],
) -> dict[str, Any]:
    """Exercise real model hooks before launching the full validation job."""

    function = flexible["categories"][0]["funcs"][0]
    argument = flexible["categories"][0]["args"][0]
    prompt = str(function["template"]).format(arg=argument)
    input_ids = bundle.lens_model.encode(prompt, max_length=512)
    layer = int(bundle.lens.source_layers[len(bundle.lens.source_layers) // 2])
    with ActivationRecorder(bundle.layers, at=[layer]) as recorder:
        with torch.no_grad():
            clean_logits = bundle.forward_logits(input_ids)[0, -1].float().cpu()
    clean_activation = recorder.activations[layer][0, -1].detach().float().cpu()
    direction = encoder.dictionary(layer)[0]

    zero_transform = partial(_edit_final, kind="steer", vector=direction * 0)
    with ResidualEditor(bundle.layers, {layer: zero_transform}):
        with torch.no_grad():
            zero_logits = bundle.forward_logits(input_ids)[0, -1].float().cpu()

    identical_transform = partial(
        _edit_final, kind="replace", vector=clean_activation
    )
    with ResidualEditor(bundle.layers, {layer: identical_transform}):
        with torch.no_grad():
            identical_logits = bundle.forward_logits(input_ids)[0, -1].float().cpu()

    swap_transform = partial(
        _swap_transform,
        source_direction=encoder.dictionary(layer)[0],
        target_direction=encoder.dictionary(layer)[1],
        strength=1.0,
        positions=(-1,),
    )
    with ResidualEditor(bundle.layers, {layer: swap_transform}), ActivationRecorder(
        bundle.layers, at=[layer]
    ) as swap_recorder:
        with torch.no_grad():
            swap_logits = bundle.forward_logits(input_ids)[0, -1].float().cpu()
    swapped_activation = swap_recorder.activations[layer][0, -1].float().cpu()

    random = matched_random_direction(
        clean_activation,
        seed=context.seed,
        norm=0.20 * float(torch.linalg.vector_norm(clean_activation)),
    )
    stripped, _ = non_j_direction(
        random, encoder.dictionary(layer), k=int(context.config["jstate"]["k"])
    )
    perturbed = clean_activation + stripped
    clamp = one_shot_clamp(
        clean_activation,
        perturbed,
        layer=layer,
        encoder=encoder,
        thresholds=ClampThresholds(min_remainder_fraction=0.0),
    )
    clamp_transform = partial(_edit_final, kind="replace", vector=clamp.activation)
    with ResidualEditor(bundle.layers, {layer: clamp_transform}):
        with torch.no_grad():
            clamp_logits = bundle.forward_logits(input_ids)[0, -1].float().cpu()

    clean_state = encoder.encode(clean_activation, layer)
    swapped_state = encoder.encode(swapped_activation, layer)
    payload = {
        "schema_version": 1,
        "run_id": context.run_id,
        "status": "COMPLETED",
        "model_id": bundle.model_id,
        "model_revision": bundle.model_revision,
        "lens_revision": bundle.lens_revision,
        "lens_sha256": context.config["lens"]["sha256"],
        "lens_checkpoint_n_prompts": int(bundle.lens.n_prompts),
        "lens_fit_metadata_prompts_fitted": context.config["lens"].get(
            "metadata_prompts_fitted"
        ),
        "layer": layer,
        "position": -1,
        "zero_max_logit_error": float(torch.max(torch.abs(clean_logits - zero_logits))),
        "identical_max_logit_error": float(
            torch.max(torch.abs(clean_logits - identical_logits))
        ),
        "swap_max_logit_change": float(torch.max(torch.abs(clean_logits - swap_logits))),
        "swap_j_dense_cosine": float(
            torch.dot(clean_state.dense_scores.float(), swapped_state.dense_scores.float())
        ),
        "clamp_dense_cosine": clamp.dense_cosine,
        "clamp_top10_overlap": clamp.top10_overlap,
        "clamp_rms_drift": clamp.activation_rms_drift,
        "clamp_output_js": jensen_shannon_from_logits(clean_logits, clamp_logits),
        "null_pass": bool(
            torch.equal(clean_logits, zero_logits)
            and torch.equal(clean_logits, identical_logits)
        ),
        "clamp_pass": clamp.passed,
    }
    write_json_atomic(context.raw_dir / context.run_id / "gpu_smoke.json", payload)
    return payload


def _evaluate_gate(
    records: pd.DataFrame,
    band: list[int],
    positive: pd.DataFrame,
    config: dict[str, Any],
) -> dict[str, Any]:
    validation = records[
        records["primary"]
        & (records["split"] == "validation")
        & records["layer"].isin(band)
        & records["rank"].notna()
    ].copy()
    hit_rates: dict[str, float] = {}
    hit_rates_by_layer: dict[str, dict[str, float]] = {}
    for family in ("factual_two_hop", "order_of_operations"):
        subset = validation[
            (validation["family"] == family) & (validation["method"] == "jacobian")
        ]
        if subset.empty:
            hit_rates[family] = 0.0
            hit_rates_by_layer[family] = {}
        else:
            by_layer = {
                str(layer): float((group["rank"] <= 10).mean())
                for layer, group in subset.groupby("layer", sort=True)
            }
            hit_rates_by_layer[family] = by_layer
            hit_rates[family] = min(by_layer.values())

    pivot = validation.pivot_table(
        index=["example_id", "concept", "family"],
        columns="method",
        values="rank",
        aggfunc="min",
    ).dropna(subset=["jacobian", "logit"])
    rank_ci_payload: dict[str, Any] | None = None
    rank_pass = False
    if len(pivot) >= 2:
        differences = pivot.reset_index()
        differences["mrr_advantage"] = (
            1.0 / differences["jacobian"] - 1.0 / differences["logit"]
        )
        ci = clustered_bootstrap_ci(
            differences,
            cluster_col="example_id",
            value_col="mrr_advantage",
            n_resamples=int(config["validation"]["bootstrap_resamples"]),
            seed=int(config["reproducibility"]["bootstrap_seed"]),
        )
        rank_ci_payload = ci.__dict__
        rank_pass = ci.lower > 0

    positive_pass = False
    positive_ci_payload: dict[str, Any] | None = None
    validated_swap: dict[str, Any] | None = None
    null_threshold = float(config["statistics"]["null_js_floor"])
    if not positive.empty:
        null = positive[positive["strength"] == 0]["target_log_odds_shift"]
        actual = positive[positive["strength"] == 1]
        null_threshold = numerical_null_threshold(
            np.abs(null.to_numpy()),
            floor=float(config["statistics"]["null_js_floor"]),
            quantile=float(config["statistics"]["null_quantile"]),
        )
        if actual["example_id"].nunique() >= 2:
            ci = clustered_bootstrap_ci(
                actual,
                cluster_col="example_id",
                value_col="target_log_odds_shift",
                n_resamples=int(config["validation"]["bootstrap_resamples"]),
                seed=int(config["reproducibility"]["bootstrap_seed"]),
            )
            positive_ci_payload = ci.__dict__
            positive_pass = ci.lower > null_threshold
            if positive_pass:
                best = actual.sort_values("target_log_odds_shift", ascending=False).iloc[0]
                validated_swap = {
                    "source_surface": str(best["source"]),
                    "target_surface": str(best["target"]),
                    "example_id": str(best["example_id"]),
                    "strength": 1.0,
                }

    hit_threshold = float(config["validation"]["hit10_threshold"])
    hit_pass = all(value >= hit_threshold for value in hit_rates.values())
    passed = bool(band and hit_pass and rank_pass and positive_pass)
    return {
        "schema_version": 1,
        "passed": passed,
        "workspace_band": band,
        "hit10_by_family": hit_rates,
        "hit10_by_family_and_layer": hit_rates_by_layer,
        "hit10_threshold": hit_threshold,
        "hit_pass": hit_pass,
        "rank_advantage_ci": rank_ci_payload,
        "rank_pass": rank_pass,
        "positive_control_ci": positive_ci_payload,
        "positive_control_null_threshold": null_threshold,
        "positive_control_pass": positive_pass,
        "validated_swap": validated_swap,
    }


def _write_report(context, gate, layer_scores, records, positive) -> None:
    failure_reasons = [
        name
        for name, passed in (
            ("hidden-intermediate hit@10", gate.get("hit_pass", False)),
            ("held-out rank advantage", gate.get("rank_pass", False)),
            ("positive-control J intervention", gate.get("positive_control_pass", False)),
        )
        if not passed
    ]
    report = [
        "# Phase 0 — J-lens validation",
        "",
        "## Material Passport",
        "",
        f"- Run ID: `{context.run_id}`",
        "- Verification Status: ANALYZED" if len(records) else "- Verification Status: NOT EXECUTED",
        f"- Gate: {'PASSED' if gate['passed'] else 'FAILED'}",
        f"- Failed criteria: `{failure_reasons or 'none'}`",
        f"- Command: `{' '.join(sys.argv)}`",
        f"- Model ID/revision: `{gate['model_id']}@{gate['model_revision']}`",
        f"- Lens revision/file SHA-256: `{gate['lens_revision']} / {gate['lens_sha256']}`",
        f"- Upstream implementation commit: `{gate['upstream_commit']}`",
        f"- Model shard hashes: `{gate.get('model_weight_shards')}`",
        f"- Vendored-data manifest SHA-256: `{gate['upstream_data_manifest_sha256']}`",
        f"- Full run manifest: `results/raw/{context.run_id}/manifest.json`",
        "",
        "## Gate metrics",
        "",
        f"- Selected workspace band: `{gate['workspace_band']}`",
        f"- Hit@10 by family: `{gate['hit10_by_family']}`",
        f"- Hit@10 by family/layer: `{gate['hit10_by_family_and_layer']}`",
        f"- Workspace-band threshold sensitivity: `{gate.get('workspace_band_sensitivity')}`",
        f"- Rank-advantage CI: `{gate['rank_advantage_ci']}`",
        f"- Positive-control CI: `{gate['positive_control_ci']}`",
        f"- Positive-control null threshold: `{gate['positive_control_null_threshold']}`",
        "",
        "## Coverage and exclusions",
        "",
        f"- Raw layer/concept records: {len(records)}",
        f"- Primary records: {int(records['primary'].sum()) if len(records) else 0}",
        f"- First-16 exclusions: {int(records['excluded_first16'].sum()) if len(records) else 0}",
        f"- Literal-copy exclusions: {int(records['excluded_literal_copy'].sum()) if len(records) else 0}",
        f"- Positive-control records: {len(positive)}",
        f"- Lens checkpoint `n_prompts`: {gate.get('lens_checkpoint_n_prompts')}",
        f"- Companion fit-config `prompts_fitted`: {gate.get('lens_fit_metadata_prompts_fitted')}",
        "- The checkpoint/companion-metadata discrepancy is retained as a provenance warning.",
        "",
        "If the gate failed, this is a measurement-system failure. Later causal",
        "results must not be interpreted as evidence for H1, H2, or H3.",
        "",
    ]
    report_name = (
        "PHASE0_VALIDATION_QWEN3_6_27B.md"
        if context.config.get("run", {}).get("confirmation_model")
        else "PHASE0_VALIDATION.md"
    )
    (context.reports_dir / report_name).write_text(
        "\n".join(report), encoding="utf-8"
    )
    if not layer_scores.empty:
        layer_scores.to_parquet(context.processed_dir / "phase0_layer_scores.parquet", index=False)


def main() -> None:
    parser = standard_parser(
        "Validate the J-lens and identify an empirical workspace band",
        "configs/qwen3_5_4b.yaml",
    )
    parser.add_argument("--gpu-smoke", action="store_true")
    args = parser.parse_args()
    context = initialize_context("gpu-smoke" if args.gpu_smoke else "phase0", args)
    try:
        if args.dry_run:
            context.finish("DRY_RUN")
            return
        data_root = context.root / context.config["data"]["upstream_root"]
        multihop = upstream_multihop(data_root)
        order_ops = upstream_order_ops(data_root)
        flexible = _load_flexible(data_root)
        bundle = load_model_bundle(context.config)
        vocabulary = build_concept_vocabulary(
            bundle.tokenizer,
            size=int(context.config["jstate"]["concept_vocab_size"]),
            mandatory_surfaces=_mandatory_concepts(multihop, order_ops, flexible),
            model_id=bundle.model_id,
            model_revision=bundle.model_revision,
        )
        vocabulary.to_json(concept_vocabulary_path(context))
        encoder = JStateEncoder.from_lens(
            bundle.lens,
            bundle.unembedding_weight,
            vocabulary,
            k=int(context.config["jstate"]["k"]),
        )
        if args.gpu_smoke:
            smoke = _gpu_smoke(context, bundle, encoder, flexible)
            context.finish(
                "COMPLETED",
                smoke_pass=bool(smoke["null_pass"] and smoke["clamp_pass"]),
                smoke_metrics=smoke,
            )
            return
        raw = [
            *_readout_records(bundle, multihop, "factual_two_hop", context.config, args.limit),
            *_readout_records(bundle, order_ops, "order_of_operations", context.config, args.limit),
        ]
        records = pd.DataFrame(raw)
        band, layer_scores = select_workspace_band(
            records,
            peak_fraction=float(context.config["validation"]["band_peak_fraction"]),
            min_layers=int(context.config["validation"]["band_min_layers"]),
        )
        sensitivity = {}
        for fraction in context.config["validation"]["report_sensitivity_peak_fractions"]:
            alternative, _ = select_workspace_band(
                records,
                peak_fraction=float(fraction),
                min_layers=int(context.config["validation"]["band_min_layers"]),
            )
            sensitivity[str(float(fraction))] = alternative
        positive_raw = (
            _positive_control_records(
                bundle,
                encoder,
                vocabulary,
                flexible,
                band,
                int(context.config["validation"]["positive_control_trials"]),
            )
            if band
            else []
        )
        positive = pd.DataFrame(positive_raw)
        gate = _evaluate_gate(records, band, positive, context.config)
        artifact_manifest = json.loads(
            (context.root / "artifacts" / "MANIFEST.json").read_text(encoding="utf-8")
        )
        gate.update(
            {
                "config_digest": config_digest(context.config),
                "run_id": context.run_id,
                "model_id": bundle.model_id,
                "model_revision": bundle.model_revision,
                "lens_revision": bundle.lens_revision,
                "lens_sha256": context.config["lens"]["sha256"],
                "upstream_commit": artifact_manifest["upstream"]["commit"],
                "lens_path": str(bundle.lens_path),
                "lens_checkpoint_n_prompts": int(bundle.lens.n_prompts),
                "lens_fit_metadata_prompts_fitted": context.config["lens"].get(
                    "metadata_prompts_fitted"
                ),
                "workspace_band_sensitivity": sensitivity,
                "upstream_data_manifest_sha256": sha256_file(
                    context.root / "data" / "upstream" / "MANIFEST.json"
                ),
            }
        )
        model_role = (
            "confirmation"
            if context.config.get("run", {}).get("confirmation_model")
            else "primary"
        )
        gate["model_weight_shards"] = artifact_manifest["models"][model_role].get(
            "weight_shards", {}
        )
        run_dir = context.raw_dir / context.run_id
        records.to_parquet(run_dir / "readout_records.parquet", index=False)
        if not positive.empty:
            positive.to_parquet(run_dir / "positive_control_records.parquet", index=False)
        write_json_atomic(phase0_gate_path(context), gate)
        _write_report(context, gate, layer_scores, records, positive)
        context.finish("COMPLETED", gate_passed=gate["passed"], workspace_band=band)
    except KeyboardInterrupt:
        context.finish("FAILED", error="KeyboardInterrupt: run cancelled")
        raise
    except Exception as exc:
        context.finish("FAILED", error=f"{type(exc).__name__}: {exc}")
        raise


if __name__ == "__main__":
    main()
