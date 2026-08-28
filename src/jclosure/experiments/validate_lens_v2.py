"""Phase 0 v2 calibration and one-shot confirmatory adjudication."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import torch

from jclosure.datasets import (
    TaskExample,
    normalize_prompt,
    task_examples_from_json,
    upstream_multihop,
    upstream_order_ops,
)
from jclosure.experiments.common import (
    initialize_context,
    standard_parser,
)
from jclosure.experiments.validate_lens import _positive_control_records
from jclosure.jstate import JStateEncoder, build_nested_concept_vocabularies
from jclosure.model import load_model_bundle
from jclosure.phase0 import (
    PROTOCOL_VERSION,
    coverage_summary,
    item_mrr_advantages,
    official_pass_summary,
    rank_candidates,
    single_token_candidates,
    synonym_surfaces,
)
from jclosure.protocol import verify_protocol_freeze
from jclosure.provenance import sha256_file, write_json_atomic
from jclosure.statistics import clustered_bootstrap_ci, numerical_null_threshold


def _prompt_cluster(prompt: str) -> str:
    return hashlib.sha256(normalize_prompt(prompt).encode()).hexdigest()[:20]


def _copied_target(prompt: str, surfaces: tuple[str, ...]) -> bool:
    haystack = normalize_prompt(prompt)
    return any(normalize_prompt(surface) in haystack for surface in surfaces if surface)


def _rank_for_token(logits: torch.Tensor, token_id: int) -> int:
    scores = logits.float().reshape(-1)
    return int(torch.sum(scores > scores[int(token_id)]).item()) + 1


def _readout_records(
    bundle: Any,
    examples: list[TaskExample],
    *,
    family: str,
    max_seq_len: int,
    limit: int | None,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    layers = [int(layer) for layer in bundle.lens.source_layers]
    selected = examples if limit is None else examples[:limit]
    for example in selected:
        input_ids = bundle.lens_model.encode(example.prompt, max_length=max_seq_len)
        position = int(input_ids.shape[1] - 1)
        lens_logits, _, _ = bundle.lens.apply(
            bundle.lens_model,
            example.prompt,
            layers=layers,
            positions=[-1],
            max_seq_len=max_seq_len,
            use_jacobian=True,
        )
        logit_logits, _, _ = bundle.lens.apply(
            bundle.lens_model,
            example.prompt,
            layers=layers,
            positions=[-1],
            max_seq_len=max_seq_len,
            use_jacobian=False,
        )
        for concept in example.intermediates:
            surfaces = synonym_surfaces(concept, family=family)
            candidates = single_token_candidates(bundle.tokenizer, surfaces)
            copied = _copied_target(example.prompt, surfaces)
            candidate_payload = [candidate.to_dict() for candidate in candidates]
            for layer in layers:
                for method, logits_by_layer in (
                    ("jacobian", lens_logits),
                    ("logit", logit_logits),
                ):
                    logits = logits_by_layer[layer][0]
                    rank, winner = rank_candidates(logits, candidates)
                    candidate_ranks = [
                        {**candidate.to_dict(), "rank": _rank_for_token(logits, candidate.token_id)}
                        for candidate in candidates
                    ]
                    records.append(
                        {
                            "schema_version": 2,
                            "protocol_version": PROTOCOL_VERSION,
                            "example_id": example.example_id,
                            "prompt_cluster": _prompt_cluster(example.prompt),
                            "template_id": example.template_id,
                            "family": family,
                            "canonical_concept": concept,
                            "concept": concept,
                            "layer": layer,
                            "position": position,
                            "method": method,
                            "rank": rank,
                            "hit1": bool(rank is not None and rank <= 1),
                            "hit5": bool(rank is not None and rank <= 5),
                            "hit10": bool(rank is not None and rank <= 10),
                            "tokenizable": bool(candidates),
                            "copied_target": copied,
                            "position_lt16": position < 16,
                            "synonym_strings_json": json.dumps(list(surfaces)),
                            "synonym_candidates_json": json.dumps(candidate_payload),
                            "synonym_token_ids_json": json.dumps(
                                [candidate.token_id for candidate in candidates]
                            ),
                            "candidate_ranks_json": json.dumps(candidate_ranks),
                            "winning_surface": winner.surface if winner else None,
                            "winning_token_id": winner.token_id if winner else None,
                        }
                    )
    return records


def select_workspace_band_v2(
    records: pd.DataFrame, *, peak_fraction: float, min_layers: int
) -> tuple[list[int], pd.DataFrame]:
    selected = records[records["tokenizable"].astype(bool) & records["rank"].notna()]
    pivot = selected.pivot_table(
        index=["example_id", "family", "concept", "layer"],
        columns="method",
        values="rank",
        aggfunc="first",
    ).dropna(subset=["jacobian", "logit"])
    if pivot.empty:
        return [], pd.DataFrame()
    pivot["advantage"] = 1.0 / pivot["jacobian"] - 1.0 / pivot["logit"]
    item_layer = (
        pivot.reset_index()
        .groupby(["example_id", "family", "layer"], as_index=False)["advantage"]
        .mean()
    )
    layer_scores = (
        item_layer.groupby("layer", as_index=False)["advantage"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .rename(
            columns={
                "mean": "advantage",
                "std": "item_stability_std",
                "count": "item_count",
            }
        )
        .sort_values("layer")
    )
    layer_scores["smoothed_advantage"] = layer_scores["advantage"].rolling(
        3, center=True, min_periods=1
    ).mean()
    if layer_scores.empty or float(layer_scores["smoothed_advantage"].max()) <= 0:
        return [], layer_scores
    peak = layer_scores.loc[layer_scores["smoothed_advantage"].idxmax()]
    peak_layer = int(peak["layer"])
    threshold = float(peak["smoothed_advantage"]) * float(peak_fraction)
    eligible = {
        int(row.layer)
        for row in layer_scores.itertuples()
        if row.smoothed_advantage >= threshold and row.advantage >= 0
    }
    band = [peak_layer]
    while band[0] - 1 in eligible:
        band.insert(0, band[0] - 1)
    while band[-1] + 1 in eligible:
        band.append(band[-1] + 1)
    return (band if len(band) >= min_layers else []), layer_scores


def _mandatory_concepts(examples: list[TaskExample], flexible: dict[str, Any]) -> list[str]:
    values: set[str] = set()
    for item in examples:
        values.add(item.answer)
        for concept in item.intermediates:
            values.update(synonym_surfaces(concept, family=item.family))
    for category in flexible["categories"]:
        values.update(str(value) for value in category["args"])
        for function in category["funcs"]:
            values.add(str(function["name"]))
            values.update(str(value) for value in function["answers"].values())
    return sorted(values)


def _rank_ci(
    records: pd.DataFrame, layers: list[int], config: dict[str, Any]
) -> tuple[dict[str, Any] | None, bool]:
    item = item_mrr_advantages(records, layers=layers)
    if item.empty:
        return None, False
    clusters = records[["example_id", "prompt_cluster"]].drop_duplicates("example_id")
    item = item.merge(clusters, on="example_id", how="left")
    if item["prompt_cluster"].nunique() < 2:
        return None, False
    ci = clustered_bootstrap_ci(
        item,
        cluster_col="prompt_cluster",
        value_col="mrr_advantage",
        n_resamples=int(config["validation"]["bootstrap_resamples"]),
        seed=int(config["reproducibility"]["bootstrap_seed"]),
    )
    return ci.__dict__, bool(ci.lower > 0)


def _positive_gate(positive: pd.DataFrame, config: dict[str, Any]) -> dict[str, Any]:
    floor = float(config["statistics"]["null_js_floor"])
    if positive.empty:
        return {"passed": False, "ci": None, "null_threshold": floor}
    null = positive[positive["strength"] == 0]["target_log_odds_shift"].abs()
    actual = positive[positive["strength"] == 1]
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
    return {
        "passed": bool(ci.lower > threshold),
        "ci": ci.__dict__,
        "null_threshold": threshold,
    }


def _family_pass(summary: dict[str, Any], threshold: float) -> tuple[dict[str, float], bool]:
    values = {
        family: float(payload["jacobian"]["pass_at"]["10"])
        for family, payload in summary["families"].items()
    }
    required = ("factual_two_hop", "order_of_operations")
    return values, all(values.get(family, 0.0) >= threshold for family in required)


def _write_report(
    *,
    path: Path,
    context: Any,
    status: str,
    summary: dict[str, Any],
    sensitivity_position: dict[str, Any],
    sensitivity_copy: dict[str, Any],
    coverage: dict[str, Any],
    band_summary: dict[str, Any],
    rank_ci: dict[str, Any] | None,
    positive_gate: dict[str, Any] | None,
    gate: dict[str, Any] | None,
) -> None:
    lines = [
        "# Phase 0 v2 — Confirmatory J-lens validation" if gate else "# Phase 0 v2 — Calibration",
        "",
        "## Material Passport",
        "",
        f"- Protocol: `{PROTOCOL_VERSION}`",
        f"- Run ID: `{context.run_id}`",
        f"- Verification Status: {status}",
        f"- Command: `{' '.join(sys.argv)}`",
        f"- Config digest: `{context.config.get('_config_path')}`",
        "",
        "## Measurement",
        "",
        f"- Official-main: `{json.dumps(summary, sort_keys=True)}`",
        f"- Position>=16 sensitivity: `{json.dumps(sensitivity_position, sort_keys=True)}`",
        f"- Copy-excluded sensitivity: `{json.dumps(sensitivity_copy, sort_keys=True)}`",
        f"- Band-restricted summary: `{json.dumps(band_summary, sort_keys=True)}`",
        f"- Coverage: `{coverage}`",
        f"- Item-clustered rank-advantage CI: `{rank_ci}`",
        f"- Positive-control gate: `{positive_gate}`",
        "",
    ]
    if gate:
        lines.extend(
            [
                "## Gate",
                "",
                f"- Decision: `{'PASSED' if gate['passed'] else 'FAILED'}`",
                f"- Criteria: `{gate['criteria']}`",
                "- This adjudication is locked. A protocol change requires a separately labeled exploratory version.",
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = standard_parser(
        "Run Phase 0 v2 calibration or confirmatory validation",
        "configs/phase0_v2_calibration.yaml",
    )
    args = parser.parse_args()
    context = initialize_context("phase0-v2", args)
    try:
        mode = str(context.config.get("run", {}).get("phase0_mode", "calibration"))
        if context.config["validation"].get("protocol_version") != PROTOCOL_VERSION:
            raise RuntimeError("Phase 0 v2 runner requires phase0_protocol_v2 config")
        if args.dry_run:
            context.finish("DRY_RUN", phase0_mode=mode)
            return
        v2 = context.config["phase0_v2"]
        if mode == "confirmatory":
            gate_path = context.processed_dir / "phase0_v2_gate.json"
            if gate_path.exists():
                existing = json.loads(gate_path.read_text(encoding="utf-8"))
                if existing.get("adjudication_locked"):
                    raise RuntimeError("Phase 0 v2 confirmatory adjudication already exists and is locked")
            freeze = verify_protocol_freeze(
                context.root / v2["freeze_manifest"], root=context.root
            )
            band = [int(layer) for layer in freeze["workspace_band"]]
            positive_layer = int(freeze["positive_control_layer"])
            data_dir = context.root / v2["fresh_data_dir"]
            multihop = task_examples_from_json(data_dir / "fresh_multihop.json")
            order_ops = task_examples_from_json(data_dir / "fresh_order_ops.json")
        elif mode == "calibration":
            freeze = None
            data_root = context.root / context.config["data"]["upstream_root"]
            multihop = upstream_multihop(data_root)
            order_ops = upstream_order_ops(data_root)
            band = []
            positive_layer = -1
        else:
            raise ValueError(f"unknown phase0_mode: {mode}")

        data_root = context.root / context.config["data"]["upstream_root"]
        flexible = json.loads(
            (data_root / "experiments/flexible-generalization.json").read_text(encoding="utf-8")
        )
        examples = [*multihop, *order_ops]
        bundle = load_model_bundle(context.config)
        raw = [
            *_readout_records(
                bundle,
                multihop,
                family="factual_two_hop",
                max_seq_len=int(context.config["model"].get("max_seq_len", 512)),
                limit=args.limit,
            ),
            *_readout_records(
                bundle,
                order_ops,
                family="order_of_operations",
                max_seq_len=int(context.config["model"].get("max_seq_len", 512)),
                limit=args.limit,
            ),
        ]
        records = pd.DataFrame(raw)
        all_layers = [int(layer) for layer in bundle.lens.source_layers]
        layer_scores = pd.DataFrame()
        band_sensitivity: dict[str, list[int]] = {}
        if mode == "calibration":
            band, layer_scores = select_workspace_band_v2(
                records,
                peak_fraction=float(context.config["validation"]["band_peak_fraction"]),
                min_layers=int(context.config["validation"]["band_min_layers"]),
            )
            for fraction in context.config["validation"]["report_sensitivity_peak_fractions"]:
                alternate, _ = select_workspace_band_v2(
                    records,
                    peak_fraction=float(fraction),
                    min_layers=int(context.config["validation"]["band_min_layers"]),
                )
                band_sensitivity[str(float(fraction))] = alternate
            if band:
                positive_layer = int(
                    layer_scores[layer_scores["layer"].isin(band)]
                    .sort_values(["smoothed_advantage", "layer"], ascending=[False, True])
                    .iloc[0]["layer"]
                )

        vocabularies = build_nested_concept_vocabularies(
            bundle.tokenizer,
            sizes=tuple(int(size) for size in context.config["jstate"]["concept_vocab_sizes"]),
            mandatory_surfaces=_mandatory_concepts(examples, flexible),
            model_id=bundle.model_id,
            model_revision=bundle.model_revision,
        )
        for size, vocabulary in vocabularies.items():
            vocabulary.to_json(context.processed_dir / f"concept_vocabulary_v2_{size}.json")
        primary_vocabulary = vocabularies[int(context.config["jstate"]["concept_vocab_size"])]
        positive = pd.DataFrame()
        positive_payload: dict[str, Any] | None = None
        if positive_layer >= 0:
            encoder = JStateEncoder.from_lens(
                bundle.lens,
                bundle.unembedding_weight,
                primary_vocabulary,
                k=int(context.config["jstate"]["k"]),
                lazy=True,
                protocol_version=PROTOCOL_VERSION,
                direction_chunk_size=int(context.config["jstate"]["direction_chunk_size"]),
            )
            positive = pd.DataFrame(
                _positive_control_records(
                    bundle,
                    encoder,
                    primary_vocabulary,
                    flexible,
                    [positive_layer],
                    int(context.config["validation"]["positive_control_trials"]),
                )
            )
            positive_payload = _positive_gate(positive, context.config)

        summary = official_pass_summary(records, layers=all_layers)
        position_summary = official_pass_summary(
            records, layers=all_layers, require_position_16=True
        )
        copy_summary = official_pass_summary(records, layers=all_layers, exclude_copied=True)
        band_summary = official_pass_summary(records, layers=band) if band else {
            "protocol_version": PROTOCOL_VERSION,
            "layers": [],
            "families": {},
        }
        rank_ci, rank_pass = _rank_ci(records, all_layers, context.config)
        coverage = coverage_summary(records)
        run_dir = context.raw_dir / context.run_id
        records.to_parquet(run_dir / "readout_records_v2.parquet", index=False)
        if not positive.empty:
            positive.to_parquet(run_dir / "positive_control_records_v2.parquet", index=False)

        base = {
            "schema_version": 2,
            "protocol_version": PROTOCOL_VERSION,
            "mode": mode,
            "run_id": context.run_id,
            "model_id": bundle.model_id,
            "model_revision": bundle.model_revision,
            "lens_revision": bundle.lens_revision,
            "lens_sha256": context.config["lens"]["sha256"],
            "workspace_band": band,
            "workspace_band_sensitivity": band_sensitivity,
            "positive_control_layer": positive_layer,
            "official_main": summary,
            "position16_sensitivity": position_summary,
            "copy_excluded_sensitivity": copy_summary,
            "band_restricted": band_summary,
            "coverage": coverage,
            "rank_advantage_ci": rank_ci,
            "rank_pass": rank_pass,
            "positive_control": positive_payload,
            "dictionary_hashes": {
                str(size): vocabulary.digest for size, vocabulary in vocabularies.items()
            },
            "readout_records_sha256": sha256_file(run_dir / "readout_records_v2.parquet"),
        }
        if mode == "calibration":
            write_json_atomic(context.processed_dir / "phase0_v2_calibration.json", base)
            if not layer_scores.empty:
                layer_scores.to_parquet(
                    context.processed_dir / "phase0_v2_layer_scores.parquet", index=False
                )
            _write_report(
                path=context.reports_dir / "PHASE0_V2_CALIBRATION.md",
                context=context,
                status="ANALYZED",
                summary=summary,
                sensitivity_position=position_summary,
                sensitivity_copy=copy_summary,
                coverage=coverage,
                band_summary=band_summary,
                rank_ci=rank_ci,
                positive_gate=positive_payload,
                gate=None,
            )
            context.finish("COMPLETED", phase0_mode=mode, workspace_band=band)
            return

        hit_values, hit_pass = _family_pass(
            summary, float(context.config["validation"]["hit10_threshold"])
        )
        positive_pass = bool(positive_payload and positive_payload["passed"])
        passed = bool(hit_pass and rank_pass and positive_pass)
        gate = {
            **base,
            "passed": passed,
            "adjudication_locked": True,
            "freeze_manifest_sha256": sha256_file(context.root / v2["freeze_manifest"]),
            "criteria": {
                "hit10_by_family": hit_values,
                "hit10_threshold": float(context.config["validation"]["hit10_threshold"]),
                "hit_pass": hit_pass,
                "rank_pass": rank_pass,
                "positive_control_pass": positive_pass,
            },
        }
        write_json_atomic(context.processed_dir / "phase0_v2_gate.json", gate)
        _write_report(
            path=context.reports_dir / "PHASE0_V2_CONFIRMATORY.md",
            context=context,
            status="VERIFIED",
            summary=summary,
            sensitivity_position=position_summary,
            sensitivity_copy=copy_summary,
            coverage=coverage,
            band_summary=band_summary,
            rank_ci=rank_ci,
            positive_gate=positive_payload,
            gate=gate,
        )
        context.finish("COMPLETED", phase0_mode=mode, gate_passed=passed)
    except KeyboardInterrupt:
        context.finish("FAILED", error="KeyboardInterrupt: run cancelled")
        raise
    except Exception as exc:
        context.finish("FAILED", error=f"{type(exc).__name__}: {exc}")
        raise


if __name__ == "__main__":
    main()
