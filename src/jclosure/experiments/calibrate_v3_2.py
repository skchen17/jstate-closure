"""Fresh calibration with runtime-matched v3.2 restoration semantics."""

from __future__ import annotations

import copy
import hashlib
import json
import os
from dataclasses import asdict
from functools import partial
from pathlib import Path
from typing import Any

import pandas as pd
import torch

from jclosure.clamp_v3_2 import build_v32_schedule
from jclosure.config import config_digest
from jclosure.experiments.calibrate_v3_1 import (
    _chain_transform,
    _hook_sanity,
    _read_jsonl,
)
from jclosure.experiments.common import initialize_context, standard_parser
from jclosure.geometry import DenseJMap
from jclosure.jstate import ConceptVocabulary, JStateEncoder
from jclosure.model import load_model_bundle
from jclosure.protocol_v3_2 import verify_calibration_freeze
from jclosure.provenance import append_jsonl, sha256_file, write_json_atomic
from jclosure.recorder import ActivationRecorder, ResidualEditor
from jclosure.runtime_v3_2 import (
    PROTOCOL_V32,
    construct_initial_sequence,
    fit_naturality_models_v32,
    load_v32_domain,
    match_donor,
    replacement_transform,
    restoration_is_optimized,
    restore_sequence,
    teacher_preanswer_prefix,
    tensor_digest,
    v32_thresholds,
)
from jclosure.statistics_v3_2 import conditional_summary_dict


def _shard_invariant_config_digest(manifest: dict[str, Any]) -> str:
    config = copy.deepcopy(manifest.get("config", {}))
    if isinstance(config.get("model"), dict):
        config["model"].pop("device", None)
    return config_digest(config)


def _write_partitioned_records_v32(
    frame: pd.DataFrame,
    *,
    root: Path,
    output_root: Path,
    manifest_path: Path,
    rows_per_file: int = 50,
) -> dict[str, Any]:
    parts: list[dict[str, Any]] = []
    keys = ["l1", "restoration_method", "restoration_scope"]
    for (l1, method, scope), group in frame.groupby(keys, sort=True, dropna=False):
        partition = output_root / f"l1={int(l1)}" / f"method={method}" / f"scope={scope}"
        partition.mkdir(parents=True, exist_ok=True)
        for part_index, start in enumerate(range(0, len(group), rows_per_file)):
            path = partition / f"part-{part_index:03d}.parquet"
            chunk = group.iloc[start : start + rows_per_file]
            chunk.to_parquet(path, index=False, compression="zstd")
            parts.append({
                "path": str(path.relative_to(root)), "sha256": sha256_file(path),
                "bytes": path.stat().st_size, "rows": len(chunk), "l1": int(l1),
                "restoration_method": str(method), "restoration_scope": str(scope),
            })
    payload = {
        "schema_version": 5, "protocol_version": PROTOCOL_V32,
        "format": "partitioned_parquet_zstd", "rows": len(frame),
        "rows_per_file": rows_per_file, "parts": parts,
    }
    write_json_atomic(manifest_path, payload)
    return payload


def _load_encoder(context, bundle):
    size = int(context.config["v3_2"]["dictionary_size"])
    vocabulary = ConceptVocabulary.from_json(
        context.root / "results/processed" / f"concept_vocabulary_v2_{size}.json"
    )
    encoder = JStateEncoder.from_lens(
        bundle.lens,
        bundle.unembedding_weight,
        vocabulary,
        k=int(context.config["jstate"]["k"]),
        lazy=True,
        protocol_version=PROTOCOL_V32,
        direction_chunk_size=int(context.config["jstate"].get("direction_chunk_size", 512)),
    )
    return vocabulary, encoder, DenseJMap.from_encoder(encoder)


def _payload(root: Path, record: dict[str, Any], device) -> dict[str, Any]:
    value = torch.load(root / record["activation_path"], map_location="cpu")
    value["input_ids"] = value["input_ids"].to(device)
    return value


def _build_bank(context, bundle, *, limit: int | None) -> Path:
    layers = list(range(23, int(context.config["v3_2"]["final_layer"]) + 1))
    examples = [
        *(("naturality_fit", item) for item in load_v32_domain(context.root, "naturality_fit")),
        *(("calibration", item) for item in load_v32_domain(context.root, "calibration")),
    ]
    if limit is not None:
        examples = examples[:limit]
    directory = context.raw_dir / context.run_id / "activation_bank"
    directory.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    for index, (domain, example) in enumerate(examples):
        input_ids, answer_id, teacher_correct, exclusion = teacher_preanswer_prefix(bundle, example)
        with ActivationRecorder(bundle.layers, at=layers) as recorder:
            with torch.no_grad():
                logits = bundle.forward_logits(input_ids)[0, -1].detach().float().cpu()
        payload = {
            "input_ids": input_ids.detach().cpu(),
            "activations": {
                layer: recorder.activations[layer][0].detach().float().cpu()
                for layer in layers
            },
            "clean_logits": logits,
            "answer_id": answer_id,
        }
        tensor_path = directory / f"state-{index:05d}.pt"
        torch.save(payload, tensor_path)
        records.append(
            {
                "schema_version": 5,
                "protocol_version": PROTOCOL_V32,
                "run_id": context.run_id,
                "domain": domain,
                "prompt_id": example.example_id,
                "prompt_hash": hashlib.sha256(example.prompt.strip().encode()).hexdigest(),
                "template_id": example.template_id,
                "task_family": example.family,
                "prompt": example.prompt,
                "answer": example.answer,
                "variables": example.variables,
                "sequence_length": int(input_ids.shape[-1]),
                "teacher_correct": teacher_correct,
                "exclusion_reason": exclusion,
                "activation_path": str(tensor_path.relative_to(context.root)),
                "activation_sha256": sha256_file(tensor_path),
            }
        )
    manifest = context.raw_dir / context.run_id / "activation_bank_manifest.jsonl"
    append_jsonl(manifest, records)
    return manifest


def _run_chain(
    *,
    bundle,
    input_ids: torch.Tensor,
    candidate: torch.Tensor,
    initial_positions: tuple[int, ...],
    future_layers: list[int],
    clean_sequences: dict[int, torch.Tensor],
    donor_sequences: dict[int, torch.Tensor],
    restoration_positions: tuple[int, ...],
    dense_map,
    encoder,
    naturality,
    tolerance: float,
    optimized: bool,
    thresholds,
) -> tuple[list[dict[str, Any]], bool]:
    capture: dict[int, dict[str, Any]] = {}
    transforms: dict[int, Any] = {
        min(future_layers) - 1: replacement_transform(candidate, initial_positions)
    }
    for layer in future_layers:
        transforms[layer] = partial(
            _chain_transform,
            clean=clean_sequences[layer],
            donor=donor_sequences[layer],
            positions=restoration_positions,
            dense_map=dense_map,
            encoder=encoder,
            naturality=naturality[layer],
            tolerance=tolerance,
            optimized=optimized,
            thresholds=thresholds,
            capture=capture,
        )
    with ResidualEditor(bundle.layers, transforms):
        with torch.no_grad():
            bundle.forward_logits(input_ids)
    events = [capture[layer] for layer in future_layers if layer in capture]
    return events, bool(len(events) == len(future_layers) and all(row["passed"] for row in events))


def _calibration_rows(
    context,
    bundle,
    bank: Path,
    *,
    shard_index: int,
    shard_count: int,
    limit: int | None,
) -> list[dict[str, Any]]:
    records = _read_jsonl(bank)
    fit = [row for row in records if row["domain"] == "naturality_fit"]
    calibration = sorted(
        [row for row in records if row["domain"] == "calibration" and row["teacher_correct"]],
        key=lambda row: (row["prompt_hash"], row["prompt_id"]),
    )
    target = int(context.config["v3_2"]["attempts_per_layer"])
    if limit is not None:
        target = min(target, int(limit))
    calibration = calibration[:target]
    if len(calibration) < target:
        raise RuntimeError(f"v3.2 calibration has {len(calibration)}/{target} teacher-correct anchors")
    candidates = [int(value) for value in context.config["v3_2"]["candidate_initial_layers"]]
    l1_values = [
        layer for layer in candidates
        if int(hashlib.sha256(str(layer).encode()).hexdigest(), 16) % shard_count == shard_index
    ]
    if not l1_values:
        raise RuntimeError("v3.2 calibration shard has no L1")
    final_layer = int(context.config["v3_2"]["final_layer"])
    layers = list(range(min(l1_values), final_layer + 1))
    naturality = {
        scope: fit_naturality_models_v32(context.root, fit, layers, scope=scope, config=context.config)
        for scope in ("final", "all_non_padding")
    }
    vocabulary, encoder, dense_map = _load_encoder(context, bundle)
    thresholds = v32_thresholds(context.config)
    tolerance = float(context.config["geometry"]["formal_null_tolerance"])
    strength = float(context.config["v3_2"]["initial_strength"])
    methods = [str(value) for value in context.config["v3_2"]["restoration_methods"]]
    sanity_payload = _payload(context.root, calibration[0], bundle.hf_model.device)
    sanity = {l1: _hook_sanity(bundle, sanity_payload["input_ids"], l1) for l1 in l1_values}
    rows: list[dict[str, Any]] = []
    for l1 in l1_values:
        future_layers = list(range(l1 + 1, final_layer + 1))
        for anchor in calibration:
            donor = match_donor(anchor, fit)
            if donor is None:
                for method in methods:
                    for scope in ("final", "all_non_padding"):
                        rows.append({
                            "schema_version": 5, "protocol_version": PROTOCOL_V32,
                            "run_id": context.run_id, "prompt_id": anchor["prompt_id"],
                            "l1": l1, "initial_scope": "final", "restoration_scope": scope,
                            "restoration_method": method, "initial_valid": False,
                            "restoration_applicable": False, "chain_valid": False,
                            "exclusion_reason": "donor_length_or_template_mismatch",
                        })
                continue
            anchor_payload = _payload(context.root, anchor, bundle.hf_model.device)
            donor_payload = _payload(context.root, donor, bundle.hf_model.device)
            input_ids = anchor_payload["input_ids"]
            length = int(input_ids.shape[-1])
            initial_positions = (length - 1,)
            clean_l1 = anchor_payload["activations"][l1].to(bundle.hf_model.device)
            donor_l1 = donor_payload["activations"][l1].to(bundle.hf_model.device)
            candidate, initial_quality, initial_valid, displacement = construct_initial_sequence(
                clean_l1, donor_l1, positions=initial_positions, layer=l1,
                dense_map=dense_map, encoder=encoder, naturality=naturality["final"][l1],
                tolerance=tolerance, strength=strength, thresholds=thresholds,
                require_per_position_displacement=True,
            )
            base_id = hashlib.sha256(
                f"{anchor['prompt_id']}\x1f{donor['prompt_id']}\x1f{l1}\x1ffinal".encode()
            ).hexdigest()
            common = {
                "schema_version": 5, "protocol_version": PROTOCOL_V32,
                "run_id": context.run_id, "base_trial_id": base_id,
                "prompt_id": anchor["prompt_id"], "donor_id": donor["prompt_id"],
                "task_family": anchor["task_family"], "template_id": anchor["template_id"],
                "l1": l1, "initial_scope": "final", "sequence_length": length,
                "dictionary_size": len(vocabulary.token_ids), "dictionary_hash": vocabulary.digest,
                "initial_valid": initial_valid, "restoration_applicable": initial_valid,
                "initial_aggregate_displacement_fraction": displacement,
                "initial_positions": initial_quality, "candidate_sha256": tensor_digest(candidate),
                "restoration_layers": future_layers, "hook_sanity": sanity[l1],
            }
            if not initial_valid:
                for method in methods:
                    for scope in ("final", "all_non_padding"):
                        rows.append({
                            **common, "restoration_scope": scope, "restoration_method": method,
                            "isolated_restoration_events": [], "runtime_restoration_events": [],
                            "chain_valid": False, "hook_execution_map": [],
                            "exclusion_reason": "initial_intervention_invalid",
                        })
                continue
            clean_sequences = {
                layer: anchor_payload["activations"][layer].to(bundle.hf_model.device)
                for layer in future_layers
            }
            donor_sequences = {
                layer: donor_payload["activations"][layer].to(bundle.hf_model.device)
                for layer in future_layers
            }
            with (
                ResidualEditor(bundle.layers, {l1: replacement_transform(candidate, initial_positions)}),
                ActivationRecorder(bundle.layers, at=future_layers) as recorder,
            ):
                with torch.no_grad():
                    bundle.forward_logits(input_ids)
            observed = {
                layer: recorder.activations[layer][0].detach().float()
                for layer in future_layers
            }
            for method in methods:
                optimized = restoration_is_optimized(method)
                for scope in ("final", "all_non_padding"):
                    restoration_positions = (length - 1,) if scope == "final" else tuple(range(length))
                    isolated = []
                    for layer in future_layers:
                        _, events, passed = restore_sequence(
                            clean_sequences[layer], observed[layer], donor_sequences[layer],
                            positions=restoration_positions, layer=layer, dense_map=dense_map,
                            encoder=encoder, naturality=naturality[scope][layer],
                            tolerance=tolerance, optimized=optimized, thresholds=thresholds,
                        )
                        isolated.append({"layer": layer, "passed": passed, "events": [asdict(event) for event in events]})
                    runtime_events, chain_valid = _run_chain(
                        bundle=bundle, input_ids=input_ids, candidate=candidate,
                        initial_positions=initial_positions, future_layers=future_layers,
                        clean_sequences=clean_sequences, donor_sequences=donor_sequences,
                        restoration_positions=restoration_positions, dense_map=dense_map,
                        encoder=encoder, naturality=naturality[scope], tolerance=tolerance,
                        optimized=optimized, thresholds=thresholds,
                    )
                    mode = "persistent_final" if scope == "final" else "persistent_all"
                    schedule = build_v32_schedule(
                        mode=mode, initial_layer=l1, restoration_layers=future_layers,
                        sequence_length=length, initial_scope="final", restoration_scope=scope,
                    )
                    rows.append({
                        **common, "restoration_scope": scope, "restoration_method": method,
                        "isolated_restoration_events": isolated,
                        "runtime_restoration_events": runtime_events,
                        "chain_valid": chain_valid,
                        "hook_execution_map": [asdict(event) for event in schedule.events],
                        "exclusion_reason": None,
                    })
    return rows


def _event_pass(events: Any, layer: int) -> bool:
    if events is None:
        return False
    for event in events:
        if int(event.get("layer", -1)) == layer:
            return bool(event.get("passed", False))
    return False


def _summarize(frame: pd.DataFrame, config: dict[str, Any]) -> dict[str, Any]:
    values = config["v3_2"]
    resamples = int(values["bootstrap_resamples"])
    confidence = float(values["confidence"])
    min_applicable = int(values["minimum_restoration_applicable"])
    min_rate = float(values["minimum_restoration_conditional_rate"])
    seed = int(config["reproducibility"]["bootstrap_seed"])
    summaries: list[dict[str, Any]] = []
    for (l1, method, scope), group in frame.groupby(["l1", "restoration_method", "restoration_scope"]):
        attempted = len(group)
        initial_valid = int(group["initial_valid"].sum())
        initial_rate = initial_valid / max(1, attempted)
        intervention_eligible = (
            initial_valid >= int(values["minimum_intervention_valid_count"])
            and initial_rate >= float(values["minimum_intervention_valid_rate"])
        )
        layers = sorted({int(layer) for row in group["restoration_layers"] for layer in row})
        isolated: dict[str, Any] = {}
        runtime: dict[str, Any] = {}
        eligible_layers = []
        applicable = group["restoration_applicable"].astype(bool).to_numpy()
        for layer in layers:
            isolated_success = group["isolated_restoration_events"].map(
                lambda events, current_layer=layer: _event_pass(events, current_layer)
            ).to_numpy()
            runtime_success = group["runtime_restoration_events"].map(
                lambda events, current_layer=layer: _event_pass(events, current_layer)
            ).to_numpy()
            isolated[str(layer)] = conditional_summary_dict(
                applicable, isolated_success, minimum_applicable=min_applicable,
                minimum_rate=min_rate, n_resamples=resamples, confidence=confidence,
                seed=seed + int(l1) * 100 + layer,
            )
            runtime[str(layer)] = conditional_summary_dict(
                applicable, runtime_success, minimum_applicable=min_applicable,
                minimum_rate=min_rate, n_resamples=resamples, confidence=confidence,
                seed=seed + int(l1) * 1000 + layer,
            )
            if isolated[str(layer)]["eligible"] and runtime[str(layer)]["eligible"]:
                eligible_layers.append(layer)
        chain = conditional_summary_dict(
            applicable, group["chain_valid"].astype(bool).to_numpy(),
            minimum_applicable=min_applicable, minimum_rate=min_rate,
            n_resamples=resamples, confidence=confidence, seed=seed + int(l1),
        )
        sanity = all(all(value.values()) for value in group["hook_sanity"] if value)
        summaries.append({
            "l1": int(l1), "initial_scope": "final", "restoration_scope": str(scope),
            "restoration_method": str(method), "attempted": attempted,
            "intervention_valid": initial_valid, "intervention_valid_rate": initial_rate,
            "intervention_eligible": intervention_eligible, "restoration_applicable": initial_valid,
            "isolated_conditional": isolated, "runtime_conditional": runtime,
            "complete_chain_conditional": chain, "restoration_eligible_layers": eligible_layers,
            "hook_sanity_passed": sanity,
        })
    protocols: list[dict[str, Any]] = []
    summary_frame = pd.DataFrame(summaries)
    for (l1, method), group in summary_frame.groupby(["l1", "restoration_method"]):
        if set(group["restoration_scope"]) != {"final", "all_non_padding"}:
            continue
        if not bool(group["intervention_eligible"].all()) or not bool(group["hook_sanity_passed"].all()):
            continue
        layer_sets = [set(value) for value in group["restoration_eligible_layers"]]
        common_layers = sorted(set.intersection(*layer_sets)) if layer_sets else []
        if len(common_layers) < int(values["minimum_restoration_layers"]):
            continue
        minimum_runtime_rate = min(
            float(row["runtime_conditional"][str(layer)]["rate"])
            for _, row in group.iterrows() for layer in common_layers
        )
        protocols.append({
            "protocol_key": f"{method}:M4096:L{int(l1)}", "l1": int(l1),
            "initial_scope": "final", "restoration_method": str(method),
            "restoration_scopes": ["final", "all_non_padding"],
            "restoration_layers": common_layers,
            "selection_score": minimum_runtime_rate,
        })
    tiebreak = {name: index for index, name in enumerate(values["method_tiebreak"])}
    protocols.sort(key=lambda row: (int(row["l1"]), -float(row["selection_score"]), tiebreak[str(row["restoration_method"])]))
    return {
        "schema_version": 5, "protocol_version": PROTOCOL_V32,
        "layers": summaries, "authorized_protocols": protocols[:1],
        "behavioral_authorized": bool(protocols),
        "selection_rule": "earliest final-scope L1 with >=3 layers eligible for both restoration scopes; maximize minimum conditional runtime rate",
    }


def main() -> None:
    parser = standard_parser("Calibrate causal protocol v3.2", "configs/closure_v3_2.yaml")
    parser.add_argument("--stage", choices=("bank", "calibrate", "merge", "smoke"), default="calibrate")
    parser.add_argument("--bank-manifest")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-group-id", default=os.environ.get("JCLOSURE_SHARD_GROUP_ID", "single"))
    args = parser.parse_args()
    context = initialize_context("calibration-v3-2", args)
    try:
        verify_calibration_freeze(context.root, context.config)
        bundle = None if args.dry_run or args.stage == "merge" else load_model_bundle(context.config)
        if args.dry_run:
            context.finish("DRY_RUN", stage=args.stage)
            return
        if args.stage in {"bank", "smoke"}:
            assert bundle is not None
            bank = _build_bank(context, bundle, limit=args.limit or (8 if args.stage == "smoke" else None))
            context.finish("COMPLETED", stage=args.stage, activation_bank_manifest=str(bank.relative_to(context.root)))
            return
        bank = Path(args.bank_manifest).resolve() if args.bank_manifest else sorted(context.raw_dir.glob("calibration-v3-2-*/activation_bank_manifest.jsonl"))[-1]
        if args.stage == "calibrate":
            assert bundle is not None
            rows = _calibration_rows(context, bundle, bank, shard_index=args.shard_index, shard_count=args.shard_count, limit=args.limit)
            path = context.raw_dir / context.run_id / f"calibration-shard-{args.shard_index:03d}.parquet"
            path.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(rows).to_parquet(path, index=False)
            context.finish("COMPLETED_SHARD", stage="calibrate", shard_group_id=args.shard_group_id, shard_index=args.shard_index, shard_count=args.shard_count, activation_bank_manifest=str(bank.relative_to(context.root)), records=str(path.relative_to(context.root)), record_count=len(rows))
            return
        manifests = []
        for path in sorted(context.raw_dir.glob("calibration-v3-2-*/manifest.json")):
            value = json.loads(path.read_text())
            if value.get("status") == "COMPLETED_SHARD" and value.get("shard_group_id") == args.shard_group_id:
                manifests.append(value)
        by_index = {int(value["shard_index"]): value for value in manifests}
        if set(by_index) != set(range(args.shard_count)):
            raise RuntimeError("v3.2 merge is missing calibration shards")
        if len({_shard_invariant_config_digest(value) for value in by_index.values()}) != 1:
            raise RuntimeError("v3.2 calibration shard configs differ")
        frame = pd.concat([pd.read_parquet(context.root / by_index[index]["records"]) for index in sorted(by_index)], ignore_index=True)
        records_manifest_path = context.processed_dir / "closure_v3_2_calibration_records.json"
        records_manifest = _write_partitioned_records_v32(
            frame, root=context.root,
            output_root=context.processed_dir / "closure_v3_2_calibration_records" / context.run_id,
            manifest_path=records_manifest_path,
        )
        summary = _summarize(frame, context.config)
        summary.update({
            "run_id": context.run_id, "activation_bank_manifest": str(bank.relative_to(context.root)),
            "records": str(records_manifest_path.relative_to(context.root)),
            "record_part_count": len(records_manifest["parts"]), "attempted": len(frame),
            "source_shards": [by_index[index]["run_id"] for index in sorted(by_index)],
        })
        output = context.processed_dir / "closure_v3_2_calibration.json"
        write_json_atomic(output, summary)
        context.finish("COMPLETED", stage="merge", calibration=str(output.relative_to(context.root)), behavioral_authorized=summary["behavioral_authorized"], attempted=len(frame))
    except KeyboardInterrupt:
        context.finish("FAILED", error="KeyboardInterrupt: run cancelled")
        raise
    except Exception as exc:
        context.finish("FAILED", error=f"{type(exc).__name__}: {exc}")
        raise


if __name__ == "__main__":
    main()
