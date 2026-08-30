"""Independent direct-L1 intervention/restoration calibration for v3.1."""

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

from jclosure.config import config_digest
from jclosure.experiments.common import initialize_context, standard_parser
from jclosure.geometry import DenseJMap
from jclosure.jstate import ConceptVocabulary, JStateEncoder
from jclosure.model import load_model_bundle
from jclosure.provenance import append_jsonl, sha256_file, write_json_atomic
from jclosure.recorder import ActivationRecorder, ResidualEditor
from jclosure.runtime_v3_1 import (
    PROTOCOL_V31,
    construct_initial_sequence,
    fit_naturality_models,
    load_v31_domain,
    match_donor,
    replacement_transform,
    restore_sequence,
    select_positions,
    teacher_preanswer_prefix,
    tensor_digest,
    v31_thresholds,
)


def _shard_invariant_config_digest(manifest: dict[str, Any]) -> str:
    config = copy.deepcopy(manifest.get("config", {}))
    if isinstance(config.get("model"), dict):
        config["model"].pop("device", None)
    return config_digest(config)


def _hook_sanity(bundle, input_ids: torch.Tensor, layer: int) -> dict[str, bool]:
    with torch.no_grad():
        clean = bundle.forward_logits(input_ids).detach().cpu()

    def zero(activation: torch.Tensor, current_layer: int) -> torch.Tensor:
        del current_layer
        return activation + torch.zeros_like(activation)

    def identity(activation: torch.Tensor, current_layer: int) -> torch.Tensor:
        del current_layer
        return activation

    with ResidualEditor(bundle.layers, {layer: zero}):
        with torch.no_grad():
            zero_logits = bundle.forward_logits(input_ids).detach().cpu()
    with ResidualEditor(bundle.layers, {layer: identity}):
        with torch.no_grad():
            identity_logits = bundle.forward_logits(input_ids).detach().cpu()
    with torch.no_grad():
        rerun = bundle.forward_logits(input_ids).detach().cpu()
        cleanup = bundle.forward_logits(input_ids).detach().cpu()
    return {
        "zero_exact": bool(torch.equal(clean, zero_logits)),
        "identity_exact": bool(torch.equal(clean, identity_logits)),
        "determinism_exact": bool(torch.equal(clean, rerun)),
        "cleanup_exact": bool(torch.equal(clean, cleanup)),
        "finite": bool(
            torch.isfinite(clean).all()
            and torch.isfinite(zero_logits).all()
            and torch.isfinite(identity_logits).all()
            and torch.isfinite(rerun).all()
            and torch.isfinite(cleanup).all()
        ),
    }


def _chain_transform(
    activation: torch.Tensor,
    layer: int,
    *,
    clean: torch.Tensor,
    donor: torch.Tensor,
    positions: tuple[int, ...],
    dense_map: DenseJMap,
    encoder: JStateEncoder,
    naturality,
    tolerance: float,
    optimized: bool,
    thresholds,
    capture: dict[int, dict[str, Any]],
) -> torch.Tensor:
    restored, events, passed = restore_sequence(
        clean.to(activation.device),
        activation[0].float(),
        donor.to(activation.device),
        positions=positions,
        layer=layer,
        dense_map=dense_map,
        encoder=encoder,
        naturality=naturality,
        tolerance=tolerance,
        optimized=optimized,
        thresholds=thresholds,
    )
    capture[layer] = {
        "layer": layer,
        "passed": passed,
        "events": [asdict(event) for event in events],
    }
    return restored.unsqueeze(0).to(activation.dtype)


def _run_runtime_chain(
    *,
    bundle,
    input_ids: torch.Tensor,
    candidate: torch.Tensor,
    initial_positions: tuple[int, ...],
    future_layers: list[int],
    clean_sequences: dict[int, torch.Tensor],
    donor_sequences: dict[int, torch.Tensor],
    restoration_positions: tuple[int, ...],
    dense_map: DenseJMap,
    encoder: JStateEncoder,
    naturality: dict[int, Any],
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
    return events, bool(
        len(events) == len(future_layers) and all(event["passed"] for event in events)
    )


def _bank_manifest(context) -> Path:
    paths = sorted(
        context.raw_dir.glob("calibration-v3-1-*/activation_bank_manifest.jsonl")
    )
    if not paths:
        raise FileNotFoundError("v3.1 activation bank manifest is missing")
    return paths[-1]


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def _build_bank(context, bundle, *, limit: int | None) -> Path:
    layers = list(range(23, int(context.config["v3_1"]["final_layer"]) + 1))
    examples = [
        *(
            ("naturality_fit", item)
            for item in load_v31_domain(context.root, "naturality_fit")
        ),
        *(
            ("calibration", item)
            for item in load_v31_domain(context.root, "calibration")
        ),
    ]
    if limit is not None:
        examples = examples[:limit]
    directory = context.raw_dir / context.run_id / "activation_bank"
    directory.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    for index, (domain, example) in enumerate(examples):
        input_ids, answer_id, teacher_correct, exclusion = teacher_preanswer_prefix(
            bundle, example
        )
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
        prompt_hash = hashlib.sha256(example.prompt.strip().encode()).hexdigest()
        records.append(
            {
                "schema_version": 4,
                "protocol_version": PROTOCOL_V31,
                "run_id": context.run_id,
                "domain": domain,
                "prompt_id": example.example_id,
                "prompt_hash": prompt_hash,
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
                "activation_shapes": {
                    str(layer): list(payload["activations"][layer].shape)
                    for layer in layers
                },
            }
        )
    manifest = context.raw_dir / context.run_id / "activation_bank_manifest.jsonl"
    append_jsonl(manifest, records)
    return manifest


def _load_encoder(context, bundle):
    size = int(context.config["v3_1"]["dictionary_size"])
    vocabulary = ConceptVocabulary.from_json(
        context.root / "results/processed" / f"concept_vocabulary_v2_{size}.json"
    )
    encoder = JStateEncoder.from_lens(
        bundle.lens,
        bundle.unembedding_weight,
        vocabulary,
        k=int(context.config["jstate"]["k"]),
        lazy=True,
        protocol_version=PROTOCOL_V31,
        direction_chunk_size=int(
            context.config["jstate"].get("direction_chunk_size", 512)
        ),
    )
    return vocabulary, encoder, DenseJMap.from_encoder(encoder)


def _payload(root: Path, record: dict[str, Any], device) -> dict[str, Any]:
    value = torch.load(root / record["activation_path"], map_location="cpu")
    value["input_ids"] = value["input_ids"].to(device)
    return value


def _calibration_rows(
    context,
    bundle,
    bank: Path,
    *,
    shard_index: int,
    shard_count: int,
    limit: int | None,
):
    records = _read_jsonl(bank)
    fit = [row for row in records if row["domain"] == "naturality_fit"]
    calibration = [
        row
        for row in records
        if row["domain"] == "calibration" and row["teacher_correct"]
    ]
    target = int(context.config["v3_1"]["attempts_per_layer"])
    if limit is not None:
        target = min(target, int(limit))
    calibration = sorted(
        calibration, key=lambda row: (row["prompt_hash"], row["prompt_id"])
    )[:target]
    if len(calibration) < target:
        raise RuntimeError(
            f"v3.1 calibration has {len(calibration)}/{target} teacher-correct anchors"
        )
    all_layers = [
        int(value) for value in context.config["v3_1"]["candidate_initial_layers"]
    ]
    l1_values = [
        layer
        for layer in all_layers
        if int(hashlib.sha256(str(layer).encode()).hexdigest(), 16) % shard_count
        == shard_index
    ]
    if not l1_values:
        raise RuntimeError("v3.1 calibration shard has no L1")
    final_layer = int(context.config["v3_1"]["final_layer"])
    layers = list(range(min(l1_values), final_layer + 1))
    naturality = {
        scope: fit_naturality_models(
            context.root, fit, layers, scope=scope, config=context.config
        )
        for scope in context.config["v3_1"]["position_scopes"]
    }
    vocabulary, encoder, dense_map = _load_encoder(context, bundle)
    thresholds = v31_thresholds(context.config)
    tolerance = float(context.config["geometry"]["formal_null_tolerance"])
    strength = float(context.config["v3_1"]["initial_strength"])
    sanity_payload = _payload(context.root, calibration[0], bundle.hf_model.device)
    sanity_by_l1 = {
        l1: _hook_sanity(bundle, sanity_payload["input_ids"], l1) for l1 in l1_values
    }
    rows: list[dict[str, Any]] = []
    for l1 in l1_values:
        future_layers = list(range(l1 + 1, final_layer + 1))
        for anchor in calibration:
            donor = match_donor(anchor, fit)
            if donor is None:
                rows.append(
                    {
                        "schema_version": 4,
                        "protocol_version": PROTOCOL_V31,
                        "run_id": context.run_id,
                        "prompt_id": anchor["prompt_id"],
                        "l1": l1,
                        "position_scope": "unmatched",
                        "initial_valid": False,
                        "chain_valid": False,
                        "exclusion_reason": "donor_length_or_template_mismatch",
                    }
                )
                continue
            anchor_payload = _payload(context.root, anchor, bundle.hf_model.device)
            donor_payload = _payload(context.root, donor, bundle.hf_model.device)
            input_ids = anchor_payload["input_ids"]
            sequence_length = int(input_ids.shape[-1])
            for scope in context.config["v3_1"]["position_scopes"]:
                positions = select_positions(sequence_length, str(scope))
                clean_l1 = anchor_payload["activations"][l1].to(bundle.hf_model.device)
                donor_l1 = donor_payload["activations"][l1].to(bundle.hf_model.device)
                candidate, initial_positions, initial_valid, aggregate = (
                    construct_initial_sequence(
                        clean_l1,
                        donor_l1,
                        positions=positions,
                        layer=l1,
                        dense_map=dense_map,
                        encoder=encoder,
                        naturality=naturality[str(scope)][l1],
                        tolerance=tolerance,
                        strength=strength,
                        thresholds=thresholds,
                        require_per_position_displacement=scope == "final",
                    )
                )
                base = {
                    "schema_version": 4,
                    "protocol_version": PROTOCOL_V31,
                    "run_id": context.run_id,
                    "base_trial_id": hashlib.sha256(
                        f"{anchor['prompt_id']}\x1f{donor['prompt_id']}\x1f{l1}\x1f{scope}".encode()
                    ).hexdigest(),
                    "prompt_id": anchor["prompt_id"],
                    "donor_id": donor["prompt_id"],
                    "task_family": anchor["task_family"],
                    "template_id": anchor["template_id"],
                    "l1": l1,
                    "position_scope": scope,
                    "sequence_length": sequence_length,
                    "dictionary_size": len(vocabulary.token_ids),
                    "dictionary_hash": vocabulary.digest,
                    "initial_valid": initial_valid,
                    "initial_aggregate_displacement_fraction": aggregate,
                    "initial_positions": initial_positions,
                    "candidate_sha256": tensor_digest(candidate),
                    "restoration_layers": future_layers,
                    "hook_sanity": sanity_by_l1[l1],
                }
                if not initial_valid:
                    rows.append(
                        {
                            **base,
                            "chain_mode": "none",
                            "chain_valid": False,
                            "restoration_events": [],
                            "exclusion_reason": "initial_intervention_invalid",
                        }
                    )
                    continue
                clean_sequences = {
                    layer: anchor_payload["activations"][layer].to(
                        bundle.hf_model.device
                    )
                    for layer in future_layers
                }
                donor_sequences = {
                    layer: donor_payload["activations"][layer].to(
                        bundle.hf_model.device
                    )
                    for layer in future_layers
                }
                with (
                    ResidualEditor(
                        bundle.layers, {l1: replacement_transform(candidate, positions)}
                    ),
                    ActivationRecorder(bundle.layers, at=future_layers) as recorder,
                ):
                    with torch.no_grad():
                        bundle.forward_logits(input_ids)
                observed = {
                    layer: recorder.activations[layer][0].detach().float()
                    for layer in future_layers
                }
                isolated_events = []
                for layer in future_layers:
                    _, events, passed = restore_sequence(
                        clean_sequences[layer],
                        observed[layer],
                        donor_sequences[layer],
                        positions=positions,
                        layer=layer,
                        dense_map=dense_map,
                        encoder=encoder,
                        naturality=naturality[str(scope)][layer],
                        tolerance=tolerance,
                        optimized=False,
                        thresholds=thresholds,
                    )
                    isolated_events.append(
                        {
                            "layer": layer,
                            "passed": passed,
                            "events": [asdict(event) for event in events],
                        }
                    )
                chain_results: dict[str, Any] = {}
                final_positions = (sequence_length - 1,)
                for chain_mode, chain_positions in (
                    ("persistent_final", final_positions),
                    ("persistent_all", positions),
                ):
                    chain_events, chain_passed = _run_runtime_chain(
                        bundle=bundle,
                        input_ids=input_ids,
                        candidate=candidate,
                        initial_positions=positions,
                        future_layers=future_layers,
                        clean_sequences=clean_sequences,
                        donor_sequences=donor_sequences,
                        restoration_positions=chain_positions,
                        dense_map=dense_map,
                        encoder=encoder,
                        naturality=naturality[str(scope)],
                        tolerance=tolerance,
                        optimized=False,
                        thresholds=thresholds,
                    )
                    chain_results[chain_mode] = {
                        "passed": chain_passed,
                        "events": chain_events,
                        "positions": list(chain_positions),
                    }
                rows.append(
                    {
                        **base,
                        "chain_mode": "runtime_matched",
                        "chain_valid": all(
                            value["passed"] for value in chain_results.values()
                        ),
                        "restoration_events": isolated_events,
                        "runtime_chains": chain_results,
                        "exclusion_reason": None,
                    }
                )
    return rows


def _summarize(frame: pd.DataFrame, config: dict[str, Any]) -> dict[str, Any]:
    required = int(config["v3_1"]["valid_required"])
    minimum_later = int(config["v3_1"]["minimum_restoration_layers"])
    summaries = []
    protocols = []
    for (l1, scope), group in frame[frame["position_scope"] != "unmatched"].groupby(
        ["l1", "position_scope"]
    ):
        initial = int(group["initial_valid"].sum())
        isolated_layer_counts: dict[int, int] = {}
        chain_layer_counts: dict[str, dict[int, int]] = {
            "persistent_final": {},
            "persistent_all": {},
        }
        complete_chain_counts = {"persistent_final": 0, "persistent_all": 0}
        construction_failures = 0
        sanity_passed = True
        for _, row in group[group["initial_valid"]].iterrows():
            sanity = row.get("hook_sanity", {})
            sanity_passed = sanity_passed and bool(sanity) and all(sanity.values())
            for position in row.get("initial_positions", []):
                if position.get("construction", {}).get("status") == "FAILED":
                    construction_failures += 1
            for layer_event in row.get("restoration_events", []):
                layer = int(layer_event["layer"])
                isolated_layer_counts[layer] = isolated_layer_counts.get(
                    layer, 0
                ) + int(layer_event["passed"])
            chains = row.get("runtime_chains", {})
            for mode in complete_chain_counts:
                chain = chains.get(mode, {})
                complete_chain_counts[mode] += int(chain.get("passed", False))
                for layer_event in chain.get("events", []):
                    layer = int(layer_event["layer"])
                    values = chain_layer_counts[mode]
                    values[layer] = values.get(layer, 0) + int(layer_event["passed"])
        all_layers = set(isolated_layer_counts)
        for values in chain_layer_counts.values():
            all_layers &= set(values)
        eligible_restoration = sorted(
            layer
            for layer in all_layers
            if isolated_layer_counts[layer] >= required
            and all(
                chain_layer_counts[mode][layer] >= required
                for mode in chain_layer_counts
            )
        )
        numerical_failure_rate = construction_failures / max(
            1,
            sum(len(row.get("initial_positions", [])) for _, row in group.iterrows()),
        )
        intervention_eligible = initial >= required and numerical_failure_rate < 0.05
        chain_eligible = all(
            value >= required for value in complete_chain_counts.values()
        )
        authorized = (
            intervention_eligible
            and chain_eligible
            and sanity_passed
            and len(eligible_restoration) >= minimum_later
        )
        summaries.append(
            {
                "l1": int(l1),
                "position_scope": str(scope),
                "attempted": len(group),
                "intervention_valid": initial,
                "intervention_eligible": intervention_eligible,
                "isolated_restoration_valid_counts": {
                    str(key): value
                    for key, value in sorted(isolated_layer_counts.items())
                },
                "runtime_restoration_valid_counts": {
                    mode: {str(key): value for key, value in sorted(values.items())}
                    for mode, values in chain_layer_counts.items()
                },
                "complete_chain_valid_counts": complete_chain_counts,
                "restoration_eligible_layers": eligible_restoration,
                "hook_sanity_passed": sanity_passed,
                "optimization_numerical_failure_rate": numerical_failure_rate,
                "authorized": authorized,
            }
        )
        if authorized and scope == "all_non_padding":
            protocols.append(
                {
                    "protocol_key": f"dense_optimized:M4096:L{int(l1)}",
                    "l1": int(l1),
                    "initial_scope": "all_non_padding",
                    "restoration_layers": eligible_restoration,
                }
            )
    protocols.sort(key=lambda row: row["l1"])
    return {
        "schema_version": 4,
        "protocol_version": PROTOCOL_V31,
        "layers": summaries,
        "authorized_protocols": protocols[:1],
        "behavioral_authorized": bool(protocols),
        "selection_rule": "earliest all-non-padding L1 with >=3 restoration-eligible later layers",
    }


def main() -> None:
    parser = standard_parser(
        "Calibrate direct-L1 corrective protocol v3.1", "configs/closure_v3_1.yaml"
    )
    parser.add_argument(
        "--stage", choices=("bank", "calibrate", "merge", "smoke"), default="calibrate"
    )
    parser.add_argument("--bank-manifest")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument(
        "--shard-group-id", default=os.environ.get("JCLOSURE_SHARD_GROUP_ID", "single")
    )
    args = parser.parse_args()
    context = initialize_context("calibration-v3-1", args)
    try:
        bundle = (
            None
            if args.dry_run or args.stage == "merge"
            else load_model_bundle(context.config)
        )
        if args.dry_run:
            context.finish("DRY_RUN", stage=args.stage)
            return
        if args.stage in {"bank", "smoke"}:
            assert bundle is not None
            bank = _build_bank(
                context,
                bundle,
                limit=args.limit or (8 if args.stage == "smoke" else None),
            )
            context.finish(
                "COMPLETED",
                stage=args.stage,
                activation_bank_manifest=str(bank.relative_to(context.root)),
            )
            return
        bank = (
            Path(args.bank_manifest).resolve()
            if args.bank_manifest
            else _bank_manifest(context)
        )
        if args.stage == "calibrate":
            assert bundle is not None
            rows = _calibration_rows(
                context,
                bundle,
                bank,
                shard_index=int(args.shard_index),
                shard_count=int(args.shard_count),
                limit=args.limit,
            )
            path = (
                context.raw_dir
                / context.run_id
                / f"calibration-shard-{args.shard_index:03d}.parquet"
            )
            path.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(rows).to_parquet(path, index=False)
            context.finish(
                "COMPLETED_SHARD",
                stage="calibrate",
                shard_group_id=args.shard_group_id,
                shard_index=args.shard_index,
                shard_count=args.shard_count,
                activation_bank_manifest=str(bank.relative_to(context.root)),
                records=str(path.relative_to(context.root)),
                record_count=len(rows),
            )
            return
        manifests = []
        for path in sorted(context.raw_dir.glob("calibration-v3-1-*/manifest.json")):
            value = json.loads(path.read_text(encoding="utf-8"))
            if (
                value.get("status") == "COMPLETED_SHARD"
                and value.get("shard_group_id") == args.shard_group_id
            ):
                manifests.append(value)
        by_index = {int(value["shard_index"]): value for value in manifests}
        if set(by_index) != set(range(int(args.shard_count))):
            raise RuntimeError("v3.1 merge is missing calibration shards")
        digests = {config_digest(value["config"]) for value in by_index.values()}
        if len(digests) != 1:
            raise RuntimeError("v3.1 calibration shard configs differ")
        frame = pd.concat(
            [
                pd.read_parquet(context.root / by_index[index]["records"])
                for index in sorted(by_index)
            ],
            ignore_index=True,
        )
        output_records = context.processed_dir / "closure_v3_1_calibration.parquet"
        frame.to_parquet(output_records, index=False)
        summary = _summarize(frame, context.config)
        summary.update(
            {
                "run_id": context.run_id,
                "activation_bank_manifest": str(bank.relative_to(context.root)),
                "records": str(output_records.relative_to(context.root)),
                "attempted": len(frame),
                "source_shards": [
                    by_index[index]["run_id"] for index in sorted(by_index)
                ],
            }
        )
        output = context.processed_dir / "closure_v3_1_calibration.json"
        write_json_atomic(output, summary)
        context.finish(
            "COMPLETED",
            stage="merge",
            calibration=str(output.relative_to(context.root)),
            behavioral_authorized=summary["behavioral_authorized"],
            attempted=len(frame),
        )
    except KeyboardInterrupt:
        context.finish("FAILED", error="KeyboardInterrupt: run cancelled")
        raise
    except Exception as exc:
        context.finish("FAILED", error=f"{type(exc).__name__}: {exc}")
        raise


if __name__ == "__main__":
    main()
