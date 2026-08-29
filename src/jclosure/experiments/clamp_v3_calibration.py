"""Paired exploratory-v3 clamp calibration over layers and dictionaries."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from jclosure.baseline_guard import verify_manifest
from jclosure.clamp_v3 import (
    V3ClampThresholds,
    construct_dense_candidate,
    construct_sparse_candidate,
    validate_v3_clamp,
)
from jclosure.experiments.common import initialize_context, standard_parser
from jclosure.experiments.geometry_v3 import (
    NaturalityModel,
    _load_bank,
    _load_vocabularies,
)
from jclosure.geometry import DenseJMap
from jclosure.jstate import JStateEncoder
from jclosure.model import load_model_bundle
from jclosure.provenance import write_json_atomic
from jclosure.recorder import ResidualEditor

PROTOCOL = "exploratory_protocol_v3"


def _latest_bank_manifest(context) -> Path:
    paths = sorted(context.raw_dir.glob("geometry-v3-*/activation_bank_manifest.jsonl"))
    if not paths:
        raise FileNotFoundError("geometry v3 activation bank is missing")
    return paths[-1]


def _balanced_calibration_records(
    records: list[dict[str, Any]], attempts: int
) -> list[dict[str, Any]]:
    """Select one deterministic, maximally task-balanced calibration batch."""

    if attempts <= 0:
        raise ValueError("calibration attempts must be positive")
    by_family: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        by_family.setdefault(str(record["task_family"]), []).append(record)
    if not by_family:
        raise RuntimeError("calibration audit bank is empty")
    families = sorted(by_family)
    quotient, remainder = divmod(attempts, len(families))
    selected_by_family: dict[str, list[dict[str, Any]]] = {}
    for family_index, family in enumerate(families):
        required = quotient + int(family_index < remainder)
        ordered = sorted(
            by_family[family],
            key=lambda row: (str(row["prompt_hash"]), str(row["prompt_id"])),
        )
        if len(ordered) < required:
            raise RuntimeError(
                f"calibration family {family} has {len(ordered)} audit states; "
                f"requires {required}"
            )
        selected_by_family[family] = ordered[:required]
    selected: list[dict[str, Any]] = []
    for within_family_index in range(max(map(len, selected_by_family.values()))):
        for family in families:
            family_records = selected_by_family[family]
            if within_family_index < len(family_records):
                selected.append(family_records[within_family_index])
    if len(selected) != attempts:
        raise AssertionError("balanced calibration selector returned wrong count")
    return selected


def _calibration_trial_ids(
    anchor_id: str, donor_id: str, layer: int, method: str
) -> tuple[str, str]:
    base = hashlib.sha256(
        "\x1f".join((anchor_id, donor_id, str(layer))).encode()
    ).hexdigest()
    paired = hashlib.sha256("\x1f".join((base, method)).encode()).hexdigest()
    return base, paired


def _thresholds(config: dict[str, Any]) -> V3ClampThresholds:
    dense = config["v3_state"]["dense"]
    sparse = config["v3_state"]["sparse"]
    return V3ClampThresholds(
        dense_cosine=float(dense["cosine_threshold"]),
        dense_top10_overlap=float(dense["top10_overlap_threshold"]),
        sparse_support_f1=float(sparse["support_f1_threshold"]),
        sparse_weighted_jaccard=float(sparse["weighted_jaccard_threshold"]),
        sparse_coefficient_cosine=float(sparse["coefficient_cosine_threshold"]),
        sparse_reconstruction_cosine=float(sparse["reconstruction_cosine_threshold"]),
        rms_drift=float(config["v3_state"]["rms_drift_threshold"]),
        formal_displacement=float(config["v3_state"]["formal_displacement_fraction"]),
        sensitivity_displacement=float(
            config["v3_state"]["small_perturbation_min_fraction"]
        ),
    )


def _naturality_by_layer(context, records, layers) -> dict[int, NaturalityModel]:
    fit = [record for record in records if record["split"] == "fit"]
    output: dict[int, NaturalityModel] = {}
    for layer in layers:
        states = []
        for record in fit:
            payload = torch.load(
                context.root / record["activation_path"], map_location="cpu"
            )
            states.append(payload["activations"][layer][-1].float().numpy())
        output[layer] = NaturalityModel(
            int(context.config["geometry"]["pca_dimension"]),
            int(context.config["geometry"]["nearest_neighbors"]),
            float(context.config["geometry"]["naturality_quantile"]),
        ).fit(np.stack(states))
    return output


def _hook_sanity(bundle, payload: dict[str, Any], layers: list[int]) -> dict[str, Any]:
    input_ids = payload["input_ids"].to(bundle.hf_model.device)
    with torch.no_grad():
        clean = bundle.forward_logits(input_ids).detach().cpu()

    def zero(activation: torch.Tensor, layer: int) -> torch.Tensor:
        del layer
        return activation + torch.zeros_like(activation)

    def identity(activation: torch.Tensor, layer: int) -> torch.Tensor:
        del layer
        return activation

    with ResidualEditor(bundle.layers, {layer: zero for layer in layers}):
        with torch.no_grad():
            zero_logits = bundle.forward_logits(input_ids).detach().cpu()
    with ResidualEditor(bundle.layers, {layer: identity for layer in layers}):
        with torch.no_grad():
            identity_logits = bundle.forward_logits(input_ids).detach().cpu()
    with torch.no_grad():
        rerun = bundle.forward_logits(input_ids).detach().cpu()
    with torch.no_grad():
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
        ),
    }


def _scaled(vector: torch.Tensor, norm: float) -> torch.Tensor:
    denominator = torch.linalg.vector_norm(vector.float()).clamp_min(1e-20)
    return vector.float() * (norm / float(denominator))


def _row(
    *,
    context,
    encoder: JStateEncoder,
    dense_map: DenseJMap,
    naturality_model: NaturalityModel,
    anchor_record: dict[str, Any],
    donor_record: dict[str, Any],
    h: torch.Tensor,
    donor: torch.Tensor,
    layer: int,
    dictionary_size: int,
    method: str,
    tolerance: float,
    thresholds: V3ClampThresholds,
) -> dict[str, Any]:
    difference = donor - h
    natural_scale = float(torch.linalg.vector_norm(difference.float()).item())
    construction: dict[str, Any]
    if natural_scale <= 1e-20:
        candidate = h.clone()
        construction = {
            "status": "FAILED",
            "failure_reason": "degenerate_natural_difference",
        }
    elif method == "sparse_same_definition":
        remainder = encoder.decompose(difference, layer).remainder
        if float(torch.linalg.vector_norm(remainder.float())) <= 1e-20:
            candidate = h.clone()
            construction = {
                "status": "FAILED",
                "failure_reason": "degenerate_sparse_remainder",
            }
        else:
            preliminary = h + _scaled(
                remainder, thresholds.formal_displacement * natural_scale
            )
            candidate = construct_sparse_candidate(
                h, preliminary, layer=layer, encoder=encoder
            )
            construction = {"status": "CONSTRUCTED", "failure_reason": None}
    else:
        candidate, construction = construct_dense_candidate(
            h,
            difference,
            layer=layer,
            dense_map=dense_map,
            natural_scale=natural_scale,
            displacement_fraction=thresholds.formal_displacement,
            relative_tolerance=tolerance,
            optimized=method == "dense_optimized",
            naturality=lambda value: naturality_model.score(
                value.detach().cpu().float().numpy()
            ).natural,
            thresholds=thresholds,
        )
    finite = bool(torch.isfinite(candidate).all())
    exploding = bool(
        torch.linalg.vector_norm(candidate.float())
        > 4 * torch.linalg.vector_norm(h.float()).clamp_min(1e-20)
    )
    naturality = naturality_model.score(candidate.detach().cpu().float().numpy())
    state_definition = (
        "V3-Sparse" if method == "sparse_same_definition" else "V3-Dense"
    )
    validation = validate_v3_clamp(
        h,
        candidate,
        layer=layer,
        state_definition=state_definition,
        encoder=encoder,
        dense_map=dense_map,
        natural_scale=natural_scale,
        natural=naturality.natural,
        thresholds=thresholds,
    )
    reasons = list(validation.failure_reasons)
    if construction.get("failure_reason"):
        reasons.append(str(construction["failure_reason"]))
    if not finite:
        reasons.append("nan_or_inf")
    if exploding:
        reasons.append("activation_explosion")
    state_failure_reasons = [
        reason for reason in validation.failure_reasons if reason != "naturality"
    ]
    state_valid_before_naturality = bool(
        not state_failure_reasons
        and construction.get("status") != "FAILED"
        and finite
        and not exploding
    )
    formal_valid = state_valid_before_naturality and naturality.natural
    base_trial_id, paired = _calibration_trial_ids(
        anchor_record["prompt_id"], donor_record["prompt_id"], layer, method
    )
    sparse = validation.sparse_equality
    return {
        "schema_version": 3,
        "protocol_version": PROTOCOL,
        "run_id": context.run_id,
        "base_trial_id": base_trial_id,
        "paired_trial_id": paired,
        "prompt_id": anchor_record["prompt_id"],
        "donor_id": donor_record["prompt_id"],
        "task_family": anchor_record["task_family"],
        "template_id": anchor_record["template_id"],
        "layer": layer,
        "position": -1,
        "position_scope": "final",
        "dictionary_size": dictionary_size,
        "dictionary_hash": encoder.vocabulary.digest,
        "state_definition": state_definition,
        "method": method,
        "null_tolerance": tolerance if state_definition == "V3-Dense" else None,
        "construction_status": construction.get("status"),
        "construction_failure_reason": construction.get("failure_reason"),
        "optimization_iterations": construction.get("iterations"),
        "basis_dimension": construction.get("basis_dimension"),
        "valid": formal_valid,
        "formal_valid": formal_valid,
        "state_valid_before_naturality": state_valid_before_naturality,
        "small_perturbation_valid": validation.small_perturbation_valid,
        "exclusion_reason": None if formal_valid else ",".join(sorted(set(reasons))),
        "dense_cosine": validation.dense_cosine,
        "dense_profile_l2": validation.dense_profile_l2,
        "top10_overlap": validation.top10_overlap,
        "rms_drift": validation.rms_drift,
        "displacement_fraction": validation.displacement_fraction,
        "natural": validation.natural,
        "mahalanobis": naturality.mahalanobis,
        "knn_ratio": naturality.knn_ratio,
        "sparse_support_f1": None if sparse is None else sparse.support_f1,
        "sparse_weighted_jaccard": None if sparse is None else sparse.weighted_jaccard,
        "sparse_coefficient_cosine": None if sparse is None else sparse.coefficient_cosine,
        "sparse_coefficient_relative_l2": (
            None if sparse is None else sparse.coefficient_relative_l2
        ),
        "sparse_reconstruction_cosine": (
            None if sparse is None else sparse.reconstruction_cosine
        ),
        "sparse_reconstruction_relative_l2": (
            None if sparse is None else sparse.reconstruction_relative_l2
        ),
        "finite": finite,
        "activation_explosion": exploding,
    }


def summarize_calibration(
    frame: pd.DataFrame,
    *,
    config: dict[str, Any],
    hook_sanity: dict[str, Any],
) -> dict[str, Any]:
    required = int(config["clamp_v3"]["strict_valid_required"])
    natural_required = float(config["v3_state"]["naturality_valid_fraction"])
    optimization_max = float(
        config["clamp_v3"]["optimization_failure_max_fraction"]
    )
    rows: list[dict[str, Any]] = []
    grouped = frame.groupby(["layer", "dictionary_size", "method"], dropna=False)
    for (layer, size, method), group in grouped:
        state_valid = group[group["state_valid_before_naturality"]]
        valid = group[group["formal_valid"]]
        numerical_failure = group["construction_failure_reason"].isin(
            ["line_search_exhausted", "nan_or_inf"]
        ).sum()
        natural_fraction = (
            float(state_valid["natural"].mean()) if len(state_valid) else 0.0
        )
        reasons: list[str] = []
        if len(state_valid) < required:
            reasons.append("strict_valid_count")
        if natural_fraction < natural_required:
            reasons.append("naturality_valid_fraction")
        if numerical_failure / max(len(group), 1) >= optimization_max:
            reasons.append("optimization_numerical_failure")
        if not all(hook_sanity.values()):
            reasons.append("hook_or_determinism_control")
        if not bool(group["finite"].all()) or bool(group["activation_explosion"].any()):
            reasons.append("finite_or_activation_explosion")
        rows.append(
            {
                "layer": int(layer),
                "dictionary_size": int(size),
                "method": str(method),
                "state_definition": (
                    "V3-Sparse" if method == "sparse_same_definition" else "V3-Dense"
                ),
                "attempted": len(group),
                "strict_valid": len(state_valid),
                "strict_valid_rate": len(state_valid) / max(len(group), 1),
                "formal_natural_valid": len(valid),
                "small_perturbation_valid": int(
                    group["small_perturbation_valid"].sum()
                ),
                "natural_fraction_among_valid": natural_fraction,
                "optimization_numerical_failure": int(numerical_failure),
                "eligible": not reasons,
                "reasons": reasons,
            }
        )
    min_layers = int(config["clamp_v3"]["behavioral_min_eligible_layers"])
    by_protocol: dict[str, Any] = {}
    summary = pd.DataFrame(rows)
    for (size, method), group in summary.groupby(["dictionary_size", "method"]):
        eligible_layers = sorted(
            int(value) for value in group[group["eligible"]]["layer"].tolist()
        )
        key = f"{method}:M{int(size)}"
        eligible_l1 = [
            layer
            for layer in eligible_layers
            if len([value for value in eligible_layers if value > layer]) >= 3
        ]
        selected: list[int] = []
        if eligible_l1:
            selected.append(eligible_l1[0])
            middle = eligible_l1[len(eligible_l1) // 2]
            if middle not in selected:
                selected.append(middle)
        by_protocol[key] = {
            "dictionary_size": int(size),
            "method": str(method),
            "state_definition": (
                "V3-Sparse" if method == "sparse_same_definition" else "V3-Dense"
            ),
            "eligible_layers": eligible_layers,
            "behavioral_authorized": len(eligible_layers) >= min_layers and bool(eligible_l1),
            "selected_l1": selected,
        }
    authorized = [
        key for key, value in by_protocol.items() if value["behavioral_authorized"]
    ]
    method_preference = [
        str(value) for value in config["clamp_v3"]["method_preference"]
    ]
    selected_authorized: list[str] = []
    for size in sorted({value["dictionary_size"] for value in by_protocol.values()}):
        sparse = [
            key
            for key in authorized
            if by_protocol[key]["dictionary_size"] == size
            and by_protocol[key]["state_definition"] == "V3-Sparse"
        ]
        selected_authorized.extend(sorted(sparse))
        dense = [
            key
            for key in authorized
            if by_protocol[key]["dictionary_size"] == size
            and by_protocol[key]["state_definition"] == "V3-Dense"
        ]
        if dense:
            dense.sort(
                key=lambda key: (
                    method_preference.index(by_protocol[key]["method"])
                    if by_protocol[key]["method"] in method_preference
                    else len(method_preference),
                    key,
                )
            )
            selected_authorized.append(dense[0])
    return {
        "schema_version": 3,
        "protocol_version": PROTOCOL,
        "hook_sanity": hook_sanity,
        "layers": rows,
        "protocols": by_protocol,
        "all_behavioral_authorized_protocols": sorted(authorized),
        "behavioral_authorized_protocols": selected_authorized,
    }


def _assigned_layers(
    layers: list[int], *, shard_index: int, shard_count: int
) -> list[int]:
    if shard_count <= 0 or not 0 <= shard_index < shard_count:
        raise ValueError("invalid clamp calibration shard coordinates")
    assigned = [
        layer
        for layer in layers
        if int(hashlib.sha256(str(layer).encode()).hexdigest(), 16) % shard_count
        == shard_index
    ]
    if not assigned:
        raise RuntimeError("clamp calibration shard has no assigned layers")
    return assigned


def _latest_group_shards(
    context, *, shard_group_id: str, shard_count: int
) -> list[tuple[Path, dict[str, Any]]]:
    by_index: dict[int, tuple[str, Path, dict[str, Any]]] = {}
    for path in sorted(context.raw_dir.glob("clamp-v3-calibration-*/manifest.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("status") != "COMPLETED_SHARD":
            continue
        if payload.get("shard_group_id") != shard_group_id:
            continue
        index = int(payload["shard_index"])
        created = str(payload.get("created_at", ""))
        if index not in by_index or created > by_index[index][0]:
            by_index[index] = (created, path, payload)
    missing = sorted(set(range(shard_count)) - set(by_index))
    if missing:
        raise RuntimeError(f"clamp calibration merge is missing shards: {missing}")
    return [(by_index[index][1], by_index[index][2]) for index in range(shard_count)]


def _merge_shards(
    context,
    *,
    shard_group_id: str,
    shard_count: int,
    bank_manifest: Path,
) -> tuple[Path, dict[str, Any]]:
    selected = _latest_group_shards(
        context, shard_group_id=shard_group_id, shard_count=shard_count
    )
    expected_commit = selected[0][1].get("git_commit")
    expected_config = selected[0][1].get("config_digest")
    expected_bank = str(bank_manifest.relative_to(context.root))
    frames: list[pd.DataFrame] = []
    hook_records: list[dict[str, bool]] = []
    for _, manifest in selected:
        if manifest.get("git_commit") != expected_commit:
            raise RuntimeError("clamp calibration shards use different commits")
        if manifest.get("config_digest") != expected_config:
            raise RuntimeError("clamp calibration shards use different configs")
        if manifest.get("activation_bank_manifest") != expected_bank:
            raise RuntimeError("clamp calibration shards use different activation banks")
        frames.append(pd.read_parquet(context.root / manifest["candidate_records"]))
        hook_records.append(dict(manifest["hook_sanity"]))
    frame = pd.concat(frames, ignore_index=True)
    expected_layers = {
        int(value) for value in context.config["geometry"]["candidate_layers"]
    }
    if set(frame["layer"].astype(int)) != expected_layers:
        raise RuntimeError("merged clamp calibration does not cover every layer")
    duplicate_columns = [
        "base_trial_id",
        "method",
        "dictionary_size",
    ]
    if bool(frame.duplicated(duplicate_columns).any()):
        raise RuntimeError("merged clamp calibration contains duplicate candidates")
    hook_sanity = {
        key: all(record.get(key, False) for record in hook_records)
        for key in hook_records[0]
    }
    raw_path = context.raw_dir / context.run_id / "clamp_candidates.parquet"
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(raw_path, index=False)
    summary = summarize_calibration(
        frame, config=context.config, hook_sanity=hook_sanity
    )
    summary.update(
        {
            "run_id": context.run_id,
            "shard_group_id": shard_group_id,
            "source_shards": [manifest["run_id"] for _, manifest in selected],
            "activation_bank_manifest": expected_bank,
            "candidate_records": str(raw_path.relative_to(context.root)),
            "attempted": len(frame),
            "formal_valid": int(frame["formal_valid"].sum()),
            "v2_hash_guard": selected[0][1]["v2_hash_guard"],
        }
    )
    output = context.processed_dir / "clamp_v3_calibration.json"
    write_json_atomic(output, summary)
    pd.DataFrame(summary["layers"]).to_parquet(
        context.processed_dir / "clamp_v3_calibration.parquet", index=False
    )
    return output, summary


def main() -> None:
    parser = standard_parser(
        "Calibrate exploratory protocol v3 clamps", "configs/geometry_v3.yaml"
    )
    parser.add_argument("--bank-manifest")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-group-id", default=os.environ.get("JCLOSURE_SHARD_GROUP_ID"))
    parser.add_argument("--merge-only", action="store_true")
    args = parser.parse_args()
    context = initialize_context("clamp-v3-calibration", args)
    try:
        immutable = verify_manifest(
            context.root, context.root / "artifacts/phase0_v2_immutable.sha256.json"
        )
        bank_manifest = (
            Path(args.bank_manifest).resolve()
            if args.bank_manifest
            else _latest_bank_manifest(context)
        )
        if args.merge_only:
            if not args.shard_group_id:
                raise ValueError("--merge-only requires --shard-group-id")
            output, summary = _merge_shards(
                context,
                shard_group_id=str(args.shard_group_id),
                shard_count=int(args.shard_count),
                bank_manifest=bank_manifest,
            )
            context.finish(
                "COMPLETED",
                calibration=str(output.relative_to(context.root)),
                shard_group_id=args.shard_group_id,
                source_shards=summary["source_shards"],
                behavioral_authorized_protocols=summary[
                    "behavioral_authorized_protocols"
                ],
                attempted=summary["attempted"],
                formal_valid=summary["formal_valid"],
            )
            return
        if args.dry_run:
            context.finish(
                "DRY_RUN",
                v2_hash_guard=immutable,
                shard_index=args.shard_index,
                shard_count=args.shard_count,
                shard_group_id=args.shard_group_id,
            )
            return
        records = _load_bank(context.root, bank_manifest)
        audit = [record for record in records if record["split"] == "audit"]
        fit_by_family: dict[str, list[dict[str, Any]]] = {}
        for record in records:
            if record["split"] == "fit":
                fit_by_family.setdefault(record["task_family"], []).append(record)
        for family_records in fit_by_family.values():
            family_records.sort(
                key=lambda row: (str(row["prompt_hash"]), str(row["prompt_id"]))
            )
        attempts = int(
            context.config["clamp_v3"]["attempts_per_layer_method_dictionary"]
        )
        if args.limit is not None:
            attempts = min(attempts, args.limit)
        audit = _balanced_calibration_records(audit, attempts)
        all_layers = [
            int(value) for value in context.config["geometry"]["candidate_layers"]
        ]
        layers = _assigned_layers(
            all_layers,
            shard_index=int(args.shard_index),
            shard_count=int(args.shard_count),
        )
        naturality = _naturality_by_layer(context, records, layers)
        bundle = load_model_bundle(context.config)
        first_payload = torch.load(
            context.root / audit[0]["activation_path"], map_location="cpu"
        )
        hook_sanity = _hook_sanity(bundle, first_payload, layers)
        thresholds = _thresholds(context.config)
        tolerance = float(context.config["geometry"]["formal_null_tolerance"])
        rows: list[dict[str, Any]] = []
        completed_parts: list[str] = []
        progress_path = context.raw_dir / context.run_id / "clamp_progress.json"
        for size, vocabulary in _load_vocabularies(context).items():
            encoder = JStateEncoder.from_lens(
                bundle.lens,
                bundle.unembedding_weight,
                vocabulary,
                k=int(context.config["jstate"]["k"]),
                lazy=True,
                protocol_version=PROTOCOL,
                direction_chunk_size=int(
                    context.config["jstate"].get("direction_chunk_size", 512)
                ),
            )
            dense_map = DenseJMap.from_encoder(encoder)
            for layer in layers:
                combination_rows: list[dict[str, Any]] = []
                for index, anchor_record in enumerate(audit):
                    donors = fit_by_family[anchor_record["task_family"]]
                    donor_record = donors[index % len(donors)]
                    anchor_payload = torch.load(
                        context.root / anchor_record["activation_path"], map_location="cpu"
                    )
                    donor_payload = torch.load(
                        context.root / donor_record["activation_path"], map_location="cpu"
                    )
                    h = anchor_payload["activations"][layer][-1].to(
                        bundle.hf_model.device
                    ).float()
                    donor = donor_payload["activations"][layer][-1].to(
                        bundle.hf_model.device
                    ).float()
                    for method in context.config["clamp_v3"]["methods"]:
                        row = _row(
                            context=context,
                            encoder=encoder,
                            dense_map=dense_map,
                            naturality_model=naturality[layer],
                            anchor_record=anchor_record,
                            donor_record=donor_record,
                            h=h,
                            donor=donor,
                            layer=layer,
                            dictionary_size=size,
                            method=str(method),
                            tolerance=tolerance,
                            thresholds=thresholds,
                        )
                        rows.append(row)
                        combination_rows.append(row)
                part_path = (
                    context.raw_dir
                    / context.run_id
                    / f"clamp_part-M{size}-L{layer}.parquet"
                )
                pd.DataFrame(combination_rows).to_parquet(part_path, index=False)
                completed_parts.append(str(part_path.relative_to(context.root)))
                write_json_atomic(
                    progress_path,
                    {
                        "schema_version": 3,
                        "protocol_version": PROTOCOL,
                        "run_id": context.run_id,
                        "shard_group_id": args.shard_group_id,
                        "shard_index": args.shard_index,
                        "shard_count": args.shard_count,
                        "completed_parts": completed_parts,
                        "records_written": len(rows),
                        "status": "RUNNING",
                    },
                )
            dense_map._device_cache.clear()
            encoder._device_directions.clear()
            encoder._device_raw_directions.clear()
            torch.cuda.empty_cache()
        raw_path = (
            context.raw_dir
            / context.run_id
            / f"clamp_candidates-shard-{int(args.shard_index):03d}.parquet"
        )
        frame = pd.DataFrame(rows)
        frame.to_parquet(raw_path, index=False)
        summary = summarize_calibration(
            frame, config=context.config, hook_sanity=hook_sanity
        )
        summary.update(
            {
                "run_id": context.run_id,
                "shard_group_id": args.shard_group_id,
                "shard_index": args.shard_index,
                "shard_count": args.shard_count,
                "activation_bank_manifest": str(bank_manifest.relative_to(context.root)),
                "candidate_records": str(raw_path.relative_to(context.root)),
                "attempted": len(frame),
                "formal_valid": int(frame["formal_valid"].sum()),
                "v2_hash_guard": immutable,
            }
        )
        output = (
            context.processed_dir / f"clamp_v3_calibration_{context.run_id}.json"
        )
        write_json_atomic(output, summary)
        pd.DataFrame(summary["layers"]).to_parquet(
            context.processed_dir / f"clamp_v3_calibration_{context.run_id}.parquet",
            index=False,
        )
        write_json_atomic(
            progress_path,
            {
                "schema_version": 3,
                "protocol_version": PROTOCOL,
                "run_id": context.run_id,
                "shard_group_id": args.shard_group_id,
                "shard_index": args.shard_index,
                "shard_count": args.shard_count,
                "completed_parts": completed_parts,
                "records_written": len(rows),
                "status": "COMPLETED",
                "merged_output": str(raw_path.relative_to(context.root)),
            },
        )
        context.finish(
            "COMPLETED_SHARD",
            calibration=str(output.relative_to(context.root)),
            activation_bank_manifest=str(bank_manifest.relative_to(context.root)),
            candidate_records=str(raw_path.relative_to(context.root)),
            hook_sanity=hook_sanity,
            v2_hash_guard=immutable,
            shard_group_id=args.shard_group_id,
            shard_index=args.shard_index,
            shard_count=args.shard_count,
            completed_parts=completed_parts,
            behavioral_authorized_protocols=summary["behavioral_authorized_protocols"],
            attempted=len(frame),
            formal_valid=int(frame["formal_valid"].sum()),
        )
    except KeyboardInterrupt:
        context.finish("FAILED", error="KeyboardInterrupt: run cancelled")
        raise
    except Exception as exc:
        context.finish("FAILED", error=f"{type(exc).__name__}: {exc}")
        raise


if __name__ == "__main__":
    main()
