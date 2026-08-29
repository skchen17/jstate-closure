"""Behavioral closure and mediation runner for frozen exploratory protocol v3."""

from __future__ import annotations

import hashlib
from collections import defaultdict
from dataclasses import asdict
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from jclosure.clamp_v3 import (
    V3ClampThresholds,
    build_clamp_schedule,
    construct_sparse_candidate,
    project_dense_candidate,
    validate_v3_clamp,
)
from jclosure.experiments.common import initialize_context, standard_parser
from jclosure.experiments.geometry_v3 import NaturalityModel, _load_bank
from jclosure.geometry import DenseJMap, DenseNullProjector
from jclosure.interventions import matched_random_direction
from jclosure.jstate import ConceptVocabulary, JStateEncoder, jstate_distance
from jclosure.metrics import (
    answer_flip,
    jensen_shannon_from_logits,
    token_log_odds,
    token_probability,
)
from jclosure.model import load_model_bundle
from jclosure.protocol_v3 import verify_v3_freeze
from jclosure.provenance import append_jsonl
from jclosure.recorder import ActivationRecorder, ResidualEditor

PROTOCOL = "exploratory_protocol_v3"


def _answer_id(tokenizer: Any, answer: str) -> int | None:
    for surface in (" " + answer.strip(), answer):
        values = tokenizer.encode(surface, add_special_tokens=False)
        if len(values) == 1:
            return int(values[0])
    return None


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


def _bank_path(root: Path, freeze: dict[str, Any]) -> Path:
    values = [
        root / path
        for path in freeze["data_hashes"]
        if path.endswith("activation_bank_manifest.jsonl")
    ]
    if len(values) != 1:
        raise RuntimeError("freeze must name exactly one activation bank manifest")
    return values[0]


def _load_payload(root: Path, record: dict[str, Any]) -> dict[str, Any]:
    return torch.load(root / record["activation_path"], map_location="cpu")


def _fit_naturality(
    root: Path,
    records: list[dict[str, Any]],
    layers: list[int],
    *,
    scope: str,
    config: dict[str, Any],
) -> dict[int, NaturalityModel]:
    fit = [record for record in records if record["split"] == "fit"]
    output: dict[int, NaturalityModel] = {}
    for layer in layers:
        states: list[np.ndarray] = []
        for record in fit:
            payload = _load_payload(root, record)
            sequence = payload["activations"][layer].float().numpy()
            if scope == "final":
                states.append(sequence[-1])
            else:
                states.extend(sequence)
        output[layer] = NaturalityModel(
            int(config["geometry"]["pca_dimension"]),
            int(config["geometry"]["nearest_neighbors"]),
            float(config["geometry"]["naturality_quantile"]),
        ).fit(np.stack(states))
    return output


def _matched_donor(
    anchor: dict[str, Any],
    fit: list[dict[str, Any]],
    *,
    scope: str,
) -> dict[str, Any] | None:
    candidates = [
        record
        for record in fit
        if record["task_family"] == anchor["task_family"]
        and record["prompt_id"] != anchor["prompt_id"]
    ]
    if scope == "all_non_padding":
        candidates = [
            record
            for record in candidates
            if record["template_id"] == anchor["template_id"]
            and record["sequence_length"] == anchor["sequence_length"]
        ]
    if not candidates:
        return None
    index = int(anchor["prompt_hash"][:16], 16) % len(candidates)
    return sorted(candidates, key=lambda value: value["prompt_hash"])[index]


def _natural_collision_donor(
    root: Path,
    anchor: dict[str, Any],
    candidates: list[dict[str, Any]],
    *,
    layer: int,
    encoder: JStateEncoder,
    scope: str,
    state_definition: str,
) -> dict[str, Any] | None:
    valid = [
        record
        for record in candidates
        if record["task_family"] == anchor["task_family"]
        and record["prompt_id"] != anchor["prompt_id"]
    ]
    if scope == "all_non_padding":
        valid = [
            record
            for record in valid
            if record["template_id"] == anchor["template_id"]
            and record["sequence_length"] == anchor["sequence_length"]
        ]
    if not valid:
        return None
    anchor_payload = _load_payload(root, anchor)
    anchor_h = anchor_payload["activations"][layer][-1].float()
    anchor_state = encoder.encode(anchor_h, layer)
    ranked: list[tuple[float, float, str, dict[str, Any]]] = []
    for record in valid:
        payload = _load_payload(root, record)
        h = payload["activations"][layer][-1].float()
        metric = (
            "sparse_weighted_jaccard"
            if state_definition == "V3-Sparse"
            else "dense_cosine"
        )
        distance = jstate_distance(
            anchor_state, encoder.encode(h, layer), metric=metric
        )
        activation_distance = float(torch.linalg.vector_norm(h - anchor_h))
        ranked.append((distance, -activation_distance, record["prompt_hash"], record))
    ranked.sort(key=lambda value: (value[0], value[2]))
    near = ranked[: min(16, len(ranked))]
    near.sort(key=lambda value: (value[1], value[0], value[2]))
    return near[0][3]


def _variable_label(record: dict[str, Any]) -> int | None:
    variables = record.get("variables", {})
    family = record["task_family"]
    value: Any | None = None
    if family == "arithmetic":
        value = variables.get("c")
    elif family == "boolean_logic":
        value = variables.get("x")
    elif family == "graph_traversal":
        value = variables.get("target")
    elif family == "symbolic_planning":
        value = variables.get("query")
    elif family == "variable_binding":
        value = variables.get("query_index")
    elif family == "state_machine":
        value = variables.get("start")
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int | float):
        return int(value) % 2
    return int(hashlib.sha256(str(value).encode()).hexdigest(), 16) % 2


def _targeted_probes(
    root: Path,
    fit: list[dict[str, Any]],
    *,
    layer: int,
    min_auc: float,
) -> dict[str, tuple[torch.Tensor, float]]:
    by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in fit:
        if _variable_label(record) is not None:
            by_family[record["task_family"]].append(record)
    output: dict[str, tuple[torch.Tensor, float]] = {}
    for family, records in by_family.items():
        labels = np.array([_variable_label(record) for record in records], dtype=int)
        if len(records) < 20 or len(np.unique(labels)) < 2:
            continue
        states = np.stack(
            [
                _load_payload(root, record)["activations"][layer][-1].float().numpy()
                for record in records
            ]
        )
        folds = min(5, int(np.bincount(labels).min()))
        if folds < 2:
            continue
        model = LogisticRegression(max_iter=1000, random_state=0)
        probabilities = cross_val_predict(
            model,
            states,
            labels,
            cv=StratifiedKFold(folds, shuffle=True, random_state=0),
            method="predict_proba",
        )[:, 1]
        auc = float(roc_auc_score(labels, probabilities))
        if auc < min_auc:
            continue
        model.fit(states, labels)
        direction = torch.from_numpy(model.coef_[0]).float()
        if float(torch.linalg.vector_norm(direction)) > 1e-20:
            output[family] = (direction, auc)
    return output


def _matched_norm(direction: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    denominator = torch.linalg.vector_norm(direction.float()).clamp_min(1e-20)
    return direction.float() * (
        torch.linalg.vector_norm(reference.float()) / denominator
    )


def _state_preserving_delta(
    clean: torch.Tensor,
    difference: torch.Tensor,
    *,
    strength: float,
    layer: int,
    state_definition: str,
    encoder: JStateEncoder,
    dense_map: DenseJMap,
    tolerance: float,
) -> torch.Tensor:
    if state_definition == "V3-Sparse":
        preliminary = clean.float() + difference.float() * float(strength)
        return (
            construct_sparse_candidate(
                clean.float(), preliminary, layer=layer, encoder=encoder
            )
            - clean.float()
        )
    projector = DenseNullProjector(dense_map, layer)
    direction, basis, _ = projector.donor_projection(
        clean.float(),
        difference.float(),
        relative_tolerance=tolerance,
        sphere_tangent=True,
    )
    if basis.shape[1] == 0 or float(torch.linalg.vector_norm(direction)) <= 1e-20:
        return torch.zeros_like(clean, dtype=torch.float32)
    target = float(strength) * float(torch.linalg.vector_norm(difference.float()))
    try:
        tangent_step = projector.tangent_step_for_chord(clean, target)
    except ValueError:
        return torch.zeros_like(clean, dtype=torch.float32)
    tangent = direction.float() * (
        tangent_step / torch.linalg.vector_norm(direction.float()).clamp_min(1e-20)
    )
    return projector.retract_to_sphere(clean, tangent).float()


def _state_changing_component(
    clean: torch.Tensor,
    difference: torch.Tensor,
    *,
    layer: int,
    state_definition: str,
    encoder: JStateEncoder,
    dense_map: DenseJMap,
    tolerance: float,
) -> torch.Tensor:
    if state_definition == "V3-Sparse":
        component = encoder.decompose(difference.float(), layer).reconstruction
    else:
        projector = DenseNullProjector(dense_map, layer)
        null, _, _ = projector.donor_projection(
            clean.float(),
            difference.float(),
            relative_tolerance=tolerance,
            sphere_tangent=True,
        )
        component = difference.float() - null
    if float(torch.linalg.vector_norm(component.float())) <= 1e-20:
        return torch.zeros_like(difference, dtype=torch.float32)
    return _matched_norm(component, difference)


def _positions(schedule, layer: int) -> tuple[int, ...]:
    return tuple(
        position
        for current_layer, position in schedule.modified_layer_positions
        if current_layer == layer
    )


def _add_delta(
    activation: torch.Tensor,
    layer: int,
    *,
    delta_by_position: dict[int, torch.Tensor],
) -> torch.Tensor:
    del layer
    output = activation.clone()
    for position, delta in delta_by_position.items():
        output[:, position, :] += delta.to(output.device, output.dtype)
    return output


def _operational_clamp(
    activation: torch.Tensor,
    layer: int,
    *,
    positions: tuple[int, ...],
    clean_sequence: torch.Tensor,
    donor_sequence: torch.Tensor,
    state_definition: str,
    method: str,
    encoder: JStateEncoder,
    dense_map: DenseJMap,
    naturality: NaturalityModel,
    tolerance: float,
    thresholds: V3ClampThresholds,
    require_formal_displacement: bool,
    capture: dict[tuple[int, int], dict[str, Any]],
) -> torch.Tensor:
    output = activation.clone()
    for position in positions:
        current = activation[0, position].float()
        clean = clean_sequence[position].to(current.device).float()
        donor = donor_sequence[position].to(current.device).float()
        natural_scale = float(torch.linalg.vector_norm(donor - clean).item())
        if state_definition == "V3-Sparse":
            candidate = construct_sparse_candidate(
                clean, current, layer=layer, encoder=encoder
            )
            construction = {"status": "CONSTRUCTED", "failure_reason": None}
        else:
            candidate, construction = project_dense_candidate(
                clean,
                current,
                layer=layer,
                dense_map=dense_map,
                relative_tolerance=tolerance,
                optimized=method == "dense_optimized",
                naturality=lambda value: naturality.score(
                    value.detach().cpu().float().numpy()
                ).natural,
                thresholds=thresholds,
            )
        natural = naturality.score(candidate.detach().cpu().float().numpy()).natural
        validation = validate_v3_clamp(
            clean,
            candidate,
            layer=layer,
            state_definition=state_definition,
            encoder=encoder,
            dense_map=dense_map,
            natural_scale=natural_scale,
            natural=natural,
            thresholds=thresholds,
        )
        equality_failures = {
            reason
            for reason in validation.failure_reasons
            if reason
            not in {"displacement_below_sensitivity", "displacement_below_formal"}
        }
        clamp_valid = (
            validation.formal_valid
            if require_formal_displacement
            else not equality_failures
        )
        capture[(layer, position)] = {
            "validation": validation,
            "construction": construction,
            "clamp_valid": clamp_valid,
            "require_formal_displacement": require_formal_displacement,
        }
        output[:, position, :] = candidate.to(output.dtype)
    return output


def _run_condition(
    *,
    bundle,
    input_ids: torch.Tensor,
    answer_id: int,
    clean_logits: torch.Tensor,
    clean_by_layer: dict[int, torch.Tensor],
    donor_by_layer: dict[int, torch.Tensor],
    l0: int,
    l1: int,
    future_layers: list[int],
    source_delta: dict[int, torch.Tensor],
    condition: str,
    strength: float,
    mode: str,
    position_scope: str,
    state_definition: str,
    method: str,
    dictionary_size: int,
    encoder: JStateEncoder,
    dense_map: DenseJMap,
    naturality: dict[int, NaturalityModel],
    tolerance: float,
    thresholds: V3ClampThresholds,
) -> dict[str, Any]:
    sequence_length = input_ids.shape[-1]
    schedule = build_clamp_schedule(
        mode=mode,
        initial_layer=l1,
        future_layers=future_layers,
        position_scope=position_scope,
        sequence_length=sequence_length,
        attention_mask=torch.ones(sequence_length, dtype=torch.bool),
        explicit_positions=None,
        reasoning_span=None,
        state_definition=state_definition,
        dictionary_size=dictionary_size,
    )
    initial_positions = schedule.resolved_positions
    transforms: dict[int, Any] = {}
    if condition == "identity":
        transforms[l0] = partial(
            _add_delta,
            delta_by_position={
                position: torch.zeros_like(source_delta[position])
                for position in initial_positions
            },
        )
    elif condition != "clean":
        multiplier = 1.0 if condition == "state_preserving" else strength
        transforms[l0] = partial(
            _add_delta,
            delta_by_position={
                position: source_delta[position] * multiplier
                for position in initial_positions
            },
        )
    capture: dict[tuple[int, int], dict[str, Any]] = {}
    if condition == "state_preserving":
        for layer in schedule.selected_layers:
            transforms[layer] = partial(
                _operational_clamp,
                positions=_positions(schedule, layer),
                clean_sequence=clean_by_layer[layer],
                donor_sequence=donor_by_layer[layer],
                state_definition=state_definition,
                method=method,
                encoder=encoder,
                dense_map=dense_map,
                naturality=naturality[layer],
                tolerance=tolerance,
                thresholds=thresholds,
                require_formal_displacement=layer == l1,
                capture=capture,
            )
    record_layers = [l1, *future_layers]
    with (
        ResidualEditor(bundle.layers, transforms),
        ActivationRecorder(bundle.layers, at=record_layers) as recorder,
    ):
        with torch.no_grad():
            logits = bundle.forward_logits(input_ids)[0, -1].detach().float().cpu()
    observed = {
        layer: recorder.activations[layer][0, -1].detach().float().cpu()
        for layer in record_layers
    }
    state_metric = (
        "sparse_weighted_jaccard"
        if state_definition == "V3-Sparse"
        else "dense_cosine"
    )
    future = {
        str(layer): jstate_distance(
            encoder.encode(clean_by_layer[layer][-1].float(), layer),
            encoder.encode(observed[layer], layer),
            metric=state_metric,
        )
        for layer in future_layers
    }
    validations = [value["validation"] for value in capture.values()]
    valid = condition != "state_preserving" or (
        bool(validations) and all(value["clamp_valid"] for value in capture.values())
    )
    perturbation_locations = (
        []
        if condition == "clean"
        else [
            {"layer": l0, "position": position, "operation": condition}
            for position in initial_positions
        ]
    )
    clamp_locations = (
        [
            {"layer": layer, "position": position, "operation": "state_clamp"}
            for layer, position in schedule.modified_layer_positions
        ]
        if condition == "state_preserving"
        else []
    )
    return {
        "valid": valid,
        "exclusion_reason": (
            None
            if valid
            else ",".join(
                sorted(
                    {
                        reason
                        for validation in validations
                        for reason in validation.failure_reasons
                    }
                )
            )
        ),
        "hook_schedule": asdict(schedule),
        "actual_modified_locations": [*perturbation_locations, *clamp_locations],
        "clamp_validations": [
            {
                "layer": layer,
                "position": position,
                **asdict(value["validation"]),
                "construction": value["construction"],
                "clamp_valid": value["clamp_valid"],
                "require_formal_displacement": value[
                    "require_formal_displacement"
                ],
            }
            for (layer, position), value in sorted(capture.items())
        ],
        "metrics": {
            "js_divergence": jensen_shannon_from_logits(clean_logits, logits),
            "target_probability": token_probability(logits, answer_id),
            "target_probability_clean": token_probability(clean_logits, answer_id),
            "target_log_odds": token_log_odds(logits, answer_id),
            "target_log_odds_clean": token_log_odds(clean_logits, answer_id),
            "answer_flip": answer_flip(clean_logits, logits),
            "task_correct": int(torch.argmax(logits)) == answer_id,
            "next_layer_j_distance": next(iter(future.values()), None),
            "mean_future_j_distance": float(np.mean(list(future.values())))
            if future
            else None,
            "future_j_distances": future,
        },
    }


def main() -> None:
    parser = standard_parser(
        "Run frozen exploratory protocol v3 closure", "configs/closure_v3_pilot.yaml"
    )
    parser.add_argument("--protocol")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    args = parser.parse_args()
    context = initialize_context("closure-v3", args)
    try:
        freeze = verify_v3_freeze(context.root)
        protocol_keys = (
            [args.protocol]
            if args.protocol
            else sorted(freeze["eligible_protocols"])
        )
        if any(key not in freeze["eligible_protocols"] for key in protocol_keys):
            raise ValueError("requested protocol is not authorized by the v3 freeze")
        if args.dry_run:
            context.finish("DRY_RUN", protocols=protocol_keys)
            return
        bundle = load_model_bundle(context.config)
        bank_manifest = _bank_path(context.root, freeze)
        records = _load_bank(context.root, bank_manifest)
        audit = [
            record
            for record in records
            if record["split"] == "audit" and record["teacher_correct"]
        ]
        fit = [record for record in records if record["split"] == "fit"]
        target = int(context.config["run"]["valid_per_cell"])
        if args.limit is not None:
            target = min(target, args.limit)
        attempts: dict[str, int] = defaultdict(int)
        valid_counts: dict[str, int] = defaultdict(int)
        output_rows = 0
        for protocol_key in protocol_keys:
            protocol = freeze["eligible_protocols"][protocol_key]
            size = int(protocol["dictionary_size"])
            method = str(protocol["method"])
            state_definition = str(protocol["state_definition"])
            vocabulary = ConceptVocabulary.from_json(
                context.root
                / "results/processed"
                / f"concept_vocabulary_v2_{size}.json"
            )
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
            eligible_layers = [int(value) for value in protocol["eligible_layers"]]
            l1_values = [int(value) for value in protocol["selected_l1"]]
            all_layers = sorted(
                set(eligible_layers) | {max(0, value - 2) for value in l1_values}
            )
            naturality_by_scope = {
                scope: _fit_naturality(
                    context.root,
                    records,
                    eligible_layers,
                    scope=scope,
                    config=context.config,
                )
                for scope in context.config["closure_v3"]["position_scopes"]
            }
            probes = {
                l0: _targeted_probes(
                    context.root,
                    fit,
                    layer=l0,
                    min_auc=float(
                        context.config["closure_v3"]["targeted_probe_min_auc"]
                    ),
                )
                for l0 in {value - 2 for value in l1_values}
            }
            thresholds = _thresholds(context.config)
            tolerance = float(context.config["geometry"]["formal_null_tolerance"])
            for anchor_index, anchor in enumerate(audit):
                if int(anchor["prompt_hash"], 16) % args.shard_count != args.shard_index:
                    continue
                input_payload = _load_payload(context.root, anchor)
                input_ids = input_payload["input_ids"].to(bundle.hf_model.device)
                answer_id = _answer_id(bundle.tokenizer, anchor["teacher_answer"])
                if answer_id is None:
                    continue
                with torch.no_grad():
                    clean_logits = bundle.forward_logits(input_ids)[0, -1].float().cpu()
                clean_by_layer = {
                    layer: input_payload["activations"][layer].float()
                    for layer in all_layers
                }
                for l1 in l1_values:
                    l0 = l1 - 2
                    future_layers = [layer for layer in eligible_layers if layer > l1]
                    for scope in context.config["closure_v3"]["position_scopes"]:
                        donor = _matched_donor(anchor, fit, scope=scope)
                        if donor is None:
                            continue
                        donor_payload = _load_payload(context.root, donor)
                        sequence_length = input_ids.shape[-1]
                        positions = (
                            (sequence_length - 1,)
                            if scope == "final"
                            else tuple(range(sequence_length))
                        )
                        natural_collision = _natural_collision_donor(
                            context.root,
                            anchor,
                            fit,
                            layer=l0,
                            encoder=encoder,
                            scope=scope,
                            state_definition=state_definition,
                        )
                        for source in context.config["closure_v3"][
                            "perturbation_sources"
                        ]:
                            active_donor = donor
                            source_payload = donor_payload
                            probe_auc = None
                            if source == "natural_collision":
                                if natural_collision is None:
                                    continue
                                active_donor = natural_collision
                                source_payload = _load_payload(
                                    context.root, natural_collision
                                )
                            active_donor_by_layer = {
                                layer: source_payload["activations"][layer].float()
                                for layer in all_layers
                            }
                            source_delta: dict[int, torch.Tensor] = {}
                            for position in positions:
                                clean = clean_by_layer[l0][position].to(
                                    bundle.hf_model.device
                                )
                                difference = (
                                    source_payload["activations"][l0][position]
                                    .to(bundle.hf_model.device)
                                    .float()
                                    - clean
                                )
                                if source == "targeted_probe":
                                    probe = probes[l0].get(anchor["task_family"])
                                    if probe is None:
                                        source_delta = {}
                                        break
                                    direction, probe_auc = probe
                                    stripped = _state_preserving_delta(
                                        clean,
                                        direction,
                                        strength=1.0,
                                        layer=l0,
                                        state_definition=state_definition,
                                        encoder=encoder,
                                        dense_map=dense_map,
                                        tolerance=tolerance,
                                    )
                                    if float(torch.linalg.vector_norm(stripped)) <= 1e-20:
                                        source_delta = {}
                                        break
                                    source_delta[position] = _matched_norm(
                                        stripped, difference
                                    )
                                else:
                                    source_delta[position] = difference
                            if not source_delta:
                                continue
                            for strength in context.config["closure_v3"]["strengths"]:
                                for condition in context.config["closure_v3"]["conditions"]:
                                    if condition == "matched_random":
                                        deltas = {
                                            position: matched_random_direction(
                                                value,
                                                seed=context.seed
                                                + anchor_index
                                                + position
                                                + 1000 * l0,
                                            )
                                            for position, value in source_delta.items()
                                        }
                                    elif condition == "j_positive":
                                        deltas = {
                                            position: _state_changing_component(
                                                clean_by_layer[l0][position].to(
                                                    bundle.hf_model.device
                                                ),
                                                value,
                                                layer=l0,
                                                state_definition=state_definition,
                                                encoder=encoder,
                                                dense_map=dense_map,
                                                tolerance=tolerance,
                                            )
                                            for position, value in source_delta.items()
                                        }
                                    elif condition == "state_preserving":
                                        deltas = {
                                            position: _state_preserving_delta(
                                                clean_by_layer[l0][position].to(
                                                    bundle.hf_model.device
                                                ),
                                                value,
                                                strength=float(strength),
                                                layer=l0,
                                                state_definition=state_definition,
                                                encoder=encoder,
                                                dense_map=dense_map,
                                                tolerance=tolerance,
                                            )
                                            for position, value in source_delta.items()
                                        }
                                    else:
                                        deltas = source_delta
                                    modes = (
                                        context.config["closure_v3"]["modes"]
                                        if condition == "state_preserving"
                                        else ["single"]
                                    )
                                    for mode in modes:
                                        cell = ":".join(
                                            map(
                                                str,
                                                (
                                                    protocol_key,
                                                    anchor["task_family"],
                                                    l1,
                                                    scope,
                                                    source,
                                                    strength,
                                                    condition,
                                                    mode,
                                                ),
                                            )
                                        )
                                        if valid_counts[cell] >= target:
                                            continue
                                        attempts[cell] += 1
                                        if attempts[cell] > target * int(
                                            context.config["closure_v3"][
                                                "max_attempt_multiplier"
                                            ]
                                        ):
                                            continue
                                        result = _run_condition(
                                            bundle=bundle,
                                            input_ids=input_ids,
                                            answer_id=answer_id,
                                            clean_logits=clean_logits,
                                            clean_by_layer=clean_by_layer,
                                            donor_by_layer=active_donor_by_layer,
                                            l0=l0,
                                            l1=l1,
                                            future_layers=future_layers,
                                            source_delta=deltas,
                                            condition=condition,
                                            strength=float(strength),
                                            mode=str(mode),
                                            position_scope=str(scope),
                                            state_definition=state_definition,
                                            method=method,
                                            dictionary_size=size,
                                            encoder=encoder,
                                            dense_map=dense_map,
                                            naturality=naturality_by_scope[scope],
                                            tolerance=tolerance,
                                            thresholds=thresholds,
                                        )
                                        valid_counts[cell] += int(result["valid"])
                                        base_trial_id = hashlib.sha256(
                                            "\x1f".join(
                                                (
                                                    anchor["prompt_id"],
                                                    active_donor["prompt_id"],
                                                    str(l0),
                                                    str(l1),
                                                    str(scope),
                                                    source,
                                                    str(float(strength)),
                                                )
                                            ).encode()
                                        ).hexdigest()
                                        paired = hashlib.sha256(
                                            f"{base_trial_id}\x1f{condition}\x1f{mode}".encode()
                                        ).hexdigest()
                                        path = (
                                            context.raw_dir
                                            / context.run_id
                                            / "trials"
                                            / anchor["task_family"]
                                            / f"part-shard-{args.shard_index:03d}.jsonl"
                                        )
                                        append_jsonl(
                                            path,
                                            [
                                                {
                                                    "schema_version": 3,
                                                    "protocol_version": PROTOCOL,
                                                    "run_id": context.run_id,
                                                    "paired_trial_id": paired,
                                                    "base_trial_id": base_trial_id,
                                                    "prompt_id": anchor["prompt_id"],
                                                    "donor_id": active_donor["prompt_id"],
                                                    "template_id": anchor["template_id"],
                                                    "task_family": anchor["task_family"],
                                                    "state_definition": state_definition,
                                                    "dictionary_size": size,
                                                    "dictionary_hash": vocabulary.digest,
                                                    "layer": l1,
                                                    "l0": l0,
                                                    "future_layers": future_layers,
                                                    "position_scope": scope,
                                                    "condition": condition,
                                                    "source": source,
                                                    "strength": float(strength),
                                                    "clamp_mode": mode,
                                                    "probe_auc": probe_auc,
                                                    **result,
                                                }
                                            ],
                                        )
                                        output_rows += 1
        summary = pd.DataFrame(
            [
                {
                    "cell": cell,
                    "attempted": attempts[cell],
                    "valid": valid_counts[cell],
                    "target": target,
                }
                for cell in sorted(attempts)
            ]
        )
        summary_path = (
            context.processed_dir / f"closure_v3_cell_counts_{context.run_id}.parquet"
        )
        summary.to_parquet(summary_path, index=False)
        context.finish(
            "COMPLETED",
            freeze_status=freeze["status"],
            protocols=protocol_keys,
            trial_records=output_rows,
            cells=len(attempts),
            cells_reaching_target=sum(value >= target for value in valid_counts.values()),
            cell_summary=str(summary_path.relative_to(context.root)),
        )
    except KeyboardInterrupt:
        context.finish("FAILED", error="KeyboardInterrupt: run cancelled")
        raise
    except Exception as exc:
        context.finish("FAILED", error=f"{type(exc).__name__}: {exc}")
        raise


if __name__ == "__main__":
    main()
