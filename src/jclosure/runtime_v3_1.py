"""Shared runtime helpers for direct-L1 protocol v3.1 experiments."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from jclosure.clamp_v3 import (
    V3ClampThresholds,
    construct_dense_candidate,
    project_dense_candidate,
    validate_v3_clamp,
)
from jclosure.clamp_v3_1 import (
    validate_intervention_eligibility,
    validate_restoration_eligibility,
)
from jclosure.datasets import TaskExample, task_examples_from_json
from jclosure.experiments.geometry_v3 import NaturalityModel
from jclosure.geometry import DenseJMap
from jclosure.jstate import JStateEncoder
from jclosure.records_v3_1 import RestorationEvent

PROTOCOL_V31 = "corrective_exploratory_protocol_v3_1"


def v31_thresholds(config: dict[str, Any]) -> V3ClampThresholds:
    values = config["v3_1"]
    return V3ClampThresholds(
        dense_cosine=float(values["dense_cosine_threshold"]),
        dense_top10_overlap=float(values["top10_overlap_threshold"]),
        rms_drift=float(values["rms_drift_threshold"]),
        formal_displacement=float(values["formal_displacement_fraction"]),
        sensitivity_displacement=0.05,
    )


def answer_token_id(tokenizer: Any, answer: str) -> int | None:
    for surface in (answer.strip(), " " + answer.strip()):
        ids = tokenizer.encode(surface, add_special_tokens=False)
        if len(ids) == 1:
            return int(ids[0])
    return None


def encode_direct_prompt(bundle: Any, prompt: str) -> torch.Tensor:
    tokenizer = bundle.tokenizer
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            encoded = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=False,
                return_tensors="pt",
            )
            if isinstance(encoded, dict) or hasattr(encoded, "get"):
                input_ids = encoded.get("input_ids")
                if input_ids is None:
                    raise ValueError("chat template mapping has no input_ids")
                return input_ids.to(bundle.hf_model.device)
            return encoded.to(bundle.hf_model.device)
        except (TypeError, ValueError):
            pass
    return bundle.lens_model.encode(prompt, max_length=512)


@torch.no_grad()
def teacher_preanswer_prefix(
    bundle: Any,
    example: TaskExample,
    *,
    maximum_steps: int = 8,
) -> tuple[torch.Tensor, int | None, bool, str | None]:
    answer_id = answer_token_id(bundle.tokenizer, example.answer)
    if answer_id is None:
        return (
            encode_direct_prompt(bundle, example.prompt),
            None,
            False,
            "answer_not_single_token",
        )
    input_ids = encode_direct_prompt(bundle, example.prompt)
    eos = getattr(bundle.tokenizer, "eos_token_id", None)
    for _ in range(maximum_steps + 1):
        logits = bundle.forward_logits(input_ids)[0, -1]
        token = int(torch.argmax(logits))
        if token == answer_id:
            return input_ids, answer_id, True, None
        if eos is not None and token == int(eos):
            return input_ids, answer_id, False, "eos_before_answer"
        surface = bundle.tokenizer.decode([token], skip_special_tokens=False)
        if surface.strip():
            return input_ids, answer_id, False, "noncanonical_or_wrong_answer"
        input_ids = torch.cat(
            (input_ids, torch.tensor([[token]], device=input_ids.device)), dim=1
        )
    return input_ids, answer_id, False, "answer_not_generated"


def load_v31_domain(root: Path, domain: str) -> list[TaskExample]:
    manifest = json.loads(
        (root / "artifacts/v3_1_data_manifest.json").read_text(encoding="utf-8")
    )
    relative = manifest["causal_domains"][domain]["path"]
    return task_examples_from_json(root / relative)


def select_positions(sequence_length: int, scope: str) -> tuple[int, ...]:
    if scope == "final":
        return (sequence_length - 1,)
    if scope == "all_non_padding":
        return tuple(range(sequence_length))
    raise ValueError(f"unsupported v3.1 scope: {scope}")


def match_donor(
    anchor: dict[str, Any],
    donors: list[dict[str, Any]],
) -> dict[str, Any] | None:
    matching = [
        row
        for row in donors
        if row["prompt_id"] != anchor["prompt_id"]
        and row["template_id"] == anchor["template_id"]
        and int(row["sequence_length"]) == int(anchor["sequence_length"])
    ]
    if not matching:
        return None
    matching.sort(key=lambda row: (str(row["prompt_hash"]), str(row["prompt_id"])))
    index = int(str(anchor["prompt_hash"])[:16], 16) % len(matching)
    return matching[index]


def fit_naturality_models(
    root: Path,
    records: list[dict[str, Any]],
    layers: list[int],
    *,
    scope: str,
    config: dict[str, Any],
) -> dict[int, NaturalityModel]:
    models: dict[int, NaturalityModel] = {}
    for layer in layers:
        states: list[np.ndarray] = []
        for record in records:
            payload = torch.load(root / record["activation_path"], map_location="cpu")
            sequence = payload["activations"][layer].float().numpy()
            if scope == "final":
                states.append(sequence[-1])
            else:
                states.extend(sequence)
        models[layer] = NaturalityModel(
            int(config["v3_1"]["pca_dimension"]),
            int(config["v3_1"]["nearest_neighbors"]),
            float(config["v3_1"]["naturality_quantile"]),
        ).fit(np.stack(states))
    return models


def construct_initial_sequence(
    clean: torch.Tensor,
    donor: torch.Tensor,
    *,
    positions: tuple[int, ...],
    layer: int,
    dense_map: DenseJMap,
    encoder: JStateEncoder,
    naturality: NaturalityModel,
    tolerance: float,
    strength: float,
    thresholds: V3ClampThresholds,
    require_per_position_displacement: bool,
) -> tuple[torch.Tensor, list[dict[str, Any]], bool, float]:
    candidate_sequence = clean.clone().float()
    rows: list[dict[str, Any]] = []
    clean_block = clean[list(positions)].float()
    donor_block = donor[list(positions)].float()
    aggregate_scale = float(torch.linalg.vector_norm(donor_block - clean_block).item())
    for position in positions:
        h = clean[position].float()
        difference = donor[position].float() - h
        natural_scale = float(torch.linalg.vector_norm(difference).item())
        candidate, construction = construct_dense_candidate(
            h,
            difference,
            layer=layer,
            dense_map=dense_map,
            natural_scale=natural_scale,
            displacement_fraction=float(strength),
            relative_tolerance=tolerance,
            optimized=True,
            naturality=lambda value: naturality.score(
                value.detach().cpu().float().numpy()
            ).natural,
            thresholds=thresholds,
        )
        natural = naturality.score(candidate.detach().cpu().float().numpy()).natural
        validation = validate_v3_clamp(
            h,
            candidate,
            layer=layer,
            state_definition="V3-Dense",
            encoder=encoder,
            dense_map=dense_map,
            natural_scale=natural_scale,
            natural=natural,
            thresholds=thresholds,
        )
        finite = bool(torch.isfinite(candidate).all())
        explosion = bool(
            torch.linalg.vector_norm(candidate.float())
            > 4 * torch.linalg.vector_norm(h.float()).clamp_min(1e-20)
        )
        eligibility = validate_intervention_eligibility(
            validation,
            finite=finite,
            activation_explosion=explosion,
            construction_status=str(construction.get("status", "UNKNOWN")),
            require_position_displacement=require_per_position_displacement,
        )
        candidate_sequence[position] = candidate
        rows.append(
            {
                "position": position,
                "eligibility": asdict(eligibility),
                "construction": construction,
                "mahalanobis": naturality.score(
                    candidate.detach().cpu().float().numpy()
                ).mahalanobis,
                "knn_ratio": naturality.score(
                    candidate.detach().cpu().float().numpy()
                ).knn_ratio,
            }
        )
    aggregate_displacement = float(
        torch.linalg.vector_norm(
            candidate_sequence[list(positions)] - clean_block
        ).item()
        / max(aggregate_scale, 1e-20)
    )
    passed = bool(
        rows
        and all(row["eligibility"]["passed"] for row in rows)
        and aggregate_displacement >= thresholds.formal_displacement
    )
    return candidate_sequence, rows, passed, aggregate_displacement


def restore_sequence(
    clean: torch.Tensor,
    current: torch.Tensor,
    donor: torch.Tensor,
    *,
    positions: tuple[int, ...],
    layer: int,
    dense_map: DenseJMap,
    encoder: JStateEncoder,
    naturality: NaturalityModel,
    tolerance: float,
    optimized: bool,
    thresholds: V3ClampThresholds,
) -> tuple[torch.Tensor, list[RestorationEvent], bool]:
    output = current.clone().float()
    events: list[RestorationEvent] = []
    for position in positions:
        h_clean = clean[position].float()
        h_current = current[position].float()
        candidate, construction = project_dense_candidate(
            h_clean,
            h_current,
            layer=layer,
            dense_map=dense_map,
            relative_tolerance=tolerance,
            optimized=optimized,
            naturality=lambda value: naturality.score(
                value.detach().cpu().float().numpy()
            ).natural,
            thresholds=thresholds,
        )
        natural_scale = float(
            torch.linalg.vector_norm(donor[position].float() - h_clean).item()
        )
        natural = naturality.score(candidate.detach().cpu().float().numpy()).natural
        validation = validate_v3_clamp(
            h_clean,
            candidate,
            layer=layer,
            state_definition="V3-Dense",
            encoder=encoder,
            dense_map=dense_map,
            natural_scale=natural_scale,
            natural=natural,
            thresholds=thresholds,
        )
        finite = bool(torch.isfinite(candidate).all())
        explosion = bool(
            torch.linalg.vector_norm(candidate.float())
            > 4 * torch.linalg.vector_norm(h_clean.float()).clamp_min(1e-20)
        )
        eligibility = validate_restoration_eligibility(
            validation,
            correction=candidate - h_current,
            natural_scale=natural_scale,
            finite=finite,
            activation_explosion=explosion,
            construction_status=str(construction.get("status", "UNKNOWN")),
        )
        output[position] = candidate
        events.append(
            RestorationEvent(
                layer=layer,
                position=position,
                eligibility=eligibility,
                construction_status=str(construction.get("status", "UNKNOWN")),
                construction_failure_reason=construction.get("failure_reason"),
            )
        )
    return (
        output,
        events,
        bool(events and all(event.eligibility.passed for event in events)),
    )


def replacement_transform(
    replacement: torch.Tensor,
    positions: tuple[int, ...],
) -> Callable[[torch.Tensor, int], torch.Tensor]:
    def transform(activation: torch.Tensor, layer: int) -> torch.Tensor:
        del layer
        output = activation.clone()
        output[:, list(positions), :] = replacement[list(positions)].to(
            output.device, output.dtype
        )
        return output

    return transform


def tensor_digest(value: torch.Tensor) -> str:
    return hashlib.sha256(
        value.detach().cpu().contiguous().numpy().tobytes()
    ).hexdigest()
