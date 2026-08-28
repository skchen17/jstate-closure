"""Phase 3 causal J-state closure and persistent-clamp mediation tests."""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from jclosure.clamp import ClampThresholds, one_shot_clamp, validate_clamp
from jclosure.datasets import (
    TaskExample,
    generate_arithmetic,
    generate_boolean,
    generate_graph_traversal,
    generate_state_machines,
    generate_symbolic_planning,
    generate_variable_binding,
    upstream_multihop,
)
from jclosure.decomposition import gradient_pursuit
from jclosure.experiments.common import (
    concept_vocabulary_path,
    initialize_context,
    require_phase0_gate,
    standard_parser,
)
from jclosure.jstate import ConceptVocabulary, JStateEncoder, jstate_distance
from jclosure.metrics import (
    answer_flip,
    jensen_shannon_from_logits,
    token_log_odds,
    token_probability,
)
from jclosure.model import load_model_bundle
from jclosure.provenance import append_jsonl
from jclosure.recorder import ActivationRecorder, ResidualEditor


@dataclass
class CleanRun:
    example: TaskExample
    input_ids: torch.Tensor
    logits: torch.Tensor
    answer_token: int
    activations: dict[int, torch.Tensor]


def _single_answer_id(tokenizer: Any, answer: str) -> int | None:
    for value in (" " + answer.strip(), answer):
        ids = tokenizer.encode(value, add_special_tokens=False)
        if len(ids) == 1:
            return int(ids[0])
    return None


def _flexible_examples(root: Path) -> list[TaskExample]:
    payload = json.loads(
        (root / "experiments/flexible-generalization.json").read_text(encoding="utf-8")
    )
    examples: list[TaskExample] = []
    for category in payload["categories"]:
        for function in category["funcs"]:
            for argument in category["args"]:
                examples.append(
                    TaskExample(
                        example_id=f"{category['name']}:{function['name']}:{argument}",
                        family="flexible_function",
                        template_id=f"{category['name']}:{function['name']}",
                        prompt=str(function["template"]).format(arg=argument),
                        answer=str(function["answers"][argument]),
                        intermediates=(str(argument),),
                        variables={"argument": str(argument), "function": str(function["name"])},
                        facts=((str(argument), str(function["name"]), str(function["answers"][argument])),),
                    )
                )
    return examples


def _task_pool(context, target: int) -> list[TaskExample]:
    data_root = context.root / context.config["data"]["upstream_root"]
    return [
        *generate_arithmetic(max(target * 3, 300), seed=context.seed),
        *generate_boolean(max(target * 3, 300), seed=context.seed + 1),
        *generate_variable_binding(max(target * 3, 300), seed=context.seed + 2),
        *generate_graph_traversal(max(target * 3, 300), seed=context.seed + 3),
        *generate_symbolic_planning(max(target * 3, 300), seed=context.seed + 4),
        *generate_state_machines(max(target * 3, 300), seed=context.seed + 5),
        *upstream_multihop(data_root),
        *_flexible_examples(data_root),
    ]


def _record_clean(bundle, examples: list[TaskExample], layers: list[int], limit: int | None):
    clean_runs: list[CleanRun] = []
    for example in examples:
        answer_token = _single_answer_id(bundle.tokenizer, example.answer)
        if answer_token is None:
            continue
        input_ids = bundle.lens_model.encode(example.prompt, max_length=512)
        with ActivationRecorder(bundle.layers, at=layers) as recorder:
            with torch.no_grad():
                logits = bundle.forward_logits(input_ids)[0, -1].float().cpu()
        if int(torch.argmax(logits)) != answer_token:
            continue
        activations = {
            layer: recorder.activations[layer][0, -1].float().cpu()
            for layer in layers
        }
        clean_runs.append(
            CleanRun(
                example=example,
                input_ids=input_ids,
                logits=logits,
                answer_token=answer_token,
                activations=activations,
            )
        )
        if limit is not None and len(clean_runs) >= limit:
            break
    return clean_runs


def choose_layer_pairs(band: list[int], min_future: int) -> list[dict[str, Any]]:
    if len(band) < 3:
        return []
    indices = []
    for quantile, label in ((0.25, "early"), (0.50, "middle"), (0.75, "late")):
        index = int(round(quantile * (len(band) - 1)))
        if index not in [item[0] for item in indices]:
            indices.append((index, label))
    pairs = []
    for index, label in indices:
        l1 = band[index]
        l0 = max(0, l1 - 2)
        future = [layer for layer in band if layer > l1]
        pairs.append(
            {
                "l0": l0,
                "l1": l1,
                "label": label,
                "primary": len(future) >= min_future and label in {"early", "middle"},
                "future_layers": future,
            }
        )
    return pairs


def _natural_donor(
    anchor_index: int,
    runs: list[CleanRun],
    *,
    layer: int,
    encoder: JStateEncoder,
    rank: int = 0,
) -> int:
    anchor = runs[anchor_index]
    anchor_state = encoder.encode(anchor.activations[layer], layer)
    candidates: list[tuple[float, float, int]] = []
    anchor_remainder = encoder.decompose(anchor.activations[layer], layer).remainder
    for index, candidate in enumerate(runs):
        if index == anchor_index or candidate.example.template_id == anchor.example.template_id:
            continue
        state = encoder.encode(candidate.activations[layer], layer)
        j_distance = jstate_distance(anchor_state, state)
        remainder = encoder.decompose(candidate.activations[layer], layer).remainder
        remainder_distance = float(torch.linalg.vector_norm(remainder - anchor_remainder))
        candidates.append((j_distance, -remainder_distance, index))
    if not candidates:
        return (anchor_index + 1) % len(runs)
    candidates.sort()
    near = candidates[: min(32, len(candidates))]
    near.sort(key=lambda item: (item[1], item[0], item[2]))
    return near[rank % len(near)][2]


def _median_natural_scales(
    runs: list[CleanRun], layers: list[int]
) -> dict[tuple[str, int], float]:
    scales: dict[tuple[str, int], float] = {}
    families = sorted({run.example.family for run in runs})
    for family in families:
        selected = [run for run in runs if run.example.family == family]
        for layer in layers:
            differences = [
                float(
                    torch.linalg.vector_norm(
                        (right.activations[layer] - left.activations[layer]).float()
                    )
                )
                for left, right in zip(selected, selected[1:], strict=False)
                if left.example.template_id != right.example.template_id
            ]
            if not differences:
                differences = [
                    float(
                        torch.linalg.vector_norm(
                            (right.activations[layer] - left.activations[layer]).float()
                        )
                    )
                    for left, right in zip(selected, selected[1:], strict=False)
                ]
            if differences:
                scales[(family, layer)] = float(np.median(differences))
    return scales


def _probe_direction(
    runs: list[CleanRun], layer: int, encoder: JStateEncoder
) -> tuple[torch.Tensor | None, float | None]:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import GroupKFold, cross_val_predict

    selected = [run for run in runs if run.example.family == "arithmetic"]
    if len(selected) < 20:
        return None, None
    x = np.stack([run.activations[layer].numpy() for run in selected])
    c_values = np.array([int(run.example.variables["c"]) for run in selected])
    y = (c_values > np.median(c_values)).astype(int)
    groups = np.array([run.example.template_id + str(int(run.example.variables["c"])) for run in selected])
    if len(np.unique(groups)) < 2 or len(np.unique(y)) < 2:
        return None, None
    splits = min(5, len(np.unique(groups)))
    model = LogisticRegression(max_iter=1000, random_state=0)
    predictions = cross_val_predict(model, x, y, groups=groups, cv=GroupKFold(splits))
    score = float(accuracy_score(y, predictions))
    model.fit(x, y)
    direction = torch.from_numpy(model.coef_[0]).float()
    decomposition = gradient_pursuit(
        direction, encoder.dictionary(layer), k=encoder.k
    )
    stripped = decomposition.remainder
    if float(torch.linalg.vector_norm(stripped)) <= 1e-12:
        return None, score
    return stripped / torch.linalg.vector_norm(stripped), score


def _add_final_position(
    activation: torch.Tensor, layer: int, *, delta: torch.Tensor
) -> torch.Tensor:
    del layer
    output = activation.clone()
    output[:, -1, :] += delta.to(output.device, output.dtype)
    return output


def _clamp_final_position(
    activation: torch.Tensor,
    layer: int,
    *,
    clean: torch.Tensor,
    encoder: JStateEncoder,
    capture: dict[int, Any],
) -> torch.Tensor:
    output = activation.clone()
    current = activation[0, -1]
    result = one_shot_clamp(
        clean.to(current.device),
        current,
        layer=layer,
        encoder=encoder,
        thresholds=ClampThresholds(min_remainder_fraction=0.0),
    )
    output[:, -1, :] = result.activation
    capture[layer] = result
    return output


def _scaled(vector: torch.Tensor, target_norm: float) -> torch.Tensor:
    norm = torch.linalg.vector_norm(vector.float()).clamp_min(1e-12)
    return vector.float() * (target_norm / float(norm))


def _directions_for_source(
    source: str,
    anchor_index: int,
    donor_index: int,
    runs: list[CleanRun],
    *,
    layer: int,
    encoder: JStateEncoder,
    probe: torch.Tensor | None,
) -> dict[str, torch.Tensor] | None:
    anchor = runs[anchor_index].activations[layer]
    donor = runs[donor_index].activations[layer]
    natural = donor - anchor
    natural_norm = float(torch.linalg.vector_norm(natural.float()))
    if natural_norm <= 1e-12:
        return None
    natural_decomp = encoder.decompose(natural, layer)
    if source == "targeted_probe":
        if probe is None:
            return None
        non_j = _scaled(probe, natural_norm)
        j_component = natural_decomp.reconstruction
    elif source == "natural_collision":
        anchor_rem = encoder.decompose(anchor, layer).remainder
        donor_rem = encoder.decompose(donor, layer).remainder
        non_j = donor_rem - anchor_rem
        j_component = natural_decomp.reconstruction
    else:
        non_j = natural_decomp.remainder
        j_component = natural_decomp.reconstruction
    if float(torch.linalg.vector_norm(non_j.float())) <= 1e-12:
        return None
    non_j = _scaled(non_j, natural_norm)
    if float(torch.linalg.vector_norm(j_component.float())) <= 1e-12:
        dictionary = encoder.dictionary(layer)
        j_component = dictionary[0]
    j_component = _scaled(j_component, natural_norm)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(2_026_082_8 + anchor_index + layer)
    random = torch.randn(anchor.shape, generator=generator)
    random = _scaled(random, natural_norm)
    return {
        "non_j": non_j,
        "j_positive": j_component,
        "full_patch": natural,
        "random": random,
        "natural_scale": torch.tensor(natural_norm),
    }


def _run_condition(
    bundle,
    encoder: JStateEncoder,
    run: CleanRun,
    pair: dict[str, Any],
    *,
    condition: str,
    clamp_condition: str,
    delta: torch.Tensor | None,
    natural_scale: float,
    strength: float,
    thresholds: ClampThresholds,
) -> tuple[dict[str, Any], dict[int, torch.Tensor]]:
    l0, l1 = pair["l0"], pair["l1"]
    capture: dict[int, Any] = {}
    transforms = {}
    if delta is not None:
        transforms[l0] = partial(_add_final_position, delta=delta * strength)
    if condition == "non_j":
        transforms[l1] = partial(
            _clamp_final_position,
            clean=run.activations[l1],
            encoder=encoder,
            capture=capture,
        )
        if clamp_condition == "persistent":
            for layer in pair["future_layers"]:
                transforms[layer] = partial(
                    _clamp_final_position,
                    clean=run.activations[layer],
                    encoder=encoder,
                    capture=capture,
                )
    record_layers = [l1, *pair["future_layers"]]
    with ResidualEditor(bundle.layers, transforms), ActivationRecorder(
        bundle.layers, at=record_layers
    ) as recorder:
        with torch.no_grad():
            logits = bundle.forward_logits(run.input_ids)[0, -1].float().cpu()
    observed = {
        layer: recorder.activations[layer][0, -1].float().cpu()
        for layer in record_layers
    }
    clamp_quality = None
    if condition == "non_j":
        clamp_quality = validate_clamp(
            run.activations[l1],
            observed[l1],
            layer=l1,
            encoder=encoder,
            thresholds=thresholds,
            natural_scale=natural_scale,
        )
    future_distances = {
        str(layer): jstate_distance(
            encoder.encode(run.activations[layer], layer),
            encoder.encode(observed[layer], layer),
        )
        for layer in pair["future_layers"]
    }
    metrics = {
        "js_divergence": jensen_shannon_from_logits(run.logits, logits),
        "target_probability": token_probability(logits, run.answer_token),
        "target_probability_clean": token_probability(run.logits, run.answer_token),
        "target_log_odds": token_log_odds(logits, run.answer_token),
        "target_log_odds_clean": token_log_odds(run.logits, run.answer_token),
        "answer_flip": answer_flip(run.logits, logits),
        "task_correct": int(torch.argmax(logits)) == run.answer_token,
        "checkpoint_dense_cosine": None if clamp_quality is None else clamp_quality.dense_cosine,
        "checkpoint_top10_overlap": None if clamp_quality is None else clamp_quality.top10_overlap,
        "checkpoint_rms_drift": None if clamp_quality is None else clamp_quality.activation_rms_drift,
        "remainder_distance": None if clamp_quality is None else clamp_quality.remainder_distance,
        "remainder_fraction": None if clamp_quality is None else clamp_quality.remainder_fraction,
        "next_layer_j_distance": next(iter(future_distances.values()), None),
        "mean_future_j_distance": float(np.mean(list(future_distances.values())))
        if future_distances
        else None,
        "future_j_distances": future_distances,
    }
    valid = condition != "non_j" or bool(clamp_quality and clamp_quality.passed)
    return {
        "valid": valid,
        "exclusion_reason": None
        if valid
        else ",".join(clamp_quality.failure_reasons),
        "metrics": metrics,
    }, observed


def main() -> None:
    parser = standard_parser(
        "Run causal J-state closure and persistent-clamp experiments",
        "configs/pilot.yaml",
    )
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int)
    args = parser.parse_args()
    context = initialize_context("closure", args)
    try:
        if args.dry_run:
            context.finish("DRY_RUN")
            return
        gate = require_phase0_gate(context)
        bundle = load_model_bundle(context.config)
        vocabulary = ConceptVocabulary.from_json(
            concept_vocabulary_path(context)
        )
        encoder = JStateEncoder.from_lens(
            bundle.lens,
            bundle.unembedding_weight,
            vocabulary,
            k=int(context.config["jstate"]["k"]),
        )
        band = [int(value) for value in gate["workspace_band"]]
        pairs = choose_layer_pairs(
            band, int(context.config["closure"]["future_min_layers"])
        )
        all_layers = sorted(
            set(band)
            | {pair["l0"] for pair in pairs}
            | {pair["l1"] for pair in pairs}
        )
        target = int(context.config.get("run", {}).get("valid_per_cell", 100))
        allowed_families = None
        if context.config.get("run", {}).get("confirmation_model"):
            confirmation = context.config["confirmation_model"]
            target = int(confirmation.get("pilot_valid_per_cell", target))
            allowed_families = set(confirmation.get("task_families", []))
        shard_count = int(args.shard_count or context.config.get("run", {}).get("shard_count", 1))
        examples = [
            example
            for index, example in enumerate(_task_pool(context, target))
            if index % shard_count == args.shard_index
            and (allowed_families is None or example.family in allowed_families)
        ]
        bank_limit = args.limit or max(target * 5, 500)
        runs = _record_clean(bundle, examples, all_layers, bank_limit)
        if len(runs) < 2:
            raise RuntimeError("fewer than two teacher-correct, single-token-answer runs")
        thresholds = ClampThresholds(
            dense_cosine=float(context.config["jstate"]["dense_cosine_threshold"]),
            top10_overlap=float(context.config["jstate"]["top10_overlap_threshold"]),
            rms_drift=float(context.config["jstate"]["rms_drift_threshold"]),
            min_remainder_fraction=float(context.config["jstate"]["min_remainder_fraction"]),
        )
        natural_scales = _median_natural_scales(
            runs, [pair["l1"] for pair in pairs]
        )
        cell_valid: dict[str, int] = {}
        attempts: dict[str, int] = {}
        for pair in pairs:
            probe, probe_score = _probe_direction(runs, pair["l0"], encoder)
            if probe_score is not None and probe_score < float(
                context.config["closure"]["targeted_probe_min_score"]
            ):
                probe = None
            for anchor_index, run in enumerate(runs):
                for source in context.config["closure"]["perturbation_sources"]:
                    for replicate in range(
                        int(context.config["closure"]["max_attempt_multiplier"])
                    ):
                        donor_index = (
                            _natural_donor(
                                anchor_index,
                                runs,
                                layer=pair["l0"],
                                encoder=encoder,
                                rank=replicate,
                            )
                            if source == "natural_collision"
                            else (anchor_index + 1 + replicate) % len(runs)
                        )
                        if donor_index == anchor_index:
                            continue
                        directions = _directions_for_source(
                            source,
                            anchor_index,
                            donor_index,
                            runs,
                            layer=pair["l0"],
                            encoder=encoder,
                            probe=probe,
                        )
                        if directions is None:
                            continue
                        for strength in context.config["closure"]["strengths"]:
                            for condition in context.config["closure"]["conditions"]:
                                clamp_conditions = (
                                    context.config["closure"]["clamp_conditions"]
                                    if condition == "non_j"
                                    else ["none"]
                                )
                                for clamp_condition in clamp_conditions:
                                    cell = f"{run.example.family}:{pair['label']}:{source}:{strength}:{condition}:{clamp_condition}"
                                    if cell_valid.get(cell, 0) >= target:
                                        continue
                                    attempts[cell] = attempts.get(cell, 0) + 1
                                    if attempts[cell] > target * int(
                                        context.config["closure"]["max_attempt_multiplier"]
                                    ):
                                        continue
                                    delta = None if condition == "clean" else directions[condition]
                                    checkpoint_scale = natural_scales.get(
                                        (run.example.family, pair["l1"]),
                                        float(directions["natural_scale"]),
                                    )
                                    result, _ = _run_condition(
                                        bundle,
                                        encoder,
                                        run,
                                        pair,
                                        condition=condition,
                                        clamp_condition=clamp_condition,
                                        delta=delta,
                                        natural_scale=checkpoint_scale,
                                        strength=float(strength),
                                        thresholds=thresholds,
                                    )
                                    if result["valid"]:
                                        cell_valid[cell] = cell_valid.get(cell, 0) + 1
                                    output_path = (
                                        context.raw_dir
                                        / context.run_id
                                        / "trials"
                                        / run.example.family
                                        / source
                                        / f"part-shard-{args.shard_index:03d}.jsonl"
                                    )
                                    append_jsonl(
                                        output_path,
                                        [
                                            {
                                                "schema_version": 1,
                                                "run_id": context.run_id,
                                                "prompt_id": run.example.example_id,
                                                "template_id": run.example.template_id,
                                                "task_family": run.example.family,
                                                "l0": pair["l0"],
                                                "l1": pair["l1"],
                                                "layer_pair_label": pair["label"],
                                                "primary_layer_pair": pair["primary"],
                                                "position": -1,
                                                "source": source,
                                                "condition": condition,
                                                "clamp_condition": clamp_condition,
                                                "strength": float(strength),
                                                "donor_id": runs[donor_index].example.example_id,
                                                "donor_replicate": replicate,
                                                "probe_score": probe_score,
                                                "checkpoint_natural_scale": checkpoint_scale,
                                                "valid": result["valid"],
                                                "exclusion_reason": result["exclusion_reason"],
                                                "metrics": result["metrics"],
                                                "seed": context.seed,
                                            }
                                        ],
                                    )
        summary = pd.DataFrame(
            [
                {"cell": cell, "valid": cell_valid.get(cell, 0), "attempted": count}
                for cell, count in sorted(attempts.items())
            ]
        )
        summary.to_parquet(
            context.processed_dir / f"closure_cell_counts_{context.run_id}.parquet",
            index=False,
        )
        context.finish(
            "COMPLETED",
            teacher_correct_runs=len(runs),
            layer_pairs=pairs,
            cells=len(attempts),
            cells_reaching_target=sum(value >= target for value in cell_valid.values()),
        )
    except KeyboardInterrupt:
        context.finish("FAILED", error="KeyboardInterrupt: run cancelled")
        raise
    except Exception as exc:
        context.finish("FAILED", error=f"{type(exc).__name__}: {exc}")
        raise


if __name__ == "__main__":
    main()
