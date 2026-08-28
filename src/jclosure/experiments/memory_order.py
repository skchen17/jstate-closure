"""Phase 5: test whether short J-history repairs instantaneous non-closure."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, replace
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from jclosure.experiments.closure import _record_clean, _task_pool
from jclosure.experiments.common import (
    concept_vocabulary_path,
    concept_vocabulary_v2_path,
    initialize_context,
    require_closure_eligible_layers,
    require_phase0_gate,
    require_phase0_v2_gate,
    standard_parser,
)
from jclosure.interventions import coordinate_swap_activation
from jclosure.jstate import ConceptVocabulary, JStateEncoder
from jclosure.metrics import token_log_odds
from jclosure.model import load_model_bundle
from jclosure.provenance import sha256_file, write_json_atomic
from jclosure.recorder import ActivationRecorder, ResidualEditor


@dataclass(frozen=True)
class TraceTensors:
    states: torch.Tensor
    remainders: torch.Tensor
    answer_labels: torch.Tensor
    answer_tokens: tuple[int, ...]
    prompt_ids: tuple[str, ...]
    template_ids: tuple[str, ...]
    families: tuple[str, ...]
    layers: tuple[int, ...]
    splits: tuple[str, ...]
    input_features: torch.Tensor
    counterfactual_states: torch.Tensor | None = None
    counterfactual_swap: tuple[int, int, int] | None = None
    counterfactual_answer_deltas: torch.Tensor | None = None
    counterfactual_answer_labels: torch.Tensor | None = None
    counterfactual_target_log_odds_deltas: torch.Tensor | None = None
    counterfactual_mask: torch.Tensor | None = None


def _structured_features(
    run: Any, dimension: int = 256, *, facts_only: bool = False
) -> torch.Tensor:
    """Signed feature hashing for procedural variables and structured facts."""

    features = torch.zeros(dimension)
    items = (
        [f"{subject}|{relation}|{object_}" for subject, relation, object_ in run.example.facts]
        if facts_only
        else [f"{key}={value}" for key, value in sorted(run.example.variables.items())]
    )
    if not items:
        if not facts_only:
            items = [f"family={run.example.family}", f"template={run.example.template_id}"]
    for item in items:
        digest = hashlib.sha256(str(item).encode()).digest()
        index = int.from_bytes(digest[:4], "big") % dimension
        sign = 1.0 if digest[4] & 1 else -1.0
        features[index] += sign
    return F.normalize(features, dim=0, eps=1e-12)


def _split_name(template_id: str, family: str, seed: int) -> str:
    digest = hashlib.sha256(f"{family}\x1f{template_id}\x1f{seed}".encode()).digest()
    value = int.from_bytes(digest[:8], "big") / 2**64
    return "train" if value < 0.70 else "validation" if value < 0.85 else "test"


def extract_traces(
    runs: list[Any], layers: list[int], encoder: JStateEncoder, *, seed: int
) -> TraceTensors:
    answer_tokens = tuple(sorted({run.answer_token for run in runs}))
    answer_index = {token: index for index, token in enumerate(answer_tokens)}
    groups = sorted({(run.example.family, run.example.template_id) for run in runs})
    groups.sort(
        key=lambda value: hashlib.sha256(
            f"{value[0]}\x1f{value[1]}\x1f{seed}".encode()
        ).digest()
    )
    train_end = max(1, int(round(0.70 * len(groups))))
    validation_end = max(train_end + 1, int(round(0.85 * len(groups))))
    validation_end = min(validation_end, max(len(groups) - 1, train_end))
    assignments = {
        group: "train"
        if index < train_end
        else "validation"
        if index < validation_end
        else "test"
        for index, group in enumerate(groups)
    }
    states = torch.stack(
        [
            torch.stack(
                [
                    encoder.encode(run.activations[layer], layer).dense_scores.float()
                    for layer in layers
                ]
            )
            for run in runs
        ]
    )
    remainders = torch.stack(
        [
            torch.stack(
                [
                    encoder.decompose(run.activations[layer], layer).remainder.float()
                    for layer in layers
                ]
            )
            for run in runs
        ]
    )
    return TraceTensors(
        states=states,
        remainders=remainders,
        answer_labels=torch.tensor([answer_index[run.answer_token] for run in runs]),
        answer_tokens=answer_tokens,
        prompt_ids=tuple(run.example.example_id for run in runs),
        template_ids=tuple(run.example.template_id for run in runs),
        families=tuple(run.example.family for run in runs),
        layers=tuple(layers),
        splits=tuple(
            assignments[(run.example.family, run.example.template_id)] for run in runs
        ),
        input_features=torch.stack(
            [
                torch.cat(
                    (
                        _structured_features(run),
                        _structured_features(run, facts_only=True),
                    )
                )
                for run in runs
            ]
        ),
    )


def _swap_final(
    activation: torch.Tensor,
    layer: int,
    *,
    source_direction: torch.Tensor,
    target_direction: torch.Tensor,
    strength: float,
) -> torch.Tensor:
    del layer
    return coordinate_swap_activation(
        activation,
        source_direction,
        target_direction,
        strength=strength,
        positions=(-1,),
    )


def collect_teacher_counterfactuals(
    traces: TraceTensors,
    runs: list[Any],
    bundle: Any,
    encoder: JStateEncoder,
    vocabulary: ConceptVocabulary,
    validated_swap: dict[str, Any] | None,
    *,
    max_traces: int,
) -> TraceTensors:
    """Collect actual teacher trajectories under the Phase-0-validated swap."""

    if not validated_swap or max_traces <= 0:
        return traces
    surfaces = {
        surface.strip().casefold(): index
        for index, surface in enumerate(vocabulary.surfaces)
    }
    source = surfaces.get(str(validated_swap["source_surface"]).strip().casefold())
    target = surfaces.get(str(validated_swap["target_surface"]).strip().casefold())
    if source is None or target is None:
        return traces
    swap_step = max(0, min(len(traces.layers) // 2, len(traces.layers) - 2))
    layer = traces.layers[swap_step]
    transform = partial(
        _swap_final,
        source_direction=encoder.dictionary(layer)[source],
        target_direction=encoder.dictionary(layer)[target],
        strength=float(validated_swap.get("strength", 1.0)),
    )
    counterfactual = traces.states.clone()
    answer_deltas = torch.zeros((len(runs), len(traces.answer_tokens)))
    answer_labels = torch.full((len(runs),), -1, dtype=torch.long)
    target_log_odds_deltas = torch.zeros(len(runs))
    mask = torch.zeros(len(runs), dtype=torch.bool)
    answer_tokens = torch.tensor(traces.answer_tokens, dtype=torch.long)
    selected_rows = sorted(
        range(len(runs)),
        key=lambda row: hashlib.sha256(traces.prompt_ids[row].encode()).digest(),
    )[:max_traces]
    for row in selected_rows:
        run = runs[row]
        with ResidualEditor(bundle.layers, {layer: transform}), ActivationRecorder(
            bundle.layers, at=traces.layers[swap_step:]
        ) as recorder:
            with torch.no_grad():
                logits = bundle.forward_logits(run.input_ids)[0, -1].float().cpu()
        for future_step in range(swap_step, len(traces.layers)):
            future_layer = traces.layers[future_step]
            activation = recorder.activations[future_layer][0, -1].float().cpu()
            counterfactual[row, future_step] = encoder.encode(
                activation, future_layer
            ).dense_scores.float()
        answer_deltas[row] = logits[answer_tokens] - run.logits[answer_tokens]
        answer_labels[row] = int(torch.argmax(logits[answer_tokens]))
        target_log_odds_deltas[row] = token_log_odds(
            logits, run.answer_token
        ) - token_log_odds(run.logits, run.answer_token)
        mask[row] = True
    return replace(
        traces,
        counterfactual_states=counterfactual,
        counterfactual_swap=(swap_step, source, target),
        counterfactual_answer_deltas=answer_deltas,
        counterfactual_answer_labels=answer_labels,
        counterfactual_target_log_odds_deltas=target_log_odds_deltas,
        counterfactual_mask=mask,
    )


class TemporalPredictor(nn.Module):
    """Parameter-matched history predictor with an optional remainder oracle."""

    def __init__(
        self,
        state_dim: int,
        history: int,
        latent_dim: int,
        n_layers: int,
        n_answers: int,
        remainder_dim: int = 0,
    ) -> None:
        super().__init__()
        self.history = history
        self.state_projection = nn.Linear(state_dim, latent_dim)
        self.remainder_projection = (
            nn.Linear(remainder_dim, latent_dim) if remainder_dim else None
        )
        mixer_inputs = history * latent_dim + (latent_dim if remainder_dim else 0)
        self.mixer = nn.Sequential(
            nn.Linear(mixer_inputs, latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, latent_dim),
            nn.GELU(),
        )
        self.layer_clock = nn.Embedding(n_layers, latent_dim)
        self.state_head = nn.Linear(latent_dim, state_dim)
        self.answer_head = nn.Linear(latent_dim, n_answers)

    def forward(
        self,
        history: torch.Tensor,
        layer_index: torch.Tensor,
        remainder: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        encoded = self.state_projection(history).flatten(1)
        values = [encoded]
        if self.remainder_projection is not None:
            if remainder is None:
                raise ValueError("remainder-aware predictor requires remainder input")
            values.append(self.remainder_projection(remainder))
        hidden = self.mixer(torch.cat(values, dim=-1))
        hidden = hidden + self.layer_clock(layer_index)
        state = F.normalize(self.state_head(hidden), dim=-1, eps=1e-12)
        return state, self.answer_head(hidden)


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def build_parameter_matched_predictor(
    *,
    state_dim: int,
    history: int,
    n_layers: int,
    n_answers: int,
    budget: int,
    remainder_dim: int = 0,
) -> TemporalPredictor:
    def create(width: int) -> TemporalPredictor:
        return TemporalPredictor(
            state_dim,
            history,
            width,
            n_layers,
            n_answers,
            remainder_dim,
        )

    low, high = 4, min(4096, max(8, budget // max(state_dim, 1)))
    while count_parameters(create(high)) < budget and high < 4096:
        high = min(4096, high * 2)
    candidates: list[tuple[int, int]] = []
    while low <= high:
        middle = (low + high) // 2
        count = count_parameters(create(middle))
        candidates.append((abs(count - budget), middle))
        if count < budget:
            low = middle + 1
        else:
            high = middle - 1
    return create(min(candidates)[1])


def _fit_remainder_pca(traces: TraceTensors, dimension: int | None) -> tuple[np.ndarray, str]:
    values = traces.remainders.numpy()
    if dimension is None or dimension >= values.shape[-1]:
        return values, "full"
    train = np.array([split == "train" for split in traces.splits])
    fitted = PCA(
        n_components=min(dimension, int(train.sum()) * values.shape[1], values.shape[-1]),
        random_state=0,
    )
    fitted.fit(values[train].reshape(-1, values.shape[-1]))
    transformed = fitted.transform(values.reshape(-1, values.shape[-1])).reshape(
        values.shape[0], values.shape[1], -1
    )
    return transformed, f"pca_{transformed.shape[-1]}"


def _windows(
    traces: TraceTensors,
    history: int,
    split: str,
    remainder: np.ndarray | None,
) -> TensorDataset:
    xs: list[torch.Tensor] = []
    clocks: list[int] = []
    ys: list[torch.Tensor] = []
    labels: list[int] = []
    rems: list[torch.Tensor] = []
    for row, row_split in enumerate(traces.splits):
        if row_split != split:
            continue
        for step in range(history - 1, traces.states.shape[1] - 1):
            xs.append(traces.states[row, step - history + 1 : step + 1])
            clocks.append(step)
            ys.append(traces.states[row, step + 1])
            labels.append(int(traces.answer_labels[row]))
            if remainder is not None:
                rems.append(torch.from_numpy(remainder[row, step]).float())
    if not xs:
        raise RuntimeError(f"no {split} windows for history {history}")
    tensors: list[torch.Tensor] = [
        torch.stack(xs),
        torch.tensor(clocks),
        torch.stack(ys),
        torch.tensor(labels),
    ]
    if remainder is not None:
        tensors.append(torch.stack(rems))
    return TensorDataset(*tensors)


def _state_loss(predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    target_distribution = F.softmax(target.float(), dim=-1)
    prediction_log_distribution = F.log_softmax(predicted.float(), dim=-1)
    return F.kl_div(
        prediction_log_distribution, target_distribution, reduction="batchmean"
    ) + (1 - F.cosine_similarity(predicted.float(), target.float()).mean())


def train_predictor(
    model: TemporalPredictor,
    traces: TraceTensors,
    *,
    history: int,
    remainder: np.ndarray | None,
    epochs: int,
    device: torch.device,
    lambda_j: float,
    lambda_answer: float,
) -> None:
    dataset = _windows(traces, history, "train", remainder)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    for _ in range(epochs):
        model.train()
        for batch in loader:
            history_values, clock, target, labels, *optional = [value.to(device) for value in batch]
            predicted, answer_logits = model(
                history_values, clock, optional[0] if optional else None
            )
            loss = lambda_j * _state_loss(predicted, target)
            loss = loss + lambda_answer * F.cross_entropy(answer_logits, labels)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()


@torch.no_grad()
def evaluate_predictor(
    model: TemporalPredictor,
    traces: TraceTensors,
    *,
    history: int,
    split: str,
    remainder: np.ndarray | None,
    device: torch.device,
    sparse_k: int = 25,
) -> dict[str, float]:
    dataset = _windows(traces, history, split, remainder)
    loader = DataLoader(dataset, batch_size=64)
    cosines: list[torch.Tensor] = []
    errors: list[torch.Tensor] = []
    f1_values: list[float] = []
    coefficient_errors: list[float] = []
    correct = 0
    count = 0
    model.eval()
    for batch in loader:
        history_values, clock, target, labels, *optional = [value.to(device) for value in batch]
        predicted, answer_logits = model(history_values, clock, optional[0] if optional else None)
        cosines.append(F.cosine_similarity(predicted, target).cpu())
        errors.append(torch.mean((predicted - target) ** 2, dim=-1).cpu())
        predicted_top = torch.topk(predicted, min(sparse_k, predicted.shape[-1]), dim=-1).indices
        target_top = torch.topk(target, min(sparse_k, target.shape[-1]), dim=-1).indices
        for p_ids, t_ids, p_scores, t_scores in zip(
            predicted_top,
            target_top,
            torch.gather(predicted, 1, predicted_top),
            torch.gather(target, 1, target_top),
            strict=True,
        ):
            p_set, t_set = set(p_ids.tolist()), set(t_ids.tolist())
            intersection = len(p_set & t_set)
            f1_values.append(intersection / max(len(p_set), 1))
            coefficient_errors.append(float(F.huber_loss(p_scores, t_scores)))
        correct += int((answer_logits.argmax(-1) == labels).sum())
        count += len(labels)
    return {
        "next_j_dense_cosine": float(torch.cat(cosines).mean()),
        "next_j_mse": float(torch.cat(errors).mean()),
        "sparse_support_f1": float(np.mean(f1_values)),
        "coefficient_huber": float(np.mean(coefficient_errors)),
        "answer_accuracy": correct / max(count, 1),
        "n_windows": count,
    }


@torch.no_grad()
def rollout_predictor(
    model: TemporalPredictor,
    traces: TraceTensors,
    *,
    history: int,
    split: str,
    remainder: np.ndarray | None,
    device: torch.device,
) -> dict[str, float]:
    row_indices = [index for index, value in enumerate(traces.splits) if value == split]
    horizon_cosines: list[list[float]] = []
    final_correct = 0
    for row in row_indices:
        generated = [value.to(device) for value in traces.states[row, :history]]
        answer_logits = None
        row_cosines: list[float] = []
        for step in range(history - 1, traces.states.shape[1] - 1):
            history_values = torch.stack(generated[-history:]).unsqueeze(0)
            rem = (
                torch.from_numpy(remainder[row, step]).float().to(device).unsqueeze(0)
                if remainder is not None
                else None
            )
            predicted, answer_logits = model(
                history_values, torch.tensor([step], device=device), rem
            )
            generated.append(predicted[0])
            row_cosines.append(
                float(
                    F.cosine_similarity(
                        predicted[0], traces.states[row, step + 1].to(device), dim=0
                    )
                )
            )
        horizon_cosines.append(row_cosines)
        if answer_logits is not None:
            final_correct += int(answer_logits.argmax(-1).item() == traces.answer_labels[row])
    flat = [value for row in horizon_cosines for value in row]
    final_values = [row[-1] for row in horizon_cosines if row]
    return {
        "rollout_mean_dense_cosine": float(np.mean(flat)) if flat else math.nan,
        "rollout_final_dense_cosine": float(np.median(final_values)) if final_values else math.nan,
        "rollout_answer_accuracy": final_correct / max(len(row_indices), 1),
        "rollout_trajectories": len(row_indices),
    }


def _save_trace_artifact(context: Any, traces: TraceTensors) -> Path:
    path = context.root / "artifacts" / "traces" / f"teacher_{context.run_id}.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(traces, path)
    write_json_atomic(
        path.with_suffix(".manifest.json"),
        {
            "schema_version": 1,
            "run_id": context.run_id,
            "path": str(path.relative_to(context.root)),
            "sha256": sha256_file(path),
            "shape": list(traces.states.shape),
            "layers": list(traces.layers),
        },
    )
    return path


def main() -> None:
    parser = standard_parser("Run J-history memory-order predictors", "configs/confirm.yaml")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--budget", type=int, default=20_000_000)
    args = parser.parse_args()
    context = initialize_context("memory-order", args)
    try:
        if args.dry_run:
            context.finish("DRY_RUN")
            return
        use_v2 = (
            context.config.get("run", {}).get("phase0_protocol")
            == "phase0_protocol_v2"
        )
        if use_v2:
            gate = require_phase0_v2_gate(context)
            layer_calibration = require_closure_eligible_layers(context)
        else:
            gate = require_phase0_gate(context)
            layer_calibration = None
        bundle = load_model_bundle(context.config)
        vocabulary = ConceptVocabulary.from_json(
            concept_vocabulary_v2_path(
                context, int(context.config["jstate"]["concept_vocab_size"])
            )
            if use_v2
            else concept_vocabulary_path(context)
        )
        encoder = JStateEncoder.from_lens(
            bundle.lens,
            bundle.unembedding_weight,
            vocabulary,
            k=int(context.config["jstate"]["k"]),
            lazy=use_v2,
            protocol_version=(
                "phase0_protocol_v2" if use_v2 else "phase0_protocol_v1"
            ),
            direction_chunk_size=int(
                context.config["jstate"].get("direction_chunk_size", 512)
            ),
        )
        layers = [
            int(value)
            for value in (
                layer_calibration["eligible_layers"]
                if layer_calibration is not None
                else gate["workspace_band"]
            )
        ]
        examples = _task_pool(context, int(context.config.get("run", {}).get("valid_per_cell", 500)))
        runs = _record_clean(bundle, examples, layers, args.limit or 3000)
        traces = extract_traces(runs, layers, encoder, seed=context.seed)
        traces = collect_teacher_counterfactuals(
            traces,
            runs,
            bundle,
            encoder,
            vocabulary,
            gate.get("validated_swap"),
            max_traces=int(context.config["controller"].get("counterfactual_traces", 0)),
        )
        trace_path = _save_trace_artifact(context, traces)
        device = torch.device(f"cuda:{context.config['model'].get('device', 0)}" if torch.cuda.is_available() else "cpu")
        controller_cfg = context.config["controller"]
        results: list[dict[str, Any]] = []
        conditions: list[tuple[int, np.ndarray | None, str]] = [
            (history, None, "j_history") for history in controller_cfg["histories"] if history < len(layers)
        ]
        for dimension in [*controller_cfg["remainder_pca_dims"], None]:
            remainder, name = _fit_remainder_pca(traces, dimension)
            conditions.append((1, remainder, f"remainder_oracle_{name}"))
        for history, remainder, condition in conditions:
            model = build_parameter_matched_predictor(
                state_dim=traces.states.shape[-1],
                history=history,
                n_layers=len(layers),
                n_answers=len(traces.answer_tokens),
                budget=args.budget,
                remainder_dim=0 if remainder is None else remainder.shape[-1],
            )
            train_predictor(
                model,
                traces,
                history=history,
                remainder=remainder,
                epochs=args.epochs,
                device=device,
                lambda_j=float(controller_cfg["lambda_j"]),
                lambda_answer=float(controller_cfg["lambda_answer"]),
            )
            metrics = evaluate_predictor(
                model,
                traces,
                history=history,
                split="test",
                remainder=remainder,
                device=device,
            )
            metrics.update(
                rollout_predictor(
                    model,
                    traces,
                    history=history,
                    split="test",
                    remainder=remainder,
                    device=device,
                )
            )
            results.append(
                {
                    "schema_version": 1,
                    "run_id": context.run_id,
                    "condition": condition,
                    "history_length": history,
                    "parameter_count": count_parameters(model),
                    "target_parameter_count": args.budget,
                    "metrics": metrics,
                    "seed": context.seed,
                }
            )
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        table = pd.json_normalize(results, sep=".")
        table.to_parquet(context.processed_dir / f"memory_order_{context.run_id}.parquet", index=False)
        write_json_atomic(
            context.raw_dir / context.run_id / "memory_order.json",
            {"schema_version": 1, "run_id": context.run_id, "records": results},
        )
        context.finish("COMPLETED", teacher_traces=len(runs), trace_artifact=str(trace_path))
    except KeyboardInterrupt:
        context.finish("FAILED", error="KeyboardInterrupt: run cancelled")
        raise
    except Exception as exc:
        context.finish("FAILED", error=f"{type(exc).__name__}: {exc}")
        raise


if __name__ == "__main__":
    main()
