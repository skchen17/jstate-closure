"""Phase 6: train compact controllers and evaluate autonomous causal rollouts."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from jclosure.experiments.common import (
    initialize_context,
    require_closure_eligible_layers,
    require_phase0_gate,
    require_phase0_v2_gate,
    standard_parser,
)
from jclosure.experiments.memory_order import TraceTensors, count_parameters
from jclosure.provenance import set_seed, sha256_file, write_json_atomic


class CognitiveController(nn.Module):
    family = "base"

    def __init__(
        self,
        state_dim: int,
        width: int,
        n_layers: int,
        n_answers: int,
        feature_dim: int,
    ) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.width = width
        self.state_projection = nn.Linear(state_dim, width)
        self.feature_encoder = nn.Sequential(nn.Linear(feature_dim, width), nn.GELU())
        self.layer_clock = nn.Embedding(n_layers, width)
        self.state_head = nn.Linear(width, state_dim)
        self.answer_head = nn.Linear(width, n_answers)

    def initial_state(self, features: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.state_head(self.feature_encoder(features)), dim=-1)

    def transition(
        self,
        state: torch.Tensor,
        clock: torch.Tensor,
        context: torch.Tensor,
        memory: Any,
    ) -> tuple[torch.Tensor, torch.Tensor, Any]:
        raise NotImplementedError


class MarkovMLP(CognitiveController):
    family = "mlp"

    def __init__(self, *args: Any) -> None:
        super().__init__(*args)
        self.body = nn.Sequential(
            nn.Linear(self.width, self.width),
            nn.GELU(),
            nn.Linear(self.width, self.width),
            nn.GELU(),
        )

    def transition(self, state, clock, context, memory):
        del memory
        hidden = self.state_projection(state) + self.layer_clock(clock) + context
        hidden = hidden + self.body(hidden)
        return F.normalize(self.state_head(hidden), dim=-1), self.answer_head(hidden), None


class GRUController(CognitiveController):
    family = "gru"

    def __init__(self, *args: Any) -> None:
        super().__init__(*args)
        self.recurrence = nn.GRUCell(self.width, self.width)

    def transition(self, state, clock, context, memory):
        encoded = self.state_projection(state) + self.layer_clock(clock) + context
        hidden = self.recurrence(encoded, memory) if memory is not None else self.recurrence(encoded)
        return F.normalize(self.state_head(hidden), dim=-1), self.answer_head(hidden), hidden


class TinyTransformerController(CognitiveController):
    family = "transformer"

    def __init__(self, *args: Any) -> None:
        super().__init__(*args)
        heads = 4 if self.width % 4 == 0 else 2 if self.width % 2 == 0 else 1
        block = nn.TransformerEncoderLayer(
            self.width,
            heads,
            dim_feedforward=4 * self.width,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(block, num_layers=2)

    def transition(self, state, clock, context, memory):
        token = self.state_projection(state) + self.layer_clock(clock) + context
        sequence = token.unsqueeze(1) if memory is None else torch.cat((memory, token.unsqueeze(1)), dim=1)
        length = sequence.shape[1]
        mask = torch.triu(
            torch.full((length, length), float("-inf"), device=sequence.device), diagonal=1
        )
        hidden = self.transformer(sequence, mask=mask)[:, -1]
        return F.normalize(self.state_head(hidden), dim=-1), self.answer_head(hidden), sequence


CONTROLLERS = {
    "mlp": MarkovMLP,
    "gru": GRUController,
    "transformer": TinyTransformerController,
}


def build_budgeted_controller(
    family: str,
    *,
    state_dim: int,
    n_layers: int,
    n_answers: int,
    feature_dim: int,
    budget: int,
) -> CognitiveController:
    if family not in CONTROLLERS:
        raise ValueError(f"unknown controller family: {family}")
    cls = CONTROLLERS[family]

    def create(width: int) -> CognitiveController:
        if family == "transformer" and width % 4:
            width += 4 - width % 4
        return cls(state_dim, width, n_layers, n_answers, feature_dim)

    low, high = 4, 4096
    candidates: list[tuple[int, int, int]] = []
    while low <= high:
        middle = (low + high) // 2
        model = create(middle)
        count = count_parameters(model)
        candidates.append((abs(count - budget), count, model.width))
        del model
        if count < budget:
            low = middle + 1
        else:
            high = middle - 1
    return create(min(candidates)[2])


def _trace_path(root: Path, explicit: str | None) -> Path:
    if explicit:
        path = Path(explicit)
        return path if path.is_absolute() else root / path
    manifests = sorted((root / "artifacts" / "traces").glob("teacher_*.manifest.json"))
    if not manifests:
        raise RuntimeError("no teacher trace artifact; run scripts/run_memory_order.sh first")
    payload = json.loads(manifests[-1].read_text(encoding="utf-8"))
    path = root / payload["path"]
    if sha256_file(path) != payload["sha256"]:
        raise RuntimeError("teacher trace SHA-256 does not match its manifest")
    return path


def _trace_loader(traces: TraceTensors, split: str, batch_size: int, shuffle: bool) -> DataLoader:
    indices = [index for index, value in enumerate(traces.splits) if value == split]
    if not indices:
        raise RuntimeError(f"teacher traces contain no {split} examples")
    index = torch.tensor(indices)
    return DataLoader(
        TensorDataset(
            traces.states[index], traces.answer_labels[index], traces.input_features[index], index
        ),
        batch_size=batch_size,
        shuffle=shuffle,
    )


def controller_rollout(
    model: CognitiveController,
    states: torch.Tensor,
    features: torch.Tensor,
    *,
    standalone: bool,
    feedback_probability: float,
    intervention: tuple[int, int, int] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    context = model.feature_encoder(features)
    current = model.initial_state(features) if standalone else states[:, 0]
    predictions: list[torch.Tensor] = [current]
    answer_logits = torch.empty((len(states), 0), device=states.device)
    memory = None
    for step in range(states.shape[1] - 1):
        transition_state = current
        if intervention is not None and step == intervention[0]:
            _, source, target = intervention
            transition_state = current.clone()
            transition_state[:, [source, target]] = transition_state[:, [target, source]]
        predicted, answer_logits, memory = model.transition(
            transition_state,
            torch.full((len(states),), step, dtype=torch.long, device=states.device),
            context,
            memory,
        )
        predictions.append(predicted)
        if model.training and feedback_probability < 1.0:
            mask = torch.rand((len(states), 1), device=states.device) < feedback_probability
            current = torch.where(mask, predicted, states[:, step + 1])
        else:
            current = predicted
    return torch.stack(predictions, dim=1), answer_logits


def _trajectory_loss(predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    distributions = F.softmax(target, dim=-1)
    log_distributions = F.log_softmax(predicted, dim=-1)
    kl = F.kl_div(log_distributions, distributions, reduction="batchmean") / target.shape[1]
    cosine = 1 - F.cosine_similarity(predicted, target, dim=-1).mean()
    return kl + cosine


def train_controller(
    model: CognitiveController,
    traces: TraceTensors,
    *,
    device: torch.device,
    epochs: int,
    standalone: bool,
    lambda_j: float,
    lambda_answer: float,
    lambda_cf: float,
    warmup_fraction: float,
    max_feedback: float,
) -> dict[str, Any]:
    loader = _trace_loader(traces, "train", 16, True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    model.to(device)
    best_validation = -math.inf
    best_state: dict[str, torch.Tensor] | None = None
    patience = 0
    for epoch in range(epochs):
        progress = epoch / max(epochs - 1, 1)
        feedback = 0.0 if progress < warmup_fraction else max_feedback * (
            progress - warmup_fraction
        ) / max(1 - warmup_fraction, 1e-9)
        model.train()
        for states, labels, features, rows in loader:
            states, labels, features = states.to(device), labels.to(device), features.to(device)
            predicted, answer = controller_rollout(
                model,
                states,
                features,
                standalone=standalone,
                feedback_probability=feedback,
            )
            loss = lambda_j * _trajectory_loss(predicted[:, 1:], states[:, 1:])
            loss = loss + lambda_answer * F.cross_entropy(answer, labels)
            if (
                traces.counterfactual_states is not None
                and traces.counterfactual_mask is not None
                and traces.counterfactual_swap is not None
            ):
                selected = traces.counterfactual_mask[rows]
                cf = traces.counterfactual_states[rows[selected]].to(device)
                clean_cf = states[selected.to(device)]
                cf_features = features[selected.to(device)]
            else:
                cf = None
            if cf is not None and len(cf):
                cf_predicted, _ = controller_rollout(
                    model,
                    clean_cf,
                    cf_features,
                    standalone=False,
                    feedback_probability=1.0,
                    intervention=traces.counterfactual_swap,
                )
                loss = loss + lambda_cf * _trajectory_loss(cf_predicted[:, 1:], cf[:, 1:])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        validation = evaluate_controller(
            model, traces, split="validation", device=device, standalone=standalone
        )
        score = validation["rollout_mean_dense_cosine"] + validation["answer_accuracy"]
        if score > best_validation:
            best_validation = score
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
        if patience >= 5:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return {"epochs_completed": epoch + 1, "best_validation_score": best_validation}


@torch.no_grad()
def evaluate_controller(
    model: CognitiveController,
    traces: TraceTensors,
    *,
    split: str,
    device: torch.device,
    standalone: bool,
) -> dict[str, Any]:
    loader = _trace_loader(traces, split, 32, False)
    model.eval()
    all_cosines: list[np.ndarray] = []
    all_f1: list[float] = []
    correct = 0
    count = 0
    for states, labels, features, _ in loader:
        states, labels, features = states.to(device), labels.to(device), features.to(device)
        predicted, answer = controller_rollout(
            model, states, features, standalone=standalone, feedback_probability=1.0
        )
        cosines = F.cosine_similarity(predicted, states, dim=-1).cpu().numpy()
        all_cosines.append(cosines)
        pred_top = torch.topk(predicted, min(25, predicted.shape[-1]), dim=-1).indices
        true_top = torch.topk(states, min(25, states.shape[-1]), dim=-1).indices
        intersections = [
            len(set(a.tolist()) & set(b.tolist())) / len(a)
            for a, b in zip(pred_top.flatten(0, 1), true_top.flatten(0, 1), strict=True)
        ]
        all_f1.extend(intersections)
        correct += int((answer.argmax(-1) == labels).sum())
        count += len(labels)
    cosine = np.concatenate(all_cosines)
    return {
        "rollout_mean_dense_cosine": float(cosine.mean()),
        "rollout_final_dense_cosine": float(np.median(cosine[:, -1])),
        "rollout_horizon_cosine": [float(value) for value in cosine.mean(axis=0)],
        "sparse_support_f1": float(np.mean(all_f1)),
        "answer_accuracy": correct / max(count, 1),
        "n_trajectories": count,
    }


@torch.no_grad()
def causal_fidelity(
    model: CognitiveController,
    traces: TraceTensors,
    *,
    split: str,
    device: torch.device,
) -> dict[str, Any]:
    required = (
        traces.counterfactual_states,
        traces.counterfactual_swap,
        traces.counterfactual_answer_labels,
        traces.counterfactual_target_log_odds_deltas,
        traces.counterfactual_mask,
    )
    if any(value is None for value in required):
        return {
            "available": False,
            "reason": "validated teacher counterfactual traces are absent",
            "intervention_direction_agreement": None,
        }
    assert traces.counterfactual_mask is not None
    rows = torch.tensor(
        [
            index
            for index, value in enumerate(traces.splits)
            if value == split and bool(traces.counterfactual_mask[index])
        ]
    )
    if not len(rows):
        return {
            "available": False,
            "reason": f"no validated teacher counterfactual traces in {split} split",
            "intervention_direction_agreement": None,
        }
    assert traces.counterfactual_states is not None
    clean = traces.states[rows].to(device)
    teacher_cf = traces.counterfactual_states[rows].to(device)
    features = traces.input_features[rows].to(device)
    clean_student, clean_answer = controller_rollout(
        model, clean, features, standalone=False, feedback_probability=1.0
    )
    student_cf, cf_answer = controller_rollout(
        model,
        clean,
        features,
        standalone=False,
        feedback_probability=1.0,
        intervention=traces.counterfactual_swap,
    )
    layer, _, _ = traces.counterfactual_swap
    teacher_delta = (teacher_cf[:, layer + 1 :] - clean[:, layer + 1 :]).flatten(1)
    student_delta = (student_cf[:, layer + 1 :] - clean_student[:, layer + 1 :]).flatten(1)
    delta_cosine = F.cosine_similarity(teacher_delta, student_delta, dim=-1)
    clean_labels = traces.answer_labels[rows].to(device)
    clean_probabilities = F.softmax(clean_answer, dim=-1)
    cf_probabilities = F.softmax(cf_answer, dim=-1)
    clean_target = clean_probabilities.gather(1, clean_labels[:, None]).squeeze(1)
    cf_target = cf_probabilities.gather(1, clean_labels[:, None]).squeeze(1)
    student_log_odds_delta = torch.logit(cf_target.clamp(1e-6, 1 - 1e-6)) - torch.logit(
        clean_target.clamp(1e-6, 1 - 1e-6)
    )
    assert traces.counterfactual_target_log_odds_deltas is not None
    teacher_log_odds_delta = traces.counterfactual_target_log_odds_deltas[rows].to(device)
    teacher_direction = torch.sign(teacher_log_odds_delta)
    student_direction = torch.sign(student_log_odds_delta)
    agreement = (teacher_direction == student_direction).float().mean()
    assert traces.counterfactual_answer_labels is not None
    teacher_labels = traces.counterfactual_answer_labels[rows].to(device)
    return {
        "available": True,
        "trajectory_delta_cosine": float(delta_cosine.mean()),
        "intervention_direction_agreement": float(agreement),
        "target_answer_log_odds_direction_agreement": float(agreement),
        "decision_agreement": float((cf_answer.argmax(-1) == teacher_labels).float().mean()),
        "effect_magnitude_ratio": float(
            student_log_odds_delta.abs().mean()
            / teacher_log_odds_delta.abs().mean().clamp_min(1e-12)
        ),
    }


def _checkpoint(model: nn.Module, root: Path, run_id: str, name: str) -> dict[str, Any]:
    path = root / "artifacts" / "checkpoints" / run_id / f"{name}.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)
    return {"path": str(path.relative_to(root)), "sha256": sha256_file(path)}


def main() -> None:
    parser = standard_parser("Train compact J-state cognitive controllers", "configs/confirm.yaml")
    parser.add_argument("--trace")
    parser.add_argument("--epochs", type=int, default=30)
    args = parser.parse_args()
    context = initialize_context("distill-controller", args)
    try:
        if args.dry_run:
            context.finish("DRY_RUN")
            return
        if (
            context.config.get("run", {}).get("phase0_protocol")
            == "phase0_protocol_v2"
        ):
            require_phase0_v2_gate(context)
            require_closure_eligible_layers(context)
        else:
            require_phase0_gate(context)
        trace_path = _trace_path(context.root, args.trace)
        traces: TraceTensors = torch.load(trace_path, map_location="cpu", weights_only=False)
        config = context.config["controller"]
        device = torch.device(f"cuda:{context.config['model'].get('device', 0)}" if torch.cuda.is_available() else "cpu")
        results: list[dict[str, Any]] = []
        for seed in context.config["reproducibility"]["controller_seeds"]:
            for family in config["families"]:
                for budget in config["parameter_budgets"]:
                    for standalone in (False, True):
                        set_seed(int(seed), True)
                        model = build_budgeted_controller(
                            family,
                            state_dim=traces.states.shape[-1],
                            n_layers=traces.states.shape[1],
                            n_answers=len(traces.answer_tokens),
                            feature_dim=traces.input_features.shape[-1],
                            budget=int(budget),
                        )
                        actual = count_parameters(model)
                        training = train_controller(
                            model,
                            traces,
                            device=device,
                            epochs=args.epochs,
                            standalone=standalone,
                            lambda_j=float(config["lambda_j"]),
                            lambda_answer=float(config["lambda_answer"]),
                            lambda_cf=float(config["lambda_counterfactual"]),
                            warmup_fraction=float(config["teacher_forcing_warmup_fraction"]),
                            max_feedback=float(config["max_predicted_feedback"]),
                        )
                        metrics = evaluate_controller(
                            model, traces, split="test", device=device, standalone=standalone
                        )
                        fidelity = causal_fidelity(model, traces, split="test", device=device)
                        stable = (
                            metrics["rollout_final_dense_cosine"] >= float(config["stable_dense_cosine"])
                            and metrics["sparse_support_f1"] >= float(config["stable_sparse_f1"])
                            and metrics["answer_accuracy"]
                            >= float(config["stable_teacher_accuracy_fraction"])
                            and fidelity.get("intervention_direction_agreement") is not None
                            and fidelity["intervention_direction_agreement"]
                            >= float(config["stable_intervention_agreement"])
                        )
                        name = f"{family}-{budget}-s{seed}-{'standalone' if standalone else 'true_j0'}"
                        checkpoint = _checkpoint(model, context.root, context.run_id, name)
                        results.append(
                            {
                                "schema_version": 1,
                                "run_id": context.run_id,
                                "family": family,
                                "target_parameter_count": int(budget),
                                "parameter_count": actual,
                                "budget_relative_error": abs(actual - int(budget)) / int(budget),
                                "budget_valid": abs(actual - int(budget)) / int(budget)
                                <= float(config["budget_tolerance"]),
                                "initialization": "standalone" if standalone else "true_j0",
                                "seed": int(seed),
                                "metrics": metrics,
                                "causal_fidelity": fidelity,
                                "operationally_stable": stable,
                                "training": training,
                                "checkpoint": checkpoint,
                            }
                        )
                        del model
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
        raw = context.raw_dir / context.run_id / "controllers.json"
        write_json_atomic(raw, {"schema_version": 1, "run_id": context.run_id, "records": results})
        pd.json_normalize(results, sep=".").to_parquet(
            context.processed_dir / f"controllers_{context.run_id}.parquet", index=False
        )
        context.finish(
            "COMPLETED",
            controller_runs=len(results),
            stable_controllers=sum(bool(item["operationally_stable"]) for item in results),
            trace_sha256=sha256_file(trace_path),
        )
    except KeyboardInterrupt:
        context.finish("FAILED", error="KeyboardInterrupt: run cancelled")
        raise
    except Exception as exc:
        context.finish("FAILED", error=f"{type(exc).__name__}: {exc}")
        raise


if __name__ == "__main__":
    main()
