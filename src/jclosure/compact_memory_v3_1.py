"""Representations, parameter-matched controllers, and leakage-safe rollout."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from torch import nn


def row_cosine(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    numerator = np.sum(left * right, axis=-1)
    denominator = np.linalg.norm(left, axis=-1) * np.linalg.norm(right, axis=-1)
    return numerator / np.maximum(denominator, 1e-12)


@dataclass
class LinearRepresentation:
    family: str
    dimension: int
    mean: np.ndarray
    encoder: np.ndarray
    decoder: np.ndarray

    def encode(self, values: np.ndarray) -> np.ndarray:
        return (np.asarray(values) - self.mean) @ self.encoder

    def decode(self, values: np.ndarray) -> np.ndarray:
        reconstructed = np.asarray(values) @ self.decoder + self.mean
        norms = np.linalg.norm(reconstructed, axis=-1, keepdims=True)
        return reconstructed / np.maximum(norms, 1e-12)

    def state_dict(self) -> dict[str, Any]:
        return {
            "family": self.family,
            "dimension": self.dimension,
            "mean": self.mean,
            "encoder": self.encoder,
            "decoder": self.decoder,
        }


def fit_representation(
    family: str,
    dimension: int,
    train_state: np.ndarray,
    train_next: np.ndarray,
    feature_importance: np.ndarray | None = None,
) -> LinearRepresentation:
    x = np.asarray(train_state, dtype=np.float64)
    y = np.asarray(train_next, dtype=np.float64)
    actual = min(int(dimension), x.shape[0] - 1, x.shape[1])
    mean = x.mean(0)
    centered = x - mean
    if family == "dense_profile_pca":
        pca = PCA(actual, random_state=0).fit(x)
        encoder = pca.components_.T
    elif family == "predictive_bottleneck":
        predictor = Ridge(alpha=1.0).fit(centered, y)
        _, _, right = np.linalg.svd(predictor.coef_, full_matrices=False)
        encoder = right[:actual].T
    elif family == "sparse_j_centered":
        if feature_importance is None:
            cross_covariance = centered.T @ (y - y.mean(0)) / max(1, x.shape[0] - 1)
            scores = np.var(centered, axis=0) * np.linalg.norm(cross_covariance, axis=1)
        else:
            importance = np.asarray(feature_importance, dtype=np.float64)
            if importance.shape != (x.shape[1],):
                raise ValueError("sparse feature importance has the wrong width")
            scores = importance
        selected = np.argsort(-scores, kind="stable")[:actual]
        encoder = np.zeros((x.shape[1], actual), dtype=np.float64)
        encoder[selected, np.arange(actual)] = 1.0
    else:
        raise ValueError(f"unknown representation family: {family}")
    latent = centered @ encoder
    decoder = Ridge(alpha=1e-5, fit_intercept=False).fit(latent, centered).coef_.T
    return LinearRepresentation(
        family=family,
        dimension=actual,
        mean=mean.astype(np.float32),
        encoder=encoder.astype(np.float32),
        decoder=decoder.astype(np.float32),
    )


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


class MarkovController(nn.Module):
    def __init__(self, state_dim: int, action_count: int, width: int) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(state_dim, width),
            nn.GELU(),
            nn.Linear(width, width),
            nn.GELU(),
        )
        self.state_head = nn.Linear(width, state_dim)
        self.action_head = nn.Linear(width, action_count)

    def forward(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.body(state)
        return self.state_head(hidden), self.action_head(hidden)


class HistoryController(nn.Module):
    def __init__(
        self, state_dim: int, history: int, action_count: int, width: int
    ) -> None:
        super().__init__()
        self.history = int(history)
        self.model = MarkovController(
            state_dim * history + history, action_count, width
        )
        self.state_dim = int(state_dim)

    def forward(
        self, states: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.model(torch.cat((states.flatten(-2), mask), dim=-1))


class GRUController(nn.Module):
    def __init__(
        self, state_dim: int, memory_dim: int, action_count: int, width: int
    ) -> None:
        super().__init__()
        self.memory_dim = int(memory_dim)
        self.cell = nn.GRUCell(state_dim, memory_dim)
        self.body = nn.Sequential(
            nn.Linear(memory_dim, width), nn.GELU(), nn.Linear(width, width), nn.GELU()
        )
        self.state_head = nn.Linear(width, state_dim)
        self.action_head = nn.Linear(width, action_count)

    def forward(
        self, state: torch.Tensor, memory: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        next_memory = self.cell(state, memory)
        hidden = self.body(next_memory)
        return self.state_head(hidden), self.action_head(hidden), next_memory


def _closest_width(factory, target: int, tolerance: float) -> nn.Module:
    low, high = 1, max(2, int(math.sqrt(target) * 4))
    best = factory(low)
    while low <= high:
        middle = (low + high) // 2
        candidate = factory(middle)
        count = parameter_count(candidate)
        if abs(count - target) < abs(parameter_count(best) - target):
            best = candidate
        if count < target:
            low = middle + 1
        else:
            high = middle - 1
    count = parameter_count(best)
    if abs(count - target) / target > tolerance:
        raise RuntimeError(f"closest controller has {count} parameters, outside budget")
    return best


def build_parameter_matched_controller(
    family: str,
    *,
    state_dim: int,
    action_count: int,
    target: int,
    tolerance: float,
    history: int = 1,
    memory_dim: int = 128,
) -> nn.Module:
    if family == "markov":
        return _closest_width(
            lambda width: MarkovController(state_dim, action_count, width),
            target,
            tolerance,
        )
    if family == "history":
        return _closest_width(
            lambda width: HistoryController(state_dim, history, action_count, width),
            target,
            tolerance,
        )
    if family == "gru":
        return _closest_width(
            lambda width: GRUController(state_dim, memory_dim, action_count, width),
            target,
            tolerance,
        )
    raise ValueError(f"unknown controller family: {family}")


def _history_input(
    predicted: list[torch.Tensor], length: int
) -> tuple[torch.Tensor, torch.Tensor]:
    if not predicted:
        raise ValueError("history cannot be empty")
    seed = predicted[0]
    values = [seed] * max(0, length - len(predicted)) + predicted[-length:]
    missing = max(0, length - len(predicted))
    mask = [0.0] * missing + [1.0] * (length - missing)
    return torch.stack(values, dim=-2), torch.tensor(mask, device=seed.device).expand(
        *seed.shape[:-1], length
    )


@torch.no_grad()
def autonomous_rollout(
    model: nn.Module,
    initial_state: torch.Tensor,
    *,
    steps: int,
    family: str,
    history: int = 1,
    memory_dim: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Roll out exclusively from Z0 and the model's own predictions."""

    current = initial_state
    predicted = [current]
    actions = []
    memory = None
    if family == "gru":
        if memory_dim is None:
            memory_dim = int(model.memory_dim)
        memory = torch.zeros(*current.shape[:-1], memory_dim, device=current.device)
    for _ in range(steps):
        if family == "markov":
            next_state, action = model(current)
        elif family == "history":
            state_window, mask = _history_input(predicted, history)
            next_state, action = model(state_window, mask)
        elif family == "gru":
            assert memory is not None
            next_state, action, memory = model(current, memory)
        else:
            raise ValueError(f"unknown rollout family: {family}")
        current = next_state
        predicted.append(current)
        actions.append(action)
    return torch.stack(predicted[1:], dim=-2), torch.stack(actions, dim=-2)


def scheduled_feedback_probability(
    epoch: int,
    maximum_epochs: int,
    *,
    warmup_fraction: float,
    maximum_feedback: float,
) -> float:
    warmup = max(1, int(round(maximum_epochs * warmup_fraction)))
    if epoch < warmup:
        return 0.0
    denominator = max(1, maximum_epochs - warmup - 1)
    return min(maximum_feedback, maximum_feedback * (epoch - warmup) / denominator)
