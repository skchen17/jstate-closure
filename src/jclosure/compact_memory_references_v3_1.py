"""Remainder-aware references for the exploratory compact-memory protocol."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from torch import nn

from jclosure.compact_memory_v3_1 import parameter_count, row_cosine


@dataclass
class LinearCurrentRemainderReference:
    """Teacher-current one-step reference using train-fitted remainder PCA."""

    remainder_mean: np.ndarray
    remainder_components: np.ndarray
    predictor: Ridge

    def encode_remainder(self, remainder: np.ndarray) -> np.ndarray:
        return (np.asarray(remainder) - self.remainder_mean) @ self.remainder_components.T

    def predict(self, state: np.ndarray, remainder: np.ndarray) -> np.ndarray:
        features = np.concatenate((state, self.encode_remainder(remainder)), axis=-1)
        return np.asarray(self.predictor.predict(features), dtype=np.float32)


def fit_linear_current_remainder_reference(
    state: np.ndarray,
    remainder: np.ndarray,
    next_state: np.ndarray,
    *,
    remainder_dimension: int = 128,
) -> LinearCurrentRemainderReference:
    values = np.asarray(remainder, dtype=np.float32)
    actual = min(remainder_dimension, len(values) - 1, values.shape[1])
    pca = PCA(actual, random_state=0).fit(values)
    compressed = pca.transform(values)
    predictor = Ridge(alpha=1.0).fit(
        np.concatenate((np.asarray(state), compressed), axis=-1),
        np.asarray(next_state),
    )
    return LinearCurrentRemainderReference(
        remainder_mean=pca.mean_.astype(np.float32),
        remainder_components=pca.components_.astype(np.float32),
        predictor=predictor,
    )


class FullRemainderOneStep(nn.Module):
    """Nonlinear teacher-current reference; never used for autonomous rollout."""

    def __init__(self, state_dim: int, remainder_dim: int, width: int) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(state_dim + remainder_dim, width),
            nn.GELU(),
            nn.Linear(width, width),
            nn.GELU(),
        )
        self.head = nn.Linear(width, state_dim)

    def forward(self, state: torch.Tensor, remainder: torch.Tensor) -> torch.Tensor:
        return self.head(self.body(torch.cat((state, remainder), dim=-1)))


class AutonomousRemainderGRU(nn.Module):
    """Joint Z/R recurrent reference with self-fed autonomous transitions."""

    def __init__(
        self,
        state_dim: int,
        remainder_dim: int,
        memory_dim: int,
        action_count: int,
        width: int,
    ) -> None:
        super().__init__()
        self.state_dim = int(state_dim)
        self.remainder_dim = int(remainder_dim)
        self.memory_dim = int(memory_dim)
        self.cell = nn.GRUCell(state_dim + remainder_dim, memory_dim)
        self.body = nn.Sequential(
            nn.Linear(memory_dim, width),
            nn.GELU(),
            nn.Linear(width, width),
            nn.GELU(),
        )
        self.state_head = nn.Linear(width, state_dim)
        self.remainder_head = nn.Linear(width, remainder_dim)
        self.action_head = nn.Linear(width, action_count)

    def forward(
        self,
        state: torch.Tensor,
        remainder: torch.Tensor,
        memory: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        next_memory = self.cell(torch.cat((state, remainder), dim=-1), memory)
        hidden = self.body(next_memory)
        return (
            self.state_head(hidden),
            self.remainder_head(hidden),
            self.action_head(hidden),
            next_memory,
        )


def _closest_width(factory, target: int, tolerance: float) -> nn.Module:
    low, high = 1, max(2, int(math.sqrt(target) * 4))
    best = factory(low)
    while low <= high:
        middle = (low + high) // 2
        candidate = factory(middle)
        if abs(parameter_count(candidate) - target) < abs(
            parameter_count(best) - target
        ):
            best = candidate
        if parameter_count(candidate) < target:
            low = middle + 1
        else:
            high = middle - 1
    if abs(parameter_count(best) - target) / target > tolerance:
        raise RuntimeError("remainder reference is outside the parameter budget")
    return best


def build_full_remainder_reference(
    *,
    state_dim: int,
    remainder_dim: int,
    target: int,
    tolerance: float,
) -> FullRemainderOneStep:
    return _closest_width(
        lambda width: FullRemainderOneStep(state_dim, remainder_dim, width),
        target,
        tolerance,
    )  # type: ignore[return-value]


def build_autonomous_remainder_reference(
    *,
    state_dim: int,
    remainder_dim: int,
    memory_dim: int,
    action_count: int,
    target: int,
    tolerance: float,
) -> AutonomousRemainderGRU:
    return _closest_width(
        lambda width: AutonomousRemainderGRU(
            state_dim, remainder_dim, memory_dim, action_count, width
        ),
        target,
        tolerance,
    )  # type: ignore[return-value]


@torch.no_grad()
def autonomous_remainder_rollout(
    model: AutonomousRemainderGRU,
    initial_state: torch.Tensor,
    initial_remainder: torch.Tensor,
    *,
    steps: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Read only (Z0, R0), then feed back the model's own Z/R predictions."""

    state = initial_state
    remainder = initial_remainder
    memory = torch.zeros(
        *state.shape[:-1], model.memory_dim, device=state.device, dtype=state.dtype
    )
    states = []
    remainders = []
    actions = []
    for _ in range(steps):
        state, remainder, action, memory = model(state, remainder, memory)
        states.append(state)
        remainders.append(remainder)
        actions.append(action)
    return (
        torch.stack(states, dim=-2),
        torch.stack(remainders, dim=-2),
        torch.stack(actions, dim=-2),
    )


def decoded_one_step_metrics(
    predicted: np.ndarray, target: np.ndarray
) -> dict[str, Any]:
    cosine = row_cosine(np.asarray(predicted), np.asarray(target))
    return {
        "n": len(cosine),
        "decoded_cosine_median": float(np.median(cosine)),
        "decoded_cosine_mean": float(np.mean(cosine)),
        "decoded_distance_mean": float(np.mean(1 - cosine)),
        "finite_fraction": float(np.isfinite(predicted).all(axis=-1).mean()),
    }
