"""Models and data transforms for compact peripheral-state protocol v5.

The operational remainder used here is explicitly train-fitted: hidden states are
standardized, their component predictable from a train-only PCA of measured-J is
removed, and the residual is standardized again.  It is not claimed to be the
complete mathematical complement of J-space.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from torch import nn

FAMILIES = (
    "boolean_logic",
    "modular_arithmetic",
    "short_graph_traversal",
    "simple_state_transition",
    "variable_binding",
)
FAMILY_TO_ID = {value: index for index, value in enumerate(FAMILIES)}
ACTION_COUNT = 16
U_DIM = ACTION_COUNT + len(FAMILIES) + 2


@dataclass
class TransitionArrays:
    current_j: np.ndarray
    next_j: np.ndarray
    current_h: np.ndarray
    next_h: np.ndarray
    current_action: np.ndarray
    next_action: np.ndarray
    u: np.ndarray
    next_u: np.ndarray
    history_j: np.ndarray
    history_mask: np.ndarray
    example_id: np.ndarray
    family: np.ndarray
    step_index: np.ndarray
    horizon: np.ndarray

    def __len__(self) -> int:
        return int(len(self.current_j))


@dataclass
class RemainderTransform:
    j_mean: np.ndarray
    j_components: np.ndarray
    h_mean: np.ndarray
    h_scale: np.ndarray
    j_to_h_coef: np.ndarray
    j_to_h_intercept: np.ndarray
    remainder_scale: np.ndarray

    @classmethod
    def fit(
        cls, current_j: np.ndarray, current_h: np.ndarray, *, rank: int, seed: int
    ) -> RemainderTransform:
        actual = min(int(rank), len(current_j) - 1, current_j.shape[1])
        pca = PCA(actual, svd_solver="randomized", random_state=int(seed)).fit(
            current_j
        )
        j_latent = pca.transform(current_j).astype(np.float32)
        h_mean = current_h.mean(axis=0).astype(np.float32)
        h_scale = current_h.std(axis=0).astype(np.float32)
        h_scale[h_scale < 1e-5] = 1.0
        standardized_h = (current_h - h_mean) / h_scale
        regression = Ridge(alpha=10.0, solver="lsqr").fit(j_latent, standardized_h)
        remainder = standardized_h - regression.predict(j_latent)
        remainder_scale = remainder.std(axis=0).astype(np.float32)
        remainder_scale[remainder_scale < 1e-5] = 1.0
        return cls(
            j_mean=pca.mean_.astype(np.float32),
            j_components=pca.components_.astype(np.float32),
            h_mean=h_mean,
            h_scale=h_scale,
            j_to_h_coef=np.asarray(regression.coef_, dtype=np.float32),
            j_to_h_intercept=np.asarray(regression.intercept_, dtype=np.float32),
            remainder_scale=remainder_scale,
        )

    def transform(self, measured_j: np.ndarray, hidden: np.ndarray) -> np.ndarray:
        j_latent = (measured_j - self.j_mean) @ self.j_components.T
        standardized_h = (hidden - self.h_mean) / self.h_scale
        prediction = j_latent @ self.j_to_h_coef.T + self.j_to_h_intercept
        return ((standardized_h - prediction) / self.remainder_scale).astype(np.float32)

    def save(self, path: Path) -> None:
        np.savez_compressed(
            path,
            j_mean=self.j_mean,
            j_components=self.j_components,
            h_mean=self.h_mean,
            h_scale=self.h_scale,
            j_to_h_coef=self.j_to_h_coef,
            j_to_h_intercept=self.j_to_h_intercept,
            remainder_scale=self.remainder_scale,
        )

    @classmethod
    def load(cls, path: Path) -> RemainderTransform:
        with np.load(path, allow_pickle=False) as value:
            return cls(**{key: value[key] for key in value.files})


def build_u(
    current_actions: np.ndarray,
    families: np.ndarray,
    steps: np.ndarray,
    horizons: np.ndarray,
) -> np.ndarray:
    length = len(current_actions)
    output = np.zeros((length, U_DIM), dtype=np.float32)
    action_ids = np.asarray(current_actions, dtype=np.int64)
    if np.any((action_ids < 0) | (action_ids >= ACTION_COUNT)):
        raise ValueError("current action is outside the frozen action vocabulary")
    output[np.arange(length), action_ids] = 1
    for index, family in enumerate(families):
        output[index, ACTION_COUNT + FAMILY_TO_ID[str(family)]] = 1
    output[:, -2] = steps / np.maximum(horizons, 1)
    output[:, -1] = horizons / 32.0
    return output


def make_history(states: np.ndarray, length: int) -> tuple[np.ndarray, np.ndarray]:
    values = np.empty((len(states), length, states.shape[1]), dtype=np.float32)
    mask = np.zeros((len(states), length), dtype=np.float32)
    for index in range(len(states)):
        start = max(0, index - length + 1)
        selected = states[start : index + 1]
        missing = length - len(selected)
        values[index, :missing] = states[0]
        values[index, missing:] = selected
        mask[index, missing:] = 1
    return values, mask


def row_cosine(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    numerator = np.sum(left * right, axis=-1)
    denominator = np.linalg.norm(left, axis=-1) * np.linalg.norm(right, axis=-1)
    return numerator / np.maximum(denominator, 1e-12)


class OneStepPredictor(nn.Module):
    """Shared-capacity predictor for J-only, history, full-R, and compact-C inputs."""

    def __init__(
        self,
        j_dim: int,
        u_dim: int,
        aux_dim: int,
        width: int,
        action_count: int,
        *,
        architecture: str,
        history_length: int = 1,
    ) -> None:
        super().__init__()
        self.architecture = architecture
        self.history_length = int(history_length)
        self.j_projection = nn.Linear(j_dim, width)
        self.u_projection = nn.Linear(u_dim, width)
        self.aux_projection = nn.Linear(aux_dim, width) if aux_dim else None
        self.position = nn.Parameter(torch.zeros(max(1, history_length) + 2, width))
        if architecture == "attention":
            layer = nn.TransformerEncoderLayer(
                width,
                nhead=8,
                dim_feedforward=2 * width,
                dropout=0.0,
                batch_first=True,
                norm_first=True,
            )
            self.body: nn.Module = nn.TransformerEncoder(layer, num_layers=1)
            self.gate = None
        else:
            self.body = nn.Sequential(
                nn.LayerNorm(width),
                nn.Linear(width, width),
                nn.GELU(),
                nn.Linear(width, width),
                nn.GELU(),
            )
            self.gate = (
                nn.Linear(2 * width, width)
                if architecture == "gated_mlp" and aux_dim
                else None
            )
        self.j_head = nn.Linear(width, j_dim)
        self.action_head = nn.Linear(width, action_count)

    def forward(
        self,
        measured_j: torch.Tensor,
        u: torch.Tensor,
        auxiliary: torch.Tensor | None = None,
        history_j: torch.Tensor | None = None,
        history_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        current = self.j_projection(measured_j)
        task = self.u_projection(u)
        aux = (
            self.aux_projection(auxiliary) if self.aux_projection is not None else None
        )
        if self.architecture == "attention":
            if history_j is None:
                tokens = [current[:, None]]
                padding = [
                    torch.zeros(
                        len(current), 1, device=current.device, dtype=torch.bool
                    )
                ]
            else:
                history_tokens = self.j_projection(history_j)
                tokens = [history_tokens]
                if history_mask is None:
                    raise ValueError("history attention requires a mask")
                padding = [history_mask <= 0]
            tokens.append(task[:, None])
            padding.append(
                torch.zeros(len(current), 1, device=current.device, dtype=torch.bool)
            )
            if aux is not None:
                tokens.append(aux[:, None])
                padding.append(
                    torch.zeros(
                        len(current), 1, device=current.device, dtype=torch.bool
                    )
                )
            sequence = torch.cat(tokens, dim=1)
            sequence = sequence + self.position[: sequence.shape[1]][None]
            hidden = self.body(
                sequence, src_key_padding_mask=torch.cat(padding, dim=1)
            )[:, -1]
        else:
            base = current + task
            if aux is None:
                fused = base
            elif self.gate is not None:
                gate = torch.sigmoid(self.gate(torch.cat((base, aux), dim=-1)))
                fused = gate * aux + (1 - gate) * base
            else:
                fused = base + aux
            hidden = fused + self.body(fused)
        return F.normalize(self.j_head(hidden), dim=-1), self.action_head(hidden)


class PeripheralEncoder(nn.Module):
    def __init__(
        self,
        remainder_dim: int,
        latent_dim: int,
        j_dim: int,
        u_dim: int,
        *,
        nonlinear: bool,
    ) -> None:
        super().__init__()
        self.nonlinear = bool(nonlinear)
        if nonlinear:
            width = min(1024, max(256, 2 * latent_dim))
            self.network: nn.Module = nn.Sequential(
                nn.Linear(remainder_dim + j_dim + u_dim, width),
                nn.GELU(),
                nn.Linear(width, latent_dim),
            )
        else:
            self.network = nn.Linear(remainder_dim, latent_dim)

    def forward(
        self, remainder: torch.Tensor, measured_j: torch.Tensor, u: torch.Tensor
    ) -> torch.Tensor:
        if self.nonlinear:
            return self.network(torch.cat((remainder, measured_j, u), dim=-1))
        return self.network(remainder)


class PeripheralComposite(nn.Module):
    def __init__(self, encoder: PeripheralEncoder, predictor: OneStepPredictor):
        super().__init__()
        self.encoder = encoder
        self.predictor = predictor

    def forward(
        self,
        measured_j: torch.Tensor,
        u: torch.Tensor,
        remainder: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        compact = self.encoder(remainder, measured_j, u)
        predicted_j, action = self.predictor(measured_j, u, compact)
        return predicted_j, action, compact


class RecurrentPeripheralController(nn.Module):
    """Autonomous (J, C) transition with no access to later teacher states."""

    def __init__(
        self,
        j_dim: int,
        compact_dim: int,
        u_dim: int,
        action_count: int,
        width: int,
        *,
        architecture: str,
    ) -> None:
        super().__init__()
        self.architecture = architecture
        self.compact_dim = compact_dim
        self.input_projection = nn.Linear(j_dim + compact_dim + u_dim, width)
        self.compact_to_hidden: nn.Linear | None
        if architecture == "gru":
            self.compact_to_hidden = nn.Linear(compact_dim, width)
            self.cell: nn.Module = nn.GRUCell(width, width)
        else:
            self.compact_to_hidden = None
            self.cell = nn.Sequential(
                nn.LayerNorm(width),
                nn.Linear(width, width),
                nn.GELU(),
                nn.Linear(width, width),
                nn.GELU(),
            )
        self.j_head = nn.Linear(width, j_dim)
        self.compact_head = nn.Linear(width, compact_dim)
        self.action_head = nn.Linear(width, action_count)

    def forward(
        self, measured_j: torch.Tensor, compact: torch.Tensor, u: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        projected = self.input_projection(torch.cat((measured_j, compact, u), dim=-1))
        if self.architecture == "gru":
            assert self.compact_to_hidden is not None
            hidden = self.cell(projected, self.compact_to_hidden(compact))
            next_compact = self.compact_head(hidden)
        else:
            hidden = projected + self.cell(projected)
            update = self.compact_head(hidden)
            next_compact = (
                compact + update
                if self.architecture == "residual_recurrent"
                else update
            )
        return (
            F.normalize(self.j_head(hidden), dim=-1),
            next_compact,
            self.action_head(hidden),
        )


def bottleneck_regularizer(compact: torch.Tensor) -> torch.Tensor:
    centered = compact - compact.mean(dim=0, keepdim=True)
    standard = torch.sqrt(centered.square().mean(dim=0) + 1e-6)
    variance = F.relu(0.5 - standard).mean()
    normalized = centered / standard
    covariance = normalized.T @ normalized / max(1, len(compact) - 1)
    off_diagonal = covariance - torch.diag(torch.diagonal(covariance))
    decorrelation = off_diagonal.square().mean()
    return variance + 0.01 * decorrelation + 1e-5 * compact.square().mean()


def parameter_count(model: nn.Module) -> int:
    return sum(value.numel() for value in model.parameters() if value.requires_grad)


@torch.no_grad()
def encode_compact(
    encoder: PeripheralEncoder,
    remainder: np.ndarray,
    measured_j: np.ndarray,
    u: np.ndarray,
    device: torch.device,
    *,
    batch_size: int = 512,
) -> np.ndarray:
    encoder.eval()
    values = []
    for start in range(0, len(remainder), batch_size):
        stop = start + batch_size
        values.append(
            encoder(
                torch.from_numpy(remainder[start:stop]).float().to(device),
                torch.from_numpy(measured_j[start:stop]).float().to(device),
                torch.from_numpy(u[start:stop]).float().to(device),
            )
            .cpu()
            .numpy()
        )
    return np.concatenate(values).astype(np.float32)


def pca_encode(pca: PCA, values: np.ndarray) -> np.ndarray:
    return pca.transform(values).astype(np.float32)


def pca_payload(pca: PCA) -> dict[str, Any]:
    return {
        "mean": pca.mean_.astype(np.float32),
        "components": pca.components_.astype(np.float32),
    }


def pca_from_payload(payload: dict[str, np.ndarray]) -> PCA:
    model = PCA(int(payload["components"].shape[0]))
    model.mean_ = payload["mean"]
    model.components_ = payload["components"]
    model.n_features_in_ = int(payload["components"].shape[1])
    return model
