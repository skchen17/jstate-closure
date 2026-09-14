"""Precision, restoration-null, and strong peripheral-reference utilities for v6."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from jclosure.peripheral_v5 import row_cosine


def normalized_dense_state(hidden: np.ndarray, centered_map: np.ndarray) -> np.ndarray:
    scores = np.asarray(hidden) @ np.asarray(centered_map).T
    denominator = np.linalg.norm(scores, axis=-1, keepdims=True)
    return scores / np.maximum(denominator, np.finfo(scores.dtype).tiny)


def precision_audit(
    before_hidden: np.ndarray,
    after_hidden: np.ndarray,
    centered_map: np.ndarray,
) -> dict[str, Any]:
    """Compare f32 and f64 projection arithmetic from the same f32 activations."""

    before32 = normalized_dense_state(
        before_hidden.astype(np.float32), centered_map.astype(np.float32)
    )
    after32 = normalized_dense_state(
        after_hidden.astype(np.float32), centered_map.astype(np.float32)
    )
    before64 = normalized_dense_state(
        before_hidden.astype(np.float64), centered_map.astype(np.float64)
    )
    after64 = normalized_dense_state(
        after_hidden.astype(np.float64), centered_map.astype(np.float64)
    )
    delta32 = after32 - before32
    delta64 = after64 - before64
    norm32 = np.linalg.norm(delta32, axis=1)
    norm64 = np.linalg.norm(delta64, axis=1)
    cosine = row_cosine(delta32, delta64)
    nonzero32 = norm32 > 0
    nonzero64 = norm64 > 0
    return {
        "delta_f32": delta32.astype(np.float32),
        "delta_f64": delta64,
        "norm_f32": norm32,
        "norm_f64": norm64,
        "cosine_f32_f64": cosine,
        "nonzero_fraction_f32": float(np.mean(nonzero32)),
        "nonzero_fraction_f64": float(np.mean(nonzero64)),
        "median_norm_f32": float(np.median(norm32)),
        "median_norm_f64": float(np.median(norm64)),
        "median_relative_norm_error": float(
            np.median(np.abs(norm32 - norm64) / np.maximum(norm64, 1e-30))
        ),
        "median_cosine_f32_f64_nonzero": float(np.median(cosine[nonzero32 & nonzero64]))
        if np.any(nonzero32 & nonzero64)
        else None,
    }


def fit_causal_directions(
    deltas: np.ndarray, count: int
) -> tuple[np.ndarray, np.ndarray]:
    """Fit deterministic right-singular directions and top varying coordinates."""

    values = np.asarray(deltas, dtype=np.float64)
    centered = values - values.mean(axis=0, keepdims=True)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    directions = vh[: min(int(count), len(vh))].astype(np.float32)
    top = np.argsort(np.mean(values * values, axis=0))[::-1]
    return directions, top.astype(np.int64)


def restoration_corrected_effects(
    frame: Any,
    *,
    effect_column: str,
    persistent_condition: str,
    null_condition: str,
) -> Any:
    """Return common-pair single, artifact, persistent, corrected, and mediation."""

    pivot = frame.pivot(
        index="base_trial_id", columns="condition", values=effect_column
    )
    pivot = pivot.dropna(subset=["single", persistent_condition, null_condition])
    pivot = pivot.copy()
    pivot["restore_artifact"] = pivot[null_condition]
    pivot["corrected_persistent"] = pivot[persistent_condition] - pivot[null_condition]
    pivot["corrected_mediation"] = 1 - pivot["corrected_persistent"] / np.maximum(
        pivot["single"], 1e-20
    )
    return pivot


@dataclass(frozen=True)
class StrongLoss:
    total: torch.Tensor
    full_profile: torch.Tensor
    action: torch.Tensor
    causal_projection: torch.Tensor
    top_dimensions: torch.Tensor


class StrongPeripheralPredictor(nn.Module):
    """J/history/full-remainder one-step predictor with a residual J head.

    Every nonlinear model receives identical task/clock U inputs. Full-peripheral
    variants additionally receive the entire operational remainder vector.
    """

    def __init__(
        self,
        j_dim: int,
        remainder_dim: int,
        u_dim: int,
        width: int,
        action_count: int,
        *,
        architecture: str,
        history_length: int = 4,
    ) -> None:
        super().__init__()
        self.architecture = architecture
        self.uses_remainder = architecture.startswith("full_remainder")
        self.uses_history = architecture == "j_history_attention"
        self.history_length = int(history_length)
        self.j_projection = nn.Linear(j_dim, width)
        self.u_projection = nn.Linear(u_dim, width)
        self.remainder_projection = (
            nn.Linear(remainder_dim, width) if self.uses_remainder else None
        )
        self.position: nn.Parameter | None = None
        if architecture.endswith("attention"):
            layer = nn.TransformerEncoderLayer(
                width,
                nhead=8,
                dim_feedforward=4 * width,
                batch_first=True,
                norm_first=True,
                dropout=0.0,
            )
            self.body: nn.Module = nn.TransformerEncoder(layer, num_layers=2)
            self.position = nn.Parameter(torch.zeros(history_length + 3, width))
            self.gate = None
        elif architecture == "full_remainder_linear":
            self.body = nn.Identity()
            self.gate = None
        else:
            self.body = nn.Sequential(
                nn.LayerNorm(width),
                nn.Linear(width, 2 * width),
                nn.GELU(),
                nn.Linear(2 * width, width),
                nn.GELU(),
                nn.Linear(width, width),
            )
            self.gate = (
                nn.Linear(2 * width, width)
                if architecture == "full_remainder_gated"
                else None
            )
        self.delta_head = nn.Linear(width, j_dim)
        self.action_head = nn.Linear(width, action_count)
        self.delta_scale = nn.Parameter(torch.tensor(-2.0))

    def forward(
        self,
        measured_j: torch.Tensor,
        u: torch.Tensor,
        remainder: torch.Tensor | None = None,
        history_j: torch.Tensor | None = None,
        history_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        current = self.j_projection(measured_j)
        task = self.u_projection(u)
        peripheral = (
            self.remainder_projection(remainder)
            if self.remainder_projection is not None and remainder is not None
            else None
        )
        if self.architecture.endswith("attention"):
            tokens = []
            padding = []
            if self.uses_history and history_j is not None:
                tokens.append(self.j_projection(history_j))
                if history_mask is None:
                    raise ValueError("history attention requires history_mask")
                padding.append(history_mask <= 0)
            else:
                tokens.append(current[:, None])
                padding.append(
                    torch.zeros(
                        (len(current), 1), dtype=torch.bool, device=current.device
                    )
                )
            tokens.append(task[:, None])
            padding.append(
                torch.zeros((len(current), 1), dtype=torch.bool, device=current.device)
            )
            if peripheral is not None:
                tokens.append(peripheral[:, None])
                padding.append(
                    torch.zeros(
                        (len(current), 1), dtype=torch.bool, device=current.device
                    )
                )
            sequence = torch.cat(tokens, dim=1)
            if self.position is None:
                raise RuntimeError("attention model has no position parameter")
            sequence = sequence + self.position[: sequence.shape[1]][None]
            hidden = self.body(
                sequence, src_key_padding_mask=torch.cat(padding, dim=1)
            )[:, -1]
        else:
            base = current + task
            if peripheral is None:
                fused = base
            elif self.gate is not None:
                gate = torch.sigmoid(self.gate(torch.cat((base, peripheral), dim=1)))
                fused = gate * peripheral + (1 - gate) * base
            else:
                fused = base + peripheral
            update = self.body(fused)
            hidden = (
                fused
                if self.architecture == "full_remainder_linear"
                else fused + update
            )
        scale = F.softplus(self.delta_scale)
        predicted = F.normalize(measured_j + scale * self.delta_head(hidden), dim=1)
        return predicted, self.action_head(hidden)


class PredictivePeripheralBottleneck(nn.Module):
    """Peripheral encoder trained against future/semantic/causal target summaries."""

    def __init__(
        self,
        remainder_dim: int,
        latent_dim: int,
        target_dim: int,
        *,
        nonlinear: bool,
    ) -> None:
        super().__init__()
        if nonlinear:
            width = min(1024, max(256, 2 * latent_dim))
            self.encoder: nn.Module = nn.Sequential(
                nn.Linear(remainder_dim, width),
                nn.GELU(),
                nn.Linear(width, latent_dim),
            )
        else:
            self.encoder = nn.Linear(remainder_dim, latent_dim)
        self.decoder = nn.Linear(latent_dim, target_dim)

    def forward(self, remainder: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        latent = self.encoder(remainder)
        return latent, self.decoder(latent)


def strong_multitask_loss(
    predicted_j: torch.Tensor,
    action_logits: torch.Tensor,
    target_j: torch.Tensor,
    target_action: torch.Tensor,
    causal_directions: torch.Tensor,
    top_dimensions: torch.Tensor,
    weights: dict[str, float],
) -> StrongLoss:
    full = (1 - F.cosine_similarity(predicted_j, target_j, dim=1)).mean()
    action = F.cross_entropy(action_logits, target_action)
    causal = F.mse_loss(
        predicted_j @ causal_directions.T, target_j @ causal_directions.T
    )
    causal = causal / (
        target_j @ causal_directions.T
    ).square().mean().detach().clamp_min(1e-4)
    top = F.mse_loss(
        predicted_j.index_select(1, top_dimensions),
        target_j.index_select(1, top_dimensions),
    )
    top = top / target_j.index_select(
        1, top_dimensions
    ).square().mean().detach().clamp_min(1e-4)
    total = (
        float(weights["full_profile"]) * full
        + float(weights["action"]) * action
        + float(weights["causal_projection"]) * causal
        + float(weights["top_causal_dimensions"]) * top
    )
    return StrongLoss(total, full, action, causal, top)
