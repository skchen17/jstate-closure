"""Joint predictive bottleneck objective for compact measured-J states."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


class PredictiveBottleneck(nn.Module):
    """Encode J profiles while preserving current, future, semantic, and causal data."""

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        action_count: int,
        *,
        nonlinear: bool,
    ) -> None:
        super().__init__()
        self.encoder: nn.Module
        self.decoder: nn.Module
        if nonlinear:
            width = min(1024, max(256, 2 * latent_dim))
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, width), nn.GELU(), nn.Linear(width, latent_dim)
            )
            self.decoder = nn.Sequential(
                nn.Linear(latent_dim, width), nn.GELU(), nn.Linear(width, input_dim)
            )
        else:
            self.encoder = nn.Linear(input_dim, latent_dim)
            self.decoder = nn.Linear(latent_dim, input_dim)
        transition_width = max(128, 2 * latent_dim)
        self.transition = nn.Sequential(
            nn.Linear(latent_dim, transition_width),
            nn.GELU(),
            nn.Linear(transition_width, latent_dim),
        )
        self.semantic_head = nn.Linear(latent_dim, action_count)
        self.input_dim = int(input_dim)
        self.latent_dim = int(latent_dim)
        self.nonlinear = bool(nonlinear)

    def reconstruct(self, values: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.decoder(self.encoder(values)), dim=-1)

    def predict(self, values: torch.Tensor) -> torch.Tensor:
        latent = self.encoder(values)
        return F.normalize(self.decoder(self.transition(latent)), dim=-1)


@dataclass(frozen=True)
class PredictiveLossWeights:
    reconstruction: float = 1.0
    future_prediction: float = 1.0
    semantic_retention: float = 0.5
    causal_retention: float = 0.5


def predictive_bottleneck_loss(
    model: PredictiveBottleneck,
    current: torch.Tensor,
    future: torch.Tensor,
    actions: torch.Tensor,
    *,
    causal_clean: torch.Tensor | None,
    causal_swapped: torch.Tensor | None,
    weights: PredictiveLossWeights,
) -> tuple[torch.Tensor, dict[str, float]]:
    latent = model.encoder(current)
    reconstructed = F.normalize(model.decoder(latent), dim=-1)
    predicted_latent = model.transition(latent)
    predicted = F.normalize(model.decoder(predicted_latent), dim=-1)
    reconstruction = (1 - F.cosine_similarity(reconstructed, current, dim=-1)).mean()
    prediction = (1 - F.cosine_similarity(predicted, future, dim=-1)).mean()
    latent_alignment = (
        1
        - F.cosine_similarity(predicted_latent, model.encoder(future).detach(), dim=-1)
    ).mean()
    semantic = F.cross_entropy(model.semantic_head(predicted_latent), actions)
    causal = current.new_zeros(())
    if causal_clean is not None and causal_swapped is not None and len(causal_clean):
        original_delta = causal_swapped - causal_clean
        decoded_delta = model.reconstruct(causal_swapped) - model.reconstruct(
            causal_clean
        )
        direction = (
            1 - F.cosine_similarity(decoded_delta, original_delta, dim=-1).clamp(-1, 1)
        ).mean()
        original_norm = torch.linalg.vector_norm(original_delta, dim=-1).clamp_min(1e-8)
        decoded_norm = torch.linalg.vector_norm(decoded_delta, dim=-1).clamp_min(1e-8)
        magnitude = torch.abs(torch.log(decoded_norm / original_norm)).mean()
        causal = direction + 0.25 * magnitude
    total = (
        weights.reconstruction * reconstruction
        + weights.future_prediction * (prediction + 0.25 * latent_alignment)
        + weights.semantic_retention * semantic
        + weights.causal_retention * causal
    )
    return total, {
        "reconstruction": float(reconstruction.detach().cpu()),
        "future_prediction": float(prediction.detach().cpu()),
        "latent_alignment": float(latent_alignment.detach().cpu()),
        "semantic": float(semantic.detach().cpu()),
        "causal": float(causal.detach().cpu()),
        "total": float(total.detach().cpu()),
    }


def train_predictive_bottleneck(
    train_x: np.ndarray,
    train_y: np.ndarray,
    train_actions: np.ndarray,
    validation_x: np.ndarray,
    validation_y: np.ndarray,
    *,
    latent_dim: int,
    action_count: int,
    nonlinear: bool,
    causal_pairs: list[tuple[np.ndarray, np.ndarray]],
    weights: PredictiveLossWeights,
    epochs: int,
    patience: int,
    seed: int,
    device: torch.device,
    batch_size: int = 256,
) -> tuple[PredictiveBottleneck, list[dict[str, Any]]]:
    torch.manual_seed(int(seed))
    generator = np.random.default_rng(seed)
    model = PredictiveBottleneck(
        train_x.shape[1], latent_dim, action_count, nonlinear=nonlinear
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    causal_clean = (
        torch.from_numpy(np.stack([value[0] for value in causal_pairs]))
        .float()
        .to(device)
        if causal_pairs
        else None
    )
    causal_swapped = (
        torch.from_numpy(np.stack([value[1] for value in causal_pairs]))
        .float()
        .to(device)
        if causal_pairs
        else None
    )
    best = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    stale = 0
    history: list[dict[str, Any]] = []
    indices = np.arange(len(train_x))
    validation_current = torch.from_numpy(validation_x).float().to(device)
    validation_future = torch.from_numpy(validation_y).float().to(device)
    for epoch in range(int(epochs)):
        generator.shuffle(indices)
        losses = []
        model.train()
        for start in range(0, len(indices), batch_size):
            selected = indices[start : start + batch_size]
            current = torch.from_numpy(train_x[selected]).float().to(device)
            future = torch.from_numpy(train_y[selected]).float().to(device)
            actions = torch.from_numpy(train_actions[selected]).long().to(device)
            optimizer.zero_grad(set_to_none=True)
            loss, metrics = predictive_bottleneck_loss(
                model,
                current,
                future,
                actions,
                causal_clean=causal_clean,
                causal_swapped=causal_swapped,
                weights=weights,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(metrics)
        model.eval()
        with torch.no_grad():
            score = float(
                (
                    1
                    - F.cosine_similarity(
                        model.predict(validation_current), validation_future, dim=-1
                    )
                )
                .mean()
                .cpu()
            )
        row = {
            "epoch": epoch,
            "validation_future_cosine_loss": score,
            "training_total": float(np.mean([value["total"] for value in losses])),
        }
        history.append(row)
        if score < best - 1e-5:
            best = score
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            stale = 0
        else:
            stale += 1
            if stale >= int(patience):
                break
    if best_state is None:
        raise RuntimeError("predictive bottleneck produced no finite checkpoint")
    model.load_state_dict(best_state)
    return model, history


@torch.no_grad()
def encode_numpy(
    model: PredictiveBottleneck, values: np.ndarray, device: torch.device
) -> np.ndarray:
    model.eval()
    return model.encoder(torch.from_numpy(values).float().to(device)).cpu().numpy()


@torch.no_grad()
def decode_numpy(
    model: PredictiveBottleneck, values: np.ndarray, device: torch.device
) -> np.ndarray:
    model.eval()
    output = model.decoder(torch.from_numpy(values).float().to(device))
    return F.normalize(output, dim=-1).cpu().numpy()
