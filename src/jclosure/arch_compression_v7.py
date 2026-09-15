"""Dual-space compression utilities for architecture-aligned cache deltas."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from jclosure.cache_v7 import clone_hybrid_cache


@dataclass(frozen=True)
class DualPCA:
    mean_blocks: tuple[torch.Tensor, ...]
    centered_fit_blocks: tuple[torch.Tensor, ...]
    eigenvalues: torch.Tensor
    eigenvectors: torch.Tensor
    train_indices: tuple[int, ...]
    block_scales: tuple[float, ...]

    @property
    def rank(self) -> int:
        return int(self.eigenvalues.numel())


def flatten_blocks(payload: dict[str, torch.Tensor]) -> tuple[torch.Tensor, ...]:
    return tuple(
        payload[name].float().reshape(payload[name].shape[0], -1)
        for name in ("recurrent", "conv", "kv")
    )


def fit_dual_pca(
    blocks: tuple[torch.Tensor, ...], train_indices: list[int], tolerance: float = 1e-8
) -> DualPCA:
    means = []
    centered = []
    scales = []
    gram = torch.zeros((len(train_indices), len(train_indices)), dtype=torch.float64)
    for block in blocks:
        fit = block[train_indices].double()
        mean = fit.mean(dim=0)
        residual = fit - mean
        scale = float(torch.sqrt(torch.mean(residual.square())).clamp_min(1e-12))
        normalized = residual / scale
        gram.add_(normalized @ normalized.T)
        means.append(mean.float())
        centered.append(residual.float())
        scales.append(scale)
    values, vectors = torch.linalg.eigh(gram)
    order = torch.argsort(values, descending=True)
    values = values[order]
    vectors = vectors[:, order]
    keep = values > values[0].clamp_min(1e-20) * tolerance
    return DualPCA(
        mean_blocks=tuple(means),
        centered_fit_blocks=tuple(centered),
        eigenvalues=values[keep].float(),
        eigenvectors=vectors[:, keep].float(),
        train_indices=tuple(train_indices),
        block_scales=tuple(scales),
    )


def pca_scores(
    model: DualPCA, blocks: tuple[torch.Tensor, ...], indices: list[int]
) -> torch.Tensor:
    cross = torch.zeros((len(indices), len(model.train_indices)), dtype=torch.float64)
    for block, mean, centered, scale in zip(
        blocks,
        model.mean_blocks,
        model.centered_fit_blocks,
        model.block_scales,
        strict=True,
    ):
        residual = block[indices].double() - mean.double()
        cross.add_((residual / scale) @ (centered.double() / scale).T)
    singular = torch.sqrt(model.eigenvalues.double()).clamp_min(1e-12)
    return ((cross @ model.eigenvectors.double()) / singular).float()


def component_order(train_scores: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Rank input PCs by squared train covariance with a declared target."""

    centered_target = target.float() - target.float().mean(dim=0, keepdim=True)
    covariance = train_scores.float().T @ centered_target
    energy = covariance.square().sum(dim=1) / train_scores.square().sum(
        dim=0
    ).clamp_min(1e-12)
    return torch.argsort(energy, descending=True)


def linear_reconstruction_alpha(
    model: DualPCA,
    scores: torch.Tensor,
    selected: torch.Tensor,
) -> torch.Tensor:
    retained = torch.zeros_like(scores)
    retained[:, selected] = scores[:, selected]
    singular = torch.sqrt(model.eigenvalues).clamp_min(1e-12)
    return (retained / singular) @ model.eigenvectors.T


class ScoreAutoencoder(nn.Module):
    def __init__(self, rank: int, dimension: int) -> None:
        super().__init__()
        width = max(64, 2 * dimension)
        self.encoder = nn.Sequential(
            nn.Linear(rank, width), nn.GELU(), nn.Linear(width, dimension)
        )
        self.decoder = nn.Sequential(
            nn.Linear(dimension, width), nn.GELU(), nn.Linear(width, rank)
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(values))


def nonlinear_reconstruction_alpha(
    model: DualPCA,
    train_scores: torch.Tensor,
    test_scores: torch.Tensor,
    dimension: int,
    *,
    seed: int,
    epochs: int = 400,
) -> tuple[torch.Tensor, dict[str, Any]]:
    rank = model.rank
    if dimension >= rank:
        selected = torch.arange(rank)
        return linear_reconstruction_alpha(model, test_scores, selected), {
            "effective_dimension": rank,
            "rank_limited": True,
            "train_loss": 0.0,
        }
    torch.manual_seed(seed)
    mean = train_scores.mean(dim=0, keepdim=True)
    scale = train_scores.std(dim=0, keepdim=True).clamp_min(1e-6)
    train = (train_scores - mean) / scale
    network = ScoreAutoencoder(rank, dimension)
    optimizer = torch.optim.AdamW(network.parameters(), lr=3e-3, weight_decay=1e-4)
    for _ in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        predicted = network(train)
        loss = torch.mean((predicted - train).square())
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        decoded = network((test_scores - mean) / scale) * scale + mean
    singular = torch.sqrt(model.eigenvalues).clamp_min(1e-12)
    alpha = (decoded / singular) @ model.eigenvectors.T
    return alpha, {
        "effective_dimension": dimension,
        "rank_limited": False,
        "train_loss": float(loss.detach()),
    }


def reconstruct_blocks(
    model: DualPCA, alpha: torch.Tensor, row: int
) -> tuple[torch.Tensor, ...]:
    return tuple(
        mean + alpha[row].float() @ centered
        for mean, centered in zip(
            model.mean_blocks, model.centered_fit_blocks, strict=True
        )
    )


def apply_delta_blocks(
    clean_cache: Any,
    reconstructed: tuple[torch.Tensor, ...],
    *,
    recurrent_layers: list[int],
    attention_layers: list[int],
) -> Any:
    output = clone_hybrid_cache(clean_cache)
    recurrent, conv, kv = reconstructed
    recurrent = recurrent.reshape(len(recurrent_layers), 32, 128, 128)
    conv = conv.reshape(len(recurrent_layers), 8192, 4)
    kv = kv.reshape(len(attention_layers), 2, 4, 256)
    for index, layer in enumerate(recurrent_layers):
        target = output.layers[layer]
        target.recurrent_states = (
            target.recurrent_states.float()
            + recurrent[index].to(target.recurrent_states.device)
        ).to(target.recurrent_states.dtype)
        target.conv_states = (
            target.conv_states.float() + conv[index].to(target.conv_states.device)
        ).to(target.conv_states.dtype)
    for index, layer in enumerate(attention_layers):
        target = output.layers[layer]
        target.keys[:, :, -1, :] = (
            target.keys[:, :, -1, :].float() + kv[index, 0].to(target.keys.device)
        ).to(target.keys.dtype)
        target.values[:, :, -1, :] = (
            target.values[:, :, -1, :].float() + kv[index, 1].to(target.values.device)
        ).to(target.values.dtype)
    return output
