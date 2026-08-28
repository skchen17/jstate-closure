"""Numerically stable state, trajectory, and output metrics."""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F

from jclosure.jstate import JState, jstate_distance, jstate_similarity


def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(F.cosine_similarity(a.float().reshape(1, -1), b.float().reshape(1, -1)).item())


def rms(tensor: torch.Tensor) -> float:
    return float(torch.sqrt(torch.mean(tensor.float() ** 2)).item())


def rms_drift(reference: torch.Tensor, candidate: torch.Tensor) -> float:
    denominator = max(rms(reference), 1e-12)
    return abs(rms(candidate) - rms(reference)) / denominator


def topk_overlap(a: torch.Tensor, b: torch.Tensor, k: int = 10) -> float:
    k = min(k, a.numel(), b.numel())
    a_ids = set(torch.topk(a.float(), k).indices.tolist())
    b_ids = set(torch.topk(b.float(), k).indices.tolist())
    return len(a_ids & b_ids) / k


def jensen_shannon_from_logits(a: torch.Tensor, b: torch.Tensor) -> float:
    """Full-support Jensen-Shannon divergence in nats."""

    a_log = F.log_softmax(a.double().reshape(-1), dim=0)
    b_log = F.log_softmax(b.double().reshape(-1), dim=0)
    log_m = torch.logaddexp(a_log, b_log) - torch.log(torch.tensor(2.0, dtype=torch.double, device=a.device))
    kl_a = torch.sum(torch.exp(a_log) * (a_log - log_m))
    kl_b = torch.sum(torch.exp(b_log) * (b_log - log_m))
    return float((0.5 * (kl_a + kl_b)).item())


def token_probability(logits: torch.Tensor, token_id: int) -> float:
    return float(F.softmax(logits.float().reshape(-1), dim=0)[int(token_id)].item())


def token_log_odds(logits: torch.Tensor, token_id: int) -> float:
    probabilities = F.softmax(logits.double().reshape(-1), dim=0)
    probability = probabilities[int(token_id)].clamp(1e-15, 1 - 1e-15)
    return float(torch.log(probability / (1 - probability)).item())


def answer_flip(clean_logits: torch.Tensor, experimental_logits: torch.Tensor) -> bool:
    return int(torch.argmax(clean_logits)) != int(torch.argmax(experimental_logits))


def sparse_support_f1(predicted: Sequence[int], target: Sequence[int]) -> float:
    pred = set(int(value) for value in predicted)
    gold = set(int(value) for value in target)
    if not pred and not gold:
        return 1.0
    if not pred or not gold:
        return 0.0
    tp = len(pred & gold)
    precision = tp / len(pred)
    recall = tp / len(gold)
    return 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)


def trajectory_metrics(clean: Sequence[JState], candidate: Sequence[JState]) -> dict[str, float]:
    if len(clean) != len(candidate) or not clean:
        raise ValueError("trajectories must be non-empty and have equal lengths")
    similarities = [jstate_similarity(a, b) for a, b in zip(clean, candidate, strict=True)]
    distances = [jstate_distance(a, b) for a, b in zip(clean, candidate, strict=True)]
    return {
        "mean_dense_cosine": sum(similarities) / len(similarities),
        "min_dense_cosine": min(similarities),
        "mean_dense_cosine_distance": sum(distances) / len(distances),
        "final_dense_cosine_distance": distances[-1],
    }

