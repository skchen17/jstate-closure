"""Quality and evaluation helpers for compact-memory protocol v3.2."""

from __future__ import annotations

from typing import Any

import numpy as np


def audit_trace_payload(row: Any, payload: Any) -> tuple[str, ...]:
    reasons: list[str] = []
    required = ("states", "remainders", "sparse_states", "dispersion", "actions")
    if any(key not in payload for key in required):
        return ("missing_tensor_field",)
    lengths = {key: int(len(payload[key])) for key in required}
    if len(set(lengths.values())) != 1:
        reasons.append("tensor_length_mismatch")
    observed = lengths["states"]
    semantic = len(row.generated_semantic)
    if observed != semantic:
        reasons.append("semantic_tensor_length_mismatch")
    declared = int(row.length)
    if bool(row.parseable) and observed != declared:
        reasons.append("parseable_length_mismatch")
    if not bool(row.parseable) and observed > declared:
        reasons.append("unparseable_exceeds_declared_length")
    for key in required:
        if not np.isfinite(np.asarray(payload[key])).all():
            reasons.append(f"nonfinite_{key}")
    return tuple(sorted(set(reasons)))


def action_metrics(
    predicted: np.ndarray,
    teacher: np.ndarray,
    ground_truth: np.ndarray,
) -> dict[str, float]:
    predicted = np.asarray(predicted)
    teacher = np.asarray(teacher)
    ground_truth = np.asarray(ground_truth)
    if not (predicted.shape == teacher.shape == ground_truth.shape):
        raise ValueError("action arrays must share shape")
    return {
        "teacher_action_fidelity": float(np.mean(predicted == teacher)),
        "ground_truth_action_accuracy": float(np.mean(predicted == ground_truth)),
        "teacher_ground_truth_agreement": float(np.mean(teacher == ground_truth)),
        "teacher_final_fidelity": float(predicted[-1] == teacher[-1]),
        "ground_truth_final_accuracy": float(predicted[-1] == ground_truth[-1]),
    }


def time_to_divergence(cosines: np.ndarray, threshold: float) -> int:
    values = np.asarray(cosines, dtype=float)
    return next(
        (index + 1 for index, value in enumerate(values) if value < threshold),
        len(values) + 1,
    )


def representation_gate_reasons(
    metrics: dict[str, float | int], config: dict[str, Any]
) -> tuple[str, ...]:
    reasons: list[str] = []
    checks = (
        ("validation_reconstruction_cosine", "minimum_reconstruction_cosine", "reconstruction_retention"),
        ("phase0_pass10_retention", "minimum_phase0_pass10_retention", "semantic_retention"),
        ("causal_direction_retention", "minimum_causal_direction_retention", "causal_direction_retention"),
        ("causal_magnitude_retention", "minimum_causal_magnitude_retention", "causal_magnitude_retention"),
    )
    for metric, threshold, reason in checks:
        if float(metrics[metric]) < float(config[threshold]):
            reasons.append(reason)
    if int(metrics["causal_trials"]) < int(config["minimum_causal_trials"]):
        reasons.append("causal_trials")
    numeric = [float(metrics[key]) for key, _, _ in checks]
    if not np.isfinite(numeric).all():
        reasons.append("nonfinite")
    return tuple(sorted(set(reasons)))
