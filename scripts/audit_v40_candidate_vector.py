"""Exploratory V40 donor fidelity in the frozen answer-alphabet logit subspace.

The formula is an explicit post-calibration operationalization of V40's stated
secondary vector endpoint. It is not full-vocabulary or hidden-state fidelity.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from jclosure.protocol_v40_a4 import stage_freeze, verify_stage
from jclosure.provenance import sha256_file, write_json_atomic
from scripts.analyze_v40_a4 import CONDITIONS, bootstrap_mean, load_rows, primary_index

SOURCE = "scripts/audit_v40_candidate_vector.py"
ROLES = ("development", "validation", "independent_final")


def gap_closed(base: dict[str, float], donor: dict[str, float],
               candidate: dict[str, float]) -> float | None:
    if set(base) != set(donor) or set(base) != set(candidate):
        raise RuntimeError("V40 candidate alphabet drift")
    alphabet = sorted(base)
    b, d, c = (np.array([values[token] for token in alphabet], dtype=np.float64)
               for values in (base, donor, candidate))
    b, d, c = (values - np.mean(values) for values in (b, d, c))
    denominator = float(np.linalg.norm(b - d))
    if denominator <= 1e-8:
        return None
    return float(1.0 - np.linalg.norm(c - d) / denominator)


def analyze(root: Path, key: str, role: str) -> dict:
    verify_stage(root, f"analysis_functional_{key}_{role}")
    rows, raw_summary = load_rows(root, key, role, "functional")
    analysis_path = root / f"results/v40/processed/analysis_functional_{key}_{role}_v40.json"
    analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
    index = primary_index(rows)
    states = analysis["qualified_state_ids"]
    eligible, nonidentifiable = [], []
    for state_id in states:
        base = index[(state_id, 0, "BASE")]["candidate_logits"]
        donor = index[(state_id, 0, "DONOR_NATIVE")]["candidate_logits"]
        if gap_closed(base, donor, base) is None:
            nonidentifiable.append(state_id)
        else:
            eligible.append(state_id)
    cfg = json.loads((root / "artifacts/channel_function_v40_a4.freeze.json").read_text())["config"]
    seed = int(cfg["bootstrap_seed"]) + sum(map(ord, key + role + "vector"))
    iterations = int(cfg["bootstrap_iterations"])
    channels = {}
    for condition in CONDITIONS:
        values = [gap_closed(index[(state_id, 0, "BASE")]["candidate_logits"],
                             index[(state_id, 0, "DONOR_NATIVE")]["candidate_logits"],
                             index[(state_id, 0, condition)]["candidate_logits"])
                  for state_id in eligible]
        if any(value is None for value in values):
            raise RuntimeError("V40 identifiable candidate gap became undefined")
        channels[condition] = {
            "mean_gap_closed": bootstrap_mean(values, seed + len(condition), iterations),
            "median_gap_closed": float(np.median(values)) if values else None,
            "identifiable_state_denominator": len(eligible),
        }
    return {"model": key, "role": role, "secondary_only": True,
            "exploratory_operationalization": True,
            "vector_scope": "mean-centered task answer-alphabet logits only",
            "metric": "1 - ||candidate-donor||_2 / ||native_BASE-native_DONOR||_2",
            "sealed_states": analysis["sealed_states"],
            "native_qualified_states": len(states),
            "identifiable_qualified_states": len(eligible),
            "nonidentifiable_state_ids": nonidentifiable,
            "channels": channels,
            "functional_raw_sha256": raw_summary["raw_sha256"],
            "primary_analysis_sha256": sha256_file(analysis_path)}


def run(root: Path, key: str, role: str) -> dict:
    result = analyze(root, key, role)
    path = root / f"results/v40/processed/candidate_vector_{key}_{role}_v40.json"
    if path.exists():
        raise RuntimeError("V40 candidate vector appendix already exists")
    write_json_atomic(path, result)
    seal = stage_freeze(root, f"candidate_vector_{key}_{role}",
                        [SOURCE, str(path.relative_to(root)),
                         f"results/v40/processed/analysis_functional_{key}_{role}_v40.json"],
                        {"model": key, "role": role, "secondary_only": True,
                         "appendix_sha256": sha256_file(path)})
    return {"path": str(path.relative_to(root)), "freeze_digest": seal["freeze_digest"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=("Q", "F"))
    parser.add_argument("role", choices=ROLES)
    args = parser.parse_args()
    print(json.dumps(run(Path.cwd(), args.model, args.role), indent=2))
