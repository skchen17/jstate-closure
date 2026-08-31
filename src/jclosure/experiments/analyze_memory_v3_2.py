"""Aggregate frozen compact-memory v3.2 controller records."""

from __future__ import annotations

from jclosure.config import load_config
from jclosure.experiments.common import repository_root
from jclosure.memory_analysis_v3_2 import analyze_controller_results


def main() -> None:
    root = repository_root()
    config = load_config(root / "configs/compact_memory_v3_2.yaml")
    analyze_controller_results(
        root,
        config=config["compact_memory_v3_2"],
        n_resamples=10_000,
        confidence=0.95,
        seed=int(config["statistics"]["bootstrap_seed"]),
    )


if __name__ == "__main__":
    main()
