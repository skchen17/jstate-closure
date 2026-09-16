"""Freeze the v11 zero-cycle and channel-interaction analysis amendment."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from jclosure.protocol_v11 import build_derived_freeze, verify_derived_freeze

FREEZE_PATH = Path("artifacts/causal_geometry_v11_manifold_amendment.freeze.json")
PROTOCOL = "causal_geometry_v11_manifold_metric_amendment_1"


def build_freeze(root: Path, config: dict[str, Any]) -> dict[str, Any]:
    return build_derived_freeze(
        root,
        config,
        path=FREEZE_PATH,
        purpose=(
            "correct undefined relative cycle error for clean zero and add a "
            "predeclared channel nonadditivity norm proxy"
        ),
        inputs=[
            Path("src/jclosure/protocol_v11_manifold_amendment.py"),
            Path("src/jclosure/experiments/analyze_v11_amendment.py"),
            Path("src/jclosure/reporting_v11_amendment.py"),
            Path("scripts/run_causal_geometry_v11_manifold_amendment.sh"),
            Path("results/v11/processed/causal_state_manifold_v11.parquet"),
            Path("results/v11/processed/causal_state_manifold_v11.json"),
            Path("results/v11/processed/channelwise_causal_v11.parquet"),
            Path("results/v11/processed/channelwise_causal_v11.json"),
        ],
        payload={
            "amendment_protocol": PROTOCOL,
            "declared_issue": (
                "relative cycle error is undefined for the exactly zero clean-delta "
                "reference; the original nonzero-state cycle metrics remain valid"
            ),
            "correction": (
                "report clean-zero relative cycle as null and its absolute score-space "
                "reconstruction error separately"
            ),
            "interaction_proxy": (
                "observed joint error norm divided by root-sum-square single-channel "
                "error norms; descriptive only because error-vector cross terms were "
                "not retained"
            ),
        },
    )


def verify_freeze(root: Path, config: dict[str, Any]) -> dict[str, Any]:
    return verify_derived_freeze(root, config, FREEZE_PATH)
