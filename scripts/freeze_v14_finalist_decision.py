"""Fail closed before final-test use when no V14 method passes development gates."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from jclosure.protocol_v14 import freeze_stage, verify_base


def main() -> None:
    root = Path.cwd()
    verify_base(root)
    numerical = json.loads(
        (root / "results/v14/processed/numerical_snr_summary_v14.json").read_text()
    )
    fidelity = pd.read_parquet(
        root / "results/v14/processed/closed_loop_fidelity_development_v14.parquet"
    )
    rows = (
        fidelity.groupby(["method", "horizon"])
        .agg(
            j=("direction_cosine", "mean"),
            magnitude=("magnitude_ratio", "mean"),
            output=("output_direction_cosine", "mean"),
            semantic=("semantic_delta_agreement", "mean"),
            sign=("task_decision_sign_agreement", "mean"),
        )
        .reset_index()
    )
    eligible = {}
    for method, group in rows.groupby("method"):
        eligible[method] = bool(
            set(group.horizon) == {1, 2, 4, 8}
            and (group.j >= 0.8).all()
            and group.magnitude.between(0.8, 1.2).all()
            and (group.output >= 0.8).all()
            and (group.semantic >= 0.8).all()
            and (group.sign >= 0.8).all()
        )
    if numerical["first_all_target_jvp_reliable_epsilon"] is not None:
        raise RuntimeError(
            "numerical gate changed; finalist decision needs new protocol"
        )
    if any(eligible.values()):
        raise RuntimeError(
            "an eligible development method exists; do not freeze no-finalist"
        )
    value = freeze_stage(
        root,
        "finalist_decision",
        [
            "scripts/freeze_v14_finalist_decision.py",
            "results/v14/processed/numerical_snr_summary_v14.json",
            "results/v14/processed/closed_loop_fidelity_development_v14.parquet",
            "results/v14/processed/local_causal_curvature_v14.json",
            "results/v14/processed/transport_analysis_v14.json",
            "results/v14/processed/joint_channel_ablation_v14.json",
            "artifacts/finite_causal_control_v14_splits.freeze.json",
        ],
        {
            "finalist_status": "NO_ELIGIBLE_FINALIST",
            "development_gate_pass": eligible,
            "all_target_numerical_gate_pass": False,
            "independent_final_panel_opened": False,
            "new_independent_bank_created": False,
            "h2_remains": True,
            "h3_supported": False,
            "absolute_state_replacement_authorized": False,
            "autonomous_controller_authorized": False,
            "reason": "numerical-JVP gate and all-horizon development causal gates failed",
        },
    )
    print(value["freeze_digest"])


if __name__ == "__main__":
    main()
