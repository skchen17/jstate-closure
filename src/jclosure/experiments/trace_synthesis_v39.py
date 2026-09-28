"""Descriptive residual-stream increments and matched-model trace synthesis."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from jclosure.protocol_v39 import stage_freeze, verify_stage
from jclosure.provenance import sha256_file, write_json_atomic


OUT = Path("results/v39/processed")
SOURCE = "src/jclosure/experiments/trace_synthesis_v39.py"


def run(root: Path) -> dict:
    for key in ("F", "Q"):
        verify_stage(root, f"internal_trace_{key}_development")
        verify_stage(root, f"trajectory_analysis_{key}_development")
    plan = json.loads((root / OUT / "execution_plan_v39.json").read_text())
    panel = json.loads((root / OUT / "panel_v39.json").read_text())
    family = {row["base_trial_id"]: row["family"] for row in panel["development"]}
    rows, dependencies, summaries = [], [], {}
    for key in ("F", "Q"):
        trace_path = root / OUT / f"internal_trace_{key}_development_v39.json"
        trace = json.loads(trace_path.read_text())
        dependencies.append(str(trace_path.relative_to(root)))
        ids = [sid for f in ("boolean_logic", "modular_arithmetic", "short_graph_traversal",
                              "simple_state_transition", "variable_binding")
               for sid in plan["internal_trace_subsets"]["development"][f]]
        for n, sid in enumerate(ids, 1):
            path = root / OUT / f"internal_tensor_bank_{key}_development_v39/state_{n:03d}.npz"
            if sha256_file(path) != trace["tensor_bank_sha256"][str(path.relative_to(root))]:
                raise RuntimeError(f"V39 internal tensor bank drift: {path}")
            dependencies.append(str(path.relative_to(root)))
            with np.load(path) as bank:
                for layer in range(trace["layers"]):
                    incoming = bank[f"I_l{layer}_BLOCK_INPUT"].astype(np.float64)
                    outgoing = bank[f"I_l{layer}_BLOCK_OUTPUT"].astype(np.float64)
                    effect = bank[f"E_l{layer}_BLOCK_OUTPUT"].astype(np.float64)
                    increment = outgoing-incoming
                    rows.append({"state_id": sid, "model": key, "family": family[sid],
                                 "layer": layer,
                                 "incoming_interaction_norm": float(np.linalg.norm(incoming)),
                                 "outgoing_interaction_norm": float(np.linalg.norm(outgoing)),
                                 "residual_increment_norm": float(np.linalg.norm(increment)),
                                 "signed_increment_toward_outgoing": float(np.dot(
                                     increment, outgoing)/max(np.dot(outgoing, outgoing), 1e-12)),
                                 "increment_over_effect": float(np.linalg.norm(increment)/
                                     max(np.linalg.norm(effect), 1e-6)),
                                 "common_residual_coordinates": True})
        endpoint = pd.read_parquet(root / OUT / f"trajectory_analysis_{key}_development_v39.parquet")
        summaries[key] = {
            "states_traced": len(ids),
            "stable_next_layer_interactions": trace["stable_next_layer_interactions"],
            "q4_isolated_vs_given_q23_cosine_median": float(
                endpoint.q4_isolated_vs_given_q23_cosine.median()),
            "q4_given_q23_magnitude_ratio_median": float(
                endpoint.q4_given_q23_magnitude_ratio.median()),
            "q4_isolated_norm_median": float(endpoint.isolated_q4_norm.median()),
            "q234_effect_norm_median": float(endpoint.cumulative_q234_norm.median())}
    frame = pd.DataFrame(rows)
    table = root / OUT / "residual_interaction_curve_v39.parquet"
    frame.to_parquet(table, index=False, compression="zstd")
    for key in summaries:
        g = frame[frame.model == key]
        summaries[key]["residual_curve"] = [
            {"layer": int(li), "median_incoming_norm": float(x.incoming_interaction_norm.median()),
             "median_outgoing_norm": float(x.outgoing_interaction_norm.median()),
             "median_increment_norm": float(x.residual_increment_norm.median()),
             "median_increment_over_effect": float(x.increment_over_effect.median())}
            for li, x in g.groupby("layer")]
    result = {"models": summaries, "rows": len(frame),
              "table_sha256": sha256_file(table),
              "direct_residual_difference_only": True,
              "descriptive_not_causal_genesis": True,
              "validation_and_final_sealed": True}
    path = root / OUT / "trace_synthesis_v39.json"
    write_json_atomic(path, result)
    seal = stage_freeze(root, "trace_synthesis",
                        [SOURCE, str(path.relative_to(root)), str(table.relative_to(root)),
                         *dependencies,
                         *[f"artifacts/interaction_genesis_v39_trajectory_analysis_{key}_development.freeze.json"
                           for key in ("F", "Q")]],
                        {"summary_sha256": sha256_file(path),
                         "descriptive_not_causal_genesis": True})
    return {"freeze_digest": seal["freeze_digest"], "rows": len(frame),
            "q4_cosine": {key: value["q4_isolated_vs_given_q23_cosine_median"]
                          for key, value in summaries.items()}}


if __name__ == "__main__":
    print(json.dumps(run(Path.cwd()), indent=2))
