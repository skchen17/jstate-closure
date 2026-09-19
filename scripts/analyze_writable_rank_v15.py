"""SNR-qualified restricted rank and channel interaction, without refitting gates."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from jclosure.experiments import operator_v15 as operator
from jclosure.experiments.jvp_v12 import _spectrum
from jclosure.experiments.numerics_v14 import _panels, _slices
from jclosure.protocol_v15 import stage_freeze, verify
from jclosure.provenance import sha256_file, write_json_atomic

OUT = Path("results/v15/processed")


def _rank(matrix: np.ndarray) -> dict[str, float | int]:
    result, _ = _spectrum(matrix, [0.9, 0.95, 0.99])
    return {key: result[key] for key in ("stable_rank", "effective_rank", "rank_90", "rank_95", "rank_99")}


def main() -> None:
    root = Path.cwd()
    base = verify(root)
    stage_path = root / "artifacts/quantization_aware_actuation_v15_rank_analysis.freeze.json"
    if not stage_path.exists():
        stage = stage_freeze(root, "rank_analysis", ["scripts/analyze_writable_rank_v15.py", "results/v15/processed/finite_operator_columns_v15.parquet", "results/v15/processed/finite_operator_spectra_v15.json"], {"role": "diagnostic_train", "SNR_rule": "both signed J effects >= frozen V14 MIN_CAUSAL_EFFECT_NORM; clean-repeat floor 0 on V14 panel", "qualified_spectrum_is_restricted_submatrix": True})
    else:
        stage = json.loads(stage_path.read_text())
    frame = pd.read_parquet(root / OUT / "finite_operator_columns_v15.parquet")
    spectra = json.loads((root / OUT / "finite_operator_spectra_v15.json").read_text())["rows"]
    scales = json.loads((root / OUT / "writable_causal_rank_v15.json").read_text())["target_scales"]
    threshold = float(base["config"]["diagnostic"]["min_causal_effect_norm"])
    frame["snr_qualified"] = (frame.j_plus_norm >= threshold) & (frame.j_minus_norm >= threshold)
    qualified_path = root / OUT / "finite_operator_snr_v15.parquet"
    frame[["base_trial_id", "direction_index", "channel", "epsilon", "j_plus_norm", "j_minus_norm", "snr_qualified", "realized_state_gain", "realized_state_cosine"]].to_parquet(qualified_path, index=False, compression="zstd")
    interaction = []
    for (base_id, index), group in frame.groupby(["base_trial_id", "direction_index"]):
        if set(group.channel) != {"joint", "recurrent", "conv", "kv"}:
            continue
        by_channel = {row.channel: row for row in group.itertuples(index=False)}
        for target in operator.TARGETS:
            joint = np.asarray(getattr(by_channel["joint"], f"{target}_response"), dtype=np.float64)
            channel_vectors = {
                name: np.asarray(getattr(by_channel[name], f"{target}_response"), dtype=np.float64)
                for name in ("recurrent", "conv", "kv")
            }
            sum_channels = sum(channel_vectors.values())
            interaction.append({
                "role": "diagnostic_train", "base_trial_id": base_id,
                "direction_index": int(index), "target": target,
                "joint_effect_norm": float(np.linalg.norm(joint)),
                "recurrent_effect_norm": float(np.linalg.norm(channel_vectors["recurrent"])),
                "conv_effect_norm": float(np.linalg.norm(channel_vectors["conv"])),
                "kv_effect_norm": float(np.linalg.norm(channel_vectors["kv"])),
                "interaction_relative_l2": float(np.linalg.norm(joint - sum_channels) / max(float(np.linalg.norm(joint)), 1e-20)),
                "joint_snr_qualified": bool(by_channel["joint"].snr_qualified),
            })
    interaction_path = root / OUT / "channel_interaction_v15.parquet"
    pd.DataFrame(interaction).to_parquet(interaction_path, index=False, compression="zstd")
    channel = {
        "role": "diagnostic_train", "freeze_digest": stage["freeze_digest"],
        "interaction_records": str(interaction_path), "interaction_sha256": sha256_file(interaction_path),
        "snr_qualified_fraction": frame.groupby("channel").snr_qualified.mean().to_dict(),
        "interaction_median_by_target": pd.DataFrame(interaction).groupby("target").interaction_relative_l2.median().to_dict(),
        "channel_spectra": [row for row in spectra if row["operator"] == "finite_writeback" and row["direction_count"] == 64 and row["target"] == "stacked_normalized"],
    }
    write_json_atomic(root / OUT / "channel_actuator_geometry_v15.json", channel)
    rank_rows = []
    panel = {str(row["base_trial_id"]): row for row in _panels(root, 1)}
    for base_id, declaration in panel.items():
        joint = frame[(frame.base_trial_id == base_id) & (frame.channel == "joint")].sort_values("direction_index")
        selected = joint[joint.snr_qualified]
        if len(selected) < 2:
            rank_rows.append({"base_trial_id": base_id, "qualified_direction_count": len(selected), "rank_identifiable": False})
            continue
        finite = np.column_stack([operator.stack({name: np.asarray(getattr(row, f"{name}_response"), dtype=np.float64) for name in operator.TARGETS}, scales) for row in selected.itertuples(index=False)])
        with np.load(root / declaration["matrix_path"], allow_pickle=False) as payload:
            if sha256_file(root / declaration["matrix_path"]) != declaration["matrix_sha256"]:
                raise RuntimeError("frozen ideal matrix hash mismatch")
            ideal_matrix = payload["matrix"].astype(np.float64)
            slices = _slices(payload)
        ideal = np.column_stack([operator.stack({name: ideal_matrix[slices[name], int(i)] for name in operator.TARGETS}, scales) for i in selected.direction_index])
        rank_rows.append({"base_trial_id": base_id, "qualified_direction_count": len(selected), "rank_identifiable": True, "finite": _rank(finite), "autograd": _rank(ideal)})
    rank_summary = {"role": "diagnostic_train", "freeze_digest": stage["freeze_digest"], "raw_spectra": str(OUT / "finite_operator_spectra_v15.json"), "snr_records": str(qualified_path), "snr_sha256": sha256_file(qualified_path), "restricted_snr_qualified_rank": rank_rows, "full_probe_spectra": [row for row in spectra if row["channel"] == "joint" and row["target"] == "stacked_normalized"], "warning": "512 probes measured at one train anchor only; 64 probes across five anchors. Qualifying columns can be selection-biased; no full model-state intrinsic rank claim."}
    write_json_atomic(root / OUT / "writable_rank_analysis_v15.json", rank_summary)
    print(json.dumps({"qualified": [(x["base_trial_id"], x["qualified_direction_count"]) for x in rank_rows], "interaction": channel["interaction_median_by_target"]}, sort_keys=True))


if __name__ == "__main__":
    main()
