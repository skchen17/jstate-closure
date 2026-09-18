"""Generate all standalone V13 reports and the V13 FINAL_REPORT section."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from jclosure.provenance import sha256_file, write_json_atomic


def _read(root: Path, name: str) -> dict[str, Any]:
    return json.loads((root / name).read_text(encoding="utf-8"))


def _write(root: Path, name: str, text: str) -> None:
    path = root / "reports" / name
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def _table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    if not rows:
        return "No identified rows."
    return pd.DataFrame(rows)[columns].to_markdown(index=False, floatfmt=".3f")


def generate(root: Path) -> dict[str, Any]:
    bank = _read(root, "artifacts/causal_bank_v13_capture.freeze.json")
    scaling = _read(root, "results/v13/processed/data_dimension_scaling_v13.json")
    probe = _read(root, "results/v13/processed/causal_probe_scaling_v13.json")
    linearity = _read(root, "results/v13/processed/local_linearity_radius_v13.json")
    analysis = _read(root, "results/v13/processed/geometry_analysis_v13.json")
    development = _read(
        root, "results/v13/processed/moving_tangent_oracle_development_v13.json"
    )
    confirmatory = _read(
        root, "results/v13/processed/moving_tangent_oracle_confirmatory_v13.json"
    )
    tangent = pd.read_parquet(root / analysis["records"]["tangent_atlas"])
    target = pd.read_parquet(root / analysis["records"]["target_completeness"])
    path = pd.read_parquet(root / analysis["records"]["path_dimension"])
    scaling_rows = []
    for name, curve in scaling["curves"].items():
        if name.endswith(("/d128", "/d256", "/d512", "/d1024")):
            scaling_rows.append(
                {
                    "curve": name,
                    "N": ",".join(map(str, curve["train_sizes"])),
                    "h1 direction": ",".join(
                        f"{value:.3f}" for value in curve["direction_h1"]
                    ),
                    "status": curve["status"],
                }
            )
    scaling_frame = pd.read_parquet(root / scaling["records"])
    d512_rows = scaling_frame[
        (scaling_frame["dimension"] == 512) & (scaling_frame["status"] == "IDENTIFIED")
    ][
        [
            "method",
            "train_size",
            "direction_h1",
            "magnitude_h1",
            "semantic_cosine_h1",
            "output_direction_h1",
            "semantic_sign_h1",
        ]
    ].to_dict("records")
    _write(
        root,
        "CAUSAL_BANK_EXPANSION_V13.md",
        f"""# Causal bank expansion — V13

V13 created a new teacher-correct capture bank; neither V11 nor V12 confirmation was used for method selection.

- usable train / validation / independent final: `{bank["bank_sizes"]}`
- family counts: `{bank["family_counts"]}`
- split hashes: `{bank["split_hashes"]}`
- capture freeze: `{bank["freeze_digest"]}`

The bank contains exact BF16 REC, convolution, and all observed KV cache fields, J endpoints, selected-logit endpoints, semantic log-odds endpoints, prompt/token metadata, pair IDs, and explicit exclusion counts in the split manifests.
""",
    )
    _write(
        root,
        "DATA_DIMENSION_SCALING_V13.md",
        "# Data × dimension scaling — V13\n\n"
        + _table(scaling_rows, ["curve", "N", "h1 direction", "status"])
        + "\n\n## Fixed 512D h1 metrics\n\n"
        + _table(
            d512_rows,
            [
                "method",
                "train_size",
                "direction_h1",
                "magnitude_h1",
                "semantic_cosine_h1",
                "output_direction_h1",
                "semantic_sign_h1",
            ],
        )
        + "\n\nThe practical saturation rule was frozen at two consecutive doublings with absolute improvement below "
        + f"`{scaling['saturation_epsilon']}`. `NOT_IDENTIFIED_RANK_LIMIT` rows were retained in the parquet record; no rank was silently clamped. Local method: `{scaling['local_basis_definition']}`. These curves are frozen held-out endpoint-fidelity estimates; finite writeback is adjudicated separately by the causal oracle.\n",
    )
    probe_rows = [
        {"m": int(size), **value}
        for size, value in sorted(
            probe["probe_scaling"].items(), key=lambda item: int(item[0])
        )
    ]
    _write(
        root,
        "CAUSAL_PROBE_SCALING_V13.md",
        "# Exact causal-probe scaling — V13\n\n"
        + _table(
            probe_rows,
            [
                "m",
                "median_r90",
                "median_r95",
                "median_r99",
                "mean_stable_rank",
                "mean_effective_rank",
            ],
        )
        + "\n\nAll matrices use exact autograd JVP with Flash/memory-efficient SDP disabled. Ranks remain restricted to the frozen mixed empirical raw-state operator and are not full-state intrinsic dimensions.\n",
    )
    robustness_rows = [
        {"probe family": name, **value}
        for name, value in probe["probe_family_robustness"].items()
    ]
    _write(
        root,
        "CAUSAL_RANK_ROBUSTNESS_V13.md",
        "# Causal-rank robustness — V13\n\n"
        + _table(robustness_rows, ["probe family", "median_r95", "mean_r95"])
        + f"\n\nMean per-column sensitivities: `{probe['probe_family_mean_column_sensitivity']}`. Low-variance/high-causal directions confirmed under the frozen rule: `{analysis['low_variance_high_causal_directions']}`. A single intrinsic dimension is not reported when construction-specific estimates disagree materially.\n",
    )
    tangent_rows = (
        tangent[tangent["rank"] == 16]
        .groupby("relation")
        .agg(
            mean_angle=("mean_principal_angle_degrees", "mean"),
            max_angle=("maximum_principal_angle_degrees", "mean"),
            grassmann=("grassmann_distance", "mean"),
        )
        .reset_index()
        .to_dict("records")
    )
    _write(
        root,
        "CAUSAL_TANGENT_ATLAS_V13.md",
        "# Local causal tangent atlas — V13\n\n"
        + _table(tangent_rows, ["relation", "mean_angle", "max_angle", "grassmann"])
        + f"\n\nRank-16 same-prompt successive-position mean angle: `{analysis['same_prompt_rank16_mean_angle']:.2f}°`; across-family: `{analysis['across_family_rank16_mean_angle']:.2f}°`. Spearman associations with tangent angle: `{analysis['tangent_angle_spearman_predictors']}`.\n",
    )
    confirm_rows = analysis["independent_confirmatory_summary"]
    _write(
        root,
        "MOVING_TANGENT_CAUSAL_ORACLE_V13.md",
        "# Static and moving-tangent causal oracle — V13\n\n"
        + _table(
            confirm_rows,
            [
                "method",
                "horizon",
                "direction",
                "magnitude",
                "semantic_continuous",
                "semantic_legacy",
                "output",
                "sign",
                "gate_pass",
            ],
        )
        + f"\n\nFrozen moving method / alpha: `{analysis['selected_moving_tangent_method']}` / `{analysis['selected_moving_tangent_alpha']}`. Development panel: `{development['panel_count']}`; independent final panel: `{confirmatory['panel_count']}`. Moving tangent improves static: `{analysis['moving_tangent_improves_static']}`. All h1/h2/h4/h8 gates pass: `{analysis['moving_tangent_passes_all_horizons']}`.\n",
    )
    _write(
        root,
        "LOCAL_LINEARITY_RADIUS_V13.md",
        "# Finite perturbation linearity radius — V13\n\n"
        + f"Practical local linear radius: `{linearity['local_linear_radius']}`. Criterion: {linearity['criterion']}.\n\n"
        + _table(
            linearity["pooled"],
            [
                "scale",
                "horizon",
                "j_direction",
                "output_direction",
                "j_relative_error",
                "output_relative_error",
                "tangent_drift",
            ],
        )
        + "\n",
    )
    _write(
        root,
        "CAUSAL_PATH_DIMENSION_V13.md",
        f"""# Instantaneous rank versus writable path dimension — V13

- median instantaneous r95: `{analysis["median_instantaneous_r95"]}`
- median cumulative path r95: `{analysis["median_cumulative_path_r95"]}`
- high path dimension under the frozen rule: `{analysis["high_path_dimension"]}`

The cumulative measure is the r95 of the span of the local tangent bases visited across the observed path; it is not equated with any one local Jacobian rank.

Machine rows: `{len(path)}`.
""",
    )
    target_rows = (
        target.groupby("target_bundle")
        .agg(
            r90=("rank_90", "median"),
            r95=("rank_95", "median"),
            r99=("rank_99", "median"),
        )
        .reset_index()
        .to_dict("records")
    )
    _write(
        root,
        "CAUSAL_TARGET_COMPLETENESS_V13.md",
        "# Causal target completeness — V13\n\n"
        + _table(target_rows, ["target_bundle", "r90", "r95", "r99"])
        + f"\n\nSemantic failure present: `{analysis['semantic_failure_present']}`; consistent with material target omission under the frozen rule: `{analysis['semantic_failure_consistent_with_target_omission']}`. The complete bundle adds continuous semantic logits and selected intermediate workspace endpoints; rank increase is sensitivity evidence, not proof of a complete causal state.\n",
    )
    _write(
        root,
        "JOINT_CHANNEL_CAUSAL_GEOMETRY_V13.md",
        f"""# Joint REC / convolution / KV causal geometry — V13

- fraction of frozen probe directions with >10% score energy in at least two channels: `{analysis["cross_channel_mixed_direction_fraction"]:.3f}`
- dominant directions judged cross-channel: `{analysis["dominant_directions_cross_channel"]}`
- same-prompt tangent rotation by dominant channel: `{analysis["same_prompt_tangent_rotation_by_channel_degrees"]}`

This joint loading explains why independently factorized variance PCA can discard low-variance causal combinations spanning REC, convolution, and KV fields.
""",
    )
    _write(
        root,
        "STRICT_STATE_REPLACEMENT_V13.md",
        f"""# Strict state replacement — V13

Status: **{analysis["strict_replacement_status"]}**.

V13 representations and tangent oracles encode causal edits relative to a clean cache. They do not encode an absolute complete cache and therefore do not authorize removal of the raw scaffold. No replacement success is claimed.
""",
    )
    final_section = f"""<!-- V13_START -->
## V13 — expanded causal bank, probe scaling, and path geometry

Formal decision: **{analysis["formal_outcome"]}**.

- New bank sizes: `{bank["bank_sizes"]}`; independent final was not used for selection.
- Restricted r95 curve m=64/128/256/512: `{analysis["probe_r95_curve"]}`.
- Stable low local causal rank under the frozen expanded probes: `{analysis["stable_low_local_causal_rank"]}`.
- Low-variance/high-causal directions: `{analysis["low_variance_high_causal_directions"]}`.
- Same-prompt rank-16 tangent angle: `{analysis["same_prompt_rank16_mean_angle"]:.2f}°`.
- Moving tangent improves static: `{analysis["moving_tangent_improves_static"]}`; independent h1/h2/h4/h8 pass: `{analysis["moving_tangent_passes_all_horizons"]}`.
- Practical local linear radius: `{linearity["local_linear_radius"]}`.
- Instantaneous/cumulative median r95: `{analysis["median_instantaneous_r95"]}` / `{analysis["median_cumulative_path_r95"]}`.
- Smallest independently validated writable dimension: `{analysis["smallest_independently_validated_writable_dimension"]}`.
- Complete replacement state: **False** (`{analysis["strict_replacement_status"]}`).
- H2 remains: **{analysis["h2_remains"]}**. H3 authorized: **{analysis["h3_authorized"]}**.
- Autonomous controller authorized: **{analysis["autonomous_controller_authorized"]}**.

Historical V1–V12 conclusions remain frozen. V13 causal-edit results are not described as absolute state replacement, and restricted exact-JVP rank is not described as full raw-state intrinsic dimension.
<!-- V13_END -->
"""
    final_path = root / "reports/FINAL_REPORT.md"
    current = final_path.read_text(encoding="utf-8")
    marker = "<!-- V13_START -->"
    if marker in current:
        current = current.split(marker)[0].rstrip()
    final_path.write_text(
        current.rstrip() + "\n\n" + final_section.rstrip() + "\n", encoding="utf-8"
    )
    reports = [
        "CAUSAL_BANK_EXPANSION_V13.md",
        "DATA_DIMENSION_SCALING_V13.md",
        "CAUSAL_PROBE_SCALING_V13.md",
        "CAUSAL_RANK_ROBUSTNESS_V13.md",
        "CAUSAL_TANGENT_ATLAS_V13.md",
        "MOVING_TANGENT_CAUSAL_ORACLE_V13.md",
        "LOCAL_LINEARITY_RADIUS_V13.md",
        "CAUSAL_PATH_DIMENSION_V13.md",
        "CAUSAL_TARGET_COMPLETENESS_V13.md",
        "JOINT_CHANNEL_CAUSAL_GEOMETRY_V13.md",
        "STRICT_STATE_REPLACEMENT_V13.md",
    ]
    integrity = {
        "schema_version": 22,
        "protocol_version": "causal_path_geometry_v13",
        "reports": {
            f"reports/{name}": sha256_file(root / "reports" / name) for name in reports
        },
        "final_report_sha256": sha256_file(final_path),
    }
    write_json_atomic(
        root / "results/v13/processed/report_integrity_v13.json", integrity
    )
    return integrity


def main() -> None:
    result = generate(Path.cwd())
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
