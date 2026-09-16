"""Render all required protocol-v12 reports from machine-readable records."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import pandas as pd

from jclosure.provenance import sha256_file, write_json_atomic

REPORTS = (
    "CAUSAL_MEASUREMENT_AUDIT_V12.md",
    "DATA_RANK_SCALING_V12.md",
    "LOCAL_CAUSAL_JACOBIAN_V12.md",
    "CAUSAL_TANGENT_GEOMETRY_V12.md",
    "LOCAL_CAUSAL_ORACLE_V12.md",
    "VARIANCE_VS_CAUSAL_GEOMETRY_V12.md",
    "CAUSAL_TRAJECTORY_DYNAMICS_V12.md",
    "STRICT_STATE_REPLACEMENT_V12.md",
)


def _read(root: Path, name: str) -> dict[str, Any]:
    return json.loads((root / name).read_text(encoding="utf-8"))


def _write(root: Path, name: str, text: str) -> None:
    path = root / "reports" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def _estimate(value: dict[str, Any] | None) -> str:
    return "NA" if value is None else f"{float(value['estimate']):.3f}"


def _causal_table(summary: dict[str, Any]) -> str:
    lines = [
        "| method | h | direction | magnitude | semantic | output | sign | all-family gate |",
        "|---|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for method, horizons in summary.get("aggregates", {}).items():
        for horizon, groups in sorted(
            horizons.items(), key=lambda value: int(value[0])
        ):
            pooled = groups["pooled"]
            lines.append(
                f"| {method} | {horizon} | {_estimate(pooled.get('direction_cosine'))} | "
                f"{_estimate(pooled.get('magnitude_ratio'))} | {_estimate(pooled.get('semantic_delta_agreement'))} | "
                f"{_estimate(pooled.get('output_direction_cosine'))} | "
                f"{_estimate(pooled.get('task_decision_sign_agreement'))} | "
                f"{'PASS' if groups.get('all_family_gate_pass') else 'FAIL'} |"
            )
    return "\n".join(lines)


def _report_measurement(root: Path, value: dict[str, Any]) -> None:
    rows = [
        "| condition | continuous cosine | rank corr | top-10 | weighted top-10 | teacher norm |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, metrics in value["condition_means"].items():
        rows.append(
            f"| {name} | {metrics['semantic_vector_cosine']:.3f} | "
            f"{metrics['semantic_rank_correlation']:.3f} | {metrics['semantic_topk_overlap']:.3f} | "
            f"{metrics['semantic_weighted_topk_overlap']:.3f} | {metrics['teacher_j_effect_norm']:.3f} |"
        )
    answer = (
        "显著受 top-k metric instability 影响"
        if value["semantic_metric_instability_material"]
        else "未发现足以解释 V11 失败的显著 metric instability"
    )
    _write(
        root,
        REPORTS[0],
        f"""# Causal measurement audit — V12

本协议严格区分 predictive fidelity、intervention/writeback fidelity 与 state-replacement fidelity；三者不可互换。冻结的 0.8 semantic gate 未修改。

{chr(10).join(rows)}

结论：**{answer}**。判据为 continuous direction ≥0.95 时 top-k overlap 是否仍跌破 0.8。该审计仅使用 development bank 的 {value["pair_count"]} 个 pair。

Machine record: `{value["records"]}` (`{value["records_sha256"]}`).
""",
    )


def _report_scaling(root: Path, value: dict[str, Any]) -> None:
    slopes = "\n".join(
        f"- `{name}`: N=150→600 的 d128 direction 变化拟合为 {slope:+.3f}."
        for name, slope in value["direction_improvement_n150_to_n600_at_d128"].items()
    )
    _write(
        root,
        REPORTS[1],
        f"""# Data × rank scaling — V12

- 可用的 balanced nested train sizes: `{value["available_train_sizes"]}`。
- 请求但原始冻结 capture bank 不支持的 sizes: `{value["unavailable_train_sizes"]}`，状态为 `NOT_IDENTIFIED_BANK_SIZE`。
- 512D 有充分 empirical rank 的 sizes: `{value["identified_512_train_sizes"]}`。
- fixed-512 saturation 是否可识别: `{value["fixed_512_saturation_identified"]}`。

{slopes}

结论：**MORE_DATA_REQUIRED**。现有冻结原始状态只有 N=600 能识别 512D，因此不能声称 512D 已饱和，也不能把 rank 不足 silently clamp 成较小维度。

Offline record SHA256: `{value["offline_records_sha256"]}`. Causal record SHA256: `{value["causal_records_sha256"]}`.
""",
    )


def _report_jacobian(root: Path, value: dict[str, Any]) -> None:
    _write(
        root,
        REPORTS[2],
        f"""# Local causal Jacobian / JVP — V12

使用关闭 Flash SDP 后的 **exact autograd JVP**，不是 finite-horizon ratio。输入 operator 限制到冻结的 64 个 empirical joint-PC raw-state directions；target 是 h1/h2/h4 的 256 个 selected-J coordinates 与 16 个 logits。

- sampled local states: `{value["state_count"]}`
- mean stable rank: `{value["mean_stable_rank"]:.2f}`
- mean effective rank: `{value["mean_effective_rank"]:.2f}`
- median restricted r90/r95/r99: `{value["median_rank_90"]:.0f}` / `{value["median_rank_95"]:.0f}` / `{value["median_rank_99"]:.0f}`

这些数只估计 64-direction empirical restriction 下的 local causal dimension，**不是**完整 raw persistent state 或整个模型的全局 intrinsic dimension。

Machine record: `{value["records"]}` (`{value["records_sha256"]}`).
""",
    )


def _report_tangent(root: Path, value: dict[str, Any]) -> None:
    frame = pd.DataFrame(value["relation_rank_means"])
    table = frame.to_markdown(index=False, floatfmt=".3f")
    _write(
        root,
        REPORTS[3],
        f"""# Causal tangent geometry — V12

Principal angles are measured between right-singular subspaces in the same frozen 64-direction coordinate system.

{table}

- Large tangent rotation (frozen ≥30° rule): `{value["large_tangent_rotation"]}`.
- Local rank reached the probe boundary: `{value["local_rank_probe_limited"]}`.

`local rank low + angles large` 才支持 curved low-dimensional atlas；若 rank 接近 64-direction probe boundary，则只能报告 probe-limited evidence，不能声称全局低维。

Machine record: `{value["records"]}` (`{value["records_sha256"]}`).
""",
    )


def _report_oracle(
    root: Path,
    stage1: dict[str, Any],
    stage2: dict[str, Any],
    confirm: dict[str, Any],
    adjudication: dict[str, Any],
) -> None:
    _write(
        root,
        REPORTS[4],
        f"""# Local causal oracle — V12

Global PCA、global causal covariance basis、current-J-neighborhood local PCA 与 local causal basis 都通过同一 raw-state dual reconstruction interface 比较。Local 512D 因 512-neighbor centered rank≤511 被正式标记 `NOT_IDENTIFIED_RANK_LIMIT`。

## Development h1

{_causal_table(stage1)}

## Development staged h2/h4/h8 (h16 only after h1 pass)

{_causal_table(stage2)}

## New independent V12 confirmation

{_causal_table(confirm)}

- Confirmed authorized methods: `{confirm.get("authorized_methods", [])}`
- Smallest causally validated dimension: `{adjudication["smallest_causally_validated_dimension"]}`
- Formal outcome: **{adjudication["decision_tree_outcome"]} — {adjudication["formal_conclusion"]}**
""",
    )


def _report_variance(root: Path, value: dict[str, Any]) -> None:
    metrics = value["rank16_means"]
    _write(
        root,
        REPORTS[5],
        f"""# Variance directions vs causal directions — V12

At rank 16 in the frozen 64-direction operator:

- PCA/causal subspace overlap: `{metrics["pca_causal_subspace_overlap"]:.3f}`
- mean principal angle: `{metrics["mean_principal_angle_degrees"]:.2f}°`
- causal sensitivity captured by leading PCA directions: `{metrics["causal_sensitivity_explained_by_leading_pca"]:.3f}`
- low-variance/high-causal-sensitivity directions present: `{value["low_variance_high_causal_directions_present"]}`
- REC/conv/KV contribution: `{metrics["recurrent_contribution"]:.3f}` / `{metrics["conv_contribution"]:.3f}` / `{metrics["kv_contribution"]:.3f}`

This comparison is restricted to the frozen empirical probe domain; it does not identify the full raw-state spectrum.

Machine record: `{value["records"]}` (`{value["records_sha256"]}`).
""",
    )


def _report_trajectory(root: Path, value: dict[str, Any]) -> None:
    _write(
        root,
        REPORTS[6],
        f"""# Causal trajectory dynamics — V12

- Source split: `{value["source_split"]}`
- Error norm monotone: `{value["error_norm_monotone"]}`
- trajectory-angle change: `{value["angle_change_degrees"]:+.2f}°`
- semantic-cosine change: `{value["semantic_cosine_change"]:+.3f}`
- dominant observed failure mode: **{value["dominant_failure_mode"]}**

Each record includes `||e_h||`, teacher-trajectory angle, semantic trajectory cosine, and error components parallel/orthogonal to the teacher effect. Finite-horizon ratios are explicitly **not** reported as Jacobian eigenvalues.

Machine record: `{value["records"]}` (`{value["records_sha256"]}`).
""",
    )


def _report_replacement(root: Path, value: dict[str, Any]) -> None:
    _write(
        root,
        REPORTS[7],
        f"""# Strict state replacement — V12

- Tested scope: `{value["replacement_scope"]}`
- Raw target-state bypass detected: `{value["raw_state_bypass_detected"]}`
- Compact-only full-state verified: `{value["compact_only_full_state_verified"]}`
- Strict full-state replacement pass: `{value["strict_full_state_replacement_pass"]}`
- Reason: {value["reason"]}.

The diagnostic overwrites/reconstructs the intervention-bearing persistent channels through the strict prepared-state interface. It does **not** relabel this as complete model-cache replacement: unmodeled cache fields remain a structural scaffold. Consequently it cannot authorize H3 or an autonomous controller.

{_causal_table(value)}

Machine record: `{value["records"]}` (`{value["records_sha256"]}`).
""",
    )


def _final_section(values: dict[str, Any]) -> str:
    a = values["adjudication"]
    scaling = values["scaling"]
    jacobian = values["jacobian"]
    tangent = values["tangent"]
    trajectory = values["trajectory"]
    measurement = values["measurement"]
    replacement = values["replacement"]
    return f"""<!-- V12_START -->
## V12 — local causal geometry and intrinsic-dimension audit

Formal decision: **{a["decision_tree_outcome"]} — {a["formal_conclusion"]}**. Hypothesis remains **{a["hypothesis"]}**. Smallest independently validated writable dimension: **{a["smallest_causally_validated_dimension"]}**. Autonomous controller authorized: **{a["autonomous_controller_authorized"]}**.

### Required scientific answers

1. Semantic metric instability materially explains V11: **{measurement["semantic_metric_instability_material"]}**.
2. 512D data/rank saturation: **not identified**; only train sizes `{scaling["identified_512_train_sizes"]}` had adequate centered rank, so `MORE_DATA_REQUIRED`.
3. Local causal effective rank: stable `{jacobian["mean_stable_rank"]:.2f}`, entropy-effective `{jacobian["mean_effective_rank"]:.2f}`, restricted median r90/r95/r99 `{jacobian["median_rank_90"]:.0f}/{jacobian["median_rank_95"]:.0f}/{jacobian["median_rank_99"]:.0f}` within the frozen 64-direction probe.
4. Causal/PCA overlap: see `VARIANCE_VS_CAUSAL_GEOMETRY_V12.md`; conclusions are restricted to the 64-direction operator.
5. Low-variance/high-causal directions: **{a["low_variance_high_causal_directions_present"]}**.
6. Tangent rotation is large under the frozen rule: **{tangent["large_tangent_rotation"]}**.
7. Locally-low-dimensional/globally-curved state established: **False** unless both low local rank and successful local oracle hold; current outcome does not authorize that claim.
8. Local causal oracle clearly exceeds global PCA: adjudicated from the independent table, but **no method is authorized** unless listed here: `{a["causal_authorized_methods"]}`.
9. A 128/256/384/512D local candidate passes h1: **{bool(a["causal_authorized_methods"])}** (512D local itself was rank-limited).
10. h2/h4/h8 pass: **{bool(a["causal_authorized_methods"])}** under the all-family frozen gates.
11. Dominant trajectory failure: **{trajectory["dominant_failure_mode"]}**; finite-horizon ratios were not treated as eigenvalues.
12. Strict full-state replacement succeeded: **{replacement["strict_full_state_replacement_pass"]}**.
13. Writable dimension is proven higher than predictive dimension: **not proven globally**; current restricted causal spectrum and failed writeback remain consistent with a larger writable state.
14. Strongest supported outcome: **{a["decision_tree_outcome"]} — {a["formal_conclusion"]}**.
15. H2 remains: **{a["hypothesis"] == "H2"}**.
16. Upgrade to H3: **{a["h3_upgrade_authorized"]}**.
17. Autonomous controller authorization: **{a["autonomous_controller_authorized"]}**.

The learned state-dependent decoder was not run because the protocol did not authorize it before causal geometry and strict replacement succeeded.
<!-- V12_END -->"""


def _update_final(root: Path, section: str) -> None:
    path = root / "reports/FINAL_REPORT.md"
    text = path.read_text(encoding="utf-8")
    start, end = "<!-- V12_START -->", "<!-- V12_END -->"
    if start in text and end in text:
        text = text.split(start, 1)[0].rstrip() + "\n\n" + section + "\n"
    else:
        text = text.rstrip() + "\n\n" + section + "\n"
    path.write_text(text, encoding="utf-8")


def generate(root: Path) -> dict[str, Any]:
    values = {
        "measurement": _read(root, "results/v12/processed/measurement_audit_v12.json"),
        "scaling": _read(
            root, "results/v12/processed/data_rank_scaling_analysis_v12.json"
        ),
        "jacobian": _read(root, "results/v12/processed/local_causal_jacobian_v12.json"),
        "tangent": _read(
            root, "results/v12/processed/causal_tangent_geometry_v12.json"
        ),
        "variance": _read(root, "results/v12/processed/variance_vs_causal_v12.json"),
        "trajectory": _read(
            root, "results/v12/processed/causal_trajectory_dynamics_v12.json"
        ),
        "stage1": _read(
            root, "results/v12/processed/local_causal_oracle_stage1_v12.json"
        ),
        "stage2": _read(
            root, "results/v12/processed/local_causal_oracle_stage2_v12.json"
        ),
        "confirm": _read(root, "results/v12/processed/causal_confirmatory_v12.json"),
        "replacement": _read(
            root, "results/v12/processed/strict_state_replacement_v12.json"
        ),
        "adjudication": _read(root, "results/v12/processed/adjudication_v12.json"),
    }
    _report_measurement(root, values["measurement"])
    _report_scaling(root, values["scaling"])
    _report_jacobian(root, values["jacobian"])
    _report_tangent(root, values["tangent"])
    _report_oracle(
        root,
        values["stage1"],
        values["stage2"],
        values["confirm"],
        values["adjudication"],
    )
    _report_variance(root, values["variance"])
    _report_trajectory(root, values["trajectory"])
    _report_replacement(root, values["replacement"])
    section = _final_section(values)
    _update_final(root, section)
    subprocess.run(
        ["python", "scripts/build_complete_version_report.py", "V12"],
        cwd=root,
        check=True,
    )
    paths = [root / "reports" / value for value in REPORTS]
    paths += [root / "reports/FINAL_REPORT.md", root / "reports/V12_COMPLETE_REPORT.md"]
    manifest = {
        "protocol_version": "local_causal_geometry_v12",
        "reports": {
            str(path.relative_to(root)): {
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in paths
        },
    }
    write_json_atomic(
        root / "results/v12/processed/report_integrity_v12.json", manifest
    )
    return manifest


def main() -> None:
    root = Path.cwd()
    manifest = generate(root)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
