"""Append-only audit of the V38 three-way interaction metric semantics."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from jclosure.provenance import sha256_file, write_json_atomic


OUT = Path("results/v39/processed")
V38 = Path("results/v38/processed")
ORDER = ("R000", "R100", "R010", "R001", "R110", "R101", "R011", "R111")


def _vectors(root: Path, key: str, role: str) -> tuple[np.ndarray, list[str], dict[str, str]]:
    stages = ("singles", "pairs", "full")
    archives = {}
    hashes = {}
    for stage in stages:
        path = root / V38 / f"trajectory_{stage}_{key}_{role}_v38.npz"
        hashes[str(path.relative_to(root))] = sha256_file(path)
        archives[stage] = np.load(path)
    ids = archives["singles"]["state_ids"].tolist()
    if any(archives[stage]["state_ids"].tolist() != ids for stage in stages):
        raise RuntimeError(f"V38 {key}/{role} state-order drift")
    by = {}
    for stage in stages:
        for i, name in enumerate(archives[stage]["conditions"].tolist()):
            by[name] = archives[stage]["vectors"][:, i].astype(np.float64)
    if set(by) != set(ORDER):
        raise RuntimeError(f"V38 {key}/{role} condition set drift")
    return np.stack([by[name] for name in ORDER], axis=1), ids, hashes


def run(root: Path) -> dict:
    # Audit is deliberately prior to any V39 formal outcome or V39 design freeze.
    if (root / "artifacts/interaction_genesis_v39.freeze.json").exists():
        raise RuntimeError("V39 formal protocol already frozen before metric audit")
    (root / OUT).mkdir(parents=True, exist_ok=True)
    rows, summaries, source_hashes = [], {}, {}
    for key in ("Q", "F"):
        for role in ("development", "validation"):
            y, ids, hashes = _vectors(root, key, role)
            source_hashes.update(hashes)
            frame = pd.read_parquet(root / V38 / f"trajectory_analysis_{key}_{role}_v38.parquet")
            if frame.state_id.tolist() != ids:
                raise RuntimeError("V38 analysis alignment drift")
            e = y - y[:, :1]
            e000, e100, e010, e001, e110, e101, e011, e111 = np.moveaxis(e, 1, 0)
            interaction = e111 - e110 - e101 - e011 + e100 + e010 + e001
            effect_norm = np.linalg.norm(e111, axis=1)
            interaction_norm = np.linalg.norm(interaction, axis=1)
            if np.any(effect_norm <= 1.0):
                raise RuntimeError("V38 floor-active states require a separate denominator audit")
            dot = np.sum(interaction * e111, axis=1)
            f = interaction_norm / effect_norm
            p = dot / effect_norm**2
            c = dot / np.maximum(interaction_norm * effect_norm, 1e-12)
            old = dot / effect_norm
            if not np.all(np.abs(p) <= f + 1e-10):
                raise RuntimeError("V39 projection bound violated")
            if not np.allclose(p, f * c, atol=1e-10, rtol=1e-10):
                raise RuntimeError("V39 p=f*cos identity violated")
            if not np.allclose(frame.threeway_fraction.to_numpy(float), f, atol=1e-5, rtol=1e-5):
                raise RuntimeError("V38 stored norm fraction does not match vectors")
            if not np.allclose(frame.threeway_projection.to_numpy(float), old, atol=1e-5, rtol=1e-5):
                raise RuntimeError("V38 stored projection is not signed length component")
            sign_stability = max(float(np.mean(old > 0)), float(np.mean(old < 0)))
            record = json.loads((root / V38 / f"trajectory_analysis_{key}_{role}_v38.json").read_text())
            if abs(sign_stability - record["metrics"]["threeway_sign_stability"]) > 1e-10:
                raise RuntimeError("V38 sign stability mismatch")
            summaries[f"{key}_{role}"] = {
                "states": len(ids), "median_f": float(np.median(f)),
                "median_p": float(np.median(p)), "median_cosine": float(np.median(c)),
                "median_old_length_projection": float(np.median(old)),
                "sign_stability": sign_stability,
                "max_abs_identity_error": float(np.max(np.abs(p - f*c))),
                "max_projection_bound_excess": float(np.max(np.abs(p)-f)),
                "v38_higher_order_gate_unchanged": True,
            }
            for i, sid in enumerate(ids):
                rows.append({"model": key, "role": role, "state_id": sid,
                             "family": frame.family.iloc[i], "norm_fraction": f[i],
                             "signed_projection_coefficient": p[i], "cosine": c[i],
                             "old_length_projection": old[i],
                             "full_effect_norm": effect_norm[i],
                             "interaction_norm": interaction_norm[i],
                             "identity_residual": p[i]-f[i]*c[i],
                             "bound_satisfied": bool(abs(p[i]) <= f[i]+1e-10)})
    table = root / OUT / "v38_interaction_metric_audit_v39.parquet"
    pd.DataFrame(rows).to_parquet(table, index=False, compression="zstd")
    result = {"audit_status": "V38_PROJECTION_LABEL_AMBIGUOUS; ADD_APPEND_ONLY_AMENDMENT",
              "v38_old_projection_definition": "dot(I234,E111)/max(norm(E111),1.0): signed length component",
              "corrected_p_definition": "dot(I234,E111)/norm(E111)^2",
              "corrected_fraction_definition": "norm(I234)/norm(E111)",
              "cosine_definition": "dot(I234,E111)/(norm(I234)*norm(E111))",
              "sign_stability_definition": "max(fraction positive old projection, fraction negative old projection)",
              "v38_formal_classification_unchanged": True,
              "summaries": summaries, "source_vector_sha256": source_hashes,
              "table_sha256": sha256_file(table),
              "v38_analysis_source_sha256": sha256_file(root / "src/jclosure/experiments/analysis_v38.py")}
    path = root / OUT / "v38_interaction_metric_audit_v39.json"
    write_json_atomic(path, result)
    lines = ["# V39: V38 三阶交互指标审计与追加勘误", "",
             "V38 原始代码将 `threeway_projection` 计算为 `<I234,E111>/||E111||`（本批状态没有触发分母下限）。"
             "这是有量纲的沿效应方向分量，不是无量纲投影系数 `p=<I234,E111>/||E111||²`。"
             "因此 V38 表中该数值大于 `||I234||/||E111||` 并不违反实现，但若读作投影系数则数学含义错误。", "",
             "V38 的 `threeway_fraction` 与标准范数比一致；正负号不因正分母的额外归一化而改变，"
             "故其符号稳定性及原有门槛判定保持不变。V1–V38 文件未改动。", "",
             "|模型|角色|n|旧有量纲分量中位数|修正 p 中位数|f 中位数|cos 中位数|符号稳定性|",
             "|---|---|---:|---:|---:|---:|---:|---:|"]
    for key in ("Q", "F"):
        for role in ("development", "validation"):
            s = summaries[f"{key}_{role}"]
            lines.append(f"|{key}|{role}|{s['states']}|{s['median_old_length_projection']:.3f}|"
                         f"{s['median_p']:.3f}|{s['median_f']:.3f}|{s['median_cosine']:.3f}|"
                         f"{s['sign_stability']:.3f}|")
    lines.extend(["", "逐状态验证 `p=f×cos` 与 `|p|≤f`；中位数之间不要求满足乘法恒等式。",
                  "完整逐状态表：`results/v39/processed/v38_interaction_metric_audit_v39.parquet`。",
                  "V39 后续一律用明确标注的无量纲 `p`、范数比 `f` 和余弦 `c`，不再含糊使用“projection”。", ""])
    report = root / "reports/V39_V38_INTERACTION_METRIC_AUDIT.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    amendment = root / "reports/V38_METRIC_AMENDMENT_V39.md"
    amendment.write_text("# V38 指标追加勘误（V39 审计）\n\n" + "\n".join(lines[2:]), encoding="utf-8")
    return {"json": str(path.relative_to(root)), "parquet": str(table.relative_to(root)),
            "report": str(report.relative_to(root)), "amendment": str(amendment.relative_to(root)),
            "summaries": summaries}


if __name__ == "__main__":
    print(json.dumps(run(Path.cwd()), indent=2, ensure_ascii=False))
