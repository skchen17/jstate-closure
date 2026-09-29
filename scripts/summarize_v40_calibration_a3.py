"""Aggregate saved V40 calibration JSONL; no formal outcome adjudication."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, median

from jclosure.protocol_v40_a3 import stage_freeze, verify
from jclosure.provenance import sha256_file, write_json_atomic

SOURCE = "scripts/summarize_v40_calibration_a3.py"
CONDITIONS = ("BASE", "REC_only", "Conv_only", "KV_only", "REC_Conv",
              "REC_Conv_KV", "DONOR_NATIVE")


def load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def summarize_model(rows: list[dict], model: str) -> dict:
    valid = [r for r in rows if r["status"] == "VALID" and r["delay"] == 0]
    invalid = [r for r in rows if r["status"] != "VALID"]
    grouped = defaultdict(dict)
    for row in valid:
        key = row["state_id"]
        if row["condition"] in grouped[key]:
            raise RuntimeError(f"duplicate V40 calibration row {model}:{key}:{row['condition']}")
        grouped[key][row["condition"]] = row
    if len(grouped) != 16 or any(set(bundle) != set(CONDITIONS) for bundle in grouped.values()):
        raise RuntimeError(f"incomplete V40 calibration matrix: {model}")
    families = {}
    for family in sorted({bundle["BASE"]["family"] for bundle in grouped.values()}):
        bundles = [b for b in grouped.values() if b["BASE"]["family"] == family]
        qualified = [b for b in bundles if b["BASE"]["recipient_correct"] and
                     b["DONOR_NATIVE"]["donor_recovered"]]
        condition = {}
        for name in CONDITIONS:
            donor_recovery = [bool(b[name]["donor_recovered"]) for b in bundles]
            shifts = [b[name]["donor_minus_recipient_logit"] -
                      b["BASE"]["donor_minus_recipient_logit"] for b in bundles]
            condition[name] = {
                "donor_recovered_count": sum(donor_recovery),
                "donor_recovered_fraction": mean(donor_recovery),
                "median_margin_shift_from_base": median(shifts),
                "qualified_pair_donor_recovered_count":
                    sum(bool(b[name]["donor_recovered"]) for b in qualified),
            }
        families[family] = {
            "states": len(bundles),
            "recipient_native_correct": sum(bool(b["BASE"]["recipient_correct"])
                                            for b in bundles),
            "donor_native_correct": sum(bool(b["DONOR_NATIVE"]["donor_recovered"])
                                        for b in bundles),
            "baseline_qualified_pairs": len(qualified),
            "conditions": condition,
        }
    native = [b[name] for b in grouped.values() for name in ("BASE", "DONOR_NATIVE")]
    proof_rows = [r for r in valid if r["condition"] not in ("BASE", "DONOR_NATIVE")]
    if any(not r["native_cache_proof"]["requested_exact"] or
           not r["native_cache_proof"]["untouched_exact"] for r in proof_rows):
        raise RuntimeError(f"V40 nonexact native swap: {model}")
    return {
        "model": model, "states": len(grouped), "invalid_records": invalid,
        "valid_condition_rows": len(valid), "native_swap_exact_rows": len(proof_rows),
        "split_full_max_abs_logit_max": max(r["split_full_max_abs_logit"] for r in native),
        "split_full_max_abs_logit_median": median(r["split_full_max_abs_logit"] for r in native),
        "split_full_constrained_prediction_agreement":
            mean(r["predicted_answer"] == r["direct_full_score"]["predicted_answer"]
                 for r in native),
        "families": families,
    }


def run(root: Path, q_name: str, f_name: str) -> dict:
    verify(root)
    raw_names = {"Q": q_name, "F": f_name}
    result = {"status": "CALIBRATION_ONLY_NOT_FORMAL_V40_EVIDENCE",
              "raw_files": {}, "models": {},
              "development_validation_final_unopened": True}
    for key, name in raw_names.items():
        path = root / name
        if path.parent != root / "results/v40/raw" or not path.name.startswith(f"calibration_{key}_a3-full-"):
            raise RuntimeError(f"Unexpected V40 a3 full raw path: {name}")
        rows = load(path)
        if any(r["model"] != key or "a3-full-" not in r["run_id"] for r in rows):
            raise RuntimeError("V40 raw run identity drift")
        result["raw_files"][key] = {"path": name, "sha256": sha256_file(path)}
        result["models"][key] = summarize_model(rows, key)
    out = root / "results/v40/processed/calibration_a3_adjudication.json"
    if out.exists():
        raise RuntimeError("V40 a3 adjudication already exists")
    write_json_atomic(out, result)
    report = root / "reports/V40_CALIBRATION_A3.md"
    lines = ["# V40 — Sequential Native-Cache Calibration", "",
             "Status: calibration only; no development, validation, or independent-final outcomes.", "",
             "The a2 batched-cache pilots remain historical diagnostics. The a3 runs use native",
             "single-token continuation and exact native channel cache swaps. All counts and",
             "numbers below are generated from the saved a3 JSONL records.", "",
             "| Model | Family | States | Recipient native correct | Donor native correct | Both correct | REC donor recovery | Conv donor recovery | KV donor recovery | REC+Conv donor recovery | All-channel donor recovery |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for key in ("Q", "F"):
        x = result["models"][key]
        for family, row in x["families"].items():
            c = row["conditions"]
            lines.append("| " + " | ".join(map(str, [key, family, row["states"],
                         row["recipient_native_correct"], row["donor_native_correct"],
                         row["baseline_qualified_pairs"],
                         c["REC_only"]["donor_recovered_count"],
                         c["Conv_only"]["donor_recovered_count"],
                         c["KV_only"]["donor_recovered_count"],
                         c["REC_Conv"]["donor_recovered_count"],
                         c["REC_Conv_KV"]["donor_recovered_count"]])) + " |")
    lines.extend(["", "Native-write audit: all intervention rows have exact requested and untouched-field proofs.",
                  "", "| Model | Exact intervention rows | Max split/full logit difference | Split/full constrained-prediction agreement |",
                  "|---|---:|---:|---:|"])
    for key in ("Q", "F"):
        x = result["models"][key]
        lines.append(f"| {key} | {x['native_swap_exact_rows']} | "
                     f"{x['split_full_max_abs_logit_max']:.4f} | "
                     f"{x['split_full_constrained_prediction_agreement']:.3f} |")
    lines.extend(["", "These calibration numbers are diagnostic. A channel role, lifetime, or",
                  "trajectory mechanism requires a separately frozen formal design and held-out",
                  "task-grounded outcomes. Poor native answer recovery limits interpretation.", ""])
    report.write_text("\n".join(lines), encoding="utf-8")
    stage_freeze(root, "calibration_adjudication",
                 [SOURCE, str(out.relative_to(root)), str(report.relative_to(root)),
                  *raw_names.values()],
                 {"adjudication_sha256": sha256_file(out),
                  "report_sha256": sha256_file(report),
                  "calibration_only": True})
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--q", required=True)
    parser.add_argument("--f", required=True)
    args = parser.parse_args()
    print(json.dumps(run(Path.cwd(), args.q, args.f), indent=2))
