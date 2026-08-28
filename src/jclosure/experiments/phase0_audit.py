"""Forensic attribution of the v1 Phase 0 protocol failure."""

from __future__ import annotations

import argparse
import itertools
import json
import math
from typing import Any

import pandas as pd

from jclosure.experiments.common import repository_root
from jclosure.phase0 import official_pass_summary
from jclosure.provenance import sha256_file, write_json_atomic

FACTORS = ("official_aggregation", "all_positions", "synonym_expansion")


def _normalized(value: str) -> str:
    return " ".join(value.strip().casefold().split())


def _literal_rank(row: pd.Series) -> float:
    candidates = json.loads(row["candidate_ranks_json"])
    canonical = _normalized(str(row["canonical_concept"]))
    ranks = [
        int(candidate["rank"])
        for candidate in candidates
        if _normalized(str(candidate["surface"])) == canonical
    ]
    return float(min(ranks)) if ranks else math.nan


def _condition_metric(
    records: pd.DataFrame,
    *,
    family: str,
    official_aggregation: bool,
    all_positions: bool,
    synonym_expansion: bool,
) -> float:
    selected = records[records["family"] == family].copy()
    if not all_positions:
        selected = selected[selected["position"] >= 16]
    if not synonym_expansion:
        selected["rank"] = selected.apply(_literal_rank, axis=1)
        selected["tokenizable"] = selected["rank"].notna()
    if selected.empty:
        return 0.0
    summary = official_pass_summary(
        selected,
        layers=sorted(selected["layer"].unique()),
    )["families"].get(family, {})
    if not summary:
        return 0.0
    jacobian = summary["jacobian"]
    return float(
        jacobian["pass_at"]["10"]
        if official_aggregation
        else jacobian["strict_all_layers_sensitivity"]["10"]
    )


def _factorial(records: pd.DataFrame, family: str) -> dict[str, float]:
    values: dict[str, float] = {}
    for bits in itertools.product((False, True), repeat=3):
        state = dict(zip(FACTORS, bits, strict=True))
        key = "+".join(name for name, enabled in state.items() if enabled) or "v1_like"
        values[key] = _condition_metric(records, family=family, **state)
    return values


def _shapley(records: pd.DataFrame, family: str) -> dict[str, float]:
    cache: dict[frozenset[str], float] = {}

    def value(enabled: frozenset[str]) -> float:
        if enabled not in cache:
            cache[enabled] = _condition_metric(
                records,
                family=family,
                **{factor: factor in enabled for factor in FACTORS},
            )
        return cache[enabled]

    contributions = {factor: 0.0 for factor in FACTORS}
    permutations = list(itertools.permutations(FACTORS))
    for order in permutations:
        active: frozenset[str] = frozenset()
        for factor in order:
            updated = active | {factor}
            contributions[factor] += value(updated) - value(active)
            active = updated
    return {factor: contribution / len(permutations) for factor, contribution in contributions.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration", default="results/processed/phase0_v2_calibration.json")
    parser.add_argument("--v1-gate", default="results/processed/phase0_gate.json")
    args = parser.parse_args()
    root = repository_root()
    calibration_path = root / args.calibration
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    record_path = root / "results/raw" / calibration["run_id"] / "readout_records_v2.parquet"
    records = pd.read_parquet(record_path)
    v1_path = root / args.v1_gate
    v1 = json.loads(v1_path.read_text(encoding="utf-8"))
    families: dict[str, Any] = {}
    for family in ("factual_two_hop", "order_of_operations"):
        family_records = records[records["family"] == family]
        corrected = _condition_metric(
            records,
            family=family,
            official_aggregation=True,
            all_positions=True,
            synonym_expansion=True,
        )
        concept = family_records.drop_duplicates(["example_id", "concept"])
        families[family] = {
            "v1_reported_hit10": v1.get("hit10_by_family", {}).get(family),
            "factorial_pass10": _factorial(records, family),
            "shapley_pass10_change": _shapley(records, family),
            "corrected_retrospective_pass10": corrected,
            "true_readout_failure_fraction": 1.0 - corrected,
            "raw_examples": int(family_records["example_id"].nunique()),
            "raw_concepts": int(concept.shape[0]),
            "tokenization_excluded_concepts": int((~concept["tokenizable"].astype(bool)).sum()),
            "position_lt16_examples": int(
                family_records.drop_duplicates("example_id")["position"].lt(16).sum()
            ),
            "copy_flagged_concepts": int(concept["copied_target"].astype(bool).sum()),
        }
    output = {
        "schema_version": 2,
        "protocol_version": "phase0_protocol_v2",
        "status": "RETROSPECTIVE_CALIBRATION_ONLY",
        "v1_gate_sha256": sha256_file(v1_path),
        "v2_calibration_sha256": sha256_file(calibration_path),
        "v2_records_sha256": sha256_file(record_path),
        "families": families,
        "warning": "These results were observed before protocol freeze and cannot adjudicate Phase 0 v2.",
    }
    machine_path = root / "results/processed/phase0_protocol_audit.json"
    write_json_atomic(machine_path, output)
    lines = [
        "# Phase 0 Protocol Audit",
        "",
        "## Material Passport",
        "",
        "- Protocol: `phase0_protocol_v2`",
        "- Verification Status: ANALYZED",
        "- Role: retrospective calibration only; not confirmatory adjudication",
        f"- Machine record: `{machine_path.relative_to(root)}`",
        "",
        "## Why the old gate failed",
        "",
        "The v1 family gate used the minimum per-layer hit@10, excluded positions below 16,",
        "and ranked only literal target strings. The factorial and Shapley values below",
        "quantify how each protocol choice changes the old-data pass@10 statistic; they do",
        "not establish that the frozen fresh-data gate will pass.",
        "",
    ]
    for family, payload in families.items():
        lines.extend(
            [
                f"### {family}",
                "",
                f"- v1 reported hit@10: `{payload['v1_reported_hit10']}`",
                f"- factorial pass@10: `{payload['factorial_pass10']}`",
                f"- Shapley attribution: `{payload['shapley_pass10_change']}`",
                f"- corrected retrospective pass@10: `{payload['corrected_retrospective_pass10']}`",
                f"- residual readout-failure fraction: `{payload['true_readout_failure_fraction']}`",
                f"- coverage: `{ {k: v for k, v in payload.items() if k.endswith('examples') or k.endswith('concepts')} }`",
                "",
            ]
        )
    lines.extend(
        [
            "## Interpretation boundary",
            "",
            "All values in this report reuse calibration data already inspected under v1.",
            "Only `PHASE0_V2_CONFIRMATORY.md` may adjudicate the repaired measurement gate.",
            "",
        ]
    )
    (root / "reports/PHASE0_PROTOCOL_AUDIT.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
