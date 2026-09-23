"""Machine-check the append-only V35 finalist-omission amendment."""
from __future__ import annotations

import json
from pathlib import Path

from jclosure.experiments.development_plan_v35 import FINAL_PRIORITY, candidates
from jclosure.provenance import sha256_file, write_json_atomic
from jclosure.protocol_v36 import stage_freeze, verify

OUT = Path("results/v36/processed")
SOURCE = "src/jclosure/experiments/audit_v35_v36.py"
EVIDENCE = (
    "configs/hierarchical_read_v35.yaml",
    "src/jclosure/experiments/depth_analysis_v35.py",
    "src/jclosure/experiments/development_plan_v35.py",
    "src/jclosure/experiments/final_opening_v35.py",
    "src/jclosure/experiments/adjudicate_v35.py",
    "results/v35/processed/depth_analysis_development_v35.json",
    "results/v35/processed/depth_analysis_validation_v35.json",
    "results/v35/processed/development_plan_v35.json",
    "results/v35/processed/final_opening_v35.json",
    "results/v35/processed/v35_adjudication.json",
)


def run(root: Path):
    verify(root)
    p = root / "results/v35/processed"
    read = lambda name: json.loads((p / name).read_text())
    dev = read("depth_analysis_development_v35.json")
    val = read("depth_analysis_validation_v35.json")
    plan = read("development_plan_v35.json")
    opening = read("final_opening_v35.json")
    adjud = read("v35_adjudication.json")
    measured = {}
    for role, data in (("development", dev), ("validation", val)):
        measured[role] = {}
        for model in ("Q", "F"):
            stage = data["models"][model]["conditions"]["Q2_Q3_Q4"]
            if not stage["pass"] or stage["families_passing"] != 5:
                raise RuntimeError("V35 triple not strong in every model/role")
            measured[role][model] = {k: stage[k] for k in
                                      ("removed_fraction", "restored_fraction",
                                       "reverse_correction_cosine", "families_passing", "pass")}
    if "LOCALIZED_Q2_Q3_Q4" not in FINAL_PRIORITY:
        raise RuntimeError("V35 triple absent from frozen priority")
    if "LOCALIZED_Q2_Q3_Q4" in candidates(dev["shared_qualitative_this_role"]):
        raise RuntimeError("V35 actual candidate generator unexpectedly includes triple")
    if "LOCALIZED_Q2_Q3_Q4" in plan["shared_development_candidates"]:
        raise RuntimeError("V35 actual development plan unexpectedly includes triple")
    if opening["selected_mechanism"] != "FULL_DEPTH_ONLY" or not adjud["no_shared_smaller_organization_passed"]:
        raise RuntimeError("V35 historical decision changed")
    result = {"audit_outcome": "V35_AUDIT_ADJUDICATION_AMENDMENT_REQUIRED",
              "reporting_correction_required": True,
              "triple_strong_gate": measured,
              "frozen_priority_contains_localized_triple": True,
              "implemented_candidate_generator_contains_localized_triple": False,
              "historical_development_candidates": plan["shared_development_candidates"],
              "historical_finalist": opening["selected_mechanism"],
              "historical_no_shared_smaller_statement": adjud["no_shared_smaller_organization_passed"],
              "counterfactual_priority_if_bug_fixed": "LOCALIZED_Q2_Q3_Q4",
              "counterfactual_final_not_observed": True,
              "historical_v35_files_modified": False,
              "source_sha256": {path: sha256_file(root / path) for path in EVIDENCE},
              "report_sha256": sha256_file(root / "reports/V36_V35_ADJUDICATION_AUDIT.md")}
    path = root / OUT / "v35_audit_v36.json"
    write_json_atomic(path, result)
    seal = stage_freeze(root, "v35_audit", [SOURCE, str(path.relative_to(root)),
                        "reports/V36_V35_ADJUDICATION_AUDIT.md", *EVIDENCE],
                        {"outcome": result["audit_outcome"], "summary_sha256": sha256_file(path)})
    return {"freeze_digest": seal["freeze_digest"], "audit_outcome": result["audit_outcome"]}


if __name__ == "__main__":
    print(json.dumps(run(Path.cwd()), indent=2))
