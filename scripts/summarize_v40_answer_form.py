"""Seal V40 calibration answer-form audit and generate its report from records."""
from __future__ import annotations

import json
from pathlib import Path

from jclosure.protocol_v40_a3 import stage_freeze, verify
from jclosure.provenance import sha256_file, write_json_atomic

SOURCE = "scripts/summarize_v40_answer_form.py"


def run(root: Path) -> dict:
    verify(root)
    inputs = {}
    totals = {}
    for key in ("Q", "F"):
        path = root / f"results/v40/processed/answer_form_audit_{key}_v40.json"
        value = json.loads(path.read_text(encoding="utf-8"))
        if value["model"] != key or len(value["rows"]) != 32 or not value["development_validation_final_unopened"]:
            raise RuntimeError(f"V40 answer-form source invalid: {key}")
        if value["source_sha256"] != sha256_file(root / "scripts/audit_v40_answer_form.py"):
            raise RuntimeError("V40 answer-form code drift")
        inputs[key] = {"path": str(path.relative_to(root)), "sha256": sha256_file(path)}
        totals[key] = {
            form: sum(value["families"][family][form]["both_correct"]
                      for family in value["families"])
            for form in ("direct", "after_one_space")}
        totals[key]["families"] = value["families"]
    result = {"status": "CALIBRATION_ONLY_ANSWER_FORM_ADJUDICATION",
              "input_files": inputs, "totals": totals,
              "formal_readout_not_yet_frozen": True,
              "development_validation_final_unopened": True}
    out = root / "results/v40/processed/answer_form_adjudication_v40.json"
    if out.exists():
        raise RuntimeError("V40 answer-form adjudication already exists")
    write_json_atomic(out, result)
    lines = ["# V40 — Answer-Form Calibration Audit", "",
             "Calibration only. The two readouts compare answer logits immediately after",
             "the prompt versus after one natural space token. All numbers below are",
             "computed from saved calibration records; no development or held-out outcomes.", "",
             "| Model | Family | Both native branches correct: direct | After one space |",
             "|---|---|---:|---:|"]
    for key in ("Q", "F"):
        for family, forms in sorted(totals[key]["families"].items()):
            lines.append(f"| {key} | {family} | {forms['direct']['both_correct']} | "
                         f"{forms['after_one_space']['both_correct']} |")
    lines.extend(["", "| Model | Total direct | Total after one space |",
                  "|---|---:|---:|"])
    for key in ("Q", "F"):
        lines.append(f"| {key} | {totals[key]['direct']} | {totals[key]['after_one_space']} |")
    lines.extend(["", "Answer formatting changes apparent native competence. Falcon's",
                  "Boolean family improves after a space, but its other families have no",
                  "both-correct calibration pairs under either readout. This is a scoring",
                  "diagnostic, not evidence for a channel role or a model deficit in general.",
                  "The formal answer readout and control design must be frozen before",
                  "development outcomes are observed.", ""])
    report = root / "reports/V40_ANSWER_FORM_AUDIT.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    stage_freeze(root, "answer_form_adjudication",
                 [SOURCE, "scripts/audit_v40_answer_form.py",
                  str(out.relative_to(root)), str(report.relative_to(root)),
                  *[inputs[key]["path"] for key in ("Q", "F")]],
                 {"adjudication_sha256": sha256_file(out),
                  "report_sha256": sha256_file(report),
                  "calibration_only": True})
    return result


if __name__ == "__main__":
    print(json.dumps(run(Path.cwd()), indent=2))
