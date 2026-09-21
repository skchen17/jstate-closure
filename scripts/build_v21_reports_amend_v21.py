"""Append-only in-memory correction for two V21 reporting-only expressions."""

from __future__ import annotations

import json
from pathlib import Path

from jclosure.protocol_v21 import stage_freeze,verify_stage

SOURCE="scripts/build_v21_reports_amend_v21.py"
FROZEN="scripts/build_v21_reports.py"


def prepare(root:Path)->dict:
    verify_stage(root,"reporting_design")
    return stage_freeze(root,"reporting_source_amendment_1",
                        [SOURCE,FROZEN,"artifacts/action_coordinate_geometry_v21_reporting_design.freeze.json"],
                        {"reason":"Pre-execution review found five undefined local metric aliases in scientific-answer/final-section formatting and a displayed 'minimum cosine' that selected row zero; execute the frozen source in memory with only those reporting expressions corrected.",
                         "machine_results_or_scientific_gates_changed":False,
                         "reports_generated_before_amendment":0,
                         "final_report_V21_section_appended_before_amendment":False,
                         "final_six_action_responses_opened":False})


def run(root:Path)->dict:
    verify_stage(root,"reporting_source_amendment_1")
    text=(root/FROZEN).read_text(encoding="utf-8")
    anchor='    s4z1=model("S4","Z1",g2)\n'
    insertion=(anchor+
        '    z0=s2z0["metrics"]["unseen_direction"]["stack_relative_l2"]\n'
        '    z1=s2z1["metrics"]["unseen_direction"]["stack_relative_l2"]\n'
        '    z2=s2z2["metrics"]["unseen_direction"]["stack_relative_l2"]\n'
        '    s2=z1\n'
        '    s4=s4z1["metrics"]["unseen_direction"]["stack_relative_l2"]\n')
    if text.count(anchor)!=1:
        raise RuntimeError("V21 report alias amendment anchor drift")
    text=text.replace(anchor,insertion)
    old="{f(read('forward_ad_comparison_v21.json')['rows'][0]['cosine'],6)}"
    new="{f(min(x['cosine'] for x in read('forward_ad_comparison_v21.json')['rows']),6)}"
    if text.count(old)!=1:
        raise RuntimeError("V21 report minimum-cosine amendment anchor drift")
    text=text.replace(old,new)
    namespace={"__name__":"v21_report_amended","__file__":str(root/FROZEN)}
    exec(compile(text,str(root/FROZEN),"exec"),namespace)
    return namespace["build"](root)


if __name__=="__main__":
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument("stage",choices=("prepare","run"))
    args=parser.parse_args()
    root=Path.cwd()
    print(json.dumps(prepare(root) if args.stage=="prepare" else run(root),indent=2))
