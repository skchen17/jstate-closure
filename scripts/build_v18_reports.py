#!/usr/bin/env python3
"""Render the frozen V18 result records into standalone and cumulative reports."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path.cwd()
OUT = ROOT / "results/v18/processed"
REPORTS = ROOT / "reports"


def data(name: str) -> dict:
    return json.loads((OUT / name).read_text(encoding="utf-8"))


def artifact(name: str) -> dict:
    return json.loads((ROOT / "artifacts" / name).read_text(encoding="utf-8"))


def number(value, places: int = 4) -> str:
    return "not measured" if value is None else f"{float(value):.{places}f}"


def emit(name: str, title: str, body: list[str]) -> None:
    path = REPORTS / name
    path.write_text("\n".join([f"# {title}", "", *body]).rstrip() + "\n", encoding="utf-8")


def table(headers: list[str], rows: list[list]) -> list[str]:
    return ["| " + " | ".join(headers) + " |",
            "|" + "|".join("---" for _ in headers) + "|",
            *("| " + " | ".join(str(value) for value in row) + " |" for row in rows), ""]


def main() -> None:
    split = artifact("strong_state_context_ceiling_v18_splits.freeze.json")
    base = artifact("strong_state_context_ceiling_v18.freeze.json")
    h1 = data("strong_h1_ceiling_v18.json")
    horizons = data("horizon_context_localization_v18.json")
    history = data("history_vs_snapshot_v18.json")
    matched = data("strict_matched_intervention_v18.json")
    decision = data("v18_adjudication.json")
    model = artifact("strong_state_context_ceiling_v18_models.freeze.json")
    bank = {role: data(f"crossed_bank_{role}_v18.json") for role in ("train", "validation")}
    correction = data("state_layer_amendment_2_complete_v18.json")
    teacher = data("teacher_token_drift_audit_v18.json")
    conditional_path = OUT / "strong_conditional_raw_context_v18.json"
    conditional = data(conditional_path.name) if conditional_path.exists() else None

    counts = []
    family_counts = []
    for role in ("train", "validation"):
        files = sorted(OUT.glob(f"crossed_response_{role}_*_v18.parquet"))
        frame = pd.concat([pd.read_parquet(path, columns=["family", "horizon", "alpha", "reliability_status",
                                                            "base_trial_id"]) for path in files], ignore_index=True)
        for (horizon, alpha), part in frame.groupby(["horizon", "alpha"]):
            counts.append([role, int(horizon), number(alpha, 1), part.base_trial_id.nunique(),
                           len(part), int((part.reliability_status == "RELIABLE").sum()),
                           number((part.reliability_status == "RELIABLE").mean())])
        for family, part in frame.groupby("family"):
            family_counts.append([role, family, part.base_trial_id.nunique(),
                                  len(part), number((part.reliability_status == "RELIABLE").mean())])
    emit("CROSSED_STATE_ACTION_BANK_V18.md", "V18 crossed state × action bank", [
        "Shared eight train-calibrated finite directions, two signs, primary α=0.5; the 400/80-state horizon panels additionally test α=1.0. Unreliable rows are retained with their status.",
        "", *table(["role", "h", "α", "states", "rows", "reliable", "fraction"], counts),
        "Family balance and reliable response coverage:",
        "", *table(["role", "family", "states", "rows", "reliable fraction"], family_counts),
        f"Train/validation state IDs are disjoint and split SHA256 values are `{split['train_id_sha256']}` / `{split['validation_id_sha256']}`.",
        f"Action coordinate indices: {[x['coordinate_index'] for x in split['actions']]}. Validation responses did not select actions.",
        "", "Pre-action current J and raw REC/Conv/KV come from V13 clean layer-23 records; action-response J is read at layer 30.",
        f"A state-layer correction archived {correction['pilot_state_count']} pilot metadata records; median/max live-to-frozen J relative L2 drift was {number(correction['median_live_current_j_v13_rel_l2'])}/{number(correction['maximum_live_current_j_v13_rel_l2'])}. The frozen V13 J is the canonical current-state value. Action responses were unchanged.",
        f"Fresh V18 clean-greedy teacher tokens match the shared frozen V13 prefix in {teacher['shared_prefix_exact_count']}/{teacher['state_count']} states and match the h1 token in {teacher['h1_token_exact_count']}/{teacher['state_count']}; later-token drift is recorded, not used as a filter.",
        "The single-GPU runtime amendment retained identical h1 response stacks for all 16 actions in a read-only equivalence check."
    ])

    primary = h1["primary_model"]
    h1rows = [[entry["model"], entry["context"], entry["parameter_count"],
               number(entry["metrics"]["j"]["relative_l2"]),
               number(entry["metrics"]["stacked_normalized"]["relative_l2"])]
              for entry in h1["model_results"]]
    primary_metrics = {(entry["model"], entry["context"]): entry["metrics"]
                       for entry in h1["model_results"]}
    target_rows = []
    for target in ("j", "logits", "semantic_continuous", "workspace", "stacked_normalized"):
        for context in ("j", "j_rec_conv_kv"):
            metric = primary_metrics[(primary, context)][target]
            target_rows.append([target, context, number(metric["relative_l2"]),
                                number(metric["direction_median"]),
                                number(metric["magnitude_ratio_median"]),
                                number(metric["component_sign_agreement"])])
    emit("STRONG_STATE_CONTEXT_CEILING_V18.md", "V18 strong h1 state-context ceiling", [
        "All conditions use a unified state × action predictor and an equal 128-dimensional state feature budget. M0–M4 are fit on train states, with train-only nested hyperparameter selection; validation is held out.",
        "", *table(["model", "context", "parameters", "J rel-L2", "stack rel-L2"], h1rows),
        "Primary model target-level validation metrics:",
        "", *table(["target", "context", "rel-L2", "direction median", "magnitude ratio", "sign agreement"], target_rows),
        f"Formal primary model: `{primary}`. Full clean REC+Conv+KV over J gives absolute J/stack relative-L2 gains **{number(h1['primary_joint_raw_gain_over_j']['j'])}/{number(h1['primary_joint_raw_gain_over_j']['stacked_normalized'])}**; material gate 0.05 passed: **{h1['material_h1_raw_context_gate_passed']}**.",
        f"Paired state bootstrap 95% CI (J/stack): {h1['primary_bootstrap']['j']['gain_ci95']} / {h1['primary_bootstrap']['stacked_normalized']['gain_ci95']}.",
        f"Family stack gains: {h1['primary_family_stack_gain']}.",
        "V17 reported J/stack gains 0.0126/0.0113 on its smaller linear-kernel reused-action cohort. V18 changes cohort size, shared-action crossing and model class, so the comparison is directional rather than a paired effect estimate.",
        "A small gain does not prove conditional independence or a complete Markov state."
    ])

    curve = horizons["horizons"]
    horizon_metrics = {(entry["horizon"], entry["context"]): entry["metrics"]
                       for entry in horizons["model_results"]}
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    x = [entry["horizon"] for entry in curve]
    for target, label, color in (("j", "J response", "#2563eb"),
                                 ("stacked_normalized", "Normalized response stack", "#d97706")):
        ax.plot(x, [entry["raw_joint_gain"][target] for entry in curve],
                marker="o", linewidth=2, label=label, color=color)
    ax.axhline(0.05, color="#6b7280", linestyle="--", linewidth=1, label="Frozen material threshold")
    ax.axhline(0, color="#9ca3af", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xlabel("Teacher-forced horizon")
    ax.set_ylabel("Absolute relative-L2 gain over J+action")
    ax.set_title("V18 clean raw-state context gain by horizon")
    ax.grid(alpha=0.2)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(REPORTS / "HORIZON_CONTEXT_GAIN_V18.png", dpi=180)
    plt.close(fig)
    emit("HORIZON_CONTEXT_LOCALIZATION_V18.md", "V18 horizon context localization", [
        "The same frozen state/action panel is teacher-forced to h=1,2,4,8. Gains below are absolute relative-L2 reduction from J+action to J+full clean raw+action.",
        "", "![V18 raw-context gain by horizon](HORIZON_CONTEXT_GAIN_V18.png)",
        "", *table(["h", "J-only J rel-L2", "full-raw J rel-L2", "J-only stack rel-L2", "full-raw stack rel-L2"],
                   [[x["horizon"],
                     number(horizon_metrics[(x["horizon"], "j")]["j"]["relative_l2"]),
                     number(horizon_metrics[(x["horizon"], "j_rec_conv_kv")]["j"]["relative_l2"]),
                     number(horizon_metrics[(x["horizon"], "j")]["stacked_normalized"]["relative_l2"]),
                     number(horizon_metrics[(x["horizon"], "j_rec_conv_kv")]["stacked_normalized"]["relative_l2"])]
                    for x in curve]),
        "", *table(["h", "J gain", "stack gain", "J CI95", "stack CI95", "material gate"],
                   [[x["horizon"], number(x["raw_joint_gain"]["j"]),
                     number(x["raw_joint_gain"]["stacked_normalized"]),
                     x["bootstrap"]["j"]["gain_ci95"],
                     x["bootstrap"]["stacked_normalized"]["gain_ci95"],
                     x["material_gate_passed"]] for x in curve]),
        f"Earliest horizon passing the pre-frozen joint target, bootstrap and family gate: **{horizons['earliest_material_horizon']}**.",
        "At long horizon, both models have high prediction error; absence of incremental raw gain here is not evidence that model memory is causally irrelevant. These are intervention-response prediction results, not autonomous transition evidence."
    ])

    channel_names = ["j_rec", "j_conv", "j_kv", "j_rec_conv", "j_rec_kv", "j_conv_kv", "j_rec_conv_kv"]
    emit("CONTEXT_CHANNEL_BY_HORIZON_V18.md", "V18 REC/Conv/KV context by horizon", [
        "Absolute normalized-stack relative-L2 gains over the same J+action baseline; negative values indicate worse validation prediction.",
        "", *table(["h", *channel_names],
                   [[x["horizon"], *(number(x["gains_over_j"][name]["stacked_normalized"])
                                     for name in channel_names)] for x in curve]),
        "Channel ranking is descriptive for these finite-action response targets and does not imply physical replacement."
    ])

    emit("HISTORY_VS_SNAPSHOT_V18.md", "V18 history versus current snapshot", [
        "Train-only PCA compresses 2- and 4-position pre-action J trajectories to the same 128-dimensional budget as current-state comparators. No prior intervention action exists at this control point.",
        "", *table(["h", "history positions", "history over current J", "raw over current J", "history after raw"],
                   [[x["horizon"], x["history_length"], number(x["history_gain_over_current_j"]),
                     number(x["raw_gain_over_current_j"]), number(x["history_incremental_gain_after_raw"])]
                    for x in history["gain_curves"]]),
        "Formal last-4-history incremental gain over the current J+raw snapshot:",
        "", *table(["h", "stack gain", "state-bootstrap CI95", "all-family nonnegative"],
                   [[h, number(decision["history_incremental_tests"][str(h)]["gain"]),
                     decision["history_incremental_tests"][str(h)]["bootstrap_ci95"],
                     all(value >= 0 for value in decision["history_incremental_tests"][str(h)]["family_gain"].values())]
                    for h in (1, 2, 4, 8)]),
        f"Material history gate: **{decision['history_material']}**. Full family-wise gains are preserved in `results/v18/processed/v18_adjudication.json`.",
        "A history advantage would indicate limits of this current-snapshot representation or model class, not uniquely missing physical memory fields."
    ])

    emit("STRICT_MATCHED_INTERVENTION_V18.md", "V18 strict matched intervention", [
        "Pairs are validation-only, same family/template, prompt length difference ≤4, exact action/sign/α=0.5, with a frozen near-J threshold 0.10. No threshold was relaxed.",
        "", *table(["pair type", "candidates", "one-to-one state pairs", "measured actions", "median response divergence"],
                   [[kind, row["candidate_state_pairs"], row["selected_one_to_one_state_pairs"],
                     row["measured_action_pairs"], number(row["median_response_relative_divergence"])]
                    for kind, row in matched["pair_types"].items()]),
        f"Status: **{matched['status']}**; formal positive divergence gate: **{decision['strict_match_positive']}**.",
        "This compares naturally different clean states under matched actions; it is not a direct raw-context swap."
    ])

    if conditional is None:
        conditional_body = ["Not run: the pre-frozen stronger raw-context material gate did not authorize a conditional residual fit.",
                            f"h1 material: {decision['material_h1_raw_context']}; material horizons: {decision['material_horizons']}."]
    else:
        conditional_body = ["Fivefold state-grouped out-of-fold target residuals and train-only raw residualization; the validation base prediction is held fixed.",
                            "", *table(["h", "J conditional gain", "stack conditional gain", "stack CI95"],
                                       [[x["horizon"], number(x["gains"]["j"]["conditional_raw_gain"]),
                                         number(x["gains"]["stacked_normalized"]["conditional_raw_gain"]),
                                         x["gains"]["stacked_normalized"]["bootstrap_ci95_and_median"]]
                                        for x in conditional["results"]]),
                            "A near-zero correction is not exact conditional-independence proof."]
    emit("STRONG_CONDITIONAL_RAW_CONTEXT_V18.md", "V18 strong conditional raw-context test", conditional_body)

    compact_status = decision["compact_search_status"]
    emit("COMPACT_CONTEXT_SUFFICIENCY_V18.md", "V18 compact context sufficiency", [
        f"Frozen authorization result: **{compact_status}**.",
        "No compact C dimension or sufficiency claim exists unless a material raw-response or strict matched-context gate is passed and a separate frozen sweep is completed."
    ])
    emit("DYNAMIC_STATE_SUFFICIENCY_V18.md", "V18 dynamic state sufficiency", [
        "No compact transition experiment or autonomous controller is claimed without a response-sufficient compact finalist and new independent confirmation.",
        f"Interventional Markov candidate: **{decision['interventional_markov_state_candidate']}**. Independent final: **{decision['independent_confirmation']}**."
    ])

    h1_lookup = {(row["model"], row["context"]): row["metrics"] for row in h1["model_results"]}
    h1_j_stack = h1_lookup[(primary, "j")]["stacked_normalized"]["relative_l2"]
    channel_h1 = {name: h1_j_stack - h1_lookup[(primary, f"j_{name}")]["stacked_normalized"]["relative_l2"]
                  for name in ("rec", "conv", "kv")}
    answers = [
        f"1. Strong nonlinear h1 full-raw gain over J+action: J **{number(h1['primary_joint_raw_gain_over_j']['j'])}**, normalized stack **{number(h1['primary_joint_raw_gain_over_j']['stacked_normalized'])}** absolute relative-L2; below the 0.05 gate.",
        "2. The larger 2000/400-state nonlinear ceiling did not turn V17's small raw gain into a material gain. This does not isolate sample size from model class because cohorts/protocols differ.",
        "3. Unified M0–M4 state×action models were fit; M4 found no material h1 full-raw increment over J. The 50-condition matrix is in the ceiling report.",
        f"4. Individual h1 stack gains REC/Conv/KV: **{number(channel_h1['rec'])}/{number(channel_h1['conv'])}/{number(channel_h1['kv'])}**; Conv is largest in this small diagnostic contrast, not a stable causal channel ranking.",
        f"5. Full-raw stack gains by h1/h2/h4/h8: **{', '.join(number(x['raw_joint_gain']['stacked_normalized']) for x in curve)}**; no monotone material increase.",
        "6. No tested horizon passed the frozen material gate.",
        "7. J+action is a strong h1 predictor in this class, but near-context-sufficiency is not formally identified because strict same-J matching found no qualifying pairs.",
        "8. The tested raw persistent context did not show a material long-horizon increment; high h8 error leaves memory and representation limitations unresolved.",
        "9. Two/four-position pre-action J history did not add material predictive gain beyond current raw snapshot.",
        "10. Current-snapshot Markov sufficiency is not established by these predictive comparisons.",
        "11. Strict same-J/different-raw and controls had zero qualifying state pairs; `MATCH_NOT_IDENTIFIED`.",
        f"12. Five-family h1 full-raw stack gains: {h1['primary_family_stack_gain']}; none yields a pooled material gate.",
        f"13. Compact C search authorization: **{compact_status}**.",
        "14. No sufficient compact C dimension was identified because the search was gated off.",
        "15. J+C+action conditional raw residual gain was not measured; no authorized C exists.",
        "16. Unseen-action generalization of C was not measured; no selected C exists.",
        "17. Cross-horizon stability of C was not measured; no selected C exists.",
        "18. Next-C transition prediction was not run; no selected C exists.",
        "19. Raw incremental next-J/next-C transition gain was not run; no compact transition finalist exists.",
        "20. No Interventional Markov State candidate was identified.",
        "21. H2 remains the project-level status.",
        "22. H3 candidate state is not supported.",
        "23. Autonomous state-model/controller training is not authorized.",
        "", f"Independent confirmation: **{decision['independent_confirmation']}**; no development result is relabeled as independent final."
    ]
    emit("V18_SCIENTIFIC_ANSWERS_V18.md", "V18 answers to the 23 scientific questions", answers)

    amendments = [artifact(f"strong_state_context_ceiling_v18_{name}.freeze.json")
                  for name in ("action_selection_amendment_1", "state_layer_amendment_2",
                               "reference_alignment_amendment_3", "runtime_placement_amendment_4",
                               "teacher_drift_amendment_5", "model_binding_amendment_6",
                               "history_array_amendment_7")]
    emit("STRICT_INTERFACE_AUDIT_V18.md", "V18 strict interface and leakage audit", [
        f"Parent immutable commit: `{base['parent_commit']}`. V18 base protocol freeze: `{base['freeze_digest']}`; split freeze: `{split['freeze_digest']}`; model freeze: `{model['freeze_digest']}`.",
        "", *table(["amendment", "freeze digest", "reason"],
                   [[x["protocol_version"], x["freeze_digest"], x["reason"]] for x in amendments]),
        "V1–V17 frozen files were not modified. V18 train/validation source states are disjoint and exclude V16 development IDs. State kernels, Nyström landmarks, normalization, hyperparameter choices and OOF residuals use train only. Validation never selected actions or thresholds. Independent final remains unopened unless a finalist qualifies.",
        "The canonical V18 current J is V13 clean layer 23. Response J is V16-style layer 30. All 2000 train and 400 validation current-J vectors were checked to match the V13 frozen clean-J memmap elementwise (maximum absolute difference 0). The 401 prior pilot state records remain archived and indexed. Runtime placement was amended after exact same-state 16-action h1 equivalence verification. The model-binding amendment changed only summary lookup; archived and final h1 prediction/selection files have identical SHA-256 values. The history-array amendment changed only nested-array decoding before any history fit. V18 clean-greedy teacher-token drift from V13 is separately audited."
    ])

    commands = [
        "PYTHONPATH=src /home/user/anaconda3/bin/python -m jclosure.protocol_v18 freeze",
        "PYTHONPATH=src /home/user/anaconda3/bin/python -m jclosure.experiments.crossed_bank_v18 amend_actions",
        "PYTHONPATH=src /home/user/anaconda3/bin/python -m jclosure.experiments.crossed_bank_v18 prepare",
        "PYTHONPATH=src /home/user/anaconda3/bin/python -m jclosure.experiments.state_features_v18 extract",
        "PYTHONPATH=src /home/user/anaconda3/bin/python -m jclosure.experiments.state_features_v18 kernels",
        "PYTHONPATH=src /home/user/anaconda3/bin/python -m jclosure.experiments.state_features_v18 scores",
        "PYTHONPATH=src /home/user/anaconda3/bin/python -m jclosure.experiments.crossed_bank_v18 amend_state_layer",
        "PYTHONPATH=src /home/user/anaconda3/bin/python -m jclosure.experiments.crossed_bank_v18 amend_reference_alignment",
        "PYTHONPATH=src /home/user/anaconda3/bin/python -m jclosure.experiments.crossed_bank_v18 repair_train",
        "PYTHONPATH=src /home/user/anaconda3/bin/python scripts/build_v18_amendment_archive.py",
        "PYTHONPATH=src /home/user/anaconda3/bin/python -m jclosure.experiments.crossed_bank_v18 amend_runtime_placement",
        "PYTHONPATH=src /home/user/anaconda3/bin/python -m jclosure.experiments.crossed_bank_v18 amend_teacher_drift",
        "PYTHONPATH=src /home/user/anaconda3/bin/python -m jclosure.experiments.crossed_bank_v18 train",
        "PYTHONPATH=src /home/user/anaconda3/bin/python -m jclosure.experiments.crossed_bank_v18 validation",
        "PYTHONPATH=src /home/user/anaconda3/bin/python -m jclosure.experiments.crossed_bank_v18 aggregate_train",
        "PYTHONPATH=src /home/user/anaconda3/bin/python -m jclosure.experiments.crossed_bank_v18 aggregate_validation",
        "PYTHONPATH=src /home/user/anaconda3/bin/python -m jclosure.experiments.crossed_bank_v18 audit_teacher",
        "PYTHONPATH=src /home/user/anaconda3/bin/python -m jclosure.experiments.strong_ceiling_v18 amend_model_binding",
        "PYTHONPATH=src /home/user/anaconda3/bin/python -m jclosure.experiments.strong_ceiling_v18 h1",
        "PYTHONPATH=src /home/user/anaconda3/bin/python -m jclosure.experiments.strong_ceiling_v18 horizons",
        "PYTHONPATH=src /home/user/anaconda3/bin/python -m jclosure.experiments.history_v18 amend_history_array",
        "PYTHONPATH=src /home/user/anaconda3/bin/python -m jclosure.experiments.history_v18 scores",
        "PYTHONPATH=src /home/user/anaconda3/bin/python -m jclosure.experiments.history_v18 evaluate",
        "PYTHONPATH=src /home/user/anaconda3/bin/python -m jclosure.experiments.strict_match_v18 run",
        "PYTHONPATH=src /home/user/anaconda3/bin/python -m jclosure.experiments.decision_v18 run",
        "PYTHONPATH=src /home/user/anaconda3/bin/python -m pytest tests/test_v18.py -q",
        "PYTHONPATH=src /home/user/anaconda3/bin/python scripts/build_v18_reports.py",
        "PYTHONPATH=src /home/user/anaconda3/bin/python scripts/build_v18_integrity.py",
        "PYTHONPATH=src /home/user/anaconda3/bin/python scripts/build_complete_version_report.py V18",
    ]
    if conditional is not None:
        commands.insert(-4, "PYTHONPATH=src /home/user/anaconda3/bin/python -m jclosure.experiments.conditional_raw_v18 run")
    emit("EXECUTION_MANIFEST_V18.md", "V18 execution manifest", [
        "Working directory: `/data/CSK/J-space-project/jstate-closure`; model cache: `/data/CSK/J-space-project/.hf-cache`. Commands below record the core stages (with `HF_HOME` set to that model cache for model-loading stages). Freeze-creation commands are historical and intentionally fail if rerun on the same already-frozen workspace.",
        "", "```bash", *commands, "```", "",
        f"Bank roles: train {bank['train']['selected_states']} states, validation {bank['validation']['selected_states']} states; new independent final remains unopened.",
        f"Frozen action coordinates: {[x['coordinate_index'] for x in split['actions']]}; protocol/split/model hashes: `{base['freeze_digest']}` / `{split['freeze_digest']}` / `{model['freeze_digest']}`.",
        f"Formal result: **{decision['formal_outcome']}**. Compact search: **{compact_status}**."
    ])

    summary = [
        "<!-- V18_START -->",
        "## V18 — Strong State-Context Ceiling and Horizon Localization",
        "",
        f"Formal outcome: **{decision['formal_outcome']}**. The new crossed bank contains {bank['train']['selected_states']} train and {bank['validation']['selected_states']} validation states under eight shared signed finite action coordinates; h1/h2/h4/h8 use a fixed 400/80-state panel. The unified nonlinear h1 full-raw versus J-only absolute J/stack relative-L2 gains are **{number(h1['primary_joint_raw_gain_over_j']['j'])}/{number(h1['primary_joint_raw_gain_over_j']['stacked_normalized'])}** (frozen material gate 0.05; passed: {h1['material_h1_raw_context_gate_passed']}).",
        f"Full-raw stack gain by horizon h1/h2/h4/h8: **{', '.join(number(x['raw_joint_gain']['stacked_normalized']) for x in curve)}**. Earliest material horizon: **{horizons['earliest_material_horizon']}**. Strict matching: **{matched['status']}**; history material: **{decision['history_material']}**. Compact search: **{compact_status}**. Independent final: **{decision['independent_confirmation']}**.",
        "H2 remains. H3 candidate state and autonomous state-model training are not authorized without a response-sufficient compact state and independent V18-F confirmation. These are finite-action response results, not proof of a complete state or physical replacement. V1–V17 frozen records are unchanged. See `reports/V18_COMPLETE_REPORT.md` for all standalone reports and machine-record integrity index.",
        "<!-- V18_END -->",
    ]
    final = REPORTS / "FINAL_REPORT.md"
    existing = final.read_text(encoding="utf-8")
    if "<!-- V18_START -->" in existing:
        frozen_section = existing.split("<!-- V18_START -->", 1)[1].split("<!-- V18_END -->", 1)[0].strip()
        refreshed_section = "\n".join(summary[1:-1]).strip()
        if frozen_section != refreshed_section:
            raise RuntimeError("V18 cumulative section differs; refuse silent rewrite")
    else:
        final.write_text(existing.rstrip() + "\n\n" + "\n".join(summary) + "\n", encoding="utf-8")
    emit("V18_REPORT_INDEX_V18.md", "V18 report index", [
        "The canonical single-file bundle is `reports/V18_COMPLETE_REPORT.md`; all V18 standalone reports and processed machine records are indexed there."
    ])
    print(json.dumps({"reports_written": len(list(REPORTS.glob("*_V18.md"))),
                      "formal_outcome": decision["formal_outcome"],
                      "compact_search": compact_status}, indent=2))


if __name__ == "__main__":
    main()
