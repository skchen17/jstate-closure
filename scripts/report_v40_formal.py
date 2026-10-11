"""Generate V40 topic and complete reports strictly from sealed machine records."""
from __future__ import annotations

import json
from pathlib import Path

from jclosure.protocol_v40_a4 import verify
from jclosure.provenance import sha256_file, write_json_atomic
from scripts.build_v40_all_reports import build as build_all

SOURCE = "scripts/report_v40_formal.py"
ROLES = ("development", "validation", "independent_final")
CHANNELS = ("BASE", "REC_only", "Conv_only", "KV_only",
            "REC_Conv", "REC_Conv_KV", "DONOR_NATIVE")


def read_if(root: Path, pattern: str) -> dict | None:
    path = root / pattern
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def fraction(n: int | None, d: int | None) -> str:
    return f"{n}/{d}" if n is not None and d is not None and d else "—"


def ci(metric: dict | None) -> str:
    if not metric or metric["ci95"] is None:
        return "—"
    return f"{metric['mean']:.3f} [{metric['ci95'][0]:.3f}, {metric['ci95'][1]:.3f}]"


def write(root: Path, name: str, lines: list[str]) -> str:
    path = root / "reports" / name
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return str(path.relative_to(root))


def run(root: Path) -> dict:
    cfg = verify(root)["config"]
    pool_roles = ("calibration", "development", "validation", "independent_final")
    pools = {role: json.loads((root / f"data/v40/{role}_pool_v40.json").read_text())["items"]
             for role in pool_roles}
    pool_ids = {role: {item["state_id"] for item in items} for role, items in pools.items()}
    if any(pool_ids[left] & pool_ids[right]
           for index, left in enumerate(pool_roles) for right in pool_roles[index + 1:]):
        raise RuntimeError("V40 report pool overlap")
    functional = {(key, role): read_if(root,
        f"results/v40/processed/analysis_functional_{key}_{role}_v40.json")
        for key in ("Q", "F") for role in ROLES}
    sensitivity = {(key, role): read_if(root,
        f"results/v40/processed/split_sensitivity_{key}_{role}_v40.json")
        for key in ("Q", "F") for role in ROLES}
    candidate_vector = {(key, role): read_if(root,
        f"results/v40/processed/candidate_vector_{key}_{role}_v40.json")
        for key in ("Q", "F") for role in ROLES}
    temporal = {role: read_if(root,
        f"results/v40/processed/analysis_temporal_Q_{role}_v40.json")
        for role in ROLES}
    temporal_family = {role: read_if(root,
        f"results/v40/processed/temporal_family_Q_{role}_v40.json")
        for role in ROLES}
    trajectory = {role: read_if(root,
        f"results/v40/processed/trajectory_analysis_Q_{role}_v40.json")
        for role in ROLES}
    opening = read_if(root, "results/v40/processed/final_opening_v40.json")
    complete = (opening is not None and all(functional.values())
                and all(sensitivity.values()) and all(candidate_vector.values())
                and all(temporal.values()) and all(temporal_family.values())
                and all(trajectory.values()))
    inputs = [f"results/v40/processed/analysis_functional_{key}_{role}_v40.json"
              for key in ("Q", "F") for role in ROLES if functional[key, role]]
    inputs.extend(f"results/v40/processed/split_sensitivity_{key}_{role}_v40.json"
                  for key in ("Q", "F") for role in ROLES if sensitivity[key, role])
    inputs.extend(f"results/v40/processed/candidate_vector_{key}_{role}_v40.json"
                  for key in ("Q", "F") for role in ROLES if candidate_vector[key, role])
    inputs.extend(f"results/v40/processed/analysis_temporal_Q_{role}_v40.json"
                  for role in ROLES if temporal[role])
    inputs.extend(f"results/v40/processed/temporal_family_Q_{role}_v40.json"
                  for role in ROLES if temporal_family[role])
    inputs.extend(f"results/v40/processed/trajectory_analysis_Q_{role}_v40.json"
                  for role in ROLES if trajectory[role])
    if opening:
        inputs.append("results/v40/processed/final_opening_v40.json")

    channel_lines = ["# V40 — Channel Function Profile", "",
        "The primary readout is the externally generated task answer after one space token.",
        "Each row uses all sealed task states and separately reports the native-pair-qualified subset.",
        "Calibration and V39 outcomes are not counted below.", "",
        "| Model | Role | Complete / sealed | Native qualified | Condition | Donor recovered / all sealed | Donor recovered / qualified | Recovery Δ vs BASE, bootstrap 95% CI |",
        "|---|---|---:|---:|---|---:|---:|---:|"]
    for role in ROLES:
        for key in ("Q", "F"):
            data = functional[key, role]
            if data is None:
                continue
            for condition in CHANNELS:
                item = data["channels"][condition]
                channel_lines.append(f"| {key} | {role} | {fraction(data['complete_states'], data['sealed_states'])} | "
                    f"{data['qualified_states']} | {condition} | "
                    f"{fraction(item['donor_recovered_all_states'], item['all_state_denominator'])} | "
                    f"{fraction(item['donor_recovered_qualified'], item['qualified_denominator'])} | "
                    f"{ci(item['recovery_delta_vs_base'])} |")
    channel_lines.extend(["", "REC+Conv+KV reconstructs the complete native donor cache and serves as a positive control; its recovery is not independent evidence of channel specialization.",
                          "REC+Conv versus Conv-only is a conditional contrast, not an independent REC-storage proof.",
                          "The source-fork and shuffled-state controls are summarized in the task-grounded report.", ""])
    channel_lines.extend(["", "## Secondary answer-alphabet logit-vector fidelity", "",
        "This exploratory operationalization computes `1 − distance(condition, native donor) / distance(BASE, native donor)` after mean-centering the frozen answer-alphabet logits. Zero is BASE and one is the native donor; it is **not** full-vocabulary or hidden-state fidelity and does not establish semantic storage.", "",
        "| Model | Role | Identifiable / native qualified | REC mean [95% CI] | Conv mean [95% CI] | KV mean [95% CI] | REC+Conv mean [95% CI] |",
        "|---|---|---:|---:|---:|---:|---:|"])
    for role in ROLES:
        for key in ("Q", "F"):
            data = candidate_vector[key, role]
            if data is None:
                continue
            cells = [ci(data["channels"][name]["mean_gap_closed"])
                     for name in ("REC_only", "Conv_only", "KV_only", "REC_Conv")]
            channel_lines.append(f"| {key} | {role} | "
                f"{fraction(data['identifiable_qualified_states'], data['native_qualified_states'])} | "
                + " | ".join(cells) + " |")
    channel_lines.append("")
    p_channel = write(root, "V40_CHANNEL_FUNCTION_PROFILE.md", channel_lines)

    task_lines = ["# V40 — Task-Grounded Results", "",
        "Labels were produced by the sealed task generator before model execution. The state is the independent unit.",
        "The frozen endpoint is constrained one-token answer selection over each family's finite alphabet, not unrestricted text generation.",
        "An intervention is not called semantically successful from cosine similarity alone.", "",
        "| Model | Role | Sealed | Complete | Native qualified | BASE donor recovery | REC+Conv donor recovery | Native donor recovery |",
        "|---|---|---:|---:|---:|---:|---:|---:|"]
    for role in ROLES:
        for key in ("Q", "F"):
            data = functional[key, role]
            if data is None:
                continue
            channel = data["channels"]
            task_lines.append(f"| {key} | {role} | {data['sealed_states']} | {data['complete_states']} | "
                f"{data['qualified_states']} | "
                f"{fraction(channel['BASE']['donor_recovered_qualified'], data['qualified_states'])} | "
                f"{fraction(channel['REC_Conv']['donor_recovered_qualified'], data['qualified_states'])} | "
                f"{fraction(channel['DONOR_NATIVE']['donor_recovered_qualified'], data['qualified_states'])} |")
    for role in ROLES:
        data = functional["Q", role]
        if data is None:
            continue
        task_lines.extend(["", f"## Qwen {role} by task family", "",
            "| Family | Sealed | Qualified | REC | Conv | KV | REC+Conv |",
            "|---|---:|---:|---:|---:|---:|---:|"])
        for family, family_data in sorted(data["families"].items()):
            recovered = family_data["donor_recovered"]
            task_lines.append(f"| {family} | {family_data['sealed_states']} | {family_data['qualified_states']} | "
                f"{recovered['REC_only']} | {recovered['Conv_only']} | {recovered['KV_only']} | {recovered['REC_Conv']} |")
    for key in ("Q", "F"):
        data = functional[key, "independent_final"] or functional[key, "validation"] or functional[key, "development"]
        if data is None:
            continue
        task_lines.extend(["", f"## {key} controls ({data['role']})", "",
            "| Control | Valid rows | Unavailable | Task-correct rows | Paired-donor rows |",
            "|---|---:|---:|---:|---:|"])
        for kind, item in data["controls"].items():
            task_lines.append(f"| {kind} | {item['valid_rows']} | {item['unavailable_rows']} | "
                f"{item['correct_task_rows']} | {item['paired_donor_recovery_rows']} |")
        task_lines.extend(["", "Paired REC+Conv donor-recovery advantage over matched negative controls (state bootstrap):"])
        for kind, metric in data["recconv_recovery_vs_matched_controls"].items():
            task_lines.append(f"- {kind}: {ci(metric)} (n={metric['n']}).")
        audit = data["split_full"]
        task_lines.append(f"\nSplit/full constrained-answer agreement: {fraction(audit['candidate_prediction_agrees'], audit['audits'])}; "
                          f"maximum full-vocabulary logit difference: {audit['max_full_vocab_abs']:.4f}.")
    task_lines.extend(["", "For surface/history controls the donor and recipient external answers are identical, so their paired-donor column is merely unchanged-label correctness, not a transfer effect.",
                       "Controls are diagnostics on frozen natural states; absent baseline competence bounds semantic interpretation.", ""])
    task_lines.extend(["", "## Native split/full prediction sensitivity", "",
        "The primary analysis is unchanged; the strict subset excludes any state with a native BASE or DONOR_NATIVE constrained-answer disagreement.", "",
        "| Model | Role | Split-discordant states | Original qualified | Strict qualified | Strict Qwen gate | Strict REC+Conv recovery |",
        "|---|---|---:|---:|---:|---|---:|"])
    for role in ROLES:
        for key in ("Q", "F"):
            result = sensitivity[key, role]
            if result is None:
                continue
            recovered = result["strict_donor_recovery"]["REC_Conv"]
            task_lines.append(f"| {key} | {role} | {len(result['mismatched_state_ids'])} | "
                f"{result['original_qualified_states']} | {result['strict_qualified_states']} | "
                f"{result['strict_qwen_interpretability_gate']} | "
                f"{fraction(recovered['recovered'], recovered['n'])} |")
    p_task = write(root, "V40_TASK_GROUNDED_RESULTS.md", task_lines)

    temporal_lines = ["# V40 — Temporal Persistence", "",
        "Qwen was primary. Each delay inserts fixed shared neutral tokens before the common task suffix.",
        "Effects are empirical; no exponential decay law is assumed.", "",
        "| Role | Delay | Complete / sealed | Native qualified | REC recovery (all; qualified) | Conv recovery (all; qualified) | KV recovery (all; qualified) | REC+Conv recovery (all; qualified) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for role in ROLES:
        data = temporal[role]
        if data is None:
            continue
        for delay, point in sorted(data["profiles"].items(), key=lambda x: int(x[0])):
            ch = point["channels"]
            cells = [f"{fraction(ch[name]['donor_recovered_all_states'], ch[name]['all_state_denominator'])}; "
                     f"{fraction(ch[name]['qualified_recovery'], point['qualified_states'])}"
                     for name in ("REC_only", "Conv_only", "KV_only", "REC_Conv")]
            temporal_lines.append(f"| {role} | {delay} | "
                f"{fraction(point['complete_states'], data['sealed_states'])} | "
                f"{point['qualified_states']} | "
                + " | ".join(cells) + " |")
    if temporal["independent_final"]:
        temporal_lines.extend(["", "Independent-final median answer-margin effect as a fraction of the contemporaneous native-donor ceiling:", "",
            "| Delay | REC | Conv | KV | REC+Conv |",
            "|---:|---:|---:|---:|---:|"])
        for delay, point in sorted(temporal["independent_final"]["profiles"].items(), key=lambda x: int(x[0])):
            ch = point["channels"]
            values = [ch[name]["median_fraction_of_native_ceiling"]
                      for name in ("REC_only", "Conv_only", "KV_only", "REC_Conv")]
            temporal_lines.append(f"| {delay} | " + " | ".join(
                f"{value:.3f}" if value is not None else "—" for value in values) + " |")
    temporal_lines.extend(["", "Observed-grid half-crossing uses only states native-qualified at every delay and requires at least 16 such states.", "",
        "| Role | Common qualified | Channel | First persistent half-crossing |",
        "|---|---:|---|---:|"])
    for role in ROLES:
        data = temporal[role]
        if data is None:
            continue
        for channel, result in data["observed_grid_half_crossings"].items():
            crossing = result["observed_grid_half_crossing"]
            temporal_lines.append(f"| {role} | {data['common_qualified_states']} | {channel} | "
                f"{crossing if crossing is not None else 'not identifiable'} |")
    temporal_lines.extend(["", "A grid crossing is not a continuous or universal channel half-life.", ""])
    family_final = temporal_family["independent_final"]
    if family_final:
        temporal_lines.extend(["", "## Independent-final task-family profiles (descriptive appendix)", "",
            "Each cell uses the native-qualified states of that family at that delay; denominators may change with delay. These subgroup comparisons are exploratory.", "",
            "| Family | Delay | Qualified / sealed | REC | Conv | KV | REC+Conv |",
            "|---|---:|---:|---:|---:|---:|---:|"])
        for family, group in sorted(family_final["families"].items()):
            for delay, point in sorted(group["profiles"].items(), key=lambda item: int(item[0])):
                channel = point["channels"]
                cells = [fraction(channel[name]["donor_recovered_qualified"],
                                  channel[name]["qualified_denominator"])
                         for name in ("REC_only", "Conv_only", "KV_only", "REC_Conv")]
                temporal_lines.append(f"| {family} | {delay} | "
                    f"{fraction(point['native_qualified_states'], point['sealed_states'])} | "
                    + " | ".join(cells) + " |")
        temporal_lines.extend(["", "Bootstrap margin estimates for every family, delay, and channel are in the sealed machine appendix.", ""])
    p_temporal = write(root, "V40_TEMPORAL_PERSISTENCE.md", temporal_lines)

    comparison_lines = ["# V40 — Channel Comparison", "",
        "Qwen is the primary model; Falcon is a descriptive comparison when its paired native-answer baseline is insufficient.",
        "Identical channel roles were never required.", "",
        "| Role | Qwen qualified | Qwen gate | Falcon qualified | Falcon scope |",
        "|---|---:|---|---:|---|"]
    for role in ROLES:
        q, f = functional["Q", role], functional["F", role]
        if q is None or f is None:
            continue
        comparison_lines.append(f"| {role} | {fraction(q['qualified_states'], q['sealed_states'])} | "
            f"{q['baseline_qualification_gate']} | {fraction(f['qualified_states'], f['sealed_states'])} | "
            "descriptive only; no preregistered Falcon gate |")
    comparison_lines.extend(["", "A weak Falcon baseline here is an assay limitation, not a general claim about Falcon ability or its channel ontology.", ""])
    p_comparison = write(root, "V40_CHANNEL_COMPARISON.md", comparison_lines)

    trajectory_lines = ["# V40 — Trajectory Causality", "",
        "Qwen early/middle/late REC groups are fixed thirds of its 24 recurrent layers.",
        "Each factorial condition starts from recipient REC/KV with donor Conv as a common background; all future computation runs naturally.",
        "The primary three-way contrast is on the externally labelled donor-versus-recipient answer margin.", "",
        "| Role | Delay | Complete / sealed | Qualified | R000 donor recovery (all; qualified) | R111 donor recovery (all; qualified) | Three-way mean [95% CI] | Gate |",
        "|---|---:|---:|---:|---:|---:|---:|---|"]
    for role in ROLES:
        data = trajectory[role]
        if data is None:
            continue
        for delay, point in sorted(data["delays"].items(), key=lambda x: int(x[0])):
            ch = point["conditions"]
            r000 = f"{fraction(ch['R000']['donor_recovered_all_complete'], ch['R000']['all_state_denominator'])}; " + fraction(ch["R000"]["qualified_donor_recovered"], point["qualified_states"])
            r111 = f"{fraction(ch['R111']['donor_recovered_all_complete'], ch['R111']['all_state_denominator'])}; " + fraction(ch["R111"]["qualified_donor_recovered"], point["qualified_states"])
            trajectory_lines.append(f"| {role} | {delay} | "
                f"{fraction(point['complete_states'], data['sealed_states'])} | "
                f"{point['qualified_states']} | "
                f"{r000} | {r111} | "
                f"{ci(point['three_way_margin_interaction'])} | {point['task_grounded_three_way_gate']} |")
    trajectory_final = trajectory["independent_final"]
    if trajectory_final:
        trajectory_lines.extend(["", "## Independent-final task-family interaction", "",
            "| Delay | Family | Qualified states | Median three-way answer-margin interaction |",
            "|---:|---|---:|---:|"])
        for delay, point in sorted(trajectory_final["delays"].items(), key=lambda item: int(item[0])):
            for family, entry in sorted(point["family_interactions"].items()):
                value = entry["median_three_way_margin"]
                display = f"{value:.3f}" if value is not None else "—"
                trajectory_lines.append(f"| {delay} | {family} | {entry['qualified_states']} | "
                                        f"{display} |")
        trajectory_lines.extend(["", "## Final-answer hidden geometry at the last recurrent checkpoint", "",
            "| Delay | Condition | Median cosine to native donor |",
            "|---:|---|---:|"])
        for delay, point in sorted(trajectory_final["delays"].items(), key=lambda item: int(item[0])):
            reference = point["conditions"]["R000"]["median_hidden_cosine_to_donor"]
            candidates = [name for name in reference if name.startswith("final_answer_space:")]
            late = max(candidates, key=lambda name: int(name.split(":")[1])) if candidates else None
            for condition in ("R000", "R100", "R010", "R001", "R111"):
                value = point["conditions"][condition]["median_hidden_cosine_to_donor"].get(late)
                trajectory_lines.append(f"| {delay} | {condition} | "
                    + (f"{value:.4f}" if value is not None else "—") + " |")
    trajectory_lines.extend(["", "Hidden-state cosine summaries and checkpoint REC/Conv norms are in the sealed machine records.",
        "Donor-like hidden geometry without task recovery is not task-grounded mediation.",
        "This factorial tests depth composition on one channel background; it does not identify a unique primitive or all token-time mechanisms.", ""])
    p_trajectory = write(root, "V40_TRAJECTORY_CAUSALITY.md", trajectory_lines)

    q_final = functional["Q", "independent_final"]
    q_temporal_final = temporal["independent_final"]
    q_traj_final = trajectory["independent_final"]
    task_effect = (None if q_final is None else
                   all(metric["ci95"] is not None and metric["ci95"][0] > 0
                       for metric in q_final["recconv_recovery_vs_matched_controls"].values()))
    traj_gate = (None if q_traj_final is None else all(
        q_traj_final["delays"][str(delay)]["task_grounded_three_way_gate"]
        for delay in (0, 8)))
    complete_lines = ["# V40 — Complete Report" if complete else "# V40 — Progress Report (Not Final)", "",
        "Functional and Temporal Decomposition of Persistent State in Hybrid Language Models", "",
        "V40 is " + ("execution-complete across its sealed Qwen functional, temporal, and trajectory branches and Falcon functional comparison."
                      if complete else "not yet execution-complete; missing formal branches remain unclaimed."),
        "V39 and the original V40 calibration are unchanged historical inputs, not formal V40 outcomes.", "",
        "## Frozen design and inference scope", "",
        "The state is the independent unit. Task labels were generated before model execution; four state-ID-disjoint pools contain "
        + ", ".join(f"{role}={len(pools[role])}" for role in pool_roles) + " states.",
        f"Qwen is primary and Falcon is descriptive comparison. The fixed readout is {cfg['readout']}; "
        f"uncertainty uses {cfg['bootstrap_iterations']} state-unit bootstrap resamples and 95% percentile intervals.",
        "Formal function, delayed continuation, and REC-depth factorial outcomes are reported separately; controls and invalid/unavailable rows remain visible. "
        "The answer-alphabet vector metric is secondary and exploratory, not a substitute for externally labelled recovery.", "",
        "## Scientific assessment", "",
        f"- Qwen independent-final task-grounded REC+Conv donor-recovery advantage over both matched mismatch/shuffled controls, positive 95% lower bounds: **{'pending' if task_effect is None else task_effect}**.",
        f"- Qwen independent-final task-grounded positive REC three-way depth interaction at both tested delays: **{'pending' if traj_gate is None else traj_gate}**.",
        "- Channel-specific empirical delay profiles and any identifiable observed-grid half-crossings are in `V40_TEMPORAL_PERSISTENCE.md`; no universal decay law or semantic storage role is inferred.",
        "- Falcon remains a comparison only where its native paired-answer baseline is interpretable; no shared or model-specific ontology is assumed from an inadequate assay.", "",
        "The broad V40 hypothesis succeeds only if channel roles, temporal specialization, trajectory causality, and external task grounding are each supported. A partial or negative result is a completed experiment, not proof of the untested claims.",
        "See the five topic reports and `V40_ALL_REPORTS.md` for the full version-local account.", ""]
    if complete and q_final and q_temporal_final and q_traj_final:
        q_channels = q_final["channels"]
        qualified = q_final["qualified_states"]
        profile = ", ".join(
            f"{name}={q_channels[name]['donor_recovered_qualified']}/{qualified}"
            for name in ("REC_only", "Conv_only", "KV_only", "REC_Conv", "REC_Conv_KV"))
        final_controls = q_final["recconv_recovery_vs_matched_controls"]
        delay_zero = q_temporal_final["profiles"]["0"]
        delay_32 = q_temporal_final["profiles"]["32"]
        temporal_profile = ", ".join(
            f"{name} {delay_zero['channels'][name]['qualified_recovery']}/{delay_zero['qualified_states']} "
            f"at delay 0 and {delay_32['channels'][name]['qualified_recovery']}/{delay_32['qualified_states']} "
            "at delay 32"
            for name in ("REC_only", "Conv_only", "KV_only", "REC_Conv"))
        crossing = ", ".join(
            f"{name}={item['observed_grid_half_crossing'] if item['observed_grid_half_crossing'] is not None else 'not identifiable'}"
            for name, item in q_temporal_final["observed_grid_half_crossings"].items())
        traj_zero = q_traj_final["delays"]["0"]
        traj_eight = q_traj_final["delays"]["8"]
        f_final = functional["F", "independent_final"]
        overall = ("V40 execution is complete, but its full four-part scientific success criterion is not established "
                   "because the task-grounded depth-trajectory gate failed."
                   if traj_gate is False else
                   "V40 execution is complete; interpret its four-part success criterion against all functional, "
                   "temporal, trajectory, and external-task evidence rather than a single positive contrast.")
        complete_lines.extend(["## Answer to the V40 scientific question", "",
            f"**Functional content under this assay.** Qwen completed {q_final['complete_states']}/{q_final['sealed_states']} "
            f"states; {qualified} were native-pair-qualified and the frozen Qwen gate was {q_final['baseline_qualification_gate']}. "
            f"Qualified donor-answer recovery: {profile}. This is a causal channel profile on generated tasks, "
            "not a proof that a tensor stores a semantic variable.", "",
            f"**Task grounding.** REC+Conv versus matched same-prefix mismatch: {ci(final_controls['same_prefix_mismatch'])}; "
            f"versus shuffled natural REC+Conv: {ci(final_controls['shuffled_REC_Conv'])}. "
            f"The incremental REC+Conv-versus-Conv recovery contrast is {ci(q_final['recconv_vs_conv_recovery'])}; "
            "a CI including zero does not establish a separate REC answer contribution.", "",
            f"**Temporal persistence.** Qwen native-qualified recovery by delay: {temporal_profile}. "
            "These per-delay denominators change, so the rates alone are not a same-cohort decay curve. "
            f"Common native-qualified states across all delays: {q_temporal_final['common_qualified_states']}; "
            f"observed-grid half-crossings: {crossing}. No continuous or universal half-life is inferred.", "",
            f"**Depth trajectory.** Three-way external-answer margin interaction at delay 0: "
            f"{ci(traj_zero['three_way_margin_interaction'])} (n={traj_zero['qualified_states']}); "
            f"at delay 8: {ci(traj_eight['three_way_margin_interaction'])} "
            f"(n={traj_eight['qualified_states']}). The predeclared gates were "
            f"{traj_zero['task_grounded_three_way_gate']} and {traj_eight['task_grounded_three_way_gate']}. "
            "Donor-like hidden geometry without answer recovery is not mediation.", "",
            f"**Cross-model limit.** Falcon completed {f_final['complete_states']}/{f_final['sealed_states']} "
            f"states but only {f_final['qualified_states']} were native-pair-qualified under this readout; "
            "its channel counts are descriptive and cannot establish a common architecture-independent role.", "",
            f"**Overall.** {overall} Any positive functional and empirical temporal profiles remain bounded "
            "to these generated tasks, model versions, readout, delays, and tested state pool.", ""])
    p_complete = write(root, "V40_COMPLETE_REPORT.md", complete_lines)
    all_path = str(build_all(root).relative_to(root))
    output_paths = [p_channel, p_task, p_temporal, p_comparison, p_trajectory,
                    p_complete, all_path]
    index = {"status": "EXECUTION_COMPLETE" if complete else "CHECKPOINT",
             "input_sha256": {path: sha256_file(root / path) for path in inputs},
             "report_sha256": {path: sha256_file(root / path) for path in output_paths},
             "v40_formal_results_only": True}
    index_path = root / "results/v40/processed/formal_report_index_v40.json"
    write_json_atomic(index_path, index)
    return {"status": index["status"], "reports": output_paths,
            "index": str(index_path.relative_to(root))}


if __name__ == "__main__":
    print(json.dumps(run(Path.cwd()), indent=2))
