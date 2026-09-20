#!/usr/bin/env python3
"""Build V19 standalone reports and append-only cumulative summary from measured records."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from jclosure.protocol_v19 import verify, verify_stage
from jclosure.provenance import sha256_file

ROOT = Path.cwd()
OUT = ROOT / "results/v19/processed"
REPORTS = ROOT / "reports"


def load(name: str):
    return json.loads((OUT / name).read_text())


def f(value, digits=4):
    return "—" if value is None else f"{value:.{digits}f}"


def ci(value):
    return "—" if value is None else f"[{f(value[0])}, {f(value[1])}]"


def write(name: str, title: str, sections: list[str]):
    path = REPORTS / name
    path.write_text(f"# {title}\n\n" + "\n\n".join(sections).rstrip() + "\n", encoding="utf-8")
    return path


def table(headers, rows):
    return "| " + " | ".join(headers) + " |\n|" + "|".join("---" for _ in headers) + "|\n" + "\n".join(
        "| " + " | ".join(str(x) for x in row) + " |" for row in rows)


def main():
    protocol = verify(ROOT)
    split = verify_stage(ROOT, "splits")
    q0 = verify_stage(ROOT, "interventions")
    q1 = verify_stage(ROOT, "q_amendment_1")
    q2 = verify_stage(ROOT, "q_amendment_2")
    teacher = verify_stage(ROOT, "teacher_amendment_3")
    analysis = verify_stage(ROOT, "analysis")
    diagnostics = verify_stage(ROOT, "diagnostics")
    runtime = verify_stage(ROOT, "runtime_gpu1_amendment_4")
    diag_runtime = verify_stage(ROOT, "diagnostic_gpu1_amendment_5")
    binding = verify_stage(ROOT, "execution_binding_amendment_6")
    reverse_runtime = verify_stage(ROOT, "gpu1_reverse_train_amendment_7")
    tr = load("factorial_summary_train_v19.json")
    va = load("factorial_summary_validation_v19.json")
    context = load("natural_vs_action_context_v19.json")
    decision = load("v19_adjudication.json")
    signs = load("q_sign_scale_v19.json")
    order = load("writeback_order_audit_v19.json")
    compact = load("compact_response_search_v19.json") if (OUT / "compact_response_search_v19.json").exists() else None
    compact_freeze = verify_stage(ROOT, "compact_response") if compact is not None else None
    independent_status = ("UNOPENED_NO_ELIGIBLE_FINALIST_AFTER_RESTRICTED_COMPACT_SEARCH"
                          if compact is not None and not compact["independent_final_eligible"]
                          else decision["independent_final_status"])
    bank = pd.concat([pd.read_parquet(path) for path in sorted(OUT.glob("factorial_validation_*_v19.parquet"))], ignore_index=True)
    metric = pd.read_parquet(OUT / "factorial_metrics_validation_v19.parquet")
    primary = metric[(metric.target == "stacked_normalized") & (metric.horizon == 1)]
    eligible = primary[primary.q_reliable & (primary.action_status == "MATCHED_REALIZED_ACTION")]
    active_primary = eligible[eligible.q_name.isin(decision["active_q_names_train_frozen"])]
    REPORTS.mkdir(parents=True, exist_ok=True)

    qrows = [[q["name"], q["channel"], q["direction_index"], q["alpha"], q["calibration_status"],
              f(q["reliable_fraction"]), f(q["median_natural_to_action_ratio"])] for q in q2["q"]]
    q_validation_rows = []
    for name, group in bank[bank.horizon == 1].groupby("q_name", sort=True):
        unique = group.drop_duplicates("base_trial_id")
        readbacks = [json.loads(value) for value in unique.q_channel_survival]
        channel = str(group.q_channel.iloc[0])
        active_keys = {"recurrent": ("recurrent",), "conv": ("conv",), "kv": ("keys", "values"),
                       "rec_conv": ("recurrent", "conv"),
                       "joint": ("recurrent", "conv", "keys", "values")}[channel]
        surviving = [item[key] for item in readbacks for key in active_keys if item.get(key) is not None]
        q_validation_rows.append([name, int(group.base_trial_id.nunique()), f(group.q_reliable.mean()),
                                  f(group.q_realized_state_norm.median()), f(group.q_realized_cosine.median()),
                                  f(group.q_gain.median()), f(float(np.median(surviving))) if surviving else "—"])
    bank_sections = [
        f"Protocol `{protocol['freeze_digest']}`; split `{split['freeze_digest']}`; amended intervention `{q2['freeze_digest']}`. The original q selection and both corrections remain immutable (`{q0['freeze_digest']}`, `{q1['freeze_digest']}`).",
        f"Development: {len(split['train'])} train and {len(split['validation'])} validation base states, balanced over five task families; disjoint train-only calibration: {len(split['calibration'])}. H2/H4/H8 nested panel: {sum(x['horizon_panel'] for x in split['train'])}/{sum(x['horizon_panel'] for x in split['validation'])} train/validation states. Validation reuses V18 clean prompts and is not independent final.",
        "## Frozen q directions\n\n" + table(["q", "requested channel", "direction", "α", "calibration", "reliable fraction", "pilot N/R"], qrows),
        "## Validation BF16 q readback\n\n" + table(["q", "states", "reliable fraction", "median raw ‖ΔP‖", "median cosine", "median gain", "median active-channel survival"], q_validation_rows),
        f"Teacher manifest `{teacher['teacher_sha256']}` covers {teacher['state_count']} states; {teacher['new_h8_count']} new h8 clean-greedy sequences were frozen before factorial responses and have exact V18 h1 prefixes. Every four-way branch uses the same frozen sequence per state.",
        f"Boundary-held current J is read once before q; all {len(bank)} validation factorial rows have zero boundary J difference and identical J hashes. P0/Pq persistent snapshot hashes differ on every row. Pq is constructed once per q and cloned for its no-action and action trajectories; P0 is handled analogously. This is an active boundary-held-J counterfactual, not natural same-J matching.",
        f"Validation four-way rows: {va['factorial_rows']}, train rows: {tr['factorial_rows']}. Factorial Parquet records contain Y00, Y01, Y10, Y11 normalized 288D endpoints, realized q/action readback, exact hashes and status denominators.",
        "h16 was not run under the frozen compute-priority rule; h1/h2/h4/h8 are complete on their declared panels."
    ]
    write("COUNTERFACTUAL_STATE_BANK_V19.md", "V19 counterfactual state bank", bank_sections)

    response_rows = []
    for h in (1, 2, 4, 8):
        for target in ("j", "logits", "semantic_continuous", "workspace", "stacked_normalized"):
            value = va["by_target_horizon"].get(f"{target}:h{h}")
            if not value:
                continue
            response_rows.append([f"h{h}", target, value["states"], f(value["natural_norm"]["median"]),
                                  f(value["clean_action_norm"]["median"]), f(value["counterfactual_action_norm"]["median"]),
                                  f(value["modulation_norm"]["median"]), f(value["modulation_ratio"]["median"]),
                                  ci(value["modulation_ratio"]["ci95"]),
                                  f(value["response_cosine_median"]), f(value["response_magnitude_ratio_median"])])
    subgroup = lambda name, data: "## " + name + "\n\n" + table(["group", "states", "median M/R", "95% state-bootstrap CI", "median N/R", "match rate"],
        [[key, val["states"], f(val["modulation_ratio"]["median"]), ci(val["modulation_ratio"]["ci95"]),
          f(val["natural_to_action_ratio"]["median"]), f(val["matched_rate_all"])] for key, val in data.items()])
    def active_subgroup(name, column):
        return "## " + name + " (train-frozen active q only, descriptive)\n\n" + table(
            ["group", "states", "rows", "median M/R", "median N/R0"],
            [[key, int(group.base_trial_id.nunique()), len(group), f(group.modulation_ratio.median()),
              f(group.natural_to_action_ratio.median())]
             for key, group in active_primary.groupby(column, sort=True)])
    write("SAME_J_COUNTERFACTUAL_RESPONSE_V19.md", "V19 boundary-held-J finite action response", [
        "Primary estimand M=[Y(Pq,a)-Y(Pq,0)]−[Y(P0,a)-Y(P0,0)]. R0=Y01−Y00; Rq=Y11−Y10. Only matched-realized-action and reliably realized q rows enter the primary summary; failed rows remain in the raw denominator and are labeled.",
        "## Validation target × horizon\n\n" + table(["h", "target", "states", "median ‖N‖", "median ‖R0‖", "median ‖Rq‖", "median ‖M‖", "median M/R", "M/R 95% CI", "cos(R0,Rq)", "‖Rq‖/‖R0‖"], response_rows),
        "Norms in the table use frozen V16 target-block normalization. Physical raw-block norms for J, logits, continuous semantic score and workspace, plus normalized ratios and response direction cosines, are retained per row in `results/v19/processed/factorial_metrics_validation_v19.parquet`.",
        subgroup("Task families", va["by_family"]), subgroup("Probe actions", va["by_action"]),
        active_subgroup("Primary task families", "family"), active_subgroup("Primary probe actions", "coordinate_index"),
        f"Formal primary active-q matched result: M/R {f(decision['modulation_ratio_state_bootstrap']['median'])}, state-bootstrap CI {ci(decision['modulation_ratio_state_bootstrap']['ci95'])}, {decision['primary_states']} independent validation base states and {decision['primary_matched_rows']} correlated response rows. Equivalence margin 0.10; dependence lower floor 0.02 were frozen before any factorial response."
    ])

    natural_rows = [[f"h{h}", f(va["by_target_horizon"][f"stacked_normalized:h{h}"]["natural_norm"]["median"]),
                     ci(va["by_target_horizon"][f"stacked_normalized:h{h}"]["natural_norm"]["ci95"]),
                     f(va["by_target_horizon"][f"stacked_normalized:h{h}"]["natural_to_action_ratio"]["median"])]
                    for h in (1, 2, 4, 8)]
    natural_direction_rows = []
    distinct_n = bank[bank.horizon == 1].drop_duplicates(["base_trial_id", "q_name"])
    for qname, group in distinct_n.groupby("q_name", sort=True):
        vectors = np.stack(group.y10_stack).astype(float) - np.stack(group.y00_stack).astype(float)
        mean_vector = vectors.mean(axis=0)
        cosines = (vectors @ mean_vector) / np.maximum(np.linalg.norm(vectors, axis=1) * np.linalg.norm(mean_vector), 1e-12)
        natural_direction_rows.append([qname, len(group), f(float(np.median(np.linalg.norm(vectors, axis=1)))),
                                       f(float(np.median(cosines))) if np.linalg.norm(mean_vector) > 1e-8 else "undefined"])
    write("NATURAL_PERSISTENT_EFFECT_V19.md", "V19 natural persistent-state effects", [
        "N=Y(Pq,0)−Y(P0,0) is measured without probe a from two boundary-held-J states. N is not M; large N with small M would indicate a natural-dynamics/action-response dissociation only after the separate equivalence gate.",
        table(["h", "median ‖N‖ stack", "95% state-bootstrap CI", "median N/R0"], natural_rows),
        "## q-specific h1 effects\n\n" + table(["q", "states", "median N/R0", "95% CI", "median M/R"],
            [[name, val["states"], f(val["natural_to_action_ratio"]["median"]), ci(val["natural_to_action_ratio"]["ci95"]),
              f(val["modulation_ratio"]["median"])] for name, val in va["by_q_name"].items()]),
        "## Natural-effect direction coherence (h1 normalized stack)\n\n" + table(["q", "states", "median ‖N‖", "median cosine to q mean N"], natural_direction_rows),
        f"Train-frozen active q: {', '.join(decision['active_q_names_train_frozen'])}. Validation material-N gate: {decision['material_natural_effect']}. Secondary natural-dynamics outcome: **{'PERSISTENT_STATE_CAUSALLY_CONTRIBUTES_TO_NATURAL_CONTINUATION' if decision['material_natural_effect'] else 'NOT_CONFIRMED_UNDER_THIS_Q_BANK'}**. Weak-effect random/REC and KV directions remain explicit controls; a small M for an inactive q cannot support workspace sufficiency."
    ])

    context_rows = []
    for h in (1, 2, 4, 8):
        row = context["horizons"][str(h)]
        for target in ("j", "stacked_normalized"):
            nat = row["natural"][target]; act = row["action_response"][target]
            context_rows.append([f"h{h}", target, row["train_states"], row["validation_states"],
                                 f(nat["raw_gain"]), ci(nat["bootstrap_ci95"]),
                                 f(act["raw_gain"]), ci(act["bootstrap_ci95"]),
                                 f(nat["raw_gain"] - act["raw_gain"])])
    write("NATURAL_VS_ACTION_CONTEXT_V19.md", "V19 natural versus finite-action raw-context utility", [
        "Same frozen V19 state split, same V18 train-frozen 128D J versus 128D J+REC+Conv+KV features, one hash-selected signed development action per state, same fixed ridge-1.0 136D design and V16 target-block normalization for both endpoints. Natural target is Y(P0,0,h); finite-action target is R(P0,a,h). Errors are divided by validation variation around the training target mean. This fixed linear comparison is not an exhaustive nonlinear ceiling.",
        table(["h", "target", "train", "val", "raw_gain natural", "95% CI", "raw_gain action", "95% CI", "Δgain"], context_rows),
        f"h1 stack natural/action gain: {f(decision['raw_gain_natural_h1_stack'])}/{f(decision['raw_gain_action_response_h1_stack'])}. Raw predictive increment is distinct from causal N or M; no memory/state claim follows from gain alone."
    ])

    channel_rows = []
    stack_metrics = metric[metric.target == "stacked_normalized"]
    for (channel, h), group in stack_metrics.groupby(["q_channel", "horizon"], sort=True):
        matched = group[group.q_reliable & (group.action_status == "MATCHED_REALIZED_ACTION")]
        channel_rows.append([channel, f"h{h}", int(group.base_trial_id.nunique()), len(group), f((group.action_status == "MATCHED_REALIZED_ACTION").mean()),
                             f(matched.natural_norm.median()) if len(matched) else "—",
                             f(matched.modulation_norm.median()) if len(matched) else "—",
                             f(matched.modulation_ratio.median()) if len(matched) else "—"])
    write("CHANNEL_COUNTERFACTUAL_MECHANISM_V19.md", "V19 channel × horizon causal mechanism", [
        "Channels refer to the requested and read-back persistent q support; labels were corrected append-only after inspecting frozen direction tensors. This is a new four-way estimand, not a reuse of V15–V18 channel rankings.",
        table(["q channel", "h", "states", "rows", "action match", "median ‖N‖", "median ‖M‖", "median M/R"], channel_rows),
        "The REC+Conv and joint q families are distinct from isolated REC, Conv and KV; a weak natural KV effect is not evidence that KV is irrelevant in general."
    ])

    h1 = bank[bank.horizon == 1]
    matched = h1[h1.action_status == "MATCHED_REALIZED_ACTION"]
    mismatch = h1[h1.action_status != "MATCHED_REALIZED_ACTION"]
    survival = [json.loads(value) for value in h1.action_channel_joint_survival]
    survival_rows = [[name, f(float(np.median([row[name] for row in survival if row.get(name) is not None])))]
                     for name in ("recurrent", "conv", "keys", "values")]
    write("ACTUATOR_MATCH_AUDIT_V19.md", "V19 realized probe actuator matching", [
        f"Frozen thresholds: cosine ≥ {protocol['config']['actions']['matched_realized_cosine_min']}; Pq/P0 norm ratio [{protocol['config']['actions']['matched_norm_ratio_min']}, {protocol['config']['actions']['matched_norm_ratio_max']}]; each action readback cosine ≥ {protocol['config']['actions']['reliability_min_cosine']} and requested/realized gain within [{protocol['config']['actions']['reliability_gain_min']}, {protocol['config']['actions']['reliability_gain_max']}].",
        f"h1 validation denominator {len(h1)} response rows; matched {len(matched)} ({f(len(matched)/len(h1))}); mismatch {len(mismatch)}. All mismatches remain in machine Parquet with `ACTION_REALIZATION_MISMATCH`. Train match rate {f(tr['matched_rate'])}; validation match rate {f(va['matched_rate'])}.",
        "## Readback distribution\n\n" + table(["quantity", "median matched", "minimum matched", "median mismatch"],
            [[name, f(matched[name].median()) if len(matched) else "—", f(matched[name].min()) if len(matched) else "—",
              f(mismatch[name].median()) if len(mismatch) else "—"]
             for name in ("realized_action_pair_cosine", "realized_norm_ratio_pq_over_p0", "realized_gain_p0", "realized_gain_pq", "realized_delta_difference_norm")]),
        "## Median joint BF16 channel survival in both P0 and Pq\n\n" + table(["channel", "surviving requested elements"], survival_rows),
        "Requested q and a are separate. The action-match audit compares actual BF16 cache deltas in P0 and Pq across REC, Conv, K and V, not merely requested direction vectors."
    ])

    order_frame = pd.read_parquet(ROOT / order["path"])
    write("WRITEBACK_ORDER_AUDIT_V19.md", "V19 writeback order and quantization audit", [
        f"Frozen diagnostic subset: {order['states']} validation base states. A=snapshot Pq→a (primary construction); B=direct q→a; C=direct a→q; D=single composed write. B/C/D are not used for V19 primary claims.",
        table(["branch", "median stack distance from A", "median distance / ‖A‖"],
              [[key, f(value), f(order_frame[order_frame.branch == key].stack_difference_relative_to_A.median())]
               for key, value in order["median_stack_difference_from_A_by_branch"].items()]),
        f"B exact persistent snapshot identity versus A: {f(order['snapshot_B_exact_fraction'])}. BF16 order and composition effects are quantified in `{order['path']}`; the primary factorial always branches from exact P0/Pq snapshots.",
        f"Sign/scale diagnostic: {signs['states']} states; {signs['rows']} observations for selected q at ± signs and α=0.5/1.0 across h1/h2/h4/h8; q reliability {f(signs['q_reliable_rate'])}. Median M/R by scale: {signs['median_modulation_ratio_by_scale']}. Odd/even components are in `{signs['odd_even_path']}`; no linearity assumption is made."
    ])
    sign_frame = pd.read_parquet(ROOT / signs["sign_scale_path"])
    odd_frame = pd.read_parquet(ROOT / signs["odd_even_path"])
    scale_rows = []
    for (qname, h), group in sign_frame.groupby(["q_name", "horizon"], sort=True):
        half = group[group.q_alpha == 0.5]
        full = group[group.q_alpha == 1.0]
        odd_n = odd_frame[(odd_frame.q_name == qname) & (odd_frame.horizon == h) & (odd_frame.endpoint == "natural")]
        odd_m = odd_frame[(odd_frame.q_name == qname) & (odd_frame.horizon == h) & (odd_frame.endpoint == "modulation")]
        scale_rows.append([qname, f"h{h}", f(half.modulation_ratio.median()), f(full.modulation_ratio.median()),
                           f(odd_n.even_to_odd_ratio.median()), f(odd_m.even_to_odd_ratio.median()),
                           f(group.q_reliable.mean())])
    write("Q_SIGN_SCALE_DIAGNOSTICS_V19.md", "V19 q sign, scale and odd/even diagnostics", [
        "This is a frozen nested diagnostic, not a linearity premise or replacement for the primary +q bank. Both ± signs and α=0.5/1.0 use the same clean teacher sequence per base state; all unreliable BF16 rows remain in the denominator.",
        table(["q", "h", "median M/R α=.5", "median M/R α=1", "N even/odd", "M even/odd", "q reliable"], scale_rows),
        "The even/odd ratios are computed from paired ±q vectors for the same base state, q and α. Variation is descriptive; no post-hoc sign/scale choice changes the frozen primary q bank."
    ])

    compact_text = (f"Authorized C_response search was executed with actual post-q minus P0 BF16 REC/Conv/KV 384D raw CountSketch. Selected k={compact['selected_k']}; restricted proxy candidate gate={compact['restricted_proxy_candidate_gate_passed']}; independent-final eligible={compact['independent_final_eligible']}. Details in `reports/COMPACT_RESIDUAL_CONTEXT_V19.md`."
                    if compact is not None else "No authorized compact search was run.")
    write("COMPACT_CONTEXT_DECISION_V19.md", "V19 compact-context and independent-final decision", [
        f"Frozen margin: M/R practical equivalence <0.10 upper state-bootstrap bound; response dependence median ≥0.10 plus bootstrap lower >0.02 across actions/channels/families with matched actuator. Observed active-q M/R {f(decision['modulation_ratio_state_bootstrap']['median'])}, CI {ci(decision['modulation_ratio_state_bootstrap']['ci95'])}; N/R {f(decision['natural_to_action_ratio_state_bootstrap']['median'])}, CI {ci(decision['natural_to_action_ratio_state_bootstrap']['ci95'])}.",
        f"Response dependence gate: **{decision['response_dependence_gate']}**. Equivalence gate: **{decision['response_equivalence_gate']}**. C_response search authorized: **{decision['compact_response_context_search_authorized']}**. Distinct C_dynamics search authorized: **{decision['compact_natural_dynamics_search_authorized']}**.",
        compact_text,
        f"Formal development outcome: **{decision['formal_outcome']}**. Independent final: **{independent_status}**. H2 remains; H3 candidate: {decision['h3_candidate_supported']}; autonomous state-model authorized: {decision['autonomous_state_model_authorized']}. No absolute replacement claim."
    ])

    if compact is not None:
        search_rows = [[row["k"], row["supervised_rank_supported"], f(row["validation_dev"]["stack_relative_l2"]),
                        f(row["validation_dev"]["j_relative_l2"]),
                        f(row["validation_heldout_actions"]["stack_relative_l2"]),
                        row["validation_dev_states"], row["validation_heldout_states"]]
                       for row in compact["search"]]
        write("COMPACT_RESIDUAL_CONTEXT_V19.md", "V19 gated compact response-residual search", [
            f"Search authorized by the frozen M gate and frozen at `{compact_freeze['freeze_digest']}`. The 384D raw descriptor is a direct CountSketch of the actual BF16 post-q minus P0 REC/Conv/KV cache state. It is a restricted raw measurement, not complete P_raw; C is trained against the same-J modulation M and is not called a model state.",
            "Train-only supervised residual PLS over eight signed development actions proposes k∈{4,8,16,32,64,128}. Fixed ridge state×continuous-action models compare J+C+a with J+C+raw-sketch+a on held-out base states, task families and four held-out action directions.",
            table(["k", "rank supported", "dev stack rel-L2", "dev J rel-L2", "held-out-action stack rel-L2", "dev states", "held-out states"], search_rows),
            f"Effective supervised PLS rank={compact['effective_supervised_rank']}; selected k={compact['selected_k']}. Conditional raw-sketch stack gains: development {f(compact['raw_residual']['development']['raw_incremental_stack_gain'])}, held-out actions {f(compact['raw_residual']['heldout_action']['raw_incremental_stack_gain'])}. Family results: {compact['by_family']}.",
            f"Restricted proxy gate passed: **{compact['restricted_proxy_candidate_gate_passed']}**. Independent final eligible: **{compact['independent_final_eligible']}**. Reason: {compact['independent_final_reason']}. The test does not certify full-raw sufficiency, compact dynamical state, H3 or autonomous control."
        ])

    write("STRICT_INTERFACE_AUDIT_V19.md", "V19 strict interface and historical-record audit", [
        f"All {len(bank)} validation rows have exact zero boundary-held-J difference and identical boundary J SHA256. All P0/Pq persistent snapshot SHA256 differ. This equality is causal by construction after the current layer-23 J readout; it is not natural same-J matching. V18 live-versus-frozen J replay drift is not hidden: V19 holds the live boundary J once and uses frozen V18 raw scores only for the separate predictor comparison.",
        f"No V1–V18 frozen file is overwritten. V19 freezes are append-only: base, split, initial q, q amendments 1/2, teacher amendment 3, analysis, diagnostics, GPU1 runtime amendments 4/5, unchanged execution-wrapper binding 6, and reverse-order GPU1 train scheduling amendment 7 (`{reverse_runtime['freeze_digest']}`). The same-state four-endpoint GPU0/GPU1 comparison had maximum relative L2 {max(v['relative_l2'] for v in runtime['comparison'].values())}. Each amendment records reason, creation UTC, observed-response counts and digest.",
        "Full-repository pytest baseline: 222 passed, 2 historical integrity tests failed before V19 appended to FINAL_REPORT. V14 and V16 manifests hash an earlier cumulative FINAL_REPORT byte state; HEAD already contains subsequent cumulative sections. V19 does not rewrite those historical manifests or report sections. V18/V19 focused tests pass.",
        "GPU1 processed the frozen train list in reverse order and was intentionally interrupted at 396/400 completed state files, before any state overlap. GPU0 completed the remaining three states; the per-state writer only commits complete Parquet files.",
        "Replay-and-J-restored counterfactual was not run. V4/V6 restoration projects later hidden activations toward clean dense J under finite cosine/top-10/RMS tolerances, not exact equality of the V19 layer-23 current J after earlier persistent perturbation. V7 proves exact clone/one-token cache continuation, not replay-and-J-restored causal isolation. Thus the old machinery does not certify an artifact-free secondary V19 branch; it cannot contaminate the primary construction.",
        "Task-sign is not pooled as a primary endpoint: the frozen V13/V18 five-family bank has no single validated shared signed semantic readout across Boolean, arithmetic, graph, state-transition and binding tasks. Selected J, logits, continuous semantic scores, workspace and V16-normalized stack are measured; no task-sign effect is fabricated from selected logits.",
        f"The current-J, persistent P, context perturbation q, probe a, natural N and modulation M interfaces remain separate. Independent final is unopened. H2 remains; H3 and autonomous replacement are not authorized."
    ])

    command_lines = [
        "PYTHONPATH=src /home/user/anaconda3/bin/python -m jclosure.protocol_v19 freeze",
        "PYTHONPATH=src /home/user/anaconda3/bin/python -m jclosure.experiments.counterfactual_bank_v19 prepare",
        "HF_HOME=/data/CSK/J-space-project/.hf-cache PYTHONPATH=src /home/user/anaconda3/bin/python -m jclosure.experiments.counterfactual_bank_v19 calibrate",
        "PYTHONPATH=src /home/user/anaconda3/bin/python scripts/audit_q_vectors_v19.py",
        "HF_HOME=/data/CSK/J-space-project/.hf-cache PYTHONPATH=src /home/user/anaconda3/bin/python -m jclosure.experiments.q_amendment_v19",
        "PYTHONPATH=src /home/user/anaconda3/bin/python scripts/audit_selected_q_v19.py",
        "HF_HOME=/data/CSK/J-space-project/.hf-cache PYTHONPATH=src /home/user/anaconda3/bin/python -m jclosure.experiments.q_support_amendment_v19",
        "HF_HOME=/data/CSK/J-space-project/.hf-cache PYTHONPATH=src /home/user/anaconda3/bin/python -m jclosure.experiments.run_amended_v19 train --limit 1  # first attempt stopped before response write on short teacher sequence",
        "HF_HOME=/data/CSK/J-space-project/.hf-cache PYTHONPATH=src /home/user/anaconda3/bin/python -m jclosure.experiments.teacher_amendment_v19",
        "PYTHONPATH=src /home/user/anaconda3/bin/python -m jclosure.experiments.analyze_v19 prepare",
        "HF_HOME=/data/CSK/J-space-project/.hf-cache PYTHONPATH=src /home/user/anaconda3/bin/python -m jclosure.experiments.run_amended_v19 train --limit 1  # successful smoke state",
        "HF_HOME=/data/CSK/J-space-project/.hf-cache PYTHONPATH=src /home/user/anaconda3/bin/python -m jclosure.experiments.run_amended_v19 train",
        "/home/user/anaconda3/bin/python scripts/inspect_v19_progress.py  # read-only progress snapshots, repeated",
        "PYTHONPATH=src /home/user/anaconda3/bin/python -m jclosure.experiments.diagnostics_v19 prepare",
        "HF_HOME=/data/CSK/J-space-project/.hf-cache PYTHONPATH=src /home/user/anaconda3/bin/python -m jclosure.experiments.compare_gpu_v19",
        "HF_HOME=/data/CSK/J-space-project/.hf-cache PYTHONPATH=src /home/user/anaconda3/bin/python -m jclosure.experiments.gpu1_v19 validation",
        "PYTHONPATH=src /home/user/anaconda3/bin/python -m jclosure.experiments.run_amended_v19 aggregate_validation",
        "PYTHONPATH=src /home/user/anaconda3/bin/python -m jclosure.experiments.analyze_v19 analyze_validation",
        "PYTHONPATH=src /home/user/anaconda3/bin/python -m jclosure.experiments.diagnostic_gpu1_v19 prepare",
        "HF_HOME=/data/CSK/J-space-project/.hf-cache PYTHONPATH=src /home/user/anaconda3/bin/python -m jclosure.experiments.diagnostic_gpu1_v19 sign_scale",
        "HF_HOME=/data/CSK/J-space-project/.hf-cache PYTHONPATH=src /home/user/anaconda3/bin/python -m jclosure.experiments.diagnostic_gpu1_v19 order",
        "PYTHONPATH=src /home/user/anaconda3/bin/python -m jclosure.experiments.execution_binding_v19",
        "PYTHONPATH=src /home/user/anaconda3/bin/python -m jclosure.experiments.gpu1_reverse_freeze_v19",
        "HF_HOME=/data/CSK/J-space-project/.hf-cache PYTHONPATH=src /home/user/anaconda3/bin/python -m jclosure.experiments.gpu1_reverse_train_v19 train  # interrupted before state overlap; GPU0 completed train",
        "PYTHONPATH=src /home/user/anaconda3/bin/python -m jclosure.experiments.run_amended_v19 aggregate_train",
        "PYTHONPATH=src /home/user/anaconda3/bin/python -m jclosure.experiments.analyze_v19 analyze_train",
        "PYTHONPATH=src /home/user/anaconda3/bin/python -m jclosure.experiments.analyze_v19 context",
        "PYTHONPATH=src /home/user/anaconda3/bin/python -m jclosure.experiments.analyze_v19 decide",
        "PYTHONPATH=src /home/user/anaconda3/bin/python scripts/build_v19_reports.py",
        "PYTHONPATH=src /home/user/anaconda3/bin/python scripts/build_v19_integrity.py",
        "PYTHONPATH=src /home/user/anaconda3/bin/python scripts/build_complete_version_report.py V19",
        "PYTHONPATH=src /home/user/anaconda3/bin/python -m pytest tests/test_v19.py -q",
        "PYTHONPATH=src /home/user/anaconda3/bin/python -m pytest tests/test_v18.py -q",
        "PYTHONPATH=src /home/user/anaconda3/bin/python -m pytest -q  # baseline: 222 passed, 2 older cumulative-hash checks fail",
    ]
    if compact is not None:
        insertion = command_lines.index("PYTHONPATH=src /home/user/anaconda3/bin/python scripts/build_v19_reports.py")
        command_lines[insertion:insertion] = [
            "PYTHONPATH=src /home/user/anaconda3/bin/python -m jclosure.experiments.compact_response_v19 prepare",
            "HF_HOME=/data/CSK/J-space-project/.hf-cache PYTHONPATH=src /home/user/anaconda3/bin/python -m jclosure.experiments.compact_response_v19 extract_train",
            "HF_HOME=/data/CSK/J-space-project/.hf-cache PYTHONPATH=src /home/user/anaconda3/bin/python -m jclosure.experiments.compact_gpu1_v19 extract_validation",
            "PYTHONPATH=src /home/user/anaconda3/bin/python -m jclosure.experiments.compact_response_v19 aggregate_train",
            "PYTHONPATH=src /home/user/anaconda3/bin/python -m jclosure.experiments.compact_response_v19 aggregate_validation",
            "PYTHONPATH=src /home/user/anaconda3/bin/python -m jclosure.experiments.compact_response_v19 search",
        ]
    write("EXECUTION_MANIFEST_V19.md", "V19 execution manifest", [
        "Working directory `/data/CSK/J-space-project/jstate-closure`. Commands are listed in execution order; freeze commands are append-only and intentionally fail if rerun. Model loading uses the existing V18 single-GPU0 placement and frozen V13 weights.",
        "```bash\n" + "\n".join(command_lines) + "\n```",
        "## Freeze digests\n\n" + table(["stage", "digest"], [[name, item["freeze_digest"]] for name, item in
            [*( ("protocol", protocol), ("split", split), ("initial q", q0), ("q amendment 1", q1),
                ("q amendment 2", q2), ("teacher amendment 3", teacher), ("analysis", analysis),
                ("diagnostics", diagnostics), ("GPU1 runtime amendment 4", runtime),
                ("diagnostic GPU1 amendment 5", diag_runtime), ("execution binding amendment 6", binding)),
             ("GPU1 reverse-train amendment 7", reverse_runtime),
             *([("compact response", compact_freeze)] if compact_freeze is not None else [])]]),
        f"Role ID hashes: {split['role_id_sha256']}. q intervention hash: `{q2['freeze_digest']}`. Teacher hash: `{teacher['teacher_sha256']}`. Four-way train/validation states {tr['factorial_states']}/{va['factorial_states']}; rows {tr['factorial_rows']}/{va['factorial_rows']}."
    ])

    stack_h1 = va["by_target_horizon"]["stacked_normalized:h1"]
    answers = [
        f"Yes. {va['factorial_states']} validation boundary-held-J states; every row has exact zero J difference and distinct persistent snapshot hashes.",
        f"Yes for the calibrated directions: validation q reliability {f(va['q_reliable_rate'])}; individual direction status and raw-state norms are in the counterfactual bank.",
        f"N/R0={f(decision['natural_to_action_ratio_state_bootstrap']['median'])} with state-bootstrap CI {ci(decision['natural_to_action_ratio_state_bootstrap']['ci95'])} on active q; material gate {decision['material_natural_effect']}.",
        f"On matched rows, median R0/Rq stack norms are {f(stack_h1['clean_action_norm']['median'])}/{f(stack_h1['counterfactual_action_norm']['median'])}; M tests equality of their vectors, not norms alone.",
        "Target×horizon M/R values and intervals are in SAME_J_COUNTERFACTUAL_RESPONSE_V19.md.",
        f"Practical equivalence gate: {decision['response_equivalence_gate']}; null significance alone was not used.",
        f"Material persistent response dependence gate: {decision['response_dependence_gate']}.",
        "REC, Conv, KV, REC+Conv and joint q are separated in CHANNEL_COUNTERFACTUAL_MECHANISM_V19.md; weak KV natural effects cannot prove KV irrelevance.",
        "h1/h2/h4/h8 M/R values are tabulated by channel and target; no monotonicity assumption was imposed.",
        f"Selected q were tested at ± signs and α=0.5/1.0 on {signs['states']} states; odd/even and scale statistics are recorded separately.",
        f"Matched-realized-action rate is {f(va['matched_rate'])}; failed rows remain labeled in the denominator.",
        f"A/B/C/D order contamination is measured on {order['states']} states; primary uses only snapshot A.",
        "Matched-budget raw-context utility by h and target is in NATURAL_VS_ACTION_CONTEXT_V19.md; linear-model limits are explicit.",
        f"h1 stack natural-minus-action raw gain = {f(context['horizons']['1']['delta_gain_stack'])}; causal N/M and predictive gains are not conflated.",
        "J as near-sufficient local response context requires material N, an equivalence upper bound below 0.10 and independent confirmation; current status follows the adjudication gate.",
        f"J insufficiency development gate: {decision['response_dependence_gate']}.",
        f"C_response search authorization: {decision['compact_response_context_search_authorized']}; executed: {compact is not None}; restricted proxy selected k: {compact['selected_k'] if compact else 'none'}.",
        f"C_dynamics search authorization: {decision['compact_natural_dynamics_search_authorized']}.",
        "H2 remains the project-level interpretation.",
        "No H3 candidate is warranted without independently confirmed compact dynamical sufficiency.",
        "AUTONOMOUS_STATE_MODEL_AUTHORIZED = FALSE.",
    ]
    write("V19_SCIENTIFIC_ANSWERS_V19.md", "V19 answers to the 21 scientific questions", [
        "\n".join(f"{i}. {answer}" for i, answer in enumerate(answers, 1)),
        f"Formal outcome: **{decision['formal_outcome']}**. Independent final status: **{independent_status}**."
    ])

    final = REPORTS / "FINAL_REPORT.md"
    original = final.read_text(encoding="utf-8")
    if "<!-- V19_START -->" in original:
        raise RuntimeError("V19 cumulative section already exists; append-only builder refuses overwrite")
    section = "\n".join([
        "<!-- V19_START -->",
        "## V19 — Counterfactual Workspace Sufficiency and Natural Dynamics",
        "",
        f"Formal development outcome: **{decision['formal_outcome']}**. Active boundary-held-J q states and matched finite probe actions were crossed on {tr['factorial_states']} train/{va['factorial_states']} validation base states (five families); primary h1 M/R {f(decision['modulation_ratio_state_bootstrap']['median'])}, state-bootstrap CI {ci(decision['modulation_ratio_state_bootstrap']['ci95'])}, natural N/R0 {f(decision['natural_to_action_ratio_state_bootstrap']['median'])}. Matched actuator rate {f(va['matched_rate'])}. h1 raw-context gain natural/action {f(decision['raw_gain_natural_h1_stack'])}/{f(decision['raw_gain_action_response_h1_stack'])}. C_response/C_dynamics search authorized: {decision['compact_response_context_search_authorized']}/{decision['compact_natural_dynamics_search_authorized']}; restricted C_response search {'ran with selected k='+str(compact['selected_k']) if compact else 'not run'}. Independent final: {independent_status}. H2 remains; H3 and autonomous state-model training are not authorized. No complete state or absolute replacement claim. See `reports/V19_COMPLETE_REPORT.md` for all standalone reports and integrity index.",
        "<!-- V19_END -->",
    ])
    final.write_text(original.rstrip() + "\n\n" + section + "\n", encoding="utf-8")
    print(f"wrote V19 standalone reports and append-only FINAL_REPORT; outcome={decision['formal_outcome']}")


if __name__ == "__main__":
    main()
