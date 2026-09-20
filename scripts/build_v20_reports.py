#!/usr/bin/env python3
"""Build all standalone V20 reports and append the cumulative adjudication."""

from __future__ import annotations

import json
from pathlib import Path

from jclosure.experiments import operator_bank_v20 as bank
from jclosure.protocol_v20 import verify, verify_stage
from jclosure.provenance import sha256_file


ROOT = Path.cwd()
REPORTS = ROOT / "reports"
DATA = ROOT / bank.OUT
PENDING: dict[str, str] = {}


def load(name: str) -> dict:
    return json.loads((DATA / name).read_text(encoding="utf-8"))


def num(value, decimals=4) -> str:
    return "not measured" if value is None else f"{value:.{decimals}f}"


def table(columns: list[str], rows: list[list]) -> str:
    return "| " + " | ".join(columns) + " |\n|" + "|".join("---" for _ in columns) + "|\n" + "\n".join(
        "| " + " | ".join(str(x) for x in row) + " |" for row in rows)


def write(name: str, title: str, body: str) -> None:
    target = REPORTS / name
    if target.exists():
        raise RuntimeError(f"Refusing to overwrite V20 report: {target}")
    PENDING[name] = f"# {title}\n\n{body.strip()}\n"


def main() -> None:
    base = verify(ROOT)
    split = verify_stage(ROOT, "splits")
    actions = verify_stage(ROOT, "actions")
    analysis = verify_stage(ROOT, "operator_analysis")
    confirmation = load("independent_v19_confirmation_v20.json")
    q = load("q_locality_v20.json")
    train = load("response_operator_operator_train_v20.json")
    validation = load("response_operator_operator_validation_v20.json")
    oracle = load("oracle_operator_search_v20.json")
    diagnostic = load("operator_action_diagnostics_v20.json")
    geometry = load("operator_geometry_v20.json")
    alias = load("operator_aliasing_v20.json")
    adjudication = load("v20_adjudication.json")
    if adjudication["operator_compactness_gate"]:
        raise RuntimeError("This report route is only for the failed-oracle, conditionally ineligible path")
    REPORTS.mkdir(exist_ok=True)
    active = confirmation["modulation_ratio"]
    qsmall = [x for x in q["by_q_scale"] if x["alpha"] == 0.25]
    chosen = next(x for x in oracle["model_sweep"] if x["model"] == oracle["selected_model_for_diagnostics"]
                  and x["k"] == oracle["selected_k_for_diagnostics"])
    held = chosen["metrics"]["unseen_direction_positive"]
    sign = chosen["metrics"]["unseen_sign_train_direction"]
    both = chosen["metrics"]["unseen_direction_and_sign"]
    seen = chosen["metrics"]["seen_direction_new_state"]
    amp = [x for x in diagnostic["by_spec"].values() if x["kind"].startswith("unseen_amplitude")]
    pairs = [x for x in diagnostic["by_spec"].values() if x["kind"] == "unseen_pair"]
    dense = [x for x in diagnostic["by_spec"].values() if x["kind"] == "unseen_dense"]
    family_rows = [[family, num(value)] for family, value in confirmation["family_modulation_median"].items()]
    qrows = [[x["q_name"], num(x["median_N_R"]), num(x["median_M_R"]),
              num(x["median_M_per_q_norm"]), num(x["median_operator_angle_degrees"]),
              num(x["reliable_matched_fraction"])] for x in qsmall]
    rank_rows = [[x["k"], num(x["train_positive_fingerprint_explained_variance"]),
                  num(x.get("train_fingerprint_relative_l2")),
                  num(x.get("validation_train_action_fingerprint_relative_l2"))] for x in oracle["rank_curve"]]
    sweep_rows = [[x["model"], x["k"],
                   num(x["metrics"]["seen_direction_new_state"]["stack_relative_l2"]),
                   num(x["metrics"]["unseen_direction_positive"]["stack_relative_l2"]),
                   num(x["metrics"]["unseen_sign_train_direction"]["stack_relative_l2"]),
                   str(x["preliminary_cross_action_gate"])] for x in oracle["model_sweep"]]
    manifold_rows = [[role, qname, row["states"], row["r95"], num(row["effective_rank"])]
                     for role, panel in geometry["manifold_dimensions"].items()
                     for qname, row in panel.items()]
    diag_rows = [[name, row["kind"], num(row["metrics"]["stack_relative_l2"]),
                  num(row["metrics"]["stack_cosine_median"]), num(row["action_reliable_rate"]),
                  num(row["matched_vs_P0_rate"])] for name, row in diagnostic["by_spec"].items()]
    file_index = [[x["path"], x["base_states"], x["operator_states"], x["response_rows"],
                   num(x["action_reliable_rate"]), x["sha256"]]
                  for x in train["outputs"] + validation["outputs"]]
    write("V19_INDEPENDENT_CONFIRMATION_V20.md", "V20 independent confirmation of V19-B", f"""
The new bank has **{confirmation['independent_states']} base states**, disjoint from V18/V19 designated roles. The boundary J is bitwise identical and persistent snapshots differ on every tested pair. This confirmation preceded V20 operator fitting.

Active q: median M/R **{num(active['median'])}**, state-bootstrap 95% CI **[{num(active['ci95'][0])}, {num(active['ci95'][1])}]**, {active['rows']} matched rows; realized-action match {num(confirmation['matched_action_rate'])}. Median N/R0 {num(confirmation['natural_to_action_ratio']['median'])}; N={num(confirmation['natural_norm_median'])}, R0={num(confirmation['clean_action_norm_median'])}, Rq={num(confirmation['counterfactual_action_norm_median'])}, M={num(confirmation['modulation_norm_median'])}. Response cosine={num(confirmation['response_cosine_median'])}; ||Rq||/||R0||={num(confirmation['response_magnitude_ratio_median'])}.

{table(['family', 'median M/R'], family_rows)}

Formal result: **{confirmation['outcome']}**. This establishes persistent-state modulation of the tested finite responses, not compactness or dynamics. Raw rows: `{confirmation['raw_path']}` (`{confirmation['raw_sha256']}`).
""")
    write("OPERATOR_RESPONSE_BANK_V20.md", "V20 crossed operator-response bank", f"""
For every persistent state P and shared frozen action a, `R_P(a)=Y(P,a)-Y(P,0)`. Y is the V16-normalized 288-D stack: J[0:128], logits[128:160], continuous semantic target[160:192], workspace[192:288]. Each base contributes natural P0 and three boundary-held-J Pq states, each crossed with the same **18 opened action directions** and both signs. The six final directions remain sealed. Train/validation/final action partitions: **12/6/6**, from **24 historically train-calibrated directions**; 32 was the target, 16 the minimum. The 24 retained directions meet the frozen historical selection and V20 train-only writeback checks. The third random control had 198/200 historical reliability; this was disclosed in an append-only amendment before operator responses.

Train: {train['base_states']} bases, {train['operator_states']} operator states, {train['response_rows']} response rows. Validation: {validation['base_states']} bases, {validation['operator_states']} operator states, {validation['response_rows']} response rows. Direction tensors are referenced by frozen SHA/index/sign/alpha, not copied as multi-GB files. Every row stores requested/read-back actuator norms, cosine, gain, channel survival, no-action baseline, response and boundary-J identity.

{table(['file', 'bases', 'operator states', 'rows', 'action reliable', 'SHA256'], file_index)}

Train IDs SHA `{split['role_id_sha256']['operator_train']}`; validation IDs SHA `{split['role_id_sha256']['operator_validation']}`. Action hashes: `{json.dumps(actions['action_hashes'], sort_keys=True)}`. Source design freeze `{verify_stage(ROOT, 'operator_design')['freeze_digest']}`.
""")
    write("OPERATOR_INTRINSIC_DIMENSION_V20.md", "V20 empirical operator dimension", f"""
The oracle coordinate is centered SVD of **positive train-action response fingerprints only**, fitted on training operator states. Held-out action responses never enter the coordinate. The frozen sweep tests k=2,4,8,16,32,64,128, subject to empirical rank, with five continuous-action decoders.

{table(['k', 'train explained fraction', 'train fingerprint rel L2', 'validation train-action rel L2'], rank_rows)}

Empirical train fingerprint rank is {oracle['effective_train_fingerprint_rank']}. This rank curve describes the 12-action inference fingerprint, not global dimension of P or of all possible actions. A low reconstruction error here alone is insufficient for cross-action compactness.

{table(['model', 'k', 'seen rel L2', 'unseen direction rel L2', 'unseen sign rel L2', 'all preliminary gates'], sweep_rows)}

Frozen gates: held-out J and full-stack median cosine ≥ {analysis['heldout_gates']['cosine_min']}, relative L2 ≤ {analysis['heldout_gates']['relative_l2_max']}, norm ratio in [{analysis['heldout_gates']['norm_ratio_min']},{analysis['heldout_gates']['norm_ratio_max']}], each-family L2 ≤ {analysis['heldout_gates']['family_relative_l2_max']}, plus sign/scale/composition diagnostics. `k_operator_min` = **not identified**.
""")
    write("ORACLE_OPERATOR_STATE_V20.md", "V20 oracle operator coordinate", f"""
`C_oracle(P)` is inferred from the same state's 12 positive train-action responses. Decoder fitting uses only train-state positive train-action responses and the continuous raw-direction descriptor (cosines to frozen train anchors plus channel norms). Validation directions, signs, amplitudes and action combinations do not supervise either coordinate or decoder. It is not an encoder from raw P.

Selected diagnostic model: `{oracle['selected_model_for_diagnostics']}`, k={oracle['selected_k_for_diagnostics']}; selected solely by validation unseen-direction stack relative L2. Seen-direction new-state L2={num(seen['stack_relative_l2'])}; unseen-direction L2={num(held['stack_relative_l2'])}; unseen-sign train-direction L2={num(sign['stack_relative_l2'])}. Preliminary gate={oracle['preliminary_operator_compactness_gate']}; full diagnostic gate={diagnostic['final_operator_gate']}. No compact operator-state is established; no inference about raw P compression follows.

Descriptor SHA `{verify_stage(ROOT, 'operator_analysis')['descriptor_sha256']}`; analysis freeze `{analysis['freeze_digest']}`.
""")
    write("UNSEEN_ACTION_GENERALIZATION_V20.md", "V20 unseen action generalization", f"""
Every validation operator state supplies its coordinate using **train actions only**. Frozen action partition is independent of state partition. Best diagnostic model metrics:

{table(['test', 'stack relative L2', 'J relative L2', 'stack median cosine', 'J median cosine', 'stack median norm ratio'], [[label, num(metric['stack_relative_l2']), num(metric['j_relative_l2']), num(metric['stack_cosine_median']), num(metric['j_cosine_median']), num(metric['stack_norm_ratio_median'])] for label, metric in [('seen direction/new state', seen), ('unseen direction', held), ('unseen sign/train direction', sign), ('unseen direction+sign', both)]])}

Family-wise unseen-direction relative L2: `{json.dumps(held['family_stack_relative_l2'], sort_keys=True)}`. A new state with a seen action is not evidence of action generalization. Final six directions were not opened because no encoder finalist was eligible.
""")
    write("OPERATOR_ROTATION_GEOMETRY_V20.md", "V20 finite operator rotation, gain and rank", f"""
Each state's positive shared-action response matrix has 18 rows × 288 normalized outputs. Its SVD yields r95; same-J P0/Pq pairs supply principal angles, gain, output-space Procrustes residual, spectrum divergence and rank change. This is finite-response geometry, not exact JVP geometry.

{geometry['state_count']} operator-state spectra and {geometry['pair_count']} same-base comparisons. Natural median r95={num(geometry['median_natural_state_operator_r95'])}; counterfactual median r95={num(geometry['median_counterfactual_state_operator_r95'])}. Pair median principal angle={num(geometry['median_pair_principal_angle_degrees'])}°, gain Pq/P0={num(geometry['median_pair_gain_ratio'])}, rank change={num(geometry['median_pair_rank_change'])}, Procrustes relative residual={num(geometry['median_pair_procrustes_relative_residual'])}. Channel-wise summaries: `{json.dumps(geometry['by_channel'], sort_keys=True)}`.

Historical V13 exact-JVP median instantaneous r95={num(geometry['V13_exact_JVP_historical_median_instantaneous_r95'])}, cumulative path r95={num(geometry['V13_exact_JVP_historical_median_cumulative_path_r95'])}. Paired V20-state JVP subspaces were **not measured**; the historical ranks are context, not proof that tangent rotation and finite operator modulation are the same mechanism. Pair rows: `{geometry['pair_path']}`.
""")
    write("NATURAL_VS_COUNTERFACTUAL_OPERATOR_V20.md", "V20 natural versus counterfactual operator manifolds", f"""
The distributions are separated: clean P0 states versus same-boundary-J Pq states. Centered equal-base action-response fingerprints yield r95 and entropy effective rank:

{table(['role', 'distribution / q', 'states', 'r95', 'entropy rank'], manifold_rows)}

This is an empirical rank on the frozen 18-direction local finite-action panel. It neither proves a global manifold dimension nor implies physical cache compression. Natural J→oracle-coordinate validation R²={num(alias['natural_J_to_oracle_C_R2'])}; natural relative L2={num(alias['natural_J_to_oracle_C_relative_l2'])}; counterfactual relative L2={num(alias['counterfactual_J_to_oracle_C_relative_l2'])}. The J predictor was fitted on natural training states only.
""")
    write("STATE_ALIASING_V20.md", "V20 workspace-state aliasing", f"""
Across {alias['same_J_pair_count']} validation P0/Pq pairs, boundary J is exactly equal: **{alias['all_boundary_J_exact_equal']}**. Median standardized oracle-coordinate distance={num(alias['median_same_J_oracle_coordinate_distance'])}; median train-action fingerprint difference={num(alias['median_same_J_fingerprint_relative_difference'])}; fraction exceeding frozen coordinate and response thresholds={num(alias['workspace_state_aliasing_pair_fraction'])}. Operational `WORKSPACE_STATE_ALIASING` observed: **{alias['workspace_state_aliasing_observed']}**.

The same J necessarily gives the same deterministic J-only prediction within a pair, while the measured train-action fingerprint can differ. This is an operational finite-action aliasing result, not a complete POMDP or Markov-state proof. Pair file `{alias['pair_path']}` (`{alias['pair_sha256']}`).
""")
    ineligible = "**Not eligible / not executed:** the frozen oracle cross-action compactness gate failed. No raw-state encoder was trained and no result is imputed."
    write("RAW_TO_OPERATOR_ENCODER_V20.md", "V20 raw P to operator encoder", f"""
{ineligible}

Accordingly there is no measured `E(P)→C_oracle` result for REC, Conv, KV, full raw P, PLS, architecture-aware, interaction or nonlinear encoders. `C_oracle` remains response-derived and cannot be called a deployable compact state. Historical V19 raw-state encoder failure is not silently recycled as a V20 result.
""")
    write("CONDITIONAL_RESPONSE_SUFFICIENCY_V20.md", "V20 conditional raw-response sufficiency", f"""
{ineligible}

`raw_incremental_gain = error(J,C,a) − error(J,C,P_raw,a)` was **not measured**. There is no equivalence CI or family-wise sufficiency result; the ≤0.02 practical margin cannot be invoked. H3 and physical replacement remain unauthorized.
""")
    write("CHANNEL_OPERATOR_STATE_V20.md", "V20 persistent-channel operator-state audit", f"""
{ineligible}

Equal-capacity REC, Conv, KV, pairwise and all-channel raw-P→C encoders were not trained; no channel-wise recoverability ranking is claimed. Descriptive q-channel finite-operator rotation/gain metrics are in `OPERATOR_ROTATION_GEOMETRY_V20.md` and are a different estimand.
""")
    write("ACTION_COMPOSITION_GENERALIZATION_V20.md", "V20 unseen scale, pair and dense actions", f"""
The pre-frozen 10-base, 40-operator-state diagnostic panel measures 2× amplitude on one train and one validation direction, two unseen validation-direction pairs, and a five-direction dense mixture. The model sees each new continuous action descriptor but was never trained on the exact diagnostic combination. Action writeback and P0/Pq matching remain audited.

{table(['action', 'type', 'stack relative L2', 'median cosine', 'reliable rate', 'matched rate'], diag_rows)}

Measured joint action is compared with the model's `G(C,a+b)`; additionally, additivity residual `||R(a+b)-R(a)-R(b)||/||R(a+b)||` is `{json.dumps(diagnostic['composition_interaction'], sort_keys=True)}`. Additivity is not assumed. The joint sign/scale/composition gate is **{diagnostic['unseen_scale_composition_gate']}**. Raw diagnostics: `results/v20/processed/operator_action_diagnostics_v20.parquet`.
""")
    write("STRICT_INTERFACE_AUDIT_V20.md", "V20 strict inference and claim audit", f"""
1. The tested object is the local finite-action response operator `R_P(a)`, not physical P compression.
2. `C_oracle` sees only positive train-action response fingerprints from the same state. Held-out direction responses, negative signs, unseen scales, pairs, dense actions, teacher caches and final directions do not enter its inference or fit.
3. The response decoder sees continuous raw-direction-derived action features, not action-ID one-hot labels. Its five frozen model classes and k grid were set before validation responses.
4. The V19-B confirmation bank ({confirmation['independent_states']} bases) was disjoint from designated V18/V19 roles and completed before operator representation learning. Its positive result is independent of the V20 oracle failure.
5. All measured actuator failures remain in the denominator; no operator state was silently deleted. Train coordinate inference reliable states: {oracle['train_coordinate_inference_reliable_states']}/{oracle['train_operator_states']}; validation: {oracle['validation_coordinate_inference_reliable_states']}/{oracle['validation_operator_states']}.
6. Final held-out directions: `{adjudication['final_heldout_action_direction_status']}`. Independent V20 final: `{adjudication['independent_V20_final_status']}`. Neither is described as a passed test.
7. No compact causal response-state, dynamical Markov state, autonomous controller, cache replacement or H3 claim is authorized.
""")
    freeze_names = [x.name for x in sorted((ROOT / "artifacts").glob("compact_causal_response_operator_v20*.freeze.json"))]
    write("EXECUTION_MANIFEST_V20.md", "V20 execution and integrity manifest", f"""
Base protocol digest: `{base['freeze_digest']}`; config SHA `{base['config_hash']}`; parent Git commit `{base['parent_commit']}`. State role hashes: `{json.dumps(split['role_id_sha256'], sort_keys=True)}`. Action partition hashes: `{json.dumps(actions['action_hashes'], sort_keys=True)}`. Operator q and action design digest: `{verify_stage(ROOT, 'operator_design')['freeze_digest']}`. q-scale freeze `{q['scale_freeze_digest']}`; diagnostic design `{diagnostic['design_digest']}`; aliasing design `{alias['design_digest']}`.

Frozen manifests ({len(freeze_names)}):
""" + "\n".join(f"- `{name}` — `{sha256_file(ROOT / 'artifacts' / name)}`" for name in freeze_names) + f"""

Development response bank: {train['response_rows']} train + {validation['response_rows']} validation rows. Independent V19-B bank: {confirmation['all_rows']} four-way rows. q-locality validation: {q['rows']} rows. Diagnostic action panel: {diagnostic['diagnostic_operator_states']} operator states. All output files and SHA256 values are indexed in `results/v20/processed/v20_integrity_index.json` after reports are built.

Executed module classes: `protocol_v20 freeze`; `operator_bank_v20 prepare`; `teacher_v20`; `independent_confirm_v20`; `actions_v20`; `q_locality_v20`; `response_operator_v20`; `operator_model_v20`; `operator_geometry_v20`; `operator_diagnostics_v20`; `operator_aliasing_v20`; `adjudicate_v20`. Sharded measurements ran on both GPUs. The frozen manifests and output records retain design hashes and completed counts.
""")
    answers = [
        f"Yes: independent V19-B, {confirmation['independent_states']} bases, M/R {num(active['median'])} (95% CI {num(active['ci95'][0])}–{num(active['ci95'][1])}).",
        f"Yes descriptively: at αq=0.25 all five selected q have reliable matched rate 1.0 and M/R {', '.join(num(x['median_M_R']) for x in qsmall)}.",
        "`Φ(P)` is the concatenation of 12 positive train-action 288-D response vectors; 18 directions × 2 signs were measured for diagnostics.",
        f"Train-positive fingerprint r95/energy curve is tabulated in OPERATOR_INTRINSIC_DIMENSION_V20.md; effective numerical rank {oracle['effective_train_fingerprint_rank']}.",
        "Family-wise finite operator r95 is available in the singular-spectra Parquet; no common rank is assumed.",
        "Natural and Pq empirical manifold dimensions are listed separately in NATURAL_VS_COUNTERFACTUAL_OPERATOR_V20.md.",
        "No oracle coordinate passed all frozen cross-action gates, although train-action coordinates were constructed.",
        "`k_operator_min` is not identified in k={2,4,8,16,32,64,128}.",
        f"Selected-model unseen-direction stack L2={num(held['stack_relative_l2'])}; preliminary gate={oracle['preliminary_operator_compactness_gate']}.",
        f"Unseen sign L2={num(sign['stack_relative_l2'])}; unseen direction+sign L2={num(both['stack_relative_l2'])}.",
        "Unseen amplitude results for train and validation directions are in ACTION_COMPOSITION_GENERALIZATION_V20.md.",
        "Two unseen pair actions and one dense mixture were measured; metrics and actuator audits are in ACTION_COMPOSITION_GENERALIZATION_V20.md.",
        f"Same-J operator mean principal angle median={num(geometry['median_pair_principal_angle_degrees'])}°.",
        f"Same-J median gain ratio Pq/P0={num(geometry['median_pair_gain_ratio'])}.",
        f"Same-J median r95 change={num(geometry['median_pair_rank_change'])}; full spectra and divergence are retained.",
        "V13 exact-JVP historical rank is compared, but paired V20-state JVP subspaces were not measured.",
        f"Natural J→oracle C validation R²={num(alias['natural_J_to_oracle_C_R2'])}.",
        f"Same J predicts identical C within each P0/Pq pair; counterfactual J→C relative L2={num(alias['counterfactual_J_to_oracle_C_relative_l2'])}.",
        f"Operational workspace-state aliasing observed={alias['workspace_state_aliasing_observed']} on {alias['same_J_pair_count']} exact-J pairs.",
        "Raw REC/Conv/KV→C was not eligible or trained after oracle gate failure.",
        "No V20 channel-wise raw-P→C recoverability ranking was measured.",
        "Same-J raw-P encoder response fidelity was not eligible or measured.",
        "Raw P incremental information after J+C+a was not measured.",
        "No practical-equivalence claim: raw gain and bootstrap CI were not measured.",
        "Held-out family-wise oracle errors are reported; no all-family compact-state gate passed.",
        f"Independent V20 final status: {adjudication['independent_V20_final_status']}.",
        "No compact causal response-state candidate was established.",
        "V21 dynamical-state search is not authorized.",
        "H2 remains the project-level interpretation.",
        "H3 is not authorized.",
        "Autonomous state-model training is not authorized.",
    ]
    write("V20_SCIENTIFIC_ANSWERS_V20.md", "V20 answers to the 31 scientific questions", "\n".join(
        f"{i}. {answer}" for i, answer in enumerate(answers, 1)) +
        f"\n\nFormal outcome: **{adjudication['formal_V20_outcome']}**. This is failure to identify compactness in the frozen tested model/action regime, not proof of mathematical nonexistence.")
    summary = f"""## V20 — Compact Causal Response Operator State

Independent V19-B confirmation passed on **{confirmation['independent_states']}** disjoint base states: same J, distinct P, matched finite action, M/R **{num(active['median'])}** (95% state-bootstrap CI **[{num(active['ci95'][0])}, {num(active['ci95'][1])}]**). At smaller reliable αq=0.25, all five selected active q retained nonzero descriptive modulation.

The crossed V20 response bank contains **{train['base_states']} training** and **{validation['base_states']} validation** base states, each with natural P0 plus three same-J Pq states, crossed with **18** shared measured directions and both signs; **6** final directions remain sealed. The train-action oracle fingerprint is `Φ_train(P)=[R_P(a1),…,R_P(a12)]`, with 288-D normalized response per action. Five models × k=2–128 were tested; best validation diagnostic model `{oracle['selected_model_for_diagnostics']}` at k={oracle['selected_k_for_diagnostics']} had unseen-direction stack relative L2 **{num(held['stack_relative_l2'])}**, unseen-sign **{num(sign['stack_relative_l2'])}**. The full frozen direction/sign/scale/composition gate did **not** pass; `k_operator_min` is **not identified**.

Finite operator pair geometry: median principal angle **{num(geometry['median_pair_principal_angle_degrees'])}°**, gain Pq/P0 **{num(geometry['median_pair_gain_ratio'])}**, r95 change **{num(geometry['median_pair_rank_change'])}**. Operational same-J workspace-state aliasing observed: **{alias['workspace_state_aliasing_observed']}**. Historical V13 JVP rank is not a paired V20 subspace test.

Formal result: **{adjudication['formal_V20_outcome']}**. Raw P→C encoder, conditional raw-gain equivalence, channel encoder audit and independent V20 final were **not eligible / unopened**, not passed or failed empirical tests. `V21_DYNAMIC_STATE_SEARCH_AUTHORIZED = FALSE`; H2 remains; H3 and autonomous training remain unauthorized. No physical cache replacement is licensed.

Protocol digest `{base['freeze_digest']}`; action hashes `{json.dumps(actions['action_hashes'], sort_keys=True)}`; V20 processed integrity index `results/v20/processed/v20_integrity_index.json`. See `reports/V20_COMPLETE_REPORT.md` for every standalone report in one file."""
    final = REPORTS / "FINAL_REPORT.md"
    current = final.read_text(encoding="utf-8")
    if "<!-- V20_START -->" in current or "<!-- V20_END -->" in current:
        raise RuntimeError("V20 cumulative FINAL_REPORT section already exists")
    for name, content in PENDING.items():
        (REPORTS / name).write_text(content, encoding="utf-8")
    final.write_text(current.rstrip() + "\n\n<!-- V20_START -->\n" + summary + "\n<!-- V20_END -->\n", encoding="utf-8")
    print(json.dumps({"standalone_reports_written": 15, "final_report_appended": True,
                      "formal_outcome": adjudication["formal_V20_outcome"]}, indent=2))


if __name__ == "__main__":
    main()
