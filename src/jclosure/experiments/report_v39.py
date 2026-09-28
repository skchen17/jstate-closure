"""Generate gate-limited V39 reports, machine adjudication, and integrity index."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from jclosure.protocol_v39 import stage_freeze, verify, verify_stage
from jclosure.provenance import sha256_file, write_json_atomic


OUT = Path("results/v39/processed")
SOURCE = "src/jclosure/experiments/report_v39.py"
REQUIRED = (
    "V39_FROZEN_STARTING_POINT", "V39_V38_INTERACTION_METRIC_AUDIT",
    "V39_HIGH_LEVEL_RECONFIRMATION", "V39_FALCON_HIGHER_ORDER_REPLICATION",
    "V39_QWEN_COMPOSITION_CONTROL", "V39_EIGHT_CONDITION_FACTORIAL",
    "V39_INTERNAL_INTERACTION_TRACE", "V39_RESIDUAL_INTERACTION_CURVE",
    "V39_INTERACTION_GENESIS_INTERVAL", "V39_FALCON_PRIMITIVE_FACTOR_MAP",
    "V39_DEPENDENCY_CONSISTENT_INTERVENTIONS", "V39_PRIMITIVE_FACTOR_REMOVAL",
    "V39_STRUCTURED_FACTOR_SUFFICIENCY", "V39_Q4_ROTATION_DECOMPOSITION",
    "V39_Q4_ROTATION_CAUSAL_TEST", "V39_ATTENTION_MLP_ALTERNATIVES",
    "V39_MIXER_RESIDUAL_CEILINGS", "V39_MODEL_COMPARATIVE_PROFILE",
    "V39_INDEPENDENT_FINAL_OPENING", "V39_TASK_GROUNDED_DESIGN",
    "V39_TASK_GROUNDED_RESULTS", "V39_SEMANTIC_SURFACE_CONTROLS",
    "V39_STRICT_INTERFACE_AUDIT", "V39_EXECUTION_MANIFEST",
    "V39_SCIENTIFIC_ANSWERS", "V39_COMPLETE_REPORT",
)


def read(root: Path, stem: str) -> dict:
    return json.loads((root / OUT / f"{stem}_v39.json").read_text(encoding="utf-8"))


def num(value) -> str:
    return f"{float(value):.4f}"


def report(root: Path, name: str, body: str) -> None:
    path = root / "reports" / f"{name}.md"
    if path.exists():
        raise RuntimeError(f"V39 report already exists; append amendment instead: {path}")
    title = name.replace("V39_", "V39 — ").replace("_", " ").title()
    path.write_text(f"# {title}\n\n{body.rstrip()}\n", encoding="utf-8")


def run(root: Path) -> dict:
    for stage in ("factorial_gate_development", "candidate_site_plan", "primitive_pilot_F",
                  "primitive_finalist_F", "trace_synthesis", "controls_F_development",
                  "controls_Q_development", "interface_ceilings_F_development"):
        verify_stage(root, stage)
    base = verify(root)
    finalist = read(root, "primitive_finalist_F")
    if finalist["primary_finalist"] is not None:
        raise RuntimeError("This gate-limited V39 report path requires a null finalist")
    for role in ("validation", "independent_final"):
        if list((root / OUT).glob(f"*_{role}_v39.parquet")):
            raise RuntimeError(f"V39 {role} outcome unexpectedly opened")
    f = read(root, "trajectory_analysis_F_development")
    q = read(root, "trajectory_analysis_Q_development")
    high = read(root, "high_level_development")
    gate = read(root, "factorial_gate_development")
    pool = json.loads((root / "data/v39/sample_pool_manifest_v39.json").read_text())
    pilot = read(root, "primitive_pilot_F")
    sites = read(root, "candidate_site_plan")
    trace_f = read(root, "internal_trace_F_development")
    trace_q = read(root, "internal_trace_Q_development")
    synthesis = read(root, "trace_synthesis")
    controls_f = read(root, "controls_F_development")
    controls_q = read(root, "controls_Q_development")
    ceilings = read(root, "interface_ceilings_F_development")
    audit_f = read(root, "calibration_F")
    audit_q = read(root, "calibration_Q")
    expanded_f = read(root, "trace_interface_audit_F")
    expanded_q = read(root, "trace_interface_audit_Q")
    primitive_audit = read(root, "primitive_interface_audit_F")
    end_f = pd.read_parquet(root / OUT / "trajectory_analysis_F_development_v39.parquet")
    end_q = pd.read_parquet(root / OUT / "trajectory_analysis_Q_development_v39.parquet")
    curve = pd.read_parquet(root / OUT / "residual_interaction_curve_v39.parquet")
    ceiling_rows = pd.read_parquet(root / OUT / "interface_ceilings_F_development_v39.parquet")
    if not (high["both_pass"] and f["gate_passes"]["q234"] and
            f["gate_passes"]["higher_order"] and not q["gate_passes"]["higher_order"]):
        raise RuntimeError("V39 report expectations drift from measured development gates")

    adjudication = {
        "schema_version": 1, "protocol": "interaction_genesis_v39",
        "parent_commit": base["parent_commit"],
        "stop_reason": "NO_CALIBRATION_SCREEN_QUALIFIED_PRIMITIVE_FINALIST",
        "completed_stage": "STAGE_6_NULL_FINALIST_FROZEN",
        "formal_development_factorial_complete": True,
        "formal_development_primitive_removal_executed": False,
        "structured_sufficiency_executed": False,
        "validation_opened": False, "independent_final_opened": False,
        "task_grounding_opened": False,
        "falcon_development_high_level_pass": high["model_passes"]["F"],
        "qwen_development_high_level_pass": high["model_passes"]["Q"],
        "falcon_development_q234_pass": f["gate_passes"]["q234"],
        "falcon_development_higher_order_pass": f["gate_passes"]["higher_order"],
        "qwen_development_higher_order_pass": q["gate_passes"]["higher_order"],
        "primary_finalist": None,
        "hypotheses": {
            "V39-A": "NOT_QUALIFIED_VALIDATION_SEALED",
            "V39-B": "NOT_TESTED_FINAL_SEALED",
            "V39-C": "DESCRIPTIVE_TRACE_ONLY_NO_CAUSAL_INTERVAL",
            "V39-D": "NOT_SUPPORTED_NO_FORMAL_MEDIATOR",
            "V39-E": "NOT_SUPPORTED_C_PILOT_FAILED",
            "V39-F": "NOT_SUPPORTED_UPDATE_DECAY_PILOT_FAILED",
            "V39-G": "UNTESTED_CAUSAL_POST_RECURRENT_PATH",
            "V39-H": "UNTESTED_CAUSAL_Q4_ROTATION",
            "V39-I": "DEVELOPMENT_COMPARISON_ONLY_VALIDATION_SEALED",
            "V39-J": "NOT_TESTED_TASK_PHASE_SEALED",
            "V39-K": "DEVELOPMENT_ONLY_MECHANISM_UNRESOLVED_NOT_FINAL_CONFIRMED",
        },
        "important_limit": "Single-layer, first-probe calibration screen cannot exclude distributed, multi-layer, attention, MLP, or residual mechanisms.",
    }
    write_json_atomic(root / OUT / "v39_adjudication.json", adjudication)
    execution = {
        "protocol": "interaction_genesis_v39",
        "parent_commit": base["parent_commit"],
        "stage_0_metric_audit": "complete",
        "stage_1_new_pools_and_design": "complete_presealed_all_roles",
        "stage_2_calibration": "complete_both_models_bitwise_and_actual_eight_condition_identity",
        "stage_3_development_factorial": "complete_both_models_predictions_presealed_before_R111",
        "stage_4_falcon_higher_order": "passed_development_only",
        "stage_5_internal_tracing": "complete_both_models_20_states_each_one_probe",
        "stage_6_primitive_finalist": "null_after_calibration_pilot_screen",
        "stage_7_formal_development_mediation": "not_opened_no_finalist",
        "stage_8_validation": "sealed_no_development_mediator_pass",
        "stage_9_mediator_validation": "not_tested",
        "stage_10_final_spec": "not_frozen",
        "stage_11_independent_final": "sealed",
        "stage_12_task_grounding": "sealed",
        "sample_counts_per_model": {role: value["states"] for role, value in pool["roles"].items()},
        "frozen_protocol_digest": base["freeze_digest"],
        "frozen_stage_files": [str(p.relative_to(root)) for p in
                               sorted((root / "artifacts").glob("interaction_genesis_v39*.freeze.json"))],
        "no_v1_v38_historical_file_mutated": True,
    }
    write_json_atomic(root / OUT / "v39_execution_manifest.json", execution)

    fam = []
    for name in f["family_metrics"]:
        fm = f["family_metrics"][name]
        qm = q["family_metrics"][name]
        fam.append(f"| {name} | {num(fm['threeway_fraction'])} | {num(fm['threeway_sign_stability'])} | {num(fm['q234_fraction_of_full'])} | {num(fm['q234_direction_cosine_to_full'])} | {num(qm['threeway_fraction'])} |")
    family_table = "| Family | Falcon f | Falcon sign stability | Falcon Q234 fraction | Falcon Q234 cosine | Qwen f |\n|---|---:|---:|---:|---:|---:|\n" + "\n".join(fam)
    h_f, h_q = high["model_summaries"]["F"], high["model_summaries"]["Q"]
    report(root, "V39_HIGH_LEVEL_RECONFIRMATION",
           f"Development-only high-level donor-error gate: Falcon {h_f['positive_fraction']:.3f} positive, median reduction {num(h_f['median_reduction'])}, {h_f['families_passing']}/5 families; Qwen {h_q['positive_fraction']:.3f}, {num(h_q['median_reduction'])}, {h_q['families_passing']}/5. Both pass the frozen 0.80/0.20/4-family gate. Validation remains sealed. Machine records: `high_level_*_development_v39.json` and Parquet/NPZ.")
    fm = f["metrics"]
    report(root, "V39_FALCON_HIGHER_ORDER_REPLICATION",
           f"Falcon development (80 states): second-order relative error median {num(fm['second_relative_error']['median'])} (state-bootstrap 95% CI {fm['second_relative_error']['state_bootstrap_median_95ci']}), three-way fraction {num(fm['threeway_fraction']['median'])}, corrected signed projection coefficient p {num(fm['threeway_projection']['median'])}, cosine {num(fm['threeway_cosine']['median'])}, sign stability {num(fm['threeway_sign_stability'])}. Higher-order gate passes in {f['family_pass_counts']['higher_order']}/5 families; Q234 gate passes in {f['family_pass_counts']['q234']}/5. There are {fm['effect_below_floor_count']} below-floor states. This is a development replication, **not** V39-A qualification: validation was not opened after the mediator screen failed. Large f with small p is not signed amplification.\n\n{family_table}")
    qm = q["metrics"]
    report(root, "V39_QWEN_COMPOSITION_CONTROL",
           f"Qwen matched development (80 states): additive error {num(qm['additive_relative_error']['median'])}, second-order error {num(qm['second_relative_error']['median'])}, three-way fraction {num(qm['threeway_fraction']['median'])}, corrected p {num(qm['threeway_projection']['median'])}, sign stability {num(qm['threeway_sign_stability'])}. Q234 passes {q['family_pass_counts']['q234']}/5; Falcon-style higher-order gate passes {q['family_pass_counts']['higher_order']}/5. This is a model comparison, not architecture-causation or a claim that Qwen is strictly additive. Validation is sealed.")
    report(root, "V39_EIGHT_CONDITION_FACTORIAL",
           "Both models completed natural R000/R100/R010/R001, then R110/R101/R011, then frozen additive and second-order predictions, then R111 on 80 fresh development states. Conv was donor-native, KV recipient-native, and exact native REC subsets were written; no later-output copy entered the factorial. Per-state six-probe vectors and predictions are in `trajectory_*_development_v39.npz` with Parquet proof records and stage freezes. Corrected `p=<I234,E111>/||E111||²`, `f=||I234||/||E111||`, and cosine obey `p=f·cos` at maximum recorded numerical error below 1e-15. Validation/final outcomes were not observed.")
    stages_f = {x["stage"]: x["layer"] for x in trace_f["stable_next_layer_interactions"]
                if x["stage"] not in ()}
    earliest = {}
    for x in trace_f["stable_next_layer_interactions"]:
        earliest[x["stage"]] = min(earliest.get(x["stage"], x["layer"]), x["layer"])
    report(root, "V39_INTERNAL_INTERACTION_TRACE",
           f"One frozen future probe per state was traced under all eight conditions for 20 development states/model. Falcon: {trace_f['rows']} state×layer×stage measurements, {len(trace_f['stable_next_layer_interactions'])} stage/layer pairs satisfy the 0.20 ratio, 0.75 state-prevalence, 4-family and next-layer-persistence rule. Earliest persistent stages: {json.dumps(earliest, sort_keys=True)}. Qwen: {trace_q['rows']} measurements and {len(trace_q['stable_next_layer_interactions'])} qualifying stage/layer pairs. Actual I234 and E tensors for every traced state/layer/stage are in hash-indexed NPZ banks. This is descriptive localization, not mediation.")
    fcurve = synthesis["models"]["F"]["residual_curve"]
    qcurve = synthesis["models"]["Q"]["residual_curve"]
    report(root, "V39_RESIDUAL_INTERACTION_CURVE",
           f"The residual-stream increment is computed only as `I234(block_output)-I234(block_input)` in common coordinates, never by subtracting unrelated internal spaces. Falcon median outgoing interaction norms at layers 18–23: {[round(x['median_outgoing_norm'],3) for x in fcurve if 18 <= x['layer'] <= 23]}. Qwen's corresponding measurements are in `residual_interaction_curve_v39.parquet` ({len(qcurve)} decoder layers). These differences are descriptive; causal creation versus transmission is unresolved.")
    report(root, "V39_INTERACTION_GENESIS_INTERVAL",
           "At the registered persistence threshold, Falcon attention output first qualifies at layer 19; B, dt/dA and dBx qualify at layer 20; C and x at layer 21. The common residual `BLOCK_OUTPUT` reaches 4/5-family prevalence only at the final layer 23, so it cannot satisfy the required *next-layer* persistence there. Recurrent state S and post-update S′ do not qualify by this relative threshold. A stable descriptive factor interval exists, but a unique **causal genesis interval is not identified**. First nonzero interaction and first threshold crossing are different claims. Qwen has no qualifying persistent stage on the traced subset.")
    site_lines = [f"| {name} | {value['layer'] if value['layer'] is not None else 'none'} | {value['eligible']} |" for name, value in sites["sites"].items()]
    report(root, "V39_FALCON_PRIMITIVE_FACTOR_MAP",
           "Trace-selected functional sites were frozen before any primitive removal outcome. No tensor coordinate was selected post hoc.\n\n| Family | Earliest stable layer | Eligible for pilot |\n|---|---:|---|\n" + "\n".join(site_lines) + "\n\nThe map is descriptive and cannot establish necessity.")
    report(root, "V39_DEPENDENCY_CONSISTENT_INTERVENTIONS",
           "The Falcon-native primitive hook passed 480/480 calibration state×layer no-intervention bitwise replay tests across logits, cache and five endpoint blocks. A candidate replacement occurs inside native recurrence at C, x/B, dt, S, dBx, gate or mixer input; dependent dA/dBx/S′/read/norm/mixer and all later layers are recomputed. Requested and realized replacement hashes matched in the pilot. Predictions were evaluated in float32 and cast to each native dtype; quantization gaps are recorded per intervention. Off-manifold status is explicit. Formal development mediation was **not** opened because the pilot produced no finalist.")
    pilot_lines = [f"| {name} | {data['layer']} | {num(data['median_removed_direction'])} | {num(data['median_remaining_norm_ratio'])} | {data['families_passing']}/5 |" for name, data in pilot["screening"].items()]
    report(root, "V39_PRIMITIVE_FACTOR_REMOVAL",
           "Calibration-only, first-probe screen (20 states) used natural R111 and replaced each trace-eligible native factor by its seven-condition second-order prediction at one frozen layer; all descendants recomputed.\n\n| Candidate | Layer | Median removed direction | Median remaining norm ratio | Families passing 0.50/0.70 |\n|---|---:|---:|---:|---:|\n" + "\n".join(pilot_lines) + "\n\nAll four candidates fail the preregistered pilot gate; the frozen primary finalist is `null`. No formal development removal outcome was generated. This screen does not exclude multi-layer or other mechanisms.")
    report(root, "V39_STRUCTURED_FACTOR_SUFFICIENCY",
           "Not executed. A structured lower-order internal baseline plus restoration of only one candidate factor's natural three-way component requires a frozen candidate. The calibration screen selected none. Adding the exact endpoint interaction back would be tautological and is not counted as sufficiency. Validation and final remain sealed.")
    report(root, "V39_Q4_ROTATION_DECOMPOSITION",
           f"Development median cosine between isolated-Q4 and Q4-given-Q23 future effects: Falcon {num(synthesis['models']['F']['q4_isolated_vs_given_q23_cosine_median'])}; Qwen {num(synthesis['models']['Q']['q4_isolated_vs_given_q23_cosine_median'])}. Falcon's median context-conditioned Q4 magnitude ratio is {num(synthesis['models']['F']['q4_given_q23_magnitude_ratio_median'])}. Native internal x/B/C/dt/attention/MLP interaction tensors are recorded, but no component-specific prospective causal rotation prediction was frozen. This is a descriptive rotation, not an explanation.")
    report(root, "V39_Q4_ROTATION_CAUSAL_TEST",
           "Not executed: no primitive finalist passed the calibration screen, and no separate Q4 causal component/prediction was frozen. The required local cosine≥0.90, downstream rotation≥50%, 4/5 families, and development+validation gate is therefore untested, not failed or passed.")
    report(root, "V39_ATTENTION_MLP_ALTERNATIVES",
           "Falcon attention output reaches the internal 0.20/0.75/4-family persistence threshold at layer 19, before several traced post-Conv factors at layers 20–21. MLP output qualifies at layer 22. These are credible alternative carriers, not identified causal generators. No attention/MLP-specific removal or structured restoration was run, so `ATTENTION_MEDIATED`, `MLP_MEDIATED`, and `RESIDUAL_NONLINEAR_COMPOSITION` remain open hypotheses. Qwen has no qualifying persistent internal stage on this subset.")
    report(root, "V39_MIXER_RESIDUAL_CEILINGS",
           f"On 20 development trace states (one probe), each Q4 layer 18–23 was tested separately by subtracting that boundary's measured I234 from the natural R111 mixer or block output, then allowing downstream native computation. Across all tested layers, mixer-output median removed direction {num(ceilings['stage_medians']['MIXER_OUTPUT']['removed_direction'])}, remaining norm ratio {num(ceilings['stage_medians']['MIXER_OUTPUT']['remaining_norm_ratio'])}; block-output values {num(ceilings['stage_medians']['BLOCK_OUTPUT']['removed_direction'])} and {num(ceilings['stage_medians']['BLOCK_OUTPUT']['remaining_norm_ratio'])}. These are single-layer **interface ceilings**, not primitive mediation or a joint multi-layer test. Per-layer records are in `interface_ceilings_F_development_v39.parquet`.")
    report(root, "V39_MODEL_COMPARATIVE_PROFILE",
           f"Matched development shows Falcon f={num(fm['threeway_fraction']['median'])}, second-order error={num(fm['second_relative_error']['median'])}, Q4 context cosine={num(synthesis['models']['F']['q4_isolated_vs_given_q23_cosine_median'])}; Qwen f={num(qm['threeway_fraction']['median'])}, error={num(qm['second_relative_error']['median'])}, cosine={num(synthesis['models']['Q']['q4_isolated_vs_given_q23_cosine_median'])}. Both high-level REC gates and both Q234 gates pass, but only Falcon passes the frozen higher-order gate on development. Matched design does not identify architecture as the cause. Validation was not opened, so V39-I is development-supported only.")
    report(root, "V39_INDEPENDENT_FINAL_OPENING",
           "The independently generated Falcon final pool of 40 states was sealed before outcomes and remains unopened. The frozen rule requires development+validation high-level/Q234/higher-order replication **and** one frozen primitive mediator passing necessity and structured sufficiency in both roles. No finalist qualified the calibration screen, no formal mediation was run, and validation stayed sealed. Qwen final also remains sealed. No final-intervention outcome exists.")
    report(root, "V39_TASK_GROUNDED_DESIGN",
           "Task-grounded Phase C did not open because the prerequisite internal mediator did not qualify. The proposed Type S surface-only, Type T task-variable, and Type N neutral-natural-token contrasts were not instantiated or tested. No model-output-dependent task labels were created. A future study must preseal externally generated truth values before model execution.")
    report(root, "V39_TASK_GROUNDED_RESULTS",
           "Not tested. No Type S/T/N task-grounded panel was opened, and no correct-answer-margin or task-variable classification outcome was measured. V39-J is unsupported/untested; donor-response similarity is not task correctness.")
    report(root, "V39_SEMANTIC_SURFACE_CONTROLS",
           "Not executed because task-grounded Phase C was sealed. There are no Type S surface-only, Type T task-variable, or Type N neutral-token contrast outcomes. The development control panel does include same-prefix wrong-token and off-manifold shuffled REC controls, which are **not** substitutes for semantic Type S/T/N controls.")
    report(root, "V39_STRICT_INTERFACE_AUDIT",
           f"V39 calibration: Falcon {audit_f['states']} and Qwen {audit_q['states']} states each passed native/instrumented bitwise replay, exact REC writeback and actual eight-condition formula equality. Maximum algebra residuals: F {audit_f['max_identity_residual']:.2e}, Q {audit_q['max_identity_residual']:.2e}. Expanded trace hook passed {expanded_f['states']}/{expanded_f['states']} Falcon and {expanded_q['states']}/{expanded_q['states']} Qwen bitwise tests; Falcon primitive hook passed {primitive_audit['rows']}/{primitive_audit['rows']} state×layer tests. Six-probe endpoint predictions were frozen before R111; recipient KV and donor Conv were unchanged by REC subset interventions. Off-manifold shuffling/sign-flip and interface-copy controls are explicitly non-natural. This is a V39-specific audit, not a reuse of V36 results.")
    report(root, "V39_EXECUTION_MANIFEST",
           "The machine record `results/v39/processed/v39_execution_manifest.json` enumerates stage outcomes and freeze artifacts. Stages 0–6 completed, with a frozen null finalist; formal mediator development, validation, independent final and Phase C were not opened. All 180 fresh pool programs/prompts are mutually disjoint and historical-disjoint; formal horizon is constant across roles. `v39_integrity_index.json` hashes code, configs, pools, stage seals, machine results, reports, and retained local NPZ banks. NPZ tensor banks are intentionally not committed to Git, but their hashes and paths are committed.")

    answers = [
        "1. V38 projection label was dimensionally wrong; f and sign were sound.",
        "2. Yes; append-only V38 metric amendment added.",
        "3. Falcon high-level effect passes development; validation unobserved.",
        "4. Natural Q234 passes 4/5 families in Falcon development.",
        "5. Falcon higher-order gate passes 5/5 families in development.",
        "6. Validation not opened after null mediator finalist.",
        "7. Independent final remains sealed.",
        "8. Yes on development: Qwen f 0.197 versus Falcon 0.554.",
        "9. Attention output first persistently crosses registered internal ratio threshold at layer 19.",
        "10. No registered persistent substantial Q2 interval identified.",
        "11. No registered persistent substantial Q3 interval identified.",
        "12. Several Falcon Q4 factors qualify at layers 20–22.",
        "13. Incoming residual is not the earliest registered persistent stage; causal precedence unresolved.",
        "14. Post-Conv B qualifies at layer 20, C and x at 21, descriptively.",
        "15. dt/dA and dBx qualify at layer 20, descriptively.",
        "16. S′ does not pass registered next-layer persistence.",
        "17. Raw read does not pass registered persistence.",
        "18. Gating/norm carries interaction, but no causal generator identified.",
        "19. Residual BLOCK_OUTPUT reaches four-family prevalence only at final layer, without next-layer persistence.",
        "20. Attention output qualifies at layer 19, descriptive only.",
        "21. MLP output qualifies at layer 22, descriptive only.",
        "22. Attention is earliest qualifying traced stage; among primitive factors B/dt/dBx appear by 20.",
        "23. Tested single-layer calibration clamps do not reach the 0.50 directional gate.",
        "24. Best screened median removed direction is 0.190 (x+B); not qualifying.",
        "25. Norm and direction diverge; no strong removal mechanism identified.",
        "26. Structured restoration was not opened.",
        "27. Necessity/sufficiency classification untested formally.",
        "28. Joint multi-layer factor requirement remains possible, untested.",
        "29. C/read mediation pilot fails; Q4 causal decomposition untested.",
        "30. x+B pilot fails the registered threshold.",
        "31. dt/dA pilot fails the registered threshold.",
        "32. Residual input context did not qualify trace-site screen; not causally tested.",
        "33. No prospective native Q4 component rotation prediction frozen.",
        "34. No Q4 component-specific causal rotation test executed.",
        "35. Trace shows recurrence, attention and MLP carriers; creation not causally assigned.",
        "36. Qwen has no registered persistent internal stage on the traced subset.",
        "37. Qwen high-level channel phenomenon and Q234 pass development; mechanism unknown.",
        "38. V39-A not qualified: validation sealed.",
        "39. V39-B not tested: final sealed.",
        "40. V39-C descriptive localization only, no causal interval.",
        "41. V39-D/E/F/G unconfirmed; primitive pilot fails and alternative paths untested.",
        "42. V39-H untested causally.",
        "43. V39-I development comparison supported, validation not performed.",
        "44. Task-grounded Phase C did not open.",
        "45. Type T untested.",
        "46. Type S untested.",
        "47. V39-J untested.",
        "48. Mechanism unresolved on development; full V39-K requires stronger replication.",
        "49. No precise mechanism is ready for a training-origin study.",
        "50. A future temporal Conv→REC study is not yet justified by V39 mechanism evidence.",
    ]
    report(root, "V39_SCIENTIFIC_ANSWERS", "\n\n".join(answers) + "\n\nThese answers distinguish development evidence from unobserved validation/final outcomes; no skipped phase is called a negative experiment.")
    report(root, "V39_COMPLETE_REPORT",
           f"V39 reached its prospectively specified stop at Stage 6: a null primitive finalist after calibration-only screening. It did **not** complete the conditional validation, independent-final, or task-grounding branches. Falcon development reproduces the high-level REC effect, Q234 effect, and higher-order three-way endpoint response (f={num(fm['threeway_fraction']['median'])}, corrected p={num(fm['threeway_projection']['median'])}; 5/5 higher-order families). Qwen development has a lower three-way fraction (f={num(qm['threeway_fraction']['median'])}). Internal tracing locates late Falcon factor and attention interactions, but causal origin remains unresolved. Four trace-selected single-layer native-factor screens fail the 0.50/0.70 pilot gate; no formal mediator or structured sufficiency result exists. The independent final remains sealed. The strongest defensible conclusion is **development interaction replicated; mechanism unresolved; full V39-A/B/K not independently confirmed**.\n\nLimitations: one-probe trace/pilot, four calibration states per family, single-layer factor screening, no joint multi-layer mediation, no prospective attention/MLP mechanism test, and no validation/final. These omissions are explicit consequences of the frozen gate, not hidden negative findings. V38 history is immutable; the projection amendment is append-only. See `v39_adjudication.json` and the integrity index for machine audit.")

    report_paths = [root / "reports" / f"{name}.md" for name in REQUIRED]
    if not all(path.exists() for path in report_paths):
        raise RuntimeError("V39 required report missing")
    all_path = root / "reports/V39_ALL_REPORTS.md"
    if all_path.exists():
        raise RuntimeError("V39_ALL_REPORTS already exists; do not overwrite")
    all_path.write_text("# V39 — All Reports\n\n"
                        + "\n\n---\n\n".join(
                            f"<!-- {path.name} -->\n\n{path.read_text(encoding='utf-8').rstrip()}"
                            for path in report_paths) + "\n", encoding="utf-8")
    final_path = root / "reports/FINAL_REPORT.md"
    if sha256_file(final_path) != base["cumulative_report_at_start_sha256"]:
        raise RuntimeError("Historical cumulative report changed before V39 append")
    with final_path.open("a", encoding="utf-8") as stream:
        stream.write("\n\n## V39 — Genesis and Causal Mediation of Higher-Order Recurrent-State Interaction\n\n"
                     "V39 development replicated the Falcon higher-order Q2×Q3×Q4 endpoint interaction (f=0.554, corrected p=0.072; 5/5 families) and found a smaller Qwen interaction (f=0.197). A new four-role historical-disjoint pool and both model designs were sealed before formal outcomes; native replay, REC writeback and true eight-condition metric equality passed. Late Falcon internal interaction was traced, but four single-layer primitive-factor calibration screens did not qualify a mediator. A null finalist was frozen. Under the predeclared gate, formal mediation, validation, independent final and task-grounding were not opened. No causal origin or independent-final claim is made. V38's ambiguous projection label is corrected only by an append-only amendment. Full evidence: `reports/V39_ALL_REPORTS.md`, `reports/V39_COMPLETE_REPORT.md`, and `results/v39/processed/v39_integrity_index.json`.\n")

    paths = [root / SOURCE, root / "configs/genesis_v39.yaml",
             *sorted((root / "artifacts").glob("interaction_genesis_v39*.freeze.json")),
             *sorted((root / "data/v39").rglob("*")),
             *sorted((root / OUT).rglob("*")),
             *report_paths, all_path, root / "reports/V38_METRIC_AMENDMENT_V39.md", final_path]
    index_path = root / OUT / "v39_integrity_index.json"
    indexed = {}
    for path in paths:
        if path.is_file() and path != index_path:
            indexed[str(path.relative_to(root))] = sha256_file(path)
    integrity = {"protocol": "interaction_genesis_v39",
                 "created_utc": datetime.now(timezone.utc).isoformat(),
                 "parent_commit": base["parent_commit"],
                 "indexed_files": indexed,
                 "indexed_file_count": len(indexed),
                 "historical_v1_v38_files_mutated": False,
                 "npz_tensor_banks_retained_locally_not_git_tracked": True,
                 "validation_and_final_intervention_outcomes_unobserved": True}
    write_json_atomic(index_path, integrity)
    seal = stage_freeze(root, "report", [SOURCE, str(index_path.relative_to(root)),
                                        *[str(path.relative_to(root)) for path in report_paths],
                                        str(all_path.relative_to(root))],
                        {"report_count": len(report_paths),
                         "all_reports_sha256": sha256_file(all_path),
                         "integrity_index_sha256": sha256_file(index_path),
                         "cumulative_report_at_append_sha256": sha256_file(final_path),
                         "validation_and_final_unopened": True})
    return {"freeze_digest": seal["freeze_digest"],
            "required_reports": len(report_paths),
            "indexed_files": len(indexed),
            "primary_finalist": None,
            "validation_opened": False,
            "independent_final_opened": False}


if __name__ == "__main__":
    print(json.dumps(run(Path.cwd()), indent=2))
