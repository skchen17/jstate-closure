"""Build the V38 topic reports, one-file report, and machine integrity index."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from jclosure.protocol_v38 import stage_freeze, verify, verify_stage
from jclosure.provenance import sha256_file, write_json_atomic


OUT=Path("results/v38/processed")
SOURCE="src/jclosure/experiments/report_v38.py"
ROLES=("development","validation")
KEYS=("Q","F")
TOPICS=(
"FROZEN_STARTING_POINT","HIGH_LEVEL_RECONFIRMATION","Q234_NATURAL_REPLICATION",
"FACTORIAL_DESIGN","EIGHT_CONDITION_REC_FACTORIAL","ADDITIVE_PREDICTION",
"PAIRWISE_INTERACTIONS","SECOND_ORDER_PREDICTION","THREE_WAY_INTERACTION",
"COMPOSITION_CLASS_ADJUDICATION","INCREMENTAL_NATURAL_ACCUMULATION",
"CONTEXT_DEPENDENT_MARGINAL_EFFECTS","MARGINAL_EFFECT_ROTATION",
"CROSS_MODEL_COMPOSITION","Q1_CONTROL","TRAJECTORY_INTERNAL_TRACE",
"DOSE_RESPONSE","JVP_FINITE_COMPARISON","TASK_GROUNDED_DESIGN",
"TASK_GROUNDED_RESULTS","SEMANTIC_SURFACE_CONTROLS","CHANNEL_TASK_FUNCTION",
"TEMPORAL_DELAY_DESIGN","TEMPORAL_CHANNEL_PROFILE","STRICT_INTERFACE_AUDIT",
"EXECUTION_MANIFEST","SCIENTIFIC_ANSWERS")


def j(root,name):return json.loads((root/OUT/name).read_text())
def f(x):return f"{float(x):.3f}"
def table(headers,rows):return "|"+"|".join(headers)+"|\n|"+"|".join("---" for _ in headers)+"|\n"+"\n".join("|"+"|".join(map(str,row))+"|" for row in rows)+"\n"
def med(s,name):return s["metrics"][name]["median"]


def run(root:Path)->dict:
    base=verify(root)
    verify_stage(root,"final_closed")
    for key in KEYS:
        for role in ROLES:
            for stage in ("calibration_"+key,"high_level_"+key+"_"+role,
                          "trajectory_singles_"+key+"_"+role,
                          "trajectory_pairs_"+key+"_"+role,
                          "trajectory_predict_"+key+"_"+role,
                          "trajectory_full_"+key+"_"+role,
                          "trajectory_analysis_"+key+"_"+role,
                          "controls_"+key+"_"+role,
                          "task_diagnostic_"+key+"_"+role):
                verify_stage(root,stage)
    verify_stage(root,"dose_F_development")
    verify_stage(root,"task_design")
    for key in KEYS:verify_stage(root,f"trajectory_trace_{key}_development")
    pool=json.loads((root/"data/v38/sample_pool_manifest_v38.json").read_text())
    comp={role:j(root,f"composition_{role}_v38.json") for role in ROLES}
    high={(key,role):j(root,f"high_level_{key}_{role}_v38.json") for key in KEYS for role in ROLES}
    ana={(key,role):j(root,f"trajectory_analysis_{key}_{role}_v38.json") for key in KEYS for role in ROLES}
    frames={(key,role):pd.read_parquet(root/OUT/f"trajectory_analysis_{key}_{role}_v38.parquet")
            for key in KEYS for role in ROLES}
    ctr={(key,role):j(root,f"controls_{key}_{role}_v38.json") for key in KEYS for role in ROLES}
    ctrframes={(key,role):pd.read_parquet(root/OUT/f"controls_{key}_{role}_v38.parquet")
               for key in KEYS for role in ROLES}
    task={(key,role):j(root,f"task_diagnostic_{key}_{role}_v38.json") for key in KEYS for role in ROLES}
    dose=j(root,"dose_F_development_v38.json")
    cfg=base["config"]
    docs={}
    def add(name,body):docs[name]="# V38 — "+name.replace("_"," ").title()+"\n\n"+body.strip()+"\n"
    hrows=[]; arows=[]; qrows=[]; pairs=[]; margins=[]; crows=[]; trows=[]
    for key in KEYS:
        for role in ROLES:
            H,A,C,T=high[key,role],ana[key,role],ctr[key,role],task[key,role]
            D=frames[key,role]
            hrows.append((key,role,H["states"],f(H["positive_fraction"]),f(H["median_reduction"]),H["families_passing"]))
            arows.append((key,role,f(med(A,"additive_relative_error")),f(med(A,"additive_cosine")),
                          f(med(A,"second_relative_error")),f(med(A,"second_cosine")),
                          f(med(A,"relative_error_improvement")),A["family_pass_counts"]))
            fm=A["family_metrics"]
            qfam=sum(x["q234_fraction_of_full"]>=cfg["q234_reconstruction_gate"]["median_fraction_min"]
                     and x["q234_direction_cosine_to_full"]>=cfg["q234_reconstruction_gate"]["median_cosine_min"]
                     for x in fm.values())
            qrows.append((key,role,f(med(A,"q234_fraction_of_full")),
                          f(med(A,"q234_direction_cosine_to_full")),qfam))
            pairs.append((key,role,*[f(D[x].median()) for x in
                           ("pair23_fraction","pair24_fraction","pair34_fraction",
                            "pair23_projection","pair24_projection","pair34_projection")]))
            margins.append((key,role,f(D.q3_isolated_vs_given_q2_cosine.median()),
                            f(D.q4_isolated_vs_given_q23_cosine.median()),
                            f(D.q3_given_q2_magnitude_ratio.median()),
                            f(D.q4_given_q23_magnitude_ratio.median())))
            crows.append((key,role,C["states"],f(ctrframes[key,role].q234_error.median()),
                          f(ctrframes[key,role].q1234_error.median()),
                          f((ctrframes[key,role].q234_error-ctrframes[key,role].q1234_error).median()),
                          f(ctrframes[key,role].q1_increment_norm.median())))
            trows.append((key,role,T["valid_states"],f(T["positive_margin_change_fraction"]),
                          f(T["median_margin_change"]),f(T["exact_answer_accuracy"]["R000"]),
                          f(T["exact_answer_accuracy"]["R111"])))
    high_table=table(("model","role","n","positive","median reduction","families"),hrows)
    pred_table=table(("model","role","add err","add cos","2nd err","2nd cos","improve","family passes"),arows)
    q_table=table(("model","role","Q234/full benefit","cos to full","families ≥ thresholds"),qrows)
    pair_table=table(("model","role","||I23||/||E||","||I24||/||E||","||I34||/||E||",
                      "proj I23","proj I24","proj I34"),pairs)
    margin_table=table(("model","role","cos Q3 alone|Q2","cos Q4 alone|Q23",
                        "Q3 magnitude ratio","Q4 magnitude ratio"),margins)
    control_table=table(("model","role","n","Q234 donor error","all Q donor error",
                         "paired Q1 error gain","Q1 vector increment"),crows)
    task_table=table(("model","role","valid","margin positive","median margin Δ",
                      "R000 accuracy","R111 accuracy"),trows)
    add("FROZEN_STARTING_POINT",f"Parent commit `{base['parent_commit']}`; V1–V37 reports/results unchanged and V37 outcomes excluded from V38 formal decisions. New pools contain calibration/development/validation/final = 20/80/40/40 states per model, five families, all cross-role program/prompt-disjoint and historically disjoint. Validation/final were sealed before intervention responses. Pool hashes and model/checkpoint/tokenizer hashes are in `artifacts/trajectory_composition_v38.freeze.json` and `data/v38/sample_pool_manifest_v38.json`.\n\nImportant distribution shift: modular arithmetic uses horizon 2 in calibration/development and horizon 3 in validation/final, after a sealed, outcome-blind amendment. This limits same-distribution interpretation. V18 training leftovers were not used in formal roles.")
    add("HIGH_LEVEL_RECONFIRMATION","Fresh donor Conv + donor REC versus donor Conv with recipient REC, with recipient-native KV and six-probe endpoints:\n\n"+high_table+"\nAll four model×role cells pass ≥0.80 positive, ≥0.20 median reduction and ≥4 families. REC-only weakness on the frozen control subset is separately shown in `V38_STRICT_INTERFACE_AUDIT.md`.")
    add("Q234_NATURAL_REPLICATION","R111 is exact native Q2+Q3+Q4 donor REC writeback on donor Conv; no future outputs copied. Recovery is donor-error gain relative to the full REC+Conv gain (states with benefit >1).\n\n"+q_table+"\nAggregate medians are strong in both models/roles. The additionally frozen 4/5 family Q234 gate fails for F development (3/5), so broad family-uniform replication must not be claimed for that cell.")
    add("FACTORIAL_DESIGN","Fixed background: donor Conv, recipient REC, recipient-native KV. Q2/Q3/Q4 are the 7–12/13–18/19–24 recurrent-layer groups. Conditions: `R000`, `R100`, `R010`, `R001`, `R110`, `R101`, `R011`, `R111`; group membership and six probes are in `execution_plan_v38.json` and model designs. All algebra uses state-wise six-probe vectors.\n\n`Eabc=Yabc−Y000`; additive=`E100+E010+E001`; second-order=`E110+E101+E011−E100−E010−E001`; three-way=`E111−E110−E101−E011+E100+E010+E001`. The pasted expressions contained typographic `*` artifacts; the standard inclusion–exclusion formulas were frozen before outcomes.")
    add("EIGHT_CONDITION_REC_FACTORIAL","Every development (80/model) and validation (40/model) state has all eight REC conditions. Before R111, singles and pairs were sealed and both predictions hashed. Each condition uses exact native REC-state replacement only; Conv/KV were checked unchanged, and all later computations evolved natively. State-wise outputs are in `trajectory_*_v38.parquet` and local ignored `.npz` arrays; per-condition REC hashes are in the `proof_json` column.")
    add("ADDITIVE_PREDICTION","Prospective additive prediction of R111 from unopened singles:\n\n"+pred_table+"\nQ has low median additive error but material second-order improvement, so it does not pass the registered additive class. F additive error/cosine fail directly. Bootstrap CIs are stored per model/role in `trajectory_analysis_*_v38.json`.")
    add("PAIRWISE_INTERACTIONS","Exact state-wise pair terms and donor-effect-direction projections (negative projection is cancelling, not amplifying):\n\n"+pair_table+"\nF shows large, mostly cancelling pair contributions; Q pair terms are smaller. The largest median norm term is I23 in both models, but no serial-circuit interpretation follows. Full distributions and per-family metrics are machine-readable.")
    add("SECOND_ORDER_PREDICTION","Predictions were sealed before R111.\n\n"+pred_table+"\nQ second-order errors are low, but the ≥30% improvement and ≥4-family pairwise gate is not met (family counts 3/5 development, 2/5 validation). F remains well above 0.30 relative error. Thus no model satisfies the formal pairwise class in both roles.")
    add("THREE_WAY_INTERACTION",table(("model","role","median fraction","median projection","sign stability","higher family pass"),
              [(key,role,f(med(ana[key,role],"threeway_fraction")),
                f(med(ana[key,role],"threeway_projection")),
                f(ana[key,role]["metrics"]["threeway_sign_stability"]),
                ana[key,role]["family_pass_counts"]["higher_order"])
               for key in KEYS for role in ROLES])+"\nF has a large, positive, stable residual beyond second order in both roles; Q does not. This is an interaction identity, not a localized circuit proof.")
    add("COMPOSITION_CLASS_ADJUDICATION","Development: Q = no registered class, F = higher-order (5/5 families). Validation: same pattern. No single class passes development+validation in both models. Therefore cross-model A/B/C/E/F are not confirmed; the preregistered independent final is not eligible and remains sealed (40 states/model). `final_gate_v38.json` records this decision. At cross-model scope the outcome is no shared stable rule; F-specific higher-order replication is a secondary architecture-specific observation, not a shared formal class.")
    add("INCREMENTAL_NATURAL_ACCUMULATION","The cumulative sequence is R000→R100→R110→R111. Per-state increments are Δ2=`E100`, Δ3|2=`E110−E100`, Δ4|23=`E111−E110`, compared against isolated `E010` and `E001`. These are re-expressions of the frozen eight-condition vectors, not extra experiments. Median isolated regional effect norms:\n\n"+
        table(("model","role","Q2","Q3","Q4","Q23","Q234"),
              [(key,role,*[f(frames[key,role][x].median()) for x in
                 ("isolated_q2_norm","isolated_q3_norm","isolated_q4_norm",
                  "cumulative_q23_norm","cumulative_q234_norm")])
               for key in KEYS for role in ROLES]))
    add("CONTEXT_DEPENDENT_MARGINAL_EFFECTS","Q3 given Q2 versus isolated Q3, and Q4 given Q2+Q3 versus isolated Q4:\n\n"+margin_table+"\nF shows stronger directional context dependence, especially for Q4; ratios near one show that rotation can dominate without large magnitude change. Additional Q4|Q2 and Q4|Q3 cosine columns are in the analysis Parquet files. No seriality is inferred.")
    add("MARGINAL_EFFECT_ROTATION","Same-state cosine and magnitude ratios for conditional versus isolated regional increments:\n\n"+margin_table+"\nQ directions remain near aligned (median cosines ≈0.95–0.97); F Q4|Q23 rotates more (≈0.62 development, 0.69 validation), while median magnitude ratios remain near 1. This is a vector-space property of the tested interventions only.")
    add("CROSS_MODEL_COMPOSITION","Q: additive prediction is fairly accurate but second-order gains are material and the pairwise 4/5-family rule fails; no stable registered class. F: second-order prediction remains poor, while a positive three-way term is stable in 5/5 families across development and validation. Numeric equality was never required. The models do not share a winning qualitative class; a universal correction/composition mechanism is not established.")
    add("Q1_CONTROL","Frozen 4/family development and 2/family validation control subsets compare natural Q234 versus Q1+Q234 (all donor REC on donor Conv):\n\n"+control_table+"\nQ1 is not dispensable on these subsets: adding it further reduces donor error. This is secondary and does not reopen layer search.")
    add("TRAJECTORY_INTERNAL_TRACE","Five frozen development states/model (one/family), all eight conditions, first frozen probe: 40 hash-only rows/model. Q2/Q3/Q4 boundary hidden, Conv/REC incoming states, post-Conv factors, transformed controls, true update, recurrent read, mixer output, and logits were captured as hashes in `trajectory_trace_*_development_v38.parquet`. The Q/F boundary layer indices are in trace summaries. These are explanatory one-probe traces, not the formal six-probe composition evidence; no future output was copied.")
    dose_rows=[(lam,f(vals["additive_relative_error"]),f(vals["second_relative_error"]))
               for lam,vals in dose["by_lambda"].items()]
    add("DOSE_RESPONSE","F-only 20-state diagnostic after its development higher-order class was frozen. λ=0.25/0.5 interpolations are off natural trajectories; λ=1 exactly replays natural R111 (max absolute discrepancy "+f(dose["lambda_1_replay_max_abs"])+").\n\n"+
        table(("λ","additive relative error","second-order relative error"),dose_rows)+
        "\nThe small-λ conditions are not more additive here; error is larger than at λ=1. This does not support the proposed small-perturbation-additive → natural-nonlinear pattern. No Q dose class was opened because Q had no frozen development class. V38-D is not confirmed.")
    add("JVP_FINITE_COMPARISON","Optional JVP was not run. The F interpolation diagnostic cannot substitute for a local Jacobian. No claim about local linearization failure is made.")
    add("TASK_GROUNDED_DESIGN","Phase B was authorized because F passed a development composition class and the high-level gate held in both models. `task_design_v38.json` froze 4/family development and 2/family validation states, generator-derived first answer action, model-specific one-token candidates, and one common newline probe before task-specific outcomes. All eight REC conditions plus recipient/REC-only/joint/donor/KV-only were measured. Crucial limitation: no controlled Type S or Type T semantic/surface prompt forks; this design cannot establish strict task-variable mediation.")
    add("TASK_GROUNDED_RESULTS","One-token answer diagnostic (invalid/coverage-fail states excluded):\n\n"+task_table+"\nR111 does not consistently improve the externally defined correct-answer margin or accuracy over R000. Neither model shows a stable task-grounded effect. Probe/format and missing semantic contrasts further limit interpretation; V38-G is not established.")
    add("SEMANTIC_SURFACE_CONTROLS","No valid Type S surface-only or Type T task-variable-changing same-prefix forks were created in V38. The wrong-token same-prefix REC control was run in `controls_*_v38.parquet`, but it is not a substitute for a controlled task-variable contrast. Accordingly, donor-like vectors are not interpreted as semantic content.")
    add("CHANNEL_TASK_FUNCTION","The Task Phase B diagnostic includes recipient, REC-only, Conv-only R000, REC+Conv joint, KV-only, and Q234 with recipient-native KV. Per-condition one-token answer accuracies are in `task_diagnostic_*_v38.json`. With no positive task-grounded mediation and no Type S/T forks, it does not identify a functional channel specialization. REC-only is weak in donor-vector error on frozen control subsets, a distinct measurement.")
    add("TEMPORAL_DELAY_DESIGN","Phase C required stable task-grounded Phase B evidence; this gate failed. No delay sequence was selected or executed. Model Conv-window size and suffix-flush behavior were not inferred.")
    add("TEMPORAL_CHANNEL_PROFILE","Not opened: no time-dependent Conv/REC/KV causal profile was measured. V38-H and any claim of information transfer to REC are unsupported.")
    add("STRICT_INTERFACE_AUDIT","Exact native REC writeback, donor Conv preservation, recipient-native KV, instrumented bitwise replay and six-probe continuation passed fresh V38 calibration. On frozen control subsets, median donor errors:\n\n"+
        table(("model","role","recipient","REC-only","Conv-only","joint","wrong token","shuffle*","signflip*","read ceiling"),
              [(key,role,*[f(ctr[key,role]["medians"][x]) for x in
                ("recipient_error","rec_only_error","conv_error","joint_error","wrong_error",
                 "shuffle_error","signflip_error","interface_error")])
               for key in KEYS for role in ROLES])+
        "\n`*` Off-manifold controls, not natural states. Wrong token is same-prefix natural alternative. Q234 read-interface output patch is explicitly an interface ceiling and never counted as a natural mechanism. Raw tensor arrays remain local/ignored by Git; table records, hashes and seals are versioned.")
    # Freeze/order audit is checked here, not merely asserted in prose.
    stage_order=[]
    for role in ROLES:
        for key in KEYS:
            names=[f"trajectory_{stage}_{key}_{role}" for stage in ("singles","pairs","predict","full")]
            times=[verify_stage(root,name)["created_utc"] for name in names]
            if times!=sorted(times) or len(set(times))!=4:
                raise RuntimeError(f"V38 stage order violated: {key}:{role}")
            stage_order.append((key,role,*times))
    add("EXECUTION_MANIFEST","Base freeze `"+base["freeze_digest"]+"`; design freeze `"+
        verify_stage(root,"design")["freeze_digest"]+"`; development adjudication freeze `"+
        verify_stage(root,"composition_development")["freeze_digest"]+
        "`; validation adjudication freeze `"+verify_stage(root,"composition_validation")["freeze_digest"]+
        "`; final-closed freeze `"+verify_stage(root,"final_closed")["freeze_digest"]+"`.\n\n"
        "For every model×role, single conditions preceded pair conditions, both predictions were sealed before R111, and stage timestamps were verified. Independent final was never executed. Machine paths/hashes are listed in `v38_integrity_index.json`; `.npz` vectors are retained locally and ignored by Git per repository policy.")
    answers=[
        "1–2. High-level conditional benefit and aggregate natural Q234 restoration replicate in both models; F development fails the additionally frozen 4/5 Q234 family-uniform gate.",
        "3–8. Isolated Q2/Q3/Q4 and pair responses were measured as same-state vectors; exact norms/distributions are in analysis Parquet files.",
        "9–12. Q additive prediction is close but not a formal additive-class pass; F additive prediction fails. Exact median errors/cosines are in the additive report.",
        "13–17. I23/I24/I34 are explicit vectors. F pair terms are large and generally cancelling; I23 has the largest median norm in both models. No serial-circuit inference.",
        "18–22. Q second-order fit is good but family/improvement gate fails; F second-order fit fails. F three-way residual is large, positive and sign-stable; Q three-way is not stable.",
        "23–24. Q and F do not require the same registered composition class; no shared cross-model class passes.",
        "25–29. Conditional Q3/Q4 increments differ from isolated increments, mainly by stronger directional rotation in F; ratios are near one.",
        "30. Q1 adds a measurable donor-directed increment on the frozen control subsets.",
        "31–33. F-only off-manifold dose interpolation does not become more additive at small λ; V38-D is unsupported. JVP was optional and not run.",
        "34–37. Donor-vector fidelity is strong, but the generator-defined correct-answer diagnostic does not improve consistently; no semantic/surface Type S/T contrast or task-grounded channel function is established.",
        "38–40. Temporal phase remained closed; Conv-window or REC persistence claims were not tested.",
        "41–48. Cross-model V38-A/B/C/E/F/G/H not confirmed; V38-D unsupported. At cross-model scope V38-I (no shared stable rule) applies, while F has a replicating within-model higher-order pattern.",
        "49. No cross-model composition rule is licensed for a third-model frozen test. F-specific three-way sign/fraction is a candidate only, not a shared law.",
        "50. A targeted exploratory F training-origin study is scientifically permissible after its replicated model-specific pattern; a shared-law V39 training-origin study is not yet justified."]
    add("SCIENTIFIC_ANSWERS","\n\n".join(answers))
    reports=root/"reports"
    paths=[]
    for name in TOPICS:
        path=reports/f"V38_{name}.md"
        if path.exists():raise RuntimeError(f"V38 report already exists: {path}")
        path.write_text(docs[name],encoding="utf-8")
        paths.append(path)
    complete=("# V38 — Trajectory-Level Causal Composition of Persistent State\n\n"
              "## Result\n\nBoth models replicate the high-level REC–Conv effect and broad natural Q234 reconstruction. "
              "F has a prospectively replicated, positive three-way Q2/Q3/Q4 interaction (5/5 families development and validation). "
              "Q has no registered additive/pairwise/higher-order class despite relatively accurate second-order vector prediction. "
              "No shared cross-model class qualifies, so independent final remains sealed (40 states/model).\n\n"
              "## Prediction evidence\n\n"+pred_table+"\n"
              "Q234 natural reconstruction:\n\n"+q_table+"\n"
              "F development does not meet the extra 4/5-family Q234 reconstruction gate (3/5); "
              "the modular-arithmetic horizon shifts from 2 to 3 in validation/final. Both limitations constrain inference.\n\n"
              "## Secondary phases\n\nThe frozen controls confirm exact native intervention semantics; Q1 is not negligible. "
              "F-only off-manifold interpolation does not support a small-λ additive regime. "
              "Task answer-token diagnostics do not show stable improvement and lack Type S/T contrasts; "
              "task-grounded V38-G is not established. Temporal Phase C and optional JVP were not run.\n\n"
              "## Audit\n\nAll roles were generated afresh, historically disjoint and sealed before formal outcomes. "
              "For each model/role, singles→pairs→hashed additive/second-order predictions→R111 ordering was verified. "
              "V1–V37 were not edited. Machine outputs and hashes are in `results/v38/processed/v38_integrity_index.json`. "
              "See `V38_ALL_REPORTS.md` for every topic in one file.\n")
    complete_path=reports/"V38_COMPLETE_REPORT.md"
    complete_path.write_text(complete,encoding="utf-8")
    paths.append(complete_path)
    all_path=reports/"V38_ALL_REPORTS.md"
    all_path.write_text("# V38 — All Reports\n\n"+"\n\n---\n\n".join(
        (reports/f"V38_{name}.md").read_text(encoding="utf-8") for name in (*TOPICS,"COMPLETE_REPORT")),
        encoding="utf-8")
    paths.append(all_path)
    final=root/"reports/FINAL_REPORT.md"
    old=final.read_text(encoding="utf-8")
    if "<!-- V38_START -->" in old:raise RuntimeError("V38 final report already appended")
    final.write_text(old.rstrip()+"\n\n<!-- V38_START -->\n## V38 — Trajectory-Level Causal Composition of Persistent State\n\n"
                     "Q/F both replicate REC–Conv high-level benefit and natural Q234 effect. F has a replicated model-specific higher-order trajectory term; Q has no preregistered class, so no shared class and no independent-final opening. Task-grounded and temporal claims remain unestablished. See `reports/V38_COMPLETE_REPORT.md` and `reports/V38_ALL_REPORTS.md`.\n<!-- V38_END -->\n",encoding="utf-8")
    # Index intentionally excludes itself and the report seal to avoid cycles.
    indexed=[root/"configs/trajectory_v38.yaml",root/"src/jclosure/protocol_v38.py",root/SOURCE,
             *sorted((root/"data/v38").glob("*")),
             *sorted((root/OUT).glob("*")),
             *sorted((root/"artifacts").glob("trajectory_composition_v38*.freeze.json")),
             *paths,final]
    index_path=root/OUT/"v38_integrity_index.json"
    indexed=[p for p in indexed if p.is_file() and p!=index_path]
    index={"protocol":"V38","parent_commit":base["parent_commit"],
           "model_specs":{key:{k:base["model_specs"][key][k] for k in
                           ("id","revision","weight_sha256","tokenizer_sha256","config_sha256")}
                          for key in KEYS},
           "stage_order_verified":True,
           "prediction_frozen_before_R111_each_model_role":True,
           "validation_pool_presealed":True,"independent_final_pool_presealed":True,
           "independent_final_opened":False,
           "modular_horizon_shift":pool["horizon_override"],
           "npz_policy":"local machine outputs hashed but ignored by Git; not in remote Git clone",
           "files_sha256":{str(p.relative_to(root)):sha256_file(p) for p in indexed}}
    write_json_atomic(index_path,index)
    seal=stage_freeze(root,"report",[SOURCE,str(index_path.relative_to(root)),
                      str(all_path.relative_to(root)),str(complete_path.relative_to(root)),
                      str(final.relative_to(root))],
                      {"reports":len(paths),"integrity_index_sha256":sha256_file(index_path),
                       "independent_final_opened":False})
    return {"freeze_digest":seal["freeze_digest"],"topic_reports":len(TOPICS),
            "reports_total":len(paths),"integrity_files":len(index["files_sha256"]),
            "independent_final_opened":False}


if __name__=="__main__":print(json.dumps(run(Path.cwd()),indent=2))
