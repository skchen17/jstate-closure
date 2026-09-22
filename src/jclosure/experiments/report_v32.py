"""Generate the V32 reports, single-file bundle and integrity manifest from records."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from jclosure.protocol_v32 import stage_freeze, verify, verify_stage
from jclosure.provenance import sha256_file, write_json_atomic

OUT = Path("results/v32/processed")
SOURCE = "src/jclosure/experiments/report_v32.py"
ORDER = (
    "REC_CORRECTION_REPLICATION", "GAIN_ROTATION_CORRECTION_DECOMPOSITION", "CONTEXTUAL_REC_CORRECTION",
    "CROSS_STATE_REC_CORRECTION", "CROSS_TOKEN_REC_CORRECTION", "NEXT_TOKEN_CORRECTION_TRACE",
    "FIRST_CORRECTION_SITE", "CORRECTION_INTERCEPTION", "CORRECTION_REVERSE_TRANSPLANT",
    "REC_CONV_LAYER_INTERACTION_MAP", "SAME_VS_CROSS_LAYER_ROUTING", "GATE_MEDIATION",
    "QKV_UPDATE_MEDIATION", "RESIDUAL_INTEGRATION", "CORRECTION_DIMENSION", "MULTI_PROBE_CORRECTION",
    "MULTI_HORIZON_CORRECTION", "STRICT_INTERFACE_AUDIT", "EXECUTION_MANIFEST", "SCIENTIFIC_ANSWERS", "COMPLETE_REPORT",
)


def fmt(x):
    return "n/a" if x is None or pd.isna(x) else f"{float(x):.3f}"


def table(headers, rows):
    values = [[str(x).replace("|", "\\|") for x in row] for row in rows]
    return "| " + " | ".join(headers) + " |\n|" + "|".join("---" for _ in headers) + "|\n" + "\n".join("| " + " | ".join(row) + " |" for row in values) + "\n"


def run(root: Path):
    verify_stage(root, "final_opening")
    cfg = verify(root)["config"]
    adj = json.loads((root / OUT / "v32_adjudication.json").read_text())
    design = json.loads((root / OUT / "design_v32.json").read_text())
    site = adj["selected_site"]
    f = {r: pd.read_parquet(root / OUT / f"factorial_{r}_v32.parquet") for r in ("development", "validation")}
    c = {r: pd.read_parquet(root / OUT / f"context_{r}_v32.parquet") for r in f}
    t = {r: pd.read_parquet(root / OUT / f"next_token_trace_{r}_v32.parquet") for r in f}
    i = {r: pd.read_parquet(root / OUT / f"site_intervention_{r}_v32.parquet") for r in f}
    lm = {r: pd.read_parquet(root / OUT / f"rec_conv_layer_map_{r}_v32.parquet") for r in f}
    sub = {r: pd.read_parquet(root / OUT / f"subsite_{r}_v32.parquet") for r in f}
    reports = {}
    def add(name, title, body):
        reports[name] = f"# {title} — V32\n\n{body.strip()}\n"

    g = adj["factorial"]
    repl_rows = [(r, g[r]["rows"], fmt(g[r]["positive_fraction"]), fmt(g[r]["median_l2_reduction"]), fmt(g[r]["reduction_bootstrap_lower"]), fmt(g[r]["median_alignment"]), fmt(g[r]["alignment_bootstrap_lower"]), g[r]["replication_families_passing"], g[r]["replication_pass"], g[r]["alignment_pass"]) for r in f]
    family_rows = [(r, fam, d["rows"], fmt(d["positive_fraction"]), fmt(d["median_l2_reduction"]), fmt(d["median_alignment"])) for r in f for fam,d in g[r]["family"].items()]
    category_rows = [(r, cat, len(group), fmt(group.relative_conv_error_reduction.median()), fmt(group.residual_alignment_cosine.median()), fmt(group.improvement_positive.mean())) for r in f for cat,group in f[r][f[r].primary].groupby("token_category")]
    add("REC_CORRECTION_REPLICATION", "REC Correction Replication", f"Exact native factorials used new 100 development and 50 validation states, one primary ordinary natural token fork per state, six prewrite probes, and recipient-native KV. The independent 50-state final remains sealed. Relative improvement is `(||Ydonor−Y01||−||Ydonor−Y11||)/||Ydonor−Y01||`; state-bootstrap lower bounds use 2,000 resamples.\n\n{table(('role','n','positive fraction','median reduction','bootstrap LB','median alignment','alignment LB','families','replication gate','alignment gate'),repl_rows)}\n{table(('role','family','n','positive fraction','median reduction','median alignment'),family_rows)}\n{table(('role','frozen token category','n','median reduction','median alignment','positive fraction'),category_rows)}\nREC-only remains weak: relative donor L2 medians development/validation {fmt(f['development'][f['development'].primary].rec_only_relative_l2.median())}/{fmt(f['validation'][f['validation'].primary].rec_only_relative_l2.median())}; Conv-only is {fmt(f['development'][f['development'].primary].conv_relative_l2.median())}/{fmt(f['validation'][f['validation'].primary].conv_relative_l2.median())}, joint is {fmt(f['development'][f['development'].primary].joint_relative_l2.median())}/{fmt(f['validation'][f['validation'].primary].joint_relative_l2.median())}. This establishes a conditional causal *effect of exact REC field replacement* on the tested donor-fidelity endpoint, not a localized mechanism or REC as an independent carrier. Sources: `factorial_*_v32.parquet`, matching vectors and audits.")

    rot = adj["rotation"]
    add("GAIN_ROTATION_CORRECTION_DECOMPOSITION", "Gain, Rotation and Correction Decomposition", f"Development/validation joint gain ratios (norm RC / norm C) are {fmt(f['development'][f['development'].primary].rec_gain_ratio.median())}/{fmt(f['validation'][f['validation'].primary].rec_gain_ratio.median())}. Per-row scalar alpha projects the joint move onto Conv; this is an oracle descriptive upper bound, not a prospective predictor. On validation the median scalar-gain donor L2 is {fmt(f['validation'][f['validation'].primary].scalar_gain_error_relative.median())}, versus true joint {fmt(f['validation'][f['validation'].primary].joint_relative_l2.median())}; true joint beats that scalar in {fmt(f['validation'][f['validation'].primary].joint_beats_scalar_gain.mean())} of rows.\n\nThe fixed global rotation comparison fits a rank-{rot['subspace_rank']} orthogonal Procrustes map on development moves only, retaining identity outside that development subspace. Validation median donor L2 is {fmt(rot['median_rotation_relative_error'])}, versus true joint {fmt(rot['median_true_joint_relative_error'])}; joint wins in {fmt(rot['fraction_true_joint_beats_rotation'])} of rows. This rejects only the tested low-capacity fixed rotation, not every possible context-dependent rotation.\n\nThe sequential decomposition of conditional REC response on development gives median normalized gain component {fmt(f['development'][f['development'].primary].gain_fraction_of_conditional_norm.median())}, residual-target component {fmt(f['development'][f['development'].primary].residual_correction_fraction_of_conditional_norm.median())}, and remaining orthogonal component {fmt(f['development'][f['development'].primary].orthogonal_remainder_fraction_of_conditional_norm.median())}. These overlapping norm fractions do not sum to one; they are descriptive, whereas exact factorial error reduction is causal.")

    ctx = adj["context"]
    context_rows = [(r, x, fmt(v)) for r in f for x,v in ctx[r]["condition_median_error"].items()]
    add("CONTEXTUAL_REC_CORRECTION", "Contextual REC Correction", f"For each of 10 development and 10 validation targets, target Conv and recipient KV are fixed. Matched REC is compared with recipient, wrong-token, same-family wrong-state, cross-family wrong-state, shuffled, sign-flipped and random same-norm REC. All variants use the same six probes.\n\n{table(('role','REC condition','median relative donor L2'),context_rows)}\nMatched REC beats all four primary off-context controls in development {fmt(ctx['development']['matched_better_than_all_four_fraction'])} and validation {fmt(ctx['validation']['matched_better_than_all_four_fraction'])} of targets. Cross-state REC is off-manifold; its failure alone does not identify state-specific gating. The mapping was frozen before context-specific responses but after factorial observations, so V32-C is **not** promoted to a fully preregistered formal outcome. Sources: `context_mapping_*_v32.json`, `context_*_v32.parquet`.")

    add("CROSS_STATE_REC_CORRECTION", "Cross-State REC Correction", f"Same-family and cross-family wrong-state source IDs are in frozen `context_mapping_*_v32.json`. Development matched/same-family/cross-family median relative donor errors are {fmt(ctx['development']['condition_median_error']['MATCHED_REC'])}/{fmt(ctx['development']['condition_median_error']['SAME_FAMILY_WRONG_STATE_REC'])}/{fmt(ctx['development']['condition_median_error']['CROSS_FAMILY_WRONG_STATE_REC'])}; validation values are {fmt(ctx['validation']['condition_median_error']['MATCHED_REC'])}/{fmt(ctx['validation']['condition_median_error']['SAME_FAMILY_WRONG_STATE_REC'])}/{fmt(ctx['validation']['condition_median_error']['CROSS_FAMILY_WRONG_STATE_REC'])}. This is target-probe cross-state correction fidelity. Direct effect-space cosine between states is **not comparable** because the six frozen probe IDs differ by state; it is not reported as though measured. Different incoming states can place REC off the natural target manifold. Therefore this is a bounded transfer-failure observation, not proof that REC stores a specific missing content or that a gate is state-specific.")

    same_token = []
    for role in f:
        primary = f[role][f[role].primary]
        counts = primary.groupby("token_id").state_id.nunique()
        second = f[role][~f[role].primary]
        paired = primary.set_index("state_id").relative_conv_error_reduction.reindex(second.state_id).to_numpy() if len(second) else []
        within_difference = abs(second.relative_conv_error_reduction.to_numpy()-paired) if len(second) else []
        same_token.append((role, len(counts), int((counts>=2).sum()), int(counts.max()), len(second), fmt(second.relative_conv_error_reduction.median()) if len(second) else "n/a", fmt(pd.Series(within_difference).median()) if len(second) else "n/a"))
    add("CROSS_TOKEN_REC_CORRECTION", "Cross-Token REC Correction", f"The design froze two candidate contrasts per state, from different response-blind surface categories where eligible. Secondary factorial contrasts were evaluated in 25 development and 10 validation states, without fitting. The primary library also repeats token IDs across independent incoming states.\n\n{table(('role','distinct primary tokens','tokens in ≥2 states','max states/token','secondary contrasts','secondary median reduction','median within-state absolute difference'),same_token)}\nWithin-state primary/secondary comparisons share probes and are directly comparable as donor-error reductions. Cross-state response-vector cosines are not, because each state's probe IDs differ. Category-stratified results appear in `V32_REC_CORRECTION_REPLICATION.md`. The observed variation is not a fitted token/state interaction law or REC content code. The full per-row records and token/state hashes are in `factorial_*_v32.parquet` and `design_v32.json`.")

    trace_rows = [(r, boundary, len(group), fmt(group.conditional_norm.median()), fmt(group.boundary_residual_alignment.median())) for r in t for boundary,group in t[r].groupby("boundary")]
    growth_rows = [(r, int(layer), fmt(group.conditional_norm.median()), fmt(group.boundary_residual_alignment.median()), fmt(group.final_residual_alignment.median())) for r in t for layer,group in t[r][t[r].boundary=="post_block_residual"].groupby("layer")]
    add("NEXT_TOKEN_CORRECTION_TRACE", "Next-Token Correction Trace", f"On the frozen 10+10 tracing states, one of six prewrite probes was recorded across all 32 post-block residual outputs and 24 recurrent layers' qkv projection, raw gate-a/b projection, normalized recurrent read and recurrent output. The primary causal endpoints remain six-probe signatures.\n\n{table(('role','architecture boundary','rows','median conditional norm','median boundary residual alignment'),trace_rows)}\nLayerwise post-block growth is reported below; norms and cosines are descriptive and are not donor-fidelity interventions.\n\n{table(('role','layer','median conditional norm','boundary target cosine','final residual target cosine'),growth_rows)}\nRaw gate projections are not themselves gate activation values; sigmoid/softplus inside the kernel was not intercepted. Likewise the functional convolution output and delta-update term were not directly hookable with the existing module interface. Thus trace coverage is partial, and no first *causal* site follows from activation divergence. Machine source: `next_token_trace_*_v32.parquet`.")

    first_rows = []
    for r in t:
        x = t[r][(t[r].boundary=="recurrent_output") & (t[r].conditional_norm>1e-6)].groupby("state_id").layer.min()
        first_rows.append((r,len(x),fmt(x.median()),int(x.min()) if len(x) else "n/a",int(x.max()) if len(x) else "n/a"))
    add("FIRST_CORRECTION_SITE", "First Measurable Correction Site", f"A recurrent output is called measurable when its Y11−Y01 norm exceeds 1e−6 on the first frozen probe; this is a numerical descriptive threshold, not a noise-calibrated causal threshold.\n\n{table(('role','states','median first layer','min','max'),first_rows)}\nThe development-only candidate rule selected recurrent-output layer {site['candidate_layer']} by maximal median alignment ({fmt(site['development_median_final_residual_alignment'])}) with the final residual donor-minus-Conv target. Validation did not select a new site. The first measurable layer is not the first causal mediator, and the selected site must pass bidirectional intervention to qualify.")

    loc = adj["localization"]
    loc_rows = [(r,loc[r]["rows"],loc[r]["success_rows"],fmt(loc[r]["median_removed_fraction"]),fmt(loc[r]["median_restored_fraction"]),fmt(loc[r]["median_remove_cosine"]),fmt(loc[r]["median_restore_cosine"]),loc[r]["families_passing"],loc[r]["pass"]) for r in f]
    add("CORRECTION_INTERCEPTION", "Correction Interception", f"At the frozen layer-{site['candidate_layer']} recurrent-operator output, replace Y11's component with its same-probe Y01 value and continue the suffix. The same input cache, token and position are retained; requested/realized output hashes are equal in all probes. Removed benefit is measured against the six-probe donor error.\n\n{table(('role','n','bidirectional successes','median removed','median restored','remove cosine','restore cosine','families','formal gate'),loc_rows)}\nNo formal localized necessity claim is supported. A single candidate was selected prospectively; its failure does not establish that no other local site exists. Source: `site_intervention_*_v32.parquet`, `site_intervention_audit_*_v32.parquet`.")

    add("CORRECTION_REVERSE_TRANSPLANT", "Correction Reverse Transplant", f"The reverse experiment inserts the matched Y11 recurrent-operator output into the Y01 Conv-only branch, one probe at a time. Exact field hash equality and baseline replay are required. Validation median restored benefit is {fmt(loc['validation']['median_restored_fraction'])}, below the frozen ≥0.50 gate; median correction-direction cosine is {fmt(loc['validation']['median_restore_cosine'])}, below ≥0.80. Development values are {fmt(loc['development']['median_restored_fraction'])}/{fmt(loc['development']['median_restore_cosine'])}. Therefore layer {site['candidate_layer']} is not a bidirectionally validated correction site; final remains closed.")

    map_rows = [(r, a, b, fmt(v)) for r in f for (a,b),v in lm[r].groupby(["REC_group","Conv_group"]).paired_l2_improvement.median().items()]
    add("REC_CONV_LAYER_INTERACTION_MAP", "REC × Conv Layer Interaction Map", f"The frozen four recurrent-layer groups and four Conv-layer groups form 16 exact-native pairings, each tested on 10 development and 10 validation states with six probes. The score is `(||donor−Conv_j||−||donor−REC_i+Conv_j||)/||donor−recipient||`. It is not an anatomical mediation score.\n\n{table(('role','REC group','Conv group','median donor-L2 improvement'),map_rows)}\nNo individual group pair was selected from validation, and no single-layer refinement or formal pairwise gate was frozen. V32-H is therefore not established. Source: `rec_conv_layer_map_*_v32.parquet`.")

    route_rows = [(r, fmt(lm[r][lm[r].same_group].paired_l2_improvement.median()), fmt(lm[r][lm[r].REC_before_Conv].paired_l2_improvement.median()), fmt(lm[r][lm[r].REC_after_Conv].paired_l2_improvement.median()), fmt(lm[r][~lm[r].same_group].paired_l2_improvement.median())) for r in f]
    add("SAME_VS_CROSS_LAYER_ROUTING", "Same- versus Cross-Layer Routing", f"{table(('role','same group','REC earlier','REC later','all cross group'),route_rows)}\nThese summaries average different partial Conv baselines and are descriptive. Same group is not the same physical layer; group index order does not prove causal anatomy. No predeclared superiority or sufficiency threshold for a group pairing was met/assessed, and no claim that cross-layer routing is required is made.")

    def component_rows(names):
        return [(r,name,fmt(group.removed_benefit_fraction.median()),fmt(group.restored_benefit_fraction.median()),fmt(group.remove_direction_cosine.median()),fmt(group.restore_direction_cosine.median())) for r in f for name,group in sub[r][sub[r].component.isin(names)].groupby("component")]
    add("GATE_MEDIATION", "Gate Mediation", f"Raw gate-a and gate-b projection traces vary under the factorial branches, but sigmoid/softplus gate values and recurrent-kernel update terms were not independently recorded. Diagnostic exact-output patches at the development-nominated layer are below. Component probes were specified after the primary factorial plan and cannot become formal V32-E evidence.\n\n{table(('role','component','median removed','median restored','remove cosine','restore cosine'),component_rows(('gate_a','gate_b')))}\nPatching a raw projection output can perturb downstream gate computation; it does not isolate a unique state-specific gate law. V32-E remains unconfirmed.")

    add("QKV_UPDATE_MEDIATION", "QKV and Update Mediation", f"The architecture exposes a joint pre-convolution qkv projection and a normalized recurrent read. The q/k/v post-convolution split and fused delta-update were not individually writable with the audited interface. Exact diagnostic patches at the frozen primary layer yield:\n\n{table(('role','component','median removed','median restored','remove cosine','restore cosine'),component_rows(('qkv_projection','normalized_recurrent_read')))}\nThese checks do not isolate q, k, v, beta or the update term, and the component list was not part of the original primary candidate-selection rule. V32-F is unconfirmed, not falsified for untested update variables.")

    add("RESIDUAL_INTEGRATION", "Residual Integration", f"The diagnostic `post_residual_mlp_normalized_input` patch replaces the post-attention layernorm *output* entering the MLP; it does not replace the entire residual sum.\n\n{table(('role','component','median removed','median restored','remove cosine','restore cosine'),component_rows(('post_residual_mlp_normalized_input',)))}\nBecause the residual skip path and recurrent cache update remain separate, this cannot adjudicate whether correction is instantiated at residual addition. V32-G remains unconfirmed.")

    rank = adj["correction_dimensionality"]
    add("CORRECTION_DIMENSION", "Correction Dimensionality", f"The centered 100-row development matrix of six-probe conditional REC moves has descriptive r90/r95/r99 = {rank['r90']}/{rank['r95']}/{rank['r99']} and effective rank {fmt(rank['effective_rank'])}; its maximal empirical rank is 99. These are ranks of a sampled *response*, not compact state dimensions or transferable content. No rank concentration threshold was frozen for a causal basis and no basis reconstruction intervention was authorized. The optional causal subspace question is unanswered.")

    add("MULTI_PROBE_CORRECTION", "Multi-Probe Correction", f"Every primary factorial, context comparison, 4×4 layer-map row and causal interception/reverse transplant used the concatenated six-probe normalized response containing J, selected logits, semantic log-probabilities, workspace, broad vocabulary and late residual. One-probe sublayer traces were only used for candidate nomination, never as the decisive causal endpoint. Frozen probe IDs and hashes are per state in `design_v32.json`; `factorial_*_v32.parquet` stores the hash on each row. The V32-A/B results therefore survive this frozen multi-probe signature; no generalization to arbitrary future-token distributions is claimed.")

    add("MULTI_HORIZON_CORRECTION", "Multi-Horizon Correction", "Primary h1 factorial and localization were completed. The frozen rule permits h2/h4 only after an h1 localized mechanism passes development and validation. The nominated site failed, so h2/h4 were not opened and remain unanswered. No h8 experiment was required. This is protocol gating, not evidence of absent longer-horizon effects.")

    audit_rows = []
    for r in f:
        fa = pd.read_parquet(root / OUT / f"factorial_audit_{r}_v32.parquet")
        ia = pd.read_parquet(root / OUT / f"site_intervention_audit_{r}_v32.parquet")
        sa = pd.read_parquet(root / OUT / f"subsite_audit_{r}_v32.parquet")
        audit_rows.append((r,len(fa),bool(fa.writeback_exact.all()),len(ia),bool(ia[['remove_exact','insert_exact','input_cache_unchanged']].all().all()),len(sa),bool(sa.exact.all())))
    add("STRICT_INTERFACE_AUDIT", "Strict Interface Audit", f"{table(('role','factorial rows','native exact','site probe patches','site exact','diagnostic probe patches','diagnostic exact'),audit_rows)}\nIncoming prefix and cache-channel hashes, donor/recipient full-state hashes, token-pair and six-probe hashes, target Conv and recipient KV hashes, layer-group hashes, requested/realized component hashes, cache lengths and role IDs are in the design and raw audit records. Hooks are context-managed and removed even on failure. V32 final-state response count is zero. The factorial audit's non-outcome `field_count` metadata incorrectly records 120 for the three conditions; an append-only `audit_metadata_amendment_v32.json` corrects this to 24+24+48=96 without altering the raw records, exact equality checks or outcomes. The trace lacks a direct convolution-output and recurrent-kernel update interception; those are explicitly untested, not silently described as measured.")

    base = verify(root)
    freezes = [(p.name,json.loads(p.read_text()).get("freeze_digest", "")[:16]) for p in sorted((root/"artifacts").glob("rec_conv_mechanism_v32*.freeze.json"))]
    add("EXECUTION_MANIFEST", "Execution Manifest", f"Parent `{base['parent_commit']}`; base freeze `{base['freeze_digest']}`. Response-blind panels: 25 calibration, 100 development, 50 validation, 50 unopened independent final. Excluded V28–V31 formal states; 40 ordinary candidate tokens in five surface classes; six frozen probes per state. Primary factorial 100/50 rows, secondary token contrasts 25/10, next-token trace 10/10, context controls 10/10, 4×4 group map 10/10 and bidirectional nominated-site intervention 10/10.\n\n{table(('freeze artifact','digest prefix'),freezes)}\nThe predeclared candidate is recurrent-output layer {site['candidate_layer']}. Context mapping and secondary component probes have a narrower prospective status than the primary factorial and nominated-site test; this difference is preserved in adjudication. The independent final was not opened; h2/h4 not run. Raw records, hashes, tests and the integrity index are in `results/v32/processed/`, `tests/test_v32.py` and `v32_integrity_index.json`.")

    o = adj["formal_outcomes"]
    answers = [
        f"1. Yes: {g['development']['rows']}/{g['validation']['rows']} fresh primary states pass the frozen replication gate.",
        f"2. Yes: {g['development']['replication_families_passing']}/5 development and {g['validation']['replication_families_passing']}/5 validation families.",
        f"3. Yes in this tested design: REC-only median relative donor L2 {fmt(f['validation'][f['validation'].primary].rec_only_relative_l2.median())}.",
        f"4. Yes: Conv-only {fmt(f['validation'][f['validation'].primary].conv_relative_l2.median())} versus REC-only {fmt(f['validation'][f['validation'].primary].rec_only_relative_l2.median())} median donor L2.",
        f"5. Yes: {int(f['development'][f['development'].primary].improvement_positive.sum())}/100 and {int(f['validation'][f['validation'].primary].improvement_positive.sum())}/50.",
        f"6. Development/validation median cosine {fmt(g['development']['median_alignment'])}/{fmt(g['validation']['median_alignment'])} with lower bounds {fmt(g['development']['alignment_bootstrap_lower'])}/{fmt(g['validation']['alignment_bootstrap_lower'])}.",
        f"7. No under the per-row best scalar model: validation joint beats scalar in {fmt(f['validation'][f['validation'].primary].joint_beats_scalar_gain.mean())} of rows.",
        f"8. No under the rank-{rot['subspace_rank']} development-fitted fixed Procrustes model; full context-dependent rotations are not excluded.",
        "9. Yes as an estimand description and exact-field causal error reduction, not as a localized anatomical mechanism.",
        f"10. Matched REC has lower error than sampled wrong-state REC in all {ctx['validation']['states']} validation context targets; off-manifold caveat applies.",
        f"11. Matched REC beats wrong-token REC in all {ctx['validation']['states']} validation context targets.",
        f"12. Matched REC beats shuffled REC in all {ctx['validation']['states']} validation context targets.",
        "13. Cross-state transfers differ, but a state-conditioned rule is not isolated.",
        "14. Within-state secondary token contrasts differ; no stable token law is isolated.",
        f"15. Recurrent-output difference is measurable from layer {int(t['validation'][(t['validation'].boundary=='recurrent_output') & (t['validation'].conditional_norm>1e-6)].layer.min())} on the tested first probe; this is not a causal site.",
        f"16. No localized site qualifies; nominated layer {site['candidate_layer']} fails bidirectional gates.",
        f"17. Layer {site['candidate_layer']} interception removes a median {fmt(loc['validation']['median_removed_fraction'])} of REC benefit, below the strict joint gate.",
        f"18. Reverse insertion restores a median {fmt(loc['validation']['median_restored_fraction'])}, below 0.50.",
        "19. A frozen 4×4 REC×Conv group map is reported; no pair was formally qualified.",
        "20. Same-group and cross-group scores differ descriptively; no same-layer privilege is established.",
        "21. Cross-layer routing necessity is not established.",
        "22. Raw gate projections change in the trace; actual transformed gates were not independently captured.",
        "23. Diagnostic gate projection interception/reversal does not support a formal gate-mediation claim.",
        "24. Joint pre-convolution qkv and normalized read were patched diagnostically; individual q/k/v or delta-update mediation remains untested.",
        "25. Only post-residual layernorm output was patched diagnostically; residual-add mediation remains unresolved.",
        "26. One nominated site fails; neither universal localization nor distributed necessity is proven.",
        f"27. Conditional response is geometrically broad in this sample: centered r90/r95/r99={rank['r90']}/{rank['r95']}/{rank['r99']} out of at most 99.",
        "28. No causal correction-basis reconstruction was run; no rank threshold licensed it.",
        "29. Yes for A/B: six frozen probes and a broad response signature are used in every primary causal endpoint.",
        "30. h2/h4 not opened because h1 local-mechanism qualification failed.",
        f"31. V32-A={o['V32-A_REC_CORRECTION_REPLICATED']}.",
        f"32. V32-B={o['V32-B_REC_RESIDUAL_CORRECTION_CONFIRMED']} for exact conditional effect; no anatomical claim.",
        "33. V32-C remains formally unconfirmed due narrower context mapping preregistration and off-manifold source concerns.",
        "34. V32-D/E/F/G/H are not formally supported.",
        "35. V32-I is not formally supported: tested site failure alone cannot establish distributed causation.",
        f"36. V32-J={o['V32-J_REC_CORRECTION_NOT_GENERAL']} under the tested broader prospective panel.",
        f"37. Cross-model replication authorization={adj['cross_model_replication_authorized']} because V32-B passes; no second-model result is claimed.",
    ]
    add("SCIENTIFIC_ANSWERS", "Scientific Answers", "Answers are scoped to this hybrid model, frozen h1 six-probe signature and tested source/recipient semantics.\n\n" + "\n\n".join(answers))

    outcome_rows = [(name,str(value)) for name,value in o.items()]
    add("COMPLETE_REPORT", "Complete Report", f"**Mechanism of REC–Conv Conditional Correction — How Does Recurrent State Refine the Convolutional Future Handoff?** Parent `{base['parent_commit']}`; V32 base freeze `{base['freeze_digest']}`.\n\nThe new 25/100/50/50-state design excluded V28–V31 formal states and froze ordinary natural token contrasts and six future probes before observing V32 causal responses. Exact REC+Conv reduces Conv donor error in 100/100 development and 50/50 validation primary rows; median reductions are {fmt(g['development']['median_l2_reduction'])}/{fmt(g['validation']['median_l2_reduction'])}, with residual-alignment cosines {fmt(g['development']['median_alignment'])}/{fmt(g['validation']['median_alignment'])}. Replication families passing: {g['development']['replication_families_passing']}/5 development and {g['validation']['replication_families_passing']}/5 validation; alignment families passing: {g['development']['alignment_families_passing']}/5 and {g['validation']['alignment_families_passing']}/5. REC-only remains weak. This qualifies V32-A and the precise V32-B statement that exact REC field replacement has a residual-aligned causal *conditional effect* on the tested response. It does **not** show REC carries independent future information or identify the computational mediator.\n\nThe development-only trace nominated recurrent-output layer {site['candidate_layer']}. Six-probe output interception and reverse insertion each yielded {loc['development']['success_rows']}/10 bidirectional successes in development and {loc['validation']['success_rows']}/10 in validation; exact writeback audit passed. No localized site qualifies. Matched REC beat wrong-token, wrong-state and artificial controls in {ctx['development']['states']}/10 development and {ctx['validation']['states']}/10 validation context targets, but context mapping was frozen after primary factorial observations and wrong-state transfers may be off-manifold; V32-C remains formally unconfirmed. The 4×4 layer map and gate/qkv/residual diagnostic probes do not license specific mediation or a distributed-causation proof.\n\n{table(('formal outcome','supported'),outcome_rows)}\nThe independent final stayed sealed; h2/h4 were not opened under the frozen h1 rule. Cross-model replication is *authorized for the V32-B estimand*, not performed. H2 remains true; H3, dynamic search and autonomous control remain unauthorized. Detailed methods, exclusions, raw machine records and per-report results are in the 20 supporting V32 reports and `V32_ALL_REPORTS.md`.")

    if set(reports) != set(ORDER):
        raise RuntimeError("V32 report set mismatch")
    report_dir = root / "reports"
    for name in ORDER:
        path = report_dir / f"V32_{name}.md"
        if path.exists():
            raise RuntimeError(f"V32 report already exists: {path.name}")
        path.write_text(reports[name])
    pieces = ["# V32 — All Reports in One File\n\nThe 21 standalone V32 reports are reproduced below in frozen order.\n"]
    for index, name in enumerate(ORDER,1):
        path = report_dir / f"V32_{name}.md"
        pieces.append(f"\n---\n\n<!-- {index:02d}: {path.name}; sha256={sha256_file(path)} -->\n\n{path.read_text()}")
    bundle = report_dir / "V32_ALL_REPORTS.md"
    if bundle.exists():
        raise RuntimeError("V32 all-reports bundle already exists")
    bundle.write_text("".join(pieces))
    cumulative = report_dir / "FINAL_REPORT.md"
    before = cumulative.read_text()
    if sha256_file(cumulative) != base["cumulative_report_at_start_sha256"]:
        raise RuntimeError("V32 cumulative report drift before append")
    cumulative.write_text(before.rstrip() + "\n\n---\n\n" + reports["COMPLETE_REPORT"])
    indexed = [root / "configs/rec_conv_mechanism_v32.yaml",root / "src/jclosure/protocol_v32.py",root / "tests/test_v32.py"]
    indexed += sorted((root/"src/jclosure/experiments").glob("*v32.py"))
    indexed += sorted((root/"artifacts").glob("rec_conv_mechanism_v32*.freeze.json"))
    indexed += [p for p in sorted((root/OUT).glob("*")) if p.is_file() and p.name != "v32_integrity_index.json"]
    indexed += [report_dir/f"V32_{name}.md" for name in ORDER] + [bundle,cumulative]
    indexed = [p for p in indexed if p.is_file()]
    integrity = {"schema_version": 41, "protocol_version": "rec_conv_mechanism_v32", "entries": {str(p.relative_to(root)): sha256_file(p) for p in indexed}, "entry_count": len(indexed), "independent_final_opened": False, "trace_tensors_committed": False, "historical_final_opened": False}
    index_path = root / OUT / "v32_integrity_index.json"
    write_json_atomic(index_path, integrity)
    stage = stage_freeze(root,"reports",[SOURCE,str(index_path.relative_to(root)),str(bundle.relative_to(root)),str(cumulative.relative_to(root)),"artifacts/rec_conv_mechanism_v32_final_opening.freeze.json"]+[f"reports/V32_{name}.md" for name in ORDER],{"integrity_index_sha256":sha256_file(index_path),"all_reports_sha256":sha256_file(bundle),"report_count":len(ORDER),"independent_final_opened":False})
    return {"freeze_digest":stage["freeze_digest"],"report_count":len(ORDER),"bundle_sha256":sha256_file(bundle),"integrity_entries":len(indexed),"independent_final_opened":False}


if __name__ == "__main__":
    print(json.dumps(run(Path.cwd()),indent=2))
