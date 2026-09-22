"""Generate V31 machine adjudication, 24 reports, one-file bundle and integrity index."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from jclosure.experiments.primitive_matrix_v31 import OUT, SCRATCH
from jclosure.protocol_v31 import stage_freeze, verify, verify_stage
from jclosure.provenance import sha256_file, write_json_atomic

SOURCE = "src/jclosure/experiments/report_v31.py"
ORDER = (
    "TOKEN_LIBRARY", "WRITE_PRIMITIVE_GEOMETRY", "PRIMITIVE_DICTIONARIES", "PRIMITIVE_CAUSAL_REALIZATION",
    "HELDOUT_COMPOSITION", "NONLINEAR_COMPOSITION", "PRIMITIVE_REUSE", "PRIMITIVE_EFFECT_SIGNATURES",
    "REC_CONV_INTERACTION", "REC_CONV_RESIDUAL_CORRECTION", "REC_CONV_STATE_TOKEN_DEPENDENCE",
    "CONV_DEPTH_PATTERNS", "TOKEN_CONDITIONED_DEPTH_ROUTES", "CROSS_TOKEN_PRIMITIVE_TRANSFER",
    "CROSS_STATE_PRIMITIVE_TRANSFER", "EFFECT_VS_TENSOR_PRIMITIVES", "TOKEN_LIBRARY_SCALING",
    "PRIMITIVE_SATURATION", "MULTI_HORIZON_COMPOSITION", "NATURAL_TRAJECTORY_COMPARISON",
    "STRICT_INTERFACE_AUDIT", "EXECUTION_MANIFEST", "SCIENTIFIC_ANSWERS", "COMPLETE_REPORT",
)


def f(value):
    return "n/a" if value is None or pd.isna(value) else f"{float(value):.3f}"


def table(headers, rows):
    def cell(x):
        return str(x).replace("|", "\\|").replace("\n", "\\n")
    return "| " + " | ".join(map(cell, headers)) + " |\n|" + "|".join("---" for _ in headers) + "|\n" + "\n".join("| " + " | ".join(cell(x) for x in row) + " |" for row in rows) + "\n"


def median(frame, field, condition=None):
    x = frame if condition is None else frame[frame.condition == condition]
    return f(x[field].median()) if len(x) else "n/a"


def run(root: Path):
    verify_stage(root, "final_opening")
    verify_stage(root, "record_hash_manifest")
    base = verify(root)
    d = json.loads((root / OUT / "design_v31.json").read_text())
    cal = json.loads((root / OUT / "prewrite_calibration_v31.json").read_text())
    plan = json.loads((root / OUT / "execution_plan_v31.json").read_text())
    geom = json.loads((root / OUT / "primitive_write_geometry_v31.json").read_text())
    dictionary = json.loads((root / OUT / "primitive_dictionary_fit_v31.json").read_text())
    function = json.loads((root / OUT / "function_conditioned_fit_v31.json").read_text())
    response_factor = json.loads((root / OUT / "response_factor_fit_v31.json").read_text())
    comp_fit = json.loads((root / OUT / "composition_fit_v31.json").read_text())
    comp_summary = json.loads((root / OUT / "heldout_composition_development_v31.json").read_text())
    primitive_summary = json.loads((root / OUT / "primitive_causal_development_v31.json").read_text())
    mechanism = {role: json.loads((root / OUT / f"mechanism_{role}_v31.json").read_text()) for role in ("development", "validation")}
    final = json.loads((root / OUT / "final_opening_v31.json").read_text())
    comp = pd.read_parquet(root / OUT / "heldout_composition_development_v31.parquet")
    causal = pd.read_parquet(root / OUT / "primitive_causal_development_v31.parquet")
    factorial = {role: pd.read_parquet(root / OUT / f"rec_conv_factorial_{role}_v31.parquet") for role in ("development", "validation")}
    depth = {role: pd.read_parquet(root / OUT / f"conv_depth_{role}_v31.parquet") for role in ("development", "validation")}
    energy = pd.read_parquet(root / OUT / "train_conv_depth_energy_v31.parquet")
    train_composition = pd.read_parquet(root / OUT / "composition_training_rows_v31.parquet")
    reports = {}

    def add(name, title, body):
        reports[name] = f"# {title} — V31\n\n{body.strip()}\n"

    pair_counts = pd.Series([x["role"] for x in d["token_pair_library"]]).value_counts().to_dict()
    categories = pd.DataFrame(d["token_pair_library"]).groupby(["role", "category"]).size().reset_index(name="count")
    add("TOKEN_LIBRARY", "Token Library", f"Response-blind 25-state calibration excluded all V28/V29/V30 formal IDs. Common candidate tokens at rank ≤8192: {len(cal['common_tokens'])}. The frozen anchor is ID 25 (`:`); 192 anchor→candidate pairs are split 160/16/16 TRAIN/VALIDATION/FINAL. Frozen surface-composition triples are 14/16/16; validation/final AB token IDs never enter fitting, while each A and B constituent does. All 255 states, six prewrite future probes and eligibility hashes were frozen before current-token writes.\n\n{table(('token role','frozen category','count'), categories.itertuples(index=False, name=None))}\nThe AB relation is *single-token decoded surface concatenation*, not a proven semantic/function composition. Many examples are punctuation or word fragments (e.g. `-`+`based`); ambiguous categories remain explicitly unresolved. This limits generalization beyond the tested surface proxy. No validation-composition or TOKEN_FINAL write was opened. Machine source: `results/v31/processed/design_v31.json`, hash `{d['token_library_hash']}`.")

    scale_rows = [(n, x["observed_fit_rows"], x["rank_90"], x["rank_95"], x["rank_99"]) for n, x in geom["token_library_scaling"].items()]
    add("WRITE_PRIMITIVE_GEOMETRY", "Write Primitive Geometry", f"The exact architecture-valid REC+Conv contrast has {geom['write_dimension']:,} components. The frozen 100 development fit states × 10 TRAIN contrasts produce 1000 rows. Pooled descriptive r90/r95/r99 = {geom['ranks']['r90']}/{geom['ranks']['r95']}/{geom['ranks']['r99']}; this is neither a model-state dimension nor a count of causally reusable primitives.\n\n{table(('TRAIN-token prefix','rows','r90','r95','r99'),scale_rows)}\nRank grows with token coverage. This alone cannot distinguish reusable composition from continually new causal directions. Full tensor matrix and bases stay off Git under `{SCRATCH}`; spectrum and hashes are committed.")

    model_rows = [("PCA baseline", 256, 128, dictionary["PCA"]["basis_sha256"][:16]), ("sparse dictionary", 256, 128, dictionary["SPARSE_DICTIONARY"]["basis_sha256"][:16]), ("clustered prototypes", 256, 128, dictionary["CLUSTERED_PROTOTYPES"]["basis_sha256"][:16]), ("function-conditioned", "category dependent", ", ".join(f"{k}:{v['estimable_rank']}" for k, v in function["models"].items()), sha256_file(root / OUT / "function_conditioned_fit_v31.json")[:16]), ("response factor", response_factor["training_rows"], response_factor["estimable_factor_rank"], response_factor["coordinate_npz_sha256"][:16])]
    add("PRIMITIVE_DICTIONARIES", "Primitive Dictionaries", f"P0 PCA, P1 TRAIN-only sparse dictionary, P2 TRAIN-only clustered prototypes, P3 frozen-surface-category PCA, and P4 TRAIN-only six-probe REC+Conv response factor were fit before held-out causal tests. P5 Conv-depth profiles and P6 exact REC-conditional Conv factorial are separately recorded as mechanisms, not deceptively counted as learned 128-atom dictionaries. M={{8,16,32,64,128}}, active s={{1,2,4,8,16}} when estimable. No deep autoencoder was introduced.\n\n{table(('candidate','fit span/rows','maximum estimable M/ranks','hash prefix'),model_rows)}\nCategory `UNCLASSIFIED` has only rank {function['models']['UNCLASSIFIED']['estimable_rank']}; its M64/M128 rows are explicitly not estimable. These are candidate directions, not causally validated primitives.")

    selected_conditions = ["EXACT_REC_CONV", "PCA_M32", "PCA_M64", "PCA_M128", "SPARSE_DICTIONARY_M128_s4", "SPARSE_DICTIONARY_M128_s16", "CLUSTERED_PROTOTYPES_M128_s4", "FUNCTION_CONDITIONED_M32", "RESPONSE_FACTOR_M128"]
    primitive_rows = [(name, primitive_summary["profiles"].get(name, {}).get("rows", 0), f(primitive_summary["profiles"].get(name, {}).get("median_cosine")), f(primitive_summary["profiles"].get(name, {}).get("median_magnitude")), f(primitive_summary["profiles"].get(name, {}).get("median_relative_l2")), primitive_summary["profiles"].get(name, {}).get("pass", False)) for name in selected_conditions]
    passing_dictionary = [k for k, v in primitive_summary["profiles"].items() if v["pass"] and k != "EXACT_REC_CONV"]
    add("PRIMITIVE_CAUSAL_REALIZATION", "Primitive Causal Realization", f"All {primitive_summary['tests']} development held-out AB/state pairs used the exact V29/V30 native writeback interface and six fixed future probes. Dictionary coefficients are oracle projections of the *held-out natural write* onto TRAIN-only atoms: this is a compression/reconstruction ceiling, **not** prediction of AB from A/B. Reconstruction tensor L2 is secondary; future-response cosine, magnitude and relative L2 decide.\n\n{table(('condition','rows','median cosine','median magnitude','median causal L2','gate pass'),primitive_rows)}\nPassing non-ceiling development candidates: `{passing_dictionary}`. The full machine table contains every estimable M/s condition and unavailable rows. Per-family ≥4/5 and per-row gates were applied without retuning. Because no finalist qualified, the frozen primitive validation/final panels stayed sealed.")

    comp_conditions = ("EXACT_REC_CONV", "UNIT_ADDITIVE", "GLOBAL_SCALAR_GATED", "LOW_ORDER_INTERACTION", "STATE_CONDITIONED_SCALAR_GATED", "A_ONLY", "B_ONLY", "SIGN_FLIPPED_B")
    comp_rows = [(name, comp_summary["profiles"][name]["rows"], f(comp_summary["profiles"][name]["median_cosine"]), f(comp_summary["profiles"][name]["median_magnitude"]), f(comp_summary["profiles"][name]["median_relative_l2"]), comp_summary["profiles"][name]["success_fraction"], comp_summary["profiles"][name]["pass"]) for name in comp_conditions]
    add("HELDOUT_COMPOSITION", "Held-Out Composition", f"The primary test held out the AB *single token* while its A and B constituents were individually in TRAIN. All three forks start from the same prefix cache. The 20 unseen development states × two frozen AB combinations yield 40 tests; no a→b+b→c tensor identity was used.\n\n{table(('condition','n','cosine','magnitude','causal L2','success fraction','pass'),comp_rows)}\nNo additive or low-order composition passes. Even the exact REC+Conv donor-field transplant misses the strict four-of-five-family development gate (3/5); this ceiling is not relabeled a compositional model. Frozen selection disallows trying new formulas on validation or independent final. This is failure of the *tested surface-composition proxy*, not proof that meaningful semantic primitives do not exist.")

    add("NONLINEAR_COMPOSITION", "Nonlinear Composition", f"TRAIN-only global scalar coefficients are `{[round(x, 6) for x in comp_fit['two_scalar_coefficients']]}`. The predeclared low-order bilinear formula has coefficients `{[round(x, 6) for x in comp_fit['three_scalar_coefficients']]}`; the interaction coefficient is not selected from held-out outcomes. There were {comp_fit['training_rows']} eligible TRAIN triplets across 100 fit states; training median additive *tensor* L2 = {f(train_composition.additive_write_relative_l2.median())}.\n\nHeld-out causal L2 is {f(comp_summary['profiles']['UNIT_ADDITIVE']['median_relative_l2'])} additive, {f(comp_summary['profiles']['GLOBAL_SCALAR_GATED']['median_relative_l2'])} globally gated, {f(comp_summary['profiles']['LOW_ORDER_INTERACTION']['median_relative_l2'])} low-order interaction and {f(comp_summary['profiles']['STATE_CONDITIONED_SCALAR_GATED']['median_relative_l2'])} state-conditioned gating. None reaches ≤0.30 or ≥4/5 families. Thus V31-C is not established; no large memorizing model was fit.")

    reuse_rows = []
    for model in ("SPARSE_DICTIONARY", "CLUSTERED_PROTOTYPES"):
        sub = causal[(causal.model == model) & (causal.M == 128) & (causal.active == 4)].copy()
        atom_records = []
        for _, row in sub.iterrows():
            for atom in json.loads(row.active_indices):
                atom_records.append((atom, row.token_id, row.state_id, row.family))
        grouped = pd.DataFrame(atom_records, columns=["atom", "token", "state", "family"]).groupby("atom") if atom_records else []
        counts = [(g.token.nunique(), g.state.nunique(), g.family.nunique()) for _, g in grouped] if atom_records else []
        reuse_rows.append((model, len(counts), sum(t >= 2 and s >= 2 and fam >= 2 for t, s, fam in counts), max((fam for _, _, fam in counts), default=0), primitive_summary["profiles"][f"{model}_M128_s4"]["pass"]))
    add("PRIMITIVE_REUSE", "Primitive Reuse", f"For M128/s4 oracle encodings, reuse is counted only when an atom appears with ≥2 token IDs, ≥2 states and ≥2 task families.\n\n{table(('candidate','used atoms','geometric multi-token/state/family atoms','max families/atom','causal gate'),reuse_rows)}\nA repeated OMP index is **geometric reuse only**. Since the held-out causal gate fails, no index is promoted to a reusable causal primitive. No semantic label is assigned to PCA axes or clusters.")

    add("PRIMITIVE_EFFECT_SIGNATURES", "Primitive Effect Signatures", f"The P4 response-factor directions were trained from {response_factor['training_rows']} TRAIN-only exact REC+Conv transplants on six frozen probes; rank {response_factor['estimable_factor_rank']}. Their held-out causal realization is in `primitive_causal_development_v31.parquet`. Isolated atom effects were not promoted to stable functional identities, because no dictionary passes the frozen causal gate. A TRAIN covariance or shared tensor direction alone is insufficient to claim token/state/family-stable effects. No isolated-effect transfer to validation/final was opened.")

    interaction_rows = [(role, len(factorial[role]), f(factorial[role].interaction_ratio.median()), f(factorial[role].interaction_donor_cosine.median()), f(factorial[role].rec_gain_ratio.median()), f(factorial[role].rec_direction_improvement.median())) for role in ("development", "validation")]
    add("REC_CONV_INTERACTION", "REC–Conv Interaction", f"Exact Y00 recipient, Y10 REC-only, Y01 Conv-only and Y11 REC+Conv donor fields were transplanted on the same background with recipient KV untouched. Interaction I=Y11−Y10−Y01+Y00; no additivity is assumed.\n\n{table(('role','pairs','median ‖I‖/‖donor move‖','median cos(I, donor)','REC gain ratio','joint direction improvement'),interaction_rows)}\nInteraction ratio around one quarter indicates conditionality. Gain ratio near one argues against pure scalar gain; small direction improvement and residual alignment support a corrective component. This is confined to the six-probe h1 endpoint.")

    correction_rows = [(role, len(factorial[role]), mechanism[role]["rec_improves_count"], f(factorial[role].conv_relative_l2.median()), f(factorial[role].joint_relative_l2.median()), f(factorial[role].rec_only_relative_l2.median()), f(factorial[role].residual_alignment_cosine.median()), f(factorial[role].residual_projection_fraction.median())) for role in ("development", "validation")]
    add("REC_CONV_RESIDUAL_CORRECTION", "REC–Conv Residual Correction", f"For E=Y_donor−Y_Conv and ΔREC|Conv=Y_REC+Conv−Y_Conv, the exact paired outcomes are:\n\n{table(('role','pairs','REC lowers L2','Conv L2','joint L2','REC-only L2','cos(ΔREC,E)','projection fraction'),correction_rows)}\nAll 40/40 paired errors decrease and residual alignment is positive in both roles. This independently strengthens V30's narrower conditional-REC observation, but REC-only remains weak. The frozen full-donor development family gate is only 3/5; a new post-hoc threshold for a formal V31-F finalist was **not** inserted, so independent final remains sealed. No independent REC carrier is claimed.")

    both_fact = pd.concat(factorial.values(), ignore_index=True)
    dependence_rows = [(cat, len(g), f(g.paired_l2_improvement.median()), f(g.residual_alignment_cosine.median()), f(g.interaction_ratio.median())) for cat, g in both_fact.groupby("token_category")]
    add("REC_CONV_STATE_TOKEN_DEPENDENCE", "REC–Conv State/Token Dependence", f"The REC correction varies across prewrite token categories and families; sample sizes are small and mostly unresolved surface fragments.\n\n{table(('frozen token category','pairs','median L2 improvement','median residual alignment','median interaction'),dependence_rows)}\nThis is descriptive heterogeneity, not a verified category-specific gating law. The six-probe endpoint cannot identify REC semantic content or prove a context-invariant mechanism.")

    layer_rows = [(layer, f(g.conv_energy_fraction.median()), f(g.conv_energy_fraction.quantile(.9))) for layer, g in energy.groupby("conv_layer")]
    add("CONV_DEPTH_PATTERNS", "Conv Depth Patterns", f"TRAIN-only native write contrasts yield 24,000 Conv-layer energy records (1000 writes × 24 layers). Median relative energy by layer:\n\n{table(('Conv layer','median energy fraction','p90 fraction'),layer_rows)}\nThis norm profile is not a causal layer carrier. Separately, development and validation exact partial-transplant tables include single layers, quartiles, halves, prefixes, suffixes and leave-quartile-out controls. Neither development selected a passing small route nor did the fallback full24 meet the development gate.")

    route = mechanism["development"]["route_selection"]
    category_depth = []
    for cat, group in depth["development"].groupby("token_category"):
        full = group[group.condition == "FULL_CONV"]
        variants = group[group.condition.str.startswith(("QUARTILE", "HALF", "PREFIX", "SUFFIX"))]
        best = variants.groupby("condition").relative_l2_to_donor.median().sort_values().head(1)
        category_depth.append((cat, len(full), f(full.relative_l2_to_donor.median()), best.index[0] if len(best) else "n/a", f(best.iloc[0]) if len(best) else "n/a"))
    add("TOKEN_CONDITIONED_DEPTH_ROUTES", "Token-Conditioned Depth Routes", f"Frozen development-only route rule selected `{route['name']}` as a **fallback**, not a passing route. All 24 Conv layers were retained.\n\n{table(('token category','development pairs','full-Conv L2','best descriptive group','group L2'),category_depth)}\nPer-category minima are exploratory and were not promoted to validation after selection; sparse categories and failure of the full-Conv development gate preclude a token-conditioned depth-route claim. TRAIN energy differences do not substitute for predicted-group versus wrong/random matched-group causal superiority. V31-H is not established.")

    add("CROSS_TOKEN_PRIMITIVE_TRANSFER", "Cross-Token Primitive Transfer", "All dictionary atoms are TRAIN-only, and development held-out AB IDs are token-OOD. Their oracle coefficients and exact writebacks were measured in the primitive-causal grid. However, using the held-out natural write to choose coefficients is **not** prediction or transplantation of a preidentified causal atom across tokens. No model passed the frozen development causal gate; an isolated cross-token primitive transfer finalist was therefore not taken to validation/final. Geometric index recurrence is not called causal transfer.")
    add("CROSS_STATE_PRIMITIVE_TRANSFER", "Cross-State Primitive Transfer", "The 20 development holdout states are disjoint from all 100 dictionary-fit states and all V28/V29/V30 formal panels. Candidate atoms were applied through native target-state writeback and judged on six-probe responses. No candidate passed the causal/family gate, so there is no validated state-transportable primitive. Exact native donor-field transplants are controls, not evidence of a transferable learned primitive. Independent final was not opened.")

    effect_rows = []
    for condition in ("PCA_M128", "SPARSE_DICTIONARY_M128_s16", "CLUSTERED_PROTOTYPES_M128_s16", "FUNCTION_CONDITIONED_M32", "RESPONSE_FACTOR_M128"):
        x = causal[causal.condition == condition].dropna(subset=["write_relative_l2", "relative_l2_to_donor"])
        corr = x[["write_relative_l2", "relative_l2_to_donor"]].corr(method="spearman").iloc[0, 1] if len(x) > 2 else None
        effect_rows.append((condition, len(x), f(x.write_relative_l2.median()) if len(x) else "n/a", f(x.relative_l2_to_donor.median()) if len(x) else "n/a", f(corr)))
    add("EFFECT_VS_TENSOR_PRIMITIVES", "Effect versus Tensor Primitives", f"Tensor reconstruction and causal response were recorded on the same held-out rows.\n\n{table(('candidate','estimable rows','median tensor L2','median causal L2','Spearman across rows'),effect_rows)}\nNeither small tensor error nor positive rank correlation would by itself validate a primitive; the frozen donor-cosine/magnitude/L2/family gate is primary. These oracle projections are not held-out AB prediction.")

    add("TOKEN_LIBRARY_SCALING", "Token-Library Scaling", f"Frozen nested TRAIN-token prefixes 16/32/64/128/160 give pooled r95 39/65/109/184/220.\n\n{table(('tokens','observed writes','r95'),[(n,x['observed_fit_rows'],x['rank_95']) for n,x in geom['token_library_scaling'].items()])}\nThe chosen response-blind library has 160 TRAIN tokens plus 32 held-out AB tokens; the suggested 256-token point was not pre-registered as an executable library and has no causal result. No false 256-token extrapolation is made. Coverage changes both row count and token diversity; this is descriptive scaling.")

    add("PRIMITIVE_SATURATION", "Primitive Saturation", "The frozen saturation criterion was <20% growth in *causally required primitive count* when token coverage doubles. No dictionary reaches the strict development causal gate, and no 256-token causal scale was tested. Therefore a causally required primitive vocabulary and its new-primitive rate are **not estimable**; V31-I and V31-J cannot be decided from PCA rank. Pooled write rank rises, but this neither proves nor refutes a potentially larger sparse reusable grammar.")
    add("MULTI_HORIZON_COMPOSITION", "Multi-Horizon Composition", "The frozen h2/h4 rule required an h1-qualified compositional mechanism. None passed the strict development causal/family gate. Consequently h2/h4 composition was not opened. V30's own exact-write horizon results remain historical and are not silently imported as V31 primitive persistence.")
    add("NATURAL_TRAJECTORY_COMPARISON", "Natural Trajectory Comparison", "No causally validated primitive/composition finalist was identified. An observational search for similar signatures in unconstrained natural generation was not undertaken because it would have no frozen primitive identity to compare and cannot replace a transplant test. This omission limits claims about unperturbed trajectories; no natural-use or semantic-module claim is made.")

    add("STRICT_INTERFACE_AUDIT", "Strict Interface Audit", f"All 255 prospective prefix-state IDs were response-blind frozen and exclude V28/V29/V30 formal IDs. Model-visible current-token forks share the same native incoming cache and equal cache length; donor and recipient complete token computation naturally. Exact REC, Conv and REC+Conv partial writes alter only their named native fields; recipient KV remains native. Off-manifold low-rank/dictionary `inject` clones recipient and writes only REC+Conv. Six next-token probes are prefix-logit-selected before the write and held fixed within each comparison. Incoming state, token pair, composition, probe, natural write, basis and response hashes are machine-recorded; per-atom and factorial-metric hashes are in `v31_record_hash_manifest.json`. Exact REC+Conv is a *field* ceiling, not full-cache identity. Historical reports/protocols/finals were not overwritten or reopened. The V31 50-state/TOKEN_FINAL independent final, validation composition, and validation primitive grid remain sealed. Large matrices/bases live only under `{SCRATCH}` outside Git; committed hashes identify them. The earlier 256-basis whole-GPU attempt failed OOM before any fit freeze; the identical frozen rows were streamed by columns without changing method or thresholds.")

    stage_rows = []
    for path in sorted((root / "artifacts").glob("compositional_natural_writes_v31*.freeze.json")):
        record = json.loads(path.read_text())
        stage_rows.append((path.name, record.get("freeze_digest", "")[:16], sha256_file(path)[:16]))
    add("EXECUTION_MANIFEST", "Execution Manifest", f"Parent commit `{base['parent_commit']}`; base freeze `{base['freeze_digest']}`. Frozen panels 25/120/60/50; 192 token pairs 160/16/16; surface-composition triples 14/16/16; six future probes. TRAIN fits: 100 states, 1000 natural write rows, 1382 eligible composition examples, 200 exact REC+Conv response-factor examples. Development tested 40 AB compositions and {primitive_summary['tests']} primitive contrasts; REC/Conv factorial and depth each used 20 pairs in development and 20 in validation. No independent final opened. A whole-GPU basis attempt hit concurrent GPU OOM; same frozen data were streamed by columns, with no gate/model change.\n\n{table(('freeze artifact','digest prefix','file hash prefix'),stage_rows)}\nDecision file `results/v31/processed/final_opening_v31.json` SHA-256 `{sha256_file(root / OUT / 'final_opening_v31.json')}`. Authorization remains H2 true, H3/dynamic search/autonomous controller/cross-model false. Tests are recorded in `tests/test_v31_protocol.py`; off-repository scratch is not Git-tracked.")

    outcomes = {
        "V31_A_REUSABLE_CAUSAL_WRITE_PRIMITIVES_CONFIRMED": False,
        "V31_B_COMPOSITIONAL_WRITE_GENERALIZATION": False,
        "V31_C_NONLINEAR_WRITE_COMPOSITION": False,
        "V31_D_FUNCTIONAL_PRIMITIVES_WITH_STATE_DEPENDENT_REALIZATION": False,
        "V31_E_TOKEN_FAMILY_SPECIFIC_WRITE_MODULES": False,
        "V31_F_REC_CORRECTS_CONV_WRITE": False,
        "V31_G_REC_ROTATES_OR_GATES_CONV_WRITE": False,
        "V31_H_TOKEN_CONDITIONED_CONV_DEPTH_ROUTES": False,
        "V31_I_PRIMITIVE_DICTIONARY_SATURATES": False,
        "V31_J_OPEN_ENDED_WRITE_COMPLEXITY": False,
        "V31_K_NO_STABLE_COMPOSITIONAL_STRUCTURE_IDENTIFIED": True,
    }
    adjudication = {"outcomes": outcomes, "interpretation": "No tested surface-composition or candidate dictionary passed the strict causal/family gate; V31-K is restricted to this panel and model grid. REC residual correction replicated descriptively on exact writes but no predeclared F-specific threshold/full-donor finalist qualified. Rank growth alone cannot decide saturation versus open-ended primitives.", "validation_composition_opened": False, "validation_primitive_opened": False, "independent_final_opened": False, "cross_model_replication_authorized": False, "H2_REMAINS": True, "H3_AUTHORIZED": False, "historical_final_opened": False}
    adj_path = root / OUT / "v31_adjudication.json"
    write_json_atomic(adj_path, adjudication)

    answers = [
        "1. No tested dictionary causally reconstructs held-out writes at the frozen gate.",
        "2. No causally sufficient primitive count is identified; M=8/16/32/64/128 were screened where estimable.",
        "3. Sparse s=1/2/4/8/16 were screened where supported; no active count qualified causally.",
        "4–6. Some atoms repeat geometrically across tokens, states and families, but none is validated as a causal reusable identity.",
        "7. No: held-out causal reuse gate failed.",
        "8. No: held-out surface AB effects were not predicted by tested A/B composition.",
        "9. No: unit additive causal L2 median " + f(comp_summary["profiles"]["UNIT_ADDITIVE"]["median_relative_l2"]) + ".",
        "10. No low-order nonlinear rule passed; necessity of nonlinear composition in general is undetermined.",
        "11–12. Token-OOD and joint state+token-OOD surface composition failed in development; validation stayed sealed under the selection rule.",
        "13. Stable causal effect signatures despite tensor changes were not demonstrated.",
        "14. Frozen surface categories do not establish function-specific causal primitives.",
        "15. Primitive-count saturation is not estimable because no causal dictionary passed.",
        "16. Yes, descriptive pooled r95 rises 39→220 across tested 16→160 TRAIN tokens.",
        "17. Coexistence of growing rank with stable primitive vocabulary is not demonstrated.",
        "18. Exact REC adds a positive residual-aligned correction in 40/40 development+validation pairs, but strict formal F gate was not qualified.",
        "19. Pure gain is disfavored (gain ratio near 1); correction plus small rotation is compatible, not uniquely identified.",
        "20. Correction varies by state/category, but sparse fragment classes prevent a confirmed token law.",
        "21. TRAIN Conv depth energy varies; no causally validated class-specific route.",
        "22. No predicted small depth route passed the development gate; full24 is fallback.",
        "23–24. Cross-token and cross-state oracle dictionary tests failed; no predictive primitive transport is established.",
        "25. A-only, B-only and sign-flipped controls fail; natural exact write is stronger, but no composition passes.",
        "26. h2/h4 not opened because no h1 compositional mechanism qualified.",
        "27. V31-A false/not identified.",
        "28. V31-B/C/D/E not supported.",
        "29. V31-F/G not formally confirmed; exact REC residual improvement replicated descriptively.",
        "30. V31-H not supported.",
        "31. Primitive vocabulary saturation not demonstrated.",
        "32. Open-ended primitive complexity not demonstrated; only write-rank growth is observed.",
        "33. Cross-model replication not authorized under the frozen qualification rule.",
    ]
    add("SCIENTIFIC_ANSWERS", "Scientific Answers", "These answers refer only to this hybrid model, frozen surface-token library and six-probe h1 response.\n\n" + "\n\n".join(answers) + "\n\nNo compact model-state dimension, semantic atom, or general impossibility theorem is inferred.")

    outcome_rows = [(name, value) for name, value in outcomes.items()]
    complete = f"""**Compositional Structure of Natural State Writes — Are High-Dimensional Future-Facing Writes Built from Reusable Causal Primitives?** Parent `{base['parent_commit']}`; frozen base `{base['freeze_digest']}`. Prospectively frozen 25/120/60/50 disjoint panels and 192 response-blind token pairs. Independent final remained sealed.

The 1000 TRAIN natural REC+Conv contrasts have 13,369,344 components, pooled r95={geom['ranks']['r95']}, rising 39→65→109→184→220 over 16→32→64→128→160 TRAIN tokens. This is geometric growth, not a causal primitive count.

The decisive 40-case unseen surface-AB development test failed: additive, global scalar, low-order interaction and state-conditioned scalar causal L2 medians are {f(comp_summary['profiles']['UNIT_ADDITIVE']['median_relative_l2'])}/{f(comp_summary['profiles']['GLOBAL_SCALAR_GATED']['median_relative_l2'])}/{f(comp_summary['profiles']['LOW_ORDER_INTERACTION']['median_relative_l2'])}/{f(comp_summary['profiles']['STATE_CONDITIONED_SCALAR_GATED']['median_relative_l2'])}; none passes the ≤0.30, cosine≥0.90, magnitude[0.8,1.2], ≥4/5-family gate. Sparse/prototype/function-conditioned/response-factor oracle reconstruction candidates likewise yielded no qualifying development primitive. This does **not** establish that semantic or other composition is impossible: the available AB triples are mostly surface fragments, and validation was sealed after development failures.

The exact REC×Conv factorial independently repeated a narrower result: REC lowers Conv donor error in 20/20 development and 20/20 validation pairs; residual alignment cosine medians are {f(factorial['development'].residual_alignment_cosine.median())}/{f(factorial['validation'].residual_alignment_cosine.median())}; interaction ratios {f(factorial['development'].interaction_ratio.median())}/{f(factorial['validation'].interaction_ratio.median())}. REC-only remains weak, gain ratio ≈1 and full24 Conv is a fallback, not a qualified small route. Because the frozen full-donor development gate is only 3/5 families and no F-specific numeric finalist threshold was predeclared, this replicated correction is not promoted to a formal V31-F finalist or used to open the independent final.

{table(('formal outcome','supported'),outcome_rows)}
V31-K means **no stable tested compositional structure identified**, not no structure exists. Causal primitive saturation versus open-ended primitive count remains unresolved; rank growth cannot decide it. H2 remains true; H3, dynamic state search, autonomous controller and cross-model replication remain unauthorized. Full files and hashes are in `results/v31/processed/`; `V31_ALL_REPORTS.md` reproduces each V31 report separately."""
    add("COMPLETE_REPORT", "Complete Report", complete)

    if set(reports) != set(ORDER):
        raise RuntimeError(f"V31 report set mismatch: {set(reports) ^ set(ORDER)}")
    report_dir = root / "reports"
    for name in ORDER:
        (report_dir / f"V31_{name}.md").write_text(reports[name])
    parts = ["# V31 — All Reports in One File\n\nThe 24 standalone V31 reports are reproduced below in frozen order. The individual files are authoritative.\n"]
    for i, name in enumerate(ORDER, 1):
        path = report_dir / f"V31_{name}.md"
        parts.append(f"\n---\n\n<!-- {i:02d}: {path.name}; sha256={sha256_file(path)} -->\n\n{path.read_text()}")
    bundle = report_dir / "V31_ALL_REPORTS.md"
    bundle.write_text("".join(parts))
    final_report = report_dir / "FINAL_REPORT.md"
    existing = final_report.read_text()
    if "# Complete Report — V31" not in existing:
        final_report.write_text(existing.rstrip() + "\n\n---\n\n" + reports["COMPLETE_REPORT"])
    else:
        if reports["COMPLETE_REPORT"] not in existing:
            raise RuntimeError("Existing V31 cumulative section differs; refusing overwrite")
    indexed = [root / "configs/compositional_natural_writes_v31.yaml", root / "src/jclosure/protocol_v31.py", root / "tests/test_v31_protocol.py"]
    indexed += sorted((root / "src/jclosure/experiments").glob("*v31.py"))
    indexed += sorted((root / "artifacts").glob("compositional_natural_writes_v31*.freeze.json"))
    indexed += [x for x in sorted((root / OUT).glob("*")) if x.is_file() and x.name != "v31_integrity_index.json"]
    indexed += [report_dir / f"V31_{name}.md" for name in ORDER]
    indexed += [bundle, final_report]
    indexed = [x for x in indexed if x.is_file()]
    integrity = {"schema_version": 31, "protocol": "compositional_natural_writes_v31", "entries": {str(x.relative_to(root)): sha256_file(x) for x in indexed}, "entry_count": len(indexed), "independent_final_opened": False, "validation_composition_opened": False, "validation_primitive_opened": False, "scratch_tensor_files_committed": False, "historical_final_opened": False}
    index_path = root / OUT / "v31_integrity_index.json"
    write_json_atomic(index_path, integrity)
    inputs = [SOURCE, str(index_path.relative_to(root)), str(bundle.relative_to(root)), str(final_report.relative_to(root)), str(adj_path.relative_to(root)), "artifacts/compositional_natural_writes_v31_final_opening.freeze.json"] + [f"reports/V31_{name}.md" for name in ORDER]
    freeze = stage_freeze(root, "reports", inputs, {"integrity_index_sha256": sha256_file(index_path), "all_reports_sha256": sha256_file(bundle), "complete_report_sha256": sha256_file(report_dir / "V31_COMPLETE_REPORT.md"), "report_count": len(ORDER), "independent_final_opened": False})
    return {"freeze_digest": freeze["freeze_digest"], "report_count": len(ORDER), "bundle_sha256": sha256_file(bundle), "integrity_entries": len(indexed), "final_opened": False}


if __name__ == "__main__":
    print(json.dumps(run(Path.cwd()), indent=2))
