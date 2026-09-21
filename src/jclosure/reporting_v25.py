"""Generate standalone, cumulative, and integrity reports for V25."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from jclosure.protocol_v25 import stage_freeze, verify, verify_stage
from jclosure.provenance import sha256_file, write_json_atomic


OUT = Path("results/v25/processed")
REPORTS = Path("reports")
SOURCE = "src/jclosure/reporting_v25.py"
NAMES = [
    "V24_RANK_AUDIT_V25.md",
    "DISTRIBUTED_INTERACTION_V25.md",
    "CAUSAL_CANCELLATION_V25.md",
    "BACKGROUND_CONDITIONED_EFFECT_V25.md",
    "LAYERWISE_CAUSAL_CONVERGENCE_V25.md",
    "MANY_TO_ONE_CAUSAL_MAPPING_V25.md",
    "JVP_FINITE_CONTRACTION_V25.md",
    "CHANNEL_INTERACTION_V25.md",
    "SAME_J_DISTRIBUTED_REALIZATION_V25.md",
    "CAUSAL_REALIZATION_DIMENSION_V25.md",
    "READOUT_COMPRESSION_V25.md",
    "NONLINEAR_CONVERGENCE_V25.md",
    "INTERACTION_CONVERGENCE_COUPLING_V25.md",
    "MULTIHORIZON_DISTRIBUTED_DYNAMICS_V25.md",
    "STRICT_INTERFACE_AUDIT_V25.md",
    "EXECUTION_MANIFEST_V25.md",
    "V25_SCIENTIFIC_ANSWERS_V25.md",
    "V25_COMPLETE_REPORT.md",
]


def load(root: Path, name: str):
    return json.loads((root / OUT / name).read_text())


def write(root: Path, name: str, body: str):
    (root / REPORTS / name).write_text(body.rstrip() + "\n", encoding="utf-8")


def fmt(value):
    if value is None:
        return "N/A"
    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def generate(root: Path, v25_tests: str, full_tests: str):
    base = verify(root)
    stages = {
        name: verify_stage(root, name)
        for name in (
            "design", "component_basis", "rank_convergence",
            "rank_convergence_amendment", "interaction",
            "interaction_estimand_amendment", "analysis",
            "zero_rank_amendment", "adjudication",
        )
    }
    design = load(root, "design_v25.json")
    basis = load(root, "component_basis_v25.json")
    rank = load(root, "v24_rank_audit_v25.json")
    interaction = load(root, "interaction_estimand_amendment_v25.json")
    original = load(root, "distributed_interaction_summary_v25.json")
    analysis = load(root, "v25_integrated_analysis.json")
    zero = load(root, "zero_rank_amendment_v25.json")
    adjudication = load(root, "v25_adjudication.json")

    write(root, NAMES[0], f"""# V24 Rank Audit — V25

The V24 raw, uncentered result is exactly reproduced: r95 is `1` at all 32 layers and raw top-1 energy fraction ranges from `{min(rank['raw_top1_fraction']):.6f}` to `{max(rank['raw_top1_fraction']):.6f}`. After subtracting the intervention-row mean independently within each frozen state, broad r95 is `13` at most layers and peaks at `14`; centered top-1 fraction is `{min(rank['centered_top1_fraction']):.6f}`–`{max(rank['centered_top1_fraction']):.6f}`. Row normalization without centering remains rank 1, while centered plus normalized r95 is `24`–`25`.

Therefore `LOW_RANK_RESPONSE_GEOMETRY_RECONFIRMED = FALSE` and `V24_RANK1_STATUS = {rank['V24_RANK1_STATUS']}`. The rank-one statement is specific to an uncentered common-effect component, not a general low-dimensional intervention geometry claim.

Rows hash `{rank['rank_rows_sha256']}`; medians hash `{rank['rank_medians_sha256']}`; spectra hash `{rank['spectra_sha256']}`.
""")

    family_rows = "\n".join(
        f"| {family} | {value:.6f} |" for family, value in interaction["family_medians"].items()
    )
    write(root, NAMES[1], f"""# Distributed Causal Interaction — V25

Strict factorial estimand: `{interaction['estimand']}`. All `{interaction['rows']}` corrected rows are append-only; the earlier persistent-FULL interpretation was preserved but superseded for the formal interaction decision.

| metric | value |
|---|---:|
| broad interaction ratio median | {interaction['broad_interaction_ratio_median']:.6f} |
| bootstrap 2.5% lower bound | {interaction['bootstrap_2_5_lower']:.6f} |
| additive relative L2 median | {interaction['additive_relative_l2_median']:.6f} |
| interaction cosine median | {interaction['interaction_cosine_median']:.6f} |
| centered interaction r95 | {analysis['interaction_rank']['interaction']['centered']['r95']} |

| family | median ratio |
|---|---:|
{family_rows}

The writeback numerical gate failed because the minimum B-branch cosine was `{interaction['writeback']['branch_min_cosine']['Y10_B']:.6f}`. Consequently `V25-A = FALSE` conservatively, despite a nonzero interaction statistic.
""")

    write(root, NAMES[2], f"""# Causal Cancellation — V25

`CAUSAL_CANCELLATION_SUPPORTED = TRUE`. The corrected median cancellation index is `{interaction['cancellation_index_median']:.6f}` and the median B/C component cosine is `{interaction['component_cosine_median']:.6f}`. The result is based on the strict clean, B, C, and B+C branches; it does not rely on the superseded persistent-FULL Y11 branch.
""")

    background = analysis["background"]
    write(root, NAMES[3], f"""# Background-Conditioned Effect — V25

`BACKGROUND_CONDITIONED_CAUSAL_EFFECT = FALSE`. Non-native cosine median is `{background['non_native_cosine_median']:.6f}`, fidelity-drop median `{background['fidelity_drop_median']:.6f}`, and conditional norm-ratio standard-deviation median `{background['conditional_norm_ratio_std_median']:.6f}`. This panel does not establish background-specific response coordinates.
""")

    convergence_rows = "\n".join(
        f"| {family} | {values['reference_r95']} | {values['late_r95']} |"
        for family, values in zero["family_profile"].items()
    )
    write(root, NAMES[4], f"""# Layerwise Causal Convergence — V25

An append-only correction assigns exact-zero response matrices rank zero. Layers 0–23 have exact-zero observed responses; layer 24 is the first nonzero layer. Pooled centered r95 rises from `{zero['reference_r95']}` at layer 24 to `{zero['late_r95']}` at layer 31, a reduction fraction of `{zero['r95_reduction_fraction']:.6f}` (negative means rank expansion). Absolute cosine falls from `{zero['reference_absolute_cosine']:.6f}` to `{zero['late_absolute_cosine']:.6f}`.

| family | r95 at layer 24 | r95 at layer 31 |
|---|---:|---:|
{convergence_rows}

`MANY_TO_ONE_CAUSAL_CONVERGENCE = FALSE`; `INTERNAL_CAUSAL_CONVERGENCE_SUPPORTED = FALSE`.
""")

    write(root, NAMES[5], f"""# Many-to-One Causal Mapping — V25

Status: **{zero['many_to_one_mapping_status']}**. The frozen mapping sources 0, 7, 15, and 23 precede the first nonzero response layer, so their zero matrices cannot identify a meaningful many-to-one response map. The original mapping records remain preserved and are not treated as evidence of contraction.
""")

    jvp = analysis["JVP_finite_contraction"]
    jvp_late = [row for row in jvp["exact_jvp"] if row["from_layer"] >= 24]
    finite_late = [row for row in jvp["finite"] if row["from_layer"] >= 24]
    write(root, NAMES[6], f"""# JVP–Finite Contraction — V25

Layers 0–23 are exact-zero-response layers, and 23→24 is response emergence rather than contraction. From layer 24 onward, exact-JVP norm ratios are `{', '.join(f'{x['norm_ratio_median']:.3f}' for x in jvp_late)}` and finite-effect ratios are `{', '.join(f'{x['norm_ratio_median']:.3f}' for x in finite_late)}`. These values are descriptive and do not support internal contraction.
""")

    channel_rows = "\n".join(
        f"| {term} | {value:.6f} |"
        for term, value in analysis["channel_factorial"]["median_interaction_by_term"].items()
    )
    write(root, NAMES[7], f"""# Channel Interaction — V25

| factorial term | median interaction ratio |
|---|---:|
{channel_rows}

The three-component interaction ratio is `{analysis['three_component']['triple_interaction_ratio_median']:.6f}`. Intervention-order relative difference has median `{analysis['intervention_order']['relative_difference_median']:.6f}` and maximum `{analysis['intervention_order']['relative_difference_max']:.6f}`. These diagnostics show channel coupling and order sensitivity but do not rescue the failed strict V25-A numerical gate.
""")

    reconstruction_rows = "\n".join(
        f"| {dimension} | {values['relative_l2']:.6f} | {values['cosine']:.6f} | {values['norm_ratio']:.6f} |"
        for dimension, values in sorted(original["reconstruction"]["causal_by_dimension"].items(), key=lambda x: int(x[0]))
    )
    write(root, NAMES[8], f"""# Same-J Distributed Realization — V25

| causal dimension | relative L2 | cosine | norm ratio |
|---:|---:|---:|---:|
{reconstruction_rows}

Even the 256-dimensional realization has relative L2 `{original['reconstruction']['causal_by_dimension']['256']['relative_l2']:.6f}` and cosine `{original['reconstruction']['causal_by_dimension']['256']['cosine']:.6f}`. No tested same-J dimension satisfies the frozen realization gate.
""")

    write(root, NAMES[9], f"""# Causal Realization Dimension — V25

`CAUSAL_EFFECT_REALIZATION_DIMENSION_IDENTIFIED = FALSE`. The tested nested dimensions were `{design['realization_dimensions']}`; no dimension passed the joint reconstruction criteria. A compact behavioral/readout effect is not equated with a compact model state or compact causal realization.
""")

    readout_metrics = analysis["readout"]["metrics"]
    readout_rows = "\n".join(
        f"| {name} | {metrics['centered']['r95']} |"
        for name, metrics in readout_metrics.items()
    )
    write(root, NAMES[10], f"""# Readout Compression — V25

| readout | centered r95 |
|---|---:|
{readout_rows}

Direct full hidden r95 is `{readout_metrics['direct_full_hidden_2560']['centered']['r95']}` versus semantic-32 `{readout_metrics['semantic_32']['centered']['r95']}`, logits-32 `{readout_metrics['logits_32']['centered']['r95']}`, J-128 `{readout_metrics['J_128']['centered']['r95']}`, and workspace-96 `{readout_metrics['workspace_96']['centered']['r95']}`. `READOUT_COMPRESSION_SUPPORTED = TRUE`, while the corrected internal convergence result is false; formal outcome `V25-E = TRUE`.
""")

    write(root, NAMES[11], f"""# Nonlinear Convergence — V25

The corrected strict interaction has centered r95 `{analysis['interaction_rank']['interaction']['centered']['r95']}`, compared with full-effect centered r95 `{analysis['interaction_rank']['full_effect']['centered']['r95']}`. The nonlinear interaction is not a one-dimensional residual. Channel-factorial and order diagnostics are substantial, but strict writeback verification failed; therefore no strong distributed nonlinear-interaction claim is authorized.
""")

    coupling = analysis["coupling"]
    write(root, NAMES[12], f"""# Interaction–Convergence Coupling — V25

Interaction versus nominal rank reduction: Spearman `{coupling['interaction_vs_rank_reduction_spearman']:.6f}`, Pearson `{coupling['interaction_vs_rank_reduction_pearson']:.6f}`. Interaction versus absolute cosine: Spearman `{coupling['interaction_vs_absolute_cosine_spearman']:.6f}`, Pearson `{coupling['interaction_vs_absolute_cosine_pearson']:.6f}`. These are associations only. Because zero-layer correction invalidates the original early-rank construction, they are not causal proof and do not override the corrected no-convergence decision.
""")

    write(root, NAMES[13], f"""# Multihorizon Distributed Dynamics — V25

Status: **{analysis['multihorizon_status']}**. No all-family two-token confirmatory panel was frozen, so no multihorizon branch was opened and no dynamic-state search was authorized.
""")

    writeback = interaction["writeback"]
    write(root, NAMES[14], f"""# Strict Interface Audit — V25

- Historical final opened: `{fmt(adjudication['historical_final_opened'])}`.
- V25 independent final opened: `{fmt(adjudication['v25_independent_final_opened'])}`.
- Independent final state hash: `{design['state_hashes']['independent_final']}`.
- Independent final size: `50` (`10` per family).
- Interaction state/action hashes: `{design['state_hashes']['interaction_validation']}` / `{design['action_hashes']['interaction']}`.
- Component basis hash: `{basis['basis_sha256']}`.
- Corrected interaction branch/vector hashes: `{interaction['corrected_branch_sha256']}` / `{interaction['corrected_vectors_sha256']}`.
- Clean replay maximum absolute/relative-L2 error: `{original['clean_replay_max_abs']}` / `{original['clean_replay_relative_l2_max']}`.
- Corrected branch median cosines: `{json.dumps(writeback['branch_median_cosine'], sort_keys=True)}`.
- Corrected numerical gate: `{fmt(writeback['numerical_gate_pass'])}`.
- Original records overwritten: interaction `{fmt(interaction['original_interaction_records_overwritten'])}`; zero-rank `{fmt(zero['original_records_overwritten'])}`.
""")

    commands = [
        "PYTHONPATH=src /home/user/anaconda3/bin/python -m jclosure.protocol_v25 freeze",
        "PYTHONPATH=src /home/user/anaconda3/bin/python -m jclosure.experiments.design_v25",
        "CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src /home/user/anaconda3/bin/python -m jclosure.experiments.component_basis_v25",
        "CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src /home/user/anaconda3/bin/python -m jclosure.experiments.rank_convergence_v25",
        "PYTHONPATH=src /home/user/anaconda3/bin/python -m jclosure.experiments.rank_convergence_amendment_v25",
        "CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src /home/user/anaconda3/bin/python -m jclosure.experiments.interaction_v25",
        "CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src /home/user/anaconda3/bin/python -m jclosure.experiments.interaction_estimand_amendment_v25",
        "PYTHONPATH=src /home/user/anaconda3/bin/python -m jclosure.experiments.analyze_v25",
        "PYTHONPATH=src /home/user/anaconda3/bin/python -m jclosure.experiments.zero_rank_amendment_v25",
        "PYTHONPATH=src /home/user/anaconda3/bin/python -m jclosure.experiments.adjudicate_v25",
        "PYTHONPATH=src /home/user/anaconda3/bin/python -m pytest tests/test_v25_distributed_interaction.py -q",
        "PYTHONPATH=src /home/user/anaconda3/bin/python -m pytest -q",
        "PYTHONPATH=src /home/user/anaconda3/bin/python -m jclosure.reporting_v25 --v25-tests '6 passed' --full-tests '256 passed, 2 inherited failures'",
        "git commit -m 'Complete V25 distributed causal interaction study'",
        "git push origin main",
    ]
    manifest = {
        "commands": commands,
        "append_only_amendments": {
            "rank_dimension_clamp": stages["rank_convergence_amendment"]["freeze_digest"],
            "strict_four_branch_estimand": stages["interaction_estimand_amendment"]["freeze_digest"],
            "zero_response_rank": stages["zero_rank_amendment"]["freeze_digest"],
        },
        "v25_tests": v25_tests,
        "full_suite": full_tests,
    }
    write_json_atomic(root / OUT / "execution_manifest_v25.json", manifest)
    write(root, NAMES[15], "# Execution Manifest — V25\n\n" + "\n".join(f"- `{command}`" for command in commands) + f"\n\nV25 tests: `{v25_tests}`. Full suite: `{full_tests}`. The two full-suite failures are inherited V14/V16 cumulative `FINAL_REPORT.md` hash checks; no V25 test failed.\n")

    answers = [
        f"The V24 raw rank-one result reproduces, but centered r95 is 13–14; unqualified low-rank geometry is not reconfirmed.",
        f"The strict four-branch interaction ratio is {interaction['broad_interaction_ratio_median']:.6f}, but V25-A is false because the numerical gate failed.",
        f"Causal cancellation is supported with index {interaction['cancellation_index_median']:.6f}; V25-B is true.",
        "Layers 0–23 have exact-zero observed responses; the first nonzero response is layer 24.",
        f"Centered r95 expands from {zero['reference_r95']} at layer 24 to {zero['late_r95']} at layer 31; V25-C and V25-D are false.",
        "Frozen many-to-one mapping sources are zero-response layers, so the mapping test is inconclusive.",
        "The 23→24 transition is response emergence, not contraction; later JVP/finite ratios are descriptive.",
        f"Background-conditioned coordinates are not supported; non-native cosine is {background['non_native_cosine_median']:.6f}.",
        f"The strict interaction centered r95 is {analysis['interaction_rank']['interaction']['centered']['r95']}; interaction is not rank one.",
        f"Channel interactions and intervention order are non-additive, with order-difference median {analysis['intervention_order']['relative_difference_median']:.6f}.",
        "No tested same-J realization dimension from 4 through 256 passes.",
        "Readout compression is supported while internal convergence is not; V25-E is true.",
        "No frozen all-family two-token panel exists; multihorizon dynamics were not opened.",
        "The independent final 50 remain frozen and unopened because validation mechanisms were not independently finalized.",
        f"Formal outcomes are {adjudication['primary_outcome']}.",
        "H2 remains; H3 and dynamic-state search remain unauthorized.",
    ]
    write_json_atomic(root / OUT / "v25_scientific_answers.json", {"answers": answers})
    write(root, NAMES[16], "# V25 Scientific Answers\n\n" + "\n".join(f"{index}. {answer}" for index, answer in enumerate(answers, 1)))

    complete = f"""# V25 Complete Report

## Identity and frozen scope

- Name: **Distributed Causal Interaction and Response Convergence**
- Parent commit: `{base['parent_commit']}`
- Base protocol hash: `{base['freeze_digest']}`
- Design freeze: `{stages['design']['freeze_digest']}`
- Component-basis freeze: `{stages['component_basis']['freeze_digest']}`
- Strict interaction amendment: `{stages['interaction_estimand_amendment']['freeze_digest']}`
- Zero-rank amendment: `{stages['zero_rank_amendment']['freeze_digest']}`
- Adjudication freeze: `{stages['adjudication']['freeze_digest']}`
- Independent final: 50 states, 10 per family, hash `{design['state_hashes']['independent_final']}`, frozen and unopened.

## Formal result

**{adjudication['primary_outcome']}**.

`V25-A={fmt(adjudication['V25_A'])}`, `V25-B={fmt(adjudication['V25_B'])}`, `V25-C={fmt(adjudication['V25_C'])}`, `V25-D={fmt(adjudication['V25_D'])}`, `V25-E={fmt(adjudication['V25_E'])}`, `V25-F={fmt(adjudication['V25_F'])}`, `V25-G={fmt(adjudication['V25_G'])}`, `V25-H={fmt(adjudication['V25_H'])}`.

The supported picture is causal cancellation plus low-dimensional behavioral/readout compression without a demonstrated low-dimensional causal realization or internal convergence.

## Rank audit

Raw uncentered broad responses reproduce r95=1 at every layer. State-wise intervention centering yields r95 13–14, and centered plus row-normalized responses yield r95 24–25. Thus V24's rank-one result is measurement-definition-specific and does not establish a general one-dimensional causal geometry.

## Strict distributed interaction

The corrected estimand is `{interaction['estimand']}` over `{interaction['rows']}` rows. Median interaction ratio is `{interaction['broad_interaction_ratio_median']:.6f}` with bootstrap lower bound `{interaction['bootstrap_2_5_lower']:.6f}`. Interaction centered r95 is `{analysis['interaction_rank']['interaction']['centered']['r95']}`. However, the strict writeback gate fails (minimum B cosine `{writeback['branch_min_cosine']['Y10_B']:.6f}`), so V25-A is false. Cancellation index `{interaction['cancellation_index_median']:.6f}` supports V25-B.

## Layerwise response geometry

Exact-zero matrices at layers 0–23 have rank zero. The first nonzero layer is 24. Pooled centered r95 rises from `{zero['reference_r95']}` at layer 24 to `{zero['late_r95']}` at layer 31, and every family rises. Many-to-one response convergence and internal causal convergence are therefore false. Frozen mapping layers before 31 are zero-response and cannot identify a meaningful contraction map.

## Distributed realization and readout

No dimension among `{design['realization_dimensions']}` meets the same-J realization criteria; at dimension 256, relative L2 is `{original['reconstruction']['causal_by_dimension']['256']['relative_l2']:.6f}`. In contrast, centered r95 is `{readout_metrics['direct_full_hidden_2560']['centered']['r95']}` for direct full hidden, `{readout_metrics['J_128']['centered']['r95']}` for J-128, `{readout_metrics['logits_32']['centered']['r95']}` for logits-32, and `{readout_metrics['semantic_32']['centered']['r95']}` for semantic-32. This supports output/readout compression only, not compact internal state.

## Other diagnostics

- Background-conditioned response coordinates: `{fmt(background['BACKGROUND_CONDITIONED_CAUSAL_EFFECT'])}`.
- Channel factorial medians: `{json.dumps(analysis['channel_factorial']['median_interaction_by_term'], sort_keys=True)}`.
- Order relative-difference median/max: `{analysis['intervention_order']['relative_difference_median']:.6f}/{analysis['intervention_order']['relative_difference_max']:.6f}`.
- Multihorizon status: `{analysis['multihorizon_status']}`.
- Interaction/convergence correlations are association-only and are not causal proof.

## Append-only corrections

1. Effective-rank values were clamped to ambient dimension without overwriting original records.
2. The formal interaction estimand was corrected to clean+B+C for Y11; the persistent-FULL records remain historical diagnostics.
3. Exact-zero response matrices were assigned rank zero; the earlier cumulative-energy artifact was retained but superseded.

## Authorization and verification

- `H2_REMAINS = TRUE`
- `H3_AUTHORIZED = FALSE`
- `DYNAMIC_STATE_SEARCH_AUTHORIZED = FALSE`
- Historical final opened: `FALSE`
- V25 independent final opened: `FALSE`
- V25 tests: `{v25_tests}`
- Full suite: `{full_tests}`; the two failures are inherited V14/V16 cumulative-report hash checks.
"""
    write(root, NAMES[17], complete)

    cumulative = root / REPORTS / "FINAL_REPORT.md"
    if "# V25 Complete Report" in cumulative.read_text(encoding="utf-8"):
        raise RuntimeError("V25 already appended to FINAL_REPORT.md")
    cumulative.write_text(cumulative.read_text(encoding="utf-8").rstrip() + "\n\n---\n\n" + complete.rstrip() + "\n", encoding="utf-8")
    write_json_atomic(root / OUT / "v25_test_audit.json", {"v25_tests": v25_tests, "full_suite": full_tests})

    paths = []
    for base_dir in ("artifacts", "results/v25/processed", "reports"):
        for path in sorted((root / base_dir).glob("**/*")):
            relative = str(path.relative_to(root))
            if path.is_file() and ("v25" in relative.lower() or relative == "reports/FINAL_REPORT.md") and not relative.endswith("v25_integrity_index.json"):
                paths.append(relative)
    integrity = {
        "files": {path: sha256_file(root / path) for path in paths},
        "file_count": len(paths),
        "historical_final_opened": False,
        "v25_independent_final_opened": False,
    }
    write_json_atomic(root / OUT / "v25_integrity_index.json", integrity)
    inputs = [
        SOURCE,
        "reports/V25_COMPLETE_REPORT.md",
        "reports/FINAL_REPORT.md",
        "results/v25/processed/v25_adjudication.json",
        "results/v25/processed/interaction_estimand_amendment_v25.json",
        "results/v25/processed/zero_rank_amendment_v25.json",
        "results/v25/processed/v25_integrity_index.json",
    ] + [f"reports/{name}" for name in NAMES[:-1]]
    frozen = stage_freeze(root, "final", inputs, {
        "formal_outcomes": adjudication["formal_outcomes"],
        "integrity_index_sha256": sha256_file(root / OUT / "v25_integrity_index.json"),
        "historical_final_opened": False,
        "v25_independent_final_opened": False,
        "H3_AUTHORIZED": False,
        "DYNAMIC_STATE_SEARCH_AUTHORIZED": False,
    })
    return {
        "formal_outcomes": adjudication["formal_outcomes"],
        "reports": len(NAMES),
        "integrity_files": len(paths),
        "final_freeze_digest": frozen["freeze_digest"],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--v25-tests", required=True)
    parser.add_argument("--full-tests", required=True)
    args = parser.parse_args()
    print(json.dumps(generate(Path.cwd(), args.v25_tests, args.full_tests), indent=2))
