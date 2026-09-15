"""Machine-generated reports for persistent-channel protocol v7/v7.1."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from jclosure.provenance import sha256_file, write_json_atomic


def _f(value: Any, digits: int = 6) -> str:
    return "NA" if value is None else f"{float(value):.{digits}f}"


def _ci(value: dict[str, Any] | None, digits: int = 6) -> str:
    if not value or value.get("estimate") is None:
        return "NA"
    return (
        f"{_f(value['estimate'], digits)} "
        f"[{_f(value['lower'], digits)}, {_f(value['upper'], digits)}]"
    )


def _passport(mode: str, freezes: dict[str, str]) -> str:
    hashes = "\n".join(f"- {name}: `{value}`" for name, value in freezes.items())
    return f"""## Material Passport

- Origin Skill: `academic-research-suite / experiment-agent`
- Mode: `{mode}`
- Date: `2026-09-15`
- Verification Status: `repository-grounded; machine records and 10,000-resample CIs`
- Protocols: `persistent_channel_attribution_protocol_v7`, `persistent_channel_compression_corrective_v7_1`
{hashes}
"""


def _table(headers: list[str], rows: list[list[Any]]) -> str:
    head = "| " + " | ".join(headers) + " |"
    rule = "| " + " | ".join("---" for _ in headers) + " |"
    body = ["| " + " | ".join(str(value) for value in row) + " |" for row in rows]
    return "\n".join((head, rule, *body))


def _effect_rows(effects: dict[str, Any]) -> list[list[str]]:
    rows = []
    for condition in ("clean", "kv_only", "recurrent_only", "full"):
        rows.append(
            [
                condition,
                _ci(effects["next_j_l2"][condition]),
                _ci(effects["direction_cosine_to_full"][condition]),
                _ci(effects["magnitude_ratio_to_full"][condition]),
                _ci(effects["output_js_divergence"][condition], 8),
                _ci(effects["target_log_odds_abs_change"][condition]),
            ]
        )
    return rows


def _figures(
    root: Path,
    schema: dict[str, Any],
    attribution: dict[str, Any],
    localization: dict[str, Any],
    compression: dict[str, Any],
) -> list[Path]:
    figures = root / "results/v7/figures"
    figures.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []

    components = sorted(schema["component_bytes"])
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.bar(components, [schema["component_bytes"][key] / 2**20 for key in components])
    ax.set_ylabel("cache memory (MiB)")
    ax.set_title("Qwen3.5 hybrid-cache schema")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    path = figures / "cache_components_v7.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    outputs.append(path)

    pooled = attribution["effects"]["pooled"]
    labels = ["KV only", "REC+conv", "full"]
    keys = ["kv_only", "recurrent_only", "full"]
    values = np.asarray(
        [pooled["direction_cosine_to_full"][key]["estimate"] for key in keys]
    )
    lower = np.asarray(
        [pooled["direction_cosine_to_full"][key]["lower"] for key in keys]
    )
    upper = np.asarray(
        [pooled["direction_cosine_to_full"][key]["upper"] for key in keys]
    )
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.bar(labels, values, yerr=np.vstack((values - lower, upper - values)), capsize=4)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("next-J delta cosine to full (95% CI)")
    ax.set_title("Persistent-channel causal attribution")
    fig.tight_layout()
    path = figures / "persistent_channel_attribution_v7.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    outputs.append(path)

    loc = localization["summaries"]["attribution_test"]
    mechanism_keys = ["recurrent_matrix_all", "conv_all", "recurrent_both_all"]
    mechanism_labels = ["matrix", "short-conv", "matrix+conv"]
    values = np.asarray(
        [loc[key]["direction_cosine_to_full"]["estimate"] for key in mechanism_keys]
    )
    lower = np.asarray(
        [loc[key]["direction_cosine_to_full"]["lower"] for key in mechanism_keys]
    )
    upper = np.asarray(
        [loc[key]["direction_cosine_to_full"]["upper"] for key in mechanism_keys]
    )
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.bar(
        mechanism_labels,
        values,
        yerr=np.vstack((values - lower, upper - values)),
        capsize=4,
    )
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("next-J delta cosine to full (95% CI)")
    ax.set_title("Gated DeltaNet state localization")
    fig.tight_layout()
    path = figures / "recurrent_conv_localization_v7.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    outputs.append(path)

    head_keys = [f"kv_layer_27_head_{head}" for head in range(4)]
    values = np.asarray(
        [loc[key]["direction_cosine_to_full"]["estimate"] for key in head_keys]
    )
    lower = np.asarray(
        [loc[key]["direction_cosine_to_full"]["lower"] for key in head_keys]
    )
    upper = np.asarray(
        [loc[key]["direction_cosine_to_full"]["upper"] for key in head_keys]
    )
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.bar(
        [f"head {head}" for head in range(4)],
        values,
        yerr=np.vstack((values - lower, upper - values)),
        capsize=4,
    )
    ax.set_ylabel("next-J delta cosine to full (95% CI)")
    ax.set_title("Layer-27 KV-head localization")
    fig.tight_layout()
    path = figures / "kv_head_localization_v7.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    outputs.append(path)

    frame = pd.DataFrame(compression["summaries"])
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.1))
    metrics = [
        ("predictive_gap_closed", "predictive gap closed"),
        ("causal_gap_closed", "causal gap closed"),
        ("conditional_residual_gain", "conditional residual gain"),
    ]
    for method, method_frame in frame.groupby("method", sort=True):
        method_frame = method_frame.sort_values("dimension")
        for ax, (metric, title) in zip(axes, metrics, strict=True):
            ax.plot(
                method_frame["dimension"],
                [value["estimate"] for value in method_frame[metric]],
                marker="o",
                label=method,
            )
            ax.set_xscale("log", base=2)
            ax.set_xlabel("nominal dimension")
            ax.set_title(title)
    axes[0].axhline(0.8, color="black", linestyle="--", linewidth=1)
    axes[1].axhline(0.8, color="black", linestyle="--", linewidth=1)
    axes[2].axhline(0.02, color="black", linestyle="--", linewidth=1)
    axes[0].set_ylabel("fraction")
    axes[2].legend(fontsize=7)
    fig.tight_layout()
    path = figures / "architecture_compression_v7.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    outputs.append(path)
    return outputs


def build(root: Path) -> dict[str, Any]:
    processed = root / "results/v7/processed"
    schema = json.loads((processed / "cache_schema_v7.json").read_text())
    restore = json.loads((processed / "cache_restore_v7.json").read_text())
    attribution = json.loads(
        (processed / "persistent_channel_attribution_v7.json").read_text()
    )
    localization = json.loads((processed / "channel_localization_v7.json").read_text())
    ceiling = json.loads(
        (processed / "raw_arch_channel_ceiling_v7_corrected.json").read_text()
    )
    compression = json.loads(
        (processed / "arch_compression_v7_corrected.json").read_text()
    )
    freezes = {
        "attribution freeze": attribution["source_freeze_digest"],
        "localization analysis freeze": localization["source_freeze_digest"],
        "corrective compression freeze": compression["source_freeze_digest"],
        "captured channel tensor SHA-256": ceiling["source_capture_artifact_sha256"],
    }
    passport = _passport("causal attribution and corrective compression", freezes)
    figures = _figures(root, schema, attribution, localization, compression)
    pooled = attribution["effects"]["pooled"]
    effect_table = _table(
        [
            "condition",
            "next-J L2",
            "direction cosine",
            "magnitude ratio",
            "output JS",
            "abs target log-odds delta",
        ],
        _effect_rows(pooled),
    )
    family_rows = []
    for family, count in attribution["family_counts"].items():
        effect = attribution["effects"][family]
        family_rows.append(
            [
                family,
                count,
                _ci(effect["direction_cosine_to_full"]["kv_only"]),
                _ci(effect["direction_cosine_to_full"]["recurrent_only"]),
                _ci(effect["interaction_ratio"]),
            ]
        )
    family_table = _table(
        ["family", "n", "KV cosine", "REC cosine", "interaction ratio"],
        family_rows,
    )
    cache_rows = [
        [
            component,
            schema["component_counts"][component],
            schema["component_bytes"][component],
            _f(schema["component_bytes"][component] / 2**20, 3),
        ]
        for component in sorted(schema["component_bytes"])
    ]
    cache_table = _table(["component", "tensor count", "bytes", "MiB"], cache_rows)

    attribution_report = f"""# Persistent Channel Attribution v7

{passport}

## Outcome

The v6 same-measured-J causal effect is persistently represented: exact cache
restoration passed {restore["passed"]}/{restore["attempted"]} trials with maximum
logit/hidden/J errors {_f(restore["maximum_logits_error"])}/
{_f(restore["maximum_hidden_error"])}/{_f(restore["maximum_j_error"])}. The frozen
classification is **{attribution["classification"]}**. Both KV and recurrent
channels are causally active; recurrent+conv dominates next-J direction, while
the pooled interaction ratio {_ci(pooled["interaction_ratio"])} exceeds the
frozen additive limit {_f(attribution["channel_gate"]["maximum_additive_interaction_ratio"], 2)}.
The result is therefore neither a pure-KV nor a pure-recurrent state.

## Cache schema

At sequence length {schema["sequence_length"]}, the cache occupies
{_f(schema["total_bytes"] / 2**20, 3)} MiB. Full-attention layers are
`{[index for index, kind in enumerate(schema["layer_types"]) if kind == "full_attention"]}`;
the other layers are Gated DeltaNet layers.

{cache_table}

## Paired 2×2 attribution (held-out n={attribution["formal_count"]})

{effect_table}

{family_table}

These are intervention-based effects from chimeric cache states under a shared
teacher-forced token sequence. They localize the persistence path; they do not
show that the cache channel is compact or sufficient.

## Endpoint serialization correction

The first auxiliary endpoint archive wrote the final `full` trajectory under
all condition names. Per-trial scalars and curves above were calculated before
that save and remain valid. The archive is preserved; corrective protocol v7.1
uses the frozen v6 step-1 vectors and a separate freeze.
"""

    swap_report = f"""# KV / Recurrent State Swap v7

{passport}

## Causal design and result

For every one of {attribution["formal_count"]} held-out pairs, clean, KV-only,
REC+conv-only, and full caches consumed identical clean teacher tokens. The
pooled next-J results are:

{effect_table}

Recurrent+conv reproduces cosine
{_ci(pooled["direction_cosine_to_full"]["recurrent_only"])} and magnitude
{_ci(pooled["magnitude_ratio_to_full"]["recurrent_only"])}; KV alone reproduces
cosine {_ci(pooled["direction_cosine_to_full"]["kv_only"])} and magnitude
{_ci(pooled["magnitude_ratio_to_full"]["kv_only"])}. The non-additive vector
interaction is {_ci(pooled["interaction_ratio"])}. Output-JS interaction is
{_ci(pooled["output_js_interaction"], 8)}.

## Family heterogeneity

{family_table}

The recurrence result is directionally stable across all five families, but
family n ranges from {min(attribution["family_counts"].values())} to
{max(attribution["family_counts"].values())}; family comparisons are exploratory.
No task-decision flips occur often enough here to localize a family-specific
behavioral mechanism. The causal conclusion is strongest for next-J writes and
logit-distribution changes.
"""

    loc = localization["summaries"]["attribution_test"]
    rec_rows = []
    for key in ("recurrent_matrix_all", "conv_all", "recurrent_both_all"):
        rec_rows.append(
            [
                key,
                _ci(loc[key]["direction_cosine_to_full"]),
                _ci(loc[key]["magnitude_ratio_to_full"]),
                _ci(loc[key]["output_js_divergence"], 8),
            ]
        )
    for layer in localization["selected_recurrent_layers_ranked"]:
        key = f"rec_layer_{layer}_both"
        rec_rows.append(
            [
                key,
                _ci(loc[key]["direction_cosine_to_full"]),
                _ci(loc[key]["magnitude_ratio_to_full"]),
                _ci(loc[key]["output_js_divergence"], 8),
            ]
        )
    rec_table = _table(
        ["mechanism", "direction cosine", "magnitude ratio", "output JS"], rec_rows
    )
    recurrent_report = f"""# Recurrent State Compression v7

{passport}

## Localization before compression

{rec_table}

Short-conv alone is the main recurrent-family carrier by direction and
magnitude, but matrix+conv jointly comes closest to the full mixed effect.
Layer ranking, selected only on the fit half, is
`{localization["selected_recurrent_layers_ranked"]}`; layer 30 is the largest
single recurrent layer on held-out data. Because the global classification is
mixed-interacting, compression was run on the joint causal support rather than
pretending REC is independent of KV.

## Compression verdict

The effective sample rank is {compression["effective_sample_rank"]}.
No candidate passed all frozen predictive, causal, direction, and conditional
sufficiency criteria (`nominal passing={compression["nominal_passing_count"]}`).
Consequently there is no validated recurrent compact state and no justified
minimum REC dimension. Nominal 64--512 settings are rank-limited to 32 and do
not add independently observed degrees of freedom.
"""

    kv_rows = []
    for layer in (27, 31):
        key = f"kv_layer_{layer}"
        kv_rows.append(
            [
                key,
                _ci(loc[key]["direction_cosine_to_full"]),
                _ci(loc[key]["magnitude_ratio_to_full"]),
                _ci(loc[key]["output_js_divergence"], 8),
            ]
        )
    for head in range(4):
        key = f"kv_layer_27_head_{head}"
        kv_rows.append(
            [
                key,
                _ci(loc[key]["direction_cosine_to_full"]),
                _ci(loc[key]["magnitude_ratio_to_full"]),
                _ci(loc[key]["output_js_divergence"], 8),
            ]
        )
    kv_table = _table(
        ["mechanism", "direction cosine", "magnitude ratio", "output JS"], kv_rows
    )
    kv_report = f"""# KV State Compression v7

{passport}

## Localization

{kv_table}

Layer {localization["selected_kv_layer"]} is the only attention layer before the
main measured layer and carries the measured next-J effect. Layer 31 is after
that measurement, so its next-J direction is structurally zero while it can
still affect output logits. Fit-half head ranking is
`{localization["selected_kv_heads_ranked"]}`; head 3 is largest on held-out data.
The 1/4/16/64-token windows are identical because the originating v6
intervention modified only the final cached token. That is a scope result, not
evidence that older KV positions are generally irrelevant.

## Compression verdict

KV was not compressed as an isolated sufficient state because its effect is
weaker and the frozen attribution is mixed-interacting. Joint mixed-channel
compression also failed conditional sufficiency, so no minimum KV dimension or
fixed memory-slot count is warranted.
"""

    ceiling_table = _table(
        ["model", "delta cosine", "delta RMSE"],
        [
            [
                "J only",
                _ci(ceiling["j_only_delta_cosine"]),
                _ci(ceiling["j_only_delta_rmse"]),
            ],
            [
                "J + mixed raw channel",
                _ci(ceiling["full_channel_delta_cosine"]),
                _ci(ceiling["full_channel_delta_rmse"]),
            ],
            [
                "paired gain/improvement",
                _ci(ceiling["delta_cosine_gain"]),
                _ci(ceiling["delta_rmse_improvement"]),
            ],
        ],
    )
    compact_rows = []
    for value in compression["summaries"]:
        if value["dimension"] in (16, 32):
            compact_rows.append(
                [
                    value["method"],
                    value["dimension"],
                    value["effective_dimension"],
                    _ci(value["predictive_gap_closed"]),
                    _ci(value["causal_gap_closed"]),
                    _ci(value["conditional_residual_gain"]),
                    _ci(value["causal_direction_cosine"]),
                ]
            )
    compact_table = _table(
        [
            "method",
            "nominal D",
            "effective D",
            "predictive gap",
            "causal gap",
            "residual gain",
            "direction cosine",
        ],
        compact_rows,
    )
    best = max(
        compression["summaries"],
        key=lambda value: value["predictive_gap_closed"]["estimate"],
    )
    compact_report = f"""# Architecture-Aligned Compact State v7

{passport}

## Adjudication

The persistent effect is real and mixed-interacting, but no candidate
sufficient persistent state was found. The corrected raw channel gives a
statistically positive yet practically small conditional predictive increment.
Compression then preserves a substantial causal direction but leaves about
half of the raw-channel effect recoverable from the omitted channel state.

## Raw-channel ceiling

{ceiling_table}

Authorization: `{ceiling["raw_channel_ceiling_authorized"]}`; fit/test counts
{ceiling["train_count"]}/{ceiling["test_count"]}; effective rank
{ceiling["effective_raw_rank"]}. The predictive endpoint is held-out
intervention-delta prediction. The separate direct cache-swap attribution is
the causal ceiling; these evidence types are not conflated.

## Compression Pareto

{compact_table}

The largest observed gap-closed estimate is
{_ci(best["predictive_gap_closed"])} for `{best["method"]}` at nominal
{best["dimension"]}D/effective {best["effective_dimension"]}D. Its causal gap is
{_ci(best["causal_gap_closed"])}, direction cosine
{_ci(best["causal_direction_cosine"])}, and conditional residual gain
{_ci(best["conditional_residual_gain"])}. Passing candidates:
{compression["independently_identified_passing_count"]} independently
identified ({compression["nominal_passing_count"]} nominal).

## Twelve required answers

1. **Can persistent state explain v6?** Yes. Both cache branches reproduce nonzero next-J/logit effects.
2. **Where is it?** Mixed KV and recurrent+conv with material interaction, not a purely transient local effect.
3. **Matrix or conv?** Short-conv is stronger alone; matrix+conv is strongest jointly.
4. **Which KV support?** Layer 27, especially head 3, for next measured-J; layer 31 is output-only relative to this endpoint. Token-window equality follows final-token intervention scope.
5. **Raw improvement over J-only?** Cosine gain {_ci(ceiling["delta_cosine_gain"])}; RMSE improvement {_ci(ceiling["delta_rmse_improvement"])}.
6. **Compressible?** Partially, but not to the frozen sufficiency criteria.
7. **Minimum effective dimension?** None established; even full empirical rank 32 fails.
8. **Predictive gap closed?** Best point estimate {_f(best["predictive_gap_closed"]["estimate"])} with CI {_ci(best["predictive_gap_closed"])}.
9. **Causal gap closed?** Best-candidate value {_ci(best["causal_gap_closed"])}; below the frozen lower-bound criterion.
10. **Residual information after compact state?** Yes: conditional residual gain {_ci(best["conditional_residual_gain"])}, far above the maximum 0.02.
11. **Candidate sufficient persistent state?** `{compression["candidate_sufficient_persistent_state"]}`.
12. **Ready for autonomous controller?** `{compression["autonomous_controller_authorized"]}`. The protocol correctly stops before generic recurrent dynamics training.

## Evidence and limitations

- Cache swaps are causal interventions; raw prediction and bottleneck fitting are predictive/associational.
- The split has only 33 fit and 33 test pairs, with family n=3--12.
- Empirical rank 32 makes nominal 64--512 curves non-identifying.
- The raw predictor's positive increment is small despite tight paired CIs.
- Compression is of intervention deltas, not a complete natural-state model.
- The fixed-token continuation has no independent task-semantic decoder;
  task-decision evidence is limited to token and logit changes.
- No result localizes parameters, extracts “true thoughts,” or concerns consciousness.
"""

    v6 = json.loads((root / "results/v6/processed/causal_endpoint_v6.json").read_text())
    final_report = f"""# J-State Closure Final Report

{passport}

## Current adjudication

The strongest warranted conclusion remains **H2 for the tested operational
measured-J state**. v6 established that same current measured-J plus a different
operational remainder causally changes the next within-forward measured-J write.
v7 now localizes the cross-token persistence to a **mixed, interacting KV and
Gated-DeltaNet recurrent/short-conv cache state**, with recurrent+conv dominant.
This rules out the claim that the observed effect is only a nonpersistent local
transient. It does not establish H3: no architecture-aligned compact state
passed conditional sufficiency, and no autonomous controller was trained.

## v7 decisive measurements

{effect_table}

Corrected raw-channel cosine gain is {_ci(ceiling["delta_cosine_gain"])}; the
best compression closes {_ci(best["predictive_gap_closed"])} of the direct
channel gap but leaves {_ci(best["conditional_residual_gain"])} conditional
gain. Candidate sufficient state:
`{compression["candidate_sufficient_persistent_state"]}`.

## Scientific interpretation

- **Causal:** v6 same-J intervention; v7 KV/REC cache swaps and direct compressed-cache reconstructions.
- **Predictive:** held-out raw-channel ceiling and learned component ordering.
- **Practical magnitude:** REC+conv reproduces most next-J direction, but output JS is small and answer flips are rare.
- **Uncertainty:** only {attribution["formal_count"]} held-out pairs; family cells have 3--12 items; compression rank is 32.

## Validity and fallacy scan

1. Predictive gains are not called causal; only state swaps support causal wording.
2. Finite 4096D profiles remain “measured-J,” not complete J-space.
3. Pooled and family-wise attribution are both reported.
4. Fit-half localization is separated from held-out attribution.
5. The endpoint archive defect is preserved and disclosed; corrected analysis has its own freeze.
6. Ridge alpha, split, bootstrap seed, and thresholds were not tuned after outcomes.
7. Nominal dimensions above empirical rank are not counted as independent evidence.
8. Failure to compress does not prove no compact representation exists.
9. Token-window equality is not generalized beyond final-token intervention scope.
10. Interaction prevents calling REC or KV independently sufficient.
11. No consciousness, true-thought, parameter-localization, or autonomous-controller claim is made.

## Next blocker

Collect a substantially larger independent architecture-state bank so 64--512
dimensions are identifiable, then fit channel-specific nonlinear causal
bottlenecks with explicit ordinary next-state, semantic, output, and conditional
residual endpoints. Only a candidate passing all frozen sufficiency gates should
authorize autonomous recurrent dynamics.

## Exact v7 commands

```bash
scripts/run_persistent_channels_v7.sh schema --run-suffix cache-schema
scripts/run_persistent_channels_v7.sh restore --run-suffix restore-full8
CUDA_VISIBLE_DEVICES=0 scripts/run_persistent_channels_v7.sh attribution --run-suffix attribution-full66
scripts/run_persistent_channels_v7.sh analyze --run-suffix attribution-analysis
CUDA_VISIBLE_DEVICES=0 scripts/run_arch_state_v7.sh localize --run-suffix localization-full66
scripts/run_localization_analysis_v7.sh --run-suffix analysis-corrected-r1
scripts/run_arch_compression_v7_corrective.sh freeze --run-suffix endpoint-fix-freeze
scripts/run_arch_compression_v7_corrective.sh ceiling --run-suffix corrected-endpoint-ceiling
CUDA_VISIBLE_DEVICES=0 scripts/run_arch_compression_v7_corrective.sh run --run-suffix corrected-compression-full
scripts/run_arch_compression_v7_corrective.sh analyze --run-suffix corrected-compression-analysis
scripts/build_report_v7.sh --run-suffix final-v7
```

## Provenance continuity

The v6 causal endpoint freeze remains `{v6["source_freeze_digest"]}`. All v1--v6
frozen reports/results remain unmodified; `FINAL_REPORT.md` is the declared
cumulative-report exception.
"""

    reports = {
        "PERSISTENT_CHANNEL_ATTRIBUTION_V7.md": attribution_report,
        "KV_REC_STATE_SWAP_V7.md": swap_report,
        "RECURRENT_STATE_COMPRESSION_V7.md": recurrent_report,
        "KV_STATE_COMPRESSION_V7.md": kv_report,
        "ARCH_ALIGNED_COMPACT_STATE_V7.md": compact_report,
        "FINAL_REPORT.md": final_report,
    }
    for name, content in reports.items():
        (root / "reports" / name).write_text(content, encoding="utf-8")
    manifest = {
        "schema_version": 10,
        "protocol_version": "persistent_channel_compression_corrective_v7_1",
        "source_freeze_digest": compression["source_freeze_digest"],
        "reports": {name: sha256_file(root / "reports" / name) for name in reports},
        "figures": {str(path.relative_to(root)): sha256_file(path) for path in figures},
        "candidate_sufficient_persistent_state": compression[
            "candidate_sufficient_persistent_state"
        ],
        "autonomous_controller_authorized": compression[
            "autonomous_controller_authorized"
        ],
    }
    write_json_atomic(processed / "report_build_v7.json", manifest)
    return manifest


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    print(json.dumps(build(root), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
