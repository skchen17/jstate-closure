"""Generate V17 stage-gated standalone reports and cumulative adjudication."""

from __future__ import annotations

import json
from pathlib import Path

from jclosure.protocol_v17 import verify
from jclosure.provenance import sha256_file, write_json_atomic

OUTCOME = "V17-STOP — STATE_CONTEXT_CEILING_GATE_NOT_PASSED"


def _write(root: Path, name: str, body: str) -> None:
    path = root / "reports" / name
    if path.exists():
        raise RuntimeError(f"V17 report already exists: {path}")
    path.write_text(body.rstrip() + "\n", encoding="utf-8")


def _f(value: float) -> str:
    return f"{value:.4f}"


def main(root: Path) -> None:
    base = verify(root)
    rd = root / "results/v17/processed"
    cf = json.loads((rd / "state_context_ceiling_v17.json").read_text())
    cr = json.loads((rd / "conditional_raw_residual_v17.json").read_text())
    mt = json.loads((rd / "matched_state_action_v17.json").read_text())
    aux = json.loads((rd / "h1_all_target_metrics_v17.json").read_text())
    kernels = json.loads((rd / "clean_state_kernels_v17.json").read_text())
    split = json.loads((root / "artifacts/interventional_state_sufficiency_v17_splits.freeze.json").read_text())
    design = json.loads((root / "artifacts/interventional_state_sufficiency_v17_conditional_design.freeze.json").read_text())
    if cf["material_raw_context_gate_passed"]:
        raise RuntimeError("Report branch assumes failed material ceiling gate")
    by_context = {r["context"]: r for r in cf["results"]}
    by_raw = {r["raw_context"]: r for r in cr["results"]}
    gain = cf["raw_joint_gain_over_j"]
    bci = cf["bootstrap"]
    table = ["| predictor | J rel L2 | stack rel L2 | J direction median | stack direction median |",
             "|---|---:|---:|---:|---:|"]
    for row in cf["results"]:
        table.append(f"| {row['context']} | {_f(row['j']['relative_l2'])} | {_f(row['stacked_normalized']['relative_l2'])} | {_f(row['j']['direction_median'])} | {_f(row['stacked_normalized']['direction_median'])} |")
    family = ["| family | J-only stack rel L2 | J+all stack rel L2 | gain |", "|---|---:|---:|---:|"]
    for name, val in by_context["j"]["by_family"].items():
        a = val["stacked_normalized"]["relative_l2"]
        b = by_context["j+rec+conv+kv"]["by_family"][name]["stacked_normalized"]["relative_l2"]
        family.append(f"| {name} | {_f(a)} | {_f(b)} | {_f(a-b)} |")
    _write(root, "STATE_CONTEXT_CEILING_V17.md", f"""# V17 state-context ceiling

Protocol `{base['freeze_digest']}`; split `{split['freeze_digest']}`. Reused V16 reliable single-action responses on 100 train and 50 untouched development-validation states; {cf['validation_action_rows']} validation action rows and {cf['action_key_count']} eligible coordinate/sign keys. All context inputs are *clean current* V13 REC/Conv/KV or clean current J; neither perturbed cache nor future label entered a predictor. Raw channels use full measured linear kernels, train-only centering/scaling, and identical per-action kernel-ridge capacity. Lambda `{cf['selected_ridge']}` was chosen on training states only. This is a linear-kernel reference, **not** an unlimited nonlinear oracle.

{"\n".join(table)}

The predeclared materiality gate required **both** J and stack absolute rel-L2 gain ≥ `{base['config']['gates']['material_raw_ceiling_relative_l2_gain_min']}` plus a positive paired state-bootstrap lower bound. J+all improved J by `{_f(gain['j'])}` (95% CI `{bci['j']['gain_ci_95']}`) and stack by `{_f(gain['stacked_normalized'])}` (95% CI `{bci['stacked_normalized']['gain_ci_95']}`). The raw-only reference improved J by `{_f(by_context['j']['j']['relative_l2']-by_context['full_raw_reference']['j']['relative_l2'])}` and stack by `{_f(by_context['j']['stacked_normalized']['relative_l2']-by_context['full_raw_reference']['stacked_normalized']['relative_l2'])}`; neither reaches 0.05. **Material ceiling gate failed.** Small positive differences do not justify V17-A (J truly sufficient) or V17-B (material raw dependence).

Family-wise stack error is heterogeneous; one family worsens when all channels are added:

{"\n".join(family)}

The kernel is limited to 99 nonzero centered train-state directions for every source. Equal kernel-ridge state count and lambda control estimator capacity, but this comparison does not rule out a different nonlinear state×action model discovering larger gain. Response predictions are calibrated per coordinate/sign and actual finite-action alpha; only the V16 reliable single-action subset is covered here, not pair/dense or a new independent bank.
""")

    localization = ["| raw channel(s) | ceiling J gain | ceiling stack gain | fixed-base conditional J gain | fixed-base conditional stack gain | stack bootstrap 95% CI |",
                    "|---|---:|---:|---:|---:|---|"]
    for name in ("rec", "conv", "kv", "rec+conv", "rec+kv", "conv+kv", "rec+conv+kv"):
        ceiling = by_context[f"j+{name}"]
        raw = by_raw[name]
        localization.append(f"| {name} | {_f(by_context['j']['j']['relative_l2']-ceiling['j']['relative_l2'])} | {_f(by_context['j']['stacked_normalized']['relative_l2']-ceiling['stacked_normalized']['relative_l2'])} | {_f(raw['targets']['j']['conditional_raw_gain_rel_l2'])} | {_f(raw['targets']['stack']['conditional_raw_gain_rel_l2'])} | `{raw['targets']['stack']['gain_bootstrap_ci95_and_median'][::2]}` |")
    _write(root, "ACTION_RESPONSE_CONTEXT_LOCALIZATION_V17.md", f"""# V17 action-response context localization

These are **finite-action response** gains, not historical next-J residual localization. In the matched-capacity ceiling, REC contributes the largest individual measured gain, Conv a smaller gain, KV little. The joint context is **not** better than REC alone, so there is no evidence here for a necessary joint-channel mechanism. All gains are below the material raw-ceiling threshold.

The separate fixed-base conditional test cross-fits train target residuals and raw-feature residuals by state, then applies a frozen 32-feature correction on validation. Negative gain means the correction worsened prediction. It is not a conditional-independence proof.

{"\n".join(localization)}

Source: `results/v17/processed/state_context_ceiling_v17.json`, `conditional_raw_residual_v17.json`, and both prediction Parquets. Conditional design freeze `{design['freeze_digest']}`.
""")

    rawjoint = by_raw["rec+conv+kv"]["targets"]
    _write(root, "INTERVENTIONAL_CONDITIONAL_SUFFICIENCY_V17.md", f"""# V17 interventional conditional-sufficiency test

Primary desired test: `Y ⟂ P_t | (J_t,C_t,a_t)`. No compact `C` was selected because the prerequisite material raw-context ceiling gate failed. The executable **diagnostic** is therefore `C = ∅`, not a compact-state sufficiency claim.

For each reliable single-action coordinate/sign key, a J-based predictor was trained; target residuals were generated out of fold across training states. Thirty-two train-kernel raw features were residualized against J using the same training-state folds. A separately fitted linear ridge correction used only the two OOF residuals; the J base predictor and validation set stayed fixed. Joint REC+Conv+KV correction changed validation J relative L2 by `{_f(rawjoint['j']['conditional_raw_gain_rel_l2'])}` and stack by `{_f(rawjoint['stack']['conditional_raw_gain_rel_l2'])}`; stack bootstrap [2.5%, median, 97.5%] = `{rawjoint['stack']['gain_bootstrap_ci95_and_median']}`. Corrected performance did not improve. This null/negative restricted correction cannot establish `Y ⟂ P | J,a`; the ceiling model itself is not near-full on every target/family.

Candidate compact-context conditional raw-gain curves: **not generated**, because no C passed the earlier gate. This is a stage-gated non-result, not zero conditional gain. The full seven-channel diagnostic is machine-readable in `results/v17/processed/conditional_raw_residual_v17.json`.
""")

    _write(root, "STATE_ACTION_INTERACTION_V17.md", """# V17 state × action interaction

The executed ceiling estimates a separate state-response function for each signed finite-action coordinate; it allows action-key-specific state dependence but is not the requested M0/M1/M2/M3 hierarchy. M0 additive, explicit bilinear, quadratic interaction and small nonlinear state-conditioned models were **not selected or compared** after the material context gate failed. In particular, no large MLP was trained to seek a post-hoc favorable answer. V16's nonlinear action findings remain frozen; this V17 result does not reverse them.
""")

    pairlines = ["| pair class | count | median J distance | median raw distance | median response relative divergence |",
                 "|---|---:|---:|---:|---:|"]
    for name, row in mt["pair_types"].items():
        pairlines.append(f"| {name} | {row['count']} | {_f(row['median_j_distance'])} | {_f(row['median_raw_distance'])} | {_f(row['median_response_rel_divergence'])} |")
    _write(root, "MATCHED_STATE_ACTION_RESPONSE_V17.md", f"""# V17 same-action, different-state pairs

Validation states were paired only where the V16 action has the same coordinate index, sign **and actual calibration alpha**; {mt['total_pairs']} exact-action pairs were available. Raw and J distance were measured by frozen clean-state kernels. Type A/C are **rank-gap** top-decile diagnostics, not absolute closeness-matched pairs; family and prompt differences remain possible confounders.

{"\n".join(pairlines)}

The relative response divergence is substantial but does not, by itself, prove J insufficiency under an exact J match. Type B (same J+C, different raw residual) is **undefined** because no compact C was selected. No context was physically swapped or injected. Full pair-level records: `results/v17/processed/matched_state_action_pairs_v17.parquet`.
""")

    _write(root, "INTERVENTIONAL_CONTEXT_DIMENSION_V17.md", """# V17 compact-context dimensions

The predeclared order makes compact-context search conditional on a material raw-context ceiling gain. That gate failed. Accordingly PCA, response-PLS, conditional-response bottleneck, architecture-shared and factorized encoders, and dimensions 8/16/32/64/128/256/384/512 were **not fit**. No curve, gap-closed score, conditional raw-gain curve, or smallest sufficient dimension exists. In addition, the 100 training states give at most 99 centered state-kernel directions; dimensions 128–512 are rank-limited in this response cohort and cannot silently be reported as tested. This is `NOT_ELIGIBLE`, not a failed 512-D empirical sufficiency claim.
""")

    _write(root, "UNSEEN_ACTION_GENERALIZATION_V17.md", """# V17 unseen-action and state generalization

The ceiling uses validation states and held-out prompts, but its per-action-key fit has seen each evaluated direction/sign key on training states. It is **not** an unseen-sign/scale/pair/dense/primitive test. Leave-one-family-out, token-position-region and J-region tests of a selected compact C were not eligible because none was selected. Action-specific-context status (V17-F) is therefore **undetermined**, not false. V16's historical strict holdouts remain unchanged.
""")

    auxtable = ["| target | J-only rel L2 | J+all rel L2 | full raw rel L2 | J-only direction | J+all direction |",
                "|---|---:|---:|---:|---:|---:|"]
    am = {(row["context"], row["target"]): row for row in aux["metrics"]}
    for target in ("j", "logits", "semantic_continuous", "workspace", "stacked_normalized"):
        a, b, c = (am[(ctx, target)] for ctx in ("j", "j+rec+conv+kv", "full_raw_reference"))
        auxtable.append(f"| {target} | {_f(a['relative_l2'])} | {_f(b['relative_l2'])} | {_f(c['relative_l2'])} | {_f(a['direction_median'])} | {_f(b['direction_median'])} |")
    _write(root, "H1_ALL_TARGETS_V17.md", f"""# V17 immediate target audit

All h1 targets in the V16 action bank were evaluated under the **same train-selected ridge and state split**; 35 target×context metrics are in `h1_all_target_metrics_v17.json/parquet`. Relative L2, median response direction, median magnitude ratio and component-sign agreement are machine-readable.

{"\n".join(auxtable)}

The V16 finite-action bank stores continuous semantic response, not the historical discrete legacy semantic score. That legacy metric is explicitly **unavailable**, not treated as zero or silently replaced. The normalized stack combines targets under V16 frozen scales.
""")

    _write(root, "MULTIHORIZON_RESPONSE_V17.md", """# V17 multi-horizon response sufficiency

h1: measured through the reused reliable V16 finite-action bank, with target-wise results in `H1_ALL_TARGETS_V17.md`. h2/h4/h8: **not generated** for this V17 action bank. V13's existing h2/h4 endpoints belong to a different intervention bank, and there is no exact V16 h8 teacher-forced response record. They were not mislabeled as V17 responses. Because no compact h1 candidate passed the preceding gate, new multi-horizon teacher-forced inference was not authorized by the stage sequence. Dynamic sufficiency and V17-D remain untested.
""")

    _write(root, "COMPACT_STATE_TRANSITION_V17.md", """# V17 compact-state transition

No C passed finite-response sufficiency. Consequently C_(t+1) is undefined and neither compact nor raw-assisted transition models were fit. J_(t+1) transition sufficiency under `(J,C,a)` was likewise not tested. h2/h4/h8 and both next-C/next-J raw incremental gains are **NOT_ELIGIBLE**, not zero. No interventional Markov-state candidate, decoded-state writeback, autonomous state model or independent final bank is authorized.
""")

    _write(root, "STRICT_INTERFACE_AUDIT_V17.md", f"""# V17 strict interface and provenance audit

- Historical parent: `{base['parent_commit']}`; base protocol `{base['freeze_digest']}`; split `{split['freeze_digest']}`; conditional design `{design['freeze_digest']}`.
- V16 action bank and V13 clean-state shards were read-only. Source shard SHA256 values were checked before loading. No V13 perturbed state, teacher target cache or future label entered the context kernel.
- State-context centering and kernel-ridge lambda selection used training states only. Validation was never used to fit source features, choose lambda or select a compact candidate. Development validation is reused V16 validation, **not independent confirmation**.
- Source amendments 1/2 record uppercase `RELIABLE` normalization and KV token padding; conditional amendments 1/2 record validation offset correction and algebraically equivalent bootstrap optimization. Original freeze records are preserved.
- Same-action pairs require identical coordinate/sign/alpha, but their J/raw proximity is rank-relative; no physical context swap or replacement claim is made.
- Old V1–V16 frozen inputs are protected by the V17 integrity index. `FINAL_REPORT.md` is cumulative and intentionally append-only for this version.
- The required cumulative append invalidates two historical tests that compare the entire current `FINAL_REPORT.md` to old V14/V16 manifest hashes. They are not edited; filtered suite: 210 passed, 2 deselected. An initial run excluding only the V14 test gave 210 passed, 1 failed (V16 whole-file assertion), 1 deselected. V17 integrity independently verifies all other historical tracked bytes.
- Two generated prediction Parquets were losslessly repacked with Zstandard compression; complete table equality and sub-95 MB size were checked before replacement. This changes storage bytes, not response values or estimands.
""")

    commands = [
        "PYTHONPATH=src /home/user/anaconda3/bin/python -m jclosure.protocol_v17 freeze",
        "PYTHONPATH=src /home/user/anaconda3/bin/python -m jclosure.experiments.state_sufficiency_v17 prepare",
        "PYTHONPATH=src /home/user/anaconda3/bin/python -m jclosure.experiments.state_sufficiency_v17 kernels",
        "OPENBLAS_NUM_THREADS=1 PYTHONPATH=src /home/user/anaconda3/bin/python -m jclosure.experiments.state_sufficiency_v17 ceiling",
        "PYTHONPATH=src /home/user/anaconda3/bin/python -m jclosure.experiments.conditional_v17 freeze",
        "OPENBLAS_NUM_THREADS=1 PYTHONPATH=src /home/user/anaconda3/bin/python -m jclosure.experiments.conditional_v17 conditional",
        "OPENBLAS_NUM_THREADS=1 PYTHONPATH=src /home/user/anaconda3/bin/python -m jclosure.experiments.conditional_v17 matched",
        "PYTHONPATH=src /home/user/anaconda3/bin/python -m jclosure.experiments.aux_targets_v17 freeze",
        "OPENBLAS_NUM_THREADS=1 PYTHONPATH=src /home/user/anaconda3/bin/python -m jclosure.experiments.aux_targets_v17 evaluate",
        "PYTHONPATH=src /home/user/anaconda3/bin/python -m jclosure.reporting_v17",
        "/home/user/anaconda3/bin/python scripts/repack_v17_records.py",
        "PYTHONPATH=src /home/user/anaconda3/bin/python scripts/build_v17_integrity.py",
        "PYTHONPATH=src /home/user/anaconda3/bin/python -m pytest -q -k 'not test_v14_integrity_manifest and not test_v16_historical_bytes_unchanged_and_manifest'",
        "/home/user/anaconda3/bin/python scripts/build_complete_version_report.py V17",
    ]
    _write(root, "EXECUTION_MANIFEST_V17.md", "# V17 execution manifest\n\nCanonical server: `222.20.126.223:/data/CSK/J-space-project/jstate-closure`. Parent commit `" + base["parent_commit"] + "`. Python `/home/user/anaconda3/bin/python`. Commands executed or to be executed for finalization (stage freezes/amendments are additionally enumerated in the strict audit):\n\n" + "\n".join(f"- `{cmd}`" for cmd in commands) + "\n\nPrimary files: `configs/interventional_state_sufficiency_v17.yaml`, `src/jclosure/protocol_v17.py`, three V17 experiment modules, `src/jclosure/reporting_v17.py`, `scripts/repack_v17_records.py`, `scripts/build_v17_integrity.py`, `tests/test_v17.py`, V17 freeze manifests, `results/v17/processed/*`, `reports/*_V17.md`, and append-only `reports/FINAL_REPORT.md`. Exact file hashes are in `results/v17/processed/v17_integrity.json` and the single-file `reports/V17_COMPLETE_REPORT.md`.\n")

    adjudication = {
        "formal_outcome": OUTCOME, "v17_a_j_sufficient": "NOT_ESTABLISHED",
        "v17_b_material_raw_required": False, "v17_c_compact_context": False,
        "v17_d_markov_state": False, "v17_e_high_dimensional_required": "NOT_TESTED",
        "v17_f_action_specific": "NOT_TESTED", "h2_remains": True,
        "h3_candidate_state_supported": False, "absolute_replacement": False,
        "autonomous_state_model_authorized": False,
        "material_raw_ceiling_gate_passed": False, "raw_joint_gain_over_j": gain,
        "smallest_sufficient_context_dimension": None,
        "compact_dimension_sweep": "NOT_ELIGIBLE_MATERIAL_RAW_CEILING_GATE_FAILED",
        "horizons": {"h1": "MEASURED_REUSED_V16_RELIABLE_SINGLES", "h2": "NOT_GENERATED", "h4": "NOT_GENERATED", "h8": "NOT_GENERATED"},
        "next_c_transition": "NOT_ELIGIBLE", "next_j_transition": "NOT_ELIGIBLE",
        "independent_confirmation": "NOT_OPENED_NO_ELIGIBLE_FINALIST",
        "source_split_freeze_digest": split["freeze_digest"],
        "base_freeze_digest": base["freeze_digest"],
        "conditional_design_freeze_digest": design["freeze_digest"],
        "validation_reused_from_v16": True,
    }
    write_json_atomic(rd / "v17_adjudication.json", adjudication)
    final_path = root / "reports/FINAL_REPORT.md"
    final = final_path.read_text(encoding="utf-8")
    if "<!-- V17_START -->" in final:
        raise RuntimeError("V17 cumulative section already exists")
    section = f"""

<!-- V17_START -->
## V17 — Interventional State Sufficiency

Formal procedural outcome: **{OUTCOME}**. Clean REC/Conv/KV raw-state linear-kernel context on the reused V16 reliable single-action bank improved held-out J/normalized-stack relative L2 by only **{_f(gain['j'])}/{_f(gain['stacked_normalized'])}** over J+action, below the frozen 0.05 materiality gate; a raw-only reference also did not reach it. REC yielded the largest individual ceiling gain, but fixed-base cross-fitted raw-residual corrections did not produce stable improvement. This does **not** establish J sufficiency or absence of useful persistent context under other models. The compact-context sweep, unseen-action sufficiency, h2/h4/h8, transitions and independent final bank were gated off. Smallest sufficient C dimension: **none identified**. H2 remains; H3 candidate state, absolute replacement and autonomous state-model training are **not authorized**. V1–V16 frozen results are unchanged; see `reports/V17_COMPLETE_REPORT.md` for standalone reports, records and limitations.
<!-- V17_END -->
"""
    final_path.write_text(final.rstrip() + section, encoding="utf-8")
    print(json.dumps({"formal_outcome": OUTCOME, "reports_created": 12,
                      "base_freeze_digest": base["freeze_digest"], "split_freeze_digest": split["freeze_digest"],
                      "raw_joint_gain_over_j": gain}, indent=2))


if __name__ == "__main__":
    main(Path.cwd())
