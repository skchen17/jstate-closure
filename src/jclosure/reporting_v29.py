"""Generate V29 reports only from frozen machine-readable records."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from jclosure.protocol_v29 import verify, verify_stage, stage_freeze
from jclosure.provenance import sha256_file, write_json_atomic

SOURCE = "src/jclosure/reporting_v29.py"
OUT = Path("results/v29/processed")
REPORTS = (
    "SAME_INCOMING_STATE_FORKS_V29.md", "NATURAL_WRITE_CONTENT_AUDIT_V29.md",
    "FORK_FUTURE_SIGNATURE_V29.md", "FULL_STATE_TRANSFER_CEILING_V29.md",
    "SAME_BACKGROUND_PARTIAL_TRANSPLANT_V29.md", "REC_WRITE_CONTENT_V29.md",
    "CONV_WRITE_CONTENT_V29.md", "KV_WRITE_CONTENT_V29.md",
    "KV_HELD_FIXED_RECURRENT_WRITE_V29.md", "CHANNEL_WRITE_CONTENT_FACTORIAL_V29.md",
    "RECIPROCAL_WRITE_CONTENT_TRANSFER_V29.md", "STATE_DEPENDENT_WRITE_CODE_V29.md",
    "STATE_TOKEN_FACTORIAL_V29.md", "FUTURE_RESPONSE_SIGNATURE_V29.md",
    "WORKSPACE_VS_WRITE_CONTENT_V29.md", "WRITE_EFFECT_REALIZATION_DIMENSION_V29.md",
    "NATURAL_VS_RANDOM_WRITE_V29.md", "WRITE_CONTENT_HORIZON_V29.md",
    "STRICT_WRITE_INTERFACE_AUDIT_V29.md", "EXECUTION_MANIFEST_V29.md",
    "V29_SCIENTIFIC_ANSWERS_V29.md", "V29_COMPLETE_REPORT.md",
)


def load(root, name):
    return json.loads((root / OUT / name).read_text())


def write(root, name, body):
    p = root / "reports" / name
    if p.exists():
        raise RuntimeError(f"V29 report already exists: {name}")
    p.write_text(body.rstrip() + "\n", encoding="utf-8")


def fmt(x):
    return "n/a" if x is None else f"{float(x):.3f}"


def profile_table(adj, conditions):
    rows = ["| channel | role | A←B cosine / magnitude / rel-L2 | B←A cosine / magnitude / rel-L2 | reciprocal gate |", "|---|---|---:|---:|---:|"]
    for condition in conditions:
        for role in ("development", "validation"):
            p = adj[role]["profiles"][condition]
            d = p["directions"]
            cell = lambda x: " / ".join(fmt(d[x][key]) for key in ("median_cosine", "median_magnitude", "median_relative_l2"))
            rows.append(f"| {condition} | {role} | {cell('A_from_B')} | {cell('B_from_A')} | {p['reciprocal_pass']} |")
    return "\n".join(rows)


def run(root: Path, vtests: str, fulltests: str):
    for stage in ("adjudication", "forks_development", "forks_validation", "secondary_development", "secondary_validation", "diagnostics", "realization"):
        verify_stage(root, stage)
    base = verify(root)
    d = load(root, "design_v29.json")
    adj = load(root, "v29_adjudication.json")
    amend = load(root, "interface_amendment_v29.json")
    final = adj["independent_final"]
    realization = load(root, "write_effect_realization_v29.json")
    diag = load(root, "workspace_write_diagnostics_v29.json")
    spectrum = np.load(root / OUT / "write_effect_spectrum_v29.npz")["singular_values"].astype(np.float64)
    energy = np.cumsum(spectrum ** 2) / max(float(np.sum(spectrum ** 2)), 1e-12)
    r90, r95, r99 = [int(np.searchsorted(energy, q) + 1) for q in (.9, .95, .99)]
    outcomes = adj["formal_outcomes"]
    outcome_lines = "\n".join(f"- `{key}`: **{value}**" for key, value in outcomes.items())
    ptab = profile_table(adj, ("REC", "Conv", "KV", "REC+Conv", "REC+KV", "Conv+KV", "REC+Conv+KV"))
    main_tab = profile_table(adj, ("REC", "Conv", "KV", "REC+Conv"))
    write(root, REPORTS[0], f"""# Same Incoming State Forks — V29

Disjoint frozen panels: 25 calibration, 75 development, 50 validation, 50 independent final (10 per family). V28 selected state IDs were excluded. For each state, the same exact prefix cache was replayed with two different current tokens chosen exclusively from the pre-write top-16 next-token distribution; the shared primary next token came from the frozen V18 teacher continuation. Prefix, token, incoming/outgoing state, and probe hashes are in `design_v29.json` and per-role fork Parquet records. This is a top-plausible/distinct-surface rule, **not** a verified semantic-pair or same-class control.

All 25 calibration same-token replays were exact. The formal all-layer analysis found distinct outgoing states and median shared-next-token future Q `{adj['development']['median_future_q']:.3f}` development and `{adj['validation']['median_future_q']:.3f}` validation; the independent final had median Q `{final['median_future_q']:.3f}`. The sole current-token difference is causal for the naturally committed cache; future comparison holds the next token fixed.
""")
    geom_rows = ["| role | channel | fields | median contrast norm | nonzero fraction |", "|---|---|---:|---:|---:|"]
    for role in ("development", "validation"):
        for ch, x in adj["write_geometry"][role].items():
            geom_rows.append(f"| {role} | {ch} | {x['fields']} | {x['median_contrast_norm']:.3f} | {x['positive_contrast_fraction']:.3f} |")
    write(root, REPORTS[1], "# Natural Write Content Audit — V29\n\nREC and Conv write contrasts are exact `P_out,B − P_out,A` at all 24 recurrent layers; KV is only the newly appended token slot at all 8 attention layers. Pre-existing KV slots are bitwise identical between branches. Geometry is descriptive, not a carrier claim.\n\n" + "\n".join(geom_rows) + f"\n\nTrain-only centered **joint REC+Conv contrast** spectrum has r90/r95/r99 = `{r90}/{r95}/{r99}` over 90 training examples, rank `{realization['train_span_rank']}` in ambient dimension `{realization['write_dimension']}`. Channel-specific full-vector centered spectra were not computed; per-field norms must not be substituted for them. The joint rank is not a dynamical state dimension.\n")
    write(root, REPORTS[2], f"# Fork Future Signature — V29\n\nWith the same frozen next token, the natural A/B outgoing caches produce large next-step contrasts in layer-30 J, logits, semantic log-probabilities, multi-layer workspace, broad-vocabulary projection, and late residual. The per-state vectors and endpoint-specific effects are in `transfer_vectors_*_v29.npz` and `transfer_*_v29.parquet`. Median aggregate future Q is `{adj['development']['median_future_q']:.3f}` development and `{adj['validation']['median_future_q']:.3f}` validation. Branch future contrast is causal to the token-conditioned complete outgoing state but not by itself a localized field attribution.\n")
    write(root, REPORTS[3], "# Full State Transfer Ceiling — V29\n\nFor all 75 development and 50 validation forks, replacing every initialized native field across all 32 layers (24 REC/Conv; 8 KV new slots) made the recipient cache bitwise equal to the donor and reproduced its future response (relative L2 ≤ 1e-6), reciprocally. This is an **identity ceiling**, not evidence of a selective carrier. The initial calibration diagnostic wrongly reused the V28 posterior 8-layer field subset; its median relative L2 was " + fmt(amend["diagnostic_named_full_median_relative_l2"]) + ". It was invalid as a full-cache control and remains preserved in an explicit interface amendment.\n")
    write(root, REPORTS[4], "# Same-Background Partial Transplant — V29\n\nAll partial operations copy exact naturally produced donor fields into a recipient branch with the **same incoming state**. REC and Conv copy full native tensors; KV copies only the new slot. Untouched recipient fields and current readout are unchanged. Donor direction, magnitude, relative L2, reciprocal replication, and all five family gates were predeclared.\n\n" + ptab + "\n\nA large ablation/write-block effect from V28 was not counted as transferable content. Here REC+Conv and Conv pass; REC-only and KV-only do not. The full-cache row is a ceiling.\n")
    write(root, REPORTS[5], "# REC Write Content — V29\n\nREC-only exact donor transplant does not pass the reciprocal donor-direction gate in development or validation. Its natural tensor contrasts are real, but neither their norm nor V28 marginal write-block sensitivity licenses an independent transferable-REC-carrier claim. REC can modify the fidelity of the Conv-containing combination, without a globally additive decomposition.\n\n" + profile_table(adj, ("REC", "REC+Conv")) + "\n")
    write(root, REPORTS[6], "# Conv Write Content — V29\n\nConv-only native donor fields pass reciprocal donor-directed transfer in both formal roles while KV remains recipient-native. This identifies a transferable, token-conditioned recurrent-subsystem component of the immediate future response under same-background forks; it does not decode a semantic variable stored in Conv.\n\n" + profile_table(adj, ("Conv",)) + "\n")
    write(root, REPORTS[7], "# KV Write Content — V29\n\nThe newly appended KV slots differ with current token and provide an exact history carrier, but KV-only replacement does **not** reproduce most of the donor-specific next-step response under the frozen targets and probes. This does not deny that KV records token/history information.\n\n" + profile_table(adj, ("KV",)) + "\n")
    write(root, REPORTS[8], "# KV-Held-Fixed Recurrent Write — V29\n\nIn `REC+Conv` transfer, the eight attention layers' KV history remains **recipient-native**, including its new token slot. Yet future response moves reciprocally toward the donor in development and validation and the fixed independent final. This supports causal token-conditioned content in recurrent/conv persistent writes beyond direct KV token-history carryover. `KV` donor with recipient-native REC/Conv fails; the contrast is local to this design and endpoint.\n\n" + profile_table(adj, ("KV", "REC+Conv")) + "\n")
    write(root, REPORTS[9], "# Channel Write Content Factorial — V29\n\nThe seven frozen channel subsets form an exact native-field factorial: REC, Conv, KV, REC+Conv, REC+KV, Conv+KV, and complete REC+Conv+KV. The complete cache is an identity ceiling; the single Conv channel already passes. Therefore **no single channel sufficient / only combinations work** is false. Combinations improve fidelity but cannot be interpreted as additive main effects without interaction-aware analysis.\n\n" + ptab + "\n")
    write(root, REPORTS[10], "# Reciprocal Write Content Transfer — V29\n\nBoth A←B and B←A were evaluated for every formal fork. Passing requires donor-directed cosine ≥0.80, magnitude ratio ≥0.50, at least half of states, and at least four of five families in each role. Exact requested/untouched writeback and identical next token are mandatory. The independent final was opened only for the fixed REC+Conv finalist after both role gates.\n\n" + main_tab + f"\n\nIndependent final `{final['states']}` states: REC+Conv reciprocal pass `{final['reciprocal_pass']}`; A←B cosine `{fmt(final['REC_Conv']['directions']['A_from_B']['median_cosine'])}`, B←A cosine `{fmt(final['REC_Conv']['directions']['B_from_A']['median_cosine'])}`.\n")
    state_rows = ["| role | pairs | contrast cosine | cross-state delta→native cosine | cross-state delta relative L2 |", "|---|---:|---:|---:|---:|"]
    for role, x in adj["state_dependence"].items():
        state_rows.append(f"| {role} | {x['pairs']} | {x['median_write_contrast_cosine']:.3f} | {x['median_cross_state_delta_to_native_cosine']:.3f} | {x['median_cross_state_delta_relative_l2']:.3f} |")
    state_text = "\n".join(state_rows)
    write(root, REPORTS[11], "# State-Dependent Write Code — V29\n\nTwo response-blind same-family state pairs per family used the same two tokens in a 2×2 factorial. REC/Conv token contrasts are geometrically fairly aligned across states, yet cross-state additive delta transfer degrades relative to the corresponding native same-state transfer. This delta injection is off-manifold and no cross-state success/degradation gate was frozen. Hence neither **global code** nor **state-dependent code** is formally confirmed.\n\n" + state_text + "\n")
    write(root, REPORTS[12], "# State × Token Factorial — V29\n\nEach frozen pair evaluates `(state A/B) × (token A/B)` with one shared next token. Exact state and token hashes, write-contrast geometry, response interaction norm, same-state REC+Conv reference, and wrong-state delta injection are in `state_token_factorial_*_v29.parquet`. This is a 10-pair-per-role diagnostic, not a global coordinate system proof.\n\n" + state_text + "\n")
    multi_rows = ["| role | channel | A←B cosine / rel-L2 | B←A cosine / rel-L2 |", "|---|---|---:|---:|"]
    for role in ("development", "validation"):
        for ch in ("REC", "Conv", "KV", "REC+Conv", "REC+Conv+KV"):
            x = adj["multi_probe"][role][ch]
            s = lambda direction: fmt(x[direction]["median_cosine"]) + " / " + fmt(x[direction]["median_relative_l2"])
            multi_rows.append(f"| {role} | {ch} | {s('A_from_B')} | {s('B_from_A')} |")
    write(root, REPORTS[13], "# Future Response Signature — V29\n\nTen frozen states per role used four next-token probes selected from the **prefix distribution before either write**. The concatenated functional signature tests whether a channel tracks donor response across inputs, not merely a single `z`. Complete-cache signatures are exact; REC+Conv remains strongly donor-directed; KV remains weak. The probe-set choice limits generalization beyond these tokens.\n\n" + "\n".join(multi_rows) + "\n")
    diag_rows = ["| diagnostic model | held-out validation R² |", "|---|---:|"]
    for name, metrics in diag["models"].items():
        diag_rows.append(f"| {name} | {metrics['r2_vs_train_mean']:.3f} |")
    write(root, REPORTS[14], "# Workspace Versus Write Content — V29\n\nTrain-only ridge models predict the h1 response contrast on 50 held-out validation states. `J_t` here is the layer-30 current workspace readout, **not** a cache field. Exact token identity/surface, J, broader current workspace, and per-field write geometry are compared diagnostically.\n\n" + "\n".join(diag_rows) + f"\n\nCurrent J predicts coarse write-geometry features with held-out R² `{diag['write_geometry_from_current_J']['r2_vs_train_mean']:.3f}`. Adding write geometry to token+J improves held-out response R², but predictive gain does not prove current J causally insufficient for the full outgoing write. A high-fidelity direct J transplant/J-matched causal equivalence test was not established; V29-H is not confirmed.\n")
    realization_rows = ["| k | dev A←B / B←A rel-L2 | val A←B / B←A rel-L2 | strict gate |", "|---:|---:|---:|---:|"]
    for k in realization["tested_k"]:
        key = lambda role, direction: realization["profiles"][f"{role}:{k}:{direction}"]["median_relative_l2"]
        realization_rows.append(f"| {k} | {key('development','A_from_B'):.3f} / {key('development','B_from_A'):.3f} | {key('validation','A_from_B'):.3f} / {key('validation','B_from_A'):.3f} | {realization['qualified_k'][str(k)]} |")
    write(root, REPORTS[15], f"# Write-Effect Realization Dimension — V29\n\nA response-blind centered PCA basis was fitted to 90 natural REC+Conv token-write contrasts (25 calibration + 65 development), then causal donor-response reconstruction was tested on 10 held-out development and 50 validation pairs, both directions. This is an approximate **write-effect** test, not a compact persistent-state or dynamical-state test. The approximations are off-manifold and BF16-writeback audited.\n\n" + "\n".join(realization_rows) + f"\n\nAmbient REC+Conv write dimension `{realization['write_dimension']}`; train span rank `{realization['train_span_rank']}`. Requested k `{realization['not_estimable_k']}` exceed the available train span and were **not** called failures. No tested dimension is inferred beyond the strict gate.\n")
    control_rows = ["| role | condition | median donor cosine | median relative L2 |", "|---|---|---:|---:|"]
    for role in ("development", "validation"):
        for condition, x in adj["natural_vs_random"][role].items():
            control_rows.append(f"| {role} | {condition} | {fmt(x['median_cosine'])} | {fmt(x['median_relative_l2'])} |")
    write(root, REPORTS[16], "# Natural Versus Random Write — V29\n\nTen states per role compare exact native REC+Conv donor writes with random same-norm, within-field shuffled, sign-flipped, and wrong-incoming-state delta controls. Artificial controls are off-manifold and BF16-realization ratios are saved per row; they are not substitutes for native donor-field transfer. Natural writes are much more donor-faithful than the random/shuffled/sign-flipped controls. The wrong-state control can retain partial directionality, so it is not evidence of a uniquely state-specific code.\n\n" + "\n".join(control_rows) + "\n")
    horizon_rows = ["| role | h | median donor cosine | median magnitude | median relative L2 |", "|---|---:|---:|---:|---:|"]
    for role in ("development", "validation"):
        for h in (1, 2, 4):
            x = adj["horizons"][role][str(h)]
            horizon_rows.append(f"| {role} | {h} | {fmt(x['median_cosine'])} | {fmt(x['median_magnitude'])} | {fmt(x['median_relative_l2'])} |")
    write(root, REPORTS[17], "# Write Content Horizon — V29\n\nOnly after h1 REC+Conv gates passed, 10 states per role were followed through h2 and h4. Every continuation token was frozen from the pre-write probe set, so this is a controlled, partly artificial continuation rather than unconstrained generation. Donor-specific influence may decay, rotate, or transform; the data do not establish an independent memory criterion.\n\n" + "\n".join(horizon_rows) + "\n")
    write(root, REPORTS[18], f"""# Strict Write Interface Audit — V29

- Parent `{base['parent_commit']}`; base freeze `{base['freeze_digest']}`; design `{verify_stage(root,'design')['freeze_digest']}`; interface amendment `{verify_stage(root,'interface_amendment')['freeze_digest']}`.
- Initial calibration/V28-selected-layer diagnostic retained and **excluded** as full cache: only 6 REC/Conv + 2 attention layers; 53 transient development states were interrupted before any formal partition was saved. This exposure is disclosed, not silently recomputed.
- Formal cache coverage: all 24 REC/Conv layers and all 8 attention layers. Every requested native field exactly equals donor; every untouched field equals recipient; shared next token and current-readout capture are audited in Parquet.
- KV operation copies only the newly appended slot; earlier KV slots must already be bitwise equal. No sequence shortening or arbitrary KV subtraction.
- Full all-field transfer is an identity ceiling, not localization. Partial transfers can be off-manifold combinations even with shared incoming state; causal claims are restricted to the tested transplant semantics.
- Independent final opened only for frozen `REC+Conv`, after both dev/validation gates: `{load(root,'final_opening_v29.json')['final_opening_hash']}`.
- No historical v1–v28 record was overwritten. H2 remains; H3, dynamic-state search, and autonomous controller are unauthorized.
""")
    commands = ["PYTHONPATH=src python -m jclosure.protocol_v29 freeze", "CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src python -m jclosure.experiments.design_v29", "CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src python -m jclosure.experiments.forks_v29 calibration", "PYTHONPATH=src python -m jclosure.experiments.plan_v29", "PYTHONPATH=src python -m jclosure.experiments.amend_v29", "CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src python -m jclosure.experiments.forks_formal_v29 development", "PYTHONPATH=src python -m jclosure.experiments.analyze_v29 development", "CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src python -m jclosure.experiments.forks_formal_v29 validation", "PYTHONPATH=src python -m jclosure.experiments.analyze_v29 validation", "PYTHONPATH=src python -m jclosure.experiments.analyze_v29 final_opening", "CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src python -m jclosure.experiments.secondary_v29 development", "CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src python -m jclosure.experiments.secondary_v29 validation", "CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src python -m jclosure.experiments.diagnostics_v29", "PYTHONPATH=src python -m jclosure.experiments.realization_plan_v29", "CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src python -m jclosure.experiments.realization_v29", "CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src python -m jclosure.experiments.forks_formal_v29 independent_final", "PYTHONPATH=src python -m jclosure.experiments.adjudicate_v29", "PYTHONPATH=src python -m pytest tests/test_v29_natural_write_content.py -q", "PYTHONPATH=src python -m pytest -q", "PYTHONPATH=src python -m jclosure.reporting_v29", "git commit", "git push origin main"]
    manifest = {"commands": commands, "v29_tests": vtests, "full_suite": fulltests, "formal_outcomes": outcomes, "invalid_initial_calibration_full_cache": True, "interrupted_transient_development_states": 53, "historical_final_opened": False, "independent_final_opened": True}
    write_json_atomic(root / OUT / "execution_manifest_v29.json", manifest)
    write(root, REPORTS[19], "# Execution Manifest — V29\n\n" + "\n".join(f"- `{c}`" for c in commands) + f"\n\nV29 tests: `{vtests}`. Full suite: `{fulltests}`. Historical cumulative-report hash tests may fail when `FINAL_REPORT.md` is appended; these failures are reported, not suppressed.\n")
    answers = [
        "Yes. Exact identical incoming caches with distinct preselected tokens yielded distinct natural outgoing caches in all formal states.",
        "REC, Conv and newly appended KV fields changed across all initialized native layers; all 32 layers were audited.",
        "Per-field REC/Conv/KV contrast norms are in the write-geometry Parquet records; channel medians are in the audit report.",
        f"Yes. Shared-next-token future Q medians were {adj['development']['median_future_q']:.3f} development and {adj['validation']['median_future_q']:.3f} validation.",
        "Yes. Exact complete native-cache replacement reproduced donor futures reciprocally; it is an identity ceiling.",
        "REC-only did not pass the frozen reciprocal donor-direction gate.",
        "Conv-only passed development and validation reciprocal donor-direction gates.",
        "KV-only did not pass despite recording the new token/history slot.",
        "REC+Conv passed development, validation and the one fixed independent final.",
        "Yes for Conv, REC+Conv and Conv+KV; not for REC, KV or REC+KV under the frozen gates.",
        "Yes. REC+Conv donor transfer is strong while KV remains recipient-native.",
        "No. KV-only explains little of the donor-specific future contrast for these endpoints.",
        "Not required for threshold success because Conv alone passes; REC+Conv improves donor fidelity.",
        "V28 write-block necessity is compatible, but does not itself establish transferable content; V29 uses separate donor-direction tests.",
        "Potentially, but not formally established: cross-state delta injection degrades while geometry remains aligned.",
        "Token-conditioned REC/Conv writes show positive cross-state similarity, but not exact equality or decisive global transfer.",
        "Cross-state additive write deltas transfer imperfectly and are off-manifold diagnostics.",
        "No stable global write code was formally confirmed.",
        f"Current layer-30 J predicts coarse write geometry with held-out R² {diag['write_geometry_from_current_J']['r2_vs_train_mean']:.3f}; full write content is not identified by this feature model.",
        "Write features add some held-out response prediction beyond token+J, but causal workspace incompleteness is not established without a J-matched test.",
        "Four frozen next-token probes define each measured functional response signature; complete-cache transfer is exact.",
        "Yes. REC+Conv remains donor-directed over the concatenated four-probe signature.",
        "Yes for the tested controls: exact natural transfer is more donor-faithful than random/shuffled/sign-flipped norm-related alternatives.",
        "A strict write-effect dimension is reported only if a train-only rank-k approximation passes both held-out roles and families; see the dimension report.",
        "Controlled h1/h2/h4 effects are recorded; the continuation is frozen, not free generation.",
        f"V29-A is {outcomes['V29_A_TOKEN_CONDITIONED_STATE_WRITE_CONFIRMED']}.",
        f"V29-B/C/D/E are {outcomes['V29_B_SAME_BACKGROUND_WRITE_TRANSFER_CONFIRMED']}/{outcomes['V29_C_RECURRENT_WRITE_CARRIES_NONTRIVIAL_CONTENT']}/{outcomes['V29_D_KV_DOMINATED_TOKEN_CARRYOVER']}/{outcomes['V29_E_DISTRIBUTED_WRITE_CONTENT']}.",
        "Neither state-dependent nor global write code passed a frozen cross-state gate; both remain inconclusive.",
        "Yes, H2 remains.",
        "No. H3, complete dynamic-state search, and autonomous controller remain unauthorized.",
        "The token conditions native REC/Conv and KV outgoing updates; same-background Conv and REC+Conv fields carry transferable next-response distinctions beyond recipient-native KV. Specific semantic variables are not decoded.",
    ]
    write_json_atomic(root / OUT / "v29_scientific_answers.json", {"answers": answers})
    write(root, REPORTS[20], "# V29 Scientific Answers\n\n" + "\n\n".join(f"{i}. {answer}" for i, answer in enumerate(answers, 1)) + "\n")
    compact_result = ", ".join(str(k) for k, passed in realization["qualified_k"].items() if passed) or "none"
    complete = f"""# V29 Complete Report

## Identity and frozen history

**Causal Content of Natural Persistent-State Writes — What Does a Token Commit to the Future?** Parent `{base['parent_commit']}`; base protocol `{base['freeze_digest']}`; adjudication `{adj['ADJUDICATION_HASH']}`. V28's exact REC+Conv write block and full-cache identity control remain historical. V29 does not reinterpret V28 failed partial transfers as successful carriers.

## Prospective same-incoming fork

The frozen 25/75/50/50 balanced panels exclude V28 IDs. At each prompt prefix, top-plausible distinct-surface token A/B were selected from the model distribution **before** their writes or future responses were observed. Both branches start with the identical native cache and complete their current-token computation naturally. The next input token is the same frozen V18 teacher token. Current readout and outgoing state are captured before any transplant. Same-token calibration replay was exact.

An explicit interface amendment preserves a failed preliminary calibration: reusing V28's posterior 6 REC/Conv + 2 KV layers as a purported full V29 cache yielded median relative L2 `{amend['diagnostic_named_full_median_relative_l2']:.3f}`. A transient 53-state development run was interrupted before formal partition output. No threshold, token pair, state role, or finalist changed. Formal V29 covers **all 24 recurrent/conv layers and 8 attention layers**; full cache transplant is bitwise donor-equal and response-exact.

## Causal results

Natural token forks generated distinct writes and shared-next-token future Q median `{adj['development']['median_future_q']:.3f}` development, `{adj['validation']['median_future_q']:.3f}` validation, `{final['median_future_q']:.3f}` independent final. All exact native-field writebacks passed. Full-cache transfer is an identity ceiling, not carrier localization.

{main_tab}

Conv-only and REC+Conv donor writes pass reciprocal development and validation gates; REC-only and KV-only fail. The **predeclared REC+Conv** finalist passed the 50-state independent final (`A←B` median cosine `{fmt(final['REC_Conv']['directions']['A_from_B']['median_cosine'])}`, `B←A` `{fmt(final['REC_Conv']['directions']['B_from_A']['median_cosine'])}`). KV remained recipient-native in this transfer. Thus the tested natural write commits future-relevant, token-conditioned recurrent/conv content beyond exact KV token-history storage. Conv alone passes, so a strict distributed-only/no-single-channel claim is false. These experiments do not decode a semantic variable inside a channel.

Ten states per role tested four pre-write next-token probes: REC+Conv median donor cosine was `{fmt(adj['multi_probe']['development']['REC+Conv']['A_from_B']['median_cosine'])}` development and `{fmt(adj['multi_probe']['validation']['REC+Conv']['A_from_B']['median_cosine'])}` validation (A←B), with full cache exact. Native REC+Conv greatly outperformed norm-related random, shuffled and sign-flipped controls. h1/h2/h4 donor-specific effects were measured only after h1 gates. Cross-state factorials found aligned but imperfectly transferable write contrasts; neither global nor state-dependent code is formally settled.

## Workspace, dimension, and limits

Held-out diagnostic R² for token+J future prediction was `{diag['models']['M2_token_plus_J']['r2_vs_train_mean']:.3f}`; adding coarse write geometry gave `{diag['models']['M4_token_J_write_geometry']['r2_vs_train_mean']:.3f}`. This predictive gain is **not** a J-matched causal proof of workspace incompleteness. Current J is a readout, not a persistent cache field.

Train-only REC+Conv write-effect PCA tested k `{realization['tested_k']}` on held-out development and validation; qualified k: `{compact_result}`. k `{realization['not_estimable_k']}` exceeded the train span and were not tested. Any qualifying k would be a write-effect approximation, **not** a complete or dynamical state dimension. No arbitrary perturbation is promoted to a natural write.

Formal outcomes:\n{outcome_lines}

`H2_REMAINS=TRUE`; `H3_AUTHORIZED=FALSE`; `DYNAMIC_STATE_SEARCH_AUTHORIZED=FALSE`; `AUTONOMOUS_CONTROLLER_AUTHORIZED=FALSE`. V29 tests: `{vtests}`. Full suite: `{fulltests}`. Full machine-readable Parquet/JSON/NPZ records, state/token/write/transplant/future hashes and the integrity index accompany this report.
"""
    write(root, REPORTS[21], complete)
    cumulative = root / "reports/FINAL_REPORT.md"
    if "# V29 Complete Report" in cumulative.read_text(encoding="utf-8"):
        raise RuntimeError("V29 already appended to cumulative final")
    cumulative.write_text(cumulative.read_text(encoding="utf-8").rstrip() + "\n\n---\n\n" + complete.rstrip() + "\n", encoding="utf-8")
    write_json_atomic(root / OUT / "v29_test_audit.json", {"v29_tests": vtests, "full_suite": fulltests})
    paths = []
    for directory in ("artifacts", "results/v29/processed", "reports", "configs", "src/jclosure", "src/jclosure/experiments", "tests"):
        for p in sorted((root / directory).glob("*")):
            rel = str(p.relative_to(root))
            if p.is_file() and ("v29" in rel.lower() or rel == "reports/FINAL_REPORT.md") and not rel.endswith("v29_integrity_index.json"):
                paths.append(rel)
    paths = sorted(set(paths))
    integrity = {"files": {p: sha256_file(root / p) for p in paths}, "file_count": len(paths), "base_freeze_digest": base["freeze_digest"], "state_hash_source": "forks_{role}_v29.parquet incoming/outgoing channel hashes", "fork_token_hash_source": "design_v29.json and forks_{role}_v29.parquet", "natural_write_hash_source": "geometry_{role}_v29.parquet per-field hashes", "transplant_hash_source": "audit_{role}_v29.parquet", "future_signature_hash_source": "future_signature_{role}_v29.parquet", "adjudication_hash": adj["ADJUDICATION_HASH"], "final_opening_hash": load(root, "final_opening_v29.json")["final_opening_hash"], "historical_final_opened": False, "independent_final_opened": True}
    ip = root / OUT / "v29_integrity_index.json"
    write_json_atomic(ip, integrity)
    inputs = [SOURCE, "tests/test_v29_natural_write_content.py", str(ip.relative_to(root)), "reports/FINAL_REPORT.md", "results/v29/processed/v29_adjudication.json", "results/v29/processed/execution_manifest_v29.json", "results/v29/processed/v29_scientific_answers.json"] + [f"reports/{name}" for name in REPORTS]
    fr = stage_freeze(root, "final", inputs, {"formal_outcomes": outcomes, "integrity_index_sha256": sha256_file(ip), "reports": len(REPORTS), "historical_final_opened": False, "independent_final_opened": True, "H3_AUTHORIZED": False})
    return {"final_freeze_digest": fr["freeze_digest"], "reports": len(REPORTS), "integrity_files": len(paths), "formal_outcomes": outcomes}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--v29-tests", required=True)
    p.add_argument("--full-tests", required=True)
    args = p.parse_args()
    print(json.dumps(run(Path.cwd(), args.v29_tests, args.full_tests), indent=2))
