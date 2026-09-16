"""Render protocol-v11 machine records into the six required reports."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

from jclosure.protocol_v11 import (
    BASE_FREEZE_PATH,
    BASELINE_COMMIT,
    CONFIRM_FREEZE_PATH,
    PREPARED_FREEZE_PATH,
    STAGE2_FREEZE_PATH,
)
from jclosure.provenance import sha256_file


def _read(root: Path, name: str) -> dict[str, Any]:
    return json.loads((root / name).read_text(encoding="utf-8"))


def _estimate(value: dict[str, Any] | None) -> str:
    if value is None:
        return "NA"
    return f"{float(value['estimate']):.3f}"


def _causal_table(summary: dict[str, Any], labels: list[str] | None = None) -> str:
    rows = [
        "| method/condition | h | direction | magnitude | semantic | output | sign | gate |",
        "|---|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    allowed = set(labels) if labels is not None else None
    for label, horizons in summary.get("aggregates", {}).items():
        if allowed is not None and label not in allowed:
            continue
        for horizon, groups in sorted(horizons.items(), key=lambda item: int(item[0])):
            pooled = groups["pooled"]
            rows.append(
                "| {label} | {horizon} | {direction} | {magnitude} | {semantic} | "
                "{output} | {sign} | {gate} |".format(
                    label=label,
                    horizon=horizon,
                    direction=_estimate(pooled.get("direction_cosine")),
                    magnitude=_estimate(pooled.get("magnitude_ratio")),
                    semantic=_estimate(pooled.get("semantic_delta_agreement")),
                    output=_estimate(pooled.get("output_direction_cosine")),
                    sign=_estimate(pooled.get("task_decision_sign_agreement")),
                    gate="PASS" if groups.get("all_family_gate_pass") else "FAIL",
                )
            )
    return "\n".join(rows)


def _write(root: Path, name: str, text: str) -> None:
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def _changed_files(root: Path) -> list[str]:
    tracked = subprocess.check_output(
        ["git", "diff", "--name-only", BASELINE_COMMIT], cwd=root, text=True
    ).splitlines()
    untracked = subprocess.check_output(
        ["git", "ls-files", "--others", "--exclude-standard"],
        cwd=root,
        text=True,
    ).splitlines()
    return sorted(set(tracked + untracked))


def _commands() -> list[str]:
    stages = (
        "freeze-base",
        "select-models",
        "prepare-development",
        "freeze-prepared",
        "channel-audit",
        "oracle-development",
        "factorized-stage1",
        "freeze-stage2",
        "factorized-stage2",
        "freeze-confirm",
        "prepare-confirmatory",
        "confirmatory",
        "analyze",
        "report",
    )
    return [f"bash scripts/run_causal_geometry_v11.sh {stage}" for stage in stages]


def _channel_report(root: Path, channel: dict[str, Any]) -> str:
    h1 = {
        name: values["1"]["pooled"]["selection_score"]
        for name, values in channel["aggregates"].items()
        if name != "teacher_reference"
    }
    worst = min(h1, key=h1.get)
    single = {name: h1[name] for name in ("decoded_rec", "decoded_conv", "decoded_kv")}
    primary = min(single, key=single.get)
    return f"""# Channel-wise causal decoder audit — V11

This is a frozen development-bank diagnosis. Teacher raw channels are retained except
where the condition name says `decoded`; all trajectories are teacher-forced.

{_causal_table(channel)}

The weakest single decoded channel at h1 is **{primary}**. The weakest overall decoded
combination at h1 is **{worst}**. Non-additivity is assessed in the amplification machine
record rather than inferred from marginal reconstruction error.

Machine records: `{channel["records"]}` (`{channel["records_sha256"]}`).
"""


def _oracle_report(oracle: dict[str, Any], adjudication: dict[str, Any]) -> str:
    return f"""# Oracle low-rank causal state — V11

The bases are fitted on the training split only. REC, conv, and KV coordinates are exact
block-normalized dual-PCA coordinates of raw teacher deltas; retained coordinates are
decoded by linear combinations of centered training raw deltas. No learned neural decoder
is used.

{_causal_table(oracle)}

Long-horizon authorized methods: `{oracle.get("authorized_methods", [])}`.
Decision-tree outcome: **{adjudication["decision_tree_outcome"]}** —
{adjudication["strongest_warranted_conclusion"]}.

Machine records: `{oracle["records"]}` (`{oracle["records_sha256"]}`).
"""


def _factor_report(
    stage1: dict[str, Any],
    stage2: dict[str, Any],
    confirm: dict[str, Any],
    confirm_freeze: dict[str, Any],
) -> str:
    stage2_table = (
        _causal_table(stage2)
        if stage2.get("aggregates")
        else "No factorized method passed the frozen all-family h1 gate; h2/h4 was gated."
    )
    return f"""# Factorized causal state — V11

Allocation and objective selection used validation/development data only. The tested
objectives cover cache PCA, h1 direction, semantic effect, output effect, multistep effect,
effect weighting, a composite causal objective, and variance-based manifold regularization.

## Stage 1: h1

{_causal_table(stage1)}

## Stage 2: h1/h2/h4

{stage2_table}

Frozen confirmatory methods: `{[v["method"] for v in confirm_freeze["method_specs"]]}`.

## Independent confirmation

{_causal_table(confirm)}
"""


def _manifold_report(manifold: dict[str, Any]) -> str:
    rows = [
        "| state | channel | NN dist | Mahalanobis | local PCA err | outside-128 | cycle err |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    compatibility = []
    for state, channels in manifold["aggregates"].items():
        for channel, metrics in channels.items():
            if channel == "joint_compatibility":
                compatibility.append(
                    f"- `{state}` pairwise nearest-neighbor agreement: "
                    f"{metrics['pairwise_neighbor_agreement']:.3f}; all-three: "
                    f"{metrics['all_channels_same_neighbor']:.3f}."
                )
                continue
            rows.append(
                f"| {state} | {channel} | {metrics['nearest_neighbor_distance']:.3f} | "
                f"{metrics['mahalanobis_distance']:.3f} | "
                f"{metrics['local_pca_reconstruction_error']:.3f} | "
                f"{metrics['outside_global_subspace_energy_fraction']:.3f} | "
                f"{metrics['cycle_relative_error']:.3e} |"
            )
    return """# Causal-state manifold audit — V11

All reference geometry is fitted exclusively to training teacher raw-intervention deltas.
Distances are reported in exact architecture-channel score coordinates.

{table}

## Joint channel compatibility

{compatibility}

Machine records: `{records}` (`{digest}`).
""".format(
        table="\n".join(rows),
        compatibility="\n".join(compatibility),
        records=manifold["records"],
        digest=manifold["records_sha256"],
    )


def _amplification_report(amplification: dict[str, Any]) -> str:
    rows = [
        "| source:method | h | error norm | direction error | semantic error | ratio from prior h |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for label, horizons in amplification["aggregates"].items():
        for horizon, values in sorted(horizons.items(), key=lambda item: int(item[0])):
            ratio = values["mean_error_amplification_ratio"]
            rows.append(
                f"| {label} | {horizon} | {values['mean_error_norm']:.3f} | "
                f"{values['mean_direction_error']:.3f} | {values['mean_semantic_error']:.3f} | "
                f"{ratio:.3f} |"
                if ratio is not None
                else f"| {label} | {horizon} | {values['mean_error_norm']:.3f} | "
                f"{values['mean_direction_error']:.3f} | {values['mean_semantic_error']:.3f} | NA |"
            )
    table = "\n".join(rows)
    return f"""# Causal error amplification — V11

`e_h` is recovered exactly from teacher/decoded effect norms and their cosine. Ratios use
the next measured horizon (1→2→4→8→16), so they are finite-horizon amplification proxies,
not one-step Jacobian eigenvalues. Explicit JVP/Jacobian estimation was not run.

{table}

Machine records: `{amplification["records"]}` (`{amplification["records_sha256"]}`).
"""


def _ceiling_report(ceiling: dict[str, Any]) -> str:
    rows = [
        "| target | combined 599 | architecture 1797 | B − A | metric |",
        "|---|---:|---:|---:|---|",
    ]
    for target, values in ceiling["architecture_gaps"].items():
        rows.append(
            f"| {target} | {values['combined_599']:.3f} | "
            f"{values['architecture_1797']:.3f} | "
            f"{values['architecture_minus_combined']:+.3f} | {values['metric']} |"
        )
    table = "\n".join(rows)
    return f"""# Architecture-resolved ceiling — V11

- Ceiling A: combined block-normalized raw dual-PCA, 599D.
- Ceiling B: separate REC + conv + KV coordinates, 1797D.
- Ceiling C: raw full persistent intervention, whose causal identity fidelity is 1 by definition.

{table}

The old 599D result is therefore called a **combined-reference ceiling**, not a complete raw
persistent-state ceiling.

Machine records: `{ceiling["record_artifact"]}` (`{ceiling["record_artifact_sha256"]}`).
"""


def _final_section(
    root: Path,
    base: dict[str, Any],
    prepared: dict[str, Any],
    stage2: dict[str, Any],
    confirm_freeze: dict[str, Any],
    channel: dict[str, Any],
    oracle: dict[str, Any],
    stage1: dict[str, Any],
    stage2_results: dict[str, Any],
    confirm: dict[str, Any],
    manifold: dict[str, Any],
    amplification: dict[str, Any],
    ceiling: dict[str, Any],
    adjudication: dict[str, Any],
) -> str:
    changed = "\n".join(f"- `{value}`" for value in _changed_files(root))
    commands = "\n".join(f"- `{value}`" for value in _commands())
    hashes = {
        "base": base["freeze_digest"],
        "prepared": prepared["freeze_digest"],
        "stage2": stage2["freeze_digest"],
        "confirmatory": confirm_freeze["freeze_digest"],
    }
    return f"""<!-- V11_START -->
## V11 — Architecture-resolved causal geometry

V11 used a new 50-pair independent confirmatory bank (10/family), disjoint from the 50 v10
development pairs. Development split hash:
`{base["development"]["base_trial_id_sha256"]}`; confirmatory split hash:
`{base["confirmatory"]["base_trial_id_sha256"]}`; confirmatory program hash:
`{base["confirmatory"]["program_hash_sha256"]}`.

Frozen gates: `{json.dumps(base["thresholds"]["causal_gates"], sort_keys=True)}`.
Freeze digests: `{json.dumps(hashes, sort_keys=True)}`.

### Channel-wise hybrid interventions

{_causal_table(channel)}

### Oracle low-rank sweep

{_causal_table(oracle)}

### Unified/factorized and causal-loss ablations

{_causal_table(stage1)}

Stage-2 result:

{_causal_table(stage2_results) if stage2_results.get("aggregates") else "GATED: no h1-qualified method."}

### Independent h1/h2/h4/h8/h16 causal fidelity

{_causal_table(confirm)}

### Architecture ceiling gaps

{_ceiling_report(ceiling)}

### Adjudication

- Decision-tree outcome: **{adjudication["decision_tree_outcome"]}**.
- Strongest warranted conclusion: {adjudication["strongest_warranted_conclusion"]}.
- Smallest causally validated dimension: `{adjudication["smallest_causally_validated_dimension"]}`.
- Hypothesis status: **{adjudication["hypothesis_status"]}**.
- Autonomous-controller training authorized: **{adjudication["autonomous_controller_training_authorized"]}**.
- Free continuation executed: **False**.
- Manifold record: `{manifold["records"]}`.
- Amplification record: `{amplification["records"]}`.

### Exact commands

{commands}

### V11 changed/generated files

{changed}
<!-- V11_END -->"""


def write_reports(root: Path, config: dict[str, Any]) -> dict[str, Any]:
    del config
    base = _read(root, str(BASE_FREEZE_PATH))
    prepared = _read(root, str(PREPARED_FREEZE_PATH))
    stage2 = _read(root, str(STAGE2_FREEZE_PATH))
    confirm_freeze = _read(root, str(CONFIRM_FREEZE_PATH))
    channel = _read(root, "results/v11/processed/channelwise_causal_v11.json")
    oracle = _read(root, "results/v11/processed/oracle_lowrank_causal_v11.json")
    factor1 = _read(root, "results/v11/processed/factorized_causal_stage1_v11.json")
    factor2 = _read(root, "results/v11/processed/factorized_causal_stage2_v11.json")
    confirm = _read(root, "results/v11/processed/causal_confirmatory_v11.json")
    manifold = _read(root, "results/v11/processed/causal_state_manifold_v11.json")
    amplification = _read(
        root, "results/v11/processed/causal_error_amplification_v11.json"
    )
    ceiling = _read(root, "results/v11/processed/architecture_ceiling_v11.json")
    adjudication = _read(root, "results/v11/processed/adjudication_v11.json")
    reports = {
        "reports/CHANNELWISE_CAUSAL_DECODER_AUDIT_V11.md": _channel_report(
            root, channel
        ),
        "reports/ORACLE_LOWRANK_CAUSAL_STATE_V11.md": _oracle_report(
            oracle, adjudication
        ),
        "reports/FACTORIZED_CAUSAL_STATE_V11.md": _factor_report(
            factor1, factor2, confirm, confirm_freeze
        ),
        "reports/CAUSAL_STATE_MANIFOLD_AUDIT_V11.md": _manifold_report(manifold),
        "reports/CAUSAL_ERROR_AMPLIFICATION_V11.md": _amplification_report(
            amplification
        ),
        "reports/ARCH_RESOLVED_CEILING_V11.md": _ceiling_report(ceiling),
    }
    for name, text in reports.items():
        _write(root, name, text)
    section = _final_section(
        root,
        base,
        prepared,
        stage2,
        confirm_freeze,
        channel,
        oracle,
        factor1,
        factor2,
        confirm,
        manifold,
        amplification,
        ceiling,
        adjudication,
    )
    final_path = root / "reports/FINAL_REPORT.md"
    existing = final_path.read_text(encoding="utf-8")
    start = "<!-- V11_START -->"
    end = "<!-- V11_END -->"
    if start in existing and end in existing:
        before = existing.split(start, 1)[0].rstrip()
        after = existing.split(end, 1)[1].lstrip()
        existing = f"{before}\n\n{section}\n\n{after}".rstrip() + "\n"
    else:
        existing = existing.rstrip() + "\n\n" + section + "\n"
    final_path.write_text(existing, encoding="utf-8")
    return {
        "reports": {
            name: sha256_file(root / name)
            for name in [*reports, "reports/FINAL_REPORT.md"]
        },
        "decision_tree_outcome": adjudication["decision_tree_outcome"],
        "hypothesis_status": adjudication["hypothesis_status"],
        "controller_authorized": adjudication[
            "autonomous_controller_training_authorized"
        ],
    }
