"""Staged architecture-resolved causal interventions for protocol v11."""

from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from jclosure.datasets_v8 import load_tasks
from jclosure.experiments.causal_single_v4 import _load_encoder
from jclosure.experiments.common import initialize_context, standard_parser
from jclosure.experiments.compress_persistent_v8 import (
    _capture_manifests,
    _load_raw_deltas,
)
from jclosure.experiments.decoded_causal_v10 import (
    _aggregate,
    _causal_gate,
    _effect_metrics,
    _pair_metadata,
    _raw_order,
    apply_decoded_state,
)
from jclosure.experiments.persistent_channels_v7 import (
    _prefill,
    _teacher_forced_trajectory,
    _teacher_tokens,
)
from jclosure.experiments.persistent_state_v8 import FORMAL_DATA, _semantic_ids
from jclosure.experiments.sufficiency_v9 import _load_data
from jclosure.model import load_model_bundle
from jclosure.protocol_v11 import (
    CONFIRM_FREEZE_PATH,
    PREPARED_FREEZE_PATH,
    PROTOCOL_V11,
    SCHEMA_VERSION_V11,
    STAGE2_FREEZE_PATH,
    build_derived_freeze,
    verify_base_freeze,
    verify_derived_freeze,
)
from jclosure.provenance import sha256_file, write_json_atomic

from .prepare_v11 import (
    CONFIRMATORY_STATES,
    DEVELOPMENT_STATES,
    MODEL_SPECS,
)

CHANNEL_RECORDS = Path("results/v11/processed/channelwise_causal_v11.parquet")
CHANNEL_SUMMARY = Path("results/v11/processed/channelwise_causal_v11.json")
ORACLE_RECORDS = Path("results/v11/processed/oracle_lowrank_causal_v11.parquet")
ORACLE_SUMMARY = Path("results/v11/processed/oracle_lowrank_causal_v11.json")
FACTOR_STAGE1_RECORDS = Path(
    "results/v11/processed/factorized_causal_stage1_v11.parquet"
)
FACTOR_STAGE1_SUMMARY = Path("results/v11/processed/factorized_causal_stage1_v11.json")
FACTOR_STAGE2_RECORDS = Path(
    "results/v11/processed/factorized_causal_stage2_v11.parquet"
)
FACTOR_STAGE2_SUMMARY = Path("results/v11/processed/factorized_causal_stage2_v11.json")
CONFIRM_RECORDS = Path("results/v11/processed/causal_confirmatory_v11.parquet")
CONFIRM_SUMMARY = Path("results/v11/processed/causal_confirmatory_v11.json")


def _tasks_v8(root: Path) -> dict[str, Any]:
    return {task.example_id: task for _, task in load_tasks(root / FORMAL_DATA)}


def _method_specs(root: Path) -> list[dict[str, Any]]:
    return json.loads((root / MODEL_SPECS).read_text(encoding="utf-8"))["methods"]


def _state_rows(
    root: Path,
    summary_path: Path,
    methods: list[str],
) -> tuple[dict[tuple[str, str], dict[str, Any]], dict[str, Any]]:
    summary = json.loads((root / summary_path).read_text(encoding="utf-8"))
    wanted = set(methods)
    output: dict[tuple[str, str], dict[str, Any]] = {}
    found: set[str] = set()
    for method in summary["methods"]:
        name = str(method["spec"]["method"])
        if name not in wanted:
            continue
        found.add(name)
        for declaration in method["shards"]:
            path = root / declaration["path"]
            if sha256_file(path) != declaration["sha256"]:
                raise RuntimeError(f"v11 state shard hash mismatch: {path}")
            payload = torch.load(path, map_location="cpu", weights_only=False)
            for row in payload["rows"]:
                output[(name, str(row["base_trial_id"]))] = row
    missing = wanted - found
    if missing:
        raise RuntimeError(f"v11 prepared methods missing: {sorted(missing)}")
    return output, summary


def _gate_section(context: Any) -> dict[str, Any]:
    return context.config["causal_geometry_v11"]["causal_gates"]


def _selection_score(metrics: dict[str, Any]) -> float:
    def estimate(name: str, default: float = 0.0) -> float:
        value = metrics.get(name)
        if value is None:
            return default
        return float(value["estimate"])

    magnitude = max(estimate("magnitude_ratio", 0.0), 1e-12)
    magnitude_fidelity = max(0.0, 1.0 - abs(float(np.log(magnitude))))
    return float(
        0.30 * estimate("direction_cosine")
        + 0.20 * estimate("semantic_delta_agreement")
        + 0.20 * estimate("output_direction_cosine")
        + 0.15 * estimate("task_decision_sign_agreement", 0.5)
        + 0.15 * magnitude_fidelity
    )


def _aggregate_frame(
    context: Any,
    frame: pd.DataFrame,
    method_column: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    section = context.config["causal_geometry_v11"]
    seeds = [int(value) for value in section["confirmation_seeds"]]
    resamples = int(section["bootstrap_resamples"])
    gates = _gate_section(context)
    aggregates: dict[str, Any] = {}
    authorization: dict[str, Any] = {}
    for method, method_values in frame.groupby(method_column, sort=True):
        method_key = str(method)
        aggregates[method_key] = {}
        horizon_passes: dict[str, bool] = {}
        for horizon, current in method_values.groupby("horizon", sort=True):
            groups = [("pooled", current)]
            groups.extend(
                (str(family), values)
                for family, values in current.groupby("family", sort=True)
            )
            groups.append(("effect_enriched", current[current["effect_enriched"]]))
            group_metrics: dict[str, Any] = {}
            required = []
            for name, values in groups:
                if values.empty:
                    continue
                metrics = _aggregate(values, seeds, resamples)
                gate = _causal_gate(metrics, gates)
                metrics["causal_gate_pass"] = gate
                metrics["selection_score"] = _selection_score(metrics)
                group_metrics[name] = metrics
                if name != "effect_enriched":
                    required.append(gate)
            all_family = bool(required and all(required))
            group_metrics["all_family_gate_pass"] = all_family
            aggregates[method_key][str(int(horizon))] = group_metrics
            horizon_passes[str(int(horizon))] = all_family
        authorization[method_key] = {
            "horizon_all_family_gate_pass": horizon_passes,
            "teacher_forced_causal_authorized": bool(
                horizon_passes
                and all(
                    horizon_passes.get(str(int(horizon)), False)
                    for horizon in section["authorization_horizons"]
                )
            ),
        }
    return aggregates, authorization


def _write_results(
    context: Any,
    *,
    frame: pd.DataFrame,
    records_path: Path,
    summary_path: Path,
    method_column: str,
    source_freeze_digest: str,
    extra: dict[str, Any],
) -> dict[str, Any]:
    path = context.root / records_path
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False, compression="zstd")
    aggregates, authorization = _aggregate_frame(context, frame, method_column)
    authorized = [
        method
        for method, value in authorization.items()
        if value["teacher_forced_causal_authorized"]
    ]
    summary = {
        "schema_version": SCHEMA_VERSION_V11,
        "protocol_version": PROTOCOL_V11,
        "run_id": context.run_id,
        "source_freeze_digest": source_freeze_digest,
        "strict_interface": {
            "verified": True,
            "function_sha256": hashlib.sha256(
                inspect.getsource(apply_decoded_state).encode()
            ).hexdigest(),
            "allowed_inputs": [
                "clean_cache",
                "prepared_delta",
                "teacher_forced_token",
            ],
            "raw_target_cache_available_to_compact_continuation": False,
        },
        "aggregates": aggregates,
        "authorization": authorization,
        "authorized_methods": authorized,
        "controller_authorized": bool(authorized),
        "free_continuation_executed": False,
        "records": str(records_path),
        "records_sha256": sha256_file(path),
        **extra,
    }
    write_json_atomic(context.root / summary_path, summary)
    return summary


def _raw_and_metadata(context: Any) -> tuple[dict[str, Any], dict[str, Any], Any]:
    data = _load_data(context.root)
    raw, raw_ids, _, _, _ = _load_raw_deltas(context, _capture_manifests(context.root))
    return data, _raw_order(raw, raw_ids, data["ids"]), _pair_metadata(context.root)


def _run_prepared(
    context: Any,
    base: dict[str, Any],
    *,
    split: str,
    specs: list[dict[str, Any]],
    horizons_by_method: dict[str, list[int]],
    state_summary_path: Path,
    records_path: Path,
    summary_path: Path,
    source_freeze_digest: str,
    record_type: str,
) -> dict[str, Any]:
    methods = [str(spec["method"]) for spec in specs]
    state_rows, state_summary = _state_rows(context.root, state_summary_path, methods)
    data, raw, metadata = _raw_and_metadata(context)
    id_lookup = {str(value): index for index, value in enumerate(data["ids"].tolist())}
    ids = [str(value) for value in base[split]["base_trial_ids"]]
    enriched = set(base[split].get("effect_enriched_base_trial_ids", []))
    if split == "development":
        v10 = json.loads(
            (
                context.root / "artifacts/causal_sufficiency_v10_candidates.freeze.json"
            ).read_text()
        )
        enriched = set(v10.get("effect_enriched_base_trial_ids", []))
    tasks = _tasks_v8(context.root)
    bundle = load_model_bundle(context.config)
    _, _, dense_map = _load_encoder(context, bundle)
    v8 = context.config["persistent_state_v8"]
    measured = [int(value) for value in v8["intervention"]["measured_layers"]]
    main_layer = max(measured)
    intervention_layer = int(v8["intervention"]["layer"])
    recurrent_layers = [
        int(value) for value in v8["persistent_components"]["recurrent_layers"]
    ]
    attention_layers = [
        int(value) for value in v8["persistent_components"]["attention_layers"]
    ]
    maximum_horizon = max(
        horizon for values in horizons_by_method.values() for horizon in values
    )
    records: list[dict[str, Any]] = []
    progress = context.raw_dir / context.run_id / f"{record_type}_progress.json"
    for case_index, base_id in enumerate(ids):
        source_index = id_lookup[base_id]
        pair = metadata[base_id]
        task = tasks[str(pair["prompt_id"])]
        clean = _prefill(
            bundle,
            task.prompt,
            measured_layers=measured,
            dense_map=dense_map,
            intervention_layer=intervention_layer,
            candidate=None,
        )
        tokens = _teacher_tokens(
            bundle,
            clean,
            count=maximum_horizon,
            measured_layers=measured,
            dense_map=dense_map,
        )
        clean_trajectory = _teacher_forced_trajectory(
            bundle,
            clean["cache"],
            tokens,
            prompt_length=clean["prompt_length"],
            measured_layers=measured,
            dense_map=dense_map,
        )
        teacher_delta = (
            raw["recurrent"][source_index],
            raw["conv"][source_index],
            raw["kv_full"][source_index],
        )
        teacher_cache = apply_decoded_state(
            clean["cache"],
            teacher_delta,
            recurrent_layers=recurrent_layers,
            attention_layers=attention_layers,
            prompt_length=clean["prompt_length"],
        )
        teacher_trajectory = _teacher_forced_trajectory(
            bundle,
            teacher_cache,
            tokens,
            prompt_length=clean["prompt_length"],
            measured_layers=measured,
            dense_map=dense_map,
        )
        del teacher_cache
        for spec in specs:
            method = str(spec["method"])
            row = state_rows[(method, base_id)]
            decoded_cache = apply_decoded_state(
                clean["cache"],
                (row["recurrent"], row["conv"], row["kv"]),
                recurrent_layers=recurrent_layers,
                attention_layers=attention_layers,
                prompt_length=clean["prompt_length"],
            )
            decoded_trajectory = _teacher_forced_trajectory(
                bundle,
                decoded_cache,
                tokens,
                prompt_length=clean["prompt_length"],
                measured_layers=measured,
                dense_map=dense_map,
            )
            del decoded_cache
            for horizon in horizons_by_method[method]:
                semantic_index = min(horizon - 1, len(task.semantic_actions) - 1)
                metrics = _effect_metrics(
                    clean_trajectory,
                    teacher_trajectory,
                    decoded_trajectory,
                    main_layer=main_layer,
                    horizon=horizon,
                    semantic_ids=_semantic_ids(
                        bundle.tokenizer, task.semantic_actions[semantic_index]
                    ),
                )
                records.append(
                    {
                        "schema_version": SCHEMA_VERSION_V11,
                        "protocol_version": PROTOCOL_V11,
                        "record_type": record_type,
                        "run_id": context.run_id,
                        "source_freeze_digest": source_freeze_digest,
                        "base_trial_id": base_id,
                        "prompt_id": str(pair["prompt_id"]),
                        "family": str(pair["family"]),
                        "split_role": split,
                        "method": method,
                        "method_class": str(spec["class"]),
                        "dimension": int(spec["dimension"]),
                        "objective": str(spec["objective"]),
                        "allocation": json.dumps(spec.get("allocation")),
                        "horizon": int(horizon),
                        "effect_enriched": base_id in enriched,
                        "teacher_forced": True,
                        "strict_interface": True,
                        **metrics,
                    }
                )
        write_json_atomic(
            progress,
            {
                "status": "RUNNING",
                "completed_cases": case_index + 1,
                "total_cases": len(ids),
                "last_base_trial_id": base_id,
            },
        )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    frame = pd.DataFrame(records)
    summary = _write_results(
        context,
        frame=frame,
        records_path=records_path,
        summary_path=summary_path,
        method_column="method",
        source_freeze_digest=source_freeze_digest,
        extra={
            "split_role": split,
            "methods": specs,
            "horizons_by_method": horizons_by_method,
            "state_preparation_run_id": state_summary["run_id"],
        },
    )
    write_json_atomic(
        progress,
        {"status": "COMPLETED", "completed_cases": len(ids), "total_cases": len(ids)},
    )
    return summary


def _channel_audit(context: Any, base: dict[str, Any]) -> dict[str, Any]:
    spec = next(
        value
        for value in _method_specs(context.root)
        if value["method"] == "unified_causal_d512"
    )
    state_rows, _ = _state_rows(context.root, DEVELOPMENT_STATES, [str(spec["method"])])
    data, raw, metadata = _raw_and_metadata(context)
    id_lookup = {str(value): index for index, value in enumerate(data["ids"].tolist())}
    ids = [str(value) for value in base["development"]["base_trial_ids"]]
    v10 = json.loads(
        (
            context.root / "artifacts/causal_sufficiency_v10_candidates.freeze.json"
        ).read_text()
    )
    enriched = set(v10.get("effect_enriched_base_trial_ids", []))
    conditions = {
        "decoded_rec": (True, False, False),
        "decoded_conv": (False, True, False),
        "decoded_kv": (False, False, True),
        "decoded_rec_conv": (True, True, False),
        "decoded_rec_kv": (True, False, True),
        "decoded_conv_kv": (False, True, True),
        "decoded_all": (True, True, True),
        "teacher_reference": (False, False, False),
    }
    tasks = _tasks_v8(context.root)
    bundle = load_model_bundle(context.config)
    _, _, dense_map = _load_encoder(context, bundle)
    v8 = context.config["persistent_state_v8"]
    measured = [int(value) for value in v8["intervention"]["measured_layers"]]
    main_layer = max(measured)
    intervention_layer = int(v8["intervention"]["layer"])
    recurrent_layers = [
        int(value) for value in v8["persistent_components"]["recurrent_layers"]
    ]
    attention_layers = [
        int(value) for value in v8["persistent_components"]["attention_layers"]
    ]
    horizons = [1, 2, 4, 8]
    records: list[dict[str, Any]] = []
    progress = context.raw_dir / context.run_id / "channel_audit_progress.json"
    for case_index, base_id in enumerate(ids):
        source_index = id_lookup[base_id]
        pair = metadata[base_id]
        task = tasks[str(pair["prompt_id"])]
        clean = _prefill(
            bundle,
            task.prompt,
            measured_layers=measured,
            dense_map=dense_map,
            intervention_layer=intervention_layer,
            candidate=None,
        )
        tokens = _teacher_tokens(
            bundle,
            clean,
            count=max(horizons),
            measured_layers=measured,
            dense_map=dense_map,
        )
        clean_trajectory = _teacher_forced_trajectory(
            bundle,
            clean["cache"],
            tokens,
            prompt_length=clean["prompt_length"],
            measured_layers=measured,
            dense_map=dense_map,
        )
        raw_delta = (
            raw["recurrent"][source_index],
            raw["conv"][source_index],
            raw["kv_full"][source_index],
        )
        teacher_cache = apply_decoded_state(
            clean["cache"],
            raw_delta,
            recurrent_layers=recurrent_layers,
            attention_layers=attention_layers,
            prompt_length=clean["prompt_length"],
        )
        teacher_trajectory = _teacher_forced_trajectory(
            bundle,
            teacher_cache,
            tokens,
            prompt_length=clean["prompt_length"],
            measured_layers=measured,
            dense_map=dense_map,
        )
        del teacher_cache
        decoded_row = state_rows[(str(spec["method"]), base_id)]
        decoded_delta = (
            decoded_row["recurrent"],
            decoded_row["conv"],
            decoded_row["kv"],
        )
        for condition, use_decoded in conditions.items():
            hybrid_delta = tuple(
                decoded_delta[index] if use_decoded[index] else raw_delta[index]
                for index in range(3)
            )
            hybrid_cache = apply_decoded_state(
                clean["cache"],
                hybrid_delta,
                recurrent_layers=recurrent_layers,
                attention_layers=attention_layers,
                prompt_length=clean["prompt_length"],
            )
            trajectory = _teacher_forced_trajectory(
                bundle,
                hybrid_cache,
                tokens,
                prompt_length=clean["prompt_length"],
                measured_layers=measured,
                dense_map=dense_map,
            )
            del hybrid_cache
            for horizon in horizons:
                semantic_index = min(horizon - 1, len(task.semantic_actions) - 1)
                metrics = _effect_metrics(
                    clean_trajectory,
                    teacher_trajectory,
                    trajectory,
                    main_layer=main_layer,
                    horizon=horizon,
                    semantic_ids=_semantic_ids(
                        bundle.tokenizer, task.semantic_actions[semantic_index]
                    ),
                )
                records.append(
                    {
                        "schema_version": SCHEMA_VERSION_V11,
                        "protocol_version": PROTOCOL_V11,
                        "record_type": "channelwise_hybrid_causal_audit",
                        "run_id": context.run_id,
                        "source_freeze_digest": base["freeze_digest"],
                        "base_trial_id": base_id,
                        "prompt_id": str(pair["prompt_id"]),
                        "family": str(pair["family"]),
                        "split_role": "development",
                        "condition": condition,
                        "dimension": 512,
                        "horizon": horizon,
                        "effect_enriched": base_id in enriched,
                        "teacher_forced": True,
                        "strict_interface": True,
                        **metrics,
                    }
                )
        write_json_atomic(
            progress,
            {
                "status": "RUNNING",
                "completed_cases": case_index + 1,
                "total_cases": len(ids),
                "last_base_trial_id": base_id,
            },
        )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    frame = pd.DataFrame(records)
    summary = _write_results(
        context,
        frame=frame,
        records_path=CHANNEL_RECORDS,
        summary_path=CHANNEL_SUMMARY,
        method_column="condition",
        source_freeze_digest=base["freeze_digest"],
        extra={
            "split_role": "development",
            "decoded_method": spec,
            "teacher_forced_horizons": horizons,
            "hybrid_definition": conditions,
        },
    )
    write_json_atomic(
        progress,
        {"status": "COMPLETED", "completed_cases": len(ids), "total_cases": len(ids)},
    )
    return summary


def _ranked_methods(
    summary: dict[str, Any],
    methods: list[str],
    horizons: list[int],
    *,
    require_gate: bool,
) -> list[tuple[float, str]]:
    ranked = []
    for method in methods:
        if method not in summary["aggregates"]:
            continue
        gates = summary["authorization"][method]["horizon_all_family_gate_pass"]
        if require_gate and not all(
            gates.get(str(horizon), False) for horizon in horizons
        ):
            continue
        scores = [
            float(
                summary["aggregates"][method][str(horizon)]["pooled"]["selection_score"]
            )
            for horizon in horizons
            if str(horizon) in summary["aggregates"][method]
        ]
        if scores:
            ranked.append((float(np.mean(scores)), method))
    return sorted(ranked, key=lambda value: (value[0], value[1]), reverse=True)


def _freeze_stage2(context: Any, base: dict[str, Any]) -> dict[str, Any]:
    verify_derived_freeze(context.root, context.config, PREPARED_FREEZE_PATH)
    summary = json.loads((context.root / FACTOR_STAGE1_SUMMARY).read_text())
    specs = {
        str(value["method"]): value
        for value in _method_specs(context.root)
        if str(value["class"]).startswith("factorized_")
    }
    eligible = _ranked_methods(summary, list(specs), [1], require_gate=True)
    selected: list[str] = []
    for class_name in ("factorized_no_int", "factorized_int"):
        current = [
            (score, method)
            for score, method in eligible
            if specs[method]["class"] == class_name
        ]
        if current:
            selected.append(current[0][1])
    limit = int(
        context.config["causal_geometry_v11"]["factorized"]["stage2_total_methods"]
    )
    selected.extend(method for _, method in eligible if method not in selected)
    selected = selected[:limit]
    payload = {
        "stage1_gate": "h1 all-family frozen causal gate",
        "eligible_method_count": len(eligible),
        "selected_methods": selected,
        "method_specs": [specs[value] for value in selected],
        "selection_split": "development",
        "confirmatory_used": False,
    }
    return build_derived_freeze(
        context.root,
        context.config,
        path=STAGE2_FREEZE_PATH,
        purpose="freeze h1-qualified factorized methods before h2/h4",
        inputs=[
            PREPARED_FREEZE_PATH,
            FACTOR_STAGE1_RECORDS,
            FACTOR_STAGE1_SUMMARY,
        ],
        payload=payload,
    )


def _freeze_confirm(context: Any, base: dict[str, Any]) -> dict[str, Any]:
    stage2 = verify_derived_freeze(context.root, context.config, STAGE2_FREEZE_PATH)
    specs = {str(value["method"]): value for value in _method_specs(context.root)}
    oracle = json.loads((context.root / ORACLE_SUMMARY).read_text())
    factor1 = json.loads((context.root / FACTOR_STAGE1_SUMMARY).read_text())
    factor2_path = context.root / FACTOR_STAGE2_SUMMARY
    factor2 = json.loads(factor2_path.read_text()) if factor2_path.is_file() else None
    selected: list[dict[str, Any]] = []

    def add(method: str, role: str, maximum_horizon: int) -> None:
        value = dict(specs[method])
        value["confirmatory_role"] = role
        value["maximum_confirmatory_horizon"] = maximum_horizon
        selected.append(value)

    add("unified_causal_d512", "historical_unified_baseline", 16)
    for class_name in ("oracle_joint", "oracle_factorized"):
        methods = [
            name for name, value in specs.items() if value["class"] == class_name
        ]
        ranked = _ranked_methods(oracle, methods, [1], require_gate=False)
        if ranked:
            method = ranked[0][1]
            h1_pass = oracle["authorization"][method][
                "horizon_all_family_gate_pass"
            ].get("1", False)
            h4_pass = all(
                oracle["authorization"][method]["horizon_all_family_gate_pass"].get(
                    str(horizon), False
                )
                for horizon in (1, 2, 4)
            )
            add(
                method,
                "development_selected_oracle",
                16 if h4_pass else (4 if h1_pass else 1),
            )
    factor_methods = [
        name
        for name, value in specs.items()
        if str(value["class"]).startswith("factorized_")
    ]
    factor_ranked: list[tuple[float, str]] = []
    factor_max = 1
    if factor2 is not None and stage2["selected_methods"]:
        factor_ranked = _ranked_methods(
            factor2, stage2["selected_methods"], [1, 2, 4], require_gate=True
        )
        if factor_ranked:
            factor_max = 16
    if not factor_ranked:
        factor_ranked = _ranked_methods(
            factor1, factor_methods, [1], require_gate=False
        )
        factor_max = 1
    if factor_ranked and len(selected) < int(
        context.config["causal_geometry_v11"]["confirmatory_method_limit"]
    ):
        add(
            factor_ranked[0][1],
            "development_selected_factorized",
            factor_max,
        )
    inputs = [
        STAGE2_FREEZE_PATH,
        ORACLE_RECORDS,
        ORACLE_SUMMARY,
        FACTOR_STAGE1_RECORDS,
        FACTOR_STAGE1_SUMMARY,
    ]
    if factor2 is not None:
        inputs.extend([FACTOR_STAGE2_RECORDS, FACTOR_STAGE2_SUMMARY])
    return build_derived_freeze(
        context.root,
        context.config,
        path=CONFIRM_FREEZE_PATH,
        purpose="freeze development-selected methods before independent v11 confirmation",
        inputs=inputs,
        payload={
            "selection_split": "development",
            "confirmatory_used": False,
            "method_specs": selected,
            "method_count": len(selected),
            "staged_rule": (
                "h1-only after h1 failure; h1/h2/h4 qualify a method for h8/h16"
            ),
        },
    )


def main() -> None:
    parser = standard_parser(
        "architecture-resolved causal interventions v11",
        "configs/causal_geometry_v11.yaml",
    )
    parser.add_argument(
        "--stage",
        required=True,
        choices=(
            "channel-audit",
            "oracle-development",
            "factorized-stage1",
            "freeze-stage2",
            "factorized-stage2",
            "freeze-confirm",
            "confirmatory",
        ),
    )
    args = parser.parse_args()
    context = initialize_context("causal-v11", args)
    try:
        base = verify_base_freeze(context.root, context.config)
        prepared = verify_derived_freeze(
            context.root, context.config, PREPARED_FREEZE_PATH
        )
        specs = _method_specs(context.root)
        if args.stage == "channel-audit":
            summary = _channel_audit(context, base)
        elif args.stage == "oracle-development":
            chosen = [
                value for value in specs if str(value["class"]).startswith("oracle_")
            ]
            horizons = [
                int(value)
                for value in context.config["causal_geometry_v11"]["horizons"]
            ]
            summary = _run_prepared(
                context,
                base,
                split="development",
                specs=chosen,
                horizons_by_method={str(value["method"]): horizons for value in chosen},
                state_summary_path=DEVELOPMENT_STATES,
                records_path=ORACLE_RECORDS,
                summary_path=ORACLE_SUMMARY,
                source_freeze_digest=prepared["freeze_digest"],
                record_type="oracle_lowrank_causal_projection",
            )
        elif args.stage == "factorized-stage1":
            chosen = [
                value
                for value in specs
                if str(value["class"]).startswith("factorized_")
            ]
            summary = _run_prepared(
                context,
                base,
                split="development",
                specs=chosen,
                horizons_by_method={str(value["method"]): [1] for value in chosen},
                state_summary_path=DEVELOPMENT_STATES,
                records_path=FACTOR_STAGE1_RECORDS,
                summary_path=FACTOR_STAGE1_SUMMARY,
                source_freeze_digest=prepared["freeze_digest"],
                record_type="factorized_causal_stage1",
            )
        elif args.stage == "freeze-stage2":
            summary = _freeze_stage2(context, base)
        elif args.stage == "factorized-stage2":
            stage2 = verify_derived_freeze(
                context.root, context.config, STAGE2_FREEZE_PATH
            )
            chosen = stage2["method_specs"]
            if not chosen:
                summary = {
                    "status": "GATED_NO_H1_QUALIFIED_FACTORIZED_METHOD",
                    "source_freeze_digest": stage2["freeze_digest"],
                    "methods": [],
                }
                write_json_atomic(context.root / FACTOR_STAGE2_SUMMARY, summary)
                pd.DataFrame().to_parquet(context.root / FACTOR_STAGE2_RECORDS)
            else:
                summary = _run_prepared(
                    context,
                    base,
                    split="development",
                    specs=chosen,
                    horizons_by_method={
                        str(value["method"]): [1, 2, 4] for value in chosen
                    },
                    state_summary_path=DEVELOPMENT_STATES,
                    records_path=FACTOR_STAGE2_RECORDS,
                    summary_path=FACTOR_STAGE2_SUMMARY,
                    source_freeze_digest=stage2["freeze_digest"],
                    record_type="factorized_causal_stage2",
                )
        elif args.stage == "freeze-confirm":
            summary = _freeze_confirm(context, base)
        else:
            confirm = verify_derived_freeze(
                context.root, context.config, CONFIRM_FREEZE_PATH
            )
            chosen = confirm["method_specs"]
            available = [
                int(value)
                for value in context.config["causal_geometry_v11"]["horizons"]
            ]
            horizons_by_method = {
                str(value["method"]): [
                    horizon
                    for horizon in available
                    if horizon <= int(value["maximum_confirmatory_horizon"])
                ]
                for value in chosen
            }
            summary = _run_prepared(
                context,
                base,
                split="confirmatory",
                specs=chosen,
                horizons_by_method=horizons_by_method,
                state_summary_path=CONFIRMATORY_STATES,
                records_path=CONFIRM_RECORDS,
                summary_path=CONFIRM_SUMMARY,
                source_freeze_digest=confirm["freeze_digest"],
                record_type="independent_causal_confirmatory",
            )
        context.finish("COMPLETED_V11_CAUSAL_STAGE", summary=summary)
    except Exception as exc:
        context.fail_progress(f"{type(exc).__name__}: {exc}")
        context.finish("FAILED", error=f"{type(exc).__name__}: {exc}")
        raise


if __name__ == "__main__":
    main()
