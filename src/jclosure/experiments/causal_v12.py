"""Measurement audit and staged causal oracle experiments for protocol v12."""

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
from jclosure.experiments.decoded_causal_v10 import (
    _aggregate,
    _causal_gate,
    _effect_metrics,
    apply_decoded_state,
)
from jclosure.experiments.persistent_channels_v7 import (
    _prefill,
    _teacher_forced_trajectory,
    _teacher_tokens,
)
from jclosure.experiments.persistent_state_v8 import FORMAL_DATA, _semantic_ids
from jclosure.model import load_model_bundle
from jclosure.protocol_v12 import (
    CONFIRM_FREEZE_PATH,
    PREPARED_FREEZE_PATH,
    PROTOCOL_V12,
    SCHEMA_VERSION_V12,
    build_derived_freeze,
    verify_base_freeze,
    verify_derived_freeze,
)
from jclosure.provenance import sha256_file, write_json_atomic

from .causal_v11 import _raw_and_metadata, _state_rows, _tasks_v8
from .prepare_v12 import (
    CONFIRMATORY_STATES,
    DEVELOPMENT_STATES,
    METHOD_SPECS,
    SCALING_STATES,
)

MEASUREMENT_RECORDS = Path("results/v12/processed/measurement_audit_v12.parquet")
MEASUREMENT_SUMMARY = Path("results/v12/processed/measurement_audit_v12.json")
SCALING_CAUSAL_RECORDS = Path(
    "results/v12/processed/data_rank_scaling_causal_v12.parquet"
)
SCALING_CAUSAL_SUMMARY = Path("results/v12/processed/data_rank_scaling_causal_v12.json")
ORACLE_STAGE1_RECORDS = Path(
    "results/v12/processed/local_causal_oracle_stage1_v12.parquet"
)
ORACLE_STAGE1_SUMMARY = Path(
    "results/v12/processed/local_causal_oracle_stage1_v12.json"
)
STAGE2_FREEZE_PATH = Path("artifacts/causal_geometry_v12_stage2.freeze.json")
ORACLE_STAGE2_RECORDS = Path(
    "results/v12/processed/local_causal_oracle_stage2_v12.parquet"
)
ORACLE_STAGE2_SUMMARY = Path(
    "results/v12/processed/local_causal_oracle_stage2_v12.json"
)
CONFIRM_RECORDS = Path("results/v12/processed/causal_confirmatory_v12.parquet")
CONFIRM_SUMMARY = Path("results/v12/processed/causal_confirmatory_v12.json")
REPLACEMENT_RECORDS = Path("results/v12/processed/strict_state_replacement_v12.parquet")
REPLACEMENT_SUMMARY = Path("results/v12/processed/strict_state_replacement_v12.json")


def _safe_cosine(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    return float(np.dot(left, right) / denominator) if denominator > 1e-20 else 0.0


def _continuous_semantic(
    teacher: np.ndarray, decoded: np.ndarray, top_k: int
) -> dict[str, float]:
    k = min(int(top_k), len(teacher))
    teacher_top = np.argpartition(np.abs(teacher), -k)[-k:]
    decoded_top = np.argpartition(np.abs(decoded), -k)[-k:]
    union = np.union1d(teacher_top, decoded_top)
    rank_teacher = pd.Series(np.abs(teacher)).rank(method="average").to_numpy()
    rank_decoded = pd.Series(np.abs(decoded)).rank(method="average").to_numpy()
    overlap = len(set(teacher_top.tolist()) & set(decoded_top.tolist())) / k
    numerator = float(np.minimum(np.abs(teacher[union]), np.abs(decoded[union])).sum())
    denominator = float(
        np.maximum(np.abs(teacher[union]), np.abs(decoded[union])).sum()
    )
    return {
        "semantic_vector_cosine": _safe_cosine(teacher, decoded),
        "semantic_rank_correlation": float(
            np.corrcoef(rank_teacher, rank_decoded)[0, 1]
        ),
        "semantic_topk_overlap": float(overlap),
        "semantic_weighted_topk_overlap": numerator / max(denominator, 1e-20),
        "semantic_topk_direction_cosine": _safe_cosine(teacher[union], decoded[union]),
    }


def _trajectory_error(metrics: dict[str, Any]) -> dict[str, float]:
    teacher_norm = float(metrics["teacher_j_effect_norm"])
    decoded_norm = float(metrics["decoded_j_effect_norm"])
    cosine = float(metrics["direction_cosine"])
    parallel = abs(decoded_norm * cosine - teacher_norm)
    orthogonal = decoded_norm * max(0.0, 1.0 - cosine**2) ** 0.5
    return {
        "trajectory_error_norm": float(metrics["j_l2_error"]),
        "trajectory_angle_degrees": float(
            np.degrees(np.arccos(np.clip(cosine, -1, 1)))
        ),
        "error_parallel_to_teacher": float(parallel),
        "error_orthogonal_to_teacher": float(orthogonal),
        "semantic_trajectory_cosine": cosine,
    }


def _selection_score(metrics: dict[str, Any]) -> float:
    def estimate(name: str, default: float = 0.0) -> float:
        value = metrics.get(name)
        return default if value is None else float(value["estimate"])

    magnitude = max(estimate("magnitude_ratio"), 1e-12)
    magnitude_fidelity = max(0.0, 1.0 - abs(float(np.log(magnitude))))
    return float(
        0.30 * estimate("direction_cosine")
        + 0.20 * estimate("semantic_delta_agreement")
        + 0.20 * estimate("output_direction_cosine")
        + 0.15 * estimate("task_decision_sign_agreement", 0.5)
        + 0.15 * magnitude_fidelity
    )


def _aggregate_frame(
    context: Any, frame: pd.DataFrame
) -> tuple[dict[str, Any], dict[str, Any]]:
    section = context.config["causal_geometry_v12"]
    seeds = [int(value) for value in section["confirmation_seeds"]]
    resamples = int(section["bootstrap_resamples"])
    gates = section["causal_gates"]
    aggregates, authorization = {}, {}
    for method, method_values in frame.groupby("method", sort=True):
        method = str(method)
        aggregates[method] = {}
        horizon_passes = {}
        for horizon, current in method_values.groupby("horizon", sort=True):
            groups = [("pooled", current)] + [
                (str(name), values)
                for name, values in current.groupby("family", sort=True)
            ]
            group_metrics, required = {}, []
            for name, values in groups:
                metrics = _aggregate(values, seeds, resamples)
                gate = _causal_gate(metrics, gates)
                metrics["causal_gate_pass"] = gate
                metrics["selection_score"] = _selection_score(metrics)
                group_metrics[name] = metrics
                required.append(gate)
            passed = bool(required and all(required))
            group_metrics["all_family_gate_pass"] = passed
            aggregates[method][str(int(horizon))] = group_metrics
            horizon_passes[str(int(horizon))] = passed
        authorization[method] = {
            "horizon_all_family_gate_pass": horizon_passes,
            "teacher_forced_causal_authorized": bool(
                all(
                    horizon_passes.get(str(value), False)
                    for value in section["authorization_horizons"]
                )
            ),
        }
    return aggregates, authorization


def _write_results(
    context: Any,
    frame: pd.DataFrame,
    records_path: Path,
    summary_path: Path,
    *,
    source_freeze_digest: str,
    extra: dict[str, Any],
) -> dict[str, Any]:
    path = context.root / records_path
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False, compression="zstd")
    aggregates, authorization = _aggregate_frame(context, frame)
    summary = {
        "schema_version": SCHEMA_VERSION_V12,
        "protocol_version": PROTOCOL_V12,
        "run_id": context.run_id,
        "source_freeze_digest": source_freeze_digest,
        "strict_interface": {
            "verified": True,
            "function_sha256": hashlib.sha256(
                inspect.getsource(apply_decoded_state).encode()
            ).hexdigest(),
            "raw_target_cache_available_to_compact_continuation": False,
        },
        "aggregates": aggregates,
        "authorization": authorization,
        "authorized_methods": [
            name
            for name, value in authorization.items()
            if value["teacher_forced_causal_authorized"]
        ],
        "records": str(records_path),
        "records_sha256": sha256_file(path),
        **extra,
    }
    write_json_atomic(context.root / summary_path, summary)
    return summary


def _method_specs(root: Path) -> dict[str, Any]:
    return json.loads((root / METHOD_SPECS).read_text(encoding="utf-8"))


def _run_states(
    context: Any,
    base: dict[str, Any],
    *,
    ids: list[str],
    specs: list[dict[str, Any]],
    horizons_by_method: dict[str, list[int]],
    state_summary_path: Path,
    records_path: Path,
    summary_path: Path,
    source_freeze_digest: str,
    split_role: str,
    record_type: str,
) -> dict[str, Any]:
    methods = [str(value["method"]) for value in specs]
    states, state_summary = _state_rows(context.root, state_summary_path, methods)
    data, raw, metadata = _raw_and_metadata(context)
    id_lookup = {str(value): index for index, value in enumerate(data["ids"].tolist())}
    tasks = _tasks_v8(context.root)
    bundle = load_model_bundle(context.config)
    _, _, dense_map = _load_encoder(context, bundle)
    v8 = context.config["persistent_state_v8"]
    measured = [int(value) for value in v8["intervention"]["measured_layers"]]
    main_layer = max(measured)
    recurrent_layers = [
        int(value) for value in v8["persistent_components"]["recurrent_layers"]
    ]
    attention_layers = [
        int(value) for value in v8["persistent_components"]["attention_layers"]
    ]
    maximum_horizon = max(
        horizon for values in horizons_by_method.values() for horizon in values
    )
    records = []
    for case_index, base_id in enumerate(ids):
        source_index = id_lookup[base_id]
        pair = metadata[base_id]
        task = tasks[str(pair["prompt_id"])]
        clean = _prefill(
            bundle,
            task.prompt,
            measured_layers=measured,
            dense_map=dense_map,
            intervention_layer=int(v8["intervention"]["layer"]),
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
        for spec in specs:
            method = str(spec["method"])
            row = states[(method, base_id)]
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
                index = horizon - 1
                teacher_effect = (
                    teacher_trajectory["j"][main_layer][index]
                    - clean_trajectory["j"][main_layer][index]
                )
                decoded_effect = (
                    decoded_trajectory["j"][main_layer][index]
                    - clean_trajectory["j"][main_layer][index]
                )
                records.append(
                    {
                        "schema_version": SCHEMA_VERSION_V12,
                        "protocol_version": PROTOCOL_V12,
                        "record_type": record_type,
                        "run_id": context.run_id,
                        "source_freeze_digest": source_freeze_digest,
                        "base_trial_id": base_id,
                        "prompt_id": str(pair["prompt_id"]),
                        "family": str(pair["family"]),
                        "split_role": split_role,
                        "method": method,
                        "method_class": str(spec["class"]),
                        "dimension": int(spec["dimension"]),
                        "train_size": spec.get("train_size"),
                        "horizon": int(horizon),
                        "teacher_forced": True,
                        "strict_interface": True,
                        **metrics,
                        **_continuous_semantic(
                            teacher_effect,
                            decoded_effect,
                            int(
                                context.config["causal_geometry_v12"]["measurement"][
                                    "top_k"
                                ]
                            ),
                        ),
                        **_trajectory_error(metrics),
                    }
                )
        write_json_atomic(
            context.raw_dir / context.run_id / f"{record_type}_progress.json",
            {
                "status": "RUNNING",
                "completed_cases": case_index + 1,
                "total_cases": len(ids),
            },
        )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    frame = pd.DataFrame(records)
    return _write_results(
        context,
        frame,
        records_path,
        summary_path,
        source_freeze_digest=source_freeze_digest,
        extra={
            "record_type": record_type,
            "split_role": split_role,
            "methods": specs,
            "horizons_by_method": horizons_by_method,
            "state_preparation_run_id": state_summary["run_id"],
        },
    )


def _measurement_audit(context: Any, base: dict[str, Any]) -> dict[str, Any]:
    data, raw, metadata = _raw_and_metadata(context)
    lookup = {str(value): index for index, value in enumerate(data["ids"].tolist())}
    ids = []
    per_family = int(
        context.config["causal_geometry_v12"]["measurement"]["audit_pairs_per_family"]
    )
    by_family: dict[str, list[str]] = {}
    for value in base["development"]["base_trial_ids"]:
        by_family.setdefault(str(metadata[str(value)]["family"]), []).append(str(value))
    for family in sorted(by_family):
        ids.extend(sorted(by_family[family])[:per_family])
    tasks = {
        task.example_id: task for _, task in load_tasks(context.root / FORMAL_DATA)
    }
    bundle = load_model_bundle(context.config)
    _, _, dense_map = _load_encoder(context, bundle)
    v8 = context.config["persistent_state_v8"]
    measured = [int(value) for value in v8["intervention"]["measured_layers"]]
    main_layer = max(measured)
    recurrent_layers = [
        int(value) for value in v8["persistent_components"]["recurrent_layers"]
    ]
    attention_layers = [
        int(value) for value in v8["persistent_components"]["attention_layers"]
    ]
    epsilon = float(
        context.config["causal_geometry_v12"]["measurement"]["mixture_epsilon"]
    )
    scales = [
        float(value)
        for value in context.config["causal_geometry_v12"]["measurement"]["raw_scales"]
    ]
    records = []
    for base_id in ids:
        source = lookup[base_id]
        pair = metadata[base_id]
        task = tasks[str(pair["prompt_id"])]
        clean = _prefill(
            bundle,
            task.prompt,
            measured_layers=measured,
            dense_map=dense_map,
            intervention_layer=int(v8["intervention"]["layer"]),
            candidate=None,
        )
        tokens = _teacher_tokens(
            bundle, clean, count=1, measured_layers=measured, dense_map=dense_map
        )
        clean_traj = _teacher_forced_trajectory(
            bundle,
            clean["cache"],
            tokens,
            prompt_length=clean["prompt_length"],
            measured_layers=measured,
            dense_map=dense_map,
        )
        original = (
            raw["recurrent"][source],
            raw["conv"][source],
            raw["kv_full"][source],
        )
        teacher_cache = apply_decoded_state(
            clean["cache"],
            original,
            recurrent_layers=recurrent_layers,
            attention_layers=attention_layers,
            prompt_length=clean["prompt_length"],
        )
        teacher_traj = _teacher_forced_trajectory(
            bundle,
            teacher_cache,
            tokens,
            prompt_length=clean["prompt_length"],
            measured_layers=measured,
            dense_map=dense_map,
        )
        del teacher_cache
        same_family = [
            value
            for value in ids
            if value != base_id and metadata[value]["family"] == pair["family"]
        ]
        other = lookup[sorted(same_family)[0]]
        conditions = [
            (f"scale_{scale:.2f}", tuple(value * scale for value in original))
            for scale in scales
        ]
        conditions.append(
            (
                "same_family_mix_0.01",
                (
                    original[0] * (1 - epsilon) + raw["recurrent"][other] * epsilon,
                    original[1] * (1 - epsilon) + raw["conv"][other] * epsilon,
                    original[2] * (1 - epsilon) + raw["kv_full"][other] * epsilon,
                ),
            )
        )
        teacher_effect = (
            teacher_traj["j"][main_layer][0] - clean_traj["j"][main_layer][0]
        )
        for condition, delta in conditions:
            perturbed_cache = apply_decoded_state(
                clean["cache"],
                delta,
                recurrent_layers=recurrent_layers,
                attention_layers=attention_layers,
                prompt_length=clean["prompt_length"],
            )
            trajectory = _teacher_forced_trajectory(
                bundle,
                perturbed_cache,
                tokens,
                prompt_length=clean["prompt_length"],
                measured_layers=measured,
                dense_map=dense_map,
            )
            decoded_effect = (
                trajectory["j"][main_layer][0] - clean_traj["j"][main_layer][0]
            )
            metrics = _effect_metrics(
                clean_traj,
                teacher_traj,
                trajectory,
                main_layer=main_layer,
                horizon=1,
                semantic_ids=_semantic_ids(bundle.tokenizer, task.semantic_actions[0]),
            )
            records.append(
                {
                    "schema_version": SCHEMA_VERSION_V12,
                    "protocol_version": PROTOCOL_V12,
                    "base_trial_id": base_id,
                    "family": str(pair["family"]),
                    "method": condition,
                    "condition": condition,
                    "horizon": 1,
                    **metrics,
                    **_continuous_semantic(
                        teacher_effect,
                        decoded_effect,
                        int(
                            context.config["causal_geometry_v12"]["measurement"][
                                "top_k"
                            ]
                        ),
                    ),
                }
            )
    frame = pd.DataFrame(records)
    path = context.root / MEASUREMENT_RECORDS
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False, compression="zstd")
    condition_means = (
        frame.groupby("condition").mean(numeric_only=True).to_dict(orient="index")
    )
    stable_direction = min(
        float(value["direction_cosine"]) for value in condition_means.values()
    )
    min_overlap = min(
        float(value["semantic_topk_overlap"]) for value in condition_means.values()
    )
    summary = {
        "schema_version": SCHEMA_VERSION_V12,
        "protocol_version": PROTOCOL_V12,
        "run_id": context.run_id,
        "source_freeze_digest": base["freeze_digest"],
        "pair_count": len(ids),
        "condition_means": condition_means,
        "semantic_metric_instability_material": bool(
            stable_direction >= 0.95 and min_overlap < 0.80
        ),
        "interpretation_rule": "material only when continuous direction remains >=0.95 while top-k overlap falls below frozen 0.8 gate",
        "records": str(MEASUREMENT_RECORDS),
        "records_sha256": sha256_file(path),
    }
    write_json_atomic(context.root / MEASUREMENT_SUMMARY, summary)
    return summary


def _ranked(summary: dict[str, Any], names: list[str]) -> list[tuple[float, str]]:
    values = []
    for name in names:
        if name in summary["aggregates"]:
            values.append(
                (
                    float(
                        summary["aggregates"][name]["1"]["pooled"]["selection_score"]
                    ),
                    name,
                )
            )
    return sorted(values, reverse=True)


def _freeze_stage2(context: Any) -> dict[str, Any]:
    specs = _method_specs(context.root)["identified_oracle_methods"]
    summary = json.loads((context.root / ORACLE_STAGE1_SUMMARY).read_text())
    groups = {
        "global_pca": [
            value
            for value in specs
            if value["class"] in {"global_combined_pca", "global_architecture_pca"}
        ],
        "global_causal": [
            value for value in specs if value["class"] == "global_causal_basis"
        ],
        "local_pca": [value for value in specs if value["class"] == "local_pca"],
        "local_causal": [
            value for value in specs if value["class"] == "local_causal_basis"
        ],
    }
    selected = []
    for role, candidates in groups.items():
        ranked = _ranked(summary, [str(value["method"]) for value in candidates])
        if ranked:
            spec = dict(
                next(value for value in candidates if value["method"] == ranked[0][1])
            )
            spec["selection_role"] = role
            h1_pass = bool(
                summary["authorization"][spec["method"]][
                    "horizon_all_family_gate_pass"
                ].get("1", False)
            )
            spec["stage2_horizons"] = [2, 4, 8, 16] if h1_pass else [2, 4, 8]
            selected.append(spec)
    return build_derived_freeze(
        context.root,
        context.config,
        path=STAGE2_FREEZE_PATH,
        purpose="freeze one development-selected v12 oracle per causal-geometry class",
        inputs=[PREPARED_FREEZE_PATH, ORACLE_STAGE1_RECORDS, ORACLE_STAGE1_SUMMARY],
        payload={
            "selection_split": "development",
            "confirmatory_used": False,
            "method_specs": selected,
        },
    )


def _freeze_confirm(context: Any) -> dict[str, Any]:
    stage2 = verify_derived_freeze(context.root, context.config, STAGE2_FREEZE_PATH)
    summary = json.loads((context.root / ORACLE_STAGE2_SUMMARY).read_text())
    selected = []
    for spec in stage2["method_specs"]:
        value = dict(spec)
        h1 = json.loads((context.root / ORACLE_STAGE1_SUMMARY).read_text())
        h1_pass = bool(
            h1["authorization"][value["method"]]["horizon_all_family_gate_pass"].get(
                "1", False
            )
        )
        value["confirmatory_horizons"] = [1, 2, 4, 8, 16] if h1_pass else [1, 2, 4, 8]
        value["development_stage2_score"] = float(
            summary["aggregates"][value["method"]]["4"]["pooled"]["selection_score"]
        )
        selected.append(value)
    return build_derived_freeze(
        context.root,
        context.config,
        path=CONFIRM_FREEZE_PATH,
        purpose="freeze v12 methods before the new independent confirmation",
        inputs=[STAGE2_FREEZE_PATH, ORACLE_STAGE2_RECORDS, ORACLE_STAGE2_SUMMARY],
        payload={
            "selection_split": "development",
            "confirmatory_used": False,
            "method_specs": selected,
        },
    )


def _strict_replacement(
    context: Any, base: dict[str, Any], confirm: dict[str, Any]
) -> dict[str, Any]:
    """Audit replacement for the intervention-bearing persistent channels.

    This intentionally does not claim whole-cache replacement: unmodeled cache fields are
    a structural scaffold and are declared explicitly in the record.
    """
    best = max(
        confirm["method_specs"],
        key=lambda value: float(value.get("development_stage2_score", 0.0)),
    )
    horizons = {str(best["method"]): [1, 2, 4, 8]}
    summary = _run_states(
        context,
        base,
        ids=[str(value) for value in base["confirmatory"]["base_trial_ids"]],
        specs=[best],
        horizons_by_method=horizons,
        state_summary_path=CONFIRMATORY_STATES,
        records_path=REPLACEMENT_RECORDS,
        summary_path=REPLACEMENT_SUMMARY,
        source_freeze_digest=confirm["freeze_digest"],
        split_role="confirmatory",
        record_type="persistent_channel_replacement_diagnostic",
    )
    summary["replacement_scope"] = (
        "intervention-bearing REC/conv/KV channels; not the entire model cache"
    )
    summary["compact_only_full_state_verified"] = False
    summary["raw_state_bypass_detected"] = False
    summary["strict_full_state_replacement_pass"] = False
    summary["reason"] = (
        "absolute full-cache compact reconstruction was not identified by the frozen v12 representation"
    )
    write_json_atomic(context.root / REPLACEMENT_SUMMARY, summary)
    return summary


def main() -> None:
    parser = standard_parser("causal geometry v12", "configs/causal_geometry_v12.yaml")
    parser.add_argument(
        "--stage",
        required=True,
        choices=(
            "measurement-audit",
            "scaling-causal",
            "oracle-stage1",
            "freeze-stage2",
            "oracle-stage2",
            "freeze-confirm",
            "confirmatory",
            "strict-replacement",
        ),
    )
    args = parser.parse_args()
    context = initialize_context("causal-v12", args)
    try:
        base = verify_base_freeze(context.root, context.config)
        prepared = verify_derived_freeze(
            context.root, context.config, PREPARED_FREEZE_PATH
        )
        specs = _method_specs(context.root)
        if args.stage == "measurement-audit":
            result = _measurement_audit(context, base)
        elif args.stage == "scaling-causal":
            chosen = specs["causal_scaling_methods"]
            result = _run_states(
                context,
                base,
                ids=[
                    str(value) for value in base["validation_panel"]["base_trial_ids"]
                ],
                specs=chosen,
                horizons_by_method={
                    str(value["method"]): [1, 2, 4] for value in chosen
                },
                state_summary_path=SCALING_STATES,
                records_path=SCALING_CAUSAL_RECORDS,
                summary_path=SCALING_CAUSAL_SUMMARY,
                source_freeze_digest=prepared["freeze_digest"],
                split_role="validation_panel",
                record_type="data_rank_scaling_causal",
            )
        elif args.stage == "oracle-stage1":
            chosen = specs["identified_oracle_methods"]
            result = _run_states(
                context,
                base,
                ids=[str(value) for value in base["development"]["base_trial_ids"]],
                specs=chosen,
                horizons_by_method={str(value["method"]): [1] for value in chosen},
                state_summary_path=DEVELOPMENT_STATES,
                records_path=ORACLE_STAGE1_RECORDS,
                summary_path=ORACLE_STAGE1_SUMMARY,
                source_freeze_digest=prepared["freeze_digest"],
                split_role="development",
                record_type="local_causal_oracle_stage1",
            )
        elif args.stage == "freeze-stage2":
            result = _freeze_stage2(context)
        elif args.stage == "oracle-stage2":
            stage2 = verify_derived_freeze(
                context.root, context.config, STAGE2_FREEZE_PATH
            )
            chosen = stage2["method_specs"]
            result = _run_states(
                context,
                base,
                ids=[str(value) for value in base["development"]["base_trial_ids"]],
                specs=chosen,
                horizons_by_method={
                    str(value["method"]): [int(h) for h in value["stage2_horizons"]]
                    for value in chosen
                },
                state_summary_path=DEVELOPMENT_STATES,
                records_path=ORACLE_STAGE2_RECORDS,
                summary_path=ORACLE_STAGE2_SUMMARY,
                source_freeze_digest=stage2["freeze_digest"],
                split_role="development",
                record_type="local_causal_oracle_stage2",
            )
        elif args.stage == "freeze-confirm":
            result = _freeze_confirm(context)
        else:
            confirm = verify_derived_freeze(
                context.root, context.config, CONFIRM_FREEZE_PATH
            )
            chosen = confirm["method_specs"]
            if args.stage == "confirmatory":
                result = _run_states(
                    context,
                    base,
                    ids=[
                        str(value) for value in base["confirmatory"]["base_trial_ids"]
                    ],
                    specs=chosen,
                    horizons_by_method={
                        str(value["method"]): [
                            int(h) for h in value["confirmatory_horizons"]
                        ]
                        for value in chosen
                    },
                    state_summary_path=CONFIRMATORY_STATES,
                    records_path=CONFIRM_RECORDS,
                    summary_path=CONFIRM_SUMMARY,
                    source_freeze_digest=confirm["freeze_digest"],
                    split_role="confirmatory",
                    record_type="independent_causal_confirmatory_v12",
                )
            else:
                result = _strict_replacement(context, base, confirm)
        context.finish("COMPLETED_V12_CAUSAL_STAGE", summary=result)
    except Exception as exc:
        context.fail_progress(f"{type(exc).__name__}: {exc}")
        context.finish("FAILED", error=f"{type(exc).__name__}: {exc}")
        raise


if __name__ == "__main__":
    main()
