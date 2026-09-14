"""Persistent measured-J mediation after the protocol-v4 single-arm gate."""

from __future__ import annotations

import json
from functools import partial
from typing import Any

import numpy as np
import pandas as pd
import torch

from jclosure.clamp_v3 import project_dense_candidate
from jclosure.experiments.causal_single_v4 import (
    _load_bank,
    _load_encoder,
    _load_tasks,
    _replacement,
    _rollout,
    _thresholds,
)
from jclosure.experiments.common import initialize_context, standard_parser
from jclosure.experiments.geometry_v3 import NaturalityModel
from jclosure.model import load_model_bundle
from jclosure.protocol_v4 import verify_program_freeze
from jclosure.provenance import append_jsonl, write_json_atomic
from jclosure.recorder import ActivationRecorder, ResidualEditor
from jclosure.runtime_v3_1 import encode_direct_prompt
from jclosure.single_arm_v4 import (
    aligned_rollout_metrics,
    construct_from_shared_basis,
    shared_low_singular_basis,
)
from jclosure.statistics import clustered_bootstrap_ci

PROTOCOL = "persistent_mediation_v4"


@torch.no_grad()
def _clean_activations(
    bundle: Any, prompt: str, layers: list[int]
) -> dict[int, torch.Tensor]:
    input_ids = encode_direct_prompt(bundle, prompt)
    with ActivationRecorder(bundle.layers, at=layers) as recorder:
        bundle.hf_model(input_ids=input_ids, use_cache=False)
    return {
        layer: recorder.activations[layer][0].detach().float() for layer in layers
    }


def _restoration_transform(
    activation: torch.Tensor,
    layer: int,
    *,
    clean: torch.Tensor,
    dense_map: Any,
    mode: str,
    capture: dict[int, dict[str, Any]],
    thresholds: Any,
) -> torch.Tensor:
    output = activation.clone().float()
    positions = (
        (int(output.shape[1]) - 1,)
        if mode == "persistent_final"
        else tuple(range(int(output.shape[1])))
    )
    events = []
    for position in positions:
        reference = clean[position].to(output.device).float()
        current = output[0, position].float()
        observed = float(torch.linalg.vector_norm(current - reference).item())
        if observed <= 1e-7:
            restored = current
            construction = {
                "status": "IDENTITY_NO_CHANGE",
                "failure_reason": None,
                "basis_dimension": None,
            }
            events.append(
                {
                    "layer": layer,
                    "position": position,
                    "observed_displacement": observed,
                    "correction_l2": 0.0,
                    "dense_cosine": 1.0,
                    "top10_overlap": 1.0,
                    "rms_drift": 0.0,
                    "construction_status": construction["status"],
                    "construction_failure_reason": None,
                    "passed": True,
                }
            )
            continue
        else:
            restored, construction = project_dense_candidate(
                reference,
                current,
                layer=layer,
                dense_map=dense_map,
                relative_tolerance=1e-4,
                optimized=True,
                naturality=None,
                thresholds=thresholds,
            )
            output[0, position] = restored.to(output.device, output.dtype)
        clean_state = dense_map.dense_state(reference, layer)
        restored_state = dense_map.dense_state(restored.float(), layer)
        clean_raw = dense_map.raw_scores(reference, layer)
        restored_raw = dense_map.raw_scores(restored.float(), layer)
        dense_cosine = float(
            torch.nn.functional.cosine_similarity(
                clean_state[None], restored_state[None]
            ).item()
        )
        top_clean = set(torch.topk(clean_raw, 10).indices.tolist())
        top_restored = set(torch.topk(restored_raw, 10).indices.tolist())
        reference_rms = torch.sqrt(torch.mean(reference.square())).clamp_min(1e-20)
        restored_rms = torch.sqrt(torch.mean(restored.float().square()))
        rms_drift = float(torch.abs(restored_rms - reference_rms) / reference_rms)
        events.append(
            {
                "layer": layer,
                "position": position,
                "observed_displacement": observed,
                "correction_l2": float(
                    torch.linalg.vector_norm(restored.float() - current).item()
                ),
                "dense_cosine": dense_cosine,
                "top10_overlap": len(top_clean & top_restored) / 10,
                "rms_drift": rms_drift,
                "construction_status": construction["status"],
                "construction_failure_reason": construction["failure_reason"],
                "passed": bool(
                    dense_cosine >= thresholds.dense_cosine
                    and len(top_clean & top_restored) / 10
                    >= thresholds.dense_top10_overlap
                    and rms_drift <= thresholds.rms_drift
                    and torch.isfinite(restored).all()
                ),
            }
        )
    capture[layer] = {
        "mode": mode,
        "events": events,
        "passed": bool(events and all(value["passed"] for value in events)),
    }
    return output.to(activation.dtype)


@torch.no_grad()
def _persistent_rollout(
    bundle: Any,
    task: Any,
    *,
    layer: int,
    workspace_layers: list[int],
    dense_map: Any,
    candidate: torch.Tensor,
    mode: str,
    clean_by_layer: dict[int, torch.Tensor],
    thresholds: Any,
) -> tuple[dict[str, Any], dict[int, dict[str, Any]]]:
    capture: dict[int, dict[str, Any]] = {}
    transforms: dict[int, Any] = {layer: _replacement(candidate)}
    for restore_layer in workspace_layers[1:]:
        transforms[restore_layer] = partial(
            _restoration_transform,
            clean=clean_by_layer[restore_layer],
            dense_map=dense_map,
            mode=mode,
            capture=capture,
            thresholds=thresholds,
        )
    # Reuse the audited continuation implementation by temporarily supplying all
    # layer transforms for the initial cached forward, then perform the same loop.
    input_ids = encode_direct_prompt(bundle, task.prompt)
    actions: list[str] = []
    logits_values: list[np.ndarray] = []
    j_states: list[np.ndarray] = []
    target_odds: list[float] = []
    layer_states: dict[str, np.ndarray] = {}
    from jclosure.experiments.causal_single_v4 import (  # local to avoid API growth
        _semantic_token,
        _token_ids,
    )
    from jclosure.single_arm_v4 import multiple_token_log_odds

    with (
        ResidualEditor(bundle.layers, transforms),
        ActivationRecorder(bundle.layers, at=workspace_layers) as recorder,
    ):
        outputs = bundle.hf_model(input_ids=input_ids, use_cache=True)
    logits = outputs.logits[0, -1].detach().float()
    past_key_values = outputs.past_key_values
    if past_key_values is None:
        raise RuntimeError("model did not return a KV cache for mediation rollout")
    current_profiles = {
        current_layer: dense_map.dense_state(
            recorder.activations[current_layer][0, -1].detach().float(), current_layer
        )
        for current_layer in workspace_layers
    }
    generated_count = 0
    maximum_tokens = 2 * task.horizon + max(8, round(0.25 * task.horizon))
    error = None
    while generated_count < maximum_tokens and len(actions) < task.horizon:
        token = int(torch.argmax(logits))
        action, action_error = _semantic_token(bundle.tokenizer, token, task)
        if action_error is not None:
            error = action_error
            break
        if action is not None:
            step_index = len(actions)
            j_states.append(
                current_profiles[layer].detach().cpu().numpy().astype(np.float16)
            )
            logits_values.append(logits.detach().cpu().numpy().astype(np.float16))
            target_odds.append(
                multiple_token_log_odds(
                    logits,
                    _token_ids(bundle.tokenizer, task.semantic_actions[step_index]),
                )
            )
            if step_index == 0:
                layer_states = {
                    str(key): value.detach().cpu().numpy().astype(np.float16)
                    for key, value in current_profiles.items()
                }
            actions.append(action)
        generated_count += 1
        next_token = torch.tensor([[token]], device=input_ids.device)
        attention_mask = torch.ones(
            (1, int(input_ids.shape[1]) + generated_count),
            dtype=torch.long,
            device=input_ids.device,
        )
        with ActivationRecorder(bundle.layers, at=workspace_layers) as recorder:
            continuation = bundle.hf_model(
                input_ids=next_token,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
            )
        logits = continuation.logits[0, -1].detach().float()
        past_key_values = continuation.past_key_values
        current_profiles = {
            current_layer: dense_map.dense_state(
                recorder.activations[current_layer][0, -1].detach().float(),
                current_layer,
            )
            for current_layer in workspace_layers
        }
    return (
        {
            "actions": actions,
            "expected_actions": list(task.semantic_actions),
            "parseable": len(actions) == task.horizon and error is None,
            "error": error,
            "logits": logits_values,
            "j_states": j_states,
            "target_log_odds": target_odds,
            "within_forward_j_states": layer_states,
        },
        capture,
    )


def _bootstrap_ratio(
    frame: pd.DataFrame, numerator: str, *, seed: int, n_resamples: int
) -> dict[str, Any]:
    pivot = frame.pivot(index="prompt_id", columns="mode", values="output_js_divergence")
    pivot = pivot.dropna(subset=["single", numerator])
    clusters = pivot.index.to_numpy()
    generator = np.random.default_rng(seed)
    values = np.empty(n_resamples, dtype=np.float64)
    for index in range(n_resamples):
        sampled = generator.choice(clusters, size=len(clusters), replace=True)
        current = pivot.loc[sampled]
        denominator = float(current["single"].mean())
        values[index] = 1 - float(current[numerator].mean()) / max(denominator, 1e-20)
    estimate = 1 - float(pivot[numerator].mean()) / max(
        float(pivot["single"].mean()), 1e-20
    )
    return {
        "estimate": estimate,
        "lower": float(np.quantile(values, 0.025)),
        "upper": float(np.quantile(values, 0.975)),
        "n_clusters": int(len(pivot)),
        "n_resamples": n_resamples,
    }


def _run(
    context: Any,
    bundle: Any,
    freeze: dict[str, Any],
    limit: int | None,
) -> list[dict[str, Any]]:
    source = json.loads(
        (context.processed_dir / "single_arm_v4.json").read_text(encoding="utf-8")
    )
    if not source["persistent_mediation_authorized"]:
        raise RuntimeError("single-arm gate did not authorize mediation")
    source_manifest = context.raw_dir / source["source_run_id"] / "single_arm_trials.jsonl"
    source_rows = [json.loads(line) for line in source_manifest.read_text(encoding="utf-8").splitlines()]
    accepted = {
        value["prompt_id"]: value
        for value in source_rows
        if value.get("record_type") == "causal_trial"
        and value.get("condition") == "j_preserving"
    }
    tasks = {task.example_id: task for task in _load_tasks(context.root, freeze, "causal_test")}
    bank_rows, bank_states = _load_bank(context)
    bank_by_id = {
        value["example_id"]: torch.as_tensor(bank_states[index])
        for index, value in enumerate(bank_rows)
    }
    section = context.config["single_arm_v4"]
    layer = int(section["layer"])
    layers = [layer, *[int(value) for value in section["future_workspace_layers"]]]
    _, encoder, dense_map = _load_encoder(context, bundle)
    naturality = NaturalityModel(128, 10, 0.99).fit(bank_states)
    shared = shared_low_singular_basis(
        dense_map,
        layer,
        relative_tolerance=1e-4,
        device=next(bundle.hf_model.parameters()).device,
    )
    thresholds = _thresholds(context.config)
    rows: list[dict[str, Any]] = []
    selected_items = sorted(accepted.items())
    if limit is not None:
        selected_items = selected_items[: int(limit)]
    for prompt_id, source_row in selected_items:
        task = tasks[prompt_id]
        clean_by_layer = _clean_activations(bundle, task.prompt, layers)
        clean = clean_by_layer[layer][-1]
        donor = bank_by_id[source_row["donor_id"]].to(clean.device).float()
        natural_scale = float(torch.linalg.vector_norm(donor - clean).item())
        candidate, quality, valid = construct_from_shared_basis(
            clean,
            donor,
            shared=shared,
            dense_map=dense_map,
            encoder=encoder,
            natural_scale=natural_scale,
            displacement_fraction=float(section["initial_strength"]),
            thresholds=thresholds,
            naturality=naturality,
        )
        if not valid:
            rows.append(
                {
                    "schema_version": 6,
                    "protocol_version": PROTOCOL,
                    "record_type": "attrition",
                    "prompt_id": prompt_id,
                    "exclusion_reason": "reconstructed_initial_candidate_invalid",
                    "quality": quality,
                }
            )
            continue
        clean_rollout = _rollout(
            bundle,
            task,
            layer=layer,
            workspace_layers=layers,
            dense_map=dense_map,
            candidate=None,
        )
        for mode in ("single", "persistent_final", "persistent_all"):
            if mode == "single":
                rollout = _rollout(
                    bundle,
                    task,
                    layer=layer,
                    workspace_layers=layers,
                    dense_map=dense_map,
                    candidate=candidate,
                )
                capture: dict[int, dict[str, Any]] = {}
                restoration_valid = True
            else:
                rollout, capture = _persistent_rollout(
                    bundle,
                    task,
                    layer=layer,
                    workspace_layers=layers,
                    dense_map=dense_map,
                    candidate=candidate,
                    mode=mode,
                    clean_by_layer=clean_by_layer,
                    thresholds=thresholds,
                )
                restoration_valid = bool(
                    len(capture) == len(layers) - 1
                    and all(value["passed"] for value in capture.values())
                )
            metrics = aligned_rollout_metrics(clean_rollout, rollout)
            rows.append(
                {
                    "schema_version": 6,
                    "protocol_version": PROTOCOL,
                    "record_type": "mediation_trial",
                    "run_id": context.run_id,
                    "base_trial_id": source_row["base_trial_id"],
                    "prompt_id": prompt_id,
                    "family": task.family,
                    "horizon": task.horizon,
                    "mode": mode,
                    "valid": restoration_valid,
                    "exclusion_reason": None
                    if restoration_valid
                    else "runtime_restoration_invalid",
                    "initial_quality": quality,
                    "metrics": metrics,
                    "restoration_events": [capture[key] for key in sorted(capture)],
                    "hook_execution_map": [
                        [layer, "final", "initial_perturbation"],
                        *[
                            [
                                restore_layer,
                                "final" if mode == "persistent_final" else "all_non_padding",
                                "measured_j_restoration",
                            ]
                            for restore_layer in layers[1:]
                        ],
                    ],
                }
            )
    return rows


def _merge(context: Any) -> dict[str, Any]:
    manifests = []
    for path in context.raw_dir.glob("mediation-v4-*/manifest.json"):
        value = json.loads(path.read_text(encoding="utf-8"))
        if value.get("status") == "COMPLETED_TRIALS":
            manifests.append(value)
    if not manifests:
        raise RuntimeError("no completed mediation-v4 trial run")
    source = sorted(manifests, key=lambda value: value["run_id"])[-1]
    path = context.root / source["records"]
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    frame = pd.DataFrame(rows)
    trials = frame[(frame["record_type"] == "mediation_trial") & frame["valid"]].copy()
    trials["output_js_divergence"] = trials["metrics"].map(
        lambda value: value["output_js_divergence"]
    )
    trials["future_j_trajectory_divergence"] = trials["metrics"].map(
        lambda value: value["future_j_trajectory_divergence"]
    )
    common_ids = [
        base_id
        for base_id, group in trials.groupby("base_trial_id")
        if set(group["mode"]) == {"single", "persistent_final", "persistent_all"}
    ]
    common = trials[trials["base_trial_id"].isin(common_ids)].copy()
    effects: dict[str, Any] = {}
    for metric in ("output_js_divergence", "future_j_trajectory_divergence"):
        effects[metric] = {}
        for mode, group in common.groupby("mode", sort=True):
            effects[metric][mode] = clustered_bootstrap_ci(
                group,
                cluster_col="prompt_id",
                value_col=metric,
                n_resamples=10000,
                seed=int(context.config["reproducibility"]["bootstrap_seed"]),
            ).__dict__
    ratios = {
        "M_final": _bootstrap_ratio(
            common,
            "persistent_final",
            seed=int(context.config["reproducibility"]["bootstrap_seed"]),
            n_resamples=10000,
        ),
        "M_all": _bootstrap_ratio(
            common,
            "persistent_all",
            seed=int(context.config["reproducibility"]["bootstrap_seed"]) + 1,
            n_resamples=10000,
        ),
    }
    parquet = context.processed_dir / "persistent_mediation_v4.parquet"
    serialized = frame.copy()
    for column in ("metrics", "initial_quality", "restoration_events", "hook_execution_map"):
        if column in serialized:
            serialized[column] = serialized[column].map(
                lambda value: json.dumps(value, sort_keys=True)
                if isinstance(value, dict | list)
                else value
            )
    serialized.to_parquet(parquet, index=False, compression="zstd")
    summary = {
        "schema_version": 6,
        "protocol_version": PROTOCOL,
        "run_id": context.run_id,
        "source_run_id": source["run_id"],
        "complete_paired_base_trials": len(common_ids),
        "valid_records": int(len(trials)),
        "effects": effects,
        "mediation_ratios": ratios,
        "records": str(parquet.relative_to(context.root)),
    }
    output = context.processed_dir / "persistent_mediation_v4.json"
    write_json_atomic(output, summary)
    return summary


def main() -> None:
    parser = standard_parser(
        "Run gated protocol-v4 persistent mediation", "configs/predictive_v4.yaml"
    )
    parser.add_argument("--stage", choices=("run", "merge"), required=True)
    args = parser.parse_args()
    context = initialize_context("mediation-v4", args)
    try:
        freeze = verify_program_freeze(context.root, context.config)
        if args.dry_run:
            context.finish("DRY_RUN", stage=args.stage)
            return
        if args.stage == "run":
            bundle = load_model_bundle(context.config)
            rows = _run(context, bundle, freeze, args.limit)
            path = context.raw_dir / context.run_id / "mediation_trials.jsonl"
            append_jsonl(path, rows)
            context.finish(
                "COMPLETED_TRIALS",
                records=str(path.relative_to(context.root)),
                record_count=len(rows),
            )
        else:
            summary = _merge(context)
            context.finish("COMPLETED", summary=summary)
    except KeyboardInterrupt:
        context.finish("FAILED", error="KeyboardInterrupt")
        raise
    except Exception as exc:
        context.finish("FAILED", error=f"{type(exc).__name__}: {exc}")
        raise


if __name__ == "__main__":
    main()
