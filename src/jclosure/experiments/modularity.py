"""Phase 7: separate controller reasoning, supplied facts, and surface rendering."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pandas as pd
import torch

from jclosure.experiments.common import (
    initialize_context,
    require_closure_eligible_layers,
    require_phase0_gate,
    require_phase0_v2_gate,
    standard_parser,
)
from jclosure.experiments.distill_controller import (
    build_budgeted_controller,
    evaluate_controller,
)
from jclosure.experiments.memory_order import TraceTensors
from jclosure.provenance import sha256_file, write_json_atomic


def fact_condition_features(
    features: torch.Tensor, condition: str, *, seed: int
) -> torch.Tensor:
    """Create exact/no/shuffled/distractor fact controls without altering inputs in place."""

    if features.shape[-1] % 2:
        raise ValueError("expected concatenated procedural/fact features")
    output = features.clone()
    boundary = features.shape[-1] // 2
    facts = features[:, boundary:]
    if condition == "exact_relevant":
        return output
    if condition == "no_fact":
        output[:, boundary:] = 0
        return output
    generator = torch.Generator(device="cpu").manual_seed(seed)
    if condition == "shuffled_fact":
        order = torch.randperm(len(features), generator=generator)
        output[:, boundary:] = facts[order]
        return output
    if condition == "distractor_fact":
        coordinate_order = torch.randperm(facts.shape[-1], generator=generator)
        output[:, boundary:] = facts[:, coordinate_order]
        return output
    raise ValueError(f"unknown fact condition: {condition}")


def _find_artifact(root: Path, explicit: str | None, pattern: str) -> Path:
    if explicit:
        path = Path(explicit)
        return path if path.is_absolute() else root / path
    paths = sorted(root.glob(pattern))
    if not paths:
        raise RuntimeError(f"missing required artifact matching {pattern}")
    return paths[-1]


def main() -> None:
    parser = standard_parser("Evaluate reasoning/knowledge/language modularity", "configs/confirm.yaml")
    parser.add_argument("--trace")
    parser.add_argument("--controllers")
    args = parser.parse_args()
    context = initialize_context("modularity", args)
    try:
        if args.dry_run:
            context.finish("DRY_RUN")
            return
        if (
            context.config.get("run", {}).get("phase0_protocol")
            == "phase0_protocol_v2"
        ):
            require_phase0_v2_gate(context)
            require_closure_eligible_layers(context)
        else:
            require_phase0_gate(context)
        trace_path = _find_artifact(
            context.root,
            args.trace,
            "artifacts/traces/teacher_*.pt",
        )
        controller_path = _find_artifact(
            context.root,
            args.controllers,
            "results/raw/distill-controller-*/controllers.json",
        )
        traces: TraceTensors = torch.load(trace_path, map_location="cpu", weights_only=False)
        controller_records = json.loads(controller_path.read_text(encoding="utf-8"))["records"]
        # Evaluate every budget-valid controller; reporting can select the smallest stable one.
        device = torch.device(f"cuda:{context.config['model'].get('device', 0)}" if torch.cuda.is_available() else "cpu")
        records: list[dict[str, Any]] = []
        original_features = traces.input_features
        for item in controller_records:
            if not item["budget_valid"]:
                continue
            model = build_budgeted_controller(
                item["family"],
                state_dim=traces.states.shape[-1],
                n_layers=traces.states.shape[1],
                n_answers=len(traces.answer_tokens),
                feature_dim=traces.input_features.shape[-1],
                budget=int(item["target_parameter_count"]),
            )
            checkpoint = context.root / item["checkpoint"]["path"]
            if sha256_file(checkpoint) != item["checkpoint"]["sha256"]:
                raise RuntimeError(f"checkpoint hash mismatch: {checkpoint}")
            model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
            for condition in ("no_fact", "exact_relevant", "shuffled_fact", "distractor_fact"):
                conditioned = replace(
                    traces,
                    input_features=fact_condition_features(
                        original_features, condition, seed=int(item["seed"])
                    ),
                )
                for standalone in (False, True):
                    metrics = evaluate_controller(
                        model,
                        conditioned,
                        split="test",
                        device=device,
                        standalone=standalone,
                    )
                    records.append(
                        {
                            "schema_version": 1,
                            "run_id": context.run_id,
                            "controller_family": item["family"],
                            "parameter_count": item["parameter_count"],
                            "seed": item["seed"],
                            "fact_condition": condition,
                            "initialization": "standalone" if standalone else "true_j0",
                            "metrics": metrics,
                            "claim_scope": "constructive modular implementation only",
                        }
                    )
            del model
        # Group C is a deterministic mapping from semantic class to token ID; it is not a language model.
        renderer = {
            str(class_id): {"token_id": token_id, "surface": f"token:{token_id}"}
            for class_id, token_id in enumerate(traces.answer_tokens)
        }
        raw_path = context.raw_dir / context.run_id / "modularity.json"
        write_json_atomic(
            raw_path,
            {
                "schema_version": 1,
                "run_id": context.run_id,
                "records": records,
                "surface_renderer": renderer,
                "surface_renderer_is_deterministic": True,
            },
        )
        pd.json_normalize(records, sep=".").to_parquet(
            context.processed_dir / f"modularity_{context.run_id}.parquet", index=False
        )
        context.finish(
            "COMPLETED",
            evaluations=len(records),
            trace_sha256=sha256_file(trace_path),
            controller_records_sha256=sha256_file(controller_path),
        )
    except KeyboardInterrupt:
        context.finish("FAILED", error="KeyboardInterrupt: run cancelled")
        raise
    except Exception as exc:
        context.finish("FAILED", error=f"{type(exc).__name__}: {exc}")
        raise


if __name__ == "__main__":
    main()
