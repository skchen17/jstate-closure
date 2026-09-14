"""Extract teacher-correct token-time measured-J and full-state traces for v4."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from jclosure.datasets_v4 import ProgramTraceTask, terminal_environment_prompt
from jclosure.experiments.common import initialize_context, standard_parser
from jclosure.experiments.teacher_tasks_v4 import _padded_inputs
from jclosure.geometry import DenseJMap
from jclosure.jstate import ConceptVocabulary, JStateEncoder
from jclosure.model import load_model_bundle
from jclosure.protocol_v4 import verify_program_freeze
from jclosure.provenance import sha256_file, write_json_atomic
from jclosure.recorder import ActivationRecorder

PROTOCOL = "predictive_state_v4"
ACTION_SURFACES = tuple(str(value) for value in range(10)) + tuple("ABCDEF")
ACTION_TO_ID = {value: index for index, value in enumerate(ACTION_SURFACES)}


def _tasks(
    root: Path, freeze: dict[str, Any], domain: str
) -> dict[str, ProgramTraceTask]:
    payload = json.loads(
        (root / freeze["formal_data"][domain]).read_text(encoding="utf-8")
    )
    output = {}
    for raw in payload["items"]:
        item = dict(raw)
        item["semantic_actions"] = tuple(item["semantic_actions"])
        task = ProgramTraceTask(**item)
        output[task.example_id] = task
    return output


def _teacher_frame(context: Any) -> pd.DataFrame:
    summary = json.loads(
        (context.processed_dir / "teacher_formal_v4.json").read_text(encoding="utf-8")
    )
    return pd.DataFrame(
        json.loads(line)
        for line in (context.root / summary["records"])
        .read_text(encoding="utf-8")
        .splitlines()
    )


def _encoder(context: Any, bundle: Any) -> tuple[Any, DenseJMap]:
    size = int(context.config["compact_state_v4"]["dictionary_size"])
    vocabulary = ConceptVocabulary.from_json(
        context.root / "results/processed" / f"concept_vocabulary_v2_{size}.json"
    )
    encoder = JStateEncoder.from_lens(
        bundle.lens,
        bundle.unembedding_weight,
        vocabulary,
        k=int(context.config["jstate"]["k"]),
        lazy=True,
        protocol_version=PROTOCOL,
        direction_chunk_size=int(
            context.config["jstate"].get("direction_chunk_size", 512)
        ),
    )
    return vocabulary, DenseJMap.from_encoder(encoder)


def _pooled_profiles(
    activations: dict[int, torch.Tensor],
    dense_map: DenseJMap,
    layers: list[int],
) -> tuple[torch.Tensor, torch.Tensor]:
    by_layer = []
    for layer in layers:
        hidden = activations[layer][:, -1].detach().float()
        matrix = dense_map.centered_map(
            layer, device=hidden.device, dtype=torch.float32
        )
        scores = hidden @ matrix.T
        by_layer.append(F.normalize(scores, dim=1))
    stack = torch.stack(by_layer, dim=1)
    pooled = F.normalize(stack.mean(dim=1), dim=1)
    dispersion = (1 - (stack * pooled[:, None]).sum(dim=-1)).mean(dim=1)
    return pooled, dispersion


@torch.no_grad()
def _extract_domain(
    context: Any,
    bundle: Any,
    dense_map: DenseJMap,
    freeze: dict[str, Any],
    teacher: pd.DataFrame,
    domain: str,
) -> dict[str, Any]:
    tasks = _tasks(context.root, freeze, domain)
    selected = teacher[
        (teacher["domain"] == domain)
        & teacher["parseable"].astype(bool)
        & teacher["full_trajectory_correct"].astype(bool)
    ].sort_values("example_id")
    prompts: list[str] = []
    metadata: list[dict[str, Any]] = []
    trajectory_rows = []
    cursor = 0
    for row in selected.itertuples():
        task = tasks[str(row.example_id)]
        step_prompts = [*list(row.step_prompts), terminal_environment_prompt(task)]
        if len(step_prompts) != task.horizon + 1:
            raise RuntimeError(
                "teacher-correct trace has a step-prompt length mismatch"
            )
        start = cursor
        trace_actions = (*task.semantic_actions, task.final_answer)
        for step_index, (prompt, action) in enumerate(
            zip(step_prompts, trace_actions, strict=True)
        ):
            prompts.append(str(prompt))
            metadata.append(
                {
                    "example_id": task.example_id,
                    "family": task.family,
                    "variant": task.variant,
                    "horizon": task.horizon,
                    "step_index": step_index,
                    "action": str(action),
                }
            )
            cursor += 1
        trajectory_rows.append(
            {
                "example_id": task.example_id,
                "family": task.family,
                "variant": task.variant,
                "horizon": task.horizon,
                "start": start,
                "stop": cursor,
                "teacher_correct": True,
            }
        )
    layers = [
        int(value) for value in context.config["compact_state_v4"]["workspace_layers"]
    ]
    batch_size = int(context.config["teacher_tasks_v4"]["generation_batch_size"])
    profiles: list[np.ndarray] = []
    full_states: list[np.ndarray] = []
    dispersions: list[np.ndarray] = []
    for offset in range(0, len(prompts), batch_size):
        batch_prompts = prompts[offset : offset + batch_size]
        input_ids, attention_mask = _padded_inputs(bundle, batch_prompts)
        with ActivationRecorder(bundle.layers, at=layers) as recorder:
            bundle.hf_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
            )
        pooled, dispersion = _pooled_profiles(recorder.activations, dense_map, layers)
        profiles.append(pooled.detach().cpu().numpy().astype(np.float16))
        full_states.append(
            recorder.activations[layers[0]][:, -1]
            .detach()
            .float()
            .cpu()
            .numpy()
            .astype(np.float16)
        )
        dispersions.append(dispersion.detach().cpu().numpy().astype(np.float32))
    actions = np.asarray(
        [ACTION_TO_ID[row["action"]] for row in metadata], dtype=np.int16
    )
    artifact_root = context.root / "artifacts/traces/v4" / context.run_id
    artifact_root.mkdir(parents=True, exist_ok=True)
    tensor_path = artifact_root / f"{domain}.npz"
    np.savez_compressed(
        tensor_path,
        j_profiles=np.concatenate(profiles),
        full_states=np.concatenate(full_states),
        dispersion=np.concatenate(dispersions),
        actions=actions,
    )
    metadata_path = context.raw_dir / context.run_id / f"{domain}_steps.parquet"
    pd.DataFrame(metadata).to_parquet(metadata_path, index=False, compression="zstd")
    trajectory_path = (
        context.raw_dir / context.run_id / f"{domain}_trajectories.parquet"
    )
    pd.DataFrame(trajectory_rows).to_parquet(
        trajectory_path, index=False, compression="zstd"
    )
    return {
        "domain": domain,
        "teacher_correct_trajectories": len(trajectory_rows),
        "steps": len(metadata),
        "tensor_path": str(tensor_path.relative_to(context.root)),
        "tensor_sha256": sha256_file(tensor_path),
        "step_metadata": str(metadata_path.relative_to(context.root)),
        "step_metadata_sha256": sha256_file(metadata_path),
        "trajectories": str(trajectory_path.relative_to(context.root)),
        "trajectories_sha256": sha256_file(trajectory_path),
    }


def main() -> None:
    parser = standard_parser(
        "Extract protocol-v4 teacher-correct traces", "configs/predictive_v4.yaml"
    )
    parser.add_argument(
        "--domain",
        choices=("train", "validation", "causal_test", "rollout_test", "all"),
        default="all",
    )
    args = parser.parse_args()
    context = initialize_context("traces-v4", args)
    try:
        freeze = verify_program_freeze(context.root, context.config)
        if args.dry_run:
            context.finish("DRY_RUN")
            return
        bundle = load_model_bundle(context.config)
        vocabulary, dense_map = _encoder(context, bundle)
        teacher = _teacher_frame(context)
        domains = list(freeze["formal_data"]) if args.domain == "all" else [args.domain]
        records = [
            _extract_domain(context, bundle, dense_map, freeze, teacher, domain)
            for domain in domains
        ]
        summary = {
            "schema_version": 6,
            "protocol_version": PROTOCOL,
            "run_id": context.run_id,
            "dictionary_size": len(vocabulary.token_ids),
            "dictionary_hash": vocabulary.digest,
            "program_freeze_digest": freeze["freeze_digest"],
            "domains": records,
        }
        output = context.processed_dir / "teacher_traces_v4.json"
        write_json_atomic(output, summary)
        context.finish("COMPLETED", summary=str(output.relative_to(context.root)))
    except KeyboardInterrupt:
        context.finish("FAILED", error="KeyboardInterrupt")
        raise
    except Exception as exc:
        context.finish("FAILED", error=f"{type(exc).__name__}: {exc}")
        raise


if __name__ == "__main__":
    main()
