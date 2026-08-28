"""Token-time measured-J macrostate extraction and autonomous predictors."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn

from jclosure.experiments.closure import _task_pool
from jclosure.experiments.common import (
    concept_vocabulary_v2_path,
    initialize_context,
    require_closure_eligible_layers,
    require_phase0_v2_gate,
    standard_parser,
)
from jclosure.jstate import ConceptVocabulary, JStateEncoder
from jclosure.model import load_model_bundle
from jclosure.provenance import append_jsonl, write_json_atomic
from jclosure.recorder import ActivationRecorder


def pool_workspace_band(
    layer_scores: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """T2 macrostate: normalized concept mean plus cross-layer dispersion."""

    if layer_scores.ndim < 2:
        raise ValueError("layer_scores must end in [layers, concepts]")
    mean = F.normalize(layer_scores.float().mean(dim=-2), dim=-1, eps=1e-12)
    dispersion = layer_scores.float().std(dim=-2, unbiased=False).mean(dim=-1)
    return mean, dispersion


class TokenMacroPredictor(nn.Module):
    """Parameter-counted Markov/history/GRU macrostate transition model."""

    def __init__(
        self,
        state_dim: int,
        *,
        history: int = 1,
        hidden_dim: int = 128,
        recurrent: bool = False,
        n_actions: int = 2,
    ) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.history = history
        self.recurrent = recurrent
        self.transition: nn.Module
        if recurrent:
            self.transition = nn.GRU(state_dim, hidden_dim, batch_first=True)
            transition_dim = hidden_dim
        else:
            self.transition = nn.Sequential(
                nn.Linear(state_dim * history, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
            )
            transition_dim = hidden_dim
        self.state_head = nn.Linear(transition_dim, state_dim)
        self.action_head = nn.Linear(transition_dim, n_actions)

    def forward(
        self, history_states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if history_states.ndim != 3:
            raise ValueError("history_states must be [batch,history,state]")
        if self.recurrent:
            hidden, _ = self.transition(history_states)
            features = hidden[:, -1]
        else:
            if history_states.shape[1] != self.history:
                raise ValueError("history length does not match predictor")
            features = self.transition(history_states.flatten(1))
        return F.normalize(self.state_head(features), dim=-1), self.action_head(
            features
        )


def autonomous_macro_rollout(
    predictor: TokenMacroPredictor,
    initial_history: torch.Tensor,
    *,
    horizon: int,
    intervention: tuple[int, int] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Feed every predicted macrostate back without reading teacher states."""

    if horizon < 0:
        raise ValueError("horizon must be non-negative")
    history = initial_history.clone()
    predicted: list[torch.Tensor] = []
    actions: list[torch.Tensor] = []
    for step in range(horizon):
        current = history[:, -predictor.history :]
        if intervention is not None and step == intervention[0]:
            source, target = intervention[1], (intervention[1] + 1) % current.shape[-1]
            current = current.clone()
            source_values = current[..., source].clone()
            current[..., source] = current[..., target]
            current[..., target] = source_values
        next_state, action_logits = predictor(current)
        predicted.append(next_state)
        actions.append(action_logits)
        history = torch.cat((history, next_state.unsqueeze(1)), dim=1)
    empty_state = history.new_empty(history.shape[0], 0, history.shape[-1])
    empty_action = history.new_empty(
        history.shape[0], 0, predictor.action_head.out_features
    )
    return (
        torch.stack(predicted, dim=1) if predicted else empty_state,
        torch.stack(actions, dim=1) if actions else empty_action,
    )


@dataclass
class MacroTrajectory:
    prompt_id: str
    template_id: str
    task_family: str
    t1: torch.Tensor
    t2: torch.Tensor
    dispersion: torch.Tensor
    actions: torch.Tensor


def _extract_trajectory(
    bundle: Any,
    encoder: JStateEncoder,
    prompt: str,
    *,
    prompt_id: str,
    template_id: str,
    task_family: str,
    layers: list[int],
    max_steps: int,
) -> MacroTrajectory:
    input_ids = bundle.lens_model.encode(prompt, max_length=512)
    t1_values: list[torch.Tensor] = []
    t2_values: list[torch.Tensor] = []
    dispersion_values: list[torch.Tensor] = []
    actions: list[int] = []
    eos = getattr(bundle.tokenizer, "eos_token_id", None)
    for _ in range(max_steps):
        with ActivationRecorder(bundle.layers, at=layers) as recorder:
            with torch.no_grad():
                logits = bundle.forward_logits(input_ids)[0, -1]
        scores = torch.stack(
            [
                encoder.encode(recorder.activations[layer][0, -1].float(), layer)
                .dense_scores.float()
                .cpu()
                for layer in layers
            ]
        )
        t1_values.append(scores[-1])
        pooled, dispersion = pool_workspace_band(scores)
        t2_values.append(pooled)
        dispersion_values.append(dispersion)
        token = int(torch.argmax(logits))
        actions.append(token)
        input_ids = torch.cat(
            (input_ids, torch.tensor([[token]], device=input_ids.device)), dim=1
        )
        if eos is not None and token == int(eos):
            break
    return MacroTrajectory(
        prompt_id=prompt_id,
        template_id=template_id,
        task_family=task_family,
        t1=torch.stack(t1_values),
        t2=torch.stack(t2_values),
        dispersion=torch.stack(dispersion_values),
        actions=torch.tensor(actions, dtype=torch.long),
    )


def main() -> None:
    parser = standard_parser(
        "Run token-time measured-J macrostate closure",
        "configs/confirm_v2.yaml",
    )
    parser.add_argument("--max-steps", type=int)
    args = parser.parse_args()
    context = initialize_context("token-time-closure", args)
    try:
        gate = require_phase0_v2_gate(context)
        layer_calibration = require_closure_eligible_layers(context)
        if args.dry_run:
            context.finish("DRY_RUN", gate_run_id=gate["run_id"])
            return
        layers = [int(layer) for layer in layer_calibration["eligible_layers"]]
        bundle = load_model_bundle(context.config)
        vocabulary = ConceptVocabulary.from_json(
            concept_vocabulary_v2_path(
                context, int(context.config["jstate"]["concept_vocab_size"])
            )
        )
        encoder = JStateEncoder.from_lens(
            bundle.lens,
            bundle.unembedding_weight,
            vocabulary,
            k=int(context.config["jstate"]["k"]),
            lazy=True,
            protocol_version="phase0_protocol_v2",
            direction_chunk_size=int(context.config["jstate"]["direction_chunk_size"]),
        )
        max_steps = int(
            args.max_steps
            or context.config.get("token_time", {}).get("max_generation_steps", 32)
        )
        examples = _task_pool(context, 500)
        if args.limit is not None:
            examples = examples[: args.limit]
        trajectories = [
            _extract_trajectory(
                bundle,
                encoder,
                example.prompt,
                prompt_id=example.example_id,
                template_id=example.template_id,
                task_family=example.family,
                layers=layers,
                max_steps=max_steps,
            )
            for example in examples
        ]
        raw_path = context.raw_dir / context.run_id / "token_macro_states.jsonl"
        records = []
        for trajectory in trajectories:
            for step in range(trajectory.t1.shape[0]):
                records.append(
                    {
                        "schema_version": 2,
                        "protocol_version": "phase0_protocol_v2",
                        "run_id": context.run_id,
                        "trajectory_id": trajectory.prompt_id,
                        "template_id": trajectory.template_id,
                        "task_family": trajectory.task_family,
                        "token_step": step,
                        "eligible_layers": layers,
                        "t1_dense_scores": trajectory.t1[step].tolist(),
                        "t2_dense_scores": trajectory.t2[step].tolist(),
                        "layer_dispersion": float(trajectory.dispersion[step]),
                        "semantic_action_token": int(trajectory.actions[step]),
                    }
                )
        append_jsonl(raw_path, records)
        summary = pd.DataFrame(
            [
                {
                    "trajectory_id": trajectory.prompt_id,
                    "template_id": trajectory.template_id,
                    "task_family": trajectory.task_family,
                    "steps": int(trajectory.t1.shape[0]),
                    "mean_layer_dispersion": float(trajectory.dispersion.mean()),
                }
                for trajectory in trajectories
            ]
        )
        summary.to_parquet(
            context.processed_dir / f"token_time_trajectories_{context.run_id}.parquet",
            index=False,
        )
        write_json_atomic(
            context.processed_dir / "token_time_status.json",
            {
                "schema_version": 2,
                "protocol_version": "phase0_protocol_v2",
                "run_id": context.run_id,
                "status": "TRACE_EXTRACTION_COMPLETED",
                "trajectory_count": len(trajectories),
                "record_count": len(records),
                "eligible_layers": layers,
                "warning": "Layer-depth and token-time states are distinct analyses.",
            },
        )
        context.finish(
            "COMPLETED", trajectories=len(trajectories), records=len(records)
        )
    except KeyboardInterrupt:
        context.finish("FAILED", error="KeyboardInterrupt: run cancelled")
        raise
    except Exception as exc:
        context.finish("FAILED", error=f"{type(exc).__name__}: {exc}")
        raise


if __name__ == "__main__":
    main()
