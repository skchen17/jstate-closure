"""Shared runner setup, gate checks, and artifact paths."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from jclosure.config import config_digest, load_config
from jclosure.provenance import build_manifest, set_seed, write_json_atomic


def repository_root() -> Path:
    return Path(__file__).resolve().parents[3]


def standard_parser(description: str, default_config: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", default=default_config)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device", type=int)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--confirmation-model",
        action="store_true",
        help="use independently gated confirmation_model model/lens settings",
    )
    return parser


@dataclass
class ExperimentContext:
    kind: str
    config: dict[str, Any]
    seed: int
    root: Path
    run_id: str
    manifest_path: Path
    raw_dir: Path
    processed_dir: Path
    figures_dir: Path
    reports_dir: Path

    def finish(self, status: str, **updates: Any) -> None:
        manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        manifest.update(updates)
        manifest["status"] = status
        write_json_atomic(self.manifest_path, manifest)


def initialize_context(kind: str, args: argparse.Namespace) -> ExperimentContext:
    root = repository_root()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = root / config_path
    config = load_config(config_path)
    if getattr(args, "confirmation_model", False):
        confirmation = config.get("confirmation_model")
        if not confirmation or not confirmation.get("enabled", False):
            raise ValueError("configuration does not enable confirmation_model")
        config["model"] = copy.deepcopy(confirmation["model"])
        config["lens"] = copy.deepcopy(confirmation["lens"])
        config.setdefault("run", {})["confirmation_model"] = True
    seed = int(args.seed or config["reproducibility"]["dataset_seed"])
    if args.device is not None:
        config["model"]["device"] = int(args.device)
    set_seed(seed, bool(config["reproducibility"].get("deterministic", True)))
    outputs = config["outputs"]
    output_root = Path(outputs.get("root", "."))
    if not output_root.is_absolute():
        output_root = (root / output_root).resolve()
    raw_dir = output_root / outputs["raw"]
    processed_dir = output_root / outputs["processed"]
    figures_dir = output_root / outputs["figures"]
    reports_dir = output_root / outputs["reports"]
    for directory in (raw_dir, processed_dir, figures_dir, reports_dir):
        directory.mkdir(parents=True, exist_ok=True)
    manifest = build_manifest(
        kind=kind,
        config=config,
        seed=seed,
        repo_root=root,
        command=sys.argv,
    )
    run_id = manifest["run_id"]
    manifest_path = raw_dir / run_id / "manifest.json"
    write_json_atomic(manifest_path, manifest)
    return ExperimentContext(
        kind=kind,
        config=config,
        seed=seed,
        root=root,
        run_id=run_id,
        manifest_path=manifest_path,
        raw_dir=raw_dir,
        processed_dir=processed_dir,
        figures_dir=figures_dir,
        reports_dir=reports_dir,
    )


def phase0_gate_path(context: ExperimentContext) -> Path:
    if context.config.get("run", {}).get("confirmation_model"):
        return context.processed_dir / "phase0_gate_qwen3_6_27b.json"
    return context.processed_dir / "phase0_gate.json"


def concept_vocabulary_path(context: ExperimentContext) -> Path:
    if context.config.get("run", {}).get("confirmation_model"):
        return context.processed_dir / "concept_vocabulary_qwen3_6_27b.json"
    return context.processed_dir / "concept_vocabulary.json"


def require_phase0_gate(context: ExperimentContext) -> dict[str, Any]:
    path = phase0_gate_path(context)
    if not path.exists():
        raise RuntimeError(
            "Phase 0 gate artifact is missing; run scripts/run_validation.sh first"
        )
    gate = json.loads(path.read_text(encoding="utf-8"))
    if gate.get("config_digest") != config_digest(context.config):
        # Stage configs extend the base config and legitimately change run sizes.
        pinned = gate.get("model_revision"), gate.get("lens_revision")
        current = (
            context.config["model"]["revision"],
            context.config["lens"]["revision"],
        )
        if pinned != current:
            raise RuntimeError("Phase 0 gate was produced for different artifacts")
    if not gate.get("passed", False):
        raise RuntimeError(
            "Phase 0 did not pass; later runners may be tested but causal results cannot be interpreted"
        )
    return gate
