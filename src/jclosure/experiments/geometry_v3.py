"""Exploratory protocol v3 geometry bank, spectra, and Pareto audit."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

from jclosure.baseline_guard import verify_manifest
from jclosure.clamp import one_shot_clamp
from jclosure.clamp_v3 import (
    V3ClampThresholds,
    construct_dense_candidate,
    validate_v3_clamp,
)
from jclosure.datasets import (
    TaskExample,
    generate_arithmetic,
    generate_boolean,
    generate_graph_traversal,
    generate_state_machines,
    generate_symbolic_planning,
    generate_variable_binding,
    upstream_multihop,
)
from jclosure.experiments.common import initialize_context, standard_parser
from jclosure.geometry import (
    DenseJMap,
    DenseNullProjector,
    SparseStateEquality,
    SpectrumSummary,
    maximum_feasible_displacement,
    pareto_nondominated,
)
from jclosure.jstate import ConceptVocabulary, JStateEncoder
from jclosure.metrics import rms_drift, topk_overlap
from jclosure.model import load_model_bundle
from jclosure.provenance import append_jsonl, sha256_file, write_json_atomic
from jclosure.recorder import ActivationRecorder

PROTOCOL = "exploratory_protocol_v3"


@dataclass(frozen=True)
class BankRecord:
    prompt_id: str
    task_family: str
    template_id: str
    prompt_hash: str
    split: str
    sequence_length: int
    layers: tuple[int, ...]
    activation_path: str
    activation_sha256: str
    activation_shapes: dict[str, list[int]]
    teacher_answer: str
    teacher_correct: bool
    variables: dict[str, Any]


@dataclass(frozen=True)
class NaturalityStats:
    mahalanobis: float
    knn_ratio: float
    natural: bool


class NaturalityModel:
    """Train-fitted PCA-whitened naturality envelope with deterministic 10-NN."""

    def __init__(self, dimension: int, neighbors: int, quantile: float) -> None:
        self.dimension = int(dimension)
        self.neighbors = int(neighbors)
        self.quantile = float(quantile)
        self.pca: PCA | None = None
        self.nn: NearestNeighbors | None = None
        self.fit_coordinates: np.ndarray | None = None
        self.mahalanobis_threshold: float | None = None
        self.knn_threshold: float | None = None
        self.median_knn: float | None = None

    def fit(self, states: np.ndarray) -> NaturalityModel:
        if states.ndim != 2 or len(states) < 3:
            raise ValueError("naturality fit requires at least three state vectors")
        components = min(self.dimension, states.shape[0] - 1, states.shape[1])
        self.pca = PCA(n_components=components, whiten=True, random_state=0)
        coordinates = self.pca.fit_transform(states)
        self.fit_coordinates = coordinates
        distances = np.linalg.norm(coordinates, axis=1)
        neighbor_count = min(self.neighbors + 1, len(coordinates))
        self.nn = NearestNeighbors(n_neighbors=neighbor_count, algorithm="brute")
        self.nn.fit(coordinates)
        fit_knn = self.nn.kneighbors(coordinates, return_distance=True)[0][:, 1:].mean(1)
        self.median_knn = float(np.median(fit_knn))
        self.mahalanobis_threshold = float(np.quantile(distances, self.quantile))
        self.knn_threshold = float(
            np.quantile(fit_knn / max(self.median_knn, 1e-12), self.quantile)
        )
        return self

    def score(self, state: np.ndarray) -> NaturalityStats:
        if self.pca is None or self.nn is None or self.median_knn is None:
            raise RuntimeError("naturality model is not fitted")
        coordinate = self.pca.transform(np.asarray(state)[None])[0]
        mahalanobis = float(np.linalg.norm(coordinate))
        count = min(
            self.neighbors,
            0 if self.fit_coordinates is None else len(self.fit_coordinates),
        )
        if count < 1:
            raise RuntimeError("naturality model has no fit coordinates")
        if self.mahalanobis_threshold is None or self.knn_threshold is None:
            raise RuntimeError("naturality thresholds are not fitted")
        distance = float(
            self.nn.kneighbors(coordinate[None], n_neighbors=count)[0].mean()
        )
        ratio = distance / max(self.median_knn, 1e-12)
        return NaturalityStats(
            mahalanobis=mahalanobis,
            knn_ratio=ratio,
            natural=bool(
                mahalanobis <= float(self.mahalanobis_threshold)
                and ratio <= float(self.knn_threshold)
            ),
        )


def _flexible_examples(root: Path) -> list[TaskExample]:
    payload = json.loads(
        (root / "experiments/flexible-generalization.json").read_text(encoding="utf-8")
    )
    output: list[TaskExample] = []
    for category in payload["categories"]:
        for function in category["funcs"]:
            for argument in category["args"]:
                output.append(
                    TaskExample(
                        example_id=f"geometry-flex:{category['name']}:{function['name']}:{argument}",
                        family="flexible_function",
                        template_id=f"{category['name']}:{function['name']}",
                        prompt=str(function["template"]).format(arg=argument),
                        answer=str(function["answers"][argument]),
                        intermediates=(str(argument),),
                    )
                )
    return output


def _unique_prompt_examples(
    examples: list[TaskExample], count: int, *, family: str
) -> list[TaskExample]:
    """Select a deterministic prefix with distinct normalized prompt hashes."""

    selected: list[TaskExample] = []
    seen: set[str] = set()
    for example in examples:
        digest = hashlib.sha256(example.prompt.strip().encode("utf-8")).hexdigest()
        if digest in seen:
            continue
        seen.add(digest)
        selected.append(example)
        if len(selected) == count:
            return selected
    raise RuntimeError(
        f"geometry task family {family} has only {len(selected)} unique prompts; "
        f"requires {count}"
    )


def geometry_examples(root: Path, config: dict[str, Any]) -> list[TaskExample]:
    count = int(config["run"]["geometry_states_per_family"])
    seed = int(config["reproducibility"]["dataset_seed"])
    upstream = root / config["data"]["upstream_root"]
    generated_count = count * 8
    pools = (
        generate_arithmetic(generated_count, seed=seed),
        generate_boolean(generated_count, seed=seed + 1),
        generate_graph_traversal(generated_count, seed=seed + 2),
        generate_symbolic_planning(generated_count, seed=seed + 3),
        generate_variable_binding(generated_count, seed=seed + 4),
        generate_state_machines(generated_count, seed=seed + 5),
        upstream_multihop(upstream),
        _flexible_examples(upstream),
    )
    selected = [
        _unique_prompt_examples(values, count, family=values[0].family)
        for values in pools
    ]
    output = [item for values in selected for item in values]
    prompt_hashes = {
        hashlib.sha256(item.prompt.strip().encode("utf-8")).hexdigest()
        for item in output
    }
    if len(prompt_hashes) != len(output):
        raise RuntimeError("geometry activation bank contains cross-family prompt duplicates")
    return output


def _split_map(examples: list[TaskExample]) -> dict[str, str]:
    """Split every family into exact hash-sorted halves without data fitting."""

    families: dict[str, list[TaskExample]] = defaultdict(list)
    for example in examples:
        families[example.family].append(example)
    output: dict[str, str] = {}
    for values in families.values():
        ordered = sorted(
            values,
            key=lambda item: (
                hashlib.sha256(item.prompt.encode("utf-8")).hexdigest(),
                item.example_id,
            ),
        )
        cutoff = len(ordered) // 2
        for index, example in enumerate(ordered):
            output[example.example_id] = "fit" if index < cutoff else "audit"
    return output


def _single_answer_id(tokenizer: Any, answer: str) -> int | None:
    for surface in (" " + answer.strip(), answer):
        values = tokenizer.encode(surface, add_special_tokens=False)
        if len(values) == 1:
            return int(values[0])
    return None


def build_activation_bank(
    context, bundle: Any, layers: list[int], *, limit: int | None = None
) -> Path:
    examples = geometry_examples(context.root, context.config)
    if context.config.get("run", {}).get("geometry_fit_fraction") != 0.5:
        raise ValueError("protocol v3 freezes geometry_fit_fraction at 0.50")
    if limit is not None:
        examples = examples[:limit]
    run_dir = context.raw_dir / context.run_id
    tensor_dir = run_dir / "activation_bank"
    tensor_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    splits = _split_map(examples)
    for index, example in enumerate(examples):
        input_ids = bundle.lens_model.encode(
            example.prompt, max_length=int(context.config["model"]["max_seq_len"])
        )
        with ActivationRecorder(bundle.layers, at=layers) as recorder:
            with torch.no_grad():
                logits = bundle.forward_logits(input_ids)[0, -1].detach().float().cpu()
        answer_id = _single_answer_id(bundle.tokenizer, example.answer)
        payload = {
            "schema_version": 3,
            "protocol_version": PROTOCOL,
            "prompt_id": example.example_id,
            "task_family": example.family,
            "template_id": example.template_id,
            "prompt_hash": hashlib.sha256(example.prompt.encode("utf-8")).hexdigest(),
            "input_ids": input_ids.detach().cpu(),
            "attention_mask": torch.ones_like(input_ids, dtype=torch.bool).cpu(),
            "activations": {
                int(layer): recorder.activations[layer][0].detach().to(torch.float16).cpu()
                for layer in layers
            },
            "teacher_answer": example.answer,
            "teacher_correct": bool(
                answer_id is not None and int(torch.argmax(logits)) == answer_id
            ),
            "variables": example.variables,
        }
        target = tensor_dir / f"state-{index:04d}.pt"
        torch.save(payload, target)
        record = BankRecord(
            prompt_id=example.example_id,
            task_family=example.family,
            template_id=example.template_id,
            prompt_hash=payload["prompt_hash"],
            split=splits[example.example_id],
            sequence_length=int(input_ids.shape[-1]),
            layers=tuple(layers),
            activation_path=str(target.relative_to(context.root)),
            activation_sha256=sha256_file(target),
            activation_shapes={
                str(layer): list(payload["activations"][layer].shape) for layer in layers
            },
            teacher_answer=example.answer,
            teacher_correct=payload["teacher_correct"],
            variables=example.variables,
        )
        records.append(asdict(record))
    manifest_path = run_dir / "activation_bank_manifest.jsonl"
    append_jsonl(manifest_path, records)
    family_split = (
        pd.DataFrame(records).groupby(["task_family", "split"]).size().to_dict()
    )
    unique_prompt_hashes = len({record["prompt_hash"] for record in records})
    if unique_prompt_hashes != len(records):
        raise RuntimeError("activation bank prompt hashes are not unique")
    context.finish(
        "BANK_COMPLETED",
        activation_bank_manifest=str(manifest_path.relative_to(context.root)),
        activation_bank_records=len(records),
        unique_prompt_hashes=unique_prompt_hashes,
        family_split={f"{key[0]}:{key[1]}": value for key, value in family_split.items()},
    )
    return manifest_path


def _load_bank(root: Path, manifest_path: Path) -> list[dict[str, Any]]:
    records = [
        json.loads(line)
        for line in manifest_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    for record in records:
        path = root / record["activation_path"]
        if sha256_file(path) != record["activation_sha256"]:
            raise RuntimeError(f"activation bank hash mismatch: {path}")
    return records


def _spectrum(
    matrix: torch.Tensor, *, relative_tolerances: list[float]
) -> SpectrumSummary:
    values = torch.linalg.svdvals(matrix)
    return SpectrumSummary.from_singular_values(
        values.detach().cpu(),
        rows=matrix.shape[0],
        cols=matrix.shape[1],
        dtype=matrix.dtype,
        relative_tolerances=relative_tolerances,
    )


def _relative_error(left: torch.Tensor, right: torch.Tensor) -> float:
    denominator = torch.linalg.vector_norm(right.float()).clamp_min(1e-20)
    return float(torch.linalg.vector_norm(left.float() - right.float()) / denominator)


def _local_operator_norm(
    dense_map: DenseJMap,
    h: torch.Tensor,
    layer: int,
    *,
    seed: int,
    iterations: int = 12,
) -> float:
    """Estimate the local Jacobian spectral norm without materializing its Gram."""

    generator = torch.Generator(device="cpu").manual_seed(seed)
    vector = torch.randn(h.shape, generator=generator, dtype=h.dtype).to(h.device)
    vector = F.normalize(vector, dim=0)
    for _ in range(iterations):
        mapped = dense_map.dense_state_jvp(h, vector, layer)
        vector = dense_map.dense_state_vjp(h, mapped, layer)
        norm = torch.linalg.vector_norm(vector)
        if float(norm) <= 1e-20:
            return 0.0
        vector = vector / norm
    return float(torch.linalg.vector_norm(dense_map.dense_state_jvp(h, vector, layer)))


def _local_diagnostics(
    dense_map: DenseJMap,
    h: torch.Tensor,
    layer: int,
    *,
    checks: int,
    seed: int,
    tolerances: list[float],
    full_spectrum: bool,
    map_summary: SpectrumSummary | None = None,
) -> dict[str, Any]:
    h = h.float()
    rank_status: str
    map_algebraic_rank = (
        sum(value > 0.0 for value in map_summary.singular_values)
        if map_summary is not None
        else None
    )
    if full_spectrum:
        gram = dense_map.local_jacobian_gram(h, layer).double()
        eigenvalues = torch.linalg.eigvalsh(gram).clamp_min(0)
        singular_values = eigenvalues.sqrt().flip(0)
        summary = SpectrumSummary.from_singular_values(
            singular_values,
            rows=dense_map.raw_map(layer).shape[0],
            cols=h.numel(),
            dtype=torch.float32,
            relative_tolerances=tolerances,
        )
        ranks = summary.tolerance_ranks
        rank_bounds = {key: [value, value] for key, value in ranks.items()}
        analytic_rank = (
            max(map_algebraic_rank - 1, 0)
            if map_algebraic_rank is not None
            else max(int(torch.count_nonzero(singular_values > 0)) - 1, 0)
        )
        operator_norm = float(singular_values[0]) if singular_values.numel() else 0.0
        smallest = float(singular_values[-1]) if singular_values.numel() else 0.0
        rank_status = summary.rank_status
    else:
        if map_summary is None:
            raise ValueError("map_summary is required for bounded local diagnostics")
        # P_s CA is a rank-one left projection and s is in range(CA), so its
        # exact algebraic rank is rank(CA)-1.  At a finite numerical tolerance,
        # Cauchy interlacing bounds the local rank between r-1 and r.  Recording
        # those bounds avoids an O(d_model^3) eigendecomposition for all 256
        # audit states while the preregistered 16 states retain full spectra.
        map_ranks = map_summary.tolerance_ranks
        rank_bounds = {
            key: [max(value - 1, 0), value] for key, value in map_ranks.items()
        }
        ranks = {key: bounds[0] for key, bounds in rank_bounds.items()}
        if map_algebraic_rank is None:
            raise AssertionError("bounded diagnostics require a map algebraic rank")
        analytic_rank = max(map_algebraic_rank - 1, 0)
        operator_norm = _local_operator_norm(
            dense_map, h, layer, seed=seed + 99_000
        )
        smallest = None
        summary = None
        rank_status = "NUMERICALLY_BOUNDED"
    generator = torch.Generator(device="cpu").manual_seed(seed)
    jvp_errors: list[float] = []
    vjp_errors: list[float] = []
    for _ in range(checks):
        v = torch.randn(h.shape, generator=generator, dtype=h.dtype).to(h.device)
        u = torch.randn(
            dense_map.raw_map(layer).shape[0], generator=generator, dtype=h.dtype
        ).to(h.device)
        analytic_jvp = dense_map.dense_state_jvp(h, v, layer)
        _, automatic_jvp = torch.autograd.functional.jvp(
            lambda value: dense_map.dense_state(value, layer), h, v
        )
        analytic_vjp = dense_map.dense_state_vjp(h, u, layer)
        _, automatic_vjp = torch.autograd.functional.vjp(
            lambda value: dense_map.dense_state(value, layer), h, v=u
        )
        jvp_errors.append(_relative_error(analytic_jvp, automatic_jvp))
        vjp_errors.append(_relative_error(analytic_vjp, automatic_vjp))
    structural_null = h.numel() - analytic_rank
    tangent_null = {
        key: max(h.numel() - rank - 1, 0) for key, rank in ranks.items()
    }
    radial_jvp = dense_map.dense_state_jvp(h, h, layer)
    radial_denominator = operator_norm * float(torch.linalg.vector_norm(h))
    radial_residual = (
        0.0
        if radial_denominator <= 1e-20
        else float(torch.linalg.vector_norm(radial_jvp)) / radial_denominator
    )
    return {
        "spectrum": summary.to_dict() if summary is not None else None,
        "tolerance_ranks": ranks,
        "tolerance_rank_bounds": rank_bounds,
        "analytic_rank": analytic_rank,
        "rank_status": rank_status,
        "structural_null_dimension": structural_null,
        "tolerance_null_dimensions": {
            key: h.numel() - value for key, value in ranks.items()
        },
        "tangent_null_dimensions": tangent_null,
        "radial_residual": radial_residual,
        "jvp_relative_errors": jvp_errors,
        "vjp_relative_errors": vjp_errors,
        "jvp_passed": max(jvp_errors + vjp_errors, default=0.0) <= 1e-4,
        "extremal_singular_values": [
            operator_norm,
            smallest,
        ],
        "extremal_method": "exact" if full_spectrum else "power_top_only",
    }


def _load_vocabularies(context) -> dict[int, ConceptVocabulary]:
    sizes = [int(value) for value in context.config["geometry"]["dictionary_sizes"]]
    return {
        size: ConceptVocabulary.from_json(
            context.root / "results/processed" / f"concept_vocabulary_v2_{size}.json"
        )
        for size in sizes
    }


def run_spectra(
    context,
    bundle: Any,
    bank_manifest: Path,
    *,
    shard_index: int,
    shard_count: int,
    smoke: bool,
) -> tuple[Path, Path]:
    records = _load_bank(context.root, bank_manifest)
    audit_records = [record for record in records if record["split"] == "audit"]
    if not audit_records:
        raise RuntimeError("geometry activation bank has no audit split")
    layers = [int(value) for value in context.config["geometry"]["candidate_layers"]]
    layers = [
        layer
        for layer in layers
        if int(hashlib.sha256(str(layer).encode()).hexdigest(), 16) % shard_count
        == shard_index
    ]
    if smoke:
        layers = layers[:1]
    vocabularies = _load_vocabularies(context)
    if smoke:
        vocabularies = {min(vocabularies): vocabularies[min(vocabularies)]}
    tolerances = [
        float(value)
        for value in context.config["geometry"]["spectrum_relative_tolerances"]
    ]
    map_rows: list[dict[str, Any]] = []
    local_rows: list[dict[str, Any]] = []
    for size, vocabulary in vocabularies.items():
        encoder = JStateEncoder.from_lens(
            bundle.lens,
            bundle.unembedding_weight,
            vocabulary,
            k=int(context.config["jstate"]["k"]),
            lazy=True,
            protocol_version=PROTOCOL,
            direction_chunk_size=int(context.config["jstate"].get("direction_chunk_size", 512)),
        )
        dense_map = DenseJMap.from_encoder(encoder)
        for layer in layers:
            raw = dense_map.raw_map(layer, device=bundle.hf_model.device).float()
            centered = dense_map.centered_map(layer, device=bundle.hf_model.device).float()
            centered_summary: SpectrumSummary | None = None
            for map_kind, matrix in (("A", raw), ("CA", centered)):
                summary = _spectrum(matrix, relative_tolerances=tolerances)
                if map_kind == "CA":
                    centered_summary = summary
                map_rows.append(
                    {
                        "schema_version": 3,
                        "protocol_version": PROTOCOL,
                        "record_kind": "map_spectrum",
                        "run_id": context.run_id,
                        "layer": layer,
                        "dictionary_size": size,
                        "dictionary_hash": vocabulary.digest,
                        "map_kind": map_kind,
                        **summary.to_dict(),
                        "provenance": {
                            "model_revision": bundle.model_revision,
                            "lens_revision": bundle.lens_revision,
                            "activation_bank_manifest": str(bank_manifest.relative_to(context.root)),
                        },
                    }
                )
            state_limit = 1 if smoke else len(audit_records)
            full_limit = (
                1
                if smoke
                else int(context.config["geometry"]["full_local_spectrum_states"])
            )
            full_check_count = (
                1 if smoke else int(context.config["geometry"]["jvp_checks_per_combination"])
            )
            for state_index, bank_record in enumerate(audit_records[:state_limit]):
                payload = torch.load(
                    context.root / bank_record["activation_path"], map_location="cpu"
                )
                h = payload["activations"][layer][-1].to(bundle.hf_model.device).float()
                diagnostics = _local_diagnostics(
                    dense_map,
                    h,
                    layer,
                    checks=full_check_count if state_index < full_limit else 1,
                    seed=context.seed + state_index + 1000 * layer + size,
                    tolerances=tolerances,
                    full_spectrum=state_index < full_limit,
                    map_summary=centered_summary,
                )
                local_rows.append(
                    {
                        "schema_version": 3,
                        "protocol_version": PROTOCOL,
                        "record_kind": "local_spectrum",
                        "run_id": context.run_id,
                        "prompt_id": bank_record["prompt_id"],
                        "task_family": bank_record["task_family"],
                        "layer": layer,
                        "dictionary_size": size,
                        "dictionary_hash": vocabulary.digest,
                        **diagnostics,
                        "provenance": {
                            "activation_sha256": bank_record["activation_sha256"],
                            "position": -1,
                        },
                    }
                )
            dense_map._device_cache.clear()
            encoder._device_directions.clear()
            encoder._device_raw_directions.clear()
            torch.cuda.empty_cache()
    suffix = "smoke" if smoke else f"shard-{shard_index:03d}"
    run_dir = context.raw_dir / context.run_id
    map_path = run_dir / f"map_spectra-{suffix}.parquet"
    local_path = run_dir / f"local_spectra-{suffix}.parquet"
    pd.DataFrame(map_rows).to_parquet(map_path, index=False)
    pd.DataFrame(local_rows).to_parquet(local_path, index=False)
    return map_path, local_path


def _naturality_models(
    context, records: list[dict[str, Any]], layers: list[int]
) -> dict[int, NaturalityModel]:
    fit_records = [record for record in records if record["split"] == "fit"]
    models: dict[int, NaturalityModel] = {}
    for layer in layers:
        states = []
        for record in fit_records:
            payload = torch.load(
                context.root / record["activation_path"], map_location="cpu"
            )
            states.append(payload["activations"][layer][-1].float().numpy())
        model = NaturalityModel(
            int(context.config["geometry"]["pca_dimension"]),
            int(context.config["geometry"]["nearest_neighbors"]),
            float(context.config["geometry"]["naturality_quantile"]),
        ).fit(np.stack(states))
        models[layer] = model
    return models


def _scaled(vector: torch.Tensor, target: float) -> torch.Tensor:
    norm = torch.linalg.vector_norm(vector.float()).clamp_min(1e-20)
    return vector.float() * (target / float(norm))


def _balanced_audit_records(
    records: list[dict[str, Any]], per_family: int
) -> list[dict[str, Any]]:
    families: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        families[record["task_family"]].append(record)
    selected: list[dict[str, Any]] = []
    for family in sorted(families):
        ordered = sorted(
            families[family], key=lambda row: (row["prompt_hash"], row["prompt_id"])
        )
        if len(ordered) < per_family:
            raise RuntimeError(
                f"Pareto audit family {family} has {len(ordered)} records; "
                f"requires {per_family}"
            )
        selected.extend(ordered[:per_family])
    return selected


def run_pareto(
    context,
    bundle: Any,
    bank_manifest: Path,
    *,
    limit: int | None,
    shard_index: int,
    shard_count: int,
) -> Path:
    records = _load_bank(context.root, bank_manifest)
    audit = [record for record in records if record["split"] == "audit"]
    fit = [record for record in records if record["split"] == "fit"]
    layers = [int(value) for value in context.config["geometry"]["candidate_layers"]]
    layers = [
        layer
        for layer in layers
        if int(hashlib.sha256(str(layer).encode()).hexdigest(), 16) % shard_count
        == shard_index
    ]
    if not layers:
        raise RuntimeError("Pareto shard has no assigned layers")
    naturality_models = _naturality_models(context, records, layers)
    vocabularies = _load_vocabularies(context)
    strengths = [float(value) for value in context.config["geometry"]["strengths"]]
    methods = list(context.config["geometry"]["methods"])
    tolerances = [
        float(value)
        for value in context.config["geometry"]["spectrum_relative_tolerances"]
    ]
    audit = _balanced_audit_records(
        audit, int(context.config["geometry"]["pareto_states_per_family"])
    )
    if limit is not None:
        audit = audit[:limit]
    by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in fit:
        by_family[record["task_family"]].append(record)
    rows: list[dict[str, Any]] = []
    run_dir = context.raw_dir / context.run_id
    progress_path = run_dir / "pareto_progress.json"
    completed_parts: list[str] = []
    for size, vocabulary in vocabularies.items():
        encoder = JStateEncoder.from_lens(
            bundle.lens,
            bundle.unembedding_weight,
            vocabulary,
            k=int(context.config["jstate"]["k"]),
            lazy=True,
            protocol_version=PROTOCOL,
            direction_chunk_size=int(context.config["jstate"].get("direction_chunk_size", 512)),
        )
        dense_map = DenseJMap.from_encoder(encoder)
        for layer in layers:
            projector = DenseNullProjector(dense_map, layer)
            combination_rows: list[dict[str, Any]] = []
            for anchor_index, anchor_record in enumerate(audit):
                donors = by_family[anchor_record["task_family"]]
                if not donors:
                    continue
                donor_record = donors[anchor_index % len(donors)]
                anchor_payload = torch.load(
                    context.root / anchor_record["activation_path"], map_location="cpu"
                )
                donor_payload = torch.load(
                    context.root / donor_record["activation_path"], map_location="cpu"
                )
                h = anchor_payload["activations"][layer][-1].to(bundle.hf_model.device).float()
                donor = donor_payload["activations"][layer][-1].to(bundle.hf_model.device).float()
                difference = donor - h
                natural_scale = float(torch.linalg.vector_norm(difference).item())
                if natural_scale <= 1e-12:
                    continue
                clean_dense = dense_map.dense_state(h, layer)
                clean_raw = dense_map.raw_scores(h, layer)
                clean_sparse = encoder.decompose(h, layer)
                sparse_difference = encoder.decompose(difference, layer)
                generator = torch.Generator(device="cpu").manual_seed(
                    context.seed + anchor_index + 10_000 * layer + size
                )
                random_direction = torch.randn(h.shape, generator=generator).to(h.device)
                base_directions: dict[str, tuple[torch.Tensor, str]] = {
                    "sparse_remainder": (sparse_difference.remainder, "sparse"),
                    "isotropic_random": (random_direction, "dense"),
                    "radial": (h, "dense"),
                }
                singular_values, right_vectors = projector.local_singular_system(h)
                for tolerance in tolerances:
                    approximate_basis = projector.low_singular_basis_from_system(
                        singular_values,
                        right_vectors,
                        relative_tolerance=tolerance,
                    )
                    tangent_basis = projector.tangent_intersection(
                        approximate_basis, h
                    )
                    approximate = projector.project(difference, approximate_basis)
                    tangent = projector.project(difference, tangent_basis)
                    base_directions[f"approximate_dense_null:{tolerance}"] = (
                        approximate,
                        "dense",
                    )
                    base_directions[f"norm_tangent_dense_null:{tolerance}"] = (
                        tangent,
                        "dense",
                    )
                    base_directions[f"hard_constrained:{tolerance}"] = (
                        tangent,
                        "optimized",
                    )
                    del approximate_basis
                    for label, (base_direction, state_kind) in list(base_directions.items()):
                        method, _, label_tolerance = label.partition(":")
                        if method not in methods:
                            continue
                        if label_tolerance and float(label_tolerance) != tolerance:
                            continue
                        if not label_tolerance and tolerance != tolerances[0]:
                            continue
                        direction_norm = float(torch.linalg.vector_norm(base_direction).item())
                        for strength in strengths:
                            paired = hashlib.sha256(
                                "\x1f".join(
                                    (
                                        anchor_record["prompt_id"],
                                        donor_record["prompt_id"],
                                        str(layer),
                                        str(strength),
                                        method,
                                        label_tolerance or "none",
                                    )
                                ).encode()
                            ).hexdigest()
                            exclusion = None
                            optimization_status = "NOT_APPLICABLE"
                            if direction_norm <= 1e-20:
                                candidate = h.clone()
                                exclusion = "zero_dimensional_or_degenerate_direction"
                            elif state_kind == "optimized":
                                optimization = projector.optimize_hard_constraints(
                                    h,
                                    difference,
                                    tangent_basis,
                                    target_displacement=strength * natural_scale,
                                )
                                candidate = optimization.activation
                                optimization_status = optimization.status
                                exclusion = optimization.failure_reason
                            else:
                                delta = _scaled(base_direction, strength * natural_scale)
                                if method == "norm_tangent_dense_null":
                                    delta = projector.retract_to_sphere(h, delta)
                                preliminary = h + delta
                                if state_kind == "sparse":
                                    candidate = one_shot_clamp(
                                        h,
                                        preliminary,
                                        layer=layer,
                                        encoder=encoder,
                                    ).activation
                                else:
                                    candidate = preliminary
                            delta = candidate - h
                            candidate_dense = dense_map.dense_state(candidate, layer)
                            candidate_raw = dense_map.raw_scores(candidate, layer)
                            candidate_sparse = encoder.decompose(candidate, layer)
                            sparse_equality = SparseStateEquality.compare(
                                clean_sparse, candidate_sparse
                            )
                            naturality = naturality_models[layer].score(
                                candidate.detach().cpu().float().numpy()
                            )
                            displacement = float(
                                torch.linalg.vector_norm(delta.float()).item()
                            )
                            dense_cosine = float(
                                F.cosine_similarity(clean_dense[None], candidate_dense[None])
                            )
                            jvp_ratio = float(
                                torch.linalg.vector_norm(
                                    dense_map.dense_state_jvp(h, delta, layer)
                                ).item()
                                / max(displacement, 1e-20)
                            )
                            row = {
                                "schema_version": 3,
                                "protocol_version": PROTOCOL,
                                "record_kind": "pareto",
                                "run_id": context.run_id,
                                "paired_trial_id": paired,
                                "prompt_id": anchor_record["prompt_id"],
                                "donor_id": donor_record["prompt_id"],
                                "task_family": anchor_record["task_family"],
                                "layer": layer,
                                "position": -1,
                                "position_scope": "final",
                                "dictionary_size": size,
                                "dictionary_hash": vocabulary.digest,
                                "state_definition": (
                                    "V3-Sparse" if state_kind == "sparse" else "V3-Dense"
                                ),
                                "method": method,
                                "null_tolerance": (
                                    float(label_tolerance) if label_tolerance else None
                                ),
                                "strength": strength,
                                "jvp_per_delta": jvp_ratio,
                                "dense_cosine": dense_cosine,
                                "dense_profile_l2": float(
                                    torch.linalg.vector_norm(
                                        candidate_dense - clean_dense
                                    ).item()
                                ),
                                "raw_score_scale_ratio": float(
                                    torch.linalg.vector_norm(candidate_raw).item()
                                    / torch.linalg.vector_norm(clean_raw).clamp_min(1e-20).item()
                                ),
                                "top10_overlap": topk_overlap(clean_raw, candidate_raw, 10),
                                "rms_drift": rms_drift(h, candidate),
                                "displacement_fraction": displacement / natural_scale,
                                "donor_alignment": float(
                                    F.cosine_similarity(delta[None], difference[None]).item()
                                )
                                if displacement > 1e-20
                                else 0.0,
                                "sparse_support_f1": sparse_equality.support_f1,
                                "sparse_weighted_jaccard": sparse_equality.weighted_jaccard,
                                "sparse_coefficient_cosine": sparse_equality.coefficient_cosine,
                                "sparse_coefficient_relative_l2": sparse_equality.coefficient_relative_l2,
                                "sparse_reconstruction_cosine": sparse_equality.reconstruction_cosine,
                                "sparse_reconstruction_relative_l2": sparse_equality.reconstruction_relative_l2,
                                "mahalanobis": naturality.mahalanobis,
                                "knn_ratio": naturality.knn_ratio,
                                "natural": naturality.natural,
                                "optimization_status": optimization_status,
                                "valid": exclusion is None,
                                "exclusion_reason": exclusion,
                                "provenance": {
                                    "anchor_activation_sha256": anchor_record["activation_sha256"],
                                    "donor_activation_sha256": donor_record["activation_sha256"],
                                },
                            }
                            rows.append(row)
                            combination_rows.append(row)
                del right_vectors, singular_values
            part_path = run_dir / f"pareto_part-M{size}-L{layer}.parquet"
            pd.DataFrame(combination_rows).to_parquet(part_path, index=False)
            completed_parts.append(str(part_path.relative_to(context.root)))
            write_json_atomic(
                progress_path,
                {
                    "schema_version": 3,
                    "protocol_version": PROTOCOL,
                    "run_id": context.run_id,
                    "shard_index": shard_index,
                    "shard_count": shard_count,
                    "completed_parts": completed_parts,
                    "records_written": len(rows),
                    "status": "RUNNING",
                },
            )
    filename = (
        f"pareto_records-shard-{shard_index:03d}.parquet"
        if limit is None
        else f"pareto_records-preflight-{int(limit)}.parquet"
    )
    path = context.raw_dir / context.run_id / filename
    frame = pd.DataFrame(rows)
    frame.to_parquet(path, index=False)
    summaries: list[dict[str, Any]] = []
    for keys, group in frame.groupby(
        ["layer", "dictionary_size", "state_definition", "method"], dropna=False
    ):
        row_records = group.to_dict("records")
        frontier = pareto_nondominated(
            row_records,
            maximize=("dense_cosine", "displacement_fraction"),
            minimize=("rms_drift",),
        )
        summaries.append(
            {
                "layer": int(keys[0]),
                "dictionary_size": int(keys[1]),
                "state_definition": keys[2],
                "method": keys[3],
                "records": len(group),
                "frontier_records": len(frontier),
                "max_formal_displacement": maximum_feasible_displacement(row_records),
            }
        )
    pd.DataFrame(summaries).to_parquet(
        context.processed_dir / f"pareto_summary_{context.run_id}.parquet", index=False
    )
    write_json_atomic(
        progress_path,
        {
            "schema_version": 3,
            "protocol_version": PROTOCOL,
            "run_id": context.run_id,
            "shard_index": shard_index,
            "shard_count": shard_count,
            "completed_parts": completed_parts,
            "records_written": len(rows),
            "status": "COMPLETED",
            "merged_output": str(path.relative_to(context.root)),
        },
    )
    return path


def run_candidate_smoke(context, bundle: Any, bank_manifest: Path) -> Path:
    """Exercise one local-null and optimized candidate without a naturality claim."""

    records = _load_bank(context.root, bank_manifest)
    fit = [record for record in records if record["split"] == "fit"]
    audit = [record for record in records if record["split"] == "audit"]
    if not fit or not audit:
        raise RuntimeError("candidate smoke needs at least one fit and one audit state")
    layer = int(context.config["geometry"]["candidate_layers"][0])
    size = min(int(value) for value in context.config["geometry"]["dictionary_sizes"])
    vocabulary = _load_vocabularies(context)[size]
    encoder = JStateEncoder.from_lens(
        bundle.lens,
        bundle.unembedding_weight,
        vocabulary,
        k=int(context.config["jstate"]["k"]),
        lazy=True,
        protocol_version=PROTOCOL,
        direction_chunk_size=int(context.config["jstate"].get("direction_chunk_size", 512)),
    )
    dense_map = DenseJMap.from_encoder(encoder)
    anchor = torch.load(
        context.root / audit[0]["activation_path"], map_location="cpu"
    )["activations"][layer][-1].to(bundle.hf_model.device).float()
    donor = torch.load(
        context.root / fit[0]["activation_path"], map_location="cpu"
    )["activations"][layer][-1].to(bundle.hf_model.device).float()
    difference = donor - anchor
    scale = float(torch.linalg.vector_norm(difference).item())
    tolerance = float(context.config["geometry"]["formal_null_tolerance"])
    thresholds = V3ClampThresholds()
    rows: list[dict[str, Any]] = []
    for optimized in (False, True):
        candidate, construction = construct_dense_candidate(
            anchor,
            difference,
            layer=layer,
            dense_map=dense_map,
            natural_scale=scale,
            displacement_fraction=0.20,
            relative_tolerance=tolerance,
            optimized=optimized,
            thresholds=thresholds,
        )
        validation = validate_v3_clamp(
            anchor,
            candidate,
            layer=layer,
            state_definition="V3-Dense",
            encoder=encoder,
            dense_map=dense_map,
            natural_scale=scale,
            natural=True,
            thresholds=thresholds,
        )
        rows.append(
            {
                "schema_version": 3,
                "protocol_version": PROTOCOL,
                "method": "dense_optimized" if optimized else "dense_local_null",
                "layer": layer,
                "dictionary_size": size,
                "construction": construction,
                "validation": asdict(validation),
                "naturality_status": "NOT_EVALUATED_IN_SMOKE",
            }
        )
    path = context.raw_dir / context.run_id / "candidate_smoke.json"
    write_json_atomic(path, {"records": rows})
    return path


def _latest_bank_manifest(context) -> Path:
    manifests = sorted(context.raw_dir.glob("geometry-v3-*/activation_bank_manifest.jsonl"))
    if not manifests:
        raise FileNotFoundError("no geometry v3 activation bank manifest found")
    return manifests[-1]


def main() -> None:
    parser = standard_parser(
        "Run exploratory protocol v3 geometry audit", "configs/geometry_v3.yaml"
    )
    parser.add_argument(
        "--stage", choices=("bank", "spectrum", "pareto", "all", "smoke"), default="all"
    )
    parser.add_argument("--bank-manifest")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    args = parser.parse_args()
    context = initialize_context("geometry-v3", args)
    try:
        immutable = verify_manifest(
            context.root, context.root / "artifacts/phase0_v2_immutable.sha256.json"
        )
        if args.dry_run:
            context.finish("DRY_RUN", v2_hash_guard=immutable)
            return
        bundle = load_model_bundle(context.config)
        layers = [int(value) for value in context.config["geometry"]["candidate_layers"]]
        bank_manifest = (
            Path(args.bank_manifest).resolve()
            if args.bank_manifest
            else None
        )
        if args.stage == "smoke" and bank_manifest is None:
            existing = sorted(
                context.raw_dir.glob("geometry-v3-*/activation_bank_manifest.jsonl")
            )
            if existing:
                bank_manifest = existing[-1]
            else:
                bank_manifest = build_activation_bank(context, bundle, layers, limit=2)
        if args.stage in {"bank", "all"}:
            bank_manifest = build_activation_bank(
                context, bundle, layers, limit=args.limit
            )
            if args.stage == "bank":
                return
        if bank_manifest is None:
            bank_manifest = _latest_bank_manifest(context)
        outputs: dict[str, str] = {}
        if args.stage in {"spectrum", "all", "smoke"}:
            map_path, local_path = run_spectra(
                context,
                bundle,
                bank_manifest,
                shard_index=args.shard_index,
                shard_count=args.shard_count,
                smoke=args.stage == "smoke",
            )
            outputs["map_spectra"] = str(map_path.relative_to(context.root))
            outputs["local_spectra"] = str(local_path.relative_to(context.root))
            if args.stage == "smoke":
                outputs["candidate_smoke"] = str(
                    run_candidate_smoke(context, bundle, bank_manifest).relative_to(
                        context.root
                    )
                )
        if args.stage in {"pareto", "all"}:
            outputs["pareto"] = str(
                run_pareto(
                    context,
                    bundle,
                    bank_manifest,
                    limit=args.limit,
                    shard_index=args.shard_index,
                    shard_count=args.shard_count,
                ).relative_to(
                    context.root
                )
            )
        context.finish(
            "COMPLETED",
            v2_hash_guard=immutable,
            stage=args.stage,
            limit=args.limit,
            shard_index=args.shard_index,
            shard_count=args.shard_count,
            activation_bank_manifest=str(bank_manifest.relative_to(context.root)),
            outputs=outputs,
        )
    except KeyboardInterrupt:
        context.finish("FAILED", error="KeyboardInterrupt: run cancelled")
        raise
    except Exception as exc:
        context.finish("FAILED", error=f"{type(exc).__name__}: {exc}")
        raise


if __name__ == "__main__":
    main()
