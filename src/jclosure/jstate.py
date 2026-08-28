"""Canonical sparse and dense J-state representations."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from jclosure.decomposition import DecompositionResult, gradient_pursuit


@dataclass(frozen=True)
class ConceptVocabulary:
    token_ids: tuple[int, ...]
    surfaces: tuple[str, ...]
    model_id: str | None = None
    model_revision: str | None = None

    def __post_init__(self) -> None:
        if len(self.token_ids) != len(self.surfaces):
            raise ValueError("token_ids and surfaces must have equal length")
        if len(set(self.token_ids)) != len(self.token_ids):
            raise ValueError("concept token IDs must be unique")

    def to_json(self, path: str | Path) -> None:
        payload = {
            "schema_version": 1,
            "model_id": self.model_id,
            "model_revision": self.model_revision,
            "concepts": [
                {"token_id": token_id, "surface": surface}
                for token_id, surface in zip(self.token_ids, self.surfaces, strict=True)
            ],
        }
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    @classmethod
    def from_json(cls, path: str | Path) -> ConceptVocabulary:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        concepts = payload["concepts"]
        return cls(
            token_ids=tuple(int(item["token_id"]) for item in concepts),
            surfaces=tuple(str(item["surface"]) for item in concepts),
            model_id=payload.get("model_id"),
            model_revision=payload.get("model_revision"),
        )


def _word_like(surface: str) -> bool:
    text = surface.strip()
    if not text or len(text) > 32 or any(char.isspace() for char in text):
        return False
    if text.isdecimal():
        return True
    allowed_punctuation = {"'", "-"}
    return len(text) >= 2 and any(char.isalpha() for char in text) and all(
        char.isalpha() or char in allowed_punctuation for char in text
    )


def build_concept_vocabulary(
    tokenizer: Any,
    *,
    size: int = 4096,
    mandatory_surfaces: tuple[str, ...] | list[str] = (),
    model_id: str | None = None,
    model_revision: str | None = None,
) -> ConceptVocabulary:
    """Select a deterministic set of word-like, single-token concepts."""

    if size <= 0:
        raise ValueError("size must be positive")
    mandatory_ids: set[int] = set()
    for surface in mandatory_surfaces:
        encoded = tokenizer.encode(surface, add_special_tokens=False)
        if len(encoded) == 1:
            mandatory_ids.add(int(encoded[0]))
        encoded_space = tokenizer.encode(" " + surface, add_special_tokens=False)
        if len(encoded_space) == 1:
            mandatory_ids.add(int(encoded_space[0]))

    special_ids = set(getattr(tokenizer, "all_special_ids", []))
    candidates: list[tuple[int, str]] = []
    for token_id in sorted(set(tokenizer.get_vocab().values())):
        if token_id in special_ids:
            continue
        surface = tokenizer.decode(
            [token_id], skip_special_tokens=True, clean_up_tokenization_spaces=False
        ).strip()
        if token_id in mandatory_ids or _word_like(surface):
            candidates.append((int(token_id), surface))

    selected: list[tuple[int, str]] = []
    by_id = {token_id: surface for token_id, surface in candidates}
    for token_id in sorted(mandatory_ids):
        if token_id in by_id:
            selected.append((token_id, by_id[token_id]))
    seen = {token_id for token_id, _ in selected}
    for item in candidates:
        if item[0] not in seen:
            selected.append(item)
            seen.add(item[0])
        if len(selected) >= size:
            break
    if len(selected) < size:
        raise ValueError(f"only {len(selected)} suitable concepts found; requested {size}")
    selected = selected[:size]
    return ConceptVocabulary(
        token_ids=tuple(item[0] for item in selected),
        surfaces=tuple(item[1] for item in selected),
        model_id=model_id,
        model_revision=model_revision,
    )


@dataclass(frozen=True)
class JState:
    layer: int
    position: int
    sparse_token_ids: torch.Tensor
    sparse_coefficients: torch.Tensor
    dense_scores: torch.Tensor
    raw_dense_scores: torch.Tensor
    residual_rms: float
    activation_norm: float
    reconstruction_error: float
    variance_explained: float

    def cpu(self) -> JState:
        return JState(
            layer=self.layer,
            position=self.position,
            sparse_token_ids=self.sparse_token_ids.detach().cpu(),
            sparse_coefficients=self.sparse_coefficients.detach().cpu(),
            dense_scores=self.dense_scores.detach().cpu(),
            raw_dense_scores=self.raw_dense_scores.detach().cpu(),
            residual_rms=self.residual_rms,
            activation_norm=self.activation_norm,
            reconstruction_error=self.reconstruction_error,
            variance_explained=self.variance_explained,
        )


class JStateEncoder:
    """Encode residual vectors with fixed per-layer J direction dictionaries."""

    def __init__(
        self,
        directions: dict[int, torch.Tensor],
        vocabulary: ConceptVocabulary,
        *,
        k: int = 25,
        raw_directions: dict[int, torch.Tensor] | None = None,
    ) -> None:
        if not directions:
            raise ValueError("at least one layer dictionary is required")
        self.vocabulary = vocabulary
        self.k = k
        self.directions = {
            int(layer): F.normalize(value.float(), dim=-1)
            for layer, value in directions.items()
        }
        self.raw_directions = raw_directions or directions
        for layer, dictionary in self.directions.items():
            if dictionary.shape[0] != len(vocabulary.token_ids):
                raise ValueError(f"layer {layer} dictionary and vocabulary disagree")

    @classmethod
    def from_lens(
        cls,
        lens: Any,
        unembedding_weight: torch.Tensor,
        vocabulary: ConceptVocabulary,
        *,
        k: int = 25,
    ) -> JStateEncoder:
        ids = torch.tensor(vocabulary.token_ids, dtype=torch.long)
        selected_unembedding = unembedding_weight.detach().float().cpu()[ids]
        raw: dict[int, torch.Tensor] = {}
        for layer, jacobian in lens.jacobians.items():
            raw[int(layer)] = selected_unembedding @ jacobian.detach().float().cpu()
        return cls(raw, vocabulary, k=k, raw_directions=raw)

    def dictionary(self, layer: int, device: torch.device | str | None = None) -> torch.Tensor:
        dictionary = self.directions[int(layer)]
        return dictionary if device is None else dictionary.to(device)

    def encode(self, h: torch.Tensor, layer: int, *, position: int = -1) -> JState:
        if h.ndim != 1:
            raise ValueError("J-state encoder expects one residual vector")
        dictionary = self.dictionary(layer, h.device)
        decomposition = gradient_pursuit(h, dictionary, k=self.k)
        raw_dictionary = self.raw_directions[int(layer)].to(h.device).float()
        raw_scores = raw_dictionary @ h.float()
        centered = raw_scores - raw_scores.mean()
        normalized = F.normalize(centered, dim=0, eps=1e-12)
        atom_ids = torch.tensor(
            [self.vocabulary.token_ids[int(index)] for index in decomposition.atom_indices],
            dtype=torch.long,
            device=h.device,
        )
        return JState(
            layer=int(layer),
            position=int(position),
            sparse_token_ids=atom_ids,
            sparse_coefficients=decomposition.coefficients.to(h.dtype),
            dense_scores=normalized.to(h.dtype),
            raw_dense_scores=raw_scores.to(h.dtype),
            residual_rms=float(torch.sqrt(torch.mean(decomposition.remainder.float() ** 2))),
            activation_norm=float(torch.linalg.vector_norm(h.float())),
            reconstruction_error=decomposition.reconstruction_error,
            variance_explained=decomposition.variance_explained,
        )

    def decompose(self, h: torch.Tensor, layer: int) -> DecompositionResult:
        return gradient_pursuit(h, self.dictionary(layer, h.device), k=self.k)


def encode_jstate(
    h: torch.Tensor, layer: int, encoder: JStateEncoder, *, position: int = -1
) -> JState:
    return encoder.encode(h, layer, position=position)


def _top_overlap(a: JState, b: JState, k: int = 10) -> float:
    k = min(k, a.raw_dense_scores.numel(), b.raw_dense_scores.numel())
    a_top = set(torch.topk(a.raw_dense_scores.float(), k).indices.tolist())
    b_top = set(torch.topk(b.raw_dense_scores.float(), k).indices.tolist())
    return len(a_top & b_top) / k


def _weighted_sparse_jaccard(a: JState, b: JState) -> float:
    a_map = {
        int(token): float(coef)
        for token, coef in zip(a.sparse_token_ids, a.sparse_coefficients, strict=True)
    }
    b_map = {
        int(token): float(coef)
        for token, coef in zip(b.sparse_token_ids, b.sparse_coefficients, strict=True)
    }
    keys = a_map.keys() | b_map.keys()
    denominator = sum(max(a_map.get(key, 0.0), b_map.get(key, 0.0)) for key in keys)
    if denominator <= 1e-20:
        return 1.0
    numerator = sum(min(a_map.get(key, 0.0), b_map.get(key, 0.0)) for key in keys)
    return numerator / denominator


def jstate_similarity(a: JState, b: JState, metric: str = "dense_cosine") -> float:
    if a.dense_scores.shape != b.dense_scores.shape:
        raise ValueError("J-states use different dense concept vocabularies")
    if metric == "dense_cosine":
        return float(
            F.cosine_similarity(
                a.dense_scores.float().unsqueeze(0),
                b.dense_scores.float().unsqueeze(0),
            ).item()
        )
    if metric == "top10_overlap":
        return _top_overlap(a, b, 10)
    if metric == "sparse_weighted_jaccard":
        return _weighted_sparse_jaccard(a, b)
    raise ValueError(f"unknown J-state similarity metric: {metric}")


def jstate_distance(a: JState, b: JState, metric: str = "dense_cosine") -> float:
    if metric in {"dense_cosine", "top10_overlap", "sparse_weighted_jaccard"}:
        return 1.0 - jstate_similarity(a, b, metric)
    if metric == "dense_l2":
        return float(torch.linalg.vector_norm(a.dense_scores.float() - b.dense_scores.float()))
    if metric == "rms_log_ratio":
        return abs(math.log((a.residual_rms + 1e-12) / (b.residual_rms + 1e-12)))
    raise ValueError(f"unknown J-state distance metric: {metric}")

