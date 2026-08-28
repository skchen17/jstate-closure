"""Versioned Phase 0 scoring and target-expansion utilities.

The v2 scorer follows the public Jacobian-lens evaluation description: ranks
are minimized over valid single-token target variants and layers, concept
fractions are averaged within items, and items are then weighted equally.
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from typing import Any

import pandas as pd
import torch

PROTOCOL_VERSION = "phase0_protocol_v2"

OPERATION_SYNONYMS: dict[str, tuple[str, ...]] = {
    "addition": ("+", "plus", "addition"),
    "subtraction": ("-", "minus", "subtraction"),
    "multiplication": ("*", "×", "times", "multiplication"),
    "division": ("/", "÷", "divide", "division"),
    "mod": ("%", "mod", "modulo"),
    "squared": ("²", "square", "squared"),
}

_ONES = (
    "zero",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
    "eleven",
    "twelve",
    "thirteen",
    "fourteen",
    "fifteen",
    "sixteen",
    "seventeen",
    "eighteen",
    "nineteen",
)
_TENS = ("", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety")
_INTEGER_RE = re.compile(r"^[+-]?\d+$")


def integer_to_words(value: int) -> str:
    """Return a deterministic US-English cardinal for integers up to 999,999."""

    if not -999_999 <= value <= 999_999:
        raise ValueError("integer synonym range is limited to [-999999, 999999]")
    if value < 0:
        return "minus " + integer_to_words(-value)
    if value < 20:
        return _ONES[value]
    if value < 100:
        tens, ones = divmod(value, 10)
        return _TENS[tens] if not ones else f"{_TENS[tens]}-{_ONES[ones]}"
    if value < 1_000:
        hundreds, remainder = divmod(value, 100)
        prefix = f"{_ONES[hundreds]} hundred"
        return prefix if not remainder else f"{prefix} {integer_to_words(remainder)}"
    thousands, remainder = divmod(value, 1_000)
    prefix = f"{integer_to_words(thousands)} thousand"
    return prefix if not remainder else f"{prefix} {integer_to_words(remainder)}"


def synonym_surfaces(concept: str, *, family: str) -> tuple[str, ...]:
    """Expand a canonical concept under the frozen v2 declared rules."""

    canonical = str(concept).strip()
    values: list[str] = [canonical]
    key = canonical.casefold()
    if family == "order_of_operations":
        if _INTEGER_RE.fullmatch(canonical):
            values.extend((str(int(canonical)), integer_to_words(int(canonical))))
        if key in OPERATION_SYNONYMS:
            values.extend(OPERATION_SYNONYMS[key])
    return tuple(dict.fromkeys(value for value in values if value))


def _normalized_surface(value: str) -> str:
    return " ".join(value.strip().casefold().split())


@dataclass(frozen=True)
class TokenCandidate:
    surface: str
    encoded_surface: str
    token_id: int
    decoded_surface: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def single_token_candidates(tokenizer: Any, surfaces: Iterable[str]) -> tuple[TokenCandidate, ...]:
    """Tokenize raw/leading-space variants and retain round-trip single tokens."""

    candidates: list[TokenCandidate] = []
    seen: set[int] = set()
    for surface in surfaces:
        for encoded_surface in (surface.strip(), " " + surface.strip()):
            ids = tokenizer.encode(encoded_surface, add_special_tokens=False)
            if len(ids) != 1:
                continue
            token_id = int(ids[0])
            decoded = tokenizer.decode(
                [token_id],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            if _normalized_surface(decoded) != _normalized_surface(surface):
                continue
            if token_id in seen:
                continue
            candidates.append(TokenCandidate(surface, encoded_surface, token_id, decoded))
            seen.add(token_id)
    return tuple(candidates)


def rank_candidates(
    logits: torch.Tensor, candidates: Iterable[TokenCandidate]
) -> tuple[int | None, TokenCandidate | None]:
    """Return 1-indexed best rank and the winning candidate."""

    choices = tuple(candidates)
    if not choices:
        return None, None
    scores = logits.float().reshape(-1)
    token_ids = torch.tensor([item.token_id for item in choices], device=scores.device)
    candidate_scores = scores[token_ids]
    best_score = candidate_scores.max()
    winners = [
        choices[index]
        for index in torch.nonzero(candidate_scores == best_score, as_tuple=False)
        .reshape(-1)
        .tolist()
    ]
    winner = min(winners, key=lambda item: item.token_id)
    rank = int(torch.sum(scores > best_score).item()) + 1
    return rank, winner


def _item_weighted(values: pd.DataFrame, value_col: str) -> float:
    if values.empty:
        return 0.0
    concept = (
        values.groupby(["example_id", "concept"], sort=False)[value_col]
        .first()
        .reset_index()
    )
    return float(concept.groupby("example_id", sort=False)[value_col].mean().mean())


def _method_summary(
    records: pd.DataFrame,
    *,
    method: str,
    layers: Iterable[int],
) -> dict[str, Any]:
    layer_set = {int(layer) for layer in layers}
    selected = records[
        (records["method"] == method)
        & records["layer"].isin(layer_set)
        & records["rank"].notna()
    ].copy()
    if selected.empty:
        return {
            "pass_at": {"1": 0.0, "5": 0.0, "10": 0.0},
            "per_layer": {},
            "strict_all_layers_sensitivity": {"1": 0.0, "5": 0.0, "10": 0.0},
            "best_layer": None,
            "item_count": 0,
            "concept_count": 0,
        }
    best = (
        selected.groupby(["example_id", "concept"], as_index=False, sort=False)["rank"]
        .min()
    )
    pass_at = {}
    for k in (1, 5, 10):
        best[f"hit{k}"] = best["rank"] <= k
        pass_at[str(k)] = _item_weighted(best, f"hit{k}")

    per_layer: dict[str, dict[str, float]] = {}
    for layer, layer_frame in selected.groupby("layer", sort=True):
        payload: dict[str, float] = {}
        for k in (1, 5, 10):
            working = layer_frame.copy()
            working[f"hit{k}"] = working["rank"] <= k
            payload[f"hit{k}"] = _item_weighted(working, f"hit{k}")
        per_layer[str(int(layer))] = payload
    strict = {
        str(k): min(payload[f"hit{k}"] for payload in per_layer.values())
        for k in (1, 5, 10)
    }
    best_layer = max(
        (int(layer) for layer in per_layer),
        key=lambda layer: (per_layer[str(layer)]["hit10"], -layer),
    )
    return {
        "pass_at": pass_at,
        "per_layer": per_layer,
        "strict_all_layers_sensitivity": strict,
        "best_layer": best_layer,
        "item_count": int(best["example_id"].nunique()),
        "concept_count": int(len(best)),
    }


def official_pass_summary(
    records: pd.DataFrame,
    *,
    layers: Iterable[int],
    require_position_16: bool = False,
    exclude_copied: bool = False,
) -> dict[str, Any]:
    """Aggregate v2 metrics without concept-count weighting across items."""

    selected = records.copy()
    if require_position_16:
        selected = selected[selected["position"] >= 16]
    if exclude_copied:
        selected = selected[~selected["copied_target"].astype(bool)]
    selected = selected[selected["tokenizable"].astype(bool)]
    families: dict[str, Any] = {}
    for family, frame in selected.groupby("family", sort=True):
        families[str(family)] = {
            method: _method_summary(frame, method=method, layers=layers)
            for method in ("jacobian", "logit")
        }
    return {
        "protocol_version": PROTOCOL_VERSION,
        "layers": sorted({int(layer) for layer in layers}),
        "require_position_16": require_position_16,
        "exclude_copied": exclude_copied,
        "families": families,
    }


def item_mrr_advantages(records: pd.DataFrame, *, layers: Iterable[int]) -> pd.DataFrame:
    """Return one equally weighted J-vs-logit MRR difference per item."""

    selected = records[
        records["layer"].isin({int(layer) for layer in layers})
        & records["tokenizable"].astype(bool)
        & records["rank"].notna()
    ]
    best = selected.pivot_table(
        index=["example_id", "family", "concept"],
        columns="method",
        values="rank",
        aggfunc="min",
    ).dropna(subset=["jacobian", "logit"])
    if best.empty:
        return pd.DataFrame(columns=["example_id", "family", "mrr_advantage"])
    best["mrr_advantage"] = 1.0 / best["jacobian"] - 1.0 / best["logit"]
    return (
        best.reset_index()
        .groupby(["example_id", "family"], as_index=False)["mrr_advantage"]
        .mean()
    )


def coverage_summary(records: pd.DataFrame) -> dict[str, int]:
    concept_rows = records.drop_duplicates(["example_id", "concept"])
    examples = records.drop_duplicates("example_id")
    return {
        "raw_examples": int(examples.shape[0]),
        "raw_concepts": int(concept_rows.shape[0]),
        "tokenization_excluded_concepts": int((~concept_rows["tokenizable"].astype(bool)).sum()),
        "copy_flagged_concepts": int(concept_rows["copied_target"].astype(bool).sum()),
        "official_main_copy_exclusions": 0,
        "position_lt16_examples": int((examples["position"] < 16).sum()),
        "official_main_examples": int(
            concept_rows[concept_rows["tokenizable"].astype(bool)]["example_id"].nunique()
        ),
        "position16_sensitivity_examples": int(
            concept_rows[
                concept_rows["tokenizable"].astype(bool) & (concept_rows["position"] >= 16)
            ]["example_id"].nunique()
        ),
    }


def mean_item_hit(records: pd.DataFrame, *, layer: int, family: str, k: int = 10) -> float:
    selected = records[
        (records["layer"] == layer)
        & (records["family"] == family)
        & (records["method"] == "jacobian")
        & records["tokenizable"].astype(bool)
        & records["rank"].notna()
    ].copy()
    selected["hit"] = selected["rank"] <= k
    return _item_weighted(selected, "hit")


def finite_float(value: Any, default: float = 0.0) -> float:
    numeric = float(value)
    return numeric if math.isfinite(numeric) else default
