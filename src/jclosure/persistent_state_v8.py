"""Architecture-aligned state extraction and factorial cache composition for v8."""

from __future__ import annotations

import itertools
from typing import Any

import numpy as np
import torch

from jclosure.cache_v7 import make_chimeric_cache

ATOM_ORDER = ("rec_matrix_all", "conv_all", "kv_l27_h3", "kv_other")


def factorial_conditions() -> dict[str, int]:
    """Return the preregistered 2^4 component design."""

    return {
        "+".join(name for bit, name in enumerate(ATOM_ORDER) if mask & (1 << bit))
        or "none": mask
        for mask in range(16)
    }


def named_raw_conditions() -> dict[str, str]:
    return {
        "R0": "none",
        "R1": "top_rec_l30",
        "R2": "top_rec_l28_l29_l30",
        "R3": "conv_all",
        "R4": "rec_matrix_all+conv_all",
        "R5": "kv_l27_h3",
        "R6": "rec_matrix_all+conv_all+kv_l27_h3",
        "R7": "+".join(ATOM_ORDER),
    }


def _copy_kv_complement(
    output: Any,
    clean: Any,
    perturbed: Any,
    *,
    attention_layers: list[int],
    primary_layer: int,
    primary_head: int,
    primary_position: int,
    keep_primary_perturbed: bool,
) -> None:
    for layer in attention_layers:
        for attribute in ("keys", "values"):
            source = getattr(perturbed.layers[layer], attribute, None)
            if not isinstance(source, torch.Tensor):
                continue
            value = source.detach().clone()
            if layer == primary_layer and not keep_primary_perturbed:
                original = getattr(clean.layers[layer], attribute)
                value[:, primary_head, primary_position, :] = original[
                    :, primary_head, primary_position, :
                ]
            setattr(output.layers[layer], attribute, value)


def compose_condition(
    clean: Any,
    perturbed: Any,
    condition: str,
    *,
    recurrent_layers: list[int],
    attention_layers: list[int],
    primary_layer: int,
    primary_head: int,
    primary_position: int,
) -> Any:
    """Compose a raw state condition without mutating either source cache."""

    if condition == "top_rec_l30":
        return make_chimeric_cache(
            clean,
            perturbed,
            recurrent_from_perturbed=True,
            conv_from_perturbed=False,
            recurrent_layers=[30],
        )
    if condition == "top_rec_l28_l29_l30":
        return make_chimeric_cache(
            clean,
            perturbed,
            recurrent_from_perturbed=True,
            conv_from_perturbed=False,
            recurrent_layers=[28, 29, 30],
        )
    names = set() if condition == "none" else set(condition.split("+"))
    output = make_chimeric_cache(
        clean,
        perturbed,
        recurrent_from_perturbed="rec_matrix_all" in names,
        conv_from_perturbed="conv_all" in names,
        recurrent_layers=recurrent_layers,
        kv_from_perturbed="kv_l27_h3" in names,
        attention_layers=[primary_layer],
        kv_heads=[primary_head],
        kv_positions=[primary_position],
    )
    if "kv_other" in names:
        _copy_kv_complement(
            output,
            clean,
            perturbed,
            attention_layers=attention_layers,
            primary_layer=primary_layer,
            primary_head=primary_head,
            primary_position=primary_position,
            keep_primary_perturbed="kv_l27_h3" in names,
        )
    return output


def extract_architecture_state(
    cache: Any,
    *,
    recurrent_layers: list[int],
    attention_layers: list[int],
) -> dict[str, Any]:
    """Extract exact clean/perturbed tensors, retaining KV token axes."""

    recurrent = torch.stack(
        [
            cache.layers[layer].recurrent_states.detach().cpu().to(torch.bfloat16)[0]
            for layer in recurrent_layers
        ]
    )
    conv = torch.stack(
        [
            cache.layers[layer].conv_states.detach().cpu().to(torch.bfloat16)[0]
            for layer in recurrent_layers
        ]
    )
    kv = {}
    for layer in attention_layers:
        kv[str(layer)] = {
            name: getattr(cache.layers[layer], name)
            .detach()
            .cpu()
            .to(torch.bfloat16)[0]
            for name in ("keys", "values")
        }
    return {"recurrent": recurrent, "conv": conv, "kv": kv}


def save_state_shard(path: Any, rows: list[dict[str, Any]]) -> None:
    """Serialize ignored large tensors as an auditable exact BF16 shard."""

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"format": "persistent_state_v8_bf16", "rows": rows}, path)


def atom_subsets() -> list[tuple[str, ...]]:
    return [
        tuple(ATOM_ORDER[index] for index in range(4) if mask & (1 << index))
        for mask in range(16)
    ]


def mobius_interactions(effects: dict[int, np.ndarray]) -> dict[int, np.ndarray]:
    """Möbius-decompose all main and higher-order factorial effects."""

    if set(effects) != set(range(16)):
        raise ValueError("complete four-atom factorial design required")
    interactions: dict[int, np.ndarray] = {}
    for mask in range(16):
        value = np.asarray(effects[mask], dtype=np.float64).copy()
        proper = [
            subset
            for subset in range(mask)
            if subset & mask == subset and subset != mask
        ]
        for subset in proper:
            value -= interactions[subset]
        interactions[mask] = value
    return interactions


def leave_one_out_masks() -> dict[str, int]:
    full = (1 << len(ATOM_ORDER)) - 1
    return {name: full ^ (1 << index) for index, name in enumerate(ATOM_ORDER)}


def pairwise_masks() -> dict[str, int]:
    return {
        f"{ATOM_ORDER[left]}+{ATOM_ORDER[right]}": (1 << left) | (1 << right)
        for left, right in itertools.combinations(range(len(ATOM_ORDER)), 2)
    }
