"""Typed inspection, cloning, and channel-wise swapping of hybrid Qwen caches."""

from __future__ import annotations

import copy
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from typing import Any

import torch

KV_ATTRIBUTES = ("keys", "values")
RECURRENT_ATTRIBUTES = ("recurrent_states",)
CONV_ATTRIBUTES = ("conv_states",)
ALL_STATE_ATTRIBUTES = (*KV_ATTRIBUTES, *RECURRENT_ATTRIBUTES, *CONV_ATTRIBUTES)


@dataclass(frozen=True)
class CacheTensorSchema:
    layer: int
    layer_type: str
    cache_class: str
    component: str
    shape: tuple[int, ...]
    dtype: str
    device: str
    bytes: int
    token_axis: int | None
    token_dependence: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _tensor(layer: Any, attribute: str) -> torch.Tensor | None:
    value = getattr(layer, attribute, None)
    return value if isinstance(value, torch.Tensor) else None


def describe_cache(cache: Any, layer_types: list[str]) -> list[dict[str, Any]]:
    """Describe every initialized persistent tensor without treating cache as opaque."""

    rows: list[dict[str, Any]] = []
    for index, layer in enumerate(cache.layers):
        layer_type = str(layer_types[index])
        for attribute in ALL_STATE_ATTRIBUTES:
            value = _tensor(layer, attribute)
            if value is None:
                continue
            token_axis = -2 if attribute in KV_ATTRIBUTES else None
            dependence = (
                "explicit_token_sequence"
                if attribute in KV_ATTRIBUTES
                else "recurrent_summary_of_all_prefill_tokens"
                if attribute in RECURRENT_ATTRIBUTES
                else "last_conv_kernel_inputs"
            )
            rows.append(
                CacheTensorSchema(
                    layer=index,
                    layer_type=layer_type,
                    cache_class=type(layer).__name__,
                    component=attribute,
                    shape=tuple(int(item) for item in value.shape),
                    dtype=str(value.dtype).replace("torch.", ""),
                    device=str(value.device),
                    bytes=int(value.numel() * value.element_size()),
                    token_axis=token_axis,
                    token_dependence=dependence,
                ).to_dict()
            )
    return rows


def clone_hybrid_cache(cache: Any) -> Any:
    """Clone every tensor while retaining the Transformers cache layer classes."""

    output = copy.copy(cache)
    output.layers = []
    for source_layer in cache.layers:
        target_layer = copy.copy(source_layer)
        for name, value in vars(source_layer).items():
            if isinstance(value, torch.Tensor):
                setattr(target_layer, name, value.detach().clone())
            else:
                setattr(target_layer, name, copy.deepcopy(value))
        output.layers.append(target_layer)
    return output


def _copy_selected_kv(
    target: torch.Tensor,
    source: torch.Tensor,
    *,
    heads: Iterable[int] | None,
    positions: Iterable[int] | None,
) -> torch.Tensor:
    output = target.detach().clone()
    selected_heads = list(range(output.shape[1])) if heads is None else list(heads)
    selected_positions = (
        list(range(output.shape[-2])) if positions is None else list(positions)
    )
    for head in selected_heads:
        for position in selected_positions:
            output[:, int(head), int(position), :] = source[
                :, int(head), int(position), :
            ]
    return output


def make_chimeric_cache(
    clean: Any,
    perturbed: Any,
    *,
    kv_from_perturbed: bool = False,
    recurrent_from_perturbed: bool = False,
    conv_from_perturbed: bool = False,
    attention_layers: Iterable[int] | None = None,
    recurrent_layers: Iterable[int] | None = None,
    kv_heads: Iterable[int] | None = None,
    kv_positions: Iterable[int] | None = None,
) -> Any:
    """Build a 2x2 cache chimera with optional layer/head/token localization."""

    if len(clean.layers) != len(perturbed.layers):
        raise ValueError("cache layer count mismatch")
    output = clone_hybrid_cache(clean)
    attention = (
        set(range(len(clean.layers)))
        if attention_layers is None
        else {int(value) for value in attention_layers}
    )
    recurrent = (
        set(range(len(clean.layers)))
        if recurrent_layers is None
        else {int(value) for value in recurrent_layers}
    )
    for index, (target_layer, source_layer) in enumerate(
        zip(output.layers, perturbed.layers, strict=True)
    ):
        if kv_from_perturbed and index in attention:
            for attribute in KV_ATTRIBUTES:
                source = _tensor(source_layer, attribute)
                target = _tensor(target_layer, attribute)
                if source is not None and target is not None:
                    setattr(
                        target_layer,
                        attribute,
                        _copy_selected_kv(
                            target,
                            source,
                            heads=kv_heads,
                            positions=kv_positions,
                        ),
                    )
        if index in recurrent:
            if recurrent_from_perturbed:
                source = _tensor(source_layer, "recurrent_states")
                if source is not None:
                    target_layer.recurrent_states = source.detach().clone()
            if conv_from_perturbed:
                source = _tensor(source_layer, "conv_states")
                if source is not None:
                    target_layer.conv_states = source.detach().clone()
            if recurrent_from_perturbed or conv_from_perturbed:
                target_layer.has_previous_state = getattr(
                    source_layer, "has_previous_state", True
                )
    return output


def cache_component_differences(clean: Any, perturbed: Any) -> list[dict[str, Any]]:
    """Return component-level L2/RMS differences for auditing and localization."""

    rows: list[dict[str, Any]] = []
    for index, (left_layer, right_layer) in enumerate(
        zip(clean.layers, perturbed.layers, strict=True)
    ):
        for attribute in ALL_STATE_ATTRIBUTES:
            left = _tensor(left_layer, attribute)
            right = _tensor(right_layer, attribute)
            if left is None or right is None:
                continue
            delta = right.float() - left.float()
            rows.append(
                {
                    "layer": index,
                    "component": attribute,
                    "l2": float(torch.linalg.vector_norm(delta).item()),
                    "rms": float(torch.sqrt(torch.mean(delta.square())).item()),
                    "nonzero": int(torch.count_nonzero(delta).item()),
                    "elements": int(delta.numel()),
                }
            )
    return rows


def caches_exact(left: Any, right: Any, attributes: Iterable[str]) -> bool:
    """Check exact equality for selected cache components."""

    for left_layer, right_layer in zip(left.layers, right.layers, strict=True):
        for attribute in attributes:
            left_value = _tensor(left_layer, attribute)
            right_value = _tensor(right_layer, attribute)
            if (left_value is None) != (right_value is None):
                return False
            if (
                left_value is not None
                and right_value is not None
                and not torch.equal(left_value, right_value)
            ):
                return False
    return True
