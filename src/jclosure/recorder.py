"""Residual-stream recording and editing hooks."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from typing import Any

import torch
from torch import nn


def residual_tensor(output: Any) -> torch.Tensor:
    if torch.is_tensor(output):
        return output
    if isinstance(output, (tuple, list)) and output and torch.is_tensor(output[0]):
        return output[0]
    raise TypeError(f"unsupported block output type: {type(output).__name__}")


def replace_residual(output: Any, replacement: torch.Tensor) -> Any:
    if torch.is_tensor(output):
        return replacement
    if isinstance(output, tuple):
        return (replacement, *output[1:])
    if isinstance(output, list):
        return [replacement, *output[1:]]
    raise TypeError(f"unsupported block output type: {type(output).__name__}")


class ActivationRecorder:
    """Record selected block outputs and always remove hooks on exit."""

    def __init__(
        self,
        blocks: Sequence[nn.Module],
        at: Iterable[int],
        *,
        clone: bool = True,
        detach: bool = True,
    ) -> None:
        self.blocks = blocks
        self.indices = tuple(sorted(set(int(index) for index in at)))
        self.clone = clone
        self.detach = detach
        self.activations: dict[int, torch.Tensor] = {}
        self._handles: list[torch.utils.hooks.RemovableHandle] = []

    def _hook(self, index: int) -> Callable[..., None]:
        def hook(module: nn.Module, inputs: tuple[Any, ...], output: Any) -> None:
            del module, inputs
            value = residual_tensor(output)
            if self.detach:
                value = value.detach()
            if self.clone:
                value = value.clone()
            self.activations[index] = value

        return hook

    def __enter__(self) -> ActivationRecorder:
        try:
            for index in self.indices:
                self._handles.append(
                    self.blocks[index].register_forward_hook(self._hook(index))
                )
        except Exception:
            self.remove()
            raise
        return self

    def remove(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles = []

    def __exit__(self, *exc: Any) -> None:
        self.remove()


class ResidualEditor:
    """Apply one deterministic transform per selected block output."""

    def __init__(
        self,
        blocks: Sequence[nn.Module],
        transforms: dict[int, Callable[[torch.Tensor, int], torch.Tensor]],
    ) -> None:
        self.blocks = blocks
        self.transforms = dict(sorted((int(k), v) for k, v in transforms.items()))
        self._handles: list[torch.utils.hooks.RemovableHandle] = []

    def _hook(self, index: int) -> Callable[..., Any]:
        transform = self.transforms[index]

        def hook(module: nn.Module, inputs: tuple[Any, ...], output: Any) -> Any:
            del module, inputs
            current = residual_tensor(output)
            updated = transform(current, index)
            if updated.shape != current.shape:
                raise ValueError(
                    f"layer {index} transform changed shape {current.shape} -> {updated.shape}"
                )
            if updated.device != current.device:
                raise ValueError(f"layer {index} transform changed device")
            return replace_residual(output, updated)

        return hook

    def __enter__(self) -> ResidualEditor:
        try:
            for index in self.transforms:
                self._handles.append(
                    self.blocks[index].register_forward_hook(self._hook(index))
                )
        except Exception:
            self.remove()
            raise
        return self

    def remove(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles = []

    def __exit__(self, *exc: Any) -> None:
        self.remove()

