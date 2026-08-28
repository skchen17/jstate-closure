"""Stable record schemas shared by experiment runners."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from jclosure.provenance import SCHEMA_VERSION


@dataclass(frozen=True)
class TrialRecord:
    run_id: str
    prompt_id: str
    task_family: str
    layer: int
    position: int
    intervention: str
    valid: bool
    metrics: dict[str, float | int | bool | None]
    seed: int
    template_id: str | None = None
    source: str | None = None
    strength: float | None = None
    donor_id: str | None = None
    exclusion_reason: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    schema_version: int = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

