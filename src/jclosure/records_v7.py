"""Additive records for architecture-aligned persistent-state protocol v7."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

SCHEMA_VERSION_V7 = 9
PROTOCOL_V7 = "persistent_channel_attribution_protocol_v7"


@dataclass(frozen=True)
class CacheRestoreRecord:
    run_id: str
    prompt_id: str
    family: str
    token_id: int
    passed: bool
    logits_max_abs: float
    hidden_max_abs: float
    j_max_abs: float
    recurrent_exact: bool
    conv_exact: bool
    kv_exact: bool
    metadata: dict[str, Any] = field(default_factory=dict)
    exclusion_reason: str | None = None
    schema_version: int = SCHEMA_VERSION_V7
    protocol_version: str = PROTOCOL_V7

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ChannelAttributionRecord:
    run_id: str
    base_trial_id: str
    prompt_id: str
    family: str
    split: str
    condition: str
    valid: bool
    cache_component: str
    metrics: dict[str, Any]
    cache_differences: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    exclusion_reason: str | None = None
    schema_version: int = SCHEMA_VERSION_V7
    protocol_version: str = PROTOCOL_V7

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ChannelCompressionRecord:
    run_id: str
    channel: str
    method: str
    dimension: int
    split: str
    predictive_gap_closed: float
    causal_gap_closed: float
    conditional_residual_gain: float
    causal_direction_cosine: float
    magnitude_ratio: float
    semantic_delta_agreement: float
    output_sign_agreement: float
    metadata: dict[str, Any] = field(default_factory=dict)
    schema_version: int = SCHEMA_VERSION_V7
    protocol_version: str = PROTOCOL_V7

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
