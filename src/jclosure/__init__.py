"""J-state closure measurement and reduced-order modeling tools."""

from jclosure.clamp import (
    ClampResult,
    ClampThresholds,
    MultiPositionClampResult,
    one_shot_clamp,
    one_shot_clamp_positions,
)
from jclosure.decomposition import DecompositionResult, gradient_pursuit
from jclosure.interventions import InterventionSpec
from jclosure.jstate import JState, JStateEncoder, encode_jstate

__all__ = [
    "ClampResult",
    "ClampThresholds",
    "DecompositionResult",
    "InterventionSpec",
    "JState",
    "JStateEncoder",
    "encode_jstate",
    "gradient_pursuit",
    "one_shot_clamp",
    "one_shot_clamp_positions",
    "MultiPositionClampResult",
]

__version__ = "0.1.0"
