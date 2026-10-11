"""CPU checks for V40 state-unit uncertainty estimates."""
from scripts.analyze_v40_a4 import bootstrap_mean


def test_bootstrap_mean_is_deterministic():
    left = bootstrap_mean([0.0, 1.0, 1.0, 0.0], 40, 100)
    right = bootstrap_mean([0.0, 1.0, 1.0, 0.0], 40, 100)
    assert left == right
    assert left["n"] == 4
    assert left["mean"] == .5
    assert left["ci95"][0] <= .5 <= left["ci95"][1]


def test_bootstrap_empty_is_explicit():
    assert bootstrap_mean([], 40, 100) == {"mean": None, "ci95": None, "n": 0}
