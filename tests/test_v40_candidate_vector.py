"""CPU checks for the restricted secondary V40 logit-vector fidelity metric."""
import pytest

from scripts.audit_v40_candidate_vector import gap_closed


def test_candidate_gap_closed_has_native_endpoints_and_offset_invariance():
    base = {"A": 1.0, "B": 0.0, "C": -1.0}
    donor = {"A": -1.0, "B": 0.0, "C": 1.0}
    halfway = {"A": 0.0, "B": 0.0, "C": 0.0}
    assert gap_closed(base, donor, base) == pytest.approx(0.0)
    assert gap_closed(base, donor, donor) == pytest.approx(1.0)
    assert gap_closed(base, donor, halfway) == pytest.approx(0.5)
    shifted = {key: value + 50.0 for key, value in halfway.items()}
    assert gap_closed(base, donor, shifted) == pytest.approx(0.5)


def test_candidate_gap_closed_requires_identifiable_native_gap_and_same_alphabet():
    assert gap_closed({"0": 1.0, "1": 2.0}, {"0": 2.0, "1": 3.0},
                      {"0": 1.0, "1": 2.0}) is None
    with pytest.raises(RuntimeError, match="alphabet drift"):
        gap_closed({"0": 1.0}, {"0": 2.0}, {"1": 1.0})
