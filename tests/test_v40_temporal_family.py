"""CPU checks for the descriptive V40 task-family temporal appendix."""
from scripts.audit_v40_temporal_family import channel_summary


def test_family_channel_summary_uses_state_unit_and_explicit_denominator():
    states = [{"state_id": "a"}, {"state_id": "b"}]
    rows = {
        ("a", 2, "BASE"): {"donor_minus_recipient_logit": -2.0},
        ("a", 2, "REC_only"): {"donor_minus_recipient_logit": 1.0,
                                "donor_recovered": True},
        ("b", 2, "BASE"): {"donor_minus_recipient_logit": -1.0},
        ("b", 2, "REC_only"): {"donor_minus_recipient_logit": 0.0,
                                "donor_recovered": False},
    }
    result = channel_summary(rows, states, 2, "REC_only", 42, 100)
    assert result["donor_recovered_qualified"] == 1
    assert result["qualified_denominator"] == 2
    assert result["median_margin_shift_vs_base"] == 2.0
    assert result["mean_margin_shift_vs_base"]["n"] == 2


def test_family_channel_summary_empty_subset_is_explicit():
    result = channel_summary({}, [], 8, "KV_only", 42, 100)
    assert result["qualified_denominator"] == 0
    assert result["median_margin_shift_vs_base"] is None
    assert result["mean_margin_shift_vs_base"]["ci95"] is None
