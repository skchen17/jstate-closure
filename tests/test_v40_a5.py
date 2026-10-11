"""CPU guards for prospective REC trajectory grouping and factorial map."""
from types import SimpleNamespace

from jclosure.experiments.trajectory_v40_a5 import ORDER, group_layers
from scripts.analyze_v40_trajectory_a5 import interaction


def test_ordered_recurrent_thirds_are_disjoint():
    layers = [SimpleNamespace(layer_type="linear_attention" if i % 4 != 3 else "full_attention")
              for i in range(32)]
    groups = group_layers(SimpleNamespace(model=SimpleNamespace(layers=layers)))
    assert [len(groups[name]) for name in ("early", "middle", "late")] == [8, 8, 8]
    flattened = groups["early"] + groups["middle"] + groups["late"]
    assert flattened == [i for i in range(32) if i % 4 != 3]


def test_factorial_order_is_complete():
    assert ORDER == ("R000", "R100", "R010", "R001",
                     "R110", "R101", "R011", "R111")


def test_three_way_interaction_is_zero_for_additive_values():
    values = {"R000": 0, "R100": 2, "R010": 3, "R001": 5,
              "R110": 5, "R101": 7, "R011": 8, "R111": 10}
    rows = {label: {"donor_minus_recipient_logit": value}
            for label, value in values.items()}
    assert interaction(rows) == 0
    rows["R111"]["donor_minus_recipient_logit"] = 12
    assert interaction(rows) == 2
