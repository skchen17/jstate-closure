"""CPU guards for V40 response-blind task controls and formal readout."""
from jclosure.datasets_v40 import FAMILIES, make_task
from jclosure.experiments.design_v40_a4 import alternative_value, control_prompts
from jclosure.experiments.formal_v40_a4 import CONDITIONS


def test_third_natural_value_has_external_answer():
    for family in FAMILIES:
        task = make_task(family, 204001, 0)
        other = alternative_value(task)
        if family == "boolean_logic":
            assert other is None
            continue
        value, answer = other
        assert value not in (task["recipient_fork"], task["donor_fork"])
        assert answer not in (task["recipient_answer"], task["donor_answer"])


def test_negative_controls_preserve_task_label():
    for family in FAMILIES:
        task = make_task(family, 204002, 0)
        controls = control_prompts(task, ["2", "7"])
        for kind in ("surface_only", "irrelevant_history"):
            left, right = controls[kind]
            assert left != right
            assert task["recipient_prompt"] in left
            assert task["recipient_prompt"] in right


def test_frozen_condition_family():
    assert CONDITIONS == ("BASE", "REC_only", "Conv_only", "KV_only",
                          "REC_Conv", "REC_Conv_KV", "DONOR_NATIVE")
