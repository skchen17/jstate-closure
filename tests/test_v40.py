"""CPU-only V40 task and encoded-fork safeguards."""
from jclosure.datasets_v40 import FAMILIES, generate, make_task
from jclosure.experiments.design_v40 import single_token_fork
from jclosure.experiments.sample_pool_v40_a1 import select_tasks


def test_external_labels_and_only_one_textual_fork():
    for family in FAMILIES:
        row = make_task(family, 204002, 0)
        assert row["recipient_answer"] != row["donor_answer"]
        assert row["recipient_prompt"] == row["prefix"] + row["recipient_fork"] + row["suffix"]
        assert row["donor_prompt"] == row["prefix"] + row["donor_fork"] + row["suffix"]
        assert row["labels_generated_before_model_execution"]


def test_pool_uniqueness_and_determinism():
    blocked = set()
    first = {f: generate(f, 36, 204002 + i * 100_003, blocked)
             for i, f in enumerate(FAMILIES)}
    assert sum(map(len, first.values())) == 144
    assert len(blocked) == 144
    assert first["boolean_logic"][0] == make_task("boolean_logic", 204002 + 3 * 100_003, 0)


def test_one_token_fork_audit():
    result = single_token_fork([1, 2, 3, 4], [1, 9, 3, 4])
    assert result["eligible"] and result["fork_index"] == 1
    assert not single_token_fork([1, 2], [1, 2, 3])["eligible"]
    assert not single_token_fork([1, 2, 3], [9, 8, 3])["eligible"]
    assert not single_token_fork([1, 2, 3], [1, 2, 9])["eligible"]


def test_amended_pool_rejects_prompt_collisions():
    blocked, prompts = set(), set()
    rows = select_tasks("state_transition", 36, 204002, blocked, prompts)
    assert len(rows) == 36 and len(prompts) == 72
    assert len({x["recipient_prompt"] for x in rows}) == 36
    assert len({x["donor_prompt"] for x in rows}) == 36
