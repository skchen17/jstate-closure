import json

import pandas as pd
import pytest
import torch

from jclosure.datasets import (
    fresh_probe_swap_multihop,
    generate_phase0_order_ops_holdout,
    normalize_prompt,
    upstream_multihop,
    upstream_order_ops,
)
from jclosure.phase0 import (
    integer_to_words,
    official_pass_summary,
    rank_candidates,
    single_token_candidates,
    synonym_surfaces,
)
from jclosure.protocol import (
    build_protocol_freeze,
    verify_protocol_freeze,
    write_protocol_freeze,
)


class TinyTokenizer:
    def __init__(self):
        surfaces = [
            "5",
            "five",
            "+",
            "plus",
            "addition",
            "×",
            "times",
            "multiplication",
        ]
        self.lookup = {surface: index for index, surface in enumerate(surfaces)}
        self.reverse = {index: surface for surface, index in self.lookup.items()}

    def encode(self, value, add_special_tokens=False):
        del add_special_tokens
        normalized = value.strip()
        return [self.lookup[normalized]] if normalized in self.lookup else [100, 101]

    def decode(self, ids, **kwargs):
        del kwargs
        return self.reverse[int(ids[0])]


def _records():
    rows = []
    # Item a has two concepts: one passes and one fails. Item b passes both.
    ranks = {
        ("a", "x", 0): 1,
        ("a", "x", 1): 20,
        ("a", "y", 0): 20,
        ("a", "y", 1): 20,
        ("b", "x", 0): 20,
        ("b", "x", 1): 5,
        ("b", "y", 0): 2,
        ("b", "y", 1): 2,
    }
    for (example, concept, layer), rank in ranks.items():
        for method in ("jacobian", "logit"):
            rows.append(
                {
                    "example_id": example,
                    "family": "factual_two_hop",
                    "concept": concept,
                    "layer": layer,
                    "method": method,
                    "rank": rank if method == "jacobian" else rank + 10,
                    "position": 8 if example == "a" else 20,
                    "copied_target": example == "a",
                    "tokenizable": True,
                }
            )
    return pd.DataFrame(rows)


def test_official_pass_uses_best_layer_and_item_weighting():
    summary = official_pass_summary(_records(), layers=[0, 1])
    jacobian = summary["families"]["factual_two_hop"]["jacobian"]
    # item a=1/2, item b=2/2, so item-weighted pass@10 is .75.
    assert jacobian["pass_at"]["10"] == 0.75
    assert jacobian["strict_all_layers_sensitivity"]["10"] == 0.5
    assert jacobian["best_layer"] == 0


def test_position_and_copy_are_sensitivities_not_main_exclusions():
    main = official_pass_summary(_records(), layers=[0, 1])
    position = official_pass_summary(_records(), layers=[0, 1], require_position_16=True)
    copied = official_pass_summary(_records(), layers=[0, 1], exclude_copied=True)
    assert main["families"]["factual_two_hop"]["jacobian"]["item_count"] == 2
    assert position["families"]["factual_two_hop"]["jacobian"]["item_count"] == 1
    assert copied["families"]["factual_two_hop"]["jacobian"]["item_count"] == 1


def test_synonym_expansion_and_best_single_token_rank():
    tokenizer = TinyTokenizer()
    surfaces = synonym_surfaces("5", family="order_of_operations")
    assert surfaces == ("5", "five")
    candidates = single_token_candidates(tokenizer, surfaces)
    assert {item.token_id for item in candidates} == {0, 1}
    logits = torch.zeros(8)
    logits[1] = 5
    rank, winner = rank_candidates(logits, candidates)
    assert rank == 1
    assert winner is not None and winner.surface == "five"
    assert integer_to_words(115) == "one hundred fifteen"
    assert synonym_surfaces("multiplication", family="order_of_operations") == (
        "multiplication",
        "*",
        "×",
        "times",
    )


def test_fresh_holdout_is_deterministic_and_nonoverlapping():
    root = "data/upstream/anthropic"
    first = fresh_probe_swap_multihop(root)
    second = fresh_probe_swap_multihop(root)
    assert first == second
    assert len(first) == 59
    calibration = {normalize_prompt(item.prompt) for item in upstream_multihop(root)}
    assert calibration.isdisjoint(normalize_prompt(item.prompt) for item in first)
    order_calibration = {normalize_prompt(item.prompt) for item in upstream_order_ops(root)}
    generated = generate_phase0_order_ops_holdout(
        256, seed=20260828, calibration_prompts=order_calibration
    )
    assert generated == generate_phase0_order_ops_holdout(
        256, seed=20260828, calibration_prompts=order_calibration
    )
    assert order_calibration.isdisjoint(normalize_prompt(item.prompt) for item in generated)
    counts = pd.Series(item.intermediates[1] for item in generated).value_counts()
    assert counts.max() - counts.min() <= 1


def test_freeze_manifest_detects_changed_file(tmp_path):
    root = tmp_path
    config = root / "config.yaml"
    tracked = root / "tracked.py"
    config.write_text(
        json.dumps(
            {
                "model": {},
                "lens": {},
                "jstate": {"k": 1, "concept_vocab_size": 16},
                "reproducibility": {},
                "outputs": {},
            }
        ),
        encoding="utf-8",
    )
    tracked.write_text("frozen\n", encoding="utf-8")
    manifest = build_protocol_freeze(
        root=root,
        config_path=config,
        workspace_band=[1, 2, 3],
        positive_control_layer=2,
        tracked_files=[tracked],
        thresholds={"hit10": 0.2},
        exclusion_policy={"main": "all_valid"},
    )
    path = root / "freeze.json"
    write_protocol_freeze(path, manifest)
    verify_protocol_freeze(path, root=root)
    tracked.write_text("changed\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="changed"):
        verify_protocol_freeze(path, root=root)
