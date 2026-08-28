import pandas as pd

from jclosure.experiments.dictionary_sensitivity import summarize_dictionary_effects


def test_dictionary_sensitivity_uses_only_common_valid_trials():
    rows = []
    for trial, prompt, valid_sizes in (
        ("a", "p1", {4096, 8192}),
        ("b", "p2", {4096}),
    ):
        for size in (4096, 8192):
            rows.append(
                {
                    "paired_trial_id": trial,
                    "dictionary_size": size,
                    "dictionary_hash": f"hash-{size}",
                    "valid": size in valid_sizes,
                    "condition": "non_j",
                    "clamp_condition": "single",
                    "prompt_id": prompt,
                    "metrics.js_divergence": size / 1_000_000,
                }
            )
    config = {
        "statistics": {"bootstrap_resamples": 10},
        "reproducibility": {"bootstrap_seed": 7},
    }
    summaries, common = summarize_dictionary_effects(
        pd.DataFrame(rows), dictionary_sizes=[4096, 8192], config=config
    )
    assert common == 1
    assert [row["common_valid_trials"] for row in summaries] == [1, 1]
    assert [row["valid_at_size"] for row in summaries] == [2, 1]
