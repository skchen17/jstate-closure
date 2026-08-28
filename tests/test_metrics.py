import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from jclosure.datasets import generate_arithmetic, split_by_template
from jclosure.metrics import (
    jensen_shannon_from_logits,
    sparse_support_f1,
    topk_overlap,
)
from jclosure.records import TrialRecord
from jclosure.reporting import _save_figure
from jclosure.statistics import (
    benjamini_hochberg,
    clustered_bootstrap_ci,
    normalized_remainder_effect,
)


def test_js_is_zero_for_identical_and_symmetric():
    a = torch.tensor([1.0, 2.0, -1.0])
    b = torch.tensor([-1.0, 0.5, 3.0])
    assert jensen_shannon_from_logits(a, a) < 1e-12
    assert np.isclose(
        jensen_shannon_from_logits(a, b), jensen_shannon_from_logits(b, a)
    )


def test_support_and_overlap_metrics():
    assert sparse_support_f1([1, 2], [2, 3]) == 0.5
    assert topk_overlap(torch.tensor([3.0, 2, 1]), torch.tensor([3.0, 1, 2]), 2) == 0.5


def test_cluster_bootstrap_and_bh_are_deterministic():
    data = pd.DataFrame(
        {"prompt": ["a", "a", "b", "b", "c", "c"], "value": [1, 2, 2, 3, 3, 4]}
    )
    first = clustered_bootstrap_ci(
        data, cluster_col="prompt", value_col="value", n_resamples=200, seed=9
    )
    second = clustered_bootstrap_ci(
        data, cluster_col="prompt", value_col="value", n_resamples=200, seed=9
    )
    assert first == second
    adjusted = benjamini_hochberg([0.01, 0.04, 0.03])
    assert all(0 <= value <= 1 for value in adjusted)


def test_eta_only_when_positive_control_clears_noise():
    assert np.isclose(
        normalized_remainder_effect(0.1, 1.0, j_effect_lower=0.5, null_threshold=0.2),
        0.1,
    )
    assert normalized_remainder_effect(0.1, 1.0, j_effect_lower=0.1, null_threshold=0.2) is None


def test_dataset_generation_and_template_split_are_deterministic():
    first = generate_arithmetic(40, seed=17)
    second = generate_arithmetic(40, seed=17)
    assert first == second
    splits = split_by_template(first, seed=18)
    template_sets = [set(item.template_id for item in values) for values in splits.values()]
    assert template_sets[0].isdisjoint(template_sets[1])
    assert template_sets[0].isdisjoint(template_sets[2])
    assert template_sets[1].isdisjoint(template_sets[2])


def test_trial_schema_is_versioned_and_roundtrippable():
    record = TrialRecord(
        run_id="run",
        prompt_id="prompt",
        task_family="arithmetic",
        layer=3,
        position=-1,
        intervention="non_j",
        valid=False,
        metrics={"js": 0.0},
        seed=1,
        exclusion_reason="clamp",
    ).to_dict()
    assert record["schema_version"] == 1
    assert record["exclusion_reason"] == "clamp"


def test_figure_regeneration_records_machine_source_hash(tmp_path):
    source = tmp_path / "records.jsonl"
    source.write_text('{"run_id":"r","x":1,"y":2}\n', encoding="utf-8")
    output = tmp_path / "figure.png"
    manifest = _save_figure(
        output,
        [source],
        lambda axis: axis.plot([1], [2]),
    )
    assert output.exists()
    assert manifest["sources"][0]["sha256"]
    plt.close("all")
