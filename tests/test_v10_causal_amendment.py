from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from jclosure.experiments.decoded_causal_v10 import _pair_metadata
from jclosure.experiments.decoded_causal_v10_amendment import v8_task_mapping


def test_every_frozen_confirmatory_pair_maps_to_v8_task() -> None:
    root = Path(__file__).resolve().parents[1]
    candidate = json.loads(
        (root / "artifacts/causal_sufficiency_v10_candidates.freeze.json").read_text()
    )
    context = SimpleNamespace(root=root)
    tasks = v8_task_mapping(context)
    pairs = _pair_metadata(root)
    selected = candidate["causal_confirmatory_base_trial_ids"]
    assert len(selected) == 50
    assert all(str(pairs[base_id]["prompt_id"]) in tasks for base_id in selected)
