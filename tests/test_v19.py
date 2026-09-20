from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

from jclosure.experiments.analyze_v19 import _ci_state, _ridge_predict
from jclosure.experiments.compact_response_v19 import _matrix as compact_matrix, _signs, _sketch
from jclosure.protocol_v19 import digest, verify, verify_stage


ROOT = Path(__file__).resolve().parents[1]


def test_protocol_chain():
    base = verify(ROOT)
    split = verify_stage(ROOT, "splits")
    q0 = verify_stage(ROOT, "interventions")
    q1 = verify_stage(ROOT, "q_amendment_1")
    q2 = verify_stage(ROOT, "q_amendment_2")
    teacher = verify_stage(ROOT, "teacher_amendment_3")
    runtime = verify_stage(ROOT, "runtime_gpu1_amendment_4")
    diagnostic_runtime = verify_stage(ROOT, "diagnostic_gpu1_amendment_5")
    binding = verify_stage(ROOT, "execution_binding_amendment_6")
    assert base["freeze_digest"] == digest(base)
    assert q1["prior_intervention_freeze_digest"] == q0["freeze_digest"]
    assert q2["prior_amendment_digest"] == q1["freeze_digest"]
    assert teacher["prior_q_amendment_digest"] == q2["freeze_digest"]
    assert runtime["equivalence_passed"]
    assert diagnostic_runtime["prior_runtime_digest"] == runtime["freeze_digest"]
    assert binding["q_amendment_digest"] == q2["freeze_digest"]
    ids = [row["base_trial_id"] for role in ("calibration", "train", "validation") for row in split[role]]
    assert len(ids) == len(set(ids)) == 520
    assert len(split["train"]) == 400 and len(split["validation"]) == 100


def test_state_cluster_bootstrap():
    frame = pd.DataFrame({"base_trial_id": ["a"] * 100 + ["b"] * 2,
                          "value": [0.0] * 100 + [1.0] * 2})
    result = _ci_state(frame, "value", 19, 200)
    assert result["states"] == 2
    assert result["median"] == 0.5


def test_fixed_ridge_shape_and_signal():
    rng = np.random.default_rng(19)
    x = rng.normal(size=(80, 4))
    y = x @ np.asarray([[1.0], [0.0], [0.0], [0.0]])
    pred = _ridge_predict(x[:60], y[:60], x[60:], 0.01)
    assert pred.shape == (20, 1)
    assert np.linalg.norm(pred - y[60:]) < 0.1


def test_compact_bilinear_matrix_shape():
    j = np.ones((3, 128), dtype=np.float32)
    c = np.ones((3, 4), dtype=np.float32)
    a = np.ones((3, 8), dtype=np.float32)
    raw = np.ones((3, 384), dtype=np.float32)
    small = compact_matrix(j, c, a)
    expanded = compact_matrix(j, c, a, raw)
    assert small.shape == (3, 1 + 128 + 4 + 8 + 128 * 8 + 4 * 8)
    assert expanded.shape[1] == small.shape[1] + 384 + 384 * 8


def test_raw_cache_sketch_shape_and_bf16_delta():
    recurrent = torch.zeros(3145728, dtype=torch.bfloat16)
    conv = torch.zeros(196608, dtype=torch.bfloat16)
    key = torch.zeros((1, 1, 128, 2048), dtype=torch.bfloat16)
    value = torch.zeros_like(key)
    cache = SimpleNamespace(layers=[SimpleNamespace(recurrent_states=recurrent, conv_states=conv),
                                    SimpleNamespace(keys=key, values=value)])
    signs = _signs(torch.device("cpu"))
    baseline = _sketch(cache, [0], [1], signs)
    recurrent[0] = 1.0
    changed = _sketch(cache, [0], [1], signs)
    assert baseline.shape == changed.shape == (384,)
    assert np.linalg.norm(changed - baseline) > 0


def test_factorial_identity_and_boundary_j():
    paths = sorted((ROOT / "results/v19/processed").glob("factorial_validation_*_v19.parquet"))
    assert len(paths) == 5
    frame = pd.concat([pd.read_parquet(path) for path in paths], ignore_index=True)
    assert frame.base_trial_id.nunique() == 100
    assert (frame.boundary_j_difference == 0).all()
    assert (frame.boundary_j_hash_p0 == frame.boundary_j_hash_pq).all()
    assert (frame.p0_snapshot_hash != frame.pq_snapshot_hash).all()
    assert set(frame.action_status).issubset({"MATCHED_REALIZED_ACTION", "ACTION_REALIZATION_MISMATCH"})
    sample = frame.iloc[0]
    y00, y01, y10, y11 = [np.asarray(sample[name], dtype=float) for name in
                          ("y00_stack", "y01_stack", "y10_stack", "y11_stack")]
    m = (y11 - y10) - (y01 - y00)
    assert len(m) == 288 and np.isfinite(m).all()
    assert frame.q_reliable.all()
    assert np.isfinite(frame.q_realized_state_norm.to_numpy(dtype=float)).all()
