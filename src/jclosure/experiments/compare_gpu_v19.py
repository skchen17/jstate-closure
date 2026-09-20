"""Compare four V19 endpoints on GPU1 to frozen GPU0 pilot before validation."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from jclosure.experiments import counterfactual_bank_v19 as bank
from jclosure.experiments.gpu1_v19 import load_context_gpu1
from jclosure.experiments.actuation_v15 import apply
from jclosure.experiments.operator_v15 import stack
from jclosure.protocol_v19 import stage_freeze, verify_stage
from jclosure.provenance import write_json_atomic

SOURCE = "src/jclosure/experiments/compare_gpu_v19.py"


def run(root: Path):
    split = verify_stage(root, "splits")
    q = verify_stage(root, "q_amendment_2")
    item = split["train"][0]
    prior_path = bank.SCRATCH / "train" / f"factorial_{item['base_trial_id']}.parquet"
    if not prior_path.exists():
        raise RuntimeError("GPU0 pilot record missing")
    prior = pd.read_parquet(prior_path)
    chosen = prior[(prior.q_name == "rec_conv_causal1_amended") & (prior.coordinate_index == 5)
                   & (prior.action_sign == 1) & (prior.horizon == 1)].iloc[0]
    teachers = pd.read_parquet(root / verify_stage(root, "teacher_amendment_3")["teacher_path"])
    token_row = teachers[teachers.base_trial_id == item["base_trial_id"]].iloc[0]
    tokens = [int(x) for x in token_row.teacher_tokens]
    bank.load_context_v18 = load_context_gpu1
    bundle, dense, _, values, rec, att, measured, state_layer, ws_layers, ws_count = bank._setup(root)
    old = bank._state_metadata(root, item)
    clean = bank._prefill_history(bundle, old["prompt"], measured, dense, state_layer)
    p0 = clean["cache"]
    qdesign = next(x for x in q["q"] if x["name"] == "rec_conv_causal1_amended")
    qrow = bank._q_row(values["directions"], qdesign, float(qdesign["alpha"]))
    action = next(x for x in split["actions"] if x["coordinate_index"] == 5)
    arow = bank._action_row(values["directions"], int(action["direction_index"]), 0.5, 1)
    pq = apply(p0, 1.0, qrow, rec, att, "native_fp32_add_bf16_writeback")
    caches = {"y00_stack": p0, "y01_stack": apply(p0, 1.0, arow, rec, att, "native_fp32_add_bf16_writeback"),
              "y10_stack": pq, "y11_stack": apply(pq, 1.0, arow, rec, att, "native_fp32_add_bf16_writeback")}
    scales = {key: float(v) for key, v in split["target_scales"].items()}
    comparison = {}
    for name, cache in caches.items():
        _, endpoint = bank._trajectory(bundle, dense, cache, None, tokens, 1, clean["prompt_length"],
                                       np.asarray(split["selected_j"], dtype=int), np.asarray(split["selected_logits"], dtype=int),
                                       ws_layers, ws_count, max(measured))
        observed = stack(endpoint[1], scales).astype(np.float64)
        expected = np.asarray(chosen[name], dtype=np.float64)
        comparison[name] = {"max_absolute": float(np.max(np.abs(observed - expected))),
                            "relative_l2": float(np.linalg.norm(observed - expected) / max(np.linalg.norm(expected), 1e-12))}
    passed = all(x["relative_l2"] <= 1e-6 for x in comparison.values())
    audit_path = root / bank.OUT / "gpu1_equivalence_audit_v19.json"
    write_json_atomic(audit_path, {"base_trial_id": item["base_trial_id"], "comparison": comparison,
                                   "equivalence_passed": passed, "tolerance_relative_l2": 1e-6})
    result = stage_freeze(root, "runtime_gpu1_amendment_4",
                          [SOURCE, "src/jclosure/experiments/gpu1_v19.py", str(audit_path.relative_to(root)),
                           "artifacts/counterfactual_workspace_v19_teacher_amendment_3.freeze.json"],
                          {"why": "GPU1 is available while GPU0 train factorial runs; verify all four h1 endpoints on the same frozen state, q and action before using GPU1 for validation. Same weights, dtype, BF16 writeback and tokens.",
                           "when": "after GPU0 pilot and before any V19 validation factorial response",
                           "responses_already_observed": {"factorial_train_state_records": len(list((bank.SCRATCH / "train").glob("factorial_*.parquet"))),
                                                          "factorial_validation_state_records": len(list((bank.SCRATCH / "validation").glob("factorial_*.parquet")))},
                           "equivalence_passed": passed, "comparison": comparison,
                           "runtime_device_map": {"": 1}, "tolerance_relative_l2": 1e-6,
                           "weights_changed": False})
    return result


if __name__ == "__main__":
    print(json.dumps(run(Path.cwd()), indent=2))
