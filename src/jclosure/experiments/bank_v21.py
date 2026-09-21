"""Frozen V21 state/action roles and a paired 64-probe differential–finite panel."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from jclosure.protocol_v21 import stage_freeze, verify
from jclosure.provenance import sha256_file

SOURCE = "src/jclosure/experiments/bank_v21.py"
OUT = Path("results/v21/processed")
SCRATCH = Path("/data/CSK/J-space-project/v21-action-geometry-work")


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _balanced(rows: list[dict], count: int, seed: str) -> list[dict]:
    families = sorted({x["family"] for x in rows})
    if count % len(families):
        raise RuntimeError("V21 balanced state count invalid")
    each = count // len(families)
    selected = []
    for family in families:
        current = sorted((x for x in rows if x["family"] == family),
                         key=lambda x: hashlib.sha256(f"{seed}:{x['base_trial_id']}".encode()).hexdigest())
        selected.extend(current[:each])
    return selected


def prepare(root: Path) -> dict:
    config = verify(root)["config"]
    split = json.loads((root / config["sources"]["v20_splits"]).read_text())
    action = json.loads((root / config["sources"]["v20_actions"]).read_text())
    summary = json.loads((root / "results/v13/processed/probe_directions_v13.json").read_text())
    train = split["operator_train"]
    validation = split["operator_validation"]
    if len(train) != 150 or len(validation) != 50 or set(x["base_trial_id"] for x in train) & set(x["base_trial_id"] for x in validation):
        raise RuntimeError("V21 V20 role source mismatch")
    jvp_dev = _balanced(train, 50, "V21-JVP-DEV")
    jvp_val = _balanced(validation, 25, "V21-JVP-VAL")
    anchor = _balanced(train, 30, "V21-RAW-NYSTROM")
    q_cycle = config["state_panels"]["jvp_q_cycle"]
    def qassign(rows):
        return [{"base_trial_id": x["base_trial_id"], "family": x["family"],
                 "q_name": q_cycle[i % len(q_cycle)]} for i, x in enumerate(rows)]
    train_action = action["partitions"]["train"]
    val_action = action["partitions"]["validation"]
    final_action = action["partitions"]["final_heldout"]
    opened = train_action + val_action
    if len(train_action) != 12 or len(val_action) != 6 or len(final_action) != 6:
        raise RuntimeError("V21 action partition mismatch")
    opened_idx = {int(x["direction_index"]) for x in opened}
    final_idx = {int(x["direction_index"]) for x in final_action}
    if opened_idx & final_idx:
        raise RuntimeError("V21 final directions overlap opened actions")
    chosen = [int(x["direction_index"]) for x in opened]
    required_extras = 64 - len(chosen)
    pools = summary["probe_families"]
    quotas = {"causal_weighted": 10, "architecture_balanced": 10, "random_raw": 9,
              "high_variance_pca": 9, "low_variance": 8}
    if sum(quotas.values()) != required_extras:
        raise RuntimeError("V21 probe quota mismatch")
    probe_labels = {int(index): name for name, indices in pools.items() for index in indices}
    for family, quota in quotas.items():
        candidates = [int(x) for x in pools[family] if int(x) not in opened_idx | final_idx]
        ordered = sorted(candidates, key=lambda x: hashlib.sha256(f"V21-PROBE:{family}:{x}".encode()).hexdigest())
        chosen.extend(ordered[:quota])
    if len(chosen) != 64 or len(set(chosen)) != 64 or set(chosen) & final_idx:
        raise RuntimeError("V21 paired JVP probe set invalid")
    order = [0, 1, 2, 8, 5, 6, 7, 14, 15, 11, 12, 27]
    by_coord = {int(x["coordinate_index"]): x for x in train_action}
    if set(order) != set(by_coord):
        raise RuntimeError("V21 nested train action order mismatch")
    details = {"V20_train_base_count": len(train), "V20_validation_base_count": len(validation),
               "V20_train_base_id_sha256": split["role_id_sha256"]["operator_train"],
               "V20_validation_base_id_sha256": split["role_id_sha256"]["operator_validation"],
               "jvp_development": qassign(jvp_dev), "jvp_validation": qassign(jvp_val),
               "raw_Nystrom_anchor_base_ids": [x["base_trial_id"] for x in anchor],
               "jvp_development_id_sha256": digest([x["base_trial_id"] for x in jvp_dev]),
               "jvp_validation_id_sha256": digest([x["base_trial_id"] for x in jvp_val]),
               "raw_anchor_id_sha256": digest([x["base_trial_id"] for x in anchor]),
               "train_actions": train_action, "validation_actions": val_action,
               "nested_train_action_coordinates": order,
               "sealed_final_action_sha256": action["action_hashes"]["final_heldout"],
               "sealed_final_action_count": 6,
               "jvp_probe_direction_indices": chosen,
               "jvp_probe_labels": [probe_labels[i] for i in chosen],
               "jvp_probe_sha256": digest(chosen),
               "jvp_probe_count": len(chosen),
               "V13_direction_tensor_sha256": sha256_file(root / config["sources"]["v13_directions"]),
               "V13_score_coordinate_dimension": 14397,
               "V20_final_action_responses_opened": False,
               "independent_final_states_opened": False,
               "JVP_or_finite_V21_responses_observed": 0,
               "note_on_sealed_metadata": "V20 freeze lists final direction identifiers, but V21 probe/representation design does not use their response labels or direction vectors."}
    return stage_freeze(root, "roles", [SOURCE, config["sources"]["v20_splits"],
                                        config["sources"]["v20_actions"],
                                        "results/v13/processed/probe_directions_v13.json"], details)


if __name__ == "__main__":
    result = prepare(Path.cwd())
    print(json.dumps({"freeze_digest": result["freeze_digest"],
                      "jvp_probe_sha256": result["jvp_probe_sha256"],
                      "jvp_development_states": len(result["jvp_development"]),
                      "jvp_validation_states": len(result["jvp_validation"])}, indent=2))
