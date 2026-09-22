"""Fresh paired semantic V35 states disjoint from V28–V34 formal panels."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from jclosure.experiments.runtime_v34 import hd
from jclosure.protocol_v35 import stage_freeze, verify
from jclosure.provenance import write_json_atomic

OUT = Path("results/v35/processed")
SOURCE = "src/jclosure/experiments/panel_v35.py"
ROLES = ("calibration", "development", "validation", "independent_final")


def run(root: Path):
    cfg = verify(root)["config"]
    used = {row["base_trial_id"] for version in range(28, 34)
            for role in ROLES
            for row in json.loads((root / f"results/v{version}/processed/design_v{version}.json").read_text()).get(role, [])
            if "base_trial_id" in row}
    previous = json.loads((root / "results/v34/processed/panel_v34.json").read_text())
    used.update(row["base_trial_id"] for role in ROLES for row in previous[role])
    roles = {role: [] for role in ROLES}
    supply = {}
    for family in cfg["families"]:
        for source in ("train", "validation"):
            frame = pd.read_parquet(root / cfg["panel_source"].format(source=source, family=family),
                                    columns=["base_trial_id", "prompt"])
            candidates = [x for x in frame.to_dict("records") if x["base_trial_id"] not in used]
            candidates.sort(key=lambda row: hd([cfg["seed"], family, source, row["base_trial_id"]]))
            assignment = (("calibration", 4), ("development", 16), ("independent_final", 8)) if source == "train" else (("validation", 8),)
            offset = 0
            for role, count in assignment:
                if len(candidates) < offset + count:
                    raise RuntimeError(f"V35 insufficient disjoint states {family}/{source}")
                for row in candidates[offset:offset + count]:
                    roles[role].append({"base_trial_id": row["base_trial_id"], "family": family,
                                        "source_role": source, "role": role,
                                        "prompt_sha256": hashlib.sha256(str(row["prompt"]).encode()).hexdigest()})
                offset += count
            supply[f"{family}:{source}"] = {"available_after_exclusion": len(candidates), "assigned": offset}
    ids = [row["base_trial_id"] for role in ROLES for row in roles[role]]
    if len(ids) != 180 or len(set(ids)) != 180 or set(ids) & used:
        raise RuntimeError("V35 panel disjointness violation")
    result = {**roles, "families": cfg["families"], "exclusion_versions": cfg["source_exclusions"],
              "excluded_state_count": len(used), "source_supply": supply,
              "paired_semantic_state_ids_for_both_models": True,
              "role_hashes": {role: hd([row["base_trial_id"] for row in roles[role]]) for role in ROLES},
              "causal_response_observed_before_freeze": False,
              "historical_independent_finals_reopened": False}
    OUT.mkdir(parents=True, exist_ok=True)
    path = root / OUT / "panel_v35.json"
    write_json_atomic(path, result)
    seal = stage_freeze(root, "panel", [SOURCE, str(path.relative_to(root))],
                        {"panel_hash": hd(result), "role_hashes": result["role_hashes"],
                         "causal_response_observed_before_freeze": False})
    return {"freeze_digest": seal["freeze_digest"], "excluded": len(used),
            "roles": {role: len(roles[role]) for role in ROLES}, "source_supply": supply}


if __name__ == "__main__":
    print(json.dumps(run(Path.cwd()), indent=2))
