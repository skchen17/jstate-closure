"""Seal all V40 roles before any V40 intervention outcome is observed."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from jclosure.datasets_v40 import FAMILIES, digest, generate
from jclosure.protocol_v40 import stage_freeze, verify
from jclosure.provenance import sha256_file, write_json_atomic

OUT = Path("data/v40")
ROLES = ("calibration", "development", "validation", "independent_final")
SOURCE = "src/jclosure/experiments/sample_pool_v40.py"


def historical_content(root: Path) -> tuple[set[str], set[str], dict]:
    programs: set[str] = set()
    prompts: set[str] = set()
    sources = {}
    for path in sorted((root / "data").rglob("*.json")):
        if "v40" in path.parts:
            continue
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            continue
        if not isinstance(value, dict):
            continue
        count = 0
        for row in value.get("items", []):
            if not isinstance(row, dict):
                continue
            if row.get("program_hash"):
                programs.add(str(row["program_hash"])); count += 1
            if row.get("prompt"):
                prompts.add(hashlib.sha256(str(row["prompt"]).encode()).hexdigest())
        if count:
            sources[str(path.relative_to(root))] = {"sha256": sha256_file(path),
                                                    "rows": count}
    for path in sorted((root / "results/v18/processed").glob("crossed_state_*_v18.parquet")):
        import pandas as pd
        frame = pd.read_parquet(path, columns=["prompt"])
        prompts.update(hashlib.sha256(str(p).encode()).hexdigest()
                       for p in frame.prompt)
        sources[str(path.relative_to(root))] = {"sha256": sha256_file(path),
                                                "rows": len(frame)}
    return programs, prompts, sources


def run(root: Path) -> dict:
    cfg = verify(root)["config"]
    if (root / OUT).exists() and any((root / OUT).iterdir()):
        raise RuntimeError("V40 pool exists; do not regenerate in place")
    blocked, prompts, historical = historical_content(root)
    initial_programs, initial_prompts = set(blocked), set(prompts)
    (root / OUT).mkdir(parents=True, exist_ok=True)
    summary = {}
    all_rows = []
    for role in ROLES:
        rows = []
        for fi, family in enumerate(FAMILIES):
            seed = int(cfg["pool_seeds"][role]) + fi * 100_003
            for task in generate(family, int(cfg["roles_per_family"][role]), seed, blocked):
                task["role"] = role
                for kind in ("recipient", "donor"):
                    h = hashlib.sha256(task[f"{kind}_prompt"].encode()).hexdigest()
                    if h in prompts:
                        raise RuntimeError(f"V40 prompt collision: {role}:{family}:{kind}")
                    prompts.add(h)
                    task[f"{kind}_prompt_sha256"] = h
                rows.append(task)
                all_rows.append(task)
        path = root / OUT / f"{role}_pool_v40.json"
        counts = {family: sum(x["family"] == family for x in rows) for family in FAMILIES}
        write_json_atomic(path, {"role": role, "items": rows, "family_counts": counts,
                                 "labels_generated_before_model_execution": True,
                                 "intervention_outcomes_observed_before_seal": False})
        seal = stage_freeze(root, f"pool_{role}", [SOURCE, str(path.relative_to(root))],
                            {"role": role, "states": len(rows), "family_counts": counts,
                             "state_ids_digest": digest([x["state_id"] for x in rows]),
                             "pool_sha256": sha256_file(path),
                             "intervention_outcomes_observed_before_seal": False})
        summary[role] = {"states": len(rows), "family_counts": counts,
                         "pool_sha256": sha256_file(path),
                         "seal_digest": seal["freeze_digest"]}
    if len({x["state_id"] for x in all_rows}) != len(all_rows):
        raise RuntimeError("V40 state ID overlap")
    if any(x["program_hash"] in initial_programs or
           x["recipient_prompt_sha256"] in initial_prompts or
           x["donor_prompt_sha256"] in initial_prompts for x in all_rows):
        raise RuntimeError("V40 historical overlap")
    manifest = {"protocol": "v40", "roles": summary,
                "historical_sources": historical,
                "historical_program_overlap": False, "historical_prompt_overlap": False,
                "roles_disjoint": True, "models_not_queried": True,
                "labels_generated_before_model_execution": True,
                "source_generator_sha256": sha256_file(root / "src/jclosure/datasets_v40.py")}
    path = root / OUT / "sample_pool_manifest_v40.json"
    write_json_atomic(path, manifest)
    stage_freeze(root, "sample_pools", [SOURCE, str(path.relative_to(root)),
                                       *[str(OUT / f"{r}_pool_v40.json") for r in ROLES],
                                       *[f"artifacts/channel_function_v40_pool_{r}.freeze.json"
                                         for r in ROLES]],
                 {"manifest_sha256": sha256_file(path), "models_not_queried": True})
    return manifest


if __name__ == "__main__":
    print(json.dumps(run(Path.cwd()), indent=2))
