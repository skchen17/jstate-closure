"""V40 amended pool selector: enforce prompt and program disjointness."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from jclosure.datasets_v40 import FAMILIES, digest, make_task
from jclosure.experiments.sample_pool_v40 import historical_content
from jclosure.protocol_v40 import verify as verify_initial
from jclosure.protocol_v40_a1 import stage_freeze, verify
from jclosure.provenance import sha256_file, write_json_atomic

OUT = Path("data/v40")
ROLES = ("calibration", "development", "validation", "independent_final")
SOURCE = "src/jclosure/experiments/sample_pool_v40_a1.py"


def select_tasks(family: str, count: int, seed: int, blocked: set[str],
                 prompt_hashes: set[str]) -> list[dict]:
    rows = []
    attempt = 0
    while len(rows) < count and attempt < max(10_000, count * 300):
        task = make_task(family, seed, attempt)
        attempt += 1
        hashes = {kind: hashlib.sha256(task[f"{kind}_prompt"].encode()).hexdigest()
                  for kind in ("recipient", "donor")}
        if (task["program_hash"] in blocked or
                len(set(hashes.values())) != 2 or
                any(h in prompt_hashes for h in hashes.values())):
            continue
        blocked.add(task["program_hash"])
        prompt_hashes.update(hashes.values())
        task.update({f"{kind}_prompt_sha256": h for kind, h in hashes.items()})
        rows.append(task)
    if len(rows) != count:
        raise RuntimeError(f"V40 pool shortage {family}: {len(rows)}/{count}")
    return rows


def run(root: Path) -> dict:
    verify(root)
    cfg = verify_initial(root)["config"]
    if (root / OUT).exists() and any((root / OUT).iterdir()):
        raise RuntimeError("V40 pool exists; never regenerate in place")
    blocked, prompt_hashes, historical = historical_content(root)
    initial_programs, initial_prompts = set(blocked), set(prompt_hashes)
    (root / OUT).mkdir(parents=True, exist_ok=True)
    summary, all_rows = {}, []
    for role in ROLES:
        rows = []
        for fi, family in enumerate(FAMILIES):
            seed = int(cfg["pool_seeds"][role]) + fi * 100_003
            selected = select_tasks(family, int(cfg["roles_per_family"][role]),
                                    seed, blocked, prompt_hashes)
            for task in selected:
                task["role"] = role
            rows.extend(selected)
            all_rows.extend(selected)
        counts = {family: sum(x["family"] == family for x in rows) for family in FAMILIES}
        path = root / OUT / f"{role}_pool_v40.json"
        write_json_atomic(path, {"role": role, "items": rows, "family_counts": counts,
                                 "labels_generated_before_model_execution": True,
                                 "intervention_outcomes_observed_before_seal": False,
                                 "prompt_collision_rejection_amendment": 1})
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
    manifest = {"protocol": "v40_amendment_1", "roles": summary,
                "historical_sources": historical,
                "historical_program_overlap": False, "historical_prompt_overlap": False,
                "all_role_programs_and_prompts_disjoint": True,
                "models_not_queried": True,
                "labels_generated_before_model_execution": True,
                "source_generator_sha256": sha256_file(root / "src/jclosure/datasets_v40.py")}
    path = root / OUT / "sample_pool_manifest_v40.json"
    write_json_atomic(path, manifest)
    stage_freeze(root, "sample_pools", [SOURCE, str(path.relative_to(root)),
                                       *[str(OUT / f"{r}_pool_v40.json") for r in ROLES],
                                       *[f"artifacts/channel_function_v40_a1_pool_{r}.freeze.json"
                                         for r in ROLES]],
                 {"manifest_sha256": sha256_file(path), "models_not_queried": True})
    return manifest


if __name__ == "__main__":
    print(json.dumps(run(Path.cwd()), indent=2))
