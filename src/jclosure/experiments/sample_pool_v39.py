"""Generate fresh, horizon-matched V39 pools; seal all roles before model calls."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from jclosure.datasets_v8 import generate_tasks
from jclosure.experiments.runtime_v34 import hd
from jclosure.protocol_v39 import stage_freeze, verify
from jclosure.provenance import sha256_file, write_json_atomic


SOURCE = "src/jclosure/experiments/sample_pool_v39.py"
OUT = Path("data/v39")
ROLES = ("calibration", "development", "validation", "independent_final")


def historical_content(root: Path) -> tuple[set[str], set[str], dict]:
    programs: set[str] = set()
    prompts: set[str] = set()
    sources: dict = {}
    for path in sorted((root / "data").rglob("*.json")):
        if "v39" in path.parts:
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            continue
        rows = payload.get("items", []) if isinstance(payload, dict) else []
        found = 0
        for row in rows:
            if not isinstance(row, dict):
                continue
            if row.get("program_hash"):
                programs.add(str(row["program_hash"]))
                found += 1
            if row.get("prompt"):
                prompts.add(hashlib.sha256(str(row["prompt"]).encode()).hexdigest())
        if found:
            sources[str(path.relative_to(root))] = {"sha256": sha256_file(path), "rows": found}
    for path in sorted((root / "results/v18/processed").glob("crossed_state_*_v18.parquet")):
        import pandas as pd
        frame = pd.read_parquet(path, columns=["prompt"])
        prompts.update(hashlib.sha256(str(p).encode()).hexdigest() for p in frame.prompt)
        sources[str(path.relative_to(root))] = {"sha256": sha256_file(path), "rows": len(frame)}
    return programs, prompts, sources


def run(root: Path) -> dict:
    cfg = verify(root)["config"]
    if (root / OUT).exists() and any((root / OUT).iterdir()):
        raise RuntimeError("V39 sample pool already exists; never regenerate in place")
    blocked_programs, blocked_prompts, historical_sources = historical_content(root)
    initial_programs, initial_prompts = set(blocked_programs), set(blocked_prompts)
    (root / OUT).mkdir(parents=True, exist_ok=True)
    all_rows = []
    summaries = {}
    for role in ROLES:
        rows = []
        for fi, family in enumerate(cfg["families"]):
            horizon = (cfg["calibration_generator_horizon"] if role == "calibration"
                       else cfg["formal_generator_horizon"])[family]
            count = int(cfg["roles_per_family"][role])
            seed = int(cfg["sample_pool_seeds"][role]) + fi * 100_003
            tasks = generate_tasks(family, count, seed=seed, horizon=int(horizon),
                                   blocked_hashes=blocked_programs)
            for task in tasks:
                prompt_hash = hashlib.sha256(task.prompt.encode()).hexdigest()
                if prompt_hash in blocked_prompts:
                    raise RuntimeError(f"V39 prompt collision: {role}/{family}/{task.example_id}")
                blocked_programs.add(task.program_hash)
                blocked_prompts.add(prompt_hash)
                row = {"base_trial_id": task.example_id.replace("v8-", f"v39-{role}-", 1),
                       "role": role, "family": family, "source_role": role,
                       "prompt": task.prompt, "prompt_sha256": prompt_hash,
                       "program_hash": task.program_hash, "variant": task.variant,
                       "horizon": task.horizon, "generator_seed": task.generator_seed,
                       "generator_index": task.generator_index,
                       "semantic_actions": list(task.semantic_actions),
                       "final_answer": task.final_answer, "ast": task.ast}
                rows.append(row)
                all_rows.append(row)
        if len(rows) != 5 * int(cfg["roles_per_family"][role]):
            raise RuntimeError(f"V39 {role} count mismatch")
        path = root / OUT / f"{role}_pool_v39.json"
        counts = {family: sum(row["family"] == family for row in rows)
                  for family in cfg["families"]}
        payload = {"role": role, "items": rows, "five_family_counts": counts,
                   "historical_program_overlap": False, "historical_prompt_overlap": False,
                   "model_response_observed_before_seal": False}
        write_json_atomic(path, payload)
        seal = stage_freeze(root, f"pool_{role}", [SOURCE, str(path.relative_to(root))],
                            {"role": role, "pool_sha256": sha256_file(path),
                             "state_id_hash": hd([r["base_trial_id"] for r in rows]),
                             "program_hash": hd([r["program_hash"] for r in rows]),
                             "prompt_hash": hd([r["prompt_sha256"] for r in rows]),
                             "model_response_observed_before_seal": False})
        summaries[role] = {"states": len(rows), "families": counts,
                           "pool_sha256": sha256_file(path),
                           "seal_digest": seal["freeze_digest"]}
    if len(all_rows) != 180 or len({r["program_hash"] for r in all_rows}) != 180:
        raise RuntimeError("V39 program overlap/count drift")
    if len({r["prompt_sha256"] for r in all_rows}) != 180:
        raise RuntimeError("V39 prompt overlap")
    if any(r["program_hash"] in initial_programs or r["prompt_sha256"] in initial_prompts
           for r in all_rows):
        raise RuntimeError("V39 historical collision")
    manifest = {"protocol": "v39", "roles": summaries,
                "historical_sources": historical_sources,
                "historical_program_overlap": False,
                "historical_prompt_overlap": False,
                "all_role_programs_disjoint": True,
                "all_role_prompts_disjoint": True,
                "models_not_queried": True,
                "same_formal_horizon_across_roles": True,
                "source_generator": "jclosure.datasets_v8.generate_tasks",
                "source_generator_sha256": sha256_file(root / "src/jclosure/datasets_v8.py"),
                "historical_v1_v38_results_used_in_selection": False}
    path = root / OUT / "sample_pool_manifest_v39.json"
    write_json_atomic(path, manifest)
    stage_freeze(root, "sample_pools", [SOURCE, str(path.relative_to(root)),
                                        *[str(OUT / f"{role}_pool_v39.json") for role in ROLES],
                                        *[f"artifacts/interaction_genesis_v39_pool_{role}.freeze.json"
                                          for role in ROLES]],
                 {"manifest_sha256": sha256_file(path), "models_not_queried": True})
    return manifest


if __name__ == "__main__":
    print(json.dumps(run(Path.cwd()), indent=2))
