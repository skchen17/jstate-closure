"""Generate and preseal fresh V38 pools before any model response is observed."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from jclosure.datasets_v8 import generate_tasks
from jclosure.experiments.runtime_v34 import hd
from jclosure.protocol_v38 import stage_freeze, verify
from jclosure.provenance import sha256_file, write_json_atomic


SOURCE = "src/jclosure/experiments/sample_pool_v38.py"
OUT = Path("data/v38")
ROLES = ("calibration", "development", "validation", "independent_final")


def historical_content(root: Path) -> tuple[set[str], set[str], dict]:
    programs: set[str] = set()
    prompts: set[str] = set()
    sources = {}
    for path in sorted((root / "data").rglob("*.json")):
        if "v38" in path.parts:
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
    if OUT.exists() and any(OUT.iterdir()):
        raise RuntimeError("V38 sample pool already exists; never regenerate it in place")
    blocked_programs, blocked_prompts, historical_sources = historical_content(root)
    selected_horizons_path = root / "data/v8/persistent_causal_formal.json"
    horizons = json.loads(selected_horizons_path.read_text())['selected_horizons']
    OUT.mkdir(parents=True, exist_ok=True)
    role_summaries = {}
    all_rows = []
    for role in ROLES:
        rows = []
        for fi, family in enumerate(cfg["families"]):
            count = int(cfg["roles_per_family"][role])
            seed = int(cfg["sample_pool_seeds"][role]) + fi * 100_003
            tasks = generate_tasks(family, count, seed=seed,
                                   horizon=int(horizons[family]), blocked_hashes=blocked_programs)
            for task in tasks:
                prompt_hash = hashlib.sha256(task.prompt.encode()).hexdigest()
                if prompt_hash in blocked_prompts:
                    raise RuntimeError(f"V38 prompt collision in {role}/{family}: {task.example_id}")
                blocked_programs.add(task.program_hash)
                blocked_prompts.add(prompt_hash)
                row = {"base_trial_id": task.example_id.replace("v8-", f"v38-{role}-", 1),
                       "role": role, "family": family, "source_role": role,
                       "prompt": task.prompt, "prompt_sha256": prompt_hash,
                       "program_hash": task.program_hash, "variant": task.variant,
                       "horizon": task.horizon, "generator_seed": task.generator_seed,
                       "generator_index": task.generator_index,
                       "semantic_actions": list(task.semantic_actions),
                       "final_answer": task.final_answer, "ast": task.ast}
                rows.append(row)
                all_rows.append(row)
        if len(rows) != len(cfg["families"]) * int(cfg["roles_per_family"][role]):
            raise RuntimeError(f"V38 {role} count mismatch")
        if len({row["base_trial_id"] for row in rows}) != len(rows):
            raise RuntimeError(f"V38 {role} repeated IDs")
        path = root / OUT / f"{role}_pool_v38.json"
        payload = {"role": role, "items": rows,
                   "five_family_counts": {family: sum(row["family"] == family for row in rows)
                                          for family in cfg["families"]},
                   "historical_program_overlap": False, "historical_prompt_overlap": False,
                   "model_response_observed_before_seal": False}
        write_json_atomic(path, payload)
        seal = stage_freeze(root, f"pool_{role}", [SOURCE, str(path.relative_to(root)),
                                                  str(selected_horizons_path.relative_to(root))],
                            {"role": role, "pool_sha256": sha256_file(path),
                             "state_id_hash": hd([row["base_trial_id"] for row in rows]),
                             "program_hash": hd([row["program_hash"] for row in rows]),
                             "prompt_hash": hd([row["prompt_sha256"] for row in rows]),
                             "model_response_observed_before_seal": False})
        role_summaries[role] = {"states": len(rows), "families": payload["five_family_counts"],
                                "pool_sha256": sha256_file(path), "seal_digest": seal["freeze_digest"]}
    if len(all_rows) != 180 or len({row["program_hash"] for row in all_rows}) != 180:
        raise RuntimeError("V38 role pools overlap")
    manifest = {"protocol": "v38", "roles": role_summaries,
                "historical_sources": historical_sources,
                "historical_program_overlap": False, "historical_prompt_overlap": False,
                "all_role_programs_disjoint": True, "all_role_prompts_disjoint": True,
                "models_not_queried": True,
                "source_generator": "jclosure.datasets_v8.generate_tasks",
                "source_generator_sha256": sha256_file(root / "src/jclosure/datasets_v8.py"),
                "historical_v37_results_used_in_selection": False}
    path = root / OUT / "sample_pool_manifest_v38.json"
    write_json_atomic(path, manifest)
    stage_freeze(root, "sample_pools", [SOURCE, str(path.relative_to(root)),
                                        *[str(OUT / f"{role}_pool_v38.json") for role in ROLES],
                                        *[f"artifacts/trajectory_composition_v38_pool_{role}.freeze.json"
                                          for role in ROLES]],
                 {"manifest_sha256": sha256_file(path), "models_not_queried": True})
    return manifest


if __name__ == "__main__":
    print(json.dumps(run(Path.cwd()), indent=2))
