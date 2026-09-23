"""Pre-response V38 modular-program-space exhaustion amendment and pool completion."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from jclosure.datasets_v8 import generate_tasks
from jclosure.experiments.runtime_v34 import hd
from jclosure.experiments.sample_pool_v38 import historical_content
from jclosure.protocol_v38 import stage_freeze, verify, verify_stage
from jclosure.provenance import sha256_file, write_json_atomic


OUT = Path("data/v38")
SOURCE = "src/jclosure/experiments/sample_pool_completion_v38.py"
ROLES = ("calibration", "development", "validation", "independent_final")
OVERRIDE = {"validation": {"modular_arithmetic": 3},
            "independent_final": {"modular_arithmetic": 3}}


def plan(root: Path):
    verify_stage(root, "pool_calibration")
    verify_stage(root, "pool_development")
    cfg = verify(root)["config"]
    if (root / OUT / "validation_pool_v38.json").exists() or \
       (root / OUT / "independent_final_pool_v38.json").exists():
        raise RuntimeError("V38 formal pool already exists")
    payload = {
        "schema_version": 1,
        "reason": "Historical-disjoint horizon-2 modular program space exhausted after 4 of 8 validation candidates; no model responses had been observed.",
        "previous_roles_unchanged": ["calibration", "development"],
        "remaining_roles": ["validation", "independent_final"],
        "horizon_override": OVERRIDE,
        "other_family_horizons_unchanged": True,
        "selection_seeds_unchanged": cfg["sample_pool_seeds"],
        "formal_outcomes_observed_before_plan": False,
        "distribution_shift_disclose": "Modular arithmetic is horizon 2 in calibration/development and horizon 3 in validation/final.",
        "historical_program_prompt_overlap_allowed": False,
    }
    path = root / OUT / "sample_pool_completion_plan_v38.json"
    write_json_atomic(path, payload)
    seal = stage_freeze(root, "sample_pool_completion_plan",
                        [SOURCE, str(path.relative_to(root)),
                         "artifacts/trajectory_composition_v38_pool_calibration.freeze.json",
                         "artifacts/trajectory_composition_v38_pool_development.freeze.json"],
                        {"plan_sha256": sha256_file(path),
                         "formal_outcomes_observed_before_plan": False})
    return {"freeze_digest": seal["freeze_digest"], **payload}


def complete(root: Path):
    verify_stage(root, "sample_pool_completion_plan")
    cfg = verify(root)["config"]
    blocked_programs, blocked_prompts, historical_sources = historical_content(root)
    horizon_source = root / "data/v8/persistent_causal_formal.json"
    horizons = json.loads(horizon_source.read_text())["selected_horizons"]
    summaries = {}
    all_rows = []
    for role in ("calibration", "development"):
        verify_stage(root, f"pool_{role}")
        path = root / OUT / f"{role}_pool_v38.json"
        rows = json.loads(path.read_text())["items"]
        for row in rows:
            if row["program_hash"] in blocked_programs or row["prompt_sha256"] in blocked_prompts:
                raise RuntimeError(f"V38 prior role historical collision: {role}")
            blocked_programs.add(row["program_hash"])
            blocked_prompts.add(row["prompt_sha256"])
        all_rows.extend(rows)
        summaries[role] = {"states": len(rows),
                           "families": {family: sum(r["family"] == family for r in rows)
                                        for family in cfg["families"]},
                           "pool_sha256": sha256_file(path),
                           "seal_digest": verify_stage(root, f"pool_{role}")["freeze_digest"]}
    for role in ("validation", "independent_final"):
        path = root / OUT / f"{role}_pool_v38.json"
        if path.exists():
            raise RuntimeError(f"V38 {role} pool already exists")
        rows = []
        for fi, family in enumerate(cfg["families"]):
            count = int(cfg["roles_per_family"][role])
            seed = int(cfg["sample_pool_seeds"][role]) + fi * 100_003
            horizon = OVERRIDE.get(role, {}).get(family, int(horizons[family]))
            tasks = generate_tasks(family, count, seed=seed, horizon=horizon,
                                   blocked_hashes=blocked_programs)
            for task in tasks:
                prompt_hash = hashlib.sha256(task.prompt.encode()).hexdigest()
                if prompt_hash in blocked_prompts:
                    raise RuntimeError(f"V38 prompt collision {role}/{family}")
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
        if len(rows) != 40 or len({r["base_trial_id"] for r in rows}) != 40:
            raise RuntimeError(f"V38 incomplete {role} pool")
        payload = {"role": role, "items": rows,
                   "five_family_counts": {family: sum(r["family"] == family for r in rows)
                                          for family in cfg["families"]},
                   "historical_program_overlap": False,
                   "historical_prompt_overlap": False,
                   "model_response_observed_before_seal": False,
                   "horizon_override": OVERRIDE[role]}
        write_json_atomic(path, payload)
        seal = stage_freeze(root, f"pool_{role}",
                            [SOURCE, str(path.relative_to(root)),
                             "artifacts/trajectory_composition_v38_sample_pool_completion_plan.freeze.json"],
                            {"role": role, "pool_sha256": sha256_file(path),
                             "state_id_hash": hd([r["base_trial_id"] for r in rows]),
                             "program_hash": hd([r["program_hash"] for r in rows]),
                             "prompt_hash": hd([r["prompt_sha256"] for r in rows]),
                             "model_response_observed_before_seal": False})
        summaries[role] = {"states": len(rows), "families": payload["five_family_counts"],
                           "pool_sha256": sha256_file(path), "seal_digest": seal["freeze_digest"]}
    if len(all_rows) != 180 or len({r["program_hash"] for r in all_rows}) != 180 or \
       len({r["prompt_sha256"] for r in all_rows}) != 180:
        raise RuntimeError("V38 incomplete/disjoint role pools")
    manifest = {"protocol": "v38", "roles": summaries,
                "historical_sources": historical_sources,
                "historical_program_overlap": False,
                "historical_prompt_overlap": False,
                "all_role_programs_disjoint": True,
                "all_role_prompts_disjoint": True,
                "models_not_queried": True,
                "horizon_override": OVERRIDE,
                "distribution_shift_disclosed": True,
                "source_generator": "jclosure.datasets_v8.generate_tasks",
                "source_generator_sha256": sha256_file(root / "src/jclosure/datasets_v8.py"),
                "historical_v37_results_used_in_selection": False}
    manifest_path = root / OUT / "sample_pool_manifest_v38.json"
    write_json_atomic(manifest_path, manifest)
    stage_freeze(root, "sample_pools",
                 [SOURCE, str(manifest_path.relative_to(root)),
                  *[str(OUT / f"{role}_pool_v38.json") for role in ROLES],
                  *[f"artifacts/trajectory_composition_v38_pool_{role}.freeze.json"
                    for role in ROLES]],
                 {"manifest_sha256": sha256_file(manifest_path),
                  "models_not_queried": True, "distribution_shift_disclosed": True})
    return manifest


if __name__ == "__main__":
    import sys
    print(json.dumps(plan(Path.cwd()) if sys.argv[1:] == ["plan"] else complete(Path.cwd()), indent=2))
