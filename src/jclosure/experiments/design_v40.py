"""Tokenizer-only V40 fork audit; no model forward or intervention outcomes."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml
from transformers import AutoTokenizer

from jclosure.datasets_v40 import FAMILIES, digest, generate
from jclosure.protocol_v40 import stage_freeze, verify, verify_stage
from jclosure.provenance import sha256_file, write_json_atomic

OUT = Path("results/v40/processed")
SOURCE = "src/jclosure/experiments/design_v40.py"
ROLES = ("calibration", "development", "validation", "independent_final")


def encode(tokenizer, key: str, prompt: str) -> list[int]:
    if key == "Q":
        tokens = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=True,
            add_generation_prompt=True, enable_thinking=False,
            return_tensors="pt")
        if isinstance(tokens, dict) or hasattr(tokens, "get"):
            tokens = tokens["input_ids"]
    else:
        tokens = tokenizer(prompt, add_special_tokens=True)["input_ids"]
    if hasattr(tokens, "tolist"):
        tokens = tokens.tolist()
    if len(tokens) == 1 and isinstance(tokens[0], list):
        tokens = tokens[0]
    return [int(x) for x in tokens]


def single_token_fork(recipient: list[int], donor: list[int]) -> dict:
    """Require equal-length encoded prompts with exactly one differing token."""
    if len(recipient) != len(donor):
        return {"eligible": False, "reason": "TOKEN_LENGTH_MISMATCH",
                "recipient_length": len(recipient), "donor_length": len(donor)}
    diffs = [i for i, (a, b) in enumerate(zip(recipient, donor, strict=True))
             if a != b]
    if len(diffs) != 1:
        return {"eligible": False, "reason": "NOT_ONE_TOKEN_FORK",
                "difference_count": len(diffs)}
    i = diffs[0]
    if i == 0 or i == len(recipient) - 1:
        return {"eligible": False, "reason": "NO_SHARED_PREFIX_OR_SUFFIX"}
    return {"eligible": True, "fork_index": i,
            "recipient_token_id": recipient[i], "donor_token_id": donor[i],
            "shared_prefix_hash": digest(recipient[:i]),
            "shared_suffix_hash": digest(recipient[i + 1:]),
            "recipient_ids_hash": digest(recipient), "donor_ids_hash": digest(donor),
            "encoded_length": len(recipient)}


def tokenizer_for(root: Path, key: str):
    spec_path = root / "configs/functional_mediation_v34.yaml"
    spec = yaml.safe_load(spec_path.read_text(encoding="utf-8"))["models"][key]
    location = Path(spec["local_path"])
    if sha256_file(location / "tokenizer.json") != spec["tokenizer_sha256"]:
        raise RuntimeError(f"V40 tokenizer drift: {key}")
    return AutoTokenizer.from_pretrained(location, local_files_only=True,
                                         trust_remote_code=False)


def preflight(root: Path, key: str) -> dict:
    """Check template feasibility before V40 base/pool sealing."""
    tokenizer = tokenizer_for(root, key)
    summary = {}
    cfg = yaml.safe_load((root / "configs/channel_function_v40.yaml").read_text(encoding="utf-8"))
    for family in FAMILIES:
        failures = {}
        checked = 0
        first_failure = None
        blocked = set()
        for role in ROLES:
            seed = int(cfg["pool_seeds"][role]) + 100_003 * FAMILIES.index(family)
            rows = generate(family, int(cfg["roles_per_family"][role]), seed, blocked)
            for row in rows:
                recipient = encode(tokenizer, key, row["recipient_prompt"])
                donor = encode(tokenizer, key, row["donor_prompt"])
                check = single_token_fork(recipient, donor)
                checked += 1
                if not check["eligible"]:
                    failures[check["reason"]] = failures.get(check["reason"], 0) + 1
                    if first_failure is None:
                        first_failure = {"recipient_length": len(recipient),
                                         "donor_length": len(donor),
                                         "diffs": [(i, recipient[i], donor[i],
                                                    tokenizer.decode([recipient[i]]),
                                                    tokenizer.decode([donor[i]]))
                                                   for i in range(min(len(recipient), len(donor)))
                                                   if recipient[i] != donor[i]][:8]}
        summary[family] = {"checked": checked, "failures": failures,
                           "first_failure": first_failure}
    return {"model": key, "preflight": summary, "no_model_forward": True}


def run(root: Path, key: str) -> dict:
    verify_stage(root, "sample_pools")
    cfg = verify(root)["config"]
    tokenizer = tokenizer_for(root, key)
    roles = {}
    for role in ROLES:
        verify_stage(root, f"pool_{role}")
        pool = json.loads((root / f"data/v40/{role}_pool_v40.json").read_text())
        rows = []
        for task in pool["items"]:
            check = single_token_fork(encode(tokenizer, key, task["recipient_prompt"]),
                                      encode(tokenizer, key, task["donor_prompt"]))
            if not check["eligible"]:
                raise RuntimeError(f"V40 fork ineligible {key}/{role}/{task['state_id']}: {check}")
            if task["recipient_answer"] == task["donor_answer"]:
                raise RuntimeError("V40 ground-truth contrast missing")
            rows.append({"state_id": task["state_id"], "role": role,
                         "family": task["family"], **check,
                         "recipient_answer": task["recipient_answer"],
                         "donor_answer": task["donor_answer"],
                         "labels_from_generator": True,
                         "intervention_outcome_observed": False})
        roles[role] = rows
    path = root / OUT / f"design_{key}_v40.json"
    spec = yaml.safe_load((root / cfg["model_spec_source"]).read_text(encoding="utf-8"))["models"][key]
    write_json_atomic(path, {"model_key": key, "model_id": spec["id"],
                             "roles": roles, "conditions": cfg["conditions"],
                             "delays": cfg["delays"],
                             "intervention_outcomes_observed_before_seal": False})
    seal = stage_freeze(root, f"design_{key}",
                        [SOURCE, str(path.relative_to(root)),
                         "artifacts/channel_function_v40_sample_pools.freeze.json"],
                        {"model": key, "role_counts": {r: len(x) for r, x in roles.items()},
                         "design_sha256": sha256_file(path),
                         "intervention_outcomes_observed_before_seal": False})
    return {"model": key, "role_counts": {r: len(x) for r, x in roles.items()},
            "freeze_digest": seal["freeze_digest"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=("Q", "F"))
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    print(json.dumps(preflight(Path.cwd(), args.model) if args.preflight else
                     run(Path.cwd(), args.model), indent=2))
