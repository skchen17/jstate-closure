"""Response-blind tokenizer preflight for V40 answer and neutral-token design."""
from __future__ import annotations

import json
from pathlib import Path

from jclosure.experiments.design_v40 import encode, single_token_fork, tokenizer_for

FAMILY_ANSWERS = {
    "variable_binding": tuple("0123456789"),
    "state_transition": tuple("ABCD"),
    "long_context_dependency": tuple("0123456789"),
    "boolean_logic": tuple("01"),
}
CANDIDATES = (" .", " ;", "\n", " neutral", " -", "\n.")
DELAYS = (0, 1, 2, 4, 8, 16, 32)


def check(root: Path, key: str) -> dict:
    tokenizer = tokenizer_for(root, key)
    rows = json.loads((root / "data/v40/calibration_pool_v40.json").read_text())["items"]
    alphabet = {family: {answer: tokenizer.encode(answer, add_special_tokens=False)
                         for answer in answers}
                for family, answers in FAMILY_ANSWERS.items()}
    neutral = {}
    for candidate in CANDIDATES:
        token_ids = tokenizer.encode(candidate, add_special_tokens=False)
        if len(token_ids) != 1:
            neutral[repr(candidate)] = {"token_ids": token_ids, "eligible": False,
                                        "reason": "NOT_ONE_TOKEN"}
            continue
        token_id = int(token_ids[0])
        failures = []
        for row in rows:
            base_r = encode(tokenizer, key, row["recipient_prompt"])
            base_d = encode(tokenizer, key, row["donor_prompt"])
            base = single_token_fork(base_r, base_d)
            i = base.get("fork_index")
            if not base["eligible"]:
                failures.append([row["state_id"], "BASE_INELIGIBLE"])
                continue
            for delay in DELAYS:
                prompt_r = row["prefix"] + row["recipient_fork"] + candidate * delay + row["suffix"]
                prompt_d = row["prefix"] + row["donor_fork"] + candidate * delay + row["suffix"]
                ids_r, ids_d = encode(tokenizer, key, prompt_r), encode(tokenizer, key, prompt_d)
                fork = single_token_fork(ids_r, ids_d)
                if (not fork["eligible"] or fork["fork_index"] != i or
                    ids_r[:i+1] != base_r[:i+1] or
                    ids_r[i+1:i+1+delay] != [token_id] * delay or
                    ids_r[i+1+delay:] != base_r[i+1:] or
                    ids_d[i+1+delay:] != base_d[i+1:]):
                    failures.append([row["state_id"], delay])
        neutral[repr(candidate)] = {"token_id": token_id, "eligible": not failures,
                                    "failure_count": len(failures),
                                    "examples": failures[:5]}
    return {"model": key, "candidate_answer_token_ids": alphabet,
            "neutral_candidates": neutral,
            "states_checked": len(rows), "no_model_forward": True}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=("Q", "F"))
    args = parser.parse_args()
    print(json.dumps(check(Path.cwd(), args.model), indent=2))
