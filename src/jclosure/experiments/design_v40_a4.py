"""Response-blind V40 formal readout and negative-control design."""
from __future__ import annotations

import json
from pathlib import Path

import yaml

from jclosure.experiments.calibration_v40_a2 import answer_token_ids
from jclosure.experiments.design_v40 import encode, single_token_fork, tokenizer_for
from jclosure.protocol_v40_a1 import verify_stage as verify_startup_stage
from jclosure.provenance import write_json_atomic

CONFIG = Path("configs/channel_function_v40_formal_a4.yaml")
ROLES = ("calibration", "development", "validation", "independent_final")
MODELS = ("Q", "F")


def alternative_value(task: dict) -> tuple[str, str] | None:
    """Return a third natural fork value and its externally computed answer."""
    family = task["family"]
    if family == "boolean_logic":
        return None
    if family == "state_transition":
        alphabet = "ABCD"
        occupied = {task["ast"]["recipient_state"], task["ast"]["donor_state"]}
        value = next(x for x in alphabet if x not in occupied)
        cycle = task["ast"]["cycle"]
        steps = task["ast"]["steps"]
        return value, cycle[(cycle.index(value) + steps) % 4]
    occupied = {task["recipient_fork"], task["donor_fork"]}
    value = next(x for x in "0123456789" if x not in occupied)
    return value, value


def control_prompts(task: dict, marker_pair: list[str]) -> dict:
    a, b = marker_pair
    original = task["recipient_prompt"]
    surface = (f"Presentation marker {a}.\n" + original,
               f"Presentation marker {b}.\n" + original)
    history = (f"Earlier unrelated note {a}. " + "Unrelated line. " * 12 + original,
               f"Earlier unrelated note {b}. " + "Unrelated line. " * 12 + original)
    mismatch = alternative_value(task)
    return {"surface_only": surface, "irrelevant_history": history,
            "same_prefix_mismatch": None if mismatch is None else
            (task["prefix"] + mismatch[0] + task["suffix"], mismatch[1])}


def plan(root: Path) -> dict:
    cfg = yaml.safe_load((root / CONFIG).read_text(encoding="utf-8"))
    if cfg["readout"] != "after_one_space" or cfg["temporal_delays"] != [0, 1, 2, 4, 8, 16, 32]:
        raise RuntimeError("V40 a4 design drift")
    result = {"protocol_version": cfg["protocol_version"], "models": {},
              "no_model_forward_or_intervention_outcome": True}
    for key in MODELS:
        verify_startup_stage(root, f"design_{key}")
        tokenizer = tokenizer_for(root, key)
        space_ids = tokenizer.encode(" ", add_special_tokens=False)
        neutral_ids = tokenizer.encode(" neutral", add_special_tokens=False)
        if len(space_ids) != 1 or len(neutral_ids) != 1:
            raise RuntimeError(f"V40 a4 readout/neutral is not one token: {key}")
        answer_ids = {family: answer_token_ids(tokenizer, chars)
                      for family, chars in cfg["answer_alphabets"].items()}
        model_plan = {"space_token_id": int(space_ids[0]),
                      "neutral_token_id": int(neutral_ids[0]),
                      "answer_ids": answer_ids, "roles": {}}
        for role in ROLES:
            verify_startup_stage(root, f"pool_{role}")
            items = json.loads((root / f"data/v40/{role}_pool_v40.json").read_text())['items']
            design = json.loads((root / f"results/v40/processed/design_{key}_v40.json").read_text())["roles"][role]
            if len(items) != len(design):
                raise RuntimeError("V40 a4 pool/design count drift")
            rows = []
            for task, old in zip(items, design, strict=True):
                if task["state_id"] != old["state_id"]:
                    raise RuntimeError("V40 a4 state order drift")
                base_r = encode(tokenizer, key, task["recipient_prompt"])
                base_d = encode(tokenizer, key, task["donor_prompt"])
                fork = single_token_fork(base_r, base_d)
                if not fork["eligible"] or any(fork[k] != old[k] for k in
                    ("fork_index", "recipient_token_id", "donor_token_id", "shared_prefix_hash", "shared_suffix_hash")):
                    raise RuntimeError(f"V40 a4 primary fork drift: {key}:{role}:{task['state_id']}")
                for delay in cfg["temporal_delays"]:
                    rt = task["prefix"] + task["recipient_fork"] + " neutral" * delay + task["suffix"]
                    dt = task["prefix"] + task["donor_fork"] + " neutral" * delay + task["suffix"]
                    ri, di = encode(tokenizer, key, rt), encode(tokenizer, key, dt)
                    i = fork["fork_index"]
                    if (ri[:i+1] != base_r[:i+1] or di[:i+1] != base_d[:i+1]
                        or ri[i+1:i+1+delay] != [neutral_ids[0]] * delay
                        or ri[i+1+delay:] != base_r[i+1:]
                        or di[i+1+delay:] != base_d[i+1:]):
                        raise RuntimeError(f"V40 a4 delay fork drift: {key}:{role}:{task['state_id']}:{delay}")
                controls = {}
                for kind, prompts in control_prompts(task, cfg["control_marker_pair"]).items():
                    if prompts is None:
                        controls[kind] = {"available": False, "reason": "BINARY_FORK_NO_THIRD_VALUE"}
                    elif kind == "same_prefix_mismatch":
                        text, answer = prompts
                        alt = encode(tokenizer, key, text)
                        if len(alt) != len(base_r) or alt[:fork["fork_index"]] != base_r[:fork["fork_index"]] or alt[fork["fork_index"]+1:] != base_r[fork["fork_index"]+1:]:
                            raise RuntimeError(f"V40 a4 mismatch tokenization: {key}:{role}:{task['state_id']}")
                        controls[kind] = {"available": True, "token_id": alt[fork["fork_index"]],
                                          "answer": answer, "prompt": text}
                    else:
                        left, right = prompts
                        li, ri = encode(tokenizer, key, left), encode(tokenizer, key, right)
                        check = single_token_fork(li, ri)
                        if not check["eligible"]:
                            raise RuntimeError(f"V40 a4 {kind} tokenization: {key}:{role}:{task['state_id']}:{check}")
                        controls[kind] = {"available": True, "left_prompt": left,
                                          "right_prompt": right,
                                          "fork_index": check["fork_index"],
                                          "left_token_id": check["recipient_token_id"],
                                          "right_token_id": check["donor_token_id"],
                                          "prefix_length": check["fork_index"]}
                rows.append({"state_id": task["state_id"], "family": task["family"],
                             "donor_answer": task["donor_answer"],
                             "fork_index": fork["fork_index"],
                             "recipient_token_id": fork["recipient_token_id"],
                             "donor_token_id": fork["donor_token_id"],
                             "prefix_length": fork["fork_index"],
                             "controls": controls})
            for row in rows:
                candidates = [other for other in rows if other["state_id"] != row["state_id"]
                              and other["prefix_length"] == row["prefix_length"]
                              and other["donor_answer"] != row["donor_answer"]]
                candidates.sort(key=lambda other: (other["family"] != row["family"],
                                                   other["state_id"]))
                row["controls"]["shuffled_REC_Conv"] = (
                    {"available": True, "source_state_id": candidates[0]["state_id"]}
                    if candidates else {"available": False, "reason": "NO_EQUAL_LENGTH_DONOR"})
            model_plan["roles"][role] = rows
        result["models"][key] = model_plan
    return result


if __name__ == "__main__":
    root = Path.cwd()
    path = root / "results/v40/processed/formal_design_a4_v40.json"
    if (root / "artifacts/channel_function_v40_a4.freeze.json").exists():
        raise RuntimeError("V40 a4 is frozen; design cannot be regenerated")
    value = plan(root)
    write_json_atomic(path, value)
    print(json.dumps({key: {role: len(rows) for role, rows in model["roles"].items()}
                      for key, model in value["models"].items()}, indent=2))
