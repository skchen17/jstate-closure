"""Pre-outcome structural and exact-write comparability audit for Falcon-H1."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import torch

from jclosure.experiments.runtime_v33 import audit_schema, context, field_hashes, hd, native_swap, prefix, step
from jclosure.protocol_v33 import stage_freeze, verify
from jclosure.provenance import write_json_atomic

OUT = Path("results/v33/processed")


@torch.no_grad()
def run(root: Path):
    cfg = verify(root)["config"]
    old = json.loads((root / "results/v32/processed/design_v32.json").read_text())
    example = old["calibration"][0]
    source = root / f"results/v18/processed/crossed_state_{example['source_role']}_{example['family']}_v18.parquet"
    frame = pd.read_parquet(source, filters=[[("base_trial_id", "==", example["base_trial_id"])]] )
    prompt = str(frame.iloc[0]["prompt"])
    if hashlib.sha256(prompt.encode()).hexdigest() != example["prompt_sha256"]:
        raise RuntimeError("V33 audit prompt mismatch")
    model, tokenizer = context(root)
    incoming, _, length, prefix_ids = prefix(model, tokenizer, prompt)
    logits = model(input_ids=torch.tensor(prefix_ids, device=next(model.parameters()).device), use_cache=False).logits[0, -1].float()
    anchor = tokenizer.encode(cfg["token_rule"]["anchor_surface"], add_special_tokens=False)
    if len(anchor) != 1:
        raise RuntimeError("V33 anchor is not a single token")
    rank = torch.topk(logits, k=cfg["token_rule"]["rank_max_per_state"]).indices.tolist()
    other = next(int(i) for i in rank if int(i) != anchor[0] and tokenizer.decode([int(i)]).strip())
    recipient = step(model, incoming, anchor[0], length)["cache"]
    donor = step(model, incoming, other, length)["cache"]
    if recipient.get_seq_length() != length or donor.get_seq_length() != length:
        raise RuntimeError("V33 cache position mismatch")
    rows = audit_schema(recipient)
    if len(rows) != model.config.num_hidden_layers * 4:
        raise RuntimeError("V33 missing persistent tensor")
    changes = {}
    for channels in ([], ["REC2"], ["CONV2"], ["REC2", "CONV2"], ["KV"], ["REC2", "CONV2", "KV"]):
        swapped, proof = native_swap(recipient, donor, channels)
        key = "+".join(channels) or "RECIPIENT"
        changes[key] = proof
        if len(channels) == 3:
            if field_hashes(swapped) != field_hashes(donor):
                raise RuntimeError("V33 all-channel transplant not exact donor")
    sample = rows[0]
    mapping = {"REC2": {"field": "recurrent_states", "semantic": "Mamba2 SSM matrix state after current-token recurrence", "incoming_outgoing": "pre-token state to post-token state", "persists_across_tokens": True, "shape_example": next(r["shape"] for r in rows if r["field"] == "recurrent_states")}, "CONV2": {"field": "conv_states", "semantic": "Mamba2 depthwise short-convolution rolling input history", "incoming_outgoing": "pre-token kernel history to post-token kernel history", "persists_across_tokens": True, "shape_example": next(r["shape"] for r in rows if r["field"] == "conv_states")}, "KV": {"fields": ["keys", "values"], "semantic": "attention K/V cache with RoPE-positioned appended current slot", "incoming_outgoing": "shared prefix plus current-token slot", "persists_across_tokens": True, "shape_example": next(r["shape"] for r in rows if r["field"] == "keys")}}
    result = {"model": cfg["model2"], "transformers_version": __import__("transformers").__version__, "torch_version": torch.__version__, "model_layers": model.config.num_hidden_layers, "layers_block_type": model.config.layers_block_type, "context_limit": cfg["model2"]["context_limit_for_experiment"], "prompt_sha256": example["prompt_sha256"], "prefix_token_hash": hd(prefix_ids), "fork_total_length": length, "anchor_id": anchor[0], "other_id": other, "recipient_hashes": field_hashes(recipient), "donor_hashes": field_hashes(donor), "schema": rows, "mapping": mapping, "mapping_hash": hd(mapping), "write_proofs": changes, "readout_boundary": "after current-token forward/logits; next-token probes only", "cache_lengths_valid": True, "untouched_recipient_fields_exact": True, "exact_native_field_transplant": True, "functional_structural_comparable": True, "identical_micro_mechanism_claimed": False, "future_causal_response_observed": False}
    OUT.mkdir(parents=True, exist_ok=True)
    path = root / OUT / "architecture_audit_v33.json"
    write_json_atomic(path, result)
    pd.DataFrame(rows).to_parquet(root / OUT / "architecture_fields_v33.parquet", index=False)
    stage = stage_freeze(root, "architecture", ["src/jclosure/experiments/runtime_v33.py", "src/jclosure/experiments/audit_v33.py", str(path.relative_to(root)), str((OUT / "architecture_fields_v33.parquet"))], {"mapping_hash": result["mapping_hash"], "interface_comparable": True, "future_causal_response_observed": False})
    return {"mapping_hash": result["mapping_hash"], "freeze_digest": stage["freeze_digest"], "schema_rows": len(rows), "token_ids": [anchor[0], other], "cache_length": length}


if __name__ == "__main__":
    print(json.dumps(run(Path.cwd()), indent=2))
