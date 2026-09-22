"""Exact native Falcon-H1 cache and response utilities for V33."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from jclosure.cache_v7 import clone_hybrid_cache
from jclosure.recorder import ActivationRecorder
from jclosure.experiments.transaction_v28 import thash
from jclosure.protocol_v33 import verify

FIELDS = ("recurrent_states", "conv_states", "keys", "values")


def hd(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()


def context(root: Path):
    cfg = verify(root)["config"]["model2"]
    model_path = cfg["local_path"]
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True, dtype=torch.bfloat16, device_map=cfg["device"], attn_implementation=cfg["attention_implementation"], trust_remote_code=False).eval()
    return model, tokenizer


def field_hashes(cache):
    result = {}
    for channel, field_names in (("REC2", ("recurrent_states",)), ("CONV2", ("conv_states",)), ("KV", ("keys", "values"))):
        h = hashlib.sha256()
        for layer, item in enumerate(cache.layers):
            for field in field_names:
                value = getattr(item, field, None)
                if not isinstance(value, torch.Tensor):
                    raise RuntimeError(f"V33 missing {field} in layer {layer}")
                h.update(f"{layer}:{field}:{tuple(value.shape)}:{value.dtype}:".encode())
                h.update(thash(value).encode())
        result[channel] = h.hexdigest()
    return result


def total_hash(hashes):
    return hd(hashes)


def audit_schema(cache):
    rows = []
    for layer, item in enumerate(cache.layers):
        for field in FIELDS:
            value = getattr(item, field, None)
            if not isinstance(value, torch.Tensor):
                raise RuntimeError(f"V33 missing field {field} at layer {layer}")
            rows.append({"layer": layer, "cache_layer_class": type(item).__name__, "field": field, "shape": list(value.shape), "dtype": str(value.dtype), "device": str(value.device), "sha256": thash(value)})
    return rows


def native_swap(recipient, donor, channels):
    """Clone recipient then exact-copy only named outgoing native fields.

    KV copies only the appended slot and verifies the common prefix; this
    prevents a donor-prefix history change from masquerading as a token write.
    """
    wanted = set(channels)
    if not wanted <= {"REC2", "CONV2", "KV"}:
        raise ValueError(wanted)
    if recipient.get_seq_length() != donor.get_seq_length() or len(recipient.layers) != len(donor.layers):
        raise RuntimeError("V33 cache length/layer mismatch")
    out = clone_hybrid_cache(recipient)
    touched = []
    for layer in range(len(out.layers)):
        for channel, fields in (("REC2", ("recurrent_states",)), ("CONV2", ("conv_states",)), ("KV", ("keys", "values"))):
            if channel not in wanted:
                continue
            for field in fields:
                base, source = getattr(recipient.layers[layer], field), getattr(donor.layers[layer], field)
                if base.shape != source.shape or base.dtype != source.dtype:
                    raise RuntimeError(f"V33 {field} shape/dtype mismatch layer {layer}")
                if channel == "KV":
                    if not torch.equal(base[..., :-1, :], source[..., :-1, :]):
                        raise RuntimeError("V33 donor/recipient KV history differs before fork")
                    replacement = base.detach().clone()
                    replacement[..., -1:, :] = source[..., -1:, :]
                else:
                    replacement = source.detach().clone()
                setattr(out.layers[layer], field, replacement)
                touched.append((layer, field))
    touched_set = set(touched)
    for layer in range(len(out.layers)):
        for field in FIELDS:
            actual = getattr(out.layers[layer], field)
            source = getattr(donor.layers[layer], field)
            base = getattr(recipient.layers[layer], field)
            expected = source if (layer, field) in touched_set else base
            if not torch.equal(actual, expected):
                raise RuntimeError(f"V33 exact native writeback failure {layer}:{field}")
    return out, {"channels": sorted(wanted), "requested_exact": True, "untouched_exact": True, "field_count": len(touched), "recipient_hashes": field_hashes(recipient), "donor_hashes": field_hashes(donor), "realized_hashes": field_hashes(out), "seq_length": int(out.get_seq_length())}


@torch.no_grad()
def prefix(model, tokenizer, prompt):
    ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=True).input_ids
    if ids.shape[1] < 2:
        raise RuntimeError("V33 prompt too short")
    device = next(model.parameters()).device
    out = model(input_ids=ids[:, :-1].to(device), use_cache=True)
    return clone_hybrid_cache(out.past_key_values), ids[:, -1:].to(device), int(ids.shape[1]), ids[:, :-1].tolist()


@torch.no_grad()
def step(model, cache, token_id, total_length, target_bundle=None):
    device = next(model.parameters()).device
    token = torch.as_tensor([[int(token_id)]], dtype=torch.long, device=device)
    cache_copy = clone_hybrid_cache(cache)
    if cache_copy.get_seq_length() != total_length - 1:
        raise RuntimeError(f"V33 position mismatch {cache_copy.get_seq_length()} != {total_length-1}")
    kwargs = {"input_ids": token, "past_key_values": cache_copy, "attention_mask": torch.ones((1, total_length), dtype=torch.long, device=device), "use_cache": True}
    if target_bundle is None:
        out = model(**kwargs)
        return {"cache": clone_hybrid_cache(out.past_key_values), "logits": out.logits[0, -1].float()}
    at = sorted(set(target_bundle["workspace_layers"] + [len(model.model.layers)-1]))
    with ActivationRecorder(model.model.layers, at=at, clone=True, detach=True) as recorder:
        out = model(**kwargs)
    logits = out.logits[0, -1].float()
    selected = torch.as_tensor(target_bundle["selected_logits"], device=device)
    broad = torch.as_tensor(target_bundle["broad_logits"], device=device)
    broad_sign = torch.as_tensor(target_bundle["broad_sign"], device=device, dtype=torch.float32)
    hidden = recorder.activations[len(model.model.layers)-1][0, -1].float()
    late_index = torch.as_tensor(target_bundle["late_hidden_indices"], device=hidden.device)
    targets = {"logits": logits[selected].cpu().numpy().astype(np.float32), "semantic": torch.log_softmax(logits, -1)[selected].cpu().numpy().astype(np.float32), "broad_vocabulary": (logits[broad]*broad_sign).cpu().numpy().astype(np.float32), "late_hidden": hidden[late_index].cpu().numpy().astype(np.float32), "workspace": torch.cat([recorder.activations[layer][0, -1].float()[:target_bundle["workspace_per_layer"]] for layer in target_bundle["workspace_layers"]]).cpu().numpy().astype(np.float32)}
    return {"cache": clone_hybrid_cache(out.past_key_values), "logits": logits, "targets": targets}


def signature(outputs, scales):
    order = ("logits", "semantic", "broad_vocabulary", "late_hidden", "workspace")
    return np.concatenate([np.concatenate([np.asarray(row["targets"][field], dtype=np.float64).ravel()/max(float(scales[field]), 1e-12) for field in order]) for row in outputs])
