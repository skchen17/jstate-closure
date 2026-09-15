from __future__ import annotations

import json
from types import SimpleNamespace

import torch

from jclosure.cache_v7 import (
    cache_component_differences,
    caches_exact,
    clone_hybrid_cache,
    describe_cache,
    make_chimeric_cache,
)
from jclosure.protocol_v7 import digest
from jclosure.protocol_v7_stage2 import freeze_path
from jclosure.records_v7 import CacheRestoreRecord


def _cache() -> SimpleNamespace:
    attention = SimpleNamespace(
        keys=torch.arange(24, dtype=torch.float32).reshape(1, 2, 3, 4),
        values=torch.arange(24, 48, dtype=torch.float32).reshape(1, 2, 3, 4),
    )
    recurrent = SimpleNamespace(
        recurrent_states=torch.arange(16, dtype=torch.float32).reshape(1, 1, 4, 4),
        conv_states=torch.arange(8, dtype=torch.float32).reshape(1, 2, 4),
        has_previous_state=True,
    )
    return SimpleNamespace(layers=[attention, recurrent])


def test_hybrid_cache_clone_is_tensor_independent() -> None:
    source = _cache()
    cloned = clone_hybrid_cache(source)
    assert caches_exact(source, cloned, ("keys", "recurrent_states", "conv_states"))
    cloned.layers[0].keys.add_(1)
    assert not torch.equal(source.layers[0].keys, cloned.layers[0].keys)


def test_two_by_two_chimeric_cache_channels_are_independent() -> None:
    clean = _cache()
    perturbed = clone_hybrid_cache(clean)
    perturbed.layers[0].keys.add_(10)
    perturbed.layers[0].values.add_(20)
    perturbed.layers[1].recurrent_states.add_(30)
    perturbed.layers[1].conv_states.add_(40)
    kv = make_chimeric_cache(clean, perturbed, kv_from_perturbed=True)
    rec = make_chimeric_cache(
        clean, perturbed, recurrent_from_perturbed=True, conv_from_perturbed=True
    )
    assert torch.equal(kv.layers[0].keys, perturbed.layers[0].keys)
    assert torch.equal(kv.layers[1].recurrent_states, clean.layers[1].recurrent_states)
    assert torch.equal(rec.layers[0].keys, clean.layers[0].keys)
    assert torch.equal(
        rec.layers[1].recurrent_states, perturbed.layers[1].recurrent_states
    )
    assert torch.equal(rec.layers[1].conv_states, perturbed.layers[1].conv_states)


def test_kv_localization_preserves_unselected_heads_and_positions() -> None:
    clean = _cache()
    perturbed = clone_hybrid_cache(clean)
    perturbed.layers[0].keys.add_(100)
    output = make_chimeric_cache(
        clean,
        perturbed,
        kv_from_perturbed=True,
        attention_layers=[0],
        kv_heads=[1],
        kv_positions=[-1],
    )
    assert torch.equal(
        output.layers[0].keys[:, 1, -1], perturbed.layers[0].keys[:, 1, -1]
    )
    assert torch.equal(output.layers[0].keys[:, 0], clean.layers[0].keys[:, 0])
    assert torch.equal(
        output.layers[0].keys[:, 1, :-1], clean.layers[0].keys[:, 1, :-1]
    )


def test_cache_schema_and_differences_are_component_explicit() -> None:
    clean = _cache()
    perturbed = clone_hybrid_cache(clean)
    perturbed.layers[1].conv_states.add_(1)
    schema = describe_cache(clean, ["full_attention", "linear_attention"])
    assert {row["component"] for row in schema} == {
        "keys",
        "values",
        "recurrent_states",
        "conv_states",
    }
    assert next(row for row in schema if row["component"] == "keys")["token_axis"] == -2
    differences = cache_component_differences(clean, perturbed)
    conv = next(row for row in differences if row["component"] == "conv_states")
    assert conv["nonzero"] == 8


def test_v7_record_round_trip_and_digest() -> None:
    record = CacheRestoreRecord(
        "run", "prompt", "family", 3, True, 0, 0, 0, True, True, True
    )
    payload = json.loads(json.dumps(record.to_dict()))
    assert payload["schema_version"] == 9
    assert payload["protocol_version"] == "persistent_channel_attribution_protocol_v7"
    assert digest({"a": 1}) != digest({"a": 2})
    assert freeze_path("localization").name == "channel_localization_v7.freeze.json"
    assert freeze_path("compression").name == "arch_compression_v7.freeze.json"
