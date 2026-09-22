"""Exact V34-native mixer-output patches on prospective V35 depth subsets."""
from __future__ import annotations

from contextlib import AbstractContextManager

import torch

from jclosure.experiments.runtime_v34 import hd
from jclosure.experiments.transaction_v28 import thash


class ReadPatch(AbstractContextManager):
    """Patch only selected native recurrent/mixer outputs; remove hooks on exit."""

    def __init__(self, model, key, references, source_by_layer):
        self.model, self.key = model, key
        self.references = references
        self.source_by_layer = dict(source_by_layer)
        self.handles = []
        self.audit = {}

    def __enter__(self):
        for layer, source in sorted(self.source_by_layer.items()):
            block = self.model.model.layers[layer]
            module = block.linear_attn if self.key == "Q" else block.mamba
            requested = self.references[source][layer]["RECURRENT_READ"]

            def hook(_module, _inputs, output, layer=layer, source=source, requested=requested):
                if requested.shape != output.shape or requested.dtype != output.dtype or requested.device != output.device:
                    raise RuntimeError(f"V35 read patch topology mismatch {layer}")
                realized = requested.detach().clone()
                if not torch.equal(realized, requested):
                    raise RuntimeError(f"V35 read patch writeback mismatch {layer}")
                self.audit[layer] = {"source": source, "requested_hash": thash(requested),
                                     "realized_hash": thash(realized), "native_hash": thash(output),
                                     "exact_writeback": True}
                return realized

            self.handles.append(module.register_forward_hook(hook))
        return self

    def __exit__(self, exc_type, exc, tb):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        return False

    def proof(self):
        if set(self.audit) != set(self.source_by_layer):
            raise RuntimeError("V35 incomplete depth patch")
        if not all(record["exact_writeback"] and record["requested_hash"] == record["realized_hash"]
                   for record in self.audit.values()):
            raise RuntimeError("V35 depth requested/realized mismatch")
        ordered = [(layer, record) for layer, record in sorted(self.audit.items())]
        return {"layers": len(ordered), "source_map_hash": hd(self.source_by_layer),
                "requested_hash": hd([x["requested_hash"] for _, x in ordered]),
                "realized_hash": hd([x["realized_hash"] for _, x in ordered]),
                "native_hash": hd([x["native_hash"] for _, x in ordered]),
                "per_layer": {str(layer): record for layer, record in ordered},
                "exact_writeback": True}
