"""Machine-readable per-token/state/atom/composition/depth/interaction hashes."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from jclosure.experiments.primitive_matrix_v31 import OUT, SCRATCH
from jclosure.protocol_v31 import stage_freeze, verify_stage
from jclosure.provenance import sha256_file, write_json_atomic

SOURCE = "src/jclosure/experiments/hash_manifest_v31.py"


def hd(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False, default=str).encode()).hexdigest()


def run(root: Path):
    for stage in ("design", "primitive_dictionary_fit", "mechanism_development", "mechanism_validation", "primitive_causal_development", "final_opening"):
        verify_stage(root, stage)
    design = json.loads((root / OUT / "design_v31.json").read_text())
    fit = json.loads((root / OUT / "primitive_dictionary_fit_v31.json").read_text())
    dim = fit["write_dimension"]
    atoms = {}
    for name, filename, count in (("PCA", "basis_PCA256_v31.f32", 256), ("SPARSE_DICTIONARY", "basis_SPARSE128_v31.f32", 128), ("CLUSTERED_PROTOTYPES", "basis_PROTOTYPE128_v31.f32", 128)):
        matrix = np.memmap(SCRATCH / filename, dtype=np.float32, mode="r", shape=(count, dim))
        atoms[name] = [hashlib.sha256(memoryview(np.ascontiguousarray(matrix[i]))).hexdigest() for i in range(count)]
    interaction = {}
    for role in ("development", "validation"):
        frame = pd.read_parquet(root / OUT / f"rec_conv_factorial_{role}_v31.parquet")
        interaction[role] = {f"{row.state_id}:{row.composition_id}": hd({col: getattr(row, col) for col in frame.columns}) for row in frame.itertuples(index=False)}
    route = json.loads((root / OUT / "depth_route_selection_v31.json").read_text())
    payload = {"token_pair_hashes": {x["pair_id"]: x["token_pair_hash"] for x in design["token_pair_library"]}, "incoming_state_hashes": {x["base_trial_id"]: x["incoming_state_hash"] for role in ("calibration", "development", "validation", "independent_final") for x in design[role]}, "primitive_atom_tensor_hashes": atoms, "dictionary_file_hashes": {name: fit[name]["basis_sha256"] for name in ("PCA", "SPARSE_DICTIONARY", "CLUSTERED_PROTOTYPES")}, "composition_ids_and_hashes": {x["composition_id"]: hd([x["A"], x["B"], x["AB"], x["role"]]) for group in design["surface_composition_splits"].values() for x in group}, "depth_route_layer_hash": route["layer_hash"], "interaction_metric_record_hashes": interaction, "interaction_hash_scope": "Canonical six-probe factorial metric rows, not unrecoverable response-vector bytes", "final_opening_hash": sha256_file(root / OUT / "final_opening_v31.json"), "large_primitive_tensors_off_repository": True}
    path = root / OUT / "v31_record_hash_manifest.json"
    write_json_atomic(path, payload)
    freeze = stage_freeze(root, "record_hash_manifest", [SOURCE, str(path.relative_to(root)), "artifacts/compositional_natural_writes_v31_primitive_dictionary_fit.freeze.json", "artifacts/compositional_natural_writes_v31_mechanism_validation.freeze.json", "artifacts/compositional_natural_writes_v31_final_opening.freeze.json"], {"manifest_sha256": sha256_file(path), "atom_counts": {k: len(v) for k, v in atoms.items()}, "interaction_record_counts": {k: len(v) for k, v in interaction.items()}})
    return {"freeze_digest": freeze["freeze_digest"], "atoms": {k: len(v) for k, v in atoms.items()}, "interaction_rows": {k: len(v) for k, v in interaction.items()}}


if __name__ == "__main__":
    print(json.dumps(run(Path.cwd()), indent=2))
