"""Frozen small-subset Conv-conditioning and REC-only quartile read controls."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from jclosure.experiments.depth_common_v35 import capture, patch, prepare
from jclosure.experiments.design_v34 import prompt_lookup
from jclosure.experiments.runtime_v34 import hd, load, native_swap, signature
from jclosure.experiments.transaction_v28 import thash
from jclosure.protocol_v35 import stage_freeze, verify_stage
from jclosure.provenance import sha256_file, write_json_atomic

OUT = Path("results/v35/processed")
SOURCE = "src/jclosure/experiments/controls_v35.py"


def cosine(a, b):
    norm = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / norm) if norm > 1e-12 else None


@torch.no_grad()
def run(root: Path, key: str):
    verify_stage(root, "control_plan")
    verify_stage(root, "full_gate_validation")
    plan = json.loads((root / OUT / "control_plan_v35.json").read_text())
    design = json.loads((root / OUT / f"design_{key}_v35.json").read_text())
    panel = json.loads((root / OUT / "panel_v35.json").read_text())
    lookup = prompt_lookup(root, panel)
    selected = set(plan["context_conv_rec_states"])
    factorial = pd.read_parquet(root / OUT / f"factorial_{key}_validation_v35.parquet")
    facts = np.load(root / OUT / f"factorial_vectors_{key}_validation_v35.npz")["vectors"].astype(np.float64)
    by_state = {row.state_id: row for row in factorial.itertuples()}
    model, tokenizer = load(root, key)
    rows, audits = [], []
    for n, item in enumerate((x for x in design["validation"] if x["base_trial_id"] in selected), 1):
        sid = item["base_trial_id"]
        caches = prepare(model, tokenizer, key, item, lookup[sid])
        rec, _ = native_swap(caches["recipient"], caches["donor"], ["REC"], key)
        length, bundle, scales = caches["length"], design["target_bundle"], design["calibration_clean_scales"]
        outputs = {q: {source: [] for source in ("JOINT", "REC")} for q in plan["quartiles"]}
        source_hashes = {q: {source: [] for source in ("JOINT", "REC")} for q in plan["quartiles"]}
        for probe_index, probe in enumerate(item["future_probe_tokens"]):
            references = {}
            for source, cache in (("JOINT", caches["joint"]), ("REC", rec)):
                _, ref = capture(model, key, cache, probe, length, bundle)
                references[source] = ref
            for q in plan["quartiles"]:
                layers = design["relative_depth_layers"][q]
                for source in ("JOINT", "REC"):
                    out, proof = patch(model, key, caches["conv"], probe, length, bundle,
                                       references, {layer: source for layer in layers})
                    outputs[q][source].append({"targets": out["targets"]})
                    source_hashes[q][source].append(hd([thash(references[source][layer]["RECURRENT_READ"])
                                                         for layer in layers]))
                    audits.append({"model_key": key, "state_id": sid, "quartile": q,
                                   "source": source, "probe_index": probe_index,
                                   "probe_token_id": int(probe), "recipient_KV_hash": caches["recipient_KV_hash"],
                                   "requested_hash": proof["requested_hash"],
                                   "realized_hash": proof["realized_hash"],
                                   "source_map_hash": proof["source_map_hash"],
                                   "per_layer_json": json.dumps(proof["per_layer"], sort_keys=True),
                                   "exact_writeback": proof["exact_writeback"]})
        factual = facts[int(by_state[sid].vector_index)]
        yd, yconv, yjoint = factual[4], factual[2], factual[3]
        benefit = float(np.linalg.norm(yd - yconv) - np.linalg.norm(yd - yjoint))
        correction = yjoint - yconv
        for q in plan["quartiles"]:
            vectors = {source: signature(outputs[q][source], scales) for source in ("JOINT", "REC")}
            values = {source: ((float(np.linalg.norm(yd - yconv)) -
                                float(np.linalg.norm(yd - vectors[source]))) / benefit)
                      if benefit > 1e-12 else None for source in ("JOINT", "REC")}
            rows.append({"model_key": key, "role": "validation", "state_id": sid,
                         "family": item["family"], "quartile": q, "benefit_positive": benefit > 1e-12,
                         "joint_restore": values["JOINT"], "rec_only_restore": values["REC"],
                         "joint_minus_rec_restore": values["JOINT"] - values["REC"]
                         if benefit > 1e-12 else None,
                         "joint_cosine": cosine(vectors["JOINT"] - yconv, correction),
                         "rec_only_cosine": cosine(vectors["REC"] - yconv, correction),
                         "joint_vs_rec_read_hash_different_probes": sum(a != b for a, b in zip(
                             source_hashes[q]["JOINT"], source_hashes[q]["REC"])),
                         "recipient_native_KV": True, "exact_writeback": True})
        print(f"V35 controls {key} {n}/{len(selected)}", flush=True)
    frame, audit = pd.DataFrame(rows), pd.DataFrame(audits)
    if len(frame) != 40 or len(audit) != 10 * 6 * 4 * 2 or not audit.exact_writeback.all():
        raise RuntimeError("V35 control record incomplete")
    paths = {"rows": root / OUT / f"controls_{key}_validation_v35.parquet",
             "audit": root / OUT / f"controls_audit_{key}_validation_v35.parquet"}
    frame.to_parquet(paths["rows"], index=False, compression="zstd")
    audit.to_parquet(paths["audit"], index=False, compression="zstd")
    result = {"model_key": key, "states": 10, "quartiles": {},
              "secondary_small_subset": True, "primary_KV": "recipient-native",
              "files_sha256": {name: sha256_file(path) for name, path in paths.items()}}
    for q, part in frame.groupby("quartile"):
        good = part[part.benefit_positive]
        result["quartiles"][q] = {"valid_states": len(good),
                                  "joint_restore_median": float(good.joint_restore.median()) if len(good) else None,
                                  "rec_only_restore_median": float(good.rec_only_restore.median()) if len(good) else None,
                                  "joint_minus_rec_restore_median": float(good.joint_minus_rec_restore.median()) if len(good) else None,
                                  "all_six_read_hashes_different_fraction":
                                  float((part.joint_vs_rec_read_hash_different_probes == 6).mean())}
    path = root / OUT / f"controls_{key}_validation_v35.json"
    write_json_atomic(path, result)
    seal = stage_freeze(root, f"controls_{key}", [SOURCE, *(str(x.relative_to(root)) for x in paths.values()),
                                                   str(path.relative_to(root)),
                                                   "artifacts/hierarchical_read_v35_control_plan.freeze.json",
                                                   "artifacts/hierarchical_read_v35_full_gate_validation.freeze.json"],
                        {"model_key": key, "summary_sha256": sha256_file(path)})
    return {"freeze_digest": seal["freeze_digest"], **result}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=("Q", "F"))
    args = parser.parse_args()
    print(json.dumps(run(Path.cwd(), args.model), indent=2))
