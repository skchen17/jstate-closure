"""Gated, restricted raw-sketch search for same-J response modulation M."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from jclosure.experiments import counterfactual_bank_v19 as bank
from jclosure.experiments.actuation_v15 import apply
from jclosure.experiments import geometry_v13 as geometry
from jclosure.protocol_v19 import stage_freeze, verify_stage
from jclosure.provenance import sha256_file, write_json_atomic

SOURCE = "src/jclosure/experiments/compact_response_v19.py"
FREEZE = "compact_response"
SCRATCH = Path("/data/CSK/J-space-project/v19-compact-raw-work")
WIDTHS = {"recurrent": 3145728, "conv": 196608, "kv": 1048576}
SKETCH = 128
DIMS = (4, 8, 16, 32, 64, 128)


def _action_descriptors(root: Path, actions: list[dict]) -> dict[str, list[float]]:
    directions = torch.load(root / geometry.DIRECTIONS, map_location="cpu", weights_only=False)
    vectors = {}
    for action in actions:
        index = int(action["direction_index"])
        vectors[int(action["coordinate_index"])] = [directions[name][index].float().reshape(-1) for name in ("recurrent", "conv", "kv")]
    anchor_coords = [5, 1, 7, 20]
    result = {}
    for action in actions:
        coord = int(action["coordinate_index"])
        parts = vectors[coord]
        norms = [float(torch.linalg.vector_norm(x)) for x in parts]
        total = math.sqrt(sum(x*x for x in norms))
        cosines = []
        for anchor in anchor_coords:
            other = vectors[anchor]
            dot = sum(float(torch.dot(a, b)) for a, b in zip(parts, other, strict=True))
            other_norm = math.sqrt(sum(float(torch.dot(x, x)) for x in other))
            cosines.append(dot / max(total * other_norm, 1e-12))
        result[str(coord)] = [*norms, *cosines, float(action["v16_train_median_j_effect_norm"])]
    scale = np.max(np.abs(np.asarray(list(result.values()), dtype=float)), axis=0)
    scale = np.maximum(scale, 1e-8)
    return {key: (np.asarray(value) / scale).tolist() for key, value in result.items()}


def prepare(root: Path) -> dict:
    decision = json.loads((root / bank.OUT / "v19_adjudication.json").read_text())
    if not decision["compact_response_context_search_authorized"]:
        raise RuntimeError("V19 C_response search not authorized")
    split = verify_stage(root, "splits")
    q = verify_stage(root, "q_amendment_2")
    runtime = verify_stage(root, "runtime_gpu1_amendment_4")
    descriptors = _action_descriptors(root, split["actions"])
    return stage_freeze(root, FREEZE,
                        [SOURCE, "src/jclosure/experiments/compact_gpu1_v19.py",
                         "artifacts/counterfactual_workspace_v19_runtime_gpu1_amendment_4.freeze.json",
                         "artifacts/counterfactual_workspace_v19_q_amendment_2.freeze.json",
                         "results/v19/processed/v19_adjudication.json",
                         "results/v18/processed/clean_state_scores_v18.npz",
                         str(geometry.DIRECTIONS)],
                        {"why": "Matched-action same-J modulation passed the pre-frozen V19 dependence gate; target its residual M, not generic future-J prediction.",
                         "responses_already_observed": {"train_states": 400, "validation_states": 100,
                                                        "independent_final_states": 0},
                         "trigger_decision_digest": sha256_file(root / bank.OUT / "v19_adjudication.json"),
                         "q_freeze_digest": q["freeze_digest"],
                         "gpu1_validation_extraction_runtime_digest": runtime["freeze_digest"],
                         "raw_state_descriptor": "direct_post_q_minus_p0_BF16_REC_Conv_KV_CountSketch_384D",
                         "sketch_seed": 19190, "sketch_width_per_channel": SKETCH,
                         "sketch_channel_widths": WIDTHS,
                         "candidate_dimensions": list(DIMS),
                         "representation": "train_only_supervised_residual_PLS_of_M_across_8_development_actions",
                         "model": "fixed_ridge_bilinear_state_action_lambda_1",
                         "primary_target": "V16_normalized_stack_M_288D_h1",
                         "development_actions": [5, 1, 7, 20], "heldout_actions": [11, 19, 17, 2],
                         "action_descriptors": descriptors,
                         "train_validation_split": split["freeze_digest"],
                         "selection": "minimum_development_validation_stack_relative_l2",
                         "candidate_gate": {"maximum_validation_dev_M_relative_l2": 0.5,
                                            "maximum_validation_heldout_M_relative_l2": 0.6,
                                            "maximum_raw_sketch_incremental_gain": 0.02,
                                            "requires_all_five_families": True},
                         "raw_limit": "384D CountSketch is a restricted measurement of actual Pq-P0 cache delta, not complete raw persistent state; no state sufficiency or independent-final eligibility from this search alone",
                         "independent_final_opened": False})


def _signs(device: torch.device) -> dict[str, torch.Tensor]:
    result = {}
    for offset, (name, width) in enumerate(WIDTHS.items()):
        rng = np.random.default_rng(19190 + offset)
        values = (rng.integers(0, 2, size=width, dtype=np.int8).astype(np.float32) * 2 - 1)
        result[name] = torch.from_numpy(values).to(device)
    return result


def _flatten(cache, rec: list[int], att: list[int]) -> dict[str, torch.Tensor]:
    recurrent = torch.cat([cache.layers[layer].recurrent_states.float().reshape(-1) for layer in rec])
    conv = torch.cat([cache.layers[layer].conv_states.float().reshape(-1) for layer in rec])
    kv_parts = []
    for layer in att:
        for attr in ("keys", "values"):
            value = getattr(cache.layers[layer], attr).float()
            if value.shape[-2] > 256:
                raise RuntimeError("V19 raw sketch KV prompt exceeds frozen width 256")
            padded = F.pad(value, (0, 0, 0, 256 - value.shape[-2]))
            kv_parts.append(padded.reshape(-1))
    kv = torch.cat(kv_parts)
    result = {"recurrent": recurrent, "conv": conv, "kv": kv}
    for name, tensor in result.items():
        if tensor.numel() != WIDTHS[name]:
            raise RuntimeError(f"V19 raw sketch {name} width {tensor.numel()} != {WIDTHS[name]}")
    return result


@torch.no_grad()
def _sketch(cache, rec: list[int], att: list[int], signs: dict[str, torch.Tensor]) -> np.ndarray:
    parts = _flatten(cache, rec, att)
    output = []
    for name in WIDTHS:
        tensor = parts[name]
        rows = tensor.numel() // SKETCH
        projected = (tensor.reshape(rows, SKETCH) * signs[name].reshape(rows, SKETCH)).sum(dim=0) / math.sqrt(rows)
        output.append(projected.cpu().numpy().astype(np.float32))
    return np.concatenate(output)


def _record_path(role: str, base_id: str) -> Path:
    return SCRATCH / role / f"raw_sketch_{base_id}.parquet"


def extract(root: Path, role: str, limit: int | None = None) -> dict:
    design = verify_stage(root, FREEZE)
    split = verify_stage(root, "splits")
    q = verify_stage(root, "q_amendment_2")
    bundle, dense, _, values, rec, att, measured, state_layer, ws_layers, ws_count = bank._setup(root)
    device = next(bundle.hf_model.parameters()).device
    signs = _signs(device)
    items = split[role][:limit] if limit is not None else split[role]
    (SCRATCH / role).mkdir(parents=True, exist_ok=True)
    completed = 0
    for number, item in enumerate(items, 1):
        path = _record_path(role, item["base_trial_id"])
        if path.exists():
            completed += 1
            continue
        old = bank._state_metadata(root, item)
        clean = bank._prefill_history(bundle, old["prompt"], measured, dense, state_layer)
        p0 = clean["cache"]
        baseline = _sketch(p0, rec, att, signs)
        reference = pd.read_parquet(bank.SCRATCH / role / f"factorial_{item['base_trial_id']}.parquet")
        rows = []
        for proposal in q["q"]:
            if proposal["calibration_status"] != "RELIABLE":
                continue
            row = bank._q_row(values["directions"], proposal, float(proposal["alpha"]))
            pq = apply(p0, 1.0, row, rec, att, "native_fp32_add_bf16_writeback")
            expected = reference[reference.q_name == proposal["name"]].iloc[0]
            if bank._snapshot_hash(p0, rec, att) != expected.p0_snapshot_hash or bank._snapshot_hash(pq, rec, att) != expected.pq_snapshot_hash:
                raise RuntimeError(f"V19 compact replay snapshot hash mismatch: {item['base_trial_id']} {proposal['name']}")
            raw = _sketch(pq, rec, att, signs)
            rows.append({"base_trial_id": item["base_trial_id"], "family": item["family"], "role": role,
                         "q_name": proposal["name"], "q_channel": proposal["channel"],
                         "raw_pq_minus_p0_sketch": (raw - baseline).tolist(),
                         "raw_p0_sketch": baseline.tolist(), "raw_pq_sketch": raw.tolist(),
                         "p0_snapshot_hash": expected.p0_snapshot_hash, "pq_snapshot_hash": expected.pq_snapshot_hash,
                         "compact_freeze_digest": design["freeze_digest"]})
        pd.DataFrame(rows).to_parquet(path, index=False, compression="zstd")
        completed += 1
        if number % 20 == 0 or number == len(items):
            print(f"V19 compact raw extract {role} {number}/{len(items)}", flush=True)
    return {"role": role, "completed": completed, "selected": len(items)}


def aggregate(root: Path, role: str) -> dict:
    design = verify_stage(root, FREEZE)
    split = verify_stage(root, "splits")
    outputs = []
    for family in sorted({x["family"] for x in split[role]}):
        items = [x for x in split[role] if x["family"] == family]
        paths = [_record_path(role, x["base_trial_id"]) for x in items]
        if not all(path.exists() for path in paths):
            raise RuntimeError(f"V19 compact {role}/{family} incomplete")
        frame = pd.concat([pd.read_parquet(path) for path in paths], ignore_index=True)
        path = root / bank.OUT / f"compact_raw_{role}_{family}_v19.parquet"
        frame.to_parquet(path, index=False, compression="zstd")
        outputs.append({"path": str(path.relative_to(root)), "rows": len(frame),
                        "states": int(frame.base_trial_id.nunique()), "sha256": sha256_file(path)})
    result = {"role": role, "outputs": outputs, "compact_freeze_digest": design["freeze_digest"]}
    write_json_atomic(root / bank.OUT / f"compact_raw_{role}_v19.json", result)
    return result


def _load(root: Path, role: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = pd.concat([pd.read_parquet(path) for path in sorted((root / bank.OUT).glob(f"compact_raw_{role}_*_v19.parquet"))], ignore_index=True)
    factorial = pd.concat([pd.read_parquet(path) for path in sorted((root / bank.OUT).glob(f"factorial_{role}_*_v19.parquet"))], ignore_index=True)
    factorial = factorial[(factorial.horizon == 1) & factorial.q_reliable & (factorial.action_status == "MATCHED_REALIZED_ACTION")]
    joined = factorial.merge(raw[["base_trial_id", "q_name", "raw_pq_minus_p0_sketch"]],
                             on=["base_trial_id", "q_name"], validate="many_to_one")
    return raw, joined


def _action(frame: pd.DataFrame, design: dict) -> np.ndarray:
    vectors = []
    for row in frame.itertuples():
        vector = np.asarray(design["action_descriptors"][str(int(row.coordinate_index))], dtype=np.float32)
        vectors.append(vector * float(row.action_sign * row.action_alpha))
    return np.stack(vectors)


def _matrix(j: np.ndarray, c: np.ndarray, a: np.ndarray, raw: np.ndarray | None = None) -> np.ndarray:
    pieces = [np.ones((len(j), 1), dtype=np.float32), j.astype(np.float32), c.astype(np.float32),
              a.astype(np.float32), (j[:, :, None] * a[:, None, :]).reshape(len(j), -1),
              (c[:, :, None] * a[:, None, :]).reshape(len(j), -1)]
    if raw is not None:
        pieces.extend((raw.astype(np.float32), (raw[:, :, None] * a[:, None, :]).reshape(len(j), -1)))
    return np.concatenate(pieces, axis=1)


def _ridge(x: np.ndarray, y: np.ndarray, xval: np.ndarray, lam: float = 1.0) -> np.ndarray:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    xtr = torch.from_numpy(x).to(device)
    ytr = torch.from_numpy(y).to(device)
    gram = (xtr.T @ xtr) / len(x)
    gram.diagonal().add_(lam)
    rhs = (xtr.T @ ytr) / len(x)
    weight = torch.linalg.solve(gram, rhs)
    return (torch.from_numpy(xval).to(device) @ weight).cpu().numpy().astype(np.float32)


def _metric(y: np.ndarray, pred: np.ndarray) -> dict:
    den = float(np.linalg.norm(y))
    return {"stack_relative_l2": float(np.linalg.norm(y - pred) / max(den, 1e-12)),
            "j_relative_l2": float(np.linalg.norm(y[:, :128] - pred[:, :128]) / max(np.linalg.norm(y[:, :128]), 1e-12)),
            "rows": len(y)}


def search(root: Path) -> dict:
    design = verify_stage(root, FREEZE)
    raw_train, train = _load(root, "train")
    raw_val, val = _load(root, "validation")
    with np.load(root / "results/v18/processed/clean_state_scores_v18.npz", allow_pickle=False) as payload:
        idx = {str(x): i for i, x in enumerate(payload["base_trial_id"].astype(str))}
        jscore = payload["j"][:, :128].astype(np.float32)
    qtrain = raw_train.set_index(["base_trial_id", "q_name"])
    qval = raw_val.set_index(["base_trial_id", "q_name"])
    def features(frame, raw):
        d = np.stack(frame.raw_pq_minus_p0_sketch).astype(np.float32)
        j = jscore[np.asarray([idx[x] for x in frame.base_trial_id])]
        a = _action(frame, design)
        y = np.stack(frame.y11_stack).astype(np.float32) - np.stack(frame.y10_stack).astype(np.float32) - np.stack(frame.y01_stack).astype(np.float32) + np.stack(frame.y00_stack).astype(np.float32)
        return d, j, a, y
    dtr, jtr, atr, ytr = features(train, qtrain)
    dva, jva, ava, yva = features(val, qval)
    mean = dtr.mean(axis=0); std = np.maximum(dtr.std(axis=0), 1e-5)
    dtr = (dtr - mean) / std; dva = (dva - mean) / std
    # One state×q training target concatenates all eight signed development-action M vectors.
    coordinates = [(int(c), int(s)) for c in design["development_actions"] for s in (-1, 1)]
    blocks = []
    drows = []
    for key, group in train[train.action_role == "development"].groupby(["base_trial_id", "q_name"], sort=True):
        lookup = {(int(row.coordinate_index), int(row.action_sign)): row for row in group.itertuples()}
        if not all(coord in lookup for coord in coordinates):
            continue
        mrow = []
        for coord in coordinates:
            row = lookup[coord]
            mrow.append(np.asarray(row.y11_stack) - np.asarray(row.y10_stack) - np.asarray(row.y01_stack) + np.asarray(row.y00_stack))
        blocks.append(np.concatenate(mrow))
        drows.append(np.asarray(qtrain.loc[key].raw_pq_minus_p0_sketch, dtype=np.float32))
    if len(drows) < 1000:
        raise RuntimeError("V19 compact PLS insufficient complete train state×q action panels")
    dx = (np.stack(drows) - mean) / std
    yy = np.stack(blocks).astype(np.float32)
    cross = dx.T @ (yy - yy.mean(0)) / len(dx)
    u, s, _ = np.linalg.svd(cross, full_matrices=False)
    if len(s) < max(DIMS):
        raise RuntimeError("V19 compact residual PLS matrix narrower than predeclared 128")
    effective_rank = int(np.sum(s > max(float(s[0]) * 1e-8, 1e-10)))
    results = []
    for k in DIMS:
        ctrain = dtr @ u[:, :k]
        cval = dva @ u[:, :k]
        cmean = ctrain.mean(0); cstd = np.maximum(ctrain.std(0), 1e-5)
        ctrain = (ctrain - cmean) / cstd
        cval = (cval - cmean) / cstd
        tr_mask = train.action_role.to_numpy() == "development"
        xtrain = _matrix(jtr[tr_mask], ctrain[tr_mask], atr[tr_mask])
        xval = _matrix(jva, cval, ava)
        pred = _ridge(xtrain, ytr[tr_mask], xval)
        dev = val.action_role.to_numpy() == "development"
        held = ~dev
        result = {"k": k, "supervised_rank_supported": bool(k <= effective_rank),
                  "train_rows": int(tr_mask.sum()), "validation_dev": _metric(yva[dev], pred[dev]),
                  "validation_heldout_actions": _metric(yva[held], pred[held]) if held.any() else None,
                  "validation_dev_states": int(val.loc[dev, "base_trial_id"].nunique()),
                  "validation_heldout_states": int(val.loc[held, "base_trial_id"].nunique())}
        results.append(result)
        print(f"V19 compact k={k} dev={result['validation_dev']['stack_relative_l2']:.4f} heldout={result['validation_heldout_actions']['stack_relative_l2']:.4f}", flush=True)
    supported = [row for row in results if row["supervised_rank_supported"]]
    if not supported:
        raise RuntimeError("V19 compact residual PLS has no supported candidate dimension")
    chosen = min(supported, key=lambda row: row["validation_dev"]["stack_relative_l2"])
    k = int(chosen["k"])
    ctrain = dtr @ u[:, :k]; cval = dva @ u[:, :k]
    cmean = ctrain.mean(0); cstd = np.maximum(ctrain.std(0), 1e-5)
    ctrain = (ctrain - cmean) / cstd; cval = (cval - cmean) / cstd
    tr_mask = train.action_role.to_numpy() == "development"
    xbase = _matrix(jtr[tr_mask], ctrain[tr_mask], atr[tr_mask])
    xval_base = _matrix(jva, cval, ava)
    xraw = _matrix(jtr[tr_mask], ctrain[tr_mask], atr[tr_mask], dtr[tr_mask])
    xval_raw = _matrix(jva, cval, ava, dva)
    pbase = _ridge(xbase, ytr[tr_mask], xval_base)
    praw = _ridge(xraw, ytr[tr_mask], xval_raw)
    residual = {}
    for label, mask in (("development", val.action_role.to_numpy() == "development"),
                        ("heldout_action", val.action_role.to_numpy() == "heldout")):
        a = _metric(yva[mask], pbase[mask]); b = _metric(yva[mask], praw[mask])
        residual[label] = {"compact": a, "plus_raw_sketch": b,
                           "raw_incremental_stack_gain": a["stack_relative_l2"] - b["stack_relative_l2"],
                           "raw_incremental_j_gain": a["j_relative_l2"] - b["j_relative_l2"]}
    family = {}
    for name in sorted(val.family.unique()):
        mask = val.family.to_numpy() == name
        a = _metric(yva[mask], pbase[mask]); b = _metric(yva[mask], praw[mask])
        family[name] = {"compact": a, "plus_raw_sketch": b,
                        "raw_incremental_stack_gain": a["stack_relative_l2"] - b["stack_relative_l2"]}
    gate = design["candidate_gate"]
    eligible = (chosen["validation_dev"]["stack_relative_l2"] <= gate["maximum_validation_dev_M_relative_l2"]
                and chosen["validation_heldout_actions"]["stack_relative_l2"] <= gate["maximum_validation_heldout_M_relative_l2"]
                and residual["development"]["raw_incremental_stack_gain"] <= gate["maximum_raw_sketch_incremental_gain"]
                and residual["heldout_action"]["raw_incremental_stack_gain"] <= gate["maximum_raw_sketch_incremental_gain"]
                and all(x["raw_incremental_stack_gain"] <= gate["maximum_raw_sketch_incremental_gain"] for x in family.values()))
    result = {"compact_freeze_digest": design["freeze_digest"], "train_complete_state_q_panels": len(drows),
              "effective_supervised_rank": effective_rank,
              "sketch_dimension": 384, "search": results, "selected_k": k,
              "raw_residual": residual, "by_family": family,
              "restricted_proxy_candidate_gate_passed": bool(eligible),
              "compact_response_context_candidate": bool(eligible),
              "independent_final_eligible": False,
              "independent_final_reason": "384D raw CountSketch is not full persistent P_raw and C encodes Pq-P0 contrast rather than an autonomous state; the frozen search alone cannot certify response or dynamical sufficiency",
              "h3_candidate_supported": False, "autonomous_state_model_authorized": False}
    np.savez_compressed(root / bank.OUT / "compact_response_pls_v19.npz", projection=u[:, :k].astype(np.float32),
                        raw_train_mean=mean.astype(np.float32), raw_train_std=std.astype(np.float32),
                        compact_train_mean=cmean.astype(np.float32), compact_train_std=cstd.astype(np.float32),
                        singular_values=s.astype(np.float32))
    write_json_atomic(root / bank.OUT / "compact_response_search_v19.json", result)
    return {"selected_k": k, "proxy_candidate_gate": bool(eligible),
            "dev_stack": chosen["validation_dev"]["stack_relative_l2"],
            "heldout_stack": chosen["validation_heldout_actions"]["stack_relative_l2"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("prepare", "extract_train", "extract_validation", "aggregate_train", "aggregate_validation", "search"))
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    root = Path.cwd()
    if args.stage == "prepare":
        result = prepare(root)
    elif args.stage.startswith("extract_"):
        result = extract(root, args.stage.split("_", 1)[1], args.limit)
    elif args.stage.startswith("aggregate_"):
        result = aggregate(root, args.stage.split("_", 1)[1])
    else:
        result = search(root)
    print(json.dumps(result, indent=2))
