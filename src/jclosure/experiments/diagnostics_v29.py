"""Held-out diagnostic models for current workspace, token, write geometry, and future response."""
from __future__ import annotations

import hashlib
import json
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from jclosure.experiments.forks_formal_v29 import OUT, design
from jclosure.experiments.pre_readout_v27 import metadata, prefix, step
from jclosure.experiments.transaction_v28 import context
from jclosure.protocol_v29 import verify_stage, stage_freeze
from jclosure.provenance import sha256_file, write_json_atomic

SOURCE = "src/jclosure/experiments/diagnostics_v29.py"


def token_features(item, vocab):
    out = np.zeros(len(vocab) + 14, np.float64)
    for sign, tid in ((-1, item["token_A"]), (1, item["token_B"])):
        if tid in vocab:
            out[vocab[tid]] += sign
    for sign, text in ((-1, item["token_A_text"]), (1, item["token_B_text"])):
        s = text.strip()
        off = len(vocab)
        out[off + 0] += sign * len(s)
        out[off + 1] += sign * int(s.isdigit())
        out[off + 2] += sign * int(s.isalpha())
        out[off + 3] += sign * int(s.isspace())
        out[off + 4] += sign * int(any(ch.isdigit() for ch in s))
        out[off + 5] += sign * int(any(ch.isalpha() for ch in s))
        out[off + 6] += sign * int(any(unicodedata.category(ch).startswith("P") for ch in s))
        out[off + 7] += sign * int(s.isascii())
        out[off + 8] += sign * int(s.startswith(" "))
        out[off + 9] += sign * int(s.endswith(" "))
        out[off + 10] += sign * int(len(s) == 1)
        out[off + 11] += sign * int(len(s) > 3)
        out[off + 12] += sign * int("\n" in text)
        out[off + 13] += sign * int("=" in text)
    return out


def ridge_fit_predict(train, test, ytrain, alpha=1.0):
    mean = train.mean(0)
    sd = train.std(0)
    sd[sd < 1e-8] = 1.0
    x = np.clip((train - mean) / sd, -10, 10)
    z = np.clip((test - mean) / sd, -10, 10)
    ym = ytrain.mean(0)
    gram = x @ x.T
    coef = x.T @ np.linalg.solve(gram + alpha * np.eye(len(x)), ytrain - ym)
    return z @ coef + ym


def score(y, pred, train_mean):
    mse = float(np.mean((y - pred) ** 2))
    baseline = float(np.mean((y - train_mean) ** 2))
    return {"mse": mse, "r2_vs_train_mean": 1.0 - mse / max(baseline, 1e-12), "relative_l2": float(np.linalg.norm(y - pred) / max(np.linalg.norm(y), 1e-12))}


def run(root: Path):
    for stage in ("development_analysis", "validation_analysis"):
        verify_stage(root, stage)
    d = design(root)
    bundle, dense, _, _, _, _ = context(root)
    vocab = sorted({x[t] for x in d["development"] for t in ("token_A", "token_B")})
    vocab = {tid: i for i, tid in enumerate(vocab)}
    arrays = {}
    raw_rows = []
    for role in ("development", "validation"):
        t = pd.read_parquet(root / OUT / f"transfer_{role}_v29.parquet")
        g = pd.read_parquet(root / OUT / f"geometry_{role}_v29.parquet")
        targets = np.load(root / OUT / f"transfer_vectors_{role}_v29.npz")["donor_target"]
        x_tok, x_j, x_ws, x_w, ys, fams, ids = [], [], [], [], [], [], []
        for n, item in enumerate(d[role], 1):
            m = metadata(root, item)
            incoming, _, length = prefix(bundle, str(m["prompt"]))
            a = step(bundle, dense, incoming, torch.tensor([[item["token_A"]]]), length, d, 30)
            b = step(bundle, dense, incoming, torch.tensor([[item["token_B"]]]), length, d, 30)
            row = t[(t.base_trial_id == item["base_trial_id"]) & (t.condition == "REC+Conv+KV") & (t.direction == "A_from_B")]
            if len(row) != 1:
                raise RuntimeError("future target row absent")
            future = targets[int(row.vector_index.iloc[0])].astype(np.float64)
            geo = g[g.base_trial_id == item["base_trial_id"]].sort_values(["channel", "layer", "field"])
            if len(geo) != 64:
                raise RuntimeError("write geometry field coverage incomplete")
            x_tok.append(token_features(item, vocab))
            x_j.append((b["targets"]["j"] - a["targets"]["j"]).astype(np.float64))
            x_ws.append((b["targets"]["workspace"] - a["targets"]["workspace"]).astype(np.float64))
            x_w.append(np.concatenate([geo.contrast_norm.to_numpy(np.float64), geo.A_B_cosine.fillna(0).to_numpy(np.float64)]))
            ys.append(future)
            fams.append(item["family"])
            ids.append(item["base_trial_id"])
            raw_rows.append({"role": role, "base_trial_id": item["base_trial_id"], "family": item["family"], "current_J_hash": hashlib.sha256(x_j[-1].astype(np.float32).tobytes()).hexdigest(), "current_workspace_hash": hashlib.sha256(x_ws[-1].astype(np.float32).tobytes()).hexdigest(), "write_geometry_hash": hashlib.sha256(x_w[-1].astype(np.float32).tobytes()).hexdigest(), "future_response_hash": hashlib.sha256(future.astype(np.float32).tobytes()).hexdigest()})
            if n % 10 == 0 or n == len(d[role]):
                print(f"V29 diagnostics {role} {n}/{len(d[role])}", flush=True)
        arrays[role] = {"token": np.stack(x_tok), "J": np.stack(x_j), "workspace": np.stack(x_ws), "write": np.stack(x_w), "future": np.stack(ys), "family": np.array(fams), "id": np.array(ids)}
    tr, te = arrays["development"], arrays["validation"]
    models = {
        "M0_token_identity_and_surface": ("token",),
        "M1_current_J": ("J",),
        "M2_token_plus_J": ("token", "J"),
        "M3_token_J_broader_current_workspace": ("token", "J", "workspace"),
        "M4_token_J_write_geometry": ("token", "J", "write"),
    }
    results = {}
    for name, columns in models.items():
        train = np.concatenate([tr[c] for c in columns], axis=1)
        test = np.concatenate([te[c] for c in columns], axis=1)
        pred = ridge_fit_predict(train, test, tr["future"])
        results[name] = score(te["future"], pred, tr["future"].mean(0))
        results[name]["train_feature_dim"] = int(train.shape[1])
        results[name]["family_r2"] = {f: score(te["future"][te["family"] == f], pred[te["family"] == f], tr["future"].mean(0))["r2_vs_train_mean"] for f in d["families"]}
    write_from_J = ridge_fit_predict(tr["J"], te["J"], tr["write"])
    write_prediction = score(te["write"], write_from_J, tr["write"].mean(0))
    xy = (tr["write"] - tr["write"].mean(0)).T @ (tr["future"] - tr["future"].mean(0))
    singular = np.linalg.svd(xy, compute_uv=False)
    summary = {"train_states": len(tr["id"]), "heldout_validation_states": len(te["id"]), "token_vocab_train": len(vocab), "token_identity_OOD_limitation": "exact held-out IDs unseen in train map to zero; coarse surface features remain", "fixed_ridge_alpha": 1.0, "models": results, "write_geometry_from_current_J": write_prediction, "train_cross_cov_rank": int(np.linalg.matrix_rank(xy)), "train_cross_cov_top10_singular": singular[:10].tolist(), "causal_interpretation": False, "full_write_vectors_not_used": True, "heldout_token_pair_test": True, "historical_final_opened": False}
    path = root / OUT / "workspace_write_diagnostics_v29.json"
    raw = root / OUT / "workspace_write_features_v29.parquet"
    npz = root / OUT / "workspace_write_vectors_v29.npz"
    write_json_atomic(path, summary)
    pd.DataFrame(raw_rows).to_parquet(raw, index=False, compression="zstd")
    np.savez_compressed(npz, **{f"{role}_{key}": value.astype(np.float32) if value.dtype.kind == "f" else value for role, data in arrays.items() for key, value in data.items()})
    fr = stage_freeze(root, "diagnostics", [SOURCE, str(path.relative_to(root)), str(raw.relative_to(root)), str(npz.relative_to(root)), "artifacts/natural_write_content_v29_validation_analysis.freeze.json"], {"summary_sha256": sha256_file(path), "vectors_sha256": sha256_file(npz), "causal_interpretation": False})
    return {"freeze_digest": fr["freeze_digest"], "models": results, "write_geometry_from_current_J": write_prediction}


if __name__ == "__main__":
    print(json.dumps(run(Path.cwd()), indent=2))
