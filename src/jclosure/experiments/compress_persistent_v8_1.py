"""Corrective v8.1 wrapper for the shared-family compression alignment bug.

The frozen v8 implementation assumed that every application matrix began with
the rows used to fit its encoder.  That is true for the universal and
family-specific paths, but false when a globally fitted encoder is applied to
a family-only slice.  This additive wrapper preserves every v8 input and gate,
while fitting the ordering on the global training rows and returning scores
for only the requested application rows.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import torch

from jclosure.experiments import compress_persistent_v8 as v8
from jclosure.experiments.common import initialize_context, standard_parser
from jclosure.protocol_v8 import verify_freeze
from jclosure.provenance import sha256_file, write_json_atomic

FREEZE_PATH = Path("artifacts/compression_protocol_v8_1.freeze.json")
FAILED_RUN_MANIFEST = Path(
    "results/v8/raw/"
    "compress-persistent-v8-20260915T100215Z-bcb732e4-s20260828-"
    "compression-full/manifest.json"
)
WRAPPER_PATH = Path("src/jclosure/experiments/compress_persistent_v8_1.py")
SCRIPT_PATH = Path("scripts/run_compression_v8_1.sh")
TEST_PATH = Path("tests/test_v8_1.py")


def _digest(value: dict[str, Any]) -> str:
    payload = dict(value)
    payload.pop("freeze_digest", None)
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def corrected_latent(
    method: str,
    dimension: int,
    train_features: np.ndarray,
    apply_features: np.ndarray,
    train_absolute: np.ndarray,
    train_delta: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Fit on ``train_features`` and score arbitrary ``apply_features`` rows."""

    if len(train_features) == len(apply_features):
        return v8._latent_original_v81(  # type: ignore[attr-defined]
            method,
            dimension,
            train_features,
            apply_features,
            train_absolute,
            train_delta,
        )
    combined = np.concatenate((train_features, apply_features), axis=0)
    scores, singular = v8._svd_scores(train_features, combined)
    train_scores = scores[: len(train_features)]
    apply_scores = scores[len(train_features) :]
    rank = scores.shape[1]
    dimension = min(int(dimension), rank)
    if method == "pca":
        order = np.arange(rank)
    elif method == "predictive_bottleneck":
        order = v8._order(train_scores, train_absolute)
    elif method == "causal_bottleneck":
        order = v8._order(train_scores, train_delta)
    elif method == "layerwise_fusion":
        quarters = [np.arange(start, rank, 4) for start in range(4)]
        order = np.asarray(
            [value for group in zip(*quarters, strict=False) for value in group]
        )
        order = order[order < rank]
    elif method == "nonlinear_encoder":
        torch.manual_seed(20260828 + dimension)
        fit = torch.from_numpy(train_scores).float()
        apply_tensor = torch.from_numpy(apply_scores).float()
        width = max(64, 2 * dimension)
        encoder = torch.nn.Sequential(
            torch.nn.Linear(rank, width),
            torch.nn.GELU(),
            torch.nn.Linear(width, dimension),
        )
        decoder = torch.nn.Sequential(
            torch.nn.Linear(dimension, width),
            torch.nn.GELU(),
            torch.nn.Linear(width, rank),
        )
        optimizer = torch.optim.AdamW(
            [*encoder.parameters(), *decoder.parameters()],
            lr=3e-3,
            weight_decay=1e-4,
        )
        for _ in range(200):
            optimizer.zero_grad(set_to_none=True)
            prediction = decoder(encoder(fit))
            loss = torch.mean((prediction - fit) ** 2)
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            return encoder(apply_tensor).numpy(), {
                "rank": rank,
                "train_reconstruction_loss": float(loss),
                "singular_value_max": float(singular[0]),
                "singular_value_min": float(singular[-1]),
                "corrective_alignment": True,
            }
    else:
        raise ValueError(method)
    return apply_scores[:, order[:dimension]], {
        "rank": rank,
        "singular_value_max": float(singular[0]),
        "singular_value_min": float(singular[-1]),
        "corrective_alignment": True,
    }


def _inputs(root: Path) -> list[Path]:
    return [
        root / WRAPPER_PATH,
        root / SCRIPT_PATH,
        root / TEST_PATH,
        root / "artifacts/persistent_state_v8.freeze.json",
        root / "results/v8/processed/structured_features_v8.json",
        root / FAILED_RUN_MANIFEST,
    ]


def build_freeze(root: Path, config: dict[str, Any]) -> dict[str, Any]:
    base = verify_freeze(root, config)
    inputs = _inputs(root)
    missing = [str(path.relative_to(root)) for path in inputs if not path.is_file()]
    if missing:
        raise RuntimeError(f"v8.1 corrective freeze inputs missing: {missing}")
    features = json.loads(
        (root / "results/v8/processed/structured_features_v8.json").read_text()
    )
    value: dict[str, Any] = {
        "schema_version": 11,
        "protocol_version": "structured_persistent_state_compression_v8_1",
        "reason": "shared_family_global_fit_application_alignment",
        "code_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip(),
        "base_freeze_digest": base["freeze_digest"],
        "feature_artifact": features["feature_artifact"],
        "feature_artifact_sha256": features["feature_artifact_sha256"],
        "failed_run_manifest": str(FAILED_RUN_MANIFEST),
        "hashes": {
            str(path.relative_to(root)): sha256_file(path) for path in inputs
        },
    }
    value["freeze_digest"] = _digest(value)
    write_json_atomic(root / FREEZE_PATH, value)
    return value


def verify_corrective_freeze(root: Path, config: dict[str, Any]) -> dict[str, Any]:
    base = verify_freeze(root, config)
    value = json.loads((root / FREEZE_PATH).read_text())
    if value.get("freeze_digest") != _digest(value):
        raise RuntimeError("v8.1 corrective freeze digest mismatch")
    if value.get("base_freeze_digest") != base["freeze_digest"]:
        raise RuntimeError("v8.1 base freeze mismatch")
    changed = [
        name
        for name, expected in value["hashes"].items()
        if not (root / name).is_file() or sha256_file(root / name) != expected
    ]
    if changed:
        raise RuntimeError(f"v8.1 corrective frozen input mismatch: {changed}")
    if sha256_file(root / value["feature_artifact"]) != value[
        "feature_artifact_sha256"
    ]:
        raise RuntimeError("v8.1 feature artifact mismatch")
    return value


def main() -> None:
    parser = standard_parser(
        "corrective structured persistent-state compression v8.1",
        "configs/persistent_state_v8.yaml",
    )
    parser.add_argument("--stage", required=True, choices=("freeze", "analyze"))
    args = parser.parse_args()
    context = initialize_context("compress-persistent-v8-1", args)
    try:
        if args.stage == "freeze":
            context.finish("COMPLETED_CORRECTIVE_FREEZE", freeze=build_freeze(context.root, context.config))
            return
        corrective = verify_corrective_freeze(context.root, context.config)
        base = verify_freeze(context.root, context.config)
        if args.dry_run:
            context.finish("DRY_RUN", corrective_freeze_digest=corrective["freeze_digest"])
            return
        if not hasattr(v8, "_latent_original_v81"):
            v8._latent_original_v81 = v8._latent  # type: ignore[attr-defined]
        v8._latent = corrected_latent
        v8._analysis(context, base)
        summary_path = context.root / v8.SUMMARY
        summary = json.loads(summary_path.read_text())
        summary["corrective_protocol_version"] = corrective["protocol_version"]
        summary["corrective_freeze_digest"] = corrective["freeze_digest"]
        summary["corrective_reason"] = corrective["reason"]
        write_json_atomic(summary_path, summary)
        context.finish("COMPLETED_COMPRESSION_SCREEN_V8_1", summary=summary)
    except Exception as exc:
        context.finish("FAILED", error=f"{type(exc).__name__}: {exc}")
        raise


if __name__ == "__main__":
    main()
