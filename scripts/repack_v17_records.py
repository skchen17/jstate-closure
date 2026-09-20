"""Losslessly repack two generated V17 prediction tables below GitHub's file limit."""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd


def main() -> None:
    root = Path.cwd() / "results/v17/processed"
    for name in ("state_context_ceiling_predictions_v17.parquet",
                 "conditional_raw_residual_predictions_v17.parquet"):
        original = root / name
        compressed = original.with_suffix(".zstd.parquet")
        old = pd.read_parquet(original)
        if not compressed.exists():
            old.to_parquet(compressed, index=False, compression="zstd", compression_level=19)
        new = pd.read_parquet(compressed)
        if not old.equals(new):
            raise RuntimeError(f"V17 lossless repack content mismatch: {name}")
        if compressed.stat().st_size >= 95_000_000:
            raise RuntimeError(f"V17 compressed file still too large: {name}")
        before, after = original.stat().st_size, compressed.stat().st_size
        os.replace(compressed, original)
        print(f"{name}: {len(new)} rows; {before} -> {after} bytes; equality verified")


if __name__ == "__main__":
    main()
