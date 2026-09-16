from __future__ import annotations

import pandas as pd

from jclosure.reporting_v9_amendment_1 import corrected_curve_rows


def test_v9_reporting_amendment_includes_old_and_new_next_labels() -> None:
    frame = pd.DataFrame(
        [
            {
                "family": "pooled",
                "endpoint": "next_j",
                "method": "causal_bottleneck",
                "dimension": 256,
            },
            {
                "family": "pooled",
                "endpoint": "next",
                "method": "causal_bottleneck",
                "dimension": 512,
            },
            {
                "family": "pooled",
                "endpoint": "output",
                "method": "causal_bottleneck",
                "dimension": 512,
            },
        ]
    )
    result = corrected_curve_rows(frame, rank=599, raw_bytes=100_000)
    assert result["dimension"].tolist() == [256, 512]
