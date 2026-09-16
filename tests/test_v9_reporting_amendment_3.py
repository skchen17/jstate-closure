from __future__ import annotations

import pandas as pd

from jclosure.reporting_v9_amendment_1 import corrected_curve_rows


def test_v9_figure_filter_can_isolate_comparable_conditional_rows() -> None:
    frame = pd.DataFrame(
        [
            {
                "family": "pooled",
                "endpoint": "next_j",
                "method": "causal_bottleneck",
                "dimension": 256,
                "status": "V8_FROZEN_REANALYSIS",
            },
            {
                "family": "pooled",
                "endpoint": "next",
                "method": "causal_bottleneck",
                "dimension": 512,
                "status": "COMPLETED_HELD_OUT",
            },
        ]
    )
    curves = corrected_curve_rows(frame, rank=599, raw_bytes=100_000)
    comparable = curves[curves["status"] == "COMPLETED_HELD_OUT"]
    assert comparable["dimension"].tolist() == [512]
