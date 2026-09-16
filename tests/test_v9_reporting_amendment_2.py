from __future__ import annotations

import pandas as pd

from jclosure.reporting_v9_amendment_2 import corrected_status_table


def test_v9_reporting_amendment_labels_endpoints() -> None:
    table = corrected_status_table(
        pd.DataFrame(
            [
                {
                    "endpoint": "next",
                    "method": "causal_bottleneck",
                    "dimension": 512,
                    "status": "COMPLETED_HELD_OUT",
                }
            ]
        )
    )
    assert "endpoint" in table
    assert "next" in table
