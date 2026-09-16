from __future__ import annotations

import numpy as np
import pandas as pd

from jclosure.experiments.analyze_v11_amendment import _error_norm


def test_error_norm_recovers_euclidean_law_of_cosines() -> None:
    frame = pd.DataFrame(
        {
            "teacher_j_effect_norm": [3.0],
            "decoded_j_effect_norm": [4.0],
            "direction_cosine": [0.0],
        }
    )
    assert np.allclose(_error_norm(frame), [5.0])
