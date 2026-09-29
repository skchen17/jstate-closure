"""V40 a3 sequential continuation API guard."""
import inspect

from jclosure.experiments.calibration_v40_a3 import sequential_logits


def test_a3_uses_native_one_token_step():
    source = inspect.getsource(sequential_logits)
    assert "for token in tokens" in source
    assert "step(model, work, token, length)" in source
    assert "continuation_logits" not in source
