"""CPU-only V40 calibration endpoint and condition safeguards."""
import torch

from jclosure.experiments.calibration_v40_a2 import (
    CONDITION_CHANNELS, answer_token_ids, score,
)


class TinyTokenizer:
    def encode(self, text, add_special_tokens=False):
        assert not add_special_tokens
        return {"0": [2], "1": [3], "A": [4], "B": [5]}[text]


def test_condition_map_matches_frozen_six_way_design():
    assert CONDITION_CHANNELS == {
        "BASE": (), "REC_only": ("REC",), "Conv_only": ("Conv",),
        "KV_only": ("KV",), "REC_Conv": ("REC", "Conv"),
        "REC_Conv_KV": ("REC", "Conv", "KV"),
    }


def test_external_answer_scoring():
    alphabet = answer_token_ids(TinyTokenizer(), "01")
    logits = torch.tensor([0.0, 0.0, 1.0, 3.0])
    result = score(logits, alphabet, "0", "1")
    assert result["predicted_answer"] == "1"
    assert result["donor_recovered"]
    assert not result["recipient_correct"]
    assert result["donor_minus_recipient_logit"] == 2.0
