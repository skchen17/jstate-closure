# Token Library — V31

Response-blind 25-state calibration excluded all V28/V29/V30 formal IDs. Common candidate tokens at rank ≤8192: 949. The frozen anchor is ID 25 (`:`); 192 anchor→candidate pairs are split 160/16/16 TRAIN/VALIDATION/FINAL. Frozen surface-composition triples are 14/16/16; validation/final AB token IDs never enter fitting, while each A and B constituent does. All 255 states, six prewrite future probes and eligibility hashes were frozen before current-token writes.

| token role | frozen category | count |
|---|---|---|
| TOKEN_FINAL | CJK_SURFACE_UNRESOLVED | 3 |
| TOKEN_FINAL | FUNCTION_WORD | 2 |
| TOKEN_FINAL | LEXICAL_SURFACE | 2 |
| TOKEN_FINAL | PUNCTUATION_OR_STRUCTURE | 1 |
| TOKEN_FINAL | SHORT_ALPHA_AMBIGUOUS | 1 |
| TOKEN_FINAL | UNCLASSIFIED | 7 |
| TOKEN_TRAIN | CJK_SURFACE_UNRESOLVED | 13 |
| TOKEN_TRAIN | FUNCTION_WORD | 22 |
| TOKEN_TRAIN | LEXICAL_SURFACE | 37 |
| TOKEN_TRAIN | NUMERIC | 10 |
| TOKEN_TRAIN | PUNCTUATION_OR_STRUCTURE | 57 |
| TOKEN_TRAIN | SHORT_ALPHA_AMBIGUOUS | 14 |
| TOKEN_TRAIN | UNCLASSIFIED | 7 |
| TOKEN_VALIDATION | CJK_SURFACE_UNRESOLVED | 1 |
| TOKEN_VALIDATION | LEXICAL_SURFACE | 4 |
| TOKEN_VALIDATION | PUNCTUATION_OR_STRUCTURE | 2 |
| TOKEN_VALIDATION | UNCLASSIFIED | 9 |

The AB relation is *single-token decoded surface concatenation*, not a proven semantic/function composition. Many examples are punctuation or word fragments (e.g. `-`+`based`); ambiguous categories remain explicitly unresolved. This limits generalization beyond the tested surface proxy. No validation-composition or TOKEN_FINAL write was opened. Machine source: `results/v31/processed/design_v31.json`, hash `2edd84b679a44578603125971abe78f5c85643be013ee2a174085f1982ba3ac1`.
