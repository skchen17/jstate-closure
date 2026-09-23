# V36 pre-experiment audit of V35 adjudication

Status: `V35_AUDIT_ADJUDICATION_AMENDMENT_REQUIRED` (with a reporting correction). This is an append-only interpretation amendment, not a replacement for the sealed V35 decision or an opening of its independent final.

## Verified evidence

The frozen V35 configuration predeclared `Q2_Q3_Q4` in `subset_conditions` and `late_cumulative`, and `development_plan_v35.py` included `LOCALIZED_Q2_Q3_Q4` in `FINAL_PRIORITY`. The sealed depth records show that the triple passed the same `quartile_gate` in both models and both prospective roles:

| Role | Model | REM | REST | cosine | passing families |
|---|---|---:|---:|---:|---:|
| Development | Qwen | 0.990 | 0.824 | 0.914 | 5/5 |
| Development | Falcon | 0.920 | 0.918 | 0.945 | 5/5 |
| Validation | Qwen | 1.013 | 0.825 | 0.901 | 5/5 |
| Validation | Falcon | 0.883 | 0.916 | 0.942 | 5/5 |

The discrepancy is not a family-level failure or a separate frozen threshold. `depth_analysis_v35.py:144` builds `strong_pair_passes` **only** from `pair_conditions`; `Q2_Q3_Q4` is a triple, so it never enters the shared candidate aggregation at line 239. `development_plan_v35.py:candidates()` then creates `LOCALIZED_*` entries only from `strong_quartiles + strong_pairs`, even though its priority list explicitly contains `LOCALIZED_Q2_Q3_Q4`. The development plan consequently lists only `COMPLEMENTARY_Q3_Q4`, `LOCALIZED_Q3_Q4`, and `FULL_DEPTH_ONLY`. The validation candidate calculation repeats the same omission, so `final_opening_v35.py` selected `FULL_DEPTH_ONLY`. `adjudicate_v35.py:patterns()` repeats it again, producing `no_shared_smaller_organization_passed=true`.

This is an **implementation/adjudication omission**, propagated into report wording, not an intentionally excluded mechanism class. A read-only replay of the existing gate, with the already-predeclared triple admitted, puts `LOCALIZED_Q2_Q3_Q4` in the development and validation intersection for both models. It would have preceded `FULL_DEPTH_ONLY` in the frozen `FINAL_PRIORITY`. The existing independent-final response, however, tested `FULL_DEPTH_ONLY` only; it cannot confirm the triple. No V35 final data are reopened here.

## Amendment and scientific boundary

The historical sentence “no shared strong subset” is false for the measured Q2+Q3+Q4 strong-gate subset. The defensible correction is: **a shared three-quartile subset passed the development and validation strong gates, but it was omitted from the finalist construction and never tested as the V35 independent-final mechanism.** Accordingly, V35 does not establish that no smaller read-side organization qualified for validation, nor does it establish independent-final confirmation of a hierarchy. V35-F/H cannot be retroactively promoted to confirmed. V35-I remains unconfirmed. Other negative V35 mechanism outcomes are not automatically overturned by this correction.

V36 will use neither the erroneous “no shared subset” claim nor an unearned final confirmation as a premise. Its local operator experiments require fresh panels and separate prospective gates.

## Source integrity

All checks used repository revision `9c272d501f6c0beb6ee0bf6c643dbdc77aef9475`. SHA-256 of the key sealed inputs:

| Input | SHA-256 |
|---|---|
| `configs/hierarchical_read_v35.yaml` | `65ca8e5254be7b1a2949a46f333c994869e7664d23db1609e27267725e7d4cf0` |
| `results/v35/processed/depth_analysis_development_v35.json` | `bbdf6400d3ac86873995ce9d0c7beb7e2a7b598299dcae5f1ae6304c433f6ae4` |
| `results/v35/processed/depth_analysis_validation_v35.json` | `0a6b7dd7b464d48c069a6da907b38eff28867784a0780a44a50db739243cda3d` |
| `results/v35/processed/development_plan_v35.json` | `62e5373715440e48fc7dc73edc775af47580212088053026c82293cd8e8ac7d8` |
| `results/v35/processed/final_opening_v35.json` | `48a88bf9a8e229f7dac1de1b03dd27cd46a31017b710e0401b898869028b8154` |
| `results/v35/processed/v35_adjudication.json` | `1fa4ba3f168dad0c348ba818314e0b8b73fcf5d7620b25f672d47a928ff7880a` |

No V1–V35 file was modified for this audit.
