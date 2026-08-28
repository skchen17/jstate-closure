# Closure-eligible layer calibration

## Material Passport

- Run ID: `layer-calibration-20260828T121537Z-d3f2fa00-s20260828`
- Verification Status: VERIFIED / FAILED GATE
- Exact command: `/data/CSK/J-space-project/jstate-closure/src/jclosure/experiments/calibrate_layers.py --config configs/phase0_v2_confirmatory.yaml`
- Input Phase 0 gate: `phase0-v2-20260828T120721Z-d3f2fa00-s20260828`
- Candidate layers: `[23, 24, 25, 26, 27, 28, 29]`
- Closure-eligible layers: `[]`

## Results

| layer | multihop hit@10 | order hit@10 | rank-CI lower | positive-CI lower | valid clamps | eligible |
|---:|---:|---:|---:|---:|---:|:---:|
| 23 | 0.327586 | 0.279297 | 0.094551 | 1.892754 | 0/200 | False |
| 24 | 0.517241 | 0.265625 | 0.100553 | 1.978771 | 0/200 | False |
| 25 | 0.603448 | 0.291016 | 0.097422 | 1.971748 | 0/200 | False |
| 26 | 0.448276 | 0.263672 | 0.071719 | 1.841033 | 0/200 | False |
| 27 | 0.517241 | 0.312500 | 0.087801 | 1.575911 | 0/200 | False |
| 28 | 0.551724 | 0.394531 | 0.099805 | 1.570776 | 0/200 | False |
| 29 | 0.517241 | 0.457031 | 0.088832 | 1.156316 | 0/200 | False |

All seven candidates passed the readout, family point-estimate, pooled rank-CI,
positive-control, numerical-null, deterministic-rerun, zero-strength, identity-patch,
and hook-cleanup criteria. Each failed the independently frozen clamp-valid-rate
criterion (required at least 80%). Numerical logit errors were exactly
`{"identity_max_logit_error": 0.0, "passed": true, "zero_max_logit_error": 0.0}`.

Across all 1400 balanced attempts, strict valid count was 0. Median
dense measured-J cosine was 0.964775 (threshold 0.995), median
activation RMS drift was 0.145782 (limit 0.02), and median remainder
fraction was 0.241852 (minimum 0.20). Exclusion-reason counts
were `{"dense_cosine": 1400, "rms_drift": 1400, "top10_overlap": 75}`; a trial may
contribute to more than one reason.

## Gate decision

**FAILED: no closure-eligible layer.** No thresholds or intervention source were
changed after observing this result. These records calibrate the state-construction
method; they are not Phase 3 causal effects and cannot adjudicate H1, H2, or H3.
