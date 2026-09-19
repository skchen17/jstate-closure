# JVP versus finite writeback — V14

The exact V13 autograd JVP was compared with central finite differences on five frozen train anchors, five predeclared direction families, four target blocks, and ε from `1e-5` to `2` under separately frozen base and extension protocols. This tests the finite BF16 writeback interface, not a new model weight state.

- Three clean repetitions per anchor were byte/deterministically identical at the target readout (`noise=0`); dividing by that zero floor would falsely imply infinite SNR.
- Empirical 5th-percentile nonzero writeback output floors: `{'j': 0.0009409313033672384, 'logits': 0.125, 'semantic_continuous': 0.12495647843152638, 'workspace': 0.036365347332082194}`.
- Validation-only `MIN_CAUSAL_EFFECT_NORM`: `0.00680280` (5th percentile of V13 validation raw-teacher J-effect norms across h1/h2/h4/h8).
- First ε satisfying the predeclared *all-target* median JVP equivalence rule (cosine ≥0.95 and relative L2 ≤0.20): `None`. **No reliable scale was identified through ε=2.** At that scale the perturbation is finite, not an infinitesimal check.
- All-FP32 cache failed because BF16 attention query requires matching key/value dtype. REC/conv-only FP32 with BF16 KV supported: `True`. FP64 finite-model reference is not feasible without changing frozen BF16 model weights.
- At ε=1, REC/conv-FP32/BF16-KV J median cosine/relative L2 was `0.957` / `0.354` versus ordinary V13 writeback `0.952` / `0.330`; partial FP32 does not materially fix the gate.

| ε | target | median cosine | median relative L2 | median quantization-floor SNR | pass fraction |
|---:|---|---:|---:|---:|---:|
| 1e-05 | j | nan | 1.000 | 0.00 | 0.00 |
| 1e-05 | logits | nan | 1.000 | 0.00 | 0.00 |
| 1e-05 | semantic_continuous | nan | 1.000 | 0.00 | 0.00 |
| 1e-05 | workspace | nan | 1.000 | 0.00 | 0.00 |
| 0.001 | j | 0.068 | 188.369 | 1.21 | 0.00 |
| 0.001 | logits | 0.032 | 343.015 | 1.41 | 0.00 |
| 0.001 | semantic_continuous | -0.001 | 294.408 | 1.41 | 0.00 |
| 0.001 | workspace | -0.047 | 264.754 | 1.67 | 0.00 |
| 0.1 | j | 0.271 | 3.246 | 1.43 | 0.00 |
| 0.1 | logits | 0.115 | 5.448 | 1.66 | 0.00 |
| 0.1 | semantic_continuous | 0.041 | 7.811 | 1.65 | 0.00 |
| 0.1 | workspace | 0.228 | 6.601 | 2.11 | 0.00 |
| 1 | j | 0.952 | 0.330 | 2.78 | 0.20 |
| 1 | logits | 0.835 | 0.588 | 2.14 | 0.12 |
| 1 | semantic_continuous | 0.787 | 0.716 | 2.12 | 0.08 |
| 1 | workspace | 0.701 | 1.042 | 3.89 | 0.04 |
| 2 | j | 0.982 | 0.203 | 4.92 | 0.44 |
| 2 | logits | 0.962 | 0.288 | 3.24 | 0.32 |
| 2 | semantic_continuous | 0.932 | 0.499 | 4.19 | 0.32 |
| 2 | workspace | 0.854 | 0.707 | 5.12 | 0.16 |

V13 alpha rows were retained and relabeled; no V13 result changed. Qualification uses `alpha × full-teacher J-effect norm` as a development-data proxy, not a newly measured scaled effect. h1 all-row versus SNR-qualified reinterpretation:

|   scale | direction_snr_label           |   n |     j |   output |
|--------:|:------------------------------|----:|------:|---------:|
|   0.050 | BELOW_DIRECTION_SNR_THRESHOLD |  10 | 0.119 |   -0.048 |
|   0.100 | BELOW_DIRECTION_SNR_THRESHOLD |  10 | 0.348 |    0.047 |
|   0.250 | BELOW_DIRECTION_SNR_THRESHOLD |   7 | 0.503 |    0.291 |
|   0.250 | SNR_QUALIFIED                 |   3 | 0.887 |    0.547 |
|   0.500 | SNR_QUALIFIED                 |  10 | 0.693 |    0.523 |
|   0.750 | SNR_QUALIFIED                 |  10 | 0.736 |    0.582 |
|   1.000 | SNR_QUALIFIED                 |  10 | 0.726 |    0.614 |

All `α=.05/.10` h1 rows fall below the direction threshold. At `α=.25`, the 3 qualified h1 rows improve J direction to about 0.887 but output direction remains about 0.547. Later-horizon qualified rows also fail. Thus near-zero metric instability explains part, not all, of V13 finite-control failure.

Machine records: `results/v14/processed/jvp_finite_writeback_audit_v14.parquet`, both numerical extension folders, `numerical_snr_summary_v14.parquet`, and `v13_alpha_snr_reanalysis_v14.parquet`. The original JSON audit preserves its `NaN` undefined-cosine tokens; the separately frozen strict-JSON correction maps only those tokens to `null` in `jvp_finite_writeback_audit_v14_strict.json`.
