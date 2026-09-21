# V25 Complete Report

## Identity and frozen scope

- Name: **Distributed Causal Interaction and Response Convergence**
- Parent commit: `2197d58dc6699d3a6bba467f703f267fa3dcabc7`
- Base protocol hash: `1889c4d8b1534302a241b2082fec2b243483b454ee9bc01b9749acfacfd8b8ad`
- Design freeze: `a4f3c479997194f52ef6c974a2d8e365049952394796a79cb1fa1c47ca442d28`
- Component-basis freeze: `204b6e88d6b90d2aa75af953f656d0861d3d33202e2072548bb1aa8dd0fb3144`
- Strict interaction amendment: `0236f6e4735091c1cb35c799b30d1e3ae218c76c0924167f1a2605aa039d0f72`
- Zero-rank amendment: `8be5b300f21c4c6aedc9fd814f349198e796a8d14ef236015af96e97bf5f47a2`
- Adjudication freeze: `cf08a168a4695704ed5c2678c12875d732f2c9350b06413edc96203be686c8eb`
- Independent final: 50 states, 10 per family, hash `bf2998344c5c3d2e4da4579b1db3a85f7886e2eabc12b262708129af34bb9fcf`, frozen and unopened.

## Formal result

**V25-B_CAUSAL_CANCELLATION_SUPPORTED+V25-E_READOUT_COMPRESSION_ONLY**.

`V25-A=FALSE`, `V25-B=TRUE`, `V25-C=FALSE`, `V25-D=FALSE`, `V25-E=TRUE`, `V25-F=FALSE`, `V25-G=FALSE`, `V25-H=FALSE`.

The supported picture is causal cancellation plus low-dimensional behavioral/readout compression without a demonstrated low-dimensional causal realization or internal convergence.

## Rank audit

Raw uncentered broad responses reproduce r95=1 at every layer. State-wise intervention centering yields r95 13–14, and centered plus row-normalized responses yield r95 24–25. Thus V24's rank-one result is measurement-definition-specific and does not establish a general one-dimensional causal geometry.

## Strict distributed interaction

The corrected estimand is `Y00=clean; Y10=clean+B; Y01=clean+C; Y11=clean+(B+C)` over `160` rows. Median interaction ratio is `0.340985` with bootstrap lower bound `0.319602`. Interaction centered r95 is `77`. However, the strict writeback gate fails (minimum B cosine `0.633638`), so V25-A is false. Cancellation index `0.257974` supports V25-B.

## Layerwise response geometry

Exact-zero matrices at layers 0–23 have rank zero. The first nonzero layer is 24. Pooled centered r95 rises from `56` at layer 24 to `99` at layer 31, and every family rises. Many-to-one response convergence and internal causal convergence are therefore false. Frozen mapping layers before 31 are zero-response and cannot identify a meaningful contraction map.

## Distributed realization and readout

No dimension among `[4, 8, 16, 32, 64, 128, 256]` meets the same-J realization criteria; at dimension 256, relative L2 is `0.568966`. In contrast, centered r95 is `47` for direct full hidden, `19` for J-128, `19` for logits-32, and `15` for semantic-32. This supports output/readout compression only, not compact internal state.

## Other diagnostics

- Background-conditioned response coordinates: `FALSE`.
- Channel factorial medians: `{"ConvxKV": 0.730175992084515, "RECxConv": 1.317024583107477, "RECxConvxKV": 1.2131817544658459, "RECxKV": 1.0587004306136611}`.
- Order relative-difference median/max: `0.435503/0.934490`.
- Multihorizon status: `NOT_OPENED_NO_FROZEN_ALL_FAMILY_TWO_TOKEN_CONFIRMATORY_PANEL`.
- Interaction/convergence correlations are association-only and are not causal proof.

## Append-only corrections

1. Effective-rank values were clamped to ambient dimension without overwriting original records.
2. The formal interaction estimand was corrected to clean+B+C for Y11; the persistent-FULL records remain historical diagnostics.
3. Exact-zero response matrices were assigned rank zero; the earlier cumulative-energy artifact was retained but superseded.

## Authorization and verification

- `H2_REMAINS = TRUE`
- `H3_AUTHORIZED = FALSE`
- `DYNAMIC_STATE_SEARCH_AUTHORIZED = FALSE`
- Historical final opened: `FALSE`
- V25 independent final opened: `FALSE`
- V25 tests: `6 passed in 5.89s`
- Full suite: `256 passed, 2 failed, 3 warnings in 77.35s (inherited V14/V16 FINAL_REPORT hash checks)`; the two failures are inherited V14/V16 cumulative-report hash checks.
