# V24 Complete Report

## Identity

- Name: **Causal Output Bottleneck Validation and Layer Localization**
- Parent: `4dcc412d85fb2995237c48ec755beb34fd662941`
- Protocol hash: `36ecce8f74a00071b4cae5293ba13a9db7aabc53274cd7d6d8531ab42af3c904`
- State hashes: `{"development": "ffdd34027b9881b637b85f1cca2dab58891770711a3c80cc6ce3e03893949950", "independent_final": "545c291dcae7b69bb6265cacc31b1c153fe36e03f0932568bfb881b75864726f", "mediation_basis": "cdde0503b5d03fef89ea9482292fd9fb960e6153370a6df965e6ec6c0db79279", "mediation_validation": "3608bd70eb2b3b2d51fc08a3fd883318dc1961c55332e593ca9620c5ac7727cc", "natural_transition": "65ee72641565a91f68ad6a9f3a41fbb4469bb7bfd8d9f6a13df9e9e136a79fec", "validation": "03546c5f088c31e094e56dc9683706dadc800cdc8ac9b7361917efa1e196bcbb"}`
- Action hashes: `{"heldout": "793cab02928568a7201c863993943c76da566809bb51fb6ce7f573e91fa14d5f", "mediation": "789d44b22902b45a7d811b9f2df8354ff279b20039f6a7f194a07539e8f3a5fc", "train": "764b2fa64b26b1ad9fb1831c6bb32315b44e0f08674f729ce5e0c7e765dc7467"}`
- Target projection hash: `99e08cd12c9b3751e251c36c6352a16fc6c98951021cbb1abca7b7b2b50d26f9`
- Candidate layer/k: `31/24`
- Mediation protocol hash: `029b32788b0e22e14d7a62a813532d25f360e7c8aeabf0a05e0ab7780484cfa5`
- Mediation-results amendment hash: `45564a3ba9071f66b61b7ca6f4a918d32c989cb8612472607ac20f25bd49ba31`
- Natural transitions: `2` eligible h2−h1 rows; `8` audited h1-only exclusions from ten frozen design rows.

## Outcome

Formal outcome: **V24-F_DISTRIBUTED_CAUSAL_MEDIATION**.

Representational bottleneck pass: `FALSE`. Causal mediation gate: `FALSE`. A causal mediator is not equated with a complete model state.

## Compression profile

- V23 input JVP/finite r95@256: `28.0/33.5`.
- Early/candidate/late broad hidden r95: `1.0/1.0/1.0`.
- Late T0/vocabulary r95: `1.0/1.0`.
- Formal collapse layer: `N/A`.

## Held-out coverage

| basis | explained norm | residual | cosine |
|---|---:|---:|---:|
| B_FAMILY | 0.847354 | 0.390700 | 0.920724 |
| B_GLOBAL | 0.832632 | 0.409102 | 0.912632 |
| B_LOCAL_J | 0.604833 | 0.628618 | 0.777834 |
| B_LOCAL_P | 0.727686 | 0.521835 | 0.853064 |
| B_STATE_ORACLE | 0.999457 | 0.023293 | 0.999949 |
| CONTROL_RANDOM | 0.099770 | 0.948804 | 0.315820 |
| CONTROL_RANDOM_CAUSAL | 0.737741 | 0.512111 | 0.859573 |
| CONTROL_SHUFFLED_FAMILY | 0.478126 | 0.722403 | 0.691895 |
| CONTROL_VARIANCE | 0.235224 | 0.874506 | 0.485021 |

## Causal branches

- B_ONLY L2/cosine/norm: `0.501006/0.936193/1.298022`.
- PERP_ONLY retained, 97.5% upper: `1.375170/1.389140`.
- FULL-minus-B retained: `0.797728`.
- Restoration mediated fraction: `-1.382951`.
- Transplant L2/cosine/norm: `2.290694/0.209903/2.348547`.
- Regeneration ratio: `N/A`.
- Horizon status: `H1_FAILED_NO_H2_H4_H8_OPENING`.

## Authorization

- `H2_REMAINS = TRUE`
- `H3_AUTHORIZED = FALSE`
- `DYNAMIC_STATE_SEARCH_AUTHORIZED = FALSE`
- Independent final: `FROZEN_AND_UNOPENED_NO_SINGLE_DEVELOPMENT_VALIDATION_MEDIATOR`

## Verification

V24 tests: `6 passed in 28.72s`. Full suite: `2 failed, 250 passed, 3 warnings in 42.17s (inherited V14/V16 FINAL_REPORT hash assertions)`. Historical cumulative-report hash failures remain visible.
