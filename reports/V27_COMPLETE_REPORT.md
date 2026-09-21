# V27 Complete Report

## Identity

- Name: **Pre-Readout Counterfactuals and Next-Token Causal Re-Entry**
- Parent: `0657f731e8a165425baa231d4880bab36bf01169`
- Protocol hash: `1de32375db20b199876b0ce5077056393b95cf392c216b757eb6600d1973a59e`
- Design hash: `c7ac19b558d54cdf6fc561c70f7ae1e5207d6c85e02a1dce676c4ea2617176b7`
- Initial bank hash: `09fa769987745722f535d0382096eca99b3d8903f794eced2f191ce78b35a130`
- Expanded bank hash: `d657c1c23622dfbffc87a7a0db9c6beec11739c06e1b7389c16483f44b87939d`
- Adjudication hash: `4e885a2cb4fd83e0a6c6e3f27a8ed53e378fbd388a7a5ffdef8c940928db6342`

## Prospective design and boundary

The intervention was moved to the latest valid pre-readout boundary: the cache after all prompt tokens except the final token. The model then executed the final prompt token normally, with no J/logit/semantic/residual restoration or clamping. All six repeated-forward numerical floors were exactly zero. An ordinary intervention materially changed current output, confirming invariance was not built into the boundary.

The four disjoint balanced panels contain 25 calibration, 50 development, 25 validation, and 25 unopened independent-final states. All boundaries, targets, silence thresholds, candidates, routing layers/components, gates, and opening rules were frozen before response observation.

## Primary result

**V27-F_V26_BOUNDARY_SPECIFIC_ONLY**.

`V27-A=FALSE`, `V27-B=FALSE`, `V27-C=FALSE`, `V27-D=FALSE`, `V27-E=FALSE`, `V27-F=TRUE`, `V27-G=FALSE`, `V27-H=FALSE`.

The initial architecture-resolved bank tested 975 rows; the frozen V19 same-J expansion tested 3,600 rows. Of 4,575 total rows, 4,460 passed actuation reliability. None met `PRE_READOUT_SILENT` and none met `PRE_READOUT_DISTAL`. The minimum reliable aggregate h0 Q by family ranged from `0.347766` to `0.463103`, well above the 0.10 distal ceiling.

Therefore V27 found no evidence that a tested persistent distinction can be introduced before current readout, survive the full natural current-token computation, remain current-invisible, and then reappear at h1. V26 remains valid as a post-readout future-state result, but its stronger hidden-current-state interpretation is unsupported by this panel.

## Timing diagnostic

For a predeclared joint action on 10 states, median pre-readout h0 Q was `1.153113` versus post-readout `0`; h1 Q was `0.929026` versus `1.063260`. The h1 direction cosine was `0.608038`. This diagnostic confirms both boundaries can affect the future but only the post-readout boundary guarantees current silence.

## Sequential stop

`DETAILED_ROUTING_AUTHORIZED = FALSE`. Consequently no layer trace, first causal site, restoration, transplant, factorial, workspace reconstruction, re-entry geometry, natural comparison, or independent final was opened. Empty machine-readable routing artifacts make this stop explicit and prevent absence from being mistaken for a null routing result.

## Authorization

- `H2_REMAINS = TRUE`
- `H3_AUTHORIZED = FALSE`
- `DYNAMIC_STATE_SEARCH_AUTHORIZED = FALSE`
- `AUTONOMOUS_CONTROLLER_AUTHORIZED = FALSE`

## Verification

V27 tests: `6 passed in 1.76s`. Full suite: `268 passed, 3 failed, 3 warnings in 52.61s`; the three failures are inherited cumulative-report hash checks.
