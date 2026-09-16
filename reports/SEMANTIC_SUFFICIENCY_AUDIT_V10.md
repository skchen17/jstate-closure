# Semantic Sufficiency Audit v10

<!-- V10-AMENDMENT:START -->
## Interpretation

At 512D final test, baseline=`0.6264`, compact=`0.6260`, full ceiling=`0.6260`, and compact/full retention=`1.0000`. The observational semantic representation gate passes.

The absolute probe ceiling is only about 0.626, so low absolute readout quality is a probe/target ceiling issue rather than compact-specific information loss. This is distinct from decoded-causal semantic fidelity, which fails and diagnoses the decoder/interface rather than this observational gate.
<!-- V10-AMENDMENT:END -->


Absolute semantic quality and relative semantic sufficiency are separate gates. The absolute probe ceiling is reported without requiring compact state to exceed it; relative sufficiency uses compact/full retention and full-minus-compact residual gain.

|   dimension |   semantic_baseline |   semantic_compact |   semantic_full_ceiling |   semantic_retention_relative_to_full |   semantic_residual_gain |   semantic_residual_lower |   semantic_residual_upper | semantic_relative_pass   |
|------------:|--------------------:|-------------------:|------------------------:|--------------------------------------:|-------------------------:|--------------------------:|--------------------------:|:-------------------------|
|          64 |              0.6264 |             0.6252 |                   0.626 |                              0.998722 |              0.0008      |                  -0.0024  |                    0.004  | True                     |
|          96 |              0.6264 |             0.6248 |                   0.626 |                              0.998083 |              0.0012      |                  -0.002   |                    0.0044 | True                     |
|         128 |              0.6264 |             0.6252 |                   0.626 |                              0.998722 |              0.0008      |                  -0.0028  |                    0.0044 | True                     |
|         160 |              0.6264 |             0.6268 |                   0.626 |                              1.00128  |             -0.0008      |                  -0.004   |                    0.0024 | True                     |
|         192 |              0.6264 |             0.6272 |                   0.626 |                              1.00192  |             -0.0012      |                  -0.0044  |                    0.002  | True                     |
|         224 |              0.6264 |             0.628  |                   0.626 |                              1.00319  |             -0.002       |                  -0.0052  |                    0.0012 | True                     |
|         256 |              0.6264 |             0.628  |                   0.626 |                              1.00319  |             -0.002       |                  -0.0048  |                    0.0008 | True                     |
|         288 |              0.6264 |             0.6264 |                   0.626 |                              1.00064  |             -0.0004      |                  -0.0032  |                    0.0024 | True                     |
|         320 |              0.6264 |             0.6264 |                   0.626 |                              1.00064  |             -0.0004      |                  -0.00281 |                    0.0024 | True                     |
|         352 |              0.6264 |             0.6272 |                   0.626 |                              1.00192  |             -0.0012      |                  -0.0032  |                    0.0008 | True                     |
|         384 |              0.6264 |             0.6272 |                   0.626 |                              1.00192  |             -0.0012      |                  -0.0036  |                    0.0012 | True                     |
|         416 |              0.6264 |             0.6272 |                   0.626 |                              1.00192  |             -0.0012      |                  -0.0032  |                    0.0004 | True                     |
|         448 |              0.6264 |             0.628  |                   0.626 |                              1.00319  |             -0.002       |                  -0.004   |                   -0.0004 | True                     |
|         480 |              0.6264 |             0.6276 |                   0.626 |                              1.00256  |             -0.0016      |                  -0.0032  |                   -0.0004 | True                     |
|         512 |              0.6264 |             0.626  |                   0.626 |                              1        |              1.11022e-19 |                  -0.0012  |                    0.0012 | True                     |
