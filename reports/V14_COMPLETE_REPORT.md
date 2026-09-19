# V14 complete report

> This is the canonical single-file bundle for this version. It combines the
> version-specific FINAL_REPORT section, every standalone version report, and
> an integrity index of the machine-readable records. Standalone reports remain
> preserved for direct navigation.

## Bundle provenance

- Source commit: `7aa05354178cc5a4c80f2729df0b32a749f5ebe6`
- Generated at: `2026-09-19T13:21:33.561572+00:00`
- Included standalone reports: `10`
- Indexed machine-record files: `25`
- Generator: `scripts/build_complete_version_report.py`

## Version summary and adjudication

## V14 — Finite Causal Control and Tangent Transport

Formal procedural outcome: **V14-STOP — NUMERICAL_INTERFACE_GATE_FAILED**. The BF16 persistent-state writeback creates a measured finite-effect floor; the frozen all-target exact-JVP/finite-difference equivalence gate did not pass through ε=2. `MIN_CAUSAL_EFFECT_NORM = 0.00680280`. SNR relabeling explains part of V13's small-alpha anomaly, not the later-horizon failures. First- and second-order valid radii: `None` / `None`. Train-atlas transport and holonomy are measurable, but they do not establish a global compact coordinate system.

Development closed-loop causal steering was run on ten validation cases; independent confirmation was **not** run because no numerically validated finalist was available. V14-A through V14-E are not fully established. H2 remains; H3 is not supported. Absolute state replacement and autonomous controller training are **not authorized**. V1–V13 conclusions remain frozen. See `reports/V14_COMPLETE_REPORT.md` for the full report bundle.

## Standalone report integrity index

| report | SHA256 |
|---|---|
| `reports/CAUSAL_HOLONOMY_V14.md` | `7c1bbd2b135c86a451b6a57b745c2be107d58ac5174c0c63f1fc9dba9a49d86c` |
| `reports/CAUSAL_TANGENT_TRANSPORT_V14.md` | `595388e6807960d261555c0d7501f37292b5582eded750b51fb9353818168a9f` |
| `reports/CLOSED_LOOP_CAUSAL_CONTROL_V14.md` | `c6d56e0129de23c61e7fe0f9d93b12580114512f060a41c8b0b745df9c013d99` |
| `reports/EXECUTION_MANIFEST_V14.md` | `17b952ecac6f98ee12572f44c8f3801f9276000c6f800c9d24aef844051b8fc3` |
| `reports/FINITE_CAUSAL_REACHABILITY_V14.md` | `bc4b3fd1f13212c7789a67b0500997273e23f0f875a596fd98ff6dd9869cb1f2` |
| `reports/JOINT_CHANNEL_CAUSAL_REALIZATION_V14.md` | `d8e33bf10e9e92a9f342234be779aaab1f27f2f85645f0af3bca0ef0c04a103d` |
| `reports/JVP_FINITE_WRITEBACK_AUDIT_V14.md` | `f0909258c9d9005fea8352ba18a1a6159a152b577043f6ce2bc14fc5f9c5f2b3` |
| `reports/LOCAL_CAUSAL_CURVATURE_V14.md` | `92746293d2d1c6f752047cd5056537c49a1cb0df2338471ff44c9f76a3a7e0a2` |
| `reports/STRICT_INTERFACE_AUDIT_V14.md` | `2535b44dfef95d847b8f2b4b5af3dcb54dff912c34128fbb7207d00c80e1ca10` |
| `reports/TRANSPORTED_PATH_DIMENSION_V14.md` | `5bcfb723b08667cd7d9aeba28f34cc9dcbb3f2dd031239ba904a10375fe6554a` |

## Machine-record integrity index

| record | bytes | SHA256 |
|---|---:|---|
| `results/v14/processed/clean_repeat_noise_v14.parquet` | 3956 | `00fa7cc7bd90f784129d290ae35ba55c2188bb57320a6a3f8f3a2b4799c525de` |
| `results/v14/processed/closed_loop_development_v14.json` | 4661 | `36aa38067673fbccce83f5d6b86e435b11505d5ceb0648f6f336e9b932eea7e4` |
| `results/v14/processed/closed_loop_development_v14.parquet` | 15035 | `58a7c6e804bc568b31aea15bc4a294712d47bb4b6a3e26f5eb85da8de3c0add2` |
| `results/v14/processed/closed_loop_fidelity_development_v14.parquet` | 27353 | `a39458b89fbf8f3bc1d21ab14910389e4cd74c765f5bda1f7f9a8f4c7d3413e2` |
| `results/v14/processed/closed_loop_progress_v14.json` | 49 | `48850e0a59f585cdaa8fa9ac8ad2cd800001b6db1e12f05baa88cbfe314f78d7` |
| `results/v14/processed/curvature_progress_v14.json` | 51 | `0f2a0af3e2d3535772793a1367b7cc6ee015ae3cb0bc12af682f0839a255b0c6` |
| `results/v14/processed/holonomy_loops_v14.parquet` | 7003 | `ec3737f84f69848d887aaa9a73c31a5b69b542602cec4916fdf0113efdde45fe` |
| `results/v14/processed/joint_channel_ablation_v14.json` | 912 | `f198056f129e7b8773e7ce5b6870277fd3654ceac63d5bc5ef21afd5cd787e66` |
| `results/v14/processed/joint_channel_ablation_v14.parquet` | 21270 | `726405bf2ee089e155b41d1af25852bfc8f89ad837850b51713e7fbd54bc7819` |
| `results/v14/processed/jvp_finite_writeback_audit_v14.json` | 25483 | `7f28b811936037444f909d4f6ce586041f756e6ad54579222944cd4d04a25b1c` |
| `results/v14/processed/jvp_finite_writeback_audit_v14.parquet` | 79527 | `280a16eaaf5f7478221808ba8e09a42689bbd01daf79346e44aaa9637ed1ca28` |
| `results/v14/processed/jvp_finite_writeback_audit_v14_strict.json` | 25696 | `9ceb5e5f9800ef9fa616ff0be726170fda4906889a6c80f6b6f8af12c013056f` |
| `results/v14/processed/local_causal_curvature_v14.json` | 4067 | `81bbf67ad52f6858fad1511ba0e26c91ff4bd228588e06fd6d8d1cb7a33863b5` |
| `results/v14/processed/local_causal_curvature_v14.parquet` | 43309 | `b4f77dd87dec439efa6edf02b02650e5069d3b7a9e8180fced4715c30125c5f0` |
| `results/v14/processed/local_causal_mixed_curvature_v14.parquet` | 5789 | `43fc9297caaaa170a07bb1321e4e8615e905ac659f35705ac318eefd2d470d70` |
| `results/v14/processed/numerical_audit_progress_v14.json` | 51 | `0f2a0af3e2d3535772793a1367b7cc6ee015ae3cb0bc12af682f0839a255b0c6` |
| `results/v14/processed/numerical_snr_summary_v14.json` | 6881 | `85c1fbfcc707558668edc4ebae639dd3558b8314be4569f97dbfa89c621dfd4c` |
| `results/v14/processed/numerical_snr_summary_v14.parquet` | 6274 | `9b882f3480496dffab4558aecdcfc402c9100821d178678cdb20d37256ec4d69` |
| `results/v14/processed/report_integrity_v14.json` | 1453 | `57282ef98743fdcbfa04535418474b117139e1a4c1a3aff3832a0ab9672018d2` |
| `results/v14/processed/transport_analysis_v14.json` | 3816 | `bae08997365378ef6ecaa084dcf5ba9c351ac3aa07bc30948b83c4004d29ca18` |
| `results/v14/processed/transport_pairs_v14.parquet` | 81083 | `50be599741969de90e61304aaca1afa865f2e18af8819808399cacea856a67a1` |
| `results/v14/processed/transported_path_dimension_v14.parquet` | 4825 | `fc2b55ad545dbf8a956078d07a89caddb0a036bea04857ed99d91e4747365395` |
| `results/v14/processed/v13_alpha_snr_reanalysis_v14.parquet` | 16645 | `0bcf6163a98d812111d5df11bfd5fcafee94c25193b7f519edec5bdb5c8d35cd` |
| `results/v14/processed/v13_oracle_snr_reanalysis_v14.parquet` | 69411 | `190e9e661bafae8ae2b217b54a7d77be07166ad60ae25d805c69007460e149db` |
| `results/v14/processed/v14_integrity.json` | 9699 | `16a4c8d0f5952f2447476e3adf8e5fbbe865d54bce243f8b22b9361aa67c1bb8` |

---

## Bundled report 1: `CAUSAL_HOLONOMY_V14.md`

# Path dependence and holonomy — V14

Ten train-state loops A→B→C→A were constructed, with same-prompt A/B and C chosen by nearest frozen JVP matrix within family when available. This does not guarantee that C is close in full raw state or J-state distance. Mean results by mapping:

|   coordinate_return_error |   direct_indirect_coordinate_error |   j_effect_return_error | method             |   subspace_holonomy_degrees |
|--------------------------:|-----------------------------------:|------------------------:|:-------------------|----------------------------:|
|                     0.390 |                              0.390 |                   0.118 | grassmann_geodesic |                      22.587 |
|                     1.051 |                              1.062 |                   0.314 | j_response         |                      67.996 |
|                     0.390 |                              0.390 |                   0.118 | procrustes         |                      22.587 |
|                     0.501 |                              0.332 |                   0.181 | projector          |                      49.157 |
|                     0.000 |                              0.000 |                   0.000 | unaligned          |                       0.000 |

Procrustes/geodesic coordinate return error is `0.390`, with J-effect return error `0.118`. This is measurable path dependence in the tested local atlas, not proof that a low-dimensional manifold is absent. The zero error of `unaligned` is a tautology (identity map), not evidence of flat geometry. Degree-style holonomy is geometrically interpretable only for the orthogonal maps; projector and J-response maps are not isometries. Channel-wise return error in physical cache space was not measured, so that part of the requested audit remains unidentified.

Machine records: `results/v14/processed/holonomy_loops_v14.parquet`.

---

## Bundled report 2: `CAUSAL_TANGENT_TRANSPORT_V14.md`

# Causal tangent transport — V14

V13's ten train-anchor local matrices were aligned at rank 16 with explicit sign/index-invariant mappings. Same-prompt successive-position mean fidelity:

|   j_direction_cosine |   logits_direction_cosine | method             |   raw_direction_cosine |   semantic_continuous_direction_cosine |   workspace_direction_cosine |
|---------------------:|--------------------------:|:-------------------|-----------------------:|---------------------------------------:|-----------------------------:|
|                0.584 |                     0.311 | grassmann_geodesic |                  0.796 |                                  0.282 |                        0.536 |
|                0.832 |                     0.351 | j_response         |                  0.462 |                                  0.282 |                        0.431 |
|                0.584 |                     0.311 | procrustes         |                  0.796 |                                  0.282 |                        0.536 |
|                0.597 |                     0.325 | projector          |                  0.833 |                                  0.293 |                        0.549 |
|                0.048 |                    -0.001 | unaligned          |                  0.038 |                                  0.001 |                        0.046 |

Procrustes and minimal Grassmann-geodesic endpoint transport coincide mathematically for these full-rank principal-angle alignments; their identical numbers are not independent replications. The reported `raw_direction_cosine` is actually a **512-dimensional probe-coordinate** cosine, not a metric-calibrated full raw-state cosine because the frozen probe directions need not be orthonormal. J-response alignment improves J-effect direction but can sacrifice probe-coordinate/output/semantic alignment, demonstrating a target-specific gauge choice. Only common selected logit IDs were compared across states; the comparison does not assert global coordinate identity or finite steering success.

Machine records: `results/v14/processed/transport_pairs_v14.parquet` (`1900` rows).

---

## Bundled report 3: `CLOSED_LOOP_CAUSAL_CONTROL_V14.md`

# Closed-loop finite causal control — V14 development

The tested controller uses a shared rank-9 REC/conv/KV direction basis, finite central response at ε=1 (because exact-JVP equivalence failed), ridge `0.01`, trust radius `0.5`, four steps, and backtracking `[1,.5,.25]`. At every accepted step the causal response is remeasured and the nearest V13 tangent atlas basis is retrieved. Only `1` atlas index was actually visited, so this run **did not test an effective between-basis transport switch**. It optimizes weighted future J, logits, continuous semantics, and workspace effects, **not raw target-state distance**. Horizon objectives were h1 and h1+h2+h4+h8.

Static V13, V13 moving-interpolated, and V14 closed-loop on the same validation panel:

| method                                  |   horizon |   direction |   magnitude |   output |   semantic_legacy |   semantic_continuous |   sign |
|:----------------------------------------|----------:|------------:|------------:|---------:|------------------:|----------------------:|-------:|
| closed_loop_finite_response_h1          |         1 |       0.699 |       0.981 |    0.671 |             0.430 |                 0.699 |  0.800 |
| closed_loop_finite_response_h1          |         2 |       0.535 |       1.005 |    0.514 |             0.110 |                 0.535 |  0.800 |
| closed_loop_finite_response_h1          |         4 |       0.498 |       1.000 |    0.537 |             0.070 |                 0.498 |  0.800 |
| closed_loop_finite_response_h1          |         8 |       0.511 |       1.008 |    0.491 |             0.120 |                 0.511 |  0.900 |
| closed_loop_finite_response_h1_h2_h4_h8 |         1 |       0.688 |       0.888 |    0.652 |             0.380 |                 0.619 |  0.600 |
| closed_loop_finite_response_h1_h2_h4_h8 |         2 |       0.593 |       0.905 |    0.492 |             0.080 |                 0.533 |  0.800 |
| closed_loop_finite_response_h1_h2_h4_h8 |         4 |       0.483 |       0.896 |    0.504 |             0.100 |                 0.435 |  0.800 |
| closed_loop_finite_response_h1_h2_h4_h8 |         8 |       0.524 |       0.855 |    0.488 |             0.080 |                 0.471 |  0.600 |
| moving_tangent_interpolated_oracle      |         1 |       0.827 |       0.874 |    0.818 |             0.500 |                 0.827 |  0.900 |
| moving_tangent_interpolated_oracle      |         2 |       0.725 |       0.903 |    0.590 |             0.290 |                 0.725 |  0.600 |
| moving_tangent_interpolated_oracle      |         4 |       0.599 |       0.931 |    0.580 |             0.160 |                 0.599 |  0.778 |
| moving_tangent_interpolated_oracle      |         8 |       0.497 |       0.992 |    0.475 |             0.090 |                 0.497 |  0.600 |
| static_local_causal_oracle              |         1 |       0.792 |       0.823 |    0.801 |             0.500 |                 0.792 |  0.800 |
| static_local_causal_oracle              |         2 |       0.708 |       0.901 |    0.625 |             0.330 |                 0.708 |  0.700 |
| static_local_causal_oracle              |         4 |       0.579 |       0.934 |    0.485 |             0.080 |                 0.579 |  0.778 |
| static_local_causal_oracle              |         8 |       0.540 |       0.998 |    0.500 |             0.070 |                 0.540 |  0.800 |

All-horizon frozen gate pass by method: `{'closed_loop_finite_response_h1': False, 'closed_loop_finite_response_h1_h2_h4_h8': False, 'moving_tangent_interpolated_oracle': False, 'static_local_causal_oracle': False}`. Continuous semantic fidelity is reported separately and does not replace the legacy gate. This is a development experiment, not V14-D independent confirmation. Differences versus V13 are exploratory because the controller's finite-response derivative was not validated as a local Jacobian.

The all-horizon objective did not consistently improve h4/h8 over h1-only (h4 J direction fell from about 0.498 to 0.483; h8 rose only from about 0.511 to 0.524). It did not resolve long-horizon rotation or semantic divergence. h16 was not authorized.

Per-step predicted/actual improvement, trust ratio, control norm, residual, and atlas index are in `results/v14/processed/closed_loop_development_v14.parquet`.

---

## Bundled report 4: `EXECUTION_MANIFEST_V14.md`

# V14 execution and provenance manifest

Server workspace: `/data/CSK/J-space-project/jstate-closure` on `222.20.126.223`. Parent commit: `37df399e4bba8afaca6d99721fe8ceee653e78ee`. V1–V13 tracked guard digest: `8b1829e2f9c4edea0102adc395eb626a93fc0822194d689bf7e67aebc9a1c2d0`. Diagnostic/development ID hashes: `b7efb4bab88dd3c2ab6e89ab0b017136e2cc26efe14ced28b8fb0b9867e97544` / `955ea2e3b3e0f05159838a99e88dfd9b752287a425d86a5374500f438f0547c9`. Independent-final status: `NOT_YET_CREATED_OR_SELECTED`.

Canonical commands (all from repo root with `/home/user/anaconda3/bin/python`; GPU stages use `HF_HOME=/data/CSK/J-space-project/.hf-cache`):

```bash
python -m jclosure.experiments.numerics_v14 --stage freeze
python -m jclosure.experiments.numerics_v14 --stage audit
python -m jclosure.experiments.numerics_v14_extension_1
python -m jclosure.experiments.analyze_numerics_v14
python -m jclosure.experiments.transport_v14 --stage analyze
python -m jclosure.experiments.curvature_v14
python -m jclosure.experiments.closed_loop_v14
python -m jclosure.experiments.numerics_v14_extension_2
python -m jclosure.experiments.channels_v14
python scripts/create_v13_immutable_for_v14.py
python scripts/freeze_v14_splits.py
python scripts/freeze_v14_finalist_decision.py
python scripts/normalize_v14_numeric_json.py
python -m jclosure.reporting_v14
python scripts/build_v14_integrity.py
python scripts/build_complete_version_report.py V14
python -m pytest -q
```

Frozen protocol and amendment hashes:

| freeze                                                                             | SHA256                                                           | digest                                                           |
|:-----------------------------------------------------------------------------------|:-----------------------------------------------------------------|:-----------------------------------------------------------------|
| artifacts/finite_causal_control_v14.freeze.json                                    | 8328bf510c87899b27b488a261d108e1d5feb992df0033a69dae3dcfdfedef50 | d268ee6932dbe1bfa1eeecca57c3fc3e97c48fa2793c0564a5e05f6419431cbf |
| artifacts/finite_causal_control_v14_channel_ablation.freeze.json                   | b6743b13aaf6d2205f41d97603b303b1661187bc4e2808a4074e0d7e563c1258 | 05da924b5b71038936b20005831b63145f3b24a3bcc8b282fb6b9531c6033df0 |
| artifacts/finite_causal_control_v14_closed_loop_development.freeze.json            | 21eebcc1a590b68a2c05ff07ec12eb47be5cb775a8a372917d696b704e200a18 | 53cbf8c9b6aa6092330489f4ac635bd55cad939db5cad655063cbd67a9634852 |
| artifacts/finite_causal_control_v14_curvature.freeze.json                          | 992645ee6d331724b0ed31ed94feb09501e9d6f8b0db59f02dfaf174cb6b767a | 4d27b2f63ed064c1ab10870a8eb1457280f0740b9057f5a64930a86996b0efed |
| artifacts/finite_causal_control_v14_finalist_decision.freeze.json                  | 119ec4fe1bad4db1645f4deddd7bb8909a77306f6a4509375a3a97cb20ef68d1 | ffcab29c9b960d1f4dfc490225ced4bc44613ba8841e6f0ca74185015058144c |
| artifacts/finite_causal_control_v14_numerical_partial_fp32_extension_2.freeze.json | 5c3ea5fc485da0f471f1f52e7cf1561b25be21759687472c50ee7fac73cc2a4e | 0125a0ff15632875abd1cfe81805f575762560c7f6e3f7f5e54708447afc940d |
| artifacts/finite_causal_control_v14_numerical_scale_extension_1.freeze.json        | 21b8dac6ab502f2984316866cc02cd4a00716855c36cd81a2058ff29d4923052 | a232ec478862444acd7a29ec2de7b1ff739e00710ac79f0e599f19280bcbc709 |
| artifacts/finite_causal_control_v14_snr_threshold.freeze.json                      | f83106c7c9884e3a716b8fc3dd3928383982bf6d2db8c7841b98a235ef26bdc3 | 0eb484ad5310d349e3ebe34fa39d929c5541b161afda90a4b6accf5825eb32cf |
| artifacts/finite_causal_control_v14_splits.freeze.json                             | bff312bd0211fc9f53a11779357b2d37ae01b9dd2c2567c34a53d7b31deb1bd5 | 07faac4e326aa22d3efbc852edc006f12668a7b7371865ca32461208e2a6c473 |
| artifacts/finite_causal_control_v14_strict_json_correction.freeze.json             | 00a2abdc6570dcbce50f75593cf600e2970ed1b3146bd19838134141d602e720 | 3eaecd2fa9f5218f5ebd1ea96ceff6aecd02f403303f155b72c6bfcabf6c5b97 |
| artifacts/finite_causal_control_v14_transport.freeze.json                          | 4b9f7446dfdacfbd16fb084d7c6f4b2ff4528e7f81318ab974b9b5aebee7b9ae | a62a1063af82b0c8f6766cf3a16553f8364ab292b2ead750872654f4f9e07dd7 |

Machine-record hashes are indexed in `reports/V14_COMPLETE_REPORT.md` for top-level files and in the nested extension summaries for extension records. Exact changed-file list after commit is obtained with `git diff --name-only 37df399e4bba8afaca6d99721fe8ceee653e78ee..HEAD`.

---

## Bundled report 5: `FINITE_CAUSAL_REACHABILITY_V14.md`

# Finite causal reachability — V14 development only

Teacher-target residuals were measured for the same ten V13 validation cases used by the prior development oracle. The controller never receives a raw target cache; raw teacher state is used only to evaluate the teacher response label. Residual ratio is `||Y*−Y(P_k)|| / ||Y*−Y(P_0)||` for the selected horizon objective.

|   acceptance | method                                  |   raw_ratio |   step |   weighted_ratio |
|-------------:|:----------------------------------------|------------:|-------:|-----------------:|
|        1.000 | closed_loop_finite_response_h1          |       0.736 |      1 |            0.750 |
|        0.900 | closed_loop_finite_response_h1          |       0.637 |      2 |            0.655 |
|        0.556 | closed_loop_finite_response_h1          |       0.564 |      3 |            0.564 |
|        0.600 | closed_loop_finite_response_h1          |       0.492 |      4 |            0.579 |
|        0.900 | closed_loop_finite_response_h1_h2_h4_h8 |       0.762 |      1 |            0.871 |
|        0.556 | closed_loop_finite_response_h1_h2_h4_h8 |       0.677 |      2 |            0.845 |
|        0.600 | closed_loop_finite_response_h1_h2_h4_h8 |       0.546 |      3 |            0.777 |
|        0.667 | closed_loop_finite_response_h1_h2_h4_h8 |       0.484 |      4 |            0.724 |

Terminal case categories (development diagnostics, not independent reachability claims):

| method                                  | family                  | class                      |   cases |
|:----------------------------------------|:------------------------|:---------------------------|--------:|
| closed_loop_finite_response_h1          | boolean_logic           | partial_reduction_or_stall |       2 |
| closed_loop_finite_response_h1          | modular_arithmetic      | partial_reduction_or_stall |       2 |
| closed_loop_finite_response_h1          | short_graph_traversal   | intermediate_reduction     |       2 |
| closed_loop_finite_response_h1          | simple_state_transition | partial_reduction_or_stall |       2 |
| closed_loop_finite_response_h1          | variable_binding        | partial_reduction_or_stall |       2 |
| closed_loop_finite_response_h1_h2_h4_h8 | boolean_logic           | partial_reduction_or_stall |       2 |
| closed_loop_finite_response_h1_h2_h4_h8 | modular_arithmetic      | partial_reduction_or_stall |       2 |
| closed_loop_finite_response_h1_h2_h4_h8 | short_graph_traversal   | intermediate_reduction     |       1 |
| closed_loop_finite_response_h1_h2_h4_h8 | short_graph_traversal   | no_meaningful_reduction    |       1 |
| closed_loop_finite_response_h1_h2_h4_h8 | simple_state_transition | intermediate_reduction     |       2 |
| closed_loop_finite_response_h1_h2_h4_h8 | variable_binding        | partial_reduction_or_stall |       2 |

Backtracking enforces improvement in the weighted objective, not necessarily every unweighted target or horizon. A failure here cannot establish V14-E while the local finite-response model and independent confirmatory panel remain unvalidated.

Machine records: `results/v14/processed/closed_loop_development_v14.parquet`.

---

## Bundled report 6: `JOINT_CHANNEL_CAUSAL_REALIZATION_V14.md`

# Shared-coordinate channel realization — V14

The same local rank-3 coordinates were realized jointly in REC, convolution, and KV heads. Each channel was ablated at finite scale 1.0, without fitting a large decoder. Low-effect rows are retained but not classified as channel-required under the frozen `MIN_CAUSAL_EFFECT_NORM`.

- SNR-qualified ablation rows: `45`.
- Fraction marked `JOINT_CHANNEL_REQUIRED` when one-channel removal made J cosine <0.8 or norm ratio <0.8: `0.4222222222222222`.

|   j_cosine |   j_norm_ratio | removed_channel   |   required_fraction |
|-----------:|---------------:|:------------------|--------------------:|
|      0.386 |          0.328 | conv              |               1.000 |
|      1.000 |          0.999 | kv                |               0.000 |
|      0.840 |          0.976 | recurrent         |               0.267 |

This is a finite writeback ablation, not a learned shared latent decoder or complete-state replacement. Machine records: `results/v14/processed/joint_channel_ablation_v14.parquet`.

---

## Bundled report 7: `JVP_FINITE_WRITEBACK_AUDIT_V14.md`

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

---

## Bundled report 8: `LOCAL_CAUSAL_CURVATURE_V14.md`

# Local causal curvature — V14

The top `9` local causal directions were taken from frozen V13 rank-512 JVP matrices, one train anchor per family. Central second differences at radius `1.0`, two mixed-direction pairs, and held-out radii `0.5` and `2.0` were measured. Since the infinitesimal JVP gate failed, these are **finite-scale curvature diagnostics, not validated differential Hessian estimates**.

|   radius | target              | surrogate   |   direction |   relative_l2 |   magnitude |
|---------:|:--------------------|:------------|------------:|--------------:|------------:|
|    0.500 | j                   | linear      |       0.921 |         0.541 |       1.253 |
|    0.500 | j                   | quadratic   |       0.949 |         0.422 |       1.145 |
|    0.500 | logits              | linear      |       0.867 |         0.602 |       1.096 |
|    0.500 | logits              | quadratic   |       0.907 |         0.529 |       1.166 |
|    0.500 | semantic_continuous | linear      |       0.854 |         0.681 |       1.040 |
|    0.500 | semantic_continuous | quadratic   |       0.926 |         0.463 |       1.063 |
|    0.500 | workspace           | linear      |       0.755 |         0.701 |       0.931 |
|    0.500 | workspace           | quadratic   |       0.924 |         0.408 |       1.028 |
|    2.000 | j                   | linear      |       0.643 |         2.289 |       2.595 |
|    2.000 | j                   | quadratic   |       0.838 |         2.253 |       2.878 |
|    2.000 | logits              | linear      |       0.556 |         1.257 |       1.547 |
|    2.000 | logits              | quadratic   |       0.799 |         1.453 |       2.232 |
|    2.000 | semantic_continuous | linear      |       0.535 |         1.389 |       1.561 |
|    2.000 | semantic_continuous | quadratic   |       0.804 |         1.724 |       2.315 |
|    2.000 | workspace           | linear      |       0.172 |         1.118 |       0.640 |
|    2.000 | workspace           | quadratic   |       0.933 |         0.902 |       1.587 |

- First-order valid radius under the all-target direction/magnitude/relative-error rule at tested held-out radii `0.5/2.0`: `None`.
- Second-order valid radius under the same tested radii: `None`.
- The quadratic term improves direction prediction at radius `0.5`, but relative errors remain above `0.20`; it does not validate a finite local surrogate.
- Mixed second-difference terms were retained in `results/v14/processed/local_causal_mixed_curvature_v14.parquet`.

---

## Bundled report 9: `STRICT_INTERFACE_AUDIT_V14.md`

# Strict interface and authorization audit — V14

- V1–V13 tracked-byte guard: `1883` files, digest `8b1829e2f9c4edea0102adc395eb626a93fc0822194d689bf7e67aebc9a1c2d0`; cumulative `FINAL_REPORT.md` is the declared mutable exception.
- V14 diagnostic train / development validation are disjoint; split hashes `b7efb4bab88dd3c2ab6e89ab0b017136e2cc26efe14ced28b8fb0b9867e97544` / `955ea2e3b3e0f05159838a99e88dfd9b752287a425d86a5374500f438f0547c9`.
- A new independent confirmatory bank was **not** created or used. The numerical all-target JVP gate failed and the frozen finalist decision is `NO_ELIGIBLE_FINALIST` (digest `ffcab29c9b960d1f4dfc490225ced4bc44613ba8841e6f0ca74185015058144c`). This is a required stop, not a final-test success.
- Raw teacher cache never enters the candidate control update or basis search; only teacher response labels define residuals. There is no direct target-cache bypass in the developmental controller.
- Development all-horizon gate pass: `False`. Independent h1/h2/h4/h8 gate pass: **not tested**.
- Absolute state replacement: `ABSOLUTE_STATE_REPLACEMENT_NOT_AUTHORIZED`. V13/V14 controls are delta/edit interfaces, not absolute raw-state representations.
- H2 remains. H3 not supported. `AUTONOMOUS_CONTROLLER_AUTHORIZED = FALSE`.

Formal procedural outcome: `V14-STOP — NUMERICAL_INTERFACE_GATE_FAILED`. No V14-A–E scientific branch is fully established: a writeback floor is demonstrated, but all-target JVP/finite equivalence and independently confirmed finite control are not.

---

## Bundled report 10: `TRANSPORTED_PATH_DIMENSION_V14.md`

# Transported path dimension — V14

For ten three-state train paths, naive **probe-coordinate-basis** union r95 has median `19.0` and range `18–23` at local rank 16. Transported *coordinate* dimension is by construction at most 16. These quantities live in different ambient spaces; Procrustes rotation within a basis cannot change the union rank in the original probe-coordinate space. Therefore V14 does **not** claim that transport lowered V13's cumulative rank (`11` under its different frozen rank/path panel), much less full raw-state dimension.

Machine records: `results/v14/processed/transported_path_dimension_v14.parquet`.
