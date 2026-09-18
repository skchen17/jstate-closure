# V13 complete report

> This is the canonical single-file bundle for this version. It combines the
> version-specific FINAL_REPORT section, every standalone version report, and
> an integrity index of the machine-readable records. Standalone reports remain
> preserved for direct navigation.

## Bundle provenance

- Source commit: `8270e77174f51583dd6d1dac187485450a495e63`
- Generated at: `2026-09-18T02:57:57.282090+00:00`
- Included standalone reports: `12`
- Indexed machine-record files: `29`
- Generator: `scripts/build_complete_version_report.py`

## Version summary and adjudication

## V13 — expanded causal bank, probe scaling, and path geometry

Formal decision: **V13-F — HIGH-DIMENSIONAL WRITABLE STATE**.

- New bank sizes: `{'final_test': 250, 'train': 4800, 'validation': 500}`; independent final was not used for selection.
- Restricted r95 curve m=64/128/256/512: `{'64': 5.0, '128': 6.5, '256': 8.0, '512': 9.0}`.
- Stable low local causal rank under the frozen expanded probes: `True`.
- Low-variance/high-causal directions: `False`.
- Same-prompt rank-16 tangent angle: `29.18°`.
- Moving tangent improves static: `True`; independent h1/h2/h4/h8 pass: `False`.
- Practical local linear radius: `None`.
- Instantaneous/cumulative median r95: `8.0` / `11.0`.
- Smallest independently validated writable dimension: `None`.
- Complete replacement state: **False** (`ABSOLUTE_REPLACEMENT_NOT_YET_TESTABLE_FROM_CURRENT_DELTA_REPRESENTATION`).
- H2 remains: **True**. H3 authorized: **False**.
- Autonomous controller authorized: **False**.

Historical V1–V12 conclusions remain frozen. V13 causal-edit results are not described as absolute state replacement, and restricted exact-JVP rank is not described as full raw-state intrinsic dimension.

## Standalone report integrity index

| report | SHA256 |
|---|---|
| `reports/CAUSAL_BANK_EXPANSION_V13.md` | `96c6034ec5f3a15ac7f6e35943e42067083dfabc4f91e842d1e47c24ee156247` |
| `reports/CAUSAL_PATH_DIMENSION_V13.md` | `61b0529d542f4cb3b0924528f06ab52d4cb748bcf6b68b9ec8d6f9b759f551a6` |
| `reports/CAUSAL_PROBE_SCALING_V13.md` | `424daa5bce0f5af13765a4b1b66763a0cf07f8141b02783f801eb07ac3d7fd4d` |
| `reports/CAUSAL_RANK_ROBUSTNESS_V13.md` | `1d18cc97aeea0228bcdef14284fa263fa5d8f770df05cf6fab381fc00262e0eb` |
| `reports/CAUSAL_TANGENT_ATLAS_V13.md` | `9fa83e6f15d80d96568dad8f14a4fda12c0783b8c7d927ec7f058cdbfac3f728` |
| `reports/CAUSAL_TARGET_COMPLETENESS_V13.md` | `5e9a904c8288eef9898a1b3cd857a289c3a4c8cfab5234054cac71cad99d4ced` |
| `reports/DATA_DIMENSION_SCALING_V13.md` | `33aaabba2768bde3b17241eeaee78834c5875d21c0f44a2317c863a40c8f2ce2` |
| `reports/EXECUTION_MANIFEST_V13.md` | `132c10d0ad4f1ca60fa7cc09b250e5624d723290369618254f558e17eb8983f9` |
| `reports/JOINT_CHANNEL_CAUSAL_GEOMETRY_V13.md` | `e0b91bc2d01fb9b7929912a32474861d4ca58a3016805e2fd97e0f07d9308312` |
| `reports/LOCAL_LINEARITY_RADIUS_V13.md` | `5066ec7c3831751bf1373bb99a2c2578084c70e05c86a54f0016378c0f0da00c` |
| `reports/MOVING_TANGENT_CAUSAL_ORACLE_V13.md` | `5d788774c447d34c5190e7ca9e59010c7dd81ff1c625296021298619eb01d450` |
| `reports/STRICT_STATE_REPLACEMENT_V13.md` | `c8b7b99f16f32e69627900c2ef043eeec62ee2fcc203964727f8908d78d71d38` |

## Machine-record integrity index

| record | bytes | SHA256 |
|---|---:|---|
| `results/v13/processed/causal_bank_expansion_v13.json` | 103974 | `fec5f8eca7443e81ea32ef5bbda7ba98c3f3b9108d6c9c7be8fbb63b959e1a8a` |
| `results/v13/processed/causal_capture_final_test_v13.json` | 20036 | `b584b92b2d39bb32c5071f0bd2b94007f264fa8b7e53f3f5115472cdf840f7c4` |
| `results/v13/processed/causal_capture_train_v13.json` | 333502 | `9aa75d01649d10f019e8c91607e425a2f2e3815fb3c3235449dd419144728419` |
| `results/v13/processed/causal_capture_validation_v13.json` | 38983 | `dc219f681106e7993df067b4a7edc3890e8421c43e09cca841c9527778176f40` |
| `results/v13/processed/causal_features_v13.json` | 1109 | `33320cfe55b2a18516a0b4ee99eb9669c5184f56cf9019b609069768425f969e` |
| `results/v13/processed/causal_path_dimension_v13.parquet` | 4174 | `e06f22addab76709ca21d994f4cb4c2a5c626754490a22401f940a0788a050b5` |
| `results/v13/processed/causal_probe_scaling_scalar_shard_0_v13.json` | 2027 | `b52a428817d0fc100e42b142c5c2ac5a5275ea9fd9cb4438df2a2e047526f782` |
| `results/v13/processed/causal_probe_scaling_scalar_shard_0_v13.parquet` | 139655 | `c18c91d236b71a3fb03e326b2d1e94cd2b75313d0792b77a24b3040585c1b5cb` |
| `results/v13/processed/causal_probe_scaling_scalar_shard_1_v13.json` | 2030 | `7edc9e2400a2ccf1cc5e1badda1f89f398fdee1f59a7f40dc2b21cbdac0c2d87` |
| `results/v13/processed/causal_probe_scaling_scalar_shard_1_v13.parquet` | 139895 | `a4cb9dc9a0c31253671ac09473ec18c71e7b0d5d8590d19e824de50e8770eb2f` |
| `results/v13/processed/causal_probe_scaling_v13.json` | 2255 | `fbddf7b9b4f09a0500692647ab0b7fe6aba516a16432aaad3afdad654a12b4ea` |
| `results/v13/processed/causal_probe_scaling_v13.parquet` | 264947 | `da45a7ec022c0ec2cfad43e47123cd595dd00de92bc57bcf56d19ea4487152af` |
| `results/v13/processed/causal_tangent_atlas_v13.parquet` | 21120 | `f867e9e71b00fbc77ed515dd285dfc78f5879f6b8dd609b077ab8430554c05d3` |
| `results/v13/processed/causal_target_completeness_v13.parquet` | 7340 | `80719becf24603f8e15182ad2c46019d47ff055d9f74cdae46bbc448868dadf9` |
| `results/v13/processed/data_dimension_scaling_v13.json` | 11365 | `0269ac94933c6b120603b1a92c426a4ccb2583d382bc936f7ae3ac2f8fe6af5f` |
| `results/v13/processed/data_dimension_scaling_v13.parquet` | 48315 | `6c689660c3c75ebf6e749aaa9fc40268e33c33bc00ab7eaf691979e5b14fa810` |
| `results/v13/processed/geometry_analysis_v13.json` | 9048 | `15d789c703c397725c936a2dbc41d39252d2e0475f185e62a45e7521ba5a4278` |
| `results/v13/processed/jvp_batch_equivalence_b32_v13.json` | 7540 | `17d3b68bf1a2be50ab797f1b4b5280fdea71a211a53db1c6b79672832f1cfafc` |
| `results/v13/processed/jvp_batch_equivalence_v13.json` | 308 | `fb36c9ca0fa16b47afe9f6c0c7b07dfae71f26f78e3caf3522f8efcf659ae842` |
| `results/v13/processed/jvp_scalar_shard_merge_v13.json` | 889 | `1c115b5b4e45d296454279e3ba10e505e8743a0a8467c90e6d21ca2dfbe8009e` |
| `results/v13/processed/local_linearity_radius_v13.json` | 5496 | `8d434a043126588279b32b88cd6ecff8dbcd36b3713e418356052808cfa00237` |
| `results/v13/processed/local_linearity_radius_v13.parquet` | 13201 | `c520269f5e1624c949bf20f0375e9a8450cc25329a084dfd85dad05404a693ea` |
| `results/v13/processed/moving_tangent_oracle_confirmatory_v13.json` | 6778 | `87f677cb92b4270fbb4aa6b029b1ca829415559c07d4cc0f2947ec010a6323aa` |
| `results/v13/processed/moving_tangent_oracle_confirmatory_v13.parquet` | 69695 | `afac610926b80c957a450f9cd39df2cdbdf800b7a71b550303185631342866e9` |
| `results/v13/processed/moving_tangent_oracle_development_v13.json` | 15645 | `c1adfea374c5109a2507eefa281264eeee47ca4d936c623736d24e568944782f` |
| `results/v13/processed/moving_tangent_oracle_development_v13.parquet` | 68673 | `e2de0a15b294fb9c5678a7772340bdb359026bb7f354d8242cfeb7ad890dbfde` |
| `results/v13/processed/probe_directions_v13.json` | 39914 | `2a98ff65431c7c1e8cf43f3c34ec999b15cc4194edf246331bb29e7e9ecbc5b4` |
| `results/v13/processed/report_integrity_v13.json` | 1445 | `4cac917f442f58a82a4d40de923350d94332773a022e0a05cbaaf35a31e09cd1` |
| `results/v13/processed/teacher_screen_v13.json` | 1306 | `a488672055244c117974936efb7c16dd50388626fbe6ba5e4a8e5e8d7970c52c` |

---

## Bundled report 1: `CAUSAL_BANK_EXPANSION_V13.md`

# Causal bank expansion — V13

V13 created a new teacher-correct capture bank; neither V11 nor V12 confirmation was used for method selection.

- usable train / validation / independent final: `{'final_test': 250, 'train': 4800, 'validation': 500}`
- family counts: `{'final_test': {'boolean_logic': 50, 'modular_arithmetic': 50, 'short_graph_traversal': 50, 'simple_state_transition': 50, 'variable_binding': 50}, 'train': {'boolean_logic': 960, 'modular_arithmetic': 960, 'short_graph_traversal': 960, 'simple_state_transition': 960, 'variable_binding': 960}, 'validation': {'boolean_logic': 100, 'modular_arithmetic': 100, 'short_graph_traversal': 100, 'simple_state_transition': 100, 'variable_binding': 100}}`
- split hashes: `{'final_test': 'cc49220b05cb3609554e1f79811d718d4425b950af6b7b9a393174810cfb2360', 'train': 'ebd4be05da86d3a3450bab336858d9e1869d1ef76640dcda7a8d76ba2f445ac3', 'validation': 'bed58343642dd8ad11d47454d126b3bd8d1d9a14478896baf3da39952d45a613'}`
- capture freeze: `167f69a3b5ddfd95775728f77e46fd94e1e432dd7097e531d8e801bc7169422b`

The bank contains exact BF16 REC, convolution, and all observed KV cache fields, J endpoints, selected-logit endpoints, semantic log-odds endpoints, prompt/token metadata, pair IDs, and explicit exclusion counts in the split manifests.

---

## Bundled report 2: `CAUSAL_PATH_DIMENSION_V13.md`

# Instantaneous rank versus writable path dimension — V13

- median instantaneous r95: `8.0`
- median cumulative path r95: `11.0`
- high path dimension under the frozen rule: `False`

The cumulative measure is the r95 of the span of the local tangent bases visited across the observed path; it is not equated with any one local Jacobian rank.

Machine rows: `10`.

---

## Bundled report 3: `CAUSAL_PROBE_SCALING_V13.md`

# Exact causal-probe scaling — V13

|       m |   median_r90 |   median_r95 |   median_r99 |   mean_stable_rank |   mean_effective_rank |
|--------:|-------------:|-------------:|-------------:|-------------------:|----------------------:|
|  64.000 |        4.000 |        5.000 |        8.000 |              2.460 |                 4.804 |
| 128.000 |        5.000 |        6.500 |       12.000 |              2.536 |                 5.496 |
| 256.000 |        5.000 |        8.000 |       16.500 |              2.582 |                 5.961 |
| 512.000 |        5.500 |        9.000 |       21.000 |              2.613 |                 6.321 |

All matrices use exact autograd JVP with Flash/memory-efficient SDP disabled. Ranks remain restricted to the frozen mixed empirical raw-state operator and are not full-state intrinsic dimensions.

---

## Bundled report 4: `CAUSAL_RANK_ROBUSTNESS_V13.md`

# Causal-rank robustness — V13

| probe family          |   median_r95 |   mean_r95 |
|:----------------------|-------------:|-----------:|
| architecture_balanced |        9.000 |      9.150 |
| causal_weighted       |        5.000 |      5.000 |
| high_variance_pca     |        7.500 |      7.900 |
| low_variance          |        8.500 |      8.200 |
| random_raw            |       11.000 |     11.000 |

Mean per-column sensitivities: `{'architecture_balanced': 4.660962577444547, 'causal_weighted': 2.3885091807515764, 'high_variance_pca': 0.7592895979669425, 'low_variance': 0.15593450524121333, 'random_raw': 0.7876877453097142}`. Low-variance/high-causal directions confirmed under the frozen rule: `False`. A single intrinsic dimension is not reported when construction-specific estimates disagree materially.

---

## Bundled report 5: `CAUSAL_TANGENT_ATLAS_V13.md`

# Local causal tangent atlas — V13

| relation                        |   mean_angle |   max_angle |   grassmann |
|:--------------------------------|-------------:|------------:|------------:|
| across_family                   |       32.494 |      87.365 |       2.285 |
| same_family                     |       24.799 |      77.533 |       1.887 |
| same_prompt_successive_position |       29.176 |      86.354 |       2.127 |

Rank-16 same-prompt successive-position mean angle: `29.18°`; across-family: `32.49°`. Spearman associations with tangent angle: `{'j_state_distance': 0.5449396540981422, 'raw_score_distance': 0.248726960359016, 'same_family_indicator': -0.36000498625317734, 'semantic_state_distance': 0.20809790507219364, 'token_distance': 0.23811847306224734}`.

---

## Bundled report 6: `CAUSAL_TARGET_COMPLETENESS_V13.md`

# Causal target completeness — V13

| target_bundle      |   r90 |   r95 |    r99 |
|:-------------------|------:|------:|-------:|
| complete_workspace | 5.500 | 9.000 | 21.000 |
| j_logits           | 6.000 | 9.000 | 21.500 |
| j_logits_semantic  | 5.000 | 8.000 | 19.000 |
| j_only             | 5.000 | 8.000 | 25.000 |

Semantic failure present: `True`; consistent with material target omission under the frozen rule: `False`. The complete bundle adds continuous semantic logits and selected intermediate workspace endpoints; rank increase is sensitivity evidence, not proof of a complete causal state.

---

## Bundled report 7: `DATA_DIMENSION_SCALING_V13.md`

# Data × dimension scaling — V13

| curve                           | N                  | h1 direction               | status       |
|:--------------------------------|:-------------------|:---------------------------|:-------------|
| architecture_resolved_pca/d1024 | 1200,2400,4800     | -0.135,-0.133,0.687        | DATA_LIMITED |
| architecture_resolved_pca/d128  | 600,1200,2400,4800 | -0.133,-0.174,-0.115,0.103 | DATA_LIMITED |
| architecture_resolved_pca/d256  | 600,1200,2400,4800 | -0.148,-0.058,-0.193,0.157 | DATA_LIMITED |
| architecture_resolved_pca/d512  | 600,1200,2400,4800 | -0.157,-0.106,-0.144,0.058 | DATA_LIMITED |
| global_causal_weighted/d1024    |                    |                            | DATA_LIMITED |
| global_causal_weighted/d128     | 600,1200,2400,4800 | -0.027,0.134,0.339,0.331   | DATA_LIMITED |
| global_causal_weighted/d256     | 600,1200,2400,4800 | -0.146,0.023,0.405,0.125   | DATA_LIMITED |
| global_causal_weighted/d512     | 600,1200,2400,4800 | 0.062,0.107,-0.029,0.149   | DATA_LIMITED |
| global_joint_pca/d1024          | 1200,2400,4800     | 0.510,0.081,0.635          | DATA_LIMITED |
| global_joint_pca/d128           | 600,1200,2400,4800 | -0.194,0.183,-0.130,0.192  | DATA_LIMITED |
| global_joint_pca/d256           | 600,1200,2400,4800 | 0.055,-0.141,-0.172,-0.038 | DATA_LIMITED |
| global_joint_pca/d512           | 600,1200,2400,4800 | 0.373,-0.136,-0.181,-0.011 | DATA_LIMITED |
| local_causal_basis/d1024        |                    |                            | DATA_LIMITED |
| local_causal_basis/d128         | 600,1200,2400,4800 | -0.027,0.134,0.348,0.749   | DATA_LIMITED |
| local_causal_basis/d256         | 600,1200,2400,4800 | -0.146,0.023,0.405,0.534   | DATA_LIMITED |
| local_causal_basis/d512         | 600,1200,2400,4800 | 0.062,0.107,-0.029,0.652   | DATA_LIMITED |

## Fixed 512D h1 metrics

| method                    |   train_size |   direction_h1 |   magnitude_h1 |   semantic_cosine_h1 |   output_direction_h1 |   semantic_sign_h1 |
|:--------------------------|-------------:|---------------:|---------------:|---------------------:|----------------------:|-------------------:|
| global_joint_pca          |          600 |          0.373 |          0.362 |                0.239 |                 0.068 |              0.300 |
| architecture_resolved_pca |          600 |         -0.157 |          0.357 |                0.147 |                 0.064 |              0.400 |
| global_causal_weighted    |          600 |          0.062 |          0.188 |                0.330 |                 0.018 |              0.300 |
| local_causal_basis        |          600 |          0.062 |          0.188 |                0.330 |                 0.018 |              0.300 |
| global_joint_pca          |         1200 |         -0.136 |          0.347 |                0.326 |                 0.063 |              0.400 |
| architecture_resolved_pca |         1200 |         -0.106 |          0.241 |                0.059 |                 0.015 |              0.400 |
| global_causal_weighted    |         1200 |          0.107 |          0.162 |                0.266 |                 0.066 |              0.400 |
| local_causal_basis        |         1200 |          0.107 |          0.162 |                0.266 |                 0.066 |              0.400 |
| global_joint_pca          |         2400 |         -0.181 |          0.500 |                0.309 |                 0.039 |              0.400 |
| architecture_resolved_pca |         2400 |         -0.144 |          0.494 |                0.262 |                 0.057 |              0.400 |
| global_causal_weighted    |         2400 |         -0.029 |          0.127 |               -0.325 |                 0.074 |              0.100 |
| local_causal_basis        |         2400 |         -0.029 |          0.127 |               -0.325 |                 0.074 |              0.100 |
| global_joint_pca          |         4800 |         -0.011 |          0.387 |                0.543 |                 0.037 |              0.400 |
| architecture_resolved_pca |         4800 |          0.058 |          0.331 |                0.394 |                 0.042 |              0.300 |
| global_causal_weighted    |         4800 |          0.149 |          0.203 |                0.376 |                 0.113 |              0.300 |
| local_causal_basis        |         4800 |          0.652 |          0.390 |                0.740 |                 0.125 |              0.600 |

The practical saturation rule was frozen at two consecutive doublings with absolute improvement below `0.01`. `NOT_IDENTIFIED_RANK_LIMIT` rows were retained in the parquet record; no rank was silently clamped. Local method: `family-stratified nearest-J neighborhood centered on one frozen validation representative per family; shared within family`. These curves are frozen held-out endpoint-fidelity estimates; finite writeback is adjudicated separately by the causal oracle.

---

## Bundled report 8: `EXECUTION_MANIFEST_V13.md`

# V13 execution manifest

## Scope and provenance

V13 was executed in `/data/CSK/J-space-project/jstate-closure` on server
`222.20.126.223`, starting from commit
`2b94a65daee29e5547530261194b7de34860df4e`. V1–V12 frozen files were treated
as immutable inputs. The pretrained model, dtype, estimands, split membership,
probe operator, targets, and gates were not changed after their corresponding
freezes.

## Canonical commands

The following commands are the canonical successful execution path (the complete
reproduction wrapper is `scripts/run_causal_geometry_v13_amended.sh`):

```bash
export HF_HOME=/data/CSK/J-space-project/.hf-cache
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)/src"
PYTHON_BIN=/home/user/anaconda3/bin/python

$PYTHON_BIN -m jclosure.experiments.bank_v13 --stage generate --run-suffix generate-independent
$PYTHON_BIN -m jclosure.experiments.bank_v13 --stage teacher --run-suffix teacher-independent-b16
$PYTHON_BIN -m jclosure.experiments.bank_v13 --stage generate --run-suffix regen-modular-single-digit
$PYTHON_BIN -m jclosure.experiments.bank_v13 --stage teacher --family modular_arithmetic --run-suffix teacher-modular-single-digit-b16
$PYTHON_BIN -m jclosure.experiments.bank_v13 --stage select --run-suffix select
$PYTHON_BIN -m jclosure.experiments.bank_v13 --stage guard --run-suffix guard
$PYTHON_BIN -m jclosure.experiments.bank_v13 --stage freeze --run-suffix freeze
$PYTHON_BIN -m jclosure.experiments.bank_v13 --stage capture --split train --run-suffix capture-train
$PYTHON_BIN -m jclosure.experiments.bank_v13 --stage capture --split validation --run-suffix capture-validation
$PYTHON_BIN -m jclosure.experiments.bank_v13 --stage capture --split final_test --run-suffix capture-final
$PYTHON_BIN -m jclosure.experiments.bank_v13 --stage freeze-capture --run-suffix freeze-capture

MALLOC_ARENA_MAX=2 CUDA_VISIBLE_DEVICES=1 $PYTHON_BIN -m jclosure.experiments.runtime_v13_features_v15 --target geometry --stage features --run-suffix features-memory-amendment-15-gpu1
CUDA_VISIBLE_DEVICES=0 $PYTHON_BIN -m jclosure.experiments.geometry_v13 --stage scaling --run-suffix scaling-gpu0
MALLOC_ARENA_MAX=2 CUDA_VISIBLE_DEVICES=1 $PYTHON_BIN -m jclosure.experiments.runtime_v13_features_v15 --target geometry --stage directions --run-suffix directions-gpu1
$PYTHON_BIN -m jclosure.experiments.geometry_v13 --stage freeze-jvp --run-suffix freeze-jvp
$PYTHON_BIN -m jclosure.experiments.runtime_v13 --target freeze
$PYTHON_BIN -m jclosure.experiments.runtime_v13_jvp_scalar_sharded --freeze-amendment

MALLOC_ARENA_MAX=2 CUDA_VISIBLE_DEVICES=0,1 $PYTHON_BIN -m jclosure.experiments.runtime_v13_jvp_scalar_sharded --stage jvp --run-suffix exact-scalar-shard-0 --shard-index 0 &
JVP0=$!
MALLOC_ARENA_MAX=2 CUDA_VISIBLE_DEVICES=0,1 $PYTHON_BIN -m jclosure.experiments.runtime_v13_jvp_scalar_sharded --stage jvp --run-suffix exact-scalar-shard-1 --shard-index 1 &
JVP1=$!
wait "$JVP0"; wait "$JVP1"
$PYTHON_BIN scripts/merge_v13_jvp_scalar_shards.py

MALLOC_ARENA_MAX=2 CUDA_VISIBLE_DEVICES=0,1 $PYTHON_BIN -m jclosure.experiments.runtime_v13 --target geometry --stage oracle-development --run-suffix oracle-development-v13
$PYTHON_BIN -m jclosure.experiments.geometry_v13 --stage freeze-finalists --run-suffix freeze-finalists-v13
MALLOC_ARENA_MAX=2 CUDA_VISIBLE_DEVICES=0,1 $PYTHON_BIN -m jclosure.experiments.runtime_v13 --target geometry --stage linearity --run-suffix linearity-v13
MALLOC_ARENA_MAX=2 CUDA_VISIBLE_DEVICES=0,1 $PYTHON_BIN -m jclosure.experiments.runtime_v13 --target geometry --stage oracle-confirmatory --run-suffix oracle-confirmatory-v13
$PYTHON_BIN -m jclosure.experiments.geometry_v13 --stage analyze --run-suffix analyze-v13
$PYTHON_BIN -m jclosure.reporting_v13
$PYTHON_BIN scripts/build_complete_version_report.py V13
```

## Primary freeze hashes

| freeze | SHA256 digest |
|---|---|
| V13 base protocol | `98066eedffd1255db439f99f0689aa26bb91659f06242f796658bf9b8324ba85` |
| capture bank | `167f69a3b5ddfd95775728f77e46fd94e1e432dd7097e531d8e801bc7169422b` |
| JVP inputs | `17b5b0ec8743d99a4fa5cbe91a7f5458fd97087f02c4d4c1991e57d84c61dd38` |
| dual-GPU runtime | `8aef08468ba1b5bf87785a5dff9fd252562c4cb7d749c2a3072d0e786657e6cb` |
| scalar JVP sharding | `5eac0070a1dfaaaaa52c41d539e7c476dcaa3307434710a20ba4ee6e942d82ba` |
| frozen finalist | `0f6c5531f5f288efe09fc26204adc0072c56bbbe3a7595729491ad3ff76a9ede` |

Split ID hashes are train
`6ecb1bb3867fdbe3ec7ff5494ba88204120a79035a694e2df7c7b4ad45b2e557`,
validation
`2bce67f2807608773b7176477e4509bb252e2d09816290160b49d82728810974`,
and final test
`8c8d0cc187b165f4990ea6620f344f3e6b19856d0e456017a638cc63e7364800`.
Capture split hashes are train
`ebd4be05da86d3a3450bab336858d9e1869d1ef76640dcda7a8d76ba2f445ac3`,
validation
`bed58343642dd8ad11d47454d126b3bd8d1d9a14478896baf3da39952d45a613`,
and final test
`cc49220b05cb3609554e1f79811d718d4425b950af6b7b9a393174810cfb2360`.

## Bank and feature records

- Usable bank: train `4800` (`960` per family), validation `500` (`100` per
  family), independent final `250` (`50` per family).
- Frozen train sizes: `600, 1200, 2400, 4800`.
- Combined feature dimension: `4799`; architecture-resolved dimension: `14397`.
- Feature artifact SHA256:
  `511623733a70354484db45b85586dfa9e1fab533d238aa92aa67fb8e1f4f02b5`.
- Probe-direction artifact SHA256:
  `77aeccb5dccaef52c444d94b65797a18ce2984c76b6655e8a90714976cf51891`.

## Data × dimension scaling

No d512 method satisfied the frozen two-doubling saturation criterion; every
d512 curve is recorded as `DATA_LIMITED`. The h1 direction curves for
N=`600,1200,2400,4800` are:

| method | d512 h1 direction curve |
|---|---|
| global joint PCA | `0.3728, -0.1358, -0.1808, -0.0112` |
| architecture-resolved PCA | `-0.1573, -0.1063, -0.1443, 0.0578` |
| global causal-weighted | `0.0623, 0.1069, -0.0286, 0.1493` |
| local causal basis | `0.0623, 0.1069, -0.0286, 0.6515` |

The local d512 final doubling is positive, but the overall scaling is unstable;
this does not establish a saturated compact state. The formal V13-B condition
was not selected because the global causal d512 final improvement CI includes
zero (`[-0.0858, 0.4687]`). Unsupported dimensions are explicitly recorded as
`NOT_IDENTIFIED_RANK_LIMIT`, never silently clamped.

## Exact scalar JVP probe scaling

Twenty local states were evaluated with exact scalar autograd JVP, 512 frozen
directions per state, and no direction batching. The canonical record SHA256 is
`da45a7ec022c0ec2cfad43e47123cd595dd00de92bc57bcf56d19ea4487152af`.

| probe m | median r90 | median r95 | median r99 | mean stable rank | mean effective rank |
|---:|---:|---:|---:|---:|---:|
| 64 | 4.0 | 5.0 | 8.0 | 2.460 | 4.804 |
| 128 | 5.0 | 6.5 | 12.0 | 2.536 | 5.496 |
| 256 | 5.0 | 8.0 | 16.5 | 2.582 | 5.961 |
| 512 | 5.5 | 9.0 | 21.0 | 2.613 | 6.321 |

This supports `STABLE_LOW_LOCAL_CAUSAL_RANK` within the frozen mixed probe
operator; it does not claim a full raw-state intrinsic dimension.

Construction-specific median r95 values are causal-weighted `5.0`, high-variance
PCA `7.5`, low-variance `8.5`, architecture-balanced `9.0`, and random `11.0`.
Therefore a single construction-independent intrinsic dimension is not reported.
The V13 frozen rule did not re-confirm low-variance/high-causal directions.

## Tangent, path, target, and channel geometry

- Rank-16 same-prompt successive-token mean angle: `29.176°`.
- Rank-16 across-family mean angle: `32.494°`.
- Strongest measured tangent-angle predictor: J-state distance, Spearman
  `0.545`.
- Median instantaneous r95: `8`; median cumulative path r95: `11`.
  `high_path_dimension=False` under the frozen >2× rule.
- Target-bundle median r95: J only `8`, J+logits `9`,
  J+logits+semantic `8`, complete workspace `9`. Semantic failure is not
  consistent with target omission under the frozen test.
- Cross-channel mixed-direction fraction: `0.6016`; dominant directions are
  jointly REC/conv/KV, so naive channel factorization is not supported.
- Same-prompt channel rotation: recurrent `25.40°`, conv `21.93°`, KV `36.89°`.

## Static versus moving oracle and independent confirmation

Validation selected and froze the interpolated moving tangent with `alpha=1.0`.
On the independent 25-case final bank:

| method | horizon | direction | magnitude | semantic continuous | semantic legacy | output | sign |
|---|---:|---:|---:|---:|---:|---:|---:|
| static local | 1 | 0.739 | 0.825 | 0.739 | 0.324 | 0.757 | 0.560 |
| moving interpolated | 1 | 0.772 | 0.871 | 0.772 | 0.356 | 0.797 | 0.760 |
| static local | 2 | 0.645 | 0.874 | 0.645 | 0.128 | 0.656 | 0.800 |
| moving interpolated | 2 | 0.689 | 0.897 | 0.689 | 0.192 | 0.669 | 0.680 |
| static local | 4 | 0.578 | 0.966 | 0.578 | 0.104 | 0.551 | 0.708 |
| moving interpolated | 4 | 0.590 | 0.994 | 0.590 | 0.148 | 0.567 | 0.792 |
| static local | 8 | 0.500 | 0.984 | 0.500 | 0.124 | 0.534 | 0.600 |
| moving interpolated | 8 | 0.513 | 0.965 | 0.513 | 0.124 | 0.546 | 0.640 |

Moving tangent improves static on the frozen aggregate rule, but it does not
pass the independent h1/h2/h4/h8 causal gates. The practical local linear
radius is `None`; no tested alpha met both pooled h1 J-direction and
output-direction thresholds of `0.8`.

## Numerical execution audit

An attempted independent-direction batch implementation was frozen and checked
against a complete scalar reference matrix. It was rejected, not used for the
canonical results: batch 32 had relative Frobenius error `0.01758`, minimum
column cosine `0.99855`, and changed m=256 r95 from `7` to `8`. The formal JVP
records above were consequently generated with scalar direction batch size 1.
The scalar sharded recomputation of the original first matrix was byte-identical
(same SHA256), establishing that execution sharding did not alter the operator.

## Formal adjudication

- Formal outcome: `V13-F — HIGH-DIMENSIONAL WRITABLE STATE` within the tested
  protocol.
- Stable low instantaneous local causal rank: `True`.
- Independently validated compact causal-sufficient state: `False`.
- Smallest independently validated writable dimension: `None`.
- Complete replacement state: `False`.
- Strict replacement status:
  `ABSOLUTE_REPLACEMENT_NOT_YET_TESTABLE_FROM_CURRENT_DELTA_REPRESENTATION`.
- `H2 remains = True`; `H3 authorized = False`.
- `AUTONOMOUS_CONTROLLER_AUTHORIZED = FALSE`.

---

## Bundled report 9: `JOINT_CHANNEL_CAUSAL_GEOMETRY_V13.md`

# Joint REC / convolution / KV causal geometry — V13

- fraction of frozen probe directions with >10% score energy in at least two channels: `0.602`
- dominant directions judged cross-channel: `True`
- same-prompt tangent rotation by dominant channel: `{'conv': 21.926273423433305, 'kv': 36.89069223999977, 'recurrent': 25.403882221877573}`

This joint loading explains why independently factorized variance PCA can discard low-variance causal combinations spanning REC, convolution, and KV fields.

---

## Bundled report 10: `LOCAL_LINEARITY_RADIUS_V13.md`

# Finite perturbation linearity radius — V13

Practical local linear radius: `None`. Criterion: largest alpha with pooled h1 J and output direction >= 0.8.

|   scale |   horizon |   j_direction |   output_direction |   j_relative_error |   output_relative_error |   tangent_drift |
|--------:|----------:|--------------:|-------------------:|-------------------:|------------------------:|----------------:|
|   0.050 |     1.000 |         0.119 |             -0.048 |              1.021 |                   1.010 |          10.372 |
|   0.050 |     2.000 |         0.011 |              0.036 |              0.999 |                   1.005 |          10.372 |
|   0.050 |     4.000 |        -0.019 |              0.266 |              1.001 |                   0.993 |          10.372 |
|   0.100 |     1.000 |         0.348 |              0.047 |              0.959 |                   1.018 |          10.372 |
|   0.100 |     2.000 |         0.014 |             -0.016 |              1.003 |                   1.018 |          10.372 |
|   0.100 |     4.000 |         0.034 |              0.144 |              0.997 |                   0.992 |          10.372 |
|   0.250 |     1.000 |         0.618 |              0.368 |              0.860 |                   0.929 |          10.372 |
|   0.250 |     2.000 |         0.112 |              0.127 |              1.002 |                   1.012 |          10.372 |
|   0.250 |     4.000 |         0.123 |              0.230 |              0.996 |                   0.988 |          10.372 |
|   0.500 |     1.000 |         0.693 |              0.523 |              0.786 |                   0.856 |           7.441 |
|   0.500 |     2.000 |         0.293 |              0.338 |              0.958 |                   0.923 |           7.441 |
|   0.500 |     4.000 |         0.151 |              0.236 |              1.008 |                   0.999 |           7.441 |
|   0.750 |     1.000 |         0.736 |              0.582 |              0.778 |                   0.780 |           6.628 |
|   0.750 |     2.000 |         0.330 |              0.433 |              0.935 |                   0.905 |           6.628 |
|   0.750 |     4.000 |         0.206 |              0.354 |              0.987 |                   0.953 |           6.628 |
|   1.000 |     1.000 |         0.726 |              0.614 |              0.802 |                   0.749 |           0.026 |
|   1.000 |     2.000 |         0.358 |              0.385 |              0.953 |                   0.965 |           0.026 |
|   1.000 |     4.000 |         0.288 |              0.332 |              0.960 |                   1.013 |           0.026 |

---

## Bundled report 11: `MOVING_TANGENT_CAUSAL_ORACLE_V13.md`

# Static and moving-tangent causal oracle — V13

| method                             |   horizon |   direction |   magnitude |   semantic_continuous |   semantic_legacy |   output |   sign | gate_pass   |
|:-----------------------------------|----------:|------------:|------------:|----------------------:|------------------:|---------:|-------:|:------------|
| global_causal_basis_oracle         |         1 |       0.097 |       6.111 |                 0.097 |             0.032 |    0.133 |  0.520 | False       |
| global_causal_basis_oracle         |         2 |       0.102 |       5.522 |                 0.102 |             0.032 |    0.154 |  0.600 | False       |
| global_causal_basis_oracle         |         4 |       0.162 |       3.909 |                 0.162 |             0.012 |    0.231 |  0.625 | False       |
| global_causal_basis_oracle         |         8 |       0.279 |       2.128 |                 0.279 |             0.012 |    0.373 |  0.680 | False       |
| global_pca_oracle                  |         1 |       0.256 |       0.461 |                 0.256 |             0.076 |    0.296 |  0.760 | False       |
| global_pca_oracle                  |         2 |       0.356 |       0.644 |                 0.356 |             0.056 |    0.352 |  0.480 | False       |
| global_pca_oracle                  |         4 |       0.418 |       0.858 |                 0.418 |             0.080 |    0.401 |  0.583 | False       |
| global_pca_oracle                  |         8 |       0.494 |       0.957 |                 0.494 |             0.084 |    0.526 |  0.560 | False       |
| moving_tangent_interpolated_oracle |         1 |       0.772 |       0.871 |                 0.772 |             0.356 |    0.797 |  0.760 | False       |
| moving_tangent_interpolated_oracle |         2 |       0.689 |       0.897 |                 0.689 |             0.192 |    0.669 |  0.680 | False       |
| moving_tangent_interpolated_oracle |         4 |       0.590 |       0.994 |                 0.590 |             0.148 |    0.567 |  0.792 | False       |
| moving_tangent_interpolated_oracle |         8 |       0.513 |       0.965 |                 0.513 |             0.124 |    0.546 |  0.640 | False       |
| raw_teacher_intervention           |         1 |       1.000 |       1.000 |                 1.000 |             1.000 |    1.000 |  1.000 | True        |
| raw_teacher_intervention           |         2 |       1.000 |       1.000 |                 1.000 |             1.000 |    1.000 |  1.000 | True        |
| raw_teacher_intervention           |         4 |       1.000 |       1.000 |                 1.000 |             1.000 |    1.000 |  1.000 | True        |
| raw_teacher_intervention           |         8 |       1.000 |       1.000 |                 1.000 |             1.000 |    1.000 |  1.000 | True        |
| static_local_causal_oracle         |         1 |       0.739 |       0.825 |                 0.739 |             0.324 |    0.757 |  0.560 | False       |
| static_local_causal_oracle         |         2 |       0.645 |       0.874 |                 0.645 |             0.128 |    0.656 |  0.800 | False       |
| static_local_causal_oracle         |         4 |       0.578 |       0.966 |                 0.578 |             0.104 |    0.551 |  0.708 | False       |
| static_local_causal_oracle         |         8 |       0.500 |       0.984 |                 0.500 |             0.124 |    0.534 |  0.600 | False       |

Frozen moving method / alpha: `moving_tangent_interpolated_oracle` / `1.0`. Development panel: `10`; independent final panel: `25`. Moving tangent improves static: `True`. All h1/h2/h4/h8 gates pass: `False`.

---

## Bundled report 12: `STRICT_STATE_REPLACEMENT_V13.md`

# Strict state replacement — V13

Status: **ABSOLUTE_REPLACEMENT_NOT_YET_TESTABLE_FROM_CURRENT_DELTA_REPRESENTATION**.

V13 representations and tangent oracles encode causal edits relative to a clean cache. They do not encode an absolute complete cache and therefore do not authorize removal of the raw scaffold. No replacement success is claimed.
