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
