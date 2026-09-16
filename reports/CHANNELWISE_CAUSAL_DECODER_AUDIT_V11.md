# Channel-wise causal decoder audit — V11

This is a frozen development-bank diagnosis. Teacher raw channels are retained except
where the condition name says `decoded`; all trajectories are teacher-forced.

| method/condition | h | direction | magnitude | semantic | output | sign | gate |
|---|---:|---:|---:|---:|---:|---:|:---:|
| decoded_all | 1 | 0.923 | 0.977 | 0.614 | 0.896 | 0.820 | FAIL |
| decoded_all | 2 | 0.823 | 0.980 | 0.352 | 0.795 | 0.740 | FAIL |
| decoded_all | 4 | 0.688 | 0.987 | 0.206 | 0.660 | 0.696 | FAIL |
| decoded_all | 8 | 0.532 | 0.998 | 0.086 | 0.528 | 0.660 | FAIL |
| decoded_conv | 1 | 0.934 | 0.984 | 0.672 | 0.904 | 0.900 | FAIL |
| decoded_conv | 2 | 0.837 | 0.998 | 0.366 | 0.806 | 0.760 | FAIL |
| decoded_conv | 4 | 0.705 | 0.989 | 0.242 | 0.676 | 0.761 | FAIL |
| decoded_conv | 8 | 0.544 | 1.010 | 0.110 | 0.512 | 0.740 | FAIL |
| decoded_conv_kv | 1 | 0.932 | 0.978 | 0.656 | 0.907 | 0.860 | FAIL |
| decoded_conv_kv | 2 | 0.840 | 0.998 | 0.384 | 0.805 | 0.740 | FAIL |
| decoded_conv_kv | 4 | 0.703 | 0.991 | 0.228 | 0.671 | 0.761 | FAIL |
| decoded_conv_kv | 8 | 0.554 | 1.011 | 0.132 | 0.544 | 0.680 | FAIL |
| decoded_kv | 1 | 0.970 | 0.995 | 0.760 | 0.945 | 0.820 | FAIL |
| decoded_kv | 2 | 0.911 | 0.997 | 0.500 | 0.874 | 0.800 | FAIL |
| decoded_kv | 4 | 0.861 | 0.989 | 0.406 | 0.807 | 0.783 | FAIL |
| decoded_kv | 8 | 0.770 | 1.010 | 0.266 | 0.660 | 0.740 | FAIL |
| decoded_rec | 1 | 0.942 | 0.997 | 0.674 | 0.913 | 0.940 | FAIL |
| decoded_rec | 2 | 0.833 | 0.995 | 0.382 | 0.818 | 0.740 | FAIL |
| decoded_rec | 4 | 0.697 | 0.993 | 0.222 | 0.628 | 0.804 | FAIL |
| decoded_rec | 8 | 0.530 | 1.003 | 0.096 | 0.527 | 0.700 | FAIL |
| decoded_rec_conv | 1 | 0.923 | 0.979 | 0.624 | 0.900 | 0.860 | FAIL |
| decoded_rec_conv | 2 | 0.826 | 0.991 | 0.352 | 0.802 | 0.760 | FAIL |
| decoded_rec_conv | 4 | 0.691 | 0.987 | 0.212 | 0.652 | 0.565 | FAIL |
| decoded_rec_conv | 8 | 0.529 | 1.004 | 0.072 | 0.519 | 0.620 | FAIL |
| decoded_rec_kv | 1 | 0.941 | 1.000 | 0.682 | 0.914 | 0.940 | FAIL |
| decoded_rec_kv | 2 | 0.832 | 0.998 | 0.404 | 0.814 | 0.800 | FAIL |
| decoded_rec_kv | 4 | 0.697 | 0.995 | 0.212 | 0.655 | 0.696 | FAIL |
| decoded_rec_kv | 8 | 0.527 | 1.006 | 0.080 | 0.517 | 0.720 | FAIL |
| teacher_reference | 1 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | PASS |
| teacher_reference | 2 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | PASS |
| teacher_reference | 4 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | PASS |
| teacher_reference | 8 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | PASS |

The weakest single decoded channel at h1 is **decoded_conv**. The weakest overall decoded
combination at h1 is **decoded_all**. Non-additivity is assessed in the amplification machine
record rather than inferred from marginal reconstruction error.

Machine records: `results/v11/processed/channelwise_causal_v11.parquet` (`f46ab7afc99287e6106ba6eb5ce032632d656d8e163a813b17bd583602b2063e`).


<!-- V11_MANIFOLD_AMENDMENT_1 -->
## Channel nonadditivity amendment 1

The retained records do not contain full error vectors, so exact cross terms cannot be
recovered. The frozen descriptive proxy compares observed joint error norm with the
root-sum-square of single-channel error norms. Ratio >1 is compatible with constructive
interaction; ratio <1 with cancellation.

| decoded channels | h | joint error | single RSS | joint/RSS |
|---|---:|---:|---:|---:|
| all | 1 | 0.0089 | 0.0126 | 0.706 |
| all | 2 | 0.0065 | 0.0102 | 0.643 |
| all | 4 | 0.0120 | 0.0185 | 0.649 |
| all | 8 | 0.0072 | 0.0114 | 0.633 |
| conv_kv | 1 | 0.0084 | 0.0100 | 0.843 |
| conv_kv | 2 | 0.0063 | 0.0079 | 0.797 |
| conv_kv | 4 | 0.0117 | 0.0142 | 0.822 |
| conv_kv | 8 | 0.0071 | 0.0088 | 0.807 |
| rec_conv | 1 | 0.0089 | 0.0113 | 0.785 |
| rec_conv | 2 | 0.0065 | 0.0090 | 0.721 |
| rec_conv | 4 | 0.0120 | 0.0167 | 0.721 |
| rec_conv | 8 | 0.0072 | 0.0102 | 0.713 |
| rec_kv | 1 | 0.0076 | 0.0094 | 0.816 |
| rec_kv | 2 | 0.0064 | 0.0079 | 0.813 |
| rec_kv | 4 | 0.0119 | 0.0144 | 0.830 |
| rec_kv | 8 | 0.0073 | 0.0089 | 0.821 |

Amendment freeze: `f17f62532f1136c9cc73109ecf55e2f6689b55638313979eb9c1d5068ac5c14f`. Machine records: `results/v11/processed/channel_interaction_nonadditivity_v11.parquet` (`c8014adb7c99d2c0d04fa451d671c5828c741f81cde78c6538564a16148c3a13`).
