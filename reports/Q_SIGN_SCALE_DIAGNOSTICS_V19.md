# V19 q sign, scale and odd/even diagnostics

This is a frozen nested diagnostic, not a linearity premise or replacement for the primary +q bank. Both ± signs and α=0.5/1.0 use the same clean teacher sequence per base state; all unreliable BF16 rows remain in the denominator.

| q | h | median M/R α=.5 | median M/R α=1 | N even/odd | M even/odd | q reliable |
|---|---|---|---|---|---|---|
| conv_causal1 | h1 | 0.7619 | 1.0895 | 0.7187 | 0.8781 | 1.0000 |
| conv_causal1 | h2 | 0.8023 | 1.1895 | 0.7551 | 0.7216 | 1.0000 |
| conv_causal1 | h4 | 0.7326 | 1.0498 | 1.1574 | 1.1357 | 1.0000 |
| conv_causal1 | h8 | 0.7191 | 1.0996 | 1.2770 | 1.4446 | 1.0000 |
| joint_causal11 | h1 | 0.3923 | 0.6333 | 0.5012 | 0.5806 | 1.0000 |
| joint_causal11 | h2 | 0.4954 | 0.7697 | 0.3789 | 0.5661 | 1.0000 |
| joint_causal11 | h4 | 0.5262 | 0.7297 | 0.4463 | 0.6612 | 1.0000 |
| joint_causal11 | h8 | 0.5419 | 0.7601 | 0.7520 | 1.0284 | 1.0000 |
| kv_arch14_amended | h1 | 0.0297 | 0.0301 | 1.1612 | 1.2441 | 1.0000 |
| kv_arch14_amended | h2 | 0.0630 | 0.0486 | 1.1545 | 1.3976 | 1.0000 |
| kv_arch14_amended | h4 | 0.1653 | 0.1906 | 1.7794 | 1.5648 | 1.0000 |
| kv_arch14_amended | h8 | 0.4090 | 0.4225 | 1.0897 | 1.2112 | 1.0000 |
| rec_arch4_amended | h1 | 0.5852 | 0.8085 | 0.7270 | 0.6797 | 1.0000 |
| rec_arch4_amended | h2 | 0.4367 | 0.7381 | 0.5973 | 1.0852 | 1.0000 |
| rec_arch4_amended | h4 | 0.7665 | 0.9838 | 0.5272 | 0.9791 | 1.0000 |
| rec_arch4_amended | h8 | 0.6968 | 0.8490 | 0.2722 | 0.4888 | 1.0000 |
| rec_conv_causal1_amended | h1 | 0.8080 | 1.0406 | 0.9783 | 0.7202 | 1.0000 |
| rec_conv_causal1_amended | h2 | 0.8486 | 1.2751 | 0.9072 | 0.8615 | 1.0000 |
| rec_conv_causal1_amended | h4 | 0.8020 | 1.1601 | 0.3477 | 1.2382 | 1.0000 |
| rec_conv_causal1_amended | h8 | 0.7170 | 1.1689 | 0.7423 | 1.3965 | 1.0000 |

The even/odd ratios are computed from paired ±q vectors for the same base state, q and α. Variation is descriptive; no post-hoc sign/scale choice changes the frozen primary q bank.
