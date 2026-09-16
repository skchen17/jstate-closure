# Decoded Causal State Validation v10

<!-- V10-AMENDMENT:START -->
## Authorization conclusion and strict three-way audit

Causal metadata amendment freeze: `cb0df76882a701452f3973782b44aaa0ea5a0984881d81465c5e1037ecdef3a5`; strict-interface freeze: `4b319ebffda82f7bcd155df70bbd7d0490dab9bea7fc92e8c1387ecb453d48c8`. All 50 selected pairs map to the frozen v8 task table.

Decoder full-rank feature reconstruction max error: `7.153e-07`. The compact-only function hash is `dfca86026a9d100d7f5ec01f4bc54ba81404385664ea6d91685f49dad295e679` and its allowed inputs are only clean cache, decoded delta, and teacher-forced token.

The diagnostic `decoded + (raw-decoded)` reconstruction matches full raw state to max absolute error `4.657e-10`. It intentionally violates compact-only access and is not an authorization path; it confirms that the missing raw residual is sufficient to restore the teacher reference.

No candidate passes. Magnitude is well calibrated (~0.97–1.01), but h1 semantic top-10 effect agreement is only 0.610–0.642 (threshold 0.8). Direction/output fidelity then decays strongly: for 512D, direction is 0.922/0.825/0.685/0.527 at h1/2/4/8 and output direction is 0.897/0.789/0.647/0.515.

Increasing 384→512D does not materially improve causal fidelity. This points to the encoder/decoder objective and block/channel interaction, with semantic-effect loss and long-horizon instability, rather than dimension alone. Every family fails, so task heterogeneity is not the sole explanation.
<!-- V10-AMENDMENT:END -->


Strict interface audit: `{'allowed_inputs': ['clean_cache', 'decoded_delta', 'teacher_forced_token'], 'function_sha256': 'dfca86026a9d100d7f5ec01f4bc54ba81404385664ea6d91685f49dad295e679', 'raw_test_cache_available_to_decoded_continuation': False, 'verified': True}`.

|   dimension |   horizon | group           |   count | direction               | magnitude               | semantic_delta          | output_direction        | task_sign               |   effect_weighted_direction | gate_pass   |
|------------:|----------:|:----------------|--------:|:------------------------|:------------------------|:------------------------|:------------------------|:------------------------|----------------------------:|:------------|
|         384 |         1 | effect_enriched |      25 | 0.9269 [0.9158, 0.9371] | 0.9731 [0.9501, 0.9992] | 0.5960 [0.5240, 0.6680] | 0.8870 [0.8592, 0.9110] | 0.8800 [0.7200, 1.0000] |                    0.931845 | False       |
|         384 |         1 | pooled          |      50 | 0.9193 [0.9036, 0.9327] | 0.9760 [0.9588, 0.9944] | 0.6240 [0.5660, 0.6820] | 0.8918 [0.8740, 0.9081] | 0.8400 [0.7400, 0.9400] |                    0.929212 | False       |
|         384 |        16 | effect_enriched |      25 | 0.5503 [0.5264, 0.5741] | 0.9998 [0.9702, 1.0286] | 0.1360 [0.0880, 0.1920] | 0.5082 [0.4409, 0.5716] | 0.6087 [0.3913, 0.7826] |                    0.555407 | False       |
|         384 |        16 | pooled          |      50 | 0.5236 [0.5010, 0.5460] | 0.9856 [0.9656, 1.0066] | 0.1180 [0.0840, 0.1560] | 0.5246 [0.4785, 0.5683] | 0.6170 [0.4681, 0.7447] |                    0.532488 | False       |
|         384 |         2 | effect_enriched |      25 | 0.8428 [0.8172, 0.8660] | 0.9930 [0.9720, 1.0153] | 0.4000 [0.3080, 0.4920] | 0.7974 [0.7545, 0.8342] | 0.8800 [0.7600, 1.0000] |                    0.851174 | False       |
|         384 |         2 | pooled          |      50 | 0.8198 [0.7965, 0.8406] | 0.9894 [0.9710, 1.0086] | 0.3260 [0.2700, 0.3860] | 0.7808 [0.7490, 0.8097] | 0.7600 [0.6400, 0.8800] |                    0.831163 | False       |
|         384 |         4 | effect_enriched |      25 | 0.7047 [0.6559, 0.7541] | 0.9922 [0.9668, 1.0184] | 0.2160 [0.1640, 0.2680] | 0.6794 [0.6188, 0.7372] | 0.8182 [0.6364, 0.9545] |                    0.703347 | False       |
|         384 |         4 | pooled          |      50 | 0.6823 [0.6512, 0.7139] | 0.9970 [0.9806, 1.0132] | 0.2140 [0.1760, 0.2560] | 0.6190 [0.5630, 0.6741] | 0.8043 [0.6957, 0.9130] |                    0.681284 | False       |
|         384 |         8 | effect_enriched |      25 | 0.5450 [0.5207, 0.5688] | 1.0157 [0.9892, 1.0410] | 0.1000 [0.0600, 0.1480] | 0.5351 [0.4740, 0.5915] | 0.5600 [0.3600, 0.7600] |                    0.546733 | False       |
|         384 |         8 | pooled          |      50 | 0.5305 [0.5132, 0.5480] | 1.0094 [0.9917, 1.0270] | 0.1020 [0.0740, 0.1340] | 0.5224 [0.4834, 0.5593] | 0.5600 [0.4200, 0.7000] |                    0.533069 | False       |
|         448 |         1 | effect_enriched |      25 | 0.9287 [0.9181, 0.9384] | 0.9804 [0.9571, 1.0084] | 0.5880 [0.5120, 0.6680] | 0.8984 [0.8763, 0.9184] | 0.8400 [0.6800, 0.9600] |                    0.933502 | False       |
|         448 |         1 | pooled          |      50 | 0.9227 [0.9093, 0.9342] | 0.9722 [0.9562, 0.9900] | 0.6420 [0.5840, 0.7020] | 0.8972 [0.8799, 0.9132] | 0.8000 [0.6800, 0.9000] |                    0.93159  | False       |
|         448 |        16 | effect_enriched |      25 | 0.5389 [0.5101, 0.5685] | 1.0037 [0.9725, 1.0361] | 0.0920 [0.0520, 0.1440] | 0.5025 [0.4340, 0.5638] | 0.6522 [0.4348, 0.8261] |                    0.541321 | False       |
|         448 |        16 | pooled          |      50 | 0.5345 [0.5107, 0.5578] | 1.0073 [0.9866, 1.0282] | 0.1060 [0.0760, 0.1400] | 0.5513 [0.5091, 0.5901] | 0.6809 [0.5319, 0.8085] |                    0.538953 | False       |
|         448 |         2 | effect_enriched |      25 | 0.8462 [0.8242, 0.8667] | 0.9834 [0.9672, 1.0008] | 0.4600 [0.3720, 0.5560] | 0.8262 [0.7996, 0.8514] | 0.7200 [0.5200, 0.8800] |                    0.853366 | False       |
|         448 |         2 | pooled          |      50 | 0.8244 [0.8021, 0.8443] | 0.9918 [0.9772, 1.0067] | 0.3480 [0.2860, 0.4120] | 0.8081 [0.7855, 0.8293] | 0.7600 [0.6400, 0.8800] |                    0.835289 | False       |
|         448 |         4 | effect_enriched |      25 | 0.7192 [0.6757, 0.7635] | 1.0062 [0.9839, 1.0333] | 0.2400 [0.1720, 0.3120] | 0.6714 [0.6033, 0.7346] | 0.7727 [0.5909, 0.9545] |                    0.717588 | False       |
|         448 |         4 | pooled          |      50 | 0.6885 [0.6603, 0.7171] | 0.9932 [0.9760, 1.0112] | 0.2260 [0.1840, 0.2700] | 0.6485 [0.6059, 0.6911] | 0.8261 [0.7174, 0.9348] |                    0.686805 | False       |
|         448 |         8 | effect_enriched |      25 | 0.5423 [0.5216, 0.5618] | 1.0105 [0.9813, 1.0411] | 0.1080 [0.0720, 0.1520] | 0.5190 [0.4561, 0.5743] | 0.6800 [0.4800, 0.8400] |                    0.543052 | False       |
|         448 |         8 | pooled          |      50 | 0.5396 [0.5240, 0.5547] | 1.0002 [0.9818, 1.0194] | 0.0900 [0.0640, 0.1200] | 0.5096 [0.4713, 0.5463] | 0.6800 [0.5400, 0.8000] |                    0.54268  | False       |
|         512 |         1 | effect_enriched |      25 | 0.9283 [0.9161, 0.9389] | 0.9768 [0.9549, 1.0015] | 0.5920 [0.5240, 0.6600] | 0.8941 [0.8674, 0.9165] | 0.8800 [0.7600, 1.0000] |                    0.933342 | False       |
|         512 |         1 | pooled          |      50 | 0.9223 [0.9062, 0.9352] | 0.9796 [0.9654, 0.9951] | 0.6100 [0.5480, 0.6720] | 0.8967 [0.8793, 0.9128] | 0.8400 [0.7400, 0.9400] |                    0.931788 | False       |
|         512 |        16 | effect_enriched |      25 | 0.5421 [0.5071, 0.5754] | 0.9946 [0.9647, 1.0273] | 0.1200 [0.0680, 0.1801] | 0.5028 [0.4395, 0.5600] | 0.6957 [0.4783, 0.8696] |                    0.537724 | False       |
|         512 |        16 | pooled          |      50 | 0.5477 [0.5253, 0.5694] | 0.9903 [0.9693, 1.0127] | 0.1160 [0.0840, 0.1520] | 0.5219 [0.4758, 0.5645] | 0.6809 [0.5532, 0.8085] |                    0.547526 | False       |
|         512 |         2 | effect_enriched |      25 | 0.8467 [0.8247, 0.8666] | 0.9814 [0.9656, 0.9972] | 0.4360 [0.3480, 0.5240] | 0.8116 [0.7745, 0.8445] | 0.8000 [0.6400, 0.9600] |                    0.854304 | False       |
|         512 |         2 | pooled          |      50 | 0.8249 [0.8014, 0.8457] | 0.9839 [0.9730, 0.9947] | 0.3660 [0.3060, 0.4260] | 0.7892 [0.7626, 0.8147] | 0.7600 [0.6400, 0.8800] |                    0.836575 | False       |
|         512 |         4 | effect_enriched |      25 | 0.7215 [0.6779, 0.7661] | 0.9892 [0.9658, 1.0130] | 0.2080 [0.1400, 0.2800] | 0.7003 [0.6508, 0.7496] | 0.7273 [0.5455, 0.9091] |                    0.717177 | False       |
|         512 |         4 | pooled          |      50 | 0.6851 [0.6559, 0.7158] | 0.9911 [0.9742, 1.0088] | 0.1880 [0.1460, 0.2320] | 0.6471 [0.5985, 0.6940] | 0.7391 [0.6087, 0.8696] |                    0.681451 | False       |
|         512 |         8 | effect_enriched |      25 | 0.5262 [0.5010, 0.5503] | 0.9880 [0.9685, 1.0077] | 0.1040 [0.0680, 0.1400] | 0.5036 [0.4491, 0.5575] | 0.6000 [0.4000, 0.8000] |                    0.529102 | False       |
|         512 |         8 | pooled          |      50 | 0.5273 [0.5088, 0.5454] | 0.9934 [0.9750, 1.0130] | 0.1000 [0.0720, 0.1280] | 0.5154 [0.4804, 0.5506] | 0.7000 [0.5600, 0.8200] |                    0.532402 | False       |

Authorization: `{'384': {'free_continuation_status': 'GATED_BY_TEACHER_FORCED_CAUSAL_FIDELITY', 'strict_interface_pass': True, 'teacher_forced_causal_authorized': False}, '448': {'free_continuation_status': 'GATED_BY_TEACHER_FORCED_CAUSAL_FIDELITY', 'strict_interface_pass': True, 'teacher_forced_causal_authorized': False}, '512': {'free_continuation_status': 'GATED_BY_TEACHER_FORCED_CAUSAL_FIDELITY', 'strict_interface_pass': True, 'teacher_forced_causal_authorized': False}}`.
Smallest candidate causal sufficient dimension: `None`.
