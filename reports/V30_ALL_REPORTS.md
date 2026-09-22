# V30 — All Reports in One File

All 21 V30 report files are reproduced verbatim below in frozen report order. The individual files are authoritative.

---

<!-- 01: V30_TOKEN_PAIR_LIBRARY.md; sha256=bd3e69d493e5a4e9580d4f3ec2bce516a3901ed9f562ba6c210136ced0765166 -->

# Token Pair Library — V30

Frozen calibration-only response-blind library: 88 anchor→candidate contrasts, split 72 TOKEN_TRAIN / 8 TOKEN_VALIDATION / 8 TOKEN_FINAL. Anchor token ID 25 (`:`); candidates are tokenizer tokens common at rank ≤1024 across 25 calibration prefixes. Every token's state eligibility is rank ≤1024, not post-write success. The 88 pairs exceed the suggested 24–48 because centered local k64 needs at least 65 TRAIN contrasts plus disjoint token holdouts. This library is dominated by frequent punctuation, whitespace and lexical fragments; it is **not** a semantic-command panel. No claim about arbitrary language tokens follows. The 50 independent-final states and 8 TOKEN_FINAL pairs stayed sealed.

| split | pair ID | anchor | candidate | surface | calibration mean rank |
|---|---|---|---|---|---|
| TOKEN_TRAIN | e66f17a9197e8116 | 25 | 13 | "." | 15.960 |
| TOKEN_TRAIN | 5d41467b5fa7bd54 | 25 | 487 | "\"," | 201.440 |
| TOKEN_TRAIN | 84ee36c2eba9fb91 | 25 | 1076 | "..." | 53.160 |
| TOKEN_TRAIN | 558ad73cd7ba8400 | 25 | 60 | "]" | 122.760 |
| TOKEN_TRAIN | 8f3a5bd86c9ec947 | 25 | 351 | " C" | 209.400 |
| TOKEN_TRAIN | 833c24cfdd4fe522 | 25 | 3018 | " based" | 203.120 |
| TOKEN_TRAIN | 388aebd2fd91528e | 25 | 4713 | " exit" | 172.520 |
| TOKEN_TRAIN | 33c135ede8487417 | 25 | 314 | " of" | 184.920 |
| TOKEN_TRAIN | a294d32db2f6ab9f | 25 | 3709 | "，" | 174.320 |
| TOKEN_TRAIN | 76eeac365f715d90 | 25 | 6 | "'" | 48.440 |
| TOKEN_TRAIN | 76b8d5bbf3bb61fe | 25 | 4021 | " cannot" | 167.200 |
| TOKEN_TRAIN | 45553178768a9af5 | 25 | 328 | " \"" | 60.720 |
| TOKEN_TRAIN | 9eff460a8d713c1b | 25 | 318 | " (" | 29.200 |
| TOKEN_TRAIN | 5f2eb0feb2e808c5 | 25 | 510 | "</" | 109.000 |
| TOKEN_TRAIN | 15dbd358e2e04b82 | 25 | 26 | ";" | 30.280 |
| TOKEN_TRAIN | 282170a9a0199d5a | 25 | 9 | "*" | 95.960 |
| TOKEN_TRAIN | d328c33219a654a5 | 25 | 30 | "?" | 55.120 |
| TOKEN_TRAIN | c44d895a00958f83 | 25 | 2487 | ".," | 236.600 |
| TOKEN_TRAIN | c09c0c2d3ec4770d | 25 | 2911 | "output" | 65.480 |
| TOKEN_TRAIN | a8df4bfb80ed4fba | 25 | 3 | "$" | 73.280 |
| TOKEN_TRAIN | 7a0d190cdd0446d2 | 25 | 3992 | " language" | 160.720 |
| TOKEN_TRAIN | 45a22f2a39d9d9c5 | 25 | 24 | "9" | 142.880 |
| TOKEN_TRAIN | 57ebb70295e36a97 | 25 | 5513 | "�" | 139.920 |
| TOKEN_TRAIN | 42539c9f30edb309 | 25 | 59 | "\\" | 82.440 |
| TOKEN_TRAIN | 52263ee6f31b2f06 | 25 | 95726 | "的" | 206.200 |
| TOKEN_TRAIN | a46730892ce79f06 | 25 | 321 | " and" | 95.600 |
| TOKEN_TRAIN | 7237e2774f2dd187 | 25 | 1132 | " only" | 224.360 |
| TOKEN_TRAIN | 9e48c8e2859187d3 | 25 | 263 | "on" | 233.720 |
| TOKEN_TRAIN | 8743dcb939158212 | 25 | 1116 | " ," | 139.840 |
| TOKEN_TRAIN | 402ab43b2d1049af | 25 | 561 | " The" | 214.760 |
| TOKEN_TRAIN | cba051dfb627871c | 25 | 1 | "\"" | 24.680 |
| TOKEN_TRAIN | 43e0876c15fb24a1 | 25 | 369 | " is" | 206.960 |
| TOKEN_TRAIN | 1f8c10ef9be93368 | 25 | 14 | "/" | 39.320 |
| TOKEN_TRAIN | b57a9e84024c09b6 | 25 | 57513 | "-output" | 162.240 |
| TOKEN_TRAIN | e813c01a2d9e7515 | 25 | 731 | "\">" | 76.400 |
| TOKEN_TRAIN | 13e7da7493687c4a | 25 | 506 | " at" | 210.680 |
| TOKEN_TRAIN | 0e376cf37db47dca | 25 | 437 | "(\"" | 220.680 |
| TOKEN_TRAIN | b93f94eae1468aa9 | 25 | 8984 | " Output" | 225.240 |
| TOKEN_TRAIN | b4c8b803b37098ce | 25 | 15231 | "​" | 115.800 |
| TOKEN_TRAIN | dba1e4bced5152a0 | 25 | 1118 | " first" | 162.560 |
| TOKEN_TRAIN | 685465e1c6016bb7 | 25 | 313 | " {" | 219.400 |
| TOKEN_TRAIN | e9ec38d3d8b87791 | 25 | 28 | "=" | 116.200 |
| TOKEN_TRAIN | 8d1dd9d7cfb0a084 | 25 | 62 | "_" | 176.080 |
| TOKEN_TRAIN | 4968051f67ddb74a | 25 | 22 | "7" | 208.080 |
| TOKEN_TRAIN | 43dd3568235d87ee | 25 | 641 | " ." | 213.880 |
| TOKEN_TRAIN | b02b99c61bc9ed60 | 25 | 190 | "\u0002" | 206.240 |
| TOKEN_TRAIN | 5933dcf5e62a5109 | 25 | 8 | ")" | 92.280 |
| TOKEN_TRAIN | 942efbff842ad002 | 25 | 579 | "'s" | 216.360 |
| TOKEN_TRAIN | da705031571e5383 | 25 | 420 | "=\"" | 220.680 |
| TOKEN_TRAIN | 0aae6c1752bc1518 | 25 | 0 | "!" | 87.600 |
| TOKEN_TRAIN | 951f007fcd640ea7 | 25 | 364 | " for" | 88.280 |
| TOKEN_TRAIN | 82c99419ab2ca39a | 25 | 16 | "1" | 38.040 |
| TOKEN_TRAIN | 755e0758fb3ca9ca | 25 | 18 | "3" | 122.120 |
| TOKEN_TRAIN | 451686798e79bffa | 25 | 196 | "\b" | 175.400 |
| TOKEN_TRAIN | a530748a9d635a96 | 25 | 90 | "{" | 43.520 |
| TOKEN_TRAIN | 34918c4d7145223f | 25 | 17 | "2" | 64.760 |
| TOKEN_TRAIN | e1115082386ca8a8 | 25 | 4960 | "：" | 65.120 |
| TOKEN_TRAIN | a15a0c6f580afee6 | 25 | 874 | " no" | 101.320 |
| TOKEN_TRAIN | 3bc0747297d67872 | 25 | 2468 | " output" | 36.760 |
| TOKEN_TRAIN | 0a17fd90294a9408 | 25 | 4935 | "Output" | 149.640 |
| TOKEN_TRAIN | 2553b9f518e9032f | 25 | 29 | ">" | 57.000 |
| TOKEN_TRAIN | 529b0c4814a60f06 | 25 | 1317 | " –" | 235.960 |
| TOKEN_TRAIN | b69d258257fb7648 | 25 | 20 | "5" | 148.400 |
| TOKEN_TRAIN | 7529d4e4ba404be9 | 25 | 3272 | " thought" | 164.680 |
| TOKEN_TRAIN | 0780538ab5af1140 | 25 | 3443 | " format" | 216.960 |
| TOKEN_TRAIN | 957d1e4e6f818333 | 25 | 58 | "[" | 134.400 |
| TOKEN_TRAIN | b3697f3e795ea87b | 25 | 310 | " to" | 84.400 |
| TOKEN_TRAIN | def0eafb1b0a9256 | 25 | 1710 | "。" | 78.080 |
| TOKEN_TRAIN | 044f917974f052d8 | 25 | 471 | " -" | 80.160 |
| TOKEN_TRAIN | 3483c04b68a3a844 | 25 | 12 | "-" | 26.240 |
| TOKEN_TRAIN | 224c498a7f13ccc1 | 25 | 539 | " by" | 169.400 |
| TOKEN_TRAIN | 579334d87fdd486d | 25 | 27 | "<" | 60.880 |
| TOKEN_VALIDATION | aad695dc12460e01 | 25 | 11 | "," | 29.680 |
| TOKEN_VALIDATION | cb9b2bf0fbd0f3c9 | 25 | 39456 | " nonsense" | 242.560 |
| TOKEN_VALIDATION | 3ce9a1d837cfb841 | 25 | 1639 | "\\n" | 33.760 |
| TOKEN_VALIDATION | 28d99fdeddf6c664 | 25 | 4754 | "{\"" | 189.760 |
| TOKEN_VALIDATION | 4e8388d46b469c53 | 25 | 628 | " can" | 162.560 |
| TOKEN_VALIDATION | 862d2c7053bdef4e | 25 | 1017 | " there" | 239.000 |
| TOKEN_VALIDATION | 32c727332c1c6b19 | 25 | 5792 | "-based" | 218.840 |
| TOKEN_VALIDATION | b513c56604be6862 | 25 | 3870 | " background" | 240.360 |
| TOKEN_FINAL | 66783de12006c20c | 25 | 7 | "(" | 21.400 |
| TOKEN_FINAL | d99d502b32d4f014 | 25 | 100627 | "输出" | 150.560 |
| TOKEN_FINAL | b317a23b30b0c35d | 25 | 303 | " in" | 106.720 |
| TOKEN_FINAL | 4352abe394014dad | 25 | 19 | "4" | 74.920 |
| TOKEN_FINAL | 34e5f215486e231e | 25 | 760 | "The" | 139.600 |
| TOKEN_FINAL | 0cea8d9ca24fa15d | 25 | 383 | " on" | 191.360 |
| TOKEN_FINAL | 6924e783ee66a8e1 | 25 | 15 | "0" | 67.680 |
| TOKEN_FINAL | d7655933ab15bcd3 | 25 | 21 | "6" | 229.360 |

Machine record: `results/v30/processed/design_v30.json`; library hash `31f57f4c0ca63c1838c32e5fb3ba0829b7c261eb9d3db8666793b493d6ea1775`.

---

<!-- 02: V30_LOCAL_WRITE_GEOMETRY.md; sha256=66d2d64fc37854b3771ffb331e112e2baf17d8654a490cf345d7e9db901feafa -->

# Local Write Geometry — V30

For each target, a centered local REC+Conv basis was fit only from its 70–72 eligible TOKEN_TRAIN contrasts. k=8/16/32/64 was tested; held-out token writes were not used to fit it. The 32D source–target principal-angle audit is descriptive, not a causal language proof.

| role | states | median r90 | median r95 | median r99 | median 32D principal cosine | median max angle ° | median chordal |
|---|---|---|---|---|---|---|---|
| development | 30 | 45.000 | 54.000 | 65.000 | 0.986 | 41.463 | 1.377 |
| validation | 60 | 45.000 | 54.000 | 65.000 | 0.986 | 48.548 | 1.412 |

The local TRAIN spectrum can have r95 ≈54 while the unseen-token k64 natural-write reconstruction median relative L2 remains 0.557 development and 0.558 validation. Thus TRAIN geometry does not establish unseen-token sufficiency.

---

<!-- 03: V30_GLOBAL_WRITE_BASIS.md; sha256=102560d5933960953bdc7f72c858f6edba635c41c1859e1d5b27c8efe8f4fe38 -->

# Global Write Basis — V30

The fixed 13,369,344-component REC+Conv write vector was fit on 90 development states × 8 frozen TOKEN_TRAIN contrasts = 720 rows. These are centered **affine** PCA approximations (mean + Uc), not exact write dimensions or model-state dimensions. All basis tensors remain off-repository scratch; their SHA-256 hashes are committed.

| basis | TRAIN rows | rank | r90 | r95 | r99 | basis hash prefix |
|---|---|---|---|---|---|---|
| FAMILY_boolean_logic | 144 | 143 | 48 | 59 | 79 | c899f2859477d4c7 |
| FAMILY_modular_arithmetic | 144 | 143 | 48 | 58 | 75 | 49804befb388af32 |
| FAMILY_short_graph_traversal | 144 | 143 | 50 | 60 | 77 | 506681d1be8f8527 |
| FAMILY_simple_state_transition | 144 | 143 | 51 | 61 | 79 | ef80da22f9f86a16 |
| FAMILY_variable_binding | 144 | 143 | 49 | 59 | 78 | 422ae32a24131c6d |
| GLOBAL | 720 | 719 | 74 | 131 | 320 | 371aa65ff0f79261 |
| LOFO_boolean_logic | 576 | 575 | 69 | 112 | 256 | bea60272a1f2e188 |
| LOFO_modular_arithmetic | 576 | 575 | 70 | 113 | 260 | 3ae9e51aef69f4a3 |
| LOFO_short_graph_traversal | 576 | 575 | 72 | 122 | 279 | bcd5cecd0d500377 |
| LOFO_simple_state_transition | 576 | 575 | 72 | 122 | 277 | 16d7c07ba5c7a46a |
| LOFO_variable_binding | 576 | 575 | 70 | 113 | 257 | d97e5e843d0cf2e5 |

The global r95=131 already exceeds k64. Geometry alone is not the verdict: the held-out causal OOD results below decide fidelity.

---

<!-- 04: V30_STATE_OOD.md; sha256=7da7c9c88cc051824a2f736fd446041bfa5f7a59e9c69793a3b9760cf3d112f0 -->

# Unseen-State Causal Fidelity — V30

STATE-OOD uses unseen development/validation states with TOKEN_TRAIN pairs that were eligible in the source library. The future endpoint is the concatenated normalized signature from four probes selected before writes. Strong gate: cosine ≥0.9, magnitude [0.8,1.2], relative L2 ≤0.3, success fraction ≥0.5, ≥4/5 families.

### development

| condition | n | cosine | magnitude | relative L2 |
|---|---|---|---|---|
| EXACT_REC_CONV | 60 | 0.976 | 0.978 | 0.220 |
| EXACT_CONV | 60 | 0.954 | 0.965 | 0.306 |
| EXACT_KV | 60 | 0.294 | 0.193 | 0.965 |
| GLOBAL_k32 | 60 | 0.864 | 0.912 | 0.509 |
| GLOBAL_k64 | 60 | 0.943 | 0.958 | 0.336 |
| FAMILY_k32 | 60 | 0.900 | 0.916 | 0.457 |
| FAMILY_k64 | 60 | 0.958 | 0.946 | 0.295 |
| LOFO_k32 | 60 | 0.844 | 0.882 | 0.547 |
| LOFO_k64 | 60 | 0.913 | 0.931 | 0.410 |

### validation

| condition | n | cosine | magnitude | relative L2 |
|---|---|---|---|---|
| EXACT_REC_CONV | 120 | 0.976 | 0.963 | 0.220 |
| EXACT_CONV | 120 | 0.950 | 0.958 | 0.318 |
| EXACT_KV | 120 | 0.345 | 0.195 | 0.952 |
| GLOBAL_k32 | 120 | 0.878 | 0.912 | 0.487 |
| GLOBAL_k64 | 120 | 0.942 | 0.948 | 0.342 |
| FAMILY_k32 | 120 | 0.906 | 0.938 | 0.426 |
| FAMILY_k64 | 120 | 0.954 | 0.961 | 0.305 |
| LOFO_k32 | 120 | 0.856 | 0.879 | 0.527 |
| LOFO_k64 | 120 | 0.908 | 0.925 | 0.424 |

The fixed global k64 basis approaches but does not pass the L2 gate on state OOD; exact REC+Conv remains the native causal ceiling. Family-fixed and LOFO rows are separate controls, not replacements for one global basis.

---

<!-- 05: V30_TOKEN_OOD.md; sha256=822093cc0b2b77c2df4c21b17bf8b9535ff0e6c1fda9b2f6300620956bde6686 -->

# Unseen-Token Causal Fidelity — V30

TOKEN-OOD tests 10 seen training states × 2 TOKEN_VALIDATION pairs each; no held-out token pair entered basis fitting.

| condition | n | cosine | magnitude | relative L2 |
|---|---|---|---|---|
| EXACT_REC_CONV | 20 | 0.977 | 0.977 | 0.214 |
| GLOBAL_k32 | 20 | 0.777 | 0.840 | 0.630 |
| GLOBAL_k64 | 20 | 0.819 | 0.846 | 0.589 |
| FAMILY_k64 | 20 | 0.836 | 0.850 | 0.564 |
| LOFO_k64 | 20 | 0.788 | 0.829 | 0.621 |

Even on seen states, k64 fails the causal gate on new token contrasts. Independent validation of unseen state plus unseen token is reported under JOINT-OOD. TOKEN_FINAL remains unopened.

---

<!-- 06: V30_JOINT_OOD.md; sha256=9a48cd3c0a19c329b87dffcdb0de5bec6be0e5d9926eee9257d1ecf9e7221f21 -->

# Joint State-and-Token OOD — V30

Primary difficult test: both target state and token contrast are outside the global basis fit. Each role uses 2 prospectively frozen TOKEN_VALIDATION contrasts per state.

### development

| condition | n | cosine | magnitude | relative L2 |
|---|---|---|---|---|
| EXACT_REC_CONV | 60 | 0.961 | 0.965 | 0.280 |
| EXACT_CONV | 60 | 0.929 | 0.970 | 0.371 |
| EXACT_KV | 60 | 0.361 | 0.189 | 0.950 |
| GLOBAL_k32 | 60 | 0.791 | 0.848 | 0.619 |
| GLOBAL_k64 | 60 | 0.809 | 0.846 | 0.595 |
| FAMILY_k64 | 60 | 0.798 | 0.868 | 0.608 |
| LOFO_k64 | 60 | 0.789 | 0.824 | 0.618 |

### validation

| condition | n | cosine | magnitude | relative L2 |
|---|---|---|---|---|
| EXACT_REC_CONV | 120 | 0.976 | 0.969 | 0.220 |
| EXACT_CONV | 120 | 0.950 | 0.972 | 0.315 |
| EXACT_KV | 120 | 0.275 | 0.164 | 0.967 |
| GLOBAL_k32 | 120 | 0.809 | 0.817 | 0.594 |
| GLOBAL_k64 | 120 | 0.828 | 0.826 | 0.564 |
| FAMILY_k64 | 120 | 0.829 | 0.830 | 0.565 |
| LOFO_k64 | 120 | 0.814 | 0.826 | 0.591 |

Global k64 has validation median cosine 0.828, relative L2 0.564; exact REC+Conv is 0.220 L2. A fixed compact global basis is not causally sufficient here.

---

<!-- 07: V30_FAMILY_OOD.md; sha256=a0aaf2ea8b0cecb30e95de1892f9098ffc5b7e992a4e2a514a15be86841003b2 -->

# Leave-One-Family-Out Generalization — V30

For each family, the LOFO basis excludes **all** TRAIN rows of that family. The table reports JOINT-OOD relative L2 separately; no failed family is averaged away.

| role | held-out family | exact REC+Conv | Conv | global32 | global64 | LOFO64 | local oracle64 | transport ridge64 |
|---|---|---|---|---|---|---|---|---|
| development | boolean_logic | 0.307 | 0.395 | 0.612 | 0.586 | 0.600 | 0.595 | 0.604 |
| development | modular_arithmetic | 0.189 | 0.297 | 0.558 | 0.548 | 0.579 | 0.527 | 0.531 |
| development | short_graph_traversal | 0.273 | 0.361 | 0.718 | 0.697 | 0.724 | 0.676 | 0.674 |
| development | simple_state_transition | 0.240 | 0.316 | 0.611 | 0.536 | 0.555 | 0.530 | 0.526 |
| development | variable_binding | 0.314 | 0.421 | 0.673 | 0.651 | 0.664 | 0.648 | 0.659 |
| validation | boolean_logic | 0.247 | 0.324 | 0.573 | 0.539 | 0.581 | 0.540 | 0.542 |
| validation | modular_arithmetic | 0.156 | 0.304 | 0.576 | 0.564 | 0.578 | 0.562 | 0.565 |
| validation | short_graph_traversal | 0.255 | 0.312 | 0.612 | 0.565 | 0.609 | 0.519 | 0.559 |
| validation | simple_state_transition | 0.234 | 0.319 | 0.617 | 0.599 | 0.595 | 0.570 | 0.585 |
| validation | variable_binding | 0.203 | 0.314 | 0.599 | 0.573 | 0.618 | 0.564 | 0.574 |

No k≤64 family-OOD causal representation meets the frozen ≥4/5-family rule in both development and validation. The exact natural ceiling remains distinct from compact generalization.

---

<!-- 08: V30_GLOBAL_VS_LOCAL.md; sha256=3ed0f5360c359454c6a5f34dc034f0e015e1e6b8b3acad3fc9dfefee6bd9f46a -->

# Global versus Local Causal Models — V30

M0 fixed global, M1 family-fixed, M2 target local oracle, M3 source→target local transport are compared on **held-out target-state + held-out-token** cases. All local bases and transport maps use TOKEN_TRAIN only.

| role | model | n | cosine | magnitude | causal L2 | write L2 |
|---|---|---|---|---|---|---|
| development | GLOBAL_k32 | 60 | 0.791 | 0.848 | 0.619 | 0.596 |
| development | GLOBAL_k64 | 60 | 0.809 | 0.846 | 0.595 | 0.575 |
| development | FAMILY_k64 | 60 | 0.798 | 0.868 | 0.608 | 0.574 |
| development | LOCAL_ORACLE_k32 | 60 | 0.805 | 0.878 | 0.600 | 0.581 |
| development | LOCAL_ORACLE_k64 | 60 | 0.807 | 0.882 | 0.597 | 0.557 |
| development | TRANSPORTED_LOCAL_RIDGE_k32 | 60 | 0.801 | 0.878 | 0.600 | 0.582 |
| development | TRANSPORTED_LOCAL_RIDGE_k64 | 60 | 0.807 | 0.882 | 0.600 | 0.562 |
| development | TRANSPORTED_LOCAL_PROCRUSTES_k64 | 60 | 0.803 | 0.876 | 0.600 | 0.561 |
| validation | GLOBAL_k32 | 120 | 0.809 | 0.817 | 0.594 | 0.603 |
| validation | GLOBAL_k64 | 120 | 0.828 | 0.826 | 0.564 | 0.579 |
| validation | FAMILY_k64 | 120 | 0.829 | 0.830 | 0.565 | 0.571 |
| validation | LOCAL_ORACLE_k32 | 120 | 0.826 | 0.837 | 0.565 | 0.577 |
| validation | LOCAL_ORACLE_k64 | 120 | 0.834 | 0.845 | 0.556 | 0.558 |
| validation | TRANSPORTED_LOCAL_RIDGE_k32 | 120 | 0.823 | 0.836 | 0.571 | 0.582 |
| validation | TRANSPORTED_LOCAL_RIDGE_k64 | 120 | 0.830 | 0.841 | 0.562 | 0.559 |
| validation | TRANSPORTED_LOCAL_PROCRUSTES_k64 | 120 | 0.829 | 0.839 | 0.564 | 0.561 |

Local k64 and transported k64 are close, but both miss the strong causal ceiling. Consequently M0 is insufficient and M3 does not establish a shared command; M2 failure prevents attributing this solely to a bad transport map.

---

<!-- 09: V30_CROSS_STATE_COORDINATE_TRANSPORT.md; sha256=0b00ce613d82a9760324dcf9c26cf53091ff4646d8d2f98a028328433eb32b7d -->

# Cross-State Coordinate Transport — V30

Five canonical source states were selected using only prewrite eligibility before V30 causal responses; this resource amendment changed only the source map and preserved all 280 frozen target–pair keys. Each source/target local map used ≥65 common TOKEN_TRAIN pairs, ridge α=1 or orthogonal Procrustes; target held-out token write never fit the map. The source's held-out contrast was projected into its TRAIN basis, mapped to the target's TRAIN basis, written into target-native REC+Conv cache, and judged against the target-native donor future. Raw source delta is **not** the primary test.

| family | canonical source | TRAIN pairs | source r95 | basis hash prefix |
|---|---|---|---|---|
| boolean_logic | v13-train-eef9accb656affcadbe4 | 72 | 54 | dcc5ddc5a73b11bc |
| modular_arithmetic | v13-train-b0c1c613e293661abb0b | 72 | 54 | 9901591dd29845e5 |
| short_graph_traversal | v13-train-4dad09829d6e18eb5f14 | 72 | 55 | c8a3be21588d451f |
| simple_state_transition | v13-train-551f383b519f280c1a95 | 72 | 55 | 8b953a135ecd8674 |
| variable_binding | v13-train-9ebb7221b0cd1647e246 | 72 | 54 | 57ff7d5f6a691e62 |

| condition | n | cosine | magnitude | relative L2 |
|---|---|---|---|---|
| LOCAL_ORACLE_k64 | 60 | 0.807 | 0.882 | 0.597 |
| TRANSPORTED_LOCAL_RIDGE_k64 | 60 | 0.807 | 0.882 | 0.600 |
| TRANSPORTED_LOCAL_PROCRUSTES_k64 | 60 | 0.803 | 0.876 | 0.600 |

| condition | n | cosine | magnitude | relative L2 |
|---|---|---|---|---|
| LOCAL_ORACLE_k64 | 120 | 0.834 | 0.845 | 0.556 |
| TRANSPORTED_LOCAL_RIDGE_k64 | 120 | 0.830 | 0.841 | 0.562 |
| TRANSPORTED_LOCAL_PROCRUSTES_k64 | 120 | 0.829 | 0.839 | 0.564 |

Neither map passes unseen-token causal gates. Similarity to the failing local ceiling is not shared-language confirmation. Transport hashes are per-row in `local_transport_*_v30.parquet`.

---

<!-- 10: V30_STATE_DEPENDENT_WRITE_ATLAS.md; sha256=5c2423e71c1870da636ef3c0a98e5339b7f7f7cf3689aeffcdc305ae638f7718 -->

# State-Dependent Write Atlas — V30

The 32D local principal-cosine median is 0.986 development and 0.986 validation. Some directions rotate (development median maximum principal angle 41.463°), but principal angles alone are descriptive. The defining atlas criterion requires target local k≤64 causal sufficiency on held-out tokens **and** train-only transport recovery after a weaker fixed global basis. The local oracle fails (validation k64 causal L2 0.556); therefore V30-C is not established. V29 same-background compression remains valid only for its tested regime. A fourth possibility—token-dependent write directions exceeding the tested compact span—remains open.

---

<!-- 11: V30_MULTI_PROBE_WRITE_SIGNATURE.md; sha256=13483697bcb290e1f0c90b52b5e4133924a8529797c31858ee716499084cfa39 -->

# Multi-Probe Write Signature — V30

All main V30 causal rows concatenate J, selected logits, semantic log-probabilities, workspace, broad vocabulary and late residual across **four frozen next-token probes**. This is stricter than a single next token. Validation JOINT-OOD: exact REC+Conv cosine 0.976, global k64 0.828, local oracle k64 0.834. The four probes were fixed from prefix logits before current-token natural writes. This report does not claim semantic-coordinate labels.

---

<!-- 12: V30_WRITE_HORIZON.md; sha256=34b444a2fdeef72dd2a901eaa205f2ea5f8a8dd23fa832fe66193a94e1c73cc7 -->

# Write Horizon — V30

A separate 10-state/role panel replays the same four prewrite-frozen tokens sequentially and compares h1/h2/h4; this is a single teacher-forced trajectory, not the four-probe concatenated primary endpoint.

| role | h | condition | cosine | magnitude | relative L2 |
|---|---|---|---|---|---|
| development | 1 | EXACT_REC_CONV | 0.965 | 0.977 | 0.269 |
| development | 1 | EXACT_CONV | 0.934 | 0.958 | 0.360 |
| development | 1 | GLOBAL_k32 | 0.824 | 0.861 | 0.576 |
| development | 1 | GLOBAL_k64 | 0.838 | 0.891 | 0.556 |
| development | 2 | EXACT_REC_CONV | 0.970 | 0.969 | 0.245 |
| development | 2 | EXACT_CONV | 0.938 | 0.909 | 0.372 |
| development | 2 | GLOBAL_k32 | 0.895 | 0.928 | 0.488 |
| development | 2 | GLOBAL_k64 | 0.899 | 0.876 | 0.491 |
| development | 4 | EXACT_REC_CONV | 0.925 | 0.879 | 0.383 |
| development | 4 | EXACT_CONV | 0.886 | 0.902 | 0.465 |
| development | 4 | GLOBAL_k32 | 0.749 | 0.818 | 0.669 |
| development | 4 | GLOBAL_k64 | 0.778 | 0.827 | 0.629 |
| validation | 1 | EXACT_REC_CONV | 0.974 | 1.002 | 0.227 |
| validation | 1 | EXACT_CONV | 0.948 | 0.960 | 0.335 |
| validation | 1 | GLOBAL_k32 | 0.856 | 0.888 | 0.535 |
| validation | 1 | GLOBAL_k64 | 0.877 | 0.926 | 0.500 |
| validation | 2 | EXACT_REC_CONV | 0.981 | 0.954 | 0.194 |
| validation | 2 | EXACT_CONV | 0.966 | 0.941 | 0.264 |
| validation | 2 | GLOBAL_k32 | 0.883 | 0.862 | 0.510 |
| validation | 2 | GLOBAL_k64 | 0.892 | 0.848 | 0.518 |
| validation | 4 | EXACT_REC_CONV | 0.959 | 0.922 | 0.285 |
| validation | 4 | EXACT_CONV | 0.932 | 0.926 | 0.368 |
| validation | 4 | GLOBAL_k32 | 0.747 | 0.802 | 0.692 |
| validation | 4 | GLOBAL_k64 | 0.770 | 0.827 | 0.641 |

Exact REC+Conv retains a donor-directed effect through h4 but development h4 does not meet the 0.30 L2 gate. Global compact writes do not regain qualification at later horizons. No transported-local h2/h4 claim is made because its h1 OOD gate failed.

---

<!-- 13: V30_CONV_LAYER_PROFILE.md; sha256=06bcd2ca734ea810304ae802b309374077591c18f2ee703f6a68b8667d509e76 -->

# Conv Layer Profile — V30

Exact native donor Conv fields were tested at every one of 24 recurrent layers on a prospectively frozen 10-state/role panel. All untouched cache fields are checked bitwise by the experiment. Full Conv: development cosine 0.921, L2 0.390; validation cosine 0.962, L2 0.280.

| layer | dev cos | dev L2 | val cos | val L2 |
|---|---|---|---|---|
| 0 | 0.270 | 0.963 | 0.171 | 0.988 |
| 1 | 0.156 | 0.989 | 0.055 | 1.004 |
| 2 | 0.091 | 0.997 | 0.084 | 0.997 |
| 4 | 0.073 | 0.999 | 0.052 | 0.999 |
| 5 | 0.139 | 0.990 | 0.151 | 0.989 |
| 6 | 0.062 | 0.999 | 0.055 | 0.998 |
| 8 | 0.044 | 1.000 | 0.013 | 1.001 |
| 9 | 0.028 | 1.000 | -0.017 | 1.002 |
| 10 | 0.046 | 0.999 | 0.030 | 1.000 |
| 12 | 0.069 | 0.998 | 0.116 | 0.995 |
| 13 | 0.114 | 0.995 | 0.186 | 0.989 |
| 14 | 0.108 | 0.994 | 0.062 | 0.999 |
| 16 | 0.083 | 0.997 | 0.100 | 0.996 |
| 17 | 0.293 | 0.975 | 0.303 | 0.978 |
| 18 | 0.328 | 0.961 | 0.276 | 0.970 |
| 20 | 0.306 | 0.983 | 0.351 | 0.974 |
| 21 | 0.195 | 0.989 | 0.218 | 0.990 |
| 22 | 0.264 | 0.986 | 0.238 | 0.988 |
| 24 | 0.121 | 0.994 | 0.124 | 0.995 |
| 25 | 0.144 | 0.993 | 0.192 | 0.989 |
| 26 | 0.070 | 0.998 | 0.030 | 1.001 |
| 28 | 0.252 | 0.975 | 0.274 | 0.975 |
| 29 | 0.142 | 0.992 | 0.192 | 0.985 |
| 30 | 0.431 | 0.932 | 0.488 | 0.906 |

Quartiles, halves, prefixes, suffixes, leave-quartile-out and 24 leave-single-out rows are preserved in `conv_layer_profile_*_v30.parquet`; a single layer's effect is not additive evidence of necessity.

---

<!-- 14: V30_MINIMAL_CONV_CARRIER.md; sha256=80b040b50a6110113f2a694fe2bad0345862352cfdb74032ecc03aba52a3b98f -->

# Minimal Conv Carrier — V30

The frozen development-only rule searched 19 nested/group candidates; none passed the four-probe strong gate. Full 24-layer Conv was therefore the **fallback**, not a successful small carrier. Selected set `FULL_CONV`, layers `[0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18, 20, 21, 22, 24, 25, 26, 28, 29, 30]`, SHA-256 `e3e5cfd2c0df818db77d9198d8fae8251fd7c15dcca213b62ce1a4bc3748725c`. The full set failed development (L2 0.390) but passed validation (L2 0.280). Validation-only passing subsets were not promoted or searched adaptively. Neither localized-small-set V30-F nor distributed-necessary V30-G is established.

---

<!-- 15: V30_REC_CONV_CONDITIONAL_EFFECT.md; sha256=26f46659a07d578e98222788bf29517c378caf258919081507c5b34a31b2deb8 -->

# REC–Conv Conditional Effect — V30

Exact native Y00 recipient, Y10 REC donor, Y01 Conv donor, Y11 REC+Conv donor were evaluated on the same frozen 10 states per role. Interaction is ‖Y11−Y10−Y01+Y00‖ / ‖donor−recipient‖, not an additivity assumption.

| role | states | REC improves count | Conv L2 | REC+Conv L2 | REC-only L2 | paired median gain | interaction ratio |
|---|---|---|---|---|---|---|---|
| development | 10 | 10 | 0.390 | 0.314 | 0.995 | 0.068 | 0.209 |
| validation | 10 | 10 | 0.280 | 0.212 | 1.002 | 0.068 | 0.219 |

REC adds a reproducible conditional refinement (20/20 paired L2 reductions) while REC alone remains weak. This supports V30-H in this endpoint; it does not make REC an independent carrier or turn the development REC+Conv subset into a passed compact coordinate.

---

<!-- 16: V30_WRITE_RANK_SCALING.md; sha256=ee5c01626cf0d1776c35356df373cc9294bc36b41078d579b2e1d1145b66c5d1 -->

# Write Rank Scaling — V30

Centered pooled natural REC+Conv contrast ranks are descriptive, not causal-state dimensions. The 90-state×8-pair original fit had r90/r95/r99 = 74/131/320.

Balanced state scaling (8 TRAIN pairs/state):

| states | rows | r90 | r95 | r99 |
|---|---|---|---|---|
| 10 | 80 | 19 | 31 | 58 |
| 120 | 960 | 75 | 135 | 354 |
| 20 | 160 | 36 | 58 | 115 |
| 40 | 320 | 65 | 104 | 219 |
| 80 | 640 | 74 | 128 | 305 |

Original-fit token coverage scaling (90 states):

| pairs/state | rows | r90 | r95 | r99 |
|---|---|---|---|---|
| 1 | 90 | 12 | 24 | 54 |
| 2 | 180 | 23 | 45 | 100 |
| 4 | 360 | 41 | 78 | 179 |
| 8 | 720 | 74 | 131 | 320 |

The 120-state point uses 30 extra frozen development states and deterministic prewrite-eligible TRAIN pairs selected **after development response observation**; it is explicitly supplemental and was never used for any causal basis, transport fit, gate or finalist. State r95 growth slows after 80, but token coverage 1→8 still grows 24→131; the exact write geometry is not shown saturated. Behavioral k64 failure is a separate causal observation.

---

<!-- 17: V30_NATURAL_VS_ARTIFICIAL_COORDINATES.md; sha256=e26f22b558a48091a6be9cf1ce28533b2e32789b6fe32e94817650649c07d730 -->

# Natural versus Artificial Coordinates — V30

On the predeclared 10-state/role Conv-profile panel, the TRAIN-only global k64 approximation was compared with exact natural REC+Conv, same-norm random tensor, shuffled coordinate, sign flip, wrong-source-state coordinate, wrong-token coordinate and α=.5/1.5 amplitudes. Random and shuffled reconstructions are deliberately off-manifold controls; their failure cannot adjudicate natural-write existence.

| role | condition | cosine | magnitude | relative L2 |
|---|---|---|---|---|
| development | EXACT_REC_CONV | 0.950 | 0.970 | 0.314 |
| development | GLOBAL_k64 | 0.805 | 0.825 | 0.596 |
| development | RANDOM_SAME_NORM | 0.073 | 3.359 | 3.279 |
| development | SHUFFLED_COORDINATE | 0.493 | 0.576 | 0.874 |
| development | SIGN_FLIPPED | 0.112 | 0.466 | 1.073 |
| development | WRONG_STATE_COORDINATE | 0.808 | 0.813 | 0.590 |
| development | WRONG_TOKEN_COORDINATE | 0.376 | 0.756 | 0.946 |
| development | AMPLITUDE_0_5 | 0.701 | 0.711 | 0.715 |
| development | AMPLITUDE_1_5 | 0.851 | 0.973 | 0.544 |
| validation | EXACT_REC_CONV | 0.978 | 0.969 | 0.212 |
| validation | GLOBAL_k64 | 0.863 | 0.794 | 0.508 |
| validation | RANDOM_SAME_NORM | 0.163 | 3.557 | 3.570 |
| validation | SHUFFLED_COORDINATE | 0.560 | 0.737 | 0.887 |
| validation | SIGN_FLIPPED | 0.098 | 0.513 | 1.081 |
| validation | WRONG_STATE_COORDINATE | 0.861 | 0.797 | 0.512 |
| validation | WRONG_TOKEN_COORDINATE | 0.545 | 0.858 | 0.846 |
| validation | AMPLITUDE_0_5 | 0.732 | 0.728 | 0.683 |
| validation | AMPLITUDE_1_5 | 0.886 | 0.966 | 0.471 |

Structured but inadequate global coordinates outperform random/shuffled/wrong-token controls. Wrong-state global coordinates being similar to target global coordinates does not establish causally adequate transport. Amplitude and sign do not exhibit a qualified linear control law; no per-coordinate semantic label is assigned.

---

<!-- 18: V30_STRICT_WRITE_INTERFACE_AUDIT.md; sha256=5ee532b1880ef0e9134e9194bc0af5819aa0a34947d0896f6bb97a1df7395195 -->

# Strict Write Interface Audit — V30

Each comparison starts from the same architecture-native prefix cache; design records incoming REC/Conv/KV SHA-256 channel hashes and prefix-token hashes for all 255 frozen states. Current token forks finish naturally before any transplant. REC+Conv comprises all 24 recurrent/Conv layers (48 tensors; 13,369,344 float32 contrast components); KV contains 8 appended attention slots and is not silently relabeled as absent memory. Exact Conv depth writeback verifies donor equality of touched fields and recipient equality of untouched fields for each tested condition. Global and local `inject` clone the native recipient and alter only REC+Conv at identical cache length; these PCA realizations are off-manifold approximations, not exact native donor writes. The four-probe next-token sequence and current readout are fixed before transplant. Per-row native cache hashes and contrast hashes are in Parquet/JSON; large tensor matrices/bases remain outside Git under `/data/CSK/J-space-project/v30-write-work` with committed hashes. No V1–V29 record was overwritten, and V30 independent-final responses were not opened. The 120-state rank supplement is descriptive only.

---

<!-- 19: V30_EXECUTION_MANIFEST.md; sha256=9338059ea07de7a7e45471b55922590f1530fb129fa64304df2499046c4d71d9 -->

# Execution Manifest — V30

Parent Git commit `c247c82020b265e9589493262562851234825113`; V30 protocol/config base frozen before current-token writes. Panels: 25 calibration (prewrite only), 120 development, 60 validation, 50 independent final sealed. The execution plan freezes 90 fit states, 30 development holdout, 10 seen-state TOKEN-OOD tests, 60 validation targets, 8/16/32/64 dimensions, four probes, three causal axes, five LOFO models, transport rules, Conv groups and thresholds. A pre-response transport-source amendment reduced 85 sources to 5 without changing target pairs. An interrupted CPU basis fit produced no response record; the same frozen matrix/Gram was used for streamed GPU basis multiplication. A later 120-state spectrum supplement is explicitly descriptive.

| freeze file | digest prefix | file hash prefix |
|---|---|---|
| transferable_natural_writes_v30.freeze.json | 21816450b59d1092 | 35c2b78e5f766d23 |
| transferable_natural_writes_v30_adjudication.freeze.json | e29529c64d159e57 | 7ea90b53a52ca753 |
| transferable_natural_writes_v30_causal_global_development.freeze.json | 1347ef9442ad1add | 25d8a580b424e180 |
| transferable_natural_writes_v30_causal_global_validation.freeze.json | 108c9fae164f58ec | 1f02cca48b6c3494 |
| transferable_natural_writes_v30_conv_depth_development.freeze.json | 7213ef4769be6b4a | 68023433baccaf5b |
| transferable_natural_writes_v30_conv_depth_validation.freeze.json | 7f291045111c12fb | 34e6a6e15fc25f1e |
| transferable_natural_writes_v30_coordinate_controls_development.freeze.json | fce7ebf57e579893 | 3efbc60dfc36f5d1 |
| transferable_natural_writes_v30_coordinate_controls_validation.freeze.json | 5f3f932cc1b3e3cf | a9fe2ca0346ddb3c |
| transferable_natural_writes_v30_design.freeze.json | 06a62592ed8d4d70 | 1334c65477db7c27 |
| transferable_natural_writes_v30_execution_plan.freeze.json | be4cd497e3cd2846 | d1e796ea95fb41b4 |
| transferable_natural_writes_v30_final_opening.freeze.json | 5cda4310df409b8e | d26cf91699c4b7c6 |
| transferable_natural_writes_v30_global_basis_fit.freeze.json | d29baca0c4351e1e | 9bdee033fe79202b |
| transferable_natural_writes_v30_horizon_development.freeze.json | a4fb3333d6aabb89 | 18b8b17890310b66 |
| transferable_natural_writes_v30_horizon_validation.freeze.json | 53ebebeb8b8c8f8c | 413c4873ce250101 |
| transferable_natural_writes_v30_local_source_bases.freeze.json | 39b259428d5f84bc | 61bec382ef1f29c3 |
| transferable_natural_writes_v30_local_transport_development.freeze.json | 7eac90a3b2bfd408 | c91feb469169d983 |
| transferable_natural_writes_v30_local_transport_validation.freeze.json | 2341688b15b88136 | bb35bdf36ffd8072 |
| transferable_natural_writes_v30_rank120_collect.freeze.json | a5aefce1073c4f49 | ff3ed17a02214e21 |
| transferable_natural_writes_v30_rank120_fit.freeze.json | 9928225bc24190a9 | 9e7cd5d6943308ed |
| transferable_natural_writes_v30_rank120_plan.freeze.json | ce05c8530d1c3d2f | 031bbf2cd03883bb |
| transferable_natural_writes_v30_rank_scaling.freeze.json | 211f814e3bf35781 | 9547e1945c1a2244 |
| transferable_natural_writes_v30_transport_source_amendment.freeze.json | deea887b6f7c4f18 | afeff4d7158b551e |

Formal final-opening decision `No candidate passes all frozen development and validation causal OOD gates`; final-opened=`False`; final decision file SHA-256 `c9fc21d1baab8e8de3a6b9725d8d287211e2c8230a130689c3aaf9788fe31d50`. Frozen authorization: H2 true, H3 false, dynamic search false, autonomous controller false, cross-model replication false. Machine records are under `results/v30/processed/`; off-repository scratch is not in Git.

---

<!-- 20: V30_SCIENTIFIC_ANSWERS.md; sha256=943390aacf95302df2466b48d839d23e1f4695745626be4d26fae0e40db5966f -->

# Scientific Answers — V30

The answers are restricted to the frozen token library, hybrid model and causal endpoint.

1. V29 32/64D same-background success did not survive the expanded held-out-token OOD panel.
2. Pooled exact write rank is not demonstrated saturated; token coverage r95 grows 24→131 for 1→8 pairs/state.
3. A fixed global k64 does not pass unseen-state causal gates.
4. It fails unseen token contrasts even on seen states.
5. It fails joint unseen state+token OOD in development and validation.
6. Five LOFO tests do not satisfy ≥4/5 causal replication.
7. k32 is insufficient on all primary OOD axes.
8. k64 is also insufficient on token/joint OOD.
9. Required tested behavioral k exceeds 64 or requires a different representation; no minimal larger k was measured.
10. Local 32D subspaces overlap strongly descriptively (median principal cosine ≈0.986 development).
11. Some directions rotate (median maximum principal angle ≈41° development); geometry alone is not state dependence.
12. The target local oracle is not materially better enough to pass held-out-token causal gates.
13. Source→target coordinates were transported by TRAIN-only ridge/Procrustes maps.
14. Their target-native future response did not pass the strong gate.
15. No: held-out token contrasts fail for transport and target local oracle.
16. No ≥4/5-family transported causal qualification.
17. No fixed global basis is sufficient in this tested k≤64 regime.
18. A state-dependent atlas is not established, because its local causal ceiling fails.
19. V29 local same-background compact effects remain; V30 cannot confirm local-only compact sufficiency for new tokens.
20. A full 24-single-layer Conv profile is recorded; no single layer passes the full causal claim.
21. No development-selected small Conv layer subset passes; the frozen fallback is all 24 layers.
22. Distributed necessity is not established, because even full Conv fails development four-probe gate.
23. REC improves Conv conditional fidelity in all 20 profiled states across roles.
24. REC×Conv interaction is material descriptively (median ratio ≈0.21 in both roles).
25. KV-only remains weak for this tested future endpoint, not generally unimportant.
26. Exact REC+Conv transfers across four probes; compact k32/64 does not.
27. Exact REC+Conv is donor-directed at h2/h4, but h4 development misses the strong L2 gate; compact write remains unqualified.
28. Structured global coordinates beat random/shuffled/wrong-token controls but fail the natural donor gate.
29. α=.5/1.5 effects are measured and nonlinear; no qualified linear amplitude law.
30. Sign flip does not show odd-symmetric causal control.
31. V30-A false.
32. V30-B false.
33. V30-C false.
34. Token-OOD generalization false.
35. Family-OOD generalization false.
36. Conv depth localization false; distributed necessity also not proven.
37. Cross-model replication not authorized.

No 32D/64D **model state** or exact natural-write dimensionality is claimed.

---

<!-- 21: V30_COMPLETE_REPORT.md; sha256=0b3b800932d3335bceaa25ce13cb4ee968fc637be61895e7cde935ee6e8b4d63 -->

# Complete Report — V30

**Generality and Geometry of Transferable Natural Writes — Is There a Shared Causal Write Language Across States, Tokens, and Tasks?** Parent `c247c82020b265e9589493262562851234825113`; frozen base `21816450b59d10929de62379cd212abae80cbd1997a5ad7725bcd78b7851cd36`; adjudication `e29529c64d159e57e8faedcf956c9b488b6b1d907b052bc4279bd6fa6eca95ba`.

The prospective 25/120/60/50 panels exclude V28/V29 states. A calibration-only 88-pair common-token library is split 72/8/8 TRAIN/VALIDATION/FINAL. The primary REC+Conv write contrast has 13,369,344 float32 components. Ninety fitting states × eight TRAIN pairs yield pooled r90/r95/r99 = 74/131/320; this is descriptive write geometry, **not** a model-state dimension. The supplemental 120-state r95=135 is descriptive only and did not enter causal fitting.

The decisive four-probe future-response test does **not** identify a shared ≤64D write language. In validation JOINT-OOD, exact REC+Conv native donor write has median cosine 0.976, relative L2 0.220; global k64 has 0.828/0.564. Target local-oracle k64 L2 0.556 and TRAIN-only transported ridge k64 L2 0.562 also miss the predeclared ≤0.30 gate. Because even the target local ceiling fails, this cannot isolate an inadequate transport map or prove a rotating atlas. V29's same-background 32/64D result remains valid in its own narrower regime.

Five-family LOFO and seen-state TOKEN-OOD tests likewise fail compact causal gates. The Conv all-24-layer fallback does not replicate the strict four-probe gate in both roles, so neither small-layer localization nor distributed-depth necessity is confirmed. REC conditionally improves Conv donor fidelity in 10/10 development and 10/10 validation states; REC-only remains weak. KV-only remains weak for this immediate future endpoint without denying its history-storage role. Natural exact writes outperform artificial controls. h2/h4 and amplitude/sign values are reported without promoting a failed compact model.

| formal outcome | supported |
|---|---|
| V30_A_GLOBAL_CAUSAL_WRITE_SUBSPACE_CONFIRMED | False |
| V30_B_SHARED_CAUSAL_WRITE_COORDINATES_CONFIRMED | False |
| V30_C_STATE_DEPENDENT_WRITE_ATLAS | False |
| V30_D_TOKEN_OOD_WRITE_GENERALIZATION | False |
| V30_E_FAMILY_OOD_WRITE_GENERALIZATION | False |
| V30_F_CONV_CARRIER_LOCALIZED_IN_DEPTH | False |
| V30_G_CONV_CARRIER_DISTRIBUTED_ACROSS_DEPTH | False |
| V30_H_REC_CONDITIONALLY_REFINES_CONV_WRITE | True |
| V30_I_LOCAL_COMPACT_WRITE_ONLY | False |
| V30_J_WRITE_EFFECT_DIMENSION_NOT_SATURATED | True |
| V30_K_NO_SHARED_WRITE_COORDINATE_IDENTIFIED | True |

No finalist qualified. The 50 independent-final states and TOKEN_FINAL contrasts were **not opened**; final-opening freeze `5cda4310df409b8efffc34d4a3e35b7e15e1a2a7c909b1880e6ca9281d219adb`. `H2_REMAINS=TRUE`; `H3_AUTHORIZED=FALSE`; `DYNAMIC_STATE_SEARCH_AUTHORIZED=FALSE`; `AUTONOMOUS_CONTROLLER_AUTHORIZED=FALSE`; `CROSS_MODEL_REPLICATION_AUTHORIZED=FALSE`. The original G/A/L trichotomy is incomplete under this expanded token library: a fourth possibility is that the tested ≤64D span omits token-dependent natural-write directions. Full machine records, hashes, reports and limitations follow in the separate V30 files and `V30_ALL_REPORTS.md`.
