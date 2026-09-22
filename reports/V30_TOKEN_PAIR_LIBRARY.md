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
