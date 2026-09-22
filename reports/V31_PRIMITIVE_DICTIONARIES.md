# Primitive Dictionaries — V31

P0 PCA, P1 TRAIN-only sparse dictionary, P2 TRAIN-only clustered prototypes, P3 frozen-surface-category PCA, and P4 TRAIN-only six-probe REC+Conv response factor were fit before held-out causal tests. P5 Conv-depth profiles and P6 exact REC-conditional Conv factorial are separately recorded as mechanisms, not deceptively counted as learned 128-atom dictionaries. M={8,16,32,64,128}, active s={1,2,4,8,16} when estimable. No deep autoencoder was introduced.

| candidate | fit span/rows | maximum estimable M/ranks | hash prefix |
|---|---|---|---|
| PCA baseline | 256 | 128 | 39c31ff06265fe01 |
| sparse dictionary | 256 | 128 | 02dcb023e7fc90c4 |
| clustered prototypes | 256 | 128 | 1ba6eb3cc9a0a0eb |
| function-conditioned | category dependent | CJK_SURFACE_UNRESOLVED:82, FUNCTION_WORD:137, LEXICAL_SURFACE:229, NUMERIC:59, PUNCTUATION_OR_STRUCTURE:256, SHORT_ALPHA_AMBIGUOUS:84, UNCLASSIFIED:41 | e4a1bd937bfc2036 |
| response factor | 200 | 199 | 127db59e1c6f0051 |

Category `UNCLASSIFIED` has only rank 41; its M64/M128 rows are explicitly not estimable. These are candidate directions, not causally validated primitives.
