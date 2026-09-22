# Gain, Rotation and Correction Decomposition — V32

Development/validation joint gain ratios (norm RC / norm C) are 1.012/1.013. Per-row scalar alpha projects the joint move onto Conv; this is an oracle descriptive upper bound, not a prospective predictor. On validation the median scalar-gain donor L2 is 0.287, versus true joint 0.180; true joint beats that scalar in 1.000 of rows.

The fixed global rotation comparison fits a rank-32 orthogonal Procrustes map on development moves only, retaining identity outside that development subspace. Validation median donor L2 is 0.280, versus true joint 0.180; joint wins in 1.000 of rows. This rejects only the tested low-capacity fixed rotation, not every possible context-dependent rotation.

The sequential decomposition of conditional REC response on development gives median normalized gain component 0.114, residual-target component 0.805, and remaining orthogonal component 0.583. These overlapping norm fractions do not sum to one; they are descriptive, whereas exact factorial error reduction is causal.
