# V20 answers to the 31 scientific questions

1. Yes: independent V19-B, 250 bases, M/R 0.7862 (95% CI 0.7782–0.7931).
2. Yes descriptively: at αq=0.25 all five selected q have reliable matched rate 1.0 and M/R 0.5198, 0.4186, 0.2147, 0.3456, 0.4686.
3. `Φ(P)` is the concatenation of 12 positive train-action 288-D response vectors; 18 directions × 2 signs were measured for diagnostics.
4. Train-positive fingerprint r95/energy curve is tabulated in OPERATOR_INTRINSIC_DIMENSION_V20.md; effective numerical rank 600.
5. Family-wise finite operator r95 is available in the singular-spectra Parquet; no common rank is assumed.
6. Natural and Pq empirical manifold dimensions are listed separately in NATURAL_VS_COUNTERFACTUAL_OPERATOR_V20.md.
7. No oracle coordinate passed all frozen cross-action gates, although train-action coordinates were constructed.
8. `k_operator_min` is not identified in k={2,4,8,16,32,64,128}.
9. Selected-model unseen-direction stack L2=0.9273; preliminary gate=False.
10. Unseen sign L2=1.0056; unseen direction+sign L2=1.0378.
11. Unseen amplitude results for train and validation directions are in ACTION_COMPOSITION_GENERALIZATION_V20.md.
12. Two unseen pair actions and one dense mixture were measured; metrics and actuator audits are in ACTION_COMPOSITION_GENERALIZATION_V20.md.
13. Same-J operator mean principal angle median=37.4441°.
14. Same-J median gain ratio Pq/P0=1.0399.
15. Same-J median r95 change=0.0000; full spectra and divergence are retained.
16. V13 exact-JVP historical rank is compared, but paired V20-state JVP subspaces were not measured.
17. Natural J→oracle C validation R²=0.1394.
18. Same J predicts identical C within each P0/Pq pair; counterfactual J→C relative L2=1.1608.
19. Operational workspace-state aliasing observed=True on 150 exact-J pairs.
20. Raw REC/Conv/KV→C was not eligible or trained after oracle gate failure.
21. No V20 channel-wise raw-P→C recoverability ranking was measured.
22. Same-J raw-P encoder response fidelity was not eligible or measured.
23. Raw P incremental information after J+C+a was not measured.
24. No practical-equivalence claim: raw gain and bootstrap CI were not measured.
25. Held-out family-wise oracle errors are reported; no all-family compact-state gate passed.
26. Independent V20 final status: UNOPENED_NO_FROZEN_ELIGIBLE_ENCODER_FINALIST.
27. No compact causal response-state candidate was established.
28. V21 dynamical-state search is not authorized.
29. H2 remains the project-level interpretation.
30. H3 is not authorized.
31. Autonomous state-model training is not authorized.

Formal outcome: **V20-D_NO_COMPACT_OPERATOR_DIMENSION_IDENTIFIED**. This is failure to identify compactness in the frozen tested model/action regime, not proof of mathematical nonexistence.

Append-only clarification for questions 4–8 and 25: the reported numerical rank 600 uses a near-zero singular-value tolerance and is not an intrinsic-dimension estimate. Family-wise measured 18-action finite-operator r95 medians are 7 on the natural states in all five families, and 6–7 on counterfactual states. A separate frozen nonlinear model at k=128 fits seen directions on new states (relative L2 0.2337) but fails unseen directions (1.3318), supporting **V20-C_ACTION_SPECIFIC_OPERATOR_ENCODING_ONLY** as a secondary descriptive finding. The primary V20-D gate result and V21/H3 restrictions do not change. Source: `results/v20/processed/v20_adjudication_amendment_1.json`.
