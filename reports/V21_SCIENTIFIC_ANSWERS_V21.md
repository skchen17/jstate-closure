# V21 scientific answers

1. **Primary bottleneck?** Action representation/coverage is the strongest observed practical limitation: fixed S2/G2 Z0→Z1 gains 0.6751 L2, but no practical coordinate passes; model/finite geometry remain possible contributors.
2. **Does full Φ_train beat k128?** No material gain: 0.0023 absolute L2 (<0.10 gate).
3. **Full requested coordinate vs V20 descriptor?** Yes, unseen L2 1.3137→0.6386 under fixed S2/G2, still fails 0.30.
4. **Architecture resolved Z2?** Corrected Z2 0.6329 versus Z1 0.6386: small further gain, not a pass.
5. **Realized BF16 Z3?** Best state-dependent Z3 0.6983; not a pass; 39D Nyström caveat.
6. **Coverage sufficient?** No; Z1 train-span residual median 0.7437.
7. **More actions help?** 4/8/12 decoder-action L2 0.9964, 0.6980, 0.6386.
8. **Raw-state ceiling?** S4×Z1/G2 0.6847 versus S2 0.6386; no gain for this 30-anchor raw kernel.
9. **Exact JVP predict finite unseen actions?** Direct Z4 L2 0.8062; best state-conditioned 0.6955, oracle only.
10. **Second order help?** Z5 direct Taylor 0.7922 vs JVP-only 0.8062; half-step oracle only, reliability 1.0000.
11–12. **Paired r90/r95/r99?** JVP 4.0/5.0/8.0; finite 5.0/7.0/12.0 (validation P0).
13–14. **Pq rotation?** JVP 36.7976°, finite 30.4461°.
15. **Rotations correlated?** Validation Spearman 0.5669, CI [0.1541, 0.8287].
16. **Rotation/gain/spectrum/rank?** Finite pre/post Procrustes 0.7697/0.4033, gain 0.9583, JS 0.006929, Δr95 0.0.
17. **Same V13/V20 mechanism?** Strong paired support = True; do not infer equality from rank alone.
18. **Corrected practical action coordinate identified?** False.
19. **Compact search reopened?** False.
20. **Raw P→C authorized?** False.
21–23. **H2/H3/dynamic search?** H2 remains True; H3 False; dynamic search False.

Formal result: **V21-F_CROSS_ACTION_OPERATOR_REMAINS_UNIDENTIFIED_UNDER_TESTED_PRACTICAL_FAMILIES + V21-G_ACTION_DATA_LIMITED + V21-E_SHARED_STATE_DEPENDENT_CAUSAL_GEOMETRY**. This is a tested-regime result, not proof of mathematical nonexistence or a license for H3/autonomous control. Final six remain unopened.
