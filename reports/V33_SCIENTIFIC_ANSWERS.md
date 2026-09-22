# V33 Scientific Answers

1. Yes: 24 hybrid layers expose persistent REC2/CONV2/KV, with distinct Mamba-2 micro-algebra.
2. Yes: recurrent_states and conv_states are separable persistent native tensors.
3. Yes: exact donor field replacement and untouched recipient field equality passed.
4. Yes: Conv-only relative L2 ≈0.24 versus REC-only ≈1.00 in validation.
5. Yes: joint relative L2 ≈0.126 versus Conv-only ≈0.240 in validation.
6. 100/100 development, 50/50 validation, 50/50 opened final.
7. Development 0.388, validation 0.413, final 0.363.
8. Yes: validation median 0.811, lower bound 0.725.
9. Yes: 5/5 correction and alignment families in both formal roles.
10. Yes: REC-only donor relative L2 ≈1.00 versus Conv-only ≈0.24.
11. No: true joint beats per-row best scalar in all validation rows.
12. No for tested rank-32 fixed map: true joint better in 1.000 validation rows.
13. Yes: validation median interaction ratio 0.228.
14. Yes: 20/20 frozen context targets.
15. Yes: 20/20; cross-state off-manifold caveat.
16. Yes: 20/20 versus shuffled and random; artificial controls are off-manifold.
17. Yes: all primary factorials used recipient-native KV.
18. No: KV-only median relative L2 0.984 on 20 controls.
19. Yes.
20. Yes.
21. Yes for tested scalar/additive distinction; not a universal nonlinear exclusion.
22. Yes, in frozen 20-target subset, with prospective-timing caveat.
23. No: tested high-level organization is shared.
24. No: architecture-valid interface passed.
25. No: this comparable tested model replicates; specificity not supported.
26. Yes: formal dev+validation passed.
27. Qwen transformed g/beta matched consumed kernel inputs in 24/24; Falcon transformed dt/decay reconstruction audited.
28. Qwen post-Conv q/k/v captured; Falcon post-Conv x/B/C captured, not identical q/k/v.
29. Yes, Qwen delta and Falcon dBx reconstructed against true state arguments in 24/24 layers each.
30. Not tested: no bidirectional ≥50% mediation intervention.
31. Not established.
32. Not established; anatomical differences alone do not show different mediators.
33. Yes for broader replication, not universal claims.
34. Channel-aware application experiments are justified as future tests; no benefit is claimed.

Machine evidence: `results/v33/processed/`, immutable stage freezes in `artifacts/cross_model_rec_conv_v33*.freeze.json`.
