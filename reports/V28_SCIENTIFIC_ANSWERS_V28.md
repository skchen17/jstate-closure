# V28 Scientific Answers

1. The incoming state is the prefix cache: per-layer REC and Conv tensors plus attention KV sequences.
2. The outgoing state is the same native cache after natural processing of token t, with updated REC/Conv and appended KV.
3. REC and Conv are read in the linear-attention mixer; KV is read by full attention.
4. REC/Conv update during token-t mixer execution; KV appends during attention.
5. The returned cache commits after the complete token-t forward.
6. The current mixer can depend on state updates; post-forward interception occurs after current layer-30 readout is fixed.
7. Yes, at the post-forward, pre-next-token cache boundary.
8. Yes, current readout remains exactly captured (Q=0).
9. Yes, REC+Conv old-state restoration changes h1 median Q 34.55 development and 36.82 validation.
10. The independent final median h1 Q is 35.87; all 50 states exceed 0.25.
11. Yes, REC+Conv retained-write dose medians decrease monotonically from α=0 to 1.
12. Yes, REC exact write block changes h1 in all families.
13. Yes, Conv exact write block changes h1 in all families.
14. KV previous-slot rewrite changes h1, but it is not exact old-state restoration.
15. REC+Conv and the length-preserving all-channel condition change h1; additivity fails.
16. Per-layer effects are substantial; no unique earliest commit layer is established.
17. Yes, incoming interventions materially affect current output.
18. Yes, outgoing interception selectively changes future output because current output is already fixed.
19. Yes, at the chosen temporal boundary; this does not establish internal path independence during token-t execution.
20. Yes, full same-length natural outgoing cache was transplanted.
21. Yes, full-cache transplant exactly reaches donor continuation; partial channels fail reciprocal directional gates.
22. Yes, next-token layer-30 J changes strongly under REC+Conv write block.
23. Yes, h2 and h4 effects remain nonzero but attenuate.
24. Yes, current readout and committed outgoing state are causally separable after current computation.
25. Yes, V28-A is supported.
26. Yes, V28-B is supported under exact REC+Conv old-state restoration.
27. V28-C is supported for full-cache identity transfer; V28-D/E are not formally supported.
28. Yes, H2 remains.
29. No, H3 remains unauthorized.
30. Yes for the tested temporal transaction: V26/V27 are better reconciled by read-now/write-for-next than by a pre-existing current-silent state, without a compact-state claim.
