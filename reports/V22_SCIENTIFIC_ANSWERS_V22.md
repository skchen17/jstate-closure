# V22 Scientific Answers

1. Input r90/r95/r99: JVP `[11.0, 15.0, 29.0]`; finite `[14.0, 20.0, 34.0]`.
2. Input rank is lower than the 63-direction metric domain but materially larger than the old output-only 5–7 estimate; it is not evidence that raw action space is 7-D.
3. Output U rotates: JVP `18.2923°`, finite `35.3716°`.
4. Input V rotates: JVP `18.5340°`, finite `31.4067°`.
5. JVP/finite input overlap: P0 `0.5334`, Pq `0.4576`; divergence is material at Pq.
6. A fixed global action coordinate is not supported.
7. A state-conditioned chart is structurally indicated, but the tested chart did not improve validation.
8. Reliable directions: `480` of 512.
9. Full 12/24/48/96/128 curves are in ACTION_DATA_SCALING_V22.md.
10. Overall saturation: `FALSE`.
11. Raw maximin gives the best measured local action ceiling; causal D-optimal does not.
12. Causal D-optimal does not beat random/maximin consistently.
13. Z6 beats raw Z1 in the best cross-state comparison but remains far above 0.30.
14. Z7 does not solve unseen action prediction.
15. Polynomial/RBF kernels do not pass.
16. Explicit even/odd does not materially fix unseen sign.
17. Monotone scale modeling improves scale-only L2 to `0.1979` / `0.2534`.
18. No tested coverage metric strongly predicts held-out error.
19. State-conditioned transport changes L2 by `-0.0033` and does not help.
20. No compact operator coordinate emerges.
21. `k_operator_min = NONE`.
22. Historical final six remain sealed.
23. New independent final remains frozen and unopened because no finalist qualified.
24. `RAW_TO_OPERATOR_ENCODER_AUTHORIZED = FALSE`.
25. H2 remains.
26. H3 is not authorized.
27. Dynamic-state search is not authorized.
