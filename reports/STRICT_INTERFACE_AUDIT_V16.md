# Strict interface and scientific-claim audit — V16

- V1–V15 protocols/results are immutable; only cumulative `FINAL_REPORT.md` is appended. Parent commit: `f1986b1b806112b1c337eb1084b3b82ae11ffb1f`.
- Real BF16 REC/Conv/KV writeback and readback are used. Requested and realized actions are separated; failed actions stay in the denominator.
- Model inputs use compact action coordinates; M4 uses only train-frozen current clean J summary. No raw persistent state or teacher raw target cache enters the response model/controller.
- Validation states are disjoint from model-fit states; strict sign, scale, pair, dense and family diagnostics are separately recorded. No independent V16 final bank was opened.
- V15 r95 gap is not 14 control dimensions. Polynomial fit is not a declaration of true quadratic dynamics. Action representation is not complete state representation.
- No V16 nonlinear MPC or autonomous S→S′ model was trained. `ABSOLUTE_REPLACEMENT_NOT_AUTHORIZED`; H2 remains; H3 candidate and full H3 are unsupported.
- V16-F is **not** asserted: despite the full 100/50 response bank, exact realized coordinates cover a 20/10 subset and primitive/sequence studies are diagnostic, not the comprehensive negative test required for F.

Formal branch audit: V16-A not supported (realized coordinates do not materially improve matched-subset response prediction); V16-B/C fail the frozen response gate; V16-D remains diagnostic rather than validated; V16-E is gated off before MPC/independent confirmation; V16-F is not eligible as a comprehensive negative claim.

Formal procedural result: **V16-STOP — NONLINEAR_RESPONSE_VALIDATION_GATE_NOT_PASSED**. Independent final: `NOT_CREATED_OR_OPENED`. This is not a theorem of absent compact causal control.
