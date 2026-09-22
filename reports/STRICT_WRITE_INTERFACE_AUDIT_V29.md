# Strict Write Interface Audit — V29

- Parent `8d3954ef05957ab22f27261ea7eaa217497b2b62`; base freeze `86389aa651802da41d09c0217c9f588586eab45a7274aee4d5d1ae42c2114913`; design `e9bd1b261b49e01e391d361bc1b1b2fe35a1221cbc3323913afde1f59aae78f3`; interface amendment `40851cd00d278191045604417c0f7edb4df0417f0eea1bd0e81d3c90d49b91d0`.
- Initial calibration/V28-selected-layer diagnostic retained and **excluded** as full cache: only 6 REC/Conv + 2 attention layers; 53 transient development states were interrupted before any formal partition was saved. This exposure is disclosed, not silently recomputed.
- Formal cache coverage: all 24 REC/Conv layers and all 8 attention layers. Every requested native field exactly equals donor; every untouched field equals recipient; shared next token and current-readout capture are audited in Parquet.
- KV operation copies only the newly appended slot; earlier KV slots must already be bitwise equal. No sequence shortening or arbitrary KV subtraction.
- Full all-field transfer is an identity ceiling, not localization. Partial transfers can be off-manifold combinations even with shared incoming state; causal claims are restricted to the tested transplant semantics.
- Independent final opened only for frozen `REC+Conv`, after both dev/validation gates: `399fb32c0a9e607494b1c54a116a35ebfec4f7fde754eb5a2ebcc50d95f232b5`.
- No historical v1–v28 record was overwritten. H2 remains; H3, dynamic-state search, and autonomous controller are unauthorized.
