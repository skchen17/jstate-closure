# V33 Replication Adjudication

Architecture comparable; locked development and validation correction/alignment gates pass; final confirms; Conv dominance, gain falsification, material interaction, and matched-context control pass. Level 1 qualitative, Level 2 formal, Level 3 structural: **PASS / PASS / PASS** in the tested models.

|formal outcome|supported|
|---|---|
|V33-A_CROSS_MODEL_REC_CORRECTION_REPLICATED|True|
|V33-B_CONV_DOMINANT_HANDOFF_REPLICATED|True|
|V33-C_CONDITIONAL_NOT_ADDITIVE_OR_GAIN_ONLY|True|
|V33-D_MATCHED_CONTEXT_CORRECTION_REPLICATED|True|
|V33-E_CROSS_MODEL_CHANNEL_ORGANIZATION_PARTIAL|False|
|V33-F_ARCHITECTURE_SEMANTICS_NOT_COMPARABLE|False|
|V33-G_MODEL_SPECIFIC_REC_CORRECTION|False|
|V33-H_SHARED_MICRO_MEDIATOR_IDENTIFIED|False|
|V33-I_SHARED_EFFECT_WITH_DIFFERENT_MICRO_MEDIATORS|False|

The predeclared failure branches (channel semantics differ, handoff-only, alternate carrier, no analogous organization) are not selected. V33-G is false for this model pair; failure was not observed. H/I are **not established**, rather than evidence of absent or different micro-mediators, because bidirectional mediation was not performed. Cross-model replication is not universality. Machine evidence: `results/v33/processed/`, immutable stage freezes in `artifacts/cross_model_rec_conv_v33*.freeze.json`.
