# V39 frozen starting point

V39 is a fresh experiment rooted at the pushed V38 commit
`3c33f945b22b7f66a90835861bdb829072a1ea38`. V1–V38 reports,
results, protocols, and the original V38 report are not rewritten.
The only V38-related new file is the append-only
`reports/V38_METRIC_AMENDMENT_V39.md`; the corresponding V39 audit is
`reports/V39_V38_INTERACTION_METRIC_AUDIT.md`.

The V38 audit established that its column `threeway_projection` was a
signed *length* component, `<I234,E111>/||E111||`, not the dimensionless
projection coefficient. V39 preregisters the corrected coefficient
`p=<I234,E111>/||E111||²`, fraction `f=||I234||/||E111||`, cosine, and
the identity `p=f·cos`; V38's historical gate result is not silently
relabelled as an amplification coefficient.

V39 uses the exact V38 model-specific target bundles and clean-scale
normalizers as fixed readout coordinates, with new response-blind natural
token forks and six probes for each V39 state. Falcon is the primary model;
Qwen is a comparative control. The four fresh, mutually disjoint pools are
calibration 20, development 80, validation 40, and independent final 40
states per model, each evenly distributed over five task families. The
formal generator horizon is 3 in every family and role; in particular,
V38's modular-arithmetic role shift is not repeated. All four pools were
sealed before any V39 formal intervention outcome was observed. Validation
and final are not substitutes for V18 training-source states.

Frozen artifacts and design files under `artifacts/interaction_genesis_v39*`,
`data/v39/`, and `results/v39/processed/` carry exact source hashes, pool
hashes, model weight/tokenizer/config hashes, condition mapping, endpoints,
probe IDs, thresholds, and stage order. The independent final is sealed until
the stated Falcon-specific development and validation gates, including a
frozen primitive mediator, pass. This is an opening rule, not a prediction
that they will pass.
