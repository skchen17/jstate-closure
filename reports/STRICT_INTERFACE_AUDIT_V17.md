# V17 strict interface and provenance audit

- Historical parent: `60060cb481b6fe2fe6a9bbf123611d37d4445fea`; base protocol `71c0501d3289b4ecafa5957c3121a076aee665e8967db7890ff19c3c050b77f8`; split `5096bead69209ec4a46747866ead84e5b99942f06f179216e0efd0e87d4682b9`; conditional design `1f7f47ea02310b6a7068688be4656a2fb968cd08dfa98f7c63021f94b8c7b57f`.
- V16 action bank and V13 clean-state shards were read-only. Source shard SHA256 values were checked before loading. No V13 perturbed state, teacher target cache or future label entered the context kernel.
- State-context centering and kernel-ridge lambda selection used training states only. Validation was never used to fit source features, choose lambda or select a compact candidate. Development validation is reused V16 validation, **not independent confirmation**.
- Source amendments 1/2 record uppercase `RELIABLE` normalization and KV token padding; conditional amendments 1/2 record validation offset correction and algebraically equivalent bootstrap optimization. Original freeze records are preserved.
- Same-action pairs require identical coordinate/sign/alpha, but their J/raw proximity is rank-relative; no physical context swap or replacement claim is made.
- Old V1–V16 frozen inputs are protected by the V17 integrity index. `FINAL_REPORT.md` is cumulative and intentionally append-only for this version.
- The required cumulative append invalidates two historical tests that compare the entire current `FINAL_REPORT.md` to old V14/V16 manifest hashes. They are not edited; filtered suite: 210 passed, 2 deselected. An initial run excluding only the V14 test gave 210 passed, 1 failed (V16 whole-file assertion), 1 deselected. V17 integrity independently verifies all other historical tracked bytes.
- Two generated prediction Parquets were losslessly repacked with Zstandard compression; complete table equality and sub-95 MB size were checked before replacement. This changes storage bytes, not response values or estimands.
