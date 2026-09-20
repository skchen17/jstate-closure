# V19 strict interface and historical-record audit

All 13120 validation rows have exact zero boundary-held-J difference and identical boundary J SHA256. All P0/Pq persistent snapshot SHA256 differ. This equality is causal by construction after the current layer-23 J readout; it is not natural same-J matching. V18 live-versus-frozen J replay drift is not hidden: V19 holds the live boundary J once and uses frozen V18 raw scores only for the separate predictor comparison.

No V1–V18 frozen file is overwritten. V19 freezes are append-only: base, split, initial q, q amendments 1/2, teacher amendment 3, analysis, diagnostics, GPU1 runtime amendments 4/5, unchanged execution-wrapper binding 6, and reverse-order GPU1 train scheduling amendment 7 (`7fe85a5bf5b8b7a141dbfc18fa027ba42cb82d81d9014e2e606a85460fc87962`). The same-state four-endpoint GPU0/GPU1 comparison had maximum relative L2 0.0. Each amendment records reason, creation UTC, observed-response counts and digest.

Full-repository pytest baseline: 222 passed, 2 historical integrity tests failed before V19 appended to FINAL_REPORT. V14 and V16 manifests hash an earlier cumulative FINAL_REPORT byte state; HEAD already contains subsequent cumulative sections. V19 does not rewrite those historical manifests or report sections. V18/V19 focused tests pass.

GPU1 processed the frozen train list in reverse order and was intentionally interrupted at 396/400 completed state files, before any state overlap. GPU0 completed the remaining three states; the per-state writer only commits complete Parquet files.

Replay-and-J-restored counterfactual was not run. V4/V6 restoration projects later hidden activations toward clean dense J under finite cosine/top-10/RMS tolerances, not exact equality of the V19 layer-23 current J after earlier persistent perturbation. V7 proves exact clone/one-token cache continuation, not replay-and-J-restored causal isolation. Thus the old machinery does not certify an artifact-free secondary V19 branch; it cannot contaminate the primary construction.

Task-sign is not pooled as a primary endpoint: the frozen V13/V18 five-family bank has no single validated shared signed semantic readout across Boolean, arithmetic, graph, state-transition and binding tasks. Selected J, logits, continuous semantic scores, workspace and V16-normalized stack are measured; no task-sign effect is fabricated from selected logits.

The current-J, persistent P, context perturbation q, probe a, natural N and modulation M interfaces remain separate. Independent final is unopened. H2 remains; H3 and autonomous replacement are not authorized.
