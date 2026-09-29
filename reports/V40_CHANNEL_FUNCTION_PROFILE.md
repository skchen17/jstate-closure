# V40 — Channel Function Profile

Status: **calibration-only interventions recorded; formal development not run**.

For each externally labelled natural token fork, the frozen condition map is BASE, REC-only, Conv-only, KV-only, REC+Conv, and REC+Conv+KV. The recipient/donor pair differs in one encoded token, and labels are fixed by the task generator. The proposed primary comparison is recovery of the donor versus recipient task variable after native cache-channel exchange, not donor-response cosine. Donor-vector fidelity is secondary.

The sequential native-cache calibration and answer-form audits are in `V40_CALIBRATION_A3.md` and `V40_ANSWER_FORM_AUDIT.md`. They establish execution feasibility, not a channel role. Before formal development execution, an append-only design must freeze the answer readout and implement surface-only, irrelevant-history, same-prefix-mismatch, and shuffled-state controls. Validation and independent final must not tune this design.

Machine sources: `configs/channel_function_v40.yaml`, `data/v40/sample_pool_manifest_v40.json`, `results/v40/processed/design_Q_v40.json`, and `results/v40/processed/design_F_v40.json`.
