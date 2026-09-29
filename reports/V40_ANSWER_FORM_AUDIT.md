# V40 — Answer-Form Calibration Audit

Calibration only. The two readouts compare answer logits immediately after
the prompt versus after one natural space token. All numbers below are
computed from saved calibration records; no development or held-out outcomes.

| Model | Family | Both native branches correct: direct | After one space |
|---|---|---:|---:|
| Q | boolean_logic | 2 | 2 |
| Q | long_context_dependency | 3 | 4 |
| Q | state_transition | 3 | 4 |
| Q | variable_binding | 4 | 4 |
| F | boolean_logic | 0 | 4 |
| F | long_context_dependency | 0 | 0 |
| F | state_transition | 0 | 0 |
| F | variable_binding | 0 | 0 |

| Model | Total direct | Total after one space |
|---|---:|---:|
| Q | 12 | 14 |
| F | 0 | 4 |

Answer formatting changes apparent native competence. Falcon's
Boolean family improves after a space, but its other families have no
both-correct calibration pairs under either readout. This is a scoring
diagnostic, not evidence for a channel role or a model deficit in general.
The formal answer readout and control design must be frozen before
development outcomes are observed.
