# Gate Mediation — V32

Raw gate-a and gate-b projection traces vary under the factorial branches, but sigmoid/softplus gate values and recurrent-kernel update terms were not independently recorded. Diagnostic exact-output patches at the development-nominated layer are below. Component probes were specified after the primary factorial plan and cannot become formal V32-E evidence.

| role | component | median removed | median restored | remove cosine | restore cosine |
|---|---|---|---|---|---|
| development | gate_a | -0.001 | 0.001 | -0.011 | 0.033 |
| development | gate_b | -0.001 | -0.001 | -0.031 | 0.010 |
| validation | gate_a | -0.001 | 0.003 | -0.006 | 0.037 |
| validation | gate_b | 0.001 | -0.002 | 0.014 | 0.024 |

Patching a raw projection output can perturb downstream gate computation; it does not isolate a unique state-specific gate law. V32-E remains unconfirmed.
