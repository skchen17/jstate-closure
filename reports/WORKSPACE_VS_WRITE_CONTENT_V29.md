# Workspace Versus Write Content — V29

Train-only ridge models predict the h1 response contrast on 50 held-out validation states. `J_t` here is the layer-30 current workspace readout, **not** a cache field. Exact token identity/surface, J, broader current workspace, and per-field write geometry are compared diagnostically.

| diagnostic model | held-out validation R² |
|---|---:|
| M0_token_identity_and_surface | 0.417 |
| M1_current_J | 0.391 |
| M2_token_plus_J | 0.423 |
| M3_token_J_broader_current_workspace | 0.482 |
| M4_token_J_write_geometry | 0.488 |

Current J predicts coarse write-geometry features with held-out R² `0.926`. Adding write geometry to token+J improves held-out response R², but predictive gain does not prove current J causally insufficient for the full outgoing write. A high-fidelity direct J transplant/J-matched causal equivalence test was not established; V29-H is not confirmed.
