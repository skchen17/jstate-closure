# V21 exact-JVP and finite second-order oracle ceilings

Z4 is the exact same-state, same-action JVP response on the 288-D normalized target; it is a target-side diagnostic oracle, not a deployable action coordinate. Z5 adds the finite central second difference measured at **half** the target action amplitude, so the full-amplitude response is never used as a Z5 feature. Z5 is not an exact Hessian.

| oracle diagnostic | validation unseen-direction positive L2 | negative L2 |
|---|---:|---:|
| Z4 direct JVP | 0.8062 | 0.9285 |
| Z5 direct Taylor D+Q/2 | 0.7922 | 0.7825 |
| Z5 train-fitted two-scalar map | 0.4688 | 0.5342 |
| Z4 best state-conditioned G4/G5 | 0.6955 | see machine record |
| Z5 S2 best state-conditioned G4/G5 | 0.6492 | see machine record |

Half-step actuator reliability: development **1.0000**, validation **1.0000**. Any apparent oracle improvement does not establish a practical action representation or justify opening the final six. The same-action derivative/second-difference feature must never be mixed with Z0–Z3 deployment claims.
