# Natural versus Artificial Coordinates — V30

On the predeclared 10-state/role Conv-profile panel, the TRAIN-only global k64 approximation was compared with exact natural REC+Conv, same-norm random tensor, shuffled coordinate, sign flip, wrong-source-state coordinate, wrong-token coordinate and α=.5/1.5 amplitudes. Random and shuffled reconstructions are deliberately off-manifold controls; their failure cannot adjudicate natural-write existence.

| role | condition | cosine | magnitude | relative L2 |
|---|---|---|---|---|
| development | EXACT_REC_CONV | 0.950 | 0.970 | 0.314 |
| development | GLOBAL_k64 | 0.805 | 0.825 | 0.596 |
| development | RANDOM_SAME_NORM | 0.073 | 3.359 | 3.279 |
| development | SHUFFLED_COORDINATE | 0.493 | 0.576 | 0.874 |
| development | SIGN_FLIPPED | 0.112 | 0.466 | 1.073 |
| development | WRONG_STATE_COORDINATE | 0.808 | 0.813 | 0.590 |
| development | WRONG_TOKEN_COORDINATE | 0.376 | 0.756 | 0.946 |
| development | AMPLITUDE_0_5 | 0.701 | 0.711 | 0.715 |
| development | AMPLITUDE_1_5 | 0.851 | 0.973 | 0.544 |
| validation | EXACT_REC_CONV | 0.978 | 0.969 | 0.212 |
| validation | GLOBAL_k64 | 0.863 | 0.794 | 0.508 |
| validation | RANDOM_SAME_NORM | 0.163 | 3.557 | 3.570 |
| validation | SHUFFLED_COORDINATE | 0.560 | 0.737 | 0.887 |
| validation | SIGN_FLIPPED | 0.098 | 0.513 | 1.081 |
| validation | WRONG_STATE_COORDINATE | 0.861 | 0.797 | 0.512 |
| validation | WRONG_TOKEN_COORDINATE | 0.545 | 0.858 | 0.846 |
| validation | AMPLITUDE_0_5 | 0.732 | 0.728 | 0.683 |
| validation | AMPLITUDE_1_5 | 0.886 | 0.966 | 0.471 |

Structured but inadequate global coordinates outperform random/shuffled/wrong-token controls. Wrong-state global coordinates being similar to target global coordinates does not establish causally adequate transport. Amplitude and sign do not exhibit a qualified linear control law; no per-coordinate semantic label is assigned.
