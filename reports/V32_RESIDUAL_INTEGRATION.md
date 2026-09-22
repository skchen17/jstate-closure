# Residual Integration — V32

The diagnostic `post_residual_mlp_normalized_input` patch replaces the post-attention layernorm *output* entering the MLP; it does not replace the entire residual sum.

| role | component | median removed | median restored | remove cosine | restore cosine |
|---|---|---|---|---|---|
| development | post_residual_mlp_normalized_input | 0.156 | 0.126 | 0.398 | 0.407 |
| validation | post_residual_mlp_normalized_input | 0.166 | 0.104 | 0.376 | 0.364 |

Because the residual skip path and recurrent cache update remain separate, this cannot adjudicate whether correction is instantiated at residual addition. V32-G remains unconfirmed.
