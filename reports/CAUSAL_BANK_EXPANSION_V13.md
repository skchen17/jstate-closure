# Causal bank expansion — V13

V13 created a new teacher-correct capture bank; neither V11 nor V12 confirmation was used for method selection.

- usable train / validation / independent final: `{'final_test': 250, 'train': 4800, 'validation': 500}`
- family counts: `{'final_test': {'boolean_logic': 50, 'modular_arithmetic': 50, 'short_graph_traversal': 50, 'simple_state_transition': 50, 'variable_binding': 50}, 'train': {'boolean_logic': 960, 'modular_arithmetic': 960, 'short_graph_traversal': 960, 'simple_state_transition': 960, 'variable_binding': 960}, 'validation': {'boolean_logic': 100, 'modular_arithmetic': 100, 'short_graph_traversal': 100, 'simple_state_transition': 100, 'variable_binding': 100}}`
- split hashes: `{'final_test': 'cc49220b05cb3609554e1f79811d718d4425b950af6b7b9a393174810cfb2360', 'train': 'ebd4be05da86d3a3450bab336858d9e1869d1ef76640dcda7a8d76ba2f445ac3', 'validation': 'bed58343642dd8ad11d47454d126b3bd8d1d9a14478896baf3da39952d45a613'}`
- capture freeze: `167f69a3b5ddfd95775728f77e46fd94e1e432dd7097e531d8e801bc7169422b`

The bank contains exact BF16 REC, convolution, and all observed KV cache fields, J endpoints, selected-logit endpoints, semantic log-odds endpoints, prompt/token metadata, pair IDs, and explicit exclusion counts in the split manifests.
