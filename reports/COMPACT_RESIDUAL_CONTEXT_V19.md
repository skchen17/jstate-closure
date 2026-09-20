# V19 gated compact response-residual search

Search authorized by the frozen M gate and frozen at `bf2e2ce77c5f6fdf1a0b673385a47a0d18bd1f1ec6fe7840172fd3a2c039159d`. The 384D raw descriptor is a direct CountSketch of the actual BF16 post-q minus P0 REC/Conv/KV cache state. It is a restricted raw measurement, not complete P_raw; C is trained against the same-J modulation M and is not called a model state.

Train-only supervised residual PLS over eight signed development actions proposes k∈{4,8,16,32,64,128}. Fixed ridge state×continuous-action models compare J+C+a with J+C+raw-sketch+a on held-out base states, task families and four held-out action directions.

| k | rank supported | dev stack rel-L2 | dev J rel-L2 | held-out-action stack rel-L2 | dev states | held-out states |
|---|---|---|---|---|---|---|
| 4 | True | 0.8642 | 0.8744 | 1.1204 | 100 | 15 |
| 8 | True | 0.8601 | 0.8697 | 1.1157 | 100 | 15 |
| 16 | True | 0.8594 | 0.8675 | 1.1149 | 100 | 15 |
| 32 | True | 0.8590 | 0.8670 | 1.1154 | 100 | 15 |
| 64 | True | 0.8585 | 0.8666 | 1.1148 | 100 | 15 |
| 128 | True | 0.8584 | 0.8665 | 1.1140 | 100 | 15 |

Effective supervised PLS rank=288; selected k=128. Conditional raw-sketch stack gains: development 0.1655, held-out actions -0.3615. Family results: {'boolean_logic': {'compact': {'j_relative_l2': 0.8764923810958862, 'rows': 1472, 'stack_relative_l2': 0.8690159445014474}, 'plus_raw_sketch': {'j_relative_l2': 0.7609157562255859, 'rows': 1472, 'stack_relative_l2': 0.7065411260076258}, 'raw_incremental_stack_gain': 0.16247481849382162}, 'modular_arithmetic': {'compact': {'j_relative_l2': 0.8491957783699036, 'rows': 1472, 'stack_relative_l2': 0.8562121362105605}, 'plus_raw_sketch': {'j_relative_l2': 0.7249510884284973, 'rows': 1472, 'stack_relative_l2': 0.7389465328903835}, 'raw_incremental_stack_gain': 0.11726560332017699}, 'short_graph_traversal': {'compact': {'j_relative_l2': 0.8645460605621338, 'rows': 1472, 'stack_relative_l2': 0.8532698265811037}, 'plus_raw_sketch': {'j_relative_l2': 0.7319926023483276, 'rows': 1472, 'stack_relative_l2': 0.6838076284059259}, 'raw_incremental_stack_gain': 0.16946219817517783}, 'simple_state_transition': {'compact': {'j_relative_l2': 0.8850002288818359, 'rows': 1472, 'stack_relative_l2': 0.8465645440356223}, 'plus_raw_sketch': {'j_relative_l2': 0.7513205409049988, 'rows': 1472, 'stack_relative_l2': 0.6863984415828062}, 'raw_incremental_stack_gain': 0.16016610245281615}, 'variable_binding': {'compact': {'j_relative_l2': 0.8765368461608887, 'rows': 1472, 'stack_relative_l2': 0.8828865185898146}, 'plus_raw_sketch': {'j_relative_l2': 0.7673096656799316, 'rows': 1472, 'stack_relative_l2': 0.7479497807159076}, 'raw_incremental_stack_gain': 0.13493673787390703}}.

Restricted proxy gate passed: **False**. Independent final eligible: **False**. Reason: 384D raw CountSketch is not full persistent P_raw and C encodes Pq-P0 contrast rather than an autonomous state; the frozen search alone cannot certify response or dynamical sufficiency. The test does not certify full-raw sufficiency, compact dynamical state, H3 or autonomous control.
