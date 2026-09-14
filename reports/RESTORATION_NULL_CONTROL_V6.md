# Restoration Null Control v6

## Material Passport

- Origin Skill: `academic-research-suite / experiment-agent`
- Mode: `restoration-artifact control`
- Date: `2026-09-15`
- Verification Status: `repository-grounded; machine results cited below`
- Protocol: `peripheral_foundations_protocol_v6`
- causal/null freeze digest: `f62c55d21e5e220cd21ea81c19a28bf1d5249a60bdc35a08e3eb029dd61ec12e`
- strong ceiling freeze digest: `6d4c77a3e33edf9e977e04293b73b77d153a15af091b1c496fe869e92ec23170`
- compact freeze digest: `07b51492b188b23da2ad66708cd3520e7b568a65bfd867ec019da7d301c24417`
- delivery/report freeze digest: `160d746a5b9b14e226acf876989e3196f971f321dbb9d2fb3720f7da68319840`


## Paired result

All persistent-null arms use the same layer list, position scope, dense projector, optimization implementation, and hook schedule as their corresponding persistent intervention, but replace the initial perturbation with the clean state. `E_restore_artifact` is their clean-relative effect. `corrected` is paired `E_persistent - E_restore_artifact`.

| family                  | E_single                      | E_persistent_final            | E_null_final                   | corrected_final               |   M_corrected_final | E_persistent_all              | E_null_all                     | corrected_all                 |   M_corrected_all |
|:------------------------|:------------------------------|:------------------------------|:-------------------------------|:------------------------------|--------------------:|:------------------------------|:-------------------------------|:------------------------------|------------------:|
| pooled                  | 0.002166 [0.000181, 0.005774] | 0.000167 [0.000020, 0.000434] | 0.000000 [-0.000000, 0.000000] | 0.000167 [0.000019, 0.000435] |            0.922975 | 0.000167 [0.000020, 0.000434] | 0.000000 [-0.000000, 0.000000] | 0.000167 [0.000019, 0.000435] |          0.922975 |
| boolean_logic           | 0.005749 [0.000325, 0.015569] | 0.000050 [0.000019, 0.000089] | 0.000000 [-0.000000, 0.000000] | 0.000050 [0.000019, 0.000089] |            0.991279 | 0.000050 [0.000019, 0.000089] | 0.000000 [-0.000000, 0.000000] | 0.000050 [0.000019, 0.000089] |          0.991279 |
| simple_state_transition | 0.000285 [0.000084, 0.000642] | 0.001587 [0.000035, 0.004309] | 0.000000 [-0.000000, 0.000000] | 0.001587 [0.000033, 0.004313] |           -4.5664   | 0.001587 [0.000035, 0.004309] | 0.000000 [-0.000000, 0.000000] | 0.001587 [0.000033, 0.004313] |         -4.5664   |
| modular_arithmetic      | 0.000074 [0.000030, 0.000137] | 0.000004 [0.000001, 0.000009] | 0.000000 [-0.000000, 0.000000] | 0.000004 [0.000001, 0.000009] |            0.94384  | 0.000004 [0.000001, 0.000009] | 0.000000 [-0.000000, 0.000000] | 0.000004 [0.000001, 0.000009] |          0.94384  |

The pooled single effect is `0.002166 [0.000181, 0.005774]`; corrected persistent effect is `0.000167 [0.000019, 0.000435]`, a point-estimate reduction of **92.30%**. Boolean logic reduces from `0.005749 [0.000325, 0.015569]` to `0.000050 [0.000019, 0.000089]` (**99.13%**). This supports mediation by later measured-J writes for the pooled/Boolean effects. The six-item state-transition family is not interpretable as mediation because its single-effect lower CI does not clear the frozen `1e-4` floor and persistent restoration amplifies its effect.

Mediation ratios are descriptive only when the family-specific single effect clears the preregistered noise requirement. The clean-state restoration null is essentially numerical zero, but it cannot exclude a state-dependent projector distortion that appears only after a real perturbation.

Figure: `results/v6/figures/restoration_null_v6.png`.
