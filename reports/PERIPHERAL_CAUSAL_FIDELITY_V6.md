# Peripheral Causal Fidelity v6

## Material Passport

- Origin Skill: `academic-research-suite / experiment-agent`
- Mode: `corrective causal-fidelity measurement`
- Date: `2026-09-15`
- Verification Status: `repository-grounded; machine results cited below`
- Protocol: `peripheral_foundations_protocol_v6`
- causal/null freeze digest: `f62c55d21e5e220cd21ea81c19a28bf1d5249a60bdc35a08e3eb029dd61ec12e`
- strong ceiling freeze digest: `6d4c77a3e33edf9e977e04293b73b77d153a15af091b1c496fe869e92ec23170`
- compact freeze digest: `07b51492b188b23da2ad66708cd3520e7b568a65bfd867ec019da7d301c24417`
- delivery/report freeze digest: `160d746a5b9b14e226acf876989e3196f971f321dbb9d2fb3720f7da68319840`


## Result

The repaired endpoint contains **66** paired interventions. It measures the immediate same-forward write from the layer-23 intervention to layer 24, rather than the structurally weak next-token/layer-23 endpoint used in v5. All hidden states, measured-J profiles, deltas, logits, and operational remainders are stored as float32.

- non-zero next-write delta: **1.000** in float32 and **1.000** with float64 projection arithmetic;
- delta L2: min **0.012004215**, median **0.071335446**, max **0.107683778**;
- float64-projection median L2: **0.071335440**;
- median float32/float64 direction cosine: **1.000000002**.

## What failed in v5

The attribution is more specific than “float16 erased a valid signal.” In the old artifact, **2/66** next-token/layer-23 deltas survived float16, and the float32 v6 trajectory audit likewise finds only **2/66** material divergences at that old endpoint (`J-distance > 1e-6`). Thus the dominant problem was endpoint geometry/causal masking: a layer-23 final-token edit normally cannot alter the next token's layer-23 state unless it first changes the emitted token. Float16 made sub-quantization changes unavailable, but does not explain the 64 zeros by itself. Separately, reusing the float16 candidate for v5 mediation introduced median candidate quantization L2 **0.035538**, large enough to matter relative to the repaired median layer-24 delta **0.071335**.

The float64 check isolates projection arithmetic only—the source activations remain float32—and is not a claim that model inference ran in float64.

## Family counts

|                         |   n |
|:------------------------|----:|
| boolean_logic           |  24 |
| modular_arithmetic      |  12 |
| short_graph_traversal   |  12 |
| simple_state_transition |   6 |
| variable_binding        |  12 |

## Saved evidence

- Endpoint records: `results/v6/processed/causal_endpoint_precision_v6.parquet`
- Full scientific tensor artifact (uncommitted by policy): `artifacts/causal/v6/causal-endpoint-v6-20260914T161620Z-a20b5a8c-s20260828-full-66/causal_endpoint_f32.npz`
- Artifact SHA-256: `af945e0432cd7b1f1f256f1f0c60c91ec7d9aca7956dae745387247732c0444e`
- Figure: `results/v6/figures/causal_delta_precision_v6.png`
