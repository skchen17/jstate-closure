# Closure causal report

## Status

GATED / UNINTERPRETABLE. Fresh official-compatible pass@10 was 0.775862 for multihop (58 items) and 0.716797 for order of operations (256 items). The item-clustered MRR advantage was 0.135202, 95% CI [0.109490, 0.161532]. At frozen layer 24, the intended-answer log-odds effect was 3.064270, 95% CI [1.978771, 4.221590], above the 0.0001 null envelope.

Independent closure-layer calibration then tested layers 23, 24, 25, 26, 27, 28, 29. It obtained 0/1400 strictly valid clamp trials; the best layer-level valid rate was 0.000, below the frozen 0.80 requirement. All candidate layers therefore failed only the clamp-valid-rate criterion, and the eligible set is empty.

No formal closure, dictionary-size, final-token mediation, or sequence-state mediation
effect was estimated. In particular, there is no valid value of E_R, E_J, eta, future-J
divergence, output JS divergence, or answer effect. The 1,400 calibration attempts are
measurement/calibration evidence, not Phase 3 causal trials.

The observed failure must not be read as evidence for H1 or H2: the proposed
measured-J restoration did not meet dense-cosine and RMS-drift requirements while
retaining the requested remainder displacement. Changing the threshold or intervention
source after observing this result would require a separately frozen exploratory v3.

## Commands recorded by run manifests

- `/data/CSK/J-space-project/jstate-closure/src/jclosure/experiments/validate_lens_v2.py --config configs/phase0_v2_confirmatory.yaml`
- `/data/CSK/J-space-project/jstate-closure/src/jclosure/experiments/calibrate_layers.py --config configs/phase0_v2_confirmatory.yaml`
- `/data/CSK/J-space-project/jstate-closure/src/jclosure/experiments/closure.py --config configs/pilot_v2.yaml --limit 1`
- `/data/CSK/J-space-project/jstate-closure/src/jclosure/experiments/natural_collisions.py --config configs/confirm_v2.yaml --limit 1`
- `/data/CSK/J-space-project/jstate-closure/src/jclosure/experiments/memory_order.py --config configs/confirm_v2.yaml --limit 1 --epochs 1 --budget 1000000`
- `/data/CSK/J-space-project/jstate-closure/src/jclosure/experiments/distill_controller.py --config configs/confirm_v2.yaml --epochs 1`
- `/data/CSK/J-space-project/jstate-closure/src/jclosure/experiments/dictionary_sensitivity.py --config configs/confirm_v2.yaml`
- `/data/CSK/J-space-project/jstate-closure/src/jclosure/experiments/token_time_closure.py --config configs/confirm_v2.yaml --limit 1`
- `/data/CSK/J-space-project/jstate-closure/src/jclosure/experiments/modularity.py --config configs/confirm_v2.yaml`
