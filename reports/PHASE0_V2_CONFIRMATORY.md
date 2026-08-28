# Phase 0 v2 — Confirmatory J-lens validation

## Material Passport

- Protocol: `phase0_protocol_v2`
- Run ID: `phase0-v2-20260828T120721Z-d3f2fa00-s20260828`
- Verification Status: VERIFIED / LOCKED
- Exact command: `/data/CSK/J-space-project/jstate-closure/src/jclosure/experiments/validate_lens_v2.py --config configs/phase0_v2_confirmatory.yaml`
- Model: `Qwen/Qwen3.5-4B` at `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`
- Lens revision: `16a01f309fcec900fdcec3f4cd5b64f3d00e4d5a`
- Lens SHA-256: `1f9a8f8fd593f0ffec1a9640993257ca4560f8ae3e5602315643d5cc6818534e`
- Freeze manifest SHA-256: `d8b384c9b7f87c98a106d74dbd0cc70fff6e15a53bbd54890151a4b1bd731628`
- Readout records SHA-256: `73c500ac65e7520215eca17084c3a8452a0d01c8d3a4d595588e7273790a1066`

## Fresh confirmatory results

| Family | items | concepts | pass@1 | pass@5 | pass@10 | best layer | strict-all-layers hit@10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| multihop | 58 | 58 | 0.206897 | 0.655172 | 0.775862 | 25 | 0.000000 |
| order of operations | 256 | 418 | 0.183594 | 0.484375 | 0.716797 | 29 | 0.000000 |

Fresh official-compatible pass@10 was 0.775862 for multihop (58 items) and 0.716797 for order of operations (256 items). The item-clustered MRR advantage was 0.135202, 95% CI [0.109490, 0.161532]. At frozen layer 24, the intended-answer log-odds effect was 3.064270, 95% CI [1.978771, 4.221590], above the 0.0001 null envelope.

The primary metric includes all valid positions and flags copied concepts. The
`position>=16` sensitivity retained 25 multihop items and had
pass@10 0.720000. Copy-excluded sensitivity had multihop
pass@10 0.775862 and order-of-operations pass@10
0.640000. Coverage was `{"copy_flagged_concepts": 231, "official_main_copy_exclusions": 0, "official_main_examples": 314, "position16_sensitivity_examples": 25, "position_lt16_examples": 289, "raw_concepts": 571, "raw_examples": 315, "tokenization_excluded_concepts": 95}`.

## Frozen artifacts

Nested concept-dictionary hashes: `{"16384": "fe3dd2f8a90d54c12ed5ce1113ac6cac22c24c35a675384debb4b8ceba42a2bb", "4096": "2505f795001c9321d54f204acc2d7e2cb7a77185c25fa1426b5197ea5c618cb0", "8192": "506206834710863c1424650e5aa78e7ec70cae1e267f001413b634458e5accc0"}`.
The declared synonyms are official-compatible; they are not claimed to reproduce an
unpublished Anthropic internal synonym table item by item.

## Gate

**PASSED** under every frozen conjunctive criterion. The adjudication is locked;
post-confirmation protocol changes require a separately labeled exploratory v3.
The full per-layer curves and every item-level rank remain in the machine records.
