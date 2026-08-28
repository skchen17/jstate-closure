# Phase 0 Protocol Audit

## Material Passport

- Protocol: `phase0_protocol_v2`
- Verification Status: ANALYZED
- Role: retrospective calibration only; not confirmatory adjudication
- Machine record: `results/processed/phase0_protocol_audit.json`

## Why the old gate failed

The v1 family gate used the minimum per-layer hit@10, excluded positions below 16,
and ranked only literal target strings. The factorial and Shapley values below
quantify how each protocol choice changes the old-data pass@10 statistic; they do
not establish that the frozen fresh-data gate will pass.

### factual_two_hop

- v1 reported hit@10: `0.03571428571428571`
- factorial pass@10: `{'v1_like': 0.0, 'synonym_expansion': 0.0, 'all_positions': 0.0, 'all_positions+synonym_expansion': 0.0, 'official_aggregation': 0.48148148148148145, 'official_aggregation+synonym_expansion': 0.48148148148148145, 'official_aggregation+all_positions': 0.5575396825396824, 'official_aggregation+all_positions+synonym_expansion': 0.5575396825396824}`
- Shapley attribution: `{'official_aggregation': 0.519510582010582, 'all_positions': 0.038029100529100496, 'synonym_expansion': 0.0}`
- corrected retrospective pass@10: `0.5575396825396824`
- residual readout-failure fraction: `0.44246031746031755`
- coverage: `{'raw_examples': 93, 'raw_concepts': 103, 'tokenization_excluded_concepts': 9, 'position_lt16_examples': 50, 'copy_flagged_concepts': 5}`

### order_of_operations

- v1 reported hit@10: `0.0`
- factorial pass@10: `{'v1_like': 0.0, 'synonym_expansion': 0.0, 'all_positions': 0.0, 'all_positions+synonym_expansion': 0.0, 'official_aggregation': 0.16666666666666666, 'official_aggregation+synonym_expansion': 0.3333333333333333, 'official_aggregation+all_positions': 0.39090909090909093, 'official_aggregation+all_positions+synonym_expansion': 0.7181818181818181}`
- Shapley attribution: `{'official_aggregation': 0.4156565656565656, 'all_positions': 0.16565656565656564, 'synonym_expansion': 0.13686868686868683}`
- corrected retrospective pass@10: `0.7181818181818181`
- residual readout-failure fraction: `0.28181818181818186`
- coverage: `{'raw_examples': 55, 'raw_concepts': 110, 'tokenization_excluded_concepts': 1, 'position_lt16_examples': 52, 'copy_flagged_concepts': 53}`

## Interpretation boundary

All values in this report reuse calibration data already inspected under v1.
Only `PHASE0_V2_CONFIRMATORY.md` may adjudicate the repaired measurement gate.
