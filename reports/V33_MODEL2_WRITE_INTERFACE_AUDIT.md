# V33 Model-2 Native Write Interface Audit

Exact REC2, CONV2, REC2+CONV2, KV, and full-field donor transplants passed all-layer tensor equality checks. Every nonselected field remained bitwise recipient-native; shared KV prefix remained identical and only the appended slot was replaceable. The 24-layer schema records layer class, tensor shape, dtype, device, and SHA-256 for each field. The audit used one calibration prompt and two natural current tokens (`549`, `537`), without measuring future causal response.

|condition|transplanted fields|exact requested/untouched|
|---|---:|---|
|CONV2|24|True/True|
|KV|48|True/True|
|REC2|24|True/True|
|REC2+CONV2|48|True/True|
|REC2+CONV2+KV|96|True/True|
|RECIPIENT|0|True/True|

Native partial transplant—not a full-cache identity copy—is the primary mechanism test. `architecture_fields_v33.parquet`, `architecture_audit_v33.json` and frozen mapping digest make this auditable. Machine evidence: `results/v33/processed/`, immutable stage freezes in `artifacts/cross_model_rec_conv_v33*.freeze.json`.
