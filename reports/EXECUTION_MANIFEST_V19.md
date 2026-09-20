# V19 execution manifest

Working directory `/data/CSK/J-space-project/jstate-closure`. Commands are listed in execution order; freeze commands are append-only and intentionally fail if rerun. Model loading uses the existing V18 single-GPU0 placement and frozen V13 weights.

```bash
PYTHONPATH=src /home/user/anaconda3/bin/python -m jclosure.protocol_v19 freeze
PYTHONPATH=src /home/user/anaconda3/bin/python -m jclosure.experiments.counterfactual_bank_v19 prepare
HF_HOME=/data/CSK/J-space-project/.hf-cache PYTHONPATH=src /home/user/anaconda3/bin/python -m jclosure.experiments.counterfactual_bank_v19 calibrate
PYTHONPATH=src /home/user/anaconda3/bin/python scripts/audit_q_vectors_v19.py
HF_HOME=/data/CSK/J-space-project/.hf-cache PYTHONPATH=src /home/user/anaconda3/bin/python -m jclosure.experiments.q_amendment_v19
PYTHONPATH=src /home/user/anaconda3/bin/python scripts/audit_selected_q_v19.py
HF_HOME=/data/CSK/J-space-project/.hf-cache PYTHONPATH=src /home/user/anaconda3/bin/python -m jclosure.experiments.q_support_amendment_v19
HF_HOME=/data/CSK/J-space-project/.hf-cache PYTHONPATH=src /home/user/anaconda3/bin/python -m jclosure.experiments.run_amended_v19 train --limit 1  # first attempt stopped before response write on short teacher sequence
HF_HOME=/data/CSK/J-space-project/.hf-cache PYTHONPATH=src /home/user/anaconda3/bin/python -m jclosure.experiments.teacher_amendment_v19
PYTHONPATH=src /home/user/anaconda3/bin/python -m jclosure.experiments.analyze_v19 prepare
HF_HOME=/data/CSK/J-space-project/.hf-cache PYTHONPATH=src /home/user/anaconda3/bin/python -m jclosure.experiments.run_amended_v19 train --limit 1  # successful smoke state
HF_HOME=/data/CSK/J-space-project/.hf-cache PYTHONPATH=src /home/user/anaconda3/bin/python -m jclosure.experiments.run_amended_v19 train
/home/user/anaconda3/bin/python scripts/inspect_v19_progress.py  # read-only progress snapshots, repeated
PYTHONPATH=src /home/user/anaconda3/bin/python -m jclosure.experiments.diagnostics_v19 prepare
HF_HOME=/data/CSK/J-space-project/.hf-cache PYTHONPATH=src /home/user/anaconda3/bin/python -m jclosure.experiments.compare_gpu_v19
HF_HOME=/data/CSK/J-space-project/.hf-cache PYTHONPATH=src /home/user/anaconda3/bin/python -m jclosure.experiments.gpu1_v19 validation
PYTHONPATH=src /home/user/anaconda3/bin/python -m jclosure.experiments.run_amended_v19 aggregate_validation
PYTHONPATH=src /home/user/anaconda3/bin/python -m jclosure.experiments.analyze_v19 analyze_validation
PYTHONPATH=src /home/user/anaconda3/bin/python -m jclosure.experiments.diagnostic_gpu1_v19 prepare
HF_HOME=/data/CSK/J-space-project/.hf-cache PYTHONPATH=src /home/user/anaconda3/bin/python -m jclosure.experiments.diagnostic_gpu1_v19 sign_scale
HF_HOME=/data/CSK/J-space-project/.hf-cache PYTHONPATH=src /home/user/anaconda3/bin/python -m jclosure.experiments.diagnostic_gpu1_v19 order
PYTHONPATH=src /home/user/anaconda3/bin/python -m jclosure.experiments.execution_binding_v19
PYTHONPATH=src /home/user/anaconda3/bin/python -m jclosure.experiments.gpu1_reverse_freeze_v19
HF_HOME=/data/CSK/J-space-project/.hf-cache PYTHONPATH=src /home/user/anaconda3/bin/python -m jclosure.experiments.gpu1_reverse_train_v19 train  # interrupted before state overlap; GPU0 completed train
PYTHONPATH=src /home/user/anaconda3/bin/python -m jclosure.experiments.run_amended_v19 aggregate_train
PYTHONPATH=src /home/user/anaconda3/bin/python -m jclosure.experiments.analyze_v19 analyze_train
PYTHONPATH=src /home/user/anaconda3/bin/python -m jclosure.experiments.analyze_v19 context
PYTHONPATH=src /home/user/anaconda3/bin/python -m jclosure.experiments.analyze_v19 decide
PYTHONPATH=src /home/user/anaconda3/bin/python -m jclosure.experiments.compact_response_v19 prepare
HF_HOME=/data/CSK/J-space-project/.hf-cache PYTHONPATH=src /home/user/anaconda3/bin/python -m jclosure.experiments.compact_response_v19 extract_train
HF_HOME=/data/CSK/J-space-project/.hf-cache PYTHONPATH=src /home/user/anaconda3/bin/python -m jclosure.experiments.compact_gpu1_v19 extract_validation
PYTHONPATH=src /home/user/anaconda3/bin/python -m jclosure.experiments.compact_response_v19 aggregate_train
PYTHONPATH=src /home/user/anaconda3/bin/python -m jclosure.experiments.compact_response_v19 aggregate_validation
PYTHONPATH=src /home/user/anaconda3/bin/python -m jclosure.experiments.compact_response_v19 search
PYTHONPATH=src /home/user/anaconda3/bin/python scripts/build_v19_reports.py
PYTHONPATH=src /home/user/anaconda3/bin/python scripts/build_v19_integrity.py
PYTHONPATH=src /home/user/anaconda3/bin/python scripts/build_complete_version_report.py V19
PYTHONPATH=src /home/user/anaconda3/bin/python -m pytest tests/test_v19.py -q
PYTHONPATH=src /home/user/anaconda3/bin/python -m pytest tests/test_v18.py -q
PYTHONPATH=src /home/user/anaconda3/bin/python -m pytest -q  # baseline: 222 passed, 2 older cumulative-hash checks fail
```

## Freeze digests

| stage | digest |
|---|---|
| protocol | 0e76a756936e08d71aeee7feffc6dfc8c8e18d572ecd47a29a3048404c843b1a |
| split | 9d50c11b81f98515fdb2191ec96167bae34e2908585e622084a3160c65503319 |
| initial q | 9d4b8f250fcf100cfcef4d9c3e8acfbf1e7f9bda3f6f108510aacf44271637c7 |
| q amendment 1 | 0e79c12a9d5aed4dce602d5694994afc46b89e9bbdab8fd5750d0a830093319b |
| q amendment 2 | 572a827f6fba915490e9262767a1178d1eff7a24b6a17cb55b24f0a69ed529bb |
| teacher amendment 3 | df10a2a96e423e7754cdbe3184eb366bbd40a4d5fef6dad5ee8e936ab53834c6 |
| analysis | 2080f9b2f5855522b9bffd3a424cbad6ad01dcf635397882c6e9e28095278f61 |
| diagnostics | ec73a234096b3165fa7f74e5ec977325c6cd559ba4df7f9e8da0d46440f4b1be |
| GPU1 runtime amendment 4 | a8300501b9402e859f87065cd5b1c008a87fa7f6fede7e701ef6139cfb8c241d |
| diagnostic GPU1 amendment 5 | c9e1da09054e4f950e1907aa64070ef77c7879a8a9291dd2b882feed0aded362 |
| execution binding amendment 6 | 2d8c16526c1a6fb7c8433cabd1dd08b0c42ce1a8ef8f19699a936b4d7bf3a0b8 |
| GPU1 reverse-train amendment 7 | 7fe85a5bf5b8b7a141dbfc18fa027ba42cb82d81d9014e2e606a85460fc87962 |
| compact response | bf2e2ce77c5f6fdf1a0b673385a47a0d18bd1f1ec6fe7840172fd3a2c039159d |

Role ID hashes: {'calibration': '420f8a288f32426002d7f15790a1018c3cf2d3d2a025fc347a7da5c3d72eeec7', 'train': '35ff396a35bb8c3db53510e660a9624fd98726539e63f1fb801aeca518c909c7', 'validation': '2066f27d4523df627eec6613c1827479d58edb3feb8e513655af1720b7dac135'}. q intervention hash: `572a827f6fba915490e9262767a1178d1eff7a24b6a17cb55b24f0a69ed529bb`. Teacher hash: `2143b1e795aa932aeea2ea94a09f5b368ca8449fcf5c89afb10b8604096e7aa3`. Four-way train/validation states 400/100; rows 48000/13120.
