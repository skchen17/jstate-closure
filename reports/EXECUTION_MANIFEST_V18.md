# V18 execution manifest

Working directory: `/data/CSK/J-space-project/jstate-closure`; model cache: `/data/CSK/J-space-project/.hf-cache`. Commands below record the core stages (with `HF_HOME` set to that model cache for model-loading stages). Freeze-creation commands are historical and intentionally fail if rerun on the same already-frozen workspace.

```bash
PYTHONPATH=src /home/user/anaconda3/bin/python -m jclosure.protocol_v18 freeze
PYTHONPATH=src /home/user/anaconda3/bin/python -m jclosure.experiments.crossed_bank_v18 amend_actions
PYTHONPATH=src /home/user/anaconda3/bin/python -m jclosure.experiments.crossed_bank_v18 prepare
PYTHONPATH=src /home/user/anaconda3/bin/python -m jclosure.experiments.state_features_v18 extract
PYTHONPATH=src /home/user/anaconda3/bin/python -m jclosure.experiments.state_features_v18 kernels
PYTHONPATH=src /home/user/anaconda3/bin/python -m jclosure.experiments.state_features_v18 scores
PYTHONPATH=src /home/user/anaconda3/bin/python -m jclosure.experiments.crossed_bank_v18 amend_state_layer
PYTHONPATH=src /home/user/anaconda3/bin/python -m jclosure.experiments.crossed_bank_v18 amend_reference_alignment
PYTHONPATH=src /home/user/anaconda3/bin/python -m jclosure.experiments.crossed_bank_v18 repair_train
PYTHONPATH=src /home/user/anaconda3/bin/python scripts/build_v18_amendment_archive.py
PYTHONPATH=src /home/user/anaconda3/bin/python -m jclosure.experiments.crossed_bank_v18 amend_runtime_placement
PYTHONPATH=src /home/user/anaconda3/bin/python -m jclosure.experiments.crossed_bank_v18 amend_teacher_drift
PYTHONPATH=src /home/user/anaconda3/bin/python -m jclosure.experiments.crossed_bank_v18 train
PYTHONPATH=src /home/user/anaconda3/bin/python -m jclosure.experiments.crossed_bank_v18 validation
PYTHONPATH=src /home/user/anaconda3/bin/python -m jclosure.experiments.crossed_bank_v18 aggregate_train
PYTHONPATH=src /home/user/anaconda3/bin/python -m jclosure.experiments.crossed_bank_v18 aggregate_validation
PYTHONPATH=src /home/user/anaconda3/bin/python -m jclosure.experiments.crossed_bank_v18 audit_teacher
PYTHONPATH=src /home/user/anaconda3/bin/python -m jclosure.experiments.strong_ceiling_v18 amend_model_binding
PYTHONPATH=src /home/user/anaconda3/bin/python -m jclosure.experiments.strong_ceiling_v18 h1
PYTHONPATH=src /home/user/anaconda3/bin/python -m jclosure.experiments.strong_ceiling_v18 horizons
PYTHONPATH=src /home/user/anaconda3/bin/python -m jclosure.experiments.history_v18 amend_history_array
PYTHONPATH=src /home/user/anaconda3/bin/python -m jclosure.experiments.history_v18 scores
PYTHONPATH=src /home/user/anaconda3/bin/python -m jclosure.experiments.history_v18 evaluate
PYTHONPATH=src /home/user/anaconda3/bin/python -m jclosure.experiments.strict_match_v18 run
PYTHONPATH=src /home/user/anaconda3/bin/python -m jclosure.experiments.decision_v18 run
PYTHONPATH=src /home/user/anaconda3/bin/python -m pytest tests/test_v18.py -q
PYTHONPATH=src /home/user/anaconda3/bin/python scripts/build_v18_reports.py
PYTHONPATH=src /home/user/anaconda3/bin/python scripts/build_v18_integrity.py
PYTHONPATH=src /home/user/anaconda3/bin/python scripts/build_complete_version_report.py V18
```

Bank roles: train 2000 states, validation 400 states; new independent final remains unopened.
Frozen action coordinates: [5, 1, 7, 20, 11, 19, 17, 2]; protocol/split/model hashes: `989a0cfa8f50cd269d7d35dd12f3eb3f4a71179fe187defabacf5079bad41316` / `5095039933f344b068534c92b1cff26252018b8d3bbbc4c60544830a4b280f9d` / `776f61ca1a94579b2f0fe7a8187f6ee6402a209af37ba3584b59097d1ce227b7`.
Formal result: **V18-STOP — STRICT_MATCH_NOT_IDENTIFIED_AND_NO_MATERIAL_RAW_CEILING**. Compact search: **COMPACT_CONTEXT_SEARCH_NOT_AUTHORIZED**.
