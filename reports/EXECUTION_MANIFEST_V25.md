# Execution Manifest — V25

- `PYTHONPATH=src /home/user/anaconda3/bin/python -m jclosure.protocol_v25 freeze`
- `PYTHONPATH=src /home/user/anaconda3/bin/python -m jclosure.experiments.design_v25`
- `CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src /home/user/anaconda3/bin/python -m jclosure.experiments.component_basis_v25`
- `CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src /home/user/anaconda3/bin/python -m jclosure.experiments.rank_convergence_v25`
- `PYTHONPATH=src /home/user/anaconda3/bin/python -m jclosure.experiments.rank_convergence_amendment_v25`
- `CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src /home/user/anaconda3/bin/python -m jclosure.experiments.interaction_v25`
- `CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src /home/user/anaconda3/bin/python -m jclosure.experiments.interaction_estimand_amendment_v25`
- `PYTHONPATH=src /home/user/anaconda3/bin/python -m jclosure.experiments.analyze_v25`
- `PYTHONPATH=src /home/user/anaconda3/bin/python -m jclosure.experiments.zero_rank_amendment_v25`
- `PYTHONPATH=src /home/user/anaconda3/bin/python -m jclosure.experiments.adjudicate_v25`
- `PYTHONPATH=src /home/user/anaconda3/bin/python -m pytest tests/test_v25_distributed_interaction.py -q`
- `PYTHONPATH=src /home/user/anaconda3/bin/python -m pytest -q`
- `PYTHONPATH=src /home/user/anaconda3/bin/python -m jclosure.reporting_v25 --v25-tests '6 passed' --full-tests '256 passed, 2 inherited failures'`
- `git commit -m 'Complete V25 distributed causal interaction study'`
- `git push origin main`

V25 tests: `6 passed in 5.89s`. Full suite: `256 passed, 2 failed, 3 warnings in 77.35s (inherited V14/V16 FINAL_REPORT hash checks)`. The two full-suite failures are inherited V14/V16 cumulative `FINAL_REPORT.md` hash checks; no V25 test failed.
