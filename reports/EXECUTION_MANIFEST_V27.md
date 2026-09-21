# Execution Manifest — V27

- `PYTHONPATH=src /home/user/anaconda3/bin/python -m jclosure.protocol_v27 freeze`
- `PYTHONPATH=src /home/user/anaconda3/bin/python -m jclosure.experiments.design_v27`
- `CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src /home/user/anaconda3/bin/python -m jclosure.experiments.pre_readout_v27 boundary`
- `CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src /home/user/anaconda3/bin/python -m jclosure.experiments.pre_readout_v27 bank`
- `PYTHONPATH=src /home/user/anaconda3/bin/python -m jclosure.experiments.pre_readout_expansion_v27 freeze`
- `CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src /home/user/anaconda3/bin/python -m jclosure.experiments.pre_readout_expansion_v27 bank`
- `CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src /home/user/anaconda3/bin/python -m jclosure.experiments.adjudicate_failure_v27 timing`
- `PYTHONPATH=src /home/user/anaconda3/bin/python -m jclosure.experiments.adjudicate_failure_v27 adjudicate`
- `PYTHONPATH=src /home/user/anaconda3/bin/python -m pytest tests/test_v27_pre_readout_reentry.py -q`
- `PYTHONPATH=src /home/user/anaconda3/bin/python -m pytest -q`
- `git commit -m 'Complete V27 pre-readout causal re-entry study'`
- `git push origin main`

V27 tests: `6 passed in 1.76s`. Full suite: `268 passed, 3 failed, 3 warnings in 52.61s`. The three failures are inherited cumulative `FINAL_REPORT.md` hash checks in V14, V16, and V25.
