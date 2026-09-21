# Execution Manifest — V26

- `PYTHONPATH=src /home/user/anaconda3/bin/python -m jclosure.protocol_v26 freeze`
- `PYTHONPATH=src /home/user/anaconda3/bin/python -m jclosure.experiments.design_v26`
- `CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src /home/user/anaconda3/bin/python -m jclosure.experiments.temporal_v26 floor`
- `CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src /home/user/anaconda3/bin/python -m jclosure.experiments.temporal_v26 bank`
- `CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src /home/user/anaconda3/bin/python -m jclosure.experiments.temporal_v26 early`
- `PYTHONPATH=src /home/user/anaconda3/bin/python -m jclosure.experiments.analyze_v26 early`
- `CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src /home/user/anaconda3/bin/python -m jclosure.experiments.temporal_v26 extension`
- `PYTHONPATH=src /home/user/anaconda3/bin/python -m jclosure.experiments.analyze_v26 full`
- `CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src /home/user/anaconda3/bin/python -m jclosure.experiments.temporal_v26 final`
- `PYTHONPATH=src /home/user/anaconda3/bin/python -m jclosure.experiments.analyze_v26 adjudicate`
- `PYTHONPATH=src /home/user/anaconda3/bin/python -m jclosure.experiments.diagnostics_v26`
- `PYTHONPATH=src /home/user/anaconda3/bin/python -m pytest tests/test_v26_temporal_readout_potency.py -q`
- `PYTHONPATH=src /home/user/anaconda3/bin/python -m pytest -q`
- `git commit -m 'Complete V26 temporal readout potency study'`
- `git push origin main`

V26 tests: `7 passed in 2.19s`. Full suite: `263 passed, 2 failed, 3 warnings in 58.10s (inherited V14/V16 FINAL_REPORT hash checks)`. The two failures are inherited V14/V16 cumulative FINAL_REPORT hash checks.
