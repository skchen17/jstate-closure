# Execution Manifest — V23

- `PYTHONPATH=src /home/user/anaconda3/bin/python -m jclosure.protocol_v23 freeze`
- `CUDA_VISIBLE_DEVICES=1 HF_HOME=/data/CSK/J-space-project/.hf-cache PYTHONPATH=src /home/user/anaconda3/bin/python -m jclosure.experiments.probe_design_v23 prepare`
- `CUDA_VISIBLE_DEVICES=1 HF_HOME=/data/CSK/J-space-project/.hf-cache PYTHONPATH=src /home/user/anaconda3/bin/python -m jclosure.experiments.probe_design_v23 calibrate`
- `PYTHONPATH=src /home/user/anaconda3/bin/python -m jclosure.experiments.probe_design_v23 select`
- `PYTHONPATH=src /home/user/anaconda3/bin/python -m jclosure.experiments.rank_measurement_v23 prepare`
- `CUDA_VISIBLE_DEVICES=1 HF_HOME=/data/CSK/J-space-project/.hf-cache PYTHONPATH=src /home/user/anaconda3/bin/python -m jclosure.experiments.rank_measurement_v23 jvp`
- `CUDA_VISIBLE_DEVICES=1 HF_HOME=/data/CSK/J-space-project/.hf-cache PYTHONPATH=src /home/user/anaconda3/bin/python -m jclosure.experiments.rank_measurement_v23 finite`
- `PYTHONPATH=src /home/user/anaconda3/bin/python -m jclosure.experiments.rank_measurement_v23 summarize`
- `PYTHONPATH=src /home/user/anaconda3/bin/python -m jclosure.experiments.analyze_v23 rank`
- `PYTHONPATH=src /home/user/anaconda3/bin/python -m jclosure.experiments.analyze_v23 oracle`
- `PYTHONPATH=src /home/user/anaconda3/bin/python -m jclosure.experiments.analyze_v23 adjudicate`
- `PYTHONPATH=src /home/user/anaconda3/bin/python -m pytest tests/test_v23_oracle_charts.py -q`
- `PYTHONPATH=src /home/user/anaconda3/bin/python -m pytest -q`
- `git add configs/oracle_local_action_charts_v23.yaml src/jclosure/protocol_v23.py src/jclosure/experiments/probe_design_v23.py src/jclosure/experiments/rank_measurement_v23.py src/jclosure/experiments/analyze_v23.py src/jclosure/reporting_v23.py tests/test_v23_oracle_charts.py artifacts/oracle_local_action_charts_v23*.json results/v23 reports/*.md`
- `git commit -m 'Complete V23 oracle local action charts'`
- `git push origin main`

V23 tests: `6 passed`. Full suite: `244 passed, 2 inherited failures`.
