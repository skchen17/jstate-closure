# Execution Manifest — V28

- `PYTHONPATH=src /home/user/anaconda3/bin/python -m jclosure.protocol_v28 freeze`
- `PYTHONPATH=src /home/user/anaconda3/bin/python -m jclosure.experiments.design_v28`
- `CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src /home/user/anaconda3/bin/python -m jclosure.experiments.transaction_v28 pilot`
- `PYTHONPATH=src /home/user/anaconda3/bin/python -m jclosure.experiments.transaction_v28 plan`
- `CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src /home/user/anaconda3/bin/python -m jclosure.experiments.transaction_v28 audit`
- `PYTHONPATH=src /home/user/anaconda3/bin/python -m jclosure.experiments.interventions_v28 amend`
- `CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src /home/user/anaconda3/bin/python -m jclosure.experiments.interventions_v28 development`
- `PYTHONPATH=src /home/user/anaconda3/bin/python -m jclosure.experiments.analyze_v28 development`
- `CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src /home/user/anaconda3/bin/python -m jclosure.experiments.interventions_v28 validation`
- `PYTHONPATH=src /home/user/anaconda3/bin/python -m jclosure.experiments.analyze_v28 validation`
- `CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src /home/user/anaconda3/bin/python -m jclosure.experiments.final_v28`
- `CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src /home/user/anaconda3/bin/python -m jclosure.experiments.transplant_v28 development`
- `PYTHONPATH=src /home/user/anaconda3/bin/python -m jclosure.experiments.analyze_transplant_v28 development`
- `CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src /home/user/anaconda3/bin/python -m jclosure.experiments.transplant_v28 validation`
- `PYTHONPATH=src /home/user/anaconda3/bin/python -m jclosure.experiments.analyze_transplant_v28 validation`
- `CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src /home/user/anaconda3/bin/python -m jclosure.experiments.secondary_v28 layerwise development`
- `CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src /home/user/anaconda3/bin/python -m jclosure.experiments.secondary_v28 layerwise validation`
- `CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src /home/user/anaconda3/bin/python -m jclosure.experiments.secondary_v28 horizons development`
- `CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src /home/user/anaconda3/bin/python -m jclosure.experiments.secondary_v28 horizons validation`
- `PYTHONPATH=src /home/user/anaconda3/bin/python -m jclosure.experiments.wrong_write_v28 prepare`
- `CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src /home/user/anaconda3/bin/python -m jclosure.experiments.wrong_write_v28 development`
- `CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src /home/user/anaconda3/bin/python -m jclosure.experiments.wrong_write_v28 validation`
- `PYTHONPATH=src /home/user/anaconda3/bin/python -m jclosure.experiments.adjudicate_v28`
- `PYTHONPATH=src /home/user/anaconda3/bin/python -m pytest tests/test_v28_token_state_transaction.py -q`
- `PYTHONPATH=src /home/user/anaconda3/bin/python -m pytest -q`
- `git commit -m 'Complete V28 token state transaction study'`
- `git push origin main`

V28 tests: `6 passed in 1.73s`. Full suite: `274 passed, 3 failed, 3 warnings in 51.90s`. The three failures are inherited V14/V16/V25 cumulative `FINAL_REPORT.md` hash checks.
