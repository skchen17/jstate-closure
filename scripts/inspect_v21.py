from pathlib import Path
import json
import torch

root = Path.cwd()
data = torch.load(root / 'artifacts/causal/v13/probe_directions_v13.pt', map_location='cpu', weights_only=False)
print(json.dumps({
    'score_directions_shape': list(data['score_directions'].shape),
    'channel_shapes': {key: list(data[key].shape) for key in ('recurrent', 'conv', 'kv')},
    'label_counts': {key: data['labels'].count(key) for key in sorted(set(data['labels']))},
    'selected_score_rows': {str(i): data['score_directions'][i].tolist()[:8] for i in (4, 24, 34)},
}, indent=2))
