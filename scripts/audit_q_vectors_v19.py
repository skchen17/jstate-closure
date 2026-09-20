from pathlib import Path

import torch

from jclosure.experiments import geometry_v13 as geometry

root = Path.cwd()
directions = torch.load(root / geometry.DIRECTIONS, map_location="cpu", weights_only=False)
for channel in ("recurrent", "conv", "kv"):
    values = directions[channel].float().flatten(1)
    norms = torch.linalg.vector_norm(values, dim=1)
    indices = torch.topk(norms, k=min(20, len(norms))).indices.tolist()
    print(channel, [(int(index), float(norms[index]), str(directions["labels"][index])) for index in indices])
