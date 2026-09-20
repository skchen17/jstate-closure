from pathlib import Path

import torch

from jclosure.experiments import geometry_v13 as geometry

directions = torch.load(Path.cwd() / geometry.DIRECTIONS, map_location="cpu", weights_only=False)
for index in (1, 2, 4, 7, 11, 14, 24, 39):
    print(index, {key: float(torch.linalg.vector_norm(directions[key][index].float()))
                  for key in ("recurrent", "conv", "kv")})
