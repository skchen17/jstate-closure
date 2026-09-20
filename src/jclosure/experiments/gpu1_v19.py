"""V19 GPU1 runtime after frozen same-state numerical equivalence audit."""

from __future__ import annotations

import copy
import sys
from pathlib import Path

from jclosure.experiments import actuation_v15 as actuation
from jclosure.experiments import counterfactual_bank_v19 as bank
from jclosure.experiments import run_amended_v19
from jclosure.model import load_model_bundle
from jclosure.protocol_v19 import verify_stage


def load_context_gpu1(root: Path):
    original = actuation._load_model

    def gpu1_loader(config):
        amended = copy.deepcopy(config)
        amended["model"]["device_map"] = {"": 1}
        amended["model"].pop("max_memory", None)
        amended["model"].pop("offload_folder", None)
        return load_model_bundle(amended)

    actuation._load_model = gpu1_loader
    try:
        return actuation.load_context(root)
    finally:
        actuation._load_model = original


def main():
    if sys.argv[1:2] != ["validation"]:
        raise RuntimeError("GPU1 amended runtime is frozen for validation only")
    runtime = verify_stage(Path.cwd(), "runtime_gpu1_amendment_4")
    if not runtime["equivalence_passed"]:
        raise RuntimeError("GPU1 equivalence audit failed")
    bank.load_context_v18 = load_context_gpu1
    run_amended_v19.main()


if __name__ == "__main__":
    main()
