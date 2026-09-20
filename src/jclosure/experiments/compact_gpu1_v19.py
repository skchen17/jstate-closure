"""V19 gated compact raw extraction on numerically audited GPU1."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from jclosure.experiments import compact_response_v19 as compact
from jclosure.experiments import counterfactual_bank_v19 as bank
from jclosure.experiments.gpu1_v19 import load_context_gpu1
from jclosure.protocol_v19 import verify_stage


def main():
    if sys.argv[1:] != ["extract_validation"]:
        raise RuntimeError("V19 compact GPU1 wrapper is validation extraction only")
    root = Path.cwd()
    design = verify_stage(root, "compact_response")
    runtime = verify_stage(root, "runtime_gpu1_amendment_4")
    if design["gpu1_validation_extraction_runtime_digest"] != runtime["freeze_digest"] or not runtime["equivalence_passed"]:
        raise RuntimeError("V19 compact GPU1 runtime binding invalid")
    bank.load_context_v18 = load_context_gpu1
    print(json.dumps(compact.extract(root, "validation"), indent=2))


if __name__ == "__main__":
    main()
