"""Invoke the identical frozen V34 mediation routine on an opened final panel."""
from pathlib import Path
import argparse
import json

from jclosure.experiments.mediate_v34 import run
from jclosure.protocol_v34 import verify_stage

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=("Q", "F"))
    args = parser.parse_args()
    verify_stage(Path.cwd(), "final_opening")
    print(json.dumps(run(Path.cwd(), args.model, "independent_final"), indent=2))
