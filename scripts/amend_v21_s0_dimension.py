"""Append-only V21 correction of S0 declared width before model fitting."""

import json
from pathlib import Path

from jclosure.protocol_v21 import stage_freeze, verify_stage
from jclosure.provenance import sha256_file


def main() -> None:
    root = Path.cwd()
    prior = verify_stage(root, "state_representations")
    summary = json.loads((root / "results/v21/processed/state_representations_v21.json").read_text())
    if summary["dimensions"]["S0"] != 4096:
        raise RuntimeError("V21 S0 observed width differs from amendment")
    result = stage_freeze(root, "s0_dimension_amendment_1",
                          ["scripts/amend_v21_s0_dimension.py",
                           "artifacts/action_coordinate_geometry_v21_state_representations.freeze.json",
                           "results/v21/processed/state_representations_v21.json"],
                          {"reason": "Base config labeled S0 as 128 selected J coordinates, but frozen V20 state metadata contains the complete 4096-dimensional boundary J. Use full J as the stronger J-only ceiling.",
                           "original_declared_S0": "current_boundary_J_128",
                           "corrected_operational_S0": "full_current_boundary_J_4096",
                           "S0_feature_scratch_sha256": prior["state_feature_scratch_sha256"],
                           "summary_sha256": sha256_file(root / "results/v21/processed/state_representations_v21.json"),
                           "model_fits_already_run": 0,
                           "V21_new_JVP_or_finite_responses_observed": 0,
                           "final_six_action_responses_opened": False})
    print(result["freeze_digest"])


if __name__ == "__main__":
    main()
