"""Conservative V37 formal adjudication from independently sealed records."""
from __future__ import annotations

import json
from pathlib import Path

from jclosure.protocol_v37 import stage_freeze, verify, verify_stage
from jclosure.provenance import sha256_file, write_json_atomic


OUT=Path("results/v37/processed")
SOURCE="src/jclosure/experiments/adjudication_v37.py"


def _read(root,name):
    return json.loads((root/OUT/name).read_text(encoding="utf-8"))


def run(root:Path):
    verify(root)
    for role in ("development","validation"):
        for stage in (f"high_level_{role}",f"read_gate_{role}",f"local_analysis_{role}"):
            verify_stage(root,stage)
    verify_stage(root,"cross_layer_analysis_development")
    verify_stage(root,"final_opening")
    high={r:_read(root,f"high_level_{r}_v37.json") for r in ("development","validation")}
    read={r:_read(root,f"read_gate_{r}_v37.json") for r in high}
    local={r:_read(root,f"local_analysis_{r}_v37.json") for r in high}
    cross=_read(root,"cross_layer_analysis_development_v37.json")
    opening=_read(root,"final_opening_v37.json")
    final=None
    if opening["opened"]:
        verify_stage(root,"final_analysis")
        final=_read(root,"final_analysis_v37.json")
    local_both=all(local[r]["both_models_local_prediction_pass"] for r in local)
    local_final=bool(final and final["both_models_local_read_law_pass"])
    result={
        "version":"V37",
        "high_level_conditioned_effect_reconfirmed":all(high[r]["both_pass"] for r in high),
        "full_recurrent_read_interface_reconfirmed":all(read[r]["both_pass"] for r in read),
        "q2_q3_q4_secondary_replication":all(read[r]["secondary_triple_both_pass"] for r in read),
        "local_native_read_law_development_validation_pass":local_both,
        "local_native_read_law_independent_final_pass":local_final if final is not None else None,
        "cross_layer_operator_conditioning_development_pass":cross["both_models_cross_layer_pass"],
        "cross_layer_validation_opened":False,
        "cross_layer_mechanism_confirmed":False,
        "old_state_vs_update_causal_dominance":"NOT_ADJUDICATED",
        "post_read_transformation_causal_mediation":"NOT_ADJUDICATED",
        "task_grounded_functional_restoration":"NOT_TESTED",
        "final_opened":opening["opened"],
        "selected_class":opening["selected_class"],
        "strongest_supported_level":("LEVEL_2_LOCAL_COMPUTATION" if local_both and local_final else
                                     "LEVEL_1_CAUSAL_INTERFACE" if all(read[r]["both_pass"] for r in read) else
                                     "LEVEL_0_PHENOMENOLOGY"),
        "v36_formal_results_counted":False,
        "v18_training_source_validation_counted":False,
        "v18_training_source_final_counted":False,
        "historical_v36_modified":False,
        "scope_note":"Level 2 native local law is not a unique gain/rotation explanation, cross-layer mechanism, or task function by itself.",
    }
    path=root/OUT/"v37_adjudication.json"
    write_json_atomic(path,result)
    seal=stage_freeze(root,"adjudication",
                      [SOURCE,str(path.relative_to(root)),
                       "artifacts/computational_origin_v37_final_opening.freeze.json",
                       *(["artifacts/computational_origin_v37_final_analysis.freeze.json"] if final else [])],
                      {"summary_sha256":sha256_file(path),
                       "strongest_supported_level":result["strongest_supported_level"]})
    return {"freeze_digest":seal["freeze_digest"],**result}


if __name__=="__main__":
    print(json.dumps(run(Path.cwd()),indent=2))
