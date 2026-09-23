"""Hash-check installed recurrence sources, models, tokenizers, and V37 inputs."""
from __future__ import annotations

import json
from pathlib import Path

import yaml

from jclosure.protocol_v37 import stage_freeze, verify, verify_stage
from jclosure.provenance import sha256_file, write_json_atomic


OUT=Path("results/v37/processed")
SOURCE="src/jclosure/experiments/source_audit_v37.py"
NATIVE={"Q":Path("/home/user/anaconda3/lib/python3.12/site-packages/transformers/models/qwen3_5/modeling_qwen3_5.py"),
        "F":Path("/home/user/anaconda3/lib/python3.12/site-packages/transformers/models/falcon_h1/modeling_falcon_h1.py")}


def run(root:Path):
    base=verify(root)
    for key in ("Q","F"):verify_stage(root,f"calibration_{key}")
    old=yaml.safe_load((root/"configs/operator_v36.yaml").read_text())
    records={}
    for key,path in NATIVE.items():
        digest=sha256_file(path)
        expected=old["native_equations"]["source_sha256"][key]
        if digest!=expected:raise RuntimeError(f"Native source drift for {key}")
        spec=base["model_specs"][key]
        records[key]={"model_id":spec["id"],"revision":spec["revision"],
                      "native_source":str(path),"native_source_sha256":digest,
                      "weight_sha256":spec["weight_sha256"],
                      "tokenizer_sha256":spec["tokenizer_sha256"],
                      "model_config_sha256":spec["config_sha256"],
                      "calibration_sha256":sha256_file(root/OUT/f"calibration_{key}_v37.json")}
    result={"models":records,
            "v36_native_formula_implementation_reused_only_as_engineering_basis":True,
            "v36_formal_responses_used_in_v37_adjudication":False,
            "v37_calibration_performed_independently":True,
            "v37_sample_pool_manifest_sha256":sha256_file(root/"data/v37/sample_pool_manifest_v37.json"),
            "v37_design_sha256":sha256_file(root/OUT/"execution_plan_v37.json")}
    path=root/OUT/"source_audit_v37.json"
    write_json_atomic(path,result)
    seal=stage_freeze(root,"source_audit",[SOURCE,str(path.relative_to(root)),
                     "configs/operator_v36.yaml",
                     *[f"artifacts/computational_origin_v37_calibration_{key}.freeze.json"
                       for key in ("Q","F")]],
                     {"summary_sha256":sha256_file(path),
                      "native_sources_match_prior_verified_installation":True})
    return {"freeze_digest":seal["freeze_digest"],**result}


if __name__=="__main__":
    print(json.dumps(run(Path.cwd()),indent=2))
